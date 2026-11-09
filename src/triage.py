#!/usr/bin/env python3
"""Dispute triage classifier.

Task: predict whether a complaint is at risk of a slow or unsatisfactory
resolution (label `at_risk`, defined in build_db.py) from the complaint
text plus product / issue / company. This is the triage step of dispute
case review: which cases should a human look at first.

Features: TF-IDF (1-2 grams, top 3,000 terms) over the consumer narrative,
plus one-hot product and issue, plus top-30 companies (other -> OTHER).
Models: logistic regression (balanced) vs a most-frequent-class baseline.
Split: time-based — train on complaints received 2025-01-01..2025-04-30,
test on complaints received 2025-05-01..2025-06-30 (the FOIA archive
narratives span Jan-Jun 2025). No future information leaks into
training: the split is on date_received, and the label is only known after
the company responds.
"""
import os

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import OneHotEncoder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOP_COMPANIES = 30
SPLIT_DATE = "2025-05-01"


def load_frame(db_path=None):
    import sqlite3
    db_path = db_path or os.path.join(ROOT, "data", "disputes.db")
    con = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT complaint_id, date_received, product, issue, company, "
        "company_response, timely, narrative, at_risk FROM complaints "
        "WHERE narrative IS NOT NULL AND narrative != '' "
        "AND at_risk IS NOT NULL",
        con)
    con.close()
    df["date_received"] = pd.to_datetime(df["date_received"], utc=True, format="mixed",
                                         errors="coerce").dt.tz_localize(None)
    return df


def build_features(df, vectorizer=None, encoder=None, top_companies=None):
    """Return (X, vectorizer, encoder, top_companies). Fits on first call."""
    text = df["narrative"].fillna("").astype(str)
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2),
                                     min_df=5, sublinear_tf=True)
        X_text = vectorizer.fit_transform(text)
    else:
        X_text = vectorizer.transform(text)
    company = df["company"].fillna("Unknown").astype(str)
    if top_companies is None:
        top_companies = set(company.value_counts().head(TOP_COMPANIES).index)
    company = company.where(company.isin(top_companies), "OTHER")
    cat = pd.DataFrame({
        "product": df["product"].fillna("Unknown").astype(str),
        "issue": df["issue"].fillna("Unknown").astype(str),
        "company": company,
    })
    if encoder is None:
        encoder = OneHotEncoder(handle_unknown="ignore")
        X_cat = encoder.fit_transform(cat)
    else:
        X_cat = encoder.transform(cat)
    return hstack([X_text, X_cat]).tocsr(), vectorizer, encoder, top_companies


def _metrics(y_true, y_pred, y_score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "base_rate_at_risk": round(float(np.mean(y_true)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def top_terms(vectorizer, coef, n=10):
    """Top TF-IDF terms by coefficient. coef covers text + one-hot features,
    so slice to the text part first."""
    names = np.array(vectorizer.get_feature_names_out())
    coef = coef[:len(names)]
    idx = np.argsort(coef)
    return {
        "most_at_risk": [(names[i], round(float(coef[i]), 3)) for i in idx[-n:][::-1]],
        "most_safe": [(names[i], round(float(coef[i]), 3)) for i in idx[:n]],
    }


def fit_model(df, split_date=SPLIT_DATE):
    """Fit the triage model on the train split; return artifacts dict."""
    df = df.sort_values("date_received")
    train = df[df["date_received"] < split_date]
    X_train, vec, enc, top_comp = build_features(train)
    model = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    model.fit(X_train, train["at_risk"].values)
    return {"model": model, "vectorizer": vec, "encoder": enc,
            "top_companies": top_comp, "split_date": split_date}


def score_row(row, artifacts):
    """At-risk probability for one complaint row (a pandas Series)."""
    import pandas as pd
    df = pd.DataFrame([row])
    X, _, _, _ = build_features(df, artifacts["vectorizer"],
                                artifacts["encoder"],
                                artifacts["top_companies"])
    return float(artifacts["model"].predict_proba(X)[0, 1])


def run(df=None, split_date=SPLIT_DATE):
    df = df if df is not None else load_frame()
    df = df.sort_values("date_received")
    train = df[df["date_received"] < split_date]
    test = df[df["date_received"] >= split_date]
    if len(train) < 100 or len(test) < 50:
        raise ValueError(f"not enough labeled rows: train={len(train)} test={len(test)}")
    X_train, vec, enc, top_comp = build_features(train)
    X_test, _, _, _ = build_features(test, vec, enc, top_comp)
    y_train, y_test = train["at_risk"].values, test["at_risk"].values

    model = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    model.fit(X_train, y_train)
    score = model.predict_proba(X_test)[:, 1]
    pred = (score >= 0.5).astype(int)
    lr = _metrics(y_test, pred, score)
    lr["model"] = "logistic regression (TF-IDF text + product/issue/company)"

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    d_pred = dummy.predict(X_test)
    d_score = np.full(len(y_test), float(np.mean(y_train)))
    base = _metrics(y_test, d_pred, d_score)
    base["model"] = "most-frequent-class baseline"
    base["roc_auc"] = 0.5  # a constant score carries no ranking signal

    terms = top_terms(vec, model.coef_[0])
    return {
        "label": "at_risk = timely=='No' OR company_response in "
                 "('Closed without relief', 'Untimely response'); "
                 "'In progress' rows excluded (outcome unknown)",
        "split": f"train date_received < {split_date}; test >= {split_date}",
        "n_train": int(len(train)), "n_test": int(len(test)),
        "logistic_regression": lr,
        "baseline": base,
        "top_terms": terms,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))

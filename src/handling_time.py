#!/usr/bin/env python3
"""Handling-time analysis, in the spirit of AHT (average handle time).

IMPORTANT — what this is and is not: the CFPB data has no contact-center
handle time. What it does have is (a) the date a complaint was received,
(b) the date it was sent to the company, and (c) whether the company's
response was timely. So "handling time" here means CFPB response
timeliness: days from receipt to the complaint being sent to the company,
and the share of responses flagged timely. No AHT from a real contact
center is claimed anywhere in this project.

Analyses:
  - distribution of response lag (median / mean / p75 / p90)
  - timely rate overall, by product, by issue, by top companies
  - what drives slow resolution: median lag by issue/product, plus a
    small OLS model of log1p(lag) on one-hot product+issue, with
    train/test R2 reported honestly (a low R2 is a finding, not a failure:
    most of the lag is company process, not the issue type).
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAG_BINS = [("0 days", 0, 0), ("1-7 days", 1, 7), ("8-14 days", 8, 14),
            ("15-30 days", 15, 30), ("31+ days", 31, 10**9)]


def load_frame(db_path=None):
    import sqlite3
    db_path = db_path or os.path.join(ROOT, "data", "disputes.db")
    con = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT complaint_id, date_received, date_sent_to_company, product, "
        "issue, company, company_response, timely, response_lag_days "
        "FROM complaints WHERE source = 'api'",
        con)
    con.close()
    for col in ("date_received", "date_sent_to_company"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce", format="mixed").dt.tz_localize(None)
    return df


def _lag_band(days):
    for label, lo, hi in LAG_BINS:
        if lo <= days <= hi:
            return label
    return "unknown"


def _rate_table(df, col, min_n=50, top=10):
    g = df.groupby(col).agg(
        n=("timely", "size"),
        timely_rate=("timely", lambda s: float((s == "Yes").mean())),
        median_lag=("response_lag_days", "median"))
    g = g[g["n"] >= min_n].sort_values("timely_rate")
    out = []
    for name, r in g.head(top).iterrows():
        out.append({"name": str(name), "n": int(r["n"]),
                    "timely_rate": round(float(r["timely_rate"]), 4),
                    "median_lag_days": None if pd.isna(r["median_lag"]) else float(r["median_lag"])})
    return out


def run(df=None):
    df = df if df is not None else load_frame()
    lag = df["response_lag_days"].dropna()
    timely = df["timely"].dropna()
    bands = lag.apply(_lag_band).value_counts()
    total_bands = int(bands.sum())
    dist = {label: {"n": int(bands.get(label, 0)),
                    "share": round(float(bands.get(label, 0)) / total_bands, 4)}
            for label, _, _ in LAG_BINS}
    result = {
        "definition": "response_lag_days = date_sent_to_company minus "
                      "date_received, in days. This is CFPB process time, "
                      "NOT contact-center average handle time (AHT).",
        "n_with_lag": int(len(lag)),
        "lag_median_days": float(lag.median()),
        "lag_mean_days": round(float(lag.mean()), 2),
        "lag_p75_days": float(lag.quantile(0.75)),
        "lag_p90_days": float(lag.quantile(0.90)),
        "lag_max_days": int(lag.max()),
        "timely_rate": round(float((timely == "Yes").mean()), 4),
        "n_with_timely_flag": int(len(timely)),
        "lag_distribution": dist,
        "slowest_issues_by_timely_rate": _rate_table(df, "issue"),
        "slowest_products_by_timely_rate": _rate_table(df, "product"),
        "slowest_companies_by_timely_rate": _rate_table(df, "company", min_n=100, top=10),
    }
    # What drives slow resolution: OLS on log1p(lag), one-hot product+issue.
    m = df.dropna(subset=["response_lag_days"]).copy()
    m = m[m["response_lag_days"] >= 0]
    if len(m) >= 500:
        cat = pd.DataFrame({
            "product": m["product"].fillna("Unknown").astype(str),
            "issue": m["issue"].fillna("Unknown").astype(str),
        })
        # keep only frequent categories so the model stays readable
        for c in cat.columns:
            top = set(cat[c].value_counts().head(15).index)
            cat[c] = cat[c].where(cat[c].isin(top), "OTHER")
        enc = OneHotEncoder(handle_unknown="ignore")
        X = enc.fit_transform(cat)
        y = np.log1p(m["response_lag_days"].values)
        rng = np.random.RandomState(42)
        idx = rng.permutation(len(y))
        cut = int(0.8 * len(y))
        tr, te = idx[:cut], idx[cut:]
        ols = LinearRegression()
        ols.fit(X[tr], y[tr])
        names = enc.get_feature_names_out()
        coef = ols.coef_
        order = np.argsort(coef)
        result["lag_drivers_ols"] = {
            "target": "log1p(response_lag_days)",
            "features": "one-hot product + issue (top 15 each, rest OTHER)",
            "train_r2": round(float(ols.score(X[tr], y[tr])), 4),
            "test_r2": round(float(ols.score(X[te], y[te])), 4),
            "note": "A low R2 means issue/product type explains little of "
                    "the lag; most of it is company process, which the "
                    "data does not describe.",
            "adds_most_days": [
                {"feature": str(names[i]), "coef_log_days": round(float(coef[i]), 3)}
                for i in order[-5:][::-1]],
            "adds_fewest_days": [
                {"feature": str(names[i]), "coef_log_days": round(float(coef[i]), 3)}
                for i in order[:5]],
        }
    return result


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))

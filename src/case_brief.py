#!/usr/bin/env python3
"""Evidence synthesis: turn one complaint into a case brief.

A brief is what a dispute analyst reads before touching a case: the
consumer's story (excerpt), the facts (product, issue, company, dates,
company response, timely flag), the key terms in the narrative, the most
similar past complaints (same issue, TF-IDF cosine similarity — precedent
cases with known outcomes), and a rule-based suggested next step.

The next step is a RULE, not a decision: it combines the triage model's
risk score (when one is supplied) with the recorded outcome fields.
Nothing here decides a case; a human reviews every brief.
"""
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_frame(db_path=None):
    import sqlite3
    db_path = db_path or os.path.join(ROOT, "data", "disputes.db")
    con = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM complaints "
                     "WHERE narrative IS NOT NULL AND narrative != ''",
                     con)
    con.close()
    for col in ("date_received", "date_sent_to_company"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce", format="mixed").dt.tz_localize(None)
    return df


class BriefBuilder:
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.by_id = {str(c): i for i, c in enumerate(self.df["complaint_id"])}
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                                          min_df=3, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(
            self.df["narrative"].fillna("").astype(str))
        self.risk_fn = None

    def set_risk_fn(self, fn):
        """fn(row: Series) -> float in [0, 1], the triage at-risk score."""
        self.risk_fn = fn

    def _keywords(self, i, n=8):
        row = self.matrix[i]
        if row.nnz == 0:
            return []
        names = np.array(self.vectorizer.get_feature_names_out())
        idx = np.argsort(row.data)[::-1][:n]
        cols = row.indices[idx]
        return [str(names[c]) for c in cols]

    def _similar(self, i, n=3):
        issue = self.df.loc[i, "issue"]
        same = self.df[self.df["issue"] == issue].index.tolist()
        same = [j for j in same if j != i]
        if not same:
            return []
        sims = cosine_similarity(self.matrix[i], self.matrix[same])[0]
        order = np.argsort(sims)[::-1][:n]
        out = []
        for k in order:
            j = same[k]
            r = self.df.loc[j]
            out.append({
                "complaint_id": str(r["complaint_id"]),
                "company": str(r["company"]),
                "company_response": str(r["company_response"]),
                "timely": str(r["timely"]),
                "similarity": round(float(sims[k]), 3),
            })
        return out

    def _next_step(self, row, risk):
        if str(row["company_response"]) == "In progress":
            return ("Open case: no company response recorded yet. Chase the "
                    "company response and check the timely flag.")
        if risk is not None and risk >= 0.7:
            return ("High triage risk (model score "
                    f"{risk:.2f}): escalate for human review before the "
                    "response deadline; check the similar cases for how "
                    "this issue usually resolves.")
        if str(row["timely"]) == "No":
            return ("Untimely response recorded: review why the response "
                    "was late and whether the consumer needs a follow-up.")
        if str(row["company_response"]) == "Closed without relief":
            return ("Closed without relief: review the evidence for a "
                    "goodwill or relief option, and document the reasoning.")
        if str(row["company_response"]) == "Closed with monetary relief":
            return ("Resolved with monetary relief: confirm the relief "
                    "reached the consumer and close the case file.")
        return "Standard follow-up: confirm the recorded resolution with the consumer."

    def brief(self, complaint_id):
        i = self.by_id.get(str(complaint_id))
        if i is None:
            return None
        row = self.df.loc[i]
        narrative = str(row["narrative"] or "")
        risk = self.risk_fn(row) if self.risk_fn is not None else None
        return {
            "complaint_id": str(row["complaint_id"]),
            "date_received": str(row["date_received"])[:10],
            "date_sent_to_company": str(row["date_sent_to_company"])[:10],
            "product": str(row["product"]),
            "sub_product": str(row["sub_product"]),
            "issue": str(row["issue"]),
            "sub_issue": str(row["sub_issue"]),
            "company": str(row["company"]),
            "company_response": str(row["company_response"]),
            "timely": str(row["timely"]),
            "consumer_disputed": str(row["consumer_disputed"]),
            "submitted_via": str(row["submitted_via"]),
            "response_lag_days": (None if pd.isna(row["response_lag_days"])
                                  else int(row["response_lag_days"])),
            "narrative_excerpt": narrative[:800],
            "narrative_chars": len(narrative),
            "keywords": self._keywords(i),
            "similar_cases": self._similar(i),
            "triage_risk_score": (None if risk is None else round(float(risk), 3)),
            "suggested_next_step": self._next_step(row, risk),
            "note": "Suggested next step is a rule based on the recorded "
                    "outcome fields and (when present) the triage model "
                    "score. It is not a decision; a human reviews it.",
        }


if __name__ == "__main__":
    import json
    import sys
    df = load_frame()
    b = BriefBuilder(df)
    cid = sys.argv[1] if len(sys.argv) > 1 else str(df["complaint_id"].iloc[0])
    print(json.dumps(b.brief(cid), indent=2))

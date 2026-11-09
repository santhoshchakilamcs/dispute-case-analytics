#!/usr/bin/env python3
"""Independently recompute the headline metrics straight from the raw
CSVs (not from reports/metrics.json, and not from the SQLite DB) and
cross-check them against reports/metrics.json, so the README numbers
can be checked without trusting the pipeline.

Two raw populations:
  data/raw/complaints_*.csv  — API structured sample (handling/trends checks)
  data/raw/narratives_2025.csv — FOIA archive narrative rows (triage refit)

Recomputes: row counts and date ranges, timely rate, median response
lag, top 3 issues, resolution mix, and the triage test metrics (refits
the logistic model from the raw narrative rows with the same label rule
and time split — deterministic, so the refit must match to 1e-6).
"""
import csv
import glob
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "raw")
sys.path.insert(0, os.path.join(ROOT, "src"))
import build_db


def read_api_rows():
    rows = []
    for path in sorted(glob.glob(os.path.join(RAW, "complaints_*.csv"))):
        with open(path, newline="", encoding="utf-8") as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def read_narrative_rows():
    rows = []
    for path in sorted(glob.glob(os.path.join(RAW, "narratives_*.csv"))):
        with open(path, newline="", encoding="utf-8") as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def main():
    api_rows = read_api_rows()
    narr_rows = read_narrative_rows()
    print(f"raw API rows: {len(api_rows):,}")
    print(f"raw narrative rows: {len(narr_rows):,}")
    dates = sorted(str(build_db.parse_date(r["date_received"]))
                   for r in api_rows if r.get("date_received"))
    print(f"API date range: {dates[0]}..{dates[-1]}")
    nd = sorted(str(build_db.parse_date(r["date_received"]))
                for r in narr_rows if r.get("date_received"))
    print(f"narrative date range: {nd[0]}..{nd[-1]}")
    timely = [r for r in api_rows if r.get("timely")]
    timely_rate = sum(1 for r in timely if r["timely"] == "Yes") / len(timely)
    print(f"timely rate: {timely_rate:.4f} (n={len(timely):,})")
    lags = sorted(build_db.lag_days(r.get("date_received"), r.get("date_sent_to_company"))
                  for r in api_rows)
    lags = [x for x in lags if x is not None and x >= 0]
    median_lag = lags[len(lags) // 2]
    print(f"median response lag: {median_lag} days (n={len(lags):,})")
    issues = Counter(r["issue"] for r in api_rows if r.get("issue"))
    top3 = issues.most_common(3)
    print("top 3 issues:", [(k, f"{v:,}") for k, v in top3])
    res = Counter(r["company_response"] for r in api_rows if r.get("company_response"))
    print("resolution mix top:", [(k, f"{v:,}") for k, v in res.most_common(3)])

    with open(os.path.join(ROOT, "reports", "metrics.json")) as f:
        m = json.load(f)
    assert m["dataset"]["n_api_complaints"] == len(api_rows), "n_api_complaints mismatch"
    assert m["dataset"]["n_narrative_complaints"] == len(narr_rows), "n_narrative_complaints mismatch"
    assert m["dataset"]["api_range"] == f"{dates[0]}..{dates[-1]}", "API date range mismatch"
    assert m["dataset"]["narrative_range"] == f"{nd[0]}..{nd[-1]}", "narrative date range mismatch"
    assert abs(m["handling_time"]["timely_rate"] - round(timely_rate, 4)) < 1e-9, "timely rate mismatch"
    assert abs(m["handling_time"]["lag_median_days"] - median_lag) < 1e-9, "median lag mismatch"
    assert [x["name"] for x in m["trends"]["top_issues"][:3]] == [k for k, _ in top3], "top issues mismatch"

    # refit the triage model from the raw narrative rows (same label rule,
    # same split) — deterministic, so it must match to 1e-6.
    import pandas as pd
    import triage as tg
    recs = []
    for r in narr_rows:
        lab = build_db.at_risk(r.get("timely"), r.get("company_response"))
        if lab is None or not r.get("complaint_what_happened"):
            continue
        recs.append({
            "complaint_id": r["complaint_id"],
            "date_received": pd.Timestamp(build_db.parse_date(r["date_received"])),
            "product": r["product"], "issue": r["issue"],
            "company": r["company"], "company_response": r["company_response"],
            "timely": r["timely"],
            "narrative": r["complaint_what_happened"], "at_risk": lab,
        })
    df = pd.DataFrame(recs)
    refit = tg.run(df)
    lr, ref = m["triage"]["logistic_regression"], refit["logistic_regression"]
    print(f"refit triage test: ROC-AUC {ref['roc_auc']} accuracy {ref['accuracy']} "
          f"(metrics.json: ROC-AUC {lr['roc_auc']} accuracy {lr['accuracy']})")
    assert abs(ref["roc_auc"] - lr["roc_auc"]) < 1e-6, "triage ROC-AUC mismatch"
    assert abs(ref["accuracy"] - lr["accuracy"]) < 1e-6, "triage accuracy mismatch"
    assert refit["n_test"] == m["triage"]["n_test"], "triage test size mismatch"
    print("cross-check vs reports/metrics.json: OK")


if __name__ == "__main__":
    main()

"""Tests for dispute-case-analytics. Run: .venv/bin/python -m pytest tests/ -q

Most tests run against the real downloaded data (data/disputes.db built by
src/pipeline.py). The label-rule and lag tests are pure unit tests; the
triage/handling/trends/brief tests assert the real pipeline's shape and
honest-metric properties (no invented numbers — bounds only).
"""
import json
import os
import sqlite3
import sys
import threading
import urllib.request

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import build_db
import case_brief as cb
import handling_time as ht
import trends as tr
import triage as tg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "data", "disputes.db")
METRICS = os.path.join(ROOT, "reports", "metrics.json")


# ---- unit: label rule and lag -------------------------------------------

def test_at_risk_rule():
    assert build_db.at_risk("No", "Closed with explanation") == 1
    assert build_db.at_risk("Yes", "Closed without relief") == 1
    assert build_db.at_risk("Yes", "Untimely response") == 1
    assert build_db.at_risk("Yes", "Closed with monetary relief") == 0
    assert build_db.at_risk("Yes", "Closed with explanation") == 0
    assert build_db.at_risk("Yes", "In progress") is None
    assert build_db.at_risk("", "Closed with explanation") is None
    assert build_db.at_risk(None, "Closed with explanation") is None


def test_lag_days_both_date_formats():
    assert build_db.lag_days("2025-01-10T00:00:00.000Z", "2025-01-12T00:00:00.000Z") == 2
    assert build_db.lag_days("01/10/2025", "01/12/2025") == 2
    assert build_db.lag_days("2025-01-10", "") is None
    assert build_db.lag_days("", "2025-01-12") is None
    assert build_db.lag_days("garbage", "2025-01-12") is None


# ---- real data: both populations present ----------------------------------

def test_db_has_both_populations():
    con = sqlite3.connect(DB)
    n_api = con.execute("SELECT COUNT(*) FROM complaints WHERE source='api'").fetchone()[0]
    n_narr = con.execute(
        "SELECT COUNT(*) FROM complaints WHERE narrative IS NOT NULL AND narrative != ''"
    ).fetchone()[0]
    n_total = con.execute("SELECT COUNT(*) FROM complaints").fetchone()[0]
    con.close()
    assert n_api == 20000, f"expected 20,000 API rows, got {n_api}"
    assert n_narr >= 10000, f"expected >=10,000 narrative rows, got {n_narr}"
    # API rows carry no narratives, archive rows all do, so the two counts
    # partition the table (an archive row replaces any API row with the
    # same complaint_id).
    assert n_total == n_api + n_narr


def test_narratives_are_real_text():
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT narrative FROM complaints WHERE narrative != '' LIMIT 100"
    ).fetchall()
    con.close()
    lens = [len(r[0]) for r in rows]
    assert min(lens) > 20, "narratives should be real consumer text, not stubs"
    assert sum(lens) / len(lens) > 200


# ---- triage ---------------------------------------------------------------

def test_triage_beats_baseline_and_reports_honest_metrics():
    result = tg.run()
    lr, base = result["logistic_regression"], result["baseline"]
    assert result["n_train"] >= 1000 and result["n_test"] >= 100
    assert 0.5 <= lr["roc_auc"] <= 1.0
    assert lr["roc_auc"] >= base["roc_auc"] - 0.01
    assert 0.0 <= lr["precision"] <= 1.0 and 0.0 <= lr["recall"] <= 1.0
    assert lr["confusion"]["tn"] + lr["confusion"]["fp"] + lr["confusion"]["fn"] + lr["confusion"]["tp"] == lr["n"]
    assert len(result["top_terms"]["most_at_risk"]) == 10


def test_triage_score_row_in_range():
    df = tg.load_frame()
    artifacts = tg.fit_model(df)
    row = df.iloc[0]
    score = tg.score_row(row, artifacts)
    assert 0.0 <= score <= 1.0


# ---- handling time ----------------------------------------------------------

def test_handling_time_shape_and_bounds():
    result = ht.run()
    assert result["n_with_lag"] > 10000
    assert 0 <= result["lag_median_days"] <= 30
    assert 0.0 <= result["timely_rate"] <= 1.0
    assert result["lag_p90_days"] >= result["lag_median_days"]
    assert "lag_drivers_ols" in result
    assert -1.0 <= result["lag_drivers_ols"]["test_r2"] <= 1.0
    slow = result["slowest_issues_by_timely_rate"]
    assert slow and all(0.0 <= s["timely_rate"] <= 1.0 for s in slow)


# ---- trends -----------------------------------------------------------------

def test_trends_shape_and_rates():
    result = tr.run()
    assert result["n_complaints"] == 20000
    assert result["top_issues"] and result["top_companies"]
    assert abs(sum(r["share"] for r in result["resolution_mix"]) - 1.0) < 0.05
    assert 0.0 <= result["timely_rate"] <= 1.0
    assert result["monthly_volumes"]


# ---- case briefs -------------------------------------------------------------

def test_case_brief_fields_and_similar_cases():
    df = cb.load_frame()
    builder = cb.BriefBuilder(df)
    cid = str(df["complaint_id"].iloc[0])
    brief = builder.brief(cid)
    assert brief is not None
    assert brief["complaint_id"] == cid
    assert len(brief["narrative_excerpt"]) > 20
    assert isinstance(brief["keywords"], list)
    assert isinstance(brief["similar_cases"], list)
    assert "rule" in brief["note"].lower() or "human" in brief["note"].lower()
    assert brief["suggested_next_step"]


def test_case_brief_unknown_id_returns_none():
    df = cb.load_frame().head(50)
    builder = cb.BriefBuilder(df)
    assert builder.brief("00000000") is None


# ---- metrics file and app -----------------------------------------------------

def test_metrics_json_matches_db_counts():
    with open(METRICS) as f:
        m = json.load(f)
    con = sqlite3.connect(DB)
    n_api = con.execute("SELECT COUNT(*) FROM complaints WHERE source='api'").fetchone()[0]
    n_narr = con.execute(
        "SELECT COUNT(*) FROM complaints WHERE narrative != ''").fetchone()[0]
    con.close()
    assert m["dataset"]["n_api_complaints"] == n_api
    assert m["dataset"]["n_narrative_complaints"] == n_narr
    assert m["triage"]["logistic_regression"]["roc_auc"] > 0.5


def test_app_complaints_endpoint():
    import app as appmod
    import http.server
    import socketserver

    with socketserver.TCPServer(("127.0.0.1", 0), appmod.Handler) as httpd:
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/complaints?limit=5",
                    timeout=30) as resp:
                rows = json.load(resp)
            assert isinstance(rows, list) and len(rows) == 5
            assert "complaint_id" in rows[0] and "issue" in rows[0]
        finally:
            httpd.shutdown()

#!/usr/bin/env python3
"""End-to-end pipeline: build the SQLite DB (if missing), run the triage
classifier, the handling-time analysis, and the trend dashboards, build
three example case briefs, and write reports/metrics.json +
reports/dashboard.html.
"""
import json
import os
import sqlite3
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_db
import case_brief as cb
import handling_time as ht
import trends as tr
import triage as tg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = os.path.join(ROOT, "reports")
os.makedirs(REPORTS, exist_ok=True)


def load_all(db_path):
    con = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM complaints", con)
    con.close()
    # Dates arrive mixed: API rows are ISO timestamps with a Z suffix,
    # archive rows are plain YYYY-MM-DD. Parse both as UTC, then drop the
    # timezone so the two populations compare cleanly.
    for col in ("date_received", "date_sent_to_company"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce", format="mixed").dt.tz_localize(None)
    return df


def pick_examples(df, builder, artifacts):
    """One brief per archetype: high-risk open case, timely monetary-relief
    case, untimely case. Returns the three brief dicts."""
    scored = df.copy()
    X, _, _, _ = tg.build_features(
        scored, artifacts["vectorizer"], artifacts["encoder"],
        artifacts["top_companies"])
    scored["risk"] = artifacts["model"].predict_proba(X)[:, 1]
    risk_by_id = dict(zip(scored["complaint_id"].astype(str), scored["risk"]))
    builder.set_risk_fn(lambda row: risk_by_id.get(str(row["complaint_id"])))
    picks = []
    open_risky = scored[(scored["company_response"] == "In progress")]
    if len(open_risky):
        picks.append(("High-risk open case",
                      str(open_risky.sort_values("risk", ascending=False)
                          ["complaint_id"].iloc[0])))
    else:
        # The sampled complaints are all resolved by now, so there is no
        # open case to show; use the highest-risk closed case instead.
        picks.append(("Highest triage-risk case (no open cases in the sample)",
                      str(scored.sort_values("risk", ascending=False)
                          ["complaint_id"].iloc[0])))
    relief = scored[scored["company_response"] == "Closed with monetary relief"]
    if len(relief):
        picks.append(("Timely case closed with monetary relief",
                      str(relief.sort_values("risk")["complaint_id"].iloc[0])))
    untimely = scored[scored["timely"] == "No"]
    if len(untimely):
        picks.append(("Untimely response case",
                      str(untimely.sort_values("risk", ascending=False)
                          ["complaint_id"].iloc[0])))
    return [(label, builder.brief(cid)) for label, cid in picks]


def dashboard_html(metrics):
    t = metrics["triage"]["logistic_regression"]
    b = metrics["triage"]["baseline"]
    h = metrics["handling_time"]
    trd = metrics["trends"]
    rows = ""
    for r in trd["top_issues"]:
        rows += f"<tr><td>{r['name']}</td><td>{r['n']:,}</td><td>{r['share']:.1%}</td></tr>\n"
    comp_rows = ""
    for r in trd["top_companies"]:
        comp_rows += f"<tr><td>{r['name']}</td><td>{r['n']:,}</td><td>{r['share']:.1%}</td></tr>\n"
    res_rows = ""
    for r in trd["resolution_mix"]:
        res_rows += f"<tr><td>{r['name']}</td><td>{r['n']:,}</td><td>{r['share']:.1%}</td></tr>\n"
    slow_rows = ""
    for r in h["slowest_issues_by_timely_rate"]:
        slow_rows += (f"<tr><td>{r['name']}</td><td>{r['n']:,}</td>"
                      f"<td>{r['timely_rate']:.1%}</td><td>{r['median_lag_days']}</td></tr>\n")
    mon_rows = ""
    for r in trd["monthly_volumes"]:
        mon_rows += f"<tr><td>{r['month']}</td><td>{r['n']:,}</td></tr>\n"
    brief_html = ""
    for label, br in metrics["example_briefs"]:
        if not br:
            continue
        sims = "".join(
            f"<li>{s['complaint_id']} — {s['company']} — {s['company_response']} "
            f"(timely: {s['timely']}, similarity {s['similarity']})</li>"
            for s in br["similar_cases"])
        brief_html += f"""
<h3>{label}</h3>
<p><b>Complaint {br['complaint_id']}</b> — {br['product']} / {br['issue']} — {br['company']}<br>
Received {br['date_received']}, sent to company {br['date_sent_to_company']},
response lag {br['response_lag_days']} days, timely: {br['timely']},
outcome: {br['company_response']}. Triage risk score: {br['triage_risk_score']}<br>
<i>{br['narrative_excerpt']}</i><br>
Keywords: {', '.join(br['keywords'])}<br>
Similar past cases: <ul>{sims}</ul>
<b>Suggested next step:</b> {br['suggested_next_step']}</p>"""
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Dispute Case Analytics</title>
<style>body{{font-family:system-ui,sans-serif;max-width:960px;margin:2em auto;padding:0 1em;line-height:1.5}}
table{{border-collapse:collapse;margin:1em 0}}td,th{{border:1px solid #ccc;padding:4px 10px;text-align:left}}th{{background:#f0f0f0}}
h1{{font-size:1.5em}}h2{{font-size:1.2em;margin-top:1.5em}}.note{{background:#fff8e1;padding:0.8em;border:1px solid #e6c200}}</style></head><body>
<h1>Dispute Case Analytics</h1>
<p>Personal project on real public data: CFPB Consumer Complaint Database
(consumerfinance.gov) — {metrics['dataset']['n_api_complaints']:,} structured
complaints from the public API ({metrics['dataset']['api_range']}) plus
{metrics['dataset']['n_narrative_complaints']:,} with real consumer-written
narratives from the CFPB FOIA Narratives Archive
({metrics['dataset']['narrative_range']}). This is <b>not</b> Chime data,
<b>not</b> a production dispute system, and <b>not</b> dispute-investigation
job experience. Volumes are sample volumes.</p>
<h2>Dispute triage: predict slow or unsatisfactory resolution</h2>
<p>Label: at-risk = timely 'No' OR 'Closed without relief' / 'Untimely response'.
'In progress' rows excluded (outcome unknown). Train {metrics['triage']['n_train']:,},
test {metrics['triage']['n_test']:,} (time split {metrics['triage']['split']}).</p>
<table><tr><th>Model</th><th>Accuracy</th><th>ROC-AUC</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
<tr><td>Logistic regression</td><td>{t['accuracy']:.3f}</td><td>{t['roc_auc']:.3f}</td><td>{t['precision']:.3f}</td><td>{t['recall']:.3f}</td><td>{t['f1']:.3f}</td></tr>
<tr><td>Most-frequent baseline</td><td>{b['accuracy']:.3f}</td><td>{b['roc_auc']:.3f}</td><td>{b['precision']:.3f}</td><td>{b['recall']:.3f}</td><td>{b['f1']:.3f}</td></tr></table>
<p>Words most associated with at-risk: {', '.join(w for w, _ in metrics['triage']['top_terms']['most_at_risk'])}<br>
Words most associated with safe resolution: {', '.join(w for w, _ in metrics['triage']['top_terms']['most_safe'])}</p>
<h2>Handling time (CFPB response timeliness — NOT contact-center AHT)</h2>
<p>Median response lag {h['lag_median_days']:.0f} days (mean {h['lag_mean_days']}, p75 {h['lag_p75_days']:.0f}, p90 {h['lag_p90_days']:.0f}).
Timely rate {h['timely_rate']:.1%} over {h['n_with_timely_flag']:,} complaints.</p>
<table><tr><th>Issue (slowest by timely rate)</th><th>n</th><th>Timely rate</th><th>Median lag (days)</th></tr>{slow_rows}</table>
<p>OLS on log1p(lag), product+issue one-hots: test R² {h['lag_drivers_ols']['test_r2']}.
{h['lag_drivers_ols']['note']}</p>
<h2>Trends (sample volumes)</h2>
<p>{trd['sampling_note']}</p>
<table><tr><th>Month</th><th>Complaints (API sample)</th></tr>{mon_rows}</table>
<table><tr><th>Top issue</th><th>n</th><th>Share</th></tr>{rows}</table>
<table><tr><th>Top company</th><th>n</th><th>Share</th></tr>{comp_rows}</table>
<table><tr><th>Resolution</th><th>n</th><th>Share</th></tr>{res_rows}</table>
<p>Consumer-disputed rate: {('n/a — ' + trd['consumer_disputed_note']) if trd['consumer_disputed_rate'] is None else f"{trd['consumer_disputed_rate']:.1%}"}. {trd['emerging_note']}</p>
<h2>Example case briefs (evidence synthesis)</h2>
{brief_html}
<p class="note"><b>Honest limits:</b> personal project on public CFPB data, not
Chime data, not production dispute handling, not dispute-investigation job
experience. Response lag is CFPB process time (receipt to sent-to-company),
not contact-center average handle time. Suggested next steps are rules, not
decisions; a human reviews every case. Volumes are API-sample volumes, not
total CFPB volumes. The CFPB API no longer returns narrative text (since
September 2026), so narratives come from the FOIA archive exports, which
cover complaints received through August 14, 2026 only.</p>
</body></html>"""


def main():
    db_path = os.path.join(ROOT, "data", "disputes.db")
    if not os.path.exists(db_path):
        build_db.main()
    df = load_all(db_path)
    # Two populations, kept separate on purpose: the API sample (structured
    # rows, no narratives) drives handling-time and trends; the FOIA archive
    # rows (real narratives) drive triage and case briefs.
    df_api = df[df["source"] == "api"]
    df_narr = df[df["narrative"].fillna("") != ""]
    triage = tg.run(df_narr)
    handling = ht.run(df_api)
    trends = tr.run(df_api)
    global builder_artifacts
    builder_artifacts = tg.fit_model(df_narr)
    builder = cb.BriefBuilder(df_narr)
    examples = pick_examples(df_narr, builder, builder_artifacts)
    api_dates = df_api["date_received"]
    narr_dates = df_narr["date_received"]
    metrics = {
        "dataset": {
            "sources": [
                "CFPB Consumer Complaint Database public API "
                "(consumerfinance.gov/data-research/consumer-complaints/search/api/v1/) "
                "— structured complaints, four half-year chunks 2024-2025, up to 5,000 each, "
                "most-recent-first per chunk. The API no longer returns narrative text.",
                "CFPB FOIA Reading Room, CFPB Consumer Complaint Database Narratives Archive "
                "(consumerfinance.gov/foia-requests/foia-electronic-reading-room/"
                "cfpb-consumer-complaint-database-narratives-archive/) — bimonthly full-record "
                "exports Jan-Jun 2025, narrative rows only, capped at 20,000 per export.",
            ],
            "license": "Public data published by the Consumer Financial Protection Bureau (CC0 / public domain, USA.gov)",
            "n_complaints": int(len(df)),
            "n_api_complaints": int(len(df_api)),
            "n_narrative_complaints": int(len(df_narr)),
            "api_range": f"{str(api_dates.min().date())}..{str(api_dates.max().date())}",
            "narrative_range": f"{str(narr_dates.min().date())}..{str(narr_dates.max().date())}",
            "note": "Samples, NOT a census of all CFPB complaints. API chunks are "
                    "most-recent-first samples; archive rows are narrative complaints only.",
        },
        "triage": triage,
        "handling_time": handling,
        "trends": trends,
        "example_briefs": examples,
    }
    with open(os.path.join(REPORTS, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(REPORTS, "dashboard.html"), "w") as f:
        f.write(dashboard_html(metrics))
    lr = triage["logistic_regression"]
    print("wrote reports/metrics.json and reports/dashboard.html")
    print(f"triage test ROC-AUC: {lr['roc_auc']}  accuracy: {lr['accuracy']}  "
          f"precision: {lr['precision']}  recall: {lr['recall']}  F1: {lr['f1']}")
    print(f"handling: median lag {handling['lag_median_days']:.0f} days, "
          f"timely rate {handling['timely_rate']:.1%}")
    print(f"trends: {trends['n_complaints']:,} complaints, "
          f"{trends['date_range']}")


if __name__ == "__main__":
    main()

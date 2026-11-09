#!/usr/bin/env python3
"""Loss / trend dashboards over the real complaint sample.

All volumes are counts of complaints in the downloaded API sample,
NOT total CFPB complaint volumes (the download is a sample: four
half-year chunks, up to 5,000 complaints each, most-recent-first per
chunk). Every table in the output says so. Metrics: daily and monthly
volumes, top products / issues / companies, resolution mix
(company_response), timely rate over time, consumer-disputed rate, and
emerging issues (share change, last 90 days vs prior 90 days, with the
same seasonality caution the reference project used: a rising share
can be seasonal, not a new trend).
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_frame(db_path=None):
    import sqlite3
    db_path = db_path or os.path.join(ROOT, "data", "disputes.db")
    con = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT complaint_id, date_received, product, issue, company, "
        "company_response, timely, consumer_disputed FROM complaints "
        "WHERE source = 'api'",
        con)
    con.close()
    df["date_received"] = pd.to_datetime(df["date_received"], utc=True, format="mixed",
                                         errors="coerce").dt.tz_localize(None)
    return df


def _top(df, col, n=10):
    vc = df[col].fillna("Unknown").value_counts()
    total = len(df)
    return [{"name": str(k), "n": int(v), "share": round(float(v) / total, 4)}
            for k, v in vc.head(n).items()]


def run(df=None):
    df = df if df is not None else load_frame()
    df = df.sort_values("date_received")
    total = len(df)
    monthly = (df.set_index("date_received").resample("ME").size())
    monthly = [{"month": str(k.date()), "n": int(v)} for k, v in monthly.items()]
    daily = df.groupby(df["date_received"].dt.date).size()
    resolution = _top(df, "company_response", n=10)
    timely = df["timely"].dropna()
    disputed = df["consumer_disputed"].dropna()
    disputed = disputed[disputed != ""]  # CFPB dropped this field in Jun 2026; API rows carry ""
    # emerging issues: last 90 days vs prior 90 days, by share
    dmax = df["date_received"].max()
    recent = df[df["date_received"] > dmax - pd.Timedelta(days=90)]
    prior = df[(df["date_received"] <= dmax - pd.Timedelta(days=90)) &
               (df["date_received"] > dmax - pd.Timedelta(days=180))]
    em = []
    if len(recent) >= 100 and len(prior) >= 100:
        rs = recent["issue"].value_counts(normalize=True)
        ps = prior["issue"].value_counts(normalize=True)
        for issue in set(rs.index) | set(ps.index):
            r, p = float(rs.get(issue, 0)), float(ps.get(issue, 0))
            if recent[recent["issue"] == issue].shape[0] >= 20:
                em.append({"issue": str(issue),
                           "share_prior": round(p, 4),
                           "share_recent": round(r, 4),
                           "change_pp": round((r - p) * 100, 2)})
        em = sorted(em, key=lambda e: -e["change_pp"])[:8]
    # The API returns most-recent-first, so each half-year chunk clusters
    # on a few dates near the window end; monthly volumes show spikes, not
    # a real monthly trend. Say so instead of letting the table imply one.
    top_day_share = float(df.groupby(df["date_received"].dt.date).size().max()) / total
    return {
        "note": "All volumes are counts of complaints in the downloaded "
                "API sample (four half-year chunks, up to 5,000 each, "
                "most-recent-first), NOT total CFPB complaint volumes.",
        "sampling_note": (
            f"The API sample clusters on window-end dates: the single busiest "
            f"day holds {top_day_share:.1%} of the sample. Monthly volumes "
            f"therefore spike on those days and are NOT a real monthly trend."),
        "consumer_disputed_note": (
            "CFPB removed the consumer-disputed field from exports in June "
            "2026, so no disputed values exist in this sample; the rate is "
            "reported as null, not 0%.") if len(disputed) == 0 else None,
        "n_complaints": int(total),
        "date_range": f"{str(df['date_received'].min().date())}..{str(df['date_received'].max().date())}",
        "mean_per_day": round(float(daily.mean()), 1),
        "monthly_volumes": monthly,
        "top_products": _top(df, "product"),
        "top_issues": _top(df, "issue"),
        "top_companies": _top(df, "company"),
        "resolution_mix": resolution,
        "timely_rate": round(float((timely == "Yes").mean()), 4) if len(timely) else None,
        "consumer_disputed_rate": round(float((disputed == "Yes").mean()), 4) if len(disputed) else None,
        "emerging_issues": em,
        "emerging_note": "A rising share over 90 days can be seasonal or "
                         "driven by one company; check the raw counts and "
                         "the company mix before acting on it.",
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))

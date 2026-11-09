#!/usr/bin/env python3
"""Load the downloaded CFPB complaint CSVs into data/disputes.db (SQLite).

One table, `complaints`, one row per real consumer complaint. Narratives
are stored as-is (consumer text, public data). Derived columns:
  response_lag_days - days from date_received to date_sent_to_company
  at_risk           - 1 if the outcome looks slow or unsatisfactory
                      (timely == 'No' OR company_response in
                      ('Closed without relief', 'Untimely response')),
                      0 otherwise, NULL when the outcome is unknown
                      (company_response == 'In progress' or timely blank).
"""
import csv
import glob
import os
import sqlite3
from datetime import date, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
DB_PATH = os.path.join(ROOT, "data", "disputes.db")

AT_RISK_RESPONSES = {"Closed without relief", "Untimely response"}


def parse_date(s):
    """ISO (API) and MM/DD/YYYY (bulk archive) dates."""
    if not s:
        return None
    s = s.strip()
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        pass
    try:
        return datetime.strptime(s[:10], "%m/%d/%Y").date()
    except ValueError:
        return None


def lag_days(received, sent):
    d0, d1 = parse_date(received), parse_date(sent)
    if d0 is None or d1 is None:
        return None
    return (d1 - d0).days


def at_risk(timely, response):
    if response == "In progress" or not timely:
        return None
    if timely.strip().lower() == "no" or response in AT_RISK_RESPONSES:
        return 1
    return 0


def main(db_path=DB_PATH, raw_dir=RAW):
    api_files = sorted(glob.glob(os.path.join(raw_dir, "complaints_*.csv")))
    narr_files = sorted(glob.glob(os.path.join(raw_dir, "narratives_*.csv")))
    if not api_files and not narr_files:
        raise SystemExit(f"no raw CSVs in {raw_dir}; run src/download_data.py first")
    if os.path.exists(db_path):
        os.remove(db_path)
    con = sqlite3.connect(db_path)
    con.execute("""
        CREATE TABLE complaints (
            complaint_id TEXT PRIMARY KEY,
            date_received TEXT, date_sent_to_company TEXT,
            product TEXT, sub_product TEXT, issue TEXT, sub_issue TEXT,
            company TEXT, company_response TEXT, timely TEXT,
            consumer_disputed TEXT, narrative TEXT,
            company_public_response TEXT, state TEXT, submitted_via TEXT,
            response_lag_days INTEGER, at_risk INTEGER,
            source TEXT
        )""")
    n = 0
    # API rows first (structured sample, no narratives). Archive rows are
    # loaded with OR REPLACE so a narrative row replaces any API row with
    # the same complaint_id, adding its narrative.
    for path, source, verb in ((p, "api", "IGNORE") for p in api_files):
        n += _load_file(con, path, source, verb)
    for path in narr_files:
        n += _load_file(con, path, "archive", "REPLACE")
    con.commit()
    total = con.execute("SELECT COUNT(*) FROM complaints").fetchone()[0]
    n_api = con.execute("SELECT COUNT(*) FROM complaints WHERE source='api'").fetchone()[0]
    n_narr = con.execute("SELECT COUNT(*) FROM complaints WHERE narrative != ''").fetchone()[0]
    print(f"loaded {total} complaints into {db_path} "
          f"({n_api} API structured rows, {n_narr} with narratives; rows read: {n})")
    con.close()
    return total


def _load_file(con, path, source, verb):
    n = 0
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            con.execute(
                f"INSERT OR {verb} INTO complaints VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (r.get("complaint_id"), r.get("date_received"),
                 r.get("date_sent_to_company"), r.get("product"),
                 r.get("sub_product"), r.get("issue"), r.get("sub_issue"),
                 r.get("company"), r.get("company_response"),
                 r.get("timely"), r.get("consumer_disputed"),
                 r.get("complaint_what_happened"),
                 r.get("company_public_response"), r.get("state"),
                 r.get("submitted_via"),
                 lag_days(r.get("date_received"), r.get("date_sent_to_company")),
                 at_risk(r.get("timely"), r.get("company_response")),
                 source))
            n += 1
    return n


if __name__ == "__main__":
    main()

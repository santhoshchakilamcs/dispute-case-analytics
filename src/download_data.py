#!/usr/bin/env python3
"""Download real CFPB Consumer Complaint Database data.

Two sources, both public, no key needed:

1. CFPB Complaint Search API
   https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/
   Structured complaint records (product, issue, company, response, timeliness).
   NOTE (verified 2026-11-09): the API's _source no longer includes the free-text
   narrative ("complaint_what_happened") at all, even with has_narrative=true
   (0 of 100 sampled records carried the field). CFPB stopped publishing
   narratives in September 2026 (database release 24) and moved previously
   published narratives to its FOIA Reading Room. So the API gives us structured
   rows only; narratives come from source 2 below.

2. CFPB FOIA Reading Room, "CFPB Consumer Complaint Database Narratives
   Archive" (https://www.consumerfinance.gov/foia-requests/foia-electronic-
   reading-room/cfpb-consumer-complaint-database-narratives-archive/).
   Monthly/bimonthly full-record exports (CSV inside a zip) covering complaints
   received 2011-12-01 through 2026-08-14 that were previously published with
   narratives. We pull the three Jan-Jun 2025 exports and keep only the rows
   with a non-empty narrative, capped at 20,000 per export, so the triage
   classifier trains on real consumer-written dispute text.

Outputs (data/raw/, gitignored, regenerated on first start by run.sh):
  complaints_2024_h1.csv, complaints_2024_h2.csv, complaints_2025_h1.csv,
  complaints_2025_h2.csv  -- up to 5,000 structured rows each from the API
  narratives_2025.csv    -- narrative rows from the FOIA archive exports

The API returns its first 5,000 rows per window in created_date_desc order
(most-recent-first), so each chunk is a sample of that window, NOT a census
and NOT a random sample. All volumes in this project are sample volumes.
"""
import csv
import io
import json
import os
import time
import urllib.parse
import urllib.request
import zipfile

BASE = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
os.makedirs(RAW, exist_ok=True)

CHUNKS = [
    ("complaints_2024_h1.csv", "2024-01-01", "2024-06-30"),
    ("complaints_2024_h2.csv", "2024-07-01", "2024-12-31"),
    ("complaints_2025_h1.csv", "2025-01-01", "2025-06-30"),
    ("complaints_2025_h2.csv", "2025-07-01", "2025-12-31"),
]
SIZE = 5000

FIELDS = [
    "complaint_id", "date_received", "date_sent_to_company", "product",
    "sub_product", "issue", "sub_issue", "company", "company_response",
    "timely", "consumer_disputed", "complaint_what_happened",
    "company_public_response", "state", "submitted_via",
]

# FOIA archive exports, Jan-Jun 2025. Full-record CSVs (16 columns incl.
# the narrative) inside a zip; we keep narrative rows only, capped per file.
ARCHIVE_EXPORTS = [
    ("https://files.consumerfinance.gov/f/documents/CCDB_Export_9_January_2025_through_February_2025.zip",
     "narratives_2025_jan_feb"),
    ("https://files.consumerfinance.gov/f/documents/CCDB_Export_10_March_2025_through_April_2025.zip",
     "narratives_2025_mar_apr"),
    ("https://files.consumerfinance.gov/f/documents/CCDB_Export_11_May_2025_through_June_2025.zip",
     "narratives_2025_may_jun"),
]
NARRATIVE_CAP_PER_EXPORT = 20000

# Archive CSV headers -> our field names. The archive exports carry 16
# columns (release 22, June 2026, dropped "Consumer disputed?" and
# "Consumer consent provided?"); the narrative column is called
# "Consumer complaint narrative", and dates are ISO (YYYY-MM-DD).
ARCHIVE_MAP = {
    "Complaint ID": "complaint_id",
    "Date received": "date_received",
    "Date sent to company": "date_sent_to_company",
    "Product": "product",
    "Sub-product": "sub_product",
    "Issue": "issue",
    "Sub-issue": "sub_issue",
    "Company": "company",
    "Company response to consumer": "company_response",
    "Timely response?": "timely",
    "Consumer complaint narrative": "complaint_what_happened",
    "Company public response": "company_public_response",
    "State": "state",
    "Submitted via": "submitted_via",
}


def fetch(params, tries=4):
    qs = urllib.parse.urlencode(params)
    url = BASE + "?" + qs
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "dispute-case-analytics/1.0 (personal project)"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.load(resp)
        except Exception as e:
            last = e
            print(f"  attempt {attempt + 1} failed: {e}; retrying...")
            time.sleep(5)
    raise last


def pull_chunk(fname, dmin, dmax):
    data = fetch({
        "size": SIZE,
        "sort": "created_date_desc",
        "has_narrative": "true",
        "date_received_min": dmin,
        "date_received_max": dmax,
    })
    hits = data.get("hits", {})
    total = hits.get("total")
    rows = []
    for h in hits.get("hits", []):
        src = h.get("_source", {})
        rows.append({f: src.get(f, "") for f in FIELDS})
    path = os.path.join(RAW, fname)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {path}: {len(rows)} rows (API total in window: {total})")
    return len(rows)


def pull_archive():
    """Download the FOIA archive exports and keep narrative rows only."""
    out_path = os.path.join(RAW, "narratives_2025.csv")
    kept_total = 0
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for url, tag in ARCHIVE_EXPORTS:
            print(f"downloading archive {tag} ...", flush=True)
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) "
                              "AppleWebKit/537.36 Chrome/120 Safari/537.36"})
            with urllib.request.urlopen(req, timeout=600) as resp:
                blob = resp.read()
            print(f"  got {len(blob) / 1e6:.1f} MB zip", flush=True)
            zf = zipfile.ZipFile(io.BytesIO(blob))
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            assert len(csv_names) == 1, f"expected 1 csv in zip, got {csv_names}"
            kept = 0
            seen = 0
            with zf.open(csv_names[0]) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"))
                for row in reader:
                    seen += 1
                    narrative = (row.get("Consumer complaint narrative") or "").strip()
                    if not narrative:
                        continue
                    if kept >= NARRATIVE_CAP_PER_EXPORT:
                        continue
                    mapped = {dst: (row.get(src_col) or "").strip()
                              for src_col, dst in ARCHIVE_MAP.items()}
                    w.writerow(mapped)
                    kept += 1
            print(f"  {tag}: {seen} rows scanned, {kept} narrative rows kept",
                  flush=True)
            kept_total += kept
    print(f"archive done: {kept_total} narrative rows -> {out_path}")
    return kept_total


def main():
    total = 0
    for fname, dmin, dmax in CHUNKS:
        print(f"pulling {dmin}..{dmax} ...")
        total += pull_chunk(fname, dmin, dmax)
        time.sleep(2)
    print(f"API done: {total} structured complaints across {len(CHUNKS)} chunks")
    pull_archive()
    print("done.")


if __name__ == "__main__":
    main()

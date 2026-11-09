#!/usr/bin/env bash
# Dispute Case Analytics — first start.
# Creates the venv, installs deps, downloads the real CFPB data (public API
# + FOIA Narratives Archive, no key), builds the SQLite warehouse, runs the
# pipeline + tests + verify, then serves the dashboard at
# http://127.0.0.1:8001/ . Data files are gitignored and re-downloaded on
# first start; nothing is pushed to GitHub but code.
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
.venv/bin/pip install -q -r requirements.txt
if [ ! -f data/raw/narratives_2025.csv ] || [ ! -f data/raw/complaints_2024_h1.csv ]; then
  .venv/bin/python src/download_data.py
fi
if [ ! -f data/disputes.db ]; then
  .venv/bin/python src/build_db.py
fi
.venv/bin/python src/pipeline.py
.venv/bin/python -m pytest tests/ -q
.venv/bin/python verify.py
.venv/bin/python app.py

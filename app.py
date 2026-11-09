#!/usr/bin/env python3
"""Serve the dashboard: stdlib http.server only, no extra dependencies.

  python3 app.py            -> http://127.0.0.1:8001/ (dashboard)
  http://127.0.0.1:8001/metrics        -> reports/metrics.json
  http://127.0.0.1:8001/api/brief/<complaint_id>  -> JSON case brief
  http://127.0.0.1:8001/api/complaints?limit=20&issue=... -> JSON list

The case-brief builder (TF-IDF over narratives + triage risk score) is
fitted lazily on the first /api/brief call and cached; that first call
can take ~30 seconds on the full sample.
"""
import http.server
import json
import os
import socketserver
import sys
import urllib.parse

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, "reports")
DB_PATH = os.environ.get("DISPUTE_DB_PATH",
                         os.path.join(ROOT, "data", "disputes.db"))
sys.path.insert(0, os.path.join(ROOT, "src"))

_CACHE = {}


def get_builder():
    if "builder" not in _CACHE:
        import case_brief as cb
        import triage as tg
        df = cb.load_frame(DB_PATH)
        builder = cb.BriefBuilder(df)
        artifacts = tg.fit_model(df)
        risk_by_id = {}
        import pandas as pd
        X, _, _, _ = tg.build_features(
            df, artifacts["vectorizer"], artifacts["encoder"],
            artifacts["top_companies"])
        scores = artifacts["model"].predict_proba(X)[:, 1]
        risk_by_id = dict(zip(df["complaint_id"].astype(str), scores))
        builder.set_risk_fn(lambda row: risk_by_id.get(str(row["complaint_id"])))
        _CACHE["builder"] = builder
    return _CACHE["builder"]


class Handler(http.server.SimpleHTTPRequestHandler):
    def _json(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.path = "/dashboard.html"
            self.directory = REPORTS
            return super().do_GET()
        if parsed.path == "/metrics":
            with open(os.path.join(REPORTS, "metrics.json"), "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path.startswith("/api/brief/"):
            cid = parsed.path[len("/api/brief/"):]
            brief = get_builder().brief(cid)
            if brief is None:
                return self._json({"error": "complaint not found"}, 404)
            return self._json(brief)
        if parsed.path == "/api/complaints":
            import sqlite3
            q = urllib.parse.parse_qs(parsed.query)
            limit = min(int(q.get("limit", ["20"])[0]), 100)
            issue = q.get("issue", [""])[0]
            con = sqlite3.connect(DB_PATH)
            con.row_factory = sqlite3.Row
            if issue:
                rows = con.execute(
                    "SELECT complaint_id, date_received, product, issue, "
                    "company, company_response, timely FROM complaints "
                    "WHERE issue LIKE ? ORDER BY date_received DESC LIMIT ?",
                    (f"%{issue}%", limit)).fetchall()
            else:
                rows = con.execute(
                    "SELECT complaint_id, date_received, product, issue, "
                    "company, company_response, timely FROM complaints "
                    "ORDER BY date_received DESC LIMIT ?",
                    (limit,)).fetchall()
            con.close()
            return self._json([dict(r) for r in rows])
        self.send_response(404)
        self.end_headers()

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8001"))
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
        print(f"serving dashboard at http://127.0.0.1:{port}/  (Ctrl-C to stop)")
        httpd.serve_forever()

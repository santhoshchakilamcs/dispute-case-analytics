# Dispute Case Analytics

A personal project that does the core of a dispute-risk data scientist's job
on real public data: triage incoming dispute cases by escalation risk,
measure how fast cases move, track what is coming in, and synthesize the
evidence on one case into a brief a human analyst can act on.

Built to cover the gaps in the Chime Data Scientist, Dispute Risk job
description (Chicago, $109k–$150k, Greenhouse 8904126002): dispute case
investigation / case-review workflow, agent SOPs and handling time, and
payment-dispute typologies (chargebacks, reversals, credits, recovery).

## What it is, plainly

- **Personal project on public CFPB data.** Not Chime data. Not production
  dispute handling. Not dispute-investigation job experience.
- **Handling time here is CFPB response timeliness** (days from receipt to
  the complaint being sent to the company), not contact-center average
  handle time. No AHT from a real contact center is claimed anywhere.
- **Suggested next steps are rules, not decisions.** A human reviews every
  case. The triage model ranks cases; it does not decide them.

## Data — real, two public sources, no key needed

**Source 1: CFPB Complaint Search API**
(consumerfinance.gov/data-research/consumer-complaints/search/api/v1/).
20,000 structured complaints, four half-year chunks 2024–2025, 5,000 each,
most-recent-first per chunk. Date range 2024-06-29..2025-12-31.

**Source 2: CFPB FOIA Reading Room, "CFPB Consumer Complaint Database
Narratives Archive"**
(consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/).
60,000 complaints with real consumer-written narratives, from the three
bimonthly full-record exports Jan–Jun 2025, capped at 20,000 narrative rows
per export. Date range 2025-01-01..2025-06-25.

**Why two sources:** the CFPB API no longer returns narrative text at all —
verified 2026-11-09, 0 of 100 sampled API records carried
`complaint_what_happened`, even with `has_narrative=true`. CFPB stopped
publishing narratives in September 2026 (database release 24) and moved
previously published narratives to the FOIA Reading Room, which covers
complaints received through August 14, 2026 only. So structured rows come
from the live API and narrative text comes from the archived exports. Both
are CFPB's own public files; nothing is synthetic and nothing is invented.

**Sampling caveats (read before quoting any number):**

- All volumes are sample volumes, not total CFPB volumes. The API chunks
  are most-recent-first, so each chunk clusters on a few dates near the
  window end: the single busiest day holds 25.0% of the 20,000-row API
  sample. Monthly volumes therefore spike on those days and are NOT a real
  monthly trend.
- The API sample is dominated by credit reporting: 83.8% of the 20,000
  rows are "Credit reporting or other personal consumer reports", and the
  three bureaus (TransUnion 27.3%, Equifax 27.0%, Experian 24.9%) are
  79.1% of complaints. Handling-time and trend numbers describe that
  sample, not the whole complaint population.
- CFPB removed the consumer-disputed field from exports in June 2026, so
  the disputed rate is reported as null, not 0%.
- Complaints are unverified consumer allegations. CFPB says so itself;
  publication is not a finding that a company did anything wrong.

## What it does

**1. Dispute triage classifier** (`src/triage.py`).
Predicts `at_risk` — timely == "No" OR company response "Closed without
relief" / "Untimely response" — from the narrative text (TF-IDF, 1–2 grams,
top 3,000 terms) plus product, issue, and top-30 companies. "In progress"
rows are excluded (outcome unknown). Time-based split: train on complaints
received 2025-01-01..2025-04-30 (n = 40,000), test on 2025-05-01..2025-06-25
(n = 20,000). No future information leaks into training.

| Model | Accuracy | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic regression (balanced) | 0.9586 | 0.9726 | 0.2171 | 0.7055 | 0.3320 |
| Most-frequent-class baseline | 0.9854 | 0.5000 | 0.0000 | 0.0000 | 0.0000 |

Base rate of at-risk in test: 1.46% (292 of 20,000). Confusion matrix:
TN 18,965, FP 743, FN 86, TP 206. Read the table honestly: the baseline is
*more accurate* because the positive class is rare — always predicting
"safe" is right 98.5% of the time and catches nothing. The model trades
accuracy for recall: it catches 70.5% of at-risk cases, and about 1 in 5
of its flags is a true at-risk case (precision 0.2171). For a triage queue
that is the right trade; for a decision it would not be.

Words most associated with at-risk: services, mail, collection, keep,
service, "this company", "of xxxx", "the last", only, owed. Words most
associated with safe resolution: however, financial, did, bill, "in my",
"xxxx and", is, again, "charge off", "asked for".

Leakage check (ablation on the same split): text-only ROC-AUC 0.9029,
categorical-only (product/issue/company) 0.9741, combined 0.9726. The
signal is mostly that at-risk cases concentrate in specific products and
companies — which is legitimate triage signal, known at intake — not
leaked outcome text. Narratives are written by consumers before the
company responds.

**2. Handling-time analysis** (`src/handling_time.py`), in the AHT spirit.
Response lag (receipt to sent-to-company): median 0 days, mean 0.4 days,
p75 0, p90 0, max 132 days; 96.6% of complaints are sent to the company
the same day. Timely rate 99.49% over 20,000 complaints. The honest
finding is that CFPB routing is near-instant, so the interesting variation
is in *who* is slow, not how long the queue is: student loans are the
slowest product (78.1% timely, n = 128) and "Dealing with your lender or
servicer" the slowest issue (84.3% timely, n = 83). An OLS model of
log1p(lag) on product + issue one-hots gets test R² 0.0489 — issue and
product type explain almost none of the lag; most of it is company
process, which the data does not describe.

**3. Loss / trend dashboards** (`src/trends.py`). Top products, issues,
companies, resolution mix, timely rate, emerging issues (last 90 days vs
prior 90 days — none met the threshold in this clustered sample).
Resolution mix in the API sample: Closed with explanation 54.8% (10,964),
Closed with non-monetary relief 43.8% (8,767), Closed with monetary relief
1.1% (219), Untimely response 0.3% (50). Top issues: Incorrect information
on your report 45.2% (9,046), Improper use of your report 22.8% (4,559),
Problem with a company's investigation into an existing problem 15.7%
(3,136).

**4. Evidence synthesis / case briefs** (`src/case_brief.py`). For one
complaint: the consumer's story (excerpt), the facts, the key terms in the
narrative, the three most similar past cases on the same issue (TF-IDF
cosine similarity, with their known outcomes), the triage risk score, and
a rule-based suggested next step. Three example briefs ship in
`reports/dashboard.html`: a highest-risk debt-management case (risk 1.00),
a Wells Fargo credit-card case closed with monetary relief (risk 0.00),
and a MOHELA student-loan case with an untimely response (risk 0.998).

## Verified numbers

`verify.py` recomputes the headline metrics straight from the raw CSVs
(not from `reports/metrics.json` and not from the SQLite DB) and
cross-checks them, including a full refit of the triage model from the raw
narrative rows with the same label rule and time split. It must match to
1e-6. Last run: **cross-check OK** — refit ROC-AUC 0.9726, accuracy
0.9586, identical to `reports/metrics.json`. Tests: **12 passed**
(`.venv/bin/python -m pytest tests/ -q`).

## How to run

```bash
./run.sh        # venv + deps + download + build + pipeline + tests + verify, then serves the dashboard
```

Or step by step:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/download_data.py   # API chunks + FOIA archive exports -> data/raw/
.venv/bin/python src/build_db.py        # -> data/disputes.db (80,000 rows)
.venv/bin/python src/pipeline.py        # -> reports/metrics.json + reports/dashboard.html
.venv/bin/python -m pytest tests/ -q
.venv/bin/python verify.py
.venv/bin/python app.py                # http://127.0.0.1:8001/
```

Endpoints: `/` dashboard, `/metrics` (metrics.json),
`/api/brief/<complaint_id>` (JSON case brief),
`/api/complaints?limit=20&issue=...` (JSON list). The brief builder fits
TF-IDF lazily on the first `/api/brief` call and caches it; that first
call can take ~30 seconds on the full narrative set.

## Files

| File | What it is |
|---|---|
| `src/download_data.py` | CFPB API pull + FOIA archive exports (narratives) |
| `src/build_db.py` | Loads both populations into `data/disputes.db` (`source` = api / archive) |
| `src/triage.py` | At-risk classifier, time split, baseline, top terms |
| `src/handling_time.py` | Response-lag distribution, timely rates, OLS lag drivers |
| `src/trends.py` | Volumes, top products/issues/companies, resolution mix, emerging issues |
| `src/case_brief.py` | Evidence synthesis: brief + similar cases + rule-based next step |
| `src/pipeline.py` | End-to-end run; writes `reports/metrics.json` + `reports/dashboard.html` |
| `app.py` | Stdlib-only HTTP server for the dashboard and the brief API |
| `verify.py` | Independent recomputation of the headline metrics from raw CSVs |
| `tests/test_dispute_analytics.py` | 12 pytest tests (label rule, lag, triage, handling, trends, briefs, app) |
| `run.sh` | First-start script: venv, deps, download, build, pipeline, tests, verify, serve |
| `requirements.txt` | pandas, numpy, scikit-learn, scipy, pytest |

Data files (`data/raw/`, `data/disputes.db`) are gitignored and
re-downloaded on first start; only code is pushed to GitHub.

## Honest limits

- Personal project on public CFPB data. Not Chime data, not production
  dispute handling, not dispute-investigation job experience.
- No AHT from a real contact center is claimed. Handling time is CFPB
  process time.
- No chargeback, reversal, or recovery-rate data exists in the CFPB
  public file, so none is claimed: dispute typologies here are CFPB
  products/issues (credit reporting, debt collection, credit card, …),
  not card-network chargeback reason codes.
- Narratives exist only through August 14, 2026 (CFPB stopped publishing
  them); the classifier trains on Jan–Jun 2025 narratives.
- The triage model is a ranking aid for a human queue, not a decision
  maker: at the 0.5 threshold it flags roughly 4–5 cases for every true
  at-risk case.

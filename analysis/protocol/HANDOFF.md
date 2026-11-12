# Handoff: where the RQ work stands (9 November 2026)

This page is for starting fresh. It lists what exists and what has been decided. It is not a plan.

## The questions

- **RQ1.** To what extent can a model conditioned on workload and environmental context predict the expected power and thermal behaviour of smart bus-stop devices across differing hardware configurations and operating conditions?
- **RQ2.** Does a hand-specified dependency graph detect energy anomalies more accurately than non-graph and learned-graph detectors? "Knowledge-based" was suggested in place of "hand-specified" but hasn't been adopted.

## Data and environment

- **Telemetry.** Prometheus data from smart bus-stop signs (Jetson Orin), 8 March – 6 October 2026: 109 devices, 12 agencies, 42 metrics. 97 devices report power. It lives in `dataset/smart-sign-telemetry/`, which is not in git, and its own README documents units and placeholder codes.
- **Hourly table.** `analysis/out/features.parquet` holds device × hour for all metrics, with counters turned into rates. It's built by `analysis/features.py`.
- **Hardware.** 13 devices run on 56 V and 84 on 12 V. 36 have a solar charge controller. The configurations are inferred from telemetry in `rq_common.hardware()`.
- **Environment.** A uv project with Python 3.12. Run scripts as `.venv/bin/python -I analysis/<script>.py`. The host has 96 cores and 503 GB of RAM. GitHub remote: `Lumiere-ngn/Power_AD`.

## What exists

- **Exploratory pipeline (committed):**
  - `profile_stream.py`
  - `cross_metric.py`
  - `features.py`
  - `spectral.py`
  - `seasonal.py`
  - `anomaly.py`
  - `influence.py`
  - two report builders
- **Pilot RQ code (uncommitted, exploratory only):**
  - `rq_common.py`, `rq1_nbm.py`, `rq2_graph.py`, `rq_report.py`
  - their outputs, `out/rq1.json` and `out/rq2.json`
  - a published report: https://claude.ai/artifact/JsGFMx2L2fnvfa6hWgWKXA
  - a 20-slide deck in `analysis/slides/`
- **Plans (uncommitted), in `analysis/protocol/`:**
  - **`PLAN_TEAM.md`** is the current plan: 4 weeks (9 Nov – 4 Dec), two people (A for data and RQ1, B for graph and RQ2), no Linux logs. Its Google Docs copy is "Plan for RQ1 and RQ2: two people, 4 weeks (v2, no Linux logs)" in the user's My Drive.
  - **`PLAN.md`** is the shorter 4-week plan. It still mentions the Linux logs.
  - **`PROTOCOL.md` and `protocol.toml`** are a detailed draft protocol, meant as a reference, not a checklist.
  - **The partner's "Implementation guide"** Google Doc holds a longer roadmap and a reading list. The user can't edit it.

## Decisions so far

- **Deadline:** about 4 December 2026. The user prefers simple plans over complete ones.
- **The pilot is exploratory.** Its numbers aren't results, because settings were chosen while looking at the August–October test period.
- **No Linux logs.** Real labels come from a shortlist of telemetry episodes that the operator confirms: 38 overheating episodes, 137 low-voltage episodes, and 16 hours where power ≠ voltage × current. Outages and low RAM are too common to use as labels.
- **The hand graph needs review.** Claude wrote it alone on 2 November, before any detector ran, and nobody who knows the hardware has reviewed it.
- **Nothing is committed.** Don't run experiments until the user says so.

## Pilot lessons worth keeping

- **RQ1:**
  - Workload drives power: input-power error drops from 1.52 W to 0.50 W.
  - Heat is environment-driven and hard to predict: within-device R² is 0.31 for CPU temperature.
  - 56 V devices are worse.
  - Calibrated 90 % bands cover about 82–90 %, and drift after July.
- **RQ2:**
  - The hand graph leads only with tail-probability scoring: 72 % recall against 69 % for the complete graph.
  - With max-|z| scoring the complete graph wins, 50 % against 44 %.
  - The hand graph misses cooling faults.
- **Pitfalls:**
  - **Thresholds and score scaling** were set on the test period. Use the calibration month instead.
  - **Injected values must also be fed in as parent inputs** of the graph detectors.
  - **Use per-device thresholds**, because a few devices' voltage shifts swamp a fleet-wide one.
  - **Score in the same process.** Forking a process pool after LightGBM has predicted deadlocks.

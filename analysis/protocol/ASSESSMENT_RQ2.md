# RQ2 repository assessment (Phase 0)

**Status:** 14 November 2026. Produced for Prompt 1 of the RQ2 implementation plan
(`RQ2_implementation.txt`), Steps 0.1–0.2: repository inspection and implementation
preparation. No code was modified. Read-only.

Scope of this file: what exists today, what is reusable, what is missing, and the
recommended implementation order for RQ2. It does not change any experiment design.

---

## 1. Existing architecture

The repository is a smart-sign telemetry analysis platform for *Propagation-Aware
Predictive Health Monitoring for Smart Mobility IoT Devices*. Data is Prometheus
telemetry from 109 smart bus-stop signs (Jetson Orin), 42 metrics, March–October
2026. Two research questions are being answered:

| Question | Method |
|---|---|
| **RQ1** — how well can context predict expected power and thermal behaviour? | LightGBM normal-behaviour model (`analysis/rq1.py`) |
| **RQ2** — does a hand-specified dependency graph detect energy anomalies better than non-graph and learned-graph detectors? | Graph-conditioned residual detectors (pilot in `analysis/rq2_graph.py`) |

```
analysis/
    features.py          # device × hour table (features.parquet), 42 metrics, counters → rates
    anomaly.py           # River Holt-Winters z-scores, HalfSpaceTrees, peer layer, incidents
    causal.py            # River Bayesian linear regression, Clark–West, BH FDR, July replication
    weather.py           # Open-Meteo city weather → weather.parquet
    rq_common.py         # shared load(), splits, feature groups, hardware/peer/weather columns
    rq1.py               # RQ1 dev/test stages (LightGBM + conformal bands)
    rq2_graph.py         # RQ2 pilot: graph learning, fault injection, detectors, bootstrap
    rq_report.py         # pilot report builder (needs rq1.json, rq2.json from the archive)
analysis/protocol/       # PROTOCOL.md, protocol.toml, plans, hand_graph/, fault spec
analysis/out/            # generated artifacts (parquet outputs are not committed)
archive/                 # earlier runs (2026-11-09 pilot; 2026-11-10 rq1 runs 1 and 2)
```

The data itself (`dataset/smart-sign-telemetry/`) and `features.parquet` are not
in git (`.gitignore`) and are not present in this checkout.

---

## 2. Relevant files and reusable functions

### `analysis/rq_common.py` — the shared data layer (most reusable)

| Function | Role |
|---|---|
| `load(columns, before)` | Reads `features.parquet`, drops `demo`/`show`/`buspas` (`EXCLUDED_AGENCIES`, D6/D7), adds `mem_used_gb`, hardware flags, peer environment (`peer_*`), joins `wx_*` weather |
| `hardware(f)` | Infers per-device `v56`, `mem_total`, `disk_total`, `has_mppt`, `has_env`, `has_ds` |
| `peer_env(f)` | Leave-one-out agency mean of enclosure sensors, same hour |
| `lgbm_params(**kw)` | Default LightGBM settings (500 trees, lr 0.05, 63 leaves, L2 1.0, n_jobs 8) |
| `clean(o)` | JSON-safe copy (NaN/inf → None, numpy scalars → Python) |
| Feature groups | `TIME`, `HW`, `ENV_EXT`, `ENV_LOCAL`, `WORKLOAD`, `WEATHER`, `device` |
| Constants | `CAL = 2026-07-01`, `TEST = 2026-08-01`; `POWER`, `THERMAL`, `TARGETS` |
| `LABEL`, `UNIT` | Display labels and units for the six target metrics |

### `analysis/rq1.py` — RQ1 pipeline (pattern to follow)

- Two stages: `dev` (load nothing from 1 August; choose inputs and settings on June,
  fit March–May; recheck on July, fit March–June; choose band calibration on July)
  and `test` (locked settings; fit March–June, calibrate July, score 1 August – 6
  October once; refuses to run a second time).
- Helpers worth reusing or mirroring: `prepare(before)`, `signs_with(*masks)`
  (device eligibility), `baseline()` (hour-of-day median), `score()` / `change()`
  (MAE, skill, paired device bootstrap), `band_stats()`, `margins()`, `variants()`.
- Test outputs: `rq1_test.json`, `rq1_test_predictions.parquet` (target, device,
  time, y, pred, baseline, lo, hi, flagged) — the shared residual framework for
  RQ2 can start from these predictions.

### `analysis/rq2_graph.py` — RQ2 pilot (closest existing prototype)

- 10 nodes (`NODES`), hand graph `HAND` (10 edges), `HAND_WHY`, context `CONTEXT`.
- `learn_graphs(Z, Zlag1, Zlag2)` — `none`, `hand`, `full`, `corr top-k`, `lasso`,
  `granger`, `notears`.
- `fit_model(key)` — per-device ridge on graph parents + global LightGBM on
  context and parents.
- `predict_matrix()`, `residual_z()`, `score_all()` — graph-conditioned residuals.
- `notears_linear()`, `within_slope()` — reusable structure learning and
  propagation-coefficient estimation.
- Faults F1–F6 in the pilot (F1–F9 in the protocol), scoring per detector per
  variant (max/tail × 1h/6h), device bootstrap.
- **Not a result:** the pilot chose settings while looking at August–October and
  set thresholds on the test period (see `PROTOCOL.md` and the deviation log).

### `analysis/anomaly.py` — online anomaly detection

River Holt-Winters per sign and metric, robust per-hour-of-day scale, spikes and
sustained flags, HalfSpaceTrees joint score, shared/device peer scope, incidents.
Outputs `anomalies.parquet`, `joint.parquet`, `incidents.parquet`,
`anomaly_summary.json`. This is the source of the `flag_*` columns used to clean
RQ1 training data.

### `analysis/causal.py` — online causal discovery

Bayesian linear regression with 2 lags, leave-one-cause-out, Clark–West,
Benjamini–Hochberg, shifted-cause null, July replication, same-hour partial
correlations. Output `causal.json`. Per the protocol, causal links are not used
to check or edit the hand graph.

### Protocol layer

- `protocol.toml` — machine-readable settings (authoritative after the freeze):
  splits, feature groups, RQ1/RQ2 parameters, thresholds (FAR = 1 % per device),
  fault classes F1–F9, bootstrap (2000 device resamples), Holm multiplicity,
  open decisions D1–D5.
- `PROTOCOL.md` — human-readable rationale, pilot lessons, deviation log.
- `PLAN_RQ1.md`, `PLAN_SIMPLE.md`, `PLAN_TEAM.md`, `HANDOFF.md`,
  `RESULTS_RQ1.md` — plans, status, and results.
- `hand_graph/` — draft hand graph (`edges.csv`, 10 edges; `nodes.csv`,
  `QUESTIONS_BUSPAS.md`, `sources.json`), pending hardware review (D3).
- Fault injector spec (`faults.md`) is open; it must be written by someone other
  than the hand-graph author (D3).

---

## 3. Data schemas and metric names

### `features.parquet` (device × hour)

Lead columns: `device` (category), `agency` (category), `time` (UTC), `has_data`,
`n_samples`, `hour_sin`, `hour_cos`, `dow`, `is_weekend`, `days_since_first`.
Hardware: `v56`, `mem_total`, `disk_total`, `has_mppt`, `has_env`, `has_ds`,
`agency`. Peer environment: `peer_temperature`, `peer_humidity`,
`peer_lightsensor`. Weather (`wx_*`): temperature, humidity, dew point, cloud,
wind, solar, precipitation. Workload: `deepstream_app_running`, `mem_used_gb`,
`wwan_rx_gb_rate`, `wwan_tx_gb_rate`, `pir_count_rate`,
`ds_consumer_error_total_rate`. Counters are reset-aware hourly increases
(`*_rate`); placeholder codes are excluded (e.g. GPU −256 °C, MPPT exact-100).

### RQ1 targets (`protocol.toml [rq1]`)

Power: `input_power_watts`, `orin_power_watts`, `cpu_gpu_power_watts`.
Thermal: `cpu_temp_celsius`, `gpu_temp_celsius`, `board_temp_celsius`.

### RQ2 nodes (`protocol.toml [rq2]`)

`cpu_gpu_power_watts`, `orin_power_watts`, `power` (heartbeat),
`input_power_watts`, `input_voltage_volts`, `input_current_amperes`,
`input_energy_wh_total_rate`, `cpu_temp_celsius`, `gpu_temp_celsius`,
`board_temp_celsius`.

### RQ2 faults (`protocol.toml [rq2.faults]`)

F1 parasitic load (input power, 5/10/20 %), F2 compute runaway (CPU+GPU power,
25/50/100 %), F3 energy-counter drift (ramp 5/10/20 %), F4 stuck power reading
(6–48 h), F5 supply sag (−3/−6/−12 % voltage), F6 cooling loss (+3/6/12 °C),
F7 sensor drift (1/2/4 residual SDs), F8 intermittent spikes (4/6/8 SD),
F9 single-node bias without propagation (1/2/4 residual SDs).

### Metric names used across experiments

`skill = 1 − MAE_model / MAE_baseline`, `nmae = MAE / device mean`, event recall,
precision–recall and ROC areas, detection delay, Top-1 / Top-3 localisation,
false alarms per device-week, 90 % band coverage / width / weighted interval score.

---

## 4. Existing train / calibration / test splits

From `protocol.toml [splits.development]` (authoritative):

| Period | Fit | Calibrate | Evaluate / test |
|---|---|---|---|
| Development (free use) | 8 Mar – 30 Jun 2026 | July 2026 | 1 Aug – 6 Oct 2026 |
| Confirmatory (once) | 8 Mar – 31 Aug 2026 | 1 Sep – 6 Oct 2026 | 7 Oct – end of fresh export |

RQ1 windows (`[splits.rq1]`): select = fit Mar–May, check June; recheck = fit
Mar–Jun, check July; bands select = fit Mar–May, calibrate June, check July;
final = fit Mar–Jun, calibrate July, test Aug–Oct.

RQ2 constraints (decision D1, settled 10 November 2026): no fresh export; August
to 6 October are reused as the test and scored once; every setting must be
chosen on March–July only; no tuning on the test; thresholds from the July
calibration window only; development fault episodes go in August and test
episodes in September–October, with the seed drawn at the freeze.

Note: `PLAN_TEAM.md` and `PLAN_RQ1.md` use June as a selection month inside the
fit window; the test period itself is the same (1 August – 6 October).

---

## 5. Data leakage and methodological issues

Issues found in the pilot, with the protocol-level fix for each:

| Issue | Fix in the protocol |
|---|---|
| Settings tuned while looking at August–October | D1: all settings chosen March–July; test scored once; pilot exposure disclosed |
| Thresholds and score scaling set on the test period | Thresholds from the July calibration window; full max/tail × 1h/6h × floor grid reported |
| Injected faults spread along the hand-graph paths | `faults.md` written independently of the graph author (D3); faults off graph paths (F6, F9) |
| Aggregation (max vs tail) chosen after results | Each detector's grid point chosen on development episodes; full grid reported |
| 90 % bands calibrated on July drift afterwards | Report drift; offer per-configuration, per-device and adaptive bands |
| Feature-group credit depended on order | Exact group Shapley over all 16 subsets |
| Per-device vs fleet thresholds | Both reported; per-device at the (1 − FAR) quantile |
| LightGBM fork deadlock after predictions | Score in a single process, or pre-predict before forking |

Remaining open decisions: D2 (RQ1 usefulness thresholds), D3 (hardware review of
the graph and the fault-injector author), D5 (confirmatory RQ2 now vs after the
Linux-log labels).

---

## 6. Missing components for RQ2

The implementation plan (Phases 1–13) requires the following, none of which exist
today as committed code or output:

| Phase | Deliverable |
|---|---|
| 1 — dataset and split | `rq2_dataset.parquet`, `device_inventory.csv`, `data_availability.csv`, `data_split_manifest.json` |
| 2 — graph | `configs/graph.yaml`, `configs/graph_review.csv`, graph visualisation (after D3) |
| 3 — residual framework | Shared residuals from RQ1 predictions, standardised per node per device on July scales |
| 4 — non-graph baseline | Working detector, saved predictions, context-only |
| 5 — engineering graph detector | Parent-conditioned node models, graph-conditioned residuals, node scores, graph score, localisation |
| 6 — learned graph (GDN) | Optional learned-graph detector and adjacency visualisation |
| 7 — fault injection | `injection_manifest.csv`, development and evaluation fault parquets, separate seeds |
| 8 — thresholds and freeze | Per-device thresholds from July calibration, alert persistence rules, frozen configuration |
| 9 — evaluation | Metrics by fault / device / configuration, FAR metrics, localisation, device bootstrap CIs |
| 10 — controls | 30 random DAGs, edge ablations |
| 11 — case studies | Diagnostic reports and figures for selected episodes |
| 12 — report | `rq2_report.html`, `rq2_summary.json` |

Existing assets that can be reused for these: `rq_common.load()` and the feature
groups; the RQ1 model and its saved predictions (`rq1_test_predictions.parquet`);
the anomaly flags from `anomaly.py`; the hand-graph draft in `hand_graph/`; the
structure-learning helpers in `rq2_graph.py`.

---

## 7. Recommended implementation order

| Order | Task | Completion condition |
|---|---|---|
| 1 | Repository inspection | Done — this file |
| 2 | Dataset and split preparation | Reproducible RQ2 dataset and split manifest |
| 3 | Graph specification | Reviewed graph with justified edges (after D3) |
| 4 | Shared residual framework | Validated residuals reused by all detectors |
| 5 | Non-graph baseline | Working baseline with reproducible predictions |
| 6 | Engineering graph detector | Parent-conditioned node predictions and scores |
| 7 | Detector unit tests | Graph behaviour and missing-data handling verified |
| 8 | Synthetic fault injection | Separate development and test injections |
| 9 | Threshold calibration | Frozen thresholds and alert rules |
| 10 | GDN | Optional learned-graph comparison |
| 11 | Final evaluation | Event-level results and bootstrap CIs |
| 12 | Random graph controls | Engineering graph vs random DAGs |
| 13 | Case studies and report | Reproducible figures, tables, interpretation |

If time becomes limited, the priority order in the implementation plan protects
the main comparison: shared data and residual framework → strong non-graph
detector → engineering graph detector → synthetic fault injection and evaluation
→ matched false-alarm comparison → device-level bootstrap CIs → random-graph
controls → GDN → localisation and case studies. GDN is omitted before the main
experiment is compromised.

---

## 8. Environment note for the next phase

Phase 1 needs `analysis/out/features.parquet`, which is built from
`dataset/smart-sign-telemetry/` by `analysis/features.py`. Neither is present in
this checkout. The pipeline also needs the uv environment (`uv sync`) with
`lightgbm`, `pandas`, `pyarrow`, `river`, and `scikit-learn` installed. Before
Phase 1 can run, place the dataset at `dataset/smart-sign-telemetry/` and run
`features.py` (plus `weather.py` first if weather columns are needed).
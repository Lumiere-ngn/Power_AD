# buspas — smart-sign telemetry analysis

Streaming analysis of the **smart-sign telemetry dataset** (Prometheus metrics from 109 smart bus-stop signs, 42 metrics, 274 M samples, March–October 2026) for the project *Propagation-Aware Predictive Health Monitoring for Smart Mobility IoT Devices*.

The current analysis has two parts, both computed online with [River](https://riverml.xyz):

1. **Anomaly detection**: which hours are unusual for a sign, and whether the fleet shares them.
2. **Causal relationships**: which variables lead which (conditional Granger causality, tested out of sample).

Report: `analysis/report/index.html` (open locally), published copy: https://claude.ai/artifact/9EUkYxU523BGdFcQr6J3dJ (private).

Earlier results (streaming profile, seasonal and spectral analysis, feature-influence report, pilot research-question outputs) are in `archive/2026-11-09/`; see its README.

## Setup

```bash
uv sync                       # Python 3.12, pandas 3, river, pyarrow
# place the dataset at dataset/smart-sign-telemetry/ (not in git; see dataset README)
```

The dataset is expected at `dataset/smart-sign-telemetry/prometheus/metric=*/month=*/part-000.parquet`.

## Pipeline

Run in this order; the scripts take no required arguments.

| Script | What it does | Output (in `analysis/out/`) |
|---|---|---|
| `analysis/weather.py` | Optional, needs internet. Hourly weather of each agency's city from the Open-Meteo historical API (outside temperature, humidity, dew point, cloud cover, wind, solar radiation, precipitation). | `weather.parquet`, `weather_sites.csv` |
| `analysis/features.py` | Hourly table: device × hour × all 42 metrics (counters as reset-aware hourly increases), time features, agency, and the `wx_*` weather columns when `weather.parquet` exists. | `features.parquet`, `features_columns.csv` |
| `analysis/anomaly.py` | Layer 1: River Holt-Winters forecast per sign and metric, deviation z against a running scale per hour of day; spikes and sustained deviations. Layer 2: River Half-Space Trees on all z values of a sign-hour. Layer 3: shared with peers or device-specific. Incidents and onset order. `--selftest` runs a synthetic check. | `anomalies.parquet`, `joint.parquet`, `incidents.parquet`, `anomaly_summary.json` |
| `analysis/causal.py` | Per sign, online Bayesian linear regressions on the z values (2 lags): full model against leave-one-cause-out (Clark–West test, Benjamini–Hochberg), a shifted-cause null, replication on July, same-hour partial correlations. August–October are not used. `--selftest` runs a synthetic check. | `causal.json` |
| `analysis/build_report.py` | Fills `analysis/report/main_template.html` with the two summaries; the chart kit comes from the `KIT` markers in `analysis/report/template.html`. | `analysis/report/index.html` |

```bash
for s in features anomaly causal build_report; do .venv/bin/python -I analysis/$s.py; done
```

`anomaly.py` and `causal.py` each take about 30 s on 96 cores. The large outputs (`*.parquet`) are not committed; regenerate them with the commands above.

### City weather (added 10 November 2026)

The signs carry no coordinates, so `weather.py` gives each agency one city: EXO Montreal, RTL Longueuil, TTC Toronto, Barcelona, NJT Newark, BCT Fort Lauderdale, BalHarbour Bal Harbour, LA Los Angeles. This is our reading of the agency labels and is not confirmed by BusPas. The agencies `buspas`, `demo`, `K` and `show` (18 signs) have no city, so their `wx_*` columns are empty; 87 % of the rows have weather.

```bash
.venv/bin/python -I analysis/weather.py && .venv/bin/python -I analysis/features.py
```

- The weather is city-level (grid of about 10 to 25 km). It follows the regional weather, not the shade or wind at one stop. NJT covers a whole state, so Newark is the weakest choice.
- `analysis/weather_check.py` compares the weather with the signs' own sensors on March to July (`out/weather_check.log`, `out/weather_check.csv`). For all 65 signs with a city and enough data, their own city fits their day-to-day temperature best of the six areas. The enclosure is about 3 to 4 °C warmer than outside, and the enclosure humidity does not follow the outside humidity.
- `anomaly.py` and `causal.py` do not read the `wx_*` columns. RQ1 uses them as an input group (`WEATHER` in `rq_common.py`; decision D4, settled 10 November 2026).
- Source: [Open-Meteo](https://open-meteo.com) historical weather API, data under CC BY 4.0.

## Research questions

Plans and the protocol are in `analysis/protocol/`.

**RQ1** (how well context predicts a sign's expected power and temperature) follows `analysis/protocol/PLAN_RQ1.md`; results are in `analysis/protocol/RESULTS_RQ1.md`.

| Script | What it does | Output (in `analysis/out/`) |
|---|---|---|
| `analysis/rq1.py dev` | Loads nothing from 1 August on. Chooses the inputs, the LightGBM settings and the band calibration on March to July. | `rq1_dev.json`, `rq1_locked.json` |
| `analysis/rq1.py test` | Fits on March to June, calibrates the 90 % bands on July, scores 1 August to 6 October once: the same signs later, unseen signs, unseen hardware or site, each input group dropped. Refuses to run a second time. | `rq1_test.json`, `rq1_test_predictions.parquet` |

August to October were already scored by the pilot of 2 November (`archive/2026-11-09/`). There is no fresh export, so they are reused as the test and scored once with settings chosen on March to July (decision D1).

**RQ2** is not rerun yet. `analysis/rq2_graph.py` and `analysis/rq_report.py` are the pilot scripts; `rq_report.py` needs `rq1.json` and `rq2.json`, which are in the archive. The hand-specified dependency graph in `analysis/protocol/hand_graph/` is built from documentation only; the links found by `causal.py` are not used to check or edit it.

## Data handling notes

- Placeholder codes documented in the dataset README (GPU −256 °C, battery temperature 3277x, exact 100 on MPPT fields, SoC > 100 %) are excluded.
- `device_id` changes letter case over some devices' lives; IDs are lower-cased (109 devices).
- Parquet rows are ordered by day then label set, so samples are re-sorted by time per device.
- Counters (`*_total`, `pir_count`, `wwan_*`) reset about daily; features use reset-aware increases. The first hour after a data gap holds the increase over the whole gap, so `anomaly.py` does not score it.

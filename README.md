# buspas — smart-sign telemetry analysis

Streaming statistical analysis of the **smart-sign telemetry dataset** (Prometheus metrics from ~109 smart bus-stop signs, 42 metrics, 274 M samples, March–October 2026) for the project *Propagation-Aware Predictive Health Monitoring for Smart Mobility IoT Devices*. Everything is computed with [River](https://riverml.xyz) in a single pass over the raw samples, plus pandas/pyarrow for aggregation and numpy for the FFT.

Report: `analysis/report/index.html` (open locally, or see the published copy).

## Setup

```bash
uv sync                       # Python 3.12, pandas 3, river, pyarrow
# place the dataset at dataset/smart-sign-telemetry/ (not in git; see dataset README)
```

The dataset is expected at `dataset/smart-sign-telemetry/prometheus/metric=*/month=*/part-000.parquet`.

## Pipeline

Run in this order; all scripts take no required arguments.

| Script | What it does | Output |
|---|---|---|
| `analysis/profile_stream.py` | One streaming pass per (metric, device shard): River stats, histogram sketches, gap/reset/placeholder counters, ADWIN drift, HalfSpaceTrees anomalies. ~2 min on 90 cores. | `analysis/out/profile/<metric>.json` |
| `analysis/cross_metric.py` | Hourly per-device table → streaming covariance/correlation, k-means device clusters, ADWIN regime changes on daily means, trend slopes, Holt-Winters/SNARIMAX forecast evaluation, availability. | `analysis/out/hourly.parquet`, `analysis/out/cross.json` |
| `analysis/features.py` | Modelling-ready table: device × hour × all 42 metrics (counters as reset-aware rates), time features, agency. | `analysis/out/features.parquet`, `features_columns.csv` |
| `analysis/spectral.py` | FFT periodogram per metric and device (share of variance by period, 2 h – 45 days). | `analysis/out/spectral.json` |
| `analysis/seasonal.py` | Seasonal patterns of 12 key metrics: hour-of-day profiles per month (fleet and agency), Hyndman seasonal/trend strength, calendar-season daily aggregates, weekday vs weekend. | `analysis/out/seasonal.json` |
| `analysis/anomaly.py` | Contextual anomalies (online Holt-Winters residual z-scores with a per-hour-of-day robust scale) and a peer layer that labels each flagged hour shared (propagating) or device-specific. `--selftest` runs a synthetic check. | `analysis/out/anomalies.parquet`, `analysis/out/anomaly_summary.json` |
| `analysis/build_report.py` | Condenses the outputs and renders the HTML report from `analysis/report/template.html`. | `analysis/out/report_data.json`, `analysis/report/index.html` |
| `analysis/influence.py` | Which features influence which: conditional Granger causality (VAR, BH-corrected, per device) and lead–lag cross-correlation on the deseasonalized residuals from `anomaly.py`, plus raw-data and shifted-cause null comparisons. | `analysis/out/influence.json` |
| `analysis/influence_report.py` | Separate feature-influence report; reuses the chart kit between the `KIT` markers in `template.html`. | `analysis/report/influence.html` |

```bash
for s in profile_stream cross_metric features spectral seasonal anomaly build_report influence influence_report; do .venv/bin/python -I analysis/$s.py; done
```

The large outputs (`*.parquet`, `analysis/out/profile/`) are not committed; regenerate them with the commands above.

## Data handling notes

- Placeholder codes documented in the dataset README (GPU −256 °C, battery temperature 3277x, exact 100 on MPPT fields, SoC > 100 %) are excluded from value statistics.
- `device_id` changes letter case over some devices' lives; IDs are lower-cased (109 devices).
- Parquet rows are ordered by day then label set, so samples are re-sorted by time per device before streaming.
- Counters (`*_total`, `pir_count`, `wwan_*`) reset about daily; features use reset-aware increases.

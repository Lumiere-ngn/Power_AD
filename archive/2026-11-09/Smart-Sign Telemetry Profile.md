Propagation-aware predictive health monitoring · dataset profile

# Smart-Sign Telemetry Profile

- [Overview](#overview)
- [Fleet](#fleet)
- [Data quality](#quality)
- [Metric explorer](#explorer)
- [All metrics](#glance)
- [Devices](#devices)
- [Cross-metric](#cross)
- [Seasonality](#season)
- [Regime changes](#dynamics)
- [Anomalies](#anomalies)
- [Forecastability](#forecast)
- [Methods](#methods)

## Overview

Every stored Prometheus sample was streamed once, in time order, through River estimators. Nothing was resampled before the statistics below; the hourly table used for cross-metric work is derived from the same pass.

## Fleet over time

Device reachability comes from the `up` series of the `edge` scrape job; a device counts as reporting on a day when any sample of it exists.

## Data quality

Samples are kept exactly as stored. Placeholder codes documented in the dataset README (GPU −256 °C, battery temperature 3277x, exact 100 on MPPT fields, state of charge above 100 %) were counted and excluded from value statistics. A gap is any interval between consecutive samples of one series longer than the 120 s scrape cadence.

## Metric explorer

Pick a metric to see its distribution, its daily cycle, how the fleet moved over the seven months, and every device's own statistics.

Metric

## All 42 metrics at a glance

Daily fleet mean over the whole history, one tile per metric. The value is the fleet median; click a tile to open it in the explorer.

## Devices

Each device was summarised by the median of its hourly values on the eleven metrics the whole fleet reports, standardised, and clustered with River's streaming k-means (4 clusters, 25 passes). Devices with fewer than seven days of data are unclustered.

## Cross-metric structure

Pearson correlation between hourly device means, estimated with River's streaming `EmpiricalCovariance` over all device-hours (pairwise where both metrics exist). The second chart asks the same question inside each device.

## Daily seasonality

Autocorrelation of each device's hourly series at a 24-hour lag. Values near 1 mean yesterday at this hour predicts today at this hour; the daily cycle dominates most physical signals.

## Regime changes and slow trends

ADWIN (δ = 0.002) ran over each device's *daily mean* of eight health metrics, so the day/night cycle does not trigger it; a detection is a sustained level shift. The slope is a streaming least-squares fit of daily means against time.

## Point anomalies

Per device, per metric: HalfSpaceTrees (10 trees, depth 6, one-day window) scored every raw sample on value and time-of-day; samples above the running 99.5 % score quantile are flagged. The rate is relative within each device, so a high rate means an unusually irregular series rather than an absolute fault count.

## Forecastability

Progressive validation on hourly series of the eight best-covered devices per metric: at every hour the model forecasts the next 12 hours, is scored, then learns the true value. Holt-Winters (α 0.3, β 0.05, γ 0.5, period 24) and SNARIMAX (p 3, d 1, seasonal p 1, m 24) against two baselines.

## Methods and caveats
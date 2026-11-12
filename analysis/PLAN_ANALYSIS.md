# Analysis plan: anomalies and causal links

A simplified plan of the analysis redone on 9 November 2026. The details are in the docstrings of
`anomaly.py` and `causal.py`, and in the "Methods and limits" section of `report/index.html`.

**Goal:** find the hours that are unusual for a sign, and find which variables lead which.

| Step | What it does | Script |
|---|---|---|
| 1. Prepare the data | Build one row per sign and hour, with 18 metrics. | `features.py` |
| 2. Forecast each metric | For each sign, forecast the next hour from the metric's own daily cycle. | `anomaly.py` |
| 3. Flag unusual hours | Flag an hour when the value is far from the forecast, either once (spike) or for three hours or more (sustained). | `anomaly.py` |
| 4. Score all metrics together | Give each sign-hour one joint score, to catch unusual combinations. | `anomaly.py` |
| 5. Split shared from device-specific | A flag is shared when enough peer signs flag the same metric in the same hour. | `anomaly.py` |
| 6. Group flags into incidents | Merge a sign's flagged hours that are close in time, and note which metric flagged first. | `anomaly.py` |
| 7. Find causal links | Test whether the last two hours of A improve the forecast of B, using March to June. | `causal.py` |
| 8. Check the links | Repeat the test on July, and against a time-shifted copy of the cause. | `causal.py` |
| 9. Report | Publish one page with the anomalies, the incidents and the causal graph. | `build_report.py` |

## Rules

- Everything runs online with River: each hour is scored before the model learns from it.
- August to October are not used for the causal links.
- The causal links are not compared with the hand-specified graph and are not used to edit it.

## Status

All nine steps are done (9 November 2026). The plan for RQ1 and RQ2 is in `protocol/PLAN_SIMPLE.md`, and the RQ1 steps are in `protocol/PLAN_RQ1.md`.

Added on 10 November 2026: `weather.py` adds the hourly weather of each agency's city to the table of
step 1 (columns `wx_*`). Steps 2 to 9 do not use these columns, so their results are unchanged.

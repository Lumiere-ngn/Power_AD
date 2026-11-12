# Plan for RQ1 and RQ2, 9 November to 4 December 2026

A simplified version of `PLAN_TEAM.md`. Person A does the data and RQ1; person B does the graph and RQ2.

| Week | A: data and RQ1 | B: graph and RQ2 |
|---|---|---|
| 1 (9–15 Nov) | Set the training, calibration and test periods. Build the RQ1 model of normal power and temperature. | Draft the node list and the edge table from the documentation. Send them to BusPas. |
| 2 (16–22 Nov) | Calibrate the prediction bands on July. Hand the model to B. | Freeze the graph after BusPas reviews it. Build the detectors (no graph, learned graph, hand graph) and the fault injection. |
| 3 (23–29 Nov) | Run the test once. Measure RQ1 error per device and per operating condition. | Run the test once. Compare the RQ2 detection results with the baselines. |
| 4 (30 Nov–4 Dec) | Write-up | Write-up |

## Questions for BusPas (week 1)

- Review the graph.
- Which rail each power metric reads.
- Confirm the shortlisted episodes, so we have real labels.

## Rules

- Nothing is tuned on August–October. The test runs only once, in week 3.
- The hand graph is frozen before week 3. Any later change gets a version number.
- Anomaly thresholds are set per device, from July.

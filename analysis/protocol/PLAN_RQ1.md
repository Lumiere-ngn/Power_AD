# Plan for RQ1

**Status:** 10 November 2026. Steps 1 to 6 are run by `analysis/rq1.py`; results are in [`RESULTS_RQ1.md`](RESULTS_RQ1.md). The run was redone twice the same day, first without the `demo` and `show` units (D6), then also without the `buspas` units (D7); the earlier runs are in `archive/2026-11-10-rq1-run1/` and `archive/2026-11-10-rq1-run2/`.

**RQ1.** To what extent can a model conditioned on workload and environmental context predict the expected power and thermal behaviour of smart bus-stop devices across differing hardware configurations and operating conditions?

## Decisions behind this plan

- **D1, settled 10 November.** There is no fresh export. August to 6 October are reused as the test and scored once. Every setting is chosen on March to July. The pilot of 2 November had already scored these months, and the write-up says so.
- **D4, settled 10 November.** City weather is an input group.
- **D6, settled 10 November.** The `demo` and `show` units (7 devices) are left out of training and scoring, because they are not installed at a stop. This was decided after the first test results had been read, so the test months were scored a second time. It is a deviation from "scored once" and is in the deviation log of [`PROTOCOL.md`](PROTOCOL.md).
- **D7, settled 10 November.** The `buspas` units (10 devices) are also left out, at the user's request after the second run had been read. The test months were scored a third time; this is also in the deviation log.
- **Still open.** The "useful" thresholds (D2); whether the enclosure temperature counts as environment for the thermal targets; BusPas's confirmation of the agency-to-city table.

## Steps

| # | Step | What it means |
|---|---|---|
| 1 | Fix targets and inputs | Targets: input, Orin and CPU+GPU power; CPU, GPU and board temperature. Inputs: time, hardware, workload, enclosure sensors, city weather. Never another power or thermal reading. |
| 2 | Clean the training data | Drop placeholder codes and the training hours that `anomaly.py` flagged. Scored hours are never dropped. |
| 3 | Two models | Baseline: the sign's usual value for that hour of day. Context model: LightGBM. |
| 4 | Tune and calibrate | Choose inputs and settings on June (fit March to May), recheck on July, calibrate the 90 % bands on July. |
| 5 | Test once | 1 August to 6 October: the same signs later, unseen signs, unseen hardware or site. |
| 6 | Drop each input group | Workload, environment, weather, hardware. |
| 7 | Write up | Error against the baseline, for normal and flagged hours separately. |

Two extras: the last two hours of light, enclosure temperature, outside temperature and solar radiation are offered as inputs; signs with and without a city are reported separately.

## Windows

| Use | Fit | Calibrate | Score |
|---|---|---|---|
| Choose inputs and settings | 8 Mar – 31 May | — | June |
| Recheck, drop each group | 8 Mar – 30 Jun | — | July |
| Choose the band calibration | 8 Mar – 31 May | June | July |
| Test | 8 Mar – 30 Jun | July | 1 Aug – 6 Oct |

These are `[splits.rq1]` in [`protocol.toml`](protocol.toml).

## How to run

```bash
.venv/bin/python -I analysis/rq1.py dev     # loads nothing from 1 August on; writes out/rq1_dev.json, out/rq1_locked.json
.venv/bin/python -I analysis/rq1.py test    # refuses to run if out/rq1_test.json exists (already run three times, see D6 and D7)
```

## Not in this plan

Ridge and hierarchical models, the grey-box and physics-chain models, Shapley values, and the few-days-of-data test. They are in [`PROTOCOL.md`](PROTOCOL.md) and can be added later.

`PLAN_TEAM.md` (A2, A9) tuned on August and tested on 1 September to 6 October. This plan does not tune on August and tests on all of 1 August to 6 October.

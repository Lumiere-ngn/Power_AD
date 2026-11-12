# RQ1 results

**Status:** third run, 10 November 2026, without the `demo`, `show` and `buspas` units. The two earlier runs of the same day are kept in `archive/2026-11-10-rq1-run1/` (all devices) and `archive/2026-11-10-rq1-run2/` (without `demo` and `show`). Plan: [`PLAN_RQ1.md`](PLAN_RQ1.md). Files and commands are listed at the end.

## The question

**RQ1.** To what extent can a model conditioned on workload and environmental context predict the expected power and thermal behaviour of smart bus-stop devices across differing hardware configurations and operating conditions?

## Short answer

**To a large extent for power and temperature on the devices installed at transit agencies. Less well on devices or hardware the model has never seen, and the data no longer allows a test across supply voltages.**

| Part of the question | Answer | Main evidence | Section |
|---|---|---|---|
| How well is power predicted? | Well | Input power is off by 0.49 W, about 4 % of a device's mean. That is 73 % less error than a prediction without context. | 1 |
| How well is temperature predicted? | Well, except for one device | Off by 1.5 °C for the CPU and 1.2 °C for the board. 52 of 62 devices are within 2 °C. One device is off by 11 °C. | 1 |
| Which context is needed? | Workload for power. Workload, enclosure sensors and weather for temperature. | Without workload the power error is three to five times higher. | 2 |
| Across hardware configurations? | A new device of a known type: somewhat worse. A new sensor set or solar setup: clearly worse. A new supply voltage: cannot be tested. | A device never seen costs 17 to 24 % more error. An unseen sensor set costs up to 2.5 times the board-temperature error. Only one 56 V device is left in the test. | 3 |
| Across operating conditions? | Worse on hot hours and on unusual hours | CPU error is 2.4 °C on the hottest hours, against 1.5 °C overall. Flagged hours are 1.4 to 2.7 times further off. | 4 |
| Can the predicted ranges be trusted? | They are a little too narrow | The 90 % ranges hold 84 to 86 % of the test hours. | 5 |

## How to read the numbers

| Term | Meaning |
|---|---|
| Device | One smart bus-stop sign: the unit installed at a stop, with its own computer and sensors. |
| Error | Mean absolute error: the average distance between the predicted and the measured hourly value, in W or °C. Lower is better. |
| Baseline | A prediction that uses no context: the device's usual (median) value for that hour of day during the training months. |
| Recent baseline | The same, from the last 28 days of training only. It is closer to the test season, so it is harder to beat. |
| Skill | The share of the baseline's error that the model removes. 0 = no better than the baseline, 1 = perfect. A skill of 0.70 means 70 % less error. |
| [a, b] | 95 % interval: the range the number would likely stay in with a different set of devices (2,000 resamples of devices). |
| 90 % band | A range around each prediction that should contain the measured value 90 % of the time. |
| Flagged hour | An hour that the earlier anomaly-detection step (`anomaly.py`) marked as unusual for that reading. About 1 % of the test hours. |

## What was done

**Data.** Hourly readings of 92 devices from 8 March to 6 October 2026. Three groups of units are left out of training and scoring: the 6 `demo` units and the 1 `show` unit, because they are not installed at a stop, and the 10 `buspas` units, which carry the vendor's name instead of a transit agency's (decisions D6 and D7).

**What is predicted.** Six readings per device and hour. The typical value is the mean over all test hours.

| Reading | Typical value in the test |
|---|---|
| Input power (the whole device) | 11.3 W |
| Orin power (the computer module) | 6.7 W |
| CPU+GPU power | 1.7 W |
| CPU temperature | 39.8 °C |
| GPU temperature | 39.6 °C |
| Board temperature | 32.5 °C |

**What the model is given (the context).**

| Input group | Contents |
|---|---|
| Time | Hour of day, day of week |
| Workload | Whether the video-analytics application is running, memory in use, mobile data sent and received, motion-sensor counts, application errors |
| Environment | Temperature, humidity and light measured inside the device's enclosure, and the same from other devices of the same agency |
| Weather | Hourly weather of the device's city (temperature, humidity, cloud, wind, sun, rain). Used for the temperature readings only. |
| Hardware | Supply voltage (12 V or 56 V), solar or not, whether the device has an enclosure sensor and the video-analytics application, memory and disk size, agency |

The model is never given another power or temperature reading of the device, and it is not told which device it is predicting. It is a gradient-boosted tree model (LightGBM), one per reading: 31 leaves and 200 trees for power, 31 leaves and 500 trees for temperature.

**Which months were used for what.**

| Months | Use |
|---|---|
| 8 March – 30 June | Training. Flagged hours are left out of the training data. |
| July | Setting the width of the 90 % bands |
| 1 August – 6 October | Test. Flagged hours are kept. |

Every choice (which inputs, model size, how the bands are set) was made on March to July only, by rules fixed in advance, and locked in a file before the test was run. The main test scores the 62 devices that have at least a week of data both before July and in the test. All of them have a known city.

**Disclosure.** The test months are not unseen data, for two reasons.

- No new data could be exported, so August to October had to be reused (decision D1). An earlier pilot run on 2 November had already been scored on these months, and its results were read before the plan was written.
- This is the third time the test months are scored with this plan. The first run had all devices; the second left out the `demo` and `show` units; this one also leaves out the `buspas` units. Each decision was taken after the previous run's results had been read, and in both earlier runs the units that were then removed were among the devices with the largest temperature errors (deviation log of `PROTOCOL.md`). The section "What changed across the three runs" shows the effect.

## 1. How well does the model predict?

**Power is predicted to within about 4 % of its usual level, and temperature to within 1.2 to 1.6 °C.**

| Reading | Typical value | Model error | Error as a share of the device's mean |
|---|---|---|---|
| Input power | 11.3 W | 0.49 W [0.40, 0.59] | 4.4 % [3.6, 5.2] |
| Orin power | 6.7 W | 0.30 W [0.23, 0.38] | 4.1 % [3.3, 5.3] |
| CPU+GPU power | 1.7 W | 0.20 W [0.15, 0.25] | 11.8 % [9.7, 15.1] |
| CPU temperature | 39.8 °C | 1.55 °C [1.26, 1.96] | — |
| GPU temperature | 39.6 °C | 1.63 °C [1.35, 2.04] | — |
| Board temperature | 32.5 °C | 1.17 °C [1.01, 1.34] | — |

A share is not given for temperature, because a percentage of a value in °C has no meaning.

**The model removes 73 to 83 % of the baseline's error, and 56 to 72 % of the recent baseline's.**

| Reading | Model error | Baseline error | Skill | Recent-baseline error | Skill vs recent baseline |
|---|---|---|---|---|---|
| Input power | 0.49 W | 1.80 W | 0.73 [0.64, 0.79] | 1.11 W | 0.56 [0.38, 0.67] |
| Orin power | 0.30 W | 1.53 W | 0.81 [0.75, 0.84] | 0.93 W | 0.68 [0.52, 0.77] |
| CPU+GPU power | 0.20 W | 0.90 W | 0.78 [0.72, 0.82] | 0.56 W | 0.65 [0.49, 0.74] |
| CPU temperature | 1.55 °C | 7.30 °C | 0.79 [0.73, 0.83] | 4.12 °C | 0.63 [0.52, 0.70] |
| GPU temperature | 1.63 °C | 7.66 °C | 0.79 [0.73, 0.83] | 4.25 °C | 0.62 [0.51, 0.69] |
| Board temperature | 1.17 °C | 6.79 °C | 0.83 [0.80, 0.85] | 4.13 °C | 0.72 [0.67, 0.76] |

The first baseline is easy to beat, because power and temperature drift between spring and autumn. The recent baseline is the fairer comparison.

**One device stands out.** For CPU temperature the median device is off by 1.3 °C and 52 of 62 devices are within 2 °C. The worst device, a TTC unit, is off by 11.0 °C: its CPU averages 50.5 °C in the test, against 40 °C for all devices, in a way the inputs do not explain. The next worst is off by 3.1 °C.

## 2. Which context matters?

**Power depends on the workload. Temperature depends on workload, enclosure sensors and weather together.**

The table shows how much the test error grows when one input group is taken away. A larger number means the group matters more.

| Reading | Error with all inputs | Without workload | Without environment | Without weather | Without hardware |
|---|---|---|---|---|---|
| Input power | 0.49 W | +1.10 W | +0.01 W* | not used | +0.18 W |
| Orin power | 0.30 W | +1.14 W | +0.01 W* | not used | +0.04 W |
| CPU+GPU power | 0.20 W | +0.63 W | 0.00 W* | not used | +0.02 W* |
| CPU temperature | 1.55 °C | +0.64 °C | +0.73 °C | +0.57 °C | +0.02 °C* |
| GPU temperature | 1.63 °C | +0.79 °C | +0.71 °C | +0.60 °C | +0.07 °C* |
| Board temperature | 1.17 °C | +0.24 °C | +0.90 °C | +0.60 °C | +0.08 °C |

\* The interval includes zero, so the change may be noise.

- **Power is set by the workload.** Without it the error is three to five times higher (3.3 to 4.8). Environment adds nothing. Weather is not in the power model: on July, adding it changed the error by less than 0.02 W.
- **Temperature needs all three.** Workload matters most for the GPU, the enclosure sensors most for the board, and city weather removes a further 0.6 °C on top of the enclosure sensors.
- **Hardware matters little** once the other inputs are there, except for input power.
- **The readings of the last two hours were not needed.** Giving the temperature model the light, enclosure temperature, outside temperature and sunshine of the previous two hours raised the June skill by less than the fixed tie margin (0.005), so the simpler inputs were kept.
- **Telling the model which device it is helps temperature** (−0.31 °C for the CPU, −0.16 °C for the board) and input power a little (−0.07 W). The main model leaves it out so that it also works on new devices.

## 3. Does it hold across hardware configurations?

**Partly for a new device of a known type. Less so for a sensor set or solar setup the model has not seen. Not testable for a new supply voltage.**

### 3a. Which devices the error comes from

After the exclusions, every device in the main test has a known city, and 61 of the 62 have a 12 V supply. A split by city or by supply voltage, as in the earlier runs, no longer says anything: the one 56 V device (an NJT unit) is predicted about as well as the others (CPU 0.8 °C, input power 0.54 W).

| Group | Devices | CPU temperature error | Board temperature error | Input-power error |
|---|---|---|---|---|
| Solar | 22 | 1.14 °C | 1.12 °C | 0.37 W |
| No solar | 40 | 1.79 °C | 1.19 °C | 0.56 W |

Devices with solar draw less power, so their errors in watts are smaller, but their skill on input power is lower (0.50 against 0.77).

### 3b. Devices the model has never seen

The devices are split into five parts. Each part is predicted by a model trained on the other four, on data before July, and scored on 1 August to 6 October. This covers 74 devices, 12 of which first appear after June.

| Reading | Error | Skill | Error compared with a seen device |
|---|---|---|---|
| Input power | 0.62 W [0.52, 0.74] | 0.78 | 1.18 times |
| Orin power | 0.37 W [0.29, 0.45] | 0.83 | 1.24 times |
| CPU+GPU power | 0.24 W [0.19, 0.29] | 0.81 | 1.21 times |
| CPU temperature | 1.94 °C [1.61, 2.44] | 0.76 | 1.19 times |
| GPU temperature | 2.08 °C [1.74, 2.58] | 0.75 | 1.23 times |
| Board temperature | 1.40 °C [1.23, 1.59] | 0.81 | 1.17 times |

Skill is against the usual value of all devices for that hour of day, since a new device has no history of its own. The last column compares the same hours of the same devices when the model had seen the device in training: 1.18 means 18 % more error.

A new device costs 17 to 24 % more error. The 12 devices that only appear after June are harder on input power (0.93 W) and CPU temperature (2.6 °C).

### 3c. Hardware or sites the model has never seen

All devices that share one property are removed from training, then predicted. "Times" is the error compared with the same hours when only the device itself was unseen (3b): 1.0 or less means the missing property costs nothing extra.

| Removed from training | Devices | Input power | Times | CPU temperature | Times | Board temperature | Times |
|---|---|---|---|---|---|---|---|
| Solar | 30 | 0.78 W | 1.5 | 2.27 °C | 1.3 | 1.56 °C | 1.1 |
| No solar | 44 | 1.18 W | 1.7 | 2.01 °C | 1.0 | 1.56 °C | 1.1 |
| Enclosure sensor | 58 | 0.98 W | 1.7 | 2.82 °C | 1.5 | 2.79 °C | 2.5 |
| No enclosure sensor | 16 | 1.13 W | 1.5 | 3.32 °C | 1.7 | 3.22 °C | 1.4 |
| Site: Barcelona | 14 | 1.19 W | 1.3 | 2.50 °C | 1.2 | 2.47 °C | 1.2 |
| Site: EXO | 19 | 0.78 W | 1.0 | 2.02 °C | 1.2 | 1.28 °C | 1.1 |
| Site: NJT | 7 | 0.98 W | 0.9 | 1.93 °C | 1.0 | 1.93 °C | 1.1 |
| Site: TTC | 24 | 0.26 W | 1.0 | 1.76 °C | 0.9 | 1.11 °C | 1.2 |

- **An unseen sensor set costs up to 2.5 times the error.** The board temperature suffers most when the model has never seen a device with an enclosure sensor.
- **An unseen solar setup costs 1.5 to 1.7 times** on input power.
- **An unseen site costs little**, at most 1.3 times (Barcelona).

Three rows are left out of the table and are only in `rq1_test.json`. Only one 56 V device is scored here, so removing the 56 V devices is not a test, and removing the 12 V devices leaves almost nothing to train on. Sites with three devices (BalHarbour, RTL) are too small to read.

## 4. Does it hold across operating conditions?

**Less well on hot hours and on unusual hours. The error grows only a little over the test months.**

| Condition | Result |
|---|---|
| Over time | CPU error is 1.5 °C in August, 1.6 °C in September and 1.6 °C in the first week of October. Input-power error stays at 0.45 to 0.50 W. |
| Hot hours | On each device's hottest 10 % of hours the CPU error is 2.4 °C, and the prediction is 1.3 °C too low. |
| Overall direction | The model predicts a little low: measured values are above the prediction by 0.11 W (input power) and 0.3 °C (CPU) on average. |

**Normal and flagged hours.** Flagged hours are 1.4 to 2.7 times further from the prediction than other hours. This is expected: the model predicts the normal behaviour, and these are the hours that are not normal.

| Reading | Error, normal hours | Error, flagged hours |
|---|---|---|
| Input power | 0.49 W | 0.88 W |
| Orin power | 0.29 W | 0.61 W |
| CPU+GPU power | 0.19 W | 0.52 W |
| CPU temperature | 1.54 °C | 2.14 °C |
| GPU temperature | 1.62 °C | 2.20 °C |
| Board temperature | 1.16 °C | 1.78 °C |

Leaving flagged hours out of the training data made almost no difference: keeping them changed the test error by less than 0.03 for every reading.

## 5. Can the predicted ranges be trusted?

**Not fully: the 90 % bands hold 84 to 86 % of the test hours.**

| Reading | Share of test hours inside the band | Median width of the band |
|---|---|---|
| Input power | 85.0 % [81.2, 88.4] | 1.7 W |
| Orin power | 86.4 % [81.8, 90.1] | 0.6 W |
| CPU+GPU power | 84.8 % [80.9, 87.9] | 0.4 W |
| CPU temperature | 83.9 % [78.6, 88.6] | 4.8 °C |
| GPU temperature | 84.4 % [79.5, 88.7] | 5.2 °C |
| Board temperature | 84.7 % [81.1, 88.0] | 3.8 °C |

- The bands fall 4 to 6 points short of 90 %. Without the adjustment on July they held only 73 to 83 %.
- On July, during development, they held 86 to 90 %. One month is not enough to cover the drift of the following months.
- The bands are worse for flagged hours (59 to 75 %).

## Against the proposed thresholds

The protocol proposes these thresholds for calling the model useful (decision D2). They are not yet confirmed.

| Proposed threshold | Result | Met? |
|---|---|---|
| Input-power error ≤ 5 % of the device's mean | 4.4 % [3.6, 5.2] | Yes (the interval reaches 5.2 %) |
| CPU temperature error ≤ 2.0 °C | 1.55 °C [1.26, 1.96] | Yes |
| GPU temperature error ≤ 2.0 °C | 1.63 °C [1.35, 2.04] | Yes (the interval reaches 2.04 °C) |
| Board temperature error ≤ 2.0 °C | 1.17 °C | Yes |
| Band coverage between 85 % and 95 % | 84 to 86 % | Only for Orin power (input power 84.98 %, just under) |

## What changed across the three runs

| | First run | Second run | Third run (this one) |
|---|---|---|---|
| Units left out | none | `demo`, `show` | `demo`, `show`, `buspas` |
| Devices in the main test | 69 | 66 | 62 |
| Locked settings | power and temperature 500 trees; temperature with 1–2 h readings | same as first | power 200 trees; temperature without 1–2 h readings |

Each run chose its settings again on March to July, by the same fixed rules. The third run's choice changed because, once the `buspas` units were gone, the simpler options came within the tie margin of the best June score, and the rule then takes the simpler option.

Error of each run, on all its devices and on the 62 devices of this run:

| Reading | First run, 69 devices | First run, same 62 | Second run, 66 devices | Second run, same 62 | This run, 62 devices |
|---|---|---|---|---|---|
| Input power | 0.57 W | 0.52 W | 0.56 W | 0.53 W | 0.49 W |
| Orin power | 0.34 W | 0.32 W | 0.34 W | 0.32 W | 0.30 W |
| CPU+GPU power | 0.24 W | 0.22 W | 0.24 W | 0.22 W | 0.20 W |
| CPU temperature | 2.39 °C | 1.56 °C | 2.08 °C | 1.55 °C | 1.55 °C |
| GPU temperature | 2.51 °C | 1.68 °C | 2.19 °C | 1.65 °C | 1.63 °C |
| Board temperature | 1.29 °C | 1.15 °C | 1.25 °C | 1.14 °C | 1.17 °C |

- **The lower temperature error comes entirely from no longer scoring the removed units.** On the same 62 devices, all three runs give the same CPU temperature error (1.55 °C). The removed `demo` and `buspas` units were off by up to 25 °C on CPU temperature.
- **Power is better on the same devices too**, by about 7 to 10 % (input power 0.52–0.53 W → 0.49 W). This comes from retraining without the removed units and from the smaller power model.
- **Board temperature is slightly worse** on the same devices (1.14 → 1.17 °C).
- **The results for unseen devices (3b) are not directly comparable** between the runs, because the five parts are drawn again from a different list of devices.

The "same 62" columns were computed afterwards from the earlier runs' saved predictions. They were not part of the plan.

## Limits

- **The test months are not unseen, and they were scored three times.** See the disclosure above. The removed units were chosen after their errors had been seen, so the errors here are on the devices that were easiest to keep.
- **No test across supply voltages.** Three 56 V devices are left in the data and only one is scored. The question "across differing hardware configurations" is answered for solar, enclosure sensors and sites, not for supply voltage.
- **Weather is approximate.** It is one location per agency, and the agency-to-city table is not confirmed by BusPas. One unit (`K`) has no city; it is not in the main test.
- **Whether the `buspas` units are installed at a stop is not confirmed.** They were left out because they carry the vendor's name; if some are real stop installations, they belong in the test.
- **Hardware is treated as fixed.** Supply voltage, solar and sensors are taken from each device's whole history, as if known at installation.
- **Hour of day is in UTC**, not local time.
- **The recent baseline was added late.** The first baseline is the one named in the plan. The recent one was added on 10 November, after the July check showed how much the first one drifts, and before the first test run.

## Development results (March to July)

These are the checks made before the test. They show that the choices were not made on the test months.

| Step | Trained on | Scored on | Result |
|---|---|---|---|
| Choose inputs and model size | March – May | June (41 devices) | Mean skill 0.83 for power and 0.89 for temperature |
| Recheck | March – June | July (65 devices) | Input power 0.44 W, Orin 0.23 W, CPU+GPU 0.15 W, CPU 1.42 °C, GPU 1.51 °C, board 1.04 °C |
| Choose how to set the bands | March – May, bands set on June | July | One margin for all devices was closest to 90 % (0.022 away, against 0.023 per configuration and 0.039 per device) |

The rules were fixed in advance. Inputs and model size: the highest mean June skill, for power and for temperature separately. Bands: the share of July hours inside the band closest to 90 %. In both cases, when two options are within 0.005, the simplest wins.

The test errors are 8 to 34 % above the July errors. That is the cost of predicting one to three months further ahead.

## Files and commands

| | |
|---|---|
| Code | `analysis/rq1.py`, `analysis/rq_common.py` (at commit `93eb148` plus uncommitted changes) |
| Excluded units | `EXCLUDED_AGENCIES = ["demo", "show", "buspas"]` in `analysis/rq_common.py`; decisions D6 and D7 in [`protocol.toml`](protocol.toml) |
| Development | `.venv/bin/python -I analysis/rq1.py dev` → `analysis/out/rq1_dev.json`, `rq1_locked.json` |
| Test | `.venv/bin/python -I analysis/rq1.py test` → `analysis/out/rq1_test.json`. It refuses to run when that file exists. |
| Earlier runs | `archive/2026-11-10-rq1-run1/` (all devices) and `archive/2026-11-10-rq1-run2/` (without `demo` and `show`): results text, outputs and the scripts as they were |
| Windows | `[splits.rq1]` in [`protocol.toml`](protocol.toml) |

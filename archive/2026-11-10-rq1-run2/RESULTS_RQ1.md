# RQ1 results

**Status:** second run, 10 November 2026, without the `demo` and `show` units. The first run of the same day is kept in `archive/2026-11-10-rq1-run1/`. Plan: [`PLAN_RQ1.md`](PLAN_RQ1.md). Files and commands are listed at the end.

## The question

**RQ1.** To what extent can a model conditioned on workload and environmental context predict the expected power and thermal behaviour of smart bus-stop devices across differing hardware configurations and operating conditions?

## Short answer

**To a large extent for power, and for temperature on most devices. Not reliably on hardware the model has never seen.**

| Part of the question | Answer | Main evidence | Section |
|---|---|---|---|
| How well is power predicted? | Well | Input power is off by 0.56 W, about 5 % of a device's mean. That is 70 % less error than a prediction without context. | 1 |
| How well is temperature predicted? | Well for most devices, badly for a few | Off by 2.1 °C for the CPU and 1.2 °C for the board. 54 of 66 devices are within 2 °C. Three devices are off by more than 11 °C. | 1, 3 |
| Which context is needed? | Workload for power. Workload, enclosure sensors and weather for temperature. | Without workload the power error is about three to four times higher. | 2 |
| Across hardware configurations? | A new device of a known type: somewhat worse. A new type of hardware: no. | A device never seen costs 12 to 27 % more error. An unseen supply voltage costs 1.7 times the input-power error. | 3 |
| Across operating conditions? | Worse on hot hours, on unusual hours, and further from the training months | CPU error is 3.5 °C on the hottest hours, and rises from 1.8 °C in August to 2.8 °C in October. | 4 |
| Can the predicted ranges be trusted? | They are a little too narrow | The 90 % ranges hold 82 to 86 % of the test hours. | 5 |

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

**Data.** Hourly readings of 102 devices from 8 March to 6 October 2026. The 6 `demo` units and the 1 `show` unit are left out of training and scoring, because they are not installed at a stop.

**What is predicted.** Six readings per device and hour. The typical value is the mean over all test hours.

| Reading | Typical value in the test |
|---|---|
| Input power (the whole device) | 11.4 W |
| Orin power (the computer module) | 6.7 W |
| CPU+GPU power | 1.8 W |
| CPU temperature | 40.7 °C |
| GPU temperature | 40.4 °C |
| Board temperature | 32.7 °C |

**What the model is given (the context).**

| Input group | Contents |
|---|---|
| Time | Hour of day, day of week |
| Workload | Whether the video-analytics application is running, memory in use, mobile data sent and received, motion-sensor counts, application errors |
| Environment | Temperature, humidity and light measured inside the device's enclosure, and the same from other devices of the same agency |
| Weather | Hourly weather of the device's city (temperature, humidity, cloud, wind, sun, rain). Used for the temperature readings only. |
| Hardware | Supply voltage (12 V or 56 V), solar or not, whether the device has an enclosure sensor and the video-analytics application, memory and disk size, agency |

The model is never given another power or temperature reading of the device, and it is not told which device it is predicting. It is a gradient-boosted tree model (LightGBM, 31 leaves, 500 trees), one per reading.

**Which months were used for what.**

| Months | Use |
|---|---|
| 8 March – 30 June | Training. Flagged hours are left out of the training data. |
| July | Setting the width of the 90 % bands |
| 1 August – 6 October | Test. Flagged hours are kept. |

Every choice (which inputs, model size, how the bands are set) was made on March to July only and locked in a file before the test was run. The main test scores the 66 devices that have at least a week of data both before July and in the test.

**Disclosure.** The test months are not unseen data, for two reasons.

- No new data could be exported, so August to October had to be reused (decision D1). An earlier pilot run on 2 November had already been scored on these months, and its results were read before the plan was written.
- This is the second time the test months are scored with this plan. The first run included the `demo` and `show` units. Its results were read, including a split by agency in which the `demo` units had the largest CPU-temperature errors. The decision to leave these units out came after that (decision D6, in the deviation log of `PROTOCOL.md`). The section "What changed from the first run" shows the effect.

## 1. How well does the model predict?

**Power is predicted to within about 5 % of its usual level, and temperature to within 1.2 to 2.2 °C.**

| Reading | Typical value | Model error | Error as a share of the device's mean |
|---|---|---|---|
| Input power | 11.4 W | 0.56 W [0.47, 0.66] | 5.0 % [4.2, 5.8] |
| Orin power | 6.7 W | 0.34 W [0.27, 0.41] | 4.7 % [3.8, 5.5] |
| CPU+GPU power | 1.8 W | 0.24 W [0.19, 0.29] | 13.8 % [11.6, 16.4] |
| CPU temperature | 40.7 °C | 2.08 °C [1.48, 2.85] | — |
| GPU temperature | 40.4 °C | 2.19 °C [1.58, 2.98] | — |
| Board temperature | 32.7 °C | 1.25 °C [1.08, 1.43] | — |

A share is not given for temperature, because a percentage of a value in °C has no meaning.

**The model removes 70 to 81 % of the baseline's error, and 54 to 70 % of the recent baseline's.**

| Reading | Model error | Baseline error | Skill | Recent-baseline error | Skill vs recent baseline |
|---|---|---|---|---|---|
| Input power | 0.56 W | 1.87 W | 0.70 [0.63, 0.76] | 1.22 W | 0.54 [0.39, 0.65] |
| Orin power | 0.34 W | 1.58 W | 0.78 [0.74, 0.82] | 1.00 W | 0.66 [0.53, 0.74] |
| CPU+GPU power | 0.24 W | 0.93 W | 0.75 [0.70, 0.78] | 0.61 W | 0.61 [0.49, 0.69] |
| CPU temperature | 2.08 °C | 7.70 °C | 0.73 [0.65, 0.80] | 4.69 °C | 0.56 [0.45, 0.65] |
| GPU temperature | 2.19 °C | 8.08 °C | 0.73 [0.65, 0.80] | 4.84 °C | 0.55 [0.45, 0.64] |
| Board temperature | 1.25 °C | 6.68 °C | 0.81 [0.78, 0.84] | 4.14 °C | 0.70 [0.65, 0.74] |

The first baseline is easy to beat, because power and temperature drift between spring and autumn. The recent baseline is the fairer comparison.

**The average hides a split between devices.** For CPU temperature the median device is off by 1.3 °C and 54 of 66 devices are within 2 °C, but three devices are off by more than 11 °C. Section 3 shows which ones.

## 2. Which context matters?

**Power depends on the workload. Temperature depends on workload, enclosure sensors and weather together.**

The table shows how much the test error grows when one input group is taken away. A larger number means the group matters more.

| Reading | Error with all inputs | Without workload | Without environment | Without weather | Without hardware |
|---|---|---|---|---|---|
| Input power | 0.56 W | +1.08 W | −0.05 W | not used | +0.15 W |
| Orin power | 0.34 W | +1.14 W | 0.00 W* | not used | +0.01 W* |
| CPU+GPU power | 0.24 W | +0.63 W | +0.01 W* | not used | 0.00 W* |
| CPU temperature | 2.08 °C | +0.77 °C | +0.74 °C | +0.52 °C | +0.02 °C* |
| GPU temperature | 2.19 °C | +1.10 °C | +0.72 °C | +0.50 °C | +0.05 °C* |
| Board temperature | 1.25 °C | +0.22 °C | +0.84 °C | +0.52 °C | +0.09 °C |

\* The interval includes zero, so the change may be noise.

- **Power is set by the workload.** Without it the error is about three to four times higher (2.9 to 4.3). Environment adds nothing. Weather is not in the power model: on July, adding it changed the error by less than 0.02 W.
- **Temperature needs all three.** Workload matters most for the CPU and GPU, and the enclosure sensors most for the board. City weather removes a further 0.5 °C on top of the enclosure sensors.
- **Hardware matters little** once the other inputs are there, except for input power.
- **The readings of the last two hours add little.** The temperature model is also given the light, enclosure temperature, outside temperature and sunshine of the previous two hours. Removing them changes the error by less than 0.1 °C.
- **Telling the model which device it is helps temperature** (−0.38 °C for the CPU, −0.15 °C for the board) but not power. The main model leaves it out so that it also works on new devices.

## 3. Does it hold across hardware configurations?

**Partly for a new device of a known type. No for a type of hardware the model has not seen, and not for the few 56 V devices and the devices without a city.**

### 3a. Which devices the error comes from

Same test as section 1, split by group.

| Group | Devices | CPU temperature error | Board temperature error | Input-power error |
|---|---|---|---|---|
| Devices with a known city (weather available) | 62 | 1.55 °C | 1.14 °C | 0.53 W |
| Devices without a city (`buspas` units) | 4 | 11.1 °C | 3.07 °C | 0.98 W |
| 12 V supply | 63 | 1.81 °C | 1.19 °C | 0.54 W |
| 56 V supply | 3 | 10.1 °C | 2.99 °C | 1.05 W |

The three worst devices for CPU temperature are two `buspas` units and one TTC device, and a third `buspas` unit is off by 7.3 °C. These devices run much hotter than the rest: their CPU temperature averages 45 to 84 °C in the test, against 41 °C for all devices, in a way the inputs do not explain. For the three 56 V devices (two `buspas` units and one NJT device) the model is worse than the baseline on all three temperatures.

### 3b. Devices the model has never seen

The devices are split into five parts. Each part is predicted by a model trained on the other four, on data before July, and scored on 1 August to 6 October. This covers 79 devices, 13 of which first appear after June.

| Reading | Error | Skill | Error compared with a seen device |
|---|---|---|---|
| Input power | 0.76 W [0.63, 0.90] | 0.73 | 1.27 times |
| Orin power | 0.42 W [0.34, 0.50] | 0.81 | 1.17 times |
| CPU+GPU power | 0.28 W [0.22, 0.34] | 0.78 | 1.12 times |
| CPU temperature | 2.84 °C [2.07, 3.84] | 0.67 | 1.26 times |
| GPU temperature | 2.92 °C [2.16, 3.90] | 0.67 | 1.25 times |
| Board temperature | 1.46 °C [1.27, 1.66] | 0.80 | 1.15 times |

Skill is against the usual value of all devices for that hour of day, since a new device has no history of its own. The last column compares the same hours of the same devices when the model had seen the device in training: 1.27 means 27 % more error.

A new device costs 12 to 27 % more error on power and 15 to 26 % more on temperature. The 13 devices that only appear after June are harder (input power 1.05 W, CPU 4.2 °C).

### 3c. Hardware or sites the model has never seen

All devices that share one property are removed from training, then predicted. "Times" is the error compared with the same hours when only the device itself was unseen (3b): 1.0 or less means the missing property costs nothing extra.

| Removed from training | Devices | Input power | Times | CPU temperature | Times | Board temperature | Times |
|---|---|---|---|---|---|---|---|
| 56 V supply | 4 | 2.76 W | 1.7 | 19.09 °C | 1.4 | 6.68 °C | 2.2 |
| Solar | 31 | 0.78 W | 1.3 | 2.91 °C | 1.3 | 1.54 °C | 1.2 |
| No solar | 48 | 1.16 W | 1.4 | 3.18 °C | 1.0 | 1.86 °C | 1.2 |
| Enclosure sensor | 62 | 0.78 W | 1.0 | 4.02 °C | 1.4 | 3.15 °C | 2.5 |
| No enclosure sensor | 17 | 1.35 W | 1.7 | 3.82 °C | 1.5 | 3.55 °C | 1.7 |
| Site: Barcelona | 14 | 1.49 W | 1.6 | 2.59 °C | 1.0 | 2.66 °C | 1.2 |
| Site: EXO | 19 | 0.76 W | 0.7 | 2.01 °C | 1.1 | 1.39 °C | 1.2 |
| Site: NJT | 7 | 0.93 W | 0.7 | 1.83 °C | 1.0 | 1.78 °C | 1.1 |
| Site: TTC | 24 | 0.31 W | 1.3 | 1.90 °C | 1.1 | 1.06 °C | 1.2 |

- **An unseen supply voltage or sensor set costs up to 2.5 times the error.** The board temperature suffers most.
- **Solar or not costs 1.2 to 1.4 times** on input power and board temperature.
- **An unseen site usually costs little.** Barcelona's power is the exception.

Two rows are left out of the table and are only in `rq1_test.json`. Removing the 12 V devices leaves only the few 56 V devices to train on, so it is not a fair test. Sites with three to five devices (BalHarbour, RTL, `buspas`) are too small to read.

## 4. Does it hold across operating conditions?

**Less well on hot hours, on unusual hours, and as the test moves away from the training months.**

| Condition | Result |
|---|---|
| Over time | CPU error is 1.8 °C in August, 2.2 °C in September and 2.8 °C in the first week of October. Input-power error stays at 0.53 to 0.58 W. |
| Hot hours | On each device's hottest 10 % of hours the CPU error is 3.5 °C, and the prediction is 2.3 °C too low. |
| Overall direction | The model predicts a little low: measured values are above the prediction by 0.16 W (input power) and 0.6 °C (CPU) on average. |

**Normal and flagged hours.** Flagged hours are 1.5 to 2.7 times further from the prediction than other hours. This is expected: the model predicts the normal behaviour, and these are the hours that are not normal.

| Reading | Error, normal hours | Error, flagged hours |
|---|---|---|
| Input power | 0.56 W | 0.93 W |
| Orin power | 0.34 W | 0.71 W |
| CPU+GPU power | 0.23 W | 0.57 W |
| CPU temperature | 2.05 °C | 5.54 °C |
| GPU temperature | 2.16 °C | 5.21 °C |
| Board temperature | 1.24 °C | 1.88 °C |

Leaving flagged hours out of the training data made almost no difference: keeping them changed the test error by less than 0.02 for every reading.

## 5. Can the predicted ranges be trusted?

**Not fully: the 90 % bands hold 82 to 86 % of the test hours.**

| Reading | Share of test hours inside the band | Median width of the band |
|---|---|---|
| Input power | 84.0 % [80.2, 87.3] | 1.8 W |
| Orin power | 85.7 % [81.0, 90.1] | 0.9 W |
| CPU+GPU power | 82.0 % [76.9, 86.7] | 0.5 W |
| CPU temperature | 82.4 % [77.0, 87.1] | 5.3 °C |
| GPU temperature | 83.4 % [78.2, 87.8] | 5.8 °C |
| Board temperature | 85.7 % [82.3, 88.8] | 4.1 °C |

- The bands fall 4 to 8 points short of 90 %. Without the adjustment on July they held only 72 to 78 %.
- The same shortfall already showed on July during development (84 to 88 %). One month is not enough to cover the drift of the following months.
- The bands are worse for flagged hours (58 to 74 %). For the three 56 V devices they are worse on Orin power, CPU+GPU power and board temperature (65 to 76 %).

## Against the proposed thresholds

The protocol proposes these thresholds for calling the model useful (decision D2). They are not yet confirmed.

| Proposed threshold | Result | Met? |
|---|---|---|
| Input-power error ≤ 5 % of the device's mean | 5.0 % [4.2, 5.8] | Borderline (4.99 %) |
| CPU temperature error ≤ 2.0 °C | 2.08 °C [1.48, 2.85] | No, narrowly (1.55 °C for devices with a city) |
| GPU temperature error ≤ 2.0 °C | 2.19 °C [1.58, 2.98] | No (1.65 °C for devices with a city) |
| Board temperature error ≤ 2.0 °C | 1.25 °C | Yes |
| Band coverage between 85 % and 95 % | 82 to 86 % | Only for Orin power and board temperature |

## What changed from the first run

The first run scored 69 devices, three of them `demo` units. This run scores the other 66.

| Reading | First run, 69 devices | First run, the same 66 devices | This run, 66 devices |
|---|---|---|---|
| Input power | 0.57 W | 0.55 W | 0.56 W |
| Orin power | 0.34 W | 0.34 W | 0.34 W |
| CPU+GPU power | 0.24 W | 0.24 W | 0.24 W |
| CPU temperature | 2.39 °C | 2.01 °C | 2.08 °C |
| GPU temperature | 2.51 °C | 2.13 °C | 2.19 °C |
| Board temperature | 1.29 °C | 1.24 °C | 1.25 °C |

- **The lower temperature error comes from no longer scoring the three `demo` units.** In the first run they were off by 7 to 22 °C on CPU temperature, which put them among the six hardest of the 69 devices.
- **Retraining without the `demo` and `show` units did not help the other devices.** On the same 66 devices the error is the same or slightly higher (by 0.06 °C for the CPU). These units were about 3 % of the training hours.
- **The locked settings did not change.** The same inputs, model size and band rule were chosen again on March to July.
- **The results for unseen devices (3b) are not directly comparable** between the runs, because the five parts are drawn again from a different list of devices.

The middle column was computed afterwards from the first run's saved predictions. It was not part of the plan.

## Limits

- **The test months are not unseen, and they were scored twice.** See the disclosure above.
- **Few unusual devices.** Three or four 56 V devices and four devices without a city are too few to say why they behave differently.
- **Weather is approximate.** It is one location per agency, and the agency-to-city table is not confirmed by BusPas. 11 devices have no city (the `buspas` and `K` units).
- **Hardware is treated as fixed.** Supply voltage, solar and sensors are taken from each device's whole history, as if known at installation.
- **Hour of day is in UTC**, not local time.
- **The recent baseline was added late.** The first baseline is the one named in the plan. The recent one was added on 10 November, after the July check showed how much the first one drifts, and before the first test run.

## Development results (March to July)

These are the checks made before the test. They show that the choices were not made on the test months.

| Step | Trained on | Scored on | Result |
|---|---|---|---|
| Choose inputs and model size | March – May | June (44 devices) | Mean skill 0.82 for power and 0.88 for temperature |
| Recheck | March – June | July (67 devices) | Input power 0.47 W, Orin 0.25 W, CPU+GPU 0.16 W, CPU 1.46 °C, GPU 1.59 °C, board 1.09 °C |
| Choose how to set the bands | March – May, bands set on June | July | One margin for all devices was closer to 90 % than a margin per configuration or per device |

The rules were fixed in advance. Inputs and model size: the highest mean June skill, for power and for temperature separately. Bands: the share of July hours inside the band closest to 90 %. In both cases, when two options are within 0.005, the simplest wins.

The test errors are 14 to 48 % above the July errors. That is the cost of predicting one to three months further ahead.

## Files and commands

| | |
|---|---|
| Code | `analysis/rq1.py`, `analysis/rq_common.py` (at commit `93eb148` plus uncommitted changes) |
| Excluded units | `EXCLUDED_AGENCIES = ["demo", "show"]` in `analysis/rq_common.py`; decision D6 in [`protocol.toml`](protocol.toml) |
| Development | `.venv/bin/python -I analysis/rq1.py dev` → `analysis/out/rq1_dev.json`, `rq1_locked.json` |
| Test | `.venv/bin/python -I analysis/rq1.py test` → `analysis/out/rq1_test.json`. It refuses to run when that file exists. |
| First run | `archive/2026-11-10-rq1-run1/` (results text, outputs and the scripts as they were) |
| Windows | `[splits.rq1]` in [`protocol.toml`](protocol.toml) |

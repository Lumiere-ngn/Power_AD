# RQ1 results

**Status:** final test, run once on 10 November 2026. Plan: [`PLAN_RQ1.md`](PLAN_RQ1.md). Files and commands are listed at the end.

## The question

**RQ1.** To what extent can a model conditioned on workload and environmental context predict the expected power and thermal behaviour of smart bus-stop devices across differing hardware configurations and operating conditions?

## Short answer

**To a large extent for power, and for temperature on most devices. Not reliably on hardware the model has never seen.**

| Part of the question | Answer | Main evidence | Section |
|---|---|---|---|
| How well is power predicted? | Well | Input power is off by 0.57 W, about 5 % of a device's mean. That is 70 % less error than a prediction without context. | 1 |
| How well is temperature predicted? | Well for most devices, badly for a few | Off by 2.4 °C for the CPU and 1.3 °C for the board. 54 of 69 devices are within 2 °C. Five devices are off by more than 11 °C. | 1, 3 |
| Which context is needed? | Workload for power. Workload, enclosure sensors and weather for temperature. | Without workload the power error is three to four and a half times higher. | 2 |
| Across hardware configurations? | A new device of a known type: almost as well. A new type of hardware: no. | A device never seen costs 6 to 24 % more error. An unseen supply voltage costs 1.6 times the input-power error. | 3 |
| Across operating conditions? | Worse on hot hours, on unusual hours, and further from the training months | CPU error is 4.5 °C on the hottest hours, and rises from 2.1 °C in August to 4.0 °C in October. | 4 |
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

**Data.** Hourly readings of the devices from 8 March to 6 October 2026.

**What is predicted.** Six readings per device and hour. The typical value is the mean over all test hours.

| Reading | Typical value in the test |
|---|---|
| Input power (the whole device) | 11.4 W |
| Orin power (the computer module) | 6.7 W |
| CPU+GPU power | 1.8 W |
| CPU temperature | 41.1 °C |
| GPU temperature | 40.9 °C |
| Board temperature | 32.9 °C |

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
| 1 August – 6 October | Test. Scored once. Flagged hours are kept. |

Every choice (which inputs, model size, how the bands are set) was made on March to July only and locked in a file before the test was run. The main test scores the 69 devices that have at least a week of data both before July and in the test.

**Disclosure.** No new data could be exported, so August to October had to be reused (decision D1). An earlier pilot run on 2 November had already been scored on these months, and its results were read before this plan was written. The months were used once here, with every choice made on earlier months, but they are not truly unseen data.

## 1. How well does the model predict?

**Power is predicted to within about 5 % of its usual level, and temperature to within 1.3 to 2.5 °C.**

| Reading | Typical value | Model error | Error as a share of the device's mean |
|---|---|---|---|
| Input power | 11.4 W | 0.57 W [0.47, 0.67] | 5.1 % [4.3, 6.0] |
| Orin power | 6.7 W | 0.34 W [0.27, 0.42] | 4.7 % [3.8, 5.8] |
| CPU+GPU power | 1.8 W | 0.24 W [0.19, 0.28] | 14.1 % [11.8, 16.8] |
| CPU temperature | 41.1 °C | 2.39 °C [1.66, 3.28] | — |
| GPU temperature | 40.9 °C | 2.51 °C [1.80, 3.38] | — |
| Board temperature | 32.9 °C | 1.29 °C [1.10, 1.47] | — |

A share is not given for temperature, because a percentage of a value in °C has no meaning.

**The model removes 70 to 81 % of the baseline's error, and 51 to 69 % of the recent baseline's.**

| Reading | Model error | Baseline error | Skill | Recent-baseline error | Skill vs recent baseline |
|---|---|---|---|---|---|
| Input power | 0.57 W | 1.87 W | 0.70 [0.62, 0.75] | 1.24 W | 0.54 [0.39, 0.64] |
| Orin power | 0.34 W | 1.58 W | 0.79 [0.73, 0.82] | 1.02 W | 0.67 [0.54, 0.75] |
| CPU+GPU power | 0.24 W | 0.93 W | 0.75 [0.70, 0.78] | 0.62 W | 0.62 [0.50, 0.69] |
| CPU temperature | 2.39 °C | 7.92 °C | 0.70 [0.61, 0.78] | 5.00 °C | 0.52 [0.42, 0.63] |
| GPU temperature | 2.51 °C | 8.29 °C | 0.70 [0.61, 0.77] | 5.14 °C | 0.51 [0.40, 0.62] |
| Board temperature | 1.29 °C | 6.63 °C | 0.81 [0.78, 0.84] | 4.19 °C | 0.69 [0.64, 0.74] |

The first baseline is easy to beat, because power and temperature drift between spring and autumn. The recent baseline is the fairer comparison.

**The average hides a split between devices.** For CPU temperature the median device is off by 1.4 °C and 54 of 69 devices are within 2 °C, but five devices are off by more than 11 °C. Section 3 shows which ones.

## 2. Which context matters?

**Power depends on the workload. Temperature depends on workload, enclosure sensors and weather together.**

The table shows how much the test error grows when one input group is taken away. A larger number means the group matters more.

| Reading | Error with all inputs | Without workload | Without environment | Without weather | Without hardware |
|---|---|---|---|---|---|
| Input power | 0.57 W | +1.13 W | −0.03 W | not used | +0.17 W |
| Orin power | 0.34 W | +1.15 W | +0.04 W | not used | +0.02 W* |
| CPU+GPU power | 0.24 W | +0.64 W | +0.01 W* | not used | 0.00 W* |
| CPU temperature | 2.39 °C | +0.80 °C | +0.75 °C | +0.48 °C | +0.10 °C* |
| GPU temperature | 2.51 °C | +1.06 °C | +0.72 °C | +0.40 °C | +0.07 °C* |
| Board temperature | 1.29 °C | +0.25 °C | +0.85 °C | +0.50 °C | +0.13 °C |

\* The interval includes zero, so the change may be noise.

- **Power is set by the workload.** Without it the error is about three to four and a half times higher. Environment adds nothing. Weather is not in the power model: on July, adding it changed the error by less than 0.01 W.
- **Temperature needs all three.** Workload matters most for the CPU and GPU, and the enclosure sensors most for the board. City weather removes a further 0.4 to 0.5 °C on top of the enclosure sensors.
- **Hardware matters little** once the other inputs are there, except for input power.
- **The readings of the last two hours add nothing.** The temperature model is also given the light, enclosure temperature, outside temperature and sunshine of the previous two hours. Removing them changes the error by less than 0.03 °C.
- **Telling the model which device it is helps temperature** (−0.38 °C for the CPU, −0.15 °C for the board) but not power. The main model leaves it out so that it also works on new devices.

## 3. Does it hold across hardware configurations?

**Yes for a new device of a known type. No for a type of hardware the model has not seen, and not for the few 56 V devices and the devices without a city.**

### 3a. Which devices the error comes from

Same test as section 1, split by group.

| Group | Devices | CPU temperature error | Board temperature error | Input-power error |
|---|---|---|---|---|
| Devices with a known city (weather available) | 62 | 1.56 °C | 1.15 °C | 0.52 W |
| Devices without a city (`buspas` and `demo` units) | 7 | 11.4 °C | 2.80 °C | 1.05 W |
| 12 V supply | 64 | 1.92 °C | 1.22 °C | 0.54 W |
| 56 V supply | 5 | 11.5 °C | 2.67 °C | 1.17 W |

The five worst devices for CPU temperature are two `demo` units, two `buspas` units and one TTC device. Their CPU temperature averages 50 to 84 °C in the test and varies far more than on other devices (standard deviation up to 22 °C), in a way the inputs do not explain. For the 56 V devices the model is no better than the baseline on CPU and GPU temperature.

### 3b. Devices the model has never seen

The devices are split into five parts. Each part is predicted by a model trained on the other four, on data before July, and scored on 1 August to 6 October. This covers 82 devices, 13 of which first appear after June.

| Reading | Error | Skill | Error compared with a seen device |
|---|---|---|---|
| Input power | 0.66 W [0.56, 0.77] | 0.77 | 1.06 times |
| Orin power | 0.40 W [0.33, 0.48] | 0.81 | 1.11 times |
| CPU+GPU power | 0.27 W [0.22, 0.32] | 0.78 | 1.06 times |
| CPU temperature | 3.27 °C [2.26, 4.50] | 0.63 | 1.24 times |
| GPU temperature | 3.34 °C [2.36, 4.55] | 0.63 | 1.24 times |
| Board temperature | 1.49 °C [1.31, 1.68] | 0.79 | 1.12 times |

Skill is against the usual value of all devices for that hour of day, since a new device has no history of its own. The last column compares the same hours of the same devices when the model had seen the device in training: 1.06 means 6 % more error.

A new device costs 6 to 11 % more error on power and 12 to 24 % more on temperature. The 13 devices that only appear after June are harder (input power 1.01 W, CPU 5.1 °C).

### 3c. Hardware or sites the model has never seen

All devices that share one property are removed from training, then predicted. "Times" is the error compared with the same hours when only the device itself was unseen (3b): 1.0 means the missing property costs nothing extra.

| Removed from training | Devices | Input power | Times | CPU temperature | Times | Board temperature | Times |
|---|---|---|---|---|---|---|---|
| 56 V supply | 6 | 2.44 W | 1.6 | 20.3 °C | 1.3 | 6.7 °C | 2.1 |
| Solar | 32 | 0.90 W | 1.6 | 3.1 °C | 1.0 | 1.7 °C | 1.2 |
| No solar | 50 | 1.17 W | 1.6 | 3.4 °C | 1.0 | 1.9 °C | 1.2 |
| Enclosure sensor | 65 | 0.81 W | 1.3 | 4.4 °C | 1.3 | 3.2 °C | 2.5 |
| No enclosure sensor | 17 | 1.47 W | 2.0 | 3.8 °C | 1.6 | 3.9 °C | 1.9 |
| Site: Barcelona | 14 | 1.67 W | 1.9 | 2.6 °C | 1.1 | 2.6 °C | 1.2 |
| Site: EXO | 19 | 0.70 W | 0.9 | 2.0 °C | 1.2 | 1.5 °C | 1.3 |
| Site: NJT | 7 | 1.01 W | 1.0 | 1.9 °C | 1.0 | 1.8 °C | 1.0 |
| Site: TTC | 24 | 0.32 W | 1.3 | 1.8 °C | 1.0 | 1.3 °C | 1.4 |

- **An unseen supply voltage or sensor set costs 1.3 to 2.5 times the error.**
- **Solar or not matters for input power (1.6 times) but not for temperature.**
- **An unseen site usually costs little.** Barcelona's power is the exception.

Two rows are left out of the table and are only in `rq1_test.json`. Removing the 12 V devices leaves only the few 56 V devices to train on, so it is not a fair test. Sites with three to five devices (BalHarbour, RTL, `buspas`, `demo`) are too small to read.

## 4. Does it hold across operating conditions?

**Less well on hot hours, on unusual hours, and as the test moves away from the training months.**

| Condition | Result |
|---|---|
| Over time | CPU error is 2.1 °C in August, 2.4 °C in September and 4.0 °C in the first week of October. Input-power error stays at 0.55 to 0.58 W. |
| Hot hours | On each device's hottest 10 % of hours the CPU error is 4.5 °C, and the prediction is 3.3 °C too low. |
| Overall direction | The model predicts a little low: measured values are above the prediction by 0.19 W (input power) and 0.8 °C (CPU) on average. |

**Normal and flagged hours.** Flagged hours are 1.5 to 4 times further from the prediction than other hours. This is expected: the model predicts the normal behaviour, and these are the hours that are not normal.

| Reading | Error, normal hours | Error, flagged hours |
|---|---|---|
| Input power | 0.56 W | 1.03 W |
| Orin power | 0.34 W | 0.67 W |
| CPU+GPU power | 0.23 W | 0.56 W |
| CPU temperature | 2.33 °C | 9.06 °C |
| GPU temperature | 2.46 °C | 8.27 °C |
| Board temperature | 1.28 °C | 1.97 °C |

Leaving flagged hours out of the training data made almost no difference: keeping them changed the test error by less than 0.04 for every reading.

## 5. Can the predicted ranges be trusted?

**Not fully: the 90 % bands hold 82 to 86 % of the test hours.**

| Reading | Share of test hours inside the band | Median width of the band |
|---|---|---|
| Input power | 84.4 % [80.0, 88.5] | 1.9 W |
| Orin power | 86.0 % [80.7, 90.4] | 1.0 W |
| CPU+GPU power | 81.6 % [75.1, 87.4] | 0.6 W |
| CPU temperature | 82.6 % [77.2, 87.3] | 5.4 °C |
| GPU temperature | 84.1 % [79.0, 88.6] | 6.1 °C |
| Board temperature | 86.1 % [82.9, 89.1] | 4.4 °C |

- The bands fall 4 to 8 points short of 90 %. Without the adjustment on July they held only 68 to 78 %.
- The same shortfall already showed on July during development (83 to 89 %). One month is not enough to cover the drift of the following months.
- The bands are worse for the 56 V devices (62 to 80 %) and for flagged hours (61 to 75 %).

## Against the proposed thresholds

The protocol proposes these thresholds for calling the model useful (decision D2). They are not yet confirmed.

| Proposed threshold | Result | Met? |
|---|---|---|
| Input-power error ≤ 5 % of the device's mean | 5.1 % [4.3, 6.0] | Borderline |
| CPU temperature error ≤ 2.0 °C | 2.39 °C [1.66, 3.28] | No (1.56 °C for devices with a city) |
| GPU temperature error ≤ 2.0 °C | 2.51 °C [1.80, 3.38] | No (1.68 °C for devices with a city) |
| Board temperature error ≤ 2.0 °C | 1.29 °C | Yes |
| Band coverage between 85 % and 95 % | 82 to 86 % | Only for Orin power and board temperature |

## Limits

- **The test months are not unseen.** See the disclosure above.
- **Few unusual devices.** Five or six 56 V devices and seven devices without a city are too few to say why they behave differently.
- **Weather is approximate.** It is one location per agency, and the agency-to-city table is not confirmed by BusPas. 18 devices have no city.
- **Hardware is treated as fixed.** Supply voltage, solar and sensors are taken from each device's whole history, as if known at installation.
- **Hour of day is in UTC**, not local time.
- **The recent baseline was added late.** The first baseline is the one named in the plan. The recent one was added on 10 November, after the July check showed how much the first one drifts, and before the test was run.

## Development results (March to July)

These are the checks made before the test. They show that the choices were not made on the test months.

| Step | Trained on | Scored on | Result |
|---|---|---|---|
| Choose inputs and model size | March – May | June (45 devices) | Mean skill 0.82 for power and 0.88 for temperature |
| Recheck | March – June | July (70 devices) | Input power 0.48 W, Orin 0.25 W, CPU+GPU 0.16 W, CPU 1.78 °C, GPU 1.86 °C, board 1.10 °C |
| Choose how to set the bands | March – May, bands set on June | July | One margin for all devices was closer to 90 % than a margin per configuration or per device |

The rules were fixed in advance. Inputs and model size: the highest mean June skill, for power and for temperature separately. Bands: the share of July hours inside the band closest to 90 %. In both cases, when two options are within 0.005, the simplest wins.

The test errors are 17 to 46 % above the July errors. That is the cost of predicting one to three months further ahead.

## Files and commands

| | |
|---|---|
| Code | `analysis/rq1.py` (at commit `93eb148` plus uncommitted changes) |
| Development | `.venv/bin/python -I analysis/rq1.py dev` → `analysis/out/rq1_dev.json`, `rq1_locked.json` |
| Test | `.venv/bin/python -I analysis/rq1.py test` → `analysis/out/rq1_test.json`. It refuses to run a second time. |
| Windows | `[splits.rq1]` in [`protocol.toml`](protocol.toml) |

# Protocol for RQ1 and RQ2

**Status:** draft 0.1, 3 November 2026. Not frozen. Decisions D1 and D4 were settled on 10 November 2026; RQ1 follows the shorter [`PLAN_RQ1.md`](PLAN_RQ1.md). Exact settings live in [`protocol.toml`](protocol.toml); this document gives the reasons. Where the two disagree, the TOML wins and this file is fixed.

- **RQ1.** To what extent can a model conditioned on workload and environmental context predict the expected power and thermal behaviour of smart bus-stop devices across differing hardware configurations and operating conditions?
- **RQ2.** Does a hand-specified dependency graph detect energy anomalies more accurately than non-graph and learned-graph detectors?

## How this protocol works

- **Draft.** Anything can change. Development data can be used as often as needed. For RQ1 that is March–July only, since D1 made August–October the test.
- **Freeze.** All of these happen together:
  - Open decisions D1–D5 are settled.
  - The hand graph and the fault-injection spec have been reviewed.
  - `inference.seed_test` is drawn.
  - The config check passes.
  - `meta.status` is set to `frozen`.
  - The commit hash is recorded in `meta.frozen`.
- **After the freeze.** The confirmatory run happens once. Any change goes in the [deviation log](#deviation-log) with the date, what changed, why, and whether any test result had been seen.
- **The pilot.** The earlier `rq1_nbm.py` (now in `archive/2026-11-09/code/`) and `rq2_graph.py` runs are exploratory. They shaped this design, but their numbers are not reported as results, because several settings were chosen after looking at August–October.

## What the pilot taught, and what changes

| Pilot | Problem | This protocol |
|---|---|---|
| Settings were tuned while looking at August–October | That window can no longer be a clean test | D1, settled 10 November: there is no fresh export. August–October stays the test, every setting is chosen on March–July, the test is scored once, and the write-up says the pilot saw these months |
| Per-device thresholds and the tail transform were set on the clean test period | That data isn't available at deployment, and the false-alarm rate holds by construction | Thresholds come from the calibration window; the realised false-alarm rate is reported |
| Max vs tail aggregation was picked afterwards, and the pick changed the RQ2 winner | Choosing the analysis after seeing results | Each detector's setting is chosen on development episodes; the full grid is reported |
| Injected faults spread along the same paths as the hand graph | The hand graph can win by construction | An independent injector spec, faults off the graph's paths, and faults that don't spread |
| Feature groups were added in a fixed order | Each group's credit depended on the order | Group Shapley values over all 16 subsets |
| No graph controls | Can't tell graph knowledge from plain sparsity | Random and perturbed graphs as controls |
| 90 % bands covered less than 90 % after July | Behaviour drifts | Adaptive conformal variant; per-configuration calibration |
| Injected faults were the only ground truth | Synthetic only | Real episodes from the data, then Linux-log labels |

## Shared design

### Data

- **Source.** The device × hour table `analysis/out/features.parquet`, with the derived columns from `rq_common.load()`: RAM in use, the hardware flags, and agency peer environment.
- **Carried-over hygiene.** Device IDs are lower-cased, and the placeholder codes are filtered as in `profile_stream.py`.
- **Inclusion.** A device is scored in a window only if it has at least 7 days of input power there.
- **The fleet.** 97 devices have power data, in 13 hardware combinations, several with only one to three devices. Along each factor:

  | Factor | Devices |
  |---|---|
  | 56 V supply | 13 |
  | 12 V supply | 84 |
  | Solar charge controller | 36 |
  | No DeepStream | 33 |

- **Late joiners.** Fifteen devices first report after 1 July. They are scored only as unseen devices.
- **Flag days.** The three fleet-wide shift days (10 May, 11 June, 13 July) stay in. One sensitivity run leaves them out.

### Windows

| | Fit | Calibrate | Evaluate / test |
|---|---|---|---|
| Development (free use) | 8 Mar – 30 Jun | July | 1 Aug – 6 Oct |
| Confirmatory (once) | 8 Mar – 31 Aug | 1 Sep – 6 Oct | 7 Oct – end of the fresh export |

- **Same shape in both.** Each layout has a fit window, then about a month of calibration just before the scored period.
- **What moves to the confirmatory run.** It refits with the settings locked in development; only the windows move.
- **D1, settled 10 November 2026: the fallback applies.** There is no fresh export. The test is 1 August – 6 October, scored once, and the report says the pilot saw this period.
- **RQ1 windows under D1.** Inputs and model settings are chosen on June (fit March–May) and rechecked on July (fit March–June). The band calibration is chosen on July (fit March–May, calibrate June). The final model is fitted on March–June, calibrated on July and tested on 1 August – 6 October. No hour from August on is loaded before the settings are locked (`analysis/rq1.py`, `splits.rq1` in the TOML).
- **Season.** The fresh export is autumn, colder than anything in the fit window. That makes it a test of operating conditions in its own right, and the report says so.

### Statistics

- **Devices are the independent unit.** Hours within a device are strongly correlated, so confidence intervals come from a bootstrap over devices: 2,000 resamples, 95 % percentile intervals.
- **Paired comparisons.** Every comparison uses the same devices, hours and fault episodes for each method.
- **Multiplicity.** Holm correction across the comparisons named in this protocol.
- **Effect sizes.** Always reported with their intervals, not only test outcomes.

## RQ1: predicting expected power and thermal behaviour

### What counts as an answer

"To what extent" calls for estimates, not a yes or no. Each of the following comes with a confidence interval, for every target and every test below:

- **Accuracy.** Mean absolute error (MAE) in W or °C, error relative to the device's mean level, and variance explained within each device. For thermal targets, also MAE on the hottest 10 % of hours.
- **Skill.** 1 − MAE(model) / MAE(baseline). The baseline is the device's own hour-of-day median, or the fleet's for a device the model hasn't seen.
- **Generalisation gap.** MAE on held-out configurations divided by MAE on familiar ones.
- **Calibration.** Coverage and width of the 90 % bands, plus the weighted interval score.

The thresholds for "useful" are agreed before the confirmatory run (D2). Proposed: input-power error within 5 % of the device mean, temperatures within 2 °C, and band coverage between 85 % and 95 %.

### Targets

- **Power:** input power (primary), Orin module power, CPU+GPU power.
- **Thermal:** CPU, GPU and board temperature.

### Inputs

- **Time:** hour of day, day of week, weekend.
- **Hardware:** 56 V supply, RAM, disk, solar charge controller, environment sensor fitted, DeepStream installed, agency.
- **Environment:** the enclosure's own temperature, humidity and light where fitted; the same three averaged over the agency's other devices.
- **Weather** (decision D4, settled 10 November 2026): hourly outside temperature, humidity, dew point, cloud cover, wind, solar radiation and precipitation of the agency's city.
  - The signs carry no coordinates. The city comes from the agency label, which is our reading and is not confirmed by BusPas. The weather is city-level, so it follows the regional weather, not the shade or wind at one stop.
  - The agencies `buspas`, `demo`, `K` and `show` (18 signs) have no city. Their weather is left missing, not filled.
  - Weather is one input group, so the ablation reports RQ1 with and without it.
  - A check on March to July (`weather_check.py`) found that each sign's own city fits its day-to-day temperature best, for 65 of 65 signs.
- **Workload:** DeepStream running, RAM in use, cellular receive and send, motion-sensor count, DeepStream errors.

Inputs never include another power or thermal reading, nor any past value of the target. Using them would turn the model into a forecaster, not a model of expected behaviour given context. The one exception is the secondary *physics-chain* model, which adds measured power as an input for the thermal targets. It shows whether heat is unpredictable in general, or only until the power is known.

### Models

| Model | Role |
|---|---|
| Hour-of-day median | Baseline for skill |
| Ridge regression | Simple, readable reference |
| LightGBM | Main model; inputs and settings chosen on June, with a fit on March–May |
| Hierarchical | Fleet LightGBM plus a per-device offset pulled towards zero, so it degrades gracefully for unseen devices |
| Grey-box thermal (optional) | T = a + b · ambient + R_th · P with a first-order lag |
| Physics chain | LightGBM plus measured power, thermal targets only |

Each model is fitted three ways: one model for the fleet, one per configuration and one per device. Comparing them answers whether one fleet model is enough across configurations.

### Which context matters

- **The fits.** Time is always in. The other four groups (hardware, environment, workload, device identity) are tried in all 16 combinations, for each target.
- **The credit.** Each group gets its exact Shapley share of the error reduction over the time-only model. The share doesn't depend on any order of adding groups.

### Generalisation tests

| Test | What's held out | Question |
|---|---|---|
| Over time | later months, same devices | Does expected behaviour hold across months and seasons? |
| Unseen devices | whole devices, in 5 folds; plus the late joiners | A new device of a known type |
| Unseen configuration | each value of each factor in turn: 56 V, 12 V, solar, no solar, sensor, no sensor, each agency | A hardware setup or site never seen |
| A few days of data | a new device with 0, 1, 3, 7 or 14 days of its own data | How much local data closes the gap |
| By condition | month, ambient band (< 0, 0–10, 10–20, 20–30, > 30 °C), DeepStream on/off, day/night, solar charging | Which operating conditions break it |

- **Unseen configurations over time.** That test is also run with training on earlier months and scoring on later ones. Otherwise devices held out on the same days share the weather, and the test looks easier than it is.
- **One factor at a time.** The 13 full combinations are too small to hold out on their own.

### Prediction bands

- **Base model.** Quantile models at 5, 50 and 95 %.
- **Recalibration.** Conformalised on the calibration window.
- **Four variants:**
  - fleet-wide
  - per configuration
  - per device
  - adaptive, where the margin keeps updating as hours arrive
- **Reported for each variant:** coverage, width and weighted interval score, broken down by configuration and month.

### Expectations carried from the pilot

These came out of the pilot. The confirmatory run tests them on data the pilot never saw:

- **H1a.** For the power targets, workload has the largest Shapley share.
- **H1b.** For the thermal targets, environment has the largest share among the context groups. The physics-chain model explains more within-device variance than the context-only model.
- **H1c.** The generalisation gap is above 1 when 56 V or solar devices are held out.
- **H1d.** Adaptive or per-configuration bands stay within the coverage band in every month of the test window. Fleet-wide bands do not.

## RQ2: does a hand-specified graph detect energy anomalies better?

### Hypothesis

**H2.** Every detector is held to the same false-alarm budget: 1 % of fault-free hours, with a threshold per device. Under that budget, the hand-graph detector catches more fault episodes than three comparators:
- the best non-graph detector;
- the best learned-graph detector;
- the median random graph.

"Best" is chosen on development episodes before the confirmatory run.

The answer is yes if all three Holm-adjusted 95 % intervals of the recall difference lie above zero.

### Definitions

- **Energy anomaly.** A period where power, supply, energy accounting or heat departs from what context predicts. The fault layers are:
  - supply
  - load
  - compute
  - thermal
  - sensing
  - accounting
- **Nodes.** The 10 readings in `rq2.nodes`:
  - Orin, CPU+GPU and heartbeat power
  - input power, voltage and current
  - the energy-counter rate
  - CPU, GPU and board temperature
- **Resolution.** Hourly, as in RQ1, plus one rerun at 10 minutes for short faults.

### The hand graph

This is the pilot graph, v0. Every edge has a physical or measurement reason:

| Parent → child | Reason |
|---|---|
| CPU+GPU power → Orin power | The CPU+GPU rail is part of the module rail |
| Orin power → heartbeat power | Assumed to log the same rail (definition undocumented) |
| Orin power → input power | Module draw plus display, modem and conversion losses |
| Input power → input current | I = P / V |
| Input voltage → input current | I = P / V |
| Input power → energy-counter rate | The counter integrates input power |
| CPU+GPU power → CPU temperature | Compute power heats the SoC |
| CPU+GPU power → GPU temperature | Compute power heats the SoC |
| CPU temperature → GPU temperature | Same die |
| Input power → board temperature | Input-stage dissipation heats the board sensor |

**Candidate edges, decided on physics before the freeze.**
- **Board temperature → CPU temperature.** The board is the SoC's local ambient: T_SoC ≈ T_board + R_th · P.
- **Board temperature → GPU temperature.** For the same reason, unless the GPU inherits it through CPU temperature.

The pilot suggests board → CPU temperature would help with cooling faults. That isn't a reason to add it, because edges come from physics, not recall.

**Questions for the hardware reviewer (D3):**
1. Does heartbeat power read the Orin rail?
2. Which direction should the input-power edges take? The INA238 measures voltage and current and computes power from them. Physically, the load sets power and current follows. The direction changes how a current-sensor fault spreads through the graph.
3. Do the 56 V devices have a different power stage that changes any edge?

### Fault benchmark

| Id | Layer | Fault | Root | Severities | Hours |
|---|---|---|---|---|---|
| F1 | load | Parasitic load | input power | +5, 10, 20 % of median | 6–24 |
| F2 | compute | Compute runaway | CPU+GPU power | +25, 50, 100 % of median | 6–24 |
| F3 | accounting | Energy-counter drift | energy rate | ramp to 5, 10, 20 % | 24–72 |
| F4 | sensing | Stuck power reading | input power | 6–12, 12–24, 24–48 h | 6–48 |
| F5 | supply | Supply sag | input voltage | −3, 6, 12 % | 6–24 |
| F6 | thermal | Cooling loss | CPU and GPU temperature | +3, 6, 12 °C | 6–24 |
| F7 | sensing | Sensor drift | any node | ramp to 1, 2, 4 residual SDs | 24–72 |
| F8 | sensing | Intermittent spikes | any node | 4, 6, 8 SD spikes, 3–6 per episode | 12–48 |
| F9 | sensing | Bias on one node, no spreading | each node in turn | 1, 2, 4 residual SDs | 6–24 |

- **Who writes the injector.** Its spec (`faults.md`) says how each fault spreads from device physics, with coefficients estimated on the fit window. Someone other than the hand graph's author writes it, so its paths are not copied from the graph.
- **Faults that don't follow the graph.** F6 travels a path the v0 graph doesn't model. F9 doesn't spread at all.
- **Episodes.**
  - 6 per device per fault and severity.
  - At least 48 h apart.
  - At least 80 % of hours must have input power.
  - Development episodes go in the development evaluation window.
  - Test episodes go in the test window, with the seed drawn at the freeze.
- **Sanity checks before any detector runs:**
  - A detector told the onset catches every top-severity episode.
  - The lowest severity is within 2 residual SDs on at least half the episodes, so it is genuinely hard.

### Detectors

**Main comparison: only the graph differs.**
- **Shared pieces.** Every graph detector uses the same node model, residual scaling, scoring and threshold rule.
- **The node model.** RQ1's locked context model, plus a per-device ridge on the parents' observed values. Faulty parent values are fed in as inputs, so a fault spreads to the children's predictions.

| Family | Detectors |
|---|---|
| Non-graph | context residual alone (empty graph), seasonal z, Isolation Forest, PCA residual, autoencoder |
| Learned graph, learned on fault-free fit-window data | strongest correlations (k = 2), Lasso neighbourhoods, Granger (lag 2, BH q = 0.01), NOTEARS, PC (Fisher z, α = 0.01), complete graph |
| Hand graph | v0 plus whichever candidate edges the review accepts |
| Controls | 30 random graphs with the hand graph's edge count; the hand graph with 1, 2 or 3 edges removed, added or reversed (10 draws each) |

**Second check: published forms.** GDN and USAD are each run end to end, tuned on development episodes. If the main comparison favours the hand graph but published methods beat it, the report says so.

### Tuning and thresholds

- **Tuning grid.** Score aggregation (max or tail probability) × window (1 h or 6 h) × relative floor (0.25, 0.5 or 1 % of level).
- **Choosing.** Each detector gets the grid point with its highest mean recall on development episodes.
- **Reporting.** The full grid is reported as a sensitivity check.
- **Thresholds.** Per device: the 99th percentile of the system score on the fault-free calibration window. Recall is also reported at 0.1 % and 5 %.

### Metrics

- **Primary: event recall.** An episode counts as caught if any alarm falls between its onset and its end, plus the score window's lag.
- **Secondary:**
  - realised false-alarm rate on fault-free test hours
  - recall at matched realised false-alarm rates
  - precision–recall and ROC areas
  - hours to detection
  - false alarms per device-week
  - top-1 and top-3 localisation of the root node
- **Localisation rule.** The node with the largest mean score over the episode. Graph detectors also get a secondary rule: the most upstream anomalous node whose parents look normal.

### Breakdowns

Recall differences are reported by:
- fault and severity, giving recall-versus-size curves
- 12 V vs 56 V
- solar vs no solar
- month
- ambient band

### Real episodes

- **Candidates.** Rules on the data flag possible real episodes:
  - CPU at or above 95 °C for 2 h or more
  - a daily input-voltage step over 5 % that lasts at least 3 days
  - DeepStream error rate above the device's 99.5th percentile
  - outages of 1 h or more
  - free disk under 1 GB
  - reported power more than 2 % off voltage × current
- **Labelling.**
  - A reviewer confirms or rejects each candidate from raw traces, without seeing any detector's score.
  - A second reviewer labels 20 % of them, and agreement is reported as Cohen's kappa.
- **Status as evidence.** These episodes are secondary evidence. The flagging rules read some of the same signals the detectors use, and the report says so.

### Linux logs (Phase 5)

- **What the logs add.** Independent labels for RQ2: under-voltage and over-current alarms, thermal throttling, out-of-memory kills, reboots and sensor-driver errors.
- **For RQ1 too.** Richer workload inputs: CPU and GPU load, clock speeds and throttling state.
- **Own protocol first.** The extension gets its own addendum, frozen before it runs: which messages map to which fault layer, and which fields become inputs.

## Phases and gates

| Phase | Work | Gate to the next phase |
|---|---|---|
| 0. Protocol | This document and `protocol.toml`; D1–D5; hand-graph review; `faults.md` | Supervisor sign-off |
| 1. RQ1 development | All models and tests on development windows | RQ1 settings locked in the TOML |
| 2. Fault benchmark | Injector from `faults.md`, sanity checks, real-episode labels | Sanity checks pass; kappa reported |
| 3. RQ2 development | Detectors and grid selection on development episodes | RQ2 settings and "best" comparators locked |
| 4. Freeze and confirmatory run | Freeze checklist, then RQ1 and RQ2 once on the test window | Results written up as they come out |
| 5. Linux-log extension | Addendum, then RQ1 with richer workload inputs and RQ2 with log labels | Addendum frozen before running |

## Open decisions

| | Decision | If unanswered |
|---|---|---|
| D1 | **Settled 10 November 2026:** no fresh export. August–October are reused as the test, scored once, with every setting chosen on March–July. The write-up says the pilot had already scored these months. | Not applicable |
| D2 | Thresholds that make RQ1 "useful" | The proposed values in `rq1.useful` |
| D3 | Who reviews the hand graph, and who writes `faults.md` (two different people)? | Cannot freeze |
| D4 | **Settled 10 November 2026:** city-level weather from the agency labels is an RQ1 input (see [Inputs](#inputs)). It was settled before any RQ1 run with weather, and without using August–October. Stop locations from BusPas would still improve it. | Not applicable |
| D5 | Run the confirmatory RQ2 now on synthetic and in-data episodes, or wait for the Linux-log labels? | Run now; logs as Phase 5 |

## Deviation log

| Date | Change | Reason | Test results seen? |
|---|---|---|---|
| 10 November 2026 | The `demo` (6) and `show` (1) units are removed from the data of every RQ experiment (`EXCLUDED_AGENCIES` in `analysis/rq_common.py`, decision D6). RQ1 was redone from the start: settings chosen again on March to July, then 1 August to 6 October scored a second time. The first run is kept in `archive/2026-11-10-rq1-run1/`. | These units are not installed at a stop (demonstration and show units), so they are not the devices the question is about. In the first run the `demo` units had the largest CPU-temperature errors. | **Yes.** The first run's test results, including the per-agency split, had been read before the decision. The second run is therefore not a blind test. |
| 10 November 2026 | The `buspas` units (10) are also removed from the data of every RQ experiment (`EXCLUDED_AGENCIES`, decision D7). RQ1 was redone from the start a third time: settings chosen again on March to July, then 1 August to 6 October scored a third time. The second run is kept in `archive/2026-11-10-rq1-run2/`. | Asked for by the user. These units carry the vendor's name instead of a transit agency's, have no city for the weather, and in the second run were three of the four devices with the largest CPU-temperature errors. Whether they are installed at a stop is not confirmed. | **Yes.** The first and second runs' test results had been read before the decision. The third run is therefore not a blind test. |
| 15 November 2026 | Extra breakdowns of the third-run test predictions were added to the RQ1 results (section 6): coolest and hottest hours, fixed temperature levels, whether hot hours were outside the fit months, workload level, hardware configuration. Script `analysis/rq1_extra.py`. Nothing was refitted and the test was not scored again. | A reviewer asked for them. They were not in the plan, so the results mark them as unplanned and descriptive. | Yes |

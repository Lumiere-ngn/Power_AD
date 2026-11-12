# Plan for answering RQ1 and RQ2: two people, 4 weeks

**Status:** plan, 6 November 2026, updated the same day to work without the Linux logs. Nothing in it has been run yet. It's the detailed, two-person version of [`PLAN.md`](PLAN.md): same scope, same weeks, with each task assigned.

- **RQ1.** To what extent can a model conditioned on workload and environmental context predict the expected power and thermal behaviour of smart bus-stop devices across differing hardware configurations and operating conditions?
- **RQ2.** Does a hand-specified dependency graph detect energy anomalies more accurately than non-graph and learned-graph detectors?

## Roles

| | Person A: data and RQ1 | Person B: graph and RQ2 |
|---|---|---|
| Owns | Data requests, splits, the context model, weather, real-episode labels | Hardware review, the hand graph, faults, detectors, GDN |
| Code | `rq_common.py`, `rq1_nbm.py`, new `weather.py`, new `labels.py` | `rq2_graph.py`, new `gdn.py` |
| Writes up | Data, RQ1 method and results | Graph, RQ2 method and results |
| Suits | Whoever is stronger at data wrangling and modelling | Whoever is closer to the hardware contacts |

**How the two halves connect.** RQ2's detectors predict each reading from RQ1's context model. So A hands the model to B at the end of week 1, and B uses the pilot's model until then. Only A edits `rq_common.py`, and B asks A for any change there, so the two never edit the same file.

**Git.**
- **Branches:** A works on `rq1`, B on `rq2`, both merging to `main` through pull requests on `Lumiere-ngn/Power_AD`.
- **Common base:** the pilot code is still uncommitted. Commit it to `main` first, so both branches start from the same code.

## Calendar

| Week | Dates | Goal |
|---|---|---|
| 1 | 9–13 Nov | Requests out, setup fixed, RQ1 rerun, RQ2 fixes started, shortlist of real episodes to the operator |
| 2 | 16–20 Nov | RQ2 rerun with the fixes and GDN; weather; operator-confirmed real episodes |
| 3 | 23–27 Nov | Lock everything, run the final test once, real-episode case studies |
| 4 | 30 Nov – 4 Dec | Write up, update the report and deck, slack |

## Hand-offs

| When | From → to | What |
|---|---|---|
| W1 Tue | A → B | Split dates in `rq_common.py` |
| W1 Fri | A → B | The locked RQ1 context model: feature list and LightGBM settings |
| W2 Wed | A → B | Weather columns, if any, so RQ2's node model uses the same context as RQ1 |
| W2 Wed | A → B | Size and duration of the confirmed low-voltage and overheating episodes, to check the F5 and F6 severities |
| W2 Fri | A → B | The real-episode table |
| W2 Fri | B → A | RQ2 development results, for the joint review |

---

## Week 1 (9–13 Nov)

### Together, Monday

**T1. Commit the pilot code.** Commit `rq_common.py`, `rq1_nbm.py`, `rq2_graph.py`, `rq_report.py`, the template and this folder to `main`, then create the two branches.
*Done when* both branches exist on GitHub.

**T2. Supervisor meeting, 30 minutes.** Confirm:
- the six targets: input, Orin and CPU+GPU power; CPU, GPU and board temperature;
- the four context groups: time, hardware, environment, workload;
- the fault list: the pilot's F1–F6 plus a new F7, a bias on one reading that doesn't spread to others;
- the 1 % false-alarm budget;
- the rule for "yes" on RQ2.

*Done when* the answers are written in `analysis/protocol/SETUP.md` (half a page) and the supervisor agrees.

**T3. Send every request on day 1.**
- **A sends:** a fresh Prometheus export from 7 October; maintenance and incident records; stop locations or agency cities. Linux logs are optional: ask only if they're easy to get, because nothing in this plan waits on them.
- **B sends:** the hand graph for review (see B1).

*Done when* every request is sent, with a reply date asked for.

### Person A

**A1. Split dates.** In `rq_common.py`, replace `CAL` and `TEST` with:
- fit: 8 March – 30 June
- calibrate: July
- tune: August
- test: 1 September – 6 October

Add a switch for the fresh export: fit to 31 August, calibrate on 1 September – 6 October, test on the export.
*Done when* both scripts read the same dates and B has them (Tuesday).

**A2. Rerun RQ1 on development data.** Point `rq1_nbm.py` at fit (March–June) → evaluate (August). Keep September – 6 October untouched. Keep:
- the baselines (device × hour-of-day median);
- full-context LightGBM;
- the drop-one ablations (hardware, environment, workload, device id);
- leave-device-out (5 folds);
- leave-configuration-out (56 V / 12 V, solar / no solar);
- conformal bands calibrated on July.

*Done when* `rq1.json` regenerates on development windows only.

*Status, 10 November:* replaced by `analysis/rq1.py dev`, which chooses everything on March–July and does not tune on August (decision D1 settled; see `PLAN_RQ1.md`).

**A3. Lock the context model and hand it to B.** Fix the feature list and LightGBM settings from A2's results, then write them in `SETUP.md`.
*Done when* B confirms `rq2_graph.py` uses them (Friday).

**A4. Prepare for the fresh export.** When it arrives, unpack it next to the current dataset and rerun `features.py`, with the same placeholder filters and lower-cased device ids.
*Done when* `features.parquet` runs past 7 October. If it hasn't arrived by the end of week 2, this task is dropped.

**A5. Shortlist of real episodes for the operator.** The operator is the slowest step, so the shortlist goes out in week 1. New `analysis/labels.py` flags candidate episodes in the telemetry:

| Rule | Episodes found (pilot data) | Devices |
|---|---|---|
| CPU at or above 95 °C for 2 h or more | 38 | 9 |
| 12 V devices below 11.5 V for 1 h or more | 137 | 8 |
| Reported input power more than 2 % off voltage × current | 16 hours | 10 |

Outages and RAM below 0.2 GB are left out, because they're too common to be labels: 2,202 outage episodes on 108 devices, and 4,339 low-RAM episodes on 68 devices.

Send the operator one line per episode (device, date and time, what was seen), and ask them to mark each one "real problem", "known cause" or "not sure", replying by Wednesday of week 2.
*Done when* the shortlist is sent (Friday).

### Person B

**B1. Graph review sheet.** One page, sent on Monday:
- the 10 edges with their one-line reasons (`HAND_WHY` in `rq2_graph.py`);
- the two candidate edges, board temperature → CPU and GPU temperature;
- three questions:
  1. Does heartbeat power read the Orin rail?
  2. The INA238 computes power from its own voltage and current readings. Should the edges run V, I → P instead of P, V → I?
  3. Do the 56 V devices have a different power stage?

*Done when* it's sent. Expect the reply by Friday.

**B2. Thresholds from the calibration month.** `device_thresholds` and `tail_transform` currently use the clean test period. Change both to the July calibration rows.
*Done when* no test-period row is used to set any threshold or scale.

**B3. Separate tuning faults from test faults.** `sample_episodes` currently samples over the whole test grid. Make two sets:
- tuning episodes in August, with a fixed seed;
- test episodes in 1 September – 6 October, with a different seed, not run until week 3.

*Done when* the August run touches no September or October row.

**B4. New fault F7: bias on one reading, no spread.** In `inject`, for each node in turn, add 1, 2 or 4 residual SDs for 6–24 h. Nothing else changes.
*Done when* a check that knows the onset sees every top-severity F7 episode.

**B5. Choose score settings on August faults.** For each detector, pick the setting in `VARIANTS` (max or tail, 1 h or 6 h) with the highest mean recall on the August episodes. Store the choice in `rq2.json`.
*Done when* each detector has one recorded setting.

---

## Week 2 (16–20 Nov)

### Person A

**A6. Weather (only if locations arrived by Monday).** New `analysis/weather.py`:
1. Fetch hourly air temperature, relative humidity, solar radiation and wind speed from the Open-Meteo historical archive, for each location, March–November.
2. Join them to the features table by location and UTC hour, as a new `WEATHER` group in `rq_common.py`.
3. Rerun RQ1 with and without weather.

*Done when* the change in thermal error is reported and the columns are handed to B (Wednesday). With no locations, write one line for the limitations instead.

*Status, 10 November:* steps 1 and the join of step 2 are done at city level (`analysis/weather.py`, columns `wx_*` in the features table; city from the agency label, 18 signs without a city). The `WEATHER` group is in `rq_common.py`, and `analysis/rq1.py` runs RQ1 with and without weather (`RESULTS_RQ1.md`).

**A7. Real-episode table.** Extend `labels.py` to write `out/real_episodes.parquet`, with columns device, start, end, type, source and verdict. In order of trust:
1. **Episodes the operator marked "real problem"** on the A5 shortlist.
2. **Incident records**, if they came, mapped to device and dates.
3. **Episodes the operator marked "not sure", or never answered.** You review these by eye on the raw traces, before any detector is scored on them. They're kept with the verdict "unconfirmed".

"Known cause" episodes, such as maintenance or a planned test, are dropped.

By Wednesday, give B the size and duration of the confirmed low-voltage and overheating episodes.
*Done when* the table exists with a count per type and verdict, and B has it (Friday).

**A8. RQ1 breakdowns.**
- Errors by month, ambient band, DeepStream on or off, and configuration.
- Band coverage by configuration and month.
- A first draft of the RQ1 results table: target × test → error, gain over the baseline, coverage.

*Done when* the draft table is ready for Friday's review.

### Person B

**B6. Apply the graph review (by Wednesday).** Update `HAND` and `HAND_WHY` with the edges the reviewer accepts. If there's no reply, keep the pilot graph and note it as a limitation.
*Done when* the graph is final. It doesn't change after this.

**B7. Random-graph controls.** Generate 30 random acyclic graphs with the same in- and out-degree per node as the hand graph, and run them through the same pipeline.
*Done when* the hand graph's rank among them, on August recall, is reported.

**B8. GDN.**
1. Add PyTorch to the project (`uv add torch`; CPU is enough for 10 hourly readings).
2. Adapt the official GDN code (Deng & Hooi, 2021): 12 h input window, 2–3 neighbours per node, embedding size 32, trained on clean fit-window data.
3. Feed its prediction errors into the same scoring and threshold pipeline.
4. Tune it on August faults.

*Done when* GDN appears as a detector in `rq2.json`. If it isn't working by Thursday, drop it and note it.

**B9. RQ2 development run.** First compare the F5 (supply sag) and F6 (cooling loss) severities with the real episode sizes from A. If they're far off, move the severities into the real range now, before anything is locked. Then run all detectors on the August faults, with the week-1 fixes. Name the best non-graph and best learned-graph detectors; they're fixed from here on.
*Done when* the results are ready for Friday's review.

### Together, Friday

Review the development results for RQ1 and RQ2. Write every remaining setting into `SETUP.md`: features, settings, chosen score variants, the named comparators and the graph.

---

## Week 3 (23–27 Nov)

### Together, Monday

**T4. Lock.** Merge both branches, commit `SETUP.md`, and tag the commit `final-run`. Nothing is tuned after this. Any change gets a dated note in `SETUP.md` with the reason.

### Person A

**A9. Final RQ1 run.** Test on 1 September – 6 October. With the fresh export: refit to 31 August, calibrate on 1 September – 6 October, and test on the export.
*Done when* `rq1.json` holds the final numbers.

*Status, 10 November:* run once by `analysis/rq1.py test` on 1 August – 6 October, with no fresh export (`out/rq1_test.json`, `RESULTS_RQ1.md`).

**A10. RQ1 answer.**
- The results table: target × test.
- One paragraph per part of the question: how well, which context matters, across configurations, across conditions.

*Done when* B has read it.

### Person B

**B10. Final RQ2 run.** Test faults, with the new seed, on 1 September – 6 October or the fresh export.
*Done when* `rq2.json` holds the final numbers.

**B11. Real-episode case studies.** Run the locked detectors over the periods in A's real-episode table. For each detector, report:
- which confirmed episodes it flags, and separately which unconfirmed ones;
- alarms per device-week outside any episode.

Present these as case studies next to the main result, not as the headline number. The episodes were found with rules on readings the detectors also score (CPU temperature, input voltage, input power), and that favours detectors that look at one reading at a time.
*Done when* both are reported per detector, with a short note on two or three episodes where the detectors disagree.

**B12. RQ2 answer.**
- Bootstrap 95 % confidence intervals over devices for each recall difference.
- The yes-or-no against each comparator.
- Breakdowns by fault type and 12 V / 56 V.
- Which reading each detector blames.

*Done when* A has read it.

---

## Week 4 (30 Nov – 4 Dec)

| Day | Person A | Person B |
|---|---|---|
| Mon–Tue | Write data and RQ1 sections. Update the RQ1 parts of `rq_report.py` | Write graph and RQ2 sections. Update the RQ2 parts of `rq_report.py` |
| Wed | Review B's sections | Review A's sections |
| Thu | Together: introduction, threats to validity, conclusion, future work. Update the deck | |
| Fri | Slack | |

## Threats to validity to expect

- Mostly synthetic faults.
- The real episodes were found with rules on the same telemetry the detectors read, so they aren't independent labels, even when the operator confirms them.
- A hand graph with only a light review.
- A test period the pilot already saw, if there's no fresh export.
- No weather, if there are no locations.

## Cut, for future work

- Linux logs, both as RQ1 inputs (CPU and GPU load, clock speeds) and as independent RQ2 labels. If they arrive before week 3 anyway, add them to the real-episode table as the most trusted source.
- the few-days-of-data test
- extra RQ1 models (hierarchical, physics-based)
- the PC graph and USAD
- the 10-minute rerun
- a second person labelling the real episodes

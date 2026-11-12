# Plan for answering RQ1 and RQ2 (4 weeks)

**Status:** plan, 6 November 2026. Nothing in it has been run yet.

- **RQ1.** To what extent can a model conditioned on workload and environmental context predict the expected power and thermal behaviour of smart bus-stop devices across differing hardware configurations and operating conditions?
- **RQ2.** Does a hand-specified dependency graph detect energy anomalies more accurately than non-graph and learned-graph detectors?

## Approach

Most of the code already exists from the pilot:
- `analysis/rq1_nbm.py` for RQ1 (10 November: moved to `archive/2026-11-09/code/`, replaced by `analysis/rq1.py`; see `PLAN_RQ1.md`);
- `analysis/rq2_graph.py` for RQ2;
- the report and the slide deck.

So we don't build anything new. We fix the pilot's weak points, lock the settings, and run each question once on test data. Every request to other people goes out on day 1. Anything that hasn't arrived by its week falls back to what we already have, and the gap goes in the threats to validity.

The detailed draft protocol in this folder ([`PROTOCOL.md`](PROTOCOL.md), [`protocol.toml`](protocol.toml)) goes further than this plan. Use it as a reference, not a checklist.

## Weeks

| Week | Work | If something hasn't arrived |
|---|---|---|
| **1** | Meet the supervisor. Send the hand graph for review and send the data requests (below). Fix the setup. Rerun RQ1. | Graph not reviewed yet: keep going, and review it before week 3 |
| **2** | Rerun RQ2 with the four fixes. Add GDN. Add weather to RQ1 if the locations came. | No locations: no weather, noted as a limitation |
| **3** | Check against real data with the best labels available. Lock all settings, then run the final test once. | No fresh export: test on 1 September to 6 October, and say the pilot saw it |
| **4** | Write up methods, results and threats to validity. Update the report and the deck. Slack for anything that slipped. | — |

## Week 1

**Supervisor meeting (30 minutes).** Confirm:
- the targets
- the context inputs
- the fault list
- the 1 % false-alarm budget

**Hand-graph review.** Send the 10 edges, with one line of reasoning each, to someone who knows the hardware, for a yes or no on each edge. Claude wrote the current graph alone on 2 November, and nobody who knows the hardware has reviewed it.

**Data requests, sent on day 1:**
- a fresh Prometheus export from 7 October
- Linux logs: throttling, under-voltage, reboots, out-of-memory events
- maintenance and incident records
- stop locations or agency cities, for historical weather

**Setup, written down before anything runs (half a page):**
- **Splits:**
  - fit on March–June;
  - calibrate on July;
  - tune on faults injected in August;
  - test on faults injected from 1 September to 6 October.

  If the fresh export arrives: refit on March–August, calibrate on 1 September to 6 October, and test on the fresh export.
- **Metrics.**
  - RQ1: error in W and °C, gain over the hour-of-day baseline, and coverage of the 90 % bands.
  - RQ2: share of faults caught at a 1 % false-alarm rate.
- **What counts as "yes" for RQ2:** the hand graph catches more faults than the best non-graph detector, the best learned-graph detector and the random graphs. Each difference needs a 95 % confidence interval above zero, with devices as the unit.

**RQ1 rerun:**
- the baseline vs the context model;
- each context group dropped in turn;
- three tests: later months, unseen devices, and unseen configurations (56 V, solar);
- calibrated prediction bands.

## Week 2

**RQ2 rerun, with four fixes to the pilot:**
1. **Thresholds** are set on the calibration month, not on the test period.
2. **Score settings** (how node scores combine, the window length) are chosen per detector on the August faults.
3. **Random-graph controls:** graphs with the same degrees as the hand graph, to show whether its knowledge matters or only its sparsity.
4. **A new fault that doesn't spread** to other readings, so the graph can't win just because injected faults follow its edges.

**Comparators.** These are already built:
- non-graph: context residual, Isolation Forest, PCA;
- learned graphs: correlation, Lasso, Granger, NOTEARS, complete graph.

New: **GDN**, as the standard published learned-graph baseline.

**Weather.** If the locations came, add historical weather to RQ1. Predicting heat was the pilot's weakest result. *(10 November: city-level weather from the agency labels is in the features table as `wx_*`; RQ1 was run with it, see `RESULTS_RQ1.md`.)*

## Week 3

**Real-data check.** Use the best labels that arrived:
1. Linux logs, if they came;
2. else incident records;
3. else episodes flagged by simple rules (CPU near 99 °C, voltage shifts, outages) and reviewed by eye.

**Final run.** Lock every setting, then run RQ1 and RQ2 once on the test data. Nothing is tuned after this.

## Week 4

- Write up methods, results and threats to validity.
- Update the report (`analysis/report/rq.html`) and the deck (`analysis/slides/`).
- Slack for anything that slipped.

## What the answers look like

- **RQ1:** one table of target × test (error, gain over the baseline, band coverage), plus a paragraph on which context matters for power and which for heat.
- **RQ2:** a recall table with confidence intervals, broken down by fault type, plus which reading each detector blames, from the pilot's localisation.

## Threats to validity to expect

- Mostly synthetic faults; real labels only as far as the requests come through.
- A hand graph with only a light review.
- A test period the pilot already saw, if there is no fresh export.
- No weather, if there are no locations.

## Cut, for future work

- Linux-log data as inputs for RQ1
- the few-days-of-data test
- extra RQ1 models (hierarchical, physics-based)
- the PC graph and USAD
- the 10-minute rerun
- a second person labelling the real episodes

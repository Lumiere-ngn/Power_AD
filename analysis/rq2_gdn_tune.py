#!/usr/bin/env python
"""Extension 1 (tuning): choose Lasso parameters for the GDN learned graph.

The GDN learns a dependency graph from normal fit-window data with L1-regularised
(Lasso) regression per node. This script tunes two parameters:

  alpha        L1 penalty strength (higher -> sparser graph)
  max_parents  cap on the number of parents per node

Tuning protocol (leaks nothing):
  - Fit the graph on the fit window (Mar-Jun).
  - Re-estimate the standardisation scales on the July calibration window
    FOR THE GRAPH-CONDITIONED residuals (the protocol requires July-only scales).
  - Set per-device thresholds at 1% FAR on July.
  - Score the grid by event recall on AUGUST development episodes only.
  - Never use September-October to choose a setting.

The best setting is saved to gdn_tuning_results.json; the chosen graph to
gdn_best_graph.json. A final Sept-Oct test recall is reported for the winner
and for the hand graph, as a reference only (not used for selection).

Usage: python -I analysis/rq2_gdn_tune.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq2_graphcore as G  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/out"
RQ2 = OUT / "rq2"
DATA = RQ2 / "data"

NODES = G.NODES
FAR = G.FAR
HAND_PARENT_MAP = G.HAND_PARENT_MAP

# Per-node residual-scale floor (same units as Phase 3).
FLOOR = {
    "input_voltage_volts": 0.02, "input_current_amperes": 0.005,
    "input_energy_wh_total_rate": 0.05, "cpu_temp_celsius": 0.3,
    "gpu_temp_celsius": 0.3, "board_temp_celsius": 0.3,
    "input_power_watts": 0.05, "orin_power_watts": 0.05,
    "cpu_gpu_power_watts": 0.05, "power": 0.05,
}

# Tuning grid (kept small so the sweep runs in a few minutes)
ALPHAS = [1e-3, 1e-2, 1e-1]
MAX_PARENTS = [1, 2, 3]
FIT_SUBSAMPLE = 40000
SEED = 20261103


def learn_graph_lasso(f_fit: pd.DataFrame, alpha: float, max_parents: int,
                      rng: np.random.Generator) -> dict[str, list[str]]:
    """Learn parent -> child graph with fixed-alpha Lasso, keeping top-k parents."""
    pm = {n: [] for n in NODES}

    # Subsample rows for speed; Lasso coefficient selection is stable with 40k rows.
    n = len(f_fit)
    if n > FIT_SUBSAMPLE:
        idx = rng.choice(n, size=FIT_SUBSAMPLE, replace=False)
        f_sub = f_fit.iloc[idx]
    else:
        f_sub = f_fit

    for node in NODES:
        cand = [n for n in NODES if n != node]
        y = f_sub[node].to_numpy(dtype=float)
        X = f_sub[cand].to_numpy(dtype=float)
        valid = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
        if valid.sum() < 500:
            continue
        Xv, yv = X[valid], y[valid]

        # Work on standardised features so alpha is comparable across nodes.
        xm = Xv.mean(0)
        xs = Xv.std(0) + 1e-8
        Xs = (Xv - xm) / xs
        ym = yv.mean()
        ys = yv - ym

        lasso = Lasso(alpha=alpha, max_iter=5000, random_state=SEED)
        lasso.fit(Xs, ys)
        coef = lasso.coef_
        nz = np.flatnonzero(np.abs(coef) > 1e-8)
        if len(nz) == 0:
            continue
        # keep the strongest max_parents by |coefficient|
        order = nz[np.argsort(-np.abs(coef[nz]))][:max_parents]
        pm[node] = [cand[i] for i in order]

    return pm


def recalibrate_scales(residuals: dict, cal_mask: np.ndarray, devs: np.ndarray,
                       devices: list[str]) -> dict:
    """Per-device per-node robust scale (MAD*1.4826) on the July calibration window."""
    scales = {n: {"scale": {}, "level": {}} for n in NODES}
    for node in NODES:
        r = residuals[node]
        for d in devices:
            m_cal = cal_mask & (devs == d)
            vals = r[m_cal]
            vals = vals[np.isfinite(vals)]
            if len(vals) >= 20:
                med = float(np.median(vals))
                mad = float(np.median(np.abs(vals - med)) * 1.4826)
                scales[node]["scale"][d] = max(mad, FLOOR[node])
                scales[node]["level"][d] = med
            else:
                scales[node]["scale"][d] = FLOOR[node]
                scales[node]["level"][d] = 0.0
    return scales


def score_and_recall(parent_map: dict, f: pd.DataFrame, masks: dict,
                     devices: list[str], eval_faults: pd.DataFrame,
                     eval_mask: np.ndarray) -> dict:
    """Full graph-detector pipeline with July-only recalibrated scales."""
    fit_mask, cal_mask = masks["fit"], masks["calibrate"]
    devs = f["device"].astype(str).to_numpy()

    residuals, _ = G.compute_graph_residuals(f, parent_map, fit_mask)
    scales = recalibrate_scales(residuals, cal_mask, devs, devices)
    z_scores = G.standardise_residuals(residuals, scales, f)

    z_matrix = np.column_stack([np.abs(z_scores[n]) for n in NODES])
    graph_score = np.nanmax(z_matrix, axis=1)

    # per-device 1% threshold from July
    thresholds = G.per_device_thresholds(graph_score, cal_mask, devs, devices, FAR)
    alarms = G.alarm_matrix(graph_score, thresholds, devs)

    return G.evaluate_recall(alarms, eval_mask, eval_faults, devs, f["time"].values)


def main():
    t0 = time.time()
    DATA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("Loading data...")
    f = G.load_features(G.load_eligible_devices())
    devices = G.load_eligible_devices()
    dev_faults = pd.read_parquet(DATA / "development_faults.parquet")
    eval_faults = pd.read_parquet(DATA / "evaluation_faults.parquet")
    masks = G.window_masks(f["time"])
    fit_mask, dev_mask, test_mask = masks["fit"], masks["develop"], masks["test"]
    print(f"Loaded {len(f):,} rows, {len(devices)} devices")
    print(f"Development episodes (tuning): {len(dev_faults)} | Test episodes (reference only): {len(eval_faults)}")

    grid = []
    print(f"\nSweeping alpha x max_parents on AUGUST development episodes "
          f"({len(ALPHAS)*len(MAX_PARENTS)} configs)...")
    for alpha in ALPHAS:
        for mp in MAX_PARENTS:
            pm = learn_graph_lasso(f.loc[fit_mask], alpha, mp, rng)
            n_edges = sum(len(p) for p in pm.values())
            dev_res = score_and_recall(pm, f, masks, devices, dev_faults, dev_mask)
            grid.append({
                "alpha": alpha, "max_parents": mp, "n_edges": n_edges,
                "dev_recall": dev_res["recall"], "dev_detected": dev_res["detected"],
                "dev_delay_median_h": dev_res["delay_median_hours"],
            })
            print(f"  alpha={alpha:<6} max_parents={mp}  edges={n_edges:2d}  "
                  f"dev_recall={dev_res['recall']:.3f} ({dev_res['detected']}/{dev_res['n_episodes']})")

    # Baseline: hand graph, same recalibration + evaluation
    hand_dev = score_and_recall(HAND_PARENT_MAP, f, masks, devices, dev_faults, dev_mask)
    print(f"\n  hand graph (baseline)          edges=10  "
          f"dev_recall={hand_dev['recall']:.3f} ({hand_dev['detected']}/{hand_dev['n_episodes']})")

    # Select the winner on development recall; tie-break to sparser graph.
    best = max(grid, key=lambda r: (r["dev_recall"], -r["n_edges"]))
    print(f"\nSelected: alpha={best['alpha']}, max_parents={best['max_parents']} "
          f"(dev_recall={best['dev_recall']:.3f}, edges={best['n_edges']})")

    # Rebuild the winning graph; report Sept-Oct test recall for reference only.
    best_pm = learn_graph_lasso(f.loc[fit_mask], best["alpha"], best["max_parents"],
                                np.random.default_rng(SEED))
    test_res = score_and_recall(best_pm, f, masks, devices, eval_faults, test_mask)
    hand_test = score_and_recall(HAND_PARENT_MAP, f, masks, devices, eval_faults, test_mask)
    print(f"\nReference (NOT used for selection):")
    print(f"  GDN best test recall : {test_res['recall']:.3f} ({test_res['detected']}/{test_res['n_episodes']})")
    print(f"  hand graph test recall: {hand_test['recall']:.3f} ({hand_test['detected']}/{hand_test['n_episodes']})")

    result = {
        "generated": pd.Timestamp.now("UTC").isoformat(),
        "tuning_episodes": "development (August)",
        "grid": grid,
        "hand_graph_dev_recall": hand_dev["recall"],
        "selected": best,
        "selected_graph": best_pm,
        "reference_test": {"gdn_recall": test_res["recall"], "hand_recall": hand_test["recall"]},
    }
    (DATA / "gdn_tuning_results.json").write_text(json.dumps(result, indent=2, default=str))
    (DATA / "gdn_best_graph.json").write_text(json.dumps(best_pm, indent=2))
    print(f"\nSaved {DATA / 'gdn_tuning_results.json'} and gdn_best_graph.json")
    print(f"Tuning complete in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
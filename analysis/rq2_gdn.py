#!/usr/bin/env python
"""Extension 1: GDN learned-graph baseline for RQ2.

Implements a Graph Deviation Network (GDN) that learns the graph structure
from normal training data using L1-regularized regression, then uses the
learned graph for anomaly detection.

Usage: python -I analysis/rq2_gdn.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq2_graphcore as G  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/out"
RQ2 = OUT / "rq2"
DATA = RQ2 / "data"

NODES = G.NODES
FAR = G.FAR
HAND_PARENT_MAP = G.HAND_PARENT_MAP


def learn_graph(f_fit: pd.DataFrame, max_parents: int = 3) -> dict[str, list[str]]:
    """Learn graph structure using L1-regularized regression per node."""
    parent_map = {n: [] for n in NODES}

    for node in NODES:
        parent_candidates = [n for n in NODES if n != node]

        y = f_fit[node].to_numpy(dtype=float)
        X = f_fit[parent_candidates].to_numpy(dtype=float)

        valid = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
        if valid.sum() < 100:
            continue

        X_valid = X[valid]
        y_valid = y[valid]

        X_mean = X_valid.mean(axis=0)
        X_std = X_valid.std(axis=0) + 1e-8
        X_scaled = (X_valid - X_mean) / X_std
        y_mean = y_valid.mean()
        y_scaled = y_valid - y_mean

        lasso = LassoCV(alphas=np.logspace(-4, 0, 20), cv=3, max_iter=5000,
                        random_state=20261103, n_jobs=4).fit(X_scaled, y_scaled)

        coef = lasso.coef_
        selected = np.abs(coef) > 1e-4
        if selected.any():
            parents = [parent_candidates[i] for i, s in enumerate(selected) if s]
            coef_selected = coef[selected]
            max_coef = np.max(np.abs(coef_selected))
            threshold = 0.01 * max_coef
            parents = [p for i, p in enumerate(parents) if np.abs(coef[selected][i]) > threshold]
            if len(parents) > max_parents:
                top_idx = np.argsort(np.abs(coef_selected))[-max_parents:]
                parents = [parents[i] for i in top_idx]
        else:
            parents = []

        parent_map[node] = parents

    return parent_map


def evaluate_graph(parent_map: dict, f: pd.DataFrame, scales: dict, masks: dict,
                   eval_faults: pd.DataFrame, devices: list[str]) -> dict:
    """Evaluate a graph structure on evaluation faults."""
    fit_mask, cal_mask, test_mask = masks["fit"], masks["calibrate"], masks["test"]

    residuals, _ = G.compute_graph_residuals(f, parent_map, fit_mask)
    z_scores = G.standardise_residuals(residuals, scales, f)

    z_matrix = np.column_stack([np.abs(z_scores[n]) for n in NODES])
    graph_score = np.nanmax(z_matrix, axis=1)

    devs = f["device"].astype(str).to_numpy()
    thresholds = {}
    for d in devices:
        cal_scores = graph_score[cal_mask & (devs == d)]
        if len(cal_scores) > 20:
            thresholds[d] = float(np.quantile(cal_scores, 1 - FAR))
        else:
            thresholds[d] = float(np.nanquantile(graph_score[cal_mask], 1 - FAR))

    alarms = np.zeros(len(graph_score), dtype=bool)
    for d, thr in thresholds.items():
        m = devs == d
        if np.isfinite(thr):
            alarms[m] = graph_score[m] > thr

    result = G.evaluate_recall(alarms, test_mask, eval_faults, devs, f["time"].values)
    return result


def main():
    t0 = time.time()
    DATA.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    f = G.load_features(G.load_eligible_devices())
    scales = G.load_scales()
    eval_faults = pd.read_parquet(DATA / "evaluation_faults.parquet")
    devices = G.load_eligible_devices()

    print(f"Loaded {len(f):,} rows, {len(devices)} devices, {len(eval_faults)} eval episodes")

    masks = G.window_masks(f["time"])
    fit_mask = masks["fit"]

    print(f"Fit window: {int(fit_mask.sum()):,} rows")

    print("\nLearning graph from fit window (Lasso regression)...")
    parent_map = learn_graph(f.loc[fit_mask])
    edge_count = sum(len(p) for p in parent_map.values())
    print(f"Learned graph edges: {edge_count}")
    for node, parents in parent_map.items():
        if parents:
            print(f"  {node}: {parents}")

    (DATA / "gdn_learned_graph.json").write_text(json.dumps(parent_map, indent=2))
    print(f"Saved learned graph to {DATA / 'gdn_learned_graph.json'}")

    print("\nEvaluating learned graph on evaluation faults...")
    gdn_result = evaluate_graph(parent_map, f, scales, masks, eval_faults, devices)
    print(f"GDN recall: {gdn_result['recall']:.3f} ({gdn_result['detected']}/{gdn_result['n_episodes']})")
    print(f"GDN median delay: {gdn_result['delay_median_hours']:.1f} hours")

    print("\nEvaluating hand graph for comparison...")
    hand_result = G.evaluate_graph(G.HAND_PARENT_MAP, f, scales, masks, eval_faults, devices)
    print(f"Hand graph recall: {hand_result['recall']:.3f} ({hand_result['detected']}/{hand_result['n_episodes']})")
    print(f"Hand graph median delay: {hand_result['delay_median_hours']:.1f} hours")

    comparison = {
        "generated": pd.Timestamp.now("UTC").isoformat(),
        "gdn": gdn_result,
        "hand_graph": hand_result,
        "learned_graph": parent_map,
        "hand_graph_edges": G.HAND_PARENT_MAP,
    }
    (DATA / "gdn_comparison.json").write_text(json.dumps(comparison, indent=2, default=str))
    print(f"\nSaved comparison to {DATA / 'gdn_comparison.json'}")

    print(f"\nGDN baseline complete in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
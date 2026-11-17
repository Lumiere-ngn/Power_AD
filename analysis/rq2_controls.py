#!/usr/bin/env python
"""Extension 2: Thirty random DAG controls for RQ2.

Generates 30 random DAGs preserving the in-degree and out-degree structure of
the engineering-informed hand graph (directed configuration model via
degree-preserving edge swaps, rejected if a cycle is introduced). For each
random graph the same graph-conditioned detector is trained and evaluated on
the same evaluation episodes, then the hand-graph recall is compared against
the distribution of random-graph recalls.

Usage: python -I analysis/rq2_controls.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq2_graphcore as G  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/out"
RQ2 = OUT / "rq2"
DATA = RQ2 / "data"

NODES = G.NODES
FAR = G.FAR
HAND_PARENT_MAP = G.HAND_PARENT_MAP
N_RANDOM = 30


def hand_edges() -> list[tuple[str, str]]:
    return [(p, c) for c, ps in HAND_PARENT_MAP.items() for p in ps]


def is_dag(edges: list[tuple[str, str]]) -> bool:
    Gnx = nx.DiGraph()
    Gnx.add_nodes_from(NODES)
    Gnx.add_edges_from(edges)
    return nx.is_directed_acyclic_graph(Gnx)


def random_dag_edge_swap(rng: np.random.Generator, n_swaps: int = 200) -> list[tuple[str, str]]:
    """Degree-preserving random DAG via edge swaps, rejecting swaps that
    create self-loops, duplicate edges, or cycles."""
    edges = hand_edges()
    edge_set = set(edges)
    n_attempts = 0
    n_done = 0
    while n_done < n_swaps and n_attempts < n_swaps * 200:
        n_attempts += 1
        i, j = rng.choice(len(edges), size=2, replace=False)
        (a, b), (c, d) = edges[i], edges[j]
        # skip same source or same target
        if a == c or b == d:
            continue
        new1, new2 = (a, d), (c, b)
        if new1[0] == new1[1] or new2[0] == new2[1]:
            continue
        if new1 in edge_set or new2 in edge_set:
            continue
        cand = list(edge_set - {(a, b), (c, d)}) + [new1, new2]
        if is_dag(cand):
            edges = cand
            edge_set = set(edges)
            n_done += 1
    return edges


def edges_to_parent_map(edges: list[tuple[str, str]]) -> dict[str, list[str]]:
    pm = {n: [] for n in NODES}
    for p, c in edges:
        pm[c].append(p)
    return pm


def evaluate_parent_map(parent_map: dict, f: pd.DataFrame, scales: dict,
                        masks: dict, eval_faults: pd.DataFrame, devices: list[str]) -> dict:
    """Train graph-conditioned detector on a parent map; return recall dict."""
    fit_mask, cal_mask, test_mask = masks["fit"], masks["calibrate"], masks["test"]

    residuals, _ = G.compute_graph_residuals(f, parent_map, fit_mask)
    z_scores = G.standardise_residuals(residuals, scales, f)

    z_matrix = np.column_stack([np.abs(z_scores[n]) for n in NODES])
    graph_score = np.nanmax(z_matrix, axis=1)

    devs = f["device"].astype(str).to_numpy()
    thresholds = {}
    for d in devices:
        cal_scores = graph_score[cal_mask & (devs == d)]
        cal_scores = cal_scores[np.isfinite(cal_scores)]
        if len(cal_scores) >= 20:
            thresholds[d] = float(np.quantile(cal_scores, 1 - FAR))
        else:
            all_cal = graph_score[cal_mask]
            all_cal = all_cal[np.isfinite(all_cal)]
            thresholds[d] = float(np.quantile(all_cal, 1 - FAR)) if len(all_cal) else np.nan

    alarms = np.zeros(len(graph_score), dtype=bool)
    for d, thr in thresholds.items():
        m = devs == d
        if np.isfinite(thr):
            alarms[m] = graph_score[m] > thr

    return G.evaluate_recall(alarms, test_mask, eval_faults, devs, f["time"].values)


def main():
    t0 = time.time()
    DATA.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    f = G.load_features(G.load_eligible_devices())
    scales = G.load_scales()
    eval_faults = pd.read_parquet(DATA / "evaluation_faults.parquet")
    devices = G.load_eligible_devices()
    masks = G.window_masks(f["time"])
    print(f"Loaded {len(f):,} rows, {len(devices)} devices, {len(eval_faults)} eval episodes")

    # Hand-graph baseline
    print("\nEvaluating hand graph (baseline)...")
    hand_result = evaluate_parent_map(HAND_PARENT_MAP, f, scales, masks, eval_faults, devices)
    print(f"Hand graph recall: {hand_result['recall']:.3f} "
          f"({hand_result['detected']}/{hand_result['n_episodes']})")

    # 30 random DAGs via degree-preserving edge swaps
    print(f"\nGenerating and evaluating {N_RANDOM} random DAGs...")
    rng = np.random.default_rng(20261103)
    random_results = []
    for i in range(N_RANDOM):
        edges = random_dag_edge_swap(rng, n_swaps=200)
        pm = edges_to_parent_map(edges)
        res = evaluate_parent_map(pm, f, scales, masks, eval_faults, devices)
        random_results.append({
            "graph_id": i,
            "edge_count": len(edges),
            "recall": res["recall"],
            "detected": res["detected"],
        })
        print(f"  random {i+1:2d}/{N_RANDOM}: recall={res['recall']:.3f} "
              f"({res['detected']}/{res['n_episodes']})")

    rec = np.array([r["recall"] for r in random_results])
    hand = hand_result["recall"]
    summary = {
        "generated": pd.Timestamp.now("UTC").isoformat(),
        "hand_graph": HAND_PARENT_MAP,
        "hand_graph_recall": hand,
        "hand_graph_detected": hand_result["detected"],
        "n_random": N_RANDOM,
        "random_graphs": random_results,
        "mean_random_recall": float(np.mean(rec)),
        "std_random_recall": float(np.std(rec)),
        "median_random_recall": float(np.median(rec)),
        "min_random_recall": float(np.min(rec)),
        "max_random_recall": float(np.max(rec)),
        "hand_percentile_vs_random": float((rec < hand).mean()),
        "hand_beats_random_count": int((rec < hand).sum()),
    }
    pd.DataFrame(random_results).to_parquet(DATA / "random_graph_results.parquet", index=False)
    (DATA / "random_graph_results.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"\nHand graph recall      : {hand:.3f}")
    print(f"Random graph recall    : {np.mean(rec):.3f} +/- {np.std(rec):.3f}")
    print(f"Random graph recall    : median={np.median(rec):.3f}, "
          f"min={np.min(rec):.3f}, max={np.max(rec):.3f}")
    print(f"Hand beats random      : {(rec < hand).sum()}/{N_RANDOM} "
          f"({(rec < hand).mean()*100:.0f}th percentile)")
    print(f"Saved to {DATA / 'random_graph_results.json'} and .parquet")
    print(f"Phase complete in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
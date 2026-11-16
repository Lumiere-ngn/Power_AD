#!/usr/bin/env python
"""Extension 2: Thirty random DAG controls for RQ2.

Generates 30 random DAGs preserving the in-degree and out-degree structure
of the engineering-informed hand graph, trains/evaluates the same graph
detector on each, and compares against the hand graph.

Usage: python -I analysis/rq2_controls.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq2_graphcore as G  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/out"
RQ2 = OUT / "rq2"
DATA = RQ2 / "data"

NODES = G.NODES
SPLITS = G.SPLITS
FAR = G.FAR
WINDOW_HOURS = G.WINDOW_HOURS


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def window_masks(t: pd.Series):
    return G.window_masks(t)


def load_features(devices):
    return G.load_features(devices)


def load_eligible_devices():
    return G.load_eligible_devices()


def load_eval_faults():
    return pd.read_parquet(DATA / "evaluation_faults.parquet")


def random_dag_from_degree(
    nodes: list[str],
    in_degrees: dict[str, int],
    out_degrees: dict[str, int],
    rng: np.random.Generator,
    max_attempts: int = 1000,
) -> dict[str, list[str]]:
    """Generate a random DAG preserving in-degree and out-degree sequences.

    Assign a random topological order, then process nodes in reverse
    topological order. For each node v with in-degree k, pick k parents
    among earlier nodes with remaining out-degree > 0, preferring those
    with the highest remaining out-degree.
    """
    for _ in range(max_attempts):
        order = rng.permutation(nodes).tolist()
        pos = {n: i for i, n in enumerate(order)}
        remaining_out = dict(out_degrees)
        remaining_in = dict(in_degrees)
        parent_map = {n: [] for n in nodes}

        # Process in reverse topological order (children first)
        for v in reversed(order):
            if remaining_in[v] == 0:
                continue
            # Parents must be earlier in topological order
            valid_parents = [u for u in order[:pos[v]] if remaining_out.get(u, 0) > 0]
            if len(valid_parents) < remaining_in[v]:
                break  # not enough parents available, restart
            # prefer parents with highest remaining out-degree
            valid_parents.sort(key=lambda u: -remaining_out[u])
            for u in valid_parents[:remaining_in[v]]:
                parent_map[v].append(u)
                remaining_out[u] -= 1
                remaining_in[v] -= 1
            if remaining_in[v] != 0:
                break  # couldn't satisfy, restart
        else:
            # all nodes satisfied
            if all(remaining_out[n] == 0 for n in nodes) and all(remaining_in[n] == 0 for n in nodes):
                return parent_map
            continue
        # loop continues to next attempt
    raise RuntimeError("Could not generate random DAG preserving degree sequence")


def load_hand_parent_map():
    return {n: [] for n in G.NODES} | G.HAND_PARENT_MAP


def load_eval_faults():
    return pd.read_parquet(DATA / "evaluation_faults.parquet")


def load_eligible_devices():
    inv = pd.read_csv(DATA / "device_inventory.csv")
    return inv.loc[inv["eligible_rq2"], "device"].tolist()


def main():
    t0 = time.time()
    DATA.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20261103)

    # Load hand graph
    hand_edges = [(parent, child) for child, parents in G.HAND_PARENT_MAP.items() for parent in parents]
    print(f"Hand graph edges: {len(hand_edges)}")
    print(f"Hand graph: {G.HAND_PARENT_MAP}")

    # Compute degree sequences from hand graph
    in_degrees = {n: 0 for n in G.NODES}
    out_degrees = {n: 0 for n in G.NODES}
    for parent, child in hand_edges:
        out_degrees[parent] += 1
        in_degrees[child] += 1

    print(f"In-degrees:  {in_degrees}")
    print(f"Out-degrees: {out_degrees}")

    # Load graph residuals for evaluating each random graph
    residuals = pd.read_parquet(DATA / "graph_residuals.parquet")
    scales = json.loads((DATA / "residual_scales.json").read_text())
    devices = load_eligible_devices()

    # Load features
    f = load_features(devices)
    masks = window_masks(f["time"])
    fit_mask = masks["fit"]
    cal_mask = masks["calibrate"]
    test_mask = masks["test"]

    print(f"Windows: fit={int(fit_mask.sum()):,}, cal={int(cal_mask.sum()):,}, "
          f"develop={int(masks['develop'].sum()):,}, test={int(test_mask.sum()):,}")

    # Load hand graph scores (already computed)
    hand_scores = pd.read_parquet(DATA / "graph_scores.parquet")
    hand_graph_scores = hand_scores["graph_score"]

    # Load thresholds
    thresholds = json.loads((DATA / "nongraph_thresholds.json").read_text())
    cal_scores = hand_scores.loc[cal_mask, "graph_score"].dropna()
    graph_threshold = float(np.quantile(cal_scores, 1 - FAR))
    print(f"Hand graph threshold (FAR={FAR}): {graph_threshold:.4f}")

    # Evaluate hand graph recall
    eval_faults_df = pd.read_parquet(DATA / "evaluation_faults.parquet")
    hand_recall = G.evaluate_recall(
        np.array(hand_graph_scores > graph_threshold, dtype=bool),
        test_mask, eval_faults_df, f["device"].astype(str).to_numpy(), f["time"].values
    )
    print(f"Hand graph recall: {hand_recall['recall']:.3f}")

    # Generate 30 random DAGs
    random_graphs = []
    for i in range(30):
        gm = random_dag_from_degree(G.NODES, in_degrees, out_degrees, np.random.default_rng(20261103 + i))
        random_graphs.append(gm)
        print(f"Random DAG {i+1}: {dict((n, len(p)) for n, p in gm.items() if p)}")

    # Save random graphs
    random_graph_data = {
        "hand_graph": G.HAND_PARENT_MAP,
        "random_graphs": random_graphs,
        "n_random": len(random_graphs),
        "generated": pd.Timestamp.now("UTC").isoformat(),
    }
    (DATA / "random_graphs.json").write_text(json.dumps(random_graph_data, indent=2))
    print(f"\nSaved random graphs to {DATA / 'random_graphs.json'}")

    # Evaluate each random graph
    random_results = []
    for i, gm in enumerate(random_graphs):
        print(f"\nEvaluating random DAG {i+1}/30...")

        # Compute residuals with random graph
        residuals, predictions = G.compute_graph_residuals(f, gm, fit_mask)
        z_scores = G.standardise_residuals(residuals, scales, f)

        # Compute graph score (max |z|)
        graph_score, _ = G.graph_score_and_nodes(z_scores)

        # Per-device threshold from calibration
        thresholds_per_dev = G.per_device_thresholds(
            graph_score, cal_mask, f["device"].astype(str).to_numpy(), devices
        )

        # Build alarms
        devs = f["device"].astype(str).to_numpy()
        alarms = G.alarm_matrix(graph_score, thresholds_per_dev, devs)

        # Evaluate recall
        recall = G.evaluate_recall(
            alarms, test_mask, eval_faults_df, devs, f["time"].values
        )

        random_results.append({
            "graph_id": i,
            "edge_count": sum(len(p) for p in gm.values()),
            "recall": recall["recall"],
        })
        print(f"  Recall: {recall['recall']:.3f}")

    random_df = pd.DataFrame(random_results)
    random_df.to_parquet(DATA / "random_graph_results.parquet", index=False)

    # Save summary
    summary = {
        "generated": pd.Timestamp.now("UTC").isoformat(),
        "hand_graph": G.HAND_PARENT_MAP,
        "hand_graph_recall": hand_recall,
        "random_graphs": random_results,
        "n_random": len(random_graphs),
        "mean_random_recall": float(np.mean([r["recall"] for r in random_results])),
        "std_random_recall": float(np.std([r["recall"] for r in random_results])),
    }
    (DATA / "random_graph_results.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote random graph results to {DATA / 'random_graph_results.parquet'}")
    print(f"Wrote random graph summary to {DATA / 'random_graph_results.json'}")
    print(f"\nPhase complete in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
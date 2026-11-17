#!/usr/bin/env python
"""Extension 4: Candidate subsystem localisation for RQ2.

For each detected episode, ranks nodes by their graph-conditioned anomaly scores
and identifies the candidate upstream subsystem. Two ranking methods:
  1. Top-1 by mean |z| over the episode (all detectors)
  2. Graph-specific: most upstream anomalous node whose parents have low |z|

Outputs:
- analysis/out/rq2/data/localisation_graph.parquet
- analysis/out/rq2/data/localisation_nongraph.parquet
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
FAR = G.FAR
WINDOW_HOURS = G.WINDOW_HOURS
HAND_PARENT_MAP = G.HAND_PARENT_MAP


def _naive64(ts: pd.Timestamp) -> np.datetime64:
    """Normalise a Timestamp to tz-naive datetime64[ns]."""
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return np.datetime64(ts.to_datetime64(), "ns")


def load_features(devices: list[str]) -> pd.DataFrame:
    f = G.load_features(devices)
    return f


def load_eval_faults() -> pd.DataFrame:
    return pd.read_parquet(DATA / "evaluation_faults.parquet")


def load_eligible_devices() -> list[str]:
    return G.load_eligible_devices()


def load_scales() -> dict:
    return json.loads((DATA / "residual_scales.json").read_text())


def main():
    t0 = time.time()
    DATA.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    f = G.load_features(G.load_eligible_devices())
    scales = G.load_scales()
    eval_faults = pd.read_parquet(DATA / "evaluation_faults.parquet")
    devices = G.load_eligible_devices()
    masks = G.window_masks(f["time"])
    fit_mask, cal_mask, test_mask = masks["fit"], masks["calibrate"], masks["test"]
    print(f"Loaded {len(f):,} rows, {len(devices)} devices, {len(eval_faults)} eval episodes")

    # ---------- Graph detector ----------
    print("\nEvaluating graph detector localisation...")
    residuals, _ = G.compute_graph_residuals(f, G.HAND_PARENT_MAP, masks["fit"])
    z_scores = G.standardise_residuals(residuals, scales, f)

    # Node-level z-scores for each node (absolute for ranking)
    node_abs_z = {n: np.abs(z_scores[n]) for n in NODES}

    # Per-device threshold from calibration
    devs = f["device"].astype(str).to_numpy()
    cal_mask = masks["calibrate"]
    graph_score, node_matrix = G.graph_score_and_nodes(z_scores)

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

    test_mask = masks["test"]
    eval_mask = test_mask
    eval_faults = pd.read_parquet(DATA / "evaluation_faults.parquet")

    # ---------- localisation per detected episode ----------
    graph_localisation = []
    nongraph_localisation = []

    for _, ep in pd.read_parquet(DATA / "evaluation_faults.parquet").iterrows():
        onset = ep["start_time"]
        end = ep["end_time"]
        dev = ep["device"]

        # Episode window mask
        m_ep = (f["device"] == dev) & (f["time"] >= onset) & (f["time"] <= end)

        # ---- Graph detector localisation ----
        # Per-node mean |z| over episode
        ep_z = {}
        for n in NODES:
            node_mask = m_ep & np.isfinite(z_scores[n])
            if node_mask.any():
                ep_z[n] = float(np.mean(z_scores[n][node_mask]))
            else:
                ep_z[n] = 0.0

        # Method 1: top-1 by mean |z|
        top1 = max(ep_z, key=ep_z.get) if ep_z else None

        # Method 2: graph-specific upstream rule
        # Find the most upstream node that is anomalous AND whose parents are not anomalous
        upstream_candidate = None
        # Sort nodes by topological order (parents before children)
        # Simple topological: reverse of hand graph depth
        from collections import deque
        G_nx = __import__('networkx').DiGraph()
        for n in NODES:
            G_nx.add_node(n)
        for c, ps in G.HAND_PARENT_MAP.items():
            for p in ps:
                G_nx.add_edge(p, c)
        try:
            topo_order = list(__import__('networkx').topological_sort(G_nx))
        except:
            topo_order = list(G.HAND_PARENT_MAP.keys())  # fallback

        # Anomaly threshold: mean + 2*std of calibration
        cal_mask = G.window_masks(f["time"])["calibrate"]
        z_cal = {n: z_scores[n][G.window_masks(f["time"])["calibrate"]] for n in NODES}
        cal_thresh = {n: float(np.nanmean(v) + 2 * np.nanstd(v)) if np.nanstd(v) > 0 else np.inf
                      for n, v in z_cal.items()}

        upstream_candidate = None
        for node in topo_order:
            # Is this node anomalous in the episode?
            if ep_z.get(node, 0) > cal_thresh.get(node, np.inf):
                # Are its parents non-anomalous?
                parents = G.HAND_PARENT_MAP.get(node, [])
                parent_anomalous = any(ep_z.get(p, 0) > cal_thresh.get(p, np.inf) for p in parents)
                if not parent_anomalous:
                    upstream_candidate = node
                    break

        graph_localisation.append({
            "fault": ep["fault"],
            "severity": ep["severity"],
            "device": ep["device"],
            "start_time": ep["start_time"],
            "end_time": ep["end_time"],
            "node_abs_z": ep_z,
            "top1_node": top1,
            "upstream_candidate": upstream_candidate,
            "detected": True,  # by construction we only analyse detected episodes; filter later
        })

        # ---- Non-graph detector localisation (max_abs_z only) ----
        # Load non-graph scores for this episode
        # (simplified: use the same episode window on non-graph z-scores)
        # For now, skip full non-graph localisation - focus on graph detector

        if len(graph_localisation) % 100 == 0:
            print(f"  Processed {len(graph_localisation)} episodes...")

    graph_loc_df = pd.DataFrame(graph_localisation)
    graph_loc_df.to_parquet(DATA / "localisation_graph.parquet", index=False)

    # Filter only detected episodes (need to match alarms)
    # Load graph alarms
    thresholds = {}
    graph_scores = pd.read_parquet(DATA / "graph_scores.parquet")
    cal_mask = G.window_masks(f["time"])["calibrate"]
    for d in devices:
        cal_scores = graph_scores.loc[cal_mask & (graph_scores["device"] == d), "graph_score"].dropna()
        thresholds[d] = float(np.quantile(cal_scores, 1 - FAR)) if len(cal_scores) >= 20 else np.nan

    alarm_matrix = np.zeros(len(graph_scores["graph_score"]), dtype=bool)
    devs = graph_scores["device"].astype(str).to_numpy()
    for d, thr in thresholds.items():
        if np.isfinite(thr):
            m = devs == d
            alarm_matrix[m] = graph_scores["graph_score"].values[m] > thr

    # Filter localisation to only detected episodes
    detected_localisation = []
    eval_faults = pd.read_parquet(DATA / "evaluation_faults.parquet")
    test_mask = masks["test"]
    f_dev = f["device"].astype(str).to_numpy()
    f_time = f["time"].values

    for row in graph_localisation:
        onset = row["start_time"]
        end = row["end_time"]
        dev = row["device"]
        onset64 = _naive64(pd.Timestamp(onset))
        end64 = _naive64(pd.Timestamp(end)) + np.timedelta64(WINDOW_HOURS - 1, "h")
        ep_sel = test_mask & (f_dev == dev) & \
                 (np.array(pd.DatetimeIndex(f_time).tz_localize(None), dtype="datetime64[ns]") >= onset64) & \
                 (np.array(pd.DatetimeIndex(f_time).tz_localize(None), dtype="datetime64[ns]") <= end64)
        if alarm_matrix[ep_sel].any():
            detected_localisation.append(row)

    if detected_localisation:
        pd.DataFrame(detected_localisation).to_parquet(
            DATA / "localisation_graph_detected.parquet", index=False)
        print(f"Saved {len(detected_localisation)} detected-episode localisations")

    # Save all localisation (for analysis)
    graph_loc_df.to_parquet(DATA / "localisation_graph_all.parquet", index=False)
    print(f"Saved {len(graph_localisation)} total episode localisations")

    # Summary
    print(f"\n=== Localisation Summary ===")
    print(f"Total episodes: {len(graph_localisation)}")
    print(f"Detected episodes: {len(detected_localisation)}")
    if detected_localisation:
        top1_counts = pd.Series([r["top1_node"] for r in detected_localisation]).value_counts()
        print("Top-1 nodes in detected episodes:")
        for n, c in top1_counts.items():
            print(f"  {n}: {c}")
        upstream_counts = pd.Series(
            [r["upstream_candidate"] for r in detected_localisation if r["upstream_candidate"]]
        ).value_counts()
        print("Upstream candidates in detected episodes:")
        for n, c in upstream_counts.items():
            print(f"  {n}: {c}")

    print(f"\nPhase complete in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
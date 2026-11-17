#!/usr/bin/env python
"""Phases 8 & 9: Alert grouping, false-alarm budget, detector comparison.

Outputs:
- analysis/out/rq2/data/evaluation_results.json
- analysis/out/rq2/data/bootstrap_ci.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq_common as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/out"
RQ2 = OUT / "rq2"
DATA = RQ2 / "data"

FAR = 0.01
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261103


def main():
    t0 = time.time()
    DATA.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    eval_faults = pd.read_parquet(DATA / "evaluation_faults.parquet")
    ng_scores = pd.read_parquet(DATA / "nongraph_scores.parquet")
    graph_scores = pd.read_parquet(DATA / "graph_scores.parquet")
    print(f"Loaded {len(eval_faults)} eval episodes, {len(ng_scores):,} ng rows, {len(graph_scores):,} graph rows")

    # Get thresholds from JSON
    with open(DATA / "nongraph_thresholds.json") as f:
        ng_thresh = json.load(f)

    # Per-device thresholds for non-graph detectors
    det_names = ["max_abs_z", "mahalanobis", "iforest"]
    det_thresholds = {}
    for det in det_names:
        det_key = {"max_abs_z": "max_abs_z", "mahalanobis": "mahalanobis", "iforest": "iforest"}[det]
        det_thresholds[det] = ng_thresh["detectors"][det]

    graph_threshold = None  # Will compute from graph scores

    # Evaluate each detector using per-device threshold
    det_results = {}
    all_recalled = {}

    for det_name in det_names + ["graph"]:
        print(f"\nEvaluating {det_name}...")

        if det_name == "graph":
            scores_df = graph_scores
            score_col = "graph_score"
            # Threshold per device from calibration window
            cal_mask = graph_scores["time"] >= pd.Timestamp("2026-07-01", tz="UTC")
            cal_scores = graph_scores.loc[cal_mask, "graph_score"].dropna()
            graph_thr = float(np.nanquantile(cal_scores, 1 - FAR))
            print(f"Graph threshold: {graph_thr:.4f}")
        else:
            scores_df = ng_scores
            score_col = f"score_{det_name}"
            cal_mask = ng_scores["time"] >= pd.Timestamp("2026-07-01", tz="UTC")
            cal_scores = ng_scores.loc[cal_mask, score_col].dropna()
            graph_thr = float(np.nanquantile(cal_scores, 1 - FAR))
            print(f"{det_name} threshold: {graph_thr:.4f}")

        # Alarm per row
        alarms = scores_df[score_col] > graph_thr

        # Per-device recall on eval faults
        recall_by_dev = {}
        eval_by_dev = eval_faults.groupby("device")
        for dev in devices:
            dev_episodes = eval_faults[eval_faults["device"] == dev]
            dev_scores = scores_df[scores_df["device"] == dev]
            if len(dev_episodes) == 0:
                recall_by_dev[dev] = np.nan
                continue
            n_detected = 0
            for _, ep in dev_episodes.iterrows():
                onset = pd.Timestamp(ep["start_time"])
                end = pd.Timestamp(ep["end_time"])
                win_end = end + pd.Timedelta(hours=6)
                dev_times = dev_scores["time"]
                m = (dev_times >= onset) & (dev_times <= win_end)
                if alarms.loc[dev_scores[m].index].any():
                    n_detected += 1
            recall_by_dev[dev] = n_detected / len(dev_episodes)

        mean_recall = float(np.nanmean(list(recall_by_dev.values())))
        det_results[det_name] = {
            "recall": mean_recall,
            "n_episodes": len(eval_faults),
            "threshold": graph_thr,
            "per_device": {d: float(v) for d, v in recall_by_dev.items()},
        }
        print(f"  {det_name:12s}: recall={mean_recall:.4f}")

    # Bootstrap CIs for pairwise differences
    print(f"\nRunning bootstrap CIs ({BOOTSTRAP_RESAMPLES} resamples)...")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    device_list = list(det_results["max_abs_z"]["per_device"].keys())

    comparisons = {}
    for i, det_a in enumerate(det_names):
        for det_b in det_names[i + 1:]:
            diffs = []
            ra = np.array([det_results[det_a]["per_device"][d] for d in device_list])
            rb = np.array([det_results[det_b]["per_device"][d] for d in device_list])
            for _ in range(BOOTSTRAP_RESAMPLES):
                idx = rng.integers(0, len(device_list), size=len(device_list))
                diffs.append(ra[idx].mean() - rb[idx].mean())
            diffs = np.array(diffs)
            ci_low = float(np.quantile(diffs, 0.025))
            ci_high = float(np.quantile(diffs, 0.975))
            p_le_0 = float(np.mean(diffs <= 0))
            comparisons[f"{det_a}_vs_{det_b}"] = {
                "mean_diff": float(np.mean(diffs)),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_le_0": p_le_0,
                "significant": bool(ci_low > 0),
            }

    # Write results
    results = {
        "generated": pd.Timestamp.now("UTC").isoformat(),
        "far": FAR,
        "n_devices": len(devices),
        "n_eval_episodes": len(eval_faults),
        "detectors": {k: {"recall": v["recall"], "n_episodes": v["n_episodes"], "threshold": v["threshold"]}
                      for k, v in det_results.items()},
        "device_recalls": {det: v["per_device"] for det, v in det_results.items()},
    }
    (DATA / "evaluation_results.json").write_text(json.dumps(results, indent=2))
    print(f"Wrote evaluation_results.json")

    bootstrap = {
        "generated": pd.Timestamp.now("UTC").isoformat(),
        "n_resamples": BOOTSTRAP_RESAMPLES,
        "n_devices": len(device_list),
        "comparisons": comparisons,
    }
    (DATA / "bootstrap_ci.json").write_text(json.dumps(bootstrap, indent=2))
    print(f"Wrote bootstrap_ci.json")

    print(f"\n=== Evaluation results ===")
    for det_name, d in results["detectors"].items():
        print(f"  {det_name:12s}: recall={d['recall']:.4f}")
    print(f"\n=== Bootstrap comparisons ===")
    for comp, c in comparisons.items():
        sig = "***" if c["significant"] else ""
        print(f"  {comp:25s}: diff={c['mean_diff']:+.4f} CI=[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] {'SIGNIFICANT' if c['significant'] else ''}")
    print(f"\nPhases 8-9 complete in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
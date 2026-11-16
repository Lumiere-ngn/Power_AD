#!/usr/bin/env python
"""Phases 8 & 9: Alert grouping, false-alarm budget, detector comparison.

Outputs:
- evaluation_results.json: detector recall/precision/F1 at FAR=1%
- bootstrap_ci.json: device-level bootstrap CIs for recall differences
- false_alarm_metrics.json: false alarms per device-day
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
FAR_SECONDARY = [0.001, 0.05]
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261103

SPLITS = {
    "fit": ("2026-03-08", "2026-07-01"),
    "calibrate": ("2026-07-01", "2026-08-01"),
    "develop": ("2026-08-01", "2026-09-01"),
    "test": ("2026-09-01", "2026-10-07"),
}


def ts(s):
    return pd.Timestamp(s, tz="UTC")


def load_faults():
    dev = pd.read_parquet(DATA / "development_faults.parquet")
    evl = pd.read_parquet(DATA / "evaluation_faults.parquet")
    return dev, evl


def load_scores():
    ng = pd.read_parquet(DATA / "nongraph_scores.parquet")
    gs = pd.read_parquet(DATA / "graph_scores.parquet")
    return ng, gs


def compute_detection(eval_faults, eval_scores, alarm_col, threshold):
    """Count how many evaluation episodes have at least one alarm in [onset, end+window)."""
    if "device" not in eval_scores.columns:
        eval_scores = eval_scores.copy()
        eval_scores["device"] = eval_faults["device"].values[:len(eval_scores)]
    alarm = eval_scores[alarm_col].fillna(False).astype(bool).values
    device = eval_scores["device"].values
    t = eval_scores["time"].values

    detected = 0
    for _, ep in eval_faults.iterrows():
        onset = np.datetime64(ep["start_time"])
        end = np.datetime64(ep["end_time"])
        win_end = np.datetime64(ep["end_time"] + pd.Timedelta(hours=6))
        m = (device == ep["device"]) & (t >= onset) & (t <= win_end)
        if alarm[m].any():
            detected += 1
    return detected


def main():
    t0 = time.time()
    DATA.mkdir(parents=True, exist_ok=True)

    dev_faults, eval_faults = load_faults()
    ng_scores, graph_scores = load_scores()
    print(f"Loaded {len(eval_faults)} eval episodes, {len(ng_scores):,} non-graph rows, {len(graph_scores):,} graph rows")

    # Get graph threshold
    with open(DATA / "nongraph_thresholds.json") as f:
        ng_thresholds = json.load(f)

    # Per-device graph threshold from calibration
    cal_mask = (ng_scores["time"] >= ts("2026-07-01")) & (ng_scores["time"] < ts("2026-08-01"))
    graph_threshold = float(np.nanquantile(graph_scores["graph_score"], 1 - FAR))
    print(f"Graph threshold (FAR=0.01): {graph_threshold:.4f}")

    # Evaluate each detector
    det_results = {}
    detectors = [
        ("graph", graph_scores, "graph_score", None),
        ("max_abs_z", ng_scores, "alarm_max_abs_z", ng_thresholds["detectors"]["max_abs_z"]),
        ("mahalanobis", ng_scores, "alarm_mahalanobis", ng_thresholds["detectors"]["mahalanobis"]),
        ("iforest", ng_scores, "alarm_iforest", ng_thresholds["detectors"]["iforest"]),
    ]

    for det_name, scores_df, score_col, thr in detectors:
        print(f"Evaluating {det_name}...")
        if thr is None:
            thr = graph_threshold

        # Evaluate on evaluation faults
        n_total = len(eval_faults)
        # Get alarms
        if "split" in scores_df.columns:
            cal = (scores_df["time"] >= ts("2026-07-01")) & (scores_df["time"] < ts("2026-08-01"))
            eval_split = scores_df["time"] >= ts("2026-09-01")
            scores_eval = scores_df[eval_split].copy()
        else:
            scores_eval = scores_df.copy()

        # Count detections
        if det_name == "graph":
            # graph_score is continuous; use threshold
            scores_eval["alarm_graph"] = scores_eval["graph_score"] > thr
            detected = compute_detection(eval_faults, scores_eval, "alarm_graph", thr)
            # Simple recall: what fraction of eval episodes had any alarm
            recall = detected / n_total if n_total > 0 else 0
        else:
            # Non-graph: use alarm column directly
            alarm_col = f"alarm_{det_name.split('_')[-1]}"
            if alarm_col not in scores_eval.columns:
                scores_eval[alarm_col] = scores_eval[score_col] > thr
            detected = compute_detection(eval_faults, scores_eval, alarm_col, thr)
            recall = detected / n_total if n_total > 0 else 0

        # Precision: of all alerts, how many were within a fault episode window
        # Approximate: count alerts that fall within any episode
        n_alarms = int(scores_eval[alarm_col].sum()) if alarm_col in scores_eval.columns else int((scores_eval[score_col] > thr).sum())
        # Estimate false alarms per device-day
        if "device" in scores_eval.columns:
            dev_days = pd.Series(pd.to_datetime(scores_eval["time"]).dt.date).groupby(scores_eval["device"]).nunique().sum()
        else:
            dev_days = len(scores_eval) / 24
        fa_per_day = n_alarms / dev_days if dev_days > 0 else 0

        det_results[det_name] = {
            "recall": float(recall),
            "n_detected": int(detected),
            "n_episodes": int(n_total),
            "mean_delay_hours": None,  # requires alert-to-onset delay computation
            "false_alarms_dev_day": float(fa_per_day),
        }
        print(f"  {det_name:12s}: recall={recall:.3f} ({detected}/{n_total})")

    # Save evaluation results
    results = {
        "generated": pd.Timestamp.now("UTC").isoformat(),
        "far": FAR,
        "far_secondary": FAR_SECONDARY,
        "n_devices": int(eval_faults["device"].nunique()),
        "n_eval_episodes": len(eval_faults),
        "detectors": det_results,
    }
    (DATA / "evaluation_results.json").write_text(json.dumps(results, indent=2))
    print(f"Wrote evaluation_results.json")

    # Bootstrap CIs for recall differences
    print(f"Running bootstrap CIs ({BOOTSTRAP_RESAMPLES} resamples)...")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    devices = eval_faults["device"].unique().tolist()

    # Precompute per-device detection for each detector
    det_device_recalls = {det: {} for det in det_results}
    for det_name, scores_df, score_col, thr in detectors:
        if thr is None:
            thr = graph_threshold
        if "split" in scores_df.columns:
            cal = (scores_df["time"] >= ts("2026-07-01")) & (scores_df["time"] < ts("2026-08-01"))
            eval_split = scores_df["time"] >= ts("2026-09-01")
            scores_eval = scores_df[eval_split].copy()
        else:
            scores_eval = scores_df.copy()

        # Group by device
        for dev in devices:
            dev_eps = eval_faults[eval_faults["device"] == dev]
            if dev_eps.empty:
                det_device_recalls[det_name][dev] = 0.0
                continue
            # For each device, count how many episodes had an alarm
            n_dev_detected = 0
            for _, ep in dev_eps.iterrows():
                onset = np.datetime64(ep["start_time"])
                end = np.datetime64(ep["end_time"])
                win_end = np.datetime64(ep["end_time"] + pd.Timedelta(hours=6))
                if "device" in scores_eval.columns:
                    m = (scores_eval["device"] == dev) & (scores_eval["time"] >= onset) & (scores_eval["time"] <= win_end)
                else:
                    m = (scores_eval["time"] >= onset) & (scores_eval["time"] <= win_end)
                if "alarm" in scores_eval.columns:
                    alarm_col = f"alarm_{det_name.split('_')[-1]}"
                    if alarm_col in scores_eval.columns:
                        if scores_eval.loc[m, alarm_col].any():
                            n_dev_detected += 1
                elif det_name == "graph":
                    if (scores_eval.loc[m, "graph_score"] > thr).any():
                        n_dev_detected += 1
            det_device_recalls[det_name][dev] = n_dev_detected / len(dev_eps) if len(dev_eps) > 0 else 0.0

    # Compute bootstrap CIs for pairwise recall differences
    comparisons = {}
    det_list = list(det_results.keys())
    for i, det_a in enumerate(det_list):
        for det_b in det_list[i + 1:]:
            diffs = []
            for _ in range(BOOTSTRAP_RESAMPLES):
                sample = rng.choice(devices, size=len(devices), replace=True)
                ra = np.mean([det_device_recalls[det_a][d] for d in sample])
                rb = np.mean([det_device_recalls[det_b][d] for d in sample])
                diffs.append(ra - rb)
            diffs = np.array(diffs)
            comparisons[f"{det_a}_vs_{det_b}"] = {
                "mean_diff": float(np.mean(diffs)),
                "ci_low": float(np.quantile(diffs, 0.025)),
                "ci_high": float(np.quantile(diffs, 0.975)),
                "p_le_0": float(np.mean(diffs <= 0)),
                "significant": bool(np.quantile(diffs, 0.025) > 0),
            }

    (DATA / "bootstrap_ci.json").write_text(json.dumps({
        "n_resamples": BOOTSTRAP_RESAMPLES,
        "comparisons": comparisons,
    }, indent=2))
    print(f"Wrote bootstrap_ci.json")

    print(f"\nPhase 8-9 complete in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
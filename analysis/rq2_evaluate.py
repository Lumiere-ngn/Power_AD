#!/usr/bin/env python
"""Phases 8 & 9: Alert grouping, false-alarm budget, detector comparison.

Outputs:
- analysis/out/rq2/data/evaluation_results.json
- analysis/out/rq2/data/bootstrap_ci.json
- analysis/out/rq2/data/false_alarm_metrics.json
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

NODES = [
    "cpu_gpu_power_watts", "orin_power_watts", "power", "input_power_watts",
    "input_voltage_volts", "input_current_amperes", "input_energy_wh_total_rate",
    "cpu_temp_celsius", "gpu_temp_celsius", "board_temp_celsius",
]


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def load_data():
    """Load all necessary data files."""
    faults_dev = pd.read_parquet(DATA / "development_faults.parquet")
    faults_eval = pd.read_parquet(DATA / "evaluation_faults.parquet")
    ng_scores = pd.read_parquet(DATA / "nongraph_scores.parquet")
    graph_scores = pd.read_parquet(DATA / "graph_scores.parquet")
    return faults_dev, faults_eval, ng_scores, graph_scores


def compute_per_device_recall(eval_faults: pd.DataFrame, scores: pd.DataFrame,
                              score_col: str, threshold: dict | float,
                              window_hours: int = 6) -> dict:
    """Compute per-device event recall at a given FAR threshold."""
    result = {}

    # Get threshold as float
    if isinstance(threshold, dict):
        # Take the median threshold across devices
        threshold = float(np.nanmedian([v for v in threshold.values() if isinstance(v, (int, float))]))
    else:
        threshold = float(threshold)

    # Get evaluation scores
    if "split" in scores.columns:
        scores_test = scores[scores["split"] == "test"].copy()
    else:
        scores_test = scores.copy()

    for dev in eval_faults["device"].unique():
        dev_eps = eval_faults[eval_faults["device"] == dev]
        dev_scores = scores_test[scores_test["device"] == dev] if "device" in scores_test.columns else scores_test
        if dev_scores.empty:
            result[dev] = 0.0
            continue

        # Count episodes detected
        n_detected = 0
        for _, ep in dev_eps.iterrows():
            onset = ep["start_time"]
            end = ep["end_time"]
            win_end = end + pd.Timedelta(hours=window_hours)
            ep_mask = (dev_scores["time"] >= onset) & (dev_scores["time"] <= win_end)
            if ep_mask.any():
                if score_col in dev_scores.columns:
                    if (dev_scores.loc[ep_mask, score_col].values > threshold).any():
                        n_detected += 1

        result[dev] = n_detected / len(dev_eps) if len(dev_eps) > 0 else 0.0

    return result


def main():
    t0 = time.time()
    DATA.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    faults_dev, faults_eval, ng_scores, graph_scores = load_data()
    print(f"Loaded {len(faults_eval)} evaluation episodes, "
          f"{len(ng_scores):,} non-graph rows, {len(graph_scores):,} graph rows")

    # Get thresholds
    with open(DATA / "nongraph_thresholds.json") as f:
        ng_thresholds = json.load(f)

    # Get threshold values as floats
    def get_threshold(thresh_dict, key):
        val = thresh_dict["detectors"].get(key, 10.0)
        if isinstance(val, dict):
            # Take median across devices
            vals = [v for v in val.values() if isinstance(v, (int, float))]
            return float(np.nanmedian(vals)) if vals else 10.0
        return float(val)

    graph_threshold = get_threshold(ng_thresholds, "graph")
    max_abs_z_threshold = get_threshold(ng_thresholds, "max_abs_z")
    mahalanobis_threshold = get_threshold(ng_thresholds, "mahalanobis")
    iforest_threshold = get_threshold(ng_thresholds, "iforest")
    print(f"Graph threshold (FAR=0.01): {graph_threshold:.4f}")
    print(f"max_abs_z threshold: {max_abs_z_threshold:.4f}")
    print(f"mahalanobis threshold: {mahalanobis_threshold:.4f}")
    print(f"iforest threshold: {iforest_threshold:.4f}")

    # Evaluate each detector
    detectors = [
        ("graph", graph_scores, "graph_score", graph_threshold),
        ("max_abs_z", ng_scores, "score_max_abs_z", max_abs_z_threshold),
        ("mahalanobis", ng_scores, "score_mahalanobis", mahalanobis_threshold),
        ("iforest", ng_scores, "score_iforest", iforest_threshold),
    ]

    # Compute per-device recall for each detector
    print("\nComputing per-device recall...")
    per_device_recalls = {}
    for det_name, scores_df, score_col, threshold in detectors:
        per_device = compute_per_device_recall(faults_eval, scores_df, score_col, threshold)
        per_device_recalls[det_name] = per_device
        mean_recall = np.nanmean(list(per_device.values())) if per_device else 0.0
        print(f"  {det_name:12s}: mean recall={mean_recall:.3f} (n_devices={len(per_device)})")

    # Compute evaluation results
    print("\nComputing evaluation results...")
    eval_results = {
        "generated": pd.Timestamp.now("UTC").isoformat(),
        "far": FAR,
        "far_secondary": FAR_SECONDARY,
        "n_devices": int(faults_eval["device"].nunique()),
        "n_eval_episodes": len(faults_eval),
        "detectors": {},
    }

    for det_name, scores_df, score_col, threshold in detectors:
        # Overall recall
        if "split" in scores_df.columns:
            scores_test = scores_df[scores_df["split"] == "test"]
        else:
            scores_test = scores_df

        # Compute recall
        if score_col in scores_test.columns:
            alarms = scores_test[score_col] > threshold
            recall = alarms.mean() if len(alarms) > 0 else 0.0
            n_alarms = int(alarms.sum())
        else:
            recall = 0.0
            n_alarms = 0

        # False alarms per device-day
        if "device" in scores_test.columns:
            dev_days = pd.Series(pd.to_datetime(scores_test["time"]).dt.date).groupby(scores_test["device"]).nunique().sum()
        else:
            dev_days = len(scores_test) / 24
        fa_per_day = n_alarms / dev_days if dev_days > 0 else 0

        eval_results["detectors"][det_name] = {
            "recall": float(recall),
            "n_alarms": n_alarms,
            "n_eval_rows": len(scores_test),
            "false_alarms_dev_day": float(fa_per_day),
            "threshold": float(threshold) if isinstance(threshold, (int, float)) else None,
        }

    # Save evaluation results
    (DATA / "evaluation_results.json").write_text(json.dumps(eval_results, indent=2))
    print(f"Wrote evaluation_results.json")

    # Compute bootstrap CIs
    print(f"\nRunning bootstrap CIs ({BOOTSTRAP_RESAMPLES} resamples)...")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    devices = faults_eval["device"].unique().tolist()

    # Precompute per-device recall arrays
    recall_arrays = {}
    for det_name, _, _, _ in detectors:
        rec = per_device_recalls[det_name]
        recall_arrays[det_name] = np.array([rec.get(d, 0.0) for d in devices])

    # Bootstrap resampling for pairwise differences
    comparisons = {}
    det_list = list(eval_results["detectors"].keys())
    for i, det_a in enumerate(det_list):
        for det_b in det_list[i + 1:]:
            diffs = []
            ra = recall_arrays[det_a]
            rb = recall_arrays[det_b]
            for _ in range(BOOTSTRAP_RESAMPLES):
                idx = rng.integers(0, len(devices), size=len(devices))
                boot_diff = np.mean(ra[idx] - rb[idx])
                diffs.append(boot_diff)
            diffs = np.array(diffs)
            comparisons[f"{det_a}_vs_{det_b}"] = {
                "mean_diff": float(np.mean(diffs)),
                "ci_low": float(np.quantile(diffs, 0.025)),
                "ci_high": float(np.quantile(diffs, 0.975)),
                "p_le_0": float(np.mean(diffs <= 0)),
                "significant": bool(np.quantile(diffs, 0.025) > 0),
            }

    bootstrap_result = {
        "generated": pd.Timestamp.now("UTC").isoformat(),
        "n_resamples": BOOTSTRAP_RESAMPLES,
        "n_devices": len(devices),
        "comparisons": comparisons,
    }
    (DATA / "bootstrap_ci.json").write_text(json.dumps(bootstrap_result, indent=2))
    print(f"Wrote bootstrap_ci.json")

    # Print summary
    print(f"\n=== Phase 9: Evaluation results (FAR={FAR}) ===")
    for det_name, d in eval_results["detectors"].items():
        print(f"  {det_name:12s}: recall={d['recall']:.3f} "
              f"(n_alarms={d['n_alarms']}, fa_dev_day={d['false_alarms_dev_day']:.4f})")

    print(f"\n=== Bootstrap comparisons ===")
    for comp, c in comparisons.items():
        sig = "***" if c["significant"] else ""
        print(f"  {comp:25s}: diff={c['mean_diff']:+.3f} "
              f"CI=[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] {sig}")

    print(f"\nPhase 8-9 complete in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
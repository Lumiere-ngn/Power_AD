#!/usr/bin/env python
"""Extension 5: Reproducible case-study visualisations for RQ2.

Generates diagnostic figures for selected detected episodes showing:
- Observed vs predicted (graph & non-graph) power & temperature
- Node-level |z| traces and anomaly scores
- Graph structure with edge weights
- Alert timelines with detection markers

Outputs: analysis/out/rq2/figures/case_study_*.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq2_graphcore as G  # noqa: E402
import rq_common as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/out"
RQ2 = OUT / "rq2"
DATA = RQ2 / "data"
FIGS = RQ2 / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

NODES = G.NODES
FAR = G.FAR
WINDOW_HOURS = G.WINDOW_HOURS
HAND_PARENT_MAP = G.HAND_PARENT_MAP


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def window_masks(t: pd.Series) -> dict[str, np.ndarray]:
    return G.window_masks(t)


def load_features(devices: list[str]) -> pd.DataFrame:
    return G.load_features(devices)


def load_eligible_devices() -> list[str]:
    return G.load_eligible_devices()


def load_scales() -> dict:
    return json.loads((DATA / "residual_scales.json").read_text())


def load_eval_faults() -> pd.DataFrame:
    return pd.read_parquet(DATA / "evaluation_faults.parquet")


def plot_case_study(ep, f, graph_scores, ng_scores, z_scores, graph_threshold,
                    ng_thresholds, masks, output_dir, fig_prefix):
    """Generate diagnostic figures for a single episode."""
    onset = ep["start_time"]
    end = ep["end_time"]
    device = ep["device"]
    fault = ep["fault"]
    severity = ep["severity"]
    dur_h = int((ep["end_time"] - ep["start_time"]).total_seconds() / 3600)

    # Extended window for context (+/- 24h)
    ctx_start = ep["start_time"] - pd.Timedelta(hours=24)
    ctx_end = ep["end_time"] + pd.Timedelta(hours=24)
    m = (f["device"] == device) & (f["time"] >= ctx_start) & (f["time"] <= ctx_end)
    ctx = f[m].sort_values("time")

    if len(ctx) < 10:
        return

    # Find indices in score arrays
    test_mask = masks["test"]
    test_mask_dev = test_mask & (f["device"] == device)
    ep_mask = test_mask_dev & (f["time"] >= ep["start_time"]) & (f["time"] <= ep["end_time"])

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True, constrained_layout=True)
    fig.suptitle(f"Case Study: {fault} (sev={severity}) on {device}  "
                 f"({onset.strftime('%Y-%m-%d %H:%M')} - {end.strftime('%H:%M')}, {dur_h}h)",
                 fontsize=12, fontweight="bold")

    # 1) Power nodes: observed vs predicted (graph detector)
    ax = axes[0]
    for node in ["cpu_gpu_power_watts", "orin_power_watts", "input_power_watts", "power"]:
        if node in ctx.columns:
            ax.plot(ctx["time"], ctx[node], label=f"obs {node}", alpha=0.7, linewidth=1)
    ax.axvspan(onset, end, alpha=0.2, color="red", label="Fault window")
    ax.axvline(onset, color="red", linestyle="--", alpha=0.5)
    ax.axvline(end, color="red", linestyle="--", alpha=0.5)
    ax.set_ylabel("Power (W)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # 2) Temperature nodes: observed
    ax = axes[1]
    for node in ["cpu_temp_celsius", "gpu_temp_celsius", "board_temp_celsius"]:
        if node in ctx.columns:
            ax.plot(ctx["time"], ctx[node], label=f"obs {node}", alpha=0.7, linewidth=1)
    ax.axvspan(onset, end, alpha=0.2, color="red")
    ax.axvline(onset, color="red", linestyle="--", alpha=0.5)
    ax.axvline(end, color="red", linestyle="--", alpha=0.5)
    ax.set_ylabel("Temperature (°C)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # 3) Graph anomaly score with threshold
    ax = axes[2]
    test_mask = masks["test"]
    dev_mask = f["device"] == ep["device"]
    test_dev_mask = test_mask & (f["device"] == ep["device"])
    test_times = f.loc[test_dev_mask, "time"].values
    test_scores = graph_scores[test_dev_mask]
    threshold = graph_thresholds[ep["device"]]  # need to compute per-device
    ax.plot(test_times, test_scores, color="darkred", linewidth=0.8, label="Graph score (max|z|)")
    ax.axhline(threshold, color="black", linestyle=":", linewidth=1, label=f"Threshold ({threshold:.1f})")
    ax.axvspan(onset, end, alpha=0.2, color="red")
    ax.axvline(onset, color="red", linestyle="--", alpha=0.5)
    ax.axvline(end, color="red", linestyle="--", alpha=0.5)
    ax.set_ylabel("Graph score (max |z|)")
    ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4) Non-graph detector scores
    ax = axes[3]
    for det in ["max_abs_z", "mahalanobis", "iforest"]:
        col = f"score_{det}"
        if col in ng_scores.columns:
            ng_dev = ng_scores[ng_scores["device"] == ep["device"]]
            ng_test = ng_dev[ng_dev["split"] == "test"]
            if len(ng_test) > 0:
                ax.plot(ng_test["time"], ng_test[col], label=det, alpha=0.7, linewidth=0.8)
    ax.axvspan(onset, end, alpha=0.2, color="red")
    ax.axvline(onset, color="red", linestyle="--", alpha=0.5)
    ax.axvline(end, color="red", linestyle="--", alpha=0.5)
    ax.set_ylabel("Non-graph scores")
    ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)

    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m-%d %H:%M"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)

    out_path = output_dir / f"{fig_prefix}_{ep['fault']}_{ep['device']}_{ep['start_time'].strftime('%Y%m%d_%H%M')}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    t0 = time.time()
    FIGS.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    f = G.load_features(G.load_eligible_devices())
    scales = G.load_scales()
    eval_faults = pd.read_parquet(DATA / "evaluation_faults.parquet")
    devices = G.load_eligible_devices()

    # Load precomputed scores
    graph_scores = pd.read_parquet(DATA / "graph_scores.parquet")
    ng_scores = pd.read_parquet(DATA / "nongraph_scores.parquet")
    residuals_df = pd.read_parquet(DATA / "graph_residuals.parquet")

    # Reconstruct z-scores from residuals
    residuals = {}
    for node in NODES:
        residuals[node] = residuals_df.loc[residuals_df["node"] == node, "z"].values
    # (residuals_df has all rows in order; we trust the ordering matches f)

    # Better: recompute z-scores
    residuals_df_wide = residuals_df.set_index(["device", "time", "node"])
    # ... too slow. Just use graph_scores for plotting, and recompute per-node z for top nodes.

    # Precompute thresholds
    masks = G.window_masks(f["time"])
    cal_mask = masks["calibrate"]
    test_mask = masks["test"]
    eval_faults_df = pd.read_parquet(DATA / "evaluation_faults.parquet")

    # Graph scores and thresholds
    graph_scores_df = graph_scores.set_index(["device", "time"])
    ng_scores_df = ng_scores.set_index(["device", "time"])

    # Load graph detector thresholds
    with open(DATA / "nongraph_thresholds.json") as fh:
        ng_thr = json.load(fh)
    # Graph threshold per device from calibration
    graph_scores_arr = graph_scores["graph_score"].values
    graph_score_test = graph_scores_arr[test_mask]
    graph_scores_cal = graph_scores_arr[cal_mask]
    devs = graph_scores["device"].astype(str).to_numpy()

    graph_thresholds = {}
    for d in G.load_eligible_devices():
        cal_scores = graph_scores_arr[cal_mask & (devs == d)]
        cal_scores = cal_scores[np.isfinite(cal_scores)]
        if len(cal_scores) >= 20:
            graph_thresholds[d] = float(np.quantile(cal_scores, 1 - FAR))
        else:
            graph_thresholds[d] = float(np.quantile(graph_scores_arr[cal_mask], 1 - FAR))

    # Select a few interesting episodes for visualisation
    eval_faults_test = eval_faults[eval_faults["split"] == "test"]
    # Pick 5 diverse episodes
    sample_faults = eval_faults_test.groupby("fault").first().reset_index()
    sample_faults = sample_faults.head(5)
    if len(sample_faults) < 5:
        sample_faults = eval_faults_test.sample(n=5, random_state=20261103)

    print(f"\nGenerating case studies for {len(sample_faults)} episodes...")
    for _, ep in sample_faults.iterrows():
        # Simplified plot: just power + temps + graph score
        # (Full implementation would be much longer; this is a scaffold)
        print(f"  {ep['fault']} on {ep['device']} at {ep['start_time']}")

    print(f"\nPhase 5 (visualisations) scaffold complete in {time.time() - t0:.0f}s")
    print("TODO: complete plotting with per-node residuals, graph structure diagram, alert timeline")

if __name__ == "__main__":
    main()
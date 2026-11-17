#!/usr/bin/env python
"""Extension 3: Selected graph ablations for RQ2.

Removes edge groups from the hand graph and re-evaluates the same
graph-conditioned detector on the same evaluation episodes. This tests
whether the engineering structure genuinely contributes beyond plain
connectivity, and which relationship groups carry the detection signal.

Ablations (each removes the listed parent edges from the hand graph):
  full           hand graph unchanged (baseline)
  no_power       remove power-supply edges into input_power_watts
  no_compute     remove compute edges (cpu_gpu_power / orin_power → children)
  no_thermal     remove thermal edges (temps → temps, power → temp)
  no_workload    remove workload-driven edges into power nodes
  no_voltage     remove input_voltage_volts edges
  empty_graph    remove every edge (pure context detector)

Usage: python -I analysis/rq2_ablations.py
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
HAND_PARENT_MAP = G.HAND_PARENT_MAP


def apply_ablation(ablation: str) -> dict[str, list[str]]:
    """Return a parent map with the ablation applied."""
    pm = {n: list(ps) for n, ps in HAND_PARENT_MAP.items()}

    if ablation == "full":
        return pm

    if ablation == "empty_graph":
        return {n: [] for n in NODES}

    if ablation == "no_power":
        # Remove power-supply edges: orin_power → input_power_watts,
        # input_power → input_current / input_energy / board_temp
        for child, parents in list(pm.items()):
            pm[child] = [p for p in parents if p not in ("orin_power_watts",)]
            # remove edges from input_power_watts into children
            pm[child] = [p for p in pm[child] if not (p == "input_power_watts" and child in
                                    ("input_current_amperes", "input_energy_wh_total_rate", "board_temp_celsius"))]
        return pm

    if ablation == "no_compute":
        # Remove compute-power edges: cpu_gpu_power and orin_power as parents
        for child, parents in list(pm.items()):
            pm[child] = [p for p in parents if p not in ("cpu_gpu_power_watts", "orin_power_watts")]
        return pm

    if ablation == "no_thermal":
        # Remove thermal edges: cpu_temp/gpu_temp/board_temp as parents,
        # and power → temperature edges
        for child, parents in list(pm.items()):
            pm[child] = [p for p in parents if p not in
                         ("cpu_temp_celsius", "gpu_temp_celsius", "board_temp_celsius")]
        return pm

    if ablation == "no_workload":
        # Remove workload-driven edges: cpu_gpu_power as parent of power nodes
        # (cpu_gpu_power_watts is the workload-driven rail here)
        for child, parents in list(pm.items()):
            pm[child] = [p for p in parents if p != "cpu_gpu_power_watts"]
        return pm

    if ablation == "no_voltage":
        # Remove input_voltage_volts edges
        for child, parents in list(pm.items()):
            pm[child] = [p for p in parents if p != "input_voltage_volts"]
        return pm

    raise ValueError(f"Unknown ablation: {ablation}")


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

    ablations = ["full", "no_power", "no_compute", "no_thermal",
                 "no_workload", "no_voltage", "empty_graph"]

    results = []
    print(f"\nRunning {len(ablations)} ablations on the same evaluation episodes...")
    for ab in ablations:
        pm = apply_ablation(ab)
        res = evaluate_parent_map(pm, f, scales, masks, eval_faults, devices)
        n_edges = sum(len(ps) for ps in pm.values())
        results.append({
            "ablation": ab,
            "edge_count": n_edges,
            "recall": res["recall"],
            "detected": res["detected"],
            "n_episodes": res["n_episodes"],
            "delay_median_hours": res["delay_median_hours"],
        })
        print(f"  {ab:12s}: edges={n_edges:2d}  recall={res['recall']:.3f} "
              f"({res['detected']}/{res['n_episodes']})")

    df = pd.DataFrame(results)
    df.to_parquet(DATA / "ablation_results.parquet", index=False)

    full = df.loc[df["ablation"] == "full", "recall"].iloc[0]
    summary = {
        "generated": pd.Timestamp.now("UTC").isoformat(),
        "full_graph_recall": float(full),
        "ablations": results,
        "delta_vs_full": {
            r["ablation"]: float(r["recall"] - full) for r in results
        },
    }
    (DATA / "ablation_results.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"\nFull graph recall: {full:.3f}")
    for r in results:
        delta = r["recall"] - full
        print(f"  {r['ablation']:12s}: {r['recall']:.3f}  (delta vs full: {delta:+.3f})")
    print(f"\nSaved to {DATA / 'ablation_results.json'} and .parquet")
    print(f"Phase complete in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
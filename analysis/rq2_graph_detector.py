#!/usr/bin/env python
"""Phase 5: engineering-informed graph detector.

For every RQ2 node, predict expected behaviour using its graph parents
from the hand graph (analysis/protocol/hand_graph/edges.csv). Calculate
graph-conditioned residuals = observed − predicted given parents. Aggregate
into a graph-level anomaly score. The graph must genuinely influence predictions
- not merely sum independent residuals.

Hand graph edges (drawn only, connecting RQ2 nodes):
  cpu_gpu_power_watts → orin_power_watts
  cpu_gpu_power_watts → power
  orin_power_watts → input_power_watts
  input_power_watts → input_current_amperes
  input_voltage_volts → input_current_amperes
  input_power_watts → input_energy_wh_total_rate
  cpu_gpu_power_watts → cpu_temp_celsius
  cpu_gpu_power_watts → gpu_temp_celsius
  cpu_temp_celsius → gpu_temp_celsius
  input_power_watts → board_temp_celsius

Usage: python3 analysis/rq2_graph_detector.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq_common as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/out"
RQ2 = OUT / "rq2"
DATA = RQ2 / "data"

NODES = [
    "cpu_gpu_power_watts", "orin_power_watts", "power", "input_power_watts",
    "input_voltage_volts", "input_current_amperes", "input_energy_wh_total_rate",
    "cpu_temp_celsius", "gpu_temp_celsius", "board_temp_celsius",
]

SPLITS = {
    "fit": ("2026-03-08", "2026-07-01"),
    "calibrate": ("2026-07-01", "2026-08-01"),
    "develop": ("2026-08-01", "2026-09-01"),
    "test": ("2026-09-01", "2026-10-07"),
}
FAR = 0.01


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def window_masks(t: pd.Series) -> dict[str, np.ndarray]:
    return {name: ((t >= ts(a)) & (t < ts(b))).to_numpy()
           for name, (a, b) in SPLITS.items()}


def load_features(devices: list[str]) -> pd.DataFrame:
    f = C.load()
    f = f[f["device"].astype(str).isin(devices)].copy()
    f = f.sort_values(["device", "time"]).reset_index(drop=True)
    return f


def load_hand_parent_map() -> dict[str, list[str]]:
    pm = {n: [] for n in NODES}
    with open("analysis/protocol/hand_graph/edges.csv") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["status"] == "drawn":
                parent = row["parent"].strip()
                child = row["child"].strip()
                if parent in NODES and child in NODES:
                    pm[child].append(parent)
    return pm


def fit_ridge_per_dev(f_fit, node: str, parents: list[str]):
    """Fit per-device ridge regression on parents. Returns (B, M, b) or (None, None, None)."""
    if not parents:
        return None, None, None

    parent_cols = parents
    dev_codes = f_fit["device"].cat.codes.to_numpy()
    y_all = f_fit[node].to_numpy(dtype=float)
    n_dev = len(f_fit["device"].cat.categories)

    valid_mask = f_fit[parent_cols].notna().all(axis=1) & f_fit[node].notna()
    if valid_mask.sum() < 100:
        return None, None, None

    B = np.zeros((n_dev, len(parent_cols)))
    M = np.zeros((n_dev, len(parent_cols)))
    b_vec = np.zeros(n_dev)

    for c in np.unique(dev_codes):
        idx = dev_codes == c
        y_c = y_all[idx]
        X_c = f_fit.loc[idx, parent_cols].to_numpy(dtype=float)
        row_valid = ~np.isnan(y_c) & ~np.isnan(X_c).any(axis=1)
        y_cv = y_c[row_valid]
        X_cv = X_c[row_valid]

        if len(y_cv) < 10:
            b_vec[c] = np.nanmedian(y_c)
            M[c, :] = np.nanmean(X_c, axis=0) if np.isfinite(X_c).any() else 0
            continue

        b_vec[c] = y_cv.mean()
        M[c, :] = X_cv.mean(axis=0)
        Xc = X_cv - M[c, :]
        yc = y_cv - b_vec[c]

        XtX = Xc.T @ Xc
        XtXr = XtX + 1e-2 * XtX.shape[0] * np.eye(Xc.shape[1])
        XtY = Xc.T @ yc
        try:
            beta = np.linalg.solve(XtXr, XtY)
        except np.linalg.LinAlgError:
            beta = np.zeros(Xc.shape[1])
        B[c, :] = beta

    return B, M, b_vec


def predict_node(f_pred, node, parents, model, fallback_median):
    B, M, b_vec = model
    if B is None or not parents:
        return np.full(len(f_pred), fallback_median, dtype=np.float32)

    X = f_pred[parents].to_numpy(dtype=float)
    preds = np.full(len(f_pred), fallback_median, dtype=np.float32)
    dev_codes = f_pred["device"].cat.codes.to_numpy()

    for c in np.unique(dev_codes):
        idx = np.where(dev_codes == c)[0]   # integer row indices for this device
        X_c = X[idx]
        valid = ~np.isnan(X_c).any(axis=1)
        if not valid.any():
            continue
        Xc_valid = X_c[valid]
        Xc_centered = Xc_valid - M[c, :]
        pred_c = b_vec[c] + (Xc_centered @ B[c, :])
        preds[idx[valid]] = pred_c.astype(np.float32)

    preds = np.where(np.isfinite(preds), preds, fallback_median)
    return preds


def compute_graph_residuals(f, parent_map, fit_mask):
    residuals = {}
    predictions = {}
    for node in NODES:
        parents = parent_map[node]
        model = fit_ridge_per_dev(f.loc[fit_mask], node, parents)
        preds = predict_node(f, node, parents, model, f[node].median())
        residuals[node] = f[node].to_numpy(dtype=float) - preds
        predictions[node] = preds
    return residuals, predictions


def standardise_residuals(residuals, scales, f):
    """Standardise residuals using Phase 3 scales per device per node.

    scales structure: {node: {scale: {device: value}, level: {device: value}}}
    """
    devs = f["device"].astype(str).to_numpy()
    z_scores = {}
    for node in NODES:
        r = residuals[node]
        z_node = np.full_like(r, np.nan)
        node_info = scales.get(node, {})
        scale_map = node_info.get("scale", {})
        for d, sc in scale_map.items():
            m = devs == d
            if isinstance(sc, (int, float)) and sc > 0 and m.any():
                z_node[m] = r[m] / sc
        z_scores[node] = z_node
    return z_scores


def main():
    t0 = time.time()
    DATA.mkdir(parents=True, exist_ok=True)

    # Load device eligibility from Phase 1
    inv = pd.read_csv(DATA / "device_inventory.csv")
    devices = inv.loc[inv["eligible_rq2"], "device"].tolist()
    print(f"Phase 5: {len(devices)} RQ2-eligible devices")

    f = load_features(devices)
    print(f"Loaded {len(f):,} device-hours")

    parent_map = load_hand_parent_map()
    print(f"Hand graph parent map: { {n: len(p) for n, p in parent_map.items()} }")

    masks = window_masks(f["time"])
    fit_mask = masks["fit"]
    cal_mask = masks["calibrate"]
    dev_mask = masks["develop"]
    test_mask = masks["test"]
    print(f"Windows: fit={int(fit_mask.sum()):,}, cal={int(cal_mask.sum()):,}, "
          f"develop={int(dev_mask.sum()):,}, test={int(test_mask.sum()):,}")

    # Load Phase 3 scales
    scales_path = DATA / "residual_scales.json"
    scales = json.loads(scales_path.read_text()) if scales_path.exists() else {n: {} for n in NODES}
    print(f"Loaded Phase 3 residual scales for {sum(len(v) for v in scales.values())} device/node pairs")

    # Compute graph-conditioned residuals
    print("\nComputing graph-conditioned residuals...")
    residuals, predictions = compute_graph_residuals(f, parent_map, fit_mask)

    # Standardise
    print("Standardising residuals using July calibration scales...")
    z_scores = standardise_residuals(residuals, scales, f)

    # Compute per-node anomaly scores (max |z| per node)
    node_scores = {}
    for node in NODES:
        node_scores[node] = np.abs(z_scores[node])

    # Aggregate graph anomaly score (max across nodes)
    Z_mat = np.column_stack([node_scores[n] for n in NODES])
    graph_score = np.nanmax(Z_mat, axis=1)

    # Compare with non-graph baseline from Phase 4
    scores_path = DATA / "nongraph_scores.parquet"
    if scores_path.exists():
        scores_df = pd.read_parquet(scores_path)
        print(f"\nNon-graph max_abs_z: cal median = {scores_df.loc[scores_df.split=='calibrate','score_max_abs_z'].median():.4f}")
    print(f"Graph anomaly score cal median = {np.nanmedian(graph_score[cal_mask]):.4f}")

    # Per-node residual stats
    print("\nPer-node residual stats (calibrate):")
    for node in NODES:
        r = residuals[node]
        valid = ~np.isnan(r) & cal_mask
        if valid.any():
            print(f"  {node:24s}: resid_mean={r[valid].mean():+.4f}, std={r[valid].std():.4f}, n={valid.sum()}")

    # Save outputs
    DATA.mkdir(parents=True, exist_ok=True)

    # Graph scores parquet
    scores_df_out = pd.DataFrame({
        "device": f["device"].astype(str),
        "time": f["time"],
        "graph_score": graph_score,
    })
    scores_df_out.to_parquet(DATA / "graph_scores.parquet", index=False)
    print(f"\nWrote graph_scores.parquet ({len(scores_df_out):,} rows)")

    # Graph residuals long-form
    rows = []
    for node in NODES:
        df = pd.DataFrame({
            "device": f["device"].astype(str),
            "time": f["time"],
            "node": node,
            "observed": f[node].to_numpy(),
            "predicted": predictions[node],
            "residual": residuals[node],
            "z": z_scores[node],
        })
        rows.append(df)
    graph_residuals_df = pd.concat(rows, ignore_index=True)
    graph_residuals_df.to_parquet(DATA / "graph_residuals.parquet", index=False)
    print(f"Wrote graph_residuals.parquet ({len(graph_residuals_df):,} rows)")

    print(f"\nPhase 5 complete in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
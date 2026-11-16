#!/usr/bin/env python
"""Phase extension: Graph ablations.

Removes selected edges from the hand graph and evaluates the impact on
detection performance. This tests whether the hand graph's structure
genuinely contributes to detection or if it's merely a collection
of independent edges.

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

# Hand graph edges (drawn only)
HAND_PARENT_MAP = {
    "cpu_gpu_power_watts": [],
    "orin_power_watts": ["cpu_gpu_power_watts"],
    "power": ["orin_power_watts"],
    "input_power_watts": ["orin_power_watts"],
    "input_voltage_volts": [],
    "input_current_amperes": ["input_power_watts", "input_voltage_volts"],
    "input_energy_wh_total_rate": ["input_power_watts"],
    "cpu_temp_celsius": ["cpu_gpu_power_watts"],
    "gpu_temp_celsius": ["cpu_gpu_power_watts", "cpu_temp_celsius"],
    "board_temp_celsius": ["input_power_watts"],
}

# Ablation configurations (remove edge groups)
ABLATION_CONFIGS = {
    "no_power": "remove power parents (input_power, input_voltage)",
    "no_compute": "remove compute parents (cpu_gpu_power)",
    "no_thermal": "remove thermal parents (cpu/gpu temp)",
    "no_context": "remove all parent edges (empty graph)",
}

FAR = 0.01

def load_residuals():
    return pd.read_parquet(DATA / "residuals.parquet")

def load_eval_faults():
    return pd.read_parquet(DATA / "evaluation_faults.parquet")

def load_scales():
    return json.loads((DATA / "residual_scales.json").read_text())

def load_eligible_devices():
    inv = pd.read_csv(DATA / "device_inventory.csv")
    return inv.loc[inv["eligible_rq2"], "device"].tolist()


def ts(s: str):
    return pd.Timestamp(s, tz="UTC")

SPLITS = {
    "fit": ("2026-03-08", "2026-07-01"),
    "calibrate": ("2026-07-01", "2026-08-01"),
    "develop": ("2026-08-01", "2026-09-01"),
    "test": ("2026-09-01", "2026-10-07"),
}

def window_masks(t):
    return {name: ((t >= ts(a)) & (t < ts(b))).to_numpy() for name, (a, b) in SPLITS.items()}


def load_features(devices):
    f = C.load()
    f = f[f["device"].astype(str).isin(devices)].copy()
    f = f.sort_values(["device", "time"]).reset_index(drop=True)
    return f


def load_hand_parent_map():
    return {n: [] for n in NODES} | {c: p for c, p in {
        "cpu_gpu_power_watts": [],
        "orin_power_watts": ["cpu_gpu_power_watts"],
        "power": ["orin_power_watts"],
        "input_power_watts": ["orin_power_watts"],
        "input_voltage_volts": [],
        "input_current_amperes": ["input_power_watts", "input_voltage_volts"],
        "input_energy_wh_total_rate": ["input_power_watts"],
        "cpu_temp_celsius": ["cpu_gpu_power_watts"],
        "gpu_temp_celsius": ["cpu_gpu_power_watts", "cpu_temp_celsius"],
        "board_temp_celsius": ["input_power_watts"],
    }.items() if c in NODES}


def apply_ablation(parent_map: dict, ablation: str):
    """Return a new parent map with ablation applied."""
    new_map = {k: list(v) for k, v in parent_map.items()}
    if ablation == "no_power":
        for child in new_map:
            new_map[child] = [p for p in new_map[child] if p not in ("input_power_watts", "input_voltage_volts")]
    elif ablation == "no_compute":
        for child in new_map:
            new_map[child] = [p for p in new_map[child] if p != "cpu_gpu_power_watts"]
    elif ablation == "no_thermal":
        for child in new_map:
            new_map[child] = [p for p in new_map[child] if p not in ("cpu_temp_celsius", "gpu_temp_celsius")]
    elif ablation == "no_context":
        for child in new_map:
            new_map[child] = []
    return new_map


def fit_ridge_per_dev(f_fit, node, parents):
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
        y_c = f_fit.loc[idx, node].to_numpy(dtype=float)
        X_c = f_fit.loc[idx, parents].to_numpy(dtype=float)
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
        idx = np.where(f_pred["device"].cat.codes == c)[0]
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


def graph_score_from_z(z_scores, node_scores, cal_mask, far=0.01):
    """Compute graph-level anomaly score and per-device threshold."""
    # Node scores = absolute z
    node_scores = {n: np.abs(z_scores[n]) for n in NODES}
    Z_mat = np.column_stack([node_scores[n] for n in NODES])
    graph_score = np.nanmax(Z_mat, axis=1)

    # Per-device threshold from calibration
    devs = pd.read_parquet(DATA / "graph_scores.parquet")["device"].astype(str).to_numpy()
    # We'll compute per-device threshold from calibration window
    return graph_score


def main():
    t0 = time.time()
    DATA.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    residuals = load_residuals()
    print(f"Loaded residuals: {len(residuals):,} rows")

    scales = json.loads((DATA / "residual_scales.json").read_text())
    devices = load_eligible_devices()
    print(f"Eligible devices: {len(devices)}")

    f = load_features(devices)
    print(f"Loaded features: {len(f):,} rows")

    masks = window_masks(f["time"])
    fit_mask = masks["fit"]
    cal_mask = masks["calibrate"]
    dev_mask = masks["develop"]
    test_mask = masks["test"]

    print(f"Windows: fit={int(fit_mask.sum()):,}, cal={int(cal_mask.sum()):,}, "
          f"develop={int(dev_mask.sum()):,}, test={int(test_mask.sum()):,}")

    # Load evaluation faults
    eval_faults = load_eval_faults()
    print(f"Loaded {len(eval_faults)} evaluation fault episodes")

    # Load hand graph
    hand_parent_map = load_hand_parent_map()
    print(f"Hand graph parent map loaded")

    # Compute baseline hand graph scores (using existing graph_scores)
    hand_scores = pd.read_parquet(DATA / "graph_scores.parquet")
    hand_graph_scores = hand_scores["graph_score"]

    # Load thresholds
    thresholds = json.loads((DATA / "nongraph_thresholds.json").read_text())

    # Compute graph threshold from calibration
    cal_scores = hand_scores.loc[cal_mask, "graph_score"].dropna()
    graph_threshold = float(np.quantile(cal_scores, 1 - 0.01))
    print(f"Graph threshold (FAR=0.01): {graph_threshold:.4f}")

    # Ablation results storage
    ablation_results = {
        "generated": pd.Timestamp.now("UTC").isoformat(),
        "hand_graph_threshold": graph_threshold,
        "ablations": {},
    }

    # For each ablation config
    for ablation_name, ablation_desc in ABLATION_CONFIGS.items():
        print(f"\nRunning ablation: {ablation_name} ({ablation_desc})...")

        # Apply ablation
        ablated_map = apply_ablation(hand_parent_map, ablation_name)
        print(f"  Ablated parent map: {dict((k, len(v)) for k, v in ablated_map.items() if v)}")

        # Compute residuals with ablated graph
        residuals, predictions = compute_graph_residuals(f, ablated_map, fit_mask)

        # Standardise
        z_scores = standardise_residuals(residuals, scales, f)

        # Compute graph score (max |z|)
        node_scores = {n: np.abs(z_scores[n]) for n in NODES}
        Z_mat = np.column_stack([node_scores[n] for n in NODES])
        graph_score = np.nanmax(Z_mat, axis=1)

        # Per-device threshold from calibration
        cal_scores = graph_score[cal_mask]
        dev_scores = f["device"].astype(str).to_numpy()

        # Per-device threshold
        thresholds_per_dev = {}
        for d in devices:
            dev_cal = cal_scores[dev_scores[cal_mask] == d]
            if len(dev_cal) > 20:
                thresholds_per_dev[d] = float(np.quantile(dev_cal, 1 - 0.01))
            else:
                thresholds_per_dev[d] = float(np.quantile(cal_scores, 1 - 0.01))

        # Apply thresholds and compute recall on evaluation faults
        eval_mask = f["time"] >= ts("2026-09-01")
        eval_scores = graph_score[eval_mask]
        eval_devices = f.loc[eval_mask, "device"].astype(str).to_numpy()

        # Evaluate on eval_faults
        eval_faults_df = pd.read_parquet(DATA / "evaluation_faults.parquet")

        # Build alarm matrix
        alarms = np.zeros(len(eval_scores), dtype=bool)
        for d in devices:
            m = eval_devices == d
            thresh = thresholds_per_dev.get(d, np.nan)
            if np.isfinite(thresh):
                alarms[m] = eval_scores[m] > thresh

        # Evaluate detection
        detected = 0
        for _, ep in pd.read_parquet(DATA / "evaluation_faults.parquet").iterrows():
            onset = ep["start_time"]
            end = ep["end_time"]
            window_end = end + pd.Timedelta(hours=6)

            # Check if any alarm in window
            ep_mask = (eval_devices == ep["device"]) & \
                      (f.loc[eval_mask, "time"].values >= onset) & \
                      (f.loc[eval_mask, "time"].values <= window_end)
            if alarms[ep_mask].any():
                detected += 1

        recall = detected / len(pd.read_parquet(DATA / "evaluation_faults.parquet"))

        ablation_results[ablation_name] = {
            "description": ablation_desc,
            "recall": recall,
            "detected": detected,
            "total_episodes": len(pd.read_parquet(DATA / "evaluation_faults.parquet")),
        }
        print(f"  Recall: {recall:.3f} ({detected}/{len(pd.read_parquet(DATA / 'evaluation_faults.parquet'))})")

    # Save ablation results
    ablation_results["ablations"] = {k: v for k, v in ablation_results.items() if k != "hand_graph"}
    (DATA / "ablation_results.json").write_text(json.dumps(ablation_results, indent=2, default=str))
    print(f"\nWrote ablation results to {DATA / 'ablation_results.json'}")
    print(f"Phase complete in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
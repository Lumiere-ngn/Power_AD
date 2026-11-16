"""Shared core for RQ2 extension scripts (controls, ablations, localisation).

Reuses the engineering-graph detector's ridge fits, residual standardisation,
graph scoring and event-recall evaluation so that every extension is evaluated
on exactly the same episodes and alert rules as the frozen primary experiment.

No detectors are defined here beyond the graph-conditioned residual score
already established in rq2_graph_detector.py.
"""
from __future__ import annotations

import json
import sys
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

FAR = 0.01
WINDOW_HOURS = 6


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def window_masks(t: pd.Series) -> dict[str, np.ndarray]:
    return {name: ((t >= ts(a)) & (t < ts(b))).to_numpy() for name, (a, b) in SPLITS.items()}


def load_eligible_devices() -> list[str]:
    inv = pd.read_csv(DATA / "device_inventory.csv")
    return inv.loc[inv["eligible_rq2"], "device"].tolist()


def load_features(devices: list[str]) -> pd.DataFrame:
    f = C.load()
    f = f[f["device"].astype(str).isin(devices)].copy()
    f = f.sort_values(["device", "time"]).reset_index(drop=True)
    f["device"] = f["device"].astype(str).astype("category")
    return f


def load_scales() -> dict:
    return json.loads((DATA / "residual_scales.json").read_text())


def fit_ridge_per_dev(f_fit: pd.DataFrame, node: str, parents: list[str]):
    """Per-device ridge of node on its graph parents. Returns (B, M, b) or (None,)*3."""
    if not parents:
        return None, None, None
    dev_codes = f_fit["device"].cat.codes.to_numpy()
    n_dev = len(f_fit["device"].cat.categories)
    y_all = f_fit[node].to_numpy(dtype=float)
    valid_mask = f_fit[parents].notna().all(axis=1) & f_fit[node].notna()
    if valid_mask.sum() < 100:
        return None, None, None
    B = np.zeros((n_dev, len(parents)))
    M = np.zeros((n_dev, len(parents)))
    b_vec = np.zeros(n_dev)
    for c in np.unique(dev_codes):
        idx = dev_codes == c
        y_c = y_all[idx]
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


def predict_node(f_pred: pd.DataFrame, parents: list[str], model, fallback_median: float) -> np.ndarray:
    B, M, b_vec = model
    if B is None or not parents:
        return np.full(len(f_pred), fallback_median, dtype=np.float32)
    X = f_pred[parents].to_numpy(dtype=float)
    preds = np.full(len(f_pred), fallback_median, dtype=np.float32)
    dev_codes = f_pred["device"].cat.codes.to_numpy()
    for c in np.unique(dev_codes):
        idx = np.where(dev_codes == c)[0]
        X_c = X[idx]
        valid = ~np.isnan(X_c).any(axis=1)
        if not valid.any():
            continue
        Xc_centered = X_c[valid] - M[c, :]
        pred_c = b_vec[c] + (Xc_centered @ B[c, :])
        preds[idx[valid]] = pred_c.astype(np.float32)
    preds = np.where(np.isfinite(preds), preds, fallback_median)
    return preds


def compute_graph_residuals(f: pd.DataFrame, parent_map: dict, fit_mask: np.ndarray):
    """Graph-conditioned residual per node (observed - predicted from parents)."""
    residuals, predictions = {}, {}
    for node in NODES:
        parents = parent_map.get(node, [])
        model = fit_ridge_per_dev(f.loc[fit_mask], node, parents)
        preds = predict_node(f, parents, model, f[node].median())
        residuals[node] = f[node].to_numpy(dtype=float) - preds
        predictions[node] = preds
    return residuals, predictions


def standardise_residuals(residuals: dict, scales: dict, f: pd.DataFrame) -> dict:
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


def graph_score_and_nodes(z_scores: dict) -> tuple[np.ndarray, np.ndarray]:
    """Graph score (max |z| over nodes) and the node |z| matrix."""
    node_matrix = np.column_stack([np.abs(z_scores[n]) for n in NODES])
    graph_score = np.nanmax(node_matrix, axis=1)
    return graph_score, node_matrix


def per_device_thresholds(graph_score: np.ndarray, cal_mask: np.ndarray,
                          devs: np.ndarray, devices: list[str], far: float = FAR) -> dict:
    """Per-device (1-far) quantile of the graph score on the calibration window."""
    cal_scores = graph_score[cal_mask]
    dev_cal = devs[cal_mask]
    out = {}
    for d in devices:
        vals = cal_scores[dev_cal == d]
        vals = vals[np.isfinite(vals)]
        if len(vals) >= 20:
            out[d] = float(np.quantile(vals, 1 - far))
        else:
            out[d] = float(np.nanquantile(cal_scores, 1 - far)) if np.isfinite(cal_scores).any() else np.nan
    return out


def alarm_matrix(graph_score: np.ndarray, thresholds: dict, devs: np.ndarray) -> np.ndarray:
    """Boolean alarm per row: graph_score > per-device threshold."""
    alarms = np.zeros(len(graph_score), dtype=bool)
    for d, thr in thresholds.items():
        m = devs == d
        if np.isfinite(thr):
            alarms[m] = graph_score[m] > thr
    return alarms


def evaluate_recall(alarms: np.ndarray, eval_mask: np.ndarray, eval_faults: pd.DataFrame,
                   devices: np.ndarray, times: np.ndarray) -> dict:
    """Event recall: episode counts as detected if any alarm in [onset, end+window)."""
    eval_dev = devices[eval_mask]
    eval_time = times[eval_mask]
    eval_alarm = alarms[eval_mask]

    detected = 0
    per_fault = {}
    per_severity = {}
    delays = []
    for _, ep in eval_faults.iterrows():
        onset = pd.Timestamp(ep["start_time"])
        end = pd.Timestamp(ep["end_time"])
        win_end = end + pd.Timedelta(hours=WINDOW_HOURS - 1)
        m = (eval_dev == ep["device"]) & (eval_time >= onset) & (eval_time <= win_end)
        hit = eval_alarm[m]
        found = bool(hit.any())
        if found:
            detected += 1
            per_fault[ep["fault"]] = per_fault.get(ep["fault"], 0) + 1
            per_severity[ep["severity"]] = per_severity.get(ep["severity"], 0) + 1
            # delay = first alarm hour minus onset
            first_hit = np.flatnonzero(hit)[0]
            delays.append(float((eval_time[m][first_hit] - onset).total_seconds() / 3600))
        else:
            per_fault.setdefault(ep["fault"], 0)
            per_severity.setdefault(ep["severity"], 0)

    return {
        "recall": detected / max(len(eval_faults), 1),
        "detected": detected,
        "n_episodes": len(eval_faults),
        "per_fault": per_fault,
        "per_severity": per_severity,
        "delay_median_hours": float(np.median(delays)) if delays else None,
    }


def false_alarm_rate(alarms: np.ndarray, valid_mask: np.ndarray) -> float:
    """Fraction of fault-free (valid) hours flagged."""
    v = valid_mask & np.isfinite(alarms.astype(float))
    if not v.any():
        return float("nan")
    return float(alarms[v].mean())
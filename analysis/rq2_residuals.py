#!/usr/bin/env python
"""Phase 3: shared residual framework.

Fits the RQ1-style context-only normal-behaviour model to all 10 RQ2 nodes,
computes residuals = observed - predicted, and standardises them using robust
scales (MAD) estimated on the July calibration window only.

Reuses the locked RQ1 settings per target family:
  - Power family (input_power_watts, orin_power_watts, cpu_gpu_power_watts,
    power, input_voltage_volts, input_current_amperes, input_energy_wh_total_rate):
    31 leaves, 200 trees, "pilot inputs" features (TIME + HW + ENV + WORKLOAD)
  - Thermal family (cpu_temp_celsius, gpu_temp_celsius, board_temp_celsius):
    31 leaves, 500 trees, "+ weather" features (pilot + WEATHER)

Chronological splits (RQ2):
  fit        2026-03-08 .. 2026-07-01   normal-model training
  calibrate  2026-07-01 .. 2026-08-01   residual scales and thresholds
  develop    2026-08-01 .. 2026-09-01   detector tuning (Phase 4)
  test       2026-09-01 .. 2026-10-07   final evaluation

Outputs
-------
analysis/out/rq2/residuals.parquet
  device, time, node, observed, predicted, residual, z, split

analysis/out/rq2/residual_scales.json
  Per-device, per-node robust scale (MAD * 1.4826) from July calibration

analysis/out/rq2/residual_stats.json
  Summary statistics per split and node

No anomaly detector is implemented here (Phase 3 only).
"""
from __future__ import annotations

import json
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq_common as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/out"
RQ2 = OUT / "rq2"
DATA = RQ2 / "data"

# RQ2 nodes (protocol.toml [rq2].nodes)
NODES = [
    "cpu_gpu_power_watts", "orin_power_watts", "power", "input_power_watts",
    "input_voltage_volts", "input_current_amperes", "input_energy_wh_total_rate",
    "cpu_temp_celsius", "gpu_temp_celsius", "board_temp_celsius",
]

# Node -> family assignment (mirrors RQ1)
POWER_NODES = ["input_power_watts", "orin_power_watts", "cpu_gpu_power_watts",
               "power", "input_voltage_volts", "input_current_amperes",
               "input_energy_wh_total_rate"]
THERMAL_NODES = ["cpu_temp_celsius", "gpu_temp_celsius", "board_temp_celsius"]

assert set(NODES) == set(POWER_NODES) | set(THERMAL_NODES)

# Context feature groups (protocol.toml [features])
TIME = ["hour_sin", "hour_cos", "dow", "is_weekend"]
HW = ["v56", "mem_total", "disk_total", "has_mppt", "has_env", "has_ds", "agency"]
ENV_EXT = ["peer_temperature", "peer_humidity", "peer_lightsensor"]
ENV_LOCAL = ["temperature", "humidity", "lightsensor"]
WORKLOAD = ["deepstream_app_running", "mem_used_gb", "wwan_rx_gb_rate", "wwan_tx_gb_rate",
            "pir_count_rate", "ds_consumer_error_total_rate"]
WEATHER = ["wx_temp_c", "wx_humidity_pct", "wx_dewpoint_c", "wx_cloud_pct",
           "wx_wind_ms", "wx_solar_wm2", "wx_precip_mm"]

CONTEXT = TIME + HW + ENV_EXT + ENV_LOCAL + WORKLOAD
CONTEXT_WITH_WEATHER = CONTEXT + WEATHER

# Locked RQ1 settings per family
LOCKED = {
    "power": {
        "features": CONTEXT,
        "params": {"num_leaves": 31, "n_estimators": 200},
    },
    "thermal": {
        "features": CONTEXT_WITH_WEATHER,
        "params": {"num_leaves": 31, "n_estimators": 500},
    },
}

# LGBM base params (from rq_common.lgbm_params())
LGBM_BASE = dict(
    learning_rate=0.05, min_child_samples=100, subsample=0.8,
    subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    n_jobs=8, verbose=-1, random_state=20261103,
    deterministic=True, force_row_wise=True,
)

# Chronological splits
SPLITS = {
    "fit": ("2026-03-08", "2026-07-01"),
    "calibrate": ("2026-07-01", "2026-08-01"),
    "develop": ("2026-08-01", "2026-09-01"),
    "test": ("2026-09-01", "2026-10-07"),
}

FLOOR = {
    "input_voltage_volts": 0.02,
    "input_current_amperes": 0.005,
    "input_energy_wh_total_rate": 0.05,
    "cpu_temp_celsius": 0.3,
    "gpu_temp_celsius": 0.3,
    "board_temp_celsius": 0.3,
    "input_power_watts": 0.05,
    "orin_power_watts": 0.05,
    "cpu_gpu_power_watts": 0.05,
    "power": 0.05,
}


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def window_masks(t: pd.Series) -> dict[str, np.ndarray]:
    return {name: ((t >= ts(a)) & (t < ts(b))).to_numpy() for name, (a, b) in SPLITS.items()}


def split_labels(t: pd.Series) -> pd.Series:
    lab = pd.Series("other", index=t.index, dtype=str)
    for name, m in window_masks(t).items():
        lab[m] = name
    return lab


def load_eligible_devices() -> list[str]:
    """Load the Phase 1 device inventory and return RQ2-eligible devices."""
    inv = pd.read_csv(DATA / "device_inventory.csv")
    return inv.loc[inv["eligible_rq2"], "device"].tolist()


def load_features(devices: list[str]) -> pd.DataFrame:
    """Load features.parquet for eligible devices with required columns.

    Uses rq_common.load() which already drops excluded agencies and adds
    hardware flags, peer environment and mem_used_gb.
    """
    f = C.load()
    f = f[f["device"].astype(str).isin(devices)].copy()
    f = f.sort_values(["device", "time"]).reset_index(drop=True)
    return f


def family_of(node: str) -> str:
    return "power" if node in POWER_NODES else "thermal"


def lgbm_params(node: str) -> dict:
    fam = family_of(node)
    p = LGBM_BASE.copy()
    p.update(LOCKED[fam]["params"])
    return p


def fit_node_models(f: pd.DataFrame, masks: dict[str, np.ndarray], devices: list[str]) -> dict:
    """Fit one context-only LightGBM per node on the fit window."""
    fit_mask = masks["fit"] & f[NODES].notna().any(axis=1)
    f_fit = f[fit_mask].copy()
    models = {}

    for node in NODES:
        fam = family_of(node)
        feats = LOCKED[fam]["features"]
        # Drop rows where target or features are NaN
        valid = f_fit[node].notna() & f_fit[feats].notna().all(axis=1)
        X = f_fit.loc[valid, feats]
        y = f_fit.loc[valid, node]
        if len(X) < 1000:
            raise ValueError(f"Not enough training data for {node}: {len(X)}")
        m = lgb.LGBMRegressor(**lgbm_params(node)).fit(X, y)
        models[node] = (m, feats)
        print(f"  {node:24s} fit: {len(X):,} rows, features: {len(feats)}")
    return models


def predict_all(f: pd.DataFrame, models: dict) -> dict:
    """Predict all nodes for all rows."""
    preds = {}
    for node, (m, feats) in models.items():
        preds[node] = m.predict(f[feats], num_threads=8)
    return preds


def compute_residuals(f: pd.DataFrame, preds: dict) -> dict:
    """residual = observed - predicted."""
    res = {}
    for node in NODES:
        res[node] = f[node].to_numpy(dtype=float) - preds[node]
    return res


def estimate_scales(residuals: dict, masks: dict[str, np.ndarray]) -> tuple[dict, dict]:
    """Estimate robust per-device, per-node scale (MAD * 1.4826) on July calibration window."""
    cal_mask = masks["calibrate"]
    scales = {}
    levels = {}

    for node in NODES:
        r = residuals[node][cal_mask]
        dev = f.loc[cal_mask, "device"].astype(str).to_numpy()
        levels_node = {}
        scale_node = {}

        for d in np.unique(dev):
            rd = r[dev == d]
            rd = rd[np.isfinite(rd)]
            if len(rd) > 20:
                mad = np.median(np.abs(rd - np.median(rd))) * 1.4826
                scale_node[d] = max(mad, FLOOR.get(node, 0.05))
                levels_node[d] = np.median(rd)
            else:
                scale_node[d] = FLOOR.get(node, 0.05)
                levels_node[d] = 0.0

        scales[node] = scale_node
        levels[node] = levels_node

    return scales, levels


def standardise(residuals: dict, scales: dict, levels: dict, f: pd.DataFrame) -> dict:
    """z = (residual - median_residual) / scale, per device per node."""
    devs = f["device"].astype(str).to_numpy()
    z = {}
    for node in NODES:
        r = residuals[node]
        z_node = np.full_like(r, np.nan)
        for d, sc in scales[node].items():
            m = devs == d
            if m.any():
                z_node[m] = (r[m] - levels[node][d]) / sc
        z[node] = z_node
    return z


def save_outputs(f, masks, preds, residuals, z, scales, levels, devices):
    """Save residuals.parquet and metadata JSONs."""
    DATA.mkdir(parents=True, exist_ok=True)

    # Build long-form residuals table
    rows = []
    for node in NODES:
        df = pd.DataFrame({
            "device": f["device"].astype(str),
            "time": f["time"],
            "node": node,
            "observed": f[node].to_numpy(),
            "predicted": preds[node],
            "residual": residuals[node],
            "z": z[node],
        })
        df["split"] = split_labels(df["time"])
        df = df[df["split"] != "other"]
        rows.append(df)

    residuals_df = pd.concat(rows, ignore_index=True)
    residuals_path = DATA / "residuals.parquet"
    residuals_df.to_parquet(residuals_path, index=False)
    print(f"Wrote {residuals_path} ({len(residuals_df):,} rows)")

    # Scales JSON
    scales_json = {}
    for node in NODES:
        scales_json[node] = {
            "scale": {k: float(v) for k, v in scales[node].items()},
            "level": {k: float(v) for k, v in levels[node].items()},
        }
    scales_path = DATA / "residual_scales.json"
    scales_path.write_text(json.dumps(scales_json, indent=2))
    print(f"Wrote {scales_path}")

    # Summary stats
    stats = {"generated": datetime.now(timezone.utc).isoformat(), "nodes": NODES}
    for split_name, mask in masks.items():
        if split_name == "other":
            continue
        stats[split_name] = {}
        for node in NODES:
            mask_arr = mask & np.isfinite(z[node])
            if mask_arr.any():
                stats[split_name][node] = {
                    "residual_mean": float(np.mean(residuals[node][mask_arr])),
                    "residual_std": float(np.std(residuals[node][mask_arr])),
                    "z_mean": float(np.mean(z[node][mask_arr])),
                    "z_std": float(np.std(z[node][mask_arr])),
                    "n": int(mask_arr.sum()),
                }
    stats_path = DATA / "residual_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, default=str))
    print(f"Wrote {stats_path}")


def main():
    DATA.mkdir(parents=True, exist_ok=True)

    # Load eligible devices from Phase 1
    devices = load_eligible_devices()
    print(f"Phase 3: {len(devices)} RQ2-eligible devices")

    # Load features
    global f
    f = load_features(devices)
    print(f"  Loaded {len(f):,} rows")

    # Split masks
    masks = window_masks(f["time"])
    for name, m in masks.items():
        print(f"  {name:9s}: {int(m.sum()):,} rows")

    # Fit context-only models on fit window
    print("Fitting context-only LightGBM per node...")
    models = fit_node_models(f, masks, devices)

    # Predict on all windows
    print("Predicting...")
    preds = predict_all(f, models)

    # Residuals
    print("Computing residuals...")
    residuals = compute_residuals(f, preds)

    # Estimate scales on July calibration window only
    print("Estimating robust scales on July calibration window...")
    scales, levels = estimate_scales(residuals, masks)

    # Standardise
    print("Standardising residuals...")
    z_scores = standardise(residuals, scales, levels, f)

    # Save
    print("Saving outputs...")
    save_outputs(f, masks, preds, residuals, z_scores, scales, levels, devices)

    print("\nPhase 3 complete.")


if __name__ == "__main__":
    main()
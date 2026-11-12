"""Shared data preparation for the research-question experiments (rq1.py, rq2_graph.py).

Starts from analysis/out/features.parquet (device × hour) and adds:
  * mem_used_gb = memory_total_gb - memory_free_gb (workload proxy)
  * per-device hardware configuration, constant over the device's life (median over its history):
    v56 (56 V supply vs 12 V), mem_total, disk_total, has_mppt (solar charge controller), has_env
    (heater / environment sensor), has_ds (DeepStream ever ran)
  * WEATHER names the city-weather columns that features.py already joined (wx_*)
  * peer environment: leave-one-out mean of the enclosure sensors (temperature, humidity, light) over the
    other devices of the same agency in the same hour, so devices without an environment sensor still get
    an environmental context
Feature groups and split dates used by both research questions are defined here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/out"

# temporal splits (UTC): models fit before CAL, residual scales calibrate on [CAL, TEST), evaluation from TEST
CAL = pd.Timestamp("2026-07-01", tz="UTC")
TEST = pd.Timestamp("2026-08-01", tz="UTC")

POWER = ["input_power_watts", "orin_power_watts", "cpu_gpu_power_watts"]
THERMAL = ["cpu_temp_celsius", "gpu_temp_celsius", "board_temp_celsius"]
TARGETS = POWER + THERMAL

TIME = ["hour_sin", "hour_cos", "dow", "is_weekend"]
HW = ["v56", "mem_total", "disk_total", "has_mppt", "has_env", "has_ds", "agency"]
ENV_EXT = ["peer_temperature", "peer_humidity", "peer_lightsensor"]
ENV_LOCAL = ["temperature", "humidity", "lightsensor"]
WORKLOAD = ["deepstream_app_running", "mem_used_gb", "wwan_rx_gb_rate", "wwan_tx_gb_rate",
            "pir_count_rate", "ds_consumer_error_total_rate"]
# hourly weather of the agency's city (weather.py; protocol decision D4). Missing for agencies without a city.
WEATHER = ["wx_temp_c", "wx_humidity_pct", "wx_dewpoint_c", "wx_cloud_pct", "wx_wind_ms", "wx_solar_wm2", "wx_precip_mm"]
CATEGORICAL = ["agency", "device"]

LABEL = {
    "input_power_watts": "Input power", "orin_power_watts": "Orin power", "cpu_gpu_power_watts": "CPU+GPU power",
    "power": "Heartbeat power", "input_current_amperes": "Input current", "input_voltage_volts": "Input voltage",
    "input_energy_wh_total_rate": "Energy counter", "cpu_temp_celsius": "CPU temp", "gpu_temp_celsius": "GPU temp",
    "board_temp_celsius": "Board temp",
}
UNIT = {t: "W" for t in POWER} | {t: "°C" for t in THERMAL}


def hardware(f: pd.DataFrame) -> pd.DataFrame:
    g = f.groupby("device", observed=True)
    hw = pd.DataFrame({
        "v56": (g["input_voltage_volts"].median() > 30).astype("int8"),
        "mem_total": g["memory_total_gb"].median().round(1),
        "disk_total": g["storage_total_gb"].median().round(0),
        "has_mppt": (g["mppt_charge_power"].count() > 0).astype("int8"),
        "has_env": (g["temperature"].count() > 0).astype("int8"),
        "has_ds": (g["deepstream_app_running"].max() > 0).astype("int8"),
    })
    hw.index = hw.index.astype(str)
    return hw


def peer_env(f: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-out agency mean of each enclosure sensor in the same hour."""
    out = {}
    for c in ENV_LOCAL:
        v = f[c]
        key = [f["agency"].astype(str), f["time"]]
        s = v.groupby(key).transform("sum")
        n = v.notna().groupby(key).transform("sum")
        own = v.fillna(0.0)
        out[f"peer_{c}"] = ((s - own) / (n - v.notna())).where(n - v.notna() > 0)
    return pd.DataFrame(out, index=f.index)


def load(columns: list[str] | None = None, before: pd.Timestamp | None = None) -> pd.DataFrame:
    """`before` keeps only the hours before that time, so later months cannot enter the hardware or peer columns."""
    f = pd.read_parquet(OUT / "features.parquet", columns=columns)
    if before is not None:
        f = f[f["time"] < before].reset_index(drop=True)
    f["device"] = f["device"].astype(str)
    f["mem_used_gb"] = f["memory_total_gb"] - f["memory_free_gb"]
    hw = hardware(f)
    f = f.join(hw, on="device")
    f = pd.concat([f, peer_env(f)], axis=1)
    f["device"] = f["device"].astype("category")
    f["agency"] = f["agency"].astype(str).astype("category")
    return f


def config_name(row) -> str:
    return f"{'56 V' if row.v56 else '12 V'}{' · solar' if row.has_mppt else ''}"


def lgbm_params(**kw) -> dict:
    p = dict(n_estimators=500, learning_rate=0.05, num_leaves=63, min_child_samples=100, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=8, verbose=-1)
    p.update(kw)
    return p


def clean(o):
    """JSON-safe copy: NaN / inf -> None, numpy scalars -> Python."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o

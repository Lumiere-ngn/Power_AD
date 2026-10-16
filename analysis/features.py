#!/usr/bin/env python
"""Build one modelling-ready feature table: device × hour × all 42 metrics, plus derived features.

Per (device, hour):
  * gauges / binary metrics -> hourly mean of clean samples (placeholder codes excluded)
  * counters (running totals that reset daily) -> hourly increase, reset-aware (`*_rate`)
  * `up` (edge scrape job only) -> share of successful scrapes that hour
  * samples_<metric> columns are not kept; a per-row `n_samples` gives total sample count
  * time features: hour sin/cos, day of week, weekend flag, days since the device's first sample
  * agency label (categorical) from the device's labels
Rows are a complete hourly grid from each device's first to last hour, so gaps show up as NaN.

Output: analysis/out/features.parquet (and a column dictionary analysis/out/features_columns.csv).
Usage: python -I analysis/features.py [--freq h]
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset/smart-sign-telemetry/prometheus"
OUT = ROOT / "analysis/out"

COUNTERS = {
    "device_input_energy_wh_total", "device_pir_count", "device_ds_consumer_error_total",
    "device_wwan_rx_gb", "device_wwan_tx_gb",
}
# Placeholder / impossible values documented in the dataset README; excluded before aggregation.
PLACEHOLDER = {
    "device_gpu_temp_celsius": lambda v: v <= -200,
    "device_mppt_min_battery_temperature": lambda v: v >= 1000,
    "device_mppt_max_battery_temperature": lambda v: v >= 1000,
    "device_mppt_battery_temperature": lambda v: v >= 120,
    "device_mppt_battery_charge_current": lambda v: v == 100,
    "device_mppt_battery_voltage": lambda v: v == 100,
    "device_mppt_solar_voltage": lambda v: v == 100,
    "device_battery_remaining_capacity": lambda v: v > 100,
    "device_mppt_battery_remaining_capacity": lambda v: v > 100,
    "device_mppt_battery_remaining_capacity_mppt": lambda v: v > 100,
}
short = lambda m: m.removeprefix("device_")


def load_metric(metric: str) -> pd.DataFrame:
    """Long frame of clean samples: ts (ms), value, device, agency."""
    parts = []
    for f in sorted(DATA.glob(f"metric={metric}/month=*/part-000.parquet")):
        t = pq.read_table(f, columns=["timestamp", "value", "labels"])
        lab = t.column("labels").cast(pa.large_string()).combine_chunks().dictionary_encode()
        raw, ag = [], []
        for s in lab.dictionary.to_pylist():
            d = json.loads(s)
            ok = d.get("job") == "edge" if metric == "up" else True
            raw.append((d.get("device_id") or d.get("instance")) if ok else None)
            ag.append(d.get("agency"))
        idx = lab.indices.to_numpy()
        raw_arr = np.array(raw, dtype=object)[idx]
        df = pd.DataFrame({
            "ts": t.column("timestamp").cast(pa.int64()).to_numpy(),
            "value": t.column("value").to_numpy(zero_copy_only=False),
            "raw": raw_arr,  # device id as recorded (one per scrape target)
            # device_id letter case changes over some devices' lives; the device is the lower-cased id
            "device": pd.Series(raw_arr, dtype=object).str.lower().to_numpy(),
            "agency": np.array(ag, dtype=object)[idx],
        })
        df = df[df["device"].notna() & np.isfinite(df["value"])]
        if metric in PLACEHOLDER:
            df = df[~PLACEHOLDER[metric](df["value"])]
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def aggregate(metric: str, freq_ms: int) -> pd.DataFrame:
    df = load_metric(metric)
    df["bucket"] = df["ts"] // freq_ms
    name = short(metric)
    if metric in COUNTERS:
        # reset-aware increase per scrape target: a drop means the total restarted, so the new value
        # is the increase. Two targets of one device report the same total, so combine them with max.
        df = df.sort_values(["raw", "ts"])
        prev = df.groupby("raw")["value"].shift()
        inc = df["value"] - prev
        df["inc"] = inc.where(inc >= 0, df["value"]).fillna(0.0)
        per_raw = df.groupby(["device", "bucket", "raw"]).agg(inc=("inc", "sum"), n=("value", "size"))
        g = per_raw.groupby(level=["device", "bucket"]).agg(**{f"{name}_rate": ("inc", "max"), "n": ("n", "max")})
    else:
        g = df.groupby(["device", "bucket"]).agg(**{name: ("value", "mean"), "n": ("value", "size")})
    g = g.rename(columns={"n": f"n_{name}"})
    agency = df.groupby("device")["agency"].agg(lambda s: s.mode().iloc[0] if len(s) else None)
    return g, agency


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", default="h", help="pandas offset alias for the bucket (h, 30min, D)")
    ap.add_argument("--out", default=str(OUT / "features.parquet"))
    args = ap.parse_args()
    freq_ms = int(pd.tseries.frequencies.to_offset(args.freq).nanos // 1_000_000)
    metrics = sorted(p.name.split("=", 1)[1] for p in DATA.glob("metric=*"))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(42, len(metrics))) as ex:
        results = list(ex.map(aggregate, metrics, [freq_ms] * len(metrics)))
    print(f"aggregated {len(metrics)} metrics in {time.time() - t0:.0f}s", flush=True)

    frames, agencies = zip(*results)
    wide = pd.concat(frames, axis=1)
    n_cols = [c for c in wide.columns if c.startswith("n_")]
    wide["n_samples"] = wide[n_cols].fillna(0).sum(axis=1).astype(int)
    wide = wide.drop(columns=n_cols)

    # complete hourly grid per device, so missing hours are explicit NaN rows
    grids = []
    for dev, g in wide.groupby(level="device"):
        b = g.index.get_level_values("bucket")
        grids.append(pd.MultiIndex.from_product([[dev], np.arange(b.min(), b.max() + 1)], names=["device", "bucket"]))
    full = wide.reindex(pd.MultiIndex.from_tuples([t for gi in grids for t in gi], names=["device", "bucket"]))
    full["n_samples"] = full["n_samples"].fillna(0).astype(int)
    full = full.reset_index()
    full["time"] = pd.to_datetime(full["bucket"] * freq_ms, unit="ms", utc=True)
    full = full.drop(columns="bucket")

    # time and identity features
    h = full["time"].dt.hour + full["time"].dt.minute / 60
    full["hour_sin"], full["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    full["dow"] = full["time"].dt.dayofweek.astype("int8")
    full["is_weekend"] = (full["dow"] >= 5).astype("int8")
    first = full.groupby("device")["time"].transform("min")
    full["days_since_first"] = ((full["time"] - first).dt.total_seconds() / 86400).astype("float32")
    ag = pd.concat(agencies).groupby(level=0).first()
    ag = ag.map(lambda a: {"balharbour": "BalHarbour", "demo": "demo"}.get(str(a).lower(), a) if a else None)
    full["agency"] = full["device"].map(ag).astype("category")
    full["device"] = full["device"].astype("category")
    full["has_data"] = (full["n_samples"] > 0).astype("int8")

    lead = ["device", "agency", "time", "has_data", "n_samples", "hour_sin", "hour_cos", "dow", "is_weekend", "days_since_first"]
    full = full[lead + sorted(c for c in full.columns if c not in lead)]
    full.to_parquet(args.out, index=False)

    cols = pd.DataFrame({
        "column": full.columns,
        "dtype": [str(d) for d in full.dtypes],
        "non_null": [int(full[c].notna().sum()) for c in full.columns],
        "kind": ["id/time" if c in lead else "counter rate per bucket" if c.endswith("_rate") else "share of successful scrapes" if c == "up" else "bucket mean" for c in full.columns],
    })
    cols.to_csv(OUT / "features_columns.csv", index=False)
    print(f"features: {full.shape[0]:,} rows × {full.shape[1]} columns, {full['device'].nunique()} devices, "
          f"{full['has_data'].mean():.1%} of grid rows have data -> {args.out} ({time.time() - t0:.0f}s)")
    print(cols.to_string(index=False))


if __name__ == "__main__":
    main()

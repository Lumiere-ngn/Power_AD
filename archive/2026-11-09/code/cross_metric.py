#!/usr/bin/env python
"""Cross-metric, fleet-level and forecasting analysis with River.

Stage 1 builds an hourly per-device table (mean of clean samples) for the key
metrics and saves it to analysis/out/hourly.parquet. Stage 2 streams that
table through River:
  * covariance.EmpiricalCovariance -> fleet-wide metric correlation matrix
  * stats.PearsonCorr per device for a few physically meaningful pairs
  * cluster.KMeans on standardized per-device fingerprints
  * drift.ADWIN + a streaming slope on per-device DAILY means (regime changes, trends)
  * time_series.HoltWinters / SNARIMAX vs naive baselines, progressive validation
  * availability from `up` (job=edge)
Results go to analysis/out/cross.json.

Usage: python -I analysis/cross_metric.py [--skip-hourly]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from river import cluster, covariance, drift, preprocessing, stats, time_series

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset/smart-sign-telemetry/prometheus"
OUT = ROOT / "analysis/out"
HOURLY = OUT / "hourly.parquet"
HOUR_MS = 3_600_000

# Same placeholder rules as profile_stream.py (README "Placeholder and impossible values").
PLACEHOLDER = {
    "device_gpu_temp_celsius": lambda v: v <= -200,
    "device_mppt_battery_voltage": lambda v: v == 100,
    "device_mppt_solar_voltage": lambda v: v == 100,
    "device_battery_remaining_capacity": lambda v: v > 100,
}

KEY = [
    "device_input_voltage_volts", "device_input_power_watts", "device_input_current_amperes",
    "device_cpu_gpu_power_watts", "device_orin_power_watts", "device_power",
    "device_cpu_temp_celsius", "device_gpu_temp_celsius", "device_board_temp_celsius",
    "device_temperature", "device_humidity", "device_lightsensor",
    "device_memory_free_gb", "device_storage_free_gb",
    "device_battery_remaining_capacity", "device_input_soc_poly5_percent",
    "device_mppt_charge_power", "device_mppt_battery_voltage", "device_mppt_solar_voltage",
    "device_deepstream_app_running", "up",
]
short = lambda m: m.removeprefix("device_")

# Metrics reported by (almost) the whole fleet: used for the correlation core and fingerprints.
CORE = ["input_voltage_volts", "input_power_watts", "cpu_gpu_power_watts", "orin_power_watts",
        "cpu_temp_celsius", "gpu_temp_celsius", "board_temp_celsius", "memory_free_gb",
        "storage_free_gb", "deepstream_app_running", "up"]
PAIRS = [("cpu_temp_celsius", "board_temp_celsius"), ("input_power_watts", "cpu_gpu_power_watts"),
         ("input_voltage_volts", "battery_remaining_capacity"), ("lightsensor", "mppt_charge_power"),
         ("temperature", "board_temp_celsius"), ("humidity", "temperature"),
         ("battery_remaining_capacity", "input_soc_poly5_percent"), ("cpu_temp_celsius", "cpu_gpu_power_watts")]
HEALTH = ["memory_free_gb", "storage_free_gb", "battery_remaining_capacity", "input_voltage_volts",
          "cpu_temp_celsius", "board_temp_celsius", "input_power_watts", "mppt_battery_voltage"]
FORECAST = ["board_temp_celsius", "input_voltage_volts", "input_power_watts", "lightsensor"]
HORIZON = 12


# ----------------------------------------------------------------------------- stage 1: hourly table
def hourly_for_metric(metric: str) -> pd.DataFrame:
    files = sorted(DATA.glob(f"metric={metric}/month=*/part-000.parquet"))
    parts = []
    for f in files:  # one month at a time: a whole metric's label strings overflow pyarrow's 2 GB string offsets
        t = pq.read_table(f, columns=["timestamp", "value", "labels"])
        lab = t.column("labels").cast(pa.large_string()).combine_chunks().dictionary_encode()
        dev_of = []
        for s in lab.dictionary.to_pylist():
            d = json.loads(s)
            if metric == "up" and d.get("job") != "edge":
                dev_of.append(None)  # only the device scrape job counts as device availability
            else:  # device_id letter case changes over some devices' lives
                dev_of.append((d.get("device_id") or "").lower() or d.get("instance"))
        dev_arr = np.array(dev_of, dtype=object)
        df = pd.DataFrame({
            "hour": t.column("timestamp").cast(pa.int64()).to_numpy() // HOUR_MS,
            "value": t.column("value").to_numpy(zero_copy_only=False),
            "device": dev_arr[lab.indices.to_numpy()],
        })
        df = df[df["device"].notna() & np.isfinite(df["value"])]
        if metric in PLACEHOLDER:
            df = df[~PLACEHOLDER[metric](df["value"])]
        parts.append(df.groupby(["device", "hour"], sort=True)["value"].agg(["sum", "count"]))
    g = pd.concat(parts).groupby(level=["device", "hour"]).sum().reset_index()
    g["mean"] = g["sum"] / g["count"]
    g = g.drop(columns="sum")
    g["metric"] = short(metric)
    return g


def build_hourly() -> pd.DataFrame:
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=len(KEY)) as ex:
        parts = list(ex.map(hourly_for_metric, KEY))
    h = pd.concat(parts, ignore_index=True)
    h["hour"] = pd.to_datetime(h["hour"] * HOUR_MS, unit="ms", utc=True)
    h.to_parquet(HOURLY, index=False)
    print(f"hourly table: {len(h):,} rows, {h['device'].nunique()} devices in {time.time() - t0:.0f}s", flush=True)
    return h


# ----------------------------------------------------------------------------- stage 2 helpers
def corr_from_cov(cov: covariance.EmpiricalCovariance, names: list[str]) -> dict:
    out = {}
    for a in names:
        for b in names:
            try:  # cov[a, b] is a River Cov/Var object, not a float
                cab, caa, cbb = cov[a, b].get(), cov[a, a].get(), cov[b, b].get()
                r = cab / math.sqrt(caa * cbb) if caa and cbb and caa > 0 and cbb > 0 else None
            except (KeyError, TypeError):
                r = None
            out[f"{a}|{b}"] = None if r is None or not math.isfinite(r) else r
    return out


def regime_changes(wide: pd.DataFrame) -> tuple[list, list]:
    """ADWIN and a streaming least-squares slope on per-device daily means."""
    day = wide.index.get_level_values("hour").floor("D")
    daily = wide.groupby([wide.index.get_level_values("device"), day]).mean()
    daily.index.names = ["device", "day"]
    changes, trends = [], []
    for dev, d in daily.groupby(level="device"):
        d = d.droplevel("device")
        for m in HEALTH:
            if m not in d:
                continue
            s = d[m].dropna()
            if len(s) < 30:
                continue
            adwin = drift.ADWIN(delta=0.002)
            cov, var, mean = stats.Cov(), stats.Var(), stats.Mean()
            x0 = s.index[0]
            for ts, v in s.items():
                before = adwin.estimation
                adwin.update(v)
                if adwin.drift_detected:
                    changes.append({"device": dev, "metric": m, "day": ts.strftime("%Y-%m-%d"),
                                    "before": before, "after": adwin.estimation})
                xd = (ts - x0).days
                cov.update(xd, v); var.update(xd); mean.update(v)
            slope = cov.get() / var.get() if var.get() else 0.0  # units per day
            trends.append({"device": dev, "metric": m, "days": len(s), "mean": mean.get(),
                           "slope_per_30d": slope * 30, "first": float(s.iloc[0]), "last": float(s.iloc[-1])})
    return changes, trends


def contiguous_hourly(s: pd.Series, max_gap_h: int = 6) -> pd.Series:
    """Longest run of hourly values after linearly filling gaps of <= max_gap_h hours."""
    s = s.dropna()
    if s.empty:
        return s
    full = s.resample("h").mean()
    filled = full.interpolate(limit=max_gap_h, limit_area="inside")
    isna = filled.isna().to_numpy()
    best, cur, start, best_start = 0, 0, 0, 0
    for i, na in enumerate(isna):
        if na:
            cur, start = 0, i + 1
        else:
            cur += 1
            if cur > best:
                best, best_start = cur, start
    return filled.iloc[best_start:best_start + best]


class Naive:
    def __init__(self): self.last = None
    def learn_one(self, y): self.last = y
    def forecast(self, horizon): return [self.last] * horizon


class SeasonalNaive:
    def __init__(self, m=24): self.m, self.hist = m, []
    def learn_one(self, y): self.hist.append(y)
    def forecast(self, horizon):
        if len(self.hist) < self.m:
            return [self.hist[-1]] * horizon
        return [self.hist[-self.m + (h % self.m)] for h in range(horizon)]


def make_models() -> dict:
    return {
        "naive": Naive(),
        "seasonal_naive": SeasonalNaive(24),
        "holt_winters": time_series.HoltWinters(alpha=0.3, beta=0.05, gamma=0.5, seasonality=24),
        # Undifferenced / MA variants diverge on recursive 12-step forecasts (unregularized SGD
        # coefficients > 1); first differencing with a seasonal AR term stays bounded.
        "snarimax": time_series.SNARIMAX(p=3, d=1, q=0, m=24, sp=1, sq=0),
    }


def evaluate_forecasts(vals: list[float], warm: int = 168) -> tuple[dict, list]:
    """Progressive validation: at each step forecast HORIZON ahead, then learn the true value."""
    models = make_models()
    err = {k: [0.0] * HORIZON for k in models}
    n = 0
    trace = []  # (actual, hw_1step, hw_12step) for the tail
    lo = hi = vals[0]
    for i, y in enumerate(vals):
        if i >= warm and i + HORIZON <= len(vals):
            n += 1
            pad = 0.1 * (hi - lo)
            for k, mdl in models.items():
                preds = mdl.forecast(horizon=HORIZON)
                # any deployed forecaster would clamp to the range seen so far; so do we
                preds = [min(max(p, lo - pad), hi + pad) if p is not None and math.isfinite(p) else y for p in preds]
                for h in range(HORIZON):
                    err[k][h] += abs(preds[h] - vals[i + h])
                if k == "holt_winters" and i >= len(vals) - HORIZON - 168:
                    trace.append((y, preds[0], preds[HORIZON - 1]))
        for mdl in models.values():
            mdl.learn_one(y)
        lo, hi = min(lo, y), max(hi, y)
    return {k: [e / n for e in v] for k, v in err.items()} if n else {}, trace


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-hourly", action="store_true", help="reuse analysis/out/hourly.parquet")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    h = pd.read_parquet(HOURLY) if args.skip_hourly and HOURLY.exists() else build_hourly()
    wide = h.pivot_table(index=["device", "hour"], columns="metric", values="mean")
    names = list(wide.columns)
    print(f"wide: {wide.shape}", flush=True)

    # --- fleet-wide correlation via streaming covariance ---------------------------------------
    cov = covariance.EmpiricalCovariance()
    pair_corr = defaultdict(lambda: defaultdict(stats.PearsonCorr))
    for (dev, _), row in wide.iterrows():
        x = {k: v for k, v in row.items() if v == v}
        if len(x) >= 2:
            cov.update(x)
        for a, b in PAIRS:
            if a in x and b in x:
                pair_corr[(a, b)][dev].update(x[a], x[b])
    corr = corr_from_cov(cov, names)
    pairs_out = []
    for (a, b), devs in pair_corr.items():
        rs = sorted((r.get(), d) for d, r in devs.items() if r.get() is not None and math.isfinite(r.get()))
        if rs:
            vals = [r for r, _ in rs]
            pairs_out.append({"a": a, "b": b, "n_devices": len(rs), "median_r": float(np.median(vals)),
                              "min_r": rs[0][0], "min_dev": rs[0][1], "max_r": rs[-1][0], "max_dev": rs[-1][1],
                              "per_device": {d: r for r, d in rs}})
    print(f"correlation done {time.time() - t0:.0f}s", flush=True)

    # --- device fingerprints + streaming KMeans ----------------------------------------------
    med = wide.groupby(level="device")[CORE].median()
    hours = wide.groupby(level="device").size()
    med = med[(hours >= 7 * 24)].dropna()
    scaler = preprocessing.StandardScaler()
    rows = [dict(r) for _, r in med.iterrows()]
    for r in rows:
        scaler.learn_one(r)
    km = cluster.KMeans(n_clusters=4, halflife=0.3, seed=42)
    rng = random.Random(0)
    for _ in range(25):
        rng.shuffle(rows)
        for r in rows:
            km.learn_one(scaler.transform_one(r))
    assign = {dev: km.predict_one(scaler.transform_one(dict(r))) for dev, r in med.iterrows()}
    med["cluster"] = pd.Series(assign)
    med["hours"] = hours
    clusters = {int(c): {"n": int((med["cluster"] == c).sum()),
                         "devices": sorted(med.index[med["cluster"] == c].tolist()),
                         "centroid": med[med["cluster"] == c][CORE].median().round(3).to_dict()}
                for c in sorted(med["cluster"].unique())}
    print(f"clustering done {time.time() - t0:.0f}s", flush=True)

    # --- regime changes and trends on daily means --------------------------------------------
    changes, trends = regime_changes(wide)
    print(f"regimes done: {len(changes)} changes, {len(trends)} trends {time.time() - t0:.0f}s", flush=True)

    # --- forecasting ---------------------------------------------------------------------------
    forecasts = {}
    for m in FORECAST:
        if m not in wide:
            continue
        cover = wide[m].groupby(level="device").count().sort_values(ascending=False)
        per_model, traces, used = defaultdict(lambda: np.zeros(HORIZON)), {}, 0
        for dev in cover.index[:8]:
            s = wide.loc[dev, m]
            s.index = pd.DatetimeIndex(s.index)
            seg = contiguous_hourly(s)
            if len(seg) < 24 * 21:
                continue
            res, trace = evaluate_forecasts(seg.to_numpy().tolist())
            if not res:
                continue
            used += 1
            for k, v in res.items():
                per_model[k] += np.array(v)
            if not traces:
                traces = {"device": dev, "start": seg.index[-len(trace)].isoformat(), "points": trace}
        if used:
            forecasts[m] = {"n_devices": used, "mae_by_horizon": {k: (v / used).round(4).tolist() for k, v in per_model.items()},
                            "scale_p50": float(wide[m].median()), "example": traces}
        print(f"forecast {m}: {used} devices {time.time() - t0:.0f}s", flush=True)

    # --- availability --------------------------------------------------------------------------
    up = h[h["metric"] == "up"].copy()
    up["day"] = up["hour"].dt.floor("D")
    dev_day = up.groupby(["device", "day"])["mean"].mean()
    fleet_day = dev_day.groupby(level="day").agg(["mean", "count"])
    dev_all = up.groupby("device")["mean"].mean().sort_values()
    avail = {
        "fleet_daily": {d.strftime("%Y-%m-%d"): [float(r["mean"]), int(r["count"])] for d, r in fleet_day.iterrows()},
        "per_device": dev_all.round(4).to_dict(),
        "devices_per_day_by_metric": {
            m: {d.strftime("%Y-%m-%d"): int(n) for d, n in
                h[h["metric"] == m].assign(day=lambda x: x["hour"].dt.floor("D")).groupby("day")["device"].nunique().items()}
            for m in ["up", "input_voltage_volts", "battery_remaining_capacity", "temperature", "mppt_charge_power"]
        },
    }

    # --- device directory (agency etc. from labels of `up`) ------------------------------------
    meta = {}
    for f in sorted(DATA.glob("metric=up/month=*/part-000.parquet")):
        for s in pc.unique(pq.read_table(f, columns=["labels"]).column("labels")).to_pylist():
            d = json.loads(s)
            if d.get("job") == "edge" and d.get("device_id"):
                meta[d["device_id"].lower()] = {k: d[k] for k in ("agency", "agency_device_id", "stop_code") if k in d}

    out = {
        "generated": pd.Timestamp.utcnow().isoformat(),
        "hourly_rows": int(len(h)), "n_devices": int(h["device"].nunique()),
        "metrics": names, "corr": corr, "pairs": pairs_out,
        "fingerprints": med.round(4).reset_index().to_dict(orient="records"), "clusters": clusters,
        "regime_changes": changes, "trends": trends, "forecasts": forecasts, "availability": avail,
        "devices": meta,
    }
    (OUT / "cross.json").write_text(json.dumps(out, default=float))
    print(f"wrote cross.json in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Descriptive seasonal patterns of the key health metrics, from the hourly feature table.

For each metric (and device):
  1. hour-of-day profile per calendar month (median per UTC hour), fleet and per agency, with the
     daily amplitude (p90 - p10 of the profile) and the peak / trough hour: how the daily cycle's
     size and phase move from March to October;
  2. seasonal and trend strength (Hyndman): F_s = max(0, 1 - Var(R) / Var(S + R)) and
     F_t = max(0, 1 - Var(R) / Var(T + R)) from a classical additive decomposition
     (24 h centred moving-average trend, hour-of-day seasonal over a rolling 29-day window);
  3. calendar-season daily aggregates (daily temperature max, daylight hours, solar energy, ...)
     summarised per month, fleet and per agency;
  4. weekday vs weekend hour-of-day profiles (expected to be the same: the FFT found no weekly cycle).

All hours are UTC. Reads analysis/out/features.parquet (features.py); writes analysis/out/seasonal.json.
Usage: python -I analysis/seasonal.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/out"
FEATURES = OUT / "features.parquet"

KEY = [
    "input_voltage_volts", "input_power_watts", "orin_power_watts", "cpu_temp_celsius",
    "board_temp_celsius", "temperature", "humidity", "lightsensor", "memory_free_gb",
    "storage_free_gb", "battery_remaining_capacity", "mppt_charge_power",
]
AGENCY_METRICS = ["lightsensor", "temperature", "board_temp_celsius", "input_power_watts"]
# Smallest daily amplitude (metric units) for which a weekday/weekend ratio is meaningful.
AMP_FLOOR = {
    "input_voltage_volts": 0.1, "input_power_watts": 0.5, "orin_power_watts": 0.3,
    "cpu_temp_celsius": 1.0, "board_temp_celsius": 1.0, "temperature": 1.0, "humidity": 2.0,
    "lightsensor": 100.0, "memory_free_gb": 0.1, "storage_free_gb": 0.5,
    "battery_remaining_capacity": 2.0, "mppt_charge_power": 5.0,
}
MIN_HOURS_MONTH = 120     # a (device, month) profile needs >= 10 days of data
MIN_PER_HOUR = 5          # ... and >= 5 values behind every hour-of-day median
MIN_HOURS_STRENGTH = 336  # two weeks for a decomposition
MIN_WEEKEND_DAYS = 8
MIN_DEVICES_AGENCY = 5
MIN_DEVICES_CAL = 3
DAYLIGHT_LUX = 50.0

# Daily aggregates for the calendar-season view: name -> (source metric, pandas agg, unit).
DAILY_AGGS = {
    "temperature_max": ("temperature", "max", "°C"),
    "temperature_mean": ("temperature", "mean", "°C"),
    "temperature_min": ("temperature", "min", "°C"),
    "board_temp_max": ("board_temp_celsius", "max", "°C"),
    "cpu_temp_max": ("cpu_temp_celsius", "max", "°C"),
    "humidity_mean": ("humidity", "mean", "% RH"),
    "input_power_mean": ("input_power_watts", "mean", "W"),
    "battery_min": ("battery_remaining_capacity", "min", "%"),
    "uptime": ("up", "mean", "share"),
}


def q(a, p):
    return float(np.nanpercentile(a, p))


def r(x, nd=4):
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else round(x, nd)


def hour_profiles(df: pd.DataFrame, months: list[str]) -> tuple[dict, dict]:
    """Fleet and per-agency hour-of-day profiles per metric and month."""
    g = df.groupby(["device", "month", "hour"], observed=True)
    med = g[KEY].median()
    cnt = g[KEY].count()
    agency_of = df.groupby("device", observed=True)["agency"].first()
    fleet, by_agency = {}, {}
    for m in KEY:
        pm = med[m].unstack("hour")      # (device, month) x 24
        pc = cnt[m].unstack("hour")
        ok = (pc.fillna(0) >= MIN_PER_HOUR).all(axis=1) & (pc.fillna(0).sum(axis=1) >= MIN_HOURS_MONTH)
        pm = pm[ok]
        fleet[m], by_agency[m] = {}, {}
        for mo in months:
            if mo not in pm.index.get_level_values("month"):
                continue
            block = pm.xs(mo, level="month")
            if block.empty:
                continue
            prof = block.median(axis=0).to_numpy()
            amp_dev = np.percentile(block.to_numpy(), 90, axis=1) - np.percentile(block.to_numpy(), 10, axis=1)
            fleet[m][mo] = {
                "n_devices": int(len(block)),
                "median": [r(v) for v in prof],
                "p25": [r(v) for v in block.quantile(0.25, axis=0)],
                "p75": [r(v) for v in block.quantile(0.75, axis=0)],
                "amplitude": r(q(prof, 90) - q(prof, 10)),
                "peak_hour": int(np.nanargmax(prof)), "trough_hour": int(np.nanargmin(prof)),
                "amp_dev_median": r(np.median(amp_dev)),
            }
            if m in AGENCY_METRICS:
                ag = block.index.get_level_values("device").map(agency_of)
                for a in pd.unique(ag):
                    sub = block[ag == a]
                    if len(sub) >= MIN_DEVICES_AGENCY:
                        by_agency[m].setdefault(str(a), {})[mo] = {
                            "n_devices": int(len(sub)), "median": [r(v) for v in sub.median(axis=0)],
                            "peak_hour": int(np.nanargmax(sub.median(axis=0).to_numpy())),
                        }
    return fleet, by_agency


def strength(df: pd.DataFrame) -> dict:
    """Seasonal / trend strength per device and metric from a classical additive decomposition."""
    per = {m: [] for m in KEY}
    for dev, g in df.groupby("device", observed=True):
        g = g.sort_values("time")
        hours = g["hour"].to_numpy()
        for m in KEY:
            y = g[m]
            if y.notna().sum() < MIN_HOURS_STRENGTH:
                continue
            T = y.rolling(24, center=True, min_periods=18).mean()
            d = y - T
            S = d.groupby(hours).transform(lambda s: s.rolling(29, center=True, min_periods=10).mean())
            R = d - S
            ok = R.notna() & d.notna() & T.notna()
            if ok.sum() < MIN_HOURS_STRENGTH:
                continue
            var_d, var_r = float(d[ok].var()), float(R[ok].var())
            var_ts = float((y - S)[ok].var())
            if var_d < 1e-12:
                per[m].append([str(dev), None, None, int(ok.sum())])
                continue
            fs = max(0.0, 1 - var_r / var_d)
            ft = max(0.0, 1 - var_r / var_ts) if var_ts > 1e-12 else None
            per[m].append([str(dev), r(fs), r(ft), int(ok.sum())])
    out = {}
    for m, rows in per.items():
        fs = [x[1] for x in rows if x[1] is not None]
        ft = [x[2] for x in rows if x[2] is not None]
        out[m] = {"n": len(rows), "fs_median": r(np.median(fs)) if fs else None,
                  "fs_p25": r(np.percentile(fs, 25)) if fs else None, "fs_p75": r(np.percentile(fs, 75)) if fs else None,
                  "ft_median": r(np.median(ft)) if ft else None, "per_device": rows}
    return out


def calendar(df: pd.DataFrame, months: list[str]) -> dict:
    """Daily aggregates per device-day (>= 12 h of data), summarised per month, fleet and per agency."""
    day = df.groupby(["device", "day"], observed=True)
    nh = day["has_data"].sum()
    daily = pd.DataFrame(index=nh.index)
    for name, (src, how, _) in DAILY_AGGS.items():
        daily[name] = day[src].agg(how)
    light = day["lightsensor"]
    daily["daylight_hours"] = light.apply(lambda s: float((s > DAYLIGHT_LUX).sum()) * 24 / max(1, s.notna().sum()) if s.notna().sum() >= 12 else np.nan)
    charge = day["mppt_charge_power"]
    daily["solar_wh"] = charge.apply(lambda s: float(s.sum()) * 24 / s.notna().sum() if s.notna().sum() >= 12 else np.nan)
    daily = daily[nh >= 12].reset_index()
    daily["month"] = daily["day"].dt.strftime("%Y-%m")
    daily["agency"] = daily["device"].map(df.groupby("device", observed=True)["agency"].first())
    units = {k: v[2] for k, v in DAILY_AGGS.items()} | {"daylight_hours": "h/day", "solar_wh": "Wh/day"}
    source = {k: v[0] for k, v in DAILY_AGGS.items()} | {"daylight_hours": "lightsensor", "solar_wh": "mppt_charge_power"}
    out = {}
    for name in units:
        fleet, agency = {}, {}
        for mo, g in daily.groupby("month"):
            v = g[name].dropna()
            if len(v) >= MIN_DEVICES_CAL:
                fleet[mo] = [r(v.median()), r(v.quantile(0.25)), r(v.quantile(0.75)), int(g.loc[v.index, "device"].nunique())]
            for a, ga in g.groupby("agency", observed=True):
                va = ga[name].dropna()
                if ga.loc[va.index, "device"].nunique() >= MIN_DEVICES_CAL:
                    agency.setdefault(str(a), {})[mo] = [r(va.median()), r(va.quantile(0.25)), r(va.quantile(0.75)), int(ga.loc[va.index, "device"].nunique())]
        out[name] = {"unit": units[name], "source_metric": source[name], "fleet": fleet, "agency": agency}
    return out


def weekend(df: pd.DataFrame) -> dict:
    """Weekday vs weekend hour-of-day profiles per device and metric."""
    is_we = df["dow"] >= 5
    we_days = df[is_we & (df["has_data"] == 1)].groupby("device", observed=True)["day"].nunique()
    pwd = df[~is_we].groupby(["device", "hour"], observed=True)[KEY].median()
    pwe = df[is_we].groupby(["device", "hour"], observed=True)[KEY].median()
    out = {}
    for m in KEY:
        rows = []
        a = pwd[m].unstack("hour")
        b = pwe[m].unstack("hour").reindex(a.index)
        for dev in a.index:
            if we_days.get(dev, 0) < MIN_WEEKEND_DAYS or a.loc[dev].isna().any() or b.loc[dev].isna().any():
                continue
            x, y = a.loc[dev].to_numpy(), b.loc[dev].to_numpy()
            amp = q(x, 90) - q(x, 10)
            if amp < AMP_FLOOR[m]:
                rows.append([str(dev), None, None])
                continue
            rows.append([str(dev), r(np.median(np.abs(x - y)) / amp), r(np.mean(y - x) / amp)])
        ratios = [x[1] for x in rows if x[1] is not None]
        shifts = [x[2] for x in rows if x[2] is not None]
        out[m] = {"n": len(ratios), "ratio_median": r(np.median(ratios)) if ratios else None,
                  "ratio_p25": r(np.percentile(ratios, 25)) if ratios else None,
                  "ratio_p75": r(np.percentile(ratios, 75)) if ratios else None,
                  "shift_median": r(np.median(shifts)) if shifts else None, "per_device": rows}
    return out


def main() -> None:
    t0 = time.time()
    df = pd.read_parquet(FEATURES, columns=["device", "agency", "time", "has_data", "dow", *KEY, "up"])
    df["hour"] = df["time"].dt.hour.astype("int8")
    df["month"] = df["time"].dt.strftime("%Y-%m")
    df["day"] = df["time"].dt.floor("D")
    months = sorted(df["month"].unique())
    first, last = df["time"].min(), df["time"].max()
    partial = [mo for mo in months if (mo == first.strftime("%Y-%m") and first.day > 1) or (mo == last.strftime("%Y-%m") and last.day < 28)]
    print(f"loaded {len(df):,} rows, {df['device'].nunique()} devices, months {months[0]}..{months[-1]} ({time.time() - t0:.0f}s)", flush=True)

    fleet, by_agency = hour_profiles(df, months)
    print(f"profiles done ({time.time() - t0:.0f}s)", flush=True)
    stren = strength(df)
    print(f"strength done ({time.time() - t0:.0f}s)", flush=True)
    cal = calendar(df, months)
    print(f"calendar done ({time.time() - t0:.0f}s)", flush=True)
    wk = weekend(df)
    print(f"weekend done ({time.time() - t0:.0f}s)", flush=True)

    out = {"generated": pd.Timestamp.now("UTC").isoformat(), "months": months, "partial_months": partial,
           "metrics": KEY, "agency_metrics": AGENCY_METRICS, "daylight_lux": DAYLIGHT_LUX,
           "profiles": fleet, "profiles_agency": by_agency, "strength": stren, "calendar": cal, "weekend": wk}
    (OUT / "seasonal.json").write_text(json.dumps(out, default=float))
    print(f"wrote seasonal.json ({(OUT / 'seasonal.json').stat().st_size / 1e6:.2f} MB) in {time.time() - t0:.0f}s")

    # sanity
    print("\nlight amplitude by month:", {mo: round(v["amplitude"]) for mo, v in fleet["lightsensor"].items()})
    print("light peak hour fleet:", {mo: v["peak_hour"] for mo, v in fleet["lightsensor"].items()})
    print("board temp peak hour EXO:", {mo: v["peak_hour"] for mo, v in by_agency["board_temp_celsius"].get("EXO", {}).items()},
          "| Barcelona:", {mo: v["peak_hour"] for mo, v in by_agency["board_temp_celsius"].get("Barcelona", {}).items()})
    print("temperature daily max by month:", {mo: v[0] for mo, v in cal["temperature_max"]["fleet"].items()})
    print("daylight hours by month:", {mo: v[0] for mo, v in cal["daylight_hours"]["fleet"].items()})
    print("F_s medians:", {m: v["fs_median"] for m, v in stren.items()})
    print("weekend ratio medians:", {m: v["ratio_median"] for m, v in wk.items()})


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Contextual and fleet-relative (propagation-aware) anomaly detection with River.

Stage 1, per device and key metric, in time order over the hourly feature table:
  an online additive Holt-Winters model (period 24 h) forecasts each hour one step ahead; the
  residual r = y - yhat is scaled into z = r / scale, where scale is a robust running estimate
  (80 % quantile of |r| over the last 14 days, / 1.2816 = the Gaussian-consistent sigma) with a
  per-metric floor. |z| >= Z_SPIKE flags a spike; |z| >= Z_SUSTAINED for >= SUSTAINED_HOURS
  consecutive hours flags a sustained deviation. Holt-Winters is coasted through short gaps
  (fed its own forecast so the seasonal index stays aligned with hour of day) and re-initialised
  after long ones. River's PredictiveAnomalyDetection is scored alongside for comparison.
Stage 2, cross-sectional per hour: each device's z is compared with its peers (same agency,
  else same k-means cluster, else the fleet). A flagged hour is SHARED when enough peers flag the
  same metric at the same hour (a propagating event), otherwise DEVICE-specific. Devices whose
  peer-relative deviation is large while their own is not ("non-followers") are counted too.

Reads analysis/out/features.parquet and analysis/out/cross.json; writes analysis/out/anomalies.parquet
(every scored hour) and analysis/out/anomaly_summary.json (what the report shows).
Usage: python -I analysis/anomaly.py [--selftest]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from river import anomaly, stats, time_series

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/out"
FEATURES = OUT / "features.parquet"
CROSS = OUT / "cross.json"
PROFILE_DIR = OUT / "profile"

KEY = [
    "input_voltage_volts", "input_power_watts", "orin_power_watts", "cpu_temp_celsius",
    "board_temp_celsius", "temperature", "humidity", "lightsensor", "memory_free_gb",
    "storage_free_gb", "battery_remaining_capacity", "mppt_charge_power",
]
# Smallest residual scale (metric units): keeps z finite on flat series and ignores sub-resolution
# noise. Set near each sensor's resolution, not above its natural hourly error: a 0.5 GB disk floor
# would hide the 1-2 GB fleet-wide clean-ups that ADWIN sees in the daily means.
FLOOR = {
    "input_voltage_volts": 0.1, "input_power_watts": 0.25, "orin_power_watts": 0.2,
    "cpu_temp_celsius": 1.0, "board_temp_celsius": 1.0, "temperature": 1.0, "humidity": 2.0,
    "lightsensor": 100.0, "memory_free_gb": 0.05, "storage_free_gb": 0.1,
    "battery_remaining_capacity": 2.0, "mppt_charge_power": 5.0,
}
# Metrics that are ~0 for half the day: their flags cannot be uniform over hour of day by construction.
DAYTIME_ONLY = {"lightsensor", "mppt_charge_power"}

# alpha 0.2 (the forecast section uses 0.3): a slower level keeps a sustained offset visible for
# several hours instead of absorbing it within two. beta=0 crashes River's HoltWinters; keep a tiny trend.
ALPHA, BETA, GAMMA, PERIOD = 0.2, 0.01, 0.3, 24
# Residual scale per hour of day: the model's error is far larger at sunrise/sunset than at night,
# so one scale for all hours would flag the transition hours only. 28 values per slot = 4 weeks.
SCALE_Q, SCALE_WINDOW_D, SCALE_MIN_D, Q_TO_SIGMA = 0.8, 28, 5, 1.2816
WARMUP_FIRST_H, WARMUP_RESET_H, SCALE_START_H = 168, 72, 48
COAST_MAX_H = 48
MIN_HOURS_DEVICE = 192
Z_SPIKE, Z_SUSTAINED, SUSTAINED_HOURS = 4.0, 3.0, 3
MIN_PEERS, SHARED_MIN, SHARED_SHARE = 5, 3, 0.2
Z_PEER_FLAG, Z_NONFOLLOW = 3.0, 4.0


def new_hw() -> time_series.HoltWinters:
    return time_series.HoltWinters(alpha=ALPHA, beta=BETA, gamma=GAMMA, seasonality=PERIOD)


def score_device(y: np.ndarray, hours: np.ndarray, floor: float) -> dict[str, np.ndarray]:
    """Run the online detector over one device's hourly series (NaN = missing hour); hours = UTC hour of day."""
    n = len(y)
    yhat = np.full(n, np.nan, np.float32)
    resid = np.full(n, np.nan, np.float32)
    scale = np.full(n, np.nan, np.float32)
    pad_s = np.full(n, np.nan, np.float32)
    level = np.full(n, np.nan, np.float32)
    season = np.full(n, np.nan, np.float32)
    scored = np.zeros(n, bool)
    coasted = np.zeros(n, bool)

    hw = pad = None
    learned = 0          # samples the current model has learned (real + coasted)
    gap = 0
    warm_target = WARMUP_FIRST_H
    started = False
    scale_q = [stats.RollingQuantile(q=SCALE_Q, window_size=SCALE_WINDOW_D) for _ in range(PERIOD)]
    slot_n = np.zeros(PERIOD, int)

    for i in range(n):
        v = y[i]
        h = int(hours[i])
        if np.isnan(v):
            gap += 1
            if hw is not None and learned >= PERIOD and gap <= COAST_MAX_H:
                f = hw.forecast(1)[0]
                hw.learn_one(f)
                pad.learn_one(None, f)
                learned += 1
                yhat[i], level[i], season[i] = f, hw.level[-1], hw.season[-1]
                coasted[i] = True
            elif gap > COAST_MAX_H:
                hw = pad = None
            continue
        if hw is None:
            hw, pad = new_hw(), anomaly.PredictiveAnomalyDetection(new_hw(), horizon=1, n_std=3.0)
            learned = 0
            warm_target = WARMUP_RESET_H if started else WARMUP_FIRST_H
            started = True
        gap = 0
        if learned >= PERIOD:
            f = hw.forecast(1)[0]
            if not np.isfinite(f):
                # River initialises seasonal indices as y / level: a device whose first day is all
                # zeros (solar power at night) gets a zero level and NaN forecasts. Start over.
                hw = pad = None
                continue
            rr = v - f
            sc = max(scale_q[h].get() / Q_TO_SIGMA, floor) if slot_n[h] >= SCALE_MIN_D else np.nan
            yhat[i], resid[i], scale[i] = f, rr, sc
            pad_s[i] = pad.score_one(None, v)
            if learned >= warm_target and not np.isnan(sc):
                scored[i] = True
            if learned >= SCALE_START_H and np.isfinite(rr):
                scale_q[h].update(abs(rr))
                slot_n[h] += 1
        hw.learn_one(v)
        pad.learn_one(None, v)
        learned += 1
        if hw.level:  # components exist only once the first full season has been seen
            if not (np.isfinite(hw.level[-1]) and np.isfinite(hw.season[-1])):
                hw = pad = None
                continue
            level[i], season[i] = hw.level[-1], hw.season[-1]

    z = np.where(scored, resid / scale, np.nan).astype(np.float32)
    flag = np.zeros(n, np.int8)
    big = scored & (np.abs(z) >= Z_SUSTAINED) & (np.abs(resid) >= floor)
    run = 0
    for i in range(n + 1):
        if i < n and big[i]:
            run += 1
        else:
            if run >= SUSTAINED_HOURS:
                flag[i - run:i] = 2
            run = 0
    spike = scored & (np.abs(z) >= Z_SPIKE) & (np.abs(resid) >= floor) & (flag == 0)
    flag[spike] = 1
    sign = np.where(flag > 0, np.sign(resid), 0).astype(np.int8)
    return {"yhat": yhat, "resid": resid, "scale": scale, "z": z, "pad": pad_s, "level": level,
            "season": season, "scored": scored, "coasted": coasted, "flag": flag, "sign": sign}


def per_metric(metric: str) -> pd.DataFrame:
    df = pd.read_parquet(FEATURES, columns=["device", "agency", "time", metric])
    df = df.sort_values(["device", "time"])
    parts = []
    for dev, g in df.groupby("device", observed=True, sort=False):
        y = g[metric].to_numpy(dtype=float)
        if np.isfinite(y).sum() < MIN_HOURS_DEVICE:
            continue
        res = score_device(y, g["time"].dt.hour.to_numpy(), FLOOR[metric])
        keep = ~np.isnan(res["yhat"])
        if not keep.any():
            continue
        part = pd.DataFrame({"device": dev, "agency": g["agency"].iloc[0], "time": g["time"].to_numpy()[keep],
                             "y": y[keep].astype(np.float32)})
        for k, arr in res.items():
            part[k] = arr[keep]
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "metric", metric)
    return out


def peer_layer(df: pd.DataFrame, cluster_of: dict[str, str]) -> pd.DataFrame:
    """Cross-sectional comparison of z with peers at the same hour; labels flagged hours shared / device."""
    df["cluster"] = df["device"].map(cluster_of).fillna("none").astype(str)
    sc = df["scored"].to_numpy()
    s = df[sc]
    n_ag = s.groupby(["metric", "time", "agency"], observed=True)["z"].transform("size")
    n_cl = s.groupby(["metric", "time", "cluster"], observed=True)["z"].transform("size")
    n_fl = s.groupby(["metric", "time"], observed=True)["z"].transform("size")
    group = np.select([n_ag >= MIN_PEERS, (n_cl >= MIN_PEERS) & (s["cluster"] != "none"), n_fl >= MIN_PEERS],
                      ["agency:" + s["agency"].astype(str), "cluster:" + s["cluster"], "fleet"], "none")
    s = s.assign(peer_group=group)
    gb = s.groupby(["metric", "time", "peer_group"], observed=True)["z"]
    med = gb.transform("median")
    mad = (s["z"] - med).abs().groupby([s["metric"], s["time"], s["peer_group"]], observed=True).transform("median")
    zp = (s["z"] - med) / np.maximum(1.4826 * mad, 0.5)
    size = gb.transform("size")
    big = (s["z"].abs() >= Z_PEER_FLAG).astype(int)
    nflag = big.groupby([s["metric"], s["time"], s["peer_group"]], observed=True).transform("sum") - big
    none = s["peer_group"] == "none"
    n_peers = (size - 1).where(~none, 0)
    n_peers_flagged = nflag.where(~none, 0)
    shared = n_peers_flagged >= np.maximum(SHARED_MIN, SHARED_SHARE * n_peers)
    flagged = s["flag"] > 0
    scope = np.select([~flagged, none, shared], ["none", "unknown", "shared"], "device")
    df["peer_group"] = "none"
    df["n_peers"] = 0
    df["n_peers_flagged"] = 0
    df["zp"] = np.float32(np.nan)
    df["scope"] = "none"
    df.loc[s.index, "peer_group"] = s["peer_group"].to_numpy()
    df.loc[s.index, "n_peers"] = n_peers.to_numpy().astype(np.int16)
    df.loc[s.index, "n_peers_flagged"] = n_peers_flagged.to_numpy().astype(np.int16)
    df.loc[s.index, "zp"] = zp.to_numpy().astype(np.float32)
    df.loc[s.index, "scope"] = scope
    for c in ("peer_group", "scope", "metric", "device", "agency"):
        df[c] = df[c].astype("category")
    return df


def runs(flagged: np.ndarray, times: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of consecutive flagged hours."""
    out, start = [], None
    for i in range(len(flagged)):
        contiguous = i > 0 and (times[i] - times[i - 1]) == np.timedelta64(1, "h")
        if flagged[i] and (start is None or not contiguous):
            if start is not None:
                out.append((start, i))
            start = i
        elif not flagged[i] and start is not None:
            out.append((start, i))
            start = None
        elif flagged[i] and start is not None and not contiguous:
            out.append((start, i))
            start = i
    if start is not None:
        out.append((start, len(flagged)))
    return out


def iso(t) -> str:
    return pd.Timestamp(t).strftime("%Y-%m-%d %H:%M")


def device_events(df: pd.DataFrame, top: int = 20) -> list[dict]:
    ev = []
    f = df[df["flag"] > 0]
    for (dev, m), g in f.groupby(["device", "metric"], observed=True):
        g = g.sort_values("time")
        t = g["time"].to_numpy()
        for a, b in runs(np.ones(len(g), bool), t):
            seg = g.iloc[a:b]
            k = int(np.argmax(np.abs(seg["z"].to_numpy())))
            peak = seg.iloc[k]
            ev.append({"device": str(dev), "agency": str(peak["agency"]), "metric": str(m), "start": iso(t[a]),
                       "end": iso(t[b - 1]), "hours": int(b - a), "kind": "sustained" if (seg["flag"] == 2).any() else "spike",
                       "max_z": round(float(peak["z"]), 2), "sign": int(np.sign(peak["z"])), "y": round(float(peak["y"]), 3),
                       "yhat": round(float(peak["yhat"]), 3), "scale": round(float(peak["scale"]), 4),
                       "scope": str(seg["scope"].mode().iloc[0]), "peak_time": iso(peak["time"])})
    ev.sort(key=lambda e: -abs(e["max_z"]))
    # one event per (device, metric) among the top, with +-24 h context for the chart
    seen, out = set(), []
    for e in ev:
        if (e["device"], e["metric"]) in seen:
            continue
        seen.add((e["device"], e["metric"]))
        pt = pd.Timestamp(e["peak_time"], tz="UTC")
        ctx = df[(df["device"] == e["device"]) & (df["metric"] == e["metric"]) &
                 (df["time"] >= pt - pd.Timedelta("36h")) & (df["time"] <= pt + pd.Timedelta("36h"))].sort_values("time")
        e["context"] = [[iso(r.time), None if np.isnan(r.y) else round(float(r.y), 3), round(float(r.yhat), 3),
                         None if np.isnan(r.scale) else round(float(r.scale), 4), int(r.flag)] for r in ctx.itertuples()]
        out.append(e)
        if len(out) >= top:
            break
    return out, len(ev)


def shared_events(df: pd.DataFrame, regimes: list[dict], top: int = 20) -> list[dict]:
    s = df[df["scope"] == "shared"]
    ev = []
    for (m, pg), g in s.groupby(["metric", "peer_group"], observed=True):
        hours = g.groupby("time", observed=True).agg(n=("device", "nunique"), n_peers=("n_peers", "max"),
                                                    z=("z", "median")).sort_index()
        t = hours.index.to_numpy()
        for a, b in runs(np.ones(len(hours), bool), t):
            seg = hours.iloc[a:b]
            devs = sorted(set(g[(g["time"] >= t[a]) & (g["time"] <= t[b - 1])]["device"].astype(str)))
            d0, d1 = pd.Timestamp(t[a]).floor("D"), pd.Timestamp(t[b - 1]).floor("D")
            adwin = sorted({c["device"] for c in regimes if c["metric"] == m and
                            d0 - pd.Timedelta("1D") <= pd.Timestamp(c["day"], tz="UTC") <= d1 + pd.Timedelta("1D")} & set(devs))
            ev.append({"metric": str(m), "peer_group": str(pg), "start": iso(t[a]), "end": iso(t[b - 1]), "hours": int(b - a),
                       "n_devices": int(seg["n"].max()), "n_peers": int(seg["n_peers"].max() + 1),
                       "median_z": round(float(seg["z"].median()), 2), "sign": int(np.sign(seg["z"].median())),
                       "devices": devs[:12], "n_devices_total": len(devs), "adwin_devices": adwin})
    ev.sort(key=lambda e: (-e["n_devices"], -e["hours"]))
    return ev[:top], len(ev)


def hst_baseline() -> dict:
    out = {}
    for m in KEY:
        f = PROFILE_DIR / f"device_{m}.json"
        if not f.exists():
            continue
        p = json.loads(f.read_text())
        rates = [s["n_anom"] / s["n"] for s in p["series"].values() if s.get("n_anom") is not None and s["n"] > 5000]
        if rates:
            out[m] = round(float(np.median(rates)), 5)
    return out


def summarize(df: pd.DataFrame, regimes: list[dict]) -> dict:
    per, hour_hist = {}, {}
    sc = df[df["scored"]]
    flagged = sc[sc["flag"] > 0]
    for m in KEY:
        s = sc[sc["metric"] == m]
        if s.empty:
            continue
        f = s[s["flag"] > 0]
        hours = f["time"].dt.hour
        dev = s.groupby("device", observed=True).agg(n=("z", "size"), nf=("flag", lambda x: int((x > 0).sum())))
        dev["ns"] = f[f["scope"] == "shared"].groupby("device", observed=True).size().reindex(dev.index).fillna(0).astype(int)
        dev["nd"] = f[f["scope"] == "device"].groupby("device", observed=True).size().reindex(dev.index).fillna(0).astype(int)
        az = s["z"].abs()
        per[m] = {
            "n_scored": int(len(s)), "n_flag": int(len(f)), "rate": round(len(f) / len(s), 5),
            "spike": int((f["flag"] == 1).sum()), "sustained": int((f["flag"] == 2).sum()),
            "shared": int((f["scope"] == "shared").sum()), "device": int((f["scope"] == "device").sum()),
            "unknown": int((f["scope"] == "unknown").sum()),
            "non_follower": int(((s["zp"].abs() >= Z_NONFOLLOW) & (az < Z_SUSTAINED)).sum()),
            "pad_rate": round(float((s["pad"] >= 1.0).mean()), 5), "rate_ge3": round(float((az >= 3).mean()), 5),
            "hour_hist": [int((hours == h).sum()) for h in range(24)],
            "hour_hist_shared": [int(((hours == h) & (f["scope"] == "shared")).sum()) for h in range(24)],
            "z_abs_quantiles": {k: round(float(az.quantile(v)), 3) for k, v in (("p50", .5), ("p90", .9), ("p99", .99), ("p999", .999))},
            "daytime_only": m in DAYTIME_ONLY,
            "per_device": [[str(d), int(r_.n), int(r_.nf), int(r_.ns), int(r_.nd)] for d, r_ in dev.iterrows()],
        }
    days = sorted(sc["time"].dt.strftime("%Y-%m-%d").unique())
    fd = flagged.assign(day=flagged["time"].dt.strftime("%Y-%m-%d"))
    def counts(g):
        c = g.groupby(["day", "scope"], observed=True).size().unstack(fill_value=0)
        return {k: [int(c[k].get(d, 0)) if k in c else 0 for d in days] for k in ("shared", "device")}
    daily = {"days": days, "total": counts(fd), "by_metric": {m: counts(fd[fd["metric"] == m]) for m in KEY if (fd["metric"] == m).any()}}
    devices = sorted(sc["device"].astype(str).unique())
    mat, mat_s = [], []
    pk = sc.groupby(["device", "metric"], observed=True).agg(n=("z", "size"), nf=("flag", lambda x: int((x > 0).sum())),
                                                               ns=("scope", lambda x: int((x == "shared").sum())))
    for d in devices:
        row, rows = [], []
        for m in KEY:
            if (d, m) in pk.index:
                n, nf, ns = pk.loc[(d, m)]
                row.append(round(1000 * nf / n, 2)); rows.append(round(1000 * ns / n, 2))
            else:
                row.append(None); rows.append(None)
        mat.append(row); mat_s.append(rows)
    ev_dev, n_dev_ev = device_events(df)
    ev_sh, n_sh_ev = shared_events(df, regimes)
    nf = sc[(sc["zp"].abs() >= Z_NONFOLLOW) & (sc["z"].abs() < Z_SUSTAINED) & (sc["flag"] == 0)].copy()
    nf["azp"] = nf["zp"].abs()
    nf_top = [{"device": str(r_.device), "agency": str(r_.agency), "metric": str(r_.metric), "time": iso(r_.time), "z": round(float(r_.z), 2),
               "zp": round(float(r_.zp), 2), "peer_group": str(r_.peer_group), "n_peers_flagged": int(r_.n_peers_flagged)}
              for r_ in nf.sort_values("azp", ascending=False).head(10).itertuples()]
    return {
        "generated": pd.Timestamp.now("UTC").isoformat(),
        "params": {"alpha": ALPHA, "beta": BETA, "gamma": GAMMA, "period_h": PERIOD, "scale_q": SCALE_Q,
                   "scale_window_days": SCALE_WINDOW_D, "scale_per_hour_of_day": True,
                   "warmup_first_h": WARMUP_FIRST_H, "warmup_reset_h": WARMUP_RESET_H, "coast_max_h": COAST_MAX_H,
                   "min_hours_device": MIN_HOURS_DEVICE, "z_spike": Z_SPIKE, "z_sustained": Z_SUSTAINED, "sustained_hours": SUSTAINED_HOURS,
                   "min_peers": MIN_PEERS, "shared_rule": f"peers flagged >= max({SHARED_MIN}, {SHARED_SHARE} x peers)", "z_peer_flag": Z_PEER_FLAG,
                   "z_nonfollow": Z_NONFOLLOW, "floors": FLOOR},
        "totals": {"n_scored": int(len(sc)), "n_flag": int(len(flagged)), "rate": round(len(flagged) / len(sc), 5),
                   "shared": int((flagged["scope"] == "shared").sum()), "device": int((flagged["scope"] == "device").sum()),
                   "unknown": int((flagged["scope"] == "unknown").sum()), "spike": int((flagged["flag"] == 1).sum()),
                   "sustained": int((flagged["flag"] == 2).sum()), "n_device_events": n_dev_ev, "n_shared_events": n_sh_ev,
                   "n_devices": len(devices), "non_follower": int(len(nf))},
        "per_metric": per, "daily": daily,
        "device_metric": {"devices": devices, "metrics": KEY, "per_1k": mat, "shared_per_1k": mat_s},
        "events_device": ev_dev, "events_shared": ev_sh, "non_follower": {"n": int(len(nf)), "top": nf_top},
        "hst_baseline": hst_baseline(),
    }


def selftest() -> None:
    rng = np.random.default_rng(0)
    n = 24 * 40
    t = np.arange(n)
    y = 20 + 6 * np.sin((t - 6) / 24 * 2 * np.pi) + rng.normal(0, 0.3, n)
    y[24 * 20:24 * 20 + 30] = np.nan          # 30 h gap: coasted
    y[24 * 30:24 * 30 + 60] = np.nan          # 60 h gap: reset
    y[24 * 15 + 14] += 8                      # spike, day 15 14:00
    y[24 * 37 + 3: 24 * 37 + 8] -= 8          # sustained, day 37
    res = score_device(y, t % 24, 1.0)
    z, flag, coasted, scored = res["z"], res["flag"], res["coasted"], res["scored"]
    print("scored hours:", scored.sum(), "coasted:", coasted.sum(), "first scored idx:", np.argmax(scored))
    print("spike z:", round(float(z[24 * 15 + 14]), 2), "flag:", flag[24 * 15 + 14])
    print("sustained flags:", flag[24 * 37 + 3: 24 * 37 + 8].tolist())
    print("after 30 h gap: first 6 z:", np.round(z[24 * 20 + 30: 24 * 20 + 36], 2).tolist(), "(should be small: phase kept)")
    i = 24 * 30 + 60
    print("after 60 h gap: scored resumes at +", int(np.argmax(scored[i:])), "h; z then:", np.round(z[i + np.argmax(scored[i:]): i + np.argmax(scored[i:]) + 4], 2).tolist())
    clean = scored.copy(); clean[24 * 15 + 10: 24 * 15 + 20] = False; clean[24 * 37: 24 * 38] = False
    print("false-flag rate on clean hours:", round(float((flag[clean] > 0).mean()), 4), "| |z| p99 clean:", round(float(np.nanpercentile(np.abs(z[clean]), 99)), 2))
    zeros = np.zeros(24 * 24); zeros[24 * 4:] = np.abs(np.sin(np.arange(24 * 20) / 24 * 2 * np.pi)) * 50  # all-zero first days, as solar power at night
    rz = score_device(zeros, np.arange(24 * 24) % 24, 5.0)
    print("zero-start series: scored", int(rz["scored"].sum()), "finite yhat", int(np.isfinite(rz["yhat"]).sum()))
    # the level absorbs a sustained offset within a few hours, so >= 3 of the 5 hours must be flagged, not all
    assert flag[24 * 15 + 14] == 1 and (flag[24 * 37 + 3: 24 * 37 + 8] == 2).sum() >= 3 and (flag[clean] > 0).mean() < 0.01
    assert np.isfinite(rz["yhat"][np.isfinite(rz["yhat"])]).all() and rz["scored"].sum() > 0
    print("selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=len(KEY)) as ex:
        parts = list(ex.map(per_metric, KEY))
    df = pd.concat(parts, ignore_index=True)
    print(f"stage 1: {len(df):,} forecast hours, {int(df['scored'].sum()):,} scored, {int((df['flag'] > 0).sum()):,} flagged ({time.time() - t0:.0f}s)", flush=True)
    cross = json.loads(CROSS.read_text())
    cluster_of = {d: c for c, info in cross["clusters"].items() for d in info["devices"]}
    df = peer_layer(df, cluster_of)
    print(f"stage 2: peer groups {df.loc[df['scored'], 'peer_group'].value_counts().to_dict()} ({time.time() - t0:.0f}s)", flush=True)
    df.to_parquet(OUT / "anomalies.parquet", index=False)
    summary = summarize(df, cross["regime_changes"])
    (OUT / "anomaly_summary.json").write_text(json.dumps(summary, default=float))
    print(f"wrote anomalies.parquet and anomaly_summary.json in {time.time() - t0:.0f}s\n")

    # sanity
    for m, p in summary["per_metric"].items():
        hh = p["hour_hist"]; ratio = (max(hh) / max(1, min(hh))) if not p["daytime_only"] else None
        print(f"  {m:28s} rate={100 * p['rate']:.2f}%  spike={p['spike']:6d} sustained={p['sustained']:6d}  shared={p['shared']:6d} device={p['device']:6d} unknown={p['unknown']:5d} "
              f"pad={100 * p['pad_rate']:.2f}%  |z|p99={p['z_abs_quantiles']['p99']}  hour max/min={ratio if ratio is None else round(ratio, 1)}")
    print("totals:", summary["totals"])
    print("shared events (top 8):")
    for e in summary["events_shared"][:8]:
        print(f"  {e['start']}..{e['end']} {e['metric']} {e['peer_group']} n={e['n_devices']}/{e['n_peers']} z={e['median_z']} adwin={len(e['adwin_devices'])}")


if __name__ == "__main__":
    main()

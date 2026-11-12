#!/usr/bin/env python
"""Streaming anomaly detection on the smart-sign telemetry, with River.

Everything is scored online, in time order, on the hourly feature table: each hour is scored before the
models learn from it, so no score uses later data.

Layer 1, per device and metric (River `time_series.HoltWinters`, period 24 h):
  one-step-ahead forecast, residual r = y - yhat, z = r / scale. The scale is a running robust estimate
  per hour of day (River `stats.RollingQuantile`: 80 % quantile of |r| over the last 28 days, / 1.2816)
  with a floor at the sensor's resolution. |z| >= 4 flags a spike; |z| >= 3 for >= 3 consecutive hours
  flags a sustained deviation. The model coasts through gaps up to 48 h and restarts after longer ones.
  Robust learning: an extreme hour is learned clipped at 6 scales, so one burst does not bend the
  forecast; after 24 extreme hours on the same side the change is learned in full (a new level).
  The model starts from the first day (fenced, additive seasonal offsets, no trend). For counter
  rates the first hour after a gap holds the increase over the whole gap and is not scored.
Layer 2, per device (River `anomaly.HalfSpaceTrees`): the vector of that hour's z values over the
  device's metrics. It flags hours whose combination of deviations is unusual, including hours where
  no single metric crosses its own threshold. One fleet threshold, set on hours before 2026-08-01.
Layer 3, per hour across devices: a flagged hour is SHARED when enough peers (same agency, else the
  fleet) flag the same metric in the same hour, otherwise DEVICE-specific.
Incidents: a device's flagged hours, over all metrics, merged when less than 3 h apart. The order in
  which metrics first flag inside an incident is counted per metric pair (onset order).

Reads analysis/out/features.parquet. Writes analysis/out/anomalies.parquet (every scored hour),
joint.parquet (layer 2), incidents.parquet and anomaly_summary.json (what the report shows).
Usage: python -I analysis/anomaly.py [--selftest]
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from river import anomaly, stats, time_series

# River's start-up divides by the first-day mean (0 for solar power at night); its result is replaced below
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/out"
FEATURES = OUT / "features.parquet"

# metric: (label, group, floor). The floor is the smallest residual scale, in metric units: it keeps z
# finite on flat series and sits near the sensor's resolution so that small fleet-wide steps still show.
METRICS = {
    "input_voltage_volts": ("Input voltage", "power", 0.1),
    "input_current_amperes": ("Input current", "power", 0.02),
    "input_power_watts": ("Input power", "power", 0.25),
    "orin_power_watts": ("Orin module power", "power", 0.2),
    "cpu_gpu_power_watts": ("CPU+GPU power", "power", 0.1),
    "cpu_temp_celsius": ("CPU temperature", "thermal", 1.0),
    "gpu_temp_celsius": ("GPU temperature", "thermal", 1.0),
    "board_temp_celsius": ("Board temperature", "thermal", 1.0),
    "temperature": ("Enclosure temperature", "environment", 1.0),
    "humidity": ("Humidity", "environment", 2.0),
    "lightsensor": ("Light", "environment", 100.0),
    "memory_free_gb": ("Free RAM", "compute", 0.05),
    "storage_free_gb": ("Free disk", "compute", 0.1),
    "deepstream_app_running": ("DeepStream running", "compute", 0.05),
    "wwan_tx_gb_rate": ("LTE upload", "network", 0.0005),
    "battery_remaining_capacity": ("Battery charge", "battery", 2.0),
    "mppt_charge_power": ("Solar charge power", "battery", 5.0),
    "mppt_battery_voltage": ("Battery voltage", "battery", 0.1),
}
KEY = list(METRICS)
LABEL = {m: v[0] for m, v in METRICS.items()}
FLOOR = {m: v[2] for m, v in METRICS.items()}
COUNTER_RATES = {"wwan_tx_gb_rate"}   # hourly increase of a running total
DAYTIME_ONLY = {"lightsensor", "mppt_charge_power"}  # ~0 at night, so their flags cannot be spread over the day

# Layer 1. beta = 0 crashes River's HoltWinters, so the trend term is kept tiny.
ALPHA, BETA, GAMMA, PERIOD = 0.2, 0.01, 0.3, 24
SCALE_Q, SCALE_WINDOW_D, SCALE_MIN_D, Q_TO_SIGMA = 0.8, 28, 5, 1.2816
WARMUP_FIRST_H, WARMUP_RESET_H, SCALE_START_H = 168, 72, 48
COAST_MAX_H = 48
MIN_HOURS_DEVICE = 192
Z_SPIKE, Z_SUSTAINED, SUSTAINED_HOURS = 4.0, 3.0, 3
LEARN_CLIP, TYP_SCALE_Q = 6.0, 0.75   # robust update: clip at 6 scales; typical scale = 75 % quantile over hours of day
# Layer 2
HST_TREES, HST_HEIGHT, HST_WINDOW, HST_SEED = 25, 8, 336, 42
HST_Z_CLIP, HST_MIN_COVER, HST_MIN_PRESENT = 6.0, 0.5, 0.7
HST_CAL_END, HST_Q = "2026-08-01", 0.995
# Layer 3
MIN_PEERS, SHARED_MIN, SHARED_SHARE, Z_PEER_FLAG = 5, 3, 0.2, 3.0
# Incidents
INCIDENT_GAP_H = 2


def new_hw() -> time_series.HoltWinters:
    return time_series.HoltWinters(alpha=ALPHA, beta=BETA, gamma=GAMMA, seasonality=PERIOD)


# ----------------------------------------------------------------------------- layer 1
def score_device(y: np.ndarray, hours: np.ndarray, floor: float) -> dict[str, np.ndarray]:
    """Online detector over one device's hourly series (NaN = missing hour); hours = UTC hour of day."""
    n = len(y)
    yhat = np.full(n, np.nan, np.float32)
    resid = np.full(n, np.nan, np.float32)
    scale = np.full(n, np.nan, np.float32)
    scored = np.zeros(n, bool)

    hw = None
    learned = 0          # samples the current model has learned (real + coasted)
    gap = 0
    warm_target = WARMUP_FIRST_H
    started = False
    scale_q = [stats.RollingQuantile(q=SCALE_Q, window_size=SCALE_WINDOW_D) for _ in range(PERIOD)]
    slot_n = np.zeros(PERIOD, int)
    ext_run, ext_sign = 0, 0            # consecutive extreme hours on the same side
    slot_scale = np.full(PERIOD, np.nan)   # latest scale per hour of day
    typ_scale = np.nan

    for i in range(n):
        v = y[i]
        h = int(hours[i])
        if np.isnan(v):
            gap += 1
            if hw is not None and learned >= PERIOD and gap <= COAST_MAX_H:
                # feed the model its own forecast so the seasonal slot stays aligned with hour of day
                f = hw.forecast(1)[0]
                hw.learn_one(f)
                learned += 1
                yhat[i] = f
            elif gap > COAST_MAX_H:
                hw = None
            continue
        if hw is None:
            hw = new_hw()
            learned = 0
            ext_run, ext_sign = 0, 0
            warm_target = WARMUP_RESET_H if started else WARMUP_FIRST_H
            started = True
        gap = 0
        if learned >= PERIOD:
            f = hw.forecast(1)[0]
            if not np.isfinite(f):
                # safety net: never score or learn from a non-finite forecast. Start over.
                hw = None
                continue
            rr = v - f
            sc = max(scale_q[h].get() / Q_TO_SIGMA, floor) if slot_n[h] >= SCALE_MIN_D else np.nan
            yhat[i], resid[i], scale[i] = f, rr, sc
            if learned >= warm_target and not np.isnan(sc):
                scored[i] = True
            if learned >= SCALE_START_H and np.isfinite(rr):
                scale_q[h].update(abs(rr))
                slot_n[h] += 1
        if learned < PERIOD:
            hw.learn_one(v)
            learned += 1
            if learned == PERIOD and hw._initialized:
                # River starts the seasonal terms as ratios y / level even in the additive model, and
                # the trend as the mean first-day difference. For small-valued series the ratios sit far
                # above the data (NaN for an all-zero first day), and a burst in the first day (a
                # backlog upload after an outage) stays in the seasonal term for weeks. Start instead
                # from additive offsets of the first day, fenced at 3 interquartile ranges, and no trend.
                fv = np.asarray(hw._first_values, float)
                q25, q75 = np.quantile(fv, [0.25, 0.75])
                fv = np.clip(fv, q25 - 3 * (q75 - q25), q75 + 3 * (q75 - q25))
                lvl = float(fv.mean())
                hw.level.clear(); hw.level.append(lvl)
                hw.trend.clear(); hw.trend.append(0.0)
                hw.season.clear(); hw.season.extend((fv - lvl).tolist())
            continue
        # Robust update (Huber-style cleaning, as in robust Holt-Winters). The seasonal term learns the
        # hour clipped to forecast +/- LEARN_CLIP scales of its own hour of day; the level, shared by
        # all hours, is clipped with the smaller of that scale and the typical scale over the day, so
        # a burst in a noisy hour cannot push the forecast of the quiet hours. After a full day of
        # extreme hours on the same side the gap is a level shift and the true values are learned.
        y_level = y_season = v
        if np.isfinite(sc):
            slot_scale[h] = sc
            if h == 0 or not np.isfinite(typ_scale):
                typ_scale = float(np.nanquantile(slot_scale, TYP_SCALE_Q))
            lim_s = LEARN_CLIP * sc
            lim_l = LEARN_CLIP * min(sc, typ_scale)
            if abs(rr) > lim_s:
                sign = 1 if rr > 0 else -1
                ext_run = ext_run + 1 if sign == ext_sign else 1
                ext_sign = sign
            else:
                ext_run, ext_sign = 0, 0
            if ext_run <= PERIOD:
                y_season = f + min(max(rr, -lim_s), lim_s)
                y_level = f + min(max(rr, -lim_l), lim_l)
        hw.level.update(y_level, hw.trend, hw.season)
        hw.trend.update(y_level, hw.level)
        hw.season.update(y_season, hw.level, hw.trend)
        learned += 1
        if hw.level and not (np.isfinite(hw.level[-1]) and np.isfinite(hw.season[-1])):
            hw = None

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
    return {"yhat": yhat, "resid": resid, "scale": scale, "z": z, "scored": scored, "flag": flag}


def per_metric(metric: str) -> pd.DataFrame:
    df = pd.read_parquet(FEATURES, columns=["device", "agency", "time", metric])
    df = df.sort_values(["device", "time"])
    parts = []
    for dev, g in df.groupby("device", observed=True, sort=False):
        y = np.array(g[metric], dtype=float)
        if metric in COUNTER_RATES:
            # the first hour after a gap holds the counter's increase over the whole gap: not an hourly rate
            miss = ~np.isfinite(y)
            y[np.r_[False, miss[:-1]] & ~miss] = np.nan
        if np.isfinite(y).sum() < MIN_HOURS_DEVICE:
            continue
        res = score_device(y, g["time"].dt.hour.to_numpy(), FLOOR[metric])
        keep = ~np.isnan(res["yhat"])
        if not keep.any():
            continue
        part = pd.DataFrame({"device": str(dev), "agency": str(g["agency"].iloc[0]), "time": g["time"].to_numpy()[keep],
                             "y": y[keep].astype(np.float32)})
        for k, arr in res.items():
            part[k] = arr[keep]
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "metric", metric)
    return out


# ----------------------------------------------------------------------------- layer 2
def joint_device(args) -> pd.DataFrame | None:
    """Half-space trees over one device's vector of z values, hour by hour."""
    dev, times, Z, names = args
    n = len(times)
    present = np.isfinite(Z)
    use = present.mean(axis=0) >= HST_MIN_COVER
    if use.sum() < 3:
        return None
    Z, present, names = Z[:, use], present[:, use], [m for m, u in zip(names, use) if u]
    ok = present.mean(axis=1) >= HST_MIN_PRESENT
    X = np.clip(np.where(present, Z, 0.0), -HST_Z_CLIP, HST_Z_CLIP) / (2 * HST_Z_CLIP) + 0.5
    hst = anomaly.HalfSpaceTrees(n_trees=HST_TREES, height=HST_HEIGHT, window_size=HST_WINDOW, seed=HST_SEED)
    score = np.full(n, np.nan, np.float32)
    seen = 0
    for i in range(n):
        if not ok[i]:
            continue
        x = dict(zip(names, X[i].tolist()))
        if seen >= HST_WINDOW:      # River returns 0 until the first reference window is complete
            score[i] = hst.score_one(x)
        hst.learn_one(x)
        seen += 1
    keep = np.isfinite(score)
    if not keep.any():
        return None
    absz = np.abs(np.where(present, Z, 0.0))
    return pd.DataFrame({"device": dev, "time": times[keep], "score": score[keep], "n_metrics": int(use.sum()),
                         "n_z2": (absz >= 2).sum(axis=1)[keep].astype(np.int16), "max_abs_z": absz.max(axis=1)[keep].astype(np.float32)})


def joint_layer(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    s = df[df["scored"]]
    wide = s.pivot_table(index=["device", "time"], columns="metric", values="z", observed=True)
    jobs = []
    for dev, g in wide.groupby(level="device", observed=True):
        g = g.droplevel("device").sort_index()
        jobs.append((str(dev), g.index.to_numpy(), g.to_numpy(dtype=float), list(g.columns)))
    with ProcessPoolExecutor(max_workers=min(90, len(jobs))) as ex:
        parts = [p for p in ex.map(joint_device, jobs) if p is not None]
    j = pd.concat(parts, ignore_index=True)
    cal = j.loc[j["time"] < pd.Timestamp(HST_CAL_END, tz="UTC"), "score"]
    thr = float(cal.quantile(HST_Q))
    j["flag"] = j["score"] >= thr
    return j, thr


# ----------------------------------------------------------------------------- layer 3
def peer_layer(df: pd.DataFrame) -> pd.DataFrame:
    """Compare z with peers in the same hour; label each flagged hour shared / device / unknown."""
    s = df[df["scored"]]
    n_ag = s.groupby(["metric", "time", "agency"], observed=True)["z"].transform("size")
    n_fl = s.groupby(["metric", "time"], observed=True)["z"].transform("size")
    group = np.select([n_ag >= MIN_PEERS, n_fl >= MIN_PEERS], ["agency:" + s["agency"].astype(str), "fleet"], "none")
    s = s.assign(peer_group=group)
    keys = [s["metric"], s["time"], s["peer_group"]]
    size = s.groupby(keys, observed=True)["z"].transform("size")
    big = (s["z"].abs() >= Z_PEER_FLAG).astype(int)
    nflag = big.groupby(keys, observed=True).transform("sum") - big
    none = s["peer_group"] == "none"
    n_peers = (size - 1).where(~none, 0)
    n_peers_flagged = nflag.where(~none, 0)
    shared = n_peers_flagged >= np.maximum(SHARED_MIN, SHARED_SHARE * n_peers)
    flagged = s["flag"] > 0
    scope = np.select([~flagged, none, shared], ["none", "unknown", "shared"], "device")
    df["peer_group"], df["n_peers"], df["n_peers_flagged"], df["scope"] = "none", 0, 0, "none"
    df.loc[s.index, "peer_group"] = s["peer_group"].to_numpy()
    df.loc[s.index, "n_peers"] = n_peers.to_numpy().astype(np.int16)
    df.loc[s.index, "n_peers_flagged"] = n_peers_flagged.to_numpy().astype(np.int16)
    df.loc[s.index, "scope"] = scope
    for c in ("peer_group", "scope", "metric", "device", "agency"):
        df[c] = df[c].astype("category")
    return df


# ----------------------------------------------------------------------------- incidents and events
def iso(t) -> str:
    return pd.Timestamp(t).strftime("%Y-%m-%d %H:%M")


def build_incidents(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Merge a device's flagged hours (all metrics) into incidents; count onset order per metric pair."""
    f = df[df["flag"] > 0].copy()
    f["h"] = ((f["time"] - pd.Timestamp("2026-01-01", tz="UTC")) // pd.Timedelta("1h")).astype(int)
    f = f.sort_values(["device", "h"])
    new = (f["device"] != f["device"].shift()) | (f["h"] - f["h"].shift() > INCIDENT_GAP_H + 1)
    f["incident"] = new.cumsum()
    rows, order = [], {"device": Counter(), "shared": Counter()}
    for inc, g in f.groupby("incident"):
        first = g.groupby("metric", observed=True)["h"].min().sort_values()
        k = int(np.argmax(np.abs(g["z"].to_numpy())))
        peak = g.iloc[k]
        scope = "shared" if (g["scope"] == "shared").mean() >= 0.5 else "device"
        rows.append({"incident": int(inc), "device": str(peak["device"]), "agency": str(peak["agency"]),
                     "start": g["time"].min(), "end": g["time"].max(), "hours": int(g["h"].max() - g["h"].min() + 1),
                     "flag_hours": int(len(g)), "n_metrics": int(len(first)), "metrics": ",".join(first.index.astype(str)),
                     "first_metrics": ",".join(first.index[first == first.iloc[0]].astype(str)),
                     "peak_metric": str(peak["metric"]), "peak_z": float(peak["z"]), "peak_time": peak["time"], "scope": scope})
        ms = list(first.index.astype(str))
        for i, a in enumerate(ms):
            for b in ms[i + 1:]:
                # first is sorted by onset, so a starts before b or in the same hour
                order[scope][(a, b, "before" if first[a] < first[b] else "same")] += 1
    return pd.DataFrame(rows), order


def device_events(df: pd.DataFrame, top: int = 12) -> list[dict]:
    """The largest deviation per (device, metric), with +-36 h of context for a chart."""
    f = df[df["flag"] > 0]
    idx = f["z"].abs().groupby([f["device"], f["metric"]], observed=True).idxmax()
    best = f.loc[idx].assign(az=lambda d: d["z"].abs()).sort_values("az", ascending=False)
    out, per_metric_n = [], Counter()
    for r in best.itertuples():
        if per_metric_n[r.metric] >= 2:      # at most two examples per metric, so the list is varied
            continue
        per_metric_n[r.metric] += 1
        ctx = df[(df["device"] == r.device) & (df["metric"] == r.metric) &
                 (df["time"] >= r.time - pd.Timedelta("36h")) & (df["time"] <= r.time + pd.Timedelta("36h"))].sort_values("time")
        out.append({"device": str(r.device), "agency": str(r.agency), "metric": str(r.metric), "time": iso(r.time),
                    "z": round(float(r.z), 1), "y": round(float(r.y), 3), "yhat": round(float(r.yhat), 3),
                    "kind": "sustained" if r.flag == 2 else "spike", "scope": str(r.scope),
                    "context": [[iso(c.time), None if np.isnan(c.y) else round(float(c.y), 3), round(float(c.yhat), 3), int(c.flag)]
                                for c in ctx.itertuples()]})
        if len(out) >= top:
            break
    return out


def shared_events(df: pd.DataFrame, top: int = 20) -> tuple[list[dict], int]:
    s = df[df["scope"] == "shared"]
    ev = []
    for (m, pg), g in s.groupby(["metric", "peer_group"], observed=True):
        hours = g.groupby("time", observed=True).agg(n=("device", "nunique"), n_peers=("n_peers", "max"), z=("z", "median")).sort_index()
        t = hours.index.to_numpy()
        brk = np.flatnonzero(np.diff(t) > np.timedelta64(1, "h")) + 1
        for a, b in zip(np.r_[0, brk], np.r_[brk, len(t)]):
            seg = hours.iloc[a:b]
            devs = g[(g["time"] >= t[a]) & (g["time"] <= t[b - 1])]["device"].astype(str).nunique()
            ev.append({"metric": str(m), "peer_group": str(pg), "start": iso(t[a]), "end": iso(t[b - 1]), "hours": int(b - a),
                       "n_devices": int(seg["n"].max()), "n_peers": int(seg["n_peers"].max() + 1), "n_devices_total": int(devs),
                       "median_z": round(float(seg["z"].median()), 1)})
    ev.sort(key=lambda e: (-e["n_devices"], -e["hours"]))
    return ev[:top], len(ev)


def summarize(df: pd.DataFrame, joint: pd.DataFrame, thr: float, inc: pd.DataFrame, order: dict) -> dict:
    sc = df[df["scored"]]
    flagged = sc[sc["flag"] > 0]
    per = {}
    for m in KEY:
        s = sc[sc["metric"] == m]
        if s.empty:
            continue
        f = s[s["flag"] > 0]
        hours = f["time"].dt.hour
        per[m] = {
            "label": LABEL[m], "group": METRICS[m][1], "floor": FLOOR[m], "n_devices": int(s["device"].nunique()),
            "n_scored": int(len(s)), "n_flag": int(len(f)), "rate": round(len(f) / len(s), 5),
            "spike": int((f["flag"] == 1).sum()), "sustained": int((f["flag"] == 2).sum()),
            "shared": int((f["scope"] == "shared").sum()), "device": int((f["scope"] == "device").sum()),
            "unknown": int((f["scope"] == "unknown").sum()), "up": int((f["z"] > 0).sum()), "down": int((f["z"] < 0).sum()),
            "hour_hist": [int((hours == h).sum()) for h in range(24)], "daytime_only": m in DAYTIME_ONLY,
            "z_p99": round(float(s["z"].abs().quantile(0.99)), 2),
        }
    # devices affected per day
    days = pd.date_range(sc["time"].min().floor("D"), sc["time"].max().floor("D"), freq="D")
    dkey = lambda t: t.dt.floor("D")
    n_rep = sc.groupby(dkey(sc["time"]))["device"].nunique().reindex(days, fill_value=0)
    def per_day(g):
        return g.groupby(dkey(g["time"]))["device"].nunique().reindex(days, fill_value=0)
    d_sh, d_dev = per_day(flagged[flagged["scope"] == "shared"]), per_day(flagged[flagged["scope"] == "device"])
    daily = {"days": [d.strftime("%Y-%m-%d") for d in days], "reporting": n_rep.tolist(), "shared": d_sh.tolist(), "device": d_dev.tolist()}
    share = (d_sh / n_rep.clip(lower=1)).where(n_rep >= 20, 0)
    top_days = []
    for d in share.sort_values(ascending=False).head(12).index:
        fd = flagged[(flagged["scope"] == "shared") & (dkey(flagged["time"]) == d)]
        mc = fd.groupby("metric", observed=True)["device"].nunique().sort_values(ascending=False)
        top_days.append({"day": d.strftime("%Y-%m-%d"), "devices": int(d_sh[d]), "reporting": int(n_rep[d]), "share": round(float(share[d]), 3),
                         "metrics": [[str(k), int(v)] for k, v in mc.head(4).items()]})
    # device x metric matrix (flagged hours per 1000 scored), device-specific flags only
    devices = sorted(sc["device"].astype(str).unique())
    pk = sc.groupby(["device", "metric"], observed=True).agg(n=("z", "size"))
    pk["nd"] = flagged[flagged["scope"] != "shared"].groupby(["device", "metric"], observed=True).size().reindex(pk.index).fillna(0)
    mat = [[round(1000 * pk.loc[(d, m), "nd"] / pk.loc[(d, m), "n"], 1) if (d, m) in pk.index else None for m in KEY] for d in devices]
    # per device table
    ag = sc.groupby("device", observed=True)["agency"].first().astype(str)
    jd = joint.groupby("device").agg(jn=("score", "size"), jf=("flag", "sum"))
    inc_d = inc.groupby("device").agg(n_inc=("incident", "size"), n_inc_dev=("scope", lambda s: int((s == "device").sum())),
                                      multi=("n_metrics", lambda s: int((s >= 3).sum())))
    per_device = []
    for d in devices:
        s = sc[sc["device"] == d]
        f = s[s["flag"] > 0]
        topm = f[f["scope"] != "shared"]["metric"].astype(str).value_counts()
        per_device.append({"device": d, "agency": ag[d], "hours": int(s["time"].nunique()), "n_metrics": int(s["metric"].nunique()),
                           "flag_hours": int(len(f)), "device_hours": int((f["scope"] != "shared").sum()), "shared_hours": int((f["scope"] == "shared").sum()),
                           "incidents": int(inc_d["n_inc"].get(d, 0)), "incidents_device": int(inc_d["n_inc_dev"].get(d, 0)),
                           "incidents_multi": int(inc_d["multi"].get(d, 0)), "top_metric": topm.index[0] if len(topm) else None,
                           "joint_rate": round(float(jd["jf"].get(d, 0) / jd["jn"].get(d, 1)), 4) if d in jd.index else None})
    # incidents
    top_inc = inc.assign(az=inc["peak_z"].abs()).sort_values(["n_metrics", "az"], ascending=False)
    def inc_row(r):
        return {"device": r.device, "agency": r.agency, "start": iso(r.start), "end": iso(r.end), "hours": int(r.hours), "n_metrics": int(r.n_metrics),
                "metrics": r.metrics.split(","), "first": r.first_metrics.split(","), "peak_metric": r.peak_metric, "peak_z": round(float(r.peak_z), 1), "scope": r.scope}
    incidents = {
        "n": int(len(inc)), "n_device": int((inc["scope"] == "device").sum()), "n_shared": int((inc["scope"] == "shared").sum()),
        "by_n_metrics": {sc_: [int(((inc["scope"] == sc_) & (inc["n_metrics"].clip(upper=6) == k)).sum()) for k in range(1, 7)] for sc_ in ("device", "shared")},
        "by_hours": {sc_: [int(((inc["scope"] == sc_) & inc["hours"].between(a, b)).sum()) for a, b in ((1, 1), (2, 3), (4, 6), (7, 12), (13, 24), (25, 10**6))] for sc_ in ("device", "shared")},
        "first_metric": {sc_: Counter(m for s in inc.loc[(inc["scope"] == sc_) & (inc["n_metrics"] >= 2), "first_metrics"] for m in s.split(",")).most_common() for sc_ in ("device", "shared")},
        "top_device": [inc_row(r) for r in top_inc[top_inc["scope"] == "device"].head(25).itertuples()],
        "top_shared": [inc_row(r) for r in top_inc[top_inc["scope"] == "shared"].head(10).itertuples()],
    }
    onset = {sc_: [[a, b, k, n] for (a, b, k), n in c.items()] for sc_, c in order.items()}
    # joint layer
    j = joint.merge(flagged.groupby(["device", "time"], observed=True).size().rename("n_uni").reset_index().astype({"device": str}),
                    on=["device", "time"], how="left")
    j["n_uni"] = j["n_uni"].fillna(0)
    jf = j[j["flag"]]
    edges = np.linspace(0, 1, 41)
    joint_s = {
        "threshold": round(thr, 4), "n_scored": int(len(j)), "n_flag": int(len(jf)), "rate": round(len(jf) / len(j), 5),
        "n_devices": int(j["device"].nunique()), "with_layer1": int((jf["n_uni"] > 0).sum()), "joint_only": int((jf["n_uni"] == 0).sum()),
        "layer1_hours": int((j["n_uni"] > 0).sum()), "layer1_caught": int(((j["n_uni"] > 0) & j["flag"]).sum()),
        "joint_only_n_z2_median": float(jf.loc[jf["n_uni"] == 0, "n_z2"].median()) if (jf["n_uni"] == 0).any() else None,
        "joint_only_max_z_median": round(float(jf.loc[jf["n_uni"] == 0, "max_abs_z"].median()), 2) if (jf["n_uni"] == 0).any() else None,
        "rate_after_cal": round(float(j.loc[j["time"] >= pd.Timestamp(HST_CAL_END, tz="UTC"), "flag"].mean()), 5),
        "hist": {"edges": edges.round(3).tolist(), "all": np.histogram(j["score"], edges)[0].tolist(),
                 "layer1": np.histogram(j.loc[j["n_uni"] > 0, "score"], edges)[0].tolist()},
        "score_by_n_uni": [[int(k), round(float(v), 3), int(n)] for k, (v, n) in
                           j.assign(k=j["n_uni"].clip(upper=5)).groupby("k")["score"].agg(["median", "size"]).iterrows()],
    }
    ev_sh, n_sh = shared_events(df)
    return {
        "generated": pd.Timestamp.now("UTC").isoformat(),
        "period": [iso(sc["time"].min()), iso(sc["time"].max())],
        "params": {"alpha": ALPHA, "beta": BETA, "gamma": GAMMA, "period_h": PERIOD, "scale_q": SCALE_Q, "scale_window_days": SCALE_WINDOW_D,
                   "warmup_first_h": WARMUP_FIRST_H, "warmup_reset_h": WARMUP_RESET_H, "coast_max_h": COAST_MAX_H, "min_hours_device": MIN_HOURS_DEVICE,
                   "z_spike": Z_SPIKE, "learn_clip": LEARN_CLIP, "typ_scale_q": TYP_SCALE_Q, "z_sustained": Z_SUSTAINED, "sustained_hours": SUSTAINED_HOURS,
                   "hst": {"trees": HST_TREES, "height": HST_HEIGHT, "window_h": HST_WINDOW, "z_clip": HST_Z_CLIP, "q": HST_Q, "cal_end": HST_CAL_END},
                   "min_peers": MIN_PEERS, "shared_min": SHARED_MIN, "shared_share": SHARED_SHARE, "z_peer_flag": Z_PEER_FLAG,
                   "incident_gap_h": INCIDENT_GAP_H},
        "metrics": KEY, "labels": LABEL, "groups": {m: METRICS[m][1] for m in KEY},
        "totals": {"n_devices": len(devices), "n_scored": int(len(sc)), "n_flag": int(len(flagged)), "rate": round(len(flagged) / len(sc), 5),
                   "device_hours_scored": int(sc.groupby("device", observed=True)["time"].nunique().sum()),
                   "device_hours_flagged": int(flagged.groupby("device", observed=True)["time"].nunique().sum()),
                   "shared": int((flagged["scope"] == "shared").sum()), "device": int((flagged["scope"] == "device").sum()),
                   "unknown": int((flagged["scope"] == "unknown").sum()), "spike": int((flagged["flag"] == 1).sum()),
                   "sustained": int((flagged["flag"] == 2).sum()), "n_shared_events": n_sh},
        "per_metric": per, "daily": daily, "top_days": top_days,
        "device_metric": {"devices": devices, "per_1k": mat}, "per_device": per_device,
        "incidents": incidents, "onset": onset, "joint": joint_s,
        "events_device": device_events(df), "events_shared": ev_sh,
    }


def selftest() -> None:
    rng = np.random.default_rng(0)
    n = 24 * 40
    t = np.arange(n)
    y = 20 + 6 * np.sin((t - 6) / 24 * 2 * np.pi) + rng.normal(0, 0.3, n)
    y[24 * 20:24 * 20 + 30] = np.nan          # 30 h gap: coasted
    y[24 * 30:24 * 30 + 60] = np.nan          # 60 h gap: restart
    y[24 * 15 + 14] += 8                      # spike, day 15 14:00
    y[24 * 37 + 3: 24 * 37 + 8] -= 8          # sustained, day 37
    res = score_device(y, t % 24, 1.0)
    z, flag, scored = res["z"], res["flag"], res["scored"]
    print("scored hours:", scored.sum(), "| spike z:", round(float(z[24 * 15 + 14]), 2), "flag:", flag[24 * 15 + 14],
          "| sustained flags:", flag[24 * 37 + 3: 24 * 37 + 8].tolist())
    clean = scored.copy(); clean[24 * 15 + 10: 24 * 15 + 20] = False; clean[24 * 37: 24 * 38] = False
    print("false-flag rate on clean hours:", round(float((flag[clean] > 0).mean()), 4))
    zeros = np.zeros(24 * 24); zeros[24 * 4:] = np.abs(np.sin(np.arange(24 * 20) / 24 * 2 * np.pi)) * 50
    rz = score_device(zeros, np.arange(24 * 24) % 24, 5.0)
    assert flag[24 * 15 + 14] == 1 and (flag[24 * 37 + 3: 24 * 37 + 8] == 2).sum() >= 3 and (flag[clean] > 0).mean() < 0.01
    assert rz["scored"].sum() > 0 and np.isfinite(rz["yhat"][np.isfinite(rz["yhat"])]).all()
    # layer 2: a joint shift of 2.5 sigma on 6 metrics, none of which crosses |z| = 3 by much
    Z = rng.normal(0, 1, (1500, 8)); Z[1200:1206, :6] += 2.5
    j = joint_device(("d", np.arange(1500), Z, [f"m{i}" for i in range(8)]))
    s = j.set_index("time")["score"]
    print("HST: median score", round(float(s.median()), 3), "| injected hours", s.loc[1200:1205].round(3).tolist(), "| 99.5 %:", round(float(s.quantile(0.995)), 3))
    assert s.loc[1200:1205].median() > s.quantile(0.99)
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
    print(f"layer 1: {len(df):,} forecast hours, {int(df['scored'].sum()):,} scored, {int((df['flag'] > 0).sum()):,} flagged ({time.time() - t0:.0f}s)", flush=True)
    df = peer_layer(df)
    print(f"layer 3: scopes {df.loc[df['flag'] > 0, 'scope'].value_counts().to_dict()} ({time.time() - t0:.0f}s)", flush=True)
    joint, thr = joint_layer(df)
    print(f"layer 2: {len(joint):,} device-hours scored, threshold {thr:.3f}, {int(joint['flag'].sum()):,} flagged ({time.time() - t0:.0f}s)", flush=True)
    inc, order = build_incidents(df)
    print(f"incidents: {len(inc):,} ({(inc['scope'] == 'device').sum():,} device-specific) ({time.time() - t0:.0f}s)", flush=True)
    df.to_parquet(OUT / "anomalies.parquet", index=False)
    joint.to_parquet(OUT / "joint.parquet", index=False)
    inc.to_parquet(OUT / "incidents.parquet", index=False)
    summary = summarize(df, joint, thr, inc, order)
    (OUT / "anomaly_summary.json").write_text(json.dumps(summary, default=float))
    print(f"wrote anomalies.parquet, joint.parquet, incidents.parquet and anomaly_summary.json in {time.time() - t0:.0f}s\n")
    for m, p in summary["per_metric"].items():
        print(f"  {m:28s} dev={p['n_devices']:3d} rate={100 * p['rate']:.2f}%  spike={p['spike']:6d} sustained={p['sustained']:6d}  "
              f"shared={p['shared']:6d} device={p['device']:6d} unknown={p['unknown']:5d}  |z|p99={p['z_p99']}")
    print("totals:", summary["totals"])
    print("joint:", {k: v for k, v in summary["joint"].items() if k not in ("hist",)})
    print("incidents:", {k: summary["incidents"][k] for k in ("n", "n_device", "n_shared", "by_n_metrics", "by_hours")})
    print("top days:", [(d["day"], d["devices"], d["reporting"]) for d in summary["top_days"][:8]])


if __name__ == "__main__":
    main()

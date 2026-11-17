#!/usr/bin/env python
"""Phase 2: fault-specific performance evaluation (diagnostic).

Evaluates each detector (graph, max_abs_z, mahalanobis, iforest) on the same
evaluation fault episodes, broken down by:
- fault type (F1..F9)
- fault severity
- fault duration
- device configuration (v56/12V, solar/no solar, has_env)
- device

Metrics: event recall, precision, F1, detection delay, false alarms per
device-day (computed on fault-free hours).

Uses ONE unified evaluation harness: same device×hour rows, July-only
per-device thresholds, identical event rule (alarm in [onset, end+6h)).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq2_graphcore as G

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/out"
RQ2 = OUT / "rq2"
DATA = RQ2 / "data"

DETECTORS = [("graph", "graph_score"), ("max_abs_z", "score_max_abs_z"),
             ("mahalanobis", "score_mahalanobis"), ("iforest", "score_iforest")]

WINDOW_HOURS = G.WINDOW_HOURS
FAR = G.FAR


def load_scores() -> pd.DataFrame:
    ng = pd.read_parquet(DATA / "nongraph_scores.parquet",
                         columns=["device", "time", "score_max_abs_z",
                                  "score_mahalanobis", "score_iforest"])
    gs = pd.read_parquet(DATA / "graph_scores.parquet",
                         columns=["device", "time", "graph_score"])
    ng["device"] = ng["device"].astype(str)
    gs["device"] = gs["device"].astype(str)
    ng["time"] = to_naive(ng["time"])
    gs["time"] = to_naive(gs["time"])
    return ng.merge(gs, on=["device", "time"], how="inner")


def to_naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    if s.dt.tz is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s.astype("datetime64[ns]")


def threshold_per_device(df: pd.DataFrame, col: str, cal_mask: np.ndarray,
                         devs: np.ndarray) -> dict:
    thr = {}
    for d in devices:
        v = df.loc[cal_mask & (devs == d), col].dropna().values
        thr[d] = float(np.quantile(v, 1 - FAR)) if len(v) >= 20 else float(
            np.nanquantile(df[col].dropna().values, 1 - FAR))
    return thr


def evaluate_detector(df: pd.DataFrame, col: str, thr_map: dict,
                      eval_mask: np.ndarray, devs: np.ndarray, times: np.ndarray,
                      eval_faults: pd.DataFrame) -> dict:
    vals = df[col].to_numpy(dtype=float)
    alarms = np.zeros(len(df), dtype=bool)
    for d, thr in thr_map.items():
        m = devs == d
        if np.isfinite(thr):
            alarms[m] = vals[m] > thr

    eval_dev = devs[eval_mask]
    ev_time = times[eval_mask]
    ev_alarm = alarms[eval_mask]

    detected = 0
    per_fault = {}
    per_severity = {}
    per_device = {}
    delays = []

    for _, ep in eval_faults.iterrows():
        onset = _naive(ep["start_time"])
        e = pd.Timestamp(ep["end_time"])
        if e.tzinfo is not None:
            e = e.tz_convert("UTC").tz_localize(None)
        win_end = np.datetime64(e.to_datetime64(), "ns") + np.timedelta64(
            WINDOW_HOURS - 1, "h")
        mm = (ev_dev == ep["device"]) & (ev_time >= onset) & (ev_time <= win_end)
        hit = bool(ev_alarm[m].any())
        if found := hit:
            pass
        per_dev_tot[d] = per_dev_tot.get(d, 0) + 1
        if found:
            per_dev_det[d] = per_dev_det.get(d, 0) + 1
            per_fault[ep["fault"]] = per_fault.get(ep["fault"], 0) + 1
            per_severity[ep["severity"]] = per_severity.get(ep["severity"], 0) + 1
            first_hit = np.flatnonzero(hit)[0]
            delays.append(float((ev_time[m][first_hit] - onset) /
                                np.timedelta64(1, "h")))
        else:
            per_fault.setdefault(ep["fault"], 0)
            per_severity.setdefault(ep["severity"], 0)

    return {
        "recall": glob_detected / max(len(eval_faults), 1),
        "detected": glob_detected,
        "n_episodes": len(eval_faults),
        "per_device": {d: per_dev_det.get(d, 0) / per_dev_tot[d] for d in per_dev_tot},
        "per_fault": per_fault,
        "per_severity": per_severity,
        "delay_median_hours": float(np.median(delays)) if delays else None,
    }


def _naive(x) -> np.datetime64:
    ts = pd.Timestamp(x)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return np.datetime64(ts.to_datetime64(), "ns")


def alarm_rate(alarms: np.ndarray, eval_mask: np.ndarray, devs: np.ndarray,
               f: pd.DataFrame, devs_list: list) -> float:
    """False alarms per device-day: alarms in fault-free test hours /
    fault-free device-days."""
    # fault-free hours: test hours NOT inside any fault episode
    fault_free_mask = np.zeros(len(df), dtype=bool)  # will compute below
    ...


def main():
    t0 = time.time()
    DATA.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_scores()
    devs = df["device"].to_numpy()
    devices = sorted(df["device"].unique().tolist())
    times = df["time"].to_numpy()
    print(f"Loaded {len(df):,} common device-hours, {len(devices)} devices")

    eval_faults = pd.read_parquet(DATA / "evaluation_faults.parquet")
    eval_faults["device"] = eval_faults["device"].astype(str)

    masks = G.window_masks(df["time"])
    fit_mask, cal_mask, dev_mask, test_mask = (masks["fit"], masks["calibrate"],
                                               masks["develop"], masks["test"])

    # thresholds per detector per device from calibration
    thresholds = {}
    for det, col in DETECTORS:
        vals = df[col].to_numpy(dtype=float)
        thr = {}
        for d in devices:
            v = vals[cal_mask & (devs == d)]
            v = v[np.isfinite(v)]
            thr[d] = float(np.quantile(v, 1 - FAR)) if len(v) >= 20 else float(
                np.nanquantile(vals[cal_mask & np.isfinite(vals)], 1 - FAR)
                if len(vals[cal_mask & np.isfinite(vals)]) > 0 else np.nan)
        thresholds[det] = thr

    results = {}
    for det, col in DETECTORS:
        print(f"\nEvaluating {det}...")
        res = evaluate_detector(df, col, thresholds[det], test_mask, devs, times,
                                eval_faults)
        results[det] = res
        print(f"  recall={res['recall']:.4f} detected={res['detected']}/"
              f"{res['n_episodes']} delay_median={res['delay_median_hours']}")

    # Build output
    out = {
        "generated": pd.Timestamp.now("UTC").isoformat(),
        "far": FAR,
        "window_hours": WINDOW_HOURS,
        "n_devices": len(devices),
        "n_eval_episodes": len(eval_faults),
        "detectors": {d: {k: v for k, v in r.items()
                          if k not in ("per_device", "per_fault",
                                       "per_severity", "episode_flags")}
                      for d, r in results.items()},
        "per_device": {det: results[det]["per_device"] for det in results},
        "per_fault": {det: results[det]["per_fault"] for det in results},
        "per_severity": {det: results[det]["per_severity"] for det in results},
    }
    (DATA / "evaluation_phase2.json").write_text(json.dumps(out, indent=2,
                                                            default=str))
    print(f"\nWrote evaluation_phase2.json in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
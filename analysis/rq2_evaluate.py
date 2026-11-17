#!/usr/bin/env python
"""Authoritative RQ2 detector evaluation (fair harness).

ONE pipeline for every detector:
- same device x hour rows (graph + non-graph scores merged on common keys)
- July-only, per-device thresholds
- identical event rule: alarm anywhere in [onset, end + WINDOW_HOURS - 1]
- identical missing-score handling

This replaces the earlier harness that calibrated thresholds on July-October
(leaky) and used one global threshold. Writes evaluation_results.json and
bootstrap_ci.json in a single consistent format.

Usage: python -I analysis/rq2_evaluate.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq2_graphcore as G  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/out"
RQ2 = OUT / "rq2"
DATA = RQ2 / "data"

FAR = 0.01
WINDOW_HOURS = G.WINDOW_HOURS
DETECTORS = [("graph", "graph_score"), ("max_abs_z", "score_max_abs_z"),
             ("mahalanobis", "score_mahalanobis"), ("iforest", "score_iforest")]
BOOT = 2000
SEED = 20261103


def to_naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    if s.dt.tz is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s.astype("datetime64[ns]")


def load_common() -> pd.DataFrame:
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


def main():
    t0 = time.time()
    DATA.mkdir(parents=True, exist_ok=True)

    df = load_common()
    df = df.sort_values(["device", "time"]).reset_index(drop=True)
    devices = sorted(df["device"].unique().tolist())
    devs = df["device"].to_numpy()
    times = df["time"].to_numpy()

    t = df["time"]
    cal_mask = ((t >= pd.Timestamp(*G.SPLITS["calibrate"]) if False else
                 (t >= pd.Timestamp(G.SPLITS["calibrate"][0])) &
                 (t < pd.Timestamp(G.SPLITS["calibrate"][1])))).to_numpy()
    test_mask = ((t >= pd.Timestamp(G.SPLITS["test"][0])) &
                 (t < pd.Timestamp(G.SPLITS["test"][1]))).to_numpy()

    eval_faults = pd.read_parquet(DATA / "evaluation_faults.parquet")
    print(f"{len(df):,} common rows, {len(devices)} devices, {len(eval_faults)} eval episodes")

    # Per-device thresholds from July calibration only
    thresholds = {}
    for det, col in DETECTORS:
        vals = df[col].to_numpy(dtype=float)
        thr = {}
        for d in devices:
            v = vals[cal_mask & (devs == d)]
            v = v[np.isfinite(v)]
            if len(v) >= 20:
                thr[d] = float(np.quantile(v, 1 - FAR))
            else:
                allc = vals[cal_mask]
                allc = allc[np.isfinite(allc)]
                thr[d] = float(np.quantile(allc, 1 - FAR)) if len(allc) else np.nan
        thresholds[det] = thr

    def alarms_for(det):
        vals = df[dict(DETECTORS)[det]].to_numpy(dtype=float)
        a = np.zeros(len(df), dtype=bool)
        for d, th in thresholds[det].items():
            m = devs == d
            if np.isfinite(th):
                a[m] = vals[m] > th
        return a

    results = {}
    per_device = {}
    ep_flags = {}
    for det, _ in DETECTORS:
        res = G.evaluate_recall(alarms_for(det), test_mask, eval_faults, devs, times)
        cal_rate = float(np.mean(alarms_for(det)[cal_mask]))
        res["calibrate_alarm_rate"] = cal_rate
        res["threshold_median"] = float(np.nanmedian(
            [v for v in thresholds[det].values() if np.isfinite(v)]))
        results[det] = res
        per_device[det] = res.pop("per_fault")  # keep; rename below
        print(f"  {det:12s}: recall={res['recall']:.4f} detected={res['detected']}/{res['n_episodes']} "
              f"cal_rate={cal_rate:.4f} delay={res['delay_median_hours']}")

    # per-device recall for bootstrap
    def per_device_recall(det):
        a = alarms_for(det)
        ev_dev = devs[test_mask]; ev_time = times[test_mask]; ev_alarm = a[test_mask]
        out = {}
        for d in devices:
            eps = eval_faults[eval_faults["device"] == d]
            if len(eps) == 0:
                continue
            det_n = 0
            for _, ep in eps.iterrows():
                onset = pd.Timestamp(ep["start_time"])
                if onset.tz is not None: onset = onset.tz_convert("UTC").tz_localize(None)
                end = pd.Timestamp(ep["end_time"])
                if end.tz is not None: end = end.tz_convert("UTC").tz_localize(None)
                o64 = np.datetime64(onset.to_datetime64(), "ns")
                e64 = np.datetime64(end.to_datetime64(), "ns") + np.timedelta64(WINDOW_HOURS - 1, "h")
                m = (ev_dev == d) & (ev_time >= o64) & (ev_time <= e64)
                if ev_alarm[m].any():
                    det_n += 1
            out[d] = det_n / len(eps)
        return out

    pd_recall = {det: per_device_recall(det) for det, _ in DETECTORS}

    # bootstrap
    rng = np.random.default_rng(SEED)
    boot = {}
    names = [d for d, _ in DETECTORS]
    for i, da in enumerate(names):
        for db in names[i + 1:]:
            common = [d for d in devices if d in pd_recall[da] and d in pd_recall[db]]
            ra = np.array([pd_recall[da][d] for d in common])
            rb = np.array([pd_recall[db][d] for d in common])
            diffs = []
            for _ in range(BOOT):
                idx = rng.integers(0, len(common), size=len(common))
                diffs.append(ra[idx].mean() - rb[idx].mean())
            diffs = np.array(diffs)
            boot[f"{da}_vs_{db}"] = {
                "mean_diff": float(np.mean(diffs)),
                "ci_low": float(np.quantile(diffs, 0.025)),
                "ci_high": float(np.quantile(diffs, 0.975)),
                "p_le_0": float(np.mean(diffs <= 0)),
                "significant": bool(np.quantile(diffs, 0.025) > 0 or np.quantile(diffs, 0.975) < 0),
            }

    out = {
        "generated": pd.Timestamp.now("UTC").isoformat(),
        "far": FAR, "window_hours": WINDOW_HOURS,
        "harness": "unified: July-only per-device thresholds, common score rows",
        "n_devices": len(devices), "n_eval_episodes": len(eval_faults),
        "detectors": {det: {k: v for k, v in results[det].items()} for det in results},
        "per_device_recall": pd_recall,
        "bootstrap": boot,
    }
    (DATA / "evaluation_results.json").write_text(json.dumps(out, indent=2, default=str))
    (DATA / "bootstrap_ci.json").write_text(json.dumps(
        {"generated": out["generated"], "n_resamples": BOOT,
         "n_devices": len(devices), "comparisons": boot}, indent=2))
    print(f"\nWrote evaluation_results.json + bootstrap_ci.json in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
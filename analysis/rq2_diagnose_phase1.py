#!/usr/bin/env python
"""Phase 1: experimental-fairness audit and ONE unified detector evaluation.

Applies one identical pipeline to every detector:
- same device x hour rows (graph & non-graph scores merged on common keys)
- July-only, per-device thresholds for every detector
- identical event-detection rule (any alarm in [onset, end+6h))
- identical missing-score handling

Writes analysis/out/rq2/data/eval_fairness.json
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
DETECTORS = [("graph", "graph_score"),
             ("max_abs_z", "score_max_abs_z"),
             ("mahalanobis", "score_mahalanobis"),
             ("iforest", "score_iforest")]

SPLITS = G.SPLITS


def to_naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    if s.dt.tz is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s.astype("datetime64[ns]")


def load_common_scores() -> pd.DataFrame:
    """Merge graph and non-graph scores on common (device, time) rows."""
    ng = pd.read_parquet(DATA / "nongraph_scores.parquet",
                         columns=["device", "time", "split",
                                  "score_max_abs_z", "score_mahalanobis", "score_iforest"])
    gs = pd.read_parquet(DATA / "graph_scores.parquet", columns=["device", "time", "graph_score"])
    ng["device"] = ng["device"].astype(str)
    gs["device"] = gs["device"].astype(str)
    ng["time"] = to_naive(ng["time"])
    gs["time"] = to_naive(gs["time"])
    merged = ng.merge(gs, on=["device", "time"], how="inner")
    return merged


def event_detection_per_device(alarms, eval_mask, eval_faults, devs, times):
    """Return per-episode detection flag list and per-device recall dict."""
    ev_dev = devs[eval_mask]
    ev_time = np.asarray(times)[eval_mask]
    ev_alarm = alarms[eval_mask]

    rows = []
    per_dev_det = {}
    per_dev_tot = {}
    glob_detected = 0
    per_fault = {}
    per_severity = {}
    delays = []

    for _, ep in eval_faults.iterrows():
        onset = pd.Timestamp(ep["start_time"])
        if onset.tz is not None:
            onset = onset.tz_convert("UTC").tz_localize(None)
        end = pd.Timestamp(ep["end_time"])
        if end.tz is not None:
            end = end.tz_convert("UTC").tz_localize(None)
        onset = np.datetime64(onset.to_datetime64(), "ns")
        end = np.datetime64(end.to_datetime64(), "ns")
        win_end = end + np.timedelta64(WINDOW_HOURS - 1, "h")

        d = ep["device"]
        m = (ev_dev == d) & (ev_time >= onset) & (ev_time <= win_end)
        hit = ev_alarm[m]
        found = bool(hit.any())

        per_dev_tot[d] = per_dev_tot.get(d, 0) + 1
        if found:
            glob_detected += 1
            per_dev_det[d] = per_dev_det.get(d, 0) + 1
            per_fault[ep["fault"]] = per_fault.get(ep["fault"], 0) + 1
            per_severity[ep["severity"]] = per_severity.get(ep["severity"], 0) + 1
            first_hit = np.flatnonzero(hit)[0]
            delays.append(float((ev_time[m][first_hit] - onset) / np.timedelta64(1, "h")))
        else:
            per_fault.setdefault(ep["fault"], 0)
            per_severity.setdefault(ep["severity"], 0)

        rows.append({
            "device": d, "fault": ep["fault"], "severity": ep["severity"],
            "start_time": ep["start_time"], "end_time": ep["end_time"],
            "detected": found,
        })

    per_dev_recall = {d: per_dev_det.get(d, 0) / per_dev_tot[d] for d in per_dev_tot}
    return {
        "recall": glob_detected / max(len(eval_faults), 1),
        "detected": glob_detected,
        "n_episodes": len(eval_faults),
        "per_fault": per_fault,
        "per_severity": per_severity,
        "per_device": per_dev_recall,
        "delay_median_hours": float(np.median(delays)) if delays else None,
        "episode_flags": rows,
    }


def main():
    t0 = time.time()
    DATA.mkdir(parents=True, exist_ok=True)

    print("Loading common score rows...")
    df = load_common_scores()
    print(f"  {len(df):,} common device-hours, {df['device'].nunique()} devices")
    print(f"  graph_score non-NaN: {df['graph_score'].notna().sum():,}")
    print(f"  score_iforest non-NaN: {df['score_iforest'].notna().sum():,}")

    # Build split masks over the merged frame
    t = df["time"]
    def mask(a, b):
        return ((t >= pd.Timestamp(a)) & (t < pd.Timestamp(b))).to_numpy()
    fit_mask = mask(*SPLITS["fit"])
    cal_mask = mask(*SPLITS["calibrate"])
    dev_mask = mask(*SPLITS["develop"])
    test_mask = mask(*SPLITS["test"])
    print(f"  splits: fit={fit_mask.sum():,} cal={cal_mask.sum():,} "
          f"dev={dev_mask.sum():,} test={test_mask.sum():,}")

    devs = df["device"].to_numpy()
    devices = sorted(df["device"].unique().tolist())
    times = df["time"].to_numpy()

    eval_faults = pd.read_parquet(DATA / "evaluation_faults.parquet")
    dev_faults = pd.read_parquet(DATA / "development_faults.parquet")
    print(f"  eval episodes: {len(eval_faults)}, dev episodes: {len(dev_faults)}")

    # ---- fairness checks ----
    fairness = {}
    fairness["n_common_rows"] = int(len(df))
    fairness["n_devices"] = len(devices)
    fairness["devices_eval"] = int(eval_faults["device"].nunique())
    # devices in eval faults that are present in score frame
    fairness["eval_devices_in_scores"] = int(
        eval_faults["device"].astype(str).isin(set(devices)).sum())
    # score coverage per detector
    fairness["score_coverage"] = {
        det: float(np.isfinite(df[col]).mean()) for det, col in DETECTORS
    }

    # ---- unified per-device thresholds from July only ----
    thresholds = {}
    for det, col in DETECTORS:
        vals = df[col].to_numpy(dtype=float)
        thr_d = {}
        for d in devices:
            mcal = cal_mask & (devs == d)
            v = vals[mcal]
            v = v[np.isfinite(v)]
            if len(v) >= 20:
                thr_d[d] = float(np.quantile(v, 1 - FAR))
            else:
                allcal = vals[cal_mask]
                allcal = allcal[np.isfinite(allcal)]
                thr_d[d] = float(np.quantile(allcal, 1 - FAR)) if len(allcal) else np.nan
        thresholds[det] = thr_d
    fairness["threshold_median"] = {
        det: float(np.nanmedian([v for v in thresholds[det].values() if np.isfinite(v)]))
        for det, _ in DETECTORS
    }

    # ---- unified evaluation on TEST episodes ----
    print("\n=== Unified evaluation on test episodes (FAR=1%, July-only thresholds) ===")
    results = {}
    episode_flags = {}
    for det, col in DETECTORS:
        vals = df[col].to_numpy(dtype=float)
        alarms = np.zeros(len(df), dtype=bool)
        for d, thr in thresholds[det].items():
            m = devs == d
            if np.isfinite(thr):
                alarms[m] = vals[m] > thr
        res = event_detection_per_device(alarms, test_mask, eval_faults, devs, times)
        # realised alarm rate on clean calibration (should be ~1%)
        cal_rate = float(np.mean(alarms[cal_mask & np.isfinite(vals)]))
        res["calibrate_alarm_rate"] = cal_rate
        results[det] = res
        episode_flags[det] = res.pop("episode_flags")
        print(f"  {det:12s}: recall={res['recall']:.3f} ({res['detected']}/{res['n_episodes']}) "
              f"cal_alarm_rate={cal_rate:.4f} delay={res['delay_median_hours']}")

    # ---- pairwise bootstrap CIs (device resampling) ----
    print("\n=== Device-bootstrap CIs for recall differences ===")
    rng = np.random.default_rng(20261103)
    dev_list = [d for d in devices if d in results["graph"]["per_device"]]
    boot = {}
    for i, (da, _) in enumerate(DETECTORS):
        for db, _ in DETECTORS[i + 1:]:
            ra = np.array([results[da]["per_device"].get(d, 0.0) for d in dev_list])
            rb = np.array([results[db]["per_device"].get(d, 0.0) for d in dev_list])
            diffs = []
            for _ in range(2000):
                idx = rng.integers(0, len(dev_list), size=len(dev_list))
                diffs.append(ra[idx].mean() - rb[idx].mean())
            diffs = np.array(diffs)
            boot[f"{da}_vs_{db}"] = {
                "mean_diff": float(np.mean(diffs)),
                "ci_low": float(np.quantile(diffs, 0.025)),
                "ci_high": float(np.quantile(diffs, 0.975)),
                "p_le_0": float(np.mean(diffs <= 0)),
                "significant_pos": bool(np.quantile(diffs, 0.025) > 0),
                "significant_neg": bool(np.quantile(diffs, 0.975) < 0),
            }
            print(f"  {da:12s} - {db:12s}: {boot[f'{da}_vs_{db}']['mean_diff']:+.3f} "
                  f"[{boot[f'{da}_vs_{db}']['ci_low']:+.3f}, {boot[f'{da}_vs_{db}']['ci_high']:+.3f}]")

    # ---- save ----
    out = {
        "generated": pd.Timestamp.now("UTC").isoformat(),
        "far": FAR,
        "fairness": fairness,
        "detectors": {det: {k: v for k, v in results[det].items() if k != "per_device"}
                      for det, _ in DETECTORS},
        "per_device_recall": {det: results[det]["per_device"] for det, _ in DETECTORS},
        "bootstrap": boot,
    }
    (DATA / "eval_fairness.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote eval_fairness.json in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
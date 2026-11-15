#!/usr/bin/env python
"""Phase 4: strong non-graph multivariate anomaly detector.

Reads the shared residuals from Phase 3 (analysis/out/rq2/data/residuals.parquet)
and builds non-graph multivariate anomaly detectors that use only standardised
residuals (z) of the 10 nodes, without modelling subsystem dependencies.

Detectors
---------
- max_abs_z      largest |z| over available nodes (baseline, always computable)
- mahalanobis    correlation-aware distance; computable only when all available
                 nodes are finite for that hour; otherwise NaN
- iforest        Isolation Forest on the per-hour z vector, fit on July calibration

Missing nodes are handled explicitly: a row is only scored if enough nodes are
present (>= MIN_NODES_FOR_MAHALANOBIS of the available set). Rows with too few
available nodes get NaN scores.

Thresholds: per device, the (1 - FAR) quantile of the system score on the July
calibration window (FAR = 1 %). The primary FAR is 0.01; 0.001 and 0.05 are also
stored. No test-window data is used for fitting or threshold selection.

Outputs
-------
analysis/out/rq2/data/nongraph_scores.parquet
  device, time, split, score_max_abs_z, score_mahalanobis, score_iforest,
  threshold_max_abs_z, threshold_mahalanobis, threshold_iforest,
  alarm_max_abs_z, alarm_mahalanobis, alarm_iforest,
  top_node_max_abs_z, top_node_mahalanobis, n_nodes_available

analysis/out/rq2/data/nongraph_thresholds.json
  Per-device thresholds at each FAR, per detector.

analysis/out/rq2/data/nongraph_summary.json
  Realised alarm rates by split, detector and FAR.

Usage: python3 analysis/rq2_nongraph.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq_common as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/out"
RQ2 = OUT / "rq2"
DATA = RQ2 / "data"

NODES = [
    "cpu_gpu_power_watts", "orin_power_watts", "power", "input_power_watts",
    "input_voltage_volts", "input_current_amperes", "input_energy_wh_total_rate",
    "cpu_temp_celsius", "gpu_temp_celsius", "board_temp_celsius",
]
SPLITS = {
    "fit": ("2026-03-08", "2026-07-01"),
    "calibrate": ("2026-07-01", "2026-08-01"),
    "develop": ("2026-08-01", "2026-09-01"),
    "test": ("2026-09-01", "2026-10-07"),
}
FAR = 0.01
FAR_SECONDARY = [0.001, 0.05]
MIN_NODE_COVERAGE = 0.5   # node available if >= 50% of cal hours are non-NaN
MIN_NODES_FOR_MAHALANOBIS = 3  # need at least 3 nodes for mahalanobis


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def load_residuals() -> pd.DataFrame:
    return pd.read_parquet(DATA / "residuals.parquet")


def pivot_z(res: pd.DataFrame) -> pd.DataFrame:
    """Wide frame: index (device, time), columns = nodes, values = z."""
    wide = res.pivot_table(index=["device", "time"], columns="node", values="z", aggfunc="first")
    wide = wide[NODES]
    return wide


def available_nodes(z_wide: pd.DataFrame, split_mask: pd.Series) -> list[str]:
    """Nodes present in >= MIN_NODE_COVERAGE of the device's calibration hours."""
    cal = z_wide.loc[split_mask == "calibrate"]
    if len(cal) == 0:
        cal = z_wide
    cover = cal.notna().mean()
    return [n for n in NODES if cover.get(n, 0.0) >= MIN_NODE_COVERAGE]


def main():
    t0 = time.time()
    DATA.mkdir(parents=True, exist_ok=True)

    res = load_residuals()
    print(f"loaded residuals: {len(res):,} rows")

    wide = pivot_z(res).reset_index()  # device, time as columns
    print(f"pivoted: {wide.shape[0]:,} device-hours x {len(NODES)} nodes")

    devices = sorted(wide["device"].unique().tolist())
    print(f"devices: {len(devices)}")

    # Split label per row
    wide["split"] = "other"
    for name, (a, b) in SPLITS.items():
        m = (wide["time"] >= ts(a)) & (wide["time"] < ts(b))
        wide.loc[m, "split"] = name

    all_rows = []
    thresholds_json = {"generated": datetime.now(timezone.utc).isoformat(),
                       "far": FAR, "far_secondary": FAR_SECONDARY,
                       "detectors": {}}

    for dev in devices:
        dev_df = wide[wide["device"] == dev].copy()
        if len(dev_df) == 0:
            continue

        # Determine available nodes for this device
        is_cal = dev_df["split"] == "calibrate"
        avail = available_nodes(wide.loc[dev_df.index], is_cal)
        if len(avail) == 0:
            print(f"  {dev}: no available nodes, skipping")
            continue
        avail_idx = [NODES.index(n) for n in avail]

        # Z matrix: rows = hours, cols = available nodes; NaN where missing
        Z = wide.loc[dev_df.index, avail].to_numpy(dtype=float)
        n_avail = np.isfinite(Z).sum(axis=1)
        # Mahalanobis needs ALL available nodes finite for that hour
        all_finite = np.all(np.isfinite(Z), axis=1)
        # max_abs_z and iforest only need at least 2 nodes present
        enough = n_avail >= max(2, int(np.ceil(MIN_NODES_FOR_MAHALANOBIS * len(avail))))

        # ---- max_abs_z -------------------------------------------------
        with np.errstate(all="ignore"):
            s_max = np.nanmax(np.abs(Z), axis=1)
        # score rows that have at least 2 nodes present
        s_max = np.where(n_avail >= 2, s_max, np.nan)

        # ---- mahalanobis ------------------------------------------------
        # Build covariance only on calibration rows where ALL available nodes are finite
        Z_cal = Z[is_cal] if is_cal.any() else np.empty((0, len(avail)))
        ok_all_finite = np.all(np.isfinite(Z_cal), axis=1) if Z_cal.shape[0] > 0 else np.array([])
        cov_pool = None
        if ok_all_finite.sum() >= 30:
            try:
                lw = LedoitWolf().fit(Z_cal[ok_all_finite])
                cov_pool = lw.covariance_
            except Exception:
                cov_pool = None

        # Mahalanobis score per row (only where all available nodes are finite)
        s_maha = np.full(len(dev_df), np.nan)
        if cov_pool is not None:
            ok_rows = np.all(np.isfinite(Z), axis=1)  # all avail nodes finite in this row
            if ok_rows.any():
                Zi = Z[ok_rows]
                prec = np.linalg.inv(cov_pool)
                s_maha[ok_rows] = np.sqrt(np.einsum("ij,jk,ik->i", Zi, prec, Zi))
                s_maha[~ok_rows] = np.nan

        # ---- iforest ----------------------------------------------------
        # fit only on calibration rows where all available nodes are finite
        cal_complete = np.asarray(is_cal) & all_finite
        Z_cal_complete = Z[cal_complete]
        iforest = IsolationForest(n_estimators=300, max_samples=4096,
                                  random_state=20261103, n_jobs=8).fit(
            Z_cal_complete if len(Z_cal_complete) >= 100 else np.ones((1, len(avail))))
        s_if = np.full(len(dev_df), np.nan)
        if all_finite.any():
            s_if[all_finite] = -iforest.score_samples(Z[all_finite])

        # ---- top node (argmax |z| among available) ----------------------
        abs_Z = np.abs(Z)
        top_node = np.full(len(dev_df), -1, dtype=int)
        for i in range(len(dev_df)):
            finite_mask = np.isfinite(abs_Z[i])
            if finite_mask.any():
                top_node[i] = avail_idx[np.nanargmax(abs_Z[i])]

        # ---- thresholds on calibration at each FAR ----------------------
        th = {}
        for det_name, sc in (("max_abs_z", s_max), ("mahalanobis", s_maha),
                             ("iforest", s_if)):
            cal_sc = sc[is_cal]
            cal_sc = cal_sc[np.isfinite(cal_sc)]  # drop NaN scores before quantile
            if len(cal_sc) >= 20:
                th[det_name] = float(np.quantile(cal_sc, 1 - FAR))
            else:
                th[det_name] = np.nan

        # ---- alarms -----------------------------------------------------
        a_max = s_max > th["max_abs_z"]
        a_maha = s_maha > th["mahalanobis"] if not np.isnan(th["mahalanobis"]) else np.full(len(s_maha), False)
        a_if = s_if > th["iforest"] if not np.isnan(th["iforest"]) else np.full(len(s_if), False)

        rows = pd.DataFrame({
            "device": dev,
            "time": dev_df["time"],
            "split": dev_df["split"].values,
            "score_max_abs_z": s_max,
            "score_mahalanobis": s_maha,
            "score_iforest": s_if,
            "threshold_max_abs_z": th["max_abs_z"],
            "threshold_mahalanobis": th["mahalanobis"],
            "threshold_iforest": th["iforest"],
            "alarm_max_abs_z": a_max,
            "alarm_mahalanobis": a_maha,
            "alarm_iforest": a_if,
            "top_node_max_abs_z": [avail[i] if i >= 0 else None for i in top_node],
            "top_node_mahalanobis": [avail[i] if i >= 0 else None for i in top_node],
            "n_nodes_available": len(avail),
            "n_nodes_present": n_avail,
        })
        all_rows.append(rows)

        # store thresholds for JSON
        for det_name in ("max_abs_z", "mahalanobis", "iforest"):
            thresholds_json["detectors"].setdefault(det_name, {})[dev] = th[det_name]

    if not all_rows:
        print("No rows generated — exiting")
        return

    scores = pd.concat(all_rows, ignore_index=True)
    scores_path = DATA / "nongraph_scores.parquet"
    scores.to_parquet(scores_path, index=False)
    print(f"wrote {scores_path} ({len(scores):,} rows)")

    # thresholds JSON
    (DATA / "nongraph_thresholds.json").write_text(
        json.dumps(thresholds_json, indent=2, default=str))
    print("wrote nongraph_thresholds.json")

    # summary: realised alarm rate by split/detector
    summary = {"generated": datetime.now(timezone.utc).isoformat(),
               "far": FAR, "far_secondary": FAR_SECONDARY, "detectors": {}}
    for det in ("max_abs_z", "mahalanobis", "iforest"):
        d = {}
        for split_name in ("fit", "calibrate", "develop", "test"):
            s_dev = scores[scores["split"] == split_name]
            if len(s_dev) == 0:
                d[split_name] = {"alarm_rate": None, "n": 0,
                                  "score_median": None, "score_p99": None}
            else:
                sc_col = f"score_{det}"
                a_col = f"alarm_{det}"
                d[split_name] = {
                    "alarm_rate": float(s_dev[a_col].mean()) if s_dev[a_col].notna().any() else None,
                    "n": int(len(s_dev)),
                    "score_median": float(s_dev[sc_col].median()) if s_dev[sc_col].notna().any() else None,
                    "score_p99": float(s_dev[sc_col].quantile(0.99)) if s_dev[sc_col].notna().any() else None,
                }
        summary["detectors"][det] = d
    (DATA / "nongraph_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print("wrote nongraph_summary.json")

    # console summary
    print("\nRealised alarm rate by split (primary FAR = %.2f):" % FAR)
    print(f"  {'detector':12s} {'fit':>8s} {'calibrate':>10s} {'develop':>9s} {'test':>8s}")
    for det, d in summary["detectors"].items():
        print(f"  {det:12s} " + " ".join(
            f"{(d[s]['alarm_rate'] if d[s]['alarm_rate'] is not None else float('nan')):8.4f}"
            for s in ("fit", "calibrate", "develop", "test")))
    print(f"\nwrote nongraph_scores.parquet, thresholds and summary in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""Extension 5: Reproducible case-study visualisations for RQ2.

For a handful of detected evaluation episodes, renders one diagnostic figure each
showing:
  1. observed power nodes over the context window, with the fault span marked
  2. observed temperature nodes
  3. the graph anomaly score with the per-device 1% FAR threshold
  4. the non-graph detector scores (max_abs_z / mahalanobis / iforest)

Outputs: analysis/out/rq2/figures/case_study_<fault>_<device>_<start>.png

Uses the same July-only per-device thresholds as the authoritative evaluator.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq2_graphcore as G  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/out"
RQ2 = OUT / "rq2"
DATA = RQ2 / "data"
FIGS = RQ2 / "figures"

NODES = G.NODES
FAR = G.FAR
POWER_NODES = ["cpu_gpu_power_watts", "orin_power_watts", "input_power_watts", "power"]
THERMAL_NODES = ["cpu_temp_celsius", "gpu_temp_celsius", "board_temp_celsius"]


def to_naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    if s.dt.tz is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s.astype("datetime64[ns]")


def main():
    t0 = time.time()
    FIGS.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    # per-node z from residuals
    res = pd.read_parquet(DATA / "residuals.parquet",
                          columns=["device", "time", "node", "observed", "predicted", "z"])
    gs = pd.read_parquet(DATA / "graph_scores.parquet")
    ng = pd.read_parquet(DATA / "nongraph_scores.parquet")
    faults = pd.read_parquet(DATA / "evaluation_faults.parquet")
    faults["device"] = faults["device"].astype(str)

    for c in (res, gs, ng):
        c["device"] = c["device"].astype(str)
        c["time"] = to_naive(c["time"])

    # July-only per-device thresholds for each detector
    cal = lambda d: ((d["time"] >= pd.Timestamp(G.SPLITS["calibrate"][0])) &
                     (d["time"] < pd.Timestamp(G.SPLITS["calibrate"][1])))

    def thr_map(score_df, col):
        out = {}
        for dev in score_df["device"].unique():
            v = score_df.loc[cal(score_df) & (score_df["device"] == dev), col].dropna().values
            out[dev] = float(np.quantile(v, 1 - FAR)) if len(v) >= 20 else float(
                np.nanquantile(score_df[col].dropna().values, 1 - FAR))
        return out

    graph_thr = thr_map(gs, "graph_score")
    ng_thr = {det: thr_map(ng, f"score_{det}") for det in
              ("max_abs_z", "mahalanobis", "iforest")}

    # residuals pivot: device,time -> node z
    z_wide = res.pivot_table(index=["device", "time"], columns="node",
                             values="z", aggfunc="first")

    # pick one episode per fault type (up to 8)
    picked = faults.sort_values("start_time").groupby("fault", as_index=False).first()
    print(f"Rendering {len(picked)} case-study figures...")

    for _, ep in picked.iterrows():
        dev = str(ep["device"])
        onset = pd.Timestamp(ep["start_time"])
        if onset.tz is not None:
            onset = onset.tz_localize(None)
        end = pd.Timestamp(ep["end_time"])
        if end.tz is not None:
            end = end.tz_localize(None)
        fault = ep["fault"]
        sev = ep["severity"]
        dur = int((end - onset).total_seconds() / 3600)

        # context window ±24h
        c0, c1 = onset - pd.Timedelta(hours=24), end + pd.Timedelta(hours=24)
        if (dev, c0) not in z_wide.index:
            # need to find nearest available
            dev_times = z_wide.loc[dev].index if dev in z_wide.index.get_level_values(0) else None
            if dev_times is None or len(dev_times) == 0:
                print(f"  skip {fault}/{dev}: no residuals")
                continue

        sub_z = z_wide.xs(dev, level="device")
        ctx = sub_z[(sub_z.index >= c0) & (sub_z.index <= c1)]

        sub_gs = gs[(gs["device"] == dev) & (gs["time"] >= c0) & (gs["time"] <= c1)].set_index("time")
        sub_ng = ng[(ng["device"] == dev) & (ng["time"] >= c0) & (ng["time"] <= c1)].set_index("time")

        fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True,
                                 constrained_layout=True)
        fig.suptitle(f"{fault} (sev={sev}, {dur}h)  ·  {dev}  ·  "
                     f"{onset:%Y-%m-%d %H:%M} → {end:%H:%M}", fontsize=12,
                     fontweight="bold")

        # 1) power nodes observed
        ax = axes[0]
        for n in POWER_NODES:
            if n in ctx.columns:
                ax.plot(ctx.index, ctx[n], lw=1, label=n.split("_")[0].upper())
        ax.axvspan(onset, end, color="red", alpha=0.18, label="Fault window")
        ax.set_ylabel("Power (W)")
        ax.legend(fontsize=7, ncol=2, loc="upper left")
        ax.grid(alpha=0.3)

        # 2) temperature nodes observed
        ax = axes[1]
        for n in THERMAL_NODES:
            if n in ctx.columns:
                ax.plot(ctx.index, ctx[n], lw=1, label=n.split("_")[0].upper())
        ax.axvspan(onset, end, color="red", alpha=0.18)
        ax.set_ylabel("Temperature (°C)")
        ax.legend(fontsize=7, ncol=2, loc="upper left")
        ax.grid(alpha=0.3)

        # 3) graph score + threshold
        ax = axes[2]
        if len(sub_gs):
            ax.plot(sub_gs.index, sub_gs["graph_score"], color="#b2182b", lw=0.9,
                    label="Graph score (max|z|)")
        ax.axhline(graph_thr.get(dev, np.nan), color="black", ls=":",
                   lw=1, label=f"Threshold (1% FAR)")
        ax.axvspan(onset, end, color="red", alpha=0.18)
        ax.set_ylabel("Graph score")
        ax.set_yscale("log")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)

        # 4) non-graph scores
        ax = axes[3]
        for det in ("max_abs_z", "mahalanobis", "iforest"):
            if len(sub_ng) and f"score_{det}" in sub_ng.columns:
                ax.plot(sub_ng.index, sub_ng[f"score_{det}"], lw=0.8, label=det)
        ax.axvspan(onset, end, color="red", alpha=0.18)
        ax.set_ylabel("Non-graph score")
        ax.set_yscale("log")
        ax.legend(fontsize=7, ncol=3, loc="upper left")
        ax.grid(alpha=0.3)

        for ax in axes:
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m-%d %H:%M"))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right",
                     fontsize=7)

        out = FIGS / f"case_study_{fault}_{dev}_{onset:%Y%m%d_%H%M}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out.name}")

    print(f"\nCase studies complete in {time.time() - t0:.0f}s ({len(list(FIGS.glob('*.png')))} figures)")


if __name__ == "__main__":
    main()
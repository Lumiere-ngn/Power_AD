#!/usr/bin/env python
"""FFT periodogram of every metric, per device, from the hourly feature table.

For each (metric, device): take the longest run of hourly values (gaps of <= 6 h linearly
filled), require at least 21 days so weekly cycles are resolvable, remove the linear trend,
apply a Hann window and take the real FFT. Power is normalized to fractions of the series'
variance (DC excluded) and summed into 60 log-spaced period bins from 2 h to 45 days, so
devices with different segment lengths are comparable. Per metric we keep the median device
spectrum with its interquartile band, each device's dominant period, and the share of variance
at the 24 h, 12 h, 8 h and 168 h (weekly) periods.

River has no spectral estimator, so this step uses numpy's FFT.

Reads analysis/out/features.parquet (run features.py first); writes analysis/out/spectral.json.
Usage: python -I analysis/spectral.py
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/out"
FEATURES = OUT / "features.parquet"

MIN_HOURS = 21 * 24
MAX_GAP_H = 6
EDGES = np.logspace(np.log10(2.0), np.log10(45 * 24), 61)   # period bin edges, hours
CENTERS = np.sqrt(EDGES[:-1] * EDGES[1:])
TARGETS = {"8h": 8, "12h": 12, "24h": 24, "168h": 168}
TOL = 0.04                                                    # +-4 % around a target period
NOT_METRICS = {"device", "agency", "time", "has_data", "n_samples", "hour_sin", "hour_cos", "dow",
               "is_weekend", "days_since_first"}


def longest_run(x: np.ndarray) -> np.ndarray:
    """Longest stretch without NaN after filling interior gaps of <= MAX_GAP_H hours."""
    s = pd.Series(x).interpolate(limit=MAX_GAP_H, limit_area="inside").to_numpy()
    ok = np.isfinite(s)
    if not ok.any():
        return s[:0]
    # run-length of consecutive True
    edges = np.diff(np.concatenate([[0], ok.astype(int), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    i = int(np.argmax(ends - starts))
    return s[starts[i]:ends[i]]


def spectrum(seg: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    n = len(seg)
    t = np.arange(n)
    seg = seg - np.polyval(np.polyfit(t, seg, 1), t)            # linear detrend
    if not np.any(np.abs(seg) > 1e-9):
        return None                                              # constant series: no spectrum
    p = np.abs(np.fft.rfft(seg * np.hanning(n))) ** 2
    f = np.fft.rfftfreq(n, d=1.0)                                # cycles per hour
    p, f = p[1:], f[1:]                                          # drop DC
    tot = p.sum()
    if tot <= 0:
        return None
    return 1.0 / f, p / tot                                      # period (h), fraction of variance


def per_metric(metric: str) -> tuple[str, dict | None]:
    df = pd.read_parquet(FEATURES, columns=["device", "time", metric])
    binned, dominant, shares = [], [], {k: [] for k in TARGETS}
    for dev, g in df.groupby("device", observed=True):
        seg = longest_run(g.sort_values("time")[metric].to_numpy(dtype=float))
        if len(seg) < MIN_HOURS:
            continue
        sp = spectrum(seg)
        if sp is None:
            continue
        per, frac = sp
        idx = np.digitize(per, EDGES) - 1
        keep = (idx >= 0) & (idx < len(CENTERS))
        binned.append(np.bincount(idx[keep], weights=frac[keep], minlength=len(CENTERS)))
        inrange = per <= 30 * 24                                 # longer periods are trend, not cycle
        j = int(np.argmax(np.where(inrange, frac, -1)))
        dominant.append([str(dev), round(float(per[j]), 2), round(float(frac[j]), 4), len(seg)])
        for k, target in TARGETS.items():
            w = (per >= target * (1 - TOL)) & (per <= target * (1 + TOL))
            shares[k].append([str(dev), round(float(frac[w].sum()), 4)])
    if not binned:
        return metric, None
    B = np.vstack(binned)
    med = np.median(B, axis=0)
    top = np.argsort(med)[::-1][:3]
    return metric, {
        "n_devices": len(binned),
        "median": np.round(med, 5).tolist(),
        "p25": np.round(np.percentile(B, 25, axis=0), 5).tolist(),
        "p75": np.round(np.percentile(B, 75, axis=0), 5).tolist(),
        "top_bins": [[round(float(CENTERS[i]), 2), round(float(med[i]), 4)] for i in top],
        "dominant": dominant,
        "shares": shares,
        "share_median": {k: round(float(np.median([v for _, v in s])), 4) for k, s in shares.items()},
    }


def main() -> None:
    t0 = time.time()
    names = [c for c in pq.read_schema(FEATURES).names if c not in NOT_METRICS]
    with ProcessPoolExecutor(max_workers=len(names)) as ex:
        res = dict(ex.map(per_metric, names))
    out = {"edges_h": np.round(EDGES, 3).tolist(), "centers_h": np.round(CENTERS, 3).tolist(),
           "min_hours": MIN_HOURS, "targets": TARGETS, "metrics": {k: v for k, v in res.items() if v}}
    (OUT / "spectral.json").write_text(json.dumps(out))
    print(f"spectral: {len(out['metrics'])}/{len(names)} metrics with spectra in {time.time() - t0:.0f}s")
    for m, v in sorted(out["metrics"].items(), key=lambda kv: -kv[1]["share_median"]["24h"]):
        doms = [d[1] for d in v["dominant"]]
        print(f"  {m:40s} dev={v['n_devices']:3d} 24h={v['share_median']['24h']:.3f} 12h={v['share_median']['12h']:.3f} "
              f"168h={v['share_median']['168h']:.3f} median dominant={np.median(doms):7.1f} h  top bins={v['top_bins']}")


if __name__ == "__main__":
    main()

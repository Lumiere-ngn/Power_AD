#!/usr/bin/env python
"""Which features influence which: Granger causality and lead-lag correlation on deseasonalized data.

Most metrics are correlated only because they share one driver, the sun. To see influence that is not
"both follow the daily cycle", both analyses run on the deseasonalized signal written by anomaly.py:
z = (y - Holt-Winters one-step forecast) / robust scale, per device, metric and hour.

1. Lead-lag cross-correlation: per device and pair (X, Y), corr(X(t), Y(t + k)) for k = -12..12 h;
   fleet median curve, peak lag (positive = X leads), and the share of devices that agree.
2. Conditional Granger causality, per device: a VAR(p) equation for each target Y on p lags of every
   variable the device has, against the same equation without X's lags (F-test). Conditioning on all
   other variables keeps indirect paths (A -> B -> C) from showing up as a direct A -> C link.
   Benjamini-Hochberg per device; a fleet edge is the share of devices where X -> Y is significant.
3. Robustness: the same Granger run on raw hourly values (shows the links the daily cycle fakes), and a
   null run with X circularly shifted by 3-10 days (the significant share should fall to about q).

River has no VAR / Granger estimator, so the fits are numpy least squares with scipy's F distribution.
Reads analysis/out/anomalies.parquet and analysis/out/features.parquet; writes analysis/out/influence.json.
Usage: python -I analysis/influence.py
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/out"
ANOM = OUT / "anomalies.parquet"
FEATURES = OUT / "features.parquet"

KEY = [
    "input_voltage_volts", "input_power_watts", "orin_power_watts", "cpu_temp_celsius",
    "board_temp_celsius", "temperature", "humidity", "lightsensor", "memory_free_gb",
    "storage_free_gb", "battery_remaining_capacity", "mppt_charge_power",
]
DS = "deepstream_app_running"
VARS = KEY + [DS]

LAG_CHOICES = (1, 2, 3, 6)
CCF_LAGS = 12
MIN_OVERLAP = 336          # hours of overlap for a cross-correlation curve
COVERAGE = 0.7             # a variable enters a device's VAR if present in >= 70 % of its hours
MIN_ROWS, ROWS_PER_REG = 500, 10
MIN_DS_SWITCHES = 5        # DeepStream needs some on/off transitions to be informative
Q = 0.01                   # Benjamini-Hochberg false discovery rate per device
EDGE_SHARE, EDGE_R2 = 0.5, 0.005
NULL_SHIFT_H = (72, 240)
SEED = 42


# ----------------------------------------------------------------------------- data
def load() -> tuple[dict, dict]:
    """Per device: (hours index, deseasonalized matrix) and (raw standardized matrix), columns = VARS."""
    a = pd.read_parquet(ANOM, columns=["device", "metric", "time", "z", "scored"])
    a = a[a["scored"]]
    z = a.set_index(["device", "time", "metric"])["z"].unstack("metric")
    f = pd.read_parquet(FEATURES, columns=["device", "time", *VARS])
    f["device"] = f["device"].astype(str)
    f = f.set_index(["device", "time"])
    deseason, raw = {}, {}
    for dev in sorted(set(z.index.get_level_values("device").astype(str))):
        zd = z.xs(dev, level="device") if dev in z.index.get_level_values("device") else None
        fd = f.xs(dev, level="device")
        grid = pd.date_range(fd.index.min(), fd.index.max(), freq="h")
        zd = zd.reindex(grid) if zd is not None else pd.DataFrame(index=grid)
        ds = fd[DS].reindex(grid)
        zd[DS] = ds - ds.mean()                       # binary, no daily cycle: demeaned only
        deseason[dev] = zd.reindex(columns=VARS).to_numpy(float)
        rf = fd.reindex(grid)[VARS]
        raw[dev] = ((rf - rf.mean()) / rf.std().replace(0, np.nan)).to_numpy(float)
    return deseason, raw


def usable_vars(arr: np.ndarray) -> list[int]:
    present = np.isfinite(arr)
    hours = present.any(axis=1).sum()
    if hours == 0:
        return []
    keep = []
    for j, v in enumerate(VARS):
        col = arr[:, j]
        if present[:, j].sum() < COVERAGE * hours or np.nanstd(col) < 1e-9:
            continue
        if v == DS:
            c = col[np.isfinite(col)]
            switches = int((np.diff((c > np.median(c)).astype(int)) != 0).sum())
            if switches < MIN_DS_SWITCHES:
                continue
        keep.append(j)
    return keep


# ----------------------------------------------------------------------------- lead-lag
def ccf_device(arr: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    """corr(X(t), Y(t+k)) for k = -L..L, unordered pairs i < j; positive k = X leads Y."""
    T, k = arr.shape
    out = {}
    for i in range(k):
        x = arr[:, i]
        if np.isfinite(x).sum() < MIN_OVERLAP:
            continue
        for j in range(i + 1, k):
            y = arr[:, j]
            curve = np.full(2 * CCF_LAGS + 1, np.nan)
            for n, lag in enumerate(range(-CCF_LAGS, CCF_LAGS + 1)):
                if lag >= 0:
                    xs, ys = x[:T - lag], y[lag:]
                else:
                    xs, ys = x[-lag:], y[:T + lag]
                m = np.isfinite(xs) & np.isfinite(ys)
                if m.sum() >= MIN_OVERLAP and xs[m].std() > 0 and ys[m].std() > 0:
                    curve[n] = np.corrcoef(xs[m], ys[m])[0, 1]
            if np.isfinite(curve).sum() > CCF_LAGS:
                out[(i, j)] = curve
    return out


# ----------------------------------------------------------------------------- Granger
def design(arr: np.ndarray, cols: list[int], p: int):
    """Current values and p lags of the given columns, rows complete over all of them."""
    T = arr.shape[0]
    cur = arr[p:, cols]
    lags = np.concatenate([arr[p - l:T - l, cols] for l in range(1, p + 1)], axis=1)   # lag-major blocks
    ok = np.isfinite(cur).all(axis=1) & np.isfinite(lags).all(axis=1)
    return cur[ok], lags[ok]


def rss(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    return float(e @ e), beta


def bh(pvals: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Hochberg: boolean mask of discoveries."""
    n = len(pvals)
    if n == 0:
        return np.zeros(0, bool)
    order = np.argsort(pvals)
    thresh = q * np.arange(1, n + 1) / n
    passed = pvals[order] <= thresh
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    mask = np.zeros(n, bool)
    mask[order[:k]] = True
    return mask


def granger_device(arr: np.ndarray, p: int, null_rng: np.random.Generator | None = None) -> list[tuple]:
    """Conditional Granger tests for every ordered pair among the device's usable variables.
    Returns rows (x, y, pval, partial_r2, sign, n) with x, y = indices into VARS."""
    cols = usable_vars(arr)
    k = len(cols)
    if k < 2:
        return []
    cur, lags = design(arr, cols, p)
    n = len(cur)
    if n < max(MIN_ROWS, ROWS_PER_REG * (k * p + 1)):
        return []
    X_full = np.column_stack([np.ones(n), lags])
    rows = []
    for ti in range(k):
        y = cur[:, ti]
        if y.std() < 1e-9:
            continue
        rss_f, beta = rss(X_full, y)
        for xi in range(k):
            if xi == ti:
                continue
            xcols = [1 + l * k + xi for l in range(p)]
            if null_rng is None:
                keep = [c for c in range(X_full.shape[1]) if c not in xcols]
                rss_r, _ = rss(X_full[:, keep], y)
                rf, b = rss_f, beta
                nn = n
            else:
                # null: X's history replaced by X circularly shifted by 3-10 days, refit both models
                shift = int(null_rng.integers(*NULL_SHIFT_H))
                a2 = arr.copy()
                a2[:, cols[xi]] = np.roll(arr[:, cols[xi]], shift)
                c2, l2 = design(a2, cols, p)
                nn = len(c2)
                if nn < max(MIN_ROWS, ROWS_PER_REG * (k * p + 1)):
                    continue
                Xf2 = np.column_stack([np.ones(nn), l2])
                rf, b = rss(Xf2, c2[:, ti])
                keep = [c for c in range(Xf2.shape[1]) if c not in xcols]
                rss_r, _ = rss(Xf2[:, keep], c2[:, ti])
            df2 = nn - (k * p + 1)
            if rf <= 0 or df2 <= 0:
                continue
            F = ((rss_r - rf) / p) / (rf / df2)
            pval = float(sps.f.sf(F, p, df2))
            pr2 = float((rss_r - rf) / rss_r) if rss_r > 0 else 0.0
            sign = float(np.sign(sum(b[c] for c in xcols)))
            rows.append((cols[xi], cols[ti], pval, pr2, sign, nn))
    return rows


def choose_p(deseason: dict, rng: np.random.Generator) -> tuple[int, dict]:
    """Fleet lag order by BIC on a sample of devices, all targets, rows common to p = 6."""
    devs = [d for d in deseason if len(usable_vars(deseason[d])) >= 4]
    sample = list(rng.choice(devs, size=min(15, len(devs)), replace=False))
    bic = {p: 0.0 for p in LAG_CHOICES}
    pmax = max(LAG_CHOICES)
    for d in sample:
        arr = deseason[d]
        cols = usable_vars(arr)
        k = len(cols)
        cur6, lags6 = design(arr, cols, pmax)
        n = len(cur6)
        if n < MIN_ROWS:
            continue
        for p in LAG_CHOICES:
            X = np.column_stack([np.ones(n), lags6[:, :k * p]])   # lag-major: first k*p columns = lags 1..p
            for ti in range(k):
                r_, _ = rss(X, cur6[:, ti])
                bic[p] += n * np.log(max(r_, 1e-12) / n) + X.shape[1] * np.log(n)
    return min(bic, key=bic.get), {str(p): round(v, 1) for p, v in bic.items()}


def work(args) -> tuple:
    dev, des, raw, p, seed = args
    rng = np.random.default_rng(seed)
    return (dev, ccf_device(des), granger_device(des, p), granger_device(raw, p), granger_device(des, p, rng),
            [VARS[j] for j in usable_vars(des)])


# ----------------------------------------------------------------------------- aggregate
def summarize_granger(results: dict[str, list[tuple]]) -> dict:
    """Fleet share of devices with a significant X -> Y after BH, median partial R2 and sign."""
    tested, sig, r2, sgn = {}, {}, {}, {}
    for dev, rows in results.items():
        if not rows:
            continue
        mask = bh(np.array([r[2] for r in rows]), Q)
        for (x, y, pv, pr2, s, n), m in zip(rows, mask):
            key = f"{VARS[x]}>{VARS[y]}"
            tested[key] = tested.get(key, 0) + 1
            if m:
                sig[key] = sig.get(key, 0) + 1
                r2.setdefault(key, []).append(pr2)
                sgn.setdefault(key, []).append(s)
    out = {}
    for key, n in tested.items():
        s = sig.get(key, 0)
        out[key] = {"n": n, "n_sig": s, "share": round(s / n, 4),
                    "r2_median": round(float(np.median(r2[key])), 5) if key in r2 else 0.0,
                    "sign": float(np.sign(np.median(sgn[key]))) if key in sgn else 0.0}
    return out


def main() -> None:
    t0 = time.time()
    deseason, raw = load()
    print(f"loaded {len(deseason)} devices ({time.time() - t0:.0f}s)", flush=True)
    rng = np.random.default_rng(SEED)
    p, bic = choose_p(deseason, rng)
    print(f"lag order p = {p} (BIC {bic}) ({time.time() - t0:.0f}s)", flush=True)

    jobs = [(d, deseason[d], raw[d], p, SEED + i) for i, d in enumerate(sorted(deseason))]
    with ProcessPoolExecutor(max_workers=min(64, len(jobs))) as ex:
        res = list(ex.map(work, jobs))
    print(f"per-device fits done ({time.time() - t0:.0f}s)", flush=True)

    ccf_all, g_des, g_raw, g_null, used = {}, {}, {}, {}, {}
    for dev, ccf, gd, gr, gn, uv in res:
        g_des[dev], g_raw[dev], g_null[dev], used[dev] = gd, gr, gn, uv
        for (i, j), curve in ccf.items():
            ccf_all.setdefault((i, j), []).append(curve)

    granger = summarize_granger(g_des)
    granger_raw = summarize_granger(g_raw)
    granger_null = summarize_granger(g_null)
    for key, g in granger.items():
        g["null_share"] = granger_null.get(key, {}).get("share")

    lags = list(range(-CCF_LAGS, CCF_LAGS + 1))
    ccf = {}
    for (i, j), curves in ccf_all.items():
        C = np.vstack(curves)
        med = np.nanmedian(C, axis=0)
        if not np.isfinite(med).any():
            continue
        pk = int(np.nanargmax(np.abs(med)))
        peak_lag, peak_r = lags[pk], float(med[pk])
        agree = 0
        for c in C:
            if np.isfinite(c).any():
                own = int(np.nanargmax(np.abs(c)))
                agree += int(np.sign(c[own]) == np.sign(peak_r) and abs(lags[own] - peak_lag) <= 1)
        ccf[f"{VARS[i]}|{VARS[j]}"] = {"n": len(curves), "peak_lag": peak_lag, "peak_r": round(peak_r, 4),
                                       "share_agree": round(agree / len(curves), 3),
                                       "r_lag0": round(float(med[CCF_LAGS]), 4),
                                       "curve": [None if not np.isfinite(v) else round(float(v), 4) for v in med],
                                       "q25": [None if not np.isfinite(v) else round(float(v), 4) for v in np.nanpercentile(C, 25, axis=0)],
                                       "q75": [None if not np.isfinite(v) else round(float(v), 4) for v in np.nanpercentile(C, 75, axis=0)]}

    edges = []
    for key, g in granger.items():
        if g["share"] >= EDGE_SHARE and g["r2_median"] >= EDGE_R2:
            x, y = key.split(">")
            rev = granger.get(f"{y}>{x}", {})
            edges.append([x, y, g["share"], g["r2_median"], g["sign"], g["n"],
                          bool(rev.get("share", 0) >= EDGE_SHARE and rev.get("r2_median", 0) >= EDGE_R2)])
    edges.sort(key=lambda e: (-e[2], -e[3]))
    raw_edges = [k for k, g in granger_raw.items() if g["share"] >= EDGE_SHARE]

    per_device = {}
    for dev, rows in g_des.items():
        if rows:
            mask = bh(np.array([r[2] for r in rows]), Q)
            per_device[dev] = [[VARS[x], VARS[y], float(f"{pv:.3g}"), round(pr2, 5)] for (x, y, pv, pr2, s, n), m in zip(rows, mask) if m]

    out = {"generated": pd.Timestamp.now("UTC").isoformat(),
           "params": {"p": p, "bic": bic, "q": Q, "min_rows": MIN_ROWS, "coverage": COVERAGE, "edge_share": EDGE_SHARE,
                      "edge_r2": EDGE_R2, "ccf_lags": CCF_LAGS, "min_overlap_h": MIN_OVERLAP, "null_shift_h": list(NULL_SHIFT_H),
                      "min_ds_switches": MIN_DS_SWITCHES},
           "variables": VARS, "lags": lags,
           "devices_tested": sum(1 for r in g_des.values() if r), "devices_total": len(g_des),
           "variables_used": {v: sum(v in u for u in used.values()) for v in VARS},
           "ccf": ccf, "granger": granger, "granger_raw": granger_raw, "edges": edges, "raw_edges": raw_edges,
           "per_device": per_device}
    (OUT / "influence.json").write_text(json.dumps(out, default=float))
    print(f"wrote influence.json in {time.time() - t0:.0f}s\n")

    null_shares = [g["share"] for g in granger_null.values() if g["n"] >= 10]
    print(f"devices tested {out['devices_tested']}/{len(g_des)}; edges {len(edges)} deseasonalized vs {len(raw_edges)} raw (share >= {EDGE_SHARE}); "
          f"null median share {np.median(null_shares):.3f} (q = {Q}), max {np.max(null_shares):.3f}")
    print("variables used (devices):", out["variables_used"])
    for e in edges:
        x, y = e[0], e[1]
        c = ccf.get(f"{x}|{y}") or ccf.get(f"{y}|{x}")
        lag = (c["peak_lag"] if f"{x}|{y}" in ccf else -c["peak_lag"]) if c else None
        print(f"  {x:28s} -> {y:28s} share={e[2]:.2f} r2={e[3]:.4f} sign={e[4]:+.0f} n={e[5]:3d} bidir={e[6]}  ccf peak lag={lag} r={c['peak_r'] if c else None}")


if __name__ == "__main__":
    main()

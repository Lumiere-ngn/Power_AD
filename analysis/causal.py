#!/usr/bin/env python
"""Causal relationships between telemetry variables, tested online with River.

Input: the deseasonalized series from anomaly.py, z = (value - Holt-Winters one-step forecast) / scale,
per device, variable and hour. The daily cycle is already removed, so two variables that only share
the sun do not count as related.

1. Directed links, one hour or more apart (predictive, "Granger" causality, tested out of sample):
   per device, online Bayesian linear regressions (River `linear_model.BayesianLinearRegression`, an
   exact recursive ridge) predict every variable Y(t) from the last LAGS hours of all variables.
   For each candidate cause X, a second model gets the same inputs without X. Each hour is predicted
   before the models learn from it. X -> Y holds on a device when the model with X has the lower
   squared error (Clark-West test for nested forecasts, Newey-West variance, Benjamini-Hochberg
   across all devices and pairs). Conditioning on every other variable keeps an indirect path
   A -> B -> C from showing up as A -> C.
2. Same-hour links (no direction): partial correlations of the full models' forecast errors, from
   River's online precision matrix (`covariance.EmpiricalPrecision`). Effects faster than the hourly
   table, such as power heating the processor, land here.
3. Checks: a null run where X is replaced by a copy of itself shifted by 3-10 days, and a replication
   of every link on July, which the discovery step (8 March - 30 June) never sees.
August to October are not used, so the result can serve as a learned graph without touching the
test months of the research-question protocol.

Reads analysis/out/anomalies.parquet; writes analysis/out/causal.json.
Usage: python -I analysis/causal.py [--selftest]
"""
from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from river import covariance, linear_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/out"
ANOM = OUT / "anomalies.parquet"

# variable: (label, group). Input current is left out (it is power / voltage), GPU temperature too
# (it duplicates CPU temperature), and battery voltage (few devices; battery charge covers it).
VARS = {
    "lightsensor": ("Light", "environment"),
    "temperature": ("Enclosure temperature", "environment"),
    "humidity": ("Humidity", "environment"),
    "board_temp_celsius": ("Board temperature", "thermal"),
    "cpu_temp_celsius": ("CPU temperature", "thermal"),
    "deepstream_app_running": ("DeepStream running", "compute"),
    "cpu_gpu_power_watts": ("CPU+GPU power", "power"),
    "orin_power_watts": ("Orin module power", "power"),
    "input_power_watts": ("Input power", "power"),
    "input_voltage_volts": ("Input voltage", "power"),
    "memory_free_gb": ("Free RAM", "compute"),
    "storage_free_gb": ("Free disk", "compute"),
    "wwan_tx_gb_rate": ("LTE upload", "network"),
    "mppt_charge_power": ("Solar charge power", "battery"),
    "battery_remaining_capacity": ("Battery charge", "battery"),
}
NAMES = list(VARS)

LAGS = 2
FIT_END, REP_END = "2026-07-01", "2026-08-01"   # discovery < FIT_END <= replication < REP_END
Z_CLIP = 6.0            # limits the leverage of single extreme hours
COVERAGE = 0.7          # a variable enters a device's model if scored in >= 70 % of its hours
MIN_STD = 0.25          # ... and if it actually varies (a process that never stops carries no signal)
BURN_IN = 168           # hours the models learn before their errors count
MIN_ROWS, MIN_ROWS_REP = 500, 200
NW_LAGS = 6
Q = 0.01                # Benjamini-Hochberg false discovery rate
P_REP = 0.05
MIN_DEVICES, EDGE_SHARE, EDGE_SKILL = 8, 0.5, 0.005
PCORR_MIN = 0.1
NULL_SHIFT_H = (72, 240)
SEED = 42


def norm_sf(t: float) -> float:
    return 0.5 * math.erfc(t / math.sqrt(2))


def cw_test(e_r: np.ndarray, e_f: np.ndarray, d_pred: np.ndarray) -> tuple[float, float]:
    """Clark-West test that the larger model forecasts better. Returns (one-sided p, skill)."""
    f = e_r ** 2 - e_f ** 2 + d_pred ** 2
    n = len(f)
    fc = f - f.mean()
    var = fc @ fc / n
    for k in range(1, NW_LAGS + 1):
        var += 2 * (1 - k / (NW_LAGS + 1)) * (fc[k:] @ fc[:-k]) / n
    se = math.sqrt(max(var, 1e-18) / n)
    sse_r = float(e_r @ e_r)
    return norm_sf(f.mean() / se), (1 - float(e_f @ e_f) / sse_r) if sse_r > 0 else 0.0


def run_device(args) -> dict | None:
    """Stream one device: full, leave-one-out and shifted-cause models for every target."""
    dev, times, Z, names = args
    n_all = len(times)
    fit_end, rep_end = np.datetime64(FIT_END), np.datetime64(REP_END)
    inwin = times < rep_end
    finite = np.isfinite(Z)
    Zc = np.clip(Z, -Z_CLIP, Z_CLIP)
    if not inwin.any():
        return {"device": dev, "skip": "no hours before " + REP_END}
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        use = (finite[inwin].mean(axis=0) >= COVERAGE) & (np.nanstd(Zc[inwin], axis=0) >= MIN_STD)
    if use.sum() < 3:
        return {"device": dev, "skip": "fewer than 3 usable variables"}
    names = [m for m, u in zip(names, use) if u]
    Zc, finite = Zc[:, use], finite[:, use]
    V = len(names)
    row_ok = finite.all(axis=1)
    valid = row_ok.copy()
    for k in range(1, LAGS + 1):
        valid[k:] &= row_ok[:-k]
        valid[:k] = False
    rows = np.flatnonzero(valid & inwin)
    if len(rows) < BURN_IN + MIN_ROWS:
        return {"device": dev, "skip": "too few complete hours"}
    rng = np.random.default_rng([SEED, sum(map(ord, dev))])
    Zs = np.column_stack([np.roll(np.nan_to_num(Zc[:, j]), int(rng.integers(*NULL_SHIFT_H))) for j in range(V)])

    blr = linear_model.BayesianLinearRegression
    full = [blr() for _ in range(V)]
    res = [[blr() if x != y else None for x in range(V)] for y in range(V)]
    null = [[blr() if x != y else None for x in range(V)] for y in range(V)]
    keys = [[j * 10 + k for k in range(1, LAGS + 1)] for j in range(V)]
    R = len(rows)
    tgt = Zc[rows]
    pf = np.zeros((R, V), np.float32)
    pr = np.zeros((R, V, V), np.float32)       # [row, y, x]: forecast of y without x
    pn = np.zeros((R, V, V), np.float32)       # [row, y, x]: forecast of y with shifted x
    coef, snap = None, False
    def snapshot():
        c = np.zeros((V, V, LAGS))             # [y, x, lag-1]
        for y in range(V):
            m = full[y]; m._ensure_m()
            for x in range(V):
                for k in range(LAGS):
                    c[y, x, k] = m._m_arr[m._idx[keys[x][k]]]
        return c

    for r, t in enumerate(rows):
        if not snap and times[t] >= fit_end:
            coef, snap = snapshot(), True
        x_full = {keys[j][k - 1]: Zc[t - k, j] for j in range(V) for k in range(1, LAGS + 1)}
        x_res, x_null = [], []
        for j in range(V):
            d = x_full.copy()
            for kk in keys[j]:
                del d[kk]
            x_res.append(d)
            d = x_full.copy()
            for k in range(1, LAGS + 1):
                d[keys[j][k - 1]] = Zs[t - k, j]
            x_null.append(d)
        for y in range(V):
            v = tgt[r, y]
            pf[r, y] = full[y].predict_one(x_full)
            full[y].learn_one(x_full, v)
            for x in range(V):
                if x == y:
                    continue
                pr[r, y, x] = res[y][x].predict_one(x_res[x])
                res[y][x].learn_one(x_res[x], v)
                pn[r, y, x] = null[y][x].predict_one(x_null[x])
                null[y][x].learn_one(x_null[x], v)
    if coef is None:
        coef = snapshot()

    rt = times[rows]
    counted = np.arange(R) >= BURN_IN
    disc, rep = counted & (rt < fit_end), counted & (rt >= fit_end)
    if disc.sum() < MIN_ROWS:
        return {"device": dev, "skip": "too few hours before " + FIT_END}
    has_rep = rep.sum() >= MIN_ROWS_REP
    shape = (V, V)
    p, skill, p_null = np.full(shape, np.nan), np.full(shape, np.nan), np.full(shape, np.nan)
    p_rep, skill_rep = np.full(shape, np.nan), np.full(shape, np.nan)
    e_f = tgt - pf
    for y in range(V):
        for x in range(V):
            if x == y:
                continue
            e_r, e_n = tgt[:, y] - pr[:, y, x], tgt[:, y] - pn[:, y, x]
            d, dn = pr[:, y, x] - pf[:, y], pr[:, y, x] - pn[:, y, x]
            p[x, y], skill[x, y] = cw_test(e_r[disc], e_f[disc, y], d[disc])
            p_null[x, y], _ = cw_test(e_r[disc], e_n[disc], dn[disc])
            if has_rep:
                p_rep[x, y], skill_rep[x, y] = cw_test(e_r[rep], e_f[rep, y], d[rep])
    # same-hour links: partial correlations of the full models' errors (discovery hours)
    prec = covariance.EmpiricalPrecision()
    for row in e_f[disc]:
        prec.update(dict(zip(names, row.tolist())))
    pc = np.full(shape, np.nan)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i != j:
                pc[i, j] = -prec[a, b] / math.sqrt(prec[a, a] * prec[b, b])
    return {"device": dev, "names": names, "n_disc": int(disc.sum()), "n_rep": int(rep.sum()) if has_rep else 0,
            "p": p, "skill": skill, "p_null": p_null, "p_rep": p_rep, "skill_rep": skill_rep,
            "coef": np.transpose(coef, (1, 0, 2)),    # -> [x, y, lag-1]
            "pcorr": pc, "r2_full": 1 - (e_f[disc] ** 2).sum(axis=0) / np.maximum((tgt[disc] ** 2).sum(axis=0), 1e-12)}


def load() -> list[tuple]:
    a = pd.read_parquet(ANOM, columns=["device", "metric", "time", "z", "scored"],
                        filters=[("metric", "in", NAMES), ("scored", "==", True)])
    a = a[a["time"] < pd.Timestamp(REP_END, tz="UTC")]
    jobs = []
    for dev, g in a.groupby("device", observed=True):
        w = g.pivot(index="time", columns="metric", values="z").reindex(columns=NAMES)
        grid = pd.date_range(w.index.min(), w.index.max(), freq="h")
        w = w.reindex(grid)
        jobs.append((str(dev), w.index.tz_localize(None).to_numpy().astype("datetime64[s]"), w.to_numpy(dtype=float), NAMES))
    return jobs


def bh_cut(p: np.ndarray, q: float) -> float:
    """Largest p-value accepted by Benjamini-Hochberg at rate q (0 if none)."""
    s = np.sort(p)
    ok = np.flatnonzero(s <= q * np.arange(1, len(s) + 1) / len(s))
    return float(s[ok[-1]]) if len(ok) else 0.0


def aggregate(results: list[dict], skipped: dict | None = None) -> dict:
    allp = np.concatenate([r["p"][np.isfinite(r["p"])] for r in results])
    cut = bh_cut(allp, Q)
    pairs, lag0 = {}, {}
    used = {v: sum(v in r["names"] for r in results) for v in NAMES}
    med = lambda a: round(float(np.median(a)), 4) if len(a) else None
    for x in NAMES:
        for y in NAMES:
            if x == y:
                continue
            rs = [(r, r["names"].index(x), r["names"].index(y)) for r in results if x in r["names"] and y in r["names"]]
            if not rs:
                continue
            p = np.array([r["p"][i, j] for r, i, j in rs])
            sk = np.array([r["skill"][i, j] for r, i, j in rs])
            pn = np.array([r["p_null"][i, j] for r, i, j in rs])
            c = np.array([r["coef"][i, j] for r, i, j in rs])           # devices x lags
            sig = p <= cut
            csum = c.sum(axis=1)
            sign = float(np.sign(np.median(csum[sig]))) if sig.any() else float(np.sign(np.median(csum)))
            pr_ = np.array([r["p_rep"][i, j] for r, i, j in rs if r["n_rep"]])
            sr_ = np.array([r["skill_rep"][i, j] for r, i, j in rs if r["n_rep"]])
            g = {"n": len(rs), "share": round(float(sig.mean()), 3), "skill": med(sk),
                 "skill_q": [round(float(np.quantile(sk, q)), 4) for q in (0.25, 0.75)],
                 "sign": int(sign), "sign_agree": round(float((np.sign(csum[sig]) == sign).mean()), 3) if sig.any() else None,
                 "coef": [med(c[:, k]) for k in range(LAGS)], "null_share": round(float((pn <= cut).mean()), 3),
                 "rep_n": int(len(pr_)), "rep_share": round(float((pr_ <= P_REP).mean()), 3) if len(pr_) else None, "rep_skill": med(sr_)}
            g["edge"] = bool(g["n"] >= MIN_DEVICES and g["share"] >= EDGE_SHARE and g["skill"] >= EDGE_SKILL)
            g["replicated"] = bool(g["edge"] and g["rep_n"] >= MIN_DEVICES and g["rep_share"] >= 0.5 and g["rep_skill"] > 0)
            pairs[f"{x}>{y}"] = g
            if NAMES.index(x) < NAMES.index(y):
                pc = np.array([r["pcorr"][i, j] for r, i, j in rs])
                s0 = float(np.sign(np.median(pc)))
                lag0[f"{x}|{y}"] = {"n": len(rs), "pcorr": med(pc), "pcorr_q": [round(float(np.quantile(pc, q)), 3) for q in (0.25, 0.75)],
                                    "share": round(float(((np.abs(pc) >= PCORR_MIN) & (np.sign(pc) == s0)).mean()), 3)}
                lag0[f"{x}|{y}"]["link"] = bool(len(rs) >= MIN_DEVICES and lag0[f"{x}|{y}"]["share"] >= EDGE_SHARE)
    edges = sorted((k for k, g in pairs.items() if g["edge"]), key=lambda k: -pairs[k]["share"])
    tested = [g for g in pairs.values() if g["n"] >= MIN_DEVICES]
    n_tests = int(len(allp))
    null_all = np.concatenate([r["p_null"][np.isfinite(r["p_null"])] for r in results])
    return {
        "generated": pd.Timestamp.now("UTC").isoformat(),
        "params": {"lags": LAGS, "fit_end": FIT_END, "rep_end": REP_END, "z_clip": Z_CLIP, "coverage": COVERAGE, "min_std": MIN_STD,
                   "burn_in_h": BURN_IN, "min_rows": MIN_ROWS, "min_rows_rep": MIN_ROWS_REP, "nw_lags": NW_LAGS, "q": Q, "p_rep": P_REP,
                   "min_devices": MIN_DEVICES, "edge_share": EDGE_SHARE, "edge_skill": EDGE_SKILL, "pcorr_min": PCORR_MIN, "null_shift_h": NULL_SHIFT_H},
        "variables": NAMES, "labels": {v: VARS[v][0] for v in NAMES}, "groups": {v: VARS[v][1] for v in NAMES}, "variables_used": used,
        "devices_tested": len(results), "devices_skipped": skipped or {}, "devices_rep": sum(1 for r in results if r["n_rep"]),
        "rows_disc_median": int(np.median([r["n_disc"] for r in results])), "rows_rep_median": int(np.median([r["n_rep"] for r in results if r["n_rep"]] or [0])),
        "n_tests": n_tests, "p_cut": cut, "share_sig": round(float((allp <= cut).mean()), 4), "null_share_sig": round(float((null_all <= cut).mean()), 4),
        "pairs": pairs, "edges": edges, "n_pairs_tested": len(tested),
        "lag0": lag0, "lag0_links": sorted((k for k, g in lag0.items() if g["link"]), key=lambda k: -abs(lag0[k]["pcorr"])),
        "r2_full": {v: med(np.array([r["r2_full"][r["names"].index(v)] for r in results if v in r["names"]])) for v in NAMES if used[v]},
        "per_device": [{"device": r["device"], "n_vars": len(r["names"]), "n_disc": r["n_disc"], "n_rep": r["n_rep"],
                        "n_sig": int((r["p"] <= cut).sum())} for r in results],
    }


def selftest() -> None:
    """A -> B (1 h), B -> C (2 h), D alone. A -> C is indirect and must not be reported."""
    rng = np.random.default_rng(1)
    n = 2400
    A, B, C, D = (rng.normal(size=n) for _ in range(4))
    for t in range(2, n):
        B[t] += 0.6 * A[t - 1]
        C[t] += 0.5 * B[t - 2]
    C += 0.5 * D                                   # a same-hour link between C and D
    times = (np.datetime64("2026-03-10T00:00:00") + np.arange(n) * np.timedelta64(3600, "s")).astype("datetime64[s]")
    r = run_device(("test", times, np.column_stack([A, B, C, D]), ["A", "B", "C", "D"]))
    p, pc = r["p"], r["pcorr"]
    print("p-values (rows = cause, cols = effect):\n", np.array2string(p, precision=4, suppress_small=True))
    print("skill A->B", round(r["skill"][0, 1], 3), "B->C", round(r["skill"][1, 2], 3), "| coef A->B", r["coef"][0, 1].round(2), "B->C", r["coef"][1, 2].round(2))
    print("same-hour partial correlation C|D", round(pc[2, 3], 3), "A|D", round(pc[0, 3], 3), "| null p min", round(float(np.nanmin(r["p_null"])), 4))
    assert p[0, 1] < 1e-6 and p[1, 2] < 1e-6 and p[0, 2] > 0.01 and p[1, 0] > 0.01 and p[2, 1] > 0.01
    assert pc[2, 3] > 0.3 and abs(pc[0, 3]) < 0.1
    print("selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    t0 = time.time()
    jobs = load()
    print(f"{len(jobs)} devices with scored hours before {REP_END} ({time.time() - t0:.0f}s)", flush=True)
    jobs.sort(key=lambda j: -np.isfinite(j[2]).sum())
    with ProcessPoolExecutor(max_workers=min(90, len(jobs))) as ex:
        allr = list(ex.map(run_device, jobs))
    results = [r for r in allr if "skip" not in r]
    skipped = dict(Counter(r["skip"] for r in allr if "skip" in r))
    print(f"{len(results)} devices tested, skipped: {skipped} ({time.time() - t0:.0f}s)", flush=True)
    out = aggregate(results, skipped)
    (OUT / "causal.json").write_text(json.dumps(out, default=float))
    print(f"tests {out['n_tests']:,}, significant {100 * out['share_sig']:.1f} % (BH q = {Q}, p <= {out['p_cut']:.2e}); "
          f"shifted-cause null {100 * out['null_share_sig']:.2f} %")
    print(f"directed links: {len(out['edges'])} of {out['n_pairs_tested']} ordered pairs; replicated in July: {sum(out['pairs'][k]['replicated'] for k in out['edges'])}")
    for k in out["edges"]:
        g = out["pairs"][k]
        print(f"  {k:52s} n={g['n']:3d} share={g['share']:.2f} skill={g['skill']:.4f} sign={g['sign']:+d} coef={g['coef']} "
              f"null={g['null_share']:.2f} july: n={g['rep_n']} share={g['rep_share']} skill={g['rep_skill']} {'REPL' if g['replicated'] else ''}")
    print(f"same-hour links: {len(out['lag0_links'])}")
    for k in out["lag0_links"]:
        g = out["lag0"][k]
        print(f"  {k:52s} n={g['n']:3d} pcorr={g['pcorr']:+.3f} share={g['share']:.2f}")
    print("variables used (devices):", out["variables_used"])


if __name__ == "__main__":
    main()

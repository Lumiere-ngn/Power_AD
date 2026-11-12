#!/usr/bin/env python
"""RQ1: how well does context predict a sign's expected power and temperature?  Plan: protocol/PLAN_RQ1.md.

Targets (hourly means): input, Orin and CPU+GPU power; CPU, GPU and board temperature.
Inputs: time, hardware, workload, enclosure sensors (the sign's own and its agency peers'), city weather, and
the last two hours of four inputs. Never another power or thermal reading, never a past value of a target.
Baseline: the sign's own median for that hour of day (the fleet's median for a sign the model has not seen).
Skill = 1 - MAE(model) / MAE(baseline). Temperatures follow the season, so a second, harder baseline uses only the
last 28 days of the training window ("recent baseline").

Two stages
  dev   Loads nothing from 1 August on. Chooses the inputs and the LightGBM settings on June (fit March-May),
        rechecks them on July (fit March-June), and chooses how the 90 % bands are calibrated on July
        (fit March-May, calibrate June). Writes out/rq1_dev.json and the locked settings out/rq1_locked.json.
  test  Reads the locked settings. Fits on March-June, calibrates the bands on July and scores
        1 August - 6 October once: the same signs later, unseen signs (5 folds), unseen hardware or site,
        and each input group dropped in turn. Writes out/rq1_test.json and refuses to run if it exists.

In every fit, the training hours that anomaly.py flagged for the target are removed, so the model learns normal
behaviour. Scored hours are never removed; the error is reported for flagged and other hours separately.

The `demo` and `show` units are not in the data (rq_common.EXCLUDED_AGENCIES, decision D6 of 10 November 2026).
This is the second run: the first one, with those units, is in archive/2026-11-10-rq1-run1/.

Usage: python -I analysis/rq1.py dev | test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script directory
import rq_common as C  # noqa: E402

SEED = 20261103                       # protocol.toml, inference.seed_development
JUNE = pd.Timestamp("2026-06-01", tz="UTC")
MIN_HOURS = 168                       # a sign is scored in a window if it has a week of input power there
NOMINAL = 0.90
BOOT = 2000                           # bootstrap resamples, over signs

LAG_SRC = ["lightsensor", "temperature", "wx_temp_c", "wx_solar_wm2"]
LAGS = [f"{c}_lag{k}" for c in LAG_SRC for k in (1, 2)]
ENV = C.ENV_EXT + C.ENV_LOCAL
BASE = C.TIME + C.HW + ENV + C.WORKLOAD
INPUT_SETS = {  # from simplest to richest
    "pilot inputs": BASE,
    "+ weather": BASE + C.WEATHER,
    "+ weather + lags": BASE + C.WEATHER + LAGS,
}
SETTINGS = {f"{leaves} leaves, {trees} trees": {"num_leaves": leaves, "n_estimators": trees}
            for trees in (200, 500) for leaves in (15, 31, 63, 127)}
FAMILY = {"power": C.POWER, "thermal": C.THERMAL}
GROUPS = {  # input groups dropped in turn; a lag belongs to the group of its source
    "workload": C.WORKLOAD,
    "environment": ENV + [c for c in LAGS if not c.startswith("wx_")],
    "weather": C.WEATHER + [c for c in LAGS if c.startswith("wx_")],
    "hardware": C.HW,
}
BANDS = ["fleet", "per configuration", "per sign"]   # one conformal margin for all, per supply x solar, or per sign
ALLOWED = set(BASE + C.WEATHER + LAGS + ["device"])
assert not ALLOWED & set(C.TARGETS)
# The test months have been scored twice; see the deviation log of protocol/PROTOCOL.md.
RUN = ("Second scoring of 1 August - 6 October (10 November 2026): redone without the demo and show units after the "
       "first run had been read. First run: archive/2026-11-10-rq1-run1/.")
TIE = 0.005                           # differences below this count as a tie; the simpler option wins

F: pd.DataFrame                       # set before the workers are forked
MASKS: dict[str, np.ndarray] = {}


# ---------------------------------------------------------------- data

def prepare(before: pd.Timestamp | None = None) -> None:
    """Feature table + lags of four inputs + the flags of anomaly.py. `before` cuts everything at that time."""
    global F
    f = C.load(before=before).sort_values(["device", "time"]).reset_index(drop=True)
    step = f.groupby("device", observed=True)["time"].diff().dropna()
    assert (step == pd.Timedelta(hours=1)).all(), "the hourly grid has gaps: lags would be wrong"
    g = f.groupby("device", observed=True)
    for c in LAG_SRC:
        for k in (1, 2):
            f[f"{c}_lag{k}"] = g[c].shift(k)
    f["dev"] = f["device"].astype(str)
    f["hour"] = f["time"].dt.hour.astype("int8")
    f["config"] = np.where(f["v56"] == 1, "56 V", "12 V") + np.where(f["has_mppt"] == 1, " solar", "")
    a = pd.read_parquet(C.OUT / "anomalies.parquet", columns=["metric", "device", "time", "flag"])
    a = a[(a["flag"] != 0) & a["metric"].isin(C.TARGETS) & a["device"].astype(str).isin(set(f["dev"]))]
    if before is not None:
        a = a[a["time"] < before]
    key = pd.MultiIndex.from_arrays([f["dev"], f["time"]])
    for y in C.TARGETS:
        ay = a[a["metric"] == y]
        f[f"flag_{y}"] = key.isin(pd.MultiIndex.from_arrays([ay["device"].astype(str), ay["time"]]))
        assert f[f"flag_{y}"].sum() == len(ay), (y, int(f[f"flag_{y}"].sum()), len(ay))
    F = f[f[C.TARGETS].notna().any(axis=1)].reset_index(drop=True)


def signs_with(*masks: np.ndarray) -> set[str]:
    """Signs with at least MIN_HOURS of input power in every window."""
    out = None
    for m in masks:
        n = F.loc[m].groupby("dev")["input_power_watts"].count()
        s = set(n[n >= MIN_HOURS].index)
        out = s if out is None else out & s
    return out


def train_rows(y: str, mask: str, clean: bool = True) -> np.ndarray:
    m = MASKS[mask] & F[y].notna().to_numpy()
    return m & ~F[f"flag_{y}"].to_numpy() if clean else m


def recent(tr: np.ndarray) -> np.ndarray:
    """The last 28 days of a set of training rows."""
    t = F["time"]
    return tr & (t >= t[tr].max() - pd.Timedelta(days=28)).to_numpy()


def vs_recent(y: str, tr: np.ndarray, idx: np.ndarray, yt, p, dv) -> dict:
    m = score(yt, p, baseline(y, recent(tr), idx), dv)
    return {k: m[k] for k in ("mae_baseline", "skill", "mae_baseline_ci", "skill_ci") if k in m}


# ---------------------------------------------------------------- model fits (in forked workers)

def run_job(job: dict):
    y, feats = job["target"], job["feats"]
    assert set(feats) <= ALLOWED, set(feats) - ALLOWED
    tr = train_rows(y, job["train"], job.get("clean", True))
    te = MASKS[job["pred"]] & F[y].notna().to_numpy()
    params = C.lgbm_params(**job["params"], random_state=SEED, deterministic=True, force_row_wise=True)
    if "alpha" in job:
        params |= {"objective": "quantile", "alpha": job["alpha"]}
    m = lgb.LGBMRegressor(**params).fit(F.loc[tr, feats], F.loc[tr, y])
    gain = m.booster_.feature_importance("gain")
    imp = dict(zip(feats, (gain / max(gain.sum(), 1e-12)).round(5).tolist()))
    return job["id"], np.flatnonzero(te), m.predict(F.loc[te, feats]).astype("float64"), imp


def run_jobs(jobs: list[dict], label: str) -> tuple[dict, dict]:
    t0 = time.time()
    preds, imps = {}, {}
    with ProcessPoolExecutor(max_workers=12, mp_context=mp.get_context("fork")) as ex:
        for i, (jid, idx, p, imp) in enumerate(ex.map(run_job, jobs)):
            preds[jid], imps[jid] = (idx, p), imp
            if (i + 1) % 25 == 0 or i + 1 == len(jobs):
                print(f"  {label}: {i + 1}/{len(jobs)} fits ({time.time() - t0:.0f}s)", flush=True)
    return preds, imps


class Jobs(list):
    def add(self, **j) -> int:
        j["id"] = len(self)
        self.append(j)
        return j["id"]


# ---------------------------------------------------------------- baselines and metrics

def baseline(y: str, tr: np.ndarray, idx: np.ndarray, by: str = "dev") -> np.ndarray:
    """Hour-of-day median over the training rows: per sign ("dev"), per agency, or for the fleet (None).
    Falls back to the fleet's hour-of-day median, so it also works for signs that are not in the training rows."""
    d = F.loc[tr, ["dev", "agency", "hour", y]]
    hour = F["hour"].to_numpy()[idx]
    p = pd.Series(hour).map(d.groupby("hour")[y].median()).to_numpy(dtype=float)
    p = np.where(np.isnan(p), d[y].median(), p)
    if by is None:
        return p
    k_tr, k_te = d[by].astype(str), F[by].astype(str).to_numpy()[idx]
    q = d.groupby([k_tr, "hour"])[y].median().reindex(pd.MultiIndex.from_arrays([k_te, hour])).to_numpy(dtype=float)
    if by == "dev":  # an hour of day the sign never reported in training: the sign's overall median
        q = np.where(np.isnan(q), pd.Series(k_te).map(d.groupby(k_tr)[y].median()).to_numpy(dtype=float), q)
    return np.where(np.isnan(q), p, q)


def ci(x: np.ndarray) -> list[float]:
    return [float(v) for v in np.nanpercentile(x, [2.5, 97.5])]


def per_sign(dev: np.ndarray, **cols: np.ndarray):
    codes, names = pd.factorize(dev)
    return names, np.bincount(codes).astype(float), {k: np.bincount(codes, weights=v, minlength=len(names))
                                                    for k, v in cols.items()}, codes


def score(y: np.ndarray, p: np.ndarray, b: np.ndarray, dev: np.ndarray) -> dict:
    """Error of prediction p and of baseline b. Intervals: bootstrap over signs, 95 %."""
    names, n, s, codes = per_sign(dev, a=np.abs(y - p), b=np.abs(y - b), y=y)
    mean_d = s["y"] / n
    nm = np.where(mean_d > 0, (s["a"] / n) / np.where(mean_d > 0, mean_d, 1), np.nan)
    e = y - p
    out = {"n": int(len(y)), "signs": int(len(names)), "mae": float(s["a"].sum() / n.sum()),
           "mae_baseline": float(s["b"].sum() / n.sum()), "skill": float(1 - s["a"].sum() / s["b"].sum()),
           "nmae": float(np.nanmean(nm)), "bias": float(e.mean()),
           "r2_within": float(1 - (e ** 2).sum() / max(((y - mean_d[codes]) ** 2).sum(), 1e-12))}
    if len(names) >= 5:
        d = np.random.default_rng(SEED).integers(0, len(names), size=(BOOT, len(names)))
        A, B, N = s["a"][d].sum(1), s["b"][d].sum(1), n[d].sum(1)
        out |= {"mae_ci": ci(A / N), "mae_baseline_ci": ci(B / N), "skill_ci": ci(1 - A / B),
                "nmae_ci": ci(np.nanmean(nm[d], axis=1))}
    return out


def change(y: np.ndarray, p: np.ndarray, ref: np.ndarray, dev: np.ndarray) -> dict:
    """MAE(p) - MAE(ref) on the same rows, with a paired bootstrap over signs."""
    names, n, s, _ = per_sign(dev, d=np.abs(y - p) - np.abs(y - ref))
    out = {"mae_change": float(s["d"].sum() / n.sum())}
    if len(names) >= 5:
        d = np.random.default_rng(SEED).integers(0, len(names), size=(BOOT, len(names)))
        out["mae_change_ci"] = ci(s["d"][d].sum(1) / n[d].sum(1))
    return out


def by_group(keys: dict[str, np.ndarray], y, p, b, dev) -> dict:
    out = {}
    for name, k in keys.items():
        out[name] = {str(v): score(y[k == v], p[k == v], b[k == v], dev[k == v])
                     for v in sorted(pd.unique(k)) if (k == v).sum() >= 200}
    return out


def margins(s: np.ndarray, is_cal: np.ndarray, dev: np.ndarray, cfg: np.ndarray, variant: str) -> np.ndarray:
    """Conformal margin added to both sides of the 5 %-95 % quantile band, from the calibration hours."""
    def q(v):
        return float(np.quantile(v, min(1.0, NOMINAL * (1 + 1 / len(v)))))
    fleet = q(s[is_cal])
    if variant == "fleet":
        return np.full(len(s), fleet)
    key = cfg if variant == "per configuration" else dev
    m = pd.Series(s[is_cal]).groupby(key[is_cal]).agg(lambda v: q(v.to_numpy()) if len(v) >= 100 else np.nan)
    return pd.Series(key).map(m).fillna(fleet).to_numpy(dtype=float)


def band_stats(y, lo, hi, dev, keys: dict[str, np.ndarray] | None = None) -> dict:
    inside = ((y >= lo) & (y <= hi)).astype(float)
    names, n, s, _ = per_sign(dev, i=inside)
    alpha = 1 - NOMINAL
    wis = (hi - lo) + 2 / alpha * np.maximum(lo - y, 0) + 2 / alpha * np.maximum(y - hi, 0)
    out = {"coverage": float(inside.mean()), "width_median": float(np.median(hi - lo)),
           "interval_score": float(wis.mean()), "n": int(len(y))}
    if len(names) >= 5:
        d = np.random.default_rng(SEED).integers(0, len(names), size=(BOOT, len(names)))
        out["coverage_ci"] = ci(s["i"][d].sum(1) / n[d].sum(1))
    for name, k in (keys or {}).items():
        out.setdefault("by", {})[name] = {str(v): {"coverage": float(inside[k == v].mean()),
                                                   "width_median": float(np.median((hi - lo)[k == v])),
                                                   "n": int((k == v).sum())}
                                          for v in sorted(pd.unique(k)) if (k == v).sum() >= 200}
    return out


def rows(idx: np.ndarray, y: str):
    d = F.iloc[idx]
    return d, d[y].to_numpy(dtype=float), d["dev"].to_numpy()


def drop(feats: list[str], group: str) -> list[str]:
    return [c for c in feats if c not in GROUPS[group]]


def variants(feats: list[str]) -> dict[str, dict]:
    """The runs that show what each input group adds, for one chosen input list."""
    v = {f"− {g}": {"feats": drop(feats, g)} for g in GROUPS if set(GROUPS[g]) & set(feats)}
    if set(LAGS) & set(feats):
        v["− lags"] = {"feats": [c for c in feats if c not in LAGS]}
    v["time only"] = {"feats": C.TIME}
    v["+ sign identity"] = {"feats": feats + ["device"]}
    v["training not cleaned"] = {"feats": feats, "clean": False}
    return v


def family_of(y: str) -> str:
    return "power" if y in C.POWER else "thermal"


# ---------------------------------------------------------------- stage: dev

def dev() -> None:
    t0 = time.time()
    prepare(before=C.TEST)
    assert F["time"].max() < C.TEST
    t = F["time"]
    MASKS.update(mar_may=(t < JUNE).to_numpy(), mar_jun=(t < C.CAL).to_numpy())
    june, july = ((t >= JUNE) & (t < C.CAL)).to_numpy(), (t >= C.CAL).to_numpy()
    s_june = signs_with(MASKS["mar_may"], june)
    s_july = signs_with(MASKS["mar_jun"], july)
    s_band = signs_with(MASKS["mar_may"], july)
    MASKS["june"] = june & F["dev"].isin(s_june).to_numpy()
    MASKS["july"] = july & F["dev"].isin(s_july).to_numpy()
    MASKS["band"] = (june | july) & F["dev"].isin(s_band).to_numpy()
    print(f"dev: {len(F):,} rows to {F['time'].max()}; signs scored on June {len(s_june)}, on July {len(s_july)}", flush=True)

    # 1. choose inputs and settings on June, fit on March-May
    jobs = Jobs()
    for y in C.TARGETS:
        for i_name, feats in INPUT_SETS.items():
            for s_name, params in SETTINGS.items():
                jobs.add(target=y, inputs=i_name, setting=s_name, feats=feats, params=params, train="mar_may", pred="june")
    preds, _ = run_jobs(jobs, "June selection")
    sel = {y: {} for y in C.TARGETS}
    for j in jobs:
        idx, p = preds[j["id"]]
        _, yt, dv = rows(idx, j["target"])
        b = baseline(j["target"], train_rows(j["target"], "mar_may"), idx)
        m = score(yt, p, b, dv)
        sel[j["target"]][f"{j['inputs']} | {j['setting']}"] = {k: m[k] for k in ("n", "signs", "mae", "mae_baseline", "skill")}
    chosen = {}
    order = [f"{i} | {s}" for i in INPUT_SETS for s in sorted(SETTINGS, key=lambda k: (SETTINGS[k]["num_leaves"], SETTINGS[k]["n_estimators"]))]
    for fam, targets in FAMILY.items():
        mean = {k: float(np.mean([sel[y][k]["skill"] for y in targets])) for k in order}
        best = max(mean.values())
        pick = next(k for k in order if mean[k] >= best - TIE)   # the simplest within a tie of the best
        i_name, s_name = pick.split(" | ")
        chosen[fam] = {"inputs": i_name, "setting": s_name, "features": INPUT_SETS[i_name], "params": SETTINGS[s_name],
                       "june_mean_skill": mean[pick], "june_best_mean_skill": best, "june_mean_skill_all": mean}
        print(f"  {fam}: {pick}  (mean June skill {mean[pick]:.3f}, best {best:.3f})", flush=True)

    # 2. recheck on July (fit March-June) with what each group adds; quantile fits for the bands
    jobs = Jobs()
    for y in C.TARGETS:
        c = chosen[family_of(y)]
        jobs.add(target=y, name="main", feats=c["features"], params=c["params"], train="mar_jun", pred="july")
        for name, v in variants(c["features"]).items():
            jobs.add(target=y, name=name, params=c["params"], train="mar_jun", pred="july", **v)
        for i_name, feats in INPUT_SETS.items():
            if i_name != c["inputs"]:
                jobs.add(target=y, name=f"inputs: {i_name}", feats=feats, params=c["params"], train="mar_jun", pred="july")
        for a in (0.05, 0.95):
            jobs.add(target=y, name=f"q{a}", alpha=a, feats=c["features"], params=c["params"], train="mar_may", pred="band")
    preds, _ = run_jobs(jobs, "July recheck")
    recheck, bands = {}, {}
    for y in C.TARGETS:
        mine = {j["name"]: preds[j["id"]] for j in jobs if j["target"] == y}
        idx, p_main = mine["main"]
        _, yt, dv = rows(idx, y)
        b = baseline(y, train_rows(y, "mar_jun"), idx)
        recheck[y] = {"main": score(yt, p_main, b, dv) | {"recent_baseline": vs_recent(y, train_rows(y, "mar_jun"), idx, yt, p_main, dv)}}
        for name, (jidx, p) in mine.items():
            if name != "main" and not name.startswith("q0"):
                assert np.array_equal(jidx, idx)
                recheck[y][name] = score(yt, p, b, dv) | change(yt, p, p_main, dv)
        (bidx, q05), (_, q95) = mine["q0.05"], mine["q0.95"]
        d, yb, db = rows(bidx, y)
        lo, hi = np.minimum(q05, q95), np.maximum(q05, q95)
        is_cal = (d["time"] < C.CAL).to_numpy()
        s = np.maximum(lo - yb, yb - hi)
        bands[y] = {"no calibration": band_stats(yb[~is_cal], lo[~is_cal], hi[~is_cal], db[~is_cal])}
        for v in BANDS:
            q = margins(s, is_cal, db, d["config"].to_numpy(), v)[~is_cal]
            bands[y][v] = band_stats(yb[~is_cal], lo[~is_cal] - q, hi[~is_cal] + q, db[~is_cal])
    miss = {v: float(np.mean([abs(bands[y][v]["coverage"] - NOMINAL) for y in C.TARGETS])) for v in BANDS}
    band_pick = next(v for v in BANDS if miss[v] <= min(miss.values()) + TIE)
    print(f"  bands: {band_pick}  (mean |coverage − 0.90| on July: " + ", ".join(f"{v} {m:.3f}" for v, m in miss.items()) + ")", flush=True)

    rule = ("Per target family, the inputs and settings with the highest mean skill on June (fit March-May); within "
            f"{TIE} of the best, the simplest (fewer inputs, then fewer leaves, then fewer trees). Bands: the calibration "
            f"with the smallest mean |coverage - 0.90| on July (fit March-May, calibrate June); within {TIE}, the simplest.")
    locked = {"locked_at": datetime.now(timezone.utc).isoformat(), "seed": SEED, "rule": rule,
              "excluded_agencies": C.EXCLUDED_AGENCIES,
              "data_until": str(F["time"].max()), "min_hours": MIN_HOURS, "nominal_coverage": NOMINAL,
              "family": {fam: {k: c[k] for k in ("inputs", "setting", "features", "params")} for fam, c in chosen.items()},
              "bands": band_pick,
              "final_windows": {"fit": ["2026-03-08", "2026-07-01"], "calibrate": ["2026-07-01", "2026-08-01"],
                                "test": ["2026-08-01", "2026-10-07"]}}
    res = {"generated": datetime.now(timezone.utc).isoformat(), "data_until": str(F["time"].max()),
           "signs": {"june": len(s_june), "july": len(s_july), "bands": len(s_band)},
           "label": C.LABEL, "unit": C.UNIT, "selection_june": sel, "chosen": chosen, "recheck_july": recheck,
           "bands_july": bands, "bands_miss": miss, "bands_chosen": band_pick, "rule": rule,
           "runtime_s": round(time.time() - t0, 1)}
    (C.OUT / "rq1_dev.json").write_text(json.dumps(C.clean(res), allow_nan=False, indent=1))
    (C.OUT / "rq1_locked.json").write_text(json.dumps(C.clean(locked), allow_nan=False, indent=1))
    print(f"wrote rq1_dev.json and rq1_locked.json in {time.time() - t0:.0f}s\n", flush=True)
    for y in C.TARGETS:
        m = recheck[y]["main"]
        print(f"{C.LABEL[y]:14s} July: MAE {m['mae']:.3f} {C.UNIT[y]} (baseline {m['mae_baseline']:.3f}, recent "
              f"{m['recent_baseline']['mae_baseline']:.3f}), skill {m['skill']:.3f} (recent {m['recent_baseline']['skill']:.3f}); "
              + ", ".join(f"{k} {v['mae_change']:+.3f}" for k, v in recheck[y].items() if k != "main")
              + f"; coverage {bands[y][band_pick]['coverage']:.3f}")


# ---------------------------------------------------------------- stage: test

def test() -> None:
    t0 = time.time()
    out_path = C.OUT / "rq1_test.json"
    if out_path.exists():
        sys.exit("analysis/out/rq1_test.json exists. The test runs once; a second run is a deviation to record in "
                 "protocol/PROTOCOL.md before the file is moved away.")
    locked_text = (C.OUT / "rq1_locked.json").read_text()
    locked = json.loads(locked_text)
    prepare()
    t = F["time"]
    fit, cal, tst = (t < C.CAL).to_numpy(), ((t >= C.CAL) & (t < C.TEST)).to_numpy(), (t >= C.TEST).to_numpy()
    s_time = signs_with(fit, tst)             # the same signs, later
    s_new = signs_with(tst)                   # every sign with a week of test data, scored as if never seen
    in_time, in_new = F["dev"].isin(s_time).to_numpy(), F["dev"].isin(s_new).to_numpy()
    MASKS.update(fit=fit, t_test=tst & in_time, t_band=(cal | tst) & in_time, n_test=tst & in_new)
    print(f"test: {len(F):,} rows; signs scored over time {len(s_time)}, as unseen {len(s_new)}", flush=True)

    def cfg(y):
        c = locked["family"][family_of(y)]
        return c["features"], c["params"]

    jobs = Jobs()
    for y in C.TARGETS:
        feats, params = cfg(y)
        jobs.add(exp="time", name="main", target=y, feats=feats, params=params, train="fit", pred="t_test")
        for name, v in variants(feats).items():
            jobs.add(exp="time", name=name, target=y, params=params, train="fit", pred="t_test", **v)
        for a in (0.05, 0.95):
            jobs.add(exp="band", name=f"q{a}", alpha=a, target=y, feats=feats, params=params, train="fit", pred="t_band")

    # unseen signs: 5 folds over all signs; train on the other signs before July, score the fold from August
    signs = np.array(sorted(F["dev"].unique()))
    fold = np.random.default_rng(SEED).permutation(len(signs)) % 5
    for k in range(5):
        held = F["dev"].isin(signs[fold == k]).to_numpy()
        MASKS[f"new_fit{k}"], MASKS[f"new_test{k}"] = fit & ~held, MASKS["n_test"] & held
        for y in C.TARGETS:
            feats, params = cfg(y)
            jobs.add(exp="new", fold=k, target=y, feats=feats, params=params, train=f"new_fit{k}", pred=f"new_test{k}")

    # unseen hardware or site: hold out every sign with that value; train on the rest before July
    held_out = {"56 V supply": ("v56", 1), "12 V supply": ("v56", 0), "solar": ("has_mppt", 1), "no solar": ("has_mppt", 0),
                "enclosure sensor": ("has_env", 1), "no enclosure sensor": ("has_env", 0)}
    n_ag = F.loc[MASKS["n_test"]].groupby(F["agency"].astype(str))["dev"].nunique()
    for a in sorted(n_ag[n_ag >= 3].index):
        held_out[f"site {a}"] = ("agency", a)
    for g, (col, val) in held_out.items():
        held = (F[col].astype(str) == str(val)).to_numpy()
        MASKS[f"hw_fit:{g}"], MASKS[f"hw_test:{g}"] = fit & ~held, MASKS["n_test"] & held
        for y in C.TARGETS:
            feats, params = cfg(y)
            jobs.add(exp="hw", group=g, target=y, feats=feats, params=params, train=f"hw_fit:{g}", pred=f"hw_test:{g}")

    preds, imps = run_jobs(jobs, "test")

    assert locked.get("excluded_agencies") == C.EXCLUDED_AGENCIES, "the settings were locked on other devices"
    res = {"generated": datetime.now(timezone.utc).isoformat(), "locked": locked, "run": RUN,
           "locked_sha256": hashlib.sha256(locked_text.encode()).hexdigest(),
           "git": subprocess.run(["git", "describe", "--always", "--dirty"], cwd=C.ROOT, capture_output=True, text=True).stdout.strip(),
           "windows": locked["final_windows"], "signs": {"over_time": len(s_time), "unseen": len(s_new)},
           "targets": C.TARGETS, "label": C.LABEL, "unit": C.UNIT, "useful_proposed": {"input_power_nmae": 0.05,
           "temperature_mae_c": 2.0, "coverage": [0.85, 0.95]}}
    over_time, new, hw, keep = {}, {}, {g: {} for g in held_out}, []
    for y in C.TARGETS:
        # ---- the same signs, later
        mine = {j["name"]: j["id"] for j in jobs if j["target"] == y and j["exp"] in ("time", "band")}
        idx, p = preds[mine["main"]]
        d, yt, dv = rows(idx, y)
        b = baseline(y, train_rows(y, "fit"), idx)
        keys = {"month": d["time"].dt.strftime("%Y-%m").to_numpy(),
                "city": np.where(d["wx_temp_c"].notna(), "with city weather", "no city"),
                "supply": np.where(d["v56"] == 1, "56 V", "12 V"),
                "solar": np.where(d["has_mppt"] == 1, "solar", "no solar"),
                "hours": np.where(d[f"flag_{y}"], "flagged by anomaly.py", "other hours")}
        r = {"main": score(yt, p, b, dv) | {"recent_baseline": vs_recent(y, train_rows(y, "fit"), idx, yt, p, dv)},
             "by": by_group(keys, yt, p, b, dv)}
        if y in C.THERMAL:  # the hottest 10 % of each sign's hours
            hot = yt > pd.Series(yt).groupby(dv).transform(lambda v: v.quantile(0.9)).to_numpy()
            r["hottest_10pct"] = score(yt[hot], p[hot], b[hot], dv[hot])
        r["variants"] = {}
        for name, jid in mine.items():
            if name != "main" and not name.startswith("q0"):
                jidx, pv = preds[jid]
                assert np.array_equal(jidx, idx)
                r["variants"][name] = score(yt, pv, b, dv) | change(yt, pv, p, dv)
        imp = imps[mine["main"]]
        owner = {c: g for g, cols in GROUPS.items() for c in cols} | {c: "time" for c in C.TIME}
        r["gain_share"] = {g: round(sum(v for c, v in imp.items() if owner[c] == g), 4) for g in ["time", *GROUPS]}
        # 90 % bands: quantile fits on March-June, margin from July
        (bidx, q05), (_, q95) = preds[mine["q0.05"]], preds[mine["q0.95"]]
        bd, yb, db = rows(bidx, y)
        is_cal = (bd["time"] < C.TEST).to_numpy()
        assert np.array_equal(bidx[~is_cal], idx)
        lo, hi = np.minimum(q05, q95), np.maximum(q05, q95)
        s = np.maximum(lo - yb, yb - hi)
        bkeys = {k: keys[k] for k in ("month", "supply", "hours")}
        r["bands"] = {"no calibration": band_stats(yt, lo[~is_cal], hi[~is_cal], dv, bkeys)}
        for v in BANDS:
            q = margins(s, is_cal, db, bd["config"].to_numpy(), v)[~is_cal]
            r["bands"][v] = band_stats(yt, lo[~is_cal] - q, hi[~is_cal] + q, dv, bkeys)
            if v == locked["bands"]:
                keep.append(pd.DataFrame({"target": y, "device": dv, "time": pd.DatetimeIndex(d["time"]), "y": yt, "pred": p,
                                          "baseline": b, "lo": lo[~is_cal] - q, "hi": hi[~is_cal] + q,
                                          "flagged": d[f"flag_{y}"].to_numpy()}))
        names, n, sums, _ = per_sign(dv, a=np.abs(yt - p), b=np.abs(yt - b))
        r["per_sign"] = {k: [int(n[i]), round(sums["a"][i] / n[i], 4), round(sums["b"][i] / n[i], 4)] for i, k in enumerate(names)}
        over_time[y] = r
        seen = pd.Series(p, index=idx)

        # ---- unseen signs
        parts = [preds[j["id"]] for j in jobs if j["target"] == y and j["exp"] == "new"]
        nidx, pn = np.concatenate([a for a, _ in parts]), np.concatenate([a for _, a in parts])
        o = np.argsort(nidx)
        nidx, pn = nidx[o], pn[o]
        nd, yn, dn = rows(nidx, y)
        b_fleet, b_agency = np.empty(len(nidx)), np.empty(len(nidx))
        for k in range(5):
            m = MASKS[f"new_test{k}"][nidx]
            tr = train_rows(y, f"new_fit{k}")
            b_fleet[m], b_agency[m] = baseline(y, tr, nidx[m], None), baseline(y, tr, nidx[m], "agency")
        r = {"main": score(yn, pn, b_fleet, dn), "skill_vs_agency_hour": score(yn, pn, b_agency, dn)["skill"],
             "by": by_group({"supply": np.where(nd["v56"] == 1, "56 V", "12 V"),
                             "first seen": np.where(nd["dev"].isin(s_time), "also in training period", "only after June")},
                            yn, pn, b_fleet, dn)}
        both = np.isin(nidx, idx)           # the same hours, predicted with and without the sign in training
        ps = seen.reindex(nidx[both]).to_numpy()
        r["same_hours"] = {"n": int(both.sum()), "mae_unseen": float(np.abs(yn[both] - pn[both]).mean()),
                           "mae_seen": float(np.abs(yn[both] - ps).mean())} | change(yn[both], pn[both], ps, dn[both])
        r["same_hours"]["ratio"] = r["same_hours"]["mae_unseen"] / r["same_hours"]["mae_seen"]
        new[y] = r
        unseen = pd.Series(pn, index=nidx)

        # ---- unseen hardware or site
        for g in held_out:
            jid = next(j["id"] for j in jobs if j["target"] == y and j["exp"] == "hw" and j["group"] == g)
            hidx, ph = preds[jid]
            if len(hidx) < 200:
                continue
            _, yh, dh = rows(hidx, y)
            bh = baseline(y, train_rows(y, f"hw_fit:{g}"), hidx, None)
            pu = unseen.reindex(hidx).to_numpy()
            r = score(yh, ph, bh, dh) | change(yh, ph, pu, dh)
            r["mae_as_unseen_sign"] = float(np.abs(yh - pu).mean())      # same hours, trained with that hardware or site
            r["gap"] = r["mae"] / r["mae_as_unseen_sign"]
            hw[g][y] = r

    res.update(over_time=over_time, unseen_signs=new, unseen_hardware=hw)
    lk = locked["bands"]
    res["useful"] = {
        "input_power_nmae": over_time["input_power_watts"]["main"]["nmae"],
        "temperature_mae_c": {y: over_time[y]["main"]["mae"] for y in C.THERMAL},
        "coverage": {y: over_time[y]["bands"][lk]["coverage"] for y in C.TARGETS}}
    res["runtime_s"] = round(time.time() - t0, 1)
    out_path.write_text(json.dumps(C.clean(res), allow_nan=False, indent=1))
    pd.concat(keep).to_parquet(C.OUT / "rq1_test_predictions.parquet", index=False)
    print(f"wrote rq1_test.json in {time.time() - t0:.0f}s\n", flush=True)
    for y in C.TARGETS:
        m, n_ = over_time[y]["main"], new[y]["main"]
        print(f"{C.LABEL[y]:14s} over time: MAE {m['mae']:.3f} {C.UNIT[y]} (baseline {m['mae_baseline']:.3f}), skill {m['skill']:.3f}, "
              f"coverage {over_time[y]['bands'][lk]['coverage']:.3f} | unseen signs: MAE {n_['mae']:.3f}, skill {n_['skill']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["dev", "test"])
    {"dev": dev, "test": test}[ap.parse_args().stage]()

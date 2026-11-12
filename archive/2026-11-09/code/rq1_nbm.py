#!/usr/bin/env python
"""RQ1 — normal-behaviour modelling of power and thermal behaviour.

Question: to what extent can a model conditioned on workload and environmental context predict the expected
power and thermal behaviour of the devices across differing hardware configurations and operating conditions?

Targets (hourly device means): input / Orin / CPU+GPU power, CPU / GPU / board temperature. Inputs never include
the target's own past, so the models describe expected behaviour rather than track the series.

Experiments
  S1 temporal hold-out: fit before 2026-08-01, test from 2026-08-01 on devices seen in training.
     * baselines: fleet mean, device mean, device × hour-of-day median, persistence (lag 1 h, reference only)
     * ridge on the full context; LightGBM feature ladder time → +hardware → +external env → +local env
       → +workload → +device identity; drop-one ablations; per-configuration specialists; quantile models
       (5 % / 95 %) for interval coverage; thermal targets also with measured power as the heat source
     * errors stratified by supply voltage, solar, agency, DeepStream state, ambient band and month
  S2 leave-device-out: 5 device folds over the whole history, unseen devices.
  S3 leave-configuration-out: train on one hardware configuration or agency, test on the others.

Output: analysis/out/rq1.json (+ rq1_examples.parquet for the report's example traces).
Usage: python -I analysis/rq1_nbm.py
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script directory
import rq_common as C  # noqa: E402

F: pd.DataFrame  # set before forking workers
MASKS: dict[str, np.ndarray] = {}

LADDER = {
    "time": C.TIME,
    "+hardware": C.TIME + C.HW,
    "+external env": C.TIME + C.HW + C.ENV_EXT,
    "+local env": C.TIME + C.HW + C.ENV_EXT + C.ENV_LOCAL,
    "+workload": C.TIME + C.HW + C.ENV_EXT + C.ENV_LOCAL + C.WORKLOAD,
    "+device id": C.TIME + C.HW + C.ENV_EXT + C.ENV_LOCAL + C.WORKLOAD + ["device"],
}
FULL = LADDER["+workload"]
DROP = {
    "full − workload": [c for c in FULL if c not in C.WORKLOAD],
    "full − environment": [c for c in FULL if c not in C.ENV_EXT + C.ENV_LOCAL],
    "full − hardware": [c for c in FULL if c not in C.HW],
}
HEAT = {"cpu_temp_celsius": ["cpu_gpu_power_watts", "orin_power_watts"],
        "gpu_temp_celsius": ["cpu_gpu_power_watts", "orin_power_watts"],
        "board_temp_celsius": ["input_power_watts", "orin_power_watts"]}


# ---------------------------------------------------------------- model fitting (runs in workers)

def ridge_design(d: pd.DataFrame, feats: list[str], ref: pd.DataFrame | None = None):
    num = [c for c in feats if c not in C.CATEGORICAL]
    X = d[num].astype(float)
    miss = X.isna().astype(float).add_suffix("_na")
    X = X.fillna(ref["fill"] if ref is not None else X.median())
    hour = pd.get_dummies(d["time"].dt.hour, prefix="h").astype(float)
    cats = [pd.get_dummies(d[c].astype(str), prefix=c).astype(float) for c in feats if c in C.CATEGORICAL]
    M = pd.concat([X, miss, hour, *cats], axis=1)
    if ref is None:
        ref = {"fill": d[num].astype(float).median(), "cols": M.columns, "mu": M.mean(), "sd": M.std().replace(0, 1)}
    M = M.reindex(columns=ref["cols"], fill_value=0.0)
    return ((M - ref["mu"]) / ref["sd"]).fillna(0.0).to_numpy(), ref


def run_job(job: dict):
    t0 = time.time()
    tr, te, y = MASKS[job["train"]], MASKS[job["test"]], job["target"]
    tr = tr & F[y].notna().to_numpy()
    te = te & F[y].notna().to_numpy()
    if "filter" in job:  # e.g. per-configuration specialist
        col, val = job["filter"]
        tr = tr & (F[col].to_numpy() == val)
        te = te & (F[col].to_numpy() == val)
    dtr, dte = F.loc[tr], F.loc[te]
    feats = job["feats"]
    if job["model"] == "ridge":
        Xtr, ref = ridge_design(dtr, feats)
        Xte, _ = ridge_design(dte, feats, ref)
        m = Ridge(alpha=1.0).fit(Xtr, dtr[y].to_numpy())
        pred = m.predict(Xte)
        imp = None
    else:
        params = C.lgbm_params(**({"objective": "quantile", "alpha": job["alpha"]} if "alpha" in job else {}))
        m = lgb.LGBMRegressor(**params).fit(dtr[feats], dtr[y])
        pred = m.predict(dte[feats])
        imp = dict(zip(feats, (m.booster_.feature_importance("gain") /
                               max(m.booster_.feature_importance("gain").sum(), 1e-9)).round(4).tolist()))
    return job["id"], np.flatnonzero(te), pred.astype("float32"), imp, time.time() - t0


# ---------------------------------------------------------------- metrics

def metrics(y: np.ndarray, p: np.ndarray, dev: np.ndarray) -> dict:
    e = y - p
    s = pd.Series(y).groupby(dev).transform("mean").to_numpy()
    sst = ((y - y.mean()) ** 2).sum()
    ssw = ((y - s) ** 2).sum()
    sse = (e ** 2).sum()
    return {"n": int(len(y)), "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "r2": float(1 - sse / sst), "r2_within": float(1 - sse / ssw), "bias": float(e.mean())}


def strata(d: pd.DataFrame) -> dict[str, pd.Series]:
    amb = d["temperature"].fillna(d["peer_temperature"])
    return {
        "supply": np.where(d["v56"] == 1, "56 V", "12 V"),
        "solar": np.where(d["has_mppt"] == 1, "solar (MPPT)", "mains / no MPPT"),
        "agency": d["agency"].astype(str),
        "deepstream": np.select([d["deepstream_app_running"] >= 0.5, d["deepstream_app_running"].notna()],
                                ["DeepStream on", "DeepStream off"], "unknown"),
        "ambient": pd.cut(amb, [-50, 5, 15, 25, 35, 100], labels=["< 5 °C", "5–15 °C", "15–25 °C", "25–35 °C", "> 35 °C"])
                     .astype(str).replace("nan", "unknown"),
        "month": d["time"].dt.strftime("%Y-%m"),
    }


def stratified(d: pd.DataFrame, y: str, preds: dict[str, np.ndarray]) -> dict:
    out = {}
    for name, key in strata(d).items():
        g = pd.DataFrame({"k": np.asarray(key), "y": d[y].to_numpy(), "dev": d["device"].astype(str).to_numpy(),
                          **{m: p for m, p in preds.items()}})
        rows = {}
        for k, gg in g.groupby("k"):
            if len(gg) < 200:
                continue
            rows[k] = {"n": int(len(gg)), "devices": int(gg["dev"].nunique()),
                       **{m: float(np.abs(gg["y"] - gg[m]).mean()) for m in preds}}
        out[name] = rows
    return out


# ---------------------------------------------------------------- experiment definitions

def main() -> None:
    global F
    t0 = time.time()
    F = C.load()
    for y in C.TARGETS:  # on the complete hourly grid, before rows without targets are dropped
        F[f"lag1_{y}"] = F.groupby("device", observed=True)[y].shift(1)
    F = F[F[C.TARGETS].notna().any(axis=1)].reset_index(drop=True)
    dev = F["device"].astype(str)
    fit_h = F[F["time"] < C.TEST].groupby(dev)["input_power_watts"].count()
    test_h = F[F["time"] >= C.TEST].groupby(dev)["input_power_watts"].count()
    s1_devs = sorted(set(fit_h[fit_h >= 336].index) & set(test_h[test_h >= 168].index))
    MASKS["s1_train"] = (F["time"] < C.TEST).to_numpy()
    MASKS["s1_test"] = ((F["time"] >= C.TEST) & dev.isin(s1_devs)).to_numpy()
    MASKS["cqr_train"] = (F["time"] < C.CAL).to_numpy()
    MASKS["cqr_pred"] = (((F["time"] >= C.CAL) & (F["time"] < C.TEST) & dev.isin(s1_devs)) | MASKS["s1_test"]).to_numpy()
    print(f"loaded {len(F):,} rows; S1 devices {len(s1_devs)}", flush=True)

    jobs = []

    def add(**j):
        j["id"] = len(jobs)
        jobs.append(j)

    for y in C.TARGETS:
        for name, feats in LADDER.items():
            add(exp="s1", target=y, model="lgbm", name=name, feats=feats, train="s1_train", test="s1_test")
        for name, feats in DROP.items():
            add(exp="s1", target=y, model="lgbm", name=name, feats=feats, train="s1_train", test="s1_test")
        add(exp="s1", target=y, model="ridge", name="ridge (full)", feats=FULL, train="s1_train", test="s1_test")
        for v in (0, 1):
            add(exp="s1", target=y, model="lgbm", name="specialist", feats=FULL, train="s1_train", test="s1_test",
                filter=("v56", v))
        for a in (0.05, 0.5, 0.95):
            add(exp="s1", target=y, model="lgbm", name=f"q{int(a * 100):02d}", feats=LADDER["+device id"],
                train="s1_train", test="s1_test", alpha=a)
        if y in HEAT:
            add(exp="s1", target=y, model="lgbm", name="+measured power", feats=LADDER["+device id"] + HEAT[y],
                train="s1_train", test="s1_test")
        for a in (0.05, 0.95):  # conformalised quantile regression: fit before July, calibrate on July
            add(exp="cqr", target=y, model="lgbm", name=f"cqr{int(a * 100):02d}", feats=LADDER["+device id"],
                train="cqr_train", test="cqr_pred", alpha=a)

    # S2: leave-device-out over the whole history
    devices = np.array(sorted(dev.unique()))
    gkf = GroupKFold(n_splits=5, shuffle=True, random_state=0)
    for k, (_, te_idx) in enumerate(gkf.split(devices, groups=devices)):
        te_devs = set(devices[te_idx])
        MASKS[f"s2_test{k}"] = dev.isin(te_devs).to_numpy()
        MASKS[f"s2_train{k}"] = ~MASKS[f"s2_test{k}"]
        for y in C.TARGETS:
            add(exp="s2", fold=k, target=y, model="lgbm", name="lgbm (full)", feats=FULL,
                train=f"s2_train{k}", test=f"s2_test{k}")
            add(exp="s2", fold=k, target=y, model="lgbm", name="lgbm − hardware", feats=DROP["full − hardware"],
                train=f"s2_train{k}", test=f"s2_test{k}")
            add(exp="s2", fold=k, target=y, model="ridge", name="ridge (full)", feats=FULL,
                train=f"s2_train{k}", test=f"s2_test{k}")

    # S3: leave-configuration-out (train on one group, test on the held-out group)
    groups = {"12 V → 56 V": ("v56", 1), "56 V → 12 V": ("v56", 0),
              "no solar → solar": ("has_mppt", 1), "solar → no solar": ("has_mppt", 0)}
    agencies = F.groupby("agency", observed=True)["device"].nunique()
    for a in agencies[agencies >= 3].index:
        groups[f"agency out: {a}"] = ("agency", a)
    for gname, (col, val) in groups.items():
        held = (F[col] == val).to_numpy()
        MASKS[f"s3_test:{gname}"], MASKS[f"s3_train:{gname}"] = held, ~held
        for y in C.TARGETS:
            for name, feats in (("lgbm (full)", FULL), ("lgbm − hardware", DROP["full − hardware"])):
                add(exp="s3", group=gname, target=y, model="lgbm", name=name, feats=feats,
                    train=f"s3_train:{gname}", test=f"s3_test:{gname}")

    print(f"{len(jobs)} model fits", flush=True)
    preds, imps = {}, {}
    with ProcessPoolExecutor(max_workers=12, mp_context=mp.get_context("fork")) as ex:
        for i, (jid, idx, p, imp, dt) in enumerate(ex.map(run_job, jobs)):
            preds[jid] = (idx, p)
            imps[jid] = imp
            if i % 25 == 0:
                print(f"  {i + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)

    res = {"generated": datetime.now(timezone.utc).isoformat(), "split": {"cal": str(C.CAL.date()), "test": str(C.TEST.date())},
           "s1_devices": len(s1_devs), "targets": C.TARGETS, "label": C.LABEL, "unit": C.UNIT,
           "ladder": list(LADDER), "drop": list(DROP), "feature_groups": {
               "time": C.TIME, "hardware": C.HW, "external env": C.ENV_EXT, "local env": C.ENV_LOCAL, "workload": C.WORKLOAD}}

    # ---- S1
    te = MASKS["s1_test"]
    tr = MASKS["s1_train"]
    s1, s1_strata, s1_cov, s1_imp = {}, {}, {}, {}
    F["hour"] = F["time"].dt.hour
    for y in C.TARGETS:
        idx = np.flatnonzero(te & F[y].notna().to_numpy())
        d = F.iloc[idx]
        yt = d[y].to_numpy()
        dv = d["device"].astype(str).to_numpy()
        trd = F.loc[tr & F[y].notna().to_numpy()]
        base = {
            "fleet mean": np.full(len(d), trd[y].mean()),
            "device mean": d["device"].astype(str).map(trd.groupby(trd["device"].astype(str))[y].mean()).to_numpy(),
            "device × hour": pd.MultiIndex.from_arrays([d["device"].astype(str), d["hour"]]).map(
                trd.groupby([trd["device"].astype(str), "hour"])[y].median()).to_numpy(dtype=float),
            "persistence (lag 1 h)": F[f"lag1_{y}"].iloc[idx].to_numpy(),
        }
        allp = {}
        for k, p in base.items():
            p = np.where(np.isnan(p), np.nanmean(p), p)
            allp[k] = p
        for j in jobs:
            if j["exp"] == "s1" and j["target"] == y and j["name"] not in ("specialist",) and "alpha" not in j:
                jidx, p = preds[j["id"]]
                assert np.array_equal(jidx, idx), j
                allp[j["name"] if j["model"] == "lgbm" else j["name"]] = p
        spec = np.full(len(idx), np.nan)
        for j in jobs:
            if j["exp"] == "s1" and j["target"] == y and j["name"] == "specialist":
                jidx, p = preds[j["id"]]
                spec[np.searchsorted(idx, jidx)] = p
        allp["specialist per supply"] = spec
        s1[y] = {k: metrics(yt, p, dv) for k, p in allp.items()}
        s1_strata[y] = stratified(d, y, {k: allp[k] for k in ("device × hour", "+workload", "+device id")})
        q = {j["alpha"]: preds[j["id"]][1] for j in jobs if j["exp"] == "s1" and j["target"] == y and "alpha" in j}
        lo, hi = np.minimum(q[0.05], q[0.95]), np.maximum(q[0.05], q[0.95])
        inside = (yt >= lo) & (yt <= hi)
        cov_by = {}
        for name, key in strata(d).items():
            if name in ("supply", "solar", "deepstream", "ambient"):
                g = pd.DataFrame({"k": np.asarray(key), "in": inside, "w": hi - lo})
                cov_by[name] = {k: {"coverage": float(gg["in"].mean()), "width": float(gg["w"].median()), "n": int(len(gg))}
                                for k, gg in g.groupby("k") if len(gg) >= 200}
        s1_cov[y] = {"coverage": float(inside.mean()), "width_median": float(np.median(hi - lo)),
                     "q50": metrics(yt, q[0.5], dv), "by": cov_by,
                     "per_device_coverage": pd.Series(inside).groupby(dv).mean().round(3).to_dict()}
        # conformalised quantile regression (split conformal on July), global and per device (Mondrian)
        cq = {j["alpha"]: preds[j["id"]] for j in jobs if j["exp"] == "cqr" and j["target"] == y}
        cidx, c05 = cq[0.05]
        c95 = cq[0.95][1]
        lo_c, hi_c = np.minimum(c05, c95), np.maximum(c05, c95)
        cd = F.iloc[cidx]
        is_cal = (cd["time"] < C.TEST).to_numpy()
        score = np.maximum(lo_c - cd[y].to_numpy(), cd[y].to_numpy() - hi_c)
        n_cal = is_cal.sum()
        q_glob = np.quantile(score[is_cal], min(1, 0.9 * (1 + 1 / n_cal)))
        cdev = cd["device"].astype(str).to_numpy()
        q_dev = pd.Series(score[is_cal]).groupby(cdev[is_cal]).agg(
            lambda s_: np.quantile(s_, min(1, 0.9 * (1 + 1 / len(s_)))) if len(s_) >= 100 else np.nan)
        qd = pd.Series(cdev).map(q_dev).fillna(q_glob).to_numpy()
        tm = ~is_cal
        assert np.array_equal(cidx[tm], idx), y
        cqr = {}
        for nm, qq in (("conformal (global)", q_glob), ("conformal (per device)", qd)):
            qq = np.broadcast_to(qq, score.shape)[tm]
            l_, h_ = lo_c[tm] - qq, hi_c[tm] + qq
            ins = (yt >= l_) & (yt <= h_)
            byc = {}
            for name, key in strata(d).items():
                if name in ("supply", "solar", "deepstream", "ambient", "month"):
                    g = pd.DataFrame({"k": np.asarray(key), "in": ins, "w": h_ - l_})
                    byc[name] = {k: {"coverage": float(gg["in"].mean()), "width": float(gg["w"].median()), "n": int(len(gg))}
                                 for k, gg in g.groupby("k") if len(gg) >= 200}
            cqr[nm] = {"coverage": float(ins.mean()), "width_median": float(np.median(h_ - l_)), "by": byc,
                       "per_device_coverage": pd.Series(ins).groupby(dv).mean().round(3).to_dict()}
        s1_cov[y]["conformal"] = cqr
        full_job = next(j for j in jobs if j["exp"] == "s1" and j["target"] == y and j["name"] == "+workload")
        s1_imp[y] = imps[full_job["id"]]
        # per-device MAE of the main models (distribution across the fleet)
        s1[y]["_per_device"] = {k: pd.Series(np.abs(yt - allp[k])).groupby(dv).mean().round(4).to_dict()
                                for k in ("device × hour", "+workload", "+device id")}
    res.update(s1=s1, s1_strata=s1_strata, s1_coverage=s1_cov, s1_importance=s1_imp)

    # example traces for the report: two devices (12 V and 56 V), 10 days of test data, median + 90 % band
    ex_rows = []
    for v in (0, 1):
        cand = [d_ for d_ in s1_devs if int(F.loc[F["device"].astype(str) == d_, "v56"].iloc[0]) == v]
        if not cand:
            continue
        devx = max(cand, key=lambda d_: test_h.get(d_, 0))
        for y in C.TARGETS:
            idx = np.flatnonzero(te & F[y].notna().to_numpy())
            q = {j["alpha"]: preds[j["id"]][1] for j in jobs if j["exp"] == "s1" and j["target"] == y and "alpha" in j}
            pfull = next(preds[j["id"]][1] for j in jobs if j["exp"] == "s1" and j["target"] == y and j["name"] == "+device id")
            d = F.iloc[idx][["device", "time", y]].assign(q05=q[0.05], q50=q[0.5], q95=q[0.95], pred=pfull)
            d = d[d["device"].astype(str) == devx]
            d = d[(d["time"] >= C.TEST + pd.Timedelta(days=14)) & (d["time"] < C.TEST + pd.Timedelta(days=24))]
            ex_rows.append(d.rename(columns={y: "y"}).assign(target=y, device=devx))
    pd.concat(ex_rows).to_parquet(C.OUT / "rq1_examples.parquet", index=False)

    # ---- S2
    s2 = {}
    for y in C.TARGETS:
        s2[y] = {}
        for name in ("lgbm (full)", "lgbm − hardware", "ridge (full)"):
            idx_all, p_all = [], []
            for j in jobs:
                if j["exp"] == "s2" and j["target"] == y and j["name"] == name:
                    idx_all.append(preds[j["id"]][0]); p_all.append(preds[j["id"]][1])
            idx = np.concatenate(idx_all); p = np.concatenate(p_all)
            d = F.iloc[idx]
            s2[y][name] = metrics(d[y].to_numpy(), p, d["device"].astype(str).to_numpy())
            s2[y][name]["per_device_mae"] = pd.Series(np.abs(d[y].to_numpy() - p)).groupby(
                d["device"].astype(str).to_numpy()).mean().round(4).to_dict()
        # baseline for unseen devices: agency × hour median from the other folds
        idx_all, p_all = [], []
        for k in range(5):
            trm = MASKS[f"s2_train{k}"] & F[y].notna().to_numpy()
            tem = np.flatnonzero(MASKS[f"s2_test{k}"] & F[y].notna().to_numpy())
            trd, ted = F.loc[trm], F.iloc[tem]
            prof = trd.groupby([trd["agency"].astype(str), "hour"])[y].median()
            p = pd.MultiIndex.from_arrays([ted["agency"].astype(str), ted["hour"]]).map(prof).to_numpy(dtype=float)
            p = np.where(np.isnan(p), trd[y].median(), p)
            idx_all.append(tem); p_all.append(p)
        idx = np.concatenate(idx_all); p = np.concatenate(p_all)
        d = F.iloc[idx]
        s2[y]["agency × hour"] = metrics(d[y].to_numpy(), p, d["device"].astype(str).to_numpy())
    res["s2"] = s2

    # ---- S3
    s3 = {}
    for gname in groups:
        s3[gname] = {}
        for y in C.TARGETS:
            row = {}
            for j in jobs:
                if j["exp"] == "s3" and j["group"] == gname and j["target"] == y:
                    idx, p = preds[j["id"]]
                    d = F.iloc[idx]
                    row[j["name"]] = metrics(d[y].to_numpy(), p, d["device"].astype(str).to_numpy())
                    # same devices, in-distribution (leave-device-out) error for the transfer gap
                    ldo = s2[y]["lgbm (full)"]["per_device_mae"]
                    devs = d["device"].astype(str).unique()
                    row["devices"] = int(len(devs))
                    w = d["device"].astype(str).value_counts()
                    row["ldo_mae_same_devices"] = float(sum(ldo.get(k, np.nan) * w[k] for k in devs if k in ldo) /
                                                        max(sum(w[k] for k in devs if k in ldo), 1))
            s3[gname][y] = row
    res["s3"] = s3
    res["runtime_s"] = round(time.time() - t0, 1)
    (C.OUT / "rq1.json").write_text(json.dumps(C.clean(res), allow_nan=False))
    print(f"wrote rq1.json in {time.time() - t0:.0f}s", flush=True)

    # console summary
    for y in C.TARGETS:
        print(f"\n{y}  (S1, MAE / within-device R²)")
        for k, m in s1[y].items():
            if not k.startswith("_"):
                print(f"  {k:24s} {m['mae']:7.3f}  {m['r2_within']:6.3f}")
        print(f"  coverage 90%: raw {s1_cov[y]['coverage']:.3f}, "
              + ", ".join(f"{k} {v['coverage']:.3f}" for k, v in s1_cov[y]["conformal"].items()))


if __name__ == "__main__":
    main()

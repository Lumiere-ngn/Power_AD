#!/usr/bin/env python
"""RQ2 — does a hand-specified dependency graph detect energy anomalies more accurately than non-graph and
learned-graph detectors?

Ten power / thermal nodes (hourly device means). Every residual detector predicts each node from the same
exogenous context (time, hardware, environment, workload, device id; the RQ1 model) plus the node's *parents*
in a dependency graph, with the same LightGBM settings; only the graph differs. Residuals are scaled per
device and node on a July calibration month, and the system score of an hour is the largest |z| over nodes.

Detectors
  non-graph      seasonal z (device × hour-of-day median, univariate), context NBM (graph with no edges),
                 Isolation Forest and PCA (squared prediction error) on the seasonal z vector
  hand graph     physics / domain DAG (HAND below)
  learned graph  top-k correlation (GDN-style neighbour selection), Lasso neighbourhood selection,
                 pooled Granger causality, NOTEARS linear DAG, complete graph (every other node)

Ground truth: there are no labelled faults, so energy faults are injected into the test period (from
2026-08-01) with physically consistent propagation; coefficients are estimated from the training data.
  F1 parasitic load      extra draw behind the input meter: input power, current, energy (+ board heat)
  F2 compute runaway     CPU+GPU power rises without workload change, propagating to Orin, heartbeat and
                         input power, current, energy and temperatures
  F3 energy-counter drift  the on-device energy integral drifts away from input power (ramp)
  F4 stuck power sensor  input power frozen at the onset value while current and voltage move on
  F5 supply sag          input voltage drops, current rises to hold power, small conversion loss
  F6 cooling fault       CPU / GPU temperature up without any power change (thermal control case)
Three severities each. Scores: point AUROC / AUPRC, event recall at a 1 % hourly false-alarm rate
(threshold = 99th percentile of the system score on the clean test period), detection delay, and root-cause
localisation (node with the largest mean |z| over the episode). Plus a real-data check: hours where the
reported input power disagrees with voltage × current by more than 2 %.

Output: analysis/out/rq2.json, rq2_episodes.parquet, rq2_examples.parquet.
Usage: python -I analysis/rq2_graph.py
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
import scipy.linalg as sla
import scipy.optimize as sopt
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LassoCV
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script directory
import rq_common as C  # noqa: E402

NODES = ["cpu_gpu_power_watts", "orin_power_watts", "power", "input_power_watts", "input_voltage_volts",
         "input_current_amperes", "input_energy_wh_total_rate", "cpu_temp_celsius", "gpu_temp_celsius",
         "board_temp_celsius"]
NI = {n: i for i, n in enumerate(NODES)}
# hand-specified DAG: parent -> child, each edge a physical or measurement relation
HAND = {
    "cpu_gpu_power_watts": [],                                         # driven by workload (context)
    "orin_power_watts": ["cpu_gpu_power_watts"],                       # CPU+GPU rail is part of the module rail
    "power": ["orin_power_watts"],                                     # heartbeat logs the same rail
    "input_power_watts": ["orin_power_watts"],                         # module + display / modem / losses
    "input_voltage_volts": [],                                         # supply / battery, exogenous
    "input_current_amperes": ["input_power_watts", "input_voltage_volts"],  # I = P / V
    "input_energy_wh_total_rate": ["input_power_watts"],               # on-device integral of input power
    "cpu_temp_celsius": ["cpu_gpu_power_watts"],                       # heat source (+ ambient in context)
    "gpu_temp_celsius": ["cpu_gpu_power_watts", "cpu_temp_celsius"],   # same die
    "board_temp_celsius": ["input_power_watts"],                       # input stage heat (+ ambient)
}
HAND_WHY = {
    ("cpu_gpu_power_watts", "orin_power_watts"): "CPU+GPU rail is part of the Orin module rail",
    ("orin_power_watts", "power"): "heartbeat power logs the same rail",
    ("orin_power_watts", "input_power_watts"): "module draw plus display, modem and conversion losses",
    ("input_power_watts", "input_current_amperes"): "I = P / V",
    ("input_voltage_volts", "input_current_amperes"): "I = P / V",
    ("input_power_watts", "input_energy_wh_total_rate"): "energy counter integrates input power",
    ("cpu_gpu_power_watts", "cpu_temp_celsius"): "compute power heats the SoC",
    ("cpu_gpu_power_watts", "gpu_temp_celsius"): "compute power heats the SoC",
    ("cpu_temp_celsius", "gpu_temp_celsius"): "CPU and GPU share one die",
    ("input_power_watts", "board_temp_celsius"): "input-stage dissipation heats the board sensor",
}
CONTEXT = C.TIME + C.HW + C.ENV_EXT + C.ENV_LOCAL + C.WORKLOAD + ["device"]
FLOOR = {"input_voltage_volts": 0.02, "input_current_amperes": 0.005, "input_energy_wh_total_rate": 0.05,
         "cpu_temp_celsius": 0.3, "gpu_temp_celsius": 0.3, "board_temp_celsius": 0.3}  # others: 0.05 W
K_TOP = 2  # parents per node for the top-k learned graphs (the hand graph has 1.0 on average, at most 2)

FAULTS = {
    "F1": {"name": "Parasitic load", "root": {"input_power_watts", "input_current_amperes"},
           "sev": [0.05, 0.10, 0.20], "sev_label": ["+5 % input power", "+10 %", "+20 %"], "dur": (6, 24)},
    "F2": {"name": "Compute runaway", "root": {"cpu_gpu_power_watts"},
           "sev": [0.25, 0.50, 1.00], "sev_label": ["+25 % CPU+GPU power", "+50 %", "+100 %"], "dur": (6, 24)},
    "F3": {"name": "Energy-counter drift", "root": {"input_energy_wh_total_rate"},
           "sev": [0.05, 0.10, 0.20], "sev_label": ["ramp to 5 %", "to 10 %", "to 20 %"], "dur": (24, 72)},
    "F4": {"name": "Stuck power sensor", "root": {"input_power_watts"},
           "sev": [1, 2, 3], "sev_label": ["6–12 h", "12–24 h", "24–48 h"], "dur": None},
    "F5": {"name": "Supply sag", "root": {"input_voltage_volts"},
           "sev": [0.03, 0.06, 0.12], "sev_label": ["−3 % voltage", "−6 %", "−12 %"], "dur": (6, 24)},
    "F6": {"name": "Cooling fault (thermal control)", "root": {"cpu_temp_celsius", "gpu_temp_celsius"},
           "sev": [3.0, 6.0, 12.0], "sev_label": ["+3 °C", "+6 °C", "+12 °C"], "dur": (6, 24)},
}
EPISODES_PER_DEVICE = 6
FAR = 0.01
WINDOWS = (1, 6)  # hours of the causal trailing mean applied to every detector's scores
# score variants: aggregation over nodes × window. "max": largest |z| over nodes; "tail": each node's score is
# first turned into its tail probability under the device's own clean test distribution, so a node with wide
# normal residuals cannot set the alarm level for all the others
VARIANTS = {"max-1h": ("max", 1), "max-6h": ("max", 6), "tail-1h": ("tail", 1), "tail-6h": ("tail", 6)}
REL_FLOOR = 0.005  # residual scale floor for power / voltage / current / energy: 0.5 % of the device's level

G: pd.DataFrame
MODELS: dict = {}
CALIB: dict = {}
STATE: dict = {}


# ------------------------------------------------------------------ structure learning

def notears_linear(X, lambda1=0.05, max_iter=100, h_tol=1e-8, rho_max=1e16, w_threshold=0.3):
    """NOTEARS (Zheng et al. 2018), linear SEM with least-squares loss; W[j, i] != 0 means j -> i."""
    n, d = X.shape
    rho, alpha, h = 1.0, 0.0, np.inf

    def _adj(w):
        return (w[: d * d] - w[d * d:]).reshape(d, d)

    def _func(w):
        W = _adj(w)
        R = X - X @ W
        loss, g_loss = 0.5 / n * (R ** 2).sum(), -1.0 / n * X.T @ R
        E = sla.expm(W * W)
        hv, g_h = np.trace(E) - d, E.T * W * 2
        obj = loss + 0.5 * rho * hv * hv + alpha * hv + lambda1 * w.sum()
        g = g_loss + (rho * hv + alpha) * g_h
        return obj, np.concatenate((g + lambda1, -g + lambda1), axis=None)

    def _h(W):
        return np.trace(sla.expm(W * W)) - d

    w_est = np.zeros(2 * d * d)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    for _ in range(max_iter):
        while rho < rho_max:
            w_new = sopt.minimize(_func, w_est, method="L-BFGS-B", jac=True, bounds=bnds).x
            h_new = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W = _adj(w_est)
    W[np.abs(W) < w_threshold] = 0
    return W


def learn_graphs(Z: np.ndarray, Zlag1: np.ndarray, Zlag2: np.ndarray) -> dict[str, dict[str, list[str]]]:
    """Z: seasonal z of training hours with every node present (rows), lags aligned to the same rows."""
    rng = np.random.default_rng(0)
    d = len(NODES)
    graphs = {"none": {n: [] for n in NODES}, "hand": HAND, "full": {n: [m for m in NODES if m != n] for n in NODES}}
    Zs = (Z - Z.mean(0)) / Z.std(0)
    # top-k correlation (GDN learns a top-k neighbour graph from embedding similarity)
    R = np.corrcoef(Zs.T)
    graphs["corr top-k"] = {n: [NODES[j] for j in np.argsort(-np.abs(R[i]))[1: K_TOP + 1]] for i, n in enumerate(NODES)}
    # Lasso neighbourhood selection (Meinshausen-Bühlmann), strongest K_TOP non-zero coefficients
    sub = rng.choice(len(Zs), min(40000, len(Zs)), replace=False)
    las = {}
    for i, n in enumerate(NODES):
        others = [j for j in range(d) if j != i]
        m = LassoCV(cv=3, alphas=30, n_jobs=8).fit(Zs[sub][:, others], Zs[sub, i])
        order = [others[j] for j in np.argsort(-np.abs(m.coef_)) if abs(m.coef_[j]) > 0][:K_TOP]
        las[n] = [NODES[j] for j in order]
    graphs["lasso"] = las
    # pooled Granger: F statistic of lags 1-2 of j added to an AR(2) of i, strongest K_TOP sources
    ok = np.isfinite(Zlag1).all(1) & np.isfinite(Zlag2).all(1)
    Y, L1, L2 = Z[ok], Zlag1[ok], Zlag2[ok]
    gr, gF = {}, {}
    for i, n in enumerate(NODES):
        Xr = np.column_stack([np.ones(len(Y)), L1[:, i], L2[:, i]])
        rss_r = np.sum((Y[:, i] - Xr @ np.linalg.lstsq(Xr, Y[:, i], rcond=None)[0]) ** 2)
        F = {}
        for j in range(d):
            if j == i:
                continue
            Xf = np.column_stack([Xr, L1[:, j], L2[:, j]])
            rss_f = np.sum((Y[:, i] - Xf @ np.linalg.lstsq(Xf, Y[:, i], rcond=None)[0]) ** 2)
            F[NODES[j]] = ((rss_r - rss_f) / 2) / (rss_f / (len(Y) - 5))
        gr[n] = sorted(F, key=F.get, reverse=True)[:K_TOP]
        gF[n] = {k: round(float(v), 1) for k, v in F.items()}
    graphs["granger"] = gr
    # NOTEARS linear DAG on a subsample
    sub = rng.choice(len(Zs), min(20000, len(Zs)), replace=False)
    W = notears_linear(Zs[sub])
    nt = {}
    for i, n in enumerate(NODES):
        js = [j for j in np.argsort(-np.abs(W[:, i])) if W[j, i] != 0][:3]
        nt[n] = [NODES[j] for j in js]
    graphs["notears"] = nt
    STATE["granger_F"] = gF
    STATE["notears_W"] = np.round(W, 3).tolist()
    STATE["corr"] = np.round(R, 3).tolist()
    return graphs


# ------------------------------------------------------------------ models (fit in workers)

def fit_linear(d: pd.DataFrame, node: str, parents: list[str]):
    """Per-device ridge of the node on its parents (standardised, small penalty); unused parents get 0.
    Returns (B, M, b): coefficient, parent mean and intercept per device code."""
    n_dev, k = len(G["device"].cat.categories), len(parents)
    B, M, b = np.zeros((n_dev, k)), np.zeros((n_dev, k)), np.zeros(n_dev)
    codes = d["device"].cat.codes.to_numpy()
    y_all, X_all = d[node].to_numpy(dtype=float), d[parents].to_numpy(dtype=float)
    for c in np.unique(codes):
        m = codes == c
        y, X = y_all[m], X_all[m]
        b[c] = np.nanmedian(y)
        use = np.isfinite(X).mean(0) >= 0.5
        if not use.any():
            continue
        M[c, use] = np.nanmean(X[:, use], axis=0)
        ok = np.isfinite(y) & np.isfinite(X[:, use]).all(1)
        if ok.sum() < 50:
            continue
        Xc = X[ok][:, use] - M[c, use]
        sd = Xc.std(0)
        keep = sd > 1e-9
        if not keep.any():
            continue
        Xs = Xc[:, keep] / sd[keep]
        yc = y[ok] - y[ok].mean()
        beta = np.linalg.solve(Xs.T @ Xs + 1e-2 * len(Xs) * np.eye(Xs.shape[1]), Xs.T @ yc) / sd[keep]
        idx = np.flatnonzero(use)[keep]
        B[c, idx] = beta
        b[c] = y[ok].mean()
    return B, M, b


def linear_pred(lin, codes: np.ndarray, Xp: np.ndarray, Sp: np.ndarray) -> np.ndarray:
    """Linear part for rows: missing parent hours take the parent's device × hour median, then its mean."""
    B, M, b = lin
    X = np.where(np.isfinite(Xp), Xp, Sp)
    X = np.where(np.isfinite(X), X, M[codes])
    return b[codes] + ((X - M[codes]) * B[codes]).sum(1)


def fit_model(key):
    """Hybrid node model: per-device linear in the graph parents (extrapolates exact relations such as
    I = P / V or the energy integral) plus a global LightGBM on context and parents for the remainder."""
    node, parents = key
    parents = list(parents)
    m = STATE["fit_mask"] & G[node].notna().to_numpy()
    d = G.loc[m]
    target = d[node].to_numpy(dtype=float)
    lin = None
    if parents:
        lin = fit_linear(d, node, parents)
        idx = [NI[p] for p in parents]
        target = target - linear_pred(lin, d["device"].cat.codes.to_numpy(), d[parents].to_numpy(dtype=float),
                                      seasonal_pred(d)[:, idx])
    mdl = lgb.LGBMRegressor(**C.lgbm_params()).fit(d[CONTEXT + parents], target)
    return key, (lin, mdl)


def node_inputs(D: pd.DataFrame, Y: np.ndarray) -> pd.DataFrame:
    """Context columns of D with the node columns taken from Y (the possibly injected values)."""
    X = D[CONTEXT].copy()
    for i, n in enumerate(NODES):
        X[n] = Y[:, i]
    return X


def predict_matrix(det: str, D: pd.DataFrame, Y: np.ndarray) -> np.ndarray:
    """Expected value of every node for detector det, given context and the observed (injected) parents."""
    if det == "seasonal":
        return seasonal_pred(D)
    X = node_inputs(D, Y)
    S = seasonal_pred(D)
    codes = D["device"].cat.codes.to_numpy()
    cols = []
    for n in NODES:
        key = STATE["keys"][det][n]
        lin, mdl = MODELS[key]
        parents = list(key[1])
        p = mdl.predict(X[CONTEXT + parents], num_threads=32)
        if lin is not None:
            idx = [NI[q] for q in parents]
            p = p + linear_pred(lin, codes, Y[:, idx], S[:, idx])
        cols.append(p)
    return np.column_stack(cols)


def seasonal_pred(D: pd.DataFrame) -> np.ndarray:
    prof = STATE["seasonal"]  # (device, hour) -> medians per node
    k = pd.MultiIndex.from_arrays([D["device"].astype(str), D["time"].dt.hour])
    return prof.reindex(k).to_numpy(dtype=float)


def residual_z(det: str, D: pd.DataFrame, Y: np.ndarray) -> np.ndarray:
    """Signed z matrix (rows × nodes) for one residual detector; Y holds the (possibly injected) node values."""
    P = predict_matrix(det, D, Y)
    dev = D["device"].astype(str).to_numpy()
    med, scale = CALIB[det]  # DataFrames device × node
    return (Y - P - med.reindex(dev).to_numpy()) / scale.reindex(dev).to_numpy()


def score_all(D: pd.DataFrame, Y: np.ndarray) -> dict[str, np.ndarray]:
    """Raw scores per detector: signed z (residual detectors), per-node squared error (PCA), 1-D (IForest)."""
    out = {det: residual_z(det, D, Y) for det in STATE["residual_dets"]}
    X = np.nan_to_num(out["seasonal"], nan=0.0)
    out["iforest"] = -STATE["iforest"].score_samples(X)
    pca, mu, sd = STATE["pca"]
    Xs = (X - mu) / sd
    out["pca"] = (Xs - pca.inverse_transform(pca.transform(Xs))) ** 2
    return out


def trailing_mean(X: np.ndarray, w: int) -> np.ndarray:
    """Causal w-hour mean per device on the test grid (rows are contiguous per device); NaN hours skipped."""
    if w == 1:
        return X
    df = pd.DataFrame(X)
    return df.groupby(STATE["T_codes"]).rolling(w, min_periods=1).mean().reset_index(level=0, drop=True).sort_index().to_numpy()


def node_scores(det: str, s: np.ndarray, w: int) -> np.ndarray:
    """Non-negative node-score matrix after the w-hour trailing mean (one column for IForest)."""
    return trailing_mean(s[:, None], w) if det == "iforest" else np.abs(trailing_mean(s, w))


def tail_transform(det: str, w: int, N: np.ndarray) -> np.ndarray:
    """-log10 of the exceedance probability of each node score under the device's clean test scores."""
    out = np.full(N.shape, np.nan)
    ref = STATE["ref"][det, w]
    for c, rows in STATE["dev_rows"].items():
        for j in range(N.shape[1]):
            r = ref[c][j]
            if not len(r):
                continue
            x = N[rows, j]
            ok = np.isfinite(x)
            rank = np.searchsorted(r, x[ok], side="right")
            out[rows[ok], j] = -np.log10((len(r) - rank + 1) / (len(r) + 1))
    return out


def system_and_nodes(det: str, s: np.ndarray, v: str):
    """System score per hour and the node-score matrix used for localisation (None for IForest)."""
    agg, w = VARIANTS[v]
    N = node_scores(det, s, w)
    if agg == "tail":
        N = tail_transform(det, w, N)
    with np.errstate(all="ignore"):
        sys_s = np.nanmax(np.where(np.isfinite(N), N, np.nan), axis=1)
    return sys_s, (None if det == "iforest" else N)


def device_thresholds(v: np.ndarray) -> np.ndarray:
    """Per-device threshold at the 1 % hourly false-alarm rate of the clean test period, mapped to rows."""
    q = pd.Series(np.where(STATE["valid"], v, np.nan)).groupby(STATE["T_codes"]).quantile(1 - FAR)
    return q.reindex(STATE["T_codes"]).to_numpy()


# ------------------------------------------------------------------ fault injection

def sample_episodes(T: pd.DataFrame, fault: str, sev_i: int, rng) -> list[tuple[str, int, int]]:
    """(device, start row, end row exclusive) episodes on the test grid T, >= 48 h apart."""
    spec = FAULTS[fault]
    eps = []
    for dev, g in T.groupby("device", observed=True, sort=False):
        rows = g.index.to_numpy()
        ok = g["input_power_watts"].notna().to_numpy()
        chosen = []
        for _ in range(200):
            if len(chosen) == EPISODES_PER_DEVICE:
                break
            if spec["dur"] is None:  # stuck sensor: duration is the severity
                lo, hi = [(6, 12), (12, 24), (24, 48)][sev_i]
            else:
                lo, hi = spec["dur"]
            dur = int(rng.integers(lo, hi + 1))
            s = int(rng.integers(1, len(rows) - dur))
            if not ok[s] or ok[s: s + dur].mean() < 0.8:
                continue
            if any(s < c1 + 48 and c0 < s + dur + 48 for c0, c1 in chosen):
                continue
            chosen.append((s, s + dur))
        eps += [(str(dev), int(rows[a]), int(rows[b - 1]) + 1) for a, b in chosen]
    return eps


def inject(T: pd.DataFrame, Y: np.ndarray, fault: str, sev: float, eps) -> np.ndarray:
    Y = Y.copy()
    k = STATE["coef"]
    medP, medCG = STATE["medP"], STATE["medCG"]
    for dev, a, b in eps:
        r = np.arange(a, b) - STATE["t0"]
        V = Y[r, NI["input_voltage_volts"]]
        if fault == "F1":
            dP = sev * medP[dev]
            Y[r, NI["input_power_watts"]] += dP
            Y[r, NI["input_current_amperes"]] += dP / V
            Y[r, NI["input_energy_wh_total_rate"]] += dP
            Y[r, NI["board_temp_celsius"]] += k["board_per_input"] * dP
        elif fault == "F2":
            dC = sev * max(medCG[dev], 1.0)
            Y[r, NI["cpu_gpu_power_watts"]] += dC
            Y[r, NI["orin_power_watts"]] += k["orin_per_cg"] * dC
            Y[r, NI["power"]] += k["orin_per_cg"] * dC
            dP = k["input_per_cg"] * dC
            Y[r, NI["input_power_watts"]] += dP
            Y[r, NI["input_current_amperes"]] += dP / V
            Y[r, NI["input_energy_wh_total_rate"]] += dP
            Y[r, NI["cpu_temp_celsius"]] += k["cpu_per_cg"] * dC
            Y[r, NI["gpu_temp_celsius"]] += k["gpu_per_cg"] * dC
            Y[r, NI["board_temp_celsius"]] += k["board_per_input"] * dP
        elif fault == "F3":
            Y[r, NI["input_energy_wh_total_rate"]] *= 1 + sev * np.linspace(0, 1, len(r))
        elif fault == "F4":
            p0 = Y[r[0], NI["input_power_watts"]]  # episodes start on an hour with input power present
            Y[r, NI["input_power_watts"]] = np.where(np.isfinite(Y[r, NI["input_power_watts"]]), p0, np.nan)
        elif fault == "F5":
            P = Y[r, NI["input_power_watts"]] * (1 + sev / 3)
            Vn = V * (1 - sev)
            Y[r, NI["input_power_watts"]] = P
            Y[r, NI["input_voltage_volts"]] = Vn
            Y[r, NI["input_current_amperes"]] = P / Vn
            Y[r, NI["input_energy_wh_total_rate"]] *= 1 + sev / 3
        elif fault == "F6":
            Y[r, NI["cpu_temp_celsius"]] += sev
            Y[r, NI["gpu_temp_celsius"]] += sev
            Y[r, NI["board_temp_celsius"]] += 0.2 * sev
    return Y


def run_scenario(args):
    fault, sev_i, seed = args
    t0 = time.time()
    T = STATE["T"]
    rng = np.random.default_rng(seed)
    eps = sample_episodes(T, fault, sev_i, rng)
    Y = inject(T, STATE["Y0"], fault, FAULTS[fault]["sev"][sev_i], eps)
    S = score_all(T, Y)
    lab = np.zeros(len(T), bool)
    for _, a, b in eps:
        lab[a - STATE["t0"]: b - STATE["t0"]] = True
    roots = {NI[n] for n in FAULTS[fault]["root"]}
    codes = STATE["T_codes"]
    res, recs = {}, []
    for det, s in S.items():
        res[det] = {}
        for v in VARIANTS:
            sys_s, N = system_and_nodes(det, s, v)
            fin = np.isfinite(sys_s) & STATE["valid"]
            auroc = roc_auc_score(lab[fin], sys_s[fin])
            auprc = average_precision_score(lab[fin], sys_s[fin])
            dev_auc = []
            for c in np.unique(codes[fin & lab]):
                m = fin & (codes == c)
                if 0 < lab[m].sum() < m.sum():
                    dev_auc.append(roc_auc_score(lab[m], sys_s[m]))
            alarm = (sys_s > STATE["thr"][det, v]) & STATE["valid"]
            det_n, loc_ok, delays, locs = 0, 0, [], []
            for dev, a, b in eps:
                r = np.arange(a, b) - STATE["t0"]
                hit = np.flatnonzero(alarm[r])
                found = len(hit) > 0
                loc = None
                if N is not None:
                    with np.errstate(all="ignore"):
                        mn = np.nanmean(np.where(np.isfinite(N[r]), N[r], np.nan), axis=0)
                    if np.isfinite(mn).any():
                        loc = int(np.nanargmax(mn))
                        locs.append(loc in roots)
                recs.append({"detector": det, "variant": v, "fault": fault, "sev": sev_i, "device": dev, "start": a,
                             "dur": b - a, "detected": found, "delay": int(hit[0]) if found else None,
                             "loc": NODES[loc] if loc is not None else None,
                             "loc_ok": (loc in roots) if loc is not None else None})
                if found:
                    det_n += 1
                    delays.append(hit[0])
                    loc_ok += bool(loc is not None and loc in roots)
            res[det][v] = {"auroc": float(auroc), "auprc": float(auprc), "auroc_device_mean": float(np.mean(dev_auc)),
                           "prevalence": float(lab[fin].mean()), "recall": det_n / max(len(eps), 1),
                           "delay_median": float(np.median(delays)) if delays else None,
                           "loc_acc_detected": loc_ok / det_n if det_n and N is not None else None,
                           "loc_acc_all": float(np.mean(locs)) if locs else None,
                           "alarm_rate_clean_rows": float(alarm[~lab & fin].mean())}
    return fault, sev_i, len(eps), res, recs, time.time() - t0


# ------------------------------------------------------------------ main

def within_slope(d: pd.DataFrame, y: str, xs: list[str]) -> np.ndarray:
    d = d[[y, *xs, "device"]].dropna()
    dm = d.groupby("device", observed=True)[[y, *xs]].transform(lambda s: s - s.mean())
    return np.linalg.lstsq(dm[xs].to_numpy(), dm[y].to_numpy(), rcond=None)[0]


def main() -> None:
    global G
    t0 = time.time()
    F = C.load()
    dev = F["device"].astype(str)
    cnt = lambda m: F[m].groupby(dev[m])["input_power_watts"].count()
    n_fit = cnt(F["time"] < C.CAL)
    n_cal = cnt((F["time"] >= C.CAL) & (F["time"] < C.TEST))
    n_test = cnt(F["time"] >= C.TEST)
    devs = sorted(set(n_fit[n_fit >= 336].index) & set(n_cal[n_cal >= 168].index) & set(n_test[n_test >= 168].index))
    G = F[dev.isin(devs)].sort_values(["device", "time"]).reset_index(drop=True)
    G["device"] = G["device"].astype(str).astype(pd.CategoricalDtype(devs))
    G["hour"] = G["time"].dt.hour
    fit = (G["time"] < C.CAL).to_numpy()
    cal = ((G["time"] >= C.CAL) & (G["time"] < C.TEST)).to_numpy()
    test = (G["time"] >= C.TEST).to_numpy()
    print(f"{len(devs)} devices, {fit.sum():,} fit / {cal.sum():,} cal / {test.sum():,} test grid hours", flush=True)

    # seasonal profile (fit period) and its z, used for cleaning, structure learning and three detectors
    gd = G.loc[fit]
    STATE["seasonal"] = gd.groupby([gd["device"].astype(str), "hour"])[NODES].median()
    Y_all = G[NODES].to_numpy(dtype=float)
    R0 = Y_all - seasonal_pred(G)
    mad0 = pd.DataFrame(np.abs(R0[fit] - np.nanmedian(R0[fit], 0)), columns=NODES).groupby(
        G.loc[fit, "device"].astype(str).to_numpy()).median() * 1.4826
    floor = np.array([FLOOR.get(n, 0.05) for n in NODES])
    z0 = R0 / np.maximum(mad0.reindex(G["device"].astype(str)).to_numpy(), floor)
    clean = ~(np.abs(np.nan_to_num(z0)) > 8).any(axis=1)
    STATE["fit_mask"] = fit & clean
    print(f"training rows removed by the |z| > 8 cleaning: {(fit & ~clean).sum():,} of {fit.sum():,}", flush=True)

    # structure learning on training hours with every node present
    zdf = pd.DataFrame(z0, columns=NODES)
    lag1 = zdf.groupby(G["device"], observed=True).shift(1).to_numpy()
    lag2 = zdf.groupby(G["device"], observed=True).shift(2).to_numpy()
    full_rows = STATE["fit_mask"] & np.isfinite(z0).all(1)
    graphs = learn_graphs(z0[full_rows], lag1[full_rows], lag2[full_rows])
    for gname, g in graphs.items():
        print(f"  {gname:11s} " + "; ".join(f"{C.LABEL[n]} ← {', '.join(C.LABEL[p] for p in ps) or '∅'}"
                                            for n, ps in g.items()), flush=True)

    # fit one model per distinct (node, parent set)
    STATE["keys"] = {gname: {n: (n, tuple(sorted(ps))) for n, ps in g.items()} for gname, g in graphs.items()}
    keys = sorted({k for g in STATE["keys"].values() for k in g.values()})
    print(f"fitting {len(keys)} node models", flush=True)
    with ProcessPoolExecutor(max_workers=12, mp_context=mp.get_context("fork")) as ex:
        for key, mdl in ex.map(fit_model, keys):
            MODELS[key] = mdl
    print(f"  done ({time.time() - t0:.0f}s)", flush=True)

    # calibration month: per device and node residual median and robust scale
    STATE["residual_dets"] = ["seasonal"] + list(graphs)
    Gc = G.loc[cal]
    Yc = Gc[NODES].to_numpy(dtype=float)
    cdev = Gc["device"].astype(str).to_numpy()
    level = gd.groupby(gd["device"].astype(str))[NODES].median().abs().reindex(devs).fillna(0.0)
    rel = np.array([0.0 if n in C.THERMAL else REL_FLOOR for n in NODES])
    for det in STATE["residual_dets"]:
        P = predict_matrix(det, Gc, Yc)
        R = pd.DataFrame(Yc - P, columns=NODES)
        med = R.groupby(cdev).median()
        mad = (R - med.reindex(cdev).to_numpy()).abs().groupby(cdev).median() * 1.4826
        mad = mad.reindex(devs).fillna(np.inf).to_numpy()
        scale = np.maximum(np.maximum(mad, floor), level.to_numpy() * rel)
        CALIB[det] = (med.reindex(devs), pd.DataFrame(scale, columns=NODES, index=devs))
    # calibration-month prediction quality per detector and node (clean data)
    cal_quality = {}
    for det in STATE["residual_dets"]:
        med, scale = CALIB[det]
        cal_quality[det] = {n: float(scale[n].median()) for n in NODES}

    # Isolation Forest / PCA on the signed seasonal z vector of the training hours
    Xf = np.nan_to_num(residual_z("seasonal", G.loc[STATE["fit_mask"]], Y_all[STATE["fit_mask"]]), nan=0.0)
    STATE["iforest"] = IsolationForest(n_estimators=300, max_samples=4096, random_state=0, n_jobs=16).fit(Xf)
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-9
    pca = PCA(n_components=0.9, random_state=0).fit((Xf - mu) / sd)
    STATE["pca"] = (pca, mu, sd)
    print(f"PCA keeps {pca.n_components_} of {len(NODES)} components", flush=True)

    # propagation coefficients for the injected faults (within-device slopes on the training hours)
    gf = G.loc[STATE["fit_mask"]].copy()
    gf["amb"] = gf["temperature"].fillna(gf["peer_temperature"])
    k = {"orin_per_cg": within_slope(gf, "orin_power_watts", ["cpu_gpu_power_watts"])[0],
         "input_per_cg": within_slope(gf, "input_power_watts", ["cpu_gpu_power_watts"])[0],
         "cpu_per_cg": within_slope(gf, "cpu_temp_celsius", ["cpu_gpu_power_watts", "amb"])[0],
         "gpu_per_cg": within_slope(gf, "gpu_temp_celsius", ["cpu_gpu_power_watts", "amb"])[0],
         "board_per_input": within_slope(gf, "board_temp_celsius", ["input_power_watts", "amb"])[0]}
    STATE["coef"] = {a: float(b) for a, b in k.items()}
    print("propagation coefficients", {a: round(b, 3) for a, b in STATE["coef"].items()}, flush=True)
    STATE["medP"] = gd.groupby(gd["device"].astype(str))["input_power_watts"].median().to_dict()
    STATE["medCG"] = gd.groupby(gd["device"].astype(str))["cpu_gpu_power_watts"].median().fillna(1.0).to_dict()

    # clean test period: thresholds at the 1 % false-alarm rate, real P vs V·I inconsistencies
    T = G.loc[test].reset_index(drop=True)  # each device's test hours stay contiguous (sorted by device, time)
    STATE["T"], STATE["t0"] = T, 0
    STATE["Y0"] = T[NODES].to_numpy(dtype=float)
    STATE["valid"] = np.isfinite(STATE["Y0"][:, NI["input_power_watts"]])  # hours the device reported power
    STATE["T_codes"] = T["device"].cat.codes.to_numpy()
    STATE["dev_rows"] = {c: np.flatnonzero(STATE["T_codes"] == c) for c in np.unique(STATE["T_codes"])}
    S0 = score_all(T, STATE["Y0"])
    STATE["ref"] = {}
    for det, sc in S0.items():
        for w in WINDOWS:
            N = node_scores(det, sc, w)
            STATE["ref"][det, w] = {c: [np.sort(x[np.isfinite(x)]) for x in N[rows][STATE["valid"][rows]].T]
                                    for c, rows in STATE["dev_rows"].items()}
    STATE["thr"], thr_summary = {}, {}
    for det, sc in S0.items():
        for v in VARIANTS:
            STATE["thr"][det, v] = device_thresholds(system_and_nodes(det, sc, v)[0])
            thr_summary.setdefault(det, {})[v] = float(np.nanmedian(STATE["thr"][det, v]))
    v = STATE["valid"]
    P, V, I = (STATE["Y0"][:, NI[n]] for n in ("input_power_watts", "input_voltage_volts", "input_current_amperes"))
    with np.errstate(all="ignore"):
        mism = (np.abs(P - V * I) / P > 0.02) & v
    real = {"hours": int(mism.sum()), "devices": int(T.loc[mism, "device"].nunique()), "by_detector": {}}
    clean_alarm = {}
    for det, sc in S0.items():
        sys1, N1 = system_and_nodes(det, sc, "max-1h")
        a = (sys1 > STATE["thr"][det, "max-1h"]) & v
        real["by_detector"][det] = {"recall": float(a[mism].mean()) if mism.any() else None}
        entry = {"per_day": pd.Series(a[v]).groupby(T.loc[v, "time"].dt.strftime("%Y-%m-%d").to_numpy()).mean().round(4).to_dict()}
        if N1 is not None:
            for key, rows in (("top_node", a), ("pvi_top_node", mism)):
                if rows.any():
                    top = np.nanargmax(np.where(np.isfinite(N1[rows]), N1[rows], -1), axis=1)
                    entry[key] = pd.Series([NODES[t] for t in top]).value_counts().to_dict()
        clean_alarm[det] = entry
    # devices whose supply voltage left its July behaviour (they dominated a fleet-wide threshold)
    zv = pd.Series(np.abs(S0["hand"][:, NI["input_voltage_volts"]])).groupby(T["device"].astype(str).to_numpy()).median()
    real["voltage_shift_devices"] = zv.sort_values(ascending=False).head(6).round(1).to_dict()
    print(f"clean test scored ({time.time() - t0:.0f}s); P≠V·I hours: {real['hours']}", flush=True)

    # injected scenarios, scored in this process: LightGBM's OpenMP pool is live here, and forking after it
    # has run deadlocks the children
    scen = [(f, s, 1000 + 10 * i + s) for i, f in enumerate(FAULTS) for s in range(3)]
    results, episodes = {}, []
    for fault, sev_i, n_eps, res, recs, dt in map(run_scenario, scen):
        results.setdefault(fault, {})[sev_i] = {"episodes": n_eps, "detectors": res}
        episodes += recs
        best = max(res, key=lambda d_: res[d_]["max-6h"]["recall"])
        print(f"  {fault} sev{sev_i}: {n_eps} episodes, recall hand max-6h {res['hand']['max-6h']['recall']:.2f} "
              f"tail-6h {res['hand']['tail-6h']['recall']:.2f}; best max-6h {best} {res[best]['max-6h']['recall']:.2f} "
              f"({dt:.0f}s)", flush=True)
    ep = pd.DataFrame(episodes)
    ep.to_parquet(C.OUT / "rq2_episodes.parquet", index=False)

    # device-cluster bootstrap of the event-recall difference: hand graph minus each other detector
    rng = np.random.default_rng(7)
    devs_arr = np.array(devs)
    boot = {}
    energy = [f for f in FAULTS if f != "F6"]
    samples = [rng.choice(devs_arr, len(devs_arr), replace=True) for _ in range(2000)]
    for w in VARIANTS:
        for metric in ("detected", "loc_ok"):
            e_ = ep[ep["fault"].isin(energy) & (ep["variant"] == w)]
            piv = e_.pivot_table(index=["fault", "sev", "device", "start"], columns="detector", values=metric,
                                 aggfunc="first").astype(float)
            by_dev = piv.groupby(level="device").agg(["sum", "count"])
            for det in piv.columns:
                if det == "hand" or by_dev[(det, "count")].sum() == 0:
                    continue
                diffs = []
                for sample in samples:
                    sub = by_dev.reindex(sample)
                    diffs.append(sub[("hand", "sum")].sum() / sub[("hand", "count")].sum()
                                 - sub[(det, "sum")].sum() / sub[(det, "count")].sum())
                diffs = np.array(diffs)
                boot.setdefault(w, {}).setdefault(metric, {})[det] = {
                    "diff": float(piv["hand"].mean() - piv[det].mean()), "lo": float(np.quantile(diffs, 0.025)),
                    "hi": float(np.quantile(diffs, 0.975)), "p_le_0": float((diffs <= 0).mean())}

    # one example episode per energy fault (medium severity) for the report: node |z| of hand vs complete graph
    ex_rows = []
    for f in ("F1", "F2", "F4"):
        e = ep[(ep["fault"] == f) & (ep["sev"] == 1) & (ep["detector"] == "hand") & (ep["variant"] == "max-6h") & ep["detected"]]
        if e.empty:
            continue
        e = e.sort_values("dur").iloc[len(e) // 2]
        rng_f = np.random.default_rng(1000 + 10 * list(FAULTS).index(f) + 1)
        eps = sample_episodes(T, f, 1, rng_f)
        Y = inject(T, STATE["Y0"], f, FAULTS[f]["sev"][1], eps)
        a, b = int(e["start"]) - STATE["t0"], int(e["start"]) - STATE["t0"] + int(e["dur"])
        lo, hi = max(a - 24, 0), min(b + 24, len(T))
        win = T.iloc[lo:hi]
        for det in ("hand", "full", "none", "corr top-k"):
            Zm = residual_z(det, win, Y[lo:hi])
            for i, n in enumerate(NODES):
                ex_rows.append(pd.DataFrame({"fault": f, "detector": det, "node": n, "device": e["device"],
                                             "time": win["time"].to_numpy(),
                                             "z": Zm[:, i], "y": Y[lo:hi, i], "y0": STATE["Y0"][lo:hi, i],
                                             "in_episode": (np.arange(lo, hi) >= a) & (np.arange(lo, hi) < b)}))
    if ex_rows:
        pd.concat(ex_rows).to_parquet(C.OUT / "rq2_examples.parquet", index=False)

    # summary tables
    dets = list(S0)
    summ = {}
    for w in VARIANTS:
        for det in dets:
            rows = ep[(ep["detector"] == det) & ep["fault"].isin(energy) & (ep["variant"] == w)]
            r_ = [results[f][s_]["detectors"][det][w] for f in energy for s_ in range(3)]
            summ.setdefault(w, {})[det] = {
                "recall": float(rows["detected"].mean()),
                "auprc_mean": float(np.mean([x["auprc"] for x in r_])),
                "auroc_mean": float(np.mean([x["auroc"] for x in r_])),
                "auroc_device_mean": float(np.mean([x["auroc_device_mean"] for x in r_])),
                "loc_acc": float(rows["loc_ok"].dropna().mean()) if rows["loc_ok"].notna().any() else None,
                "delay_median": float(rows.loc[rows["detected"], "delay"].median()) if rows["detected"].any() else None,
            }
    out = {"generated": datetime.now(timezone.utc).isoformat(), "devices": len(devs), "nodes": NODES,
           "label": C.LABEL, "graphs": graphs, "hand_why": {f"{a}>{b}": w for (a, b), w in HAND_WHY.items()},
           "granger_F": STATE["granger_F"], "notears_W": STATE["notears_W"], "corr": STATE["corr"],
           "faults": {f: {k_: (sorted(v) if isinstance(v, set) else v) for k_, v in s.items()} for f, s in FAULTS.items()},
           "coef": STATE["coef"], "thresholds_median": thr_summary, "variants": list(VARIANTS), "far": FAR, "episodes_per_device": EPISODES_PER_DEVICE,
           "results": results, "summary_energy": summ, "bootstrap_vs_hand": boot, "calibration_scale": cal_quality,
           "real_pvi": real, "clean_alarms": clean_alarm,
           "rows": {"fit": int(fit.sum()), "cal": int(cal.sum()), "test": int(test.sum()),
                    "cleaned": int((fit & ~clean).sum())},
           "runtime_s": round(time.time() - t0, 1)}
    (C.OUT / "rq2.json").write_text(json.dumps(C.clean(out), allow_nan=False))
    print(f"\nwrote rq2.json in {time.time() - t0:.0f}s\n", flush=True)
    for w in VARIANTS:
        print(f"\n{w} — {'detector':12s} recall  AUPRC  AUROC  devAUROC  loc   delay   (F1-F5, all severities)")
        for det, m in sorted(summ[w].items(), key=lambda x: -x[1]["recall"]):
            loc = f"{m['loc_acc']:.2f}" if m["loc_acc"] is not None else "  — "
            print(f"   {det:12s} {m['recall']:.3f}  {m['auprc_mean']:.3f}  {m['auroc_mean']:.3f}  "
                  f"{m['auroc_device_mean']:.3f}  {loc}  {m['delay_median']}")
        print("   hand − other, event recall / localisation (95 % device bootstrap):")
        for det, b in boot[w]["detected"].items():
            lb = boot[w]["loc_ok"].get(det)
            lt = f"   loc {lb['diff']:+.3f} [{lb['lo']:+.3f}, {lb['hi']:+.3f}]" if lb else ""
            print(f"     {det:12s} {b['diff']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}]{lt}")


if __name__ == "__main__":
    main()

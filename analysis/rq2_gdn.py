#!/usr/bin/env python
"""Extension 1: GDN learned-graph baseline (Deng & Hooi 2021 adaptation).

Faithful four-component GDN on RQ1 context residuals:
  1. Sensor embedding          learnable vector per node
  2. Graph structure learning  cosine similarity -> top-k neighbours
  3. Graph attention forecast  each node predicted from its own window plus
                               its k neighbours' windows
  4. Graph deviation scoring   per-node |error| robust-normalised (median/IQR)
                               on the fit window, then max over nodes

Input: RQ1 context-model residual z-scores (device x hour x node), the same
residuals the other RQ2 detectors use (reviewer recommendation D).

Protocol (no leakage): train on fit window only; per-device 1% FAR threshold
from July calibration; evaluate on the same Sep-Oct test episodes.

Usage: python3 analysis/rq2_gdn.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq2_graphcore as G  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/out"
RQ2 = OUT / "rq2"
DATA = RQ2 / "data"

NODES = G.NODES
N = len(NODES)
FAR = G.FAR
WINDOW = 12
K_NEIGHBOURS = 3
EMB_DIM = 16
EPOCHS = 40
BATCH = 512
LR = 0.01
SEED = 20261103
MAX_TRAIN_WINDOWS = 60000

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_default_dtype(torch.float32)


class GDN(nn.Module):
    """Sensor embeddings -> similarity graph -> graph-conditioned forecast."""

    def __init__(self, n=N, k=K_NEIGHBOURS, emb=EMB_DIM, window=WINDOW):
        super().__init__()
        self.n, self.k, self.w = n, k, window
        self.embed = nn.Parameter(torch.randn(n, emb))
        self.heads = nn.ModuleList(
            [nn.Linear(window * (k + 1), 1) for _ in range(n)]
        )
        self._neigh = None

    def adjacency(self):
        e = self.embed / (self.embed.norm(dim=1, keepdim=True) + 1e-8)
        sim = e @ e.t()
        sim.fill_diagonal_(-1e9)
        return torch.topk(sim, self.k, dim=1).indices

    def capture(self):
        with torch.no_grad():
            self._neigh = self.adjacency().tolist()

    def forward(self, X):
        if self._neigh is None:
            self.capture()
        B = X.shape[0]
        out = []
        for i in range(self.n):
            cols = [i] + self._neigh[i]
            xi = X[:, :, cols].reshape(B, self.w * (self.k + 1))
            out.append(self.heads[i](xi))
        return torch.cat(out, dim=1)


def load_wide() -> pd.DataFrame:
    r = pd.read_parquet(DATA / "residuals.parquet",
                        columns=["device", "time", "node", "z"])
    r["device"] = r["device"].astype(str)
    r["time"] = pd.to_datetime(r["time"])
    if r["time"].dt.tz is not None:
        r["time"] = r["time"].dt.tz_convert("UTC").dt.tz_localize(None)
    wide = r.pivot_table(index=["device", "time"], columns="node",
                         values="z", aggfunc="first").reindex(columns=NODES)
    return wide


def windows(vals: np.ndarray, window: int):
    Xs, ys = [], []
    for t in range(window, len(vals)):
        chunk = vals[t - window:t + 1]
        if np.isfinite(chunk).all():
            Xs.append(chunk[:window])
            ys.append(chunk[-1])
    if not Xs:
        return (np.empty((0, window, vals.shape[1]), np.float32),
                np.empty((0, vals.shape[1]), np.float32))
    return np.asarray(Xs, np.float32), np.asarray(ys, np.float32)


def train_gdn(wide: pd.DataFrame, fit_mask: np.ndarray) -> GDN:
    dfw = wide.reset_index()
    rng = np.random.default_rng(SEED)
    devices = sorted(dfw["device"].unique())
    sample = rng.choice(devices, min(25, len(devices)), replace=False)
    Xs, ys = [], []
    for dev in sample:
        sub = dfw[(dfw["device"] == dev) & fit_mask].sort_values("time")
        vals = sub[NODES].to_numpy(dtype=float)
        X, y = windows(vals, WINDOW)
        if len(X):
            Xs.append(X)
            ys.append(y)
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    if len(X) > MAX_TRAIN_WINDOWS:
        idx = rng.choice(len(X), MAX_TRAIN_WINDOWS, replace=False)
        X, y = X[idx], y[idx]
    print(f"  training windows: {X.shape[0]:,} ({X.shape[1]}h x {N} nodes)")

    xb = torch.from_numpy(X)
    yb = torch.from_numpy(y)
    net = GDN()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    n = len(xb)
    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        total, nb = 0.0, 0
        for b in range(0, n, BATCH):
            idx = perm[b:b + BATCH]
            opt.zero_grad()
            pred = net(xb[idx])
            loss = loss_fn(pred, yb[idx])
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  epoch {ep+1:2d}/{EPOCHS}  mse={total/max(nb,1):.4f}")
    net.capture()
    return net


def score_series(net: GDN, vals: np.ndarray, err_med: np.ndarray,
                 err_iqr: np.ndarray) -> np.ndarray:
    n = len(vals)
    score = np.full(n, np.nan)
    net.eval()
    with torch.no_grad():
        for t in range(WINDOW, n):
            chunk = vals[t - WINDOW:t + 1]
            if not np.isfinite(chunk).all():
                continue
            x = torch.from_numpy(chunk[:WINDOW].reshape(1, WINDOW, N).astype(np.float32))
            pred = net(x)[0].numpy()
            err = np.abs(vals[t] - pred)
            z = (err - err_med) / np.maximum(err_iqr, 1e-8)
            score[t] = np.nanmax(z)
    return score


def main():
    t0 = time.time()
    DATA.mkdir(parents=True, exist_ok=True)

    print("Loading residuals...")
    wide = load_wide()
    print(f"  {len(wide):,} device-hours, {wide.index.get_level_values('device').nunique()} devices")

    t = pd.DatetimeIndex(wide.index.get_level_values("time"))
    def mask(a, b):
        return np.asarray((t >= pd.Timestamp(a)) & (t < pd.Timestamp(b)))
    fit_mask = mask(*G.SPLITS["fit"])
    cal_mask = mask(*G.SPLITS["calibrate"])
    test_mask = mask(*G.SPLITS["test"])
    print(f"  splits: fit={fit_mask.sum():,} cal={cal_mask.sum():,} test={test_mask.sum():,}")

    print("\nTraining GDN...")
    net = train_gdn(wide, fit_mask)
    torch.save(net.state_dict(), RQ2 / "gdn_model.pt")
    learned = {NODES[i]: [NODES[j] for j in net._neigh[i]] for i in range(N)}
    print(f"  learned graph edges: {sum(len(v) for v in learned.values())}")
    for node, ps in learned.items():
        print(f"    {node}: {ps}")
    (DATA / "gdn_learned_graph.json").write_text(json.dumps(learned, indent=2))

    # per-node error stats from fit window (robust normalisation)
    print("\nCalibrating error normalisation on fit window...")
    dfw = wide.reset_index()
    all_err = []
    for dev in dfw["device"].unique()[:15]:
        sub = dfw[(dfw["device"] == dev) & fit_mask].sort_values("time")
        vals = sub[NODES].to_numpy(dtype=float)
        net.eval()
        with torch.no_grad():
            for tt in range(WINDOW, len(vals)):
                chunk = vals[tt - WINDOW:tt + 1]
                if not np.isfinite(chunk).all():
                    continue
                x = torch.from_numpy(chunk[:WINDOW].reshape(1, WINDOW, N).astype(np.float32))
                pred = net(x)[0].numpy()
                all_err.append(np.abs(vals[tt] - pred))
    all_err = np.asarray(all_err)
    err_med = np.nanmedian(all_err, axis=0)
    q75 = np.nanquantile(all_err, 0.75, axis=0)
    q25 = np.nanquantile(all_err, 0.25, axis=0)
    err_iqr = np.maximum(q75 - q25, 1e-6)
    print(f"  error median (first 4): {np.round(err_med[:4], 3)}")
    print(f"  error IQR   (first 4): {np.round(err_iqr[:4], 3)}")

    # score every device
    print("\nScoring devices...")
    score_by_dev, thr, times_by_dev = {}, {}, {}
    eval_faults = pd.read_parquet(DATA / "evaluation_faults.parquet")
    eval_faults["device"] = eval_faults["device"].astype(str)
    wide_by_dev = {d: g for d, g in wide.groupby(level="device")}

    for dev in wide_by_dev:
        w = wide_by_dev[dev].droplevel("device")
        vals = w[NODES].to_numpy(dtype=float)
        times = w.index.to_numpy()
        sc = score_series(net, vals, err_med, err_iqr)
        score_by_dev[dev] = sc
        times_by_dev[dev] = times
        cm = ((w.index >= pd.Timestamp(G.SPLITS["calibrate"][0])) &
              (w.index < pd.Timestamp(G.SPLITS["calibrate"][1])))
        cs = sc[cm]
        cs = cs[np.isfinite(cs)]
        thr[dev] = float(np.quantile(cs, 1 - FAR)) if len(cs) >= 20 else np.nan

    # event recall on test episodes
    print("\nEvaluating on test episodes...")
    det = 0
    per_fault = {}
    for _, ep in eval_faults.iterrows():
        dev = ep["device"]
        if dev not in score_by_dev:
            continue
        times = times_by_dev[dev]
        sc = score_by_dev[dev]
        thr_d = thr.get(dev, np.nan)
        onset = pd.Timestamp(ep["start_time"])
        if onset.tz is not None:
            onset = onset.tz_convert("UTC").tz_localize(None)
        end = pd.Timestamp(ep["end_time"])
        if end.tz is not None:
            end = end.tz_convert("UTC").tz_localize(None)
        win_end = end + pd.Timedelta(hours=G.WINDOW_HOURS - 1)
        m = (times >= np.datetime64(onset.to_datetime64(), "ns")) & \
            (times <= np.datetime64(win_end.to_datetime64(), "ns"))
        found = bool(np.nanmax(np.where(m, sc, np.nan)) > thr_d) if m.any() else False
        det += found
        per_fault.setdefault(ep["fault"], 0)
        per_fault[ep["fault"]] += int(found)

    recall = det / max(len(eval_faults), 1)
    print(f"\nGDN recall @ 1% FAR: {recall:.3f} ({det}/{len(eval_faults)})")
    print(f"  per-fault: {per_fault}")

    out = {
        "generated": pd.Timestamp.now("UTC").isoformat(),
        "model": "GDN (Deng & Hooi 2021 adaptation)",
        "input": "RQ1 context-model residuals (z)",
        "recall": recall,
        "detected": det,
        "n_episodes": len(eval_faults),
        "per_fault": per_fault,
        "learned_graph": learned,
        "hyperparameters": {"window": WINDOW, "k": K_NEIGHBOURS,
                            "emb_dim": EMB_DIM, "epochs": EPOCHS,
                            "batch": BATCH, "lr": LR, "seed": SEED},
    }
    (DATA / "gdn_results.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote gdn_results.json in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
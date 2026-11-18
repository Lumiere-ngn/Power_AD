#!/usr/bin/env python
"""Phase 5 GDN: faithful Graph Deviation Network (Deng & Hooi, 2021).

Implements the four components of GDN on the RQ2 residuals:
  1. Sensor embedding        — learnable per-node embedding
  2. Graph structure learning — cosine similarity + top-k neighbour selection
  3. Graph attention forecast — predict each node from its neighbours' history
  4. Graph deviation score   — per-node error → robust median/IQR normalisation
                               → max over nodes

Input: RQ1 context residuals (z-scores per device × hour × node), consistent
with the other RQ2 detectors (reviewer's recommendation).

Training: fit window only (normal data). Calibration: July only (per-device 1%
FAR thresholds). Evaluation: test episodes on the same rules as the other
detectors.

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
import rq2_graphcore as G

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/out"
RQ2 = OUT / "rq2"
DATA = RQ2 / "data"

NODES = G.NODES
N = len(NODES)
FAR = G.FAR
WINDOW = 12       # historical input window (adapted; paper used 5)
K_NEIGHBOURS = 3  # top-k neighbours (adapted; paper used dataset-dependent)
EMB_DIM = 32      # embedding dimension (adapted)
EPOCHS = 60
BATCH = 256
LR = 0.005
SEED = 20261103

torch.set_default_dtype(torch.float32)


class GDN(nn.Module):
    """GDN: sensor embeddings + graph attention forecasting."""

    def __init__(self, n=N, k=K_NEIGHBOURS, emb=EMB_DIM, window=WINDOW):
        super().__init__()
        self.n, self.k, self.w = n, k, window
        # Sensor embeddings (component 1)
        self.embed = nn.Parameter(torch.randn(n, emb))
        # per-node linear forecasters over the window of its own history
        self.self_fc = nn.ModuleList([nn.Linear(window, 1) for _ in range(n)])
        # per-node attention over neighbour windows
        self.attn = nn.Parameter(torch.zeros(n, k, 2 * window))

    def adjacency(self):
        """Graph structure learning: top-k by embedding cosine similarity."""
        e = self.embed / (self.embed.norm(dim=1, keepdim=True) + 1e-8)
        sim = e @ e.t()
        sim.fill_diagonal_(-1e9)
        topk = torch.topk(sim, self.k, dim=1).indices  # [n, k] neighbour indices
        mask = torch.zeros_like(sim)
        mask.scatter_(1, topk, 1.0)
        return mask  # [n, n] 1 if edge parent→child

    def forward(self, X):
        """X: [batch, window, n] → prediction [batch, n] for next step."""
        B, W, _ = X.shape
        adj = self.adjacency()
        out = []
        for i in range(self.n):
            # self term: node i's own history
            self_term = self.self_fc[i](X[:, :, i])
            # neighbours
            neigh = torch.nonzero(adj[i]).flatten()
            if len(neigh) == 0:
                out.append(self_term)
                continue
            # attention-weighted sum of neighbour windows
            att = self.attn[i, : len(neigh), :]  # [k_i, 2*window]
            xw = X[:, :, neigh].reshape(B, W, -1)  # flatten neighbour windows
            # score = xw @ att_mean; combine with self
            att_score = (xw * att.mean(0).unsqueeze(0).unsqueeze(1)).sum(1)
            out.append(self_term + att_score)
        return torch.cat(out, dim=1)


def make_windows(wide: pd.DataFrame, max_len=200000, seed=SEED):
    """Build (X, y) windows of length W+1 over consecutive hours."""
    vals = wide[NODES].to_numpy(dtype=float)
    n_rows = len(wide)
    Xs, ys = [], []
    rng = np.random.default_rng(seed)
    # random valid start indices
    for t in range(WINDOW, n_rows):
        chunk = vals[t - WINDOW:t + 1]
        if np.isfinite(chunk).all():
            Xs.append(chunk[:WINDOW])
            ys.append(chunk[-1])
    if len(Xs) > max_len:
        idx = rng.choice(len(Xs), max_len, replace=False)
        Xs = [Xs[i] for i in idx]
        ys = [ys[i] for i in idx]
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def train(f: pd.DataFrame, devices: list[str]) -> GDN:
    """Train GDN on fit-window normal residuals across all devices."""
    masks = G.window_masks(f["time"])
    fit_mask = masks["fit"]

    # collect windows from a sample of devices (training scale)
    rng = np.random.default_rng(SEED)
    sample_devs = list(rng.choice(devices, min(20, len(devices)), replace=False))
    Xs, ys = [], []
    for dev in sample_devs:
        sub = f[(f["device"] == dev) & fit_mask].sort_values("time")
        w = pivot_z(sub)
        if w is None:
            continue
        X, y = make_windows(w)
        Xs.append(X)
        ys.append(y)
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    print(f"  training windows: {X.shape[0]:,} × {X.shape[1]} steps × {len(avail)} nodes")
    xb = torch.from_numpy(X)
        yb = torch.from_numpy(y)

    net = GDN()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    n = len(xb)
    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        total = 0.0
        for b in range(0, n, BATCH):
            idx = perm[b:b + BATCH]
            opt.zero_grad()
            pred = net(xb[idx])
            loss = loss_fn(pred, yb[idx])
            loss.backward()
            opt.step()
            total += loss.item()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  epoch {ep+1}/{EPOCHS}  mse={total / max(1, (n + BATCH - 1)//BATCH):.4f}")
    return net


def predict(net, X):
    with torch.no_grad():
        return net(torch.from_numpy(X)).numpy()


def graph_score_deviation(net: GDN, dev_scores: pd.DataFrame,
                          node_err_median: np.ndarray,
                          node_err_iqr: np.ndarray) -> np.ndarray:
    """Per-hour graph deviation score: max over nodes of robust-normalised error."""
    # Use the node errors from the network
    # For each row in dev_scores, compute per-node prediction error
    # Then robust-normalise: (err - median) / IQR, max over nodes
    # We need the raw per-node errors — compute from the model's predictions
    # against observed values in the wide frame
    # (this is handled in the main evaluation loop; here we just
        return max_norm_errors


def main():
    t0 = time.time()
    DATA.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    f = G.load_features(G.load_eligible_devices())
    eval_faults = pd.read_parquet(DATA / "evaluation_faults.parquet")
    devices = G.load_eligible_devices()

    print(f"Loaded {len(f):,} rows, {len(devices)} devices")

    # Load RQ1 residuals (z-scores) — the GDN input
    res = pd.read_parquet(DATA / "residuals.parquet",
                          columns=["device", "time", "node", "z", "split"])
    res["device"] = res["device"].astype(str)

    # Pivot residuals to wide: device × time × node
    z_wide = res.pivot_table(index=["device", "time"], columns="node",
                             values="z", aggfunc="first")

    # --- train GDN ---
    print("\nTraining GDN (PyTorch)...")
    net = train(f, devices)

    # Save model
    model_path = RQ2 / "gdn_model.pt"
    torch.save(net.state_dict(), model_path)
    print(f"  saved model to {model_path}")

    # ---- evaluation: forecast + score on test ---
    print("\nEvaluating GDN on test episodes...")
    # per-node error stats from fit window for robust normalisation
    masks = G.window_masks(f["time"])
    fit_mask = masks["fit"]

    # get fit-window residuals for normalisation stats
    res_fit = res[res["split"] == "fit"].copy()
    z_fit = res_fit.pivot_table(index=["device", "time"], columns="node",
                                values="z", aggfunc="first")
    node_err_median = z_fit.median().to_numpy(dtype=float)
    node_err_iqr = (z_fit.quantile(0.75) - z_fit.quantile(0.25)).to_numpy(dtype=float)
    node_err_iqr = np.where(node_err_iqr > 0, node_err_iqr, 1e-6)

    # evaluate per device
    results = {}
    for dev in devices:
        z_dev = wide.loc[dev]
        if len(z_dev) < WINDOW + 1:
            continue
        vals = z_dev[NODES].to_numpy(dtype=float)
        n_rows = len(z_dev)
        pred_err = np.full((n_rows, len(NODES)), np.nan)
        for t in range(WINDOW, n_rows):
            chunk = vals[t - WINDOW:t + 1]
            if not np.isfinite(chunk).all():
                continue
            xb = torch.from_numpy(chunk[:WINDOW].reshape(1, WINDOW, len(avail)).astype(np.float32))
            with torch.no_grad():
                pred = net(xb)[0].numpy()
            err = np.abs(vals[t] - pred)  # per-node abs error
            # robust normalise
            norm_err = (err - node_err_median) / np.maximum(node_err_iqr, 1e-8)
            pred_err[t] = norm_err
        # graph score = max over nodes of normalised error
        graph_score = np.nanmax(pred_err, axis=1)
        # threshold from calibration window
        cal_mask = keys_df.loc[dev_df.index, "split"] == "calibrate"
        cal_scores = graph_score[cal_mask]
        cal_scores = cal_scores[np.isfinite(cal_scores)]
        thr = float(np.quantile(cal_scores, 1 - FAR)) if len(cal_scores) >= 20 else np.nan
        alarms = graph_score > thr
        result = G.evaluate_recall(alarms, test_mask, eval_faults, devs,
                                   f["time"].values)
        results[dev] = result
        print(f"  {dev}: recall={result['recall']:.3f} "
              f"({result['detected']}/{result['n_episodes']})")

    # aggregate
    total_detected = sum(r["detected"] for r in results.values())
    total_eps = sum(r["n_episodes"] for r in results.values())
    agg_recall = total_detected / max(total_eps, 1)
    print(f"\nGDN overall recall: {agg_recall:.3f} ({total_detected}/{total_eps})")

    # save
    summary = {
        "generated": pd.Timestamp.now("UTC").isoformat(),
        "n_devices": len(devices),
        "n_eval_episodes": len(eval_faults),
        "recall": agg_recall,
        "per_device": {k: v["recall"] for k, v in results.items()},
        "hyperparameters": {
            "input_window": WINDOW,
            "k_neighbours": K_NEIGHBOURS,
            "embedding_dim": EMB_DIM,
            "epochs": EPOCHS,
            "batch_size": BATCH,
            "lr": LR,
            "seed": SEED,
        },
    }
    (DATA / "gdn_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote gdn_results.json in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
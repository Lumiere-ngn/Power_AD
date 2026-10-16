#!/usr/bin/env python
"""Single-pass streaming profile of the smart-sign telemetry with River.

Every metric is streamed in time order, sample by sample, through River
estimators: fleet-level running stats and histogram sketches, per-series
(per-device) stats and quantiles, gap / duplicate / counter-reset / placeholder
counters, ADWIN drift detection and HalfSpaceTrees anomaly scoring on curated
metrics. Work is split into (metric, device-shard) tasks that run in parallel;
mergeable River objects (Mean, Var, Histogram) are combined in the parent and
one JSON per metric is written to analysis/out/profile/.

Usage: python -I analysis/profile_stream.py [--metrics m1 m2] [--workers N]
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import sys
import time
import zlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from river import anomaly, compose, drift, preprocessing, sketch, stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset/smart-sign-telemetry/prometheus"
OUT = ROOT / "analysis/out/profile"

SCRAPE_S = 120          # nominal scrape interval
GAP_S = 600             # a gap longer than this (5 missed scrapes) counts as an outage
DAY_MS = 86_400_000
LAG_DAY = 720           # samples per day at 120 s -> lag for daily autocorrelation
ROWS_PER_SHARD = 2_000_000

# Metrics that behave as running totals which fall back to 0 (README "Totals that reset").
COUNTERS = {
    "device_input_energy_wh_total", "device_pir_count", "device_ds_consumer_error_total",
    "device_wwan_rx_gb", "device_wwan_tx_gb",
}
BINARY = {"up", "device_wifi_hotspot_running", "device_deepstream_app_running", "device_mppt_load_state"}

# Placeholder / impossible values documented in the README; flagged and excluded from value stats.
PLACEHOLDER = {
    "device_gpu_temp_celsius": lambda v: v <= -200,
    "device_mppt_min_battery_temperature": lambda v: v >= 1000,
    "device_mppt_max_battery_temperature": lambda v: v >= 1000,
    "device_mppt_battery_temperature": lambda v: v >= 120,
    "device_mppt_battery_charge_current": lambda v: v == 100,
    "device_mppt_battery_voltage": lambda v: v == 100,
    "device_mppt_solar_voltage": lambda v: v == 100,
    "device_battery_remaining_capacity": lambda v: v > 100,
    "device_mppt_battery_remaining_capacity": lambda v: v > 100,
    "device_mppt_battery_remaining_capacity_mppt": lambda v: v > 100,
}

# Curated metrics for the heavier per-series analyses.
DRIFT = {
    "device_input_voltage_volts", "device_input_power_watts", "device_battery_remaining_capacity",
    "device_cpu_temp_celsius", "device_gpu_temp_celsius", "device_board_temp_celsius",
    "device_memory_free_gb", "device_storage_free_gb", "device_temperature", "device_humidity",
    "device_mppt_battery_voltage", "device_mppt_charge_power", "device_orin_power_watts",
    "device_cpu_gpu_power_watts",
}
ANOMALY = {
    "device_input_voltage_volts", "device_input_power_watts", "device_battery_remaining_capacity",
    "device_cpu_temp_celsius", "device_board_temp_celsius", "device_memory_free_gb",
}
SEASONAL = {
    "device_lightsensor", "device_mppt_solar_voltage", "device_mppt_charge_power", "device_temperature",
    "device_board_temp_celsius", "device_cpu_temp_celsius", "device_input_voltage_volts",
    "device_input_power_watts", "device_humidity", "device_battery_remaining_capacity",
}
DAILY = DRIFT | SEASONAL | {"up", "device_deepstream_app_running"}

GAP_EDGES = [150, GAP_S, 3600, DAY_MS // 1000]  # seconds: normal | missed scrapes | <1h | <1d | >=1d
GAP_NAMES = ["normal", "missed", "lt_1h", "lt_1d", "ge_1d"]


def gap_bucket(gap_s: float) -> int:
    for i, e in enumerate(GAP_EDGES):
        if gap_s <= e:
            return i
    return len(GAP_EDGES)


def sget(stat):
    """River .get() may return None before enough data; keep JSON clean."""
    v = stat.get()
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else v


class Series:
    """Streaming state for one (metric, device) series."""

    __slots__ = (
        "meta", "n", "nan", "inf", "flagged", "dup", "resets", "increase", "first", "last", "last_v",
        "gaps", "downtime", "mean", "var", "min", "max", "q", "ac", "adwin", "drifts",
        "det", "n_anom", "anoms", "daily",
    )

    def __init__(self, metric: str, meta: dict):
        self.meta = meta
        self.n = self.nan = self.inf = self.flagged = self.dup = self.resets = 0
        self.increase = 0.0
        self.first = self.last = self.last_v = None
        self.gaps = 0
        self.downtime = 0.0
        self.mean, self.var, self.min, self.max = stats.Mean(), stats.Var(), stats.Min(), stats.Max()
        self.q = [stats.Quantile(0.01), stats.Quantile(0.5), stats.Quantile(0.99)]
        self.ac = stats.AutoCorr(lag=LAG_DAY) if metric in SEASONAL else None
        self.adwin = drift.ADWIN(delta=0.002) if metric in DRIFT else None
        self.drifts = []            # (ts_ms, mean_before, mean_after)
        self.det = None
        if metric in ANOMALY:
            # HalfSpaceTrees expects features in [0, 1]; window = one day of samples.
            # Hour-of-day features let the trees learn the daily cycle instead of flagging it.
            hst = anomaly.HalfSpaceTrees(n_trees=10, height=6, window_size=LAG_DAY, seed=42)
            self.det = anomaly.QuantileFilter(compose.Pipeline(preprocessing.MinMaxScaler(), hst), q=0.995)
        self.n_anom = 0
        self.anoms = []             # heap of (score, ts_ms, value), top 50
        self.daily = defaultdict(stats.Mean) if metric in DAILY else None

    def to_dict(self) -> dict:
        d = {
            "meta": self.meta, "n": self.n, "nan": self.nan, "inf": self.inf, "flagged": self.flagged,
            "dup": self.dup, "resets": self.resets, "increase": self.increase,
            "first_ms": self.first, "last_ms": self.last, "gaps": self.gaps, "downtime_s": self.downtime,
            "mean": sget(self.mean), "var": sget(self.var), "min": sget(self.min), "max": sget(self.max),
            "p01": sget(self.q[0]), "p50": sget(self.q[1]), "p99": sget(self.q[2]),
        }
        if self.first is not None and self.last > self.first:
            d["coverage"] = self.n / ((self.last - self.first) / 1000 / SCRAPE_S + 1)
        if self.ac is not None:
            d["autocorr_1d"] = sget(self.ac)
        if self.adwin is not None:
            d["n_drifts"] = len(self.drifts)
            d["drifts"] = self.drifts[:200]
        if self.det is not None:
            d["n_anom"] = self.n_anom
            d["anoms"] = sorted(self.anoms, reverse=True)
        if self.daily is not None:
            d["daily"] = {day: [m.get(), m.n] for day, m in self.daily.items()}
        return d


def series_key(metric: str, lab: dict) -> tuple[str, dict]:
    # device_id changes letter case over some devices' lives (BP3235300111 -> bp3235300111), so normalize it
    dev = (lab.get("device_id") or "").lower() or lab.get("instance") or "?"
    job = lab.get("job", "")
    key = f"{dev}|{job}" if metric == "up" else dev
    meta = {k: lab[k] for k in ("device_id", "instance", "job", "agency", "bus_stop", "stop_code") if k in lab}
    if "device_id" in meta:
        meta["device_id"] = meta["device_id"].lower()
    return key, meta


def new_global() -> dict:
    return {
        "rows": 0, "nan": 0, "inf": 0, "flagged": 0, "dup": 0, "resets": 0,
        "mean": stats.Mean(), "var": stats.Var(), "min": stats.Min(), "max": stats.Max(),
        "hist": sketch.Histogram(max_bins=256),
        "gap_buckets": [0] * (len(GAP_EDGES) + 1),
        "hour_mean": [stats.Mean() for _ in range(24)],
        "day_mean": defaultdict(stats.Mean),
    }


def run_task(metric: str, shard: int, n_shards: int, files: list[str]) -> tuple[str, int, dict, dict, float]:
    t0 = time.time()
    g = new_global()
    series: dict[str, Series] = {}
    cache: dict[str, tuple | None] = {}
    flag = PLACEHOLDER.get(metric)
    is_counter = metric in COUNTERS
    hour_mean, day_mean, gap_buckets, hist = g["hour_mean"], g["day_mean"], g["gap_buckets"], g["hist"]
    g_mean, g_var, g_min, g_max = g["mean"], g["var"], g["min"], g["max"]

    def day_rows():
        """This shard's rows, one row group (= one UTC day) at a time, sorted by (series, time).
        Files are ordered by day then by label set, and one device can carry several label sets
        in a day (relabelled mid-day, or scraped by two targets), so file order is not time order."""
        for f in files:
            pf = pq.ParquetFile(f)
            for rg in range(pf.num_row_groups):
                tab = pf.read_row_group(rg, columns=["timestamp", "value", "labels"])
                ts = tab.column(0).cast(pa.int64()).to_numpy().tolist()
                vals = tab.column(1).to_numpy().tolist()
                labs = tab.column(2).to_pylist()
                rows = []
                for t, v, lab_s in zip(ts, vals, labs):
                    hit = cache.get(lab_s)
                    if hit is None:
                        if lab_s in cache:
                            continue  # belongs to another shard
                        key, meta = series_key(metric, json.loads(lab_s))
                        if zlib.crc32(key.encode()) % n_shards != shard:
                            cache[lab_s] = None
                            continue
                        hit = cache[lab_s] = (key, meta)
                    rows.append((hit[0], t, v, hit[1]))
                rows.sort(key=lambda r: (r[0], r[1]))
                yield from rows

    for key, t, v, meta in day_rows():
        s = series.get(key)
        if s is None:
            s = series[key] = Series(metric, meta)

        # --- time structure (all samples, regardless of value) ---
        g["rows"] += 1
        if s.last is not None and t <= s.last:
            # same timestamp twice (two scrape targets of one device), or a sample older
            # than one already seen across a day boundary: keep the first
            s.dup += 1
            g["dup"] += 1
            continue
        s.n += 1
        if s.last is None:
            s.first = t
        else:
            gap = (t - s.last) / 1000
            b = gap_bucket(gap)
            gap_buckets[b] += 1
            if gap > GAP_S:
                s.gaps += 1
                s.downtime += gap
        s.last = t

        # --- value quality ---
        if v != v:
            s.nan += 1
            g["nan"] += 1
            continue
        if v in (math.inf, -math.inf):
            s.inf += 1
            g["inf"] += 1
            continue
        if flag is not None and flag(v):
            s.flagged += 1
            g["flagged"] += 1
            continue
        if is_counter and s.last_v is not None:
            if v < s.last_v:
                s.resets += 1
                g["resets"] += 1
            else:
                s.increase += v - s.last_v
        s.last_v = v

        # --- value statistics ---
        g_mean.update(v); g_var.update(v); g_min.update(v); g_max.update(v); hist.update(v)
        hour_mean[(t // 3_600_000) % 24].update(v)
        day = t // DAY_MS
        day_mean[day].update(v)
        s.mean.update(v); s.var.update(v); s.min.update(v); s.max.update(v)
        for q in s.q:
            q.update(v)
        if s.ac is not None:
            s.ac.update(v)
        if s.daily is not None:
            s.daily[day].update(v)

        # --- dynamics ---
        if s.adwin is not None:
            before = s.adwin.estimation
            s.adwin.update(v)
            if s.adwin.drift_detected:
                s.drifts.append((t, before, s.adwin.estimation))
        if s.det is not None:
            h = (t % DAY_MS) / DAY_MS * 2 * math.pi
            x = {"v": v, "hs": math.sin(h), "hc": math.cos(h)}
            score = s.det.score_one(x)
            if s.det.classify(score):
                s.n_anom += 1
                if len(s.anoms) < 50:
                    heapq.heappush(s.anoms, (score, t, v))
                elif score > s.anoms[0][0]:
                    heapq.heapreplace(s.anoms, (score, t, v))
            s.det.learn_one(x)

    out_series = {k: s.to_dict() for k, s in series.items()}
    g["day_mean"] = dict(day_mean)
    return metric, shard, g, out_series, time.time() - t0


def merge_global(acc: dict | None, g: dict) -> dict:
    if acc is None:
        return g
    for k in ("rows", "nan", "inf", "flagged", "dup", "resets"):
        acc[k] += g[k]
    acc["mean"] += g["mean"]
    acc["var"] += g["var"]
    for k in ("min", "max"):
        v = g[k].get()
        if v is not None and math.isfinite(v):  # Min/Max start at +-inf before any update
            acc[k].update(v)
    acc["hist"] += g["hist"]
    acc["gap_buckets"] = [a + b for a, b in zip(acc["gap_buckets"], g["gap_buckets"])]
    for a, b in zip(acc["hour_mean"], g["hour_mean"]):
        a += b
    for day, m in g["day_mean"].items():
        if day in acc["day_mean"]:
            acc["day_mean"][day] += m
        else:
            acc["day_mean"][day] = m
    return acc


def hist_quantiles(hist: sketch.Histogram, qs: list[float]) -> list[float | None]:
    """Approximate quantiles from a River histogram sketch (linear within bins)."""
    bins = [(b.left, b.right, b.count) for b in hist]
    total = sum(c for _, _, c in bins)
    if not total:
        return [None] * len(qs)
    out = []
    for q in qs:
        target, cum = q * total, 0.0
        for left, right, c in bins:
            if cum + c >= target:
                frac = (target - cum) / c if c else 0
                out.append(left + (right - left) * frac)
                break
            cum += c
        else:
            out.append(bins[-1][1])
    return out


def finalize(metric: str, g: dict, series: dict) -> dict:
    day0 = min(g["day_mean"]) if g["day_mean"] else 0
    p01, p05, p25, p50, p75, p95, p99 = hist_quantiles(g["hist"], [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "metric": metric,
        "type": "counter" if metric in COUNTERS else "binary" if metric in BINARY else "gauge",
        "rows": g["rows"], "nan": g["nan"], "inf": g["inf"], "flagged": g["flagged"], "dup": g["dup"],
        "resets": g["resets"], "n_series": len(series),
        "mean": sget(g["mean"]), "var": sget(g["var"]), "min": sget(g["min"]), "max": sget(g["max"]),
        "p01": p01, "p05": p05, "p25": p25, "p50": p50, "p75": p75, "p95": p95, "p99": p99,
        "hist": [(b.left, b.right, b.count) for b in g["hist"]],
        "gap_buckets": dict(zip(GAP_NAMES, g["gap_buckets"])),
        "hour_mean": [sget(m) for m in g["hour_mean"]],
        "daily": {time.strftime("%Y-%m-%d", time.gmtime(d * 86400)): [m.get(), m.n]
                  for d, m in sorted(g["day_mean"].items())},
        "series": series,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", nargs="*", help="subset of metric names (default: all)")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 4))
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    metrics = sorted(p.name.split("=", 1)[1] for p in DATA.glob("metric=*"))
    if args.metrics:
        metrics = [m for m in metrics if m in set(args.metrics)]

    tasks = []
    for m in metrics:
        files = sorted(str(p) for p in DATA.glob(f"metric={m}/month=*/part-000.parquet"))
        rows = sum(pq.read_metadata(f).num_rows for f in files)
        n_shards = max(1, math.ceil(rows / ROWS_PER_SHARD))
        if m in ANOMALY:
            n_shards *= 2  # HalfSpaceTrees is ~10x the per-row cost of the stats
        tasks += [(rows, m, k, n_shards, files) for k in range(n_shards)]
    tasks.sort(reverse=True)  # longest first
    print(f"{len(metrics)} metrics, {len(tasks)} tasks, {args.workers} workers", flush=True)

    t0 = time.time()
    acc_g: dict[str, dict | None] = {m: None for m in metrics}
    acc_s: dict[str, dict] = {m: {} for m in metrics}
    pending = {m: sum(1 for t in tasks if t[1] == m) for m in metrics}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_task, m, k, n, files) for _, m, k, n, files in tasks]
        for fut in as_completed(futs):
            m, k, g, s, dt = fut.result()
            acc_g[m] = merge_global(acc_g[m], g)
            acc_s[m].update(s)
            pending[m] -= 1
            print(f"  {m} shard {k} done in {dt:.0f}s ({g['rows']:,} rows)", flush=True)
            if pending[m] == 0:
                res = finalize(m, acc_g[m], acc_s[m])
                (OUT / f"{m}.json").write_text(json.dumps(res))
                print(f"* {m}: {res['rows']:,} rows, {res['n_series']} series -> written", flush=True)
                acc_g[m] = None
                acc_s[m] = {}
    print(f"all done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

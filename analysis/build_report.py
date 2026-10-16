#!/usr/bin/env python
"""Condense analysis/out/profile/*.json + cross.json into report_data.json and render the HTML report.

The page template lives in analysis/report/template.html; the data is inlined at the
__DATA__ marker and the result written to analysis/report/index.html.

Usage: python -I analysis/build_report.py
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/out"
PROF = OUT / "profile"
REPORT = ROOT / "analysis/report"
DAY_S = 86400

# Display metadata: group, unit, short meaning (from the dataset README; units are the README's reading).
META = {
    "up": ("Reachability", "0/1", "Prometheus could scrape the target"),
    "device_cpu_gpu_power_watts": ("Power", "W", "CPU + GPU power draw (heartbeat)"),
    "device_input_current_amperes": ("Power", "A", "Current at the power input (INA238)"),
    "device_input_energy_wh_total": ("Power", "Wh", "Energy used, integrated on device; resets daily"),
    "device_input_power_watts": ("Power", "W", "Power at the device input (INA238)"),
    "device_input_voltage_volts": ("Power", "V", "Voltage at the power input; 12 V and 56 V populations"),
    "device_orin_power_watts": ("Power", "W", "Jetson Orin module power (INA3221)"),
    "device_power": ("Power", "W", "Device power per heartbeat service (undocumented definition)"),
    "device_battery_remaining_capacity": ("Battery / solar", "%", "Battery state of charge (power manager)"),
    "device_input_soc_poly5_percent": ("Battery / solar", "%", "SoC estimated from input voltage (rough)"),
    "device_mppt_battery_charge_current": ("Battery / solar", "A", "Current into battery from charge controller"),
    "device_mppt_battery_remaining_capacity": ("Battery / solar", "%", "SoC reported by charge controller"),
    "device_mppt_battery_remaining_capacity_mppt": ("Battery / solar", "%", "Same SoC under a second variable name"),
    "device_mppt_battery_voltage": ("Battery / solar", "V", "Battery voltage at charge controller"),
    "device_mppt_charge_power": ("Battery / solar", "W", "Charge power into battery (0 at night)"),
    "device_mppt_load_current": ("Battery / solar", "A", "Load current on controller output"),
    "device_mppt_load_power": ("Battery / solar", "W", "Load power on controller output"),
    "device_mppt_load_state": ("Battery / solar", "0/1", "Controller load output on/off"),
    "device_mppt_load_voltage": ("Battery / solar", "V", "Load voltage on controller output"),
    "device_mppt_solar_current": ("Battery / solar", "A", "Solar panel current (0 at night)"),
    "device_mppt_solar_voltage": ("Battery / solar", "V", "Solar panel voltage (0 at night)"),
    "device_mppt_total_load_operation_time": ("Battery / solar", "h?", "Cumulative load-on time"),
    "device_board_temp_celsius": ("Temperature", "°C", "Board temperature (INA238)"),
    "device_cpu_temp_celsius": ("Temperature", "°C", "CPU temperature"),
    "device_gpu_temp_celsius": ("Temperature", "°C", "GPU temperature; -256 is a sensor error"),
    "device_mppt_battery_temperature": ("Temperature", "°C", "Battery temperature (controller sensor)"),
    "device_mppt_controller_temperature": ("Temperature", "°C", "Charge controller temperature"),
    "device_mppt_max_battery_temperature": ("Temperature", "°C", "Max battery temperature recorded"),
    "device_mppt_min_battery_temperature": ("Temperature", "°C", "Min battery temperature; 3277x is a placeholder"),
    "device_temperature": ("Temperature", "°C", "Heater / environment sensor temperature"),
    "device_memory_free_gb": ("Compute / storage", "GB", "Free RAM"),
    "device_memory_total_gb": ("Compute / storage", "GB", "Total RAM"),
    "device_storage_free_gb": ("Compute / storage", "GB", "Free disk"),
    "device_storage_total_gb": ("Compute / storage", "GB", "Total disk"),
    "device_humidity": ("Sensors", "% RH", "Relative humidity"),
    "device_lightsensor": ("Sensors", "lux", "Ambient light"),
    "device_pir_count": ("Sensors", "count", "PIR motion detections; resets daily"),
    "device_wifi_hotspot_running": ("Connectivity", "0/1", "Wi-Fi hotspot service active"),
    "device_wwan_rx_gb": ("Connectivity", "GB", "Cellular data received; restarts daily"),
    "device_wwan_tx_gb": ("Connectivity", "GB", "Cellular data sent; restarts daily"),
    "device_deepstream_app_running": ("Software", "0/1", "DeepStream video-analytics process running"),
    "device_ds_consumer_error_total": ("Software", "count", "DeepStream consumer errors in log; resets daily"),
}
GROUP_ORDER = ["Reachability", "Power", "Battery / solar", "Temperature", "Compute / storage", "Sensors",
               "Connectivity", "Software"]


def short(m: str) -> str:
    return m.removeprefix("device_")


def day_str(day_idx: int) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(day_idx * DAY_S))


def merge_centroids(bins: list, max_n: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """River's Histogram += only concatenates the shard bins (they overlap and are unsorted).
    Reduce them to weighted centroids and merge the closest neighbours (Ben-Haim & Tom-Tov)."""
    pts = sorted(((l + r) / 2, c) for l, r, c in bins if c > 0)
    xs = np.array([x for x, _ in pts], dtype=float)
    cs = np.array([c for _, c in pts], dtype=float)
    while len(xs) > max_n:
        i = int(np.argmin(np.diff(xs)))
        x = (xs[i] * cs[i] + xs[i + 1] * cs[i + 1]) / (cs[i] + cs[i + 1])
        xs = np.concatenate([xs[:i], [x], xs[i + 2:]])
        cs = np.concatenate([cs[:i], [cs[i] + cs[i + 1]], cs[i + 2:]])
    return xs, cs


def centroid_quantiles(xs: np.ndarray, cs: np.ndarray, qs: list[float]) -> list[float | None]:
    if not len(xs):
        return [None] * len(qs)
    cum = np.cumsum(cs) - cs / 2  # F at each centroid
    return [float(np.interp(q * cs.sum(), cum, xs)) for q in qs]


def display_hist(xs: np.ndarray, cs: np.ndarray, p01: float, p99: float, vmin: float, vmax: float,
                 is_binary: bool, nbins: int = 60) -> dict:
    """Equal-width bins over the bulk of the distribution, with overflow counts outside it."""
    if is_binary:
        return {"lo": 0, "hi": 1, "counts": [float(cs[xs < 0.5].sum()), float(cs[xs >= 0.5].sum())],
                "below": 0, "above": 0, "binary": True}
    pad = 0.05 * (p99 - p01)
    lo, hi = max(vmin, p01 - pad), min(vmax, p99 + pad)
    if hi <= lo:
        lo, hi = vmin, vmax if vmax > vmin else vmin + 1
    idx = np.floor((xs - lo) / (hi - lo) * nbins).astype(int)
    counts = [0.0] * nbins
    below = above = 0.0
    for i, c in zip(idx, cs):
        if i < 0:
            below += c
        elif i >= nbins:
            if xs[idx == i].size and i == nbins and hi == vmax:
                counts[-1] += c  # value exactly at the top edge
            else:
                above += c
        else:
            counts[i] += c
    return {"lo": round(lo, 6), "hi": round(hi, 6), "counts": counts, "below": below, "above": above, "binary": False}


def seasonality_from_hourly() -> dict:
    """Lag-24 h autocorrelation per device from the hourly table (robust to gaps, unlike
    the raw lag-720-sample statistic); returns per metric the per-device values."""
    import pandas as pd
    h = pd.read_parquet(OUT / "hourly.parquet")
    out = {}
    for m, g in h.groupby("metric"):
        vals = []
        for dev, s in g.groupby("device"):
            x = s.set_index("hour")["mean"].resample("h").mean()
            if x.notna().sum() >= 24 * 14:
                a = x.autocorr(24)
                if a == a:
                    vals.append([dev, round(float(a), 3)])
        out[m] = vals
    return out


def r(x, nd=4):
    if x is None:
        return None
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return round(x, nd)


def main() -> None:
    cross = json.loads((OUT / "cross.json").read_text())
    profiles = {}
    for f in sorted(PROF.glob("*.json")):
        p = json.loads(f.read_text())
        profiles[p["metric"]] = p
    print(f"{len(profiles)} profiles, cross generated {cross['generated']}")
    seasonal = seasonality_from_hourly()
    spec_path = OUT / "spectral.json"
    spectral = json.loads(spec_path.read_text()) if spec_path.exists() else {"metrics": {}}
    spec_of = lambda s: spectral["metrics"].get(s) or spectral["metrics"].get(s + "_rate")  # counters are rates there

    def agency_norm(a):  # label values change case over a device's life (BalHarbour / Balharbour, DEMO / demo)
        return {"balharbour": "BalHarbour", "demo": "demo"}.get(a.lower(), a) if a else None

    devices: dict[str, dict] = defaultdict(lambda: {"agency": None, "first": None, "last": None, "metrics": 0})
    metrics_out = []
    for name in sorted(profiles, key=lambda m: (GROUP_ORDER.index(META[m][0]), m)):
        p = profiles[name]
        group, unit, meaning = META[name]
        series = p["series"]
        rows_summary = []
        band = defaultdict(list)
        for key, s in series.items():
            dev = s["meta"].get("device_id") or key.split("|")[0]
            if name == "up" and s["meta"].get("job") != "edge":
                continue  # non-device scrape targets are summarized separately
            d = devices[dev]
            d["agency"] = d["agency"] or agency_norm(s["meta"].get("agency"))
            d["first"] = min(filter(None, [d["first"], s["first_ms"]]))
            d["last"] = max(filter(None, [d["last"], s["last_ms"]]))
            d["metrics"] += 1
            rows_summary.append({
                "dev": dev, "agency": agency_norm(s["meta"].get("agency")), "n": s["n"], "mean": r(s["mean"]),
                "p01": r(s["p01"]), "p50": r(s["p50"]), "p99": r(s["p99"]), "min": r(s["min"]), "max": r(s["max"]),
                "std": r(math.sqrt(s["var"])) if s.get("var") else None,
                "coverage": r(s.get("coverage")), "gaps": s["gaps"], "downtime_h": r(s["downtime_s"] / 3600, 1),
                "flagged": s["flagged"], "nan": s["nan"], "dup": s["dup"], "resets": s["resets"],
                "increase": r(s["increase"], 2), "drifts": s.get("n_drifts"), "anom": s.get("n_anom"),
                "ac1d": r(s.get("autocorr_1d"), 3),
                "first": day_str(s["first_ms"] // 1000 // DAY_S), "last": day_str(s["last_ms"] // 1000 // DAY_S),
            })
            if s.get("daily"):
                for di, (m, n) in s["daily"].items():
                    if n >= 60 and m is not None:  # at least ~2 h of samples that day
                        band[int(di)].append(m)
        band_out = []
        for di in sorted(band):
            v = np.array(band[di])
            band_out.append([day_str(di), r(float(np.percentile(v, 10)), 3), r(float(np.median(v)), 3),
                             r(float(np.percentile(v, 90)), 3), len(v)])
        # up: non-device targets
        other_up = None
        if name == "up":
            other_up = defaultdict(lambda: {"n": 0, "up": 0.0, "series": 0})
            for key, s in series.items():
                if s["meta"].get("job") != "edge":
                    j = s["meta"].get("job", "?")
                    other_up[j]["n"] += s["n"]
                    other_up[j]["up"] += (s["mean"] or 0) * s["n"]
                    other_up[j]["series"] += 1
            other_up = {j: {"series": v["series"], "n": v["n"], "mean": r(v["up"] / v["n"]) if v["n"] else None}
                        for j, v in other_up.items()}
        # top anomalies / drifts for the examples
        anoms = []
        for key, s in series.items():
            for sc, ts, v in s.get("anoms", [])[:5]:
                anoms.append([r(sc, 3), ts, r(v, 3), s["meta"].get("device_id") or key])
        anoms.sort(reverse=True)
        xs, cs = merge_centroids(p["hist"])
        p01, p05, p25, p50, p75, p95, p99 = centroid_quantiles(xs, cs, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        anoms_by_dev, seen = [], set()
        for a in anoms:  # one example per device, highest score first
            if a[3] not in seen:
                seen.add(a[3])
                anoms_by_dev.append(a)
        metrics_out.append({
            "name": name, "short": short(name), "group": group, "unit": unit, "meaning": meaning, "type": p["type"],
            "rows": p["rows"], "n_series": len(rows_summary), "mean": r(p["mean"]),
            "std": r(math.sqrt(p["var"])) if p["var"] else None, "min": r(p["min"]), "max": r(p["max"]),
            "p01": r(p01), "p05": r(p05), "p25": r(p25), "p50": r(p50), "p75": r(p75), "p95": r(p95), "p99": r(p99),
            "hist": display_hist(xs, cs, p01, p99, p["min"], p["max"], p["type"] == "binary"),
            "hour_mean": [r(x) for x in p["hour_mean"]],
            "season24": seasonal.get(short(name), []),
            "spectrum": spec_of(short(name)),
            "daily": [[d, r(m), n] for d, (m, n) in p["daily"].items()],
            "quality": {"nan": p["nan"], "inf": p["inf"], "flagged": p["flagged"], "dup": p["dup"],
                        "resets": p["resets"], "gaps": p["gap_buckets"]},
            "series": sorted(rows_summary, key=lambda x: x["dev"]),
            "band": band_out, "other_up": other_up, "anoms": anoms_by_dev[:8],
        })

    # ---- devices directory
    avail = cross["availability"]["per_device"]
    cluster_of = {d: c for c, info in cross["clusters"].items() for d in info["devices"]}
    fp = {row["device"]: row for row in cross["fingerprints"]}
    dev_meta = cross["devices"]
    devices_out = []
    for dev, d in sorted(devices.items()):
        devices_out.append({
            "id": dev, "agency": d["agency"] or agency_norm(dev_meta.get(dev, {}).get("agency")),
            "stop": dev_meta.get(dev, {}).get("stop_code"),
            "first": day_str(d["first"] // 1000 // DAY_S), "last": day_str(d["last"] // 1000 // DAY_S),
            "days": round((d["last"] - d["first"]) / 1000 / DAY_S, 1), "metrics": d["metrics"],
            "uptime": r(avail.get(dev)), "cluster": cluster_of.get(dev),
            "hours": int(fp[dev]["hours"]) if dev in fp else None,
            "fp": {k: r(v, 3) for k, v in fp[dev].items() if k not in ("device", "cluster", "hours")} if dev in fp else None,
        })
    agencies = Counter(d["agency"] for d in devices_out)

    # ---- correlation matrix
    names = cross["metrics"]
    corr = cross["corr"]
    matrix = [[corr.get(f"{a}|{b}") for b in names] for a in names]

    # ---- regime changes & trends
    scale = {m["short"]: (m["p99"] or 0) - (m["p01"] or 0) or 1 for m in metrics_out}
    regimes = cross["regime_changes"]
    reg_by_metric = Counter(c["metric"] for c in regimes)
    reg_weekly = defaultdict(Counter)
    for c in regimes:
        wk = time.strftime("%Y-%m-%d", time.gmtime((int(time.mktime(time.strptime(c["day"], "%Y-%m-%d"))) // (7 * DAY_S)) * 7 * DAY_S))
        reg_weekly[c["metric"]][wk] += 1
    biggest = sorted(regimes, key=lambda c: -abs(c["after"] - c["before"]) / scale.get(c["metric"], 1))[:40]
    same_day = Counter((c["metric"], c["day"]) for c in regimes)  # fleet-wide events: many devices, one day
    same_day_top = [{"metric": m, "day": d, "n": n, "devices": sorted(c["device"] for c in regimes if c["metric"] == m and c["day"] == d)}
                    for (m, d), n in same_day.most_common(12) if n >= 3]
    trends = cross["trends"]
    trend_top = defaultdict(list)
    for t in trends:
        t = dict(t)
        t["rel"] = r(t["slope_per_30d"] / scale.get(t["metric"], 1), 4)
        trend_top[t["metric"]].append(t)
    trend_out = {}
    for m, lst in trend_top.items():
        lst.sort(key=lambda t: t["slope_per_30d"])
        trend_out[m] = {"n": len(lst), "median_slope_30d": r(float(np.median([t["slope_per_30d"] for t in lst]))),
                        "falling": [{k: r(v) if isinstance(v, float) else v for k, v in t.items()} for t in lst[:6]],
                        "rising": [{k: r(v) if isinstance(v, float) else v for k, v in t.items()} for t in lst[-6:][::-1]],
                        "all": [[t["device"], r(t["slope_per_30d"]), t["days"]] for t in lst]}

    data = {
        "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "overview": {
            "rows": sum(p["rows"] for p in profiles.values()), "metrics": len(profiles), "devices": len(devices_out),
            "agencies": dict(agencies.most_common()), "first_day": min(d["first"] for d in devices_out),
            "last_day": max(d["last"] for d in devices_out), "hourly_rows": cross["hourly_rows"],
            "flagged": sum(p["flagged"] for p in profiles.values()), "nan": sum(p["nan"] for p in profiles.values()),
            "dup": sum(p["dup"] for p in profiles.values()),
        },
        "metrics": metrics_out, "devices": devices_out,
        "availability": cross["availability"],
        "corr": {"names": names, "matrix": matrix}, "pairs": cross["pairs"], "clusters": cross["clusters"],
        "regimes": {"by_metric": dict(reg_by_metric), "weekly": {m: dict(c) for m, c in reg_weekly.items()},
                    "biggest": biggest, "total": len(regimes), "same_day": same_day_top},
        "trends": trend_out, "forecasts": cross["forecasts"],
        "spectral": {k: spectral[k] for k in ("edges_h", "centers_h", "min_hours", "targets") if k in spectral},
    }
    (OUT / "report_data.json").write_text(json.dumps(data))
    size = (OUT / "report_data.json").stat().st_size
    print(f"report_data.json: {size / 1e6:.2f} MB")

    tpl = REPORT / "template.html"
    if tpl.exists():
        html = tpl.read_text().replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
        (REPORT / "index.html").write_text(html)
        print(f"index.html: {len(html) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()

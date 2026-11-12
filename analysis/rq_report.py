#!/usr/bin/env python
"""Render the research-question report (RQ1 normal-behaviour modelling, RQ2 graph-based anomaly detection).

Reads analysis/out/rq1.json, rq2.json and the example parquet files, condenses them, and writes
analysis/report/rq.html from analysis/report/rq_template.html with the shared chart kit inlined (the CSS and JS
between the KIT markers in analysis/report/template.html, as influence_report.py does).

Usage: python -I analysis/rq_report.py
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/out"
REPORT = ROOT / "analysis/report"


def between(text: str, start: str, end: str) -> str:
    m = re.search(re.escape(start) + r".*?\n(.*?)" + re.escape(end), text, re.S)
    if not m:
        raise SystemExit(f"marker {start!r} … {end!r} not found in template.html")
    return m.group(1)


def rnd(o, d=4):
    if isinstance(o, dict):
        return {k: rnd(v, d) for k, v in o.items()}
    if isinstance(o, list):
        return [rnd(v, d) for v in o]
    if isinstance(o, float):
        return round(o, d)
    return o


def rq1_data() -> dict:
    r = json.loads((OUT / "rq1.json").read_text())
    s1 = {y: {k: v for k, v in m.items() if not k.startswith("_")} for y, m in r["s1"].items()}
    per_dev = {y: r["s1"][y]["_per_device"] for y in r["targets"]}
    per_dev_med = {y: {k: float(np.median(list(v.values()))) for k, v in d.items()} for y, d in per_dev.items()}
    cov = {}
    for y, c in r["s1_coverage"].items():
        cov[y] = {"raw": {"coverage": c["coverage"], "width": c["width_median"], "by": c["by"]},
                  **{k: {"coverage": v["coverage"], "width": v["width_median"], "by": v["by"],
                         "devices_below_80": int(sum(x < 0.8 for x in v["per_device_coverage"].values())),
                         "devices": len(v["per_device_coverage"])}
                     for k, v in c["conformal"].items()}}
    s2 = {y: {k: {kk: vv for kk, vv in v.items() if kk != "per_device_mae"} for k, v in m.items()} for y, m in r["s2"].items()}
    ex = pd.read_parquet(OUT / "rq1_examples.parquet")
    examples = {}
    for (dev, y), g in ex.groupby(["device", "target"]):
        g = g.sort_values("time")
        examples.setdefault(dev, {})[y] = {
            "t": (g["time"].astype("int64") // 10**6).tolist(), "y": g["y"].round(3).tolist(),
            "q05": g["q05"].round(3).tolist(), "q50": g["q50"].round(3).tolist(), "q95": g["q95"].round(3).tolist(),
            "pred": g["pred"].round(3).tolist()}
    return rnd({"targets": r["targets"], "label": r["label"], "unit": r["unit"], "ladder": r["ladder"], "drop": r["drop"],
                "groups": r["feature_groups"], "s1": s1, "per_device_median": per_dev_med, "strata": r["s1_strata"],
                "coverage": cov, "importance": r["s1_importance"], "s2": s2, "s3": r["s3"], "examples": examples,
                "s1_devices": r["s1_devices"], "split": r["split"]})


def rq2_data() -> dict:
    r = json.loads((OUT / "rq2.json").read_text())
    ep = pd.read_parquet(OUT / "rq2_episodes.parquet")
    by_fault = {}
    for w, e in ep.groupby("variant"):
        rec = e.pivot_table(index=["fault", "sev"], columns="detector", values="detected", aggfunc="mean")
        loc = e.assign(loc_ok=e["loc_ok"].astype(float)).pivot_table(index=["fault", "sev"], columns="detector",
                                                                      values="loc_ok", aggfunc="mean")
        by_fault[str(w)] = {"recall": {f"{f}|{s}": row.dropna().to_dict() for (f, s), row in rec.iterrows()},
                            "loc": {f"{f}|{s}": row.dropna().to_dict() for (f, s), row in loc.iterrows()}}
    ex = {}
    p = OUT / "rq2_examples.parquet"
    if p.exists():
        x = pd.read_parquet(p)
        for (f, det), g in x.groupby(["fault", "detector"]):
            nodes = {}
            for n, gg in g.groupby("node"):
                gg = gg.sort_values("time")
                nodes[n] = {"z": gg["z"].round(2).tolist(), "y": gg["y"].round(3).tolist(), "y0": gg["y0"].round(3).tolist()}
            g0 = g[g["node"] == g["node"].iloc[0]].sort_values("time")
            ex.setdefault(f, {"device": g["device"].iloc[0], "t": (g0["time"].astype("int64") // 10**6).tolist(),
                              "in": g0["in_episode"].astype(int).tolist(), "det": {}})["det"][det] = nodes
    keep = ["devices", "nodes", "label", "graphs", "hand_why", "faults", "coef", "far", "episodes_per_device",
            "results", "summary_energy", "bootstrap_vs_hand", "calibration_scale", "real_pvi", "rows", "windows",
            "thresholds_median", "granger_F", "notears_W", "corr"]
    d = {k: r[k] for k in keep if k in r}
    d["clean_alarms"] = {k: {kk: vv for kk, vv in v.items() if kk != "per_day"} for k, v in r["clean_alarms"].items()}
    d["by_fault"] = by_fault
    d["examples"] = ex
    return rnd(d)


def main() -> None:
    main_tpl = (REPORT / "template.html").read_text()
    kit_css = between(main_tpl, "/* KIT-CSS-START", "/* KIT-CSS-END */")
    kit_js = between(main_tpl, "// KIT-START", "// KIT-END")
    data = {"generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "metrics": [],
            "rq1": rq1_data(), "rq2": rq2_data()}
    page = (REPORT / "rq_template.html").read_text()
    page = page.replace("__KIT_CSS__", kit_css).replace("__KIT_JS__", kit_js)
    page = page.replace("__DATA__", json.dumps(data, allow_nan=False).replace("</", "<\\/"))
    (REPORT / "rq.html").write_text(page)
    print(f"rq.html: {len(page) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()

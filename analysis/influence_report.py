#!/usr/bin/env python
"""Render the feature-influence report from analysis/out/influence.json.

The page reuses the main report's chart kit: the CSS and JS between the KIT markers in
analysis/report/template.html are inlined into analysis/report/influence_template.html,
together with the data, and written to analysis/report/influence.html.

Usage: python -I analysis/influence_report.py
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/out"
REPORT = ROOT / "analysis/report"

LABELS = {
    "input_voltage_volts": "Input voltage", "input_power_watts": "Input power", "orin_power_watts": "Orin power",
    "cpu_temp_celsius": "CPU temp", "board_temp_celsius": "Board temp", "temperature": "Enclosure temp",
    "humidity": "Humidity", "lightsensor": "Light", "memory_free_gb": "Free RAM", "storage_free_gb": "Free disk",
    "battery_remaining_capacity": "Battery SoC", "mppt_charge_power": "Solar charge power",
    "deepstream_app_running": "DeepStream running",
}


def between(text: str, start: str, end: str) -> str:
    m = re.search(re.escape(start) + r".*?\n(.*?)" + re.escape(end), text, re.S)
    if not m:
        raise SystemExit(f"marker {start!r} … {end!r} not found in template.html")
    return m.group(1)


def main() -> None:
    main_tpl = (REPORT / "template.html").read_text()
    kit_css = between(main_tpl, "/* KIT-CSS-START", "/* KIT-CSS-END */")
    kit_js = between(main_tpl, "// KIT-START", "// KIT-END")
    inf = json.loads((OUT / "influence.json").read_text())
    inf.pop("per_device", None)  # large and not shown; stays in influence.json
    data = {"generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "metrics": [], "labels": LABELS, "inf": inf}
    page = (REPORT / "influence_template.html").read_text()
    page = page.replace("__KIT_CSS__", kit_css).replace("__KIT_JS__", kit_js)
    page = page.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    (REPORT / "influence.html").write_text(page)
    print(f"influence.html: {len(page) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()

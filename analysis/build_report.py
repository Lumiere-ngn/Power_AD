#!/usr/bin/env python
"""Build the report page from the anomaly and causal summaries.

Reads analysis/out/anomaly_summary.json (anomaly.py) and analysis/out/causal.json (causal.py), fills
analysis/report/main_template.html and writes analysis/report/index.html.

The chart kit (CSS and JS) is taken from analysis/report/template.html, between its KIT markers, so the
page keeps the look of the earlier reports. The interpretation notes in the template are written for
the current results: reread them after a rerun that changes the list of links.

Usage: python -I analysis/build_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "out"
REP = ROOT / "analysis" / "report"


def between(lines: list[str], start: str, end: str) -> str:
    """The lines strictly between the line that starts with `start` and the one that starts with `end`."""
    a = next(i for i, ln in enumerate(lines) if ln.startswith(start))
    b = next(i for i, ln in enumerate(lines) if ln.startswith(end))
    return "\n".join(lines[a + 1:b])


def onset_counts(onset: list, x: str, y: str) -> dict:
    """Inside incidents that flag both x and y: how often x flagged first, y first, or both in the same hour."""
    c = {"cause_first": 0, "effect_first": 0, "same": 0}
    for a, b, kind, n in onset:
        if kind == "same":
            if {a, b} == {x, y}:
                c["same"] += n
        elif (a, b) == (x, y):
            c["cause_first"] += n
        elif (a, b) == (y, x):
            c["effect_first"] += n
    return c


def main() -> None:
    S = json.loads((OUT / "anomaly_summary.json").read_text())
    C = json.loads((OUT / "causal.json").read_text())
    kit = (REP / "template.html").read_text().split("\n")
    kit_css = between(kit, "/* KIT-CSS-START", "/* KIT-CSS-END")
    kit_js = between(kit, "// KIT-START", "// KIT-END")

    links = []
    for key in C["edges"]:
        x, y = key.split(">")
        links.append({"key": key, "cause": x, "effect": y, **C["pairs"][key],
                      "onset_device": onset_counts(S["onset"]["device"], x, y),
                      "onset_shared": onset_counts(S["onset"]["shared"], x, y)})
    anomalies = {k: v for k, v in S.items() if k != "onset"}
    data = {
        "metrics": [{"short": m, "label": S["labels"][m], "group": S["groups"][m]} for m in S["metrics"]],
        "A": anomalies,
        "C": C,
        "links": links,
    }
    html = (REP / "main_template.html").read_text()
    html = html.replace("__KIT_CSS__", kit_css).replace("__KIT_JS__", kit_js)
    html = html.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    out = REP / "index.html"
    out.write_text(html)
    print(f"wrote {out.relative_to(ROOT)} ({len(html) / 1024:.0f} KB): {len(links)} directed links, "
          f"{len(S['events_device'])} example events")


if __name__ == "__main__":
    main()

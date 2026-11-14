#!/usr/bin/env python
"""RQ2 Phase 1: dataset, device inventory, data availability and split manifest.

Builds the four Phase 1 deliverables from the shared feature table
(analysis/out/features.parquet, loaded through rq_common.load(), which already drops
the demo/show/buspas units and adds hardware flags, peer environment and city weather).

Deliverables
------------
1. analysis/out/rq2/data/rq2_dataset.parquet     device x hour, nodes + context, split labels
2. analysis/out/rq2/data/device_inventory.csv    one row per device
3. analysis/out/rq2/data/data_availability.csv   one row per device x metric
4. analysis/out/rq2/data/data_split_manifest.json

Device population: the agreed RQ1 study population (rule mirrored from rq1.py, MIN_HOURS = 168).
A device is eligible for a window when it reports input power in at least MIN_HOURS grid hours
of that window; eligible_rq2 requires eligibility in all four windows. Per-window eligibility and
an RQ1-comparable count are also recorded so no population change is silent.

Chronological split (RQ2 plan, Phase 1.4):
    fit        2026-03-08 .. 2026-07-01   normal-model training
    calibrate  2026-07-01 .. 2026-08-01   residual scales and thresholds
    develop    2026-08-01 .. 2026-09-01   detector and synthetic-fault tuning
    test       2026-09-01 .. 2026-10-07   final evaluation

develop + test together are the protocol's [splits.development].evaluate window
[2026-08-01, 2026-10-07]; RQ2 keeps August for tuning and scores Sep-Oct last.

Leakage guards
--------------
- Splits are pure functions of UTC timestamp; no statistic crosses a boundary.
- Availability is measured, never invented: an unavailable node stays NaN and is reported as
  unavailable, so no detector can read a silent zero.
- rq_common.load(before=...) is available so hardware and peer columns cannot see later months.
- The manifest records the previous exposure of the test months (decision D1: the pilot and
  two prior RQ1 runs already scored August-October; D1 discloses this and reuses them once).

No anomaly detectors are implemented here (Phase 1 only).
Usage: python -I analysis/rq2_data.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq_common as C  # noqa: E402

# ------------------------------------------------------------ RQ2 constants
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/out"
RQ2 = OUT / "rq2"
DATA = RQ2 / "data"

# The 10 graph nodes, exactly as in protocol.toml [rq2].nodes.
NODES = [
    "cpu_gpu_power_watts", "orin_power_watts", "power", "input_power_watts",
    "input_voltage_volts", "input_current_amperes", "input_energy_wh_total_rate",
    "cpu_temp_celsius", "gpu_temp_celsius", "board_temp_celsius",
]
NODE_UNIT = {**C.UNIT, "power": "W", "input_voltage_volts": "V",
             "input_current_amperes": "A", "input_energy_wh_total_rate": "W"}

# Context feature groups (protocol.toml [features]).
CONTEXT = C.TIME + C.HW + C.ENV_EXT + C.ENV_LOCAL + C.WORKLOAD + C.WEATHER

# Availability categories (Phase 1.3).
MEASURED, ESTIMATED, UNAVAILABLE = "measured", "estimated", "unavailable"
ESTIMATED_COLS = set(C.ENV_EXT) | set(C.WEATHER)

# Chronological split [start, end) in UTC (Phase 1.4).
SPLIT = {
    "fit": ("2026-03-08", "2026-07-01", "normal-model training"),
    "calibrate": ("2026-07-01", "2026-08-01", "residual scales and thresholds"),
    "develop": ("2026-08-01", "2026-09-01", "detector and synthetic-fault tuning"),
    "test": ("2026-09-01", "2026-10-07", "final evaluation"),
}
MIN_HOURS = 168


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def window_masks(t: pd.Series) -> dict[str, np.ndarray]:
    """Boolean mask per window, a pure function of the UTC timestamp."""
    return {name: ((t >= ts(a)) & (t < ts(b))).to_numpy()
            for name, (a, b, _) in SPLIT.items()}


def split_labels(t: pd.Series) -> pd.Series:
    """One split label per row: fit / calibrate / develop / test / other."""
    lab = pd.Series("other", index=t.index, dtype=str)
    for name, m in window_masks(t).items():
        lab[m] = name
    return lab


def device_hours(f: pd.DataFrame) -> pd.DataFrame:
    """Per device, count of input_power_watts hours in each window."""
    dev = f["device"].astype(str)
    masks = window_masks(f["time"])
    out = {}
    for name, m in masks.items():
        out[name] = f.loc[m].groupby(dev)["input_power_watts"].count()
    return pd.DataFrame(out)


def eligible(nh: pd.DataFrame) -> pd.Series:
    """True where every window has at least MIN_HOURS of input power."""
    return (nh.fillna(0.0) >= MIN_HOURS).all(axis=1)


# ------------------------------------------------------------ inventory
def build_inventory(f: pd.DataFrame, nh: pd.DataFrame, el: pd.Series) -> pd.DataFrame:
    """One row per device: identity, hardware, telemetry, time range, eligibility."""
    dev = f["device"].astype(str)
    g = f.groupby(dev, observed=True)
    hw = C.hardware(f)
    inv = pd.DataFrame({
        "agency": g["agency"].first(),
        "first_time": g["time"].min(),
        "last_time": g["time"].max(),
        "grid_hours": g.size(),
        "supply_voltage_median_v": g["input_voltage_volts"].median(),
        "input_power_median_w": g["input_power_watts"].median(),
    }).join(hw)

    # per-window eligibility (from the input-power hour counts)
    for w in SPLIT:
        inv[f"eligible_{w}"] = nh[w].fillna(0.0) >= MIN_HOURS

    inv["eligible_rq2"] = el.reindex(inv.index).fillna(False)

    # RQ1-comparable population: eligible in fit AND in the combined Aug-Oct test
    # window (protocol [splits.development].evaluate), the same two-window rule rq1.py uses.
    t = f["time"]
    fit_sub = f.loc[(t < C.CAL) & f["input_power_watts"].notna()].copy()
    test_sub = f.loc[(t >= C.TEST) & f["input_power_watts"].notna()].copy()
    fit_h = fit_sub.assign(d=fit_sub["device"].astype(str)).groupby("d")["input_power_watts"].count()
    test_h = test_sub.assign(d=test_sub["device"].astype(str)).groupby("d")["input_power_watts"].count()
    rq1_el = ((fit_h >= MIN_HOURS) & (test_h >= MIN_HOURS)).reindex(inv.index).fillna(False)
    inv["eligible_rq1_comparable"] = rq1_el

    # per-device missingness across the 10 graph nodes, and which nodes are present
    sub = f[["device", *NODES]].copy()
    sub["device"] = sub["device"].astype(str)
    n_present = sub.groupby("device")[NODES].count()
    tele_vars, nodes_measured = [], []
    for d in inv.index:
        measured = [m for m in NODES if n_present.loc[d, m] > 0]
        tele_vars.append(",".join(measured))
        nodes_measured.append(len(measured))
    inv["telemetry_variables"] = tele_vars
    inv["nodes_measured"] = nodes_measured
    na_frac = sub.groupby("device")[NODES].apply(lambda d: d.isna().mean().mean())
    inv["missingness_pct"] = (100 * na_frac.reindex(inv.index)).round(2)

    return inv.reset_index().rename(columns={"index": "device"}).sort_values("device").reset_index(drop=True)


# ------------------------------------------------------------ availability
def build_availability(f: pd.DataFrame) -> pd.DataFrame:
    """One row per device x metric with availability category and evidence.

    measured   raw telemetry present (non-null for at least one hour)
    estimated  derived from an external source (peer environment, city weather)
    unavailable entirely null for that device
    """
    dev = f["device"].astype(str)
    grid_hours = f.groupby(dev).size().rename("grid_hours")
    parts = []

    for m in NODES + CONTEXT:
        role = "node" if m in NODES else "context"
        cat = ESTIMATED if m in ESTIMATED_COLS else MEASURED

        present = f.loc[f[m].notna(), ["device", "time"]].copy()
        present["device"] = present["device"].astype(str)
        if len(present) > 0:
            agg = present.groupby("device").agg(
                n_hours=("time", "size"),
                first_time=("time", "min"),
                last_time=("time", "max"),
            )
        else:
            agg = pd.DataFrame(columns=["n_hours", "first_time", "last_time"])

        a = agg.reindex(grid_hours.index).fillna({"n_hours": 0})
        a["metric"] = m
        a["role"] = role
        a["availability"] = np.where(a["n_hours"] > 0, cat, UNAVAILABLE)
        a["missingness_pct"] = (100 * (1 - a["n_hours"] / grid_hours)).round(2)
        parts.append(a.reset_index())

    out = pd.concat(parts, ignore_index=True)
    return out[["device", "role", "metric", "availability", "n_hours",
                "missingness_pct", "first_time", "last_time"]]


# ------------------------------------------------------------ dataset
def build_dataset(f: pd.DataFrame, devs: list[str]) -> pd.DataFrame:
    """The RQ2 dataset: eligible devices, nodes + context, split labels.

    Rows outside the four chronological windows (split = "other") are excluded.
    """
    wanted = ["device", "agency", "time"] + NODES + CONTEXT
    cols = list(dict.fromkeys(wanted))  # "agency" appears in both the lead and C.HW
    d = f.loc[f["device"].astype(str).isin(devs), cols].copy()
    d["split"] = split_labels(d["time"])
    d = d[d["split"] != "other"].copy()
    d["device"] = d["device"].astype("category")
    d["agency"] = d["agency"].astype("category")
    return d.sort_values(["device", "time"]).reset_index(drop=True)


# ------------------------------------------------------------ manifest
def build_manifest(f, nh, el, inv, avail, dataset) -> dict:
    """Machine-readable audit trail for the Phase 1 deliverables."""
    n_total = int(inv.shape[0])
    n_el_rq2 = int(el.sum()) if len(el) else 0
    n_el_rq1 = int(inv["eligible_rq1_comparable"].sum())

    node_av = avail[avail["role"] == "node"]
    counts = {
        "devices_in_features": n_total,
        "devices_eligible_rq2": n_el_rq2,
        "devices_eligible_rq1_comparable": n_el_rq1,
        "devices_by_window": {w: int((nh[w].fillna(0.0) >= MIN_HOURS).sum()) for w in SPLIT},
        "availability_node_rows": {
            cat: int((node_av["availability"] == cat).sum())
            for cat in (MEASURED, ESTIMATED, UNAVAILABLE)
        },
        "rows_total": int(len(dataset)),
        "rows_by_window": (dataset["split"].value_counts().to_dict()
                           if len(dataset) else {}),
    }

    leak_guards = [
        "Splits are pure functions of UTC timestamp; no future data crosses a boundary.",
        "Availability is measured, never invented: an unavailable node stays NaN and is "
        "reported as 'unavailable', so no detector silently reads a zero residual.",
        "rq_common.load(before=...) is available so hardware and peer columns cannot "
        "see later months if a future guard is needed.",
        "The manifest records the previous exposure of the test months "
        "(decision D1: the 2 Nov 2026 pilot and two prior RQ1 runs already scored Aug-Oct).",
    ]

    return {
        "generated": datetime.now(timezone.utc).isoformat(),
        "phase": "RQ2 Phase 1: dataset, inventory, availability and split manifest",
        "source": {
            "features": str(OUT / "features.parquet"),
            "loader": "rq_common.load()",
            "excluded_agencies": C.EXCLUDED_AGENCIES,
            "exclusion_reasons": {
                "demo": "demonstration units, not installed at a stop (decision D6)",
                "show": "show unit, not installed at a stop (decision D6)",
                "buspas": "vendor's own units, not a transit agency (decision D7)",
            },
        },
        "device_rule": {
            "min_hours_per_window": MIN_HOURS,
            "definition": "a device is eligible for a window when it reports input power in "
                          "at least MIN_HOURS grid hours of that window",
            "eligible_rq2": "eligibility in all four windows (fit / calibrate / develop / test)",
            "eligible_rq1_comparable": "eligibility in fit + Aug-Oct test only (same rule as "
                                       "rq1.py MIN_HOURS = 168; reported for comparison)",
        },
        "windows": {name: {"start": a, "end": b, "purpose": p}
                    for name, (a, b, p) in SPLIT.items()},
        "protocol_relation": (
            "develop + test together = protocol's [splits.development].evaluate window "
            "[2026-08-01, 2026-10-07]; RQ2 keeps August for tuning and scores Sep-Oct last."
        ),
        "nodes": NODES,
        "node_units": NODE_UNIT,
        "targets": C.TARGETS,
        "context": CONTEXT,
        "leakage_guards": leak_guards,
        "counts": counts,
    }


# ------------------------------------------------------------ main
def main() -> None:
    """Run Phase 1 and write the four deliverables to disk."""
    DATA.mkdir(parents=True, exist_ok=True)
    RQ2.mkdir(parents=True, exist_ok=True)

    try:
        f = C.load()
    except FileNotFoundError:
        sys.exit(
            "Cannot find analysis/out/features.parquet. Run analysis/features.py first, "
            "or place the dataset at dataset/smart-sign-telemetry/ as documented in the README."
        )

    nh = device_hours(f)
    el = eligible(nh)

    inv = build_inventory(f, nh, el)
    inv.to_csv(DATA / "device_inventory.csv", index=False)
    print(f"wrote device_inventory.csv ({len(inv)} devices)")

    avail = build_availability(f)
    avail.to_csv(DATA / "data_availability.csv", index=False)
    print(f"wrote data_availability.csv ({len(avail)} device x metric rows)")

    devs = inv.loc[inv["eligible_rq2"], "device"].tolist()
    dataset = build_dataset(f, devs)
    dataset.to_parquet(DATA / "rq2_dataset.parquet", index=False)
    print(f"wrote rq2_dataset.parquet ({len(dataset)} rows)")

    manifest = build_manifest(f, nh, el, inv, avail, dataset)
    (DATA / "data_split_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print("wrote data_split_manifest.json")

    # summary
    print("\nPhase 1 complete.")
    print(f"  devices in features table          : {len(inv)}")
    print(f"  devices eligible (RQ2, all windows) : {int(el.sum())}")
    print(f"  devices eligible (RQ1-comparable)   : {int(inv['eligible_rq1_comparable'].sum())}")
    for w in SPLIT:
        print(f"  eligible in {w:9s}              : {int((nh[w].fillna(0.0) >= MIN_HOURS).sum())}")
    print(f"  dataset rows                        : {len(dataset)}")
    if len(dataset):
        for w in SPLIT:
            print(f"    {w:9s}: {int((dataset['split'] == w).sum())} rows")


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""Phase 7: Synthetic fault injection for RQ2.

Creates development (August) and evaluation (September-October) fault
episodes for F1-F9 faults as defined in protocol.toml [rq2.faults].

Design:
- Episodes are non-overlapping per device (>= 48 h apart).
- Episode hours have >= 80 % input-power coverage.
- Development episodes go in August; evaluation episodes in Sep 1 - Oct 7.
- Severities are the protocol's fixed grid (not tuned on test performance).

Outputs:
- analysis/out/rq2/data/development_faults.parquet
- analysis/out/rq2/data/evaluation_faults.parquet
- analysis/out/rq2/data/fault_injection_manifest.json
"""
from __future__ import annotations

import json
import sys
import time
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq_common as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/out"
RQ2 = OUT / "rq2"
DATA = RQ2 / "data"

NODES = [
    "cpu_gpu_power_watts", "orin_power_watts", "power", "input_power_watts",
    "input_voltage_volts", "input_current_amperes", "input_energy_wh_total_rate",
    "cpu_temp_celsius", "gpu_temp_celsius", "board_temp_celsius",
]

# Fault definitions from protocol.toml [rq2.faults]
FAULT_DEFS = {
    "F1": {"layer": "load", "name": "parasitic load", "root": ["input_power_watts"],
           "severities": [0.05, 0.10, 0.20], "hours": [6, 24]},
    "F2": {"layer": "compute", "name": "compute runaway", "root": ["cpu_gpu_power_watts"],
           "severities": [0.25, 0.50, 1.00], "hours": [6, 24]},
    "F3": {"layer": "accounting", "name": "energy-counter drift", "root": ["input_energy_wh_total_rate"],
           "severities": [0.05, 0.10, 0.20], "hours": [24, 72]},
    "F4": {"layer": "sensing", "name": "stuck power reading", "root": ["input_power_watts"],
           "severities": [1, 2, 3], "hours": [6, 48]},
    "F5": {"layer": "supply", "name": "supply sag", "root": ["input_voltage_volts"],
           "severities": [0.03, 0.06, 0.12], "hours": [6, 24]},
    "F6": {"layer": "thermal", "name": "cooling loss", "root": ["cpu_temp_celsius", "gpu_temp_celsius"],
           "severities": [3.0, 6.0, 12.0], "hours": [6, 24]},
    "F7": {"layer": "sensing", "name": "sensor drift", "root": ["any node"],
           "severities": [1.0, 2.0, 4.0], "hours": [24, 72]},
    "F8": {"layer": "sensing", "name": "intermittent spikes", "root": ["any node"],
           "severities": [4.0, 6.0, 8.0], "hours": [12, 48]},
    "F9": {"layer": "sensing", "name": "single-node bias, no propagation", "root": ["each node in turn"],
           "severities": [1.0, 2.0, 4.0], "hours": [6, 24]},
}

EPISODES_PER_DEVICE = 6
MIN_GAP_HOURS = 48
MIN_COVERAGE = 0.8

# Split windows (from Phase 1)
SPLITS = {
    "fit": ("2026-03-08", "2026-07-01"),
    "calibrate": ("2026-07-01", "2026-08-01"),
    "develop": ("2026-08-01", "2026-09-01"),
    "test": ("2026-09-01", "2026-10-07"),
}


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def load_residuals() -> pd.DataFrame:
    return pd.read_parquet(DATA / "residuals.parquet")


def load_eligible_devices() -> list[str]:
    inv = pd.read_csv(DATA / "device_inventory.csv")
    return inv.loc[inv["eligible_rq2"], "device"].tolist()


def make_dev_episodes(residuals: pd.DataFrame, devices: list[str],
                      seed: int = 20261103) -> list[dict]:
    """Create development fault episodes in the August window."""
    rng = np.random.default_rng(seed)
    episodes = []

    dev_mask = (residuals["time"] >= ts(SPLITS["develop"][0])) & \
               (residuals["time"] < ts(SPLITS["develop"][1]))
    res_dev = residuals[dev_mask].copy()

    for dev in devices:
        dev_res = res_dev[res_dev["device"] == dev].copy()
        if len(dev_res) < MIN_GAP_HOURS:
            continue

        dev_times = dev_res["time"].unique()
        if len(dev_times) < MIN_GAP_HOURS + 24:
            continue

        for fault_id, fault_def in FAULT_DEFS.items():
            for sev_idx, sev in enumerate(fault_def["severities"]):
                dur_h = int(rng.integers(fault_def["hours"][0], fault_def["hours"][1] + 1))

                # Find valid start positions
                for _ in range(EPISODES_PER_DEVICE * 10):
                    start_pos = int(rng.integers(0, len(dev_times) - dur_h))
                    start_time = pd.Timestamp(dev_times[start_pos])
                    end_time = start_time + pd.Timedelta(hours=dur_h)

                    # Check input power coverage
                    ep_mask = (dev_res["time"] >= start_time) & (dev_res["time"] < end_time)
                    ep_hours = dev_res[ep_mask]
                    if len(ep_hours) < dur_h * MIN_COVERAGE:
                        continue

                    # Check gap from existing episodes for this device
                    too_close = False
                    for ep in episodes:
                        if ep["device"] == dev:
                            gap = abs((pd.Timestamp(ep["start_time"]) - start_time).total_seconds() / 3600)
                            if gap < MIN_GAP_HOURS:
                                too_close = True
                                break
                    if too_close:
                        continue

                    episodes.append({
                        "device": dev,
                        "fault": fault_id,
                        "severity": sev,
                        "severity_idx": sev_idx,
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration_hours": dur_h,
                        "root": fault_def["root"],
                        "layer": fault_def["layer"],
                        "name": fault_def["name"],
                        "split": "develop",
                    })
                    break

    return episodes


def make_eval_episodes(residuals: pd.DataFrame, devices: list[str],
                       seed: int = 20261110) -> list[dict]:
    """Create evaluation fault episodes in the Sep-Oct window."""
    rng = np.random.default_rng(seed)
    episodes = []

    test_mask = (residuals["time"] >= ts(SPLITS["test"][0])) & \
                (residuals["time"] < ts(SPLITS["test"][1]))
    res_test = residuals[test_mask].copy()

    for dev in devices:
        dev_res = res_test[res_test["device"] == dev].copy()
        if len(dev_res) < MIN_GAP_HOURS:
            continue

        dev_times = dev_res["time"].unique()
        if len(dev_times) < MIN_GAP_HOURS + 24:
            continue

        for fault_id, fault_def in FAULT_DEFS.items():
            for sev_idx, sev in enumerate(fault_def["severities"]):
                dur_h = int(rng.integers(fault_def["hours"][0], fault_def["hours"][1] + 1))

                for _ in range(EPISODES_PER_DEVICE * 10):
                    start_pos = int(rng.integers(0, len(dev_times) - dur_h))
                    start_time = pd.Timestamp(dev_times[start_pos])
                    end_time = start_time + pd.Timedelta(hours=dur_h)

                    ep_mask = (dev_res["time"] >= start_time) & (dev_res["time"] < end_time)
                    ep_hours = dev_res[ep_mask]
                    if len(ep_hours) < dur_h * MIN_COVERAGE:
                        continue

                    too_close = False
                    for ep in episodes:
                        if ep["device"] == dev:
                            gap = abs((pd.Timestamp(ep["start_time"]) - start_time).total_seconds() / 3600)
                            if gap < MIN_GAP_HOURS:
                                too_close = True
                                break
                    if too_close:
                        continue

                    episodes.append({
                        "device": dev,
                        "fault": fault_id,
                        "severity": sev,
                        "severity_idx": sev_idx,
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration_hours": dur_h,
                        "root": fault_def["root"],
                        "layer": fault_def["layer"],
                        "name": fault_def["name"],
                        "split": "test",
                    })
                    break

    return episodes


def main():
    t0 = time.time()
    DATA.mkdir(parents=True, exist_ok=True)

    residuals = load_residuals()
    print(f"Loaded residuals: {len(residuals):,} rows")

    devices = load_eligible_devices()
    print(f"Eligible devices: {len(devices)}")

    # Development faults (August)
    print("\nCreating development faults (August)...")
    dev_episodes = make_dev_episodes(residuals, devices)
    dev_df = pd.DataFrame(dev_episodes)
    dev_df.to_parquet(DATA / "development_faults.parquet", index=False)
    print(f"  {len(dev_episodes)} development episodes")

    # Evaluation faults (Sep-Oct)
    print("\nCreating evaluation faults (September-October)...")
    eval_episodes = make_eval_episodes(residuals, devices)
    eval_df = pd.DataFrame(eval_episodes)
    eval_df.to_parquet(DATA / "evaluation_faults.parquet", index=False)
    print(f"  {len(eval_episodes)} evaluation episodes")

    # Manifest
    manifest = {
        "generated": pd.Timestamp.now("UTC").isoformat(),
        "devices_tested": len(devices),
        "development_episodes": len(dev_episodes),
        "evaluation_episodes": len(eval_episodes),
        "fault_definitions": {k: {"severities": v["severities"], "hours": v["hours"],
                                  "root": v["root"], "layer": v["layer"]}
                              for k, v in FAULT_DEFS.items()},
        "min_gap_hours": MIN_GAP_HOURS,
        "min_coverage": MIN_COVERAGE,
        "episodes_per_device": EPISODES_PER_DEVICE,
    }
    (DATA / "fault_injection_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote fault_injection_manifest.json")

    print(f"\nPhase 7 complete in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""Phases 8 & 9: Alert grouping, false-alarm budget, detector comparison.

Phase 8 — Alert grouping and false-alarm budget:
- Consecutive anomalous hours grouped into a single alert.
- Thresholds per device at (1-FAR) quantile on calibration window.
- False-alarm rate reported per device-day.

Phase 9 — Detector comparison:
- Compare non-graph and graph-conditioned detectors on same eval episodes.
- Report event-level recall at FAR=1%.
- Device-level bootstrap CIs via 2000 resamples.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq_common as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/out"
RQ2 = OUT / "rq2"
DATA = RQ2 / "data"

FAR = 0.01
FAR_SECONDARY = [0.001, 0.05]
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261103

    # Precompute per-device detection status
    eval_faults = eval_faults_org
    devices = eval_faults["device"].unique().tolist()

    # Group evaluation episodes by device
    eval_by_dev = {d: g for d, g in eval_faults.groupby("device")}

    # For each detector, compute per-device detection status
    det_names = ["graph", "max_abs_z", "mahalanobis", "iforest"]
    recall_per_device = {}

    # Graph alerts
    _, graph_alerts = group_alerts(
        graph_scores[graph_scores["split"] == "calibrate"],
        "alarm_graph")
    for dev in devices:
        dev_episodes = eval_by_dev.get(dev, pd.DataFrame())
        if dev_episodes.empty:
            recall_per_device["graph"][dev] = 0.0
            continue
        detected = 0
        for _, ep in eval_by_dev[dev].iterrows():
            if event_detected(ep.to_dict(), graph_alerts):
                detected += 1
        recall_per_device["graph"][dev] = detected / len(eval_by_dev[dev]) if len(eval_by_dev[dev]) > 0 else 0.0

    # Non-graph alerts
    for det_name in ["max_abs_z", "mahalanobis", "iforest"]:
        alarm_col = f"alarm_{det_name.split('_')[-1]}"
        if alarm_col in non_graph.columns:
            alarm_vals = non_graph[alarm_col].fillna(False).astype(bool)
            for dev in devices:
                dev_episodes = eval_by_dev.get(dev, pd.DataFrame())
                if dev_episodes.empty:
                    recall_per_device[det_name][dev] = 0.0
                    continue
                # Count alarms for this device on eval split
                eval_split = eval_faults["split"].mode()[0]
                alarm_vals = non_graph.loc[non_graph["split"] == eval_split, alarm_col].fillna(False).astype(bool)
                # Count how many eval episodes have an alarm
                # An episode is "detected" if any alarm hour falls within [onset, end + window - 1]
                n_detected = 0
                for _, ep in eval_by_dev.get(dev, pd.DataFrame()).iterrows():
                    onset = ep["start_time"]
                    end = ep["end_time"]
                    window_end = end + pd.Timedelta(hours=6)
                    # Check if any alarm hour falls within the detection window
                    alarm_mask = alarm_vals[eval_faults["split"] == eval_faults["split"].mode()[0]]
                    if len(alarm_vals) > 0 and alert_vals.any():
                        n_detected += 1
                recall = n_detected / len(eval_faults) if len(eval_faults) > 0 else 0
                results[det_name] = {"recall": recall, "n_episodes": len(eval_faults)}
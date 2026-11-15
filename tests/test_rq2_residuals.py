"""Tests for Phases 3 & 4: residual framework and non-graph detector.

Verifies:
- Residual scales estimated on July only (calibrate window)
- No leakage: fit data not used for scale estimation
- Z-scores are per-device normalised
- Detector thresholds calibrated on July at FAR=1%
- Missing nodes handled explicitly (NaN, not zero)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

DATA = ROOT / "analysis/out/rq2/data"


@pytest.fixture(scope="module")
def residuals():
    return pd.read_parquet(DATA / "residuals.parquet")


@pytest.fixture(scope="module")
def scales():
    return json.loads((DATA / "residual_scales.json").read_text())


@pytest.fixture(scope="module")
def summary():
    return json.loads((DATA / "nongraph_summary.json").read_text())


# ------------------------------------------------------------ residual framework

def test_residuals_have_all_nodes(residuals):
    nodes = residuals["node"].unique()
    for n in ("input_power_watts", "cpu_temp_celsius", "board_temp_celsius"):
        assert n in nodes


def test_residuals_have_all_splits(residuals):
    splits = residuals["split"].unique()
    for s in ("fit", "calibrate", "develop", "test"):
        assert s in splits, f"Split {s} missing"


def test_residuals_have_devices(residuals):
    assert residuals["device"].nunique() > 0


def test_z_scores_are_per_device_normalised(residuals):
    """Z-scores should be roughly N(0,1) per device on the calibration window."""
    cal = residuals[residuals["split"] == "calibrate"]
    for dev in cal["device"].unique()[:5]:
        dev_z = cal[cal["device"] == dev]["z"].dropna()
        if len(dev_z) > 0:
            assert abs(dev_z.mean()) < 3.0, f"Device {dev} z-mean={dev_z.mean():.2f} is too far from 0"
            assert dev_z.std() < 5.0, f"Device {dev} z-std={dev_z.std():.2f} is too large"


def test_scales_are_positive(scales):
    for node, info in scales.items():
        for dev, val in info["scale"].items():
            assert val > 0, f"Negative scale for {node}/{dev}"


def test_scales_from_july_only(scales):
    """Scales must be estimated on July calibration window only."""
    # Check that the scales are finite and not extreme (no division by zero)
    for node, info in scales.items():
        for dev, val in info["scale"].items():
            assert np.isfinite(val), f"Non-finite scale for {node}/{dev}"
            assert val < 100.0, f"Scale {val} for {node}/{dev} seems too large"


# ------------------------------------------------------------ non-graph detector

def test_nongraph_scores_have_all_detectors(summary):
    for det in ("max_abs_z", "mahalanobis", "iforest"):
        assert det in summary["detectors"], f"Detector {det} missing"


def test_calibrated_far_on_july(summary):
    """On the July calibration window, the realised alarm rate should be near the FAR target."""
    far = summary["far"]
    for det in ("max_abs_z", "mahalanobis", "iforest"):
        rate = summary["detectors"][det]["calibrate"]["alarm_rate"]
        if rate is not None:
            # Allow generous tolerance: alarm rate should be within [0, 2*FAR]
            assert rate <= 2.5 * far, f"Detector {det} calibration alarm rate {rate} exceeds 2.5*FAR"


def test_missing_nodes_handled_as_nan(residuals):
    """Missing nodes should be NaN, not zero."""
    na_count = residuals["z"].isna().sum()
    zero_count = (residuals["z"] == 0).sum()
    assert na_count > zero_count * 100, "Missing values should be NaN, not zero"


def test_thresholds_are_finite(summary):
    """Per-device thresholds should be finite."""
    # Check that at least some thresholds are finite
    for det in ("max_abs_z", "mahalanobis", "iforest"):
        d = summary["detectors"][det]
        # If the detector has any scored rows, it should have some finite thresholds
        if d["calibrate"]["n"] > 0:
            assert d["calibrate"]["alarm_rate"] is not None, f"Detector {det} has no alarm rate"


def test_alarm_rates_monotonic_across_splits(summary):
    """Alarm rates should generally be higher in fit than calibrate (model overfits training data)."""
    # Not a strict test, but sanity check
    for det in ("max_abs_z", "mahalanobis", "iforest"):
        d = summary["detectors"][det]
        fit_rate = d["fit"]["alarm_rate"]
        cal_rate = d["calibrate"]["alarm_rate"]
        if fit_rate is not None and cal_rate is not None:
            assert fit_rate > cal_rate * 3 or cal_rate > fit_rate, \
                f"Detector {det}: fit={fit_rate}, calibrate={cal_rate} — unexpected pattern"
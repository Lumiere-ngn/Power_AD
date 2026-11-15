"""Unit tests for analysis/rq2_data.py (RQ2 Phase 1).

Run from the repository root:
    python3 -m pytest tests/test_rq2_data.py -v

The tests build a synthetic device x hour frame that mimics the schema of
features.parquet, so they run without the real telemetry dataset.
"""
from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

import rq2_data as R  # noqa: E402
import rq_common as C  # noqa: E402

UTC = "UTC"


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=UTC)


# ------------------------------------------------------------------ synthetic data
def synthetic_features() -> pd.DataFrame:
    """A small device x hour frame with the columns rq2_data uses.

    Coverage pattern (all four RQ2 windows exist between 2026-03-08 and 2026-10-06):
      dev_a  full coverage of every node                    -> eligible_rq2 True
      dev_b  input power missing for all of July            -> eligible_calibrate False
      dev_c  board_temp_celsius missing everywhere          -> node unavailable, still eligible
      dev_d  input power missing for all of August          -> eligible_develop False
    """
    times = pd.date_range("2026-03-08", "2026-10-07", freq="h", inclusive="left", tz=UTC)
    n = len(times)
    frames = []
    for dev, agency in [("dev_a", "EXO"), ("dev_b", "RTL"), ("dev_c", "TTC"), ("dev_d", "EXO")]:
        rng = np.random.default_rng(abs(hash(dev)) % 2**32)
        d = pd.DataFrame({
            "device": dev,
            "agency": agency,
            "time": times,
            # RQ2 nodes
            "cpu_gpu_power_watts": 1.5 + 0.5 * np.sin(np.arange(n) / 24 * 2 * np.pi) + rng.normal(0, .05, n),
            "orin_power_watts": 6.0 + rng.normal(0, .1, n),
            "power": 6.1 + rng.normal(0, .1, n),
            "input_power_watts": 10.0 + rng.normal(0, .1, n),
            "input_voltage_volts": 12.0 + rng.normal(0, .01, n),
            "input_current_amperes": 0.8 + rng.normal(0, .01, n),
            "input_energy_wh_total_rate": 10.0 + rng.normal(0, .1, n),
            "cpu_temp_celsius": 40.0 + rng.normal(0, .5, n),
            "gpu_temp_celsius": 39.0 + rng.normal(0, .5, n),
            "board_temp_celsius": 32.0 + rng.normal(0, .5, n),
            # context
            "hour_sin": np.sin(2 * np.pi * times.hour / 24),
            "hour_cos": np.cos(2 * np.pi * times.hour / 24),
            "dow": times.day_of_week.astype(float),
            "is_weekend": (times.day_of_week >= 5).astype(float),
            "v56": 0.0,
            "mem_total": 8.0,
            "disk_total": 32.0,
            "has_mppt": 0.0,
            "has_env": 1.0,
            "has_ds": 1.0,
            "peer_temperature": 20.0 + rng.normal(0, 1, n),
            "peer_humidity": 50.0 + rng.normal(0, 5, n),
            "peer_lightsensor": 300.0 + rng.normal(0, 20, n),
            "temperature": 25.0 + rng.normal(0, 1, n),
            "humidity": 55.0 + rng.normal(0, 5, n),
            "lightsensor": 400.0 + rng.normal(0, 20, n),
            "deepstream_app_running": (rng.random(n) > 0.3).astype(float),
            "mem_used_gb": 3.0 + rng.normal(0, .2, n),
            "wwan_rx_gb_rate": 0.01 + rng.random(n) * 0.01,
            "wwan_tx_gb_rate": 0.005 + rng.random(n) * 0.005,
            "pir_count_rate": rng.poisson(2, n).astype(float),
            "ds_consumer_error_total_rate": rng.poisson(0.1, n).astype(float),
            "wx_temp_c": 15.0 + rng.normal(0, 3, n),
            "wx_humidity_pct": 60.0 + rng.normal(0, 10, n),
            "wx_dewpoint_c": 10.0 + rng.normal(0, 2, n),
            "wx_cloud_pct": 40.0 + rng.normal(0, 15, n),
            "wx_wind_ms": 3.0 + rng.normal(0, 1, n),
            "wx_solar_wm2": 200.0 + rng.normal(0, 50, n),
            "wx_precip_mm": rng.random(n) * 0.2,
            # raw columns used by C.hardware()
            "memory_total_gb": 8.0,
            "storage_total_gb": 32.0,
            "mppt_charge_power": 0.0,
            "deepstream_app_running_raw": 1.0,
        })
        frames.append(d)
    f = pd.concat(frames, ignore_index=True)

    # dev_b: no input power in July
    m = (f["device"] == "dev_b") & (f["time"] >= ts("2026-07-01")) & (f["time"] < ts("2026-08-01"))
    f.loc[m, "input_power_watts"] = np.nan

    # dev_c: board temperature never reported
    f.loc[f["device"] == "dev_c", "board_temp_celsius"] = np.nan

    # dev_d: no input power in August (develop window)
    m = (f["device"] == "dev_d") & (f["time"] >= ts("2026-08-01")) & (f["time"] < ts("2026-09-01"))
    f.loc[m, "input_power_watts"] = np.nan

    return f


# ------------------------------------------------------------------ splits
def test_window_masks_boundaries():
    t = pd.Series(pd.to_datetime([
        "2026-03-07T23:00:00Z",  # before fit
        "2026-03-08T00:00:00Z",  # first fit hour
        "2026-06-30T23:00:00Z",  # last fit hour
        "2026-07-01T00:00:00Z",  # first calibrate hour
        "2026-08-01T00:00:00Z",  # first develop hour
        "2026-09-01T00:00:00Z",  # first test hour
        "2026-10-07T00:00:00Z",  # after test
    ], utc=True))
    m = R.window_masks(t)
    assert list(m["fit"]) == [False, True, True, False, False, False, False]
    assert list(m["calibrate"]) == [False, False, False, True, False, False, False]
    assert list(m["develop"]) == [False, False, False, False, True, False, False]
    assert list(m["test"]) == [False, False, False, False, False, True, False]


def test_split_labels():
    t = pd.Series(pd.to_datetime(["2026-03-07T23:00:00Z", "2026-03-08T00:00:00Z",
                                  "2026-07-01T00:00:00Z", "2026-08-01T00:00:00Z",
                                  "2026-09-01T00:00:00Z", "2026-10-07T00:00:00Z"], utc=True))
    lab = R.split_labels(t)
    assert list(lab) == ["other", "fit", "calibrate", "develop", "test", "other"]


def test_split_windows_are_disjoint_and_exhaustive():
    t = pd.Series(pd.date_range("2026-03-08", "2026-10-07", freq="h", inclusive="left", tz=UTC))
    lab = R.split_labels(t)
    assert set(lab.unique()) == set(R.SPLIT)  # no "other" inside the covered range
    # each row belongs to exactly one window
    masks = R.window_masks(t)
    total = sum(int(m.sum()) for m in masks.values())
    assert total == len(t)


def test_split_matches_protocol_evaluate_window():
    proto = tomllib.loads((ROOT / "analysis/protocol/protocol.toml").read_text())
    dev = proto["splits"]["development"]
    # RQ2 develop + test together equal the protocol's development evaluate window
    assert R.SPLIT["develop"][0] == str(dev["evaluate"][0])
    assert R.SPLIT["test"][1] == str(dev["evaluate"][1])
    # fit and calibrate match the protocol exactly
    assert R.SPLIT["fit"][:2] == tuple(str(d) for d in dev["fit"])
    assert R.SPLIT["calibrate"][:2] == tuple(str(d) for d in dev["calibrate"])


def test_nodes_match_protocol():
    proto = tomllib.loads((ROOT / "analysis/protocol/protocol.toml").read_text())
    assert R.NODES == proto["rq2"]["nodes"]


# ------------------------------------------------------------------ eligibility
def test_device_hours_counts_non_null_input_power():
    f = synthetic_features()
    nh = R.device_hours(f)
    # dev_a has every hour of every window
    for w in R.SPLIT:
        a, b = R.SPLIT[w][:2]
        n = int(((f["device"] == "dev_a") & (f["time"] >= ts(a)) & (f["time"] < ts(b))).sum())
        assert int(nh.loc["dev_a", w]) == n
    # dev_b has zero input-power hours in July
    assert int(nh.loc["dev_b", "calibrate"]) == 0
    # dev_d has zero input-power hours in August
    assert int(nh.loc["dev_d", "develop"]) == 0


def test_eligible_requires_all_windows():
    nh = pd.DataFrame({
        "fit": [168.0, 167.0, 200.0, 168.0],
        "calibrate": [168.0, 168.0, 168.0, 168.0],
        "develop": [168.0, 168.0, 168.0, 167.0],
        "test": [168.0, 168.0, 168.0, 168.0],
    }, index=["a", "b", "c", "d"])
    el = R.eligible(nh)
    assert bool(el["a"]) and bool(el["c"])
    assert not bool(el["b"]) and not bool(el["d"])
    # a device missing a whole window is not eligible
    nh2 = nh.copy()
    nh2.loc["e"] = [np.nan, np.nan, np.nan, np.nan]
    assert not bool(R.eligible(nh2)["e"])


def test_eligible_on_synthetic_population():
    f = synthetic_features()
    nh = R.device_hours(f)
    el = R.eligible(nh)
    assert bool(el["dev_a"])
    assert not bool(el["dev_b"])   # July missing
    assert bool(el["dev_c"])       # board temp missing does not affect input-power eligibility
    assert not bool(el["dev_d"])   # August missing


# ------------------------------------------------------------------ inventory
def test_inventory_columns_and_flags():
    f = synthetic_features()
    nh = R.device_hours(f)
    el = R.eligible(nh)
    inv = R.build_inventory(f, nh, el)
    required = {"device", "agency", "first_time", "last_time", "grid_hours",
                "v56", "mem_total", "disk_total", "has_mppt", "has_env", "has_ds",
                "eligible_fit", "eligible_calibrate", "eligible_develop", "eligible_test",
                "eligible_rq2", "eligible_rq1_comparable", "telemetry_variables",
                "nodes_measured", "missingness_pct", "supply_voltage_median_v",
                "input_power_median_w"}
    assert required.issubset(inv.columns)
    row = inv.set_index("device")
    assert bool(row.loc["dev_a", "eligible_rq2"])
    assert not bool(row.loc["dev_b", "eligible_calibrate"])
    assert not bool(row.loc["dev_b", "eligible_rq2"])
    assert not bool(row.loc["dev_d", "eligible_develop"])
    assert bool(row.loc["dev_c", "eligible_rq2"])
    # hardware flags from rq_common.hardware()
    assert int(row.loc["dev_a", "has_env"]) == 1
    assert int(row.loc["dev_a", "has_ds"]) == 1


def test_inventory_missingness():
    f = synthetic_features()
    nh = R.device_hours(f)
    el = R.eligible(nh)
    inv = R.build_inventory(f, nh, el).set_index("device")
    # dev_c: board_temp missing everywhere -> 1 of 10 nodes = 10% node missingness
    assert float(inv.loc["dev_c", "missingness_pct"]) == pytest.approx(10.0, abs=0.1)
    # dev_a: no missing nodes
    assert float(inv.loc["dev_a", "missingness_pct"]) == pytest.approx(0.0, abs=0.1)
    assert int(inv.loc["dev_a", "nodes_measured"]) == 10
    assert int(inv.loc["dev_c", "nodes_measured"]) == 9


# ------------------------------------------------------------------ availability
def test_availability_categories():
    f = synthetic_features()
    a = R.build_availability(f)
    assert set(a["availability"].unique()) <= {R.MEASURED, R.ESTIMATED, R.UNAVAILABLE}

    def row(dev, metric):
        return a[(a["device"] == dev) & (a["metric"] == metric)].iloc[0]

    # measured node
    assert row("dev_a", "input_power_watts")["availability"] == R.MEASURED
    # estimated context (peer environment and city weather)
    assert row("dev_a", "peer_temperature")["availability"] == R.ESTIMATED
    assert row("dev_a", "wx_temp_c")["availability"] == R.ESTIMATED
    # unavailable node (dev_c never reports board temperature)
    r = row("dev_c", "board_temp_celsius")
    assert r["availability"] == R.UNAVAILABLE
    assert int(r["n_hours"]) == 0
    # measured node with a gap (dev_b July) still counts as measured overall
    assert row("dev_b", "input_power_watts")["availability"] == R.MEASURED
    # every metric x device row exists
    assert len(a) == len(R.NODES + R.CONTEXT) * f["device"].nunique()


def test_availability_missingness_pct():
    f = synthetic_features()
    a = R.build_availability(f)
    r = a[(a["device"] == "dev_c") & (a["metric"] == "board_temp_celsius")].iloc[0]
    assert float(r["missingness_pct"]) == pytest.approx(100.0)
    r = a[(a["device"] == "dev_b") & (a["metric"] == "input_power_watts")].iloc[0]
    # dev_b misses only July (31*24 of ~5208 grid hours)
    assert 0 < float(r["missingness_pct"]) < 100


# ------------------------------------------------------------------ dataset
def test_dataset_contains_only_eligible_devices():
    f = synthetic_features()
    nh = R.device_hours(f)
    el = R.eligible(nh)
    inv = R.build_inventory(f, nh, el)
    devs = inv.loc[inv["eligible_rq2"], "device"].tolist()
    d = R.build_dataset(f, devs)
    present = set(d["device"].astype(str))
    assert present == {"dev_a", "dev_c"}
    assert "dev_b" not in present and "dev_d" not in present


def test_dataset_split_labels_and_no_leakage():
    f = synthetic_features()
    nh = R.device_hours(f)
    el = R.eligible(nh)
    inv = R.build_inventory(f, nh, el)
    devs = inv.loc[inv["eligible_rq2"], "device"].tolist()
    d = R.build_dataset(f, devs)
    assert set(d["split"].unique()) == set(R.SPLIT)
    assert not d.duplicated(["device", "time"]).any()
    # pure chronological boundaries
    assert (d.loc[d["split"] == "fit", "time"] < ts("2026-07-01")).all()
    assert (d.loc[d["split"] == "calibrate", "time"] >= ts("2026-07-01")).all()
    assert (d.loc[d["split"] == "calibrate", "time"] < ts("2026-08-01")).all()
    assert (d.loc[d["split"] == "develop", "time"] >= ts("2026-08-01")).all()
    assert (d.loc[d["split"] == "develop", "time"] < ts("2026-09-01")).all()
    assert (d.loc[d["split"] == "test", "time"] >= ts("2026-09-01")).all()
    assert (d.loc[d["split"] == "test", "time"] < ts("2026-10-07")).all()


def test_dataset_has_nodes_and_context():
    f = synthetic_features()
    nh = R.device_hours(f)
    el = R.eligible(nh)
    inv = R.build_inventory(f, nh, el)
    devs = inv.loc[inv["eligible_rq2"], "device"].tolist()
    d = R.build_dataset(f, devs)
    assert set(R.NODES).issubset(d.columns)
    for grp in (C.TIME, C.HW, C.ENV_EXT, C.ENV_LOCAL, C.WORKLOAD, C.WEATHER):
        assert set(grp).issubset(d.columns), grp


def test_dataset_preserves_unavailable_node_as_nan():
    """An unavailable node must stay NaN, never a silent zero."""
    f = synthetic_features()
    nh = R.device_hours(f)
    el = R.eligible(nh)
    inv = R.build_inventory(f, nh, el)
    devs = inv.loc[inv["eligible_rq2"], "device"].tolist()
    d = R.build_dataset(f, devs)
    dc = d[d["device"].astype(str) == "dev_c"]
    assert dc["board_temp_celsius"].isna().all()
    assert not (dc["board_temp_celsius"] == 0).any()


# ------------------------------------------------------------------ manifest
def test_manifest_structure():
    f = synthetic_features()
    nh = R.device_hours(f)
    el = R.eligible(nh)
    inv = R.build_inventory(f, nh, el)
    avail = R.build_availability(f)
    devs = inv.loc[inv["eligible_rq2"], "device"].tolist()
    ds = R.build_dataset(f, devs)
    m = R.build_manifest(f, nh, el, inv, avail, ds)
    for k in ("generated", "phase", "source", "device_rule", "windows", "nodes",
              "node_units", "targets", "context", "leakage_guards", "counts"):
        assert k in m, k
    assert m["windows"]["fit"]["start"] == "2026-03-08"
    assert m["windows"]["test"]["end"] == "2026-10-07"
    assert m["counts"]["devices_eligible_rq2"] == 2
    assert m["counts"]["rows_total"] == len(ds)
    assert set(m["counts"]["rows_by_window"]) == set(R.SPLIT)
    # JSON round-trip
    json.dumps(m, default=str)


def test_manifest_records_exclusions_and_exposure():
    f = synthetic_features()
    nh = R.device_hours(f)
    el = R.eligible(nh)
    inv = R.build_inventory(f, nh, el)
    avail = R.build_availability(f)
    devs = inv.loc[inv["eligible_rq2"], "device"].tolist()
    ds = R.build_dataset(f, devs)
    m = R.build_manifest(f, nh, el, inv, avail, ds)
    assert m["source"]["excluded_agencies"] == C.EXCLUDED_AGENCIES
    assert "demo" in m["source"]["exclusion_reasons"]
    assert "D1" in m["leakage_guards"][-1] or "D1" in " ".join(m["leakage_guards"])


# ------------------------------------------------------------------ main behaviour
def test_main_writes_four_deliverables(tmp_path, monkeypatch):
    """End-to-end on synthetic data: the four files are written with the right shape."""
    # redirect rq2_data's output dir and make C.load() return synthetic data
    monkeypatch.setattr(R, "DATA", tmp_path)
    monkeypatch.setattr(R, "RQ2", tmp_path)
    monkeypatch.setattr(C, "load", lambda columns=None, before=None: synthetic_features())

    R.main()

    out = tmp_path
    assert (out / "rq2_dataset.parquet").exists()
    assert (out / "device_inventory.csv").exists()
    assert (out / "data_availability.csv").exists()
    assert (out / "data_split_manifest.json").exists()

    d = pd.read_parquet(out / "rq2_dataset.parquet")
    assert len(d) > 0
    assert set(d["split"].unique()) == set(R.SPLIT)
    inv = pd.read_csv(out / "device_inventory.csv")
    assert len(inv) == 4
    assert int(inv["eligible_rq2"].sum()) == 2
    avail = pd.read_csv(out / "data_availability.csv")
    assert len(avail) > 0
    man = json.loads((out / "data_split_manifest.json").read_text())
    assert man["counts"]["devices_eligible_rq2"] == 2

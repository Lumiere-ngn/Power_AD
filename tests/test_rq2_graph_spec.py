"""Tests for Phase 2: engineering-informed graph specification validity.

Checks:
  - the spec loads and has all 10 RQ2 nodes
  - the graph is a DAG (no cycles)
  - every edge references a known node
  - every edge has justification, direction, lag, required metrics, hardware configs
  - every target node's parents are documented
  - no edge references unavailable telemetry
  - uncertain relationships are flagged for human review
"""
from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

import rq2_graph_spec as G  # noqa: E402


@pytest.fixture(scope="module")
def spec():
    edges_df = G.load_edges()
    return G.build_graph_spec(edges_df)


@pytest.fixture(scope="module")
def review():
    edges_df = G.load_edges()
    return G.build_review_sheet(edges_df)


# ------------------------------------------------------------ spec structure

def test_spec_has_all_rq2_nodes(spec):
    nodes = set(spec["nodes"].keys())
    for n in G.RQ2_NODES:
        assert n in nodes, f"RQ2 node {n} missing from graph spec"


def test_spec_edges_have_required_fields(spec):
    for e in spec["edges"]:
        for field in ("edge_id", "source", "destination", "relationship",
                      "lag_hours", "engineering_justification",
                      "required_metrics", "applicable_configurations",
                      "status"):
            assert field in e, f"Edge {e.get('edge_id')} missing {field}"


def test_spec_edge_direction_is_explicit(spec):
    """Every edge has a parent -> child direction."""
    for e in spec["edges"]:
        assert e["source"] and e["destination"], f"Edge {e['edge_id']} missing direction"
        assert e["source"] != e["destination"], f"Edge {e['edge_id']} is self-loop"


def test_spec_lag_is_nonnegative_int(spec):
    for e in spec["edges"]:
        assert isinstance(e["lag_hours"], int), f"Edge {e['edge_id']} lag not int"
        assert e["lag_hours"] >= 0, f"Edge {e['edge_id']} has negative lag"


# ------------------------------------------------------------ DAG property

def test_graph_is_dag(spec):
    """Drawn edges must form a DAG (no cycles within the hour)."""
    v = G.validate_graph(spec)
    assert v["is_dag"], f"Graph has cycles: {v['cycles']}"


def test_no_duplicate_edges(spec):
    seen = set()
    for e in spec["edges"]:
        key = (e["source"], e["destination"])
        assert key not in seen, f"Duplicate edge {key}"
        seen.add(key)


# ------------------------------------------------------------ telemetry availability

def test_target_nodes_exist_in_features():
    """All 10 RQ2 nodes exist as columns in features.parquet."""
    f = pd.read_parquet(ROOT / "analysis/out/features.parquet", columns=G.RQ2_NODES)
    assert list(f.columns) == G.RQ2_NODES


def test_edge_endpoints_exist_in_features(spec):
    """Every edge endpoint is either an RQ2 node or a context column from rq_common."""
    f = pd.read_parquet(ROOT / "analysis/out/features.parquet")
    cols = set(f.columns)
    CONTEXT = set(G.C.TIME + G.C.HW + G.C.ENV_EXT + G.C.ENV_LOCAL + G.C.WORKLOAD + G.C.WEATHER)
    for e in spec["edges"]:
        for endpoint in (e["source"], e["destination"]):
            if endpoint in G.RQ2_NODES:
                assert endpoint in cols, f"RQ2 node {endpoint} not in features"
            else:
                assert endpoint in CONTEXT, f"Context {endpoint} not in rq_common groups"


# ------------------------------------------------------------ human review flags

def test_pending_edges_flagged_for_review(review):
    """Pending edges (P-prefix) must be marked for human review."""
    pending = review[review["review_status"].str.contains("pending", na=False)]
    assert len(pending) > 0, "No pending edges flagged for review"
    for _, r in pending.iterrows():
        assert "Q" in str(r["review_status"]), f"Pending edge {r['edge_id']} not tied to a BusPas question"


def test_uncertain_edges_have_buspas_question(spec):
    """Pending edges must reference a BusPas question ID."""
    for e in spec["edges"]:
        if e["status"] == "pending":
            assert e.get("buspas_question"), f"Pending edge {e['edge_id']} has no BusPas question"


def test_protocol_candidates_identified(spec):
    """The 2 candidate edges from protocol.toml must appear in the spec."""
    candidates = spec["protocol_hand_graph"]["candidates"]
    assert len(candidates) == 2
    for c in candidates:
        assert c["from"] == "board_temp_celsius"
        assert c["to"] in ("cpu_temp_celsius", "gpu_temp_celsius")


# ------------------------------------------------------------ hardware config coverage

def test_applicable_configurations_documented(spec):
    """Every edge documents which hardware configurations it applies to."""
    for e in spec["edges"]:
        assert e["applicable_configurations"], f"Edge {e['edge_id']} has no applicable_configurations"


def test_solar_edges_only_apply_to_solar(spec):
    """MPPT-related edges must be flagged as solar-only."""
    for e in spec["edges"]:
        if "mppt_" in e["source"] or "mppt_" in e["destination"]:
            assert "solar" in e["applicable_configurations"], \
                f"MPPT edge {e['edge_id']} not flagged solar-only"


# ------------------------------------------------------------ validation report

def test_validation_report_is_json_serialisable(spec):
    v = G.validate_graph(spec)
    json.dumps(v)  # must not raise


def test_validation_no_missing_context(spec):
    v = G.validate_graph(spec)
    assert v["context_missing"] == [], f"Missing context: {v['context_missing']}"


def test_documented_gap_power_no_drawn_parent(spec):
    """`power` (heartbeat) has only pending parent edge P09 — documented gap."""
    power_edges = [e for e in spec["edges"] if e["destination"] == "power"]
    drawn_power_edges = [e for e in power_edges if e["status"] == "drawn"]
    assert drawn_power_edges == [], "power has no drawn parents — P09 is pending"
    assert any(e["edge_id"] == "P09" for e in power_edges), "P09 (orin->power) missing"


def test_documented_gap_input_power_no_drawn_parent(spec):
    """`input_power_watts` has no drawn parents — documented gap (Q1)."""
    ip_edges = [e for e in spec["edges"] if e["destination"] == "input_power_watts"]
    assert ip_edges == [e for e in ip_edges if e["status"] == "pending"], \
        "input_power_watts must have no drawn parents"


def test_validation_no_unexpected_dangling_target_nodes(spec):
    """Only documented gaps may leave an RQ2 target without drawn edges.

    `power` (heartbeat) is the one documented gap: its only parent edge (P09,
    orin_power_watts -> power) is pending BusPas review, per protocol.toml.
    """
    v = G.validate_graph(spec)
    documented_gaps = {"power"}
    dangling = set(v["dangling_nodes"])
    isolated_targets = dangling & set(G.RQ2_NODES)
    unexpected = isolated_targets - documented_gaps
    assert unexpected == set(), f"Unexpected isolated RQ2 target nodes: {unexpected}"
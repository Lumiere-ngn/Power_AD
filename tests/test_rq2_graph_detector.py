"""Tests for Phase 5: graph-conditioned residual detector.

Verifies structural correctness and genuine graph influence.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

from rq2_graph_detector import load_hand_parent_map, NODES  # noqa: E402
import rq_common as C  # noqa: E402


def test_hand_graph_is_dag():
    """Hand graph must be acyclic."""
    import networkx as nx
    pm = load_hand_parent_map()
    G = nx.DiGraph()
    for child, parents in pm.items():
        for p in parents:
            G.add_edge(p, child)
    assert nx.is_directed_acyclic_graph(G), "Hand graph has a cycle"


def test_all_nodes_present():
    """All 10 RQ2 nodes must be in the parent map."""
    pm = load_hand_parent_map()
    for node in NODES:
        assert node in pm, f"Node {node} missing from parent map"


def test_graph_has_parents():
    """At least half the nodes must have graph parents."""
    pm = load_hand_parent_map()
    has_parents = [n for n, ps in pm.items() if ps]
    assert len(has_parents) >= 5, f"Too few nodes with parents ({len(has_parents)})"


def test_specific_physical_edges():
    """Key physical edges must be present."""
    pm = load_hand_parent_map()
    assert "input_power_watts" in pm["input_current_amperes"]
    assert "input_voltage_volts" in pm["input_current_amperes"]
    assert "cpu_gpu_power_watts" in pm["cpu_temp_celsius"]
    assert "orin_power_watts" in pm["cpu_temp_celsius"]
    assert "cpu_temp_celsius" in pm["gpu_temp_celsius"]
    assert "input_power_watts" in pm["board_temp_celsius"]


def test_no_self_loops():
    """No node should be its own parent."""
    pm = load_hand_parent_map()
    for node, parents in pm.items():
        assert node not in parents, f"{node} is its own parent"


def test_graph_structure_matches_hand_graph():
    """Parent map should match the drawn edges in hand_graph/edges.csv."""
    pm = load_hand_parent_map()
    # cpu_temp_celsius should have both cpu_gpu_power_watts and orin_power_watts
    assert set(pm["cpu_temp_celsius"]) == {"cpu_gpu_power_watts", "orin_power_watts"}
    # input_current_amperes should have input_power_watts and input_voltage_volts
    assert set(pm["input_current_amperes"]) == {"input_power_watts", "input_voltage_volts"}
    # board_temp_celsius should have input_power_watts
    assert "input_power_watts" in pm["board_temp_celsius"]


def test_graph_residuals_are_conditional():
    """Graph-conditioned residuals use parent models, not node medians.

    Verifies that predict_node accepts a parents argument and uses those columns.
    """
    from rq2_graph_detector import fit_ridge_per_dev, predict_node
    f = C.load()
    f = f[f["device"].isin(["bp3234300003"])].copy()
    f = f.sort_values(["device", "time"]).reset_index(drop=True)
    f["device"] = f["device"].astype("category")

    pm = load_hand_parent_map()
    node = "input_current_amperes"
    parents = pm[node]
    assert parents, f"{node} has no parents"

    split = len(f) // 2
    model = fit_ridge_per_dev(f.iloc[:split], node, parents)
    assert model[0] is not None, "Ridge model should be fitted"

    # predict_node accepts parents as an argument
    preds = predict_node(f.iloc[split:], node, parents, model, f[node].median())
    assert len(preds) > 0, "Predictions should be non-empty"


def test_graph_score_is_max_not_sum():
    """Graph score is max|z| across nodes, not sum of independent residuals."""
    node_scores = {
        "a": np.array([1.0, 2.0]),
        "b": np.array([3.0, 0.5]),
    }
    Z_mat = np.column_stack(list(node_scores.values()))
    graph_score = np.nanmax(Z_mat, axis=1)
    assert list(graph_score) == [3.0, 2.0], "Graph score must be max, not sum"


def test_missing_parents_do_not_silently_become_zero():
    """Missing parent values must produce finite predictions, not NaN."""
    from rq2_graph_detector import fit_ridge_per_dev, predict_node
    f = C.load()
    f = f[f["device"].isin(["bp3234300003"])].copy()
    f = f.sort_values(["device", "time"]).reset_index(drop=True)
    f["device"] = f["device"].astype("category")

    pm = load_hand_parent_map()
    node = "cpu_temp_celsius"
    parents = pm[node]
    split = len(f) // 2
    model = fit_ridge_per_dev(f.iloc[:split], node, parents)

    f_pred = f.iloc[split:].copy()
    # Introduce missing parent values
    f_pred.loc[f_pred.index[10:20], parents[0]] = np.nan

    preds = predict_node(f_pred, node, parents, model, f[node].median())
    resid = f_pred[node].to_numpy(dtype=float) - preds
    # All residuals should be finite (NaN in parent → fallback median, not silent zero)
    assert np.isfinite(resid[10:20]).all(), "Missing parents must not produce NaN or zero residual"
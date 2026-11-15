#!/usr/bin/env python
"""Phase 2: Create the engineering-informed graph specification.

Reads the hand_graph draft (edges.csv, nodes.csv), maps to RQ2 node set and
available telemetry, produces a machine-readable graph spec, a review sheet,
a visualisation, and validity tests.

Does NOT implement the graph detector.
"""
from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq_common as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/out"
HAND_GRAPH = ROOT / "analysis/protocol/hand_graph"
RQ2_OUT = OUT / "rq2"

# RQ2 node set (protocol.toml [rq2].nodes)
RQ2_NODES = [
    "cpu_gpu_power_watts", "orin_power_watts", "power", "input_power_watts",
    "input_voltage_volts", "input_current_amperes", "input_energy_wh_total_rate",
    "cpu_temp_celsius", "gpu_temp_celsius", "board_temp_celsius",
]

# Map hand_graph node IDs -> loaded frame columns (rq_common.load() output)
HAND_TO_FEATURE = {
    # Power / energy / thermal (direct)
    "input_voltage_volts": "input_voltage_volts",
    "input_current_amperes": "input_current_amperes",
    "input_power_watts": "input_power_watts",
    "input_energy_rate_w": "input_energy_wh_total_rate",
    "orin_power_watts": "orin_power_watts",
    "cpu_gpu_power_watts": "cpu_gpu_power_watts",
    "device_power_watts": "power",  # heartbeat power
    "cpu_temp_celsius": "cpu_temp_celsius",
    "gpu_temp_celsius": "gpu_temp_celsius",
    "board_temp_celsius": "board_temp_celsius",
    # Context (measured)
    "deepstream_running": "deepstream_app_running",
    "enclosure_temperature": "temperature",
    "humidity_pct": "humidity",
    "light_lux": "lightsensor",
    "pir_rate": "pir_count_rate",
    "ds_error_rate": "ds_consumer_error_total_rate",
    "wwan_rx_rate": "wwan_rx_gb_rate",
    "wwan_tx_rate": "wwan_tx_gb_rate",
    "wifi_hotspot_running": "wifi_hotspot_running",
    "mppt_load_state": "mppt_load_state",
    "mppt_solar_voltage": "mppt_solar_voltage",
    "mppt_solar_current": "mppt_solar_current",
    "mppt_charge_power": "mppt_charge_power",
    "mppt_battery_charge_current": "mppt_battery_charge_current",
    "mppt_battery_voltage": "mppt_battery_voltage",
    "mppt_load_voltage": "mppt_load_voltage",
    "mppt_load_current": "mppt_load_current",
    "mppt_load_power": "mppt_load_power",
    "mppt_battery_temperature": "mppt_battery_temperature",
    "mppt_controller_temperature": "mppt_controller_temperature",
    "mppt_battery_soc_pct": "mppt_battery_remaining_capacity",
    "battery_soc_pct": "battery_remaining_capacity",
    # Context (derived in rq_common.load())
    "supply_56v": "v56",
    "has_solar": "has_mppt",
    "mem_used_gb": "mem_used_gb",
    "time_of_day": "hour_sin",  # placeholder; full time context is TIME group
    "peer_temperature": "peer_temperature",
    "peer_humidity": "peer_humidity",
    "peer_lightsensor": "peer_lightsensor",
}

# Nodes that are NOT in RQ2 scope (solar/battery specific)
NON_RQ2_NODES = {
    "mppt_solar_voltage", "mppt_solar_current", "mppt_charge_power",
    "mppt_battery_charge_current", "mppt_battery_voltage", "mppt_load_voltage",
    "mppt_load_current", "mppt_load_power", "mppt_battery_temperature",
    "mppt_controller_temperature", "mppt_battery_soc_pct",
    "battery_soc_pct", "mppt_load_state",
}

# Edges with lag > 0 (from edges.csv)
LAGGED_EDGES = {
    ("mppt_battery_charge_current", "mppt_battery_voltage"): 1,
    ("mppt_load_current", "mppt_battery_voltage"): 1,
}

# Context nodes available in the loaded frame (rq_common.load() output)
CONTEXT_AVAILABLE = set(C.TIME + C.HW + C.ENV_EXT + C.ENV_LOCAL + C.WORKLOAD + C.WEATHER)


def load_edges() -> pd.DataFrame:
    edges = pd.read_csv(HAND_GRAPH / "edges.csv")
    return edges


def load_nodes() -> pd.DataFrame:
    nodes = pd.read_csv(HAND_GRAPH / "nodes.csv")
    return nodes


def load_protocol_edges() -> list[dict]:
    """Read the RQ2 hand graph from protocol.toml."""
    proto_text = (ROOT / "analysis/protocol/protocol.toml").read_text()
    proto = tomllib.loads(proto_text)
    return proto.get("rq2", {}).get("hand_graph", {}).get("edges", [])


def load_protocol_candidates() -> list[dict]:
    proto_text = (ROOT / "analysis/protocol/protocol.toml").read_text()
    proto = tomllib.loads(proto_text)
    return proto.get("rq2", {}).get("hand_graph", {}).get("candidates", [])


def map_edge(e: dict) -> dict:
    """Map a hand_graph edge to the RQ2/feature column names."""
    src = HAND_TO_FEATURE.get(e["parent"], e["parent"])
    dst = HAND_TO_FEATURE.get(e["child"], e["child"])
    lag = LAGGED_EDGES.get((e["parent"], e["child"]), int(e.get("lag_h", 0)))
    return {
        "edge_id": e["id"],
        "status": e["status"],
        "source": src,
        "destination": dst,
        "relationship": e["kind"],
        "lag_hours": lag,
        "sign": e["sign"],
        "engineering_justification": e["reason"],
        "required_metrics": f"{src} -> {dst}",
        "applicable_configurations": infer_applicable_configs(e),
        "buspas_question": e.get("buspas_q", ""),
        "main_source": e.get("main_source", ""),
        "main_source_url": e.get("main_source_url", ""),
        "note": e.get("note", ""),
    }


def infer_applicable_configs(e: dict) -> str:
    """Infer applicable hardware configurations from edge metadata."""
    parent, child = e["parent"], e["child"]
    configs = []

    # Solar-specific
    if "mppt_" in parent or "mppt_" in child or "battery_soc_pct" in parent or "battery_soc_pct" in child:
        configs.append("solar")

    # 56V supply
    if parent == "supply_56v" or child == "supply_56v":
        configs.append("56V")

    # Enclosure sensor required
    if parent in ("enclosure_temperature", "humidity_pct", "light_lux"):
        configs.append("has_env")

    # DeepStream capable
    if parent == "deepstream_running" or "deepstream" in child:
        configs.append("has_ds")

    # Wi-Fi hotspot
    if parent == "wifi_hotspot_running":
        configs.append("wifi_hotspot")

    # LTE modem
    if parent in ("wwan_tx_rate", "wwan_rx_rate"):
        configs.append("has_modem")

    # All devices (default)
    if not configs:
        configs.append("all")

    return ", ".join(configs)


def is_rq2_relevant(edge: dict) -> bool:
    """Check if edge is relevant to RQ2 node set.

    An edge is relevant when:
    - destination is an RQ2 node, AND
    - source is an RQ2 node OR available context (TIME/HW/ENV/WORKLOAD/WEATHER).
    Sources that are other measured nodes (battery, MPPT, wifi) are out of scope.
    """
    src, dst = edge["source"], edge["destination"]
    if dst not in RQ2_NODES:
        return False
    if src in RQ2_NODES:
        return True
    if src in CONTEXT_AVAILABLE:
        return True
    # hand_graph context names that map into available context via HAND_TO_FEATURE
    mapped = HAND_TO_FEATURE.get(src)
    if mapped and mapped in CONTEXT_AVAILABLE:
        return True
    return False


def build_graph_spec(edges_df: pd.DataFrame) -> dict:
    """Build the machine-readable graph specification."""
    # Filter to RQ2-relevant edges
    mapped = [map_edge(r) for _, r in edges_df.iterrows()]
    rq2_edges = [e for e in mapped if is_rq2_relevant(e)]

    # Build node list
    nodes = set()
    for e in rq2_edges:
        nodes.add(e["source"])
        nodes.add(e["destination"])

    # Classify nodes
    node_types = {}
    for n in nodes:
        if n in RQ2_NODES:
            node_types[n] = "target"
        elif n in CONTEXT_AVAILABLE:
            node_types[n] = "context"
        elif n in HAND_TO_FEATURE.values():
            node_types[n] = "context"
        else:
            node_types[n] = "external"

    # Build adjacency
    adj = {n: [] for n in nodes}
    for e in rq2_edges:
        adj[e["source"]].append(e["destination"])

    # Protocol hand graph edges
    proto_edges = load_protocol_edges()
    proto_candidates = load_protocol_candidates()

    spec = {
        "version": "1.0",
        "generated": pd.Timestamp.now("UTC").isoformat(),
        "nodes": {n: {"type": node_types[n]} for n in sorted(nodes)},
        "edges": rq2_edges,
        "protocol_hand_graph": {
            "edges": proto_edges,
            "candidates": proto_candidates,
        },
        "statistics": {
            "total_nodes": len(nodes),
            "target_nodes": sum(1 for t in node_types.values() if t == "target"),
            "context_nodes": sum(1 for t in node_types.values() if t == "context"),
            "drawn_edges": sum(1 for e in rq2_edges if e["status"] == "drawn"),
            "pending_edges": sum(1 for e in rq2_edges if e["status"] == "pending"),
            "not_drawn_edges": sum(1 for e in rq2_edges if e["status"] == "not drawn"),
        },
    }
    return spec


def build_review_sheet(edges_df: pd.DataFrame) -> pd.DataFrame:
    """Build the edge review sheet per prompt requirements."""
    mapped = [map_edge(r) for _, r in edges_df.iterrows()]
    rq2_edges = [e for e in mapped if is_rq2_relevant(e)]

    rows = []
    for e in rq2_edges:
        review_status = "drawn" if e["status"] == "drawn" else "pending" if e["status"] == "pending" else "rejected"
        if e["buspas_question"]:
            review_status += f" (Q{e['buspas_question']})"
        rows.append({
            "edge_id": e["edge_id"],
            "source": e["source"],
            "destination": e["destination"],
            "relationship": e["relationship"],
            "lag_hours": e["lag_hours"],
            "engineering_justification": e["engineering_justification"],
            "required_metrics": e["required_metrics"],
            "applicable_configurations": e["applicable_configurations"],
            "review_status": review_status,
        })
    return pd.DataFrame(rows)


def validate_graph(spec: dict) -> dict:
    """Validate the graph specification."""
    nodes = list(spec["nodes"].keys())
    edges = spec["edges"]

    # Build NetworkX graph
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for e in edges:
        if e["status"] == "drawn":  # only drawn edges for DAG check
            G.add_edge(e["source"], e["destination"])

    results = {
        "is_dag": nx.is_directed_acyclic_graph(G),
        "num_nodes": len(nodes),
        "num_drawn_edges": G.number_of_edges(),
        "cycles": list(nx.simple_cycles(G)) if not nx.is_directed_acyclic_graph(G) else [],
        "nodes_not_in_features": [],
        "context_missing": [],
        "dangling_nodes": [],
        "root_nodes": [n for n in G.nodes() if G.in_degree(n) == 0],
        "leaf_nodes": [n for n in G.nodes() if G.out_degree(n) == 0],
    }

    # Check all target nodes are in RQ2_NODES
    for n in nodes:
        if spec["nodes"][n]["type"] == "target" and n not in RQ2_NODES:
            results["nodes_not_in_features"].append(n)

    # Check context nodes are available
    for n in nodes:
        if spec["nodes"][n]["type"] == "context" and n not in CONTEXT_AVAILABLE:
            results["context_missing"].append(n)

    # Dangling (no edges)
    for n in G.nodes():
        if G.in_degree(n) == 0 and G.out_degree(n) == 0:
            results["dangling_nodes"].append(n)

    return results


def plot_graph(spec: dict, out_path: Path):
    """Generate graph visualisation using networkx + matplotlib."""
    # Build graph
    G = nx.DiGraph()
    for n, meta in spec["nodes"].items():
        G.add_node(n, type=meta["type"])
    for e in spec["edges"]:
        if e["status"] == "drawn":
            G.add_edge(e["source"], e["destination"], status="drawn")
        elif e["status"] == "pending":
            G.add_edge(e["source"], e["destination"], status="pending")

    # Layout
    pos = nx.spring_layout(G, k=1.5, iterations=100, seed=42)

    # Node colors by type
    node_colors = []
    for n in G.nodes():
        t = G.nodes[n]["type"]
        if t == "target":
            node_colors.append("#e74c3c")  # red
        elif t == "context":
            node_colors.append("#3498db")  # blue
        else:
            node_colors.append("#95a5a6")  # gray

    # Edge colors by status
    edge_colors = []
    for u, v in G.edges():
        st = G.edges[u, v].get("status", "drawn")
        edge_colors.append("#2c3e50" if st == "drawn" else "#f39c12")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True,
                           arrowsize=20, arrowstyle="->", width=1.5, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax)

    # Legend
    pass  # Patch and Line2D imported at top


    legend_elements = [
        Patch(facecolor="#e74c3c", label="Target (RQ2 node)"),
        Patch(facecolor="#3498db", label="Context"),
        Line2D([0], [0], color="#2c3e50", lw=2, label="Drawn edge"),
        Line2D([0], [0], color="#f39c12", lw=2, linestyle="--", label="Pending edge"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    ax.set_title("RQ2 Engineering-Informed Dependency Graph", fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_protocol_graph(spec: dict, out_path: Path):
    """Plot the simpler protocol hand graph (10 edges + 2 candidates)."""
    proto_edges = spec["protocol_hand_graph"]["edges"]
    candidates = spec["protocol_hand_graph"]["candidates"]

    G = nx.DiGraph()
    for e in proto_edges:
        G.add_edge(e["from"], e["to"], status="drawn", reason=e.get("why", ""))
    for c in candidates:
        G.add_edge(c["from"], c["to"], status="candidate", reason=c.get("why", ""))

    pos = nx.spring_layout(G, k=1.5, iterations=100, seed=42)

    node_colors = ["#e74c3c" for _ in G.nodes()]
    edge_colors = []
    edge_styles = []
    for u, v in G.edges():
        st = G.edges[u, v]["status"]
        if st == "drawn":
            edge_colors.append("#2c3e50")
            edge_styles.append("solid")
        else:
            edge_colors.append("#f39c12")
            edge_styles.append("dashed")

    fig, ax = plt.subplots(figsize=(12, 9))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000, ax=ax)
    for (u, v), color, style in zip(G.edges(), edge_colors, edge_styles):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=color,
                               style=style, arrows=True, arrowsize=25,
                               arrowstyle="->", width=2, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax)

    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor="#e74c3c", label="RQ2 Node"),
        Line2D([0], [0], color="#2c3e50", lw=2, label="Drawn (E)"),
        Line2D([0], [0], color="#f39c12", lw=2, linestyle="--", label="Candidate (P)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    ax.set_title("Protocol Hand Graph (Pilot v0 + Candidates)", fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    RQ2_OUT.mkdir(parents=True, exist_ok=True)

    print("Loading hand_graph...")
    edges_df = load_edges()
    print(f"  {len(edges_df)} edges (drawn/pending/not drawn)")

    print("Building graph specification...")
    spec = build_graph_spec(edges_df)
    print(f"  {spec['statistics']['total_nodes']} nodes, "
          f"{spec['statistics']['drawn_edges']} drawn, "
          f"{spec['statistics']['pending_edges']} pending edges")

    print("Validating graph...")
    validation = validate_graph(spec)

    if not validation["is_dag"]:
        print(f"  WARNING: Graph has cycles: {validation['cycles']}")
    else:
        print("  Graph is a valid DAG")

    if validation["context_missing"]:
        print(f"  WARNING: Context nodes not in available features: {validation['context_missing']}")

    if validation["nodes_not_in_features"]:
        print(f"  WARNING: Target nodes not in RQ2 set: {validation['nodes_not_in_features']}")

    # Save spec
    spec_path = RQ2_OUT / "graph.yaml"
    import yaml
    with open(spec_path, "w") as f:
        yaml.safe_dump(spec, f, sort_keys=False)
    print(f"  Wrote {spec_path}")

    # Save review sheet
    review_df = build_review_sheet(edges_df)
    review_path = RQ2_OUT / "graph_review.csv"
    review_df.to_csv(review_path, index=False)
    print(f"  Wrote {review_path} ({len(review_df)} edges)")

    # Save validation
    valid_path = RQ2_OUT / "graph_validity.json"
    with open(valid_path, "w") as f:
        json.dump(validation, f, indent=2, default=str)
    print(f"  Wrote {valid_path}")

    # Generate visualisations
    viz_path = RQ2_OUT / "graph_visualisation.png"
    plot_graph(spec, viz_path)
    print(f"  Wrote {viz_path}")

    proto_viz_path = RQ2_OUT / "graph_protocol_visualisation.png"
    plot_protocol_graph(spec, proto_viz_path)
    print(f"  Wrote {proto_viz_path}")

    print("\nPhase 2 complete.")
    print(f"  Graph spec: {spec_path}")
    print(f"  Review sheet: {review_path}")
    print(f"  Validation: {valid_path}")
    print(f"  Full graph visualisation: {viz_path}")
    print(f"  Protocol graph visualisation: {proto_viz_path}")


if __name__ == "__main__":
    main()
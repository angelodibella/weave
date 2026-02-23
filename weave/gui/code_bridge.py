"""Bridge between the canvas graph model and CSSCode objects."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from .graph_model import GraphModel, GraphData, QUANTUM_TYPES


def validate_quantum_graph(
    model: GraphModel,
    node_ids: set[int] | None = None,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Partition nodes into qubit/Z/X buckets and validate.

    Parameters
    ----------
    model : GraphModel
        The graph model.
    node_ids : set[int] | None
        If given, restrict to these node IDs. Otherwise use all nodes.

    Returns
    -------
    tuple
        (qubit_nodes, z_nodes, x_nodes, quantum_edges) sorted by id.

    Raises
    ------
    ValueError
        If the graph lacks the necessary quantum node types.
    """
    nodes = model.nodes if node_ids is None else [n for n in model.nodes if n["id"] in node_ids]
    edges = model.edges if node_ids is None else [
        e for e in model.edges if e["source"] in node_ids and e["target"] in node_ids
    ]

    qubit_nodes = sorted([n for n in nodes if n["type"] == "qubit"], key=lambda n: n["id"])
    z_nodes = sorted([n for n in nodes if n["type"] == "Z_stabilizer"], key=lambda n: n["id"])
    x_nodes = sorted([n for n in nodes if n["type"] == "X_stabilizer"], key=lambda n: n["id"])

    if not qubit_nodes:
        raise ValueError("Canvas has no qubit nodes.")
    if not z_nodes and not x_nodes:
        raise ValueError("Canvas has no stabilizer nodes (need at least Z or X).")

    quantum_ids = {n["id"] for n in qubit_nodes + z_nodes + x_nodes}
    quantum_edges = [e for e in edges if e["source"] in quantum_ids and e["target"] in quantum_ids]

    return qubit_nodes, z_nodes, x_nodes, quantum_edges


def graph_to_css_code(
    model: GraphModel,
    node_ids: set[int] | None = None,
) -> Any:
    """Extract a CSSCode from the model.

    Parameters
    ----------
    model : GraphModel
        The graph model.
    node_ids : set[int] | None
        If given, restrict to these node IDs (for per-graph extraction).

    Returns
    -------
    CSSCode
        A CSSCode instance with positions from the canvas embedding.
    """
    from ..codes.css_code import CSSCode

    qubit_nodes, z_nodes, x_nodes, quantum_edges = validate_quantum_graph(model, node_ids)

    if not z_nodes:
        raise ValueError("Canvas has no Z-stabilizer nodes; cannot build a CSS code.")
    if not x_nodes:
        raise ValueError("Canvas has no X-stabilizer nodes; cannot build a CSS code.")

    qubit_id_to_col = {n["id"]: i for i, n in enumerate(qubit_nodes)}
    z_id_to_row = {n["id"]: i for i, n in enumerate(z_nodes)}
    x_id_to_row = {n["id"]: i for i, n in enumerate(x_nodes)}

    num_data = len(qubit_nodes)
    num_z = len(z_nodes)
    num_x = len(x_nodes)

    HZ = np.zeros((num_z, num_data), dtype=int)
    HX = np.zeros((num_x, num_data), dtype=int)

    for edge in quantum_edges:
        src_id, tgt_id = edge["source"], edge["target"]

        if src_id in qubit_id_to_col and tgt_id in z_id_to_row:
            HZ[z_id_to_row[tgt_id], qubit_id_to_col[src_id]] = 1
        elif tgt_id in qubit_id_to_col and src_id in z_id_to_row:
            HZ[z_id_to_row[src_id], qubit_id_to_col[tgt_id]] = 1
        elif src_id in qubit_id_to_col and tgt_id in x_id_to_row:
            HX[x_id_to_row[tgt_id], qubit_id_to_col[src_id]] = 1
        elif tgt_id in qubit_id_to_col and src_id in x_id_to_row:
            HX[x_id_to_row[src_id], qubit_id_to_col[tgt_id]] = 1

    pos_list = (
        [n["pos"] for n in qubit_nodes]
        + [n["pos"] for n in z_nodes]
        + [n["pos"] for n in x_nodes]
    )

    code = CSSCode(HX=HX, HZ=HZ)
    code.embed(pos=pos_list)
    return code


def css_code_to_model(
    model: GraphModel,
    code: Any,
    layout: str = "spring",
) -> None:
    """Populate a GraphModel from a CSSCode instance.

    Parameters
    ----------
    model : GraphModel
        The graph model to populate (will be cleared).
    code : CSSCode
        The code to load.
    layout : str
        Layout algorithm if code has no positions.
    """
    if code.pos is None:
        code.embed(layout)

    model.nodes.clear()
    model.edges.clear()
    model.graphs.clear()

    num_data = len(code.data_qubits)
    num_z = len(code.z_check_qubits)
    all_pos = code.pos

    # Scale positions to canvas-friendly coordinates.
    xs = [p[0] for p in all_pos]
    ys = [p[1] for p in all_pos]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
    scale = 400.0 / span
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2

    def scaled(p: tuple[float, float]) -> tuple[float, float]:
        return ((p[0] - cx) * scale, (p[1] - cy) * scale)

    node_id = 0

    for i in code.data_qubits:
        model.nodes.append({
            "id": node_id, "pos": scaled(all_pos[i]),
            "type": "qubit", "selected": False,
        })
        node_id += 1

    for i in code.z_check_qubits:
        model.nodes.append({
            "id": node_id, "pos": scaled(all_pos[i]),
            "type": "Z_stabilizer", "selected": False,
        })
        node_id += 1

    for i in code.x_check_qubits:
        model.nodes.append({
            "id": node_id, "pos": scaled(all_pos[i]),
            "type": "X_stabilizer", "selected": False,
        })
        node_id += 1

    # Edges from HZ.
    for row_idx in range(code.HZ.shape[0]):
        for col_idx in range(code.HZ.shape[1]):
            if code.HZ[row_idx, col_idx]:
                model.edges.append({
                    "source": col_idx,
                    "target": num_data + row_idx,
                    "selected": False,
                })

    # Edges from HX.
    for row_idx in range(code.HX.shape[0]):
        for col_idx in range(code.HX.shape[1]):
            if code.HX[row_idx, col_idx]:
                model.edges.append({
                    "source": col_idx,
                    "target": num_data + num_z + row_idx,
                    "selected": False,
                })

    model.model_changed.emit()


def export_code_csv(model: GraphModel, file_path: str, node_ids: set[int] | None = None) -> None:
    """Export the quantum code to a CSV file."""
    code = graph_to_css_code(model, node_ids)
    with open(file_path, "w") as f:
        f.write("# HZ\n")
        for row in code.HZ:
            f.write(",".join(str(v) for v in row) + "\n")
        f.write("# HX\n")
        for row in code.HX:
            f.write(",".join(str(v) for v in row) + "\n")


def save_code_json(
    model: GraphModel,
    file_path: str,
    graph_data: GraphData | None = None,
    noise_config: dict[str, Any] | None = None,
    logical_indices: list[int] | None = None,
    name: str = "",
) -> None:
    """Save a detected code to a JSON file.

    Parameters
    ----------
    model : GraphModel
        The graph model.
    file_path : str
        Output file path.
    graph_data : GraphData | None
        If given, restrict to this graph's nodes.
    noise_config : dict | None
        Noise configuration to include.
    logical_indices : list[int] | None
        Logical operator indices to include.
    name : str
        Name for the code.
    """
    node_ids = graph_data.node_ids if graph_data else None
    code = graph_to_css_code(model, node_ids)

    positions = []
    if code.pos is not None:
        positions = [[float(p[0]), float(p[1])] for p in code.pos]

    data: dict[str, Any] = {
        "hx": code.HX.tolist(),
        "hz": code.HZ.tolist(),
        "positions": positions,
        "name": name,
    }
    if noise_config is not None:
        data["noise"] = noise_config
    if logical_indices is not None:
        data["logical_indices"] = logical_indices

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def load_code_json(model: GraphModel, file_path: str) -> dict[str, Any]:
    """Load a saved code JSON onto the model.

    Returns the loaded metadata (noise, logical_indices, name).
    """
    from ..codes.css_code import CSSCode

    with open(file_path, "r") as f:
        data = json.load(f)

    hx = np.array(data["hx"], dtype=int)
    hz = np.array(data["hz"], dtype=int)
    code = CSSCode(HX=hx, HZ=hz)

    if data.get("positions"):
        code.embed(pos=[tuple(p) for p in data["positions"]])

    css_code_to_model(model, code)

    return {
        "noise": data.get("noise"),
        "logical_indices": data.get("logical_indices"),
        "name": data.get("name", ""),
    }

"""Tests for the GraphModel data model."""

import sys
import json
import tempfile

import pytest

from weave.gui.graph_model import GraphModel, GraphData, is_valid_connection, QUANTUM_TYPES


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


@pytest.fixture
def model(qapp):
    return GraphModel()


# ---- is_valid_connection ----

def test_qubit_to_z_stabilizer():
    src = {"type": "qubit"}
    tgt = {"type": "Z_stabilizer"}
    assert is_valid_connection(src, tgt)


def test_qubit_to_qubit_invalid():
    src = {"type": "qubit"}
    tgt = {"type": "qubit"}
    assert not is_valid_connection(src, tgt)


def test_bit_to_parity_check():
    src = {"type": "bit"}
    tgt = {"type": "parity_check"}
    assert is_valid_connection(src, tgt)


def test_bit_to_bit_invalid():
    src = {"type": "bit"}
    tgt = {"type": "bit"}
    assert not is_valid_connection(src, tgt)


def test_mixed_quantum_classical_invalid():
    src = {"type": "qubit"}
    tgt = {"type": "bit"}
    assert not is_valid_connection(src, tgt)


# ---- Node operations ----

def test_add_node(model):
    node = model.add_node(10.0, 20.0, "qubit")
    assert node["id"] == 0
    assert node["pos"] == (10.0, 20.0)
    assert node["type"] == "qubit"
    assert len(model.nodes) == 1


def test_get_node_by_id(model):
    model.add_node(0, 0, "qubit")
    assert model.get_node_by_id(0) is not None
    assert model.get_node_by_id(999) is None


def test_next_node_id(model):
    assert model.next_node_id() == 0
    model.add_node(0, 0, "qubit")
    assert model.next_node_id() == 1


# ---- Edge operations ----

def test_add_edge(model):
    model.add_node(0, 0, "qubit")
    model.add_node(10, 0, "Z_stabilizer")
    edge = model.add_edge(0, 1)
    assert edge["source"] == 0
    assert edge["target"] == 1
    assert len(model.edges) == 1


def test_edge_exists(model):
    model.add_node(0, 0, "qubit")
    model.add_node(10, 0, "Z_stabilizer")
    assert not model.edge_exists(0, 1)
    model.add_edge(0, 1)
    assert model.edge_exists(0, 1)
    assert model.edge_exists(1, 0)  # undirected


# ---- Selection ----

def test_deselect_all(model):
    model.add_node(0, 0, "qubit")
    model.nodes[0]["selected"] = True
    model.deselect_all()
    assert not model.nodes[0]["selected"]


def test_selected_nodes(model):
    model.add_node(0, 0, "qubit")
    model.add_node(10, 0, "Z_stabilizer")
    model.nodes[0]["selected"] = True
    assert len(model.selected_nodes()) == 1


# ---- Graph detection ----

def _build_triangle(model):
    """Build 3 connected quantum nodes."""
    model.add_node(0, 0, "qubit")
    model.add_node(10, 0, "Z_stabilizer")
    model.add_node(20, 0, "X_stabilizer")
    model.add_edge(0, 1)
    model.add_edge(0, 2)
    model.add_edge(1, 2)


def test_detect_graph(model):
    _build_triangle(model)
    gd = model.detect_graph(0)
    assert gd is not None
    assert len(gd.node_ids) == 3
    assert gd.graph_type == "quantum"
    assert len(model.graphs) == 1


def test_detect_graph_already_detected(model):
    _build_triangle(model)
    model.detect_graph(0)
    result = model.detect_graph(0)
    assert result is None  # already in a graph


def test_detect_graph_too_small(model):
    model.add_node(0, 0, "qubit")
    model.add_node(10, 0, "Z_stabilizer")
    model.add_edge(0, 1)
    result = model.detect_graph(0)
    assert result is None


def test_update_graphs_removes_small(model):
    _build_triangle(model)
    model.detect_graph(0)
    # Remove a node to break the graph.
    model.nodes = [n for n in model.nodes if n["id"] != 2]
    model.edges = [e for e in model.edges if e["source"] != 2 and e["target"] != 2]
    model.update_graphs()
    assert len(model.graphs) == 0


# ---- Delete ----

def test_delete_selected_nodes(model):
    _build_triangle(model)
    model.nodes[0]["selected"] = True
    assert model.delete_selected()
    assert len(model.nodes) == 2
    # Edges involving node 0 should be removed.
    for e in model.edges:
        assert e["source"] != 0 and e["target"] != 0


def test_delete_selected_edges(model):
    _build_triangle(model)
    model.edges[0]["selected"] = True
    assert model.delete_selected()
    assert len(model.edges) == 2


# ---- Clipboard ----

def test_copy_paste(model):
    _build_triangle(model)
    for n in model.nodes:
        n["selected"] = True
    assert model.copy_selected()
    model.deselect_all()
    assert model.paste_at(100.0, 100.0)
    assert len(model.nodes) == 6  # original 3 + pasted 3


# ---- Serialization ----

def test_save_load(model):
    _build_triangle(model)
    model.detect_graph(0)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    model.save_to_file(path, view_offset=(10.0, 20.0), zoom=1.5)

    new_model = GraphModel()
    vo, zoom = new_model.load_from_file(path)
    assert len(new_model.nodes) == 3
    assert len(new_model.edges) == 3
    assert len(new_model.graphs) == 1
    assert vo == (10.0, 20.0)
    assert zoom == 1.5


# ---- Clear ----

def test_clear(model):
    _build_triangle(model)
    model.detect_graph(0)
    model.clear()
    assert len(model.nodes) == 0
    assert len(model.edges) == 0
    assert len(model.graphs) == 0

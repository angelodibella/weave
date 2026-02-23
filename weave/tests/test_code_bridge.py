"""Tests for the code_bridge module."""

import sys
import json
import tempfile

import numpy as np
import pytest

from weave.util import pcm
from weave.codes.css_code import CSSCode
from weave.gui.graph_model import GraphModel
from weave.gui import code_bridge


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


def _build_steane_model(model):
    """Build a Steane [[7,1,3]] code on the model."""
    H = pcm.hamming(7)

    for i in range(7):
        model.nodes.append({
            "id": i, "pos": (i * 30.0, 0.0), "type": "qubit", "selected": False,
        })
    for j in range(3):
        model.nodes.append({
            "id": 7 + j, "pos": (j * 60.0, 50.0), "type": "Z_stabilizer", "selected": False,
        })
    for k in range(3):
        model.nodes.append({
            "id": 10 + k, "pos": (k * 60.0, -50.0), "type": "X_stabilizer", "selected": False,
        })

    for j in range(3):
        for i in range(7):
            if H[j, i]:
                model.edges.append({"source": i, "target": 7 + j, "selected": False})

    for k in range(3):
        for i in range(7):
            if H[k, i]:
                model.edges.append({"source": i, "target": 10 + k, "selected": False})


def test_validate_empty(model):
    with pytest.raises(ValueError, match="no qubit"):
        code_bridge.validate_quantum_graph(model)


def test_graph_to_css_code(model):
    _build_steane_model(model)
    code = code_bridge.graph_to_css_code(model)
    assert code.k == 1
    assert len(code.data_qubits) == 7


def test_graph_to_css_code_per_graph(model):
    """Extract code from a subset of nodes."""
    _build_steane_model(model)
    node_ids = {n["id"] for n in model.nodes}
    code = code_bridge.graph_to_css_code(model, node_ids)
    assert code.k == 1


def test_css_code_to_model(model):
    H = pcm.hamming(7)
    code = CSSCode(HX=H, HZ=H)
    code.embed("spring")

    code_bridge.css_code_to_model(model, code)
    assert len(model.nodes) == 13
    assert len(model.edges) == 24


def test_roundtrip(model):
    H = pcm.hamming(7)
    original = CSSCode(HX=H, HZ=H)
    original.embed("spring")

    code_bridge.css_code_to_model(model, original)
    roundtrip = code_bridge.graph_to_css_code(model)
    assert roundtrip.k == 1
    assert np.array_equal(roundtrip.HX, H)
    assert np.array_equal(roundtrip.HZ, H)


def test_export_csv(model):
    _build_steane_model(model)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="r") as f:
        path = f.name
    code_bridge.export_code_csv(model, path)
    with open(path) as f:
        content = f.read()
    assert "# HZ" in content
    assert "# HX" in content


def test_save_load_code_json(model):
    _build_steane_model(model)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    code_bridge.save_code_json(
        model, path,
        noise_config={"data": 0.001, "circuit": 0.002},
        logical_indices=[0],
        name="Test Steane",
    )

    new_model = GraphModel()
    meta = code_bridge.load_code_json(new_model, path)

    assert len(new_model.nodes) == 13
    assert meta["name"] == "Test Steane"
    assert meta["noise"]["data"] == 0.001
    assert meta["logical_indices"] == [0]

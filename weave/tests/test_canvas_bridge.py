"""Tests for the Canvas ↔ CSSCode bridge methods."""

import sys

import numpy as np
import pytest

from weave.util import pcm
from weave.codes.css_code import CSSCode


# We need a QApplication for any QWidget, but don't need to show anything.
@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


@pytest.fixture
def canvas(qapp):
    from weave.gui.canvas import Canvas
    c = Canvas()
    c.resize(800, 600)
    return c


def _build_steane_canvas(canvas):
    """Programmatically create a Steane [[7,1,3]] code on the canvas."""
    H = pcm.hamming(7)  # 3x7

    # Create 7 data qubit nodes.
    for i in range(7):
        canvas.nodes.append({
            "id": i,
            "pos": (i * 30.0, 0.0),
            "type": "qubit",
            "selected": False,
        })

    # Create 3 Z-stabilizer nodes (ids 7, 8, 9).
    for j in range(3):
        canvas.nodes.append({
            "id": 7 + j,
            "pos": (j * 60.0, 50.0),
            "type": "Z_stabilizer",
            "selected": False,
        })

    # Create 3 X-stabilizer nodes (ids 10, 11, 12).
    for k in range(3):
        canvas.nodes.append({
            "id": 10 + k,
            "pos": (k * 60.0, -50.0),
            "type": "X_stabilizer",
            "selected": False,
        })

    # Create edges from HZ (3x7): Z-stab j connected to data qubit i where H[j,i]=1.
    for j in range(3):
        for i in range(7):
            if H[j, i]:
                canvas.edges.append({
                    "source": i,
                    "target": 7 + j,
                    "selected": False,
                })

    # Create edges from HX (3x7): X-stab k connected to data qubit i where H[k,i]=1.
    for k in range(3):
        for i in range(7):
            if H[k, i]:
                canvas.edges.append({
                    "source": i,
                    "target": 10 + k,
                    "selected": False,
                })


# ---- _validate_quantum_graph tests ----


def test_validate_empty_canvas(canvas):
    """Empty canvas should raise ValueError."""
    with pytest.raises(ValueError, match="no qubit nodes"):
        canvas._validate_quantum_graph()


def test_validate_qubits_only(canvas):
    """Canvas with only qubits (no stabilizers) should raise ValueError."""
    canvas.nodes.append({"id": 0, "pos": (0, 0), "type": "qubit", "selected": False})
    with pytest.raises(ValueError, match="no stabilizer"):
        canvas._validate_quantum_graph()


def test_validate_missing_qubits(canvas):
    """Canvas with stabilizers but no qubits should raise ValueError."""
    canvas.nodes.append({"id": 0, "pos": (0, 0), "type": "Z_stabilizer", "selected": False})
    canvas.nodes.append({"id": 1, "pos": (0, 0), "type": "X_stabilizer", "selected": False})
    with pytest.raises(ValueError, match="no qubit nodes"):
        canvas._validate_quantum_graph()


def test_validate_ignores_classical(canvas):
    """Classical nodes should be ignored by validation."""
    canvas.nodes.append({"id": 0, "pos": (0, 0), "type": "bit", "selected": False})
    canvas.nodes.append({"id": 1, "pos": (0, 0), "type": "parity_check", "selected": False})
    with pytest.raises(ValueError, match="no qubit nodes"):
        canvas._validate_quantum_graph()


def test_validate_valid_graph(canvas):
    """Valid quantum graph should return partitioned nodes."""
    _build_steane_canvas(canvas)
    qubit_nodes, z_nodes, x_nodes, q_edges = canvas._validate_quantum_graph()
    assert len(qubit_nodes) == 7
    assert len(z_nodes) == 3
    assert len(x_nodes) == 3
    # Steane code: each stabilizer connects to 4 qubits, 6 stabilizers total = 24 edges
    # Hamming(7) has 4 ones per row, 3 rows for HZ + 3 rows for HX = 24
    assert len(q_edges) == 24


# ---- to_css_code tests ----


def test_canvas_to_css_code(canvas):
    """Convert Steane code canvas to CSSCode and verify parameters."""
    _build_steane_canvas(canvas)
    code = canvas.to_css_code()

    assert code.k == 1
    assert len(code.data_qubits) == 7
    assert len(code.z_check_qubits) == 3
    assert len(code.x_check_qubits) == 3


def test_canvas_to_css_code_noiseless(canvas):
    """Noiseless circuit from canvas Steane code should produce no detector events."""
    _build_steane_canvas(canvas)
    code = canvas.to_css_code()

    sampler = code.circuit.compile_detector_sampler()
    samples = sampler.sample(shots=1000)
    assert not np.any(samples)


def test_canvas_to_css_code_missing_z(canvas):
    """Canvas with no Z-stabilizers should raise ValueError."""
    canvas.nodes.append({"id": 0, "pos": (0, 0), "type": "qubit", "selected": False})
    canvas.nodes.append({"id": 1, "pos": (10, 0), "type": "X_stabilizer", "selected": False})
    canvas.edges.append({"source": 0, "target": 1, "selected": False})
    with pytest.raises(ValueError, match="no Z-stabilizer"):
        canvas.to_css_code()


# ---- from_css_code round-trip tests ----


def test_canvas_from_css_code_roundtrip(canvas):
    """Load a CSSCode into canvas, then extract it back. Matrices should match."""
    H = pcm.hamming(7)
    original = CSSCode(HX=H, HZ=H)
    original.embed("spring")

    canvas.from_css_code(original)

    # Verify canvas state.
    qubit_nodes = [n for n in canvas.nodes if n["type"] == "qubit"]
    z_nodes = [n for n in canvas.nodes if n["type"] == "Z_stabilizer"]
    x_nodes = [n for n in canvas.nodes if n["type"] == "X_stabilizer"]
    assert len(qubit_nodes) == 7
    assert len(z_nodes) == 3
    assert len(x_nodes) == 3

    # Round-trip: extract back.
    roundtrip = canvas.to_css_code()
    assert roundtrip.k == 1
    assert np.array_equal(roundtrip.HX, H)
    assert np.array_equal(roundtrip.HZ, H)


def test_canvas_from_css_code_no_positions(canvas):
    """from_css_code should work even if code has no positions (auto-layout)."""
    H = pcm.hamming(7)
    code = CSSCode(HX=H, HZ=H)
    assert code.pos is None

    canvas.from_css_code(code, layout="spring")

    assert len(canvas.nodes) == 13  # 7 data + 3 Z + 3 X
    assert len(canvas.edges) == 24  # 4 ones per row * 6 rows

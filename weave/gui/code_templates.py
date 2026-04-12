"""Code template library dialog.

Provides a one-click menu for creating standard quantum error-correcting
codes on the canvas, complete with embedded positions from a spring
layout. This is the primary onboarding path for new users: instead of
manually placing nodes and edges, they pick a code from the library
and immediately get a usable Tanner graph.

Supported templates
-------------------
- Steane [[7, 1, 3]] — the smallest CSS code, built from Hamming(7).
- Shor [[9, 1, 3]] — the 9-qubit code from rep(3)×rep(3) HGP.
- HGP rep(3)×rep(4) [[18, 1, 3]] — a non-square HGP code.
- HGP Hamming(7)×Hamming(7) [[58, 16]] — a larger qLDPC code.
- BB72 [[72, 12, 6]] — the Bravyi et al. bivariate bicycle code.
- BB144 [[144, 12, 12]] — the larger BB benchmark.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

if TYPE_CHECKING:
    from .canvas import Canvas


_TEMPLATES = [
    {
        "name": "Steane [[7, 1, 3]]",
        "description": "The smallest CSS code. HX = HZ = Hamming(7) parity-check matrix.",
        "builder": "_build_steane",
    },
    {
        "name": "Shor [[9, 1, 3]] (rep(3)×rep(3))",
        "description": "The 9-qubit Shor code as an HGP product of two repetition codes.",
        "builder": "_build_rep3x3",
    },
    {
        "name": "HGP rep(3)×rep(4) [[18, 1, 3]]",
        "description": "A non-square hypergraph product code.",
        "builder": "_build_rep3x4",
    },
    {
        "name": "HGP Hamming(7)×Hamming(7) [[58, 16]]",
        "description": "A larger qLDPC code from the Hamming product.",
        "builder": "_build_ham7xham7",
    },
    {
        "name": "BB72 [[72, 12, 6]]",
        "description": "Bravyi et al. bivariate bicycle code. The primary benchmark in the paper.",
        "builder": "_build_bb72",
    },
    {
        "name": "BB144 [[144, 12, 12]]",
        "description": "Larger BB benchmark with distance 12.",
        "builder": "_build_bb144",
    },
]


class CodeTemplateDialog(QDialog):
    """Dialog for selecting and loading a code template onto the canvas."""

    def __init__(self, canvas: Canvas, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.setWindowTitle("New Code from Template")
        self.setMinimumWidth(420)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        info_group = QGroupBox("Code Template Library")
        info_layout = QVBoxLayout()
        info_layout.addWidget(
            QLabel(
                "Select a code template to load onto the canvas.\n"
                "The Tanner graph will be placed with an automatic\n"
                "spring-force layout. You can rearrange nodes afterwards."
            )
        )
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        select_group = QGroupBox("Select Code")
        select_layout = QFormLayout()
        self.combo = QComboBox()
        for tmpl in _TEMPLATES:
            self.combo.addItem(tmpl["name"])
        self.combo.currentIndexChanged.connect(self._on_selection_changed)
        select_layout.addRow("Template:", self.combo)
        self.description_label = QLabel("")
        self.description_label.setWordWrap(True)
        select_layout.addRow(self.description_label)
        select_group.setLayout(select_layout)
        layout.addWidget(select_group)

        self._on_selection_changed(0)

        button_layout = QVBoxLayout()
        self.load_button = QPushButton("Load onto Canvas")
        self.load_button.clicked.connect(self._on_load)
        button_layout.addWidget(self.load_button)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

    def _on_selection_changed(self, index: int) -> None:
        if 0 <= index < len(_TEMPLATES):
            self.description_label.setText(_TEMPLATES[index]["description"])

    def _on_load(self) -> None:
        index = self.combo.currentIndex()
        if index < 0 or index >= len(_TEMPLATES):
            return
        builder_name = _TEMPLATES[index]["builder"]
        builder_fn = globals()[builder_name]
        code = builder_fn()
        _load_code_onto_canvas(self.canvas, code)
        self.accept()


# =============================================================================
# Template builders
# =============================================================================


def _build_steane():
    from ..codes.css_code import CSSCode
    from ..util import pcm

    H = pcm.hamming(7)
    return CSSCode(HX=H, HZ=H)


def _build_rep3x3():
    from ..codes import HypergraphProductCode
    from ..util import pcm

    return HypergraphProductCode(pcm.repetition(3), pcm.repetition(3))


def _build_rep3x4():
    from ..codes import HypergraphProductCode
    from ..util import pcm

    return HypergraphProductCode(pcm.repetition(3), pcm.repetition(4))


def _build_ham7xham7():
    from ..codes import HypergraphProductCode
    from ..util import pcm

    return HypergraphProductCode(pcm.hamming(7), pcm.hamming(7))


def _build_bb72():
    from ..codes.bb import build_bb72

    return build_bb72()


def _build_bb144():
    from ..codes.bb import build_bb144

    return build_bb144()


# =============================================================================
# Canvas loader
# =============================================================================


def _load_code_onto_canvas(canvas: Canvas, code) -> None:
    """Clear the canvas and populate it with the code's Tanner graph.

    Uses NetworkX spring layout to produce 2D positions, then creates
    nodes and edges on the canvas model.
    """

    canvas.model.clear()

    # Build a NetworkX graph from the parity-check matrices.
    G = nx.Graph()
    data_qubits = list(range(len(code.data_qubits)))
    z_checks = list(range(len(code.z_check_qubits)))
    x_checks = list(range(len(code.x_check_qubits)))

    # Add nodes.
    for i in data_qubits:
        G.add_node(("data", i))
    for i in z_checks:
        G.add_node(("Z", i))
    for i in x_checks:
        G.add_node(("X", i))

    # Add edges from HZ (data ↔ Z-check).
    for row_idx in range(code.HZ.shape[0]):
        for col_idx in range(code.HZ.shape[1]):
            if code.HZ[row_idx, col_idx]:
                G.add_edge(("Z", row_idx), ("data", col_idx))

    # Add edges from HX (data ↔ X-check).
    for row_idx in range(code.HX.shape[0]):
        for col_idx in range(code.HX.shape[1]):
            if code.HX[row_idx, col_idx]:
                G.add_edge(("X", row_idx), ("data", col_idx))

    # Spring layout.
    pos = nx.spring_layout(G, seed=42, k=2.0, iterations=100)

    # Scale positions to canvas coordinates.
    scale = 40.0 * canvas.node_radius
    center_x = canvas.width() / 2
    center_y = canvas.height() / 2

    # Create canvas nodes.
    node_map: dict[tuple, int] = {}
    for key, (x, y) in pos.items():
        kind, idx = key
        if kind == "data":
            node_type = "qubit"
        elif kind == "Z":
            node_type = "Z_stabilizer"
        else:
            node_type = "X_stabilizer"
        canvas_x = center_x + x * scale
        canvas_y = center_y + y * scale
        node_id = canvas.model.add_node(canvas_x, canvas_y, node_type)
        node_map[key] = node_id

    # Create canvas edges.
    for u, v in G.edges():
        canvas.model.add_edge(node_map[u], node_map[v])

    canvas.update()

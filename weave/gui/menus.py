"""Menu construction for the canvas — hamburger menu and context menu."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QMenu, QWidgetAction, QHBoxLayout, QLabel, QWidget,
)
from PySide6.QtGui import QPainter, QPen, QIcon, QPixmap, QPainterPath, QAction
from PySide6.QtCore import Qt, QPointF

from .components import ToggleSwitch
from .drawing import get_crossing_number
from .graph_model import is_valid_connection

if TYPE_CHECKING:
    from .canvas import Canvas


# ------------------------------------------------------------------
# Icon helpers
# ------------------------------------------------------------------

def _make_clear_icon(fg_color) -> QIcon:
    icon = QIcon()
    pixmap = QPixmap(16, 16)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QPen(fg_color, 1))
    painter.setBrush(Qt.NoBrush)
    painter.drawRect(4, 5, 8, 9)
    painter.drawLine(3, 5, 13, 5)
    painter.drawLine(6, 3, 10, 3)
    painter.drawLine(6, 3, 6, 5)
    painter.drawLine(10, 3, 10, 5)
    painter.drawLine(6, 7, 6, 12)
    painter.drawLine(8, 7, 8, 12)
    painter.drawLine(10, 7, 10, 12)
    painter.end()
    icon.addPixmap(pixmap)
    return icon


def _make_save_icon(fg_color) -> QIcon:
    icon = QIcon()
    pixmap = QPixmap(16, 16)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QPen(fg_color, 1))
    painter.setBrush(Qt.NoBrush)
    painter.drawRect(3, 3, 10, 10)
    painter.drawRect(5, 4, 6, 3)
    painter.drawRect(11, 8, 2, 3)
    painter.end()
    icon.addPixmap(pixmap)
    return icon


def _make_load_icon(fg_color) -> QIcon:
    icon = QIcon()
    pixmap = QPixmap(16, 16)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QPen(fg_color, 1))
    painter.setBrush(Qt.NoBrush)
    painter.drawRect(3, 5, 10, 8)
    path = QPainterPath()
    path.moveTo(3, 5)
    path.lineTo(6, 3)
    path.lineTo(9, 3)
    path.lineTo(9, 5)
    painter.drawPath(path)
    painter.end()
    icon.addPixmap(pixmap)
    return icon


# ------------------------------------------------------------------
# Hamburger menu
# ------------------------------------------------------------------

def build_hamburger_menu(canvas: Canvas) -> QMenu:
    """Create the hamburger dropdown menu."""
    theme = canvas.theme_manager
    menu = QMenu(canvas)
    menu.setWindowFlags(menu.windowFlags() | Qt.FramelessWindowHint)
    menu.setAttribute(Qt.WA_TranslucentBackground)
    menu.setStyleSheet(theme.get_menu_style())

    menu.aboutToHide.connect(canvas._on_menu_hide)

    label_style = (
        f"font-family: 'Segoe UI', 'Helvetica Neue', sans-serif; "
        f"font-size: 12px; color: {theme.foreground.name()};"
    )

    # --- Show Crossings toggle ---
    crossings_widget = QWidget()
    crossings_layout = QHBoxLayout(crossings_widget)
    crossings_layout.setContentsMargins(8, 4, 8, 4)
    crossings_label = QLabel("Show Crossings")
    crossings_label.setStyleSheet(label_style)
    canvas.crossings_toggle = ToggleSwitch(canvas.show_crossings, crossings_widget)
    canvas.crossings_toggle.toggled.connect(canvas._on_crossings_toggled)
    crossings_layout.addWidget(crossings_label)
    crossings_layout.addStretch()
    crossings_layout.addWidget(canvas.crossings_toggle)
    crossings_action = QWidgetAction(canvas)
    crossings_action.setDefaultWidget(crossings_widget)
    menu.addAction(crossings_action)

    # --- Dark Mode toggle ---
    dark_mode_widget = QWidget()
    dark_mode_layout = QHBoxLayout(dark_mode_widget)
    dark_mode_layout.setContentsMargins(8, 4, 8, 4)
    dark_mode_label = QLabel("Dark Mode")
    dark_mode_label.setStyleSheet(label_style)
    canvas.dark_mode_toggle = ToggleSwitch(theme.dark_mode, dark_mode_widget)
    canvas.dark_mode_toggle.toggled.connect(canvas._on_dark_mode_toggled)
    dark_mode_layout.addWidget(dark_mode_label)
    dark_mode_layout.addStretch()
    dark_mode_layout.addWidget(canvas.dark_mode_toggle)
    dark_mode_action = QWidgetAction(canvas)
    dark_mode_action.setDefaultWidget(dark_mode_widget)
    menu.addAction(dark_mode_action)

    menu.addSeparator()

    # --- Canvas actions ---
    clear_action = menu.addAction("Clear Canvas", canvas._clear_canvas)
    clear_action.setIcon(_make_clear_icon(theme.foreground))

    save_action = menu.addAction("Save Canvas", canvas._save_canvas)
    save_action.setIcon(_make_save_icon(theme.foreground))

    load_action = menu.addAction("Load Canvas", canvas._load_canvas)
    load_action.setIcon(_make_load_icon(theme.foreground))

    menu.addSeparator()

    menu.addAction("Simulate...", canvas._run_simulation)
    menu.addAction("Export Code...", canvas._export_code)

    menu.addSeparator()

    # --- Crossing count display ---
    crossing_widget = QWidget()
    crossing_layout = QHBoxLayout(crossing_widget)
    crossing_layout.setContentsMargins(8, 4, 8, 4)
    crossing_label = QLabel(f"Crossings: {get_crossing_number(canvas.model)}")
    crossing_label.setStyleSheet(label_style)
    crossing_layout.addWidget(crossing_label)
    crossing_layout.addStretch()
    crossing_action = QWidgetAction(canvas)
    crossing_action.setDefaultWidget(crossing_widget)
    menu.addAction(crossing_action)

    return menu


def update_crossing_display(canvas: Canvas) -> None:
    """Update the crossing count label in the hamburger menu."""
    if not canvas._hamburger_menu:
        return
    for action in canvas._hamburger_menu.actions():
        if isinstance(action, QWidgetAction):
            widget = action.defaultWidget()
            if widget:
                for child in widget.findChildren(QLabel):
                    if "Crossings:" in child.text():
                        child.setStyleSheet(
                            f"font-family: 'Segoe UI', 'Helvetica Neue', sans-serif; "
                            f"font-size: 12px; color: {canvas.theme_manager.foreground.name()};"
                        )
                        child.setText(f"Crossings: {get_crossing_number(canvas.model)}")


# ------------------------------------------------------------------
# Context menu
# ------------------------------------------------------------------

def build_context_menu(canvas: Canvas, event) -> QMenu:
    """Create the right-click context menu."""
    theme = canvas.theme_manager
    menu = QMenu(canvas)
    menu.setWindowFlags(menu.windowFlags() | Qt.FramelessWindowHint)
    menu.setAttribute(Qt.WA_TranslucentBackground)
    style = theme.get_menu_style(is_context_menu=True)
    menu.setStyleSheet(style)

    pos = event.pos()

    # Classical node options.
    menu.addAction("New Bit", lambda: canvas.add_node_at(pos, "bit"))
    menu.addAction("New Parity Check", lambda: canvas.add_node_at(pos, "parity_check"))

    # Quantum node options.
    quantum_menu = menu.addMenu("New Quantum Node")
    quantum_menu.setWindowFlags(quantum_menu.windowFlags() | Qt.FramelessWindowHint)
    quantum_menu.setAttribute(Qt.WA_TranslucentBackground)
    quantum_menu.setStyleSheet(style)
    quantum_menu.addAction("New Qubit", lambda: canvas.add_node_at(pos, "qubit"))
    quantum_menu.addAction("New Z-Stabilizer", lambda: canvas.add_node_at(pos, "Z_stabilizer"))
    quantum_menu.addAction("New X-Stabilizer", lambda: canvas.add_node_at(pos, "X_stabilizer"))

    node = canvas.input_handler.get_node_at(pos)

    if node:
        if not node["selected"] and len(canvas.model.selected_nodes()) >= 1:
            canvas.model.deselect_all()
        node["selected"] = True
        canvas.update()

        menu.addSeparator()

        detect_action = QAction("Detect", menu)
        detect_action.triggered.connect(lambda: canvas.model.detect_graph(node["id"]))
        menu.addAction(detect_action)

        nodes_in_comp, _ = canvas.model.detect_connected_component(node["id"])
        detect_action.setEnabled(len(nodes_in_comp) > 2)

        for graph in canvas.model.graphs:
            if node["id"] in graph.node_ids:
                detect_action.setEnabled(False)
                break

        # Per-graph actions for detected graphs.
        for graph in canvas.model.graphs:
            if node["id"] in graph.node_ids:
                menu.addSeparator()
                menu.addAction("Configure && Simulate...",
                               lambda g=graph: canvas._run_graph_simulation(g))
                menu.addAction("Save Code...",
                               lambda g=graph: canvas._save_graph_code(g))
                break

        copy_action = menu.addAction("Copy", lambda: canvas.model.copy_selected())
        copy_action.setEnabled(len(canvas.model.selected_nodes()) > 0)

    if not node:
        menu.addSeparator()
    paste_action = menu.addAction("Paste", lambda: canvas._paste_at_widget(pos))
    paste_action.setEnabled(canvas.model.can_paste())

    if node:
        menu.addSeparator()
        menu.addAction("Export Code...", canvas._export_code)
        menu.addAction("Simulate...", canvas._run_simulation)

    return menu, node

"""Slim orchestrator canvas widget — delegates to graph_model, drawing, input_handler, menus, and code_bridge."""

from __future__ import annotations

from PySide6.QtWidgets import QWidget, QFileDialog, QMessageBox
from PySide6.QtGui import QPainter, QMouseEvent, QKeyEvent, QWheelEvent, QAction
from PySide6.QtCore import Qt, QPointF, QPropertyAnimation, QEasingCurve, Property

from .theme import ThemeManager
from .components import MenuIcon
from .graph_model import GraphModel, GraphData
from .input_handler import InputHandler
from . import drawing as draw
from . import menus as menu_builder
from . import code_bridge


class Canvas(QWidget):
    """Interactive canvas for editing quantum error-correcting codes."""

    def __init__(self, parent=None, dark_mode=False):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)

        # Theme.
        self.theme_manager = ThemeManager(dark_mode=dark_mode)

        # Data model.
        self.model = GraphModel(self)
        self.node_radius = self.model.node_radius

        # View transformation.
        self._view_offset = QPointF(0, 0)
        self._zoom = 1.0

        # Interaction state (owned by canvas, mutated by InputHandler).
        self.pan_active = False
        self.last_pan_point = None
        self.dragged_node = None
        self.drag_offset = QPointF(0, 0)
        self.graph_drag: GraphData | None = None
        self.graph_drag_initial_positions: dict[int, tuple] = {}

        self.selecting = False
        self.selection_rect_start = None
        self.selection_rect = None
        self.selection_mode = None

        self.drag_start = None
        self._drag_start_positions: dict[int, tuple] = {}

        self.shift_pending_toggle = None
        self._shift_press_node = None
        self._shift_press_pos = None
        self._shift_press_was_selected = False

        self.grid_mode = False
        self.grid_size = 2 * self.node_radius

        self.show_crossings = True

        # Delegates.
        self.input_handler = InputHandler(self)

        # Animations.
        self._setup_animations()

        # Hamburger menu.
        self._hamburger_menu = None
        self.create_hamburger_menu()

    # ------------------------------------------------------------------
    # Properties for Qt animation
    # ------------------------------------------------------------------

    def get_zoom(self):
        return self._zoom

    def set_zoom(self, value):
        self._zoom = value
        self.update()

    zoom_level = Property(float, get_zoom, set_zoom)

    def get_view_offset(self):
        return self._view_offset

    def set_view_offset(self, value):
        self._view_offset = value
        self.update()

    pan_offset = Property(QPointF, get_view_offset, set_view_offset)

    # ------------------------------------------------------------------
    # Animations
    # ------------------------------------------------------------------

    def smooth_zoom_to(self, new_zoom, center_pos=None):
        if center_pos is None:
            center_pos = QPointF(self.width() / 2, self.height() / 2)

        old_zoom = self._zoom
        screen_x = center_pos.x()
        screen_y = center_pos.y()
        world_x = (screen_x - self._view_offset.x()) / old_zoom
        world_y = (screen_y - self._view_offset.y()) / old_zoom

        new_offset = QPointF(screen_x - world_x * new_zoom, screen_y - world_y * new_zoom)

        self._zoom_animation.stop()
        self._pan_animation.stop()

        self._zoom_animation.setStartValue(old_zoom)
        self._zoom_animation.setEndValue(new_zoom)
        self._pan_animation.setStartValue(self._view_offset)
        self._pan_animation.setEndValue(new_offset)

        self._zoom_animation.start()
        self._pan_animation.start()

    def _setup_animations(self):
        self._zoom_animation = QPropertyAnimation(self, b"zoom_level")
        self._zoom_animation.setDuration(150)
        self._zoom_animation.setEasingCurve(QEasingCurve.OutCubic)

        self._pan_animation = QPropertyAnimation(self, b"pan_offset")
        self._pan_animation.setDuration(150)
        self._pan_animation.setEasingCurve(QEasingCurve.OutCubic)

    # ------------------------------------------------------------------
    # Hamburger menu
    # ------------------------------------------------------------------

    def create_hamburger_menu(self):
        if hasattr(self, "hamburger"):
            self.hamburger.deleteLater()
        self.hamburger = MenuIcon(self, self.theme_manager)
        self.hamburger.clicked.connect(self.toggle_hamburger_menu)
        self._position_hamburger()

    def toggle_hamburger_menu(self):
        if self._hamburger_menu is None:
            self.hamburger.set_open(True)
            self._hamburger_menu = menu_builder.build_hamburger_menu(self)
            hamburger_br = self.hamburger.mapToGlobal(
                QPointF(self.hamburger.width(), self.hamburger.height()).toPoint()
            )
            pos = QPointF(
                hamburger_br.x() - self._hamburger_menu.sizeHint().width(),
                hamburger_br.y() + 5,
            ).toPoint()
            self._hamburger_menu.popup(pos)
        else:
            self.hamburger.set_open(False)
            self._hamburger_menu.close()
            self._hamburger_menu = None

    def _position_hamburger(self):
        margin = 10
        self.hamburger.move(self.width() - self.hamburger.width() - margin, margin)

    def _on_menu_hide(self):
        if hasattr(self, "hamburger"):
            self.hamburger.set_open(False)
        self._hamburger_menu = None

    def _on_crossings_toggled(self, checked):
        self.show_crossings = checked
        self.update()

    def _on_dark_mode_toggled(self, checked):
        from PySide6.QtWidgets import QLabel, QWidgetAction

        self.theme_manager.set_dark_mode(checked)
        self.update()

        if self._hamburger_menu:
            self._hamburger_menu.setStyleSheet(self.theme_manager.get_menu_style())
            for action in self._hamburger_menu.actions():
                if isinstance(action, QWidgetAction):
                    widget = action.defaultWidget()
                    if widget:
                        for child in widget.findChildren(QLabel):
                            child.setStyleSheet(
                                f"font-family: 'Segoe UI', 'Helvetica Neue', sans-serif; "
                                f"font-size: 12px; color: {self.theme_manager.foreground.name()};"
                            )
            menu_builder.update_crossing_display(self)
            if hasattr(self, "hamburger"):
                self.hamburger.update()

    # ------------------------------------------------------------------
    # Qt event overrides — delegates to InputHandler
    # ------------------------------------------------------------------

    def contextMenuEvent(self, event):
        menu, node = menu_builder.build_context_menu(self, event)
        menu.exec(event.globalPos())
        self.update()
        if node:
            if len(self.model.selected_nodes()) == 1:
                node["selected"] = False
            self.update()

    def keyPressEvent(self, event: QKeyEvent):
        if not self.input_handler.handle_key_press(event):
            self.update()
            super().keyPressEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        self.input_handler.handle_wheel(event)

    def mousePressEvent(self, event: QMouseEvent):
        self.input_handler.handle_mouse_press(event)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self.input_handler.handle_mouse_release(event)
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if not self.input_handler.handle_double_click(event):
            super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        self.input_handler.handle_mouse_move(event)
        super().mouseMoveEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_hamburger()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), self.theme_manager.background)

        draw.draw_grid(painter, self.theme_manager, self._view_offset, self._zoom,
                       self.width(), self.height())

        if self.grid_mode:
            draw.draw_snap_grid(painter, self.theme_manager, self._view_offset, self._zoom,
                                self.grid_size, self.width(), self.height())

        if self.selecting and self.selection_rect is not None:
            draw.draw_selection_rect(painter, self.theme_manager, self.selection_rect,
                                     self._view_offset, self._zoom)

        painter.save()
        painter.translate(self._view_offset)
        painter.scale(self._zoom, self._zoom)

        draw.draw_graph_borders(painter, self.model, self.theme_manager)
        draw.draw_edges(painter, self.model, self.theme_manager)
        if self.show_crossings:
            draw.draw_crossings(painter, self.model, self.theme_manager)
        draw.draw_nodes(painter, self.model, self.theme_manager)

        painter.restore()

    # ------------------------------------------------------------------
    # Public interface (backward compatibility)
    # ------------------------------------------------------------------

    @property
    def nodes(self):
        return self.model.nodes

    @nodes.setter
    def nodes(self, value):
        self.model.nodes = value

    @property
    def edges(self):
        return self.model.edges

    @edges.setter
    def edges(self, value):
        self.model.edges = value

    @property
    def graphs(self):
        return self.model.graphs

    @graphs.setter
    def graphs(self, value):
        self.model.graphs = value

    @property
    def clipboard_nodes(self):
        return self.model.clipboard_nodes

    @clipboard_nodes.setter
    def clipboard_nodes(self, value):
        self.model.clipboard_nodes = value

    @property
    def clipboard_edges(self):
        return self.model.clipboard_edges

    @clipboard_edges.setter
    def clipboard_edges(self, value):
        self.model.clipboard_edges = value

    @property
    def clipboard_center(self):
        return self.model.clipboard_center

    @clipboard_center.setter
    def clipboard_center(self, value):
        self.model.clipboard_center = value

    def get_node_by_id(self, node_id):
        return self.model.get_node_by_id(node_id)

    def can_paste(self):
        return self.model.can_paste()

    def add_node_at(self, pos, node_type):
        adjusted_x = (pos.x() - self._view_offset.x()) / self._zoom
        adjusted_y = (pos.y() - self._view_offset.y()) / self._zoom
        if self.grid_mode:
            adjusted_x = round(adjusted_x / self.grid_size) * self.grid_size
            adjusted_y = round(adjusted_y / self.grid_size) * self.grid_size
        self.model.add_node(adjusted_x, adjusted_y, node_type)
        self.update()

    def save_to_file(self, filename):
        self.model.save_to_file(
            filename,
            view_offset=(self._view_offset.x(), self._view_offset.y()),
            zoom=self._zoom,
        )

    def load_from_file(self, filename):
        view_offset, zoom = self.model.load_from_file(filename)
        self._view_offset = QPointF(view_offset[0], view_offset[1])
        self._zoom = zoom
        self.update()

    # ------------------------------------------------------------------
    # Code bridge (backward compatibility)
    # ------------------------------------------------------------------

    def _validate_quantum_graph(self):
        return code_bridge.validate_quantum_graph(self.model)

    def to_css_code(self):
        return code_bridge.graph_to_css_code(self.model)

    def from_css_code(self, css_code, layout="spring"):
        code_bridge.css_code_to_model(self.model, css_code, layout)
        self._view_offset = QPointF(self.width() / 2, self.height() / 2)
        self._zoom = 1.0
        self.update()

    # ------------------------------------------------------------------
    # Actions triggered by menus
    # ------------------------------------------------------------------

    def _clear_canvas(self):
        self.model.clear()
        self.update()

    def _save_canvas(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Canvas", "", "JSON Files (*.json)")
        if file_path:
            if not file_path.endswith(".json"):
                file_path += ".json"
            self.save_to_file(file_path)

    def _load_canvas(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Canvas", "", "JSON Files (*.json)")
        if file_path:
            self.load_from_file(file_path)

    def _export_code(self):
        try:
            css_code = self.to_css_code()
        except ValueError as e:
            QMessageBox.warning(self, "Export Error", str(e))
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Export Code", "", "CSV Files (*.csv)")
        if not file_path:
            return
        if not file_path.endswith(".csv"):
            file_path += ".csv"
        code_bridge.export_code_csv(self.model, file_path)

    def _run_simulation(self):
        try:
            css_code = self.to_css_code()
        except ValueError as e:
            QMessageBox.warning(self, "Simulation Error", str(e))
            return

        from .simulation import SimulationDialog
        dialog = SimulationDialog(css_code, self)
        dialog.exec()

    def _run_graph_simulation(self, graph_data: GraphData):
        """Open simulation dialog for a specific detected graph."""
        try:
            css_code = code_bridge.graph_to_css_code(self.model, graph_data.node_ids)
        except ValueError as e:
            QMessageBox.warning(self, "Simulation Error", str(e))
            return

        from .simulation import SimulationDialog
        dialog = SimulationDialog(css_code, self, graph_data=graph_data)
        dialog.exec()

    def _save_graph_code(self, graph_data: GraphData):
        """Save a detected graph's code to a JSON file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Code", "", "JSON Files (*.json)"
        )
        if not file_path:
            return
        if not file_path.endswith(".json"):
            file_path += ".json"
        try:
            code_bridge.save_code_json(
                self.model, file_path,
                graph_data=graph_data,
                noise_config=graph_data.noise_config,
                logical_indices=graph_data.logical_indices,
                name=graph_data.name,
            )
        except ValueError as e:
            QMessageBox.warning(self, "Save Error", str(e))

    def _paste_at_widget(self, pos):
        """Paste at a widget position (used by context menu)."""
        wx = (pos.x() - self._view_offset.x()) / self._zoom
        wy = (pos.y() - self._view_offset.y()) / self._zoom
        if self.grid_mode:
            wx = round(wx / self.grid_size) * self.grid_size
            wy = round(wy / self.grid_size) * self.grid_size
        self.model.paste_at(wx, wy)
        self.update()

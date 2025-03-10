import json
import math
from typing import Any

from PySide6.QtWidgets import QWidget, QMenu, QWidgetAction, QHBoxLayout, QLabel
from PySide6.QtGui import QPainter, QPen, QColor, QMouseEvent, QKeyEvent, QWheelEvent, QRadialGradient, QIcon, QPixmap, \
    QLinearGradient, QBrush
from PySide6.QtCore import Qt, QPointF, QRectF, QPropertyAnimation, QEasingCurve, Property

from .theme import ThemeManager
from .components import ToggleSwitch, MenuIcon
from ..util.graph import find_edge_crossings, line_intersection


def _is_valid_connection(source: dict[str, Any], target: dict[str, Any]) -> bool:
    """
    Check if a connection between two nodes is valid.
    """
    quantum_types = {"qubit", "Z_stabilizer", "X_stabilizer"}
    classical_types = {"bit", "parity_check"}

    # If both nodes are quantum, allow only qubit–stabilizer connections. If both are classical, allow only
    # different types.
    if source["type"] in quantum_types and target["type"] in quantum_types:
        return ((source["type"] == "qubit" and target["type"] in {"Z_stabilizer", "X_stabilizer"}) or
                (target["type"] == "qubit" and source["type"] in {"Z_stabilizer", "X_stabilizer"}))
    elif source["type"] in classical_types and target["type"] in classical_types:
        return source["type"] != target["type"]
    else:
        return False


def _distance_point_to_segment(p: QPointF, a: QPointF, b: QPointF) -> float:
    """
    Compute the distance from point p to the line segment ab.

    Parameters
    ----------
    p : QPointF
        The point.
    a : QPointF
        Start point of the segment.
    b : QPointF
        End point of the segment.

    Returns
    -------
    float
        The distance from p to the segment.
    """
    ax, ay = a.x(), a.y()
    bx, by = b.x(), b.y()
    px, py = p.x(), p.y()
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)


class Canvas(QWidget):
    """
    An interactive canvas for editing quantum error-correcting codes.
    """

    def __init__(self, parent=None, dark_mode=False):
        """
        Initialize the canvas.

        Parameters
        ----------
        parent : QWidget, optional
            The parent widget (default is None).
        dark_mode : bool, optional
            Whether to start in dark mode (default is False).
        """
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)

        # Initialize theme manager.
        self.theme_manager = ThemeManager(dark_mode=dark_mode)

        # World model: nodes and edges.
        # Each node is a dict: {'id', 'pos', 'type', 'selected'}
        # Each edge is a dict: {'source', 'target', 'selected'}
        self.nodes = []
        self.edges = []
        self.node_radius = 10

        # View transformation parameters.
        self._view_offset = QPointF(0, 0)  # in widget coordinates
        self._zoom = 1.0

        # State for panning and dragging.
        self.pan_active = False
        self.last_pan_point = None  # QPointF
        self.dragged_node = None
        self.drag_offset = QPointF(0, 0)

        # Selection state.
        self.selecting = False  # whether a rectangular selection is active
        self.selection_rect_start = None  # the starting world coordinate of the selection
        self.selection_rect = None  # current selection rectangle (x_min, y_min, x_max, y_max)
        self.selection_mode = None  # "node" or "edge" selection mode

        # Drag state.
        self.drag_start = None  # world coordinate of the start of a drag
        self._drag_start_positions = {}  # dictionary to store initial positions of selected nodes when starting a drag

        # Selection state.
        self.shift_pending_toggle = None  # will hold a reference to the node pending deselection
        self._shift_press_node = None  # store node clicked with shift
        self._shift_press_pos = None  # store position of shift-click
        self._shift_press_was_selected = False  # store node's selection state before shift-click

        # Display options.
        self.show_crossings = True

        # Initialize animations.
        self._setup_animations()

        # Create hamburger menu.
        self._hamburger_menu = None
        self.create_hamburger_menu()

    # ------------------------------------------------------------
    # Property Getters and Setters
    # ------------------------------------------------------------

    def get_zoom(self):
        return self._zoom

    def set_zoom(self, value):
        self._zoom = value
        self.update()

    # Define the zoom property for Qt animation.
    zoom_level = Property(float, get_zoom, set_zoom)

    def get_view_offset(self):
        return self._view_offset

    def set_view_offset(self, value):
        self._view_offset = value
        self.update()

    # Define the view_offset property for Qt animation.
    pan_offset = Property(QPointF, get_view_offset, set_view_offset)

    # ------------------------------------------------------------
    # Animations
    # ------------------------------------------------------------

    def _setup_animations(self):
        """Setup animations for UI elements."""
        # Zoom animation.
        self._zoom_animation = QPropertyAnimation(self, b"zoom_level")
        self._zoom_animation.setDuration(150)
        self._zoom_animation.setEasingCurve(QEasingCurve.OutCubic)

        # Pan animation.
        self._pan_animation = QPropertyAnimation(self, b"pan_offset")
        self._pan_animation.setDuration(150)
        self._pan_animation.setEasingCurve(QEasingCurve.OutCubic)

    def smooth_zoom_to(self, new_zoom, center_pos=None):
        """
        Smoothly zoom to a new level with animation.

        Parameters
        ----------
        new_zoom : float
            Target zoom level.
        center_pos : QPointF, optional
            Center point for the zoom operation. If None, uses widget center.
        """
        if center_pos is None:
            center_pos = QPointF(self.width() / 2, self.height() / 2)

        old_zoom = self._zoom

        # Calculate the world point under the cursor.
        world_point = QPointF(
            (center_pos.x() - self._view_offset.x()) / old_zoom,
            (center_pos.y() - self._view_offset.y()) / old_zoom
        )

        # Calculate new view offset to keep world point fixed.
        new_offset = center_pos - world_point * new_zoom

        # Stop any running animations.
        self._zoom_animation.stop()
        self._pan_animation.stop()

        # Setup and start zoom animation.
        self._zoom_animation.setStartValue(old_zoom)
        self._zoom_animation.setEndValue(new_zoom)

        # Setup and start pan animation.
        self._pan_animation.setStartValue(self._view_offset)
        self._pan_animation.setEndValue(new_offset)

        # Start animations together.
        self._zoom_animation.start()
        self._pan_animation.start()

    def _set_dark_mode(self, checked):
        # Remember if the hamburger menu was open.
        hamburger_was_open = self._hamburger_menu is not None

        # Update theme.
        self.theme_manager.set_dark_mode(checked)

        # Force widget update to reflect new colors.
        self.update()

        # Only close and reopen the menu if it was open.
        if hamburger_was_open:
            # Store the menu position before closing.
            menu_pos = None
            if self._hamburger_menu:
                menu_pos = self._hamburger_menu.pos()
                self._hamburger_menu.close()

            # Recreate the menu with new theme.
            self._hamburger_menu = None
            self._create_hamburger_menu()

            # Restore menu position if we had one.
            if menu_pos:
                self._hamburger_menu.move(menu_pos)

            # Keep hamburger icon in open state.
            if hasattr(self, 'hamburger'):
                self.hamburger.set_open(True)

    def _set_show_crossings(self, checked):
        self.show_crossings = checked
        self.update()

    def keyPressEvent(self, event: QKeyEvent):
        """
        Ctrl+0: Reset zoom to default.
        Ctrl+=: Zoom in.
        Ctrl+-: Zoom out.
        Ctrl+A: Select all nodes.
        Escape: Deselect all objects.
        Delete/Backspace: Remove selected nodes/edges.
        """
        if event.key() == Qt.Key_A and event.modifiers() & Qt.ControlModifier:
            # Select all nodes.
            for node in self.nodes:
                node['selected'] = True
            self.update()
            return
        elif event.key() == Qt.Key_0 and event.modifiers() & Qt.ControlModifier:
            # Reset zoom with animation.
            self.smooth_zoom_to(1.0)
            return
        elif event.key() == Qt.Key_Equal and event.modifiers() & Qt.ControlModifier:
            # Zoom in with animation.
            new_zoom = min(self._zoom * 1.2, 5.0)
            self.smooth_zoom_to(new_zoom)
            return
        elif event.key() == Qt.Key_Minus and event.modifiers() & Qt.ControlModifier:
            # Zoom out with animation.
            new_zoom = max(self._zoom / 1.2, 0.2)
            self.smooth_zoom_to(new_zoom)
            return
        elif event.key() == Qt.Key_Escape:
            self._deselect_all()
        elif event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            selected_nodes = self._get_selected_nodes()
            if selected_nodes:
                ids_to_remove = {node['id'] for node in selected_nodes}
                self.nodes = [n for n in self.nodes if n['id'] not in ids_to_remove]
                self.edges = [e for e in self.edges if
                              e['source'] not in ids_to_remove and e['target'] not in ids_to_remove]
                self._deselect_all()
            else:
                selected_edges = self._get_selected_edges()
                if selected_edges:
                    for edge in selected_edges:
                        self.edges.remove(edge)
                    self._deselect_all()
            self.update()

        self.update()
        super().keyPressEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        factor = 1.1 if delta > 0 else 0.9

        # Calculate new zoom level within limits.
        new_zoom = self._zoom * factor
        new_zoom = max(0.2, min(new_zoom, 5.0))

        # Don't animate tiny changes.
        if abs(new_zoom - self._zoom) < 0.01:
            return

        # Perform smooth zoom centered on cursor.
        self.smooth_zoom_to(new_zoom, event.position())

    # ------------------------------------------------------------
    # Main Menu
    # ------------------------------------------------------------

    def create_hamburger_menu(self):
        """Create the hamburger menu button."""
        # Remove old hamburger if it exists.
        if hasattr(self, 'hamburger'):
            self.hamburger.deleteLater()

        # Create new modern hamburger icon.
        self.hamburger = MenuIcon(self, self.theme_manager)
        self.hamburger.clicked.connect(self.toggle_hamburger_menu)

        # Position the hamburger.
        self._position_hamburger()

    def toggle_hamburger_menu(self):
        """Toggle the hamburger menu open/closed state."""
        if self._hamburger_menu is None:
            # Create menu.
            self.hamburger.set_open(True)
            self._create_hamburger_menu()
        else:
            # Close menu.
            self.hamburger.set_open(False)
            self._hamburger_menu.close()
            self._hamburger_menu = None

    def _create_hamburger_menu(self):
        """Create and display the hamburger menu."""
        self._hamburger_menu = QMenu(self)
        self._hamburger_menu.setWindowFlags(self._hamburger_menu.windowFlags() | Qt.FramelessWindowHint)
        self._hamburger_menu.setAttribute(Qt.WA_TranslucentBackground)
        self._hamburger_menu.setStyleSheet(self.theme_manager.get_menu_style())

        # Connect to aboutToHide to ensure hamburger icon resets when menu closes.
        self._hamburger_menu.aboutToHide.connect(self._on_menu_hide)

        # Create actions for toggles.
        crossings_widget = QWidget()
        crossings_layout = QHBoxLayout(crossings_widget)
        crossings_layout.setContentsMargins(8, 4, 8, 4)

        crossings_label = QLabel("Show Crossings")
        crossings_label.setStyleSheet(
            f"font-family: 'Segoe UI', 'Helvetica Neue', sans-serif; font-size: 12px; color: {self.theme_manager.foreground.name()};"
        )

        # Create an explicitly initialized toggle for show crossings.
        self.crossings_toggle = ToggleSwitch(self.show_crossings, crossings_widget)
        self.crossings_toggle.toggled.connect(self._on_crossings_toggled)

        crossings_layout.addWidget(crossings_label)
        crossings_layout.addStretch()
        crossings_layout.addWidget(self.crossings_toggle)

        crossings_action = QWidgetAction(self)
        crossings_action.setDefaultWidget(crossings_widget)
        self._hamburger_menu.addAction(crossings_action)

        # Create dark mode toggle widget.
        dark_mode_widget = QWidget()
        dark_mode_layout = QHBoxLayout(dark_mode_widget)
        dark_mode_layout.setContentsMargins(8, 4, 8, 4)

        dark_mode_label = QLabel("Dark Mode")
        dark_mode_label.setStyleSheet(
            f"font-family: 'Segoe UI', 'Helvetica Neue', sans-serif; font-size: 12px; color: {self.theme_manager.foreground.name()};"
        )

        # Explicitly create a new toggle switch with the current theme state.
        self.dark_mode_toggle = ToggleSwitch(self.theme_manager.dark_mode, dark_mode_widget)
        self.dark_mode_toggle.toggled.connect(self._on_dark_mode_toggled)

        dark_mode_layout.addWidget(dark_mode_label)
        dark_mode_layout.addStretch()
        dark_mode_layout.addWidget(self.dark_mode_toggle)

        dark_mode_action = QWidgetAction(self)
        dark_mode_action.setDefaultWidget(dark_mode_widget)
        self._hamburger_menu.addAction(dark_mode_action)

        # Add a separator.
        self._hamburger_menu.addSeparator()

        # Add Clear Canvas action.
        clear_action = self._hamburger_menu.addAction("Clear Canvas", self._clear_canvas)
        clear_action.setIcon(self._get_clear_icon())

        # Add separator.
        self._hamburger_menu.addSeparator()

        # Add crossing number display.
        crossing_widget = QWidget()
        crossing_layout = QHBoxLayout(crossing_widget)
        crossing_layout.setContentsMargins(8, 4, 8, 4)

        crossing_label = QLabel(f"Crossings: {self._get_crossing_number()}")
        crossing_label.setStyleSheet(
            f"font-family: 'Segoe UI', 'Helvetica Neue', sans-serif; font-size: 12px; color: {self.theme_manager.foreground.name()};"
        )
        crossing_layout.addWidget(crossing_label)
        crossing_layout.addStretch()

        crossing_action = QWidgetAction(self)
        crossing_action.setDefaultWidget(crossing_widget)
        self._hamburger_menu.addAction(crossing_action)

        # Position and show the menu.
        hamburger_bottom_right = self.hamburger.mapToGlobal(
            QPointF(self.hamburger.width(), self.hamburger.height()).toPoint())
        pos = QPointF(hamburger_bottom_right.x() - self._hamburger_menu.sizeHint().width(),
                      hamburger_bottom_right.y() + 5).toPoint()
        self._hamburger_menu.popup(pos)

    def _create_toggle_widget(self, label_text, initial, callback):
        """
        Create a toggle widget with label.

        Parameters
        ----------
        label_text : str
            Text label for the toggle.
        initial : bool
            Initial state of the toggle.
        callback : function
            Function to call when toggle state changes.

        Returns
        -------
        QWidgetAction
            The widget action for the menu.
        """
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(8, 4, 8, 4)

        # Create label first.
        label = QLabel(label_text)
        label.setStyleSheet(
            f"font-family: 'Segoe UI', 'Helvetica Neue', sans-serif; font-size: 12px; color: {self.theme_manager.foreground.name()};"
        )

        # Create toggle with explicit starting state.
        toggle = ToggleSwitch(initial, widget)
        toggle.toggled.connect(callback)

        # Add label and toggle in correct order (label on left, toggle on right).
        layout.addWidget(label)
        layout.addStretch()
        layout.addWidget(toggle)

        action = QWidgetAction(self)
        action.setDefaultWidget(widget)
        return action

    # TODO: Switch to icon files as assets.
    def _get_clear_icon(self):
        """Create a clear/trash icon for the menu."""
        icon = QIcon()
        pixmap = QPixmap(16, 16)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw a simple trash can icon.
        painter.setPen(QPen(self.theme_manager.foreground, 1))
        painter.setBrush(Qt.NoBrush)

        # Draw the trash can body.
        painter.drawRect(4, 5, 8, 9)

        # Draw the lid.
        painter.drawLine(3, 5, 13, 5)

        # Draw the handle.
        painter.drawLine(6, 3, 10, 3)
        painter.drawLine(6, 3, 6, 5)
        painter.drawLine(10, 3, 10, 5)

        # Draw lines inside to represent trash.
        painter.drawLine(6, 7, 6, 12)
        painter.drawLine(8, 7, 8, 12)
        painter.drawLine(10, 7, 10, 12)

        painter.end()

        icon.addPixmap(pixmap)
        return icon

    # ------------------------------------------------------------

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.setWindowFlags(menu.windowFlags() | Qt.FramelessWindowHint)
        menu.setAttribute(Qt.WA_TranslucentBackground)

        # Get context menu style.
        style = self.theme_manager.get_menu_style(is_context_menu=True)
        menu.setStyleSheet(style)

        # Classical node options.
        menu.addAction("New Bit", lambda: self.add_node_at(event.pos(), "bit"))
        menu.addAction("New Parity Check", lambda: self.add_node_at(event.pos(), "parity_check"))

        # Quantum node options.
        quantum_menu = menu.addMenu("New Quantum Node")
        quantum_menu.setWindowFlags(quantum_menu.windowFlags() | Qt.FramelessWindowHint)
        quantum_menu.setAttribute(Qt.WA_TranslucentBackground)

        quantum_menu.setStyleSheet(style)

        quantum_menu.addAction("New Qubit", lambda: self.add_node_at(event.pos(), "qubit"))
        quantum_menu.addAction("New Z-Stabilizer", lambda: self.add_node_at(event.pos(), "Z_stabilizer"))
        quantum_menu.addAction("New X-Stabilizer", lambda: self.add_node_at(event.pos(), "X_stabilizer"))

        # Save option.
        menu.addAction("Save Code as CSV", lambda: print("Save functionality not implemented yet."))
        menu.exec(event.globalPos())
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        """
        Left-click: Selects nodes or edges and begins dragging or panning.
        Ctrl+left-click: Creates an edge between nodes.
        """
        pos = event.position()  # QPointF
        if event.button() == Qt.LeftButton:
            clicked_node = self._get_node_at(pos)
            if clicked_node:
                if event.modifiers() & Qt.ControlModifier:
                    selected_nodes = self._get_selected_nodes()
                    if selected_nodes:
                        for sn in selected_nodes:
                            if sn != clicked_node and _is_valid_connection(sn, clicked_node):
                                if not self._edge_exists(sn['id'], clicked_node['id']):
                                    self.edges.append({
                                        'source': sn['id'],
                                        'target': clicked_node['id'],
                                        'selected': False
                                    })
                        self._deselect_all()  # optionally, clear selection after edge creation
                    else:
                        clicked_node['selected'] = True
                    self.update()
                elif event.modifiers() & Qt.ShiftModifier:
                    # Record the node, its press position, and its initial selection state.
                    self._shift_press_node = clicked_node
                    self._shift_press_pos = event.position()
                    self._shift_press_was_selected = clicked_node.get('selected', False)
                    # Record drag-start positions for all selected nodes.
                    self._drag_start_positions = {n['id']: n['pos'] for n in self._get_selected_nodes()}
                    self.drag_start = event.position()
                else:
                    # Normal click: clear all selections and select only the clicked node.
                    self._deselect_all()
                    clicked_node['selected'] = True
                    self._drag_start_positions = {clicked_node['id']: clicked_node['pos']}
                    self.drag_start = event.position()
                self.update()
            else:
                clicked_edge = self._get_edge_at(pos)
                if clicked_edge:
                    if event.modifiers() & Qt.ShiftModifier:
                        clicked_edge['selected'] = True
                    else:
                        self._deselect_all()
                        clicked_edge['selected'] = True
                else:
                    if event.modifiers() & Qt.ShiftModifier:
                        self.selecting = True
                        self.drag_start = None
                        self._drag_start_positions = {}
                        world_pos = ((pos.x() - self._view_offset.x()) / self._zoom,
                                     (pos.y() - self._view_offset.y()) / self._zoom)
                        self.selection_rect_start = world_pos
                        self.selection_mode = "edge" if event.modifiers() & Qt.ControlModifier else "node"
                    else:
                        self._deselect_all()
                        self.pan_active = True
                        self.last_pan_point = pos
                self.update()
        self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.drag_start is not None:
            delta = event.position() - self.drag_start
            delta_world = (delta.x() / self._zoom, delta.y() / self._zoom)
            for node in self._get_selected_nodes():
                init_pos = self._drag_start_positions[node['id']]
                node['pos'] = (init_pos[0] + delta_world[0], init_pos[1] + delta_world[1])
            self.update()

        pos = event.position()
        if self.selecting:
            current = ((pos.x() - self._view_offset.x()) / self._zoom,
                       (pos.y() - self._view_offset.y()) / self._zoom)
            x1, y1 = self.selection_rect_start
            x2, y2 = current
            self.selection_rect = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            self.update()

        if self.dragged_node is not None:
            new_center = pos - self.drag_offset
            world_x = (new_center.x() - self._view_offset.x()) / self._zoom
            world_y = (new_center.y() - self._view_offset.y()) / self._zoom
            self.dragged_node['pos'] = (world_x, world_y)
            self.update()
        elif self.pan_active and self.last_pan_point is not None:
            delta = pos - self.last_pan_point
            self._view_offset += delta
            self.last_pan_point = pos
            self.update()

        if self.drag_start is not None:
            delta = event.position() - self.drag_start
            # If the movement is greater than a small threshold, cancel the pending toggle.
            if delta.manhattanLength() > 10:
                self.shift_pending_toggle = None

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.pan_active:
            self.pan_active = False
            self.last_pan_point = None

        if self.dragged_node is not None:
            self.dragged_node = None

        if self.selecting and self.selection_rect is not None:
            x_min, y_min, x_max, y_max = self.selection_rect
            if self.selection_mode == "node":
                for node in self.nodes:
                    x, y = node['pos']
                    if x_min <= x <= x_max and y_min <= y <= y_max:
                        node['selected'] = True
            elif self.selection_mode == "edge":
                for edge in self.edges:
                    source = self.get_node_by_id(edge['source'])
                    target = self.get_node_by_id(edge['target'])
                    if source and target:
                        x1, y1 = source['pos']
                        x2, y2 = target['pos']

                        # Select edge if both endpoints are inside the rectangle.
                        if (x_min <= x1 <= x_max and y_min <= y1 <= y_max and
                                x_min <= x2 <= x_max and y_min <= y2 <= y_max):
                            edge['selected'] = True
            self.selecting = False
            self.selection_rect_start = None
            self.selection_rect = None
            self.selection_mode = None
            self.drag_start = None
            self._drag_start_positions = {}
            self.update()

        self.drag_start = None
        self._drag_start_positions = {}

        if hasattr(self, '_shift_press_node') and self._shift_press_node is not None:
            # If there was minimal movement (i.e. no drag occurred), toggle the node's selection.
            if (event.position() - self._shift_press_pos).manhattanLength() < 10:
                # Toggle the selection: if it was selected, deselect it; if it was not, select it.
                self._shift_press_node['selected'] = not self._shift_press_was_selected
            self._shift_press_node = None
            self.update()

        super().mouseReleaseEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_hamburger()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        # Fill background.
        painter.fillRect(self.rect(), self.theme_manager.background)

        # ----- Draw grid (triangular lattice) in widget coordinates -----
        spacing = 20
        dot_radius = 1
        painter.setPen(QPen(self.theme_manager.grid, 0.5))

        # Compute grid indices from view_offset and widget dimensions.
        vox = self._view_offset.x()
        voy = self._view_offset.y()
        width = self.width()
        height = self.height()
        min_n = math.floor((-vox) / spacing)
        max_n = math.ceil((width - vox) / spacing)
        min_m = math.floor((-voy) / spacing)
        max_m = math.ceil((height - voy) / spacing)

        for m in range(min_m, max_m + 1):
            y = m * spacing + voy
            row_offset = spacing / 2 if (m % 2 != 0) else 0
            for n in range(min_n, max_n + 1):
                x = n * spacing + vox + row_offset
                painter.drawEllipse(QPointF(x, y), dot_radius, dot_radius)

        # Draw selection area.
        if self.selecting and self.selection_rect is not None:
            painter.setPen(Qt.NoPen)
            # Create a semi-transparent selection color.
            selection_color = QColor(self.theme_manager.selected)
            painter.setBrush(selection_color)

            # Convert world coordinates back to widget coordinates.
            rect = QRectF(
                self.selection_rect[0] * self._zoom + self._view_offset.x(),
                self.selection_rect[1] * self._zoom + self._view_offset.y(),
                (self.selection_rect[2] - self.selection_rect[0]) * self._zoom,
                (self.selection_rect[3] - self.selection_rect[1]) * self._zoom
            )
            painter.drawRoundedRect(rect, 5, 5)

            # Reset the brush to avoid affecting other drawing.
            painter.setBrush(Qt.NoBrush)

        # ----- Draw world objects (nodes and edges) -----
        painter.save()
        painter.translate(self._view_offset)
        painter.scale(self._zoom, self._zoom)

        # Draw edges.
        self._draw_edges(painter)

        # Draw crossings.
        if self.show_crossings:
            self._draw_crossings(painter)

        # Draw nodes.
        self._draw_nodes(painter)

        painter.restore()

    def _draw_edges(self, painter):
        for edge in self.edges:
            source = self.get_node_by_id(edge['source'])
            target = self.get_node_by_id(edge['target'])
            if source is None or target is None:
                continue

            src_center = QPointF(source['pos'][0], source['pos'][1])
            tgt_center = QPointF(target['pos'][0], target['pos'][1])

            # Check if the nodes are quantum: no clipping prevention is necessary.
            if source["type"] in {"bit", "parity_check"} and target["type"] in {"bit", "parity_check"}:
                dx = tgt_center.x() - src_center.x()
                dy = tgt_center.y() - src_center.y()
                dist = math.hypot(dx, dy)
                if dist == 0:
                    continue

                # Compute margins so edges begin at node boundaries.
                margin_source = self._get_margin(source, dx, dy)
                margin_target = self._get_margin(target, -dx, -dy)
                if margin_source + margin_target > dist:
                    continue

                src = QPointF(src_center.x() + dx / dist * margin_source,
                              src_center.y() + dy / dist * margin_source)
                tgt = QPointF(tgt_center.x() - dx / dist * margin_target,
                              tgt_center.y() - dy / dist * margin_target)
            else:
                src, tgt = src_center, tgt_center

            pen = QPen(self.theme_manager.foreground, 0.8)
            pen.setCapStyle(Qt.FlatCap)
            painter.setPen(pen)
            painter.drawLine(src, tgt)

            if edge.get('selected', False):
                highlight_size = 5
                highlight_color = self.theme_manager.selected

                pen = QPen(highlight_color, highlight_size)
                pen.setCapStyle(Qt.FlatCap)
                painter.setPen(pen)
                painter.drawLine(src, tgt)

    def _draw_crossings(self, painter):
        # Filter quantum nodes and edges.
        quantum_types = {"qubit", "Z_stabilizer", "X_stabilizer"}
        qnodes = {node['id']: node for node in self.nodes if node['type'] in quantum_types}
        qedges = [edge for edge in self.edges
                  if self.get_node_by_id(edge['source']) and self.get_node_by_id(edge['target'])
                  and self.get_node_by_id(edge['source'])['type'] in quantum_types
                  and self.get_node_by_id(edge['target'])['type'] in quantum_types]

        if not qnodes or not qedges:
            return

        # Build a mapping from quantum node id to a continuous index and a list of positions.
        qnode_ids = list(qnodes.keys())
        id_to_index = {node_id: i for i, node_id in enumerate(qnode_ids)}
        pos_list = [qnodes[node_id]['pos'] for node_id in qnode_ids]

        # Build edge list as tuples of indices.
        edge_list = []
        for edge in qedges:
            try:
                i = id_to_index[edge['source']]
                j = id_to_index[edge['target']]
                edge_list.append((i, j))
            except KeyError:
                continue

        # For each crossing, compute approximate intersection and draw a diamond.
        crossings = find_edge_crossings(pos_list, edge_list)
        for crossing in crossings:
            # Each crossing is a frozenset of two edges, extract them.
            edge_pair = list(crossing)
            e1, e2 = edge_pair[0], edge_pair[1]

            # Get the endpoints (in world coordinates) for each edge.
            def get_endpoints(e):
                try:
                    n1 = qnodes[qnode_ids[e[0]]]['pos']
                    n2 = qnodes[qnode_ids[e[1]]]['pos']
                    return n1, n2
                except (KeyError, IndexError):
                    return None, None

            a, b = get_endpoints(e1)
            c, d = get_endpoints(e2)

            if None in (a, b, c, d):
                continue

            ip = line_intersection(a, b, c, d)
            if ip is not None:
                size = 4  # size of the square in world units

                # Draw the diamond.
                painter.save()
                painter.translate(ip[0], ip[1])
                painter.rotate(45)

                # Use a light color that contrasts with the background.
                if self.theme_manager.dark_mode:
                    diamond_color = QColor(255, 255, 255, 180)
                else:
                    diamond_color = QColor(0, 0, 0, 180)

                painter.setBrush(diamond_color)
                painter.setPen(Qt.NoPen)
                painter.drawRect(QRectF(-size / 2, -size / 2, size, size))
                painter.restore()

    def _draw_nodes(self, painter):
        for node in self.nodes:
            x = node['pos'][0]
            y = node['pos'][1]
            r = self.node_radius
            l = 1.86 * r
            node_type = node['type']

            # Determine if node is selected.
            is_selected = node.get('selected', False)

            # Set node color and pen.
            node_color = self.theme_manager.get_node_color(node_type)
            pen = QPen(self.theme_manager.foreground, 1)

            # Draw selection highlight if selected.
            if is_selected:
                # Draw a bigger highlight circle/square behind the node.
                highlight_size = 2
                highlight_color = self.theme_manager.selected

                painter.setPen(Qt.NoPen)
                painter.setBrush(highlight_color)

                if node_type == "bit" or node_type == "qubit":
                    painter.drawEllipse(QPointF(x, y), r + highlight_size, r + highlight_size)
                else:
                    painter.drawRoundedRect(QRectF(x - l / 2 - highlight_size, y - l / 2 - highlight_size,
                                                   l + 2 * highlight_size, l + 2 * highlight_size), 1, 1)

            # Draw the actual node.
            if node_type in {"bit", "parity_check"}:
                # Classical nodes have outlines.
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)

                if node_type == "bit":
                    painter.drawEllipse(QPointF(x, y), r, r)
                else:
                    painter.drawRect(QRectF(x - l / 2, y - l / 2, l, l))
            else:
                # Quantum nodes have fills.
                painter.setPen(Qt.NoPen)
                painter.setBrush(node_color)

                if node_type == "qubit":
                    painter.drawEllipse(QPointF(x, y), r, r)
                else:  # Z or X stabilizer
                    painter.drawRect(QRectF(x - l / 2, y - l / 2, l, l))

            # Reset brush for next node.
            painter.setBrush(Qt.NoBrush)

    def save_to_file(self, filename):
        """
        Save the current nodes and edges to a JSON file.

        Parameters
        ----------
        filename : str
            The path to the file.
        """
        data = {'nodes': self.nodes, 'edges': self.edges}
        with open(filename, 'w') as f:
            json.dump(data, f)

    def load_from_file(self, filename):
        """
        Load nodes and edges from a JSON file.

        Parameters
        ----------
        filename : str
            The path to the file.
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        self.nodes = data.get('nodes', [])
        self.edges = data.get('edges', [])
        self.update()

    def add_node_at(self, pos, node_type):
        """
        Add a new node at the given widget position.

        The position is converted to world coordinates based on the current view_offset and zoom.

        Parameters
        ----------
        pos : QPointF
            The position in widget coordinates.
        node_type : str
            The type of node to create ('bit' or 'parity_check').
        """
        adjusted_x = (pos.x() - self._view_offset.x()) / self._zoom
        adjusted_y = (pos.y() - self._view_offset.y()) / self._zoom
        new_node = {
            'id': len(self.nodes),
            'pos': (adjusted_x, adjusted_y),
            'type': node_type,
            'selected': False
        }
        self.nodes.append(new_node)
        self.update()

    def get_node_by_id(self, node_id):
        """
        Return the node with the specified id.

        Parameters
        ----------
        node_id : int
            The node identifier.

        Returns
        -------
        dict or None
            The node dictionary, or None if not found.
        """
        for node in self.nodes:
            if node['id'] == node_id:
                return node
        return None

    def _get_margin(self, node, dx, dy):
        """
        Compute the margin for edge clipping from a node's center.

        For 'bit' nodes, the margin is simply node_radius. For square nodes ('parity_check'), compute the distance to
        the square's boundary along the ray (dx, dy) and add an epsilon.

        Parameters
        ----------
        node : dict
            The node dictionary.
        dx : float
            Difference in x from source to target.
        dy : float
            Difference in y from source to target.

        Returns
        -------
        float
            The margin distance.
        """
        r = self.node_radius
        if node['type'] == "bit":
            return r
        else:
            if dx == 0 and dy == 0:
                return r
            dist = math.hypot(dx, dy)
            cos_theta = abs(dx) / dist
            sin_theta = abs(dy) / dist
            epsilon = 0.5
            return r / max(cos_theta, sin_theta) - epsilon

    def _get_node_at(self, pos):
        """
        Return the node at the given widget position, if any.

        Parameters
        ----------
        pos : QPointF
            The position in widget coordinates.

        Returns
        -------
        dict or None
            The node at the position, or None if not found.
        """
        adjusted_x = (pos.x() - self._view_offset.x()) / self._zoom
        adjusted_y = (pos.y() - self._view_offset.y()) / self._zoom
        for node in self.nodes:
            nx, ny = node['pos']
            dx = adjusted_x - nx
            dy = adjusted_y - ny
            if (dx * dx + dy * dy) <= (self.node_radius * self.node_radius):
                return node
        return None

    def _deselect_all_nodes(self):
        for node in self.nodes:
            node['selected'] = False

    def _deselect_all_edges(self):
        for edge in self.edges:
            edge['selected'] = False

    def _deselect_all(self):
        self._deselect_all_nodes()
        self._deselect_all_edges()

    def _get_selected_nodes(self):
        return [node for node in self.nodes if node.get('selected', False)]

    def _get_selected_edges(self):
        return [edge for edge in self.edges if edge.get('selected', False)]

    def _get_selected_node(self):
        for node in self.nodes:
            if node.get('selected', False):
                return node
        return None

    def _get_selected_edge(self):
        for edge in self.edges:
            if edge.get('selected', False):
                return edge
        return None

    def _edge_exists(self, source_id, target_id):
        """
        Check if an edge already exists between two nodes (undirected).

        Parameters
        ----------
        source_id : int
            The source node ID.
        target_id : int
            The target node ID.

        Returns
        -------
        bool
            True if the edge exists, False otherwise.
        """
        for edge in self.edges:
            if ((edge['source'] == source_id and edge['target'] == target_id) or
                    (edge['source'] == target_id and edge['target'] == source_id)):
                return True
        return False

    def _get_edge_at(self, pos):
        """
        Return the edge at the given widget position, if any.

        Parameters
        ----------
        pos : QPointF
            The position in widget coordinates.

        Returns
        -------
        dict or None
            The edge at the position, or None if not found.
        """
        world_pos = QPointF((pos.x() - self._view_offset.x()) / self._zoom,
                            (pos.y() - self._view_offset.y()) / self._zoom)
        threshold = 5 / self._zoom
        for edge in self.edges:
            source = self.get_node_by_id(edge['source'])
            target = self.get_node_by_id(edge['target'])
            if source is None or target is None:
                continue
            a = QPointF(source['pos'][0], source['pos'][1])
            b = QPointF(target['pos'][0], target['pos'][1])
            if _distance_point_to_segment(world_pos, a, b) <= threshold:
                return edge
        return None

    def _position_hamburger(self):
        margin = 10
        self.hamburger.move(self.width() - self.hamburger.width() - margin, margin)

    def _get_crossing_number(self) -> int:
        quantum_types = {"qubit", "Z_stabilizer", "X_stabilizer"}
        qnodes = {node['id']: node for node in self.nodes if node['type'] in quantum_types}
        qedges = [edge for edge in self.edges
                  if self.get_node_by_id(edge['source']) and self.get_node_by_id(edge['target'])
                  and self.get_node_by_id(edge['source'])['type'] in quantum_types
                  and self.get_node_by_id(edge['target'])['type'] in quantum_types]
        if qnodes and qedges:
            qnode_ids = list(qnodes.keys())
            pos_list = [qnodes[node_id]['pos'] for node_id in qnode_ids]
            edge_list = []
            for edge in qedges:
                try:
                    i = qnode_ids.index(edge['source'])
                    j = qnode_ids.index(edge['target'])
                    edge_list.append((i, j))
                except ValueError:
                    continue
            crossings = find_edge_crossings(pos_list, edge_list)
            return len(crossings)
        return 0

    def _clear_canvas(self):
        self.nodes = []
        self.edges = []
        self.update()

    def _on_menu_hide(self):
        if hasattr(self, 'hamburger'):
            self.hamburger.set_open(False)
        self._hamburger_menu = None

    def _create_custom_toggle(self, label_text, initial, callback):
        """
        Create a toggle widget that doesn't auto-close the menu.

        Parameters
        ----------
        label_text : str
            Text label for the toggle.
        initial : bool
            Initial state of the toggle.
        callback : function
            Function to call when toggle state changes.

        Returns
        -------
        QWidget
            The widget containing the toggle.
        """
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(8, 4, 8, 4)

        # Create label.
        label = QLabel(label_text)
        label.setStyleSheet(
            f"font-family: 'Segoe UI', 'Helvetica Neue', sans-serif; font-size: 12px; color: {self.theme_manager.foreground.name()};"
        )

        # Create toggle with pre-set state.
        toggle = ToggleSwitch(initial, widget)

        # Connect the toggle after setting up the preventMenuClose attribute.
        toggle.toggled.connect(lambda checked: self._handle_toggle(callback, checked))

        # Add label and toggle in correct order.
        layout.addWidget(label)
        layout.addStretch()
        layout.addWidget(toggle)

        return widget

    def _handle_toggle(self, callback, checked):
        """
        Handle toggle changes without closing the menu.

        Parameters
        ----------
        callback : function
            The callback to call.
        checked : bool
            The new toggle state.
        """
        callback(checked)
        # Keep the hamburger menu open.
        if self._hamburger_menu:
            # Refocus on menu to prevent auto-close.
            self._hamburger_menu.setFocus()

    def _on_crossings_toggled(self, checked):
        self.show_crossings = checked
        self.update()

    def _on_dark_mode_toggled(self, checked):
        # Update theme immediately.
        self.theme_manager.set_dark_mode(checked)

        # Force canvas to update with new colors.
        self.update()

        # Update the existing menu in-place if it's open.
        if self._hamburger_menu:
            # Store the current position.
            menu_pos = self._hamburger_menu.pos()

            # Update menu styles without closing it.
            self._hamburger_menu.setStyleSheet(self.theme_manager.get_menu_style())

            # Update all toggle widgets in the menu.
            for action in self._hamburger_menu.actions():
                if isinstance(action, QWidgetAction):
                    widget = action.defaultWidget()
                    if widget:
                        # Update all labels in the widget.
                        for child in widget.findChildren(QLabel):
                            child.setStyleSheet(
                                f"font-family: 'Segoe UI', 'Helvetica Neue', sans-serif; "
                                f"font-size: 12px; color: {self.theme_manager.foreground.name()};"
                            )

            # Update the crossing number display if present.
            self._update_crossing_display()

            # Update the hamburger icon.
            if hasattr(self, 'hamburger'):
                self.hamburger.update()

    def _update_crossing_display(self):
        if not self._hamburger_menu:
            return

        # Look for the crossing number display
        for action in self._hamburger_menu.actions():
            if isinstance(action, QWidgetAction):
                widget = action.defaultWidget()
                if widget:
                    # Look for a label with "Crossings:" text
                    for child in widget.findChildren(QLabel):
                        if "Crossings:" in child.text():
                            # Update the label with current theme
                            child.setStyleSheet(
                                f"font-family: 'Segoe UI', 'Helvetica Neue', sans-serif; "
                                f"font-size: 12px; color: {self.theme_manager.foreground.name()};"
                            )
                            # Also update the count in case it changed
                            child.setText(f"Crossings: {self._get_crossing_number()}")

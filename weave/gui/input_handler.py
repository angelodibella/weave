"""Mouse and keyboard event handling for the canvas."""

from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

from PySide6.QtGui import QMouseEvent, QKeyEvent, QWheelEvent
from PySide6.QtCore import Qt, QPointF

from .graph_model import GraphModel, GraphData, is_valid_connection

if TYPE_CHECKING:
    from .canvas import Canvas


def distance_point_to_segment(p: QPointF, a: QPointF, b: QPointF) -> float:
    """Compute the distance from point *p* to segment *ab*."""
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


class InputHandler:
    """Processes mouse/keyboard events and mutates the canvas state accordingly."""

    def __init__(self, canvas: Canvas) -> None:
        self.canvas = canvas

    @property
    def model(self) -> GraphModel:
        return self.canvas.model

    # ------------------------------------------------------------------
    # Hit testing
    # ------------------------------------------------------------------

    def get_node_at(self, pos: QPointF) -> dict[str, Any] | None:
        adjusted_x = (pos.x() - self.canvas._view_offset.x()) / self.canvas._zoom
        adjusted_y = (pos.y() - self.canvas._view_offset.y()) / self.canvas._zoom
        for node in self.model.nodes:
            nx, ny = node["pos"]
            dx = adjusted_x - nx
            dy = adjusted_y - ny
            if dx * dx + dy * dy <= self.model.node_radius ** 2:
                return node
        return None

    def get_edge_at(self, pos: QPointF) -> dict[str, Any] | None:
        world_pos = QPointF(
            (pos.x() - self.canvas._view_offset.x()) / self.canvas._zoom,
            (pos.y() - self.canvas._view_offset.y()) / self.canvas._zoom,
        )
        threshold = 10 / self.canvas._zoom
        for edge in self.model.edges:
            source = self.model.get_node_by_id(edge["source"])
            target = self.model.get_node_by_id(edge["target"])
            if source is None or target is None:
                continue
            a = QPointF(source["pos"][0], source["pos"][1])
            b = QPointF(target["pos"][0], target["pos"][1])
            if distance_point_to_segment(world_pos, a, b) <= threshold:
                return edge
        return None

    def get_graph_at(self, pos: QPointF) -> GraphData | None:
        world_pos = QPointF(
            (pos.x() - self.canvas._view_offset.x()) / self.canvas._zoom,
            (pos.y() - self.canvas._view_offset.y()) / self.canvas._zoom,
        )
        for graph in self.model.graphs:
            nodes = [n for n in self.model.nodes if n["id"] in graph.node_ids]
            if len(nodes) <= 2:
                continue
            min_x = min(n["pos"][0] for n in nodes)
            min_y = min(n["pos"][1] for n in nodes)
            max_x = max(n["pos"][0] for n in nodes)
            max_y = max(n["pos"][1] for n in nodes)

            padding = 20
            rect_f = (min_x - padding, min_y - padding,
                      max_x - min_x + 2 * padding, max_y - min_y + 2 * padding)

            from PySide6.QtCore import QRectF
            rect = QRectF(*rect_f)
            border_width = 10 / self.canvas._zoom
            outer_rect = rect.adjusted(-border_width, -border_width, border_width, border_width)
            inner_rect = rect.adjusted(border_width, border_width, -border_width, -border_width)

            if outer_rect.contains(world_pos) and not inner_rect.contains(world_pos):
                return graph
        return None

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def widget_to_world(self, pos: QPointF) -> tuple[float, float]:
        x = (pos.x() - self.canvas._view_offset.x()) / self.canvas._zoom
        y = (pos.y() - self.canvas._view_offset.y()) / self.canvas._zoom
        return x, y

    def snap_to_grid(self, x: float, y: float) -> tuple[float, float]:
        if self.canvas.grid_mode:
            gs = self.canvas.grid_size
            x = round(x / gs) * gs
            y = round(y / gs) * gs
        return x, y

    # ------------------------------------------------------------------
    # Key press
    # ------------------------------------------------------------------

    def handle_key_press(self, event: QKeyEvent) -> bool:
        """Handle a key press. Returns True if the event was consumed."""
        key = event.key()
        ctrl = bool(event.modifiers() & Qt.ControlModifier)

        if key == Qt.Key_0 and ctrl:
            self.canvas.smooth_zoom_to(1.0)
            return True
        elif key == Qt.Key_A and ctrl:
            for node in self.model.nodes:
                node["selected"] = True
            self.canvas.update()
            return True
        elif key == Qt.Key_C and ctrl:
            self.model.copy_selected()
            return True
        elif key == Qt.Key_V and ctrl:
            cursor_pos = self.canvas.mapFromGlobal(self.canvas.cursor().pos())
            wx, wy = self.widget_to_world(QPointF(cursor_pos))
            wx, wy = self.snap_to_grid(wx, wy)
            self.model.paste_at(wx, wy)
            self.canvas.update()
            return True
        elif key == Qt.Key_O and ctrl:
            self.canvas._load_canvas()
            return True
        elif key == Qt.Key_S and ctrl:
            self.canvas._save_canvas()
            return True
        elif key == Qt.Key_Equal and ctrl:
            new_zoom = min(self.canvas._zoom * 1.2, 5.0)
            self.canvas.smooth_zoom_to(new_zoom)
            return True
        elif key == Qt.Key_Minus and ctrl:
            new_zoom = max(self.canvas._zoom / 1.2, 0.2)
            self.canvas.smooth_zoom_to(new_zoom)
            return True
        elif key == Qt.Key_G:
            self.canvas.grid_mode = not self.canvas.grid_mode
            self.canvas.update()
            return True
        elif key == Qt.Key_Escape:
            self.model.deselect_all()
            self.canvas.update()
            return True
        elif key in (Qt.Key_Delete, Qt.Key_Backspace):
            self.model.delete_selected()
            self.canvas.update()
            return True
        return False

    # ------------------------------------------------------------------
    # Mouse press
    # ------------------------------------------------------------------

    def handle_mouse_press(self, event: QMouseEvent) -> None:
        pos = event.position()
        if event.button() != Qt.LeftButton:
            return

        clicked_graph = self.get_graph_at(pos)
        if clicked_graph:
            if event.modifiers() & Qt.ShiftModifier:
                clicked_graph.selected = True
            else:
                self.model.deselect_all()
                clicked_graph.selected = True

            self.canvas.graph_drag = clicked_graph
            self.canvas.graph_drag_initial_positions = {
                n["id"]: n["pos"]
                for n in self.model.nodes
                if n["id"] in clicked_graph.node_ids
            }
            self.canvas.drag_start = pos
            self.canvas.update()
            return

        clicked_node = self.get_node_at(pos)
        if clicked_node:
            if event.modifiers() & Qt.ControlModifier:
                selected_nodes = self.model.selected_nodes()
                if selected_nodes:
                    for sn in selected_nodes:
                        if sn != clicked_node and is_valid_connection(sn, clicked_node):
                            if not self.model.edge_exists(sn["id"], clicked_node["id"]):
                                self.model.add_edge(sn["id"], clicked_node["id"])
                                self.model.update_graphs()
                    self.model.deselect_all()
                else:
                    clicked_node["selected"] = True
            elif event.modifiers() & Qt.ShiftModifier:
                self.canvas._shift_press_node = clicked_node
                self.canvas._shift_press_pos = event.position()
                self.canvas._shift_press_was_selected = clicked_node.get("selected", False)
                self.canvas._drag_start_positions = {
                    n["id"]: n["pos"] for n in self.model.selected_nodes()
                }
                self.canvas.drag_start = event.position()
            else:
                self.model.deselect_all()
                clicked_node["selected"] = True
                self.canvas._drag_start_positions = {clicked_node["id"]: clicked_node["pos"]}
                self.canvas.drag_start = event.position()
            self.canvas.update()
            return

        clicked_edge = self.get_edge_at(pos)
        if clicked_edge:
            if event.modifiers() & Qt.ShiftModifier:
                clicked_edge["selected"] = True
            else:
                self.model.deselect_all()
                clicked_edge["selected"] = True
            self.canvas.update()
            return

        # Empty space click.
        if event.modifiers() & Qt.ShiftModifier:
            self.canvas.selecting = True
            self.canvas.drag_start = None
            self.canvas._drag_start_positions = {}
            world_pos = self.widget_to_world(pos)
            self.canvas.selection_rect_start = world_pos
            self.canvas.selection_mode = "edge" if event.modifiers() & Qt.ControlModifier else "node"
        else:
            self.model.deselect_all()
            self.canvas.pan_active = True
            self.canvas.last_pan_point = pos
        self.canvas.update()

    # ------------------------------------------------------------------
    # Mouse release
    # ------------------------------------------------------------------

    def handle_mouse_release(self, event: QMouseEvent) -> None:
        if self.canvas.pan_active:
            self.canvas.pan_active = False
            self.canvas.last_pan_point = None

        if self.canvas.dragged_node is not None:
            self.canvas.dragged_node = None
            self.model.update_graphs()

        if self.canvas.graph_drag is not None:
            self.canvas.graph_drag = None
            self.canvas.graph_drag_initial_positions = {}

        if self.canvas.selecting and self.canvas.selection_rect is not None:
            x_min, y_min, x_max, y_max = self.canvas.selection_rect
            if self.canvas.selection_mode == "node":
                for node in self.model.nodes:
                    x, y = node["pos"]
                    if x_min <= x <= x_max and y_min <= y <= y_max:
                        node["selected"] = True
            elif self.canvas.selection_mode == "edge":
                for edge in self.model.edges:
                    source = self.model.get_node_by_id(edge["source"])
                    target = self.model.get_node_by_id(edge["target"])
                    if source and target:
                        x1, y1 = source["pos"]
                        x2, y2 = target["pos"]
                        if (x_min <= x1 <= x_max and y_min <= y1 <= y_max
                                and x_min <= x2 <= x_max and y_min <= y2 <= y_max):
                            edge["selected"] = True
            self.canvas.selecting = False
            self.canvas.selection_rect_start = None
            self.canvas.selection_rect = None
            self.canvas.selection_mode = None
            self.canvas.drag_start = None
            self.canvas._drag_start_positions = {}
            self.canvas.update()

        self.canvas.drag_start = None
        self.canvas._drag_start_positions = {}

        if self.canvas._shift_press_node is not None:
            if (event.position() - self.canvas._shift_press_pos).manhattanLength() < 10:
                self.canvas._shift_press_node["selected"] = not self.canvas._shift_press_was_selected
            self.canvas._shift_press_node = None
            self.canvas.update()

        self.model.update_graphs()

    # ------------------------------------------------------------------
    # Mouse move
    # ------------------------------------------------------------------

    def handle_mouse_move(self, event: QMouseEvent) -> None:
        pos = event.position()

        # Drag selected nodes.
        if self.canvas.drag_start is not None and self.canvas.graph_drag is None:
            delta = event.position() - self.canvas.drag_start
            delta_world = (delta.x() / self.canvas._zoom, delta.y() / self.canvas._zoom)
            for node in self.model.selected_nodes():
                if node["id"] not in self.canvas._drag_start_positions:
                    continue
                init_pos = self.canvas._drag_start_positions[node["id"]]
                new_x = init_pos[0] + delta_world[0]
                new_y = init_pos[1] + delta_world[1]
                new_x, new_y = self.snap_to_grid(new_x, new_y)
                node["pos"] = (new_x, new_y)
            self.model.update_graphs()
            self.canvas.update()

        # Drag a graph.
        if self.canvas.graph_drag is not None and self.canvas.drag_start is not None:
            delta = event.position() - self.canvas.drag_start
            delta_world = (delta.x() / self.canvas._zoom, delta.y() / self.canvas._zoom)
            for node in self.model.nodes:
                if node["id"] in self.canvas.graph_drag.node_ids:
                    init_pos = self.canvas.graph_drag_initial_positions[node["id"]]
                    new_x = init_pos[0] + delta_world[0]
                    new_y = init_pos[1] + delta_world[1]
                    new_x, new_y = self.snap_to_grid(new_x, new_y)
                    node["pos"] = (new_x, new_y)
            self.canvas.update()
            return

        # Lasso selection.
        if self.canvas.selecting:
            current = self.widget_to_world(pos)
            x1, y1 = self.canvas.selection_rect_start
            x2, y2 = current
            self.canvas.selection_rect = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            self.canvas.update()

        # Node drag.
        if self.canvas.dragged_node is not None:
            new_center = pos - self.canvas.drag_offset
            world_x = (new_center.x() - self.canvas._view_offset.x()) / self.canvas._zoom
            world_y = (new_center.y() - self.canvas._view_offset.y()) / self.canvas._zoom
            self.canvas.dragged_node["pos"] = (world_x, world_y)
            self.model.update_graphs()
            self.canvas.update()
        elif self.canvas.pan_active and self.canvas.last_pan_point is not None:
            delta = pos - self.canvas.last_pan_point
            self.canvas._view_offset += delta
            self.canvas.last_pan_point = pos
            self.canvas.update()

        # Cancel pending shift toggle on large move.
        if self.canvas.drag_start is not None:
            delta = event.position() - self.canvas.drag_start
            if delta.manhattanLength() > 10:
                self.canvas.shift_pending_toggle = None

    # ------------------------------------------------------------------
    # Wheel (zoom)
    # ------------------------------------------------------------------

    def handle_wheel(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        factor = 1.1 if delta > 0 else 0.9
        new_zoom = self.canvas._zoom * factor
        new_zoom = max(0.2, min(new_zoom, 5.0))
        if abs(new_zoom - self.canvas._zoom) < 0.01:
            return
        self.canvas.smooth_zoom_to(new_zoom, event.position())
        event.accept()

    # ------------------------------------------------------------------
    # Double click
    # ------------------------------------------------------------------

    def handle_double_click(self, event: QMouseEvent) -> bool:
        """Returns True if handled."""
        pos = event.position()
        graph = self.get_graph_at(pos)
        if graph:
            graph.selected = False
            for node in self.model.nodes:
                if node["id"] in graph.node_ids:
                    node["selected"] = True
            self.canvas._drag_start_positions = {
                n["id"]: n["pos"]
                for n in self.model.nodes
                if n["id"] in graph.node_ids and n["selected"]
            }
            self.canvas.drag_start = pos
            self.canvas.update()
            event.accept()
            return True
        return False

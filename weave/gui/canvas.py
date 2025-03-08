import json
import math
from typing import Any

from PySide6.QtWidgets import QWidget, QMenu
from PySide6.QtGui import QPainter, QPen, QColor, QMouseEvent, QKeyEvent, QWheelEvent
from PySide6.QtCore import Qt, QPointF, QRectF

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

    def __init__(self, parent=None):
        """
        Initialize the canvas.

        Parameters
        ----------
        parent : QWidget, optional
            The parent widget (default is None).
        """
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)

        # World model: nodes and edges.
        # Each node is a dict: {'id', 'pos', 'type', 'selected'}
        # Each edge is a dict: {'source', 'target', 'selected'}
        self.nodes = []
        self.edges = []
        self.node_radius = 10

        # View transformation parameters.
        self.view_offset = QPointF(0, 0)  # in widget coordinates
        self.zoom = 1.0

        # State for panning and dragging.
        self.pan_active = False
        self.last_pan_point = None  # QPointF
        self.dragged_node = None
        self.drag_offset = QPointF(0, 0)

        self.selecting = False  # Whether a rectangular selection is active.
        self.selection_rect_start = None  # The starting world coordinate of the selection.
        self.selection_rect = None  # Current selection rectangle (x_min, y_min, x_max, y_max).
        self.selection_mode = None  # "node" or "edge" selection mode.
        self._drag_start_positions = {}  # Dictionary to store initial positions of selected nodes when starting a drag.
        self.drag_start = None  # World coordinate of the start of a drag.

        self.show_crossings = True

    def paintEvent(self, event):
        """
        Draw the grid, nodes, and edges.

        The grid is rendered in widget coordinates as a continuous triangular lattice.
        World objects (nodes and edges) are drawn with the current view_offset and zoom.
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        # ----- Draw grid (triangular lattice) in widget coordinates -----
        spacing = 20
        dot_radius = 1
        painter.setPen(QPen(QColor("lightgray")))

        # Compute grid indices from view_offset and widget dimensions.
        vox = self.view_offset.x()
        voy = self.view_offset.y()
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
            # Create a light gray color with 50% transparency.
            selection_color = QColor(211, 211, 211, 128)
            painter.setBrush(selection_color)
            # Convert world coordinates back to widget coordinates.
            rect = QRectF(
                self.selection_rect[0] * self.zoom + self.view_offset.x(),
                self.selection_rect[1] * self.zoom + self.view_offset.y(),
                (self.selection_rect[2] - self.selection_rect[0]) * self.zoom,
                (self.selection_rect[3] - self.selection_rect[1]) * self.zoom
            )
            painter.drawRoundedRect(rect, 5, 5)
            # Reset the brush to avoid affecting other drawing.
            painter.setBrush(Qt.NoBrush)

        # ----- Draw world objects (nodes and edges) -----
        painter.save()
        painter.translate(self.view_offset)
        painter.scale(self.zoom, self.zoom)

        # Draw edges with thin pen and clipped to node perimeters.
        for edge in self.edges:
            pen = QPen(QColor("green") if edge.get('selected', False) else QColor("black"), 0.8)
            painter.setPen(pen)

            source = self.get_node_by_id(edge['source'])
            target = self.get_node_by_id(edge['target'])
            if source is None or target is None:
                continue

            src_center = QPointF(source['pos'][0], source['pos'][1])
            tgt_center = QPointF(target['pos'][0], target['pos'][1])

            # Check if the nodes are quantum: no clipping prevention is necessary.
            if source["type"] not in {"bit", "parity_check"}:
                painter.drawLine(src_center, tgt_center)
                continue

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

            new_src = QPointF(src_center.x() + dx / dist * margin_source,
                              src_center.y() + dy / dist * margin_source)
            new_tgt = QPointF(tgt_center.x() - dx / dist * margin_target,
                              tgt_center.y() - dy / dist * margin_target)
            painter.drawLine(new_src, new_tgt)

        # Draw quantum crossings if enabled.
        if self.show_crossings:
            # Filter quantum nodes and edges.
            quantum_types = {"qubit", "Z_stabilizer", "X_stabilizer"}
            qnodes = {node['id']: node for node in self.nodes if node['type'] in quantum_types}
            qedges = [edge for edge in self.edges
                      if self.get_node_by_id(edge['source']) and self.get_node_by_id(edge['target'])
                      and self.get_node_by_id(edge['source'])['type'] in quantum_types
                      and self.get_node_by_id(edge['target'])['type'] in quantum_types]
            if qnodes and qedges:
                # Build a mapping from quantum node id to a continuous index and a list of positions.
                qnode_ids = list(qnodes.keys())
                id_to_index = {node_id: i for i, node_id in enumerate(qnode_ids)}
                pos_list = [qnodes[node_id]['pos'] for node_id in qnode_ids]

                # Build edge list as tuples of indices.
                edge_list = []
                for edge in qedges:
                    i = id_to_index[edge['source']]
                    j = id_to_index[edge['target']]
                    edge_list.append((i, j))

                # For each crossing, compute approximate intersection and draw a diamond.
                crossings = find_edge_crossings(pos_list, edge_list)
                for crossing in crossings:
                    # Each crossing is a frozenset of two edges, extract them.
                    edge_pair = list(crossing)
                    e1, e2 = edge_pair[0], edge_pair[1]

                    # Get the endpoints (in world coordinates) for each edge.
                    def get_endpoints(e):
                        n1 = qnodes[qnode_ids[e[0]]]['pos']
                        n2 = qnodes[qnode_ids[e[1]]]['pos']
                        return n1, n2

                    a, b = get_endpoints(e1)
                    c, d = get_endpoints(e2)
                    ip = line_intersection(a, b, c, d)
                    if ip is not None:
                        size = 4  # size of the square in world units
                        painter.save()
                        painter.translate(ip[0], ip[1])
                        painter.rotate(45)
                        painter.setBrush(QColor("black"))
                        painter.setPen(Qt.NoPen)
                        painter.drawRect(QRectF(-size / 2, -size / 2, size, size))
                        painter.restore()

        # Draw nodes.
        for node in self.nodes:
            x = node['pos'][0]
            y = node['pos'][1]
            r = self.node_radius
            l = 1.86 * r
            node_type = node['type']
            if node_type in {"bit", "parity_check"}:
                painter.setPen(QPen(QColor("green"), 1) if node.get('selected', False) else QColor("black"))
                if node_type == "bit":
                    painter.drawEllipse(QPointF(x, y), r, r)
                else:
                    painter.drawRect(QRectF(x - l / 2, y - l / 2, l, l))
            else:
                painter.setPen(QPen(QColor("green"), 1) if node.get('selected', False) else Qt.NoPen)
                if node_type == "qubit":
                    painter.setBrush(QColor("#D3D3D3"))
                    painter.drawEllipse(QPointF(x, y), r, r)
                elif node_type == "Z_stabilizer":
                    painter.setBrush(QColor("#ADD8E6"))
                    painter.drawRect(QRectF(x - l / 2, y - l / 2, l, l))
                elif node_type == "X_stabilizer":
                    painter.setBrush(QColor("#FFC0CB"))
                    painter.drawRect(QRectF(x - l / 2, y - l / 2, l, l))
                painter.setBrush(QColor("transparent"))

        painter.restore()

    def wheelEvent(self, event: QWheelEvent):
        """
        Zoom the view relative to the cursor position.

        The view_offset is adjusted so that the world point under the cursor remains fixed.
        """
        old_zoom = self.zoom
        delta = event.angleDelta().y()
        factor = 1.05 if delta > 0 else 0.95
        new_zoom = old_zoom * factor
        new_zoom = max(0.2, min(new_zoom, 5.0))
        cursor_pos = event.position()  # QPointF
        self.view_offset = cursor_pos - (new_zoom / old_zoom) * (cursor_pos - self.view_offset)
        self.zoom = new_zoom
        self.update()

    def keyPressEvent(self, event: QKeyEvent):
        """
        Handle key events.

        Ctrl+0 resets zoom to default.
        Escape deselects all objects.
        Delete/Backspace removes selected nodes (and their edges) or selected edges.
        """
        if event.key() == Qt.Key_0 and event.modifiers() & Qt.ControlModifier:
            self.zoom = 1.0
            self.update()
            return
        if event.key() == Qt.Key_Escape:
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

    def contextMenuEvent(self, event):
        """
        Display a context menu for creating nodes or saving the code.
        """
        menu = QMenu(self)

        # Classical node options.
        menu.addAction("New Bit", lambda: self.add_node_at(event.pos(), "bit"))
        menu.addAction("New Parity Check", lambda: self.add_node_at(event.pos(), "parity_check"))

        # Quantum node options.
        quantum_menu = menu.addMenu("New Quantum Node")
        quantum_menu.addAction("New Qubit", lambda: self.add_node_at(event.pos(), "qubit"))
        quantum_menu.addAction("New Z-Stabilizer", lambda: self.add_node_at(event.pos(), "Z_stabilizer"))
        quantum_menu.addAction("New X-Stabilizer", lambda: self.add_node_at(event.pos(), "X_stabilizer"))

        # Toggle crossings.
        menu.addAction("Toggle Crossings", lambda: self._toggle_crossings())

        # Save option.
        menu.addAction("Save Code as CSV", lambda: print("Save functionality not implemented yet."))
        menu.exec(event.globalPos())
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        """
        Handle mouse press events.

        Left-click selects nodes or edges and begins dragging or panning.
        Ctrl+left-click creates an edge between nodes.
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
                        self._deselect_all()  # Optionally, clear selection after edge creation.
                    else:
                        clicked_node['selected'] = True
                    self.update()
                elif event.modifiers() & Qt.ShiftModifier:
                    # If the node is not already selected, add it to the selection.
                    if not clicked_node.get('selected', False):
                        clicked_node['selected'] = True
                    # Otherwise, leave it selected so that dragging moves all selected nodes.
                    # Record starting positions for all selected nodes.
                    self._drag_start_positions = {n['id']: n['pos'] for n in self._get_selected_nodes()}
                    self.drag_start = event.position()
                else:
                    self._deselect_all()
                    clicked_node['selected'] = True
                    self._drag_start_positions = {clicked_node['id']: clicked_node['pos']}
                    self.drag_start = event.position()
            else:
                clicked_edge = self._get_edge_at(pos)
                if clicked_edge:
                    if event.modifiers() & Qt.ShiftModifier:
                        clicked_edge['selected'] = True
                    else:
                        self._deselect_all()
                        clicked_edge['selected'] = True
                    self.update()
                else:
                    # If shift (or ctrl+shift) is held, start a rectangular selection.
                    if event.modifiers() & Qt.ShiftModifier:
                        self.selecting = True
                        # Clear any previous drag state to prevent nodes from moving.
                        self.drag_start = None
                        self._drag_start_positions = {}
                        world_pos = ((pos.x() - self.view_offset.x()) / self.zoom,
                                     (pos.y() - self.view_offset.y()) / self.zoom)
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
        """
        Handle mouse move events.

        Updates node positions when dragging or adjusts view_offset when panning.
        """
        if self.drag_start is not None:
            delta = event.position() - self.drag_start
            delta_world = (delta.x() / self.zoom, delta.y() / self.zoom)
            for node in self._get_selected_nodes():
                init_pos = self._drag_start_positions[node['id']]
                node['pos'] = (init_pos[0] + delta_world[0], init_pos[1] + delta_world[1])
            self.update()

        pos = event.position()
        if self.selecting:
            current = ((pos.x() - self.view_offset.x()) / self.zoom,
                       (pos.y() - self.view_offset.y()) / self.zoom)
            x1, y1 = self.selection_rect_start
            x2, y2 = current
            self.selection_rect = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            self.update()

        if self.dragged_node is not None:
            new_center = pos - self.drag_offset
            world_x = (new_center.x() - self.view_offset.x()) / self.zoom
            world_y = (new_center.y() - self.view_offset.y()) / self.zoom
            self.dragged_node['pos'] = (world_x, world_y)
            self.update()
        elif self.pan_active and self.last_pan_point is not None:
            delta = pos - self.last_pan_point
            self.view_offset += delta
            self.last_pan_point = pos
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """
        Handle mouse release events, ending panning or dragging.
        """
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

        super().mouseReleaseEvent(event)

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
        adjusted_x = (pos.x() - self.view_offset.x()) / self.zoom
        adjusted_y = (pos.y() - self.view_offset.y()) / self.zoom
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
            epsilon = 0.25
            return r / max(cos_theta, sin_theta) + epsilon

    def _get_node_at(self, pos):
        """
        Return the node at the given widget position, if any.

        The position is converted to world coordinates.
        """
        adjusted_x = (pos.x() - self.view_offset.x()) / self.zoom
        adjusted_y = (pos.y() - self.view_offset.y()) / self.zoom
        for node in self.nodes:
            nx, ny = node['pos']
            dx = adjusted_x - nx
            dy = adjusted_y - ny
            if (dx * dx + dy * dy) <= (self.node_radius * self.node_radius):
                return node
        return None

    def _deselect_all_nodes(self):
        """Deselect all nodes."""
        for node in self.nodes:
            node['selected'] = False

    def _deselect_all_edges(self):
        """Deselect all edges."""
        for edge in self.edges:
            edge['selected'] = False

    def _deselect_all(self):
        """Deselect all nodes and edges."""
        self._deselect_all_nodes()
        self._deselect_all_edges()

    def _get_selected_nodes(self):
        """Return a list of all currently selected nodes."""
        return [node for node in self.nodes if node.get('selected', False)]

    def _get_selected_edges(self):
        """Return a list of all currently selected edges."""
        return [edge for edge in self.edges if edge.get('selected', False)]

    def _get_selected_node(self):
        """
        Return the selected node, if any.
        """
        for node in self.nodes:
            if node.get('selected', False):
                return node
        return None

    def _get_selected_edge(self):
        """
        Return the selected edge, if any.
        """
        for edge in self.edges:
            if edge.get('selected', False):
                return edge
        return None

    def _edge_exists(self, source_id, target_id):
        """
        Check if an edge already exists between two nodes (undirected).
        """
        for edge in self.edges:
            if ((edge['source'] == source_id and edge['target'] == target_id) or
                    (edge['source'] == target_id and edge['target'] == source_id)):
                return True
        return False

    def _get_edge_at(self, pos):
        """
        Return the edge at the given widget position, if any.

        The position is converted to world coordinates.
        """
        world_pos = QPointF((pos.x() - self.view_offset.x()) / self.zoom,
                            (pos.y() - self.view_offset.y()) / self.zoom)
        threshold = 5 / self.zoom
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

    def _toggle_crossings(self):
        self.show_crossings = not self.show_crossings

"""Rendering logic for the canvas — nodes, edges, crossings, grid, graph borders."""

from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

from PySide6.QtGui import QPainter, QPen, QColor, QPainterPath
from PySide6.QtCore import Qt, QPointF, QRectF

from ..util.graph import find_edge_crossings, line_intersection
from .graph_model import GraphModel, QUANTUM_TYPES

if TYPE_CHECKING:
    from .theme import ThemeManager


def draw_grid(
    painter: QPainter,
    theme: ThemeManager,
    view_offset: QPointF,
    zoom: float,
    width: int,
    height: int,
) -> None:
    """Draw the background dot grid."""
    base_spacing = 10
    min_dot_spacing = 15

    dot_spacing = base_spacing
    while dot_spacing * zoom < min_dot_spacing:
        dot_spacing *= 2

    dot_radius = 1
    painter.setPen(QPen(theme.grid, 0.5))

    vox = view_offset.x()
    voy = view_offset.y()

    min_n = math.floor(-vox / (dot_spacing * zoom))
    max_n = math.ceil((width - vox) / (dot_spacing * zoom))
    min_m = math.floor(-voy / (dot_spacing * zoom))
    max_m = math.ceil((height - voy) / (dot_spacing * zoom))

    max_dots = 2000
    total_dots = (max_n - min_n + 1) * (max_m - min_m + 1)
    skip = max(1, int(math.sqrt(total_dots / max_dots))) if total_dots > max_dots else 1

    for m in range(min_m, max_m + 1, skip):
        y = m * (dot_spacing * zoom) + voy
        row_offset = (dot_spacing * zoom) / 2 if (m % 2 != 0) else 0
        for n in range(min_n, max_n + 1, skip):
            x = n * (dot_spacing * zoom) + vox + row_offset
            painter.drawEllipse(QPointF(x, y), dot_radius, dot_radius)


def draw_snap_grid(
    painter: QPainter,
    theme: ThemeManager,
    view_offset: QPointF,
    zoom: float,
    grid_size: float,
    width: int,
    height: int,
) -> None:
    """Draw the snap grid overlay."""
    min_grid_spacing = 20
    screen_grid_size = grid_size * zoom

    grid_scale = 1
    while screen_grid_size < min_grid_spacing:
        grid_scale *= 2
        screen_grid_size = grid_size * grid_scale * zoom

    world_grid_size = grid_size * grid_scale

    min_world_x = (0 - view_offset.x()) / zoom
    max_world_x = (width - view_offset.x()) / zoom
    min_world_y = (0 - view_offset.y()) / zoom
    max_world_y = (height - view_offset.y()) / zoom

    world_start_x = math.floor(min_world_x / world_grid_size) * world_grid_size
    world_end_x = math.ceil(max_world_x / world_grid_size) * world_grid_size
    world_start_y = math.floor(min_world_y / world_grid_size) * world_grid_size
    world_end_y = math.ceil(max_world_y / world_grid_size) * world_grid_size

    x_count = int((world_end_x - world_start_x) / world_grid_size) + 1
    y_count = int((world_end_y - world_start_y) / world_grid_size) + 1

    max_lines = 100
    skip = max(1, int(math.sqrt((x_count + y_count) / max_lines * 2))) if x_count + y_count > max_lines else 1

    painter.save()
    painter.translate(view_offset)
    painter.scale(zoom, zoom)

    pen = QPen(theme.selected, 1 / zoom)
    painter.setPen(pen)

    for i in range(0, y_count, skip):
        y = world_start_y + i * world_grid_size
        painter.drawLine(world_start_x, y, world_end_x, y)

    for i in range(0, x_count, skip):
        x = world_start_x + i * world_grid_size
        painter.drawLine(x, world_start_y, x, world_end_y)

    painter.restore()


def draw_selection_rect(
    painter: QPainter,
    theme: ThemeManager,
    selection_rect: tuple[float, float, float, float],
    view_offset: QPointF,
    zoom: float,
) -> None:
    """Draw the lasso/rectangular selection area."""
    painter.setPen(Qt.NoPen)
    selection_color = QColor(theme.selected)
    painter.setBrush(selection_color)

    rect = QRectF(
        selection_rect[0] * zoom + view_offset.x(),
        selection_rect[1] * zoom + view_offset.y(),
        (selection_rect[2] - selection_rect[0]) * zoom,
        (selection_rect[3] - selection_rect[1]) * zoom,
    )
    painter.drawRoundedRect(rect, 5, 5)
    painter.setBrush(Qt.NoBrush)


def _get_margin(node: dict[str, Any], dx: float, dy: float, node_radius: int) -> float:
    """Compute the margin for edge clipping from a node's center."""
    r = node_radius
    if node["type"] == "bit":
        return r
    if dx == 0 and dy == 0:
        return r
    dist = math.hypot(dx, dy)
    cos_theta = abs(dx) / dist
    sin_theta = abs(dy) / dist
    epsilon = 0.5
    return r / max(cos_theta, sin_theta) - epsilon


def draw_edges(painter: QPainter, model: GraphModel, theme: ThemeManager) -> None:
    """Draw all edges."""
    for edge in model.edges:
        source = model.get_node_by_id(edge["source"])
        target = model.get_node_by_id(edge["target"])
        if source is None or target is None:
            continue

        src_center = QPointF(source["pos"][0], source["pos"][1])
        tgt_center = QPointF(target["pos"][0], target["pos"][1])

        if source["type"] in {"bit", "parity_check"} and target["type"] in {"bit", "parity_check"}:
            dx = tgt_center.x() - src_center.x()
            dy = tgt_center.y() - src_center.y()
            dist = math.hypot(dx, dy)
            if dist == 0:
                continue

            margin_source = _get_margin(source, dx, dy, model.node_radius)
            margin_target = _get_margin(target, -dx, -dy, model.node_radius)
            if margin_source + margin_target > dist:
                continue

            src = QPointF(
                src_center.x() + dx / dist * margin_source,
                src_center.y() + dy / dist * margin_source,
            )
            tgt = QPointF(
                tgt_center.x() - dx / dist * margin_target,
                tgt_center.y() - dy / dist * margin_target,
            )
        else:
            src, tgt = src_center, tgt_center

        pen = QPen(theme.foreground, 0.8)
        pen.setCapStyle(Qt.FlatCap)
        painter.setPen(pen)
        painter.drawLine(src, tgt)

        if edge.get("selected", False):
            highlight_pen = QPen(theme.selected, 5)
            highlight_pen.setCapStyle(Qt.FlatCap)
            painter.setPen(highlight_pen)
            painter.drawLine(src, tgt)


def draw_crossings(painter: QPainter, model: GraphModel, theme: ThemeManager) -> None:
    """Draw crossing diamonds at edge intersection points."""
    qnodes = {n["id"]: n for n in model.nodes if n["type"] in QUANTUM_TYPES}
    qedges = [
        e for e in model.edges
        if model.get_node_by_id(e["source"]) and model.get_node_by_id(e["target"])
        and model.get_node_by_id(e["source"])["type"] in QUANTUM_TYPES
        and model.get_node_by_id(e["target"])["type"] in QUANTUM_TYPES
    ]

    if not qnodes or not qedges:
        return

    qnode_ids = list(qnodes.keys())
    id_to_index = {nid: i for i, nid in enumerate(qnode_ids)}
    pos_list = [qnodes[nid]["pos"] for nid in qnode_ids]

    edge_list = []
    for edge in qedges:
        try:
            i = id_to_index[edge["source"]]
            j = id_to_index[edge["target"]]
            edge_list.append((i, j))
        except KeyError:
            continue

    crossings = find_edge_crossings(pos_list, edge_list)
    for crossing in crossings:
        edge_pair = list(crossing)
        e1, e2 = edge_pair[0], edge_pair[1]

        def get_endpoints(e: tuple[int, int]) -> tuple[Any, Any]:
            try:
                n1 = qnodes[qnode_ids[e[0]]]["pos"]
                n2 = qnodes[qnode_ids[e[1]]]["pos"]
                return n1, n2
            except (KeyError, IndexError):
                return None, None

        a, b = get_endpoints(e1)
        c, d = get_endpoints(e2)

        if None in (a, b, c, d):
            continue

        ip = line_intersection(a, b, c, d)
        if ip is not None:
            size = 4
            painter.save()
            painter.translate(ip[0], ip[1])
            painter.rotate(45)
            painter.setBrush(theme.crossing)
            painter.setPen(Qt.NoPen)
            painter.drawRect(QRectF(-size / 2, -size / 2, size, size))
            painter.restore()


def draw_nodes(painter: QPainter, model: GraphModel, theme: ThemeManager) -> None:
    """Draw all nodes."""
    for node in model.nodes:
        x = node["pos"][0]
        y = node["pos"][1]
        r = model.node_radius
        l = 1.86 * r
        node_type = node["type"]
        is_selected = node.get("selected", False)

        node_color = theme.get_node_color(node_type)
        pen = QPen(theme.foreground, 1)

        if is_selected:
            highlight_size = 2
            highlight_color = theme.selected
            painter.setPen(Qt.NoPen)
            painter.setBrush(highlight_color)

            if node_type in ("bit", "qubit"):
                painter.drawEllipse(QPointF(x, y), r + highlight_size, r + highlight_size)
            else:
                painter.drawRoundedRect(
                    QRectF(x - l / 2 - highlight_size, y - l / 2 - highlight_size,
                           l + 2 * highlight_size, l + 2 * highlight_size),
                    1, 1,
                )

        if node_type in {"bit", "parity_check"}:
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            if node_type == "bit":
                painter.drawEllipse(QPointF(x, y), r, r)
            else:
                painter.drawRect(QRectF(x - l / 2, y - l / 2, l, l))
        else:
            painter.setPen(Qt.NoPen)
            painter.setBrush(node_color)
            if node_type == "qubit":
                painter.drawEllipse(QPointF(x, y), r, r)
            else:
                painter.drawRect(QRectF(x - l / 2, y - l / 2, l, l))

        painter.setBrush(Qt.NoBrush)


def draw_graph_borders(painter: QPainter, model: GraphModel, theme: ThemeManager) -> None:
    """Draw borders around detected graphs."""
    for graph in model.graphs:
        nodes = [n for n in model.nodes if n["id"] in graph.node_ids]
        if len(nodes) <= 2:
            continue

        border_color = theme.graph_quantum if graph.graph_type == "quantum" else theme.graph_classical

        min_x = min(n["pos"][0] for n in nodes)
        min_y = min(n["pos"][1] for n in nodes)
        max_x = max(n["pos"][0] for n in nodes)
        max_y = max(n["pos"][1] for n in nodes)

        padding = 20
        rect = QRectF(min_x - padding, min_y - padding,
                       max_x - min_x + 2 * padding, max_y - min_y + 2 * padding)

        painter.setPen(QPen(border_color, 1.5))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(rect, 3, 3)

        if graph.selected:
            highlight_pen = QPen(theme.selected, 5)
            highlight_pen.setCapStyle(Qt.FlatCap)
            painter.setPen(highlight_pen)
            painter.drawRoundedRect(rect, 3, 3)


def get_crossing_number(model: GraphModel) -> int:
    """Count the number of edge crossings among quantum edges."""
    qnodes = {n["id"]: n for n in model.nodes if n["type"] in QUANTUM_TYPES}
    qedges = [
        e for e in model.edges
        if model.get_node_by_id(e["source"]) and model.get_node_by_id(e["target"])
        and model.get_node_by_id(e["source"])["type"] in QUANTUM_TYPES
        and model.get_node_by_id(e["target"])["type"] in QUANTUM_TYPES
    ]
    if qnodes and qedges:
        qnode_ids = list(qnodes.keys())
        pos_list = [qnodes[nid]["pos"] for nid in qnode_ids]
        edge_list = []
        for edge in qedges:
            try:
                i = qnode_ids.index(edge["source"])
                j = qnode_ids.index(edge["target"])
                edge_list.append((i, j))
            except ValueError:
                continue
        return len(find_edge_crossings(pos_list, edge_list))
    return 0

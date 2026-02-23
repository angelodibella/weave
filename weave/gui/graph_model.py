"""Data model for the canvas graph editor."""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from PySide6.QtCore import QObject, Signal


QUANTUM_TYPES = frozenset({"qubit", "Z_stabilizer", "X_stabilizer"})
CLASSICAL_TYPES = frozenset({"bit", "parity_check"})


def is_valid_connection(source: dict[str, Any], target: dict[str, Any]) -> bool:
    """Check if a connection between two nodes is valid."""
    if source["type"] in QUANTUM_TYPES and target["type"] in QUANTUM_TYPES:
        return (
            (source["type"] == "qubit" and target["type"] in {"Z_stabilizer", "X_stabilizer"})
            or (target["type"] == "qubit" and source["type"] in {"Z_stabilizer", "X_stabilizer"})
        )
    elif source["type"] in CLASSICAL_TYPES and target["type"] in CLASSICAL_TYPES:
        return source["type"] != target["type"]
    return False


@dataclass
class GraphData:
    """Data for a detected graph (connected component)."""
    node_ids: set[int]
    graph_type: str  # 'quantum' or 'classical'
    selected: bool = False
    name: str = ""
    css_code: Any = None  # CSSCode | None — avoid circular import
    noise_config: dict[str, Any] | None = None
    logical_indices: list[int] | None = None


class GraphModel(QObject):
    """Central data model for the canvas.

    Owns all nodes, edges, and detected graphs. Emits signals
    when data changes so the canvas and other modules can react.
    """

    model_changed = Signal()
    selection_changed = Signal()
    graph_detected = Signal(object)   # GraphData
    graph_removed = Signal(object)    # GraphData

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.nodes: list[dict[str, Any]] = []
        self.edges: list[dict[str, Any]] = []
        self.graphs: list[GraphData] = []
        self.node_radius: int = 10

        # Clipboard state.
        self.clipboard_nodes: list[dict[str, Any]] = []
        self.clipboard_edges: list[dict[str, Any]] = []
        self.clipboard_center: tuple[float, float] | None = None

    # ------------------------------------------------------------------
    # Node helpers
    # ------------------------------------------------------------------

    def get_node_by_id(self, node_id: int) -> dict[str, Any] | None:
        for node in self.nodes:
            if node["id"] == node_id:
                return node
        return None

    def next_node_id(self) -> int:
        return max((n["id"] for n in self.nodes), default=-1) + 1

    def add_node(self, x: float, y: float, node_type: str) -> dict[str, Any]:
        node = {
            "id": self.next_node_id(),
            "pos": (x, y),
            "type": node_type,
            "selected": False,
        }
        self.nodes.append(node)
        self.model_changed.emit()
        return node

    def selected_nodes(self) -> list[dict[str, Any]]:
        return [n for n in self.nodes if n.get("selected", False)]

    def selected_edges(self) -> list[dict[str, Any]]:
        return [e for e in self.edges if e.get("selected", False)]

    # ------------------------------------------------------------------
    # Edge helpers
    # ------------------------------------------------------------------

    def edge_exists(self, source_id: int, target_id: int) -> bool:
        for edge in self.edges:
            if (
                (edge["source"] == source_id and edge["target"] == target_id)
                or (edge["source"] == target_id and edge["target"] == source_id)
            ):
                return True
        return False

    def add_edge(self, source_id: int, target_id: int) -> dict[str, Any]:
        edge = {"source": source_id, "target": target_id, "selected": False}
        self.edges.append(edge)
        self.model_changed.emit()
        return edge

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def deselect_all(self) -> None:
        for node in self.nodes:
            node["selected"] = False
        for edge in self.edges:
            edge["selected"] = False
        for graph in self.graphs:
            graph.selected = False
        self.selection_changed.emit()

    # ------------------------------------------------------------------
    # Graph detection
    # ------------------------------------------------------------------

    def detect_connected_component(self, node_id: int) -> tuple[list[dict], list[dict]]:
        """BFS to find the connected component containing *node_id*."""
        all_nodes = {n["id"]: n for n in self.nodes}
        if not all_nodes:
            return [], []

        visited: set[int] = set()
        queue: deque[int] = deque([node_id])
        visited.add(node_id)

        while queue:
            current = queue.popleft()
            for edge in self.edges:
                if edge["source"] == current and edge["target"] not in visited:
                    queue.append(edge["target"])
                    visited.add(edge["target"])
                elif edge["target"] == current and edge["source"] not in visited:
                    queue.append(edge["source"])
                    visited.add(edge["source"])

        component_nodes = [all_nodes[nid] for nid in visited if nid in all_nodes]
        component_edges = [
            e for e in self.edges if e["source"] in visited and e["target"] in visited
        ]
        return component_nodes, component_edges

    @staticmethod
    def determine_graph_type(nodes: list[dict]) -> str:
        for node in nodes:
            if node["type"] in QUANTUM_TYPES:
                return "quantum"
        return "classical"

    def detect_graph(self, node_id: int) -> GraphData | None:
        """Detect and register the graph containing *node_id*."""
        # Already in a graph?
        for g in self.graphs:
            if node_id in g.node_ids:
                return None

        nodes, _ = self.detect_connected_component(node_id)
        if len(nodes) <= 2:
            return None

        gd = GraphData(
            node_ids={n["id"] for n in nodes},
            graph_type=self.determine_graph_type(nodes),
        )
        self.graphs.append(gd)
        self.graph_detected.emit(gd)
        self.model_changed.emit()
        return gd

    def update_graphs(self) -> None:
        """Refresh graph membership after topology changes."""
        to_remove: list[GraphData] = []

        for graph in self.graphs:
            if not graph.node_ids:
                to_remove.append(graph)
                continue

            existing_ids = {n["id"] for n in self.nodes}
            any_id = next(iter(graph.node_ids & existing_ids), None)
            if any_id is None:
                to_remove.append(graph)
                continue

            nodes, _ = self.detect_connected_component(any_id)
            if len(nodes) <= 2:
                to_remove.append(graph)
                continue

            graph.node_ids = {n["id"] for n in nodes}
            graph.graph_type = self.determine_graph_type(nodes)

        for graph in to_remove:
            self.graphs.remove(graph)
            self.graph_removed.emit(graph)

    # ------------------------------------------------------------------
    # Clipboard
    # ------------------------------------------------------------------

    def copy_selected(self) -> bool:
        selected = self.selected_nodes()
        if not selected:
            return False

        selected_ids = {n["id"] for n in selected}
        self.clipboard_nodes = [n.copy() for n in selected]
        self.clipboard_edges = [
            e.copy() for e in self.edges
            if e["source"] in selected_ids and e["target"] in selected_ids
        ]
        if self.clipboard_nodes:
            avg_x = sum(n["pos"][0] for n in self.clipboard_nodes) / len(self.clipboard_nodes)
            avg_y = sum(n["pos"][1] for n in self.clipboard_nodes) / len(self.clipboard_nodes)
            self.clipboard_center = (avg_x, avg_y)
        return True

    def can_paste(self) -> bool:
        return len(self.clipboard_nodes) > 0

    def paste_at(self, world_x: float, world_y: float) -> bool:
        if not self.clipboard_nodes or self.clipboard_center is None:
            return False

        offset_x = world_x - self.clipboard_center[0]
        offset_y = world_y - self.clipboard_center[1]

        next_id = self.next_node_id()
        id_mapping: dict[int, int] = {}

        self.deselect_all()

        for old_node in self.clipboard_nodes:
            new_node = old_node.copy()
            old_id = new_node["id"]
            new_node["id"] = next_id
            id_mapping[old_id] = next_id
            next_id += 1
            old_x, old_y = new_node["pos"]
            new_node["pos"] = (old_x + offset_x, old_y + offset_y)
            new_node["selected"] = True
            self.nodes.append(new_node)

        for old_edge in self.clipboard_edges:
            new_edge = old_edge.copy()
            new_edge["source"] = id_mapping[old_edge["source"]]
            new_edge["target"] = id_mapping[old_edge["target"]]
            new_edge["selected"] = False
            self.edges.append(new_edge)

        self.model_changed.emit()
        return True

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------

    def clear(self) -> None:
        self.nodes.clear()
        self.edges.clear()
        self.graphs.clear()
        self.model_changed.emit()

    # ------------------------------------------------------------------
    # Delete selected
    # ------------------------------------------------------------------

    def delete_selected(self) -> bool:
        """Delete selected nodes, edges, or graphs. Returns True if anything was deleted."""
        selected_nodes = self.selected_nodes()
        if selected_nodes:
            ids_to_remove = {n["id"] for n in selected_nodes}
            self.nodes = [n for n in self.nodes if n["id"] not in ids_to_remove]
            self.edges = [
                e for e in self.edges
                if e["source"] not in ids_to_remove and e["target"] not in ids_to_remove
            ]
            self.deselect_all()
            self.update_graphs()
            self.model_changed.emit()
            return True

        selected_edges = self.selected_edges()
        if selected_edges:
            for edge in selected_edges:
                self.edges.remove(edge)
            self.deselect_all()
            self.update_graphs()
            self.model_changed.emit()
            return True

        selected_graphs = [g for g in self.graphs if g.selected]
        if selected_graphs:
            for g in selected_graphs:
                self.graphs.remove(g)
            self.model_changed.emit()
            return True

        return False

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save_to_file(self, filename: str, view_offset: tuple[float, float] = (0, 0), zoom: float = 1.0) -> None:
        serializable_graphs = [
            {"node_ids": list(g.node_ids), "type": g.graph_type}
            for g in self.graphs
        ]
        data = {
            "nodes": [{k: v for k, v in node.items() if k != "selected"} for node in self.nodes],
            "edges": [{k: v for k, v in edge.items() if k != "selected"} for edge in self.edges],
            "graphs": serializable_graphs,
            "view_offset": list(view_offset),
            "zoom": zoom,
        }
        with open(filename, "w") as f:
            json.dump(data, f)

    def load_from_file(self, filename: str) -> tuple[tuple[float, float], float]:
        """Load model from file. Returns (view_offset, zoom)."""
        with open(filename, "r") as f:
            data = json.load(f)

        self.nodes = [dict(node, selected=False) for node in data.get("nodes", [])]
        self.edges = [dict(edge, selected=False) for edge in data.get("edges", [])]

        self.graphs = []
        for gd in data.get("graphs", []):
            self.graphs.append(
                GraphData(node_ids=set(gd["node_ids"]), graph_type=gd["type"])
            )

        view_offset = tuple(data.get("view_offset", [0, 0]))
        zoom = data.get("zoom", 1.0)
        self.model_changed.emit()
        return view_offset, zoom  # type: ignore[return-value]

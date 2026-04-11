"""JSON-polyline embedding: arbitrary precomputed routed polylines.

Loads a map from Tanner-graph edges to 3D polylines from a JSON document
or file. The schema is stable across weave versions via
`SCHEMA_VERSION`. Useful for importing hand-authored layouts, frozen
reference embeddings from other tools (HAL, bbstim), and regression
fixtures shipped in `benchmarks/fixtures/`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from ...geometry import Point3
from ..embedding import IREdge, IRPolyline, RoutingGeometry


def _to_point3(p: Sequence[float]) -> Point3:
    if len(p) == 2:
        return (float(p[0]), float(p[1]), 0.0)
    if len(p) == 3:
        return (float(p[0]), float(p[1]), float(p[2]))
    raise ValueError(f"Expected a 2D or 3D point, got length {len(p)}.")


def _to_polyline(points: Sequence[Sequence[float]]) -> IRPolyline:
    if len(points) < 2:
        raise ValueError(f"Polyline must have at least 2 points, got {len(points)}.")
    return tuple(_to_point3(p) for p in points)


@dataclass(frozen=True)
class JsonPolylineEmbedding:
    """Embedding backed by a precomputed map from edges to polylines.

    Unlike :class:`StraightLineEmbedding`, every edge can have its own
    multi-segment polyline — useful for biplanar routes, surface
    geodesics exported as polyline approximations, or any layout that
    weave does not yet know how to generate natively.

    Parameters
    ----------
    positions : dict[int, Point3]
        Per-qubit 3D positions.
    edge_polylines : dict[tuple[int, int], tuple[Point3, ...]]
        Routed polylines keyed by `(source, target)`. Each polyline
        must contain at least 2 points.
    name : str, optional
        Label for provenance and debugging; defaults to
        ``"json_polyline"``.

    Notes
    -----
    Both internal storage fields are dicts, so this embedding is
    declared frozen but is **not** hashable at runtime. Frozen here
    protects field reassignment only; the convention is that the dicts
    are not mutated after construction.
    """

    SCHEMA_VERSION: ClassVar[int] = 1

    positions: dict[int, Point3]
    edge_polylines: dict[IREdge, IRPolyline]
    name: str = "json_polyline"

    def __post_init__(self) -> None:
        for edge, poly in self.edge_polylines.items():
            if len(poly) < 2:
                raise ValueError(
                    f"Polyline for edge {edge} must have at least 2 points, got {len(poly)}."
                )

    # ------------------------------------------------------------------
    # Embedding protocol
    # ------------------------------------------------------------------

    @property
    def schema_version(self) -> int:
        return self.SCHEMA_VERSION

    @property
    def surface_name(self) -> str:
        return "json_polyline"

    def node_position(self, qubit_idx: int) -> Point3:
        if qubit_idx not in self.positions:
            raise IndexError(f"No position defined for qubit {qubit_idx}.")
        return self.positions[qubit_idx]

    def routing_geometry(self, active_edges: Sequence[IREdge]) -> RoutingGeometry:
        edges_map: dict[IREdge, IRPolyline] = {}
        for edge in active_edges:
            if edge not in self.edge_polylines:
                raise KeyError(
                    f"No polyline defined for edge {edge}; "
                    f"known edges: {sorted(self.edge_polylines.keys())[:5]}..."
                )
            edges_map[edge] = self.edge_polylines[edge]
        return RoutingGeometry(edges=edges_map, name=self.name)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "type": "json_polyline",
            "name": self.name,
            "positions": {str(k): list(v) for k, v in sorted(self.positions.items())},
            "edges": [
                {
                    "source": int(u),
                    "target": int(v),
                    "polyline": [list(p) for p in poly],
                }
                for (u, v), poly in sorted(self.edge_polylines.items())
            ],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> JsonPolylineEmbedding:
        if data.get("type") != "json_polyline":
            raise ValueError(f"Expected type='json_polyline', got {data.get('type')!r}.")
        version = data.get("schema_version")
        if version != cls.SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version {version}; expected {cls.SCHEMA_VERSION}."
            )
        positions = {int(k): _to_point3(v) for k, v in data["positions"].items()}
        edge_polylines = {
            (int(e["source"]), int(e["target"])): _to_polyline(e["polyline"]) for e in data["edges"]
        }
        return cls(
            positions=positions,
            edge_polylines=edge_polylines,
            name=data.get("name", "json_polyline"),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> JsonPolylineEmbedding:
        """Load from a JSON file on disk."""
        with Path(path).open() as f:
            data = json.load(f)
        return cls.from_json(data)

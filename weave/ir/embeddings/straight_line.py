"""Straight-line embedding: every edge is a 2-point polyline at its endpoints.

This is the trivial embedding used by default throughout weave. It adapts
the existing `CSSCode.pos` style (a list of 2D or 3D positions) into the
:class:`~weave.ir.embedding.Embedding` protocol without loss of
information.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from ...geometry import Point3
from ..embedding import IREdge, IRPolyline, RoutingGeometry


def _to_point3(p: Sequence[float]) -> Point3:
    """Lift a 2D or 3D numeric sequence into a `Point3`."""
    if len(p) == 2:
        return (float(p[0]), float(p[1]), 0.0)
    if len(p) == 3:
        return (float(p[0]), float(p[1]), float(p[2]))
    raise ValueError(f"Expected a 2D or 3D point, got length {len(p)}.")


@dataclass(frozen=True)
class StraightLineEmbedding:
    """Embedding in which every routed edge is a straight segment.

    Takes an ordered tuple of node positions (one per qubit index). The
    `__init__` accepts already-normalized `tuple[Point3, ...]`; use the
    :meth:`from_positions` factory for the ergonomic path that accepts
    2D sequences and lifts them to `z = 0`.

    Parameters
    ----------
    positions : tuple[Point3, ...]
        One 3D position per qubit, indexed by the qubit's integer index.
    name : str
        Human-readable label; defaults to ``"straight_line"``.
    """

    SCHEMA_VERSION: ClassVar[int] = 1

    positions: tuple[Point3, ...]
    name: str = "straight_line"

    def __post_init__(self) -> None:
        # Coerce list inputs to tuples for hashability.
        if not isinstance(self.positions, tuple):
            object.__setattr__(self, "positions", tuple(self.positions))
        # Validate every stored position.
        for i, p in enumerate(self.positions):
            if len(p) != 3:
                raise ValueError(
                    f"Position {i} has length {len(p)}, expected 3. "
                    f"Use StraightLineEmbedding.from_positions() to lift 2D points."
                )

    @classmethod
    def from_positions(
        cls,
        positions: Sequence[Sequence[float]],
        *,
        name: str = "straight_line",
    ) -> StraightLineEmbedding:
        """Build from a sequence of 2D or 3D positions, lifting 2D to `z = 0`.

        Parameters
        ----------
        positions : Sequence[Sequence[float]]
            One position per qubit. Each entry may be `(x, y)` or
            `(x, y, z)`.
        name : str, optional
            Label for the embedding.

        Returns
        -------
        StraightLineEmbedding
            A new embedding with lifted positions.
        """
        pts = tuple(_to_point3(p) for p in positions)
        return cls(positions=pts, name=name)

    # ------------------------------------------------------------------
    # Embedding protocol
    # ------------------------------------------------------------------

    @property
    def schema_version(self) -> int:
        return self.SCHEMA_VERSION

    @property
    def surface_name(self) -> str:
        return "plane"

    def node_position(self, qubit_idx: int) -> Point3:
        if not 0 <= qubit_idx < len(self.positions):
            raise IndexError(f"qubit_idx {qubit_idx} out of range [0, {len(self.positions)}).")
        return self.positions[qubit_idx]

    def routing_geometry(self, active_edges: Sequence[IREdge]) -> RoutingGeometry:
        edges_map: dict[IREdge, IRPolyline] = {}
        n = len(self.positions)
        for u, v in active_edges:
            if not 0 <= u < n:
                raise IndexError(f"Edge source {u} out of range [0, {n}).")
            if not 0 <= v < n:
                raise IndexError(f"Edge target {v} out of range [0, {n}).")
            edges_map[(u, v)] = (self.positions[u], self.positions[v])
        return RoutingGeometry(edges=edges_map, name=self.name)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "type": "straight_line",
            "name": self.name,
            "positions": [list(p) for p in self.positions],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> StraightLineEmbedding:
        if data.get("type") != "straight_line":
            raise ValueError(f"Expected type='straight_line', got {data.get('type')!r}.")
        version = data.get("schema_version")
        if version != cls.SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version {version}; expected {cls.SCHEMA_VERSION}."
            )
        positions = tuple(_to_point3(p) for p in data["positions"])
        return cls(
            positions=positions,
            name=data.get("name", "straight_line"),
        )

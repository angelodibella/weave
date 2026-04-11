"""The `Embedding` protocol, `RoutingGeometry` output type, and shared aliases.

An embedding fixes a 3D position for every qubit and a routed polyline for
every active Tanner-graph edge. It is the geometry half of the compiler
input signature `(CSSCode, Embedding, Schedule, Kernel, LocalNoise)`.

The protocol is structural (`@runtime_checkable`): any object with the
required attributes and methods qualifies as an `Embedding`, regardless
of inheritance. Concrete implementations in :mod:`weave.ir.embeddings`
are frozen dataclasses with validation in `__post_init__` and full JSON
round-trip.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ..geometry import Point3

IREdge = tuple[int, int]
"""A directed Tanner-graph edge: `(source_qubit_idx, target_qubit_idx)`."""

IRPolyline = tuple[Point3, ...]
"""An immutable polyline: a tuple of 3D points."""


@dataclass(frozen=True)
class RoutingGeometry:
    """Routed polylines for a set of active Tanner-graph edges.

    Produced by :meth:`Embedding.routing_geometry` as a map from
    `(source, target)` qubit pairs to the 3D polyline traversed by that
    edge's physical wire.

    Notes
    -----
    The `edges` field is a mutable dict, so `RoutingGeometry` is
    declared frozen but is **not** hashable at runtime: calling
    `hash(rg)` raises `TypeError`. Frozen here protects field
    reassignment only. The convention is that the dict contents are
    not mutated after construction; callers that need true immutability
    should treat it as read-only.

    Parameters
    ----------
    edges : dict[tuple[int, int], tuple[Point3, ...]]
        Map from Tanner-graph edges to their 3D routed polylines. Every
        polyline must contain at least 2 points.
    name : str, optional
        Label for provenance and debugging; empty by default.

    Raises
    ------
    ValueError
        If any polyline has fewer than 2 points.
    """

    edges: dict[IREdge, IRPolyline]
    name: str = ""

    def __post_init__(self) -> None:
        for edge, poly in self.edges.items():
            if len(poly) < 2:
                raise ValueError(
                    f"Polyline for edge {edge} must have at least 2 points, got {len(poly)}."
                )

    def __len__(self) -> int:
        return len(self.edges)

    def __contains__(self, edge: object) -> bool:
        return edge in self.edges

    def __getitem__(self, edge: IREdge) -> IRPolyline:
        return self.edges[edge]


@runtime_checkable
class Embedding(Protocol):
    """Structural type for a routed embedding of a Tanner graph.

    An embedding pins qubit positions in 3D and emits routed polylines
    for the gate blocks that are simultaneously active in a given
    schedule step. The geometry-aware compiler consumes embeddings to
    derive the retained correlated-noise channel from routed
    separations.

    Implementations in :mod:`weave.ir.embeddings`:

    * :class:`~weave.ir.embeddings.StraightLineEmbedding` — trivial
      2-point polylines; adapts today's 2D `CSSCode.pos` attribute into
      the IR.
    * :class:`~weave.ir.embeddings.JsonPolylineEmbedding` — arbitrary
      precomputed polylines loaded from a JSON document.

    Planned (tracked in `private/plan.md`):

    * ``ColumnEmbedding`` — 4-column single-layer layouts for BB codes
      (PR 10).
    * ``BiplanarEmbedding`` — bounded-thickness layer routing (PR 10).
    * ``SurfaceEmbedding`` — geodesic routing on a non-flat
      :class:`~weave.surface.Surface` (PR 13).

    Notes
    -----
    The :meth:`routing_geometry` method currently takes an explicit list
    of active edges. Once the `Schedule` IR lands (PR 4), the compiler
    will call it once per `ScheduleStep`, passing the extracted edge
    tuples. Embeddings that need schedule context (e.g. BB layer
    routing that depends on the algebraic term name) can add an
    overload in PR 10 without breaking existing implementations.
    """

    @property
    def schema_version(self) -> int:
        """Stable schema version for JSON serialization."""
        ...

    @property
    def surface_name(self) -> str:
        """Identifier for the underlying surface (e.g. ``"plane"``)."""
        ...

    def node_position(self, qubit_idx: int) -> Point3:
        """Return the 3D position of a qubit."""
        ...

    def routing_geometry(self, active_edges: Sequence[IREdge]) -> RoutingGeometry:
        """Return routed polylines for the given active edges.

        Parameters
        ----------
        active_edges : Sequence[tuple[int, int]]
            Ordered sequence of `(source, target)` edges to route.

        Returns
        -------
        RoutingGeometry
            A map from each requested edge to its routed polyline.
        """
        ...

    def to_json(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        ...


def load_embedding(data: dict[str, Any]) -> Embedding:
    """Load any registered embedding type from its JSON dict.

    Dispatches on the ``type`` field to the appropriate concrete
    class's `from_json` method.

    Parameters
    ----------
    data : dict
        A dict produced by some `Embedding.to_json()` call.

    Returns
    -------
    Embedding
        The reconstructed embedding.

    Raises
    ------
    ValueError
        If `type` is missing or unrecognized.
    """
    # Local import avoids a circular module load: weave.ir.embeddings
    # imports from weave.ir.embedding (this module).
    from .embeddings import JsonPolylineEmbedding, StraightLineEmbedding

    emb_type = data.get("type")
    if emb_type == "straight_line":
        return StraightLineEmbedding.from_json(data)
    if emb_type == "json_polyline":
        return JsonPolylineEmbedding.from_json(data)
    raise ValueError(
        f"Unknown embedding type {emb_type!r}; expected one of 'straight_line', 'json_polyline'."
    )

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
from .route import RouteID

IREdge = tuple[int, int]
"""A directed Tanner-graph edge: `(source_qubit_idx, target_qubit_idx)`.

This is the legacy PR 2 route identifier. PR 4 introduced
:class:`~weave.ir.route.RouteID` as the structured replacement.
`RoutingGeometry` and the :class:`Embedding` protocol accept both for
backward compatibility: tuple inputs are auto-lifted to a `RouteID`
with `step_tick=0`, `term_name=None`, `instance=0`.
"""

IRPolyline = tuple[Point3, ...]
"""An immutable polyline: a tuple of 3D points."""


@dataclass(frozen=True)
class RoutingGeometry:
    """Routed polylines for a set of active Tanner-graph edges.

    Produced by :meth:`Embedding.routing_geometry` as a map from
    :class:`~weave.ir.route.RouteID` values to the 3D polyline
    traversed by that edge's physical wire.

    Backward compatibility
    ----------------------
    Legacy `(source, target)` tuple keys are accepted in the
    constructor and auto-lifted to `RouteID(source, target, 0, None, 0)`
    in `__post_init__`. `__contains__` and `__getitem__` accept either
    a `RouteID` or a tuple; tuple lookups match the lifted default
    metadata (`step_tick=0`, `term_name=None`, `instance=0`).

    Notes
    -----
    The `edges` field is a mutable dict, so `RoutingGeometry` is
    declared frozen but is **not** hashable at runtime: calling
    `hash(rg)` raises `TypeError`. Frozen here protects field
    reassignment only.

    Parameters
    ----------
    edges : dict
        Map from routes to their 3D routed polylines. Keys may be
        `RouteID` instances (preferred) or `(source, target)` tuples
        (lifted to `RouteID` with default metadata). Every polyline
        must contain at least 2 points.
    name : str, optional
        Label for provenance and debugging; empty by default.

    Raises
    ------
    ValueError
        If any polyline has fewer than 2 points.
    TypeError
        If a key is neither a `RouteID` nor a 2-tuple of ints.
    """

    edges: dict[RouteID, IRPolyline]
    name: str = ""

    def __post_init__(self) -> None:
        # Lift tuple keys to RouteID for backward compatibility.
        lifted: dict[RouteID, IRPolyline] = {}
        for key, poly in self.edges.items():
            if isinstance(key, RouteID):
                rid = key
            elif isinstance(key, tuple) and len(key) == 2:
                rid = RouteID(source=int(key[0]), target=int(key[1]))
            else:
                raise TypeError(
                    f"RoutingGeometry edge key must be RouteID or (source, target) "
                    f"tuple, got {type(key).__name__}: {key!r}"
                )
            if len(poly) < 2:
                raise ValueError(
                    f"Polyline for edge {rid} must have at least 2 points, got {len(poly)}."
                )
            lifted[rid] = poly
        object.__setattr__(self, "edges", lifted)

    def __len__(self) -> int:
        return len(self.edges)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, RouteID):
            return key in self.edges
        if isinstance(key, tuple) and len(key) == 2:
            return RouteID(source=int(key[0]), target=int(key[1])) in self.edges
        return False

    def __getitem__(self, key: RouteID | IREdge) -> IRPolyline:
        if isinstance(key, RouteID):
            return self.edges[key]
        if isinstance(key, tuple) and len(key) == 2:
            return self.edges[RouteID(source=int(key[0]), target=int(key[1]))]
        raise TypeError(
            f"RoutingGeometry key must be RouteID or (source, target) tuple, "
            f"got {type(key).__name__}: {key!r}"
        )


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

    def routing_geometry(self, active_edges: Sequence[RouteID | IREdge]) -> RoutingGeometry:
        """Return routed polylines for the given active edges.

        Parameters
        ----------
        active_edges : Sequence[RouteID | tuple[int, int]]
            Ordered sequence of routes to lay out. Each entry may be a
            :class:`~weave.ir.route.RouteID` (preferred) or a legacy
            `(source, target)` tuple (auto-lifted).

        Returns
        -------
        RoutingGeometry
            A map from each requested route to its routed polyline,
            keyed by `RouteID`.
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

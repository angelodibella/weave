"""Typed intermediate representation for the weave compiler.

This module houses the immutable, JSON-round-trippable objects that the
geometry-aware compiler consumes and produces. It is a pure-Python layer
with no dependency on Stim, networkx, or any GUI code, so it can run
headless in CI and in benchmarks.

See `private/plan.md` for the full specification and the roadmap for
subsequent PRs (`Schedule` in PR 4, `Kernel` relocation in PR 3,
`CompiledExtraction` in PR 8).

Currently shipping
------------------
* :class:`Embedding` — structural type for routed embeddings.
* :class:`RoutingGeometry` — compiler output type for one step's polylines.
* :data:`IREdge`, :data:`IRPolyline` — type aliases.
* :class:`~weave.ir.embeddings.StraightLineEmbedding` — trivial 2-point
  polylines; adapts today's `CSSCode.pos` attribute.
* :class:`~weave.ir.embeddings.JsonPolylineEmbedding` — load arbitrary
  precomputed polylines from a JSON document.
* :func:`load_embedding` — dispatch helper for JSON deserialization.
"""

from .embedding import (
    Embedding,
    IREdge,
    IRPolyline,
    RoutingGeometry,
    load_embedding,
)
from .embeddings import JsonPolylineEmbedding, StraightLineEmbedding

__all__ = [
    "Embedding",
    "IREdge",
    "IRPolyline",
    "JsonPolylineEmbedding",
    "RoutingGeometry",
    "StraightLineEmbedding",
    "load_embedding",
]

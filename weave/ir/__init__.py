"""Typed intermediate representation for the weave compiler.

This module houses the immutable, JSON-round-trippable objects that the
geometry-aware compiler consumes and produces. It is a pure-Python layer
with no dependency on Stim, networkx, or any GUI code, so it can run
headless in CI and in benchmarks.

See `private/plan.md` for the full specification and the roadmap for
subsequent PRs (`Schedule` in PR 4, `CompiledExtraction` in PR 8).

Currently shipping
------------------
Embeddings
    :class:`Embedding`, :class:`RoutingGeometry`, :data:`IREdge`,
    :data:`IRPolyline`,
    :class:`~weave.ir.embeddings.StraightLineEmbedding`,
    :class:`~weave.ir.embeddings.JsonPolylineEmbedding`,
    :func:`load_embedding`.
Kernels
    :class:`Kernel`,
    :class:`CrossingKernel`,
    :class:`RegularizedPowerLawKernel`,
    :class:`ExponentialKernel`,
    :func:`load_kernel`.
Noise
    :class:`LocalNoiseConfig`,
    :class:`GeometryNoiseConfig`,
    :data:`GeometryScope`.
"""

from .embedding import (
    Embedding,
    IREdge,
    IRPolyline,
    RoutingGeometry,
    load_embedding,
)
from .embeddings import JsonPolylineEmbedding, StraightLineEmbedding
from .kernel import (
    CrossingKernel,
    ExponentialKernel,
    Kernel,
    RegularizedPowerLawKernel,
    load_kernel,
)
from .noise import GeometryNoiseConfig, GeometryScope, LocalNoiseConfig

__all__ = [
    "CrossingKernel",
    "Embedding",
    "ExponentialKernel",
    "GeometryNoiseConfig",
    "GeometryScope",
    "IREdge",
    "IRPolyline",
    "JsonPolylineEmbedding",
    "Kernel",
    "LocalNoiseConfig",
    "RegularizedPowerLawKernel",
    "RoutingGeometry",
    "StraightLineEmbedding",
    "load_embedding",
    "load_kernel",
]

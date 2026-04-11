"""Typed intermediate representation for the weave compiler.

This module houses the immutable, JSON-round-trippable objects that the
geometry-aware compiler consumes and produces. It is a pure-Python layer
with no dependency on Stim, networkx, or any GUI code, so it can run
headless in CI and in benchmarks.

See `private/plan.md` for the full specification and the roadmap for
subsequent PRs (compiler in PR 5, propagation analyzer in PR 7,
`CompiledExtraction` in PR 9).

Currently shipping
------------------
Routes
    :class:`RouteID`, :func:`route_id_sort_key`.
Embeddings
    :class:`Embedding`, :class:`RoutingGeometry`, :data:`IREdge`,
    :data:`IRPolyline`,
    :class:`~weave.ir.embeddings.StraightLineEmbedding`,
    :class:`~weave.ir.embeddings.JsonPolylineEmbedding`,
    :func:`load_embedding`.
Route metrics
    :class:`RoutePairMetric`,
    :class:`MinDistanceMetric`,
    :func:`load_route_metric`.
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
Schedule
    :class:`Schedule`, :class:`ScheduleStep`,
    :class:`TwoQubitEdge`, :class:`SingleQubitEdge`,
    :data:`ScheduleEdge`, :data:`ScheduleRole`, :data:`QubitRole`,
    :data:`InteractionSector`,
    :func:`default_css_schedule`.
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
from .route import RouteID, route_id_sort_key
from .route_metric import MinDistanceMetric, RoutePairMetric, load_route_metric
from .schedule import (
    InteractionSector,
    QubitRole,
    Schedule,
    ScheduleEdge,
    ScheduleRole,
    ScheduleStep,
    SingleQubitEdge,
    TwoQubitEdge,
    default_css_schedule,
)

__all__ = [
    "CrossingKernel",
    "Embedding",
    "ExponentialKernel",
    "GeometryNoiseConfig",
    "GeometryScope",
    "IREdge",
    "IRPolyline",
    "InteractionSector",
    "JsonPolylineEmbedding",
    "Kernel",
    "LocalNoiseConfig",
    "MinDistanceMetric",
    "QubitRole",
    "RegularizedPowerLawKernel",
    "RouteID",
    "RoutePairMetric",
    "RoutingGeometry",
    "Schedule",
    "ScheduleEdge",
    "ScheduleRole",
    "ScheduleStep",
    "SingleQubitEdge",
    "StraightLineEmbedding",
    "TwoQubitEdge",
    "default_css_schedule",
    "load_embedding",
    "load_kernel",
    "load_route_metric",
    "route_id_sort_key",
]

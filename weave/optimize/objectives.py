r"""Objective functions for the exposure-based embedding optimizer.

The central object an optimizer minimizes is the *kernel exposure
scale* :math:`J_\kappa` of a reference family `\mathcal{R}_X`:

.. math::

    J_\kappa(\varphi)
    \;=\;
    \max_{L \in \mathcal{R}_X}
    \sum_{\substack{e \in \mathcal{E}(\varphi) \\ \mathrm{supp}_{\mathrm{data}}(e) \subseteq L}}
    p_\kappa(d_\varphi(e)),

where `\varphi` is the current embedding, `\mathcal{E}(\varphi)` is
the set of retained single-pair events in one cycle of the
schedule, `d_\varphi(e)` is the routed distance between the two
edges of event `e` under `\varphi`, and
`p_\kappa(d) = \sin^2(\tau J_0 \kappa(d))` is the twirled pair
probability (or its weak-coupling approximation). The optimizer
searches over the embedding space keeping the code and the
schedule fixed.

This module provides:

- :class:`PairEventTemplate` — a frozen record containing the
  schedule-dependent quantities of one pair event
  (`tick_index`, the two edges, the sector, the propagated data
  support). Only the embedding-dependent quantities — routed
  distance and pair probability — are recomputed per optimizer
  iteration.
- :class:`ExposureTemplate` — a bundle of per-event templates
  together with a precomputed *event-to-support* mapping, filtered
  to events that contribute to at least one reference support.
  This is the shape the inner optimizer loop queries.
- :func:`compute_bb_ibm_event_template` — a BB-specific fast path
  for schedules produced by
  :func:`weave.codes.bb.ibm_schedule`. Derives every pair event's
  propagated data support analytically, in `O(\mathrm{events})`
  time, without running the generic propagator.
- :func:`compute_event_template_generic` — the schedule-agnostic
  version that calls :func:`weave.compiler.geometry_pass.compute_provenance`.
  Slow but works for any schedule.
- :func:`j_kappa` and :func:`j_cross` — the two scalar objective
  functions the optimizer minimizes.

Analytical fast path (BB-specific)
----------------------------------
For a BB code scheduled via
:func:`weave.codes.bb.ibm_schedule`, each cycle tick has one
monomial action per data block, so every data qubit is control (or
target) of at most one CNOT per tick. The pair fault at a tick is
therefore `P \otimes P \otimes I \otimes \dots` with `P \in \{X,
Z\}` depending on the sector, and its propagation through the rest
of the cycle leaves *exactly* the two participating data qubits in
the final data-level image (see the module docstring of
`weave.codes.bb.schedule` for the step-by-step derivation). This
observation eliminates the `~5 ms` per-event propagation cost that
makes the generic pass a non-starter for inner-loop use: on BB72
the template is assembled in `O(7560)` operations in milliseconds.
We pin the analytical formula against the generic propagator in
`test_optimize.py::TestTemplateCorrectness`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from ..analysis.propagation import propagate_single_pair_event
from ..geometry.pair import (
    exact_twirled_pair_probability,
    weak_pair_probability,
)
from ..ir import (
    CrossingKernel,
    Embedding,
    ExponentialKernel,
    Kernel,
    MinDistanceMetric,
    RegularizedPowerLawKernel,
    RoutePairMetric,
    Schedule,
    TwoQubitEdge,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..codes.bb.bb_code import BivariateBicycleCode
    from ..codes.css_code import CSSCode


InteractionSector = Literal["X", "Z"]


__all__ = [
    "ExposureTemplate",
    "NumpyExposureTemplate",
    "PairEventTemplate",
    "compute_bb_ibm_event_template",
    "compute_event_template_generic",
    "j_cross",
    "j_kappa",
    "j_kappa_numpy",
    "prepare_exposure_template",
]


# =============================================================================
# Data classes
# =============================================================================


@dataclass(frozen=True)
class PairEventTemplate:
    """Schedule-dependent half of a pair event.

    Stores everything the optimizer needs that does *not* depend on
    the current embedding: the cycle tick, the two CNOT edges, the
    sector, and the propagated data-qubit support. The
    embedding-dependent half (routed distance and pair probability)
    is recomputed per iteration.
    """

    tick_index: int
    edge_a: tuple[int, int]
    edge_b: tuple[int, int]
    sector: InteractionSector
    data_support: tuple[int, ...]


@dataclass(frozen=True)
class ExposureTemplate:
    """Pre-filtered event table for a given reference family.

    Only events whose propagated data support is contained in at
    least one reference support survive the filter — the rest
    contribute `0` to `J_\\kappa` regardless of the embedding.

    Parameters
    ----------
    events : tuple[PairEventTemplate, ...]
        The filtered events in iteration order.
    event_to_supports : tuple[tuple[int, ...], ...]
        Parallel to `events`: each entry lists the indices into
        `reference_family` whose support contains this event's
        `data_support`.
    reference_family : tuple[tuple[int, ...], ...]
        The reference supports used to build the filter. Each
        entry is a sorted tuple of data-qubit indices.
    num_supports : int
        `len(reference_family)` — cached for convenience.
    """

    events: tuple[PairEventTemplate, ...] = field(default_factory=tuple)
    event_to_supports: tuple[tuple[int, ...], ...] = field(default_factory=tuple)
    reference_family: tuple[tuple[int, ...], ...] = field(default_factory=tuple)
    num_supports: int = 0


# =============================================================================
# Template construction
# =============================================================================


def compute_bb_ibm_event_template(
    bb_code: BivariateBicycleCode, schedule: Schedule
) -> list[PairEventTemplate]:
    r"""Build the pair-event template for a BB code via the fast path.

    Valid only for schedules produced by
    :func:`weave.codes.bb.ibm_schedule` (or any schedule in which
    every data qubit participates in at most one CNOT per tick and
    the pair fault propagates analytically to the participating
    data qubits). On BB72 this runs in milliseconds — the
    generic propagator would take tens of seconds.

    For each `cnot_layer` step of `schedule.cycle_steps` and each
    sector `S \in \{X, Z\}`, enumerate the unordered pairs of
    sector-tagged edges and record

    - `data_support = sorted(edge_a.control, edge_b.control)` when
      `S = X` (Z-check CNOTs `data → z-ancilla`), or
    - `data_support = sorted(edge_a.target, edge_b.target)` when
      `S = Z` (X-check CNOTs `x-ancilla → data`).

    Shared-data-qubit pairs (which never arise in a BB ibm_schedule
    layer — each qubit is touched at most once) are skipped
    defensively.

    Parameters
    ----------
    bb_code : BivariateBicycleCode
        The code whose syndrome extraction the schedule implements.
        Not strictly required for the analytical formula (we read
        everything from `schedule`), but we keep it in the signature
        so that future BB-specific optimizations can see the code
        directly.
    schedule : Schedule
        The ibm_schedule output. Must have `cnot_layer` steps tagged
        with per-edge `interaction_sector` in `{"X", "Z"}`.

    Returns
    -------
    list[PairEventTemplate]
        One template per pair event. Sorted by
        `(tick_index, sector, edge_a, edge_b)`.
    """
    _ = bb_code  # reserved for future BB-aware extensions
    templates: list[PairEventTemplate] = []
    for step in schedule.cycle_steps:
        if step.role != "cnot_layer":
            continue
        for sector in ("X", "Z"):
            edges = [
                e
                for e in step.active_edges
                if isinstance(e, TwoQubitEdge) and e.interaction_sector == sector
            ]
            if len(edges) < 2:
                continue
            for i, j in combinations(range(len(edges)), 2):
                edge_a = edges[i]
                edge_b = edges[j]
                if sector == "X":
                    data_a = edge_a.control
                    data_b = edge_b.control
                else:
                    data_a = edge_a.target
                    data_b = edge_b.target
                if data_a == data_b:
                    continue
                data_support = tuple(sorted((data_a, data_b)))
                templates.append(
                    PairEventTemplate(
                        tick_index=step.tick_index,
                        edge_a=(edge_a.control, edge_a.target),
                        edge_b=(edge_b.control, edge_b.target),
                        sector=cast(InteractionSector, sector),
                        data_support=data_support,
                    )
                )
    templates.sort(key=lambda t: (t.tick_index, t.sector, t.edge_a, t.edge_b))
    return templates


def compute_event_template_generic(code: CSSCode, schedule: Schedule) -> list[PairEventTemplate]:
    r"""Build the pair-event template via the full propagator.

    Schedule-agnostic fallback: iterates every `cnot_layer` step of
    `schedule.cycle_steps` and every sector-relevant pair of edges,
    calls :func:`weave.analysis.propagation.propagate_single_pair_event`
    to walk the cycle, and records each propagated data support.

    This bypasses the geometry pass (which would need a concrete
    kernel and embedding, and filters zero-probability events)
    because the template is embedding-independent: once the
    schedule is fixed, every pair event's *propagated data support*
    is determined by the circuit structure alone. The
    template can then be reused across every optimizer iteration
    (which only varies the embedding).

    Slow for large schedules: on BB72 with `ibm_schedule`, the
    generic walker takes tens of seconds because every pair event
    requires a full-cycle Pauli walk. For BB codes on
    :func:`weave.codes.bb.ibm_schedule` prefer the analytical
    fast path :func:`compute_bb_ibm_event_template` instead —
    both return identical results, pinned by
    `test_fast_and_generic_match_on_bb72`.
    """
    _ = code
    data_qubits = frozenset(q for q, role in schedule.qubit_roles.items() if role == "data")
    templates: list[PairEventTemplate] = []
    for step in schedule.cycle_steps:
        if step.role != "cnot_layer":
            continue
        for sector in ("X", "Z"):
            edges = [
                e
                for e in step.active_edges
                if isinstance(e, TwoQubitEdge) and e.interaction_sector == sector
            ]
            if len(edges) < 2:
                continue
            for i, j in combinations(range(len(edges)), 2):
                edge_a = edges[i]
                edge_b = edges[j]
                result = propagate_single_pair_event(
                    schedule=schedule,
                    step_tick=step.tick_index,
                    edge_a=edge_a,
                    edge_b=edge_b,
                    sector=cast(InteractionSector, sector),
                    data_qubits=data_qubits,
                    end_block="cycle",
                )
                data_support = tuple(sorted(result.data_pauli.support))
                templates.append(
                    PairEventTemplate(
                        tick_index=step.tick_index,
                        edge_a=(edge_a.control, edge_a.target),
                        edge_b=(edge_b.control, edge_b.target),
                        sector=cast(InteractionSector, sector),
                        data_support=data_support,
                    )
                )
    templates.sort(key=lambda t: (t.tick_index, t.sector, t.edge_a, t.edge_b))
    return templates


def prepare_exposure_template(
    template: Sequence[PairEventTemplate],
    reference_family: Sequence[Sequence[int]],
) -> ExposureTemplate:
    r"""Filter a raw template by a reference family.

    Drops every event whose propagated data support is not contained
    in any reference support (those contribute `0` to
    `J_\kappa`), and precomputes the event-to-support index map that
    the inner optimizer loop uses.

    Parameters
    ----------
    template : Sequence[PairEventTemplate]
        The raw template from
        :func:`compute_bb_ibm_event_template` or
        :func:`compute_event_template_generic`.
    reference_family : Sequence[Sequence[int]]
        The reference supports. Each support is a sorted or
        unsorted iterable of data-qubit indices.

    Returns
    -------
    ExposureTemplate
    """
    family_frozen = tuple(frozenset(int(q) for q in s) for s in reference_family)
    family_sorted = tuple(tuple(sorted(s)) for s in family_frozen)

    filtered_events: list[PairEventTemplate] = []
    filtered_supports: list[tuple[int, ...]] = []
    for ev in template:
        ev_support = frozenset(ev.data_support)
        if not ev_support:
            continue
        containing = tuple(idx for idx, f in enumerate(family_frozen) if ev_support.issubset(f))
        if not containing:
            continue
        filtered_events.append(ev)
        filtered_supports.append(containing)

    return ExposureTemplate(
        events=tuple(filtered_events),
        event_to_supports=tuple(filtered_supports),
        reference_family=family_sorted,
        num_supports=len(family_sorted),
    )


# =============================================================================
# Objective functions
# =============================================================================


def j_kappa(
    embedding: Embedding,
    exposure_template: ExposureTemplate,
    kernel: Kernel,
    *,
    route_metric: RoutePairMetric | None = None,
    J0: float = 1.0,
    tau: float = 1.0,
    use_weak_limit: bool = False,
) -> float:
    r"""Kernel exposure scale :math:`J_\kappa(\varphi)`.

    For the given embedding `\varphi`, walks every filtered event in
    `exposure_template`, computes its routed distance and pair
    probability under `kernel`, and accumulates into the per-support
    exposure. Returns the max over the reference family.

    Parameters
    ----------
    embedding : Embedding
        The current embedding; queried for polyline positions.
    exposure_template : ExposureTemplate
        Precomputed event table, filtered by a reference family.
    kernel : Kernel
        Proximity kernel `\kappa(d)`.
    route_metric : RoutePairMetric, optional
        Route-pair reduction. Defaults to minimum polyline distance.
    J0, tau : float
        Physical coupling parameters. Default to 1.
    use_weak_limit : bool
        If True, use the quadratic weak-coupling approximation
        `(τ J_0 κ)^2`. Otherwise use the exact
        `sin^2(τ J_0 κ)`.

    Returns
    -------
    float
        `max_L \mathcal{E}(L; \varphi)` with `L` ranging over the
        reference family in the template. Returns `0.0` for an
        empty template.
    """
    if exposure_template.num_supports == 0 or not exposure_template.events:
        return 0.0

    metric = route_metric or MinDistanceMetric()
    per_support_exposure = [0.0] * exposure_template.num_supports

    for ev, containing in zip(
        exposure_template.events, exposure_template.event_to_supports, strict=True
    ):
        # Build the two polylines for this event under the current
        # embedding. Both are 2-point straight segments between the
        # edge endpoints — this matches StraightLineEmbedding and
        # every ColumnEmbedding descendant's routing, which is the
        # only case the optimizer currently supports.
        poly_a = (
            embedding.node_position(ev.edge_a[0]),
            embedding.node_position(ev.edge_a[1]),
        )
        poly_b = (
            embedding.node_position(ev.edge_b[0]),
            embedding.node_position(ev.edge_b[1]),
        )
        d = float(metric(poly_a, poly_b))
        if use_weak_limit:
            p = weak_pair_probability(d, tau, J0, kernel)
        else:
            p = exact_twirled_pair_probability(d, tau, J0, kernel)
        if p == 0.0:
            continue
        for idx in containing:
            per_support_exposure[idx] += p

    return max(per_support_exposure)


# =============================================================================
# Vectorized fast path
# =============================================================================


@dataclass(frozen=True)
class NumpyExposureTemplate:
    """NumPy-backed view of an :class:`ExposureTemplate` for fast j_kappa.

    The optimizer's inner loop re-evaluates :math:`J_\\kappa` thousands
    of times on the same schedule-derived template with different
    embeddings. That loop is dominated by (a) segment-segment
    distance computation and (b) the per-support exposure
    accumulation. Both vectorize cleanly in NumPy once the template
    is reshaped into flat index arrays.

    Parameters
    ----------
    edge_indices : np.ndarray
        Shape `(n_events, 4)`, dtype `int64`. Column `k` is the
        qubit index of the `k`-th endpoint of the event (ordered
        `edge_a[0], edge_a[1], edge_b[0], edge_b[1]`).
    flat_event_idx : np.ndarray
        Shape `(n_containings,)`, dtype `int64`. One entry per
        (event, containing-support) pair, giving the event index.
    flat_support_idx : np.ndarray
        Shape `(n_containings,)`, dtype `int64`. Parallel to
        `flat_event_idx`, giving the support index.
    num_supports : int
        Number of supports in the reference family.
    reference_family : tuple[tuple[int, ...], ...]
        The supports themselves, carried for provenance.
    """

    edge_indices: np.ndarray
    flat_event_idx: np.ndarray
    flat_support_idx: np.ndarray
    num_supports: int
    reference_family: tuple[tuple[int, ...], ...]

    @classmethod
    def from_exposure_template(cls, exposure_template: ExposureTemplate) -> NumpyExposureTemplate:
        """Build a vectorized view from an :class:`ExposureTemplate`."""
        n_events = len(exposure_template.events)
        edge_indices = np.zeros((n_events, 4), dtype=np.int64)
        flat_event_list: list[int] = []
        flat_support_list: list[int] = []
        for i, (ev, containing) in enumerate(
            zip(
                exposure_template.events,
                exposure_template.event_to_supports,
                strict=True,
            )
        ):
            edge_indices[i, 0] = ev.edge_a[0]
            edge_indices[i, 1] = ev.edge_a[1]
            edge_indices[i, 2] = ev.edge_b[0]
            edge_indices[i, 3] = ev.edge_b[1]
            for s in containing:
                flat_event_list.append(i)
                flat_support_list.append(s)
        return cls(
            edge_indices=edge_indices,
            flat_event_idx=np.asarray(flat_event_list, dtype=np.int64),
            flat_support_idx=np.asarray(flat_support_list, dtype=np.int64),
            num_supports=exposure_template.num_supports,
            reference_family=exposure_template.reference_family,
        )


def _segment_segment_distance_vec(
    a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray
) -> np.ndarray:
    r"""Vectorized minimum distance between stacks of 3D segments.

    Takes four `(n, 3)` arrays of segment endpoints and returns an
    `(n,)` array of segment-to-segment distances. Implements the
    same algorithm as :func:`weave.geometry.segment_distance` —
    clamped-interior closest approach when the segments are not
    parallel, plus a fallback over the four
    endpoint-to-other-segment distances to robustly handle
    parallel and collinear cases.

    Numerically consistent with the scalar version to 1e-12 for
    typical inputs.
    """
    eps = 1e-12
    u = a1 - a0
    v = b1 - b0
    w0 = a0 - b0
    a = np.einsum("ij,ij->i", u, u)
    b = np.einsum("ij,ij->i", u, v)
    c = np.einsum("ij,ij->i", v, v)
    d_prod = np.einsum("ij,ij->i", u, w0)
    e = np.einsum("ij,ij->i", v, w0)
    denom = a * c - b * b

    safe_denom = np.where(denom > eps, denom, 1.0)
    s = np.clip((b * e - c * d_prod) / safe_denom, 0.0, 1.0)
    t = np.clip((a * e - b * d_prod) / safe_denom, 0.0, 1.0)
    closest_a = a0 + s[:, None] * u
    closest_b = b0 + t[:, None] * v
    parametric_dist = np.linalg.norm(closest_a - closest_b, axis=-1)
    # Where the segments are parallel, the parametric formula is
    # degenerate — mark its value as infinity so the endpoint
    # fallback wins.
    parametric_dist = np.where(denom > eps, parametric_dist, np.inf)

    def _point_to_segment(p: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
        qv = q1 - q0
        qv_sq = np.einsum("ij,ij->i", qv, qv)
        qv_sq_safe = np.where(qv_sq > eps, qv_sq, 1.0)
        tt = np.einsum("ij,ij->i", p - q0, qv) / qv_sq_safe
        tt = np.clip(tt, 0.0, 1.0)
        closest = q0 + tt[:, None] * qv
        return np.linalg.norm(p - closest, axis=-1)

    d1 = _point_to_segment(a0, b0, b1)
    d2 = _point_to_segment(a1, b0, b1)
    d3 = _point_to_segment(b0, a0, a1)
    d4 = _point_to_segment(b1, a0, a1)
    endpoint_min = np.minimum(np.minimum(d1, d2), np.minimum(d3, d4))
    return np.minimum(parametric_dist, endpoint_min)


def _kernel_vec(kernel: Kernel, d: np.ndarray) -> np.ndarray:
    """Evaluate a kernel on a distance array.

    Uses a vectorized fast path for the three shipped kernel types
    (:class:`CrossingKernel`, :class:`RegularizedPowerLawKernel`,
    :class:`ExponentialKernel`) and falls back to a per-element
    scalar call for user-supplied kernels that do not match.
    """
    if isinstance(kernel, CrossingKernel):
        return (np.abs(d) < 1e-12).astype(float)
    if isinstance(kernel, RegularizedPowerLawKernel):
        return (1.0 + d / kernel.r0) ** (-kernel.alpha)
    if isinstance(kernel, ExponentialKernel):
        return np.exp(-d / kernel.xi)
    return np.array([float(kernel(float(di))) for di in d])


def j_kappa_numpy(
    positions_array: np.ndarray,
    numpy_template: NumpyExposureTemplate,
    kernel: Kernel,
    *,
    J0: float = 1.0,
    tau: float = 1.0,
    use_weak_limit: bool = False,
) -> float:
    r"""Vectorized :math:`J_\kappa` evaluation.

    Equivalent in result to :func:`j_kappa` but typically 10×-30×
    faster on BB72 thanks to vectorized distance and kernel
    computations plus :func:`numpy.add.at`-based support
    accumulation.

    Parameters
    ----------
    positions_array : np.ndarray
        Shape `(n_qubits, 3)` array of 3D qubit positions.
    numpy_template : NumpyExposureTemplate
        Precomputed event edge indices and flat support-index
        arrays — build it once via
        :meth:`NumpyExposureTemplate.from_exposure_template` and
        reuse across optimizer iterations.
    kernel : Kernel
        Proximity kernel. Matched against the shipped kernel
        classes for the vectorized fast path; falls back to scalar
        evaluation on user-defined kernels.
    J0, tau : float
        Physical parameters.
    use_weak_limit : bool
        If True, replace `sin^2(\tau J_0 \kappa)` with
        `(\tau J_0 \kappa)^2`.

    Returns
    -------
    float
        `max_L \mathcal{E}(L; \varphi)` — the maximum per-support
        exposure over the reference family.
    """
    if numpy_template.edge_indices.shape[0] == 0:
        return 0.0
    idx = numpy_template.edge_indices
    a0 = positions_array[idx[:, 0]]
    a1 = positions_array[idx[:, 1]]
    b0 = positions_array[idx[:, 2]]
    b1 = positions_array[idx[:, 3]]
    d = _segment_segment_distance_vec(a0, a1, b0, b1)
    kappa_vals = _kernel_vec(kernel, d)
    x = tau * J0 * kappa_vals
    p = x * x if use_weak_limit else np.sin(x) ** 2
    per_support = np.zeros(numpy_template.num_supports, dtype=float)
    np.add.at(per_support, numpy_template.flat_support_idx, p[numpy_template.flat_event_idx])
    return float(per_support.max())


def j_cross(
    embedding: Embedding,
    exposure_template: ExposureTemplate,
    *,
    route_metric: RoutePairMetric | None = None,
) -> int:
    r"""Support-crossing number `\#\{e \in \mathcal{E} : d_\varphi(e) = 0\}`.

    A specialization of :func:`j_kappa` with the combinatorial
    crossing kernel: the number of filtered events whose routed
    distance is zero is the support-crossing matching number. For
    straight-line routings this is zero iff the two polylines are
    disjoint and nonzero iff they intersect.

    We compute it directly (rather than via :func:`j_kappa` with
    :class:`CrossingKernel`) so the return type is :class:`int`
    and we avoid floating-point comparisons.

    Parameters
    ----------
    embedding : Embedding
        The current embedding; queried for polyline positions.
    exposure_template : ExposureTemplate
        Precomputed event table.
    route_metric : RoutePairMetric, optional
        Route-pair reduction. Defaults to minimum polyline distance.
        A zero return from the metric is taken as "crossing".

    Returns
    -------
    int
        Maximum crossing count over the reference supports in the
        exposure template.
    """
    if exposure_template.num_supports == 0 or not exposure_template.events:
        return 0
    metric = route_metric or MinDistanceMetric()
    per_support_count = [0] * exposure_template.num_supports
    # Use the same `1e-12` tolerance as `CrossingKernel` so this
    # function and `j_kappa(..., CrossingKernel())` agree up to
    # the `sin^2(\tau J_0)` conversion factor. Strict `d == 0.0`
    # would miss pairs whose segment-segment distance sits at
    # floating-point noise below the threshold.
    for ev, containing in zip(
        exposure_template.events, exposure_template.event_to_supports, strict=True
    ):
        poly_a = (
            embedding.node_position(ev.edge_a[0]),
            embedding.node_position(ev.edge_a[1]),
        )
        poly_b = (
            embedding.node_position(ev.edge_b[0]),
            embedding.node_position(ev.edge_b[1]),
        )
        d = float(metric(poly_a, poly_b))
        if abs(d) < 1e-12:
            for idx in containing:
                per_support_count[idx] += 1
    return max(per_support_count)

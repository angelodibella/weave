r"""Geometry pass: produce `ProvenanceRecord`s for a routed schedule.

The geometry pass is the second half of ``compile_extraction``. It is
invoked when ``geometry_noise.enabled`` (i.e. ``J0 > 0``) and turns a
routed :class:`~weave.ir.Embedding` and a :class:`~weave.ir.Schedule`
into a list of :class:`~weave.ir.ProvenanceRecord` — one record per
pair-fault event that will be emitted as a Stim ``CORRELATED_ERROR``
instruction in each round of the compiled circuit.

Algorithm
---------
For every ``cnot_layer`` step in ``schedule.cycle_steps``, and for
each CSS sector :math:`S \in \{X, Z\}`:

1. **Filter edges by sector.** Collect sector-relevant two-qubit
   edges. In the default ``geometry_scope == "full_cycle"`` mode an
   edge is sector-relevant when its ``interaction_sector`` equals
   :math:`S` or is ``None``; in ``"theory_reduced"`` mode only
   explicitly-tagged edges count. Steps with fewer than two relevant
   edges produce no pair events in that sector.
2. **Query routed polylines.** Ask the embedding for the
   :class:`~weave.ir.RoutingGeometry` of the filtered edges so every
   pair can be reduced to a scalar distance by the configured
   :class:`~weave.ir.RoutePairMetric`.
3. **Enumerate unordered pairs.** For every pair :math:`(e_a, e_b)`:

   - Compute :math:`d = \mathrm{route\_metric}(\pi_a, \pi_b)`.
   - Compute the pair probability
     :math:`p(d) = \sin^2(\tau J_0 \kappa(d))` (or its weak-coupling
     quadratic approximation if ``use_weak_limit`` is set). Skip
     events with ``pair_probability == 0``.
   - Call :func:`~weave.analysis.propagate_single_pair_event` to
     propagate the corresponding single-pair block fault through the
     remainder of the cycle, restricted to data qubits. Skip events
     whose data-level image has weight zero — those are pair faults
     the schedule cancels out on its own.
   - Emit one :class:`~weave.ir.ProvenanceRecord` summarising the
     event. The per-qubit Pauli symbols and support come from the
     propagated residual so they capture whatever kind of data-level
     error the CNOT structure ultimately produces (not just the
     sector's starting convention).

The pass is deliberately schedule-agnostic: it makes no BB-specific
assumptions. Schedules that violate the weight-$\le 2$ assumption of
§II.D (see :func:`~weave.analysis.verify_weight_le_2_assumption`) are
still accepted — their provenance records simply carry weight-3+ data
supports, which the user can audit post-hoc.

References
----------
- Di Bella, *Geometry-induced correlated noise in qLDPC syndrome
  extraction* (PRX Quantum, under review, 2026). §II.D — retained
  single-and-pair channel derivation; Assumption 2 (weight $\le 2$).
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, get_args

from ..analysis.propagation import propagate_single_pair_event
from ..geometry.pair import (
    exact_twirled_pair_probability,
    weak_pair_probability,
)
from ..ir import (
    Embedding,
    GeometryNoiseConfig,
    InteractionSector,
    Kernel,
    ProvenanceRecord,
    RouteID,
    RoutePairMetric,
    Schedule,
    ScheduleStep,
    TwoQubitEdge,
)

if TYPE_CHECKING:
    from ..analysis.pauli import Pauli

__all__ = ["compute_provenance"]


def compute_provenance(
    schedule: Schedule,
    embedding: Embedding,
    kernel: Kernel,
    route_metric: RoutePairMetric,
    geometry_noise: GeometryNoiseConfig,
) -> list[ProvenanceRecord]:
    """Walk the cycle and produce one record per emitted pair event.

    Parameters
    ----------
    schedule : Schedule
        The extraction schedule whose cycle is scanned for parallel
        pair events.
    embedding : Embedding
        The routed embedding used to compute polylines for every
        sector-relevant edge.
    kernel : Kernel
        Proximity kernel :math:`\\kappa(d)`.
    route_metric : RoutePairMetric
        Route-pair reduction (default: minimum polyline distance).
    geometry_noise : GeometryNoiseConfig
        Physical parameters; also provides the `enabled` flag and
        `geometry_scope`.

    Returns
    -------
    list[ProvenanceRecord]
        One record per `(tick, sector, edge pair)` event with
        non-zero probability and non-zero data-level image. The list
        is sorted by `(tick_index, sector, edge_a, edge_b)` for
        deterministic downstream iteration.
    """
    if not geometry_noise.enabled:
        return []

    data_qubits = frozenset(q for q, role in schedule.qubit_roles.items() if role == "data")
    records: list[ProvenanceRecord] = []

    for step in schedule.cycle_steps:
        if step.role != "cnot_layer":
            continue
        for sector in get_args(InteractionSector):
            sector_edges = _sector_relevant_edges(step, sector, geometry_noise.geometry_scope)
            if len(sector_edges) < 2:
                continue

            # Query the embedding for all sector-relevant polylines in
            # this step. One RouteID per edge, tagged with the step's
            # tick_index and the edge's term_name for disambiguation.
            route_ids = [
                RouteID(
                    source=edge.control,
                    target=edge.target,
                    step_tick=step.tick_index,
                    term_name=edge.term_name,
                )
                for edge in sector_edges
            ]
            routing_geometry = embedding.routing_geometry(route_ids)

            # Enumerate unordered pairs, canonicalized by index so
            # iteration order matches combinations(range(n), 2).
            for i, j in combinations(range(len(sector_edges)), 2):
                edge_a = sector_edges[i]
                edge_b = sector_edges[j]
                poly_a = routing_geometry[route_ids[i]]
                poly_b = routing_geometry[route_ids[j]]
                distance = float(route_metric(poly_a, poly_b))

                if geometry_noise.use_weak_limit:
                    pair_prob = weak_pair_probability(
                        distance, geometry_noise.tau, geometry_noise.J0, kernel
                    )
                else:
                    pair_prob = exact_twirled_pair_probability(
                        distance, geometry_noise.tau, geometry_noise.J0, kernel
                    )
                if pair_prob == 0.0:
                    continue

                data_pauli = _propagate_pair(
                    schedule=schedule,
                    step=step,
                    edge_a=edge_a,
                    edge_b=edge_b,
                    sector=sector,
                    data_qubits=data_qubits,
                )
                data_support = tuple(sorted(data_pauli.support))
                if not data_support:
                    # Pair fault cancels on the data level — nothing
                    # to inject. Skip the event silently.
                    continue
                data_pauli_symbols = tuple(data_pauli.pauli_on(q) for q in data_support)

                records.append(
                    ProvenanceRecord(
                        tick_index=step.tick_index,
                        edge_a=(edge_a.control, edge_a.target),
                        edge_b=(edge_b.control, edge_b.target),
                        sector=sector,
                        routed_distance=distance,
                        pair_probability=pair_prob,
                        data_support=data_support,
                        data_pauli_symbols=data_pauli_symbols,
                    )
                )

    records.sort(key=_provenance_sort_key)
    return records


# =============================================================================
# Internals
# =============================================================================


def _sector_relevant_edges(
    step: ScheduleStep,
    sector: InteractionSector,
    geometry_scope: str,
) -> list[TwoQubitEdge]:
    """Return the sector-relevant two-qubit edges of a step.

    In `"full_cycle"` scope an edge with `interaction_sector is None`
    is treated as sector-agnostic and included in both sectors' pair
    enumerations. In `"theory_reduced"` only explicitly-tagged edges
    count.
    """
    out: list[TwoQubitEdge] = []
    for edge in step.active_edges:
        if not isinstance(edge, TwoQubitEdge):
            continue
        tag = edge.interaction_sector
        if tag == sector or tag is None and geometry_scope == "full_cycle":
            out.append(edge)
    return out


def _propagate_pair(
    *,
    schedule: Schedule,
    step: ScheduleStep,
    edge_a: TwoQubitEdge,
    edge_b: TwoQubitEdge,
    sector: InteractionSector,
    data_qubits: frozenset[int],
) -> Pauli:
    """Propagate one sector pair fault to its data-level image."""
    result = propagate_single_pair_event(
        schedule=schedule,
        step_tick=step.tick_index,
        edge_a=edge_a,
        edge_b=edge_b,
        sector=sector,
        data_qubits=data_qubits,
        end_block="cycle",
    )
    return result.data_pauli


def _provenance_sort_key(
    rec: ProvenanceRecord,
) -> tuple[int, str, tuple[int, int], tuple[int, int]]:
    return (rec.tick_index, rec.sector, rec.edge_a, rec.edge_b)

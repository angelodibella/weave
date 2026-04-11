r"""Weight-$\le 2$ propagation validation for correlated-noise schedules.

The correlated single-and-pair noise channel derived in the PRX-Quantum-
under-review paper *Geometry-induced correlated noise in qLDPC syndrome
extraction* (Di Bella 2026, §II.D) relies on a structural assumption
which, for reference, we reproduce here:

    **Assumption 2 (Weight-$\le 2$ propagation).** Let `e, e'` be two
    gate blocks active in the same tick of a sector-relevant CNOT
    layer, and let `P_e ⊗ P_{e'}` be the microscopic block-Pauli of
    the retained pair channel for that sector. Propagating
    `P_e ⊗ P_{e'}` through the remainder of the schedule — including
    all subsequent Cliffords and ancilla measurements — yields a
    data-level Pauli of Hamming weight at most 2.

This assumption is the reason the retained channel can be modelled as
a *two-body* correlated error at the decoder level. It holds trivially
for non-interleaved schedules whose CNOTs are scheduled one at a time
(e.g. :func:`~weave.ir.default_css_schedule`), because the "parallel
pair" precondition is never satisfied. It also holds for the
sector-colored BB depth-8 schedule in the paper, but does *not* hold
in general: an arbitrary parallel schedule may propagate pair faults
to data-weight `> 2`, in which case the retained channel is a
conservative but inaccurate approximation of the true fault model.

This module provides a schedule-agnostic checker that either certifies
the assumption holds or returns a structured report of the violating
parallel-pair events.

Algorithm
---------
Given a schedule and a CSS sector (`"X"` or `"Z"`):

1. Walk `schedule.cycle_steps` and select every `cnot_layer` step
   whose `active_edges` contain at least two :class:`TwoQubitEdge`
   instances carrying the requested sector (or with an unset sector,
   which is treated as sector-agnostic). Steps with fewer than two
   sector-relevant edges cannot produce a pair fault and are skipped.
2. For each such step, enumerate the unordered pairs
   `(edge_a, edge_b)` and build the single-pair fault via
   :func:`~weave.analysis.propagation.build_single_pair_fault`.
3. Propagate the fault through the rest of the cycle and the tail via
   :func:`~weave.analysis.propagation.propagate_fault`, restricting to
   data qubits. Record `(step_tick, edge_a, edge_b, data_weight)`.
4. Collect pass / fail events into a :class:`ValidationReport`.

The resulting report surfaces the specific edges that cause a
violation so schedule authors can see exactly which CNOT layer needs
to be re-ordered or un-paired.

References
----------
- Di Bella, *Geometry-induced correlated noise in qLDPC syndrome
  extraction* (PRX Quantum, under review, 2026). §II.D, Assumption 2.
- Gottesman, *Stabilizer Codes and Quantum Error Correction*, Caltech
  PhD thesis, arXiv:quant-ph/9705052 (1997), for the underlying
  fault-propagation machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from ..ir import InteractionSector, Schedule, ScheduleStep, TwoQubitEdge
from .propagation import (
    FaultLocation,
    build_single_pair_fault,
    propagate_fault,
)

__all__ = [
    "PairEventResult",
    "ValidationReport",
    "verify_weight_le_2_assumption",
]


# =============================================================================
# Per-event record
# =============================================================================


@dataclass(frozen=True)
class PairEventResult:
    """Outcome of propagating one sector-pair event in one CNOT step.

    Parameters
    ----------
    tick_index : int
        `ScheduleStep.tick_index` of the CNOT layer.
    edge_a, edge_b : TwoQubitEdge
        The two simultaneously-active edges forming the pair.
    data_weight : int
        Hamming weight of the data-level residual after propagation
        through the rest of the cycle and the tail.
    flipped_ancilla_count : int
        Number of ancilla measurements flipped by this fault. Reported
        for diagnostics — not used in the assumption check itself.
    passed : bool
        `True` iff `data_weight <= 2`.
    """

    tick_index: int
    edge_a: TwoQubitEdge
    edge_b: TwoQubitEdge
    data_weight: int
    flipped_ancilla_count: int
    passed: bool


# =============================================================================
# Aggregate report
# =============================================================================


@dataclass(frozen=True)
class ValidationReport:
    """Verdict and event list for a weight-$\\le 2$ check.

    Parameters
    ----------
    sector : {"X", "Z"}
        The sector the check was run for.
    schedule_name : str
        `Schedule.name` of the schedule under test.
    events : tuple[PairEventResult, ...]
        One entry per propagated pair event, in (tick, edge-pair)
        order. Empty if the schedule has no sector-relevant parallel
        pair events — which is the common case for naive serial
        schedules and is still a *pass*.
    violations : tuple[PairEventResult, ...]
        The subset of `events` with `passed = False`.
    passed : bool
        `True` iff `len(violations) == 0`. An empty `events` tuple
        also counts as a pass (there is nothing to violate).
    """

    sector: InteractionSector
    schedule_name: str
    events: tuple[PairEventResult, ...] = field(default_factory=tuple)
    violations: tuple[PairEventResult, ...] = field(default_factory=tuple)
    passed: bool = True

    def summary(self) -> str:
        """One-line human-readable summary for logs and CLI output."""
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"[{verdict}] sector={self.sector} schedule={self.schedule_name!r} "
            f"events={len(self.events)} violations={len(self.violations)}"
        )


# =============================================================================
# Verification entry point
# =============================================================================


def verify_weight_le_2_assumption(
    schedule: Schedule,
    sector: InteractionSector,
) -> ValidationReport:
    r"""Check the weight-$\le 2$ propagation assumption for a sector.

    Iterates every `cnot_layer` step in `schedule.cycle_steps`,
    collects the sector-relevant edges, forms all unordered pairs,
    and propagates each pair fault through the remainder of the cycle
    and the tail. A pair event passes iff its data-level residual has
    weight $\le 2$.

    The check is exact — it uses the same symplectic propagator as
    :mod:`weave.analysis.propagation`. It is also schedule-agnostic:
    the only structural requirement is that `cycle_steps` contain
    :class:`~weave.ir.TwoQubitEdge` instances tagged with
    `interaction_sector == sector` (or with `interaction_sector is
    None`, in which case the edge is treated as sector-agnostic and
    included).

    Parameters
    ----------
    schedule : Schedule
        The schedule to validate. The walker reads `cycle_steps` for
        pair enumeration and walks the remainder of the cycle + tail
        for propagation.
    sector : {"X", "Z"}
        Which sector to validate. `"X"` checks the retained channel
        driving X-error propagation (Z-check CNOTs in CSS); `"Z"`
        checks the Z-error channel (X-check CNOTs).

    Returns
    -------
    ValidationReport
        The structured verdict. Schedules with no sector-relevant
        parallel pairs return an empty-events `passed=True` report.

    Notes
    -----
    * The number of events scales as the sum over sector-relevant
      CNOT steps of `C(k, 2)`, where `k` is the number of
      sector-relevant edges in that step. For typical CSS schedules
      this is small (a handful of parallel checks times a few pairs),
      so the check is not a performance concern.
    * Faults are injected at the tick of the parallel pair, *before*
      the CNOTs at that tick fire, matching the paper's convention:
      the retained channel models noise that "happens at" the pair
      location and then gets conjugated forward.
    """
    data_qubits = frozenset(q for q, role in schedule.qubit_roles.items() if role == "data")

    events: list[PairEventResult] = []
    violations: list[PairEventResult] = []

    for step in schedule.cycle_steps:
        if step.role != "cnot_layer":
            continue
        sector_edges = _sector_relevant_edges(step, sector)
        if len(sector_edges) < 2:
            continue
        for edge_a, edge_b in combinations(sector_edges, 2):
            event = _propagate_pair(
                schedule=schedule,
                step=step,
                edge_a=edge_a,
                edge_b=edge_b,
                sector=sector,
                data_qubits=data_qubits,
            )
            events.append(event)
            if not event.passed:
                violations.append(event)

    return ValidationReport(
        sector=sector,
        schedule_name=schedule.name,
        events=tuple(events),
        violations=tuple(violations),
        passed=not violations,
    )


# =============================================================================
# Internals
# =============================================================================


def _sector_relevant_edges(step: ScheduleStep, sector: InteractionSector) -> list[TwoQubitEdge]:
    """Return the edges in `step` that contribute to `sector`.

    An edge is sector-relevant iff (a) it is a :class:`TwoQubitEdge`
    and (b) its `interaction_sector` is either `sector` or `None`. A
    declared sector different from the requested one excludes the
    edge from that sector's pair enumeration.
    """
    out: list[TwoQubitEdge] = []
    for edge in step.active_edges:
        if not isinstance(edge, TwoQubitEdge):
            continue
        if edge.interaction_sector is None or edge.interaction_sector == sector:
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
) -> PairEventResult:
    """Propagate one `(edge_a, edge_b)` pair fault and grade it."""
    num_qubits = max(schedule.qubits) + 1 if schedule.qubits else 0
    initial = build_single_pair_fault(edge_a, edge_b, sector, num_qubits)
    location = FaultLocation(block="cycle", tick_index=step.tick_index)
    result = propagate_fault(
        schedule=schedule,
        initial_fault=initial,
        injection_location=location,
        data_qubits=data_qubits,
        end_block="cycle",
    )
    data_weight = result.data_weight
    return PairEventResult(
        tick_index=step.tick_index,
        edge_a=edge_a,
        edge_b=edge_b,
        data_weight=data_weight,
        flipped_ancilla_count=len(result.ancilla_flips),
        passed=data_weight <= 2,
    )

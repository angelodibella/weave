"""Tests for `verify_weight_le_2_assumption` on hand-crafted schedules.

The weight-$\\le 2$ assumption is automatically satisfied by
`default_css_schedule` because its cycle fires one CNOT at a time,
so there are no parallel pair events to check; we test that the
validator reports an empty, passing verdict in that case.

For a genuine end-to-end check we construct two tiny parallel
schedules by hand:

1. A 2-data / 2-ancilla Z-check-style schedule where two parallel
   CNOTs are measured, and the pair fault trivially ends with
   weight-2 data residual. Expected: PASS with 1 event.
2. A 3-data / 2-ancilla X-check-style schedule with an extra
   post-pair CNOT that spreads a Z-sector pair fault to a third
   data qubit, giving weight-3 residual. Expected: FAIL with
   violation reported at the pair tick.

The schedules are constructed directly as :class:`Schedule` objects
so the tests don't depend on the geometry compiler or any BB code
family.
"""

from __future__ import annotations

import pytest

from weave.analysis.validation import (
    PairEventResult,
    ValidationReport,
    verify_weight_le_2_assumption,
)
from weave.codes.css_code import CSSCode
from weave.ir import (
    Schedule,
    ScheduleEdge,
    ScheduleStep,
    SingleQubitEdge,
    TwoQubitEdge,
    default_css_schedule,
)
from weave.ir.schedule import QubitRole, ScheduleRole
from weave.util import pcm

# ---------------------------------------------------------------------------
# Helpers for building tiny schedules by hand
# ---------------------------------------------------------------------------


def _make_step(
    tick: int,
    role: ScheduleRole,
    edges: list[ScheduleEdge],
    all_qubits: frozenset[int],
) -> ScheduleStep:
    active: set[int] = set()
    for e in edges:
        for q in e.qubits:
            active.add(q)
    idle = all_qubits - active
    return ScheduleStep(
        tick_index=tick,
        role=role,
        active_edges=tuple(edges),
        active_qubits=frozenset(active),
        idle_qubits=idle,
    )


# ---------------------------------------------------------------------------
# Baseline: default_css_schedule has no parallel pair events
# ---------------------------------------------------------------------------


class TestDefaultScheduleVacuouslyPasses:
    def test_steane_x_sector_empty_events(self):
        """Steane `default_css_schedule` fires one CNOT per tick,
        so no parallel-pair event exists in either sector."""
        H = pcm.hamming(7)
        code = CSSCode(HX=H, HZ=H, rounds=1)
        schedule = default_css_schedule(code)
        report = verify_weight_le_2_assumption(schedule, sector="X")
        assert isinstance(report, ValidationReport)
        assert report.passed is True
        assert report.events == ()
        assert report.violations == ()

    def test_steane_z_sector_empty_events(self):
        H = pcm.hamming(7)
        code = CSSCode(HX=H, HZ=H, rounds=1)
        schedule = default_css_schedule(code)
        report = verify_weight_le_2_assumption(schedule, sector="Z")
        assert report.passed is True
        assert report.events == ()


# ---------------------------------------------------------------------------
# Positive parallel case: weight-2 pair fault stays weight-2
# ---------------------------------------------------------------------------


@pytest.fixture
def passing_parallel_x_schedule() -> Schedule:
    """A 2-data / 2-ancilla schedule with one parallel CNOT tick.

    Data: {0, 1}. Ancillas: {2, 3}. The cycle has a single parallel
    `cnot_layer` step that fires both Z-checks simultaneously, then
    measures both ancillas. An X-sector pair fault `X_0 X_1` picks
    up `X_2 X_3` at the CNOT step and is exactly cancelled by the
    ancilla `MR` that follows, leaving weight-2 data residual.
    """
    qubits = frozenset({0, 1, 2, 3})
    roles: dict[int, QubitRole] = {0: "data", 1: "data", 2: "z_ancilla", 3: "z_ancilla"}

    cycle: list[ScheduleStep] = []
    # Tick 0: parallel CNOTs, both in the X sector (data -> Z-ancilla).
    cycle.append(
        _make_step(
            tick=0,
            role="cnot_layer",
            edges=[
                TwoQubitEdge(gate="CNOT", control=0, target=2, interaction_sector="X"),
                TwoQubitEdge(gate="CNOT", control=1, target=3, interaction_sector="X"),
            ],
            all_qubits=qubits,
        )
    )
    # Tick 1: MR on ancillas.
    cycle.append(
        _make_step(
            tick=1,
            role="meas",
            edges=[
                SingleQubitEdge(gate="MR", qubit=2),
                SingleQubitEdge(gate="MR", qubit=3),
            ],
            all_qubits=qubits,
        )
    )

    return Schedule(
        head_steps=(),
        cycle_steps=tuple(cycle),
        tail_steps=(),
        qubits=qubits,
        qubit_roles=roles,
        name="parallel_pair_passing",
    )


class TestPassingParallelPair:
    def test_report_passed(self, passing_parallel_x_schedule):
        report = verify_weight_le_2_assumption(passing_parallel_x_schedule, sector="X")
        assert report.passed is True

    def test_report_event_count_and_weight(self, passing_parallel_x_schedule):
        report = verify_weight_le_2_assumption(passing_parallel_x_schedule, sector="X")
        # Exactly one pair event (two edges → C(2,2) = 1 pair).
        assert len(report.events) == 1
        ev = report.events[0]
        assert isinstance(ev, PairEventResult)
        # Pair X fault propagates to X_0 X_1 after ancilla elimination.
        assert ev.data_weight == 2
        assert ev.passed is True
        assert ev.tick_index == 0
        # Both ancillas were flipped by the pair fault.
        assert ev.flipped_ancilla_count == 2

    def test_z_sector_finds_no_events(self, passing_parallel_x_schedule):
        """The schedule has only X-sector CNOTs, so Z-sector validation
        has nothing to enumerate."""
        report = verify_weight_le_2_assumption(passing_parallel_x_schedule, sector="Z")
        assert report.passed is True
        assert report.events == ()


# ---------------------------------------------------------------------------
# Negative case: weight-3 violation in the Z sector
# ---------------------------------------------------------------------------


@pytest.fixture
def violating_parallel_z_schedule() -> Schedule:
    """A 3-data / 2-ancilla schedule that violates weight-$\\le 2$
    propagation in the Z sector.

    Data: {0, 1, 2}. Ancillas: {3, 4}. Cycle:
      tick 0: CNOT(3,0) [Z], CNOT(4,1) [Z]              # parallel pair
      tick 1: CNOT(2,3)                                 # data 2 controls ancilla 3
      tick 2: MR(3), MR(4)

    Pair fault in the Z sector: `Z_0 Z_1` (Z on the pair's targets).
    The CNOT Z-propagation rule is
    `z[control] ← z[control] ⊕ z[target]`, i.e. Z spreads from target
    to control.

    - tick 0: CNOT(3,0) spreads Z_0 up to control 3 → fault gains Z_3.
              CNOT(4,1) spreads Z_1 up to control 4 → fault gains Z_4.
              Running fault: Z_0 Z_1 Z_3 Z_4.
    - tick 1: CNOT(2,3) spreads Z_3 (now on qubit 3, the target of
              THIS cnot) down to qubit 2 (the control):
              `z[2] ← z[2] ⊕ z[3] = 0 ⊕ 1 = 1`. Fault: Z_0 Z_1 Z_2 Z_3 Z_4.
    - tick 2: MR(3) measures Z on qubit 3. Z commutes with Z, so
              no flip; but MR clears qubit 3's fault entries.
              Similarly MR(4). Fault: Z_0 Z_1 Z_2.

    Data residual weight = 3 → the weight-$\\le 2$ assumption is
    violated, as the validator must detect. This schedule is a
    minimal counterexample and is NOT representative of the paper's
    sector-colored BB schedule, which is itself proven to satisfy the
    assumption.
    """
    qubits = frozenset({0, 1, 2, 3, 4})
    roles: dict[int, QubitRole] = {
        0: "data",
        1: "data",
        2: "data",
        3: "x_ancilla",
        4: "x_ancilla",
    }

    cycle: list[ScheduleStep] = []
    # Parallel pair CNOT tick (both Z sector).
    cycle.append(
        _make_step(
            tick=0,
            role="cnot_layer",
            edges=[
                TwoQubitEdge(gate="CNOT", control=3, target=0, interaction_sector="Z"),
                TwoQubitEdge(gate="CNOT", control=4, target=1, interaction_sector="Z"),
            ],
            all_qubits=qubits,
        )
    )
    # Post-pair CNOT: data 2 controls ancilla 3. The fault's Z_3
    # component (picked up in tick 0) propagates down to Z_2 here.
    # Untagged (sector-agnostic) so it never generates its own
    # pair event — it is alone at this tick anyway.
    cycle.append(
        _make_step(
            tick=1,
            role="cnot_layer",
            edges=[
                TwoQubitEdge(gate="CNOT", control=2, target=3),
            ],
            all_qubits=qubits,
        )
    )
    # MR on ancillas.
    cycle.append(
        _make_step(
            tick=2,
            role="meas",
            edges=[
                SingleQubitEdge(gate="MR", qubit=3),
                SingleQubitEdge(gate="MR", qubit=4),
            ],
            all_qubits=qubits,
        )
    )

    return Schedule(
        head_steps=(),
        cycle_steps=tuple(cycle),
        tail_steps=(),
        qubits=qubits,
        qubit_roles=roles,
        name="parallel_pair_violating",
    )


class TestViolatingParallelPair:
    def test_report_failed(self, violating_parallel_z_schedule):
        report = verify_weight_le_2_assumption(violating_parallel_z_schedule, sector="Z")
        assert report.passed is False

    def test_violation_details(self, violating_parallel_z_schedule):
        report = verify_weight_le_2_assumption(violating_parallel_z_schedule, sector="Z")
        # One pair tick, one pair → one event, which is the violation.
        assert len(report.events) == 1
        assert len(report.violations) == 1
        violation = report.violations[0]
        assert violation.tick_index == 0  # the parallel pair tick
        assert violation.data_weight == 3
        assert violation.passed is False

    def test_summary_contains_fail_and_counts(self, violating_parallel_z_schedule):
        report = verify_weight_le_2_assumption(violating_parallel_z_schedule, sector="Z")
        summary = report.summary()
        assert "FAIL" in summary
        assert "violations=1" in summary
        assert "sector=Z" in summary


# ---------------------------------------------------------------------------
# Sector filtering
# ---------------------------------------------------------------------------


class TestSectorFiltering:
    def test_steps_with_fewer_than_two_sector_edges_are_skipped(self):
        """A step with one X-edge and one Z-edge has zero parallel
        pairs in either sector and must be silently skipped."""
        qubits = frozenset({0, 1, 2, 3})
        roles: dict[int, QubitRole] = {
            0: "data",
            1: "data",
            2: "z_ancilla",
            3: "x_ancilla",
        }
        step = _make_step(
            tick=0,
            role="cnot_layer",
            edges=[
                TwoQubitEdge(gate="CNOT", control=0, target=2, interaction_sector="X"),
                TwoQubitEdge(gate="CNOT", control=3, target=1, interaction_sector="Z"),
            ],
            all_qubits=qubits,
        )
        sched = Schedule(
            head_steps=(),
            cycle_steps=(step,),
            tail_steps=(),
            qubits=qubits,
            qubit_roles=roles,
            name="mixed_sector",
        )
        for sector in ("X", "Z"):
            report = verify_weight_le_2_assumption(sched, sector=sector)
            assert report.passed is True
            assert report.events == ()

"""Tests for the v2 `Schedule` IR and `default_css_schedule`.

Covers `TwoQubitEdge`, `SingleQubitEdge`, `ScheduleStep`, `Schedule`,
all invariants, full JSON round-trip, and the `default_css_schedule`
factory on the Steane code.
"""

from __future__ import annotations

import json

import pytest

from weave.codes.css_code import CSSCode
from weave.ir import (
    Schedule,
    ScheduleStep,
    SingleQubitEdge,
    TwoQubitEdge,
    default_css_schedule,
)
from weave.util import pcm

# =============================================================================
# TwoQubitEdge
# =============================================================================


class TestTwoQubitEdge:
    def test_basic_construction(self):
        e = TwoQubitEdge(gate="CNOT", control=0, target=7)
        assert e.gate == "CNOT"
        assert e.control == 0
        assert e.target == 7
        assert e.interaction_sector is None
        assert e.term_name is None

    def test_qubits_property(self):
        e = TwoQubitEdge(gate="CNOT", control=3, target=12)
        assert e.qubits == (3, 12)

    def test_with_sector_and_term_name(self):
        e = TwoQubitEdge(
            gate="CNOT",
            control=0,
            target=7,
            interaction_sector="X",
            term_name="HZ[0,0]",
        )
        assert e.interaction_sector == "X"
        assert e.term_name == "HZ[0,0]"

    def test_rejects_control_equals_target(self):
        with pytest.raises(ValueError, match="must be distinct"):
            TwoQubitEdge(gate="CNOT", control=5, target=5)

    def test_frozen(self):
        from dataclasses import FrozenInstanceError

        e = TwoQubitEdge(gate="CNOT", control=0, target=1)
        with pytest.raises(FrozenInstanceError):
            e.control = 2  # type: ignore[misc]

    def test_equality(self):
        a = TwoQubitEdge(gate="CNOT", control=0, target=1, interaction_sector="X")
        b = TwoQubitEdge(gate="CNOT", control=0, target=1, interaction_sector="X")
        assert a == b
        assert hash(a) == hash(b)

    def test_to_from_json_roundtrip(self):
        e = TwoQubitEdge(
            gate="CNOT",
            control=3,
            target=12,
            interaction_sector="X",
            term_name="HZ[0,3]",
        )
        data = e.to_json()
        restored = TwoQubitEdge.from_json(data)
        assert restored == e

    def test_to_json_has_kind_discriminator(self):
        data = TwoQubitEdge(gate="CNOT", control=0, target=1).to_json()
        assert data["kind"] == "two_qubit"


# =============================================================================
# SingleQubitEdge
# =============================================================================


class TestSingleQubitEdge:
    def test_basic_construction(self):
        e = SingleQubitEdge(gate="H", qubit=5)
        assert e.gate == "H"
        assert e.qubit == 5

    def test_qubits_property(self):
        e = SingleQubitEdge(gate="R", qubit=3)
        assert e.qubits == (3,)

    def test_various_gates(self):
        for gate in ["H", "X", "Y", "Z", "S", "R", "RX", "M", "MX", "MR", "MRX", "I"]:
            e = SingleQubitEdge(gate=gate, qubit=0)  # type: ignore[arg-type]
            assert e.gate == gate

    def test_to_from_json_roundtrip(self):
        e = SingleQubitEdge(gate="MR", qubit=7, term_name="meas_ancilla")
        data = e.to_json()
        restored = SingleQubitEdge.from_json(data)
        assert restored == e

    def test_to_json_has_kind_discriminator(self):
        data = SingleQubitEdge(gate="H", qubit=0).to_json()
        assert data["kind"] == "single_qubit"


# =============================================================================
# ScheduleStep invariants
# =============================================================================


class TestScheduleStepInvariants:
    def _cnot_step(self):
        """A valid cnot_layer step."""
        return ScheduleStep(
            tick_index=0,
            role="cnot_layer",
            active_edges=(TwoQubitEdge(gate="CNOT", control=0, target=1),),
            active_qubits=frozenset({0, 1}),
            idle_qubits=frozenset({2, 3}),
        )

    def test_basic_construction(self):
        step = self._cnot_step()
        assert step.tick_index == 0
        assert step.role == "cnot_layer"
        assert len(step.active_edges) == 1
        assert step.duration == 1.0

    def test_coerces_list_to_tuple(self):
        step = ScheduleStep(
            tick_index=0,
            role="cnot_layer",
            active_edges=[TwoQubitEdge(gate="CNOT", control=0, target=1)],  # type: ignore[arg-type]
            active_qubits=frozenset({0, 1}),
            idle_qubits=frozenset(),
        )
        assert isinstance(step.active_edges, tuple)

    def test_coerces_set_to_frozenset(self):
        step = ScheduleStep(
            tick_index=0,
            role="cnot_layer",
            active_edges=(TwoQubitEdge(gate="CNOT", control=0, target=1),),
            active_qubits={0, 1},  # type: ignore[arg-type]
            idle_qubits={2, 3},  # type: ignore[arg-type]
        )
        assert isinstance(step.active_qubits, frozenset)
        assert isinstance(step.idle_qubits, frozenset)

    def test_rejects_overlapping_active_and_idle(self):
        with pytest.raises(ValueError, match="disjoint"):
            ScheduleStep(
                tick_index=0,
                role="cnot_layer",
                active_edges=(TwoQubitEdge(gate="CNOT", control=0, target=1),),
                active_qubits=frozenset({0, 1}),
                idle_qubits=frozenset({1, 2}),
            )

    def test_rejects_cnot_layer_with_single_qubit_edge(self):
        with pytest.raises(ValueError, match="cnot_layer"):
            ScheduleStep(
                tick_index=0,
                role="cnot_layer",
                active_edges=(SingleQubitEdge(gate="H", qubit=0),),
                active_qubits=frozenset({0}),
                idle_qubits=frozenset(),
            )

    def test_rejects_single_q_with_two_qubit_edge(self):
        with pytest.raises(ValueError, match="single_q"):
            ScheduleStep(
                tick_index=0,
                role="single_q",
                active_edges=(TwoQubitEdge(gate="CNOT", control=0, target=1),),
                active_qubits=frozenset({0, 1}),
                idle_qubits=frozenset(),
            )

    def test_rejects_meas_with_two_qubit_edge(self):
        with pytest.raises(ValueError, match="meas"):
            ScheduleStep(
                tick_index=0,
                role="meas",
                active_edges=(TwoQubitEdge(gate="CNOT", control=0, target=1),),
                active_qubits=frozenset({0, 1}),
                idle_qubits=frozenset(),
            )

    def test_rejects_multi_edge_sharing_qubit(self):
        with pytest.raises(ValueError, match="touched by multiple edges"):
            ScheduleStep(
                tick_index=0,
                role="cnot_layer",
                active_edges=(
                    TwoQubitEdge(gate="CNOT", control=0, target=1),
                    TwoQubitEdge(gate="CNOT", control=0, target=2),  # 0 repeats
                ),
                active_qubits=frozenset({0, 1, 2}),
                idle_qubits=frozenset(),
            )

    def test_rejects_edge_qubit_not_in_active(self):
        with pytest.raises(ValueError, match="active_qubits must contain"):
            ScheduleStep(
                tick_index=0,
                role="cnot_layer",
                active_edges=(TwoQubitEdge(gate="CNOT", control=0, target=5),),
                active_qubits=frozenset({0}),  # missing 5
                idle_qubits=frozenset(),
            )

    def test_rejects_zero_duration(self):
        with pytest.raises(ValueError, match="duration must be positive"):
            ScheduleStep(
                tick_index=0,
                role="cnot_layer",
                active_edges=(TwoQubitEdge(gate="CNOT", control=0, target=1),),
                active_qubits=frozenset({0, 1}),
                idle_qubits=frozenset(),
                duration=0.0,
            )

    def test_rejects_negative_duration(self):
        with pytest.raises(ValueError, match="duration must be positive"):
            ScheduleStep(
                tick_index=0,
                role="cnot_layer",
                active_edges=(TwoQubitEdge(gate="CNOT", control=0, target=1),),
                active_qubits=frozenset({0, 1}),
                idle_qubits=frozenset(),
                duration=-1.0,
            )

    def test_parallel_cnot_layer_valid(self):
        """Two CNOTs on disjoint qubits in one tick is valid."""
        step = ScheduleStep(
            tick_index=0,
            role="cnot_layer",
            active_edges=(
                TwoQubitEdge(gate="CNOT", control=0, target=1),
                TwoQubitEdge(gate="CNOT", control=2, target=3),
            ),
            active_qubits=frozenset({0, 1, 2, 3}),
            idle_qubits=frozenset(),
        )
        assert len(step.active_edges) == 2


class TestScheduleStepJson:
    def test_roundtrip(self):
        step = ScheduleStep(
            tick_index=3,
            role="cnot_layer",
            active_edges=(
                TwoQubitEdge(
                    gate="CNOT",
                    control=0,
                    target=7,
                    interaction_sector="X",
                    term_name="HZ[0,0]",
                ),
            ),
            active_qubits=frozenset({0, 7}),
            idle_qubits=frozenset({1, 2, 3}),
            duration=1.5,
        )
        data = step.to_json()
        restored = ScheduleStep.from_json(data)
        assert restored == step

    def test_roundtrip_empty_tick_barrier(self):
        step = ScheduleStep(
            tick_index=0,
            role="tick_barrier",
            active_edges=(),
            active_qubits=frozenset(),
            idle_qubits=frozenset({0, 1}),
        )
        restored = ScheduleStep.from_json(step.to_json())
        assert restored == step


# =============================================================================
# Schedule invariants and roundtrip
# =============================================================================


class TestScheduleInvariants:
    def _trivial_schedule(self) -> Schedule:
        """A minimal valid schedule."""
        return Schedule(
            head_steps=(),
            cycle_steps=(
                ScheduleStep(
                    tick_index=0,
                    role="cnot_layer",
                    active_edges=(TwoQubitEdge(gate="CNOT", control=0, target=1),),
                    active_qubits=frozenset({0, 1}),
                    idle_qubits=frozenset(),
                ),
            ),
            tail_steps=(),
            qubits=frozenset({0, 1}),
            qubit_roles={0: "data", 1: "z_ancilla"},
        )

    def test_basic_schedule(self):
        s = self._trivial_schedule()
        assert s.cycle_depth == 1
        assert len(s.all_steps()) == 1
        assert s.schema_version == 1

    def test_coerces_qubits_to_frozenset(self):
        s = Schedule(
            head_steps=(),
            cycle_steps=(),
            tail_steps=(),
            qubits={0, 1, 2},  # type: ignore[arg-type]
            qubit_roles={0: "data", 1: "data", 2: "z_ancilla"},
        )
        assert isinstance(s.qubits, frozenset)

    def test_rejects_missing_role_for_qubit(self):
        with pytest.raises(ValueError, match="qubit_roles keys must equal"):
            Schedule(
                head_steps=(),
                cycle_steps=(),
                tail_steps=(),
                qubits=frozenset({0, 1, 2}),
                qubit_roles={0: "data", 1: "z_ancilla"},  # missing 2
            )

    def test_rejects_extra_role_for_nonexistent_qubit(self):
        with pytest.raises(ValueError, match="qubit_roles keys must equal"):
            Schedule(
                head_steps=(),
                cycle_steps=(),
                tail_steps=(),
                qubits=frozenset({0, 1}),
                qubit_roles={0: "data", 1: "z_ancilla", 2: "x_ancilla"},
            )

    def test_rejects_non_monotonic_tick_index_in_cycle(self):
        step_a = ScheduleStep(
            tick_index=2,
            role="single_q",
            active_edges=(SingleQubitEdge(gate="H", qubit=0),),
            active_qubits=frozenset({0}),
            idle_qubits=frozenset(),
        )
        step_b = ScheduleStep(
            tick_index=1,  # goes backwards
            role="single_q",
            active_edges=(SingleQubitEdge(gate="X", qubit=0),),
            active_qubits=frozenset({0}),
            idle_qubits=frozenset(),
        )
        with pytest.raises(ValueError, match="not strictly increasing"):
            Schedule(
                head_steps=(),
                cycle_steps=(step_a, step_b),
                tail_steps=(),
                qubits=frozenset({0}),
                qubit_roles={0: "data"},
            )

    def test_tick_index_resets_across_blocks(self):
        """head tick 0, cycle tick 0, tail tick 0 is valid (resets per block)."""
        reset_step = ScheduleStep(
            tick_index=0,
            role="reset",
            active_edges=(SingleQubitEdge(gate="R", qubit=0),),
            active_qubits=frozenset({0}),
            idle_qubits=frozenset(),
        )
        cycle_step = ScheduleStep(
            tick_index=0,  # resets
            role="single_q",
            active_edges=(SingleQubitEdge(gate="H", qubit=0),),
            active_qubits=frozenset({0}),
            idle_qubits=frozenset(),
        )
        meas_step = ScheduleStep(
            tick_index=0,  # resets again
            role="meas",
            active_edges=(SingleQubitEdge(gate="M", qubit=0),),
            active_qubits=frozenset({0}),
            idle_qubits=frozenset(),
        )
        s = Schedule(
            head_steps=(reset_step,),
            cycle_steps=(cycle_step,),
            tail_steps=(meas_step,),
            qubits=frozenset({0}),
            qubit_roles={0: "data"},
        )
        assert s.cycle_depth == 1

    def test_rejects_step_qubit_not_in_schedule(self):
        with pytest.raises(ValueError, match="qubits not in schedule"):
            Schedule(
                head_steps=(),
                cycle_steps=(
                    ScheduleStep(
                        tick_index=0,
                        role="cnot_layer",
                        active_edges=(TwoQubitEdge(gate="CNOT", control=0, target=99),),
                        active_qubits=frozenset({0, 99}),  # 99 not in schedule
                        idle_qubits=frozenset(),
                    ),
                ),
                tail_steps=(),
                qubits=frozenset({0, 1}),
                qubit_roles={0: "data", 1: "z_ancilla"},
            )


class TestScheduleJson:
    def test_trivial_roundtrip(self):
        s = Schedule(
            head_steps=(),
            cycle_steps=(),
            tail_steps=(),
            qubits=frozenset({0, 1}),
            qubit_roles={0: "data", 1: "z_ancilla"},
            name="trivial",
        )
        restored = Schedule.from_json(s.to_json())
        assert restored == s

    def test_rich_schedule_roundtrip(self):
        s = Schedule(
            head_steps=(
                ScheduleStep(
                    tick_index=0,
                    role="reset",
                    active_edges=(
                        SingleQubitEdge(gate="R", qubit=0),
                        SingleQubitEdge(gate="R", qubit=1),
                    ),
                    active_qubits=frozenset({0, 1}),
                    idle_qubits=frozenset(),
                ),
            ),
            cycle_steps=(
                ScheduleStep(
                    tick_index=0,
                    role="cnot_layer",
                    active_edges=(
                        TwoQubitEdge(
                            gate="CNOT",
                            control=0,
                            target=1,
                            interaction_sector="X",
                            term_name="HZ[0,0]",
                        ),
                    ),
                    active_qubits=frozenset({0, 1}),
                    idle_qubits=frozenset(),
                ),
            ),
            tail_steps=(),
            qubits=frozenset({0, 1}),
            qubit_roles={0: "data", 1: "z_ancilla"},
            name="rich",
        )
        restored = Schedule.from_json(s.to_json())
        assert restored == s

    def test_rejects_wrong_type(self):
        with pytest.raises(ValueError, match="type='schedule'"):
            Schedule.from_json(
                {
                    "schema_version": 1,
                    "type": "embedding",
                    "head_steps": [],
                    "cycle_steps": [],
                    "tail_steps": [],
                    "qubits": [],
                    "qubit_roles": [],
                }
            )

    def test_rejects_wrong_schema_version(self):
        with pytest.raises(ValueError, match="schema_version"):
            Schedule.from_json(
                {
                    "schema_version": 999,
                    "type": "schedule",
                    "head_steps": [],
                    "cycle_steps": [],
                    "tail_steps": [],
                    "qubits": [],
                    "qubit_roles": [],
                }
            )


# =============================================================================
# default_css_schedule on Steane
# =============================================================================


@pytest.fixture
def steane_code():
    H = pcm.hamming(7)
    return CSSCode(HX=H, HZ=H, rounds=1)


class TestDefaultCssScheduleSteane:
    """The PR 4 flagship acceptance tests for `default_css_schedule`."""

    def test_structural_counts_z_memory(self, steane_code):
        """Steane z_memory: 1 head (R), 31 cycle (12 Z-CNOT + 3×6 X-bracket + 1 MR),
        1 tail (M)."""
        s = default_css_schedule(steane_code, experiment="z_memory")
        assert len(s.head_steps) == 1
        assert len(s.cycle_steps) == 31
        assert len(s.tail_steps) == 1

    def test_structural_counts_x_memory(self, steane_code):
        """Steane x_memory: 2 head (RX data, R ancilla), 31 cycle, 1 tail (MX)."""
        s = default_css_schedule(steane_code, experiment="x_memory")
        assert len(s.head_steps) == 2
        assert len(s.cycle_steps) == 31
        assert len(s.tail_steps) == 1

    def test_head_step_z_memory_is_all_R(self, steane_code):
        s = default_css_schedule(steane_code, experiment="z_memory")
        head = s.head_steps[0]
        assert head.role == "reset"
        assert len(head.active_edges) == 13  # 7 data + 3 z + 3 x
        assert all(isinstance(e, SingleQubitEdge) and e.gate == "R" for e in head.active_edges)

    def test_head_step_x_memory_uses_RX_for_data(self, steane_code):
        s = default_css_schedule(steane_code, experiment="x_memory")
        rx_step = s.head_steps[0]
        assert rx_step.role == "reset"
        assert all(isinstance(e, SingleQubitEdge) and e.gate == "RX" for e in rx_step.active_edges)
        assert len(rx_step.active_edges) == 7  # data qubits only

    def test_qubit_roles_partition(self, steane_code):
        s = default_css_schedule(steane_code)
        data_count = sum(1 for r in s.qubit_roles.values() if r == "data")
        z_anc = sum(1 for r in s.qubit_roles.values() if r == "z_ancilla")
        x_anc = sum(1 for r in s.qubit_roles.values() if r == "x_ancilla")
        assert data_count == 7
        assert z_anc == 3
        assert x_anc == 3

    def test_cycle_has_12_z_cnots(self, steane_code):
        """Hamming(7) has row weight 4; 3 Z-checks × 4 data = 12 Z-CNOTs."""
        s = default_css_schedule(steane_code)
        z_cnots = [
            step
            for step in s.cycle_steps
            if step.role == "cnot_layer"
            and len(step.active_edges) == 1
            and isinstance(step.active_edges[0], TwoQubitEdge)
            and step.active_edges[0].interaction_sector == "X"
        ]
        assert len(z_cnots) == 12

    def test_cycle_has_12_x_cnots(self, steane_code):
        s = default_css_schedule(steane_code)
        x_cnots = [
            step
            for step in s.cycle_steps
            if step.role == "cnot_layer"
            and len(step.active_edges) == 1
            and isinstance(step.active_edges[0], TwoQubitEdge)
            and step.active_edges[0].interaction_sector == "Z"
        ]
        assert len(x_cnots) == 12

    def test_cycle_has_6_h_gates(self, steane_code):
        """3 X-checks × 2 H gates (before/after bracket) = 6 H steps."""
        s = default_css_schedule(steane_code)
        h_steps = [
            step
            for step in s.cycle_steps
            if step.role == "single_q"
            and len(step.active_edges) == 1
            and isinstance(step.active_edges[0], SingleQubitEdge)
            and step.active_edges[0].gate == "H"
        ]
        assert len(h_steps) == 6

    def test_cycle_ends_with_mr(self, steane_code):
        s = default_css_schedule(steane_code)
        last_step = s.cycle_steps[-1]
        assert last_step.role == "meas"
        assert len(last_step.active_edges) == 6  # 3 z + 3 x ancillas
        assert all(
            isinstance(e, SingleQubitEdge) and e.gate == "MR" for e in last_step.active_edges
        )

    def test_tail_is_single_m_step(self, steane_code):
        s = default_css_schedule(steane_code, experiment="z_memory")
        tail = s.tail_steps[0]
        assert tail.role == "meas"
        assert len(tail.active_edges) == 7  # all data qubits
        assert all(isinstance(e, SingleQubitEdge) and e.gate == "M" for e in tail.active_edges)

    def test_tail_x_memory_uses_MX(self, steane_code):
        s = default_css_schedule(steane_code, experiment="x_memory")
        tail = s.tail_steps[0]
        assert all(isinstance(e, SingleQubitEdge) and e.gate == "MX" for e in tail.active_edges)

    def test_idle_qubits_are_disjoint_from_active_every_step(self, steane_code):
        s = default_css_schedule(steane_code)
        for step in s.all_steps():
            assert step.active_qubits & step.idle_qubits == frozenset()

    def test_every_step_covers_all_qubits(self, steane_code):
        """Every step's active ∪ idle covers all schedule qubits."""
        s = default_css_schedule(steane_code)
        for step in s.all_steps():
            assert step.active_qubits | step.idle_qubits == s.qubits

    def test_schedule_json_roundtrip_on_steane(self, steane_code):
        """PR 4 flagship: Steane default schedule round-trips through JSON."""
        s = default_css_schedule(steane_code, name="steane_default_z")
        data = s.to_json()
        # Survives json.dumps/loads.
        serialized = json.loads(json.dumps(data))
        restored = Schedule.from_json(serialized)
        assert restored == s

    def test_z_cnot_term_names_match_HZ_indices(self, steane_code):
        """Each Z-CNOT's `term_name` identifies the HZ entry."""
        s = default_css_schedule(steane_code)
        z_cnot_edges = [
            step.active_edges[0]
            for step in s.cycle_steps
            if step.role == "cnot_layer"
            and isinstance(step.active_edges[0], TwoQubitEdge)
            and step.active_edges[0].interaction_sector == "X"
        ]
        # 12 Z-CNOTs, term names of the form HZ[row, col]
        term_names = {e.term_name for e in z_cnot_edges}  # type: ignore[union-attr]
        assert len(term_names) == 12
        assert all(name is not None and name.startswith("HZ[") for name in term_names)

    def test_cnot_directions_z_check_data_is_control(self, steane_code):
        """For Z-checks, data is the control, ancilla is the target."""
        s = default_css_schedule(steane_code)
        data_set = set(steane_code.data_qubits)
        z_check_set = set(steane_code.z_check_qubits)
        for step in s.cycle_steps:
            if (
                step.role == "cnot_layer"
                and isinstance(step.active_edges[0], TwoQubitEdge)
                and step.active_edges[0].interaction_sector == "X"
            ):
                e = step.active_edges[0]
                assert e.control in data_set
                assert e.target in z_check_set

    def test_cnot_directions_x_check_ancilla_is_control(self, steane_code):
        """For X-checks, ancilla is the control, data is the target."""
        s = default_css_schedule(steane_code)
        data_set = set(steane_code.data_qubits)
        x_check_set = set(steane_code.x_check_qubits)
        for step in s.cycle_steps:
            if (
                step.role == "cnot_layer"
                and isinstance(step.active_edges[0], TwoQubitEdge)
                and step.active_edges[0].interaction_sector == "Z"
            ):
                e = step.active_edges[0]
                assert e.control in x_check_set
                assert e.target in data_set

    def test_invalid_experiment_rejected(self, steane_code):
        with pytest.raises(ValueError, match="experiment must be"):
            default_css_schedule(steane_code, experiment="magic")  # type: ignore[arg-type]

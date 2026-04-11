"""Schedule walker tests for the fault propagator.

These tests pin down three guarantees of
:func:`weave.analysis.propagation.propagate_fault`:

1. **Ancilla elimination.** On a well-formed CSS schedule, any fault
   reaches the final data measurement with zero residual on ancilla
   qubits — every ancilla is measured and reset at the end of the
   cycle and at the end of the tail.
2. **Sector-correct pair building.** `build_single_pair_fault` places
   X's on the control qubits in the X sector and Z's on the target
   qubits in the Z sector, matching the microscopic-block identification
   in the PRX-Quantum-under-review paper's §II.D.
3. **Propagation through a CNOT chain.** A known fault (X on a data
   qubit that participates in a Z-check) flips the check it runs into
   and propagates no farther. We check the flip is recorded and the
   data residual has weight 1.

The reference schedule is `default_css_schedule` applied to the
[[7, 1, 3]] Steane code, for which the hand-checked expectations below
are mathematically unambiguous.
"""

from __future__ import annotations

import pytest

from weave.analysis.pauli import Pauli
from weave.analysis.propagation import (
    AncillaFlip,
    FaultLocation,
    PropagationResult,
    build_single_pair_fault,
    propagate_fault,
    propagate_single_pair_event,
)
from weave.codes.css_code import CSSCode
from weave.ir import (
    Schedule,
    ScheduleStep,
    TwoQubitEdge,
    default_css_schedule,
)
from weave.util import pcm

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def steane_schedule() -> Schedule:
    """`default_css_schedule` applied to the Steane [[7,1,3]] code."""
    H = pcm.hamming(7)
    code = CSSCode(HX=H, HZ=H, rounds=1)
    return default_css_schedule(code)


@pytest.fixture
def steane_data_qubits(steane_schedule: Schedule) -> frozenset[int]:
    return frozenset(q for q, role in steane_schedule.qubit_roles.items() if role == "data")


@pytest.fixture
def steane_num_qubits(steane_schedule: Schedule) -> int:
    return max(steane_schedule.qubits) + 1


# ---------------------------------------------------------------------------
# Ancilla elimination
# ---------------------------------------------------------------------------


class TestAncillaElimination:
    def test_zero_fault_stays_zero(self, steane_schedule, steane_data_qubits, steane_num_qubits):
        """An identity fault propagates to identity, with no ancilla flips."""
        initial = Pauli.identity(steane_num_qubits)
        result = propagate_fault(
            schedule=steane_schedule,
            initial_fault=initial,
            injection_location=FaultLocation(block="cycle", tick_index=0),
            data_qubits=steane_data_qubits,
        )
        assert result.data_weight == 0
        assert result.full_pauli.is_identity()
        assert result.ancilla_flips == ()

    def test_data_fault_leaves_no_ancilla_residual(
        self, steane_schedule, steane_data_qubits, steane_num_qubits
    ):
        """Any data-only fault ends with zero weight on ancillas.

        After one full cycle every ancilla has been measured and reset
        (the cycle ends with `MR` on all ancillas), so the fault must
        be fully localized on data qubits at the end of the cycle.
        We stop walking at the end of the cycle to preserve the data
        residual for inspection — the tail would measure data qubits
        and clear them.
        """
        for data_q in sorted(steane_data_qubits):
            initial = Pauli.single_x(data_q, steane_num_qubits)
            result = propagate_fault(
                schedule=steane_schedule,
                initial_fault=initial,
                injection_location=FaultLocation(block="head", tick_index=0),
                data_qubits=steane_data_qubits,
                end_block="cycle",
            )
            # No residual on ancilla qubits.
            ancillas = set(steane_schedule.qubits) - set(steane_data_qubits)
            for q in ancillas:
                assert result.full_pauli.pauli_on(q) == "I", (
                    f"residual on ancilla {q} after propagating X_{data_q}"
                )
            # The data X itself survives on `data_q` (no data meas yet).
            assert result.data_pauli.support == frozenset({data_q})

    def test_ancilla_x_fault_is_absorbed_by_measure_reset(
        self, steane_schedule, steane_data_qubits, steane_num_qubits
    ):
        """An X fault on a Z-ancilla flips its measurement and then dies.

        Z-ancillas are measured in the Z basis at the end of the cycle.
        An X on a Z-ancilla anticommutes with Z, flips the record, and
        is absorbed by the measure-and-reset.
        """
        z_ancilla = next(
            q for q, role in steane_schedule.qubit_roles.items() if role == "z_ancilla"
        )
        initial = Pauli.single_x(z_ancilla, steane_num_qubits)
        result = propagate_fault(
            schedule=steane_schedule,
            initial_fault=initial,
            injection_location=FaultLocation(block="cycle", tick_index=0),
            data_qubits=steane_data_qubits,
            end_block="cycle",
        )
        # The measurement on that ancilla was flipped.
        assert any(flip.qubit == z_ancilla and flip.basis == "Z" for flip in result.ancilla_flips)
        # No residual on ancillas.
        ancillas = set(steane_schedule.qubits) - set(steane_data_qubits)
        for q in ancillas:
            assert result.full_pauli.pauli_on(q) == "I"


# ---------------------------------------------------------------------------
# build_single_pair_fault
# ---------------------------------------------------------------------------


class TestBuildSinglePairFault:
    def test_x_sector_places_x_on_controls(self):
        edge_a = TwoQubitEdge(gate="CNOT", control=0, target=4, interaction_sector="X")
        edge_b = TwoQubitEdge(gate="CNOT", control=1, target=5, interaction_sector="X")
        p = build_single_pair_fault(edge_a, edge_b, "X", num_qubits=6)
        assert p.support == frozenset({0, 1})
        assert p.x_support == frozenset({0, 1})
        assert p.z_support == frozenset()

    def test_z_sector_places_z_on_targets(self):
        edge_a = TwoQubitEdge(gate="CNOT", control=4, target=0, interaction_sector="Z")
        edge_b = TwoQubitEdge(gate="CNOT", control=5, target=1, interaction_sector="Z")
        p = build_single_pair_fault(edge_a, edge_b, "Z", num_qubits=6)
        assert p.support == frozenset({0, 1})
        assert p.z_support == frozenset({0, 1})
        assert p.x_support == frozenset()

    def test_shared_qubit_rejected(self):
        edge_a = TwoQubitEdge(gate="CNOT", control=0, target=4)
        edge_b = TwoQubitEdge(gate="CNOT", control=0, target=5)
        with pytest.raises(ValueError, match="disjoint"):
            build_single_pair_fault(edge_a, edge_b, "X", num_qubits=6)

    def test_sector_mismatch_rejected(self):
        edge_a = TwoQubitEdge(gate="CNOT", control=0, target=4, interaction_sector="X")
        edge_b = TwoQubitEdge(gate="CNOT", control=1, target=5, interaction_sector="Z")
        with pytest.raises(ValueError, match="interaction_sector"):
            build_single_pair_fault(edge_a, edge_b, "X", num_qubits=6)


# ---------------------------------------------------------------------------
# End-to-end propagation through a CNOT chain
# ---------------------------------------------------------------------------


class TestPropagationEndToEnd:
    def test_x_fault_flips_all_zchecks_that_touch_qubit(
        self, steane_schedule, steane_data_qubits, steane_num_qubits
    ):
        """An X on data qubit `q` flips exactly the Z-checks that touch `q`.

        In `default_css_schedule`, each Z-check is built via a sequence
        of CNOTs with `control = data_q` and `target = z_ancilla`, all
        in the X sector. Propagating an X_q through CNOT(q, z_anc)
        flips Z_{z_anc} (by `CNOT: X_c -> X_c X_t`), which anticommutes
        with the ancilla's terminal measurement. So the set of flipped
        Z-ancilla measurements equals exactly the Z-checks that act on
        qubit `q`. We stop the walker at the end of the cycle so the
        data X on `data_q` survives (no tail measurement clearing it).
        """
        H = pcm.hamming(7)
        code = CSSCode(HX=H, HZ=H, rounds=1)
        z_ancilla_set = set(code.z_check_qubits)

        for col, data_q in enumerate(code.data_qubits):
            initial = Pauli.single_x(data_q, steane_num_qubits)
            result = propagate_fault(
                schedule=steane_schedule,
                initial_fault=initial,
                injection_location=FaultLocation(block="head", tick_index=0),
                data_qubits=steane_data_qubits,
                end_block="cycle",
            )

            # Expected Z-checks: those with HZ[:, col] == 1.
            expected_rows = {i for i in range(H.shape[0]) if H[i, col]}
            expected_zancillas = {code.z_check_qubits[i] for i in expected_rows}
            got_zancillas = {
                flip.qubit
                for flip in result.ancilla_flips
                if flip.basis == "Z" and flip.qubit in z_ancilla_set
            }
            assert got_zancillas == expected_zancillas, (
                f"X on data qubit {data_q}: expected Z-flips "
                f"{expected_zancillas}, got {got_zancillas}"
            )

            # And the data residual still has weight 1 on `data_q`.
            assert result.data_pauli.support == frozenset({data_q})
            assert result.data_pauli.pauli_on(data_q) == "X"

    def test_propagate_single_pair_event_returns_full_result(
        self, steane_schedule, steane_data_qubits
    ):
        """The single-pair wrapper routes to propagate_fault correctly."""
        # Steane default_css_schedule has one CNOT per tick, so there
        # is no actual "pair" tick — but we can still call the helper
        # on two arbitrary disjoint edges at tick 0.
        edge_a = TwoQubitEdge(gate="CNOT", control=0, target=7, interaction_sector="X")
        edge_b = TwoQubitEdge(gate="CNOT", control=1, target=8, interaction_sector="X")
        result = propagate_single_pair_event(
            schedule=steane_schedule,
            step_tick=0,
            edge_a=edge_a,
            edge_b=edge_b,
            sector="X",
            data_qubits=steane_data_qubits,
        )
        assert isinstance(result, PropagationResult)
        assert result.initial_fault.weight == 2


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestPropagationErrors:
    def test_wrong_num_qubits_raises(self, steane_schedule, steane_data_qubits):
        bad = Pauli.identity(3)  # schedule has 13 qubits.
        with pytest.raises(ValueError, match="qubits"):
            propagate_fault(
                schedule=steane_schedule,
                initial_fault=bad,
                injection_location=FaultLocation(block="cycle", tick_index=0),
                data_qubits=steane_data_qubits,
            )

    def test_unsupported_gate_raises(self, steane_data_qubits):
        """A CZ gate in a cnot_layer step is rejected at propagation time.

        The propagator only implements CNOT for two-qubit Cliffords at
        present; CZ would need separate propagation rules. This test
        documents that limitation rather than hiding it.
        """
        qubits = frozenset({0, 1})
        step = ScheduleStep(
            tick_index=0,
            role="cnot_layer",
            active_edges=(TwoQubitEdge(gate="CZ", control=0, target=1),),
            active_qubits=qubits,
            idle_qubits=frozenset(),
        )
        sched = Schedule(
            head_steps=(),
            cycle_steps=(step,),
            tail_steps=(),
            qubits=qubits,
            qubit_roles={0: "data", 1: "data"},
            name="cz_only",
        )
        with pytest.raises(NotImplementedError, match="CZ"):
            propagate_fault(
                schedule=sched,
                initial_fault=Pauli.single_x(0, 2),
                injection_location=FaultLocation(block="cycle", tick_index=0),
                data_qubits=frozenset({0, 1}),
            )


# ---------------------------------------------------------------------------
# AncillaFlip record type
# ---------------------------------------------------------------------------


def test_ancilla_flip_is_hashable():
    """`AncillaFlip` is frozen and can be stored in sets."""
    f1 = AncillaFlip(block="cycle", tick_index=3, qubit=7, basis="Z")
    f2 = AncillaFlip(block="cycle", tick_index=3, qubit=7, basis="Z")
    assert f1 == f2
    assert hash(f1) == hash(f2)
    assert len({f1, f2}) == 1

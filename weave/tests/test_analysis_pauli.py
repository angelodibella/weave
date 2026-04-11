"""Algebraic tests for the symplectic Pauli primitive.

These tests pin down the mathematical guarantees the rest of the
propagator depends on:

- The symplectic representation identifies Paulis mod phases, so
  Pauli products are XORs and commutation is the symplectic inner
  product over F_2.
- Clifford conjugation rules match Gottesman (1997) Table 3.1 exactly.
- Clifford conjugation preserves the Pauli group (closure), the
  weight of any identity operator is zero, and measurements flip
  outcomes iff the fault anticommutes with the measured Pauli.

We do not test the phase of the conjugated Pauli because the
representation is deliberately phase-free; see the module docstring.
"""

from __future__ import annotations

import itertools

import pytest

from weave.analysis.pauli import (
    Pauli,
    measure_x,
    measure_z,
    propagate_cnot,
    propagate_h,
    propagate_i,
    propagate_s,
    propagate_x,
    propagate_y,
    propagate_z,
)

# ---------------------------------------------------------------------------
# Constructors and basic properties
# ---------------------------------------------------------------------------


class TestPauliConstruction:
    def test_identity_has_zero_weight(self):
        p = Pauli.identity(5)
        assert p.num_qubits == 5
        assert p.weight == 0
        assert p.is_identity()
        assert p.support == frozenset()

    def test_single_x_has_unit_weight_on_qubit(self):
        p = Pauli.single_x(2, 5)
        assert p.num_qubits == 5
        assert p.weight == 1
        assert p.support == frozenset({2})
        assert p.x_support == frozenset({2})
        assert p.z_support == frozenset()
        assert p.pauli_on(2) == "X"
        assert p.pauli_on(0) == "I"

    def test_single_y_is_x_times_z(self):
        py = Pauli.single_y(1, 3)
        product = Pauli.single_x(1, 3) * Pauli.single_z(1, 3)
        assert py == product
        assert py.pauli_on(1) == "Y"

    def test_from_string_round_trips(self):
        for s in ["I", "X", "Y", "Z", "IXZY", "XXZZYY"]:
            p = Pauli.from_string(s)
            # pauli_on reconstructs the string.
            reconstructed = "".join(p.pauli_on(i) for i in range(len(s)))
            assert reconstructed == s

    def test_from_string_rejects_garbage(self):
        with pytest.raises(ValueError, match="Unknown Pauli symbol"):
            Pauli.from_string("XxZ")

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            Pauli(x=(True, False), z=(False,))


# ---------------------------------------------------------------------------
# Multiplication and commutation
# ---------------------------------------------------------------------------


class TestPauliAlgebra:
    def test_product_is_xor(self):
        # X * Y = Z (up to phase).
        assert Pauli.from_string("X") * Pauli.from_string("Y") == Pauli.from_string("Z")
        # Y * Z = X.
        assert Pauli.from_string("Y") * Pauli.from_string("Z") == Pauli.from_string("X")
        # X * X = I (involution).
        assert Pauli.from_string("X") * Pauli.from_string("X") == Pauli.from_string("I")

    def test_tensor_product_multiplies_componentwise(self):
        p = Pauli.from_string("XI") * Pauli.from_string("IZ")
        assert p == Pauli.from_string("XZ")

    def test_commutation_same_pauli(self):
        # Any Pauli commutes with itself.
        for s in ["I", "X", "Y", "Z", "XZ", "YXZI"]:
            p = Pauli.from_string(s)
            assert p.commutes_with(p)
            assert not p.anticommutes_with(p)

    def test_single_qubit_anticommutation_table(self):
        # X and Z anticommute; X and Y anticommute; Y and Z anticommute.
        x = Pauli.from_string("X")
        y = Pauli.from_string("Y")
        z = Pauli.from_string("Z")
        assert x.anticommutes_with(z)
        assert x.anticommutes_with(y)
        assert y.anticommutes_with(z)
        # Commutation is symmetric.
        assert z.anticommutes_with(x)

    def test_different_qubit_paulis_commute(self):
        p = Pauli.from_string("XI")
        q = Pauli.from_string("IZ")
        assert p.commutes_with(q)

    def test_tensor_parity_determines_commutation(self):
        # XX and ZZ on two qubits: two anticommuting single-qubit pairs
        # cancel to commuting overall.
        xx = Pauli.from_string("XX")
        zz = Pauli.from_string("ZZ")
        assert xx.commutes_with(zz)
        # XZ and ZX also commute (cross terms cancel).
        assert Pauli.from_string("XZ").commutes_with(Pauli.from_string("ZX"))
        # XZ and XX anticommute (one cancellation, one mismatch).
        assert Pauli.from_string("XZ").anticommutes_with(Pauli.from_string("XX"))

    def test_multiply_mismatched_sizes_raises(self):
        with pytest.raises(ValueError, match="different qubit counts"):
            Pauli.from_string("X") * Pauli.from_string("XI")


# ---------------------------------------------------------------------------
# Clifford propagation rules (Gottesman 1997, Table 3.1)
# ---------------------------------------------------------------------------


class TestCliffordPropagation:
    def test_h_swaps_x_and_z(self):
        # H X H = Z.
        assert propagate_h(Pauli.from_string("X"), 0) == Pauli.from_string("Z")
        # H Z H = X.
        assert propagate_h(Pauli.from_string("Z"), 0) == Pauli.from_string("X")
        # H Y H = -Y (phase-free: Y).
        assert propagate_h(Pauli.from_string("Y"), 0) == Pauli.from_string("Y")
        # H I H = I.
        assert propagate_h(Pauli.from_string("I"), 0) == Pauli.from_string("I")

    def test_h_is_involution(self):
        for s in ["I", "X", "Y", "Z"]:
            p = Pauli.from_string(s)
            assert propagate_h(propagate_h(p, 0), 0) == p

    def test_s_sends_x_to_y(self):
        # S X S† = Y.
        assert propagate_s(Pauli.from_string("X"), 0) == Pauli.from_string("Y")
        # S Z S† = Z.
        assert propagate_s(Pauli.from_string("Z"), 0) == Pauli.from_string("Z")
        # S Y S† = -X (phase-free: X).
        assert propagate_s(Pauli.from_string("Y"), 0) == Pauli.from_string("X")
        # S I S† = I.
        assert propagate_s(Pauli.from_string("I"), 0) == Pauli.from_string("I")

    def test_cnot_x_control_spreads_to_target(self):
        # CNOT (control=0, target=1) sends X_0 to X_0 X_1.
        p = Pauli.from_string("XI")
        out = propagate_cnot(p, control=0, target=1)
        assert out == Pauli.from_string("XX")

    def test_cnot_z_target_spreads_to_control(self):
        # CNOT sends Z_1 to Z_0 Z_1.
        p = Pauli.from_string("IZ")
        out = propagate_cnot(p, control=0, target=1)
        assert out == Pauli.from_string("ZZ")

    def test_cnot_leaves_x_target_alone(self):
        p = Pauli.from_string("IX")
        out = propagate_cnot(p, control=0, target=1)
        assert out == Pauli.from_string("IX")

    def test_cnot_leaves_z_control_alone(self):
        p = Pauli.from_string("ZI")
        out = propagate_cnot(p, control=0, target=1)
        assert out == Pauli.from_string("ZI")

    def test_cnot_y_control_becomes_y_x(self):
        # Y = X*Z (phase-free). CNOT: X_0 -> X_0 X_1, Z_0 unchanged.
        # So Y_0 = X_0 Z_0 -> X_0 X_1 Z_0 = Y_0 X_1.
        p = Pauli.from_string("YI")
        out = propagate_cnot(p, control=0, target=1)
        assert out == Pauli.from_string("YX")

    def test_cnot_is_involution(self):
        # CNOT^2 = I.
        for bits in itertools.product(["I", "X", "Y", "Z"], repeat=2):
            p = Pauli.from_string("".join(bits))
            twice = propagate_cnot(propagate_cnot(p, 0, 1), 0, 1)
            assert twice == p

    def test_cnot_rejects_same_control_target(self):
        with pytest.raises(ValueError, match="must differ"):
            propagate_cnot(Pauli.from_string("XX"), control=1, target=1)

    def test_pauli_gate_propagation_is_identity(self):
        # X, Y, Z, I gates are phase-free no-ops on a fault.
        for s in ["IX", "XY", "ZZ", "YI"]:
            p = Pauli.from_string(s)
            assert propagate_x(p, 0) == p
            assert propagate_y(p, 1) == p
            assert propagate_z(p, 0) == p
            assert propagate_i(p, 1) == p


# ---------------------------------------------------------------------------
# Measurements / ancilla elimination
# ---------------------------------------------------------------------------


class TestMeasurements:
    def test_measure_z_detects_x_fault(self):
        p = Pauli.single_x(0, 2)
        flipped, reduced = measure_z(p, 0)
        assert flipped is True
        # The X component on qubit 0 is cleared.
        assert reduced == Pauli.identity(2)

    def test_measure_z_ignores_z_fault(self):
        p = Pauli.single_z(0, 2)
        flipped, reduced = measure_z(p, 0)
        assert flipped is False
        assert reduced == Pauli.identity(2)

    def test_measure_x_detects_z_fault(self):
        p = Pauli.single_z(0, 2)
        flipped, reduced = measure_x(p, 0)
        assert flipped is True
        assert reduced == Pauli.identity(2)

    def test_measure_leaves_other_qubits_untouched(self):
        # Fault on qubit 1 should survive measurement of qubit 0.
        p = Pauli.from_string("IX")
        flipped, reduced = measure_z(p, 0)
        assert flipped is False
        assert reduced == Pauli.from_string("IX")

    def test_measure_z_y_fault_flips(self):
        # Y has an X component, so measure_z sees an anticommutation.
        p = Pauli.single_y(0, 1)
        flipped, reduced = measure_z(p, 0)
        assert flipped is True
        assert reduced == Pauli.identity(1)

    def test_measure_out_of_range_raises(self):
        with pytest.raises(IndexError):
            measure_z(Pauli.identity(2), 5)

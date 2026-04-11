"""Tests for residual-error enumeration and the Strikis Δ(E) functional.

Every numeric expectation in this file is hand-verified against the
Steane [[7, 1, 3]] code (via `pcm.hamming(7)`), for which the answers
are unambiguous:

* The code has quantum distance 3. So for the trivial residual `E = 0`,
  Strikis Definition 1 gives `Δ(0) = 1 + 3 = 4`.
* A nontrivial weight-3 logical `L` is itself a valid witness for
  Δ via the `D = 0` branch; `Δ(L) = 1 + 0 = 1`.
* Any stabilizer residual is in the trivial coset, so the minimum
  logical weight modulo stabilizers is still 3, and `Δ(S) = 4`.

The hook-residual enumerator is a direct implementation of a
combinatorial definition and is checked by enumeration.

References
----------
- A. Strikis, D. E. Browne, M. E. Beverland, arXiv:2603.05481, §IV.1,
  Definitions 1 and 3.
"""

from __future__ import annotations

import numpy as np
import pytest

from weave.analysis.pauli import Pauli
from weave.analysis.residual import (
    ResidualError,
    effective_distance_upper_bound,
    enumerate_hook_residuals_z_sector,
    residual_distance,
)
from weave.codes.css_code import CSSCode
from weave.util import pcm

# ---------------------------------------------------------------------------
# ResidualError dataclass
# ---------------------------------------------------------------------------


class TestResidualError:
    def test_basic_construction(self):
        r = ResidualError(data_support=(1, 3, 5), weight=3, label="test")
        assert r.data_support == (1, 3, 5)
        assert r.weight == 3
        assert r.label == "test"
        assert r.flipped_ancilla_ticks == ()

    def test_support_sorted_automatically(self):
        r = ResidualError(data_support=(5, 1, 3), weight=3)
        assert r.data_support == (1, 3, 5)

    def test_weight_mismatch_raises(self):
        with pytest.raises(ValueError, match="weight"):
            ResidualError(data_support=(0, 1), weight=3)

    def test_from_pauli_intersects_with_data_qubits(self):
        """`from_pauli` drops ancilla-qubit components of the fault."""
        p = Pauli.from_string("XIXII")  # support {0, 2}
        data = frozenset({0, 1, 2})
        r = ResidualError.from_pauli(p, data_qubits=data, label="hook")
        # Support intersected with data = {0, 2}.
        assert r.data_support == (0, 2)
        assert r.weight == 2
        assert r.label == "hook"

    def test_from_pauli_drops_ancilla_components(self):
        # Qubit 3 is an ancilla — must not appear in data_support.
        p = Pauli.from_string("XIZXI")  # support {0, 2, 3}
        data = frozenset({0, 1, 2})
        r = ResidualError.from_pauli(p, data_qubits=data)
        assert r.data_support == (0, 2)

    def test_as_binary_vector(self):
        r = ResidualError(data_support=(1, 3), weight=2)
        vec = r.as_binary_vector(num_data_qubits=5)
        np.testing.assert_array_equal(vec, np.array([0, 1, 0, 1, 0], dtype=np.uint8))


# ---------------------------------------------------------------------------
# residual_distance — Strikis Definition 1
# ---------------------------------------------------------------------------


class TestResidualDistanceSteane:
    """Δ(E) for the Steane [[7, 1, 3]] code."""

    @pytest.fixture
    def HX(self):
        return pcm.hamming(7)

    @pytest.fixture
    def HZ(self):
        return pcm.hamming(7)

    def test_empty_residual_equals_distance_plus_one(self, HX, HZ):
        """Δ(0) = 1 + d_Steane = 4."""
        empty = ResidualError(data_support=(), weight=0, label="empty")
        # X-sector: h_commute = HZ, h_stab = HX.
        delta = residual_distance(empty, h_commute=HZ, h_stab=HX)
        assert delta == 4

    def test_nontrivial_logical_residual_has_delta_1(self, HX, HZ):
        """A weight-3 X-logical residual `L` has `L + L = 0`, so D = 0
        is a valid (logical) completion via u = L itself, giving
        Δ(L) = 1 + 0 = 1."""
        # Find a nontrivial X-logical as a ker(HZ) vector outside span(HX).
        # We know the Steane has logicals of weight 3 — e.g. [1,1,1,0,0,0,0]
        # corresponds to an X-logical for the standard Hamming(7)
        # convention if it is in the nullspace. We verify below.
        code = CSSCode(HX=HX, HZ=HZ)
        x_logicals, _ = code.find_logicals()
        # Pick the first X-logical.
        lx = x_logicals[0]
        support = tuple(int(i) for i in np.nonzero(lx)[0])
        residual = ResidualError(data_support=support, weight=len(support), label="L_X")
        delta = residual_distance(residual, h_commute=HZ, h_stab=HX)
        assert delta == 1

    def test_stabilizer_residual_matches_empty(self, HX, HZ):
        """A stabilizer residual is in the trivial coset, so its
        Δ equals Δ(0) = 4. This holds because the ambient
        nontrivial-logical minimum is insensitive to multiplication
        by stabilizers."""
        # Use the first row of HX as a stabilizer; it's in span(HX) by
        # definition and in ker(HZ) by the CSS condition.
        stab_row = HX[0]
        support = tuple(int(i) for i in np.nonzero(stab_row)[0])
        residual = ResidualError(data_support=support, weight=len(support), label="S")
        # Sanity check: the stabilizer IS in span(HX).
        delta = residual_distance(residual, h_commute=HZ, h_stab=HX)
        # For a stabilizer: we need u ∈ ker(HZ) \ span(HX). u + stab is
        # also a nontrivial logical with the same logical class.
        # Minimum |u + stab| = Steane distance = 3, so Δ = 4.
        assert delta == 4

    def test_returns_inf_when_no_logicals(self):
        """Codes with no logicals (trivial nullspace mod stabilizer)
        give Δ = ∞ for the empty residual."""
        # A code with HX = HZ = identity has ker(HZ) = 0, so there's
        # no logical operator at all.
        n = 3
        H = np.eye(n, dtype=int)
        empty = ResidualError(data_support=(), weight=0)
        delta = residual_distance(empty, h_commute=H, h_stab=H)
        assert delta == float("inf")

    def test_shape_mismatch_raises(self):
        empty = ResidualError(data_support=(), weight=0)
        with pytest.raises(ValueError, match="columns"):
            residual_distance(
                empty,
                h_commute=np.zeros((1, 5), dtype=int),
                h_stab=np.zeros((1, 4), dtype=int),
            )

    def test_k_guard_rejects_huge_nullspace(self):
        """k_guard prevents `O(2^k)` enumeration on huge codes."""
        n = 25
        H = np.zeros((1, n), dtype=int)  # trivial check, dim ker = 25
        empty = ResidualError(data_support=(), weight=0)
        with pytest.raises(ValueError, match="infeasible"):
            residual_distance(empty, h_commute=H, h_stab=H, k_guard=20)


# ---------------------------------------------------------------------------
# effective_distance_upper_bound
# ---------------------------------------------------------------------------


class TestEffectiveDistanceUpperBound:
    def test_empty_residual_set_returns_inf(self):
        """No residuals → nothing to minimize → ∞."""
        H = pcm.hamming(7)
        result = effective_distance_upper_bound([], h_commute=H, h_stab=H)
        assert result == float("inf")

    def test_single_residual_equals_residual_distance(self):
        H = pcm.hamming(7)
        empty = ResidualError(data_support=(), weight=0)
        single = effective_distance_upper_bound([empty], h_commute=H, h_stab=H)
        assert single == residual_distance(empty, h_commute=H, h_stab=H)

    def test_min_over_residual_set(self):
        """The bound is the minimum Δ over the residuals.

        We construct a set containing (i) the empty residual, which
        has Δ = 4 for Steane, and (ii) a Steane X-logical, which has
        Δ = 1. The min is 1.
        """
        H = pcm.hamming(7)
        code = CSSCode(HX=H, HZ=H)
        x_logicals, _ = code.find_logicals()
        lx = x_logicals[0]
        support = tuple(int(i) for i in np.nonzero(lx)[0])

        residuals = [
            ResidualError(data_support=(), weight=0, label="empty"),
            ResidualError(data_support=support, weight=len(support), label="L_X"),
        ]
        result = effective_distance_upper_bound(residuals, h_commute=H, h_stab=H)
        assert result == 1


# ---------------------------------------------------------------------------
# enumerate_hook_residuals_z_sector
# ---------------------------------------------------------------------------


class TestHookResidualEnumeration:
    def test_enumeration_matches_definition(self):
        """For targets `[i_1, ..., i_w]`, residual `ell` has support
        `{i_ell, ..., i_w}`, enumerated for `ell = 2, ..., w`."""
        targets = [10, 11, 12, 13, 14]
        residuals = enumerate_hook_residuals_z_sector(targets, label_prefix="bbtest")
        # 4 residuals (ell from 2 to 5).
        assert len(residuals) == 4
        # ell=2: {11, 12, 13, 14}
        assert residuals[0].data_support == (11, 12, 13, 14)
        assert residuals[0].weight == 4
        # ell=5: {14}
        assert residuals[-1].data_support == (14,)
        assert residuals[-1].weight == 1

    def test_labels_are_prefixed(self):
        residuals = enumerate_hook_residuals_z_sector([0, 1, 2, 3], label_prefix="BB")
        # ell = 2, 3, 4.
        labels = [r.label for r in residuals]
        assert labels == ["BB_2", "BB_3", "BB_4"]

    def test_single_target_empty(self):
        """A one-CNOT check has no hook positions (ell starts at 2)."""
        residuals = enumerate_hook_residuals_z_sector([0])
        assert residuals == []

    def test_two_targets_has_one_residual(self):
        residuals = enumerate_hook_residuals_z_sector([0, 1])
        assert len(residuals) == 1
        assert residuals[0].data_support == (1,)

    def test_descending_weight_sequence(self):
        residuals = enumerate_hook_residuals_z_sector([100, 101, 102, 103, 104, 105])
        weights = [r.weight for r in residuals]
        # ell = 2..6: weights 5, 4, 3, 2, 1.
        assert weights == [5, 4, 3, 2, 1]

"""Tests for the BB code family and pure-L logical enumeration.

Every parameter assertion is pinned against Bravyi et al., *High-
threshold and low-overhead fault-tolerant quantum memory*, Nature
**627**, 778 (2024), arXiv:2308.07915, Table I. The pure-L
enumeration is checked against the workbook support in the plan
document: for BB72, `{3, 12, 21, 24, 27, 33}` must appear among the
36 minimum-weight pure-L Z-logical representatives.
"""

from __future__ import annotations

import numpy as np
import pytest

from weave.codes.bb import (
    BivariateBicycleCode,
    build_bb72,
    build_bb90,
    build_bb108,
    build_bb144,
    enumerate_pure_L_minwt_logicals,
)
from weave.codes.bb.algebra import (
    ker_A_basis,
    ker_BT_basis,
    pure_L_stabilizer_basis,
)
from weave.codes.css_code import CSSCode

# ---------------------------------------------------------------------------
# [[n, k, d]] parameters
# ---------------------------------------------------------------------------


class TestBB72:
    """Plan acceptance test #1: BB72 has n=72, k=12, distance=6."""

    def test_n_and_k(self):
        bb = build_bb72()
        assert bb.n == 72
        assert bb.k == 12

    def test_distance(self):
        bb = build_bb72()
        assert bb.distance() == 6

    def test_name_and_block_size(self):
        bb = build_bb72()
        assert bb.name == "BB72"
        assert bb.l == 6
        assert bb.m == 6
        assert bb.block_size == 36

    def test_css_condition(self):
        bb = build_bb72()
        assert np.all((bb.HX @ bb.HZ.T) % 2 == 0)

    def test_inherits_csscode(self):
        bb = build_bb72()
        assert isinstance(bb, CSSCode)


class TestBB90:
    def test_parameters(self):
        bb = build_bb90()
        assert bb.n == 90
        assert bb.k == 8
        assert bb.distance() == 10
        assert bb.l == 15
        assert bb.m == 3


class TestBB108:
    """Plan acceptance test #3: BB108 distance.

    Note: the published BB108 code (Bravyi et al. 2024, Table I row
    3) has distance 10, not 12. The plan text says 12, which we
    read as a typo — we pin the assertion to the Bravyi value and
    keep a `known_distance=10` in the factory.
    """

    def test_parameters(self):
        bb = build_bb108()
        assert bb.n == 108
        assert bb.k == 8
        assert bb.distance() == 10
        assert bb.l == 9
        assert bb.m == 6


class TestBB144:
    def test_parameters(self):
        bb = build_bb144()
        assert bb.n == 144
        assert bb.k == 12
        assert bb.distance() == 12
        assert bb.l == 12
        assert bb.m == 6


# ---------------------------------------------------------------------------
# Flat-index round-trip
# ---------------------------------------------------------------------------


class TestFlatIndexing:
    def test_flat_and_unflat_invert(self):
        """`unflat_index(flat_index(i, j, block)) == (i, j, block)`
        for every group element in both blocks."""
        bb = build_bb72()
        for block in ("L", "R"):
            for i in range(bb.l):
                for j in range(bb.m):
                    q = bb.flat_index(i, j, block)
                    assert bb.unflat_index(q) == (i, j, block)

    def test_block_ranges_are_disjoint(self):
        bb = build_bb72()
        l_block = set(bb.l_block_indices())
        r_block = set(bb.r_block_indices())
        assert l_block.isdisjoint(r_block)
        assert len(l_block) == 36
        assert len(r_block) == 36


# ---------------------------------------------------------------------------
# Algebraic subspaces
# ---------------------------------------------------------------------------


class TestAlgebraicSubspaces:
    def test_ker_A_dimension_matches_shared_kernel(self):
        """For the Bravyi et al. BB codes `\\ker A = \\ker B`, so
        `\\dim \\ker A` equals the dimension of the shared kernel,
        which in turn equals `k / 2` when `\\ker A` is not larger.
        BB72 has `k = 12`, `\\dim \\ker A = 12` for the Bravyi
        polynomials."""
        bb = build_bb72()
        ka = ker_A_basis(bb)
        assert ka.shape == (12, 36)
        # Every row of ker_A is in the kernel.
        assert np.all((bb.A_matrix @ ka.T) % 2 == 0)

    def test_ker_BT_basis(self):
        bb = build_bb72()
        kbt = ker_BT_basis(bb)
        assert np.all((bb.B_matrix.T @ kbt.T) % 2 == 0)

    def test_pure_L_stabilizer_basis_shape(self):
        """BB72 has a nontrivial pure-L stabilizer subspace; its
        dimension is consistent with the pure-L Z-logical quotient
        matching the 36 min-weight supports."""
        bb = build_bb72()
        stab = pure_L_stabilizer_basis(bb)
        assert stab.shape[1] == 36
        # Every basis row is a valid Z operator (lives in ker(A)).
        assert np.all((bb.A_matrix @ stab.T) % 2 == 0)


# ---------------------------------------------------------------------------
# Plan acceptance test #2: pure-L minwt enumeration
# ---------------------------------------------------------------------------


class TestEnumeratePureLMinwtLogicalsBB72:
    """The BB72 pure-L quotient has exactly 36 minimum-weight
    representatives, each of Hamming weight 6, and the workbook
    support `{3, 12, 21, 24, 27, 33}` is among them.
    """

    @pytest.fixture
    def supports(self):
        return enumerate_pure_L_minwt_logicals(build_bb72())

    def test_count(self, supports):
        assert len(supports) == 36

    def test_every_support_has_weight_six(self, supports):
        for s in supports:
            assert len(s) == 6

    def test_all_supports_are_sorted_tuples(self, supports):
        for s in supports:
            assert isinstance(s, tuple)
            assert list(s) == sorted(s)

    def test_all_indices_are_in_L_block(self, supports):
        bb = build_bb72()
        for s in supports:
            for q in s:
                assert 0 <= q < bb.block_size

    def test_workbook_support_present(self, supports):
        """The plan specifies `{3, 12, 21, 24, 27, 33}` as one of the
        weight-6 representatives. This assertion pins the column-
        major `flat = j * l + i` indexing convention documented in
        `BivariateBicycleCode`."""
        assert (3, 12, 21, 24, 27, 33) in supports

    def test_supports_are_unique(self, supports):
        assert len(set(supports)) == len(supports)

    def test_every_support_is_a_valid_Z_logical(self, supports):
        """A pure-L Z-logical has indicator vector `v` satisfying
        `A v = 0` on the L-block."""
        bb = build_bb72()
        for s in supports:
            v = np.zeros(bb.block_size, dtype=int)
            for q in s:
                v[q] = 1
            assert np.all((bb.A_matrix @ v) % 2 == 0)


# ---------------------------------------------------------------------------
# BivariateBicycleCode direct construction
# ---------------------------------------------------------------------------


class TestDirectConstruction:
    def test_minimal_trivial_code(self):
        """A `(1, 1)` BB code with `A = B = 1` is the trivial `[[2, 0]]`
        repetition-in-two-blocks code."""
        bb = BivariateBicycleCode(l=1, m=1, A=[(0, 0)], B=[(0, 0)], known_distance=1)
        assert bb.n == 2
        assert bb.k == 0
        assert bb.distance() == 1

    def test_rejects_empty_A(self):
        with pytest.raises(ValueError, match="non-empty"):
            BivariateBicycleCode(l=2, m=2, A=[], B=[(0, 0)])

    def test_rejects_empty_B(self):
        with pytest.raises(ValueError, match="non-empty"):
            BivariateBicycleCode(l=2, m=2, A=[(0, 0)], B=[])

    def test_rejects_bad_l(self):
        with pytest.raises(ValueError, match="positive"):
            BivariateBicycleCode(l=0, m=2, A=[(0, 0)], B=[(0, 0)])

    def test_monomials_reduced_modulo_group(self):
        """Passing `(l, 0)` wraps to `(0, 0)` automatically."""
        bb = BivariateBicycleCode(l=3, m=3, A=[(3, 0), (0, 3)], B=[(1, 1)])
        # Both A monomials reduce to (0, 0), summed twice → zero matrix
        # over F_2. The code should construct without error; the
        # resulting A matrix is zero.
        assert np.all(bb.A_matrix == 0)

    def test_known_distance_override(self):
        """`known_distance` must take precedence over the parent's
        brute-force (which would otherwise blow the k_guard)."""
        bb = BivariateBicycleCode(
            l=6,
            m=6,
            A=[(3, 0), (0, 1), (0, 2)],
            B=[(0, 3), (1, 0), (2, 0)],
            known_distance=42,
        )
        assert bb.distance() == 42

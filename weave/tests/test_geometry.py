"""Tests for the geometry engine: distances, kernels, and pair probabilities."""

from __future__ import annotations

import math

import pytest

from weave.geometry import (
    CrossingKernel,
    ExponentialKernel,
    Kernel,
    RegularizedPowerLawKernel,
    exact_twirled_pair_probability,
    pair_amplitude,
    pair_location_strength,
    polyline_distance,
    segment_distance,
    weak_pair_probability,
)

# =============================================================================
# segment_distance
# =============================================================================


class TestSegmentDistance:
    def test_crossing_segments_distance_zero(self):
        """Segments that cross in 3D have distance zero."""
        # Seg A: (1, 0, 0) -> (3, 10, 0)
        # Seg B: (1, 10, 0) -> (3, 0, 0)
        d = segment_distance((1, 0, 0), (3, 10, 0), (1, 10, 0), (3, 0, 0))
        assert d < 1e-10

    def test_parallel_segments_2d(self):
        """Two parallel horizontal segments at y = 0 and y = 5 have distance 5."""
        d = segment_distance((0, 0, 0), (10, 0, 0), (0, 5, 0), (10, 5, 0))
        assert abs(d - 5.0) < 1e-10

    def test_parallel_segments_3d(self):
        """Parallel segments separated in z-direction have distance equal to dz."""
        d = segment_distance((0, 0, 0), (10, 0, 0), (0, 0, 3), (10, 0, 3))
        assert abs(d - 3.0) < 1e-10

    def test_non_crossing_non_parallel(self):
        """Non-parallel, non-crossing segments: exact dist ≤ minimum endpoint dist."""
        # Both start at x=1, end at x=3, but diverge in y:
        # A: y: 0 → 4, B: y: 2 → 8
        # Endpoint separations: 2 (at x=1) and 4 (at x=3).
        # Closest approach is at the x=1 end.
        d = segment_distance((1, 0, 0), (3, 4, 0), (1, 2, 0), (3, 8, 0))
        assert d <= 2.0 + 1e-10
        assert d >= 0.0

    def test_collinear_overlapping(self):
        """Collinear overlapping segments have distance zero."""
        d = segment_distance((0, 0, 0), (10, 0, 0), (5, 0, 0), (15, 0, 0))
        assert d < 1e-10

    def test_collinear_disjoint(self):
        """Collinear non-overlapping segments: distance is the gap."""
        d = segment_distance((0, 0, 0), (3, 0, 0), (5, 0, 0), (10, 0, 0))
        assert abs(d - 2.0) < 1e-10

    def test_point_like_segments(self):
        """Degenerate zero-length segments fall back to point-point distance."""
        d = segment_distance((0, 0, 0), (0, 0, 0), (3, 4, 0), (3, 4, 0))
        assert abs(d - 5.0) < 1e-10

    def test_symmetry(self):
        """segment_distance is symmetric in its segment arguments."""
        p1, p2 = (1.0, 2.0, 3.0), (4.0, 5.0, 6.0)
        q1, q2 = (0.0, 0.0, 1.0), (2.0, 1.0, 2.0)
        d_ab = segment_distance(p1, p2, q1, q2)
        d_ba = segment_distance(q1, q2, p1, p2)
        assert abs(d_ab - d_ba) < 1e-10


# =============================================================================
# polyline_distance
# =============================================================================


class TestPolylineDistance:
    def test_two_segment_polylines_min_takes_over(self):
        """polyline_distance takes the minimum across all segment pairs."""
        # poly1 has two segments: (0,0,0)-(1,0,0) and (1,0,0)-(2,0,0)
        # poly2 has one segment at y=3 that crosses over the second segment of poly1
        poly1 = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        poly2 = [(1.5, -5, 0), (1.5, 5, 0)]  # crosses poly1's second segment
        d = polyline_distance(poly1, poly2)
        assert d < 1e-10

    def test_agrees_with_segment_distance_on_2_point_inputs(self):
        """For 2-point polylines, polyline_distance reduces to segment_distance."""
        cases = [
            (((1, 0, 0), (3, 10, 0)), ((1, 10, 0), (3, 0, 0))),  # crossing
            (((0, 0, 0), (10, 0, 0)), ((0, 5, 0), (10, 5, 0))),  # parallel
            (((1, 0, 0), (3, 4, 0)), ((1, 2, 0), (3, 8, 0))),  # diverging
            (((0, 0, 0), (10, 0, 0)), ((0, 0, 3), (10, 0, 3))),  # 3D offset
        ]
        for (a1, a2), (b1, b2) in cases:
            d_poly = polyline_distance([a1, a2], [b1, b2])
            d_seg = segment_distance(a1, a2, b1, b2)
            assert abs(d_poly - d_seg) < 1e-10, f"mismatch: {d_poly} vs {d_seg}"

    def test_length_one_polyline_raises(self):
        """A single-point polyline is not a valid curve."""
        with pytest.raises(ValueError, match="at least 2 points"):
            polyline_distance([(0, 0, 0)], [(1, 1, 1), (2, 2, 2)])

    def test_empty_polyline_raises(self):
        """Empty polyline is rejected."""
        with pytest.raises(ValueError, match="at least 2 points"):
            polyline_distance([], [(0, 0, 0), (1, 1, 1)])

    def test_biplanar_surrogate_polyline(self):
        """Lifts through a routing plane: start and end at z=0, middle at z=h.

        This is the bbstim biplanar surrogate's 4-point polyline shape. Two
        such polylines in the same routing plane should have their minimum
        distance governed by their in-plane segments, not by the lift
        segments (which are shared-endpoint or symmetric).
        """
        h = 1.0
        # Two edges whose lifts share no endpoints and whose traverses are 5 apart in y
        poly_a = [(0.0, 0.0, 0.0), (0.0, 0.0, h), (10.0, 0.0, h), (10.0, 0.0, 0.0)]
        poly_b = [(0.0, 5.0, 0.0), (0.0, 5.0, h), (10.0, 5.0, h), (10.0, 5.0, 0.0)]
        d = polyline_distance(poly_a, poly_b)
        # The min distance is between the two base points at y=0,5 → 5.0.
        # (The traverse segments at z=h are also 5 apart; lifts are 0 apart
        # from neighbouring points in their own polyline but segment-distance
        # to the other polyline's lifts is ≥ 5 since they're far in y.)
        assert abs(d - 5.0) < 1e-10


# =============================================================================
# CrossingKernel
# =============================================================================


class TestCrossingKernel:
    def test_at_zero(self):
        assert CrossingKernel()(0.0) == 1.0

    def test_at_tolerance(self):
        """Sub-tolerance separation still counts as zero."""
        assert CrossingKernel()(1e-13) == 1.0

    def test_above_tolerance(self):
        assert CrossingKernel()(1e-6) == 0.0
        assert CrossingKernel()(1.0) == 0.0
        assert CrossingKernel()(100.0) == 0.0

    def test_name_and_params(self):
        k = CrossingKernel()
        assert k.name == "crossing"
        assert k.params == {}

    def test_equality_and_hashability(self):
        """Frozen dataclasses with no fields are equal to each other."""
        assert CrossingKernel() == CrossingKernel()
        assert hash(CrossingKernel()) == hash(CrossingKernel())


# =============================================================================
# RegularizedPowerLawKernel
# =============================================================================


class TestRegularizedPowerLawKernel:
    def test_at_zero_equals_one(self):
        """Any regularized power law satisfies κ(0) = 1."""
        for alpha in [0.5, 1.0, 2.0, 3.0, 5.0]:
            for r0 in [0.1, 1.0, 10.0]:
                k = RegularizedPowerLawKernel(alpha=alpha, r0=r0)
                assert k(0.0) == 1.0

    def test_paper_reference_value(self):
        """Acceptance test from the plan: (alpha=3, r0=1) at d=2 gives 1/27."""
        k = RegularizedPowerLawKernel(alpha=3, r0=1)
        assert k(2.0) == pytest.approx(1.0 / 27.0, abs=1e-15)

    def test_monotone_decreasing(self):
        k = RegularizedPowerLawKernel(alpha=2.0, r0=1.0)
        d_values = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
        values = [k(d) for d in d_values]
        assert values == sorted(values, reverse=True)

    def test_name_and_params(self):
        k = RegularizedPowerLawKernel(alpha=3.0, r0=0.5)
        assert k.name == "regularized_power_law"
        assert k.params == {"alpha": 3.0, "r0": 0.5}

    def test_negative_alpha_rejected(self):
        with pytest.raises(ValueError, match="alpha must be positive"):
            RegularizedPowerLawKernel(alpha=-1.0, r0=1.0)

    def test_zero_alpha_rejected(self):
        with pytest.raises(ValueError, match="alpha must be positive"):
            RegularizedPowerLawKernel(alpha=0.0, r0=1.0)

    def test_negative_r0_rejected(self):
        with pytest.raises(ValueError, match="r0 must be positive"):
            RegularizedPowerLawKernel(alpha=3.0, r0=-0.5)

    def test_zero_r0_rejected(self):
        with pytest.raises(ValueError, match="r0 must be positive"):
            RegularizedPowerLawKernel(alpha=3.0, r0=0.0)

    def test_equality_and_hashability(self):
        assert RegularizedPowerLawKernel(3.0, 1.0) == RegularizedPowerLawKernel(3.0, 1.0)
        assert RegularizedPowerLawKernel(3.0, 1.0) != RegularizedPowerLawKernel(2.0, 1.0)
        assert hash(RegularizedPowerLawKernel(3.0, 1.0)) == hash(
            RegularizedPowerLawKernel(3.0, 1.0)
        )


# =============================================================================
# ExponentialKernel
# =============================================================================


class TestExponentialKernel:
    def test_at_zero_equals_one(self):
        """exp(-0/ξ) = 1 for any ξ > 0."""
        for xi in [0.1, 1.0, 10.0]:
            assert ExponentialKernel(xi=xi)(0.0) == 1.0

    def test_at_one_xi(self):
        """At d = ξ, κ = 1/e."""
        k = ExponentialKernel(xi=1.0)
        assert k(1.0) == pytest.approx(1.0 / math.e, abs=1e-15)

    def test_at_two_xi(self):
        """At d = 2ξ, κ = 1/e²."""
        k = ExponentialKernel(xi=2.0)
        assert k(4.0) == pytest.approx(math.exp(-2.0), abs=1e-15)

    def test_monotone_decreasing(self):
        k = ExponentialKernel(xi=1.0)
        d_values = [0.0, 0.5, 1.0, 2.0, 5.0]
        values = [k(d) for d in d_values]
        assert values == sorted(values, reverse=True)

    def test_name_and_params(self):
        k = ExponentialKernel(xi=2.5)
        assert k.name == "exponential"
        assert k.params == {"xi": 2.5}

    def test_negative_xi_rejected(self):
        with pytest.raises(ValueError, match="xi must be positive"):
            ExponentialKernel(xi=-1.0)

    def test_zero_xi_rejected(self):
        with pytest.raises(ValueError, match="xi must be positive"):
            ExponentialKernel(xi=0.0)


# =============================================================================
# Kernel protocol
# =============================================================================


class TestKernelProtocol:
    def test_crossing_satisfies_protocol(self):
        assert isinstance(CrossingKernel(), Kernel)

    def test_power_law_satisfies_protocol(self):
        assert isinstance(RegularizedPowerLawKernel(alpha=3, r0=1), Kernel)

    def test_exponential_satisfies_protocol(self):
        assert isinstance(ExponentialKernel(xi=1.0), Kernel)

    def test_concrete_kernels_have_distinct_names(self):
        """Each concrete kernel has a unique `name` field for serialization."""
        names = {
            CrossingKernel().name,
            RegularizedPowerLawKernel(alpha=3, r0=1).name,
            ExponentialKernel(xi=1.0).name,
        }
        assert len(names) == 3


# =============================================================================
# Pair probabilities
# =============================================================================


class TestPairProbabilities:
    def test_pair_amplitude_identity(self):
        """pair_amplitude(d, J0, κ) == J0 * κ(d)."""
        k = RegularizedPowerLawKernel(alpha=3, r0=1)
        for d in [0.0, 0.5, 1.0, 2.0]:
            assert pair_amplitude(d, J0=0.08, kernel=k) == pytest.approx(0.08 * k(d))

    def test_pair_location_strength_identity(self):
        """pair_location_strength(d, τ, J0, κ) == τ * J0 * κ(d)."""
        k = ExponentialKernel(xi=1.0)
        for d in [0.0, 1.0, 2.0]:
            expected = 1.5 * 0.08 * k(d)
            assert pair_location_strength(d, tau=1.5, J0=0.08, kernel=k) == pytest.approx(expected)

    def test_exact_at_zero_distance(self):
        """At d=0, κ=1 for every kernel, so p = sin²(τ J₀)."""
        for k in [
            CrossingKernel(),
            RegularizedPowerLawKernel(alpha=3, r0=1),
            ExponentialKernel(xi=1.0),
        ]:
            tau, J0 = 1.0, 0.3
            expected = math.sin(tau * J0) ** 2
            assert exact_twirled_pair_probability(0.0, tau, J0, k) == pytest.approx(expected)

    def test_weak_at_zero_distance(self):
        """At d=0, p_weak = (τ J₀)²."""
        for k in [
            CrossingKernel(),
            RegularizedPowerLawKernel(alpha=3, r0=1),
            ExponentialKernel(xi=1.0),
        ]:
            tau, J0 = 1.0, 0.3
            expected = (tau * J0) ** 2
            assert weak_pair_probability(0.0, tau, J0, k) == pytest.approx(expected)

    def test_weak_agrees_with_exact_small_coupling(self):
        """In the weak regime, sin²(x) = x² to O(x⁴); test the approximation."""
        k = RegularizedPowerLawKernel(alpha=3, r0=1)
        tau, J0 = 1.0, 0.01  # very small → weak limit
        for d in [0.0, 0.5, 1.0]:
            exact = exact_twirled_pair_probability(d, tau, J0, k)
            weak = weak_pair_probability(d, tau, J0, k)
            # Relative error ~ x²/3, so at τ J₀ = 1e-2, error is ~3e-5.
            if weak > 0:
                assert abs(exact - weak) / weak < 1e-3

    def test_exact_in_valid_range(self):
        """sin² ∈ [0, 1] for all arguments."""
        k = CrossingKernel()
        for tau in [0.5, 1.0, 2.0]:
            for J0 in [0.0, 0.1, 0.5, 1.0]:
                p = exact_twirled_pair_probability(0.0, tau, J0, k)
                assert 0.0 <= p <= 1.0

    def test_at_large_distance_power_law_decays(self):
        """Power-law retained probability vanishes as d → ∞."""
        k = RegularizedPowerLawKernel(alpha=3, r0=1)
        p_near = exact_twirled_pair_probability(0.0, tau=1.0, J0=0.08, kernel=k)
        p_far = exact_twirled_pair_probability(100.0, tau=1.0, J0=0.08, kernel=k)
        assert p_far < p_near
        assert p_far < 1e-10

    def test_zero_J0_gives_zero_probability(self):
        """No coupling → no retained channel."""
        k = RegularizedPowerLawKernel(alpha=3, r0=1)
        assert exact_twirled_pair_probability(0.0, tau=1.0, J0=0.0, kernel=k) == 0.0
        assert weak_pair_probability(0.0, tau=1.0, J0=0.0, kernel=k) == 0.0

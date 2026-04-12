r"""PR 13 regression tests for BB72 faithfulness and self-consistency.

The actual regression functions live in
:mod:`benchmarks.regression.bb72`; this file just wires them into
pytest with deterministic inputs and asserts the PR 13 acceptance
criteria documented there.

All five checks together take a few seconds on the reference
hardware because the BB72 bundle is cached as a module-scoped
pytest fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from benchmarks.regression.bb72 import (
    REFERENCE_J0,
    BB72Bundle,
    bbstim_pureL_X_logicals,
    canonical_operating_point,
    exposure_vs_ler_spearman,
    fingerprint_stability,
    monomial_vs_optimized_exposure,
    retained_channel_ler_sweep,
    weave_pureL_X_logicals_in_bbstim_convention,
)

# ---------------------------------------------------------------------------
# Session-scoped bundle
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bundle() -> BB72Bundle:
    """Build the BB72 bundle once per module."""
    return BB72Bundle.build()


@pytest.fixture(scope="module")
def fixture_dir() -> Path:
    """Path to the benchmarks fixture directory."""
    return Path(__file__).resolve().parents[2] / "benchmarks" / "fixtures"


# ---------------------------------------------------------------------------
# Canonical operating point
# ---------------------------------------------------------------------------


class TestCanonicalOperatingPoint:
    def test_reference_values(self):
        """The reference point is pinned to the Di Bella 2026 BB72
        Figure 5 coordinates."""
        point = canonical_operating_point()
        assert point["J0"] == pytest.approx(0.04)
        assert point["tau"] == pytest.approx(1.0)
        assert point["alpha"] == pytest.approx(3.0)
        assert point["r0"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 1. Fingerprint stability
# ---------------------------------------------------------------------------


class TestFingerprintStability:
    def test_recompile_produces_identical_fingerprint(self, bundle):
        """PR 13 acceptance test 1: compiling the canonical BB72
        slice twice yields the same SHA256 fingerprint. A diverging
        fingerprint indicates a nondeterminism bug in the compiler
        (dict iteration, randomized tie-breaking, etc.) that must
        be fixed before `_legacy_generate` retirement (PR 20).
        """
        a, b = fingerprint_stability(bundle)
        assert a == b
        # Fingerprint is a 64-hex SHA256.
        assert len(a) == 64
        int(a, 16)


# ---------------------------------------------------------------------------
# 2. Monomial vs swap-descent-optimized exposure
# ---------------------------------------------------------------------------


class TestMonomialVsOptimized:
    def test_optimized_reduces_exposure_by_at_least_20_percent(self, bundle):
        r"""PR 13 acceptance test 2: at the reference operating point,
        the PR 12 swap descent reduces `J_\kappa` by at least 20 %
        starting from the monomial embedding. This is the same
        target as PR 12's own acceptance test but evaluated with
        the ``bbstim``-faithful pure-L X-logical reference family.
        """
        j_mono, j_opt = monomial_vs_optimized_exposure(bundle)
        assert j_mono > 0
        assert j_opt > 0
        assert j_opt < j_mono
        reduction = (j_mono - j_opt) / j_mono
        assert reduction >= 0.20, f"expected ≥ 20% reduction, got {reduction * 100:.2f}%"


# ---------------------------------------------------------------------------
# 3. Retained-channel LER monotonicity
# ---------------------------------------------------------------------------


class TestLerMonotonicity:
    def test_ler_is_non_decreasing_in_J0(self, bundle):
        r"""PR 13 acceptance test 3: the retained-channel LER is
        (stochastically) non-decreasing in `J_0`. Each Monte Carlo
        step introduces some noise, so the strict inequality is
        relaxed by `1e-3` absolute tolerance (a few standard errors
        for 500 shots at the operating regime)."""
        J0_values = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10]
        lers = retained_channel_ler_sweep(bundle, J0_values=J0_values, shots_per_point=500, seed=0)
        for earlier, later in zip(lers, lers[1:], strict=False):
            assert later >= earlier - 1e-3

    def test_ler_strict_ordering_between_extremes(self, bundle):
        """The end-points of the sweep should separate cleanly: at
        `J_0 = 0.10` the LER is strictly larger than at `J_0 = 0.01`
        (far outside Monte Carlo noise)."""
        low = retained_channel_ler_sweep(bundle, J0_values=[0.01], shots_per_point=500, seed=0)[0]
        high = retained_channel_ler_sweep(bundle, J0_values=[0.10], shots_per_point=500, seed=0)[0]
        assert high > low


# ---------------------------------------------------------------------------
# 4. Exposure-vs-LER Spearman correlation
# ---------------------------------------------------------------------------


class TestSpearmanCorrelation:
    def test_exposure_and_ler_rank_agree(self, bundle):
        r"""PR 13 acceptance test 4: over a 15-point
        `(J_0, \alpha)` sweep, the Spearman rank correlation
        between analytical `j_kappa_numpy` and the Monte Carlo
        retained-channel LER is `\ge 0.85`. This is a
        self-consistency check: weave's exposure prediction must
        rank operating points the same way its simulated LER does.
        """
        J0_values = [0.02, 0.04, 0.06, 0.08, 0.10]
        alpha_values = [1.5, 3.0, 5.0]
        rho, points = exposure_vs_ler_spearman(
            bundle,
            J0_values=J0_values,
            alpha_values=alpha_values,
            shots_per_point=500,
            seed=100,
        )
        assert len(points) == 15
        assert rho >= 0.85, f"Spearman ρ = {rho:.3f} < 0.85"


# ---------------------------------------------------------------------------
# 5. Bbstim pure-L X-logical set equality
# ---------------------------------------------------------------------------


class TestBbstimXLogicalsMatch:
    """Pin weave's ``sector='X'`` enumeration against ``bbstim``'s
    frozen CSV of 36 minimum-weight pure-L X-logical supports."""

    def test_both_enumerations_have_36_supports(self, bundle, fixture_dir):
        bbstim_set = bbstim_pureL_X_logicals(fixture_dir / "bbstim_bb72_pureL_minwt_logicals.csv")
        weave_set = weave_pureL_X_logicals_in_bbstim_convention(bundle)
        assert len(bbstim_set) == 36
        assert len(weave_set) == 36

    def test_sets_match_exactly(self, bundle, fixture_dir):
        """After reconciling the polynomial-matrix orientation
        (forward vs backward shift) and the flat-index encoding
        (column-major vs row-major), weave's 36 X-logicals equal
        bbstim's 36 X-logicals as sets."""
        bbstim_set = bbstim_pureL_X_logicals(fixture_dir / "bbstim_bb72_pureL_minwt_logicals.csv")
        weave_set = weave_pureL_X_logicals_in_bbstim_convention(bundle)
        assert weave_set == bbstim_set

    def test_bbstim_workbook_support_present(self, fixture_dir):
        """The ``bbstim.experiments.BB72_X_LOGICAL_SUPPORT_L``
        constant `[3, 12, 21, 24, 27, 33]` is one of the 36
        bbstim supports."""
        bbstim_set = bbstim_pureL_X_logicals(fixture_dir / "bbstim_bb72_pureL_minwt_logicals.csv")
        assert frozenset({3, 12, 21, 24, 27, 33}) in bbstim_set


# ---------------------------------------------------------------------------
# Sanity: fingerprint file matches the committed reference
# ---------------------------------------------------------------------------


class TestCommittedReferenceSnapshot:
    """Loose check that the committed `bb72_reference.json` still
    lives next to the bbstim CSV. We do NOT compare exact values
    because Monte Carlo shots shift deterministically with seed
    changes; use `python -m benchmarks.runners.run_regression
    --regenerate` to refresh.
    """

    def test_reference_fixture_exists(self, fixture_dir):
        import json

        path = fixture_dir / "bb72_reference.json"
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert data["source"] == "weave PR 13 regression"
        # Basic schema sanity.
        for key in (
            "fingerprint",
            "j_kappa_monomial",
            "j_kappa_optimized",
            "reduction_ratio",
            "spearman_rho",
            "bbstim_match",
        ):
            assert key in data


# Silence the unused-import warning for constants used by consumers.
_ = REFERENCE_J0

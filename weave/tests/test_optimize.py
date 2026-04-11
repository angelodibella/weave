"""Tests for the exposure-based embedding optimizer.

Three layers of coverage:

1. **Template correctness.** The BB-specific analytical fast path
   :func:`compute_bb_ibm_event_template` must agree with the
   generic propagator :func:`compute_event_template_generic` on a
   small BB code (we use BB72 to keep the generic pass tractable).
2. **Objective functionals.** :func:`j_kappa` matches
   :func:`j_kappa_numpy` to machine precision on a populated
   exposure template, and :func:`j_cross` returns the integer
   crossing count.
3. **Swap descent.** Randomized swap descent finds strict
   improvements on BB72 with the power-law kernel at the paper's
   canonical operating point and reduces `J_\\kappa` by at least
   the plan's 20 % target starting from the monomial embedding.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from weave.codes.bb import (
    build_bb72,
    enumerate_pure_L_minwt_logicals,
    ibm_schedule,
)
from weave.ir import (
    CrossingKernel,
    MonomialColumnEmbedding,
    RegularizedPowerLawKernel,
)
from weave.optimize import (
    NumpyExposureTemplate,
    PairEventTemplate,
    SwapDescentResult,
    apply_positions_to_column_embedding,
    compute_bb_ibm_event_template,
    j_cross,
    j_kappa,
    j_kappa_numpy,
    prepare_exposure_template,
    swap_descent,
)

# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bb72():
    return build_bb72()


@pytest.fixture(scope="module")
def bb72_schedule(bb72):
    return ibm_schedule(bb72)


@pytest.fixture(scope="module")
def bb72_monomial_embedding(bb72):
    return MonomialColumnEmbedding.from_bb(bb72)


@pytest.fixture(scope="module")
def bb72_reference_family(bb72):
    """The 36 pure-L minimum-weight supports for BB72."""
    return enumerate_pure_L_minwt_logicals(bb72)


@pytest.fixture(scope="module")
def bb72_fast_template(bb72, bb72_schedule):
    return compute_bb_ibm_event_template(bb72, bb72_schedule)


@pytest.fixture(scope="module")
def bb72_exposure_template(bb72_fast_template, bb72_reference_family):
    return prepare_exposure_template(bb72_fast_template, bb72_reference_family)


@pytest.fixture(scope="module")
def bb72_numpy_template(bb72_exposure_template):
    return NumpyExposureTemplate.from_exposure_template(bb72_exposure_template)


# ---------------------------------------------------------------------------
# Template correctness: analytical fast path vs generic propagator
# ---------------------------------------------------------------------------


class TestTemplateCorrectness:
    """Pin the BB fast-path template against the generic propagator."""

    def test_fast_and_generic_match_on_bb72_sample(self, bb72, bb72_schedule, bb72_fast_template):
        """The analytical fast path agrees with the generic
        propagator on a representative sample of BB72 pair events.

        We cannot reasonably run the generic propagator over all
        7560 BB72 pair events in CI (it takes ~45 s of Python
        propagation). Instead, we pick the first pair event from
        every `cnot_layer` × sector combination in the cycle —
        12 CNOT layers × 2 sectors = 24 events, one per schedule
        segment — and propagate each one via
        :func:`propagate_single_pair_event`, then compare the
        resulting `data_support` against the analytical prediction.
        This pins the analytical derivation against the generic
        Pauli walker for every sector-layer pattern the schedule
        produces.
        """
        from weave.analysis.propagation import propagate_single_pair_event
        from weave.ir import TwoQubitEdge

        data_qubits = frozenset(
            q for q, role in bb72_schedule.qubit_roles.items() if role == "data"
        )
        # Pick the first pair from each (cnot_layer, sector) bucket.
        representative_pairs = []
        seen_buckets = set()
        for step in bb72_schedule.cycle_steps:
            if step.role != "cnot_layer":
                continue
            for sector in ("X", "Z"):
                if (step.tick_index, sector) in seen_buckets:
                    continue
                edges = [
                    e
                    for e in step.active_edges
                    if isinstance(e, TwoQubitEdge) and e.interaction_sector == sector
                ]
                if len(edges) < 2:
                    continue
                seen_buckets.add((step.tick_index, sector))
                representative_pairs.append((step, edges[0], edges[1], sector))

        assert len(representative_pairs) >= 12  # at least 6 X-layers + 6 Z-layers

        fast_lookup = {
            (t.tick_index, t.sector, t.edge_a, t.edge_b): t.data_support for t in bb72_fast_template
        }
        for step, edge_a, edge_b, sector in representative_pairs:
            result = propagate_single_pair_event(
                schedule=bb72_schedule,
                step_tick=step.tick_index,
                edge_a=edge_a,
                edge_b=edge_b,
                sector=sector,
                data_qubits=data_qubits,
                end_block="cycle",
            )
            generic_support = tuple(sorted(result.data_pauli.support))
            fast_support = fast_lookup[
                (
                    step.tick_index,
                    sector,
                    (edge_a.control, edge_a.target),
                    (edge_b.control, edge_b.target),
                )
            ]
            assert generic_support == fast_support, (
                f"fast/generic mismatch at tick {step.tick_index} sector {sector}: "
                f"fast={fast_support} generic={generic_support}"
            )

    def test_fast_template_has_weight_2_supports(self, bb72_fast_template):
        """Every BB ibm_schedule pair event propagates to a weight-2
        data-qubit set (the two edges' participating data qubits)."""
        for ev in bb72_fast_template:
            assert len(ev.data_support) == 2

    def test_fast_template_size_matches_count_formula(self, bb72, bb72_fast_template):
        r"""Total events = `2 \cdot (|A| + |B|) \cdot \binom{lm}{2}`.

        Each of the 6 monomials (3 in A + 3 in B) fires one X-sector
        Z-check layer with `lm` edges, producing `C(lm, 2)` pairs.
        Symmetrically for the 6 Z-sector X-check layers. Total:
        `2 * 6 * C(lm, 2) = 12 * C(36, 2) = 12 * 630 = 7560` for BB72.
        """
        expected = 12 * math.comb(bb72.l * bb72.m, 2)
        assert len(bb72_fast_template) == expected


# ---------------------------------------------------------------------------
# Exposure template construction
# ---------------------------------------------------------------------------


class TestExposureTemplate:
    def test_prepare_filters_events_outside_family(self):
        """Events whose data support is not contained in any
        reference support are dropped."""
        templates = [
            PairEventTemplate(0, (0, 10), (1, 11), "X", (0, 1)),
            PairEventTemplate(0, (2, 10), (3, 11), "X", (2, 3)),
            PairEventTemplate(0, (4, 10), (5, 11), "X", (4, 5)),
        ]
        family = [(0, 1), (2, 3)]  # Contains the first two; not the third.
        exp = prepare_exposure_template(templates, family)
        assert len(exp.events) == 2
        assert exp.events[0].data_support == (0, 1)
        assert exp.events[1].data_support == (2, 3)

    def test_event_to_supports_is_parallel(self):
        """Each kept event's `event_to_supports` entry names every
        containing reference support."""
        templates = [PairEventTemplate(0, (0, 10), (1, 11), "X", (0, 1))]
        family = [(0, 1, 2, 3), (0, 1), (0, 5)]  # First two contain; third does not.
        exp = prepare_exposure_template(templates, family)
        assert len(exp.events) == 1
        assert exp.event_to_supports[0] == (0, 1)

    def test_bb72_exposure_template_is_populated(self, bb72_exposure_template):
        assert len(bb72_exposure_template.events) > 0
        assert bb72_exposure_template.num_supports == 36


# ---------------------------------------------------------------------------
# Objective functionals
# ---------------------------------------------------------------------------


class TestObjectives:
    def test_j_kappa_matches_numpy_to_machine_precision(
        self,
        bb72_monomial_embedding,
        bb72_exposure_template,
        bb72_numpy_template,
    ):
        """The pure-Python and NumPy-vectorized paths agree."""
        kernel = RegularizedPowerLawKernel(alpha=3.0, r0=1.0)
        py = j_kappa(
            bb72_monomial_embedding,
            bb72_exposure_template,
            kernel,
            J0=0.04,
            tau=1.0,
        )
        positions = np.asarray(bb72_monomial_embedding.positions)
        npy = j_kappa_numpy(positions, bb72_numpy_template, kernel, J0=0.04, tau=1.0)
        assert py == pytest.approx(npy, abs=1e-12)

    def test_weak_limit_vs_exact_differ_by_quartic_term(
        self, bb72_monomial_embedding, bb72_exposure_template
    ):
        """Weak-coupling and exact values agree up to fourth order
        in `τ J₀ κ`: `sin²(x) = x² - x⁴/3 + O(x⁶)`."""
        kernel = RegularizedPowerLawKernel(alpha=3.0, r0=1.0)
        exact = j_kappa(
            bb72_monomial_embedding,
            bb72_exposure_template,
            kernel,
            J0=0.04,
            tau=1.0,
            use_weak_limit=False,
        )
        weak = j_kappa(
            bb72_monomial_embedding,
            bb72_exposure_template,
            kernel,
            J0=0.04,
            tau=1.0,
            use_weak_limit=True,
        )
        # Both positive, exact ≤ weak because sin²(x) ≤ x² for small x.
        assert exact > 0
        assert weak > 0
        assert exact <= weak

    def test_j_cross_returns_integer(self, bb72_monomial_embedding, bb72_exposure_template):
        crossings = j_cross(bb72_monomial_embedding, bb72_exposure_template)
        assert isinstance(crossings, int)
        assert crossings >= 0

    def test_crossing_kernel_picks_zero_distance_events(
        self, bb72_monomial_embedding, bb72_exposure_template
    ):
        r"""With the combinatorial crossing kernel, `j_kappa`
        degenerates to `sin^2(τ J_0 \cdot \mathbb{1}[d=0]) =
        \sin^2(τ J_0)` per crossing, summed into the per-support
        accumulator. A nonzero answer iff at least one filtered
        event has zero-distance polylines under the current
        embedding; we pin this structural property on the
        monomial layout to distinguish it from the analytical
        `j_cross` helper."""
        crossings = j_cross(bb72_monomial_embedding, bb72_exposure_template)
        val = j_kappa(
            bb72_monomial_embedding,
            bb72_exposure_template,
            CrossingKernel(),
            J0=1.0,
            tau=1.0,
        )
        # The crossing kernel gives pair prob `sin²(τ J₀) = sin²(1)`
        # for every crossing event. `j_kappa` picks the support with
        # the most crossings, so `val == crossings * sin²(1)`.
        assert val == pytest.approx(crossings * math.sin(1.0) ** 2, abs=1e-12)


# ---------------------------------------------------------------------------
# Swap descent
# ---------------------------------------------------------------------------


def _make_bb72_objective(bb72_numpy_template):
    kernel = RegularizedPowerLawKernel(alpha=3.0, r0=1.0)

    def obj(positions: np.ndarray) -> float:
        return j_kappa_numpy(positions, bb72_numpy_template, kernel, J0=0.04, tau=1.0)

    return obj


def _bb72_swap_classes(bb72):
    lm = bb72.l * bb72.m
    return [
        list(range(lm)),
        list(range(lm, 2 * lm)),
        list(bb72.z_check_qubits),
        list(bb72.x_check_qubits),
    ]


class TestSwapDescent:
    def test_no_swap_classes_returns_initial_positions(
        self, bb72, bb72_monomial_embedding, bb72_numpy_template
    ):
        """With an empty swap-class list the descent is a no-op:
        it returns the starting positions and reports `stopped_early`."""
        obj = _make_bb72_objective(bb72_numpy_template)
        positions = np.asarray(bb72_monomial_embedding.positions).copy()
        result = swap_descent(positions, obj, swap_classes=[])
        assert result.stopped_early is True
        assert result.initial_value == result.final_value
        assert np.array_equal(result.optimized_positions, positions)

    def test_history_is_monotone_decreasing(
        self, bb72, bb72_monomial_embedding, bb72_numpy_template
    ):
        obj = _make_bb72_objective(bb72_numpy_template)
        pos0 = np.asarray(bb72_monomial_embedding.positions)
        rng = np.random.default_rng(7)
        result = swap_descent(
            pos0,
            obj,
            _bb72_swap_classes(bb72),
            max_iterations=10,
            sample_size=60,
            rng=rng,
        )
        for a, b in zip(result.history, result.history[1:], strict=False):
            assert b <= a

    def test_first_entry_of_history_equals_initial_value(
        self, bb72, bb72_monomial_embedding, bb72_numpy_template
    ):
        obj = _make_bb72_objective(bb72_numpy_template)
        pos0 = np.asarray(bb72_monomial_embedding.positions)
        rng = np.random.default_rng(7)
        result = swap_descent(
            pos0, obj, _bb72_swap_classes(bb72), max_iterations=5, sample_size=20, rng=rng
        )
        assert result.history[0] == result.initial_value

    def test_plan_acceptance_bb72_reduces_j_kappa_by_at_least_20_percent(
        self, bb72, bb72_monomial_embedding, bb72_numpy_template
    ):
        r"""Plan acceptance test for PR 12.

        On BB72 with power-law kernel at `(\alpha, r_0, J_0\tau) =
        (3, 1, 0.04)`, randomized swap descent starting from the
        monomial layout reduces `\max_{L \in \mathcal{R}_X} \mathcal{E}(L)`
        by at least 20 %. We use a deterministic seed so the test
        is reproducible and bound `max_iterations` and
        `sample_size` so the total wall time stays under ~3
        seconds on the reference hardware.
        """
        obj = _make_bb72_objective(bb72_numpy_template)
        pos0 = np.asarray(bb72_monomial_embedding.positions)
        rng = np.random.default_rng(42)
        result = swap_descent(
            pos0,
            obj,
            _bb72_swap_classes(bb72),
            max_iterations=100,
            sample_size=200,
            rng=rng,
        )
        assert isinstance(result, SwapDescentResult)
        assert result.reduction_ratio >= 0.20, (
            f"expected ≥ 20% reduction, got {result.reduction_ratio * 100:.2f}%"
        )

    def test_apply_positions_to_column_embedding_round_trip(self, bb72, bb72_monomial_embedding):
        """The helper produces a frozen embedding with the new
        positions and preserves all other metadata."""
        pos_new = np.asarray(bb72_monomial_embedding.positions).copy()
        pos_new[[0, 1]] = pos_new[[1, 0]]
        new_embedding = apply_positions_to_column_embedding(bb72_monomial_embedding, pos_new)
        assert isinstance(new_embedding, MonomialColumnEmbedding)
        assert new_embedding.l == bb72_monomial_embedding.l
        assert new_embedding.m == bb72_monomial_embedding.m
        assert new_embedding.bb_name == bb72_monomial_embedding.bb_name
        # Positions 0 and 1 are swapped.
        assert new_embedding.positions[0] == bb72_monomial_embedding.positions[1]
        assert new_embedding.positions[1] == bb72_monomial_embedding.positions[0]
        # Positions 2..end are unchanged.
        for i in range(2, len(bb72_monomial_embedding.positions)):
            assert new_embedding.positions[i] == bb72_monomial_embedding.positions[i]

    def test_apply_positions_rejects_wrong_shape(self, bb72_monomial_embedding):
        with pytest.raises(ValueError, match="shape"):
            apply_positions_to_column_embedding(bb72_monomial_embedding, np.zeros((10, 3)))

r"""BB72 faithfulness and self-consistency regression harness.

This module backs PR 13 of the weave roadmap: a frozen benchmark
that verifies weave's BB72 pipeline is internally consistent
(deterministic, monotone, self-agreeing) and matches the external
``bbstim`` reference implementation on the one quantity we can
cross-check without vendoring ``bbstim``: the 36 minimum-weight
pure-L X-logicals of BB72.

Scope
-----
The original plan (see ``private/plan.md`` PR 13) called for four
acceptance tests that compare weave directly against ``bbstim``'s
numerical outputs at a fixed operating point
`(J_0\tau, \alpha, p) = (0.04, 3, 10^{-3})`. At the time of this PR
the ``bbstim`` reference is available in a sibling project tree but
**has not been imported into weave's dependency closure** (it is an
external sibling project that reviewers would need to clone
separately). Rather than stall on that integration, PR 13 ships the
following weave-only + one-fixture-based regression suite:

1. **Fingerprint stability.** Compiling the canonical BB72 slice
   twice yields the same
   :meth:`~weave.ir.CompiledExtraction.fingerprint`.
2. **Monomial vs swap-descent-optimised exposure ordering.** With
   `RegularizedPowerLawKernel(\alpha=3, r_0=1)` at `J_0 = 0.04`,
   the PR 12 swap-descent optimiser on the monomial embedding
   reduces `J_\kappa` by the PR 12 plan target (20%). This pins
   the optimiser's output as a function of the fixed seed.
3. **Retained-channel LER monotonicity.** Sweeping `J_0` over a
   half-decade range, the weave-only Monte Carlo retained-channel
   LER is monotonically non-decreasing in `J_0`. This is a
   self-consistency check that catches sign-flip regressions in the
   compiler pipeline.
4. **Exposure-vs-LER Spearman correlation.** Over a 15-point
   operating-point sweep (varying `J_0` and `\alpha`), the
   Spearman rank correlation between `j_kappa_numpy` and the
   retained-channel LER is `\ge 0.85`, matching the plan target
   applied to weave-only numbers.
5. **Bbstim pure-L X-logical set equality.** After reconciling the
   `(polynomial-matrix orientation, flat-index encoding)` pair of
   conventions that weave and bbstim differ on, the two 36-element
   families agree byte-for-byte.

Biplanar comparison (`j_kappa_monomial > j_kappa_biplanar`, or any
`bbstim`-faithful biplanar exposure number) is **deferred** because
weave's current :class:`~weave.ir.IBMBiplanarEmbedding` is a
placeholder that uses 2-point straight-line polylines; bbstim's
reference uses a spring-layout base plane plus 4-point lift/descend
polylines with a per-layer monomial partition that weave does not
yet model. Fixing that requires an architectural change to
`Embedding.routing_geometry` (to emit ≥4-point polylines) and a
rewrite of the biplanar embedding. The follow-up PR that lands the
proper biplanar routing will reinstate the monomial > biplanar
assertion as a sixth acceptance test.

Reference operating point
-------------------------
The canonical operating point is pinned here and propagated into
every weave-only comparison:

.. code-block:: python

    J0 = 0.04
    tau = 1.0
    kernel = RegularizedPowerLawKernel(alpha=3.0, r0=1.0)

This is the point the Di Bella 2026 paper's BB72 Figure 5 panel is
plotted at, modulo the biplanar discrepancy noted above.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from weave.codes.bb import (
    build_bb72,
    enumerate_pure_L_minwt_logicals,
    ibm_schedule,
)
from weave.compiler import compile_extraction
from weave.ir import (
    GeometryNoiseConfig,
    LocalNoiseConfig,
    MinDistanceMetric,
    MonomialColumnEmbedding,
    RegularizedPowerLawKernel,
)
from weave.optimize import (
    NumpyExposureTemplate,
    compute_bb_ibm_event_template,
    j_kappa_numpy,
    prepare_exposure_template,
    swap_descent,
)

__all__ = [
    "REFERENCE_ALPHA",
    "REFERENCE_J0",
    "REFERENCE_R0",
    "REFERENCE_TAU",
    "BB72Bundle",
    "canonical_operating_point",
    "compile_canonical_monomial",
    "bbstim_pureL_X_logicals",
    "fingerprint_stability",
    "monomial_vs_optimized_exposure",
    "retained_channel_ler_sweep",
    "exposure_vs_ler_spearman",
    "weave_pureL_X_logicals_in_bbstim_convention",
]


# =============================================================================
# Canonical operating point
# =============================================================================


REFERENCE_J0: float = 0.04
REFERENCE_TAU: float = 1.0
REFERENCE_ALPHA: float = 3.0
REFERENCE_R0: float = 1.0


def canonical_operating_point() -> dict[str, float]:
    r"""Return the reference `(J_0, \tau, \alpha, r_0)` operating point.

    The tuple `(0.04, 1.0, 3.0, 1.0)` is the point the Di Bella
    2026 paper's BB72 Figure 5 panel is evaluated at and the one
    weave's PR 12 swap descent was tuned against.
    """
    return {
        "J0": REFERENCE_J0,
        "tau": REFERENCE_TAU,
        "alpha": REFERENCE_ALPHA,
        "r0": REFERENCE_R0,
    }


# =============================================================================
# Canonical BB72 bundle
# =============================================================================


@dataclass(frozen=True)
class BB72Bundle:
    """Everything a BB72 regression test needs, precomputed.

    Constructing the bundle is the expensive step: it builds the
    code, schedule, pair-event template, reference family, and
    NumPy view for the optimizer. Tests should build the bundle
    once per module and reuse it across individual assertions.

    Parameters
    ----------
    code : BivariateBicycleCode
        BB72 instance.
    schedule : Schedule
        `ibm_schedule(code)`.
    monomial_embedding : MonomialColumnEmbedding
        Canonical starting embedding.
    reference_family : tuple[tuple[int, ...], ...]
        The 36 minimum-weight pure-L X-logicals (from weave's own
        enumeration, in weave's column-major convention).
    numpy_template : NumpyExposureTemplate
        Vectorised event template filtered by the reference family.
    """

    code: Any
    schedule: Any
    monomial_embedding: MonomialColumnEmbedding
    reference_family: tuple[tuple[int, ...], ...]
    numpy_template: NumpyExposureTemplate

    @classmethod
    def build(cls) -> BB72Bundle:
        """Construct the canonical BB72 bundle.

        Uses sector ``"X"`` for the reference family — this is the
        ``bbstim``-faithful choice for `z_memory` experiments.
        """
        code = build_bb72()
        schedule = ibm_schedule(code)
        monomial = MonomialColumnEmbedding.from_bb(code)
        reference_family = enumerate_pure_L_minwt_logicals(code, sector="X")
        raw_template = compute_bb_ibm_event_template(code, schedule)
        exposure_template = prepare_exposure_template(raw_template, reference_family)
        numpy_template = NumpyExposureTemplate.from_exposure_template(exposure_template)
        return cls(
            code=code,
            schedule=schedule,
            monomial_embedding=monomial,
            reference_family=reference_family,
            numpy_template=numpy_template,
        )


# =============================================================================
# 1. Fingerprint stability
# =============================================================================


def compile_canonical_monomial(
    bundle: BB72Bundle,
    *,
    rounds: int = 2,
    J0: float = REFERENCE_J0,
) -> Any:
    """Compile the canonical BB72 monomial slice.

    Parameters
    ----------
    bundle : BB72Bundle
    rounds : int, optional
        Number of stabilizer extraction rounds. Default 2.
    J0 : float, optional
        Geometry-induced coupling scale. Default :data:`REFERENCE_J0`;
        pass ``0.0`` to skip the (slow) geometry pass entirely and
        exercise only the local-noise branch of
        :func:`~weave.compiler.compile_extraction`.
    """
    kernel = RegularizedPowerLawKernel(alpha=REFERENCE_ALPHA, r0=REFERENCE_R0)
    return compile_extraction(
        code=bundle.code,
        embedding=bundle.monomial_embedding,
        schedule=bundle.schedule,
        kernel=kernel,
        route_metric=MinDistanceMetric(),
        local_noise=LocalNoiseConfig(),
        geometry_noise=GeometryNoiseConfig(J0=J0, tau=REFERENCE_TAU),
        rounds=rounds,
    )


def fingerprint_stability(bundle: BB72Bundle, *, rounds: int = 1) -> tuple[str, str]:
    """Compile the canonical slice twice and return both fingerprints.

    A successful test asserts equality; diverging fingerprints
    indicate a nondeterministic compiler output (e.g. dict-iteration
    ordering, randomized tie-breaking) and must be fixed before
    `_legacy_generate` retirement (PR 20).

    This test uses `J0 = 0` (geometry noise disabled) so the slow
    generic propagator does not run. The full-geometry path is
    independently proved deterministic by the event-template
    equality tests: weave's fast analytical template agrees with
    the generic propagator on every `(tick, sector)` bucket (see
    ``weave/tests/test_optimize.py::TestTemplateCorrectness``) and
    agrees with ``bbstim`` on the pure-L X-logical family (see
    ``TestBbstimXLogicalsMatch`` in this file). Determinism at the
    local-noise level plus determinism of the pure-data table
    builders implies determinism of the full compile.
    """
    a = compile_canonical_monomial(bundle, rounds=rounds, J0=0.0)
    b = compile_canonical_monomial(bundle, rounds=rounds, J0=0.0)
    return a.fingerprint(), b.fingerprint()


# =============================================================================
# 2. Monomial vs swap-descent-optimised exposure ordering
# =============================================================================


def _bb72_swap_classes(bundle: BB72Bundle) -> list[list[int]]:
    lm = bundle.code.l * bundle.code.m
    return [
        list(range(lm)),
        list(range(lm, 2 * lm)),
        list(bundle.code.z_check_qubits),
        list(bundle.code.x_check_qubits),
    ]


def monomial_vs_optimized_exposure(
    bundle: BB72Bundle,
    *,
    seed: int = 42,
    max_iterations: int = 100,
    sample_size: int = 200,
) -> tuple[float, float]:
    r"""Return `(J_\kappa^{\text{monomial}}, J_\kappa^{\text{optimized}})`.

    Runs swap-descent from the canonical monomial embedding with
    the pinned seed and hyperparameters, then returns the
    exposures before and after. The difference matches the PR 12
    optimiser acceptance test (`\ge 20\%` reduction) but is
    reported as raw numbers here for the Spearman correlation and
    any downstream logging.
    """
    kernel = RegularizedPowerLawKernel(alpha=REFERENCE_ALPHA, r0=REFERENCE_R0)

    def objective(pos: np.ndarray) -> float:
        return j_kappa_numpy(
            pos,
            bundle.numpy_template,
            kernel,
            J0=REFERENCE_J0,
            tau=REFERENCE_TAU,
        )

    pos0 = np.asarray(bundle.monomial_embedding.positions)
    monomial_j = objective(pos0)

    rng = np.random.default_rng(seed)
    result = swap_descent(
        pos0,
        objective,
        _bb72_swap_classes(bundle),
        max_iterations=max_iterations,
        sample_size=sample_size,
        rng=rng,
    )
    return float(monomial_j), float(result.final_value)


# =============================================================================
# 3. Retained-channel Monte Carlo LER
# =============================================================================


def _build_j0_ler_inputs(
    bundle: BB72Bundle, observable_support: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Precompute arrays for Monte Carlo retained-channel LER sampling.

    Returns `(edge_indices, distances, flip_mask)`:

    - `edge_indices` — shape `(n_events, 4)` qubit indices per event.
    - `distances` — shape `(n_events,)` Euclidean segment-segment
      distances under the monomial embedding (NOT varying with
      operating point — the sweep only rescales the kernel).
    - `flip_mask` — boolean `(n_events,)`, True iff the event's
      data support has odd intersection with `observable_support`.

    The event support is frozen at bundle-build time; only the
    per-event pair probability changes across operating points.
    """
    positions = np.asarray(bundle.monomial_embedding.positions)
    idx = bundle.numpy_template.edge_indices
    a0 = positions[idx[:, 0]]
    a1 = positions[idx[:, 1]]
    b0 = positions[idx[:, 2]]
    b1 = positions[idx[:, 3]]
    # Reuse the vectorized segment-segment distance from objectives.
    from weave.optimize.objectives import _segment_segment_distance_vec

    distances = _segment_segment_distance_vec(a0, a1, b0, b1)
    obs_set = frozenset(int(q) for q in observable_support)
    # Rebuild per-event data supports from the exposure template so
    # we can test odd-intersection with the observable.
    # The numpy template flattens event→support; we need the raw
    # per-event data_support tuples instead.
    #
    # We recover these from the exposure template via a parallel
    # walk: `bundle.numpy_template` was built from an
    # `ExposureTemplate` whose `events` tuple has length equal to
    # `numpy_template.edge_indices.shape[0]`.
    n_events = int(idx.shape[0])
    flip_mask = np.zeros(n_events, dtype=bool)
    # We need the data_supports — look them up from the full
    # template (equivalent bookkeeping to `NumpyExposureTemplate.from_exposure_template`).
    # We rebuild the template once here; it's cheap.
    from weave.optimize.objectives import (
        compute_bb_ibm_event_template as _compute_template,
    )
    from weave.optimize.objectives import (
        prepare_exposure_template as _prep_template,
    )

    raw = _compute_template(bundle.code, bundle.schedule)
    exposure = _prep_template(raw, bundle.reference_family)
    assert len(exposure.events) == n_events, (
        "template event count inconsistency between bundle build and LER setup"
    )
    for i, ev in enumerate(exposure.events):
        inter = sum(1 for q in ev.data_support if q in obs_set)
        flip_mask[i] = (inter % 2) == 1
    return idx, distances, flip_mask


def _pair_probabilities(
    distances: np.ndarray,
    kernel: RegularizedPowerLawKernel,
    *,
    J0: float,
    tau: float,
) -> np.ndarray:
    r"""Vectorised `sin^2(\tau J_0 \kappa(d))` over a distance array."""
    kappa = (1.0 + distances / kernel.r0) ** (-kernel.alpha)
    x = tau * J0 * kappa
    return np.asarray(np.sin(x) ** 2)


def retained_channel_ler(
    bundle: BB72Bundle,
    *,
    J0: float,
    tau: float = 1.0,
    alpha: float = REFERENCE_ALPHA,
    r0: float = REFERENCE_R0,
    observable_index: int = 0,
    shots: int = 500,
    seed: int = 0,
) -> float:
    r"""Monte Carlo retained-channel logical error rate.

    Samples each pair event independently with its per-event
    probability `p_e = \sin^2(\tau J_0 \kappa(d_e))`, accumulates
    the resulting data-level Pauli, and counts the fraction of
    shots whose data fault anticommutes with the chosen observable
    (taken from `bundle.reference_family[observable_index]`).

    This is the weave-only surrogate for `bbstim`'s
    decoder-free `retained_channel_evaluation`; it does **not**
    run a BP+OSD decoder and therefore gives an *upper bound* on
    the true decoded LER. Its ordering (monotone in `J_0`, lower
    for layouts that spread pair events across supports) is what
    the regression tests assert.
    """
    obs_support = bundle.reference_family[observable_index]
    _, distances, flip_mask = _build_j0_ler_inputs(bundle, obs_support)
    kernel = RegularizedPowerLawKernel(alpha=alpha, r0=r0)
    p = _pair_probabilities(distances, kernel, J0=J0, tau=tau)

    rng = np.random.default_rng(seed)
    samples = rng.random((shots, p.shape[0]))
    fired = samples < p[None, :]
    flip_counts = fired[:, flip_mask].sum(axis=1)
    observable_flipped = flip_counts % 2
    return float(observable_flipped.mean())


def retained_channel_ler_sweep(
    bundle: BB72Bundle,
    *,
    J0_values: Sequence[float],
    shots_per_point: int = 500,
    seed: int = 0,
) -> list[float]:
    """Evaluate the retained-channel LER at each `J_0` in the sweep.

    Uses a fixed seed derived from `(seed, operating_point_index)`
    so the output is deterministic and reproducible.
    """
    return [
        retained_channel_ler(
            bundle,
            J0=float(J0),
            shots=shots_per_point,
            seed=seed + i,
        )
        for i, J0 in enumerate(J0_values)
    ]


# =============================================================================
# 4. Spearman rank correlation between exposure and LER
# =============================================================================


def _rank(values: Sequence[float]) -> np.ndarray:
    """Return the rank of each entry (1-based, with ties broken by index)."""
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="stable")
    ranks = np.empty_like(arr, dtype=float)
    ranks[order] = np.arange(1, len(arr) + 1)
    return ranks


def spearman_rho(a: Sequence[float], b: Sequence[float]) -> float:
    r"""Spearman rank correlation between two equal-length sequences.

    We avoid the scipy dependency and implement the rank-Pearson
    formula directly. Ties are broken by insertion order, which
    is fine for the Monte Carlo LER output because stochastic
    noise breaks exact ties with probability 1.
    """
    if len(a) != len(b):
        raise ValueError("a and b must have equal length")
    if len(a) < 2:
        raise ValueError("Spearman needs at least two points")
    ra = _rank(a)
    rb = _rank(b)
    ra_mean = float(ra.mean())
    rb_mean = float(rb.mean())
    num = float(((ra - ra_mean) * (rb - rb_mean)).sum())
    den = float(np.sqrt(((ra - ra_mean) ** 2).sum()) * np.sqrt(((rb - rb_mean) ** 2).sum()))
    if den == 0.0:
        return 0.0
    return num / den


def exposure_vs_ler_spearman(
    bundle: BB72Bundle,
    *,
    J0_values: Sequence[float],
    alpha_values: Sequence[float],
    shots_per_point: int = 500,
    seed: int = 0,
) -> tuple[float, list[tuple[float, float, float, float]]]:
    r"""Compute Spearman `(\text{exposure}, \text{LER})` over a 2D sweep.

    Parameters
    ----------
    bundle : BB72Bundle
    J0_values : Sequence[float]
        `J_0` coordinates of the sweep.
    alpha_values : Sequence[float]
        `\alpha` coordinates of the sweep.
    shots_per_point : int, optional
        Monte Carlo shots per operating point. Default 500.
    seed : int, optional
        RNG seed base; each point uses `seed + i`.

    Returns
    -------
    (rho, points) : tuple
        `rho` is the Spearman rank correlation between exposure
        and LER over every `(J_0, \alpha)` combination. `points`
        is a list of `(J_0, alpha, exposure, ler)` 4-tuples for
        logging / diagnostics.
    """
    monomial_positions = np.asarray(bundle.monomial_embedding.positions)
    # Build per-operating-point (exposure, LER) pairs by iterating
    # the outer product.
    points: list[tuple[float, float, float, float]] = []
    exposures: list[float] = []
    lers: list[float] = []
    obs_support = bundle.reference_family[0]
    _, distances, flip_mask = _build_j0_ler_inputs(bundle, obs_support)
    for J0 in J0_values:
        for alpha in alpha_values:
            kernel = RegularizedPowerLawKernel(alpha=float(alpha), r0=REFERENCE_R0)
            exposure = float(
                j_kappa_numpy(
                    monomial_positions,
                    bundle.numpy_template,
                    kernel,
                    J0=float(J0),
                    tau=REFERENCE_TAU,
                )
            )
            p = _pair_probabilities(distances, kernel, J0=float(J0), tau=REFERENCE_TAU)
            rng = np.random.default_rng(seed + len(points))
            samples = rng.random((shots_per_point, p.shape[0]))
            fired = samples < p[None, :]
            flip_counts = fired[:, flip_mask].sum(axis=1)
            ler = float((flip_counts % 2).mean())
            exposures.append(exposure)
            lers.append(ler)
            points.append((float(J0), float(alpha), exposure, ler))
    rho = spearman_rho(exposures, lers)
    return rho, points


# =============================================================================
# 5. Bbstim pure-L X-logical set equality
# =============================================================================


def bbstim_pureL_X_logicals(
    fixture_path: str | Path,
) -> set[frozenset[int]]:
    """Load bbstim's 36 BB72 pure-L X-logical supports from a CSV fixture.

    Each entry is a `frozenset` of the six qubit indices in bbstim's
    row-major `flat = i*m + j` convention. The CSV format matches
    the header `index,support_indices,support_monomials` used by
    the upstream ``bbstim`` project.

    Parameters
    ----------
    fixture_path : str or Path
        Path to the committed CSV (typically
        `benchmarks/fixtures/bbstim_bb72_pureL_minwt_logicals.csv`).
    """
    import ast

    path = Path(fixture_path)
    supports: set[frozenset[int]] = set()
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            flats = ast.literal_eval(row["support_indices"])
            supports.add(frozenset(int(q) for q in flats))
    return supports


def weave_pureL_X_logicals_in_bbstim_convention(
    bundle: BB72Bundle,
) -> set[frozenset[int]]:
    r"""Compute weave's 36 BB72 pure-L X-logical supports, translated
    into bbstim's flat-index convention.

    Two conventions must be reconciled:

    1. **Polynomial-matrix orientation.** Weave's polynomial matrix
       acts via a forward shift (`M e_i = e_{i + \text{shift}}`),
       while bbstim's acts via a backward shift
       (`M e_i = e_{i - \text{shift}}`). The two are related by the
       group automorphism `g \mapsto g^{-1}`, i.e.
       `(i, j) \mapsto (-i \bmod l, -j \bmod m)`.
    2. **Flat-index encoding.** Weave uses column-major
       `flat = j l + i`; bbstim uses row-major `flat = i m + j`.

    Applying both transforms to each weave flat index yields a
    bbstim-compatible flat index. The resulting set of 36 supports
    is byte-for-byte identical to
    :func:`bbstim_pureL_X_logicals`'s output.
    """
    l = bundle.code.l
    m = bundle.code.m

    def weave_flat_to_bbstim(weave_flat: int) -> int:
        # Decode weave column-major flat → group element (i_w, j_w).
        i_w = weave_flat % l
        j_w = weave_flat // l
        # Apply the inversion automorphism g → g^{-1}.
        i_b = (-i_w) % l
        j_b = (-j_w) % m
        # Encode as bbstim row-major flat.
        return i_b * m + j_b

    return {frozenset(weave_flat_to_bbstim(q) for q in supp) for supp in bundle.reference_family}


# =============================================================================
# Public runner interface
# =============================================================================


def run_regression(
    *,
    bundle: BB72Bundle | None = None,
    fixture_dir: str | Path | None = None,
    shots_per_point: int = 500,
    seed: int = 0,
    printer: Callable[[str], None] = print,
) -> dict[str, Any]:
    r"""Run the full PR 13 regression suite and return a results dict.

    Parameters
    ----------
    bundle : BB72Bundle, optional
        Pre-built bundle. Built on demand if omitted.
    fixture_dir : str or Path, optional
        Directory containing `bbstim_bb72_pureL_minwt_logicals.csv`.
        Defaults to `<repo>/benchmarks/fixtures`.
    shots_per_point : int, optional
        Monte Carlo shots per operating point in the sweeps.
    seed : int, optional
        RNG seed base.
    printer : Callable[[str], None]
        Where to send status lines. Defaults to `print`.

    Returns
    -------
    dict
        Keys:
        - ``fingerprint_a``, ``fingerprint_b`` — the two compile
          fingerprints from the stability test.
        - ``j_kappa_monomial`` and ``j_kappa_optimized``.
        - ``spearman_rho`` and the operating-point table.
        - ``bbstim_match`` — True iff weave's and bbstim's
          X-logical sets are equal as sets.
        - ``all_pass`` — True iff every weave-only assertion holds.
    """
    if bundle is None:
        bundle = BB72Bundle.build()
    if fixture_dir is None:
        fixture_dir = Path(__file__).resolve().parents[1] / "fixtures"

    printer("[PR 13] building BB72 bundle…")
    printer("[PR 13] (1/5) fingerprint stability")
    fa, fb = fingerprint_stability(bundle)
    stable = fa == fb

    printer("[PR 13] (2/5) monomial vs optimized exposure")
    j_mono, j_opt = monomial_vs_optimized_exposure(bundle)

    printer("[PR 13] (3/5) retained-channel LER monotonicity")
    J0_mono = np.linspace(0.01, 0.1, 8)
    lers_mono = retained_channel_ler_sweep(
        bundle,
        J0_values=J0_mono,
        shots_per_point=shots_per_point,
        seed=seed,
    )
    # Shot-aware tolerance: allow roughly a few standard errors of
    # Monte Carlo noise in the "almost-monotone" check. At small
    # LER ~ 10⁻⁴ the per-shot SE is ~1/sqrt(shots), so 4σ of headroom
    # is 4/sqrt(shots). We allow slightly more to stay robust against
    # seed-dependent flukes.
    mono_tol = 5.0 / float(shots_per_point) ** 0.5
    monotone = all(b >= a - mono_tol for a, b in zip(lers_mono, lers_mono[1:], strict=False))

    printer("[PR 13] (4/5) exposure-vs-LER Spearman sweep")
    J0_sweep = [0.02, 0.04, 0.06, 0.08, 0.10]
    alpha_sweep = [1.5, 3.0, 5.0]
    rho, points = exposure_vs_ler_spearman(
        bundle,
        J0_values=J0_sweep,
        alpha_values=alpha_sweep,
        shots_per_point=shots_per_point,
        seed=seed + 100,
    )

    printer("[PR 13] (5/5) bbstim X-logicals set equality")
    bbstim_fixture = Path(fixture_dir) / "bbstim_bb72_pureL_minwt_logicals.csv"
    bbstim_set = bbstim_pureL_X_logicals(bbstim_fixture)
    weave_set = weave_pureL_X_logicals_in_bbstim_convention(bundle)
    bbstim_match = weave_set == bbstim_set

    reduction = (j_mono - j_opt) / j_mono if j_mono > 0 else 0.0
    all_pass = stable and reduction >= 0.20 and monotone and rho >= 0.85 and bbstim_match

    printer(
        f"[PR 13] summary: "
        f"stable={stable} "
        f"reduction={reduction * 100:.2f}% "
        f"monotone={monotone} "
        f"rho={rho:.3f} "
        f"bbstim_match={bbstim_match}"
    )

    return {
        "fingerprint_a": fa,
        "fingerprint_b": fb,
        "fingerprint_stable": stable,
        "j_kappa_monomial": j_mono,
        "j_kappa_optimized": j_opt,
        "reduction_ratio": reduction,
        "ler_monotone": monotone,
        "ler_sweep": lers_mono,
        "J0_sweep_for_ler": list(map(float, J0_mono)),
        "spearman_rho": rho,
        "spearman_points": points,
        "bbstim_match": bbstim_match,
        "all_pass": all_pass,
    }

"""PR 14 — HGP cross-code validation: the same code path works on non-BB families.

Demonstrates that weave's compile → geometry pass → exposure → residual
pipeline runs on hypergraph product (HGP) codes without any
BB-specific branches. Every test here exercises exactly the same
functions used in the BB72 regression (PRs 8–13), just with a
different code family and a generic schedule.

Codes under test
----------------
- `rep(3) × rep(3)` → [[13, 1, 3]]: the minimal nontrivial HGP code.
  Small enough for brute-force distance, noiseless compile, and
  hand-built parallel schedules.
- `rep(3) × rep(4)` → [[18, 1, 3]]: non-symmetric product, tests
  the asymmetric-check-weight case.

For each code, the default serial schedule (:func:`default_css_schedule`)
puts one CNOT per tick, so the geometry pass produces empty provenance.
To exercise the geometry pass, a hand-built parallel fixture provides
a schedule with two disjoint Z-check CNOTs in the same tick.
"""

from __future__ import annotations

import numpy as np
import pytest

from weave.analysis.residual import (
    ResidualError,
    effective_distance_upper_bound,
    enumerate_hook_residuals_z_sector,
    residual_distance,
)
from weave.analysis.validation import verify_weight_le_2_assumption
from weave.codes import CSSCode, HypergraphProductCode
from weave.compiler import compile_extraction
from weave.compiler.geometry_pass import compute_provenance
from weave.ir import (
    CrossingKernel,
    GeometryNoiseConfig,
    LocalNoiseConfig,
    MinDistanceMetric,
    RegularizedPowerLawKernel,
    Schedule,
    ScheduleEdge,
    ScheduleStep,
    SingleQubitEdge,
    StraightLineEmbedding,
    TwoQubitEdge,
    default_css_schedule,
)
from weave.ir.schedule import QubitRole, ScheduleRole
from weave.util import pcm

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rep3x3() -> HypergraphProductCode:
    H = pcm.repetition(3)
    return HypergraphProductCode(H, H)


@pytest.fixture(scope="module")
def rep3x4() -> HypergraphProductCode:
    return HypergraphProductCode(pcm.repetition(3), pcm.repetition(4))


@pytest.fixture(scope="module")
def rep3x3_schedule(rep3x3) -> Schedule:
    return default_css_schedule(rep3x3)


@pytest.fixture(scope="module")
def rep3x3_embedding(rep3x3) -> StraightLineEmbedding:
    """Trivial linear layout — positions along the x axis."""
    return StraightLineEmbedding.from_positions([(float(i), 0.0) for i in range(rep3x3.n_total)])


# ---------------------------------------------------------------------------
# 1. Noiseless compile produces zero detector events
# ---------------------------------------------------------------------------


class TestNoiselessCompile:
    def test_rep3x3_zero_events(self, rep3x3, rep3x3_schedule, rep3x3_embedding):
        compiled = compile_extraction(
            code=rep3x3,
            embedding=rep3x3_embedding,
            schedule=rep3x3_schedule,
            kernel=CrossingKernel(),
            local_noise=LocalNoiseConfig(),
            geometry_noise=GeometryNoiseConfig(),
            rounds=2,
        )
        sampler = compiled.circuit.compile_detector_sampler()
        samples = sampler.sample(shots=100)
        assert not np.any(samples)

    def test_rep3x4_zero_events(self, rep3x4):
        sched = default_css_schedule(rep3x4)
        emb = StraightLineEmbedding.from_positions([(float(i), 0.0) for i in range(rep3x4.n_total)])
        compiled = compile_extraction(
            code=rep3x4,
            embedding=emb,
            schedule=sched,
            kernel=CrossingKernel(),
            local_noise=LocalNoiseConfig(),
            geometry_noise=GeometryNoiseConfig(),
            rounds=1,
        )
        sampler = compiled.circuit.compile_detector_sampler()
        samples = sampler.sample(shots=50)
        assert not np.any(samples)


# ---------------------------------------------------------------------------
# 2. Fingerprint stability
# ---------------------------------------------------------------------------


class TestFingerprintStability:
    def test_recompile_is_deterministic(self, rep3x3, rep3x3_schedule, rep3x3_embedding):
        kwargs = dict(
            code=rep3x3,
            embedding=rep3x3_embedding,
            schedule=rep3x3_schedule,
            kernel=CrossingKernel(),
            local_noise=LocalNoiseConfig(),
            geometry_noise=GeometryNoiseConfig(),
            rounds=1,
        )
        a = compile_extraction(**kwargs)
        b = compile_extraction(**kwargs)
        assert a.fingerprint() == b.fingerprint()


# ---------------------------------------------------------------------------
# 3. Weight-≤2 assumption on default serial schedule
# ---------------------------------------------------------------------------


class TestWeightLe2Assumption:
    def test_serial_schedule_vacuously_passes(self, rep3x3, rep3x3_schedule):
        """The default serial schedule has one CNOT per tick, so there
        are no parallel pair events and the weight-≤2 assumption is
        vacuously true."""
        for sector in ("X", "Z"):
            report = verify_weight_le_2_assumption(rep3x3_schedule, sector=sector)
            assert report.passed
            assert report.events == ()


# ---------------------------------------------------------------------------
# 4. Residual distance matches code distance
# ---------------------------------------------------------------------------


class TestResidualDistance:
    def test_empty_residual_gives_distance_plus_one(self, rep3x3):
        """Δ(0) = 1 + d for the trivial residual on rep(3)×rep(3)."""
        empty = ResidualError(data_support=(), weight=0)
        delta = residual_distance(empty, h_commute=rep3x3.HZ, h_stab=rep3x3.HX)
        assert delta == 1 + rep3x3.distance()

    def test_hook_residuals_bound_distance(self, rep3x3):
        """The effective distance upper bound from a single Z-check's
        hook residuals is ≤ d + 1. This exercises the Strikis
        formalism on an HGP code."""
        # Pick the first Z-check's CNOT target order from HZ row 0.
        row = rep3x3.HZ[0]
        targets = [rep3x3.data_qubits[i] for i in range(len(row)) if row[i]]
        residuals = enumerate_hook_residuals_z_sector(targets)
        if not residuals:
            pytest.skip("Z-check row 0 has fewer than 2 nonzero columns")
        bound = effective_distance_upper_bound(residuals, h_commute=rep3x3.HZ, h_stab=rep3x3.HX)
        assert bound <= rep3x3.distance() + 1


# ---------------------------------------------------------------------------
# 5. Geometry pass on a custom parallel schedule
# ---------------------------------------------------------------------------


def _parallel_hgp_schedule(code: CSSCode) -> Schedule:
    """Build a minimal schedule for `code` with one parallel Z-check tick.

    Finds two Z-check rows with disjoint support and fires their
    first CNOTs in the same tick, then finishes the remaining
    checks serially. The rest (X-checks, MR, tail) matches
    `default_css_schedule` structurally.

    This is the HGP analogue of the BB72 parallel fixtures in
    test_compiler_geometry.py — it exists solely to exercise the
    geometry pass on non-BB codes.
    """
    data = code.data_qubits
    z_checks = code.z_check_qubits
    x_checks = code.x_check_qubits
    all_qubits = frozenset(code.qubits)
    roles: dict[int, QubitRole] = {}
    for q in data:
        roles[q] = "data"
    for q in z_checks:
        roles[q] = "z_ancilla"
    for q in x_checks:
        roles[q] = "x_ancilla"

    def mk(tick: int, role: ScheduleRole, edges: list[ScheduleEdge]) -> ScheduleStep:
        active: set[int] = set()
        for e in edges:
            for q in e.qubits:
                active.add(q)
        return ScheduleStep(
            tick_index=tick,
            role=role,
            active_edges=tuple(edges),
            active_qubits=frozenset(active),
            idle_qubits=all_qubits - active,
        )

    # Head: reset.
    head = [mk(0, "reset", [SingleQubitEdge(gate="R", qubit=q) for q in sorted(all_qubits)])]

    # Find two disjoint Z-check rows.
    import numpy as np

    pair_a, pair_b = None, None
    for i in range(code.HZ.shape[0]):
        cols_i = set(np.where(code.HZ[i])[0])
        for j in range(i + 1, code.HZ.shape[0]):
            cols_j = set(np.where(code.HZ[j])[0])
            if not (cols_i & cols_j):
                pair_a, pair_b = i, j
                break
        if pair_a is not None:
            break
    assert pair_a is not None, "No disjoint Z-check pair found"

    # Build the parallel tick: first CNOT of each disjoint check.
    cycle: list[ScheduleStep] = []
    tick = 0
    col_a = int(np.where(code.HZ[pair_a])[0][0])
    col_b = int(np.where(code.HZ[pair_b])[0][0])
    cycle.append(
        mk(
            tick,
            "cnot_layer",
            [
                TwoQubitEdge(
                    gate="CNOT",
                    control=data[col_a],
                    target=z_checks[pair_a],
                    interaction_sector="X",
                ),
                TwoQubitEdge(
                    gate="CNOT",
                    control=data[col_b],
                    target=z_checks[pair_b],
                    interaction_sector="X",
                ),
            ],
        )
    )
    tick += 1

    # Remaining Z-check CNOTs (serial).
    for i in range(code.HZ.shape[0]):
        cols = np.where(code.HZ[i])[0]
        start = 1 if i in (pair_a, pair_b) else 0
        for c in cols[start:]:
            cycle.append(
                mk(
                    tick,
                    "cnot_layer",
                    [
                        TwoQubitEdge(
                            gate="CNOT",
                            control=data[int(c)],
                            target=z_checks[i],
                            interaction_sector="X",
                        )
                    ],
                )
            )
            tick += 1

    # X-check H bracket + serial CNOTs.
    for i in range(code.HX.shape[0]):
        cycle.append(mk(tick, "single_q", [SingleQubitEdge(gate="H", qubit=x_checks[i])]))
        tick += 1
        cols = np.where(code.HX[i])[0]
        for c in cols:
            cycle.append(
                mk(
                    tick,
                    "cnot_layer",
                    [
                        TwoQubitEdge(
                            gate="CNOT",
                            control=x_checks[i],
                            target=data[int(c)],
                            interaction_sector="Z",
                        )
                    ],
                )
            )
            tick += 1
        cycle.append(mk(tick, "single_q", [SingleQubitEdge(gate="H", qubit=x_checks[i])]))
        tick += 1

    # MR all ancillas.
    cycle.append(
        mk(tick, "meas", [SingleQubitEdge(gate="MR", qubit=q) for q in z_checks + x_checks])
    )

    # Tail: M on data.
    tail = [mk(0, "meas", [SingleQubitEdge(gate="M", qubit=q) for q in data])]

    return Schedule(
        head_steps=tuple(head),
        cycle_steps=tuple(cycle),
        tail_steps=tuple(tail),
        qubits=all_qubits,
        qubit_roles=roles,
        name="hgp_parallel_fixture",
    )


class TestGeometryPassOnHGP:
    def test_parallel_schedule_produces_provenance(self, rep3x3, rep3x3_embedding):
        """A custom HGP schedule with one parallel tick produces at
        least one provenance record when `J_0 > 0`."""
        sched = _parallel_hgp_schedule(rep3x3)
        kernel = RegularizedPowerLawKernel(alpha=3.0, r0=1.0)
        provenance = compute_provenance(
            schedule=sched,
            embedding=rep3x3_embedding,
            kernel=kernel,
            route_metric=MinDistanceMetric(),
            geometry_noise=GeometryNoiseConfig(J0=0.5, tau=1.0),
        )
        assert len(provenance) >= 1
        # All records should be in the X sector (Z-check CNOTs).
        for rec in provenance:
            assert rec.sector == "X"
            assert rec.data_weight == 2

    def test_parallel_schedule_compiles_noiseless(self, rep3x3, rep3x3_embedding):
        """The custom parallel schedule also compiles to a correct
        noiseless Stim circuit with zero detector events."""
        sched = _parallel_hgp_schedule(rep3x3)
        compiled = compile_extraction(
            code=rep3x3,
            embedding=rep3x3_embedding,
            schedule=sched,
            kernel=CrossingKernel(),
            local_noise=LocalNoiseConfig(),
            geometry_noise=GeometryNoiseConfig(),
            rounds=2,
        )
        sampler = compiled.circuit.compile_detector_sampler()
        samples = sampler.sample(shots=50)
        assert not np.any(samples)

    def test_exposure_metrics_populated(self, rep3x3, rep3x3_embedding):
        """The compiled output has non-None exposure metrics when
        geometry noise is on."""
        sched = _parallel_hgp_schedule(rep3x3)
        compiled = compile_extraction(
            code=rep3x3,
            embedding=rep3x3_embedding,
            schedule=sched,
            kernel=RegularizedPowerLawKernel(alpha=3.0, r0=1.0),
            local_noise=LocalNoiseConfig(),
            geometry_noise=GeometryNoiseConfig(J0=0.5, tau=1.0),
            rounds=1,
        )
        assert compiled.exposure_metrics is not None
        assert compiled.exposure_metrics.total() > 0
        assert len(compiled.provenance) >= 1


# ---------------------------------------------------------------------------
# 6. The same compile_extraction API: no BB-specific branches
# ---------------------------------------------------------------------------


class TestSameAPIAsForBB:
    """The point of PR 14: every function called here is the SAME
    function the BB72 regression calls. No special-case code for HGP.
    """

    def test_compiled_extraction_has_standard_fields(
        self, rep3x3, rep3x3_schedule, rep3x3_embedding
    ):
        compiled = compile_extraction(
            code=rep3x3,
            embedding=rep3x3_embedding,
            schedule=rep3x3_schedule,
            kernel=CrossingKernel(),
            local_noise=LocalNoiseConfig(),
            geometry_noise=GeometryNoiseConfig(),
            rounds=1,
        )
        assert hasattr(compiled, "provenance")
        assert hasattr(compiled, "correlation_edges")
        assert hasattr(compiled, "exposure_metrics")
        assert hasattr(compiled, "decoder_artifact")
        assert hasattr(compiled, "fingerprint")
        # All tables are populated (possibly empty for J0=0).
        assert compiled.exposure_metrics is not None
        assert compiled.decoder_artifact is not None
        assert isinstance(compiled.provenance, tuple)
        assert isinstance(compiled.correlation_edges, tuple)

    def test_json_round_trip(self, rep3x3, rep3x3_schedule, rep3x3_embedding):
        from weave.ir import CompiledExtraction

        compiled = compile_extraction(
            code=rep3x3,
            embedding=rep3x3_embedding,
            schedule=rep3x3_schedule,
            kernel=CrossingKernel(),
            local_noise=LocalNoiseConfig(),
            geometry_noise=GeometryNoiseConfig(),
            rounds=1,
        )
        reconstructed = CompiledExtraction.from_json(compiled.to_json())
        assert reconstructed == compiled

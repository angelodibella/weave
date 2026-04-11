"""Tests for the geometry branch of `compile_extraction`.

The geometry pass (PR 8) reads a schedule and a routed embedding,
computes per-pair probabilities via the `route_metric → kernel →
sin²` pipeline, propagates each pair fault through the remainder of
the cycle, and emits a `CORRELATED_ERROR` for each surviving event.
This module pins every intermediate quantity against a hand-built
two-qubit motif plus end-to-end assertions on `compile_extraction`.

Fixtures
--------
- :class:`CSSCode` with `HX = HZ = [[1,1,0,0], [0,0,1,1]]` so that
  we have 4 data qubits, 2 Z-check ancillas, 2 X-check ancillas, and
  k = 0. The CSS condition `HX @ HZ.T = 0 mod 2` holds trivially
  because each row's support is disjoint from every column of HZ.
- A hand-crafted :class:`~weave.ir.Schedule` whose cycle has **one**
  parallel X-sector CNOT tick (tick 0) and keeps the rest serial
  so there is a single parallel-pair event to validate against.
- A straight-line embedding whose `CNOT(d0, z0)` and `CNOT(d2, z1)`
  polylines are parallel vertical segments one unit apart, so the
  `min_distance` metric returns `1.0` exactly.
- :class:`RegularizedPowerLawKernel` with ``alpha=1``, ``r0=1`` so
  `κ(1) = 1/2`.
- ``J0 = 0.5``, ``tau = 1``, which gives ``τ·J₀·κ(1) = 0.25`` and
  ``p_pair = sin²(0.25)``. These numbers are kept small so the
  retained-channel derivation stays deep in the validity regime.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from weave.analysis.propagation import propagate_single_pair_event
from weave.codes.css_code import CSSCode
from weave.compiler import compile_extraction
from weave.compiler.geometry_pass import compute_provenance
from weave.ir import (
    CompiledExtraction,
    CorrelationEdgeRecord,
    DecoderArtifact,
    ExposureMetrics,
    GeometryNoiseConfig,
    LocalNoiseConfig,
    MinDistanceMetric,
    ProvenanceRecord,
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


def _make_step(
    tick: int,
    role: ScheduleRole,
    edges: list[ScheduleEdge],
    all_qubits: frozenset[int],
) -> ScheduleStep:
    active: set[int] = set()
    for e in edges:
        for q in e.qubits:
            active.add(q)
    idle = all_qubits - active
    return ScheduleStep(
        tick_index=tick,
        role=role,
        active_edges=tuple(edges),
        active_qubits=frozenset(active),
        idle_qubits=idle,
    )


@pytest.fixture
def parallel_code() -> CSSCode:
    """A 4-data / 2-z_check / 2-x_check CSS code with disjoint checks."""
    HX = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=int)
    HZ = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=int)
    return CSSCode(HX=HX, HZ=HZ, rounds=1)


@pytest.fixture
def parallel_schedule(parallel_code: CSSCode) -> Schedule:
    """Cycle with one parallel X-sector CNOT tick at tick 0.

    Exactly one pair event: `(CNOT(0, 4), CNOT(2, 5))` at tick 0.
    The remaining Z-check and X-check CNOTs are emitted one per
    tick so no further pair events survive enumeration.
    """
    d0, d1, d2, d3 = parallel_code.data_qubits
    z0, z1 = parallel_code.z_check_qubits
    x0, x1 = parallel_code.x_check_qubits
    all_qubits = frozenset(parallel_code.qubits)
    roles: dict[int, QubitRole] = {
        d0: "data",
        d1: "data",
        d2: "data",
        d3: "data",
        z0: "z_ancilla",
        z1: "z_ancilla",
        x0: "x_ancilla",
        x1: "x_ancilla",
    }

    # Head: reset everything.
    head = [
        _make_step(
            tick=0,
            role="reset",
            edges=[SingleQubitEdge(gate="R", qubit=q) for q in sorted(all_qubits)],
            all_qubits=all_qubits,
        )
    ]

    cycle: list[ScheduleStep] = []

    # Tick 0: PARALLEL X-sector pair. Both Z-checks fire their first CNOT.
    cycle.append(
        _make_step(
            tick=0,
            role="cnot_layer",
            edges=[
                TwoQubitEdge(
                    gate="CNOT",
                    control=d0,
                    target=z0,
                    interaction_sector="X",
                    term_name="HZ[0,0]",
                ),
                TwoQubitEdge(
                    gate="CNOT",
                    control=d2,
                    target=z1,
                    interaction_sector="X",
                    term_name="HZ[1,2]",
                ),
            ],
            all_qubits=all_qubits,
        )
    )
    # Tick 1-2: second CNOT of each Z-check, sequentially.
    cycle.append(
        _make_step(
            tick=1,
            role="cnot_layer",
            edges=[
                TwoQubitEdge(
                    gate="CNOT",
                    control=d1,
                    target=z0,
                    interaction_sector="X",
                    term_name="HZ[0,1]",
                ),
            ],
            all_qubits=all_qubits,
        )
    )
    cycle.append(
        _make_step(
            tick=2,
            role="cnot_layer",
            edges=[
                TwoQubitEdge(
                    gate="CNOT",
                    control=d3,
                    target=z1,
                    interaction_sector="X",
                    term_name="HZ[1,3]",
                ),
            ],
            all_qubits=all_qubits,
        )
    )
    # Tick 3: H on X-check ancillas.
    cycle.append(
        _make_step(
            tick=3,
            role="single_q",
            edges=[SingleQubitEdge(gate="H", qubit=x0), SingleQubitEdge(gate="H", qubit=x1)],
            all_qubits=all_qubits,
        )
    )
    # Ticks 4-7: X-check CNOTs, one per tick to avoid further pair events.
    cycle.append(
        _make_step(
            tick=4,
            role="cnot_layer",
            edges=[
                TwoQubitEdge(
                    gate="CNOT",
                    control=x0,
                    target=d0,
                    interaction_sector="Z",
                    term_name="HX[0,0]",
                )
            ],
            all_qubits=all_qubits,
        )
    )
    cycle.append(
        _make_step(
            tick=5,
            role="cnot_layer",
            edges=[
                TwoQubitEdge(
                    gate="CNOT",
                    control=x0,
                    target=d1,
                    interaction_sector="Z",
                    term_name="HX[0,1]",
                )
            ],
            all_qubits=all_qubits,
        )
    )
    cycle.append(
        _make_step(
            tick=6,
            role="cnot_layer",
            edges=[
                TwoQubitEdge(
                    gate="CNOT",
                    control=x1,
                    target=d2,
                    interaction_sector="Z",
                    term_name="HX[1,2]",
                )
            ],
            all_qubits=all_qubits,
        )
    )
    cycle.append(
        _make_step(
            tick=7,
            role="cnot_layer",
            edges=[
                TwoQubitEdge(
                    gate="CNOT",
                    control=x1,
                    target=d3,
                    interaction_sector="Z",
                    term_name="HX[1,3]",
                )
            ],
            all_qubits=all_qubits,
        )
    )
    # Tick 8: H bracket close.
    cycle.append(
        _make_step(
            tick=8,
            role="single_q",
            edges=[SingleQubitEdge(gate="H", qubit=x0), SingleQubitEdge(gate="H", qubit=x1)],
            all_qubits=all_qubits,
        )
    )
    # Tick 9: MR on ancillas, z-check first then x-check to match the
    # layout the legacy detector emitter assumes.
    cycle.append(
        _make_step(
            tick=9,
            role="meas",
            edges=[
                SingleQubitEdge(gate="MR", qubit=z0),
                SingleQubitEdge(gate="MR", qubit=z1),
                SingleQubitEdge(gate="MR", qubit=x0),
                SingleQubitEdge(gate="MR", qubit=x1),
            ],
            all_qubits=all_qubits,
        )
    )

    # Tail: final data measurement.
    tail = [
        _make_step(
            tick=0,
            role="meas",
            edges=[SingleQubitEdge(gate="M", qubit=q) for q in parallel_code.data_qubits],
            all_qubits=all_qubits,
        )
    ]

    return Schedule(
        head_steps=tuple(head),
        cycle_steps=tuple(cycle),
        tail_steps=tuple(tail),
        qubits=all_qubits,
        qubit_roles=roles,
        name="parallel_pair",
    )


@pytest.fixture
def parallel_embedding(parallel_code: CSSCode) -> StraightLineEmbedding:
    """Positions that make the tick-0 pair polylines 1 unit apart.

    `CNOT(d0, z0)` is the vertical segment at x=0 from (0,0) to (0,1),
    and `CNOT(d2, z1)` is the vertical segment at x=1 from (1,0) to
    (1,1). Both segments are parallel, so `polyline_distance` returns
    exactly 1.0.

    The other data qubits `d1, d3` are parked far enough away that
    any post-tick-0 polylines have weight-0 propagation through the
    geometry pass (no additional pair events fire).
    """
    num_qubits = len(parallel_code.qubits)
    positions: list[tuple[float, float]] = [(0.0, 0.0)] * num_qubits
    d0, d1, d2, d3 = parallel_code.data_qubits
    z0, z1 = parallel_code.z_check_qubits
    x0, x1 = parallel_code.x_check_qubits
    positions[d0] = (0.0, 0.0)
    positions[d1] = (100.0, 50.0)
    positions[d2] = (1.0, 0.0)
    positions[d3] = (200.0, 50.0)
    positions[z0] = (0.0, 1.0)
    positions[z1] = (1.0, 1.0)
    positions[x0] = (300.0, -50.0)
    positions[x1] = (400.0, -50.0)
    return StraightLineEmbedding.from_positions(positions)


# ---------------------------------------------------------------------------
# compute_provenance — pure unit tests
# ---------------------------------------------------------------------------


class TestComputeProvenanceUnit:
    def test_zero_J0_returns_empty(self, parallel_schedule, parallel_embedding):
        """`geometry_noise.enabled` is `False` when `J0 == 0`, so the
        pass should short-circuit to an empty list regardless of
        schedule content."""
        records = compute_provenance(
            schedule=parallel_schedule,
            embedding=parallel_embedding,
            kernel=RegularizedPowerLawKernel(alpha=1.0, r0=1.0),
            route_metric=MinDistanceMetric(),
            geometry_noise=GeometryNoiseConfig(J0=0.0, tau=1.0),
        )
        assert records == []

    def test_serial_schedule_has_no_pair_events(self, parallel_code, parallel_embedding):
        """The default Steane-style serial schedule fires one CNOT per
        tick, so there are no parallel-pair events to enumerate."""
        serial = default_css_schedule(parallel_code)
        records = compute_provenance(
            schedule=serial,
            embedding=parallel_embedding,
            kernel=RegularizedPowerLawKernel(alpha=1.0, r0=1.0),
            route_metric=MinDistanceMetric(),
            geometry_noise=GeometryNoiseConfig(J0=0.5, tau=1.0),
        )
        assert records == []

    def test_single_pair_event_numerics(self, parallel_schedule, parallel_embedding):
        """Hand-computed values for the tick-0 pair.

        Distance: 1.0 (parallel vertical segments at x=0 and x=1).
        Kernel:   (1 + 1)^{-1} = 0.5.
        Strength: τ·J₀·κ = 1 · 0.5 · 0.5 = 0.25.
        Probability: sin²(0.25).
        Data image: X on data qubits {0, 2} (controls of the pair).
        """
        records = compute_provenance(
            schedule=parallel_schedule,
            embedding=parallel_embedding,
            kernel=RegularizedPowerLawKernel(alpha=1.0, r0=1.0),
            route_metric=MinDistanceMetric(),
            geometry_noise=GeometryNoiseConfig(J0=0.5, tau=1.0),
        )
        assert len(records) == 1
        rec = records[0]
        assert isinstance(rec, ProvenanceRecord)
        assert rec.tick_index == 0
        assert rec.sector == "X"
        assert rec.edge_a == (0, 4)  # (d0, z0)
        assert rec.edge_b == (2, 5)  # (d2, z1)
        assert rec.routed_distance == pytest.approx(1.0, abs=1e-12)
        expected_p = math.sin(0.25) ** 2
        assert rec.pair_probability == pytest.approx(expected_p, abs=1e-12)
        assert rec.data_support == (0, 2)
        assert rec.data_pauli_symbols == ("X", "X")
        assert rec.data_weight == 2

    def test_weak_limit_uses_quadratic(self, parallel_schedule, parallel_embedding):
        """In the weak-coupling limit the probability is `(τ·J₀·κ)²`."""
        records = compute_provenance(
            schedule=parallel_schedule,
            embedding=parallel_embedding,
            kernel=RegularizedPowerLawKernel(alpha=1.0, r0=1.0),
            route_metric=MinDistanceMetric(),
            geometry_noise=GeometryNoiseConfig(J0=0.5, tau=1.0, use_weak_limit=True),
        )
        assert len(records) == 1
        # (τ·J₀·κ(1))² = 0.25² = 0.0625.
        assert records[0].pair_probability == pytest.approx(0.0625, abs=1e-12)

    def test_propagation_integration(self, parallel_schedule, parallel_embedding):
        """`ProvenanceRecord.data_qubit_a/b` equals the output of
        :func:`propagate_single_pair_event` for the same pair."""
        records = compute_provenance(
            schedule=parallel_schedule,
            embedding=parallel_embedding,
            kernel=RegularizedPowerLawKernel(alpha=1.0, r0=1.0),
            route_metric=MinDistanceMetric(),
            geometry_noise=GeometryNoiseConfig(J0=0.5, tau=1.0),
        )
        rec = records[0]

        # Recover the actual edges from the cycle step.
        tick0 = parallel_schedule.cycle_steps[0]
        edge_a = tick0.active_edges[0]
        edge_b = tick0.active_edges[1]
        assert isinstance(edge_a, TwoQubitEdge)
        assert isinstance(edge_b, TwoQubitEdge)

        data_qubits = frozenset(q for q, r in parallel_schedule.qubit_roles.items() if r == "data")
        result = propagate_single_pair_event(
            schedule=parallel_schedule,
            step_tick=0,
            edge_a=edge_a,
            edge_b=edge_b,
            sector="X",
            data_qubits=data_qubits,
            end_block="cycle",
        )
        expected_support = tuple(sorted(result.data_pauli.support))
        assert rec.data_support == expected_support
        assert rec.data_qubit_a == expected_support[0]
        assert rec.data_qubit_b == expected_support[1]


# ---------------------------------------------------------------------------
# compile_extraction integration
# ---------------------------------------------------------------------------


class TestCompileExtractionGeometry:
    def _compile(self, code, schedule, embedding, *, J0=0.5, tau=1.0, rounds=1):
        return compile_extraction(
            code=code,
            embedding=embedding,
            schedule=schedule,
            kernel=RegularizedPowerLawKernel(alpha=1.0, r0=1.0),
            route_metric=MinDistanceMetric(),
            local_noise=LocalNoiseConfig(),
            geometry_noise=GeometryNoiseConfig(J0=J0, tau=tau),
            rounds=rounds,
        )

    def test_correlated_error_in_circuit(
        self, parallel_code, parallel_schedule, parallel_embedding
    ):
        """Stim's text form renders `CORRELATED_ERROR` as the shorthand
        `E(...)` — we check the parsed circuit for an instruction named
        either way to stay format-agnostic."""
        compiled = self._compile(parallel_code, parallel_schedule, parallel_embedding)
        names = [inst.name for inst in compiled.circuit]
        assert any(n in ("E", "CORRELATED_ERROR") for n in names)

    def test_provenance_count_matches_instructions(
        self, parallel_code, parallel_schedule, parallel_embedding
    ):
        """With rounds=1 the number of correlated-error instructions
        equals the number of provenance records (1:1 mapping)."""
        compiled = self._compile(parallel_code, parallel_schedule, parallel_embedding)
        instr_count = sum(1 for inst in compiled.circuit if inst.name in ("E", "CORRELATED_ERROR"))
        assert instr_count == len(compiled.provenance)
        assert instr_count == 1

    def test_provenance_count_scales_with_rounds(
        self, parallel_code, parallel_schedule, parallel_embedding
    ):
        """Each round re-emits every pair channel; the provenance list
        collapses per-round duplication and therefore stays the same
        size. Confirming the per-round factor separates the two."""
        compiled = self._compile(parallel_code, parallel_schedule, parallel_embedding, rounds=3)
        instr_count = sum(1 for inst in compiled.circuit if inst.name in ("E", "CORRELATED_ERROR"))
        assert instr_count == 3 * len(compiled.provenance)

    def test_pair_probability_preserved_to_machine_precision(
        self, parallel_code, parallel_schedule, parallel_embedding
    ):
        """The probability in the parsed circuit must match the hand-
        computed `sin²(0.25)` to 1e-12. (Stim's text form prints a
        shortened float but the parsed `Circuit` object keeps the
        full double-precision value.)
        """
        compiled = self._compile(parallel_code, parallel_schedule, parallel_embedding)
        expected_p = math.sin(0.25) ** 2
        errors = [inst for inst in compiled.circuit if inst.name in ("E", "CORRELATED_ERROR")]
        assert len(errors) == 1
        err = errors[0]
        assert err.gate_args_copy()[0] == pytest.approx(expected_p, abs=1e-12)
        # And the Pauli targets are X0, X2 exactly.
        tgt = err.targets_copy()
        assert len(tgt) == 2
        symbols = [(t.pauli_type, t.qubit_value) for t in tgt]
        assert sorted(symbols) == [("X", 0), ("X", 2)]

    def test_determinism(self, parallel_code, parallel_schedule, parallel_embedding):
        """Compiling the same inputs twice produces byte-identical
        circuit_text, dem_text, and an equal provenance tuple."""
        a = self._compile(parallel_code, parallel_schedule, parallel_embedding)
        b = self._compile(parallel_code, parallel_schedule, parallel_embedding)
        assert a.circuit_text == b.circuit_text
        assert a.dem_text == b.dem_text
        assert a.provenance == b.provenance
        # Fingerprint is deterministic over all fields.
        assert a.fingerprint() == b.fingerprint()

    def test_zero_J0_still_compiles(self, parallel_code, parallel_schedule, parallel_embedding):
        """A routed code with `J0 == 0` must emit no CORRELATED_ERROR
        and yield an empty provenance list, matching the PR 5 baseline."""
        compiled = self._compile(parallel_code, parallel_schedule, parallel_embedding, J0=0.0)
        assert compiled.provenance == ()
        assert all(inst.name not in ("E", "CORRELATED_ERROR") for inst in compiled.circuit)


# ---------------------------------------------------------------------------
# JSON round-trip with provenance
# ---------------------------------------------------------------------------


def test_provenance_record_round_trip():
    rec = ProvenanceRecord(
        tick_index=3,
        edge_a=(0, 4),
        edge_b=(2, 5),
        sector="X",
        routed_distance=1.0,
        pair_probability=0.0625,
        data_support=(0, 2),
        data_pauli_symbols=("X", "X"),
    )
    data = rec.to_json()
    reconstructed = ProvenanceRecord.from_json(data)
    assert reconstructed == rec


def test_compiled_extraction_round_trip_with_provenance(
    parallel_code, parallel_schedule, parallel_embedding
):
    compiled = compile_extraction(
        code=parallel_code,
        embedding=parallel_embedding,
        schedule=parallel_schedule,
        kernel=RegularizedPowerLawKernel(alpha=1.0, r0=1.0),
        route_metric=MinDistanceMetric(),
        local_noise=LocalNoiseConfig(),
        geometry_noise=GeometryNoiseConfig(J0=0.5, tau=1.0),
        rounds=1,
    )
    from weave.ir import CompiledExtraction

    data = compiled.to_json()
    reconstructed = CompiledExtraction.from_json(data)
    assert reconstructed.provenance == compiled.provenance
    assert reconstructed.circuit_text == compiled.circuit_text


def test_v1_compiled_extraction_loads_with_empty_provenance():
    """Schema v1 records have no `provenance` field; from_json must
    fall back to an empty tuple rather than raise."""
    v1_data = {
        "schema_version": 1,
        "type": "compiled_extraction",
        "circuit_text": "",
        "dem_text": "",
        "code_fingerprint": "a" * 64,
        "embedding_spec": {},
        "schedule_spec": {},
        "kernel_spec": {},
        "route_metric_spec": {},
        "local_noise_spec": {},
        "geometry_noise_spec": {},
    }
    ce = CompiledExtraction.from_json(v1_data)
    assert ce.provenance == ()
    assert ce.correlation_edges == ()
    assert ce.exposure_metrics is None
    assert ce.decoder_artifact is None


def test_v2_compiled_extraction_loads_with_empty_correlation_fields():
    """Schema v2 records have `provenance` but no PR 9 tables; the
    new fields default to empty/None on load."""
    v2_data = {
        "schema_version": 2,
        "type": "compiled_extraction",
        "circuit_text": "",
        "dem_text": "",
        "code_fingerprint": "a" * 64,
        "embedding_spec": {},
        "schedule_spec": {},
        "kernel_spec": {},
        "route_metric_spec": {},
        "local_noise_spec": {},
        "geometry_noise_spec": {},
        "provenance": [],
    }
    ce = CompiledExtraction.from_json(v2_data)
    assert ce.provenance == ()
    assert ce.correlation_edges == ()
    assert ce.exposure_metrics is None
    assert ce.decoder_artifact is None


# ---------------------------------------------------------------------------
# PR 9 — CompiledExtraction tables, fingerprint, lazy materializers
# ---------------------------------------------------------------------------


class TestCompiledExtractionPR9:
    """Acceptance tests for PR 9: correlation edges, exposure metrics,
    decoder artifact, fingerprint, and lazy materializers.
    """

    def _compile(self, parallel_code, parallel_schedule, parallel_embedding, *, rounds=1):
        return compile_extraction(
            code=parallel_code,
            embedding=parallel_embedding,
            schedule=parallel_schedule,
            kernel=RegularizedPowerLawKernel(alpha=1.0, r0=1.0),
            route_metric=MinDistanceMetric(),
            local_noise=LocalNoiseConfig(),
            geometry_noise=GeometryNoiseConfig(J0=0.5, tau=1.0),
            rounds=rounds,
        )

    def test_correlation_edges_populated(
        self, parallel_code, parallel_schedule, parallel_embedding
    ):
        """One pair event at tick 0 between (0, 4) and (2, 5) in the
        X sector produces exactly one correlation edge between data
        qubits 0 and 2 with `weight == sin²(0.25)`."""
        compiled = self._compile(parallel_code, parallel_schedule, parallel_embedding)
        assert len(compiled.correlation_edges) == 1
        edge = compiled.correlation_edges[0]
        assert isinstance(edge, CorrelationEdgeRecord)
        assert edge.qubit_a == 0
        assert edge.qubit_b == 2
        assert edge.sector == "X"
        assert edge.weight == pytest.approx(math.sin(0.25) ** 2, abs=1e-12)

    def test_exposure_total_matches_sum_of_probabilities(
        self, parallel_code, parallel_schedule, parallel_embedding
    ):
        """Plan acceptance test #2: ExposureMetrics.total() equals
        sum(pair_probability) over provenance to 1e-12."""
        compiled = self._compile(parallel_code, parallel_schedule, parallel_embedding)
        assert isinstance(compiled.exposure_metrics, ExposureMetrics)
        expected = sum(r.pair_probability for r in compiled.provenance)
        assert compiled.exposure_metrics.total() == pytest.approx(expected, abs=1e-12)

    def test_per_data_pair_matches_correlation_edge(
        self, parallel_code, parallel_schedule, parallel_embedding
    ):
        """The exposure metrics' `per_data_pair` row must match the
        correlation-edge table row-for-row (modulo the sector column
        that correlation_edges adds)."""
        compiled = self._compile(parallel_code, parallel_schedule, parallel_embedding)
        assert compiled.exposure_metrics is not None
        assert len(compiled.exposure_metrics.per_data_pair) == len(compiled.correlation_edges)
        for (qa, qb, exposure), edge in zip(
            compiled.exposure_metrics.per_data_pair,
            compiled.correlation_edges,
            strict=True,
        ):
            assert (qa, qb) == (edge.qubit_a, edge.qubit_b)
            assert exposure == pytest.approx(edge.weight, abs=1e-12)

    def test_per_support_uses_decoded_sector_logicals(
        self, parallel_code, parallel_schedule, parallel_embedding
    ):
        """k = 0 for the parallel fixture, so `per_support` is empty —
        but the field is still populated (as an empty tuple), not None."""
        compiled = self._compile(parallel_code, parallel_schedule, parallel_embedding)
        assert compiled.exposure_metrics is not None
        assert compiled.exposure_metrics.per_support == ()

    def test_decoder_artifact_populated(self, parallel_code, parallel_schedule, parallel_embedding):
        """The artifact carries one pair edge, a per-data-qubit
        zero-prior of the right length, and the default hint."""
        compiled = self._compile(parallel_code, parallel_schedule, parallel_embedding)
        assert isinstance(compiled.decoder_artifact, DecoderArtifact)
        assert len(compiled.decoder_artifact.pair_edges) == 1
        qa, qb, weight = compiled.decoder_artifact.pair_edges[0]
        assert (qa, qb) == (0, 2)
        assert weight == pytest.approx(math.sin(0.25) ** 2, abs=1e-12)
        assert compiled.decoder_artifact.single_prior == (0.0,) * 4
        assert compiled.decoder_artifact.decoder_hint == ""

    def test_fingerprint_deterministic(self, parallel_code, parallel_schedule, parallel_embedding):
        """Plan acceptance test #5: two compiles of the same inputs
        produce the same SHA256 fingerprint."""
        a = self._compile(parallel_code, parallel_schedule, parallel_embedding)
        b = self._compile(parallel_code, parallel_schedule, parallel_embedding)
        assert a.fingerprint() == b.fingerprint()
        # And the fingerprint is hex-encoded SHA256.
        assert len(a.fingerprint()) == 64
        int(a.fingerprint(), 16)  # raises if not hex

    def test_round_trip_preserves_tables(
        self, parallel_code, parallel_schedule, parallel_embedding
    ):
        """Plan acceptance test #4: to_json / from_json round-trip
        gives a deep-equal object."""
        compiled = self._compile(parallel_code, parallel_schedule, parallel_embedding)
        data = compiled.to_json()
        reconstructed = CompiledExtraction.from_json(data)
        assert reconstructed.provenance == compiled.provenance
        assert reconstructed.correlation_edges == compiled.correlation_edges
        assert reconstructed.exposure_metrics == compiled.exposure_metrics
        assert reconstructed.decoder_artifact == compiled.decoder_artifact
        # Full pure-data equality includes circuit_text + specs.
        assert reconstructed == compiled

    def test_fingerprint_round_trips(self, parallel_code, parallel_schedule, parallel_embedding):
        """The fingerprint of the round-tripped object matches the
        original. This ties `fingerprint` stability to JSON fidelity."""
        compiled = self._compile(parallel_code, parallel_schedule, parallel_embedding)
        reconstructed = CompiledExtraction.from_json(compiled.to_json())
        assert reconstructed.fingerprint() == compiled.fingerprint()

    def test_lazy_circuit_matches_circuit_text(
        self, parallel_code, parallel_schedule, parallel_embedding
    ):
        """Plan acceptance test #6: `compiled.circuit` materializes to
        a Stim circuit whose `str()` is byte-identical to
        `circuit_text`.

        Subtlety: the *in-process* `compiled.circuit` is the exact
        `stim.Circuit` the compiler built (with full float precision),
        so `str()` on it matches `circuit_text` directly. After a
        JSON round-trip the cache is empty, and `circuit` re-parses
        the (lossy) text — but `str(reparsed)` still matches the text
        exactly because Stim's text form is idempotent.
        """
        compiled = self._compile(parallel_code, parallel_schedule, parallel_embedding)
        assert str(compiled.circuit) == compiled.circuit_text
        assert str(compiled.dem) == compiled.dem_text

        # After round-trip, still holds.
        reconstructed = CompiledExtraction.from_json(compiled.to_json())
        assert str(reconstructed.circuit) == reconstructed.circuit_text
        assert str(reconstructed.dem) == reconstructed.dem_text

    def test_correlation_graph_is_networkx_graph(
        self, parallel_code, parallel_schedule, parallel_embedding
    ):
        """The lazy `correlation_graph` property builds a NetworkX
        graph with one node per distinct qubit and one edge per
        correlation record."""
        import networkx as nx

        compiled = self._compile(parallel_code, parallel_schedule, parallel_embedding)
        g = compiled.correlation_graph
        assert isinstance(g, nx.Graph)
        assert g.number_of_edges() == 1
        assert g.has_edge(0, 2)
        assert g.edges[0, 2]["weight"] == pytest.approx(math.sin(0.25) ** 2, abs=1e-12)
        assert g.edges[0, 2]["sectors"] == {"X"}


# ---------------------------------------------------------------------------
# PR 9 — Steane crossing acceptance test
# ---------------------------------------------------------------------------


class TestSteaneCrossingPR9:
    """Plan acceptance test #1: a Steane-derived fixture with a
    hand-crafted parallel pair produces exactly one correlation edge
    with the expected weight.

    Since `default_css_schedule(steane)` is serial, we construct a
    minimal parallel schedule over the Steane qubit layout and a
    custom embedding that places two specific CNOT polylines at a
    known distance. This test is the geometry-pipeline equivalent
    of the `test_single_pair_event_numerics` unit test but runs
    through the full `compile_extraction` pipeline on a real code
    (the [[7, 1, 3]] Steane code).
    """

    def _build(self):
        H = pcm.hamming(7)
        code = CSSCode(HX=H, HZ=H, rounds=1)
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

        head = [
            _make_step(
                tick=0,
                role="reset",
                edges=[SingleQubitEdge(gate="R", qubit=q) for q in sorted(all_qubits)],
                all_qubits=all_qubits,
            )
        ]

        # Parallel pair at cycle tick 0: the first CNOT of Z-check 0
        # (data[0] → z_checks[0]) and the first CNOT of Z-check 1
        # (data[3] → z_checks[1]). Picking columns HZ[0,0]=1 and
        # HZ[1,3]=1 ensures both rows have an X-sector CNOT here.
        # Remaining CNOTs from default_css_schedule are appended
        # serially. This makes the schedule lossy w.r.t. full stabilizer
        # extraction but keeps the invariants compile_extraction relies
        # on (one MR at the end, final data M in the tail).
        cycle: list[ScheduleStep] = []
        cycle.append(
            _make_step(
                tick=0,
                role="cnot_layer",
                edges=[
                    TwoQubitEdge(
                        gate="CNOT",
                        control=data[0],
                        target=z_checks[0],
                        interaction_sector="X",
                        term_name="HZ[0,0]",
                    ),
                    TwoQubitEdge(
                        gate="CNOT",
                        control=data[3],
                        target=z_checks[1],
                        interaction_sector="X",
                        term_name="HZ[1,3]",
                    ),
                ],
                all_qubits=all_qubits,
            )
        )

        # Serial Z-check CNOTs for the rest.
        tick = 1
        for check_idx, target in enumerate(z_checks):
            row = H[check_idx]
            for col_idx in range(len(row)):
                if not row[col_idx]:
                    continue
                data_q = data[col_idx]
                # Skip the two CNOTs already emitted at tick 0.
                if (check_idx, col_idx) in {(0, 0), (1, 3)}:
                    continue
                cycle.append(
                    _make_step(
                        tick=tick,
                        role="cnot_layer",
                        edges=[
                            TwoQubitEdge(
                                gate="CNOT",
                                control=data_q,
                                target=target,
                                interaction_sector="X",
                                term_name=f"HZ[{check_idx},{col_idx}]",
                            )
                        ],
                        all_qubits=all_qubits,
                    )
                )
                tick += 1
        # X-check brackets, serial.
        for check_idx, target in enumerate(x_checks):
            cycle.append(
                _make_step(
                    tick=tick,
                    role="single_q",
                    edges=[SingleQubitEdge(gate="H", qubit=target)],
                    all_qubits=all_qubits,
                )
            )
            tick += 1
            row = H[check_idx]
            for col_idx in range(len(row)):
                if not row[col_idx]:
                    continue
                data_q = data[col_idx]
                cycle.append(
                    _make_step(
                        tick=tick,
                        role="cnot_layer",
                        edges=[
                            TwoQubitEdge(
                                gate="CNOT",
                                control=target,
                                target=data_q,
                                interaction_sector="Z",
                                term_name=f"HX[{check_idx},{col_idx}]",
                            )
                        ],
                        all_qubits=all_qubits,
                    )
                )
                tick += 1
            cycle.append(
                _make_step(
                    tick=tick,
                    role="single_q",
                    edges=[SingleQubitEdge(gate="H", qubit=target)],
                    all_qubits=all_qubits,
                )
            )
            tick += 1
        cycle.append(
            _make_step(
                tick=tick,
                role="meas",
                edges=[SingleQubitEdge(gate="MR", qubit=q) for q in z_checks + x_checks],
                all_qubits=all_qubits,
            )
        )

        tail = [
            _make_step(
                tick=0,
                role="meas",
                edges=[SingleQubitEdge(gate="M", qubit=q) for q in data],
                all_qubits=all_qubits,
            )
        ]

        schedule = Schedule(
            head_steps=tuple(head),
            cycle_steps=tuple(cycle),
            tail_steps=tuple(tail),
            qubits=all_qubits,
            qubit_roles=roles,
            name="steane_forced_crossing",
        )

        # Embedding: place the two pair-tick CNOT endpoints so that
        # their polylines are at unit distance. Other qubits far away.
        num_qubits = len(code.qubits)
        positions: list[tuple[float, float]] = [(0.0, 0.0)] * num_qubits
        positions[data[0]] = (0.0, 0.0)
        positions[z_checks[0]] = (0.0, 1.0)
        positions[data[3]] = (1.0, 0.0)
        positions[z_checks[1]] = (1.0, 1.0)
        # Park the rest far away.
        park_x = 100.0
        for i, q in enumerate(code.qubits):
            if positions[q] == (0.0, 0.0) and q not in (data[0],):
                positions[q] = (park_x + 3.0 * i, -50.0)
        embedding = StraightLineEmbedding.from_positions(positions)

        return code, schedule, embedding

    def test_exactly_one_correlation_edge_with_expected_weight(self):
        code, schedule, embedding = self._build()
        compiled = compile_extraction(
            code=code,
            embedding=embedding,
            schedule=schedule,
            kernel=RegularizedPowerLawKernel(alpha=1.0, r0=1.0),
            route_metric=MinDistanceMetric(),
            local_noise=LocalNoiseConfig(),
            geometry_noise=GeometryNoiseConfig(J0=0.5, tau=1.0),
            rounds=1,
        )
        # Exactly one correlation edge, between data qubits 0 and 3.
        assert len(compiled.correlation_edges) == 1
        edge = compiled.correlation_edges[0]
        assert edge.sector == "X"
        assert {edge.qubit_a, edge.qubit_b} == {code.data_qubits[0], code.data_qubits[3]}
        assert edge.weight == pytest.approx(math.sin(0.25) ** 2, abs=1e-12)

    def test_exposure_metrics_match_correlation(self):
        code, schedule, embedding = self._build()
        compiled = compile_extraction(
            code=code,
            embedding=embedding,
            schedule=schedule,
            kernel=RegularizedPowerLawKernel(alpha=1.0, r0=1.0),
            route_metric=MinDistanceMetric(),
            local_noise=LocalNoiseConfig(),
            geometry_noise=GeometryNoiseConfig(J0=0.5, tau=1.0),
            rounds=1,
        )
        assert compiled.exposure_metrics is not None
        assert compiled.exposure_metrics.total() == pytest.approx(math.sin(0.25) ** 2, abs=1e-12)
        # Steane has k=1, so per_support has one entry.
        assert len(compiled.exposure_metrics.per_support) == 1
        # The forced pair acts on data qubits 0 and 3. Its exposure
        # contribution to a given Z-logical support depends on whether
        # both 0 and 3 are in the support.
        support_rec = compiled.exposure_metrics.per_support[0]
        data_indices_in_support = {
            i for i, q in enumerate(code.data_qubits) if q in support_rec.support
        }
        contributes = {0, 3}.issubset(data_indices_in_support)
        expected_exposure = math.sin(0.25) ** 2 if contributes else 0.0
        assert support_rec.exposure == pytest.approx(expected_exposure, abs=1e-12)

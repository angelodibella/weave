"""PR 17 — decoder artifact adapter tests.

Exercises the three adapter methods on `DecoderArtifact`:

1. `to_bposd_decoder(dem)` — builds a `stimbposd.BPOSD` that
   successfully decodes at least one shot.
2. `to_pymatching(dem)` — builds a `pymatching.Matching` that
   accepts the non-decomposed DEM and decodes.
3. `to_pair_prior_dict()` — returns a dict with the expected
   structure.

The test circuit is a compiled Steane [[7,1,3]] code with a
custom parallel schedule and `J_0 > 0`, so the geometry pass
produces at least one `CORRELATED_ERROR` in the DEM.
"""

from __future__ import annotations

import numpy as np
import pytest

from weave.codes.css_code import CSSCode
from weave.compiler import compile_extraction
from weave.ir import (
    DecoderArtifact,
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
)
from weave.ir.schedule import QubitRole, ScheduleRole
from weave.util import pcm

# ---------------------------------------------------------------------------
# Fixture: Steane with parallel schedule + geometry noise
# ---------------------------------------------------------------------------


def _steane_parallel_compiled():
    """Build a Steane [[7,1,3]] compiled extraction with at least one
    CORRELATED_ERROR in the DEM, suitable for decoder testing.

    Uses a custom schedule with one parallel CNOT tick so the
    geometry pass produces provenance.
    """
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

    head = [mk(0, "reset", [SingleQubitEdge(gate="R", qubit=q) for q in sorted(all_qubits)])]

    # Parallel tick: first CNOT of Z-check 0 and Z-check 2 (disjoint).
    cycle: list[ScheduleStep] = []
    # Z-check 0 touches data cols where HZ[0] is 1.
    cols_0 = [i for i, v in enumerate(H[0]) if v]
    cols_2 = [i for i, v in enumerate(H[2]) if v]
    cycle.append(
        mk(
            0,
            "cnot_layer",
            [
                TwoQubitEdge(
                    gate="CNOT", control=data[cols_0[0]], target=z_checks[0], interaction_sector="X"
                ),
                TwoQubitEdge(
                    gate="CNOT", control=data[cols_2[0]], target=z_checks[2], interaction_sector="X"
                ),
            ],
        )
    )
    tick = 1
    # Remaining Z-check CNOTs serial.
    for ci in range(H.shape[0]):
        cols = [i for i, v in enumerate(H[ci]) if v]
        start = 1 if ci in (0, 2) else 0
        for c in cols[start:]:
            cycle.append(
                mk(
                    tick,
                    "cnot_layer",
                    [
                        TwoQubitEdge(
                            gate="CNOT",
                            control=data[c],
                            target=z_checks[ci],
                            interaction_sector="X",
                        )
                    ],
                )
            )
            tick += 1
    # X-check brackets serial.
    for ci in range(H.shape[0]):
        cycle.append(mk(tick, "single_q", [SingleQubitEdge(gate="H", qubit=x_checks[ci])]))
        tick += 1
        cols = [i for i, v in enumerate(H[ci]) if v]
        for c in cols:
            cycle.append(
                mk(
                    tick,
                    "cnot_layer",
                    [
                        TwoQubitEdge(
                            gate="CNOT",
                            control=x_checks[ci],
                            target=data[c],
                            interaction_sector="Z",
                        )
                    ],
                )
            )
            tick += 1
        cycle.append(mk(tick, "single_q", [SingleQubitEdge(gate="H", qubit=x_checks[ci])]))
        tick += 1
    cycle.append(
        mk(tick, "meas", [SingleQubitEdge(gate="MR", qubit=q) for q in z_checks + x_checks])
    )

    tail = [mk(0, "meas", [SingleQubitEdge(gate="M", qubit=q) for q in data])]

    sched = Schedule(
        head_steps=tuple(head),
        cycle_steps=tuple(cycle),
        tail_steps=tuple(tail),
        qubits=all_qubits,
        qubit_roles=roles,
        name="steane_parallel",
    )

    # Positions with the parallel pair 1 unit apart.
    positions: list[tuple[float, float]] = [(0.0, 0.0)] * len(code.qubits)
    positions[data[cols_0[0]]] = (0.0, 0.0)
    positions[z_checks[0]] = (0.0, 1.0)
    positions[data[cols_2[0]]] = (1.0, 0.0)
    positions[z_checks[2]] = (1.0, 1.0)
    for i, q in enumerate(code.qubits):
        if positions[q] == (0.0, 0.0) and q != data[cols_0[0]]:
            positions[q] = (5.0 + i, 10.0)
    emb = StraightLineEmbedding.from_positions(positions)

    compiled = compile_extraction(
        code=code,
        embedding=emb,
        schedule=sched,
        kernel=RegularizedPowerLawKernel(alpha=1.0, r0=1.0),
        route_metric=MinDistanceMetric(),
        local_noise=LocalNoiseConfig(p_cnot=0.001),
        geometry_noise=GeometryNoiseConfig(J0=0.5, tau=1.0),
        rounds=1,
    )
    return compiled


@pytest.fixture(scope="module")
def compiled():
    return _steane_parallel_compiled()


# ---------------------------------------------------------------------------
# to_pair_prior_dict
# ---------------------------------------------------------------------------


class TestPairPriorDict:
    def test_returns_dict_of_pairs_to_floats(self, compiled):
        d = compiled.decoder_artifact.to_pair_prior_dict()
        assert isinstance(d, dict)
        for key, val in d.items():
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert isinstance(val, float)
            assert val >= 0

    def test_keys_match_pair_edges(self, compiled):
        d = compiled.decoder_artifact.to_pair_prior_dict()
        expected_keys = {(a, b) for a, b, _ in compiled.decoder_artifact.pair_edges}
        assert set(d.keys()) == expected_keys

    def test_empty_artifact_gives_empty_dict(self):
        artifact = DecoderArtifact()
        assert artifact.to_pair_prior_dict() == {}


# ---------------------------------------------------------------------------
# to_bposd_decoder
# ---------------------------------------------------------------------------


class TestBPOSDDecoder:
    def test_decoder_accepts_compiled_dem(self, compiled):
        """stimbposd.BPOSD accepts the non-decomposed DEM."""
        decoder = compiled.decoder_artifact.to_bposd_decoder(
            compiled.dem, max_bp_iters=5, osd_order=5
        )
        assert decoder is not None

    def test_decoder_decodes_at_least_one_shot(self, compiled):
        """Plan acceptance test: BPOSD successfully decodes at least
        one shot on the compiled Steane circuit."""
        decoder = compiled.decoder_artifact.to_bposd_decoder(
            compiled.dem, max_bp_iters=10, osd_order=10
        )
        sampler = compiled.circuit.compile_detector_sampler()
        shots, obs = sampler.sample(shots=10, separate_observables=True)
        predictions = decoder.decode_batch(shots)
        # At least one shot should decode without error.
        errors = np.any(predictions != obs, axis=1)
        assert not np.all(errors), "BPOSD failed to decode any shot correctly"


# ---------------------------------------------------------------------------
# to_pymatching
# ---------------------------------------------------------------------------


class TestPyMatching:
    def test_matching_accepts_compiled_dem(self, compiled):
        """PyMatching accepts the non-decomposed DEM and builds a
        matching graph."""
        matching = compiled.decoder_artifact.to_pymatching(compiled.dem)
        assert matching is not None
        assert matching.num_nodes >= 1

    def test_matching_decodes_at_least_one_shot(self, compiled):
        """Plan acceptance test: PyMatching successfully decodes at
        least one shot."""
        matching = compiled.decoder_artifact.to_pymatching(compiled.dem)
        sampler = compiled.circuit.compile_detector_sampler()
        shots, obs = sampler.sample(shots=10, separate_observables=True)
        predictions = matching.decode_batch(shots)
        errors = np.any(predictions != obs, axis=1)
        assert not np.all(errors), "PyMatching failed to decode any shot correctly"


# ---------------------------------------------------------------------------
# Provenance sanity: compiled has non-empty pair data
# ---------------------------------------------------------------------------


class TestCompiledHasPairData:
    def test_provenance_is_non_empty(self, compiled):
        assert len(compiled.provenance) >= 1

    def test_decoder_artifact_has_pair_edges(self, compiled):
        assert compiled.decoder_artifact.num_pair_edges >= 1

    def test_dem_has_errors(self, compiled):
        assert compiled.dem.num_errors >= 1
        assert compiled.dem.num_detectors >= 1

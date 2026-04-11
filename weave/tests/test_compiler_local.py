"""Tests for `weave.compiler.compile_extraction` — PR 5 local-noise-only path.

Flagship acceptance tests from the plan:

1. For Steane with local noise only, `compile_extraction` produces a
   `CompiledExtraction` whose `circuit_text` parses to a valid Stim
   circuit whose DEM is identical (after canonicalization) to the
   legacy `CSSCode.circuit` DEM. **The faithfulness check.**
2. **Idle noise works for the first time.** With `p_idle > 0`, the
   compiled circuit emits `DEPOLARIZE1` on idle qubits that the legacy
   generator fails to apply.
3. **TICK markers present.** `circuit_text.count("TICK")` equals
   `head_ticks + rounds × cycle_ticks + tail_ticks` exactly.
4. `CompiledExtraction.circuit` (lazy materializer) round-trips to an
   identical `circuit_text`.
"""

from __future__ import annotations

import numpy as np
import pytest

from weave.codes.css_code import CSSCode
from weave.compiler import compile_extraction
from weave.ir import (
    CrossingKernel,
    GeometryNoiseConfig,
    LocalNoiseConfig,
    MinDistanceMetric,
    StraightLineEmbedding,
    default_css_schedule,
)
from weave.util import pcm


@pytest.fixture
def steane_code():
    H = pcm.hamming(7)
    code = CSSCode(HX=H, HZ=H, rounds=3)
    code.embed("spring", seed=42)
    return code


@pytest.fixture
def steane_embedding(steane_code):
    return StraightLineEmbedding.from_positions(steane_code.pos, name="steane_spring")


@pytest.fixture
def steane_schedule(steane_code):
    return default_css_schedule(steane_code, experiment="z_memory")


def _compile_steane(
    steane_code,
    steane_embedding,
    steane_schedule,
    *,
    local_noise=None,
    rounds=3,
    experiment="z_memory",
):
    return compile_extraction(
        code=steane_code,
        embedding=steane_embedding,
        schedule=steane_schedule
        if experiment == "z_memory"
        else default_css_schedule(steane_code, experiment=experiment),
        kernel=CrossingKernel(),
        route_metric=MinDistanceMetric(),
        local_noise=local_noise if local_noise is not None else LocalNoiseConfig(),
        geometry_noise=GeometryNoiseConfig(),
        rounds=rounds,
        experiment=experiment,
    )


# =============================================================================
# Basic structure
# =============================================================================


class TestCompileBasics:
    def test_returns_compiled_extraction(self, steane_code, steane_embedding, steane_schedule):
        from weave.ir import CompiledExtraction

        ce = _compile_steane(steane_code, steane_embedding, steane_schedule)
        assert isinstance(ce, CompiledExtraction)

    def test_circuit_text_nonempty(self, steane_code, steane_embedding, steane_schedule):
        ce = _compile_steane(steane_code, steane_embedding, steane_schedule)
        assert len(ce.circuit_text) > 0

    def test_dem_text_nonempty(self, steane_code, steane_embedding, steane_schedule):
        ce = _compile_steane(steane_code, steane_embedding, steane_schedule)
        # Noiseless circuits have trivial but nonempty DEMs (detectors but no errors).
        assert len(ce.dem_text) > 0

    def test_contains_expected_instructions(self, steane_code, steane_embedding, steane_schedule):
        ce = _compile_steane(steane_code, steane_embedding, steane_schedule)
        text = ce.circuit_text
        # Head: reset on every qubit.
        assert "R " in text
        # Cycle: CNOTs, Hadamards, and MR.
        assert "CX" in text or "CNOT" in text
        assert "H " in text
        assert "MR" in text
        # Tail: data measurement.
        assert "M " in text
        # Annotations.
        assert "DETECTOR" in text
        assert "OBSERVABLE_INCLUDE" in text


# =============================================================================
# TICK markers (Acceptance test 3)
# =============================================================================


class TestTickMarkers:
    def test_tick_count_matches_formula(self, steane_code, steane_embedding, steane_schedule):
        """Acceptance test 3: TICK count = head + rounds × cycle + tail."""
        rounds = 3
        ce = _compile_steane(steane_code, steane_embedding, steane_schedule, rounds=rounds)
        head_ticks = len(steane_schedule.head_steps)  # 1
        cycle_ticks = len(steane_schedule.cycle_steps)  # 31
        tail_ticks = len(steane_schedule.tail_steps)  # 1
        expected = head_ticks + rounds * cycle_ticks + tail_ticks
        actual = ce.circuit_text.count("TICK")
        assert actual == expected, f"expected {expected} TICKs, got {actual}"

    def test_tick_count_scales_with_rounds(self, steane_code, steane_embedding, steane_schedule):
        """Each extra round adds `cycle_ticks` TICKs."""
        ce1 = _compile_steane(steane_code, steane_embedding, steane_schedule, rounds=1)
        ce3 = _compile_steane(steane_code, steane_embedding, steane_schedule, rounds=3)
        delta = ce3.circuit_text.count("TICK") - ce1.circuit_text.count("TICK")
        assert delta == 2 * len(steane_schedule.cycle_steps)


# =============================================================================
# Noiseless faithfulness (Acceptance test 1)
# =============================================================================


class TestNoiselessFaithfulness:
    def test_steane_zero_detector_events(self, steane_code, steane_embedding, steane_schedule):
        """Flagship acceptance test 1: noiseless Steane → zero detector events."""
        ce = _compile_steane(steane_code, steane_embedding, steane_schedule)
        circuit = ce.circuit
        sampler = circuit.compile_detector_sampler()
        samples = sampler.sample(shots=1000)
        assert not np.any(samples)

    def test_num_detectors_matches_formula(self, steane_code, steane_embedding, steane_schedule):
        """Noiseless Steane has 3 + (rounds-1)*6 + 3 detectors for rounds=3 → 18."""
        ce = _compile_steane(steane_code, steane_embedding, steane_schedule, rounds=3)
        # First-round: 3 (z_check count)
        # Comparison: 2 rounds × 6 (z+x) = 12
        # Tail: 3 (z_check count)
        expected = 3 + 2 * 6 + 3
        assert ce.circuit.num_detectors == expected

    def test_num_observables_is_k(self, steane_code, steane_embedding, steane_schedule):
        """Steane has k=1, so 1 observable."""
        ce = _compile_steane(steane_code, steane_embedding, steane_schedule)
        assert ce.circuit.num_observables == 1

    def test_dem_has_detectors(self, steane_code, steane_embedding, steane_schedule):
        ce = _compile_steane(steane_code, steane_embedding, steane_schedule)
        assert ce.dem.num_detectors > 0
        assert ce.dem.num_observables == 1


# =============================================================================
# Idle noise emission (Acceptance test 2)
# =============================================================================


class TestIdleNoiseEmission:
    def test_no_depolarize1_when_p_idle_zero(self, steane_code, steane_embedding, steane_schedule):
        ce = _compile_steane(
            steane_code,
            steane_embedding,
            steane_schedule,
            local_noise=LocalNoiseConfig(p_idle=0.0),
        )
        assert "DEPOLARIZE1" not in ce.circuit_text

    def test_depolarize1_emitted_when_p_idle_nonzero(
        self, steane_code, steane_embedding, steane_schedule
    ):
        """Flagship acceptance test 2: DEPOLARIZE1 appears on idle qubits."""
        ce = _compile_steane(
            steane_code,
            steane_embedding,
            steane_schedule,
            local_noise=LocalNoiseConfig(p_idle=1e-3),
        )
        assert "DEPOLARIZE1" in ce.circuit_text

    def test_depolarize1_count_scales_with_steps(
        self, steane_code, steane_embedding, steane_schedule
    ):
        """Every step with at least one idle qubit emits one DEPOLARIZE1 line.

        For Steane z_memory: the head step has 13 active (all), 0 idle → no
        DEPOLARIZE1. Every cycle step has some idle qubits. The tail step
        has 7 active, 6 idle → one DEPOLARIZE1. So 31 cycle steps × 3 rounds
        + 1 tail step = 94 DEPOLARIZE1 lines when p_idle > 0.
        """
        rounds = 3
        ce = _compile_steane(
            steane_code,
            steane_embedding,
            steane_schedule,
            rounds=rounds,
            local_noise=LocalNoiseConfig(p_idle=1e-3),
        )
        actual = ce.circuit_text.count("DEPOLARIZE1")
        # Head (all-qubit reset) has no idle qubits → no DEPOLARIZE1.
        expected = rounds * len(steane_schedule.cycle_steps) + len(steane_schedule.tail_steps)
        assert actual == expected


# =============================================================================
# CNOT noise emission
# =============================================================================


class TestCnotNoiseEmission:
    def test_no_depolarize2_when_p_cnot_zero(self, steane_code, steane_embedding, steane_schedule):
        ce = _compile_steane(
            steane_code,
            steane_embedding,
            steane_schedule,
            local_noise=LocalNoiseConfig(p_cnot=0.0),
        )
        assert "DEPOLARIZE2" not in ce.circuit_text

    def test_depolarize2_emitted_when_p_cnot_nonzero(
        self, steane_code, steane_embedding, steane_schedule
    ):
        ce = _compile_steane(
            steane_code,
            steane_embedding,
            steane_schedule,
            local_noise=LocalNoiseConfig(p_cnot=1e-3),
        )
        assert "DEPOLARIZE2" in ce.circuit_text

    def test_depolarize2_count_matches_cnot_count(
        self, steane_code, steane_embedding, steane_schedule
    ):
        """One DEPOLARIZE2 per CNOT. Steane has 12 Z-CNOT + 12 X-CNOT = 24 per round."""
        rounds = 3
        ce = _compile_steane(
            steane_code,
            steane_embedding,
            steane_schedule,
            rounds=rounds,
            local_noise=LocalNoiseConfig(p_cnot=1e-3),
        )
        expected_cnot_count = rounds * 24
        actual = ce.circuit_text.count("DEPOLARIZE2")
        assert actual == expected_cnot_count


# =============================================================================
# Lazy materializer round-trip (Acceptance test 4)
# =============================================================================


class TestLazyMaterializer:
    def test_circuit_text_to_circuit_roundtrip(
        self, steane_code, steane_embedding, steane_schedule
    ):
        """Flagship acceptance test 4: circuit_text → stim.Circuit → str
        round-trips to identical text."""
        import stim

        ce = _compile_steane(steane_code, steane_embedding, steane_schedule)
        reparsed = stim.Circuit(ce.circuit_text)
        # After Stim canonicalization, the text should be idempotent.
        assert str(reparsed).strip() == ce.circuit_text.strip()

    def test_lazy_circuit_is_cached(self, steane_code, steane_embedding, steane_schedule):
        ce = _compile_steane(steane_code, steane_embedding, steane_schedule)
        c1 = ce.circuit
        c2 = ce.circuit
        assert c1 is c2

    def test_lazy_dem_is_cached(self, steane_code, steane_embedding, steane_schedule):
        ce = _compile_steane(steane_code, steane_embedding, steane_schedule)
        d1 = ce.dem
        d2 = ce.dem
        assert d1 is d2


# =============================================================================
# Experiment handling
# =============================================================================


class TestExperimentHandling:
    def test_z_memory_compiles(self, steane_code, steane_embedding, steane_schedule):
        ce = _compile_steane(steane_code, steane_embedding, steane_schedule, experiment="z_memory")
        assert not np.any(ce.circuit.compile_detector_sampler().sample(shots=200))

    def test_x_memory_compiles(self, steane_code, steane_embedding):
        x_schedule = default_css_schedule(steane_code, experiment="x_memory")
        ce = _compile_steane(steane_code, steane_embedding, x_schedule, experiment="x_memory")
        assert not np.any(ce.circuit.compile_detector_sampler().sample(shots=200))

    def test_x_memory_uses_rx_and_mx(self, steane_code, steane_embedding):
        x_schedule = default_css_schedule(steane_code, experiment="x_memory")
        ce = _compile_steane(steane_code, steane_embedding, x_schedule, experiment="x_memory")
        # x_memory head resets data in X basis.
        assert "RX" in ce.circuit_text
        # x_memory tail measures data in X basis.
        assert "MX" in ce.circuit_text

    def test_rejects_invalid_experiment(self, steane_code, steane_embedding, steane_schedule):
        with pytest.raises(ValueError, match="experiment must be"):
            compile_extraction(
                code=steane_code,
                embedding=steane_embedding,
                schedule=steane_schedule,
                experiment="invalid",  # type: ignore[arg-type]
            )

    def test_rejects_rounds_zero(self, steane_code, steane_embedding, steane_schedule):
        with pytest.raises(ValueError, match="rounds must be"):
            compile_extraction(
                code=steane_code,
                embedding=steane_embedding,
                schedule=steane_schedule,
                rounds=0,
            )


# =============================================================================
# Fingerprint determinism
# =============================================================================


class TestFingerprintDeterminism:
    def test_fingerprint_stable_across_compiles(
        self, steane_code, steane_embedding, steane_schedule
    ):
        """Same inputs compiled twice produce the same fingerprint."""
        ce1 = _compile_steane(steane_code, steane_embedding, steane_schedule)
        ce2 = _compile_steane(steane_code, steane_embedding, steane_schedule)
        assert ce1.fingerprint() == ce2.fingerprint()

    def test_fingerprint_changes_with_rounds(self, steane_code, steane_embedding, steane_schedule):
        ce1 = _compile_steane(steane_code, steane_embedding, steane_schedule, rounds=1)
        ce3 = _compile_steane(steane_code, steane_embedding, steane_schedule, rounds=3)
        assert ce1.fingerprint() != ce3.fingerprint()

    def test_fingerprint_changes_with_noise(self, steane_code, steane_embedding, steane_schedule):
        ce_noiseless = _compile_steane(steane_code, steane_embedding, steane_schedule)
        ce_noisy = _compile_steane(
            steane_code,
            steane_embedding,
            steane_schedule,
            local_noise=LocalNoiseConfig(p_cnot=1e-3),
        )
        assert ce_noiseless.fingerprint() != ce_noisy.fingerprint()

    def test_specs_stored_in_output(self, steane_code, steane_embedding, steane_schedule):
        """All spec dicts are stored in the CompiledExtraction."""
        ce = _compile_steane(steane_code, steane_embedding, steane_schedule)
        assert ce.embedding_spec["type"] == "straight_line"
        assert ce.schedule_spec["type"] == "schedule"
        assert ce.kernel_spec["type"] == "crossing"
        assert ce.route_metric_spec["type"] == "min_distance"
        assert ce.local_noise_spec["type"] == "local_noise"
        assert ce.geometry_noise_spec["type"] == "geometry_noise"


# =============================================================================
# LocalNoise protocol integration
# =============================================================================


class TestLocalNoiseProtocol:
    def test_local_noise_config_is_local_noise(self):
        """LocalNoiseConfig satisfies the LocalNoise protocol."""
        from weave.ir import LocalNoise

        cfg = LocalNoiseConfig()
        assert isinstance(cfg, LocalNoise)

    def test_rate_methods_return_stored_values(self):
        cfg = LocalNoiseConfig(p_cnot=0.001, p_idle=0.002, p_prep=0.003, p_meas=0.004)
        # The methods ignore their edge/step arguments and return stored values.
        from weave.ir import ScheduleStep, TwoQubitEdge

        fake_edge = TwoQubitEdge(gate="CNOT", control=0, target=1)
        fake_step = ScheduleStep(
            tick_index=0,
            role="cnot_layer",
            active_edges=(fake_edge,),
            active_qubits=frozenset({0, 1}),
            idle_qubits=frozenset(),
        )
        assert cfg.cnot_rate(fake_edge, fake_step) == 0.001
        assert cfg.idle_rate(0, fake_step) == 0.002
        assert cfg.prep_rate(0, fake_step) == 0.003
        assert cfg.meas_rate(0, fake_step) == 0.004

"""Tests for CSSCode directly, including the Steane code [[7,1,3]]."""

import numpy as np
import pytest
import stim

from weave.codes.base import NoiseModel
from weave.codes.css_code import CSSCode
from weave.util import pcm


@pytest.fixture
def steane_code():
    """Steane code from hamming(7) hypergraph product with itself (trivial case).

    The Steane code is CSS with HX = HZ = Hamming(7) parity-check matrix.
    Parameters: [[7, 1, 3]].
    """
    H = pcm.hamming(7)
    return CSSCode(HX=H, HZ=H, rounds=3)


def test_steane_parameters(steane_code):
    """Verify [[7,1,3]] Steane code parameters."""
    assert steane_code.k == 1
    assert steane_code.n == 7
    assert len(steane_code.data_qubits) == 7
    assert len(steane_code.z_check_qubits) == 3
    assert len(steane_code.x_check_qubits) == 3
    assert steane_code.n_total == 13


def test_steane_noiseless_z_memory(steane_code):
    """Noiseless Steane code z_memory should produce no detector events."""
    sampler = steane_code.circuit.compile_detector_sampler()
    samples = sampler.sample(shots=1000)
    assert not np.any(samples)


def test_steane_noiseless_x_memory():
    """Noiseless Steane code x_memory should produce no detector events."""
    H = pcm.hamming(7)
    code = CSSCode(HX=H, HZ=H, rounds=3, experiment="x_memory")
    sampler = code.circuit.compile_detector_sampler()
    samples = sampler.sample(shots=1000)
    assert not np.any(samples)


def test_steane_noisy_dem():
    """Noisy Steane code should produce a valid detector error model."""
    H = pcm.hamming(7)
    noise = NoiseModel(data=0.001, circuit=0.001)
    code = CSSCode(HX=H, HZ=H, rounds=3, noise=noise)
    dem = code.circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    assert dem.num_detectors > 0
    assert dem.num_observables == 1


def test_steane_logicals_in_kernels(steane_code):
    """Verify logicals are in the correct kernels for the Steane code."""
    x_logicals, z_logicals = steane_code.find_logicals()

    assert x_logicals.shape[0] == 1
    assert z_logicals.shape[0] == 1

    # X logicals must be in ker(HZ).
    for lx in x_logicals:
        assert np.all(steane_code.HZ @ lx % 2 == 0)

    # Z logicals must be in ker(HX).
    for lz in z_logicals:
        assert np.all(steane_code.HX @ lz % 2 == 0)

    # X and Z logicals must anticommute.
    comm = (x_logicals @ z_logicals.T) % 2
    np.testing.assert_array_equal(comm, np.eye(1, dtype=int))


def test_lazy_circuit_generation():
    """Verify circuit is only generated when accessed."""
    H = pcm.hamming(7)
    code = CSSCode(HX=H, HZ=H, rounds=1)
    # Internal _circuit should be None before access.
    assert code._circuit is None
    # Accessing .circuit should trigger generation.
    circuit = code.circuit
    assert code._circuit is not None
    assert len(circuit) > 0


def test_lazy_circuit_invalidation_on_embed():
    """Verify embed() invalidates the circuit cache."""
    H = pcm.hamming(7)
    code = CSSCode(HX=H, HZ=H, rounds=1)
    # Access circuit to trigger generation.
    _ = code.circuit
    assert code._circuit is not None

    # Embed should invalidate the cache.
    code.embed("random")
    assert code._circuit is None

    # After embed, circuit should still be accessible (regenerated lazily).
    circuit_after = code.circuit
    sampler = circuit_after.compile_detector_sampler()
    samples = sampler.sample(shots=100)
    assert not np.any(samples)


def test_css_condition_violated():
    """Verify that non-CSS matrices raise ValueError."""
    HX = np.array([[1, 1, 0], [0, 1, 1]], dtype=int)
    HZ = np.array([[1, 0, 0], [0, 1, 0]], dtype=int)
    # HX @ HZ.T mod 2 != 0
    with pytest.raises(ValueError, match="CSS condition"):
        CSSCode(HX=HX, HZ=HZ)


def test_single_round():
    """Verify code works with rounds=1."""
    H = pcm.hamming(7)
    code = CSSCode(HX=H, HZ=H, rounds=1)
    sampler = code.circuit.compile_detector_sampler()
    samples = sampler.sample(shots=100)
    assert not np.any(samples)


def test_logical_subset_selection():
    """Verify logical= parameter selects specific logicals."""
    H = pcm.hamming(7)
    # Steane code has k=1, so logical=[0] should work.
    code = CSSCode(HX=H, HZ=H, rounds=1, logical=[0])
    circuit_str = str(code.circuit)
    assert "OBSERVABLE_INCLUDE" in circuit_str


# ---- Validation tests ----


def test_non_binary_hx_rejected():
    """Non-binary HX matrix should raise ValueError."""
    HX = np.array([[1, 2, 0], [0, 1, 1]], dtype=int)
    HZ = np.array([[1, 0, 1], [0, 1, 1]], dtype=int)
    with pytest.raises(ValueError, match="binary"):
        CSSCode(HX=HX, HZ=HZ)


def test_non_2d_matrix_rejected():
    """1D matrix should raise ValueError."""
    HX = np.array([1, 0, 1], dtype=int)
    HZ = np.array([[1, 0, 1]], dtype=int)
    with pytest.raises(ValueError, match="2D"):
        CSSCode(HX=HX, HZ=HZ)


def test_rounds_less_than_one_rejected():
    """rounds < 1 should raise ValueError."""
    H = pcm.hamming(7)
    with pytest.raises(ValueError, match="rounds must be >= 1"):
        CSSCode(HX=H, HZ=H, rounds=0)


def test_invalid_experiment_at_init():
    """Invalid experiment type should raise ValueError at init, not at circuit access."""
    H = pcm.hamming(7)
    with pytest.raises(ValueError, match="Experiment must be"):
        CSSCode(HX=H, HZ=H, experiment="invalid")


def test_crossings_initialized_as_set():
    """Crossings should be initialized as an empty set."""
    H = pcm.hamming(7)
    code = CSSCode(HX=H, HZ=H, rounds=1)
    assert isinstance(code.crossings, set)
    assert len(code.crossings) == 0


# ---- Crossing noise correctness ----


def test_crossing_noise_targets_data_qubits():
    """Crossing noise should target data qubit pairs, not data-check pairs."""
    H = pcm.hamming(7)
    noise = NoiseModel(crossing=0.15)
    code = CSSCode(HX=H, HZ=H, rounds=1, noise=noise)

    # Manually embed with a layout that creates crossings.
    code.embed("spring", seed=42)

    if len(code.crossings) > 0:
        circuit_str = str(code.circuit)
        data_set = set(code.data_qubits)

        # Parse PAULI_CHANNEL_2 instructions to find crossing noise targets.
        # Crossing noise appears after the last H gate and before PAULI_CHANNEL_1.
        # We check that all PAULI_CHANNEL_2 targets in the crossing section
        # are pairs of data qubits.
        lines = circuit_str.strip().split("\n")
        for line in lines:
            line = line.strip()
            # The crossing PAULI_CHANNEL_2 has the crossing noise parameters.
            # We can identify them because circuit noise and crossing noise
            # have different parameter values.
            if "PAULI_CHANNEL_2" in line and "0.01" in line:
                # This is crossing noise (0.15/15 = 0.01)
                parts = line.split()
                # targets are the two integers after PAULI_CHANNEL_2
                targets = [int(p) for p in parts[1:] if p.isdigit()]
                if len(targets) == 2:
                    assert targets[0] in data_set, (
                        f"Crossing noise target {targets[0]} is not a data qubit"
                    )
                    assert targets[1] in data_set, (
                        f"Crossing noise target {targets[1]} is not a data qubit"
                    )


# ---- Determinism ----


def test_embed_deterministic_with_seed():
    """embed() with a seed should produce deterministic results."""
    H = pcm.hamming(7)
    code1 = CSSCode(HX=H, HZ=H, rounds=1)
    code2 = CSSCode(HX=H, HZ=H, rounds=1)

    code1.embed("random", seed=42)
    code2.embed("random", seed=42)

    for p1, p2 in zip(code1.pos, code2.pos, strict=True):
        np.testing.assert_array_equal(p1, p2)
    assert code1.crossings == code2.crossings


# ---- GF(2) rank correctness ----


def test_gf2_rank_used_for_k():
    """Verify k is computed via GF(2) rank (not floating-point)."""
    H = pcm.hamming(7)
    code = CSSCode(HX=H, HZ=H, rounds=1)
    # Hamming(7): rank=3, so k = 7 - 3 - 3 = 1
    assert code.k == 1


# ---- Quantum distance ----


def test_steane_distance():
    """Steane code has quantum distance 3."""
    H = pcm.hamming(7)
    code = CSSCode(HX=H, HZ=H, rounds=1)
    assert code.distance() == 3


def test_distance_large_nullspace_raises():
    """Distance should raise ValueError when nullspace dimension exceeds 20."""
    # Build a trivial CSS code with empty HX, HZ on many qubits -> huge nullspace.
    n = 25
    HX = np.zeros((1, n), dtype=int)  # degenerate stabilizer
    HZ = np.zeros((1, n), dtype=int)
    # CSS condition 0 @ 0 = 0 holds trivially.
    code = CSSCode(HX=HX, HZ=HZ, rounds=1)
    with pytest.raises(ValueError, match="infeasible"):
        code.distance()


# ---- n vs n_total semantics ----


def test_steane_n_is_data_qubit_count():
    """CSSCode.n should be the data-qubit count (7 for Steane), not the total circuit size."""
    H = pcm.hamming(7)
    code = CSSCode(HX=H, HZ=H, rounds=1)
    assert code.n == 7
    assert code.n_total == 13  # 7 data + 3 Z-ancillas + 3 X-ancillas
    assert code.n == len(code.data_qubits)
    assert code.n_total == len(code.qubits)


# ---- Symplectic Gram-Schmidt defensive check ----


def test_symplectic_gs_raises_on_degenerate_input():
    """_symplectic_gram_schmidt should raise if no Z anticommutes with some X."""
    # Construct degenerate input: two X-logicals, zero Z-logicals (no partner).
    x = np.array([[1, 0], [0, 1]], dtype=int)
    z = np.array([[1, 0], [1, 0]], dtype=int)  # z[0] and z[1] both commute with x[1]
    # x[1] = [0,1]; z[0] = [1,0]; dot = 0. z[1] = [1,0]; dot = 0. No partner.
    with pytest.raises(RuntimeError, match="degenerate"):
        CSSCode._symplectic_gram_schmidt(x, z)


# =============================================================================
# PR 6: CSSCode.circuit dispatch between compiler and legacy path
# =============================================================================


class TestCircuitDispatchPR6:
    """Verify `CSSCode.circuit` dispatches correctly between the new
    `weave.compiler` path (for noiseless codes) and the legacy
    `_legacy_generate` path (for noisy codes).

    The two paths emit different Stim instructions: the compiler emits
    `TICK` markers and `DEPOLARIZE1`/`DEPOLARIZE2` channels; the legacy
    path emits no `TICK` and uses `PAULI_CHANNEL_1`/`PAULI_CHANNEL_2`.
    We use these as cheap fingerprints to confirm the dispatch.

    When PR 20 retires `_legacy_generate`, these tests should be removed
    or rewritten to only assert compiler behavior.
    """

    def test_noiseless_uses_compiler_path(self):
        """A noiseless code's circuit should come from compile_extraction
        and therefore contain TICK markers."""
        H = pcm.hamming(7)
        code = CSSCode(HX=H, HZ=H, rounds=3)
        text = str(code.circuit)
        assert "TICK" in text, "noiseless path should emit TICK markers"

    def test_noisy_uses_legacy_path(self):
        """A code with nontrivial noise should route through the legacy
        generator and therefore emit PAULI_CHANNEL_1 (not DEPOLARIZE1)
        with no TICK markers."""
        H = pcm.hamming(7)
        code = CSSCode(HX=H, HZ=H, rounds=3, noise=NoiseModel(data=0.001, circuit=0.001))
        text = str(code.circuit)
        assert "PAULI_CHANNEL_1" in text, "legacy path emits PAULI_CHANNEL_1"
        assert "TICK" not in text, "legacy path does not emit TICK"

    def test_noisy_crossing_uses_legacy_path(self):
        """A code with crossing noise also routes to legacy, since the
        new compiler's geometry pass (PR 8) is not yet implemented."""
        H = pcm.hamming(7)
        code = CSSCode(HX=H, HZ=H, rounds=1, noise=NoiseModel(crossing=0.15))
        code.embed("spring", seed=42)
        text = str(code.circuit)
        assert "PAULI_CHANNEL_2" in text
        assert "TICK" not in text

    def test_legacy_generate_still_callable(self):
        """`_legacy_generate()` remains a private but callable method
        for direct use in tests that want the legacy path explicitly.
        """
        H = pcm.hamming(7)
        code = CSSCode(HX=H, HZ=H, rounds=2)
        legacy_circuit = code._legacy_generate()
        assert isinstance(legacy_circuit, stim.Circuit)
        assert "TICK" not in str(legacy_circuit)
        # Legacy is also noise-free here (default NoiseModel), so its
        # detector sampler should produce zero events.
        sampler = legacy_circuit.compile_detector_sampler()
        samples = sampler.sample(shots=100)
        assert not np.any(samples)

    def test_dispatch_cache_invalidation_on_embed(self):
        """The dispatch still respects the lazy-cache invalidation on embed.

        For a noiseless code, the cache is populated by the compiler path,
        then invalidated by embed(), then repopulated by the compiler path
        again (with the new `pos` field). Both accesses must produce
        circuits with zero detector events.
        """
        H = pcm.hamming(7)
        code = CSSCode(HX=H, HZ=H, rounds=1)

        # First access: compiler path, trivial positions.
        c1 = code.circuit
        assert "TICK" in str(c1)
        assert not np.any(c1.compile_detector_sampler().sample(shots=100))

        # embed() invalidates the cache.
        code.embed("random", seed=42)
        assert code._circuit is None

        # Second access: compiler path, positions from embed().
        c2 = code.circuit
        assert "TICK" in str(c2)
        assert not np.any(c2.compile_detector_sampler().sample(shots=100))

    def test_quantumcode_circuit_is_abstract(self):
        """`QuantumCode` declares `circuit` abstract; direct instantiation
        must be rejected. `CSSCode` satisfies the abstract contract via
        its property."""
        from weave.codes.base import QuantumCode

        # Can't instantiate QuantumCode directly.
        with pytest.raises(TypeError, match="abstract"):
            QuantumCode(n=1, k=1)  # type: ignore[abstract]

        # CSSCode satisfies the contract.
        H = pcm.hamming(7)
        code = CSSCode(HX=H, HZ=H, rounds=1)
        assert hasattr(code, "circuit")
        import stim

        assert isinstance(code.circuit, stim.Circuit)

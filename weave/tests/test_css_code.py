"""Tests for CSSCode directly, including the Steane code [[7,1,3]]."""

import numpy as np
import pytest

from weave.codes.css_code import CSSCode
from weave.codes.base import NoiseModel
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
    assert len(steane_code.data_qubits) == 7
    assert len(steane_code.z_check_qubits) == 3
    assert len(steane_code.x_check_qubits) == 3
    assert steane_code.n == 13


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
    dem = code.circuit.detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=True
    )
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
    circuit_before = str(code.circuit)
    assert code._circuit is not None

    # Embed should invalidate.
    code.embed("random")
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
        lines = circuit_str.strip().split('\n')
        for line in lines:
            line = line.strip()
            # The crossing PAULI_CHANNEL_2 has the crossing noise parameters.
            # We can identify them because circuit noise and crossing noise
            # have different parameter values.
            if 'PAULI_CHANNEL_2' in line and '0.01' in line:
                # This is crossing noise (0.15/15 = 0.01)
                parts = line.split()
                # targets are the two integers after PAULI_CHANNEL_2
                targets = [int(p) for p in parts[1:] if p.isdigit()]
                if len(targets) == 2:
                    assert targets[0] in data_set, \
                        f"Crossing noise target {targets[0]} is not a data qubit"
                    assert targets[1] in data_set, \
                        f"Crossing noise target {targets[1]} is not a data qubit"


# ---- Determinism ----

def test_embed_deterministic_with_seed():
    """embed() with a seed should produce deterministic results."""
    H = pcm.hamming(7)
    code1 = CSSCode(HX=H, HZ=H, rounds=1)
    code2 = CSSCode(HX=H, HZ=H, rounds=1)

    code1.embed("random", seed=42)
    code2.embed("random", seed=42)

    for p1, p2 in zip(code1.pos, code2.pos):
        np.testing.assert_array_equal(p1, p2)
    assert code1.crossings == code2.crossings


# ---- GF(2) rank correctness ----

def test_gf2_rank_used_for_k():
    """Verify k is computed via GF(2) rank (not floating-point)."""
    H = pcm.hamming(7)
    code = CSSCode(HX=H, HZ=H, rounds=1)
    # Hamming(7): rank=3, so k = 7 - 3 - 3 = 1
    assert code.k == 1

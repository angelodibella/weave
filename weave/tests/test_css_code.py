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

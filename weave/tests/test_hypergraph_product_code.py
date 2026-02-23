import numpy as np
import pytest

from weave.codes.hypergraph_product_code import HypergraphProductCode
from weave.codes.base import NoiseModel
from weave.util import pcm


@pytest.fixture
def rep3_matrices():
    """Return a pair of repetition(3) parity-check matrices."""
    H = pcm.repetition(3)  # Shape: (2, 3)
    return H, H


@pytest.fixture
def rep3_hamming7_matrices():
    """Return repetition(3) and Hamming(7) parity-check matrices."""
    H_rep = pcm.repetition(3)  # Shape: (2, 3)
    H_ham = pcm.hamming(7)  # Shape: (3, 7)
    return H_rep, H_ham


def test_hp_code_parameters(rep3_matrices):
    """Verify basic code parameters for rep(3) x rep(3)."""
    H1, H2 = rep3_matrices
    code = HypergraphProductCode(H1, H2, rounds=1)
    assert code.k == 1
    assert len(code.data_qubits) == 13
    assert len(code.z_check_qubits) == 6
    assert len(code.x_check_qubits) == 6
    assert code.n == 25


def test_hp_code_stim_circuit(rep3_matrices):
    """Verify that the Stim circuit contains key instructions."""
    H1, H2 = rep3_matrices
    code = HypergraphProductCode(H1, H2, rounds=1)
    circuit_str = str(code.circuit)
    for cmd in ["R", "M", "DETECTOR", "OBSERVABLE_INCLUDE"]:
        assert cmd in circuit_str


def test_hp_code_noiseless_detectors(rep3_matrices):
    """Noiseless circuit should produce no detector events."""
    H1, H2 = rep3_matrices
    code = HypergraphProductCode(H1, H2, rounds=3)
    sampler = code.circuit.compile_detector_sampler()
    samples = sampler.sample(shots=100)
    assert not np.any(samples)


def test_hp_code_noiseless_x_memory(rep3_matrices):
    """Noiseless x_memory circuit should produce no detector events."""
    H1, H2 = rep3_matrices
    code = HypergraphProductCode(H1, H2, rounds=3, experiment="x_memory")
    sampler = code.circuit.compile_detector_sampler()
    samples = sampler.sample(shots=100)
    assert not np.any(samples)


def test_hp_code_noisy_dem(rep3_matrices):
    """Noisy circuit should produce a valid detector error model."""
    H1, H2 = rep3_matrices
    noise = NoiseModel(data=0.01, circuit=0.01)
    code = HypergraphProductCode(H1, H2, rounds=3, noise=noise)
    dem = code.circuit.detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=True
    )
    assert dem.num_detectors > 0
    assert dem.num_observables == 1


def test_hp_code_logicals_count(rep3_hamming7_matrices):
    """
    Verify that logical operator extraction returns the expected number.

    For repetition(3): k1 = 3 - 2 = 1; for Hamming(7): k2 = 7 - 3 = 4;
    Hence, the product code should have k1*k2 = 4 logical operators.
    """
    H_rep, H_ham = rep3_hamming7_matrices
    code = HypergraphProductCode(H_rep, H_ham, rounds=1)
    x_logicals, z_logicals = code.find_logicals()
    assert x_logicals.shape[0] == 4
    assert z_logicals.shape[0] == 4


def test_hp_code_logicals_validity(rep3_hamming7_matrices):
    """Verify that extracted logicals are in the correct kernels."""
    H_rep, H_ham = rep3_hamming7_matrices
    code = HypergraphProductCode(H_rep, H_ham, rounds=1)
    x_logicals, z_logicals = code.find_logicals()

    # X logicals must be in ker(HZ)
    for lx in x_logicals:
        assert np.all(code.HZ @ lx % 2 == 0)

    # Z logicals must be in ker(HX)
    for lz in z_logicals:
        assert np.all(code.HX @ lz % 2 == 0)


def test_hp_code_logicals_symplectic_pairing(rep3_hamming7_matrices):
    """Verify that X and Z logicals form a symplectic basis."""
    H_rep, H_ham = rep3_hamming7_matrices
    code = HypergraphProductCode(H_rep, H_ham, rounds=1)
    x_logicals, z_logicals = code.find_logicals()
    k = x_logicals.shape[0]

    comm_matrix = (x_logicals @ z_logicals.T) % 2
    np.testing.assert_array_equal(
        comm_matrix, np.eye(k, dtype=int),
        err_msg="Logical operators are not in a symplectic basis"
    )


def test_hp_code_logicals_symplectic_large():
    """Test symplectic pairing for a larger code (hamming(7) x hamming(7), k=16)."""
    Hh = pcm.hamming(7)
    code = HypergraphProductCode(Hh, Hh, rounds=1)
    x_logicals, z_logicals = code.find_logicals()

    assert x_logicals.shape[0] == 16
    assert z_logicals.shape[0] == 16

    comm_matrix = (x_logicals @ z_logicals.T) % 2
    np.testing.assert_array_equal(comm_matrix, np.eye(16, dtype=int))


def test_hp_code_parity_check_dimensions(rep3_hamming7_matrices):
    """
    Verify hypergraph product matrix dimensions.

    For repetition(3) (2x3) and Hamming(7) (3x7):
      HX: (2*7) x (3*7 + 2*3) = 14 x 27.
      HZ: (3*3) x 27 = 9 x 27.
    """
    H_rep, H_ham = rep3_hamming7_matrices
    code = HypergraphProductCode(H_rep, H_ham, rounds=1)
    assert code.HX.shape == (14, 27)
    assert code.HZ.shape == (9, 27)


def test_hp_code_invalid_experiment(rep3_matrices):
    """Verify that an invalid experiment type raises a ValueError."""
    H1, H2 = rep3_matrices
    with pytest.raises(ValueError):
        HypergraphProductCode(H1, H2, rounds=1, experiment="invalid")


def test_hp_code_embed(rep3_matrices):
    """Verify that embed() works and regenerates the circuit."""
    H1, H2 = rep3_matrices
    code = HypergraphProductCode(H1, H2, rounds=2)
    circuit_before = str(code.circuit)

    code.embed("random")
    assert code.graph is not None
    assert code.pos is not None

    # Circuit should still be valid after re-embedding
    sampler = code.circuit.compile_detector_sampler()
    samples = sampler.sample(shots=100)
    assert not np.any(samples)

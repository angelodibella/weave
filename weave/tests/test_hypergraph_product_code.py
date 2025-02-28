import pytest

from weave.codes.hypergraph_product_code import HypergraphProductCode
from weave.codes.base import NoiseModel
from weave.util import pcm


@pytest.fixture
def simple_clists():
    """
    Return a simple clist pair using a repetition code of length 4.
    """
    H = pcm.repetition(4)  # Shape: (3, 4)
    clist = pcm.to_clist(H)
    return clist, clist


@pytest.fixture
def rep3_hamming7():
    """
    Return a clist pair for a repetition(3) code and a Hamming(7) code.

    For repetition(3): H has shape (2, 3).
    For Hamming(7): H has shape (3, 7).
    """
    H_rep = pcm.repetition(3)  # Shape: (2, 3)
    clist_rep = pcm.to_clist(H_rep)
    H_ham = pcm.hamming(7)  # Shape: (3, 7)
    clist_ham = pcm.to_clist(H_ham)
    return clist_rep, clist_ham


def test_hp_code_graph_dimensions(simple_clists):
    """
    Verify that the Tanner graph has the expected number of nodes.
    """
    clist1, clist2 = simple_clists
    code = HypergraphProductCode(clist1, clist2, rounds=1, pos="random", noise=NoiseModel())
    expected_nodes = len(code.data_qubits) + len(code.z_check_qubits) + len(code.x_check_qubits)
    assert code.graph.number_of_nodes() == expected_nodes


def test_hp_code_crossings(simple_clists):
    """
    Verify that the crossing number is an integer ≥ 0.
    """
    clist1, clist2 = simple_clists
    code = HypergraphProductCode(clist1, clist2, rounds=1, pos="random", noise=NoiseModel())
    assert isinstance(code.crossing_number(), int)
    assert code.crossing_number() >= 0


def test_hp_code_stim_circuit(simple_clists):
    """
    Verify that the Stim circuit contains key instructions.
    """
    clist1, clist2 = simple_clists
    noise = NoiseModel(data=0.0, z_check=0.0, x_check=0.0, circuit=0.0, crossing=0.0)
    code = HypergraphProductCode(clist1, clist2, rounds=1, pos="random", noise=noise, experiment="z_memory")
    circuit_str = str(code.circuit)
    for cmd in ["R", "M", "DETECTOR", "OBSERVABLE_INCLUDE"]:
        assert cmd in circuit_str


def test_hp_code_logicals(rep3_hamming7):
    """
    Verify that logical operator extraction returns the expected number (4) of logicals.

    For repetition(3): k = 3 - 2 = 1; for Hamming(7): k = 7 - 3 = 4;
    Hence, the product code should have 1*4 = 4 logical operators.
    """
    clist1, clist2 = rep3_hamming7
    code = HypergraphProductCode(clist1, clist2, rounds=1, pos="random", noise=NoiseModel(), experiment="z_memory")
    x_logicals, z_logicals = code.find_logicals()
    assert x_logicals.shape[0] == 4, f"Expected 4 X logicals, got {x_logicals.shape[0]}"
    assert z_logicals.shape[0] == 4, f"Expected 4 Z logicals, got {z_logicals.shape[0]}"


def test_hp_code_parity_check_dimensions(rep3_hamming7):
    """
    Verify that the hypergraph product matrices have the expected dimensions.

    For repetition(3) (2x3) and Hamming(7) (3x7):
      Expected HX: (2*7) x (3*7 + 2*3) = 14 x 27.
      Expected HZ: (3*3) x 27 = 9 x 27.
    """
    clist1, clist2 = rep3_hamming7
    code = HypergraphProductCode(clist1, clist2, rounds=1, pos="random", noise=NoiseModel())
    HX, HZ = code.HX, code.HZ
    assert HX.shape == (14, 27), f"Unexpected HX dimensions: {HX.shape}"
    assert HZ.shape == (9, 27), f"Unexpected HZ dimensions: {HZ.shape}"


def test_hp_code_invalid_experiment(simple_clists):
    """
    Verify that an invalid experiment type raises a ValueError.
    """
    clist1, clist2 = simple_clists
    with pytest.raises(ValueError):
        HypergraphProductCode(clist1, clist2, rounds=1, pos="random", noise=NoiseModel(), experiment="invalid")

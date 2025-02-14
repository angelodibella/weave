import pytest
from weave.codes.hypergraph_product_code import HypergraphProductCode
from weave.codes.base import NoiseModel
from weave.util import pcm

@pytest.fixture
def simple_clists():
    # Generate a simple clist from a repetition code (n=4) for both inputs.
    H = pcm.repetition(4)
    clist = pcm.to_clist(H)
    return clist, clist  # Use the same clist for both for simplicity.

def test_hypergraph_product_code_dimensions(simple_clists):
    clist1, clist2 = simple_clists
    code = HypergraphProductCode(clist1, clist2, rounds=1, pos="random", noise=NoiseModel())
    # The expected number of nodes is the sum of data, Z-check, and X-check qubits.
    expected_nodes = len(code.data_qubits) + len(code.z_check_qubits) + len(code.x_check_qubits)
    assert code.graph.number_of_nodes() == expected_nodes

def test_hypergraph_product_code_crossings(simple_clists):
    clist1, clist2 = simple_clists
    code = HypergraphProductCode(clist1, clist2, rounds=1, pos="spring", noise=NoiseModel())
    # The crossing number should be an integer greater than or equal to zero.
    assert isinstance(code.crossing_number(), int)
    assert code.crossing_number() >= 0

def test_hypergraph_product_code_stim_circuit(simple_clists):
    clist1, clist2 = simple_clists
    noise = NoiseModel(data=0.0, z_check=0.0, x_check=0.0, circuit=0.0, crossing=0.0)
    code = HypergraphProductCode(clist1, clist2, rounds=1, pos="random", noise=noise, experiment="z_memory")
    # Convert the Stim circuit to a string and check for key instructions.
    circuit_str = str(code.circuit)
    # Check that the circuit includes reset ("R"), measurement ("M"), and detector ("DETECTOR") operations.
    assert "R" in circuit_str
    assert "M" in circuit_str
    assert "DETECTOR" in circuit_str
    # Check for observable include instructions (assuming logical operators are present).
    assert "OBSERVABLE_INCLUDE" in circuit_str

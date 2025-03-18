"""
Tests for core C++ functionality
"""
import pytest
import weave


def test_hypergraph_product_code_creation():
    """Test creating and using a HypergraphProductCode."""
    # Create empty code
    code = weave.HypergraphProductCode()
    
    # Create and initialize with parity check matrices
    pcx = [[1, 0, 1], [0, 1, 1]]
    pcz = [[1, 1, 0], [1, 0, 1]]
    
    # Generate the code
    code.generate(pcx, pcz)
    
    # Get parameters
    params = code.get_parameters()
    assert len(params) == 3, "Should return [n, k, d]"
    
    # Get stabilizers
    stabilizers = code.get_stabilizers()
    assert len(stabilizers) > 0, "Should return at least one stabilizer"
import numpy as np
import pytest
from weave.codes.base import NoiseModel

def test_noise_model_numeric():
    # Test that numeric inputs are converted correctly to lists.
    nm = NoiseModel(data=0.3, z_check=0.2, x_check=0.1, circuit=0.15, crossing=0.2)
    # For data, expected: [0.1, 0.1, 0.1]
    np.testing.assert_almost_equal(nm.data, [0.1, 0.1, 0.1])
    # For z_check and x_check: each should be value/3 repeated 3 times.
    np.testing.assert_almost_equal(nm.z_check, [0.06666667]*3, decimal=6)
    np.testing.assert_almost_equal(nm.x_check, [0.03333333]*3, decimal=6)
    # For circuit and crossing: expected lists of length 15.
    assert len(nm.circuit) == 15
    assert len(nm.crossing) == 15

def test_noise_model_list():
    # Provide lists of the correct length.
    data_list = [0.1, 0.1, 0.1]
    circuit_list = [0.01] * 15
    crossing_list = [0.02] * 15
    nm = NoiseModel(data=data_list, z_check=data_list, x_check=data_list,
                    circuit=circuit_list, crossing=crossing_list)
    np.testing.assert_almost_equal(nm.data, data_list)
    np.testing.assert_almost_equal(nm.circuit, circuit_list)
    np.testing.assert_almost_equal(nm.crossing, crossing_list)

def test_noise_model_invalid_length():
    # Test that providing an invalid list length raises an assertion error.
    with pytest.raises(AssertionError):
        # Data should have length 3; providing a list of 2 should trigger an error.
        NoiseModel(data=[0.1, 0.1])

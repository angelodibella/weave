import numpy as np
import pytest
from weave.codes.base import NoiseModel


def test_noise_model_numeric():
    # Test that numeric inputs are converted correctly to lists.
    nm = NoiseModel(data=0.3, z_check=0.2, x_check=0.1, circuit=0.15, crossing=0.2)

    # For data, expected: [0.1, 0.1, 0.1]
    np.testing.assert_almost_equal(nm.data, [0.1, 0.1, 0.1])

    # For z_check and x_check: each should be value/3 repeated 3 times.
    np.testing.assert_almost_equal(nm.z_check, [0.06666667] * 3, decimal=6)
    np.testing.assert_almost_equal(nm.x_check, [0.03333333] * 3, decimal=6)

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
    # Test that providing an invalid list length raises a ValueError.
    with pytest.raises(ValueError):
        NoiseModel(data=[0.1, 0.1])


def test_noise_model_zero():
    """Default NoiseModel should have all zeros."""
    nm = NoiseModel()
    assert all(v == 0.0 for v in nm.data)
    assert all(v == 0.0 for v in nm.circuit)
    assert all(v == 0.0 for v in nm.crossing)


def test_noise_model_int_input():
    """Integer noise values should work (not just floats)."""
    nm = NoiseModel(data=0, circuit=0, crossing=0)
    assert all(v == 0.0 for v in nm.data)


def test_noise_model_invalid_circuit_length():
    """Circuit noise with wrong length should raise ValueError."""
    with pytest.raises(ValueError):
        NoiseModel(circuit=[0.01] * 10)


def test_noise_model_invalid_crossing_length():
    """Crossing noise with wrong length should raise ValueError."""
    with pytest.raises(ValueError):
        NoiseModel(crossing=[0.01] * 3)


def test_noise_model_negative_scalar():
    """Negative scalar noise should raise ValueError."""
    with pytest.raises(ValueError, match="non-negative"):
        NoiseModel(data=-0.1)


def test_noise_model_negative_list_element():
    """Negative element in noise list should raise ValueError."""
    with pytest.raises(ValueError, match="non-negative"):
        NoiseModel(data=[0.1, -0.05, 0.1])


def test_noise_model_sum_exceeds_one_scalar():
    """Scalar noise that results in sum > 1 should raise ValueError."""
    # data=3.1 => each element is 3.1/3 ≈ 1.033, sum ≈ 3.1 > 1
    with pytest.raises(ValueError, match="sum"):
        NoiseModel(data=3.1)


def test_noise_model_sum_exceeds_one_list():
    """List noise summing > 1 should raise ValueError."""
    with pytest.raises(ValueError, match="sum"):
        NoiseModel(data=[0.4, 0.4, 0.3])


def test_noise_model_immutable():
    """NoiseModel values should be tuples (immutable)."""
    nm = NoiseModel(data=0.3)
    assert isinstance(nm.data, tuple)
    assert isinstance(nm.circuit, tuple)
    assert isinstance(nm.crossing, tuple)
    assert isinstance(nm.z_check, tuple)
    assert isinstance(nm.x_check, tuple)


def test_noise_model_boundary_sum():
    """Noise parameters summing to exactly 1 should be accepted."""
    # 3 params summing to exactly 1.0
    nm = NoiseModel(data=[0.4, 0.3, 0.3])
    np.testing.assert_almost_equal(sum(nm.data), 1.0)

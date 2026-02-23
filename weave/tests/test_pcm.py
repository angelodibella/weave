import numpy as np
import pytest
from weave.util import pcm


def test_repetition():
    # For n=4, repetition should return a matrix of shape (3,4).
    H = pcm.repetition(4)
    assert H.shape == (3, 4)

    # Each row should have exactly two 1s (adjacent bits).
    for i in range(H.shape[0]):
        assert np.sum(H[i]) == 2
        assert H[i, i] == 1
        assert H[i, i + 1] == 1


def test_hamming_valid():
    # For n=7, Hamming should return a matrix of shape (m,7) where m = ceil(log2(8)) = 3.
    H = pcm.hamming(7)
    assert H.shape == (3, 7)

    # Verify that all elements are 0 or 1.
    assert np.all((H == 0) | (H == 1))


def test_hamming_invalid():
    # For an invalid n (not equal to 2^m - 1), hamming should raise a ValueError.
    with pytest.raises(ValueError):
        pcm.hamming(6)


def test_hypergraph_product_dimensions():
    # Create simple matrices using repetition codes.
    H1 = pcm.repetition(4)  # shape (3,4)
    H2 = pcm.repetition(5)  # shape (4,5)
    HX, HZ = pcm.hypergraph_product(H1, H2, reordered=False)
    # Expected dimensions:
    # HX: (r1*n2) x (n1*n2 + r1*r2) = (3*5) x (4*5 + 3*4) = 15 x (20+12) = 15 x 32.
    # HZ: (n1*r2) x (n1*n2 + r1*r2) = (4*4) x 32 = 16 x 32.
    assert HX.shape == (15, 32)
    assert HZ.shape == (16, 32)


def test_css_commutation():
    # Test that HX * HZ^T mod 2 equals the zero matrix.
    H1 = pcm.repetition(4)
    H2 = pcm.repetition(5)
    HX, HZ = pcm.hypergraph_product(H1, H2, reordered=False)

    product = np.mod(np.dot(HX, HZ.T), 2)
    assert np.all(product == 0)


def test_to_matrix_and_to_clist():
    # Create a repetition code matrix, convert to clist, then back to a matrix, and ensure the reconstructed matrix
    # matches the original.
    H = pcm.repetition(4)
    clist = pcm.to_clist(H)
    H_reconstructed = pcm.to_matrix(clist)
    np.testing.assert_array_equal(H, H_reconstructed)

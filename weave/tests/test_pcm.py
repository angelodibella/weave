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


# ---- GF(2) linear algebra tests ----

def test_row_echelon_identity():
    I = np.eye(3, dtype=int)
    reduced, rank, _, pivots = pcm.row_echelon(I)
    assert rank == 3
    assert pivots == [0, 1, 2]
    np.testing.assert_array_equal(reduced, I)


def test_row_echelon_rank_deficient():
    # Row 2 = Row 0 + Row 1 over GF(2)
    M = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=int)
    reduced, rank, _, pivots = pcm.row_echelon(M)
    assert rank == 2
    assert len(pivots) == 2


def test_row_echelon_transform():
    H = pcm.hamming(7)
    reduced, rank, transform, pivots = pcm.row_echelon(H)
    # transform @ H should equal reduced (mod 2)
    np.testing.assert_array_equal((transform @ H) % 2, reduced)


def test_nullspace_basic():
    # Repetition code: nullspace should be the all-ones vector.
    H = pcm.repetition(4)
    ker = pcm.nullspace(H)
    assert ker.shape[0] == 1  # k = n - rank = 4 - 3 = 1
    # Verify it's actually in the nullspace.
    assert np.all(H @ ker[0] % 2 == 0)


def test_nullspace_hamming():
    H = pcm.hamming(7)
    ker = pcm.nullspace(H)
    # Hamming(7,4): nullspace dimension = 7 - 3 = 4
    assert ker.shape[0] == 4
    # Every nullspace vector should satisfy H @ v = 0 mod 2.
    for v in ker:
        assert np.all(H @ v % 2 == 0)


def test_nullspace_full_rank():
    # Full-rank square matrix should have empty nullspace.
    M = np.eye(5, dtype=int)
    ker = pcm.nullspace(M)
    assert ker.shape[0] == 0


def test_row_basis():
    H = pcm.hamming(7)
    basis = pcm.row_basis(H)
    assert basis.shape[0] == 3  # rank of Hamming(7) is 3
    # Basis rows should be independent.
    assert np.linalg.matrix_rank(basis.astype(float)) == 3


def test_row_basis_with_dependent_rows():
    # Add a dependent row.
    H = pcm.repetition(3)  # 2x3
    H_ext = np.vstack([H, (H[0] + H[1]) % 2])  # 3x3, rank still 2
    basis = pcm.row_basis(H_ext)
    assert basis.shape[0] == 2


def test_nullspace_and_rank_sum():
    """rank + nullity = n for any binary matrix."""
    for H in [pcm.repetition(5), pcm.hamming(7), pcm.hamming(15)]:
        n = H.shape[1]
        rank = pcm.row_echelon(H)[1]
        nullity = pcm.nullspace(H).shape[0]
        assert rank + nullity == n, f"rank({rank}) + nullity({nullity}) != n({n})"


# ---- Distance tests ----

def test_distance_repetition():
    """Repetition code of length n has distance n."""
    H = pcm.repetition(5)
    assert pcm.distance(H) == 5


def test_distance_hamming():
    """Hamming(7,4,3) code has distance 3."""
    H = pcm.hamming(7)
    assert pcm.distance(H) == 3


# ---- Hypergraph product with asymmetric inputs ----

def test_hypergraph_product_asymmetric():
    """Test HP code where r1 > n1 (more checks than bits in H1.T)."""
    H1 = pcm.repetition(3)  # (2, 3)
    H2 = pcm.hamming(7)     # (3, 7)
    HX, HZ = pcm.hypergraph_product(H1, H2, reordered=True)
    # CSS condition must hold.
    assert np.all(np.mod(HX @ HZ.T, 2) == 0)


def test_hypergraph_product_reordered_preserves_css():
    """Both reordered and non-reordered should satisfy CSS condition."""
    H1 = pcm.repetition(4)
    H2 = pcm.repetition(3)
    for reordered in (True, False):
        HX, HZ = pcm.hypergraph_product(H1, H2, reordered=reordered)
        assert np.all(np.mod(HX @ HZ.T, 2) == 0)


# ---- Row-reduce and find_pivot_columns convenience ----

def test_row_reduce():
    """row_reduce should return reduced form."""
    M = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=int)
    reduced = pcm.row_reduce(M)
    assert reduced.shape == M.shape


def test_find_pivot_columns():
    """find_pivot_columns should return correct pivots for identity."""
    I = np.eye(4, dtype=int)
    assert pcm.find_pivot_columns(I) == [0, 1, 2, 3]


# ---- Repetition code validation ----

def test_repetition_n_less_than_2():
    """repetition(n) should raise ValueError for n < 2."""
    with pytest.raises(ValueError, match="must be >= 2"):
        pcm.repetition(1)

    with pytest.raises(ValueError, match="must be >= 2"):
        pcm.repetition(0)


# ---- Distance with k>1 (combination enumeration) ----

def test_distance_k_greater_than_1():
    """Test distance computation for a code where the minimum-weight codeword
    is a combination of basis vectors, not an individual basis vector."""
    # Hamming(15, 11, 3) has k=11, but distance is still 3.
    H = pcm.hamming(15)
    assert pcm.distance(H) == 3


def test_distance_repetition_code_combinations():
    """Repetition code should still return correct distance with new implementation."""
    for n in [3, 4, 5, 7]:
        H = pcm.repetition(n)
        assert pcm.distance(H) == n

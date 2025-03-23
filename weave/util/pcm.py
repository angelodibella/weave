"""Parity-check matrix utilities for classical and quantum error correction."""

from typing import List, Tuple, Union, Optional
import numpy as np
from ldpc import mod2


def repetition(n: int) -> np.ndarray:
    """
    Construct the parity-check matrix for a repetition code.

    Parameters
    ----------
    n : int
        Length of the repetition code.

    Returns
    -------
    np.ndarray
        A (n-1) x n parity-check matrix.
    """
    H = np.zeros((n - 1, n), dtype=int)
    np.fill_diagonal(H, 1)
    for i in range(n - 1):
        H[i, i + 1] = 1
    return H


def hamming(n: int) -> np.ndarray:
    """
    Construct the parity-check matrix for a Hamming code.

    The parameters must satisfy n = 2**m - 1 for some integer m.

    Parameters
    ----------
    n : int
        Length of the Hamming code.

    Returns
    -------
    np.ndarray
        An m x n parity-check matrix.

    Raises
    ------
    ValueError
        If n is not of the form 2**m - 1.
    """
    m = int(np.ceil(np.log2(n + 1)))
    if 2**m - 1 != n:
        raise ValueError("Invalid n for a Hamming code. Ensure n = 2^m - 1.")

    H = np.zeros((m, n), dtype=int)
    for i in range(1, n + 1):
        binary_str = format(i, f"0{m}b")[::-1]
        H[:, i - 1] = [int(bit) for bit in binary_str]
    return H


# TODO: Address interleaving.
def hypergraph_product(
    H1: np.ndarray, H2: np.ndarray, reordered: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the hypergraph product of two parity-check matrices.

    Given H1 (r1 x n1) and H2 (r2 x n2), the product yields:

        HX = [ H1 ⊗ I(n2)  |  I(r1) ⊗ H2^T ]
        HZ = [ I(n1) ⊗ H2  |  H1^T ⊗ I(r2) ]

    Parameters
    ----------
    H1 : np.ndarray
        First parity-check matrix of shape (r1, n1).
    H2 : np.ndarray
        Second parity-check matrix of shape (r2, n2).
    reordered : bool, optional
        Whether to interleave columns in a canonical order (default is True).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple (HX, HZ) of binary matrices.
    """
    r1, n1 = H1.shape
    r2, n2 = H2.shape

    HX_left = np.kron(H1, np.eye(n2, dtype=int))
    HX_right = np.kron(np.eye(r1, dtype=int), H2.T)
    HX = np.append(HX_left, HX_right, axis=1)

    HZ_left = np.kron(np.eye(n1, dtype=int), H2)
    HZ_right = np.kron(H1.T, np.eye(r2, dtype=int))
    HZ = np.append(HZ_left, HZ_right, axis=1)

    if reordered:
        HX = _reorder_matrix(HX, HX_left, HX_right, n1, r1)
        HZ = _reorder_matrix(HZ, HZ_left, HZ_right, n1, r1)

    return HX.astype(int), HZ.astype(int)


def _reorder_matrix(
    full_matrix: np.ndarray,
    left_part: np.ndarray,
    right_part: np.ndarray,
    n1: int,
    r1: int,
) -> np.ndarray:
    """
    Helper function to reorder matrices by interleaving columns.

    Parameters
    ----------
    full_matrix : np.ndarray
        The original concatenated matrix.
    left_part : np.ndarray
        The left part of the matrix to reorder.
    right_part : np.ndarray
        The right part of the matrix to reorder.
    n1 : int
        First dimension parameter.
    r1 : int
        Second dimension parameter.

    Returns
    -------
    np.ndarray
        The reordered matrix.
    """
    left_split = np.split(left_part, n1, axis=1)
    right_split = np.split(right_part, r1, axis=1)
    parts = []
    for i in range(n1):
        parts.append(left_split[i])
        if i < r1:
            parts.append(right_split[i])
    return np.concatenate(parts, axis=1)


def to_clist(H: np.ndarray) -> List:
    """
    Convert a parity-check matrix to its classical list (clist) representation.

    The clist starts with "B" tokens for each bit (column in H) followed by sequences
    starting with "C" and then the indices of bits that are 1 in each check row.

    Parameters
    ----------
    H : np.ndarray
        The parity-check matrix.

    Returns
    -------
    List
        The clist representation.
    """
    clist = ["B"] * H.shape[1]
    for row in H:
        clist.append("C")
        for i, val in enumerate(row):
            if val == 1:
                clist.append(i)
    return clist


def to_matrix(clist: List) -> np.ndarray:
    """
    Reconstruct a parity-check matrix from its clist representation.

    Parameters
    ----------
    clist : List
        The classical list representation, with an initial block of "B" tokens followed by
        check sequences introduced by "C".

    Returns
    -------
    np.ndarray
        The reconstructed parity-check matrix.
    """
    num_bits = clist.count("B")
    rows = []
    i = 0
    while i < len(clist):
        if clist[i] == "C":
            one_hot = np.zeros(num_bits, dtype=int)
            i += 1
            while i < len(clist) and not isinstance(clist[i], str):
                one_hot[clist[i]] = 1
                i += 1
            rows.append(one_hot)
        else:
            i += 1
    return np.array(rows, dtype=int)


def find_pivot_columns(matrix: np.ndarray) -> List[int]:
    """
    Find the pivot columns of a binary matrix using Gaussian elimination over GF(2).

    Parameters
    ----------
    matrix : np.ndarray
        A binary matrix (elements 0 or 1).

    Returns
    -------
    List[int]
        Indices of the pivot columns.
    """
    # Create a copy to avoid modifying the original matrix
    mat_copy = matrix.copy()
    rows, cols = mat_copy.shape
    pivot_columns = []
    row_index = 0

    for col_index in range(cols):
        pivot_found = False
        for i in range(row_index, rows):
            if mat_copy[i, col_index] == 1:
                pivot_columns.append(col_index)
                pivot_found = True
                if i != row_index:
                    mat_copy[[i, row_index]] = mat_copy[[row_index, i]]
                for j in range(rows):
                    if j != row_index and mat_copy[j, col_index] == 1:
                        mat_copy[j] = (mat_copy[j] + mat_copy[row_index]) % 2
                row_index += 1
                break
        if row_index >= rows:
            break
    return pivot_columns


def row_reduce(matrix: np.ndarray) -> np.ndarray:
    """
    Perform Gaussian elimination over GF(2) to get a row-reduced form.

    Parameters
    ----------
    matrix : np.ndarray
        A binary matrix to row-reduce.

    Returns
    -------
    np.ndarray
        The row-reduced form of the matrix.
    """
    result = mod2.row_echelon(matrix)[0]
    return result


def nullspace(matrix: np.ndarray) -> np.ndarray:
    """
    Compute the nullspace of a binary matrix over GF(2).

    Parameters
    ----------
    matrix : np.ndarray
        A binary matrix.

    Returns
    -------
    np.ndarray
        A matrix whose rows form a basis for the nullspace.
    """
    return mod2.nullspace(matrix)


def distance(H: np.ndarray) -> int:
    """
    Compute the minimum distance of a code defined by parity-check matrix H.

    For small codes only, as this is a brute-force computation.

    Parameters
    ----------
    H : np.ndarray
        The parity-check matrix.

    Returns
    -------
    int
        The minimum distance of the code.
    """
    n = H.shape[1]
    ker = nullspace(H)

    if ker.shape[0] == 0:  # Trivial code
        return float("inf")

    # Compute weights of all non-zero codewords
    min_weight = float("inf")
    for codeword in ker:
        weight = np.sum(codeword)
        if 0 < weight < min_weight:
            min_weight = weight

    return min_weight

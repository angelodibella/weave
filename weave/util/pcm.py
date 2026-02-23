"""Parity-check matrix utilities for classical and quantum error correction."""

from typing import List, Tuple, Union, Optional
import numpy as np


# =============================================================================
# GF(2) Linear Algebra
# =============================================================================

def row_echelon(matrix: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray, List[int]]:
    """
    Perform Gaussian elimination over GF(2) to get row echelon form.

    Parameters
    ----------
    matrix : np.ndarray
        A binary matrix (elements 0 or 1).

    Returns
    -------
    Tuple[np.ndarray, int, np.ndarray, List[int]]
        (reduced_matrix, rank, transformation_matrix, pivot_columns).
        - reduced_matrix: Row echelon form of the input.
        - rank: The rank of the matrix.
        - transformation_matrix: The row operations applied (M such that M @ matrix = reduced).
        - pivot_columns: Indices of the pivot columns.
    """
    mat = np.array(matrix, dtype=int) % 2
    rows, cols = mat.shape
    transform = np.eye(rows, dtype=int)
    pivots = []
    row_idx = 0

    for col in range(cols):
        # Find pivot in this column.
        pivot = None
        for r in range(row_idx, rows):
            if mat[r, col] == 1:
                pivot = r
                break
        if pivot is None:
            continue

        pivots.append(col)

        # Swap rows.
        if pivot != row_idx:
            mat[[row_idx, pivot]] = mat[[pivot, row_idx]]
            transform[[row_idx, pivot]] = transform[[pivot, row_idx]]

        # Eliminate all other 1s in this column.
        for r in range(rows):
            if r != row_idx and mat[r, col] == 1:
                mat[r] = (mat[r] + mat[row_idx]) % 2
                transform[r] = (transform[r] + transform[row_idx]) % 2

        row_idx += 1

    return mat, len(pivots), transform, pivots


def row_basis(matrix: np.ndarray) -> np.ndarray:
    """
    Compute a row basis of a binary matrix over GF(2).

    Returns the independent rows from the row echelon form.

    Parameters
    ----------
    matrix : np.ndarray
        A binary matrix.

    Returns
    -------
    np.ndarray
        A matrix whose rows form a basis for the row space.
    """
    reduced, rank, _, _ = row_echelon(matrix)
    return reduced[:rank].copy()


def nullspace(matrix: np.ndarray) -> np.ndarray:
    """
    Compute the nullspace of a binary matrix over GF(2).

    Finds all vectors x such that matrix @ x = 0 (mod 2).

    Parameters
    ----------
    matrix : np.ndarray
        A binary matrix of shape (m, n).

    Returns
    -------
    np.ndarray
        A matrix of shape (k, n) whose rows form a basis for the nullspace,
        where k = n - rank(matrix).
    """
    mat = np.array(matrix, dtype=int) % 2
    m, n = mat.shape

    # Augment with identity on the right: [matrix | I_n] transposed approach.
    # We work with the transpose and find the left nullspace, then transpose back.
    # Equivalently: row-reduce [matrix; I_n]^T and extract kernel vectors.

    # Standard approach: row-reduce matrix, then read off nullspace from free columns.
    reduced, rank, _, pivots = row_echelon(mat)

    if rank == n:
        return np.zeros((0, n), dtype=int)

    # Identify free (non-pivot) columns.
    pivot_set = set(pivots)
    free_cols = [c for c in range(n) if c not in pivot_set]

    # Build nullspace vectors: for each free column, construct a vector.
    null_vectors = []
    for fc in free_cols:
        vec = np.zeros(n, dtype=int)
        vec[fc] = 1
        # For each pivot column, determine the value from the reduced matrix.
        for i, pc in enumerate(pivots):
            vec[pc] = reduced[i, fc]
        null_vectors.append(vec)

    if not null_vectors:
        return np.zeros((0, n), dtype=int)

    return np.array(null_vectors, dtype=int)


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
    return row_echelon(matrix)[0]


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
    return row_echelon(matrix)[3]


# =============================================================================
# Code Constructions
# =============================================================================

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

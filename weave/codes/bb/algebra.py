r"""Algebraic helpers for bivariate bicycle codes.

This module exposes the BB-specific machinery the compiler, tests,
and optimizer need on top of the generic :class:`CSSCode` machinery:

* :func:`ker_A_basis` / :func:`ker_BT_basis` — bases for the pure-L
  and pure-R Z-logical subspaces.
* :func:`pure_L_stabilizer_basis` — the stabilizer subgroup visible
  inside the L-block.
* :func:`enumerate_pure_L_minwt_logicals` — all minimum-weight
  representatives of the pure-L logical quotient, as sorted qubit
  supports.

Pure-L Z-logicals
-----------------
Given a BB code with `H_X = [A \mid B]` and `H_Z = [B^\top \mid A^\top]`,
a vector `v` in the L-block defines a pure-L operator `(v, 0)` that
commutes with every X-stabilizer iff `A v = 0`, i.e. `v \in \ker A`.
Every element of the stabilizer group has the form `(B c, A c)` for
some `c \in \mathbb{F}_2^{lm}`, obtained by summing rows of `H_Z`
with coefficient vector `c`. Requiring the R-component to vanish
forces `c \in \ker A`, and the L-component of such a stabilizer is
`B c`. The pure-L stabilizer subspace is therefore

.. math::

    S_L \;=\; B \!\!\cdot\! \ker(A) \;\subseteq\; \ker(A),

where the inclusion holds because `AB = BA` forces `A(B c) = B(A c) = 0`
for every `c \in \ker A`. The pure-L Z-logical quotient is
`\ker A / S_L`, and its minimum-weight coset leaders are the objects
the exposure optimizer and the PR 11 workbook fixture enumerate.

For the Bravyi et al. Table I BB codes, `\dim \ker A = k` and
`\dim S_L = k/2`, so the quotient has dimension `k/2` and enumerating
it costs `2^{k/2}` operations — e.g. 64 for BB72.

References
----------
- Bravyi et al. 2024, arXiv:2308.07915. §II and appendix A for the
  polynomial algebra.
- Di Bella, PRX Quantum under review 2026. §II.C for the pure-q(L)
  reference family used by the optimizer's `J_\kappa` objective.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

from ...util import pcm

if TYPE_CHECKING:
    from .bb_code import BivariateBicycleCode

__all__ = [
    "enumerate_pure_L_minwt_logicals",
    "ker_A_basis",
    "ker_BT_basis",
    "pure_L_stabilizer_basis",
]


# =============================================================================
# Subspace bases
# =============================================================================


def ker_A_basis(bb_code: BivariateBicycleCode) -> np.ndarray:
    r"""Basis for `\ker A`, the pure-L valid Z operators.

    A vector `v \in \mathbb{F}_2^{lm}` defines a pure-L operator
    `(v, 0)` that commutes with all X-stabilizers iff `A v = 0`.
    """
    return pcm.nullspace(bb_code.A_matrix)


def ker_BT_basis(bb_code: BivariateBicycleCode) -> np.ndarray:
    r"""Basis for `\ker B^\top`.

    Used by the symmetric R-block enumeration and by algebraic
    identities that characterize the BB code distance.
    """
    return pcm.nullspace(bb_code.B_matrix.T)


def pure_L_stabilizer_basis(bb_code: BivariateBicycleCode) -> np.ndarray:
    r"""Basis for the pure-L stabilizer subspace `S_L = B \cdot \ker(A)`.

    These are the pure-L vectors already stabilizer-equivalent to the
    identity. Because `A B = B A`, the action of `B` on `\ker A` maps
    into `\ker A`, so `S_L \subseteq \ker A` and its row span gives
    the trivial-logical subspace inside the pure-L quotient.
    """
    ker_A = pcm.nullspace(bb_code.A_matrix)
    if ker_A.shape[0] == 0:
        return np.zeros((0, bb_code.block_size), dtype=int)
    # For each basis vector c of ker(A), compute `B c` as a column
    # vector. Row-stacked form: image[i] = (B @ ker_A[i]^T).T = ker_A[i] @ B^T.
    image = (ker_A @ bb_code.B_matrix.T) % 2
    return pcm.row_basis(image)


# =============================================================================
# Minimum-weight pure-L logical enumeration
# =============================================================================


def enumerate_pure_L_minwt_logicals(
    bb_code: BivariateBicycleCode,
) -> tuple[tuple[int, ...], ...]:
    r"""Enumerate every minimum-weight pure-L Z-logical support.

    Walks the quotient `\ker A / S_L`, finds the minimum Hamming
    weight over all nontrivial cosets, and returns the sorted tuple
    of every coset representative that achieves that minimum. Each
    returned tuple is a sorted set of L-block qubit indices (so
    `0 \le q < bb_code.block_size`).

    The enumeration is exhaustive over `\ker A`, so it is cheap
    only when `\dim \ker A` is small (which is the case for every
    Bravyi et al. Table I code — the shared kernel has dimension 4-6).

    Parameters
    ----------
    bb_code : BivariateBicycleCode

    Returns
    -------
    tuple[tuple[int, ...], ...]
        Sorted supports (one per minimum-weight element). Every
        entry is a sorted tuple of integers in `[0, lm)`. The outer
        tuple itself is sorted in lexicographic order for determinism.

    Notes
    -----
    The plan's acceptance test expects this function to return
    exactly 36 weight-6 supports on BB72. If it does not, either the
    `(l, m)` / polynomial convention or the flat-index order differs
    from the plan's reference workbook. The module docstring pins
    the convention; the test file pins the count.
    """
    ker_A = ker_A_basis(bb_code)
    stab = pure_L_stabilizer_basis(bb_code)
    dim = ker_A.shape[0]
    lm = bb_code.block_size

    if dim == 0:
        return ()

    # First pass: find the minimum weight over nontrivial cosets.
    # Second pass: collect every coset rep that hits that minimum.
    # We walk all 2^dim elements of ker(A) once and classify.
    best_by_coset: dict[int, int] = {}
    reps_by_coset: dict[int, list[tuple[int, ...]]] = {}

    # Precompute the stabilizer row-echelon form for fast coset lookup.
    if stab.shape[0] > 0:
        stab_mat, _, _, stab_pivots = pcm.row_echelon(stab)
    else:
        stab_mat = np.zeros((0, lm), dtype=int)
        stab_pivots = []

    def _coset_key(v: np.ndarray) -> tuple[int, ...]:
        """Reduce `v` modulo the stabilizer row space and hash the result."""
        reduced = v.copy() % 2
        for i, pivot in enumerate(stab_pivots):
            if reduced[pivot]:
                reduced = (reduced + stab_mat[i]) % 2
        return tuple(int(x) for x in reduced)

    # Iterate all 2^dim elements of ker_A.
    for mask in range(2**dim):
        v = np.zeros(lm, dtype=int)
        for bit in range(dim):
            if mask & (1 << bit):
                v = (v + ker_A[bit]) % 2
        if not v.any():
            continue
        key = hash(_coset_key(v))
        weight = int(v.sum())
        support = tuple(int(q) for q in np.where(v)[0])
        if key not in best_by_coset or weight < best_by_coset[key]:
            best_by_coset[key] = weight
            reps_by_coset[key] = [support]
        elif weight == best_by_coset[key]:
            reps_by_coset[key].append(support)

    if not best_by_coset:
        return ()

    min_weight = min(best_by_coset.values())
    collected: set[tuple[int, ...]] = set()
    for key, weight in best_by_coset.items():
        if weight == min_weight:
            for rep in reps_by_coset[key]:
                collected.add(rep)

    return tuple(sorted(collected))


# =============================================================================
# Internal utility: bit-iteration over a subspace basis
# =============================================================================


def _iter_subspace(basis: np.ndarray) -> list[np.ndarray]:
    """Return every element of the subspace spanned by `basis`.

    Exponential in `basis.shape[0]`; intended for small bases only.
    Exposed as an internal helper for unit tests.
    """
    dim = basis.shape[0]
    lm = basis.shape[1]
    # The empty-basis subspace contains only the zero vector.
    if dim == 0:
        return [np.zeros(lm, dtype=int)]
    out: list[np.ndarray] = []
    for mask in range(2**dim):
        v = np.zeros(lm, dtype=int)
        for bit in range(dim):
            if mask & (1 << bit):
                v = (v + basis[bit]) % 2
        out.append(v)
    return out


# Exposed for downstream modules that need a curried enumeration.
# Suppress unused-import warnings on `combinations` — reserved for the
# PR 11 optimizer pass that walks weight-k supports directly.
_ = combinations

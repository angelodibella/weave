r"""Algebraic helpers for bivariate bicycle codes.

This module exposes the BB-specific machinery the compiler, tests,
and optimizer need on top of the generic :class:`CSSCode` machinery.
It enumerates minimum-weight *pure-L* logicals — logical
representatives supported entirely in the first data-qubit block —
in either the X sector or the Z sector, and computes the
subspace-dimension bookkeeping the optimizer's reference family
queries.

Pure-L logical quotients
------------------------
Given a BB code with `H_X = [A \mid B]` and `H_Z = [B^\top \mid A^\top]`,
a pure-L vector `u` (i.e. an operator `(u, 0)` supported in the
L-block only) defines:

- a **pure-L Z-logical** iff it commutes with every X-stabilizer row
  of `H_X = [A \mid B]`. Commutation against a row `(A_i, B_i)`
  gives `A_i \cdot u = 0`, i.e. `u \in \ker A`. The stabilizer
  subspace visible in the L-block is
  `S_L^{(Z)} = B \cdot \ker A`, so the quotient is
  `\ker A \,/\, (B \cdot \ker A)`.
- a **pure-L X-logical** iff it commutes with every Z-stabilizer row
  of `H_Z = [B^\top \mid A^\top]`. Commutation against a row
  `(B^\top_i, A^\top_i)` gives `B^\top_i \cdot u = 0`, i.e.
  `B^\top u = 0` (equivalently `u \in \ker B^\top`). The
  stabilizer subspace visible in the L-block is
  `T_L = \{\lambda A : \lambda \in \ker B\,\mathrm{(left)}\}`,
  i.e. the row space of `\ker(B^\top) \cdot A`. The quotient is
  `\ker B^\top \,/\, T_L`. This is the formulation used by the
  ``bbstim`` reference implementation (Di Bella 2026).

For `z_memory` experiments (where the decoder is correcting X
errors to preserve Z-observables), **the physically relevant
reference family is the pure-L X-logicals** — an X fault becomes a
logical error iff its support is equivalent to an X-logical modulo
stabilizers. The Z-logical enumeration is retained for symmetric
`x_memory` experiments and for backward compatibility with PR 10's
original tests.

The :func:`enumerate_pure_L_minwt_logicals` function takes a
`sector` argument (`"X"` or `"Z"`) and dispatches to the
appropriate quotient. For BB codes, the :mod:`bbstim` paper
convention uses `sector="X"`.

References
----------
- Bravyi et al. 2024, arXiv:2308.07915. §II and appendix A for the
  polynomial algebra.
- Di Bella, *Geometry-induced correlated noise in qLDPC syndrome
  extraction*, PRX Quantum under review 2026.
  `bbstim/algebra.py::enumerate_pure_L_minwt_logicals` (reference
  implementation used to pin the `sector="X"` faithfulness test).
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Literal

import numpy as np

from ...util import pcm

if TYPE_CHECKING:
    from .bb_code import BivariateBicycleCode


Sector = Literal["X", "Z"]


__all__ = [
    "Sector",
    "enumerate_pure_L_minwt_logicals",
    "ker_A_basis",
    "ker_BT_basis",
    "pure_L_stabilizer_basis",
    "pure_L_X_stabilizer_basis",
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
    r"""Z-sector pure-L stabilizer subspace `S_L^{(Z)} = B \cdot \ker A`.

    Pure-L vectors that are already Z-stabilizer-equivalent to the
    identity. Because `A B = B A`, the action of `B` on `\ker A` maps
    into `\ker A`, so `S_L^{(Z)} \subseteq \ker A` and its row span
    is the trivial-Z-logical subspace inside the pure-L quotient.
    """
    ker_A = pcm.nullspace(bb_code.A_matrix)
    if ker_A.shape[0] == 0:
        return np.zeros((0, bb_code.block_size), dtype=int)
    # For each basis vector c of ker(A), compute `B c` as a column
    # vector. Row-stacked form: image[i] = (B @ ker_A[i]^T).T = ker_A[i] @ B^T.
    image = (ker_A @ bb_code.B_matrix.T) % 2
    return pcm.row_basis(image)


def pure_L_X_stabilizer_basis(bb_code: BivariateBicycleCode) -> np.ndarray:
    r"""X-sector pure-L stabilizer subspace `T_L = \{\lambda A : \lambda \in \ker B\}`.

    Summing X-stabilizer rows `(A_i, B_i)` with a row vector
    `\lambda` gives `(\lambda A, \lambda B)`. For the R-component
    to vanish we need `\lambda B = 0`, i.e. `\lambda` in the left
    null space of `B`, which is the same as the ordinary null space
    of `B^\top`. The L-component is then `\lambda A`, and the row
    span over all such `\lambda` is the trivial-X-logical subspace
    inside `\ker B^\top`.

    This is the "`T_L`" subspace from the ``bbstim`` reference
    implementation.
    """
    left_null_B = pcm.nullspace(bb_code.B_matrix.T)  # rows are λ with λB = 0
    if left_null_B.shape[0] == 0:
        return np.zeros((0, bb_code.block_size), dtype=int)
    image = (left_null_B @ bb_code.A_matrix) % 2
    return pcm.row_basis(image)


# =============================================================================
# Minimum-weight pure-L logical enumeration
# =============================================================================


def enumerate_pure_L_minwt_logicals(
    bb_code: BivariateBicycleCode,
    *,
    sector: Sector = "Z",
) -> tuple[tuple[int, ...], ...]:
    r"""Enumerate every minimum-weight pure-L logical support.

    Walks the pure-L logical quotient for the chosen sector, finds
    the minimum Hamming weight over all nontrivial cosets, and
    returns the sorted tuple of every coset representative that
    achieves that minimum. Each returned tuple is a sorted set of
    L-block qubit indices (so `0 \le q < bb_code.block_size`).

    Sector dispatch
    ---------------
    - ``sector="Z"`` (default, backward compat with PR 10): enumerates
      pure-L Z-logicals via `\ker A \,/\, (B \cdot \ker A)`. Relevant
      for `x_memory` experiments.
    - ``sector="X"`` (matches ``bbstim``): enumerates pure-L
      X-logicals via `\ker B^\top \,/\, T_L` where
      `T_L = \{\lambda A : \lambda \in \ker B\,\text{(left)}\}`.
      Relevant for `z_memory` experiments; this is the
      reference family used by the Di Bella 2026 paper.

    The enumeration is exhaustive over the kernel, so it is cheap
    only when the kernel dimension is small (which is the case for
    every Bravyi et al. Table I code — the shared kernel has
    dimension 4–6).

    Parameters
    ----------
    bb_code : BivariateBicycleCode
    sector : {"X", "Z"}, optional
        Which pure-L logical sector to enumerate. Default ``"Z"``
        for backward compatibility; set to ``"X"`` for the
        ``bbstim``-faithful BB72 workbook family.

    Returns
    -------
    tuple[tuple[int, ...], ...]
        Sorted supports (one per minimum-weight element). Every
        entry is a sorted tuple of integers in `[0, lm)`. The outer
        tuple itself is sorted in lexicographic order for
        determinism.
    """
    if sector == "Z":
        kernel_basis = ker_A_basis(bb_code)
        stab = pure_L_stabilizer_basis(bb_code)
    elif sector == "X":
        kernel_basis = ker_BT_basis(bb_code)
        stab = pure_L_X_stabilizer_basis(bb_code)
    else:
        raise ValueError(f"sector must be 'X' or 'Z', got {sector!r}")

    dim = kernel_basis.shape[0]
    lm = bb_code.block_size

    if dim == 0:
        return ()

    # First pass: find the minimum weight over nontrivial cosets.
    # Second pass: collect every coset rep that hits that minimum.
    # We walk all 2^dim elements of the kernel once and classify.
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

    # Iterate all 2^dim elements of the kernel.
    for mask in range(2**dim):
        v = np.zeros(lm, dtype=int)
        for bit in range(dim):
            if mask & (1 << bit):
                v = (v + kernel_basis[bit]) % 2
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

r"""Residual-error enumeration and effective-distance diagnostics.

This module implements the residual-error formalism of

    A. Strikis, D. E. Browne, M. E. Beverland,
    *High-performance syndrome extraction circuits for quantum codes*,
    arXiv:2603.05481 (March 2026),

adapted to weave's :class:`~weave.ir.Schedule` IR. The key definitions
(verbatim from §IV.1 of that paper) are:

.. math::

    \Delta(E) \;:=\; 1 \;+\; \min_{D \in \mathbb{F}_2^n}
    \Bigl\{\, |D| \;:\; (E + D) \in \ker(H_X) \setminus \mathrm{span}(H_Z) \,\Bigr\},

i.e. the minimum number of additional *data-level* errors needed to
extend a residual error `E` to a nontrivial logical operator, plus one
for the original hook. The "residual distance" of a residual set `R`
is `min_{E ∈ R} Δ(E)`; their Theorem 1 states that for non-interleaved
SECs it coincides with the *circuit distance* `d_circ`.

The residual errors themselves are Pauli faults that survive
propagation through the remaining Cliffords of a syndrome extraction
cycle and the subsequent ancilla measurements. In the single-ancilla
ZZ-check case of Strikis et al., these are

.. math::

    \mathcal{E}_{Z, j}
    \;=\; \Bigl\{\, E_\ell := \sum_{s=\ell}^{w} e_{i_s} \,:\, \ell = 2, \dots, w \,\Bigr\},

one per "hook position" in the CNOT sequence. In the generalized
setting this module supports, we enumerate residuals by walking the
schedule via :func:`~weave.analysis.propagation.propagate_fault` and
collecting the data-level images of the relevant faults.

The module exposes:

* :class:`ResidualError` — a pure-data record of a residual (data-level
  Pauli support + which ancilla measurements it flipped).
* :func:`residual_distance` — the `Δ(E)` of a single residual against
  a code's parity-check matrices.
* :func:`effective_distance_upper_bound` — `min_{E ∈ R} Δ(E)` over a
  residual set, which is an upper bound on the circuit distance (and
  for non-interleaved SECs coincides with it).
* :func:`enumerate_hook_residuals_z_sector` — a helper that produces
  the `E_ℓ` set for a single ancilla-targeted CNOT sequence.

This PR ships the data-level enumeration and distance functions. The
schedule-walker-based "collect all retained residuals from a
schedule" function will land alongside PR 8 when the geometry pass
needs it.

References
----------
- A. Strikis, D. E. Browne, M. E. Beverland, *High-performance
  syndrome extraction circuits for quantum codes*, arXiv:2603.05481
  (2026). §IV.1, Definitions 1 and 3; §IV.2, Theorem 1.
- Di Bella, *Geometry-induced correlated noise in qLDPC syndrome
  extraction* (PRX Quantum, under review, 2026). §II.D for the
  retained single-and-pair channel from which the residual
  enumeration derives.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from ..util import pcm
from .pauli import Pauli

__all__ = [
    "ResidualError",
    "effective_distance_upper_bound",
    "enumerate_hook_residuals_z_sector",
    "residual_distance",
]


# =============================================================================
# Residual record
# =============================================================================


@dataclass(frozen=True)
class ResidualError:
    r"""A residual fault surviving one cycle of syndrome extraction.

    Parameters
    ----------
    data_support : tuple[int, ...]
        Sorted data qubit indices where the fault acts (the support
        of the data-level Pauli after ancilla elimination). For a
        single-sector `ZZ`-check residual this is the set
        `{i_ℓ, i_{ℓ+1}, ..., i_w}` from Strikis Definition 1.
    weight : int
        Hamming weight of `data_support`, i.e. `len(data_support)`.
        Stored separately for convenience.
    flipped_ancilla_ticks : tuple[tuple[int, int], ...]
        `(tick_index, qubit)` pairs for the ancilla measurements this
        residual flipped during propagation. Empty tuple for residuals
        that do not trigger any syndrome bit — those are *undetectable*
        at the level of the schedule alone and are the ones that
        determine the effective distance.
    label : str
        Optional human-readable label used for debugging and test
        assertions (e.g. ``"HZ[0]_hook_3"``).
    """

    data_support: tuple[int, ...]
    weight: int
    flipped_ancilla_ticks: tuple[tuple[int, int], ...] = ()
    label: str = ""

    def __post_init__(self) -> None:
        if tuple(sorted(self.data_support)) != tuple(self.data_support):
            object.__setattr__(self, "data_support", tuple(sorted(self.data_support)))
        if self.weight != len(self.data_support):
            raise ValueError(
                f"weight ({self.weight}) does not match len(data_support) "
                f"({len(self.data_support)}) for residual {self.label!r}."
            )

    @classmethod
    def from_pauli(
        cls,
        pauli: Pauli,
        *,
        data_qubits: frozenset[int],
        flipped_ancilla_ticks: tuple[tuple[int, int], ...] = (),
        label: str = "",
    ) -> ResidualError:
        """Build a residual from a phase-free Pauli on the full schedule.

        Takes the intersection of `pauli.support` with `data_qubits`,
        which is the data-level image of the residual.
        """
        data_support = tuple(sorted(pauli.support & data_qubits))
        return cls(
            data_support=data_support,
            weight=len(data_support),
            flipped_ancilla_ticks=flipped_ancilla_ticks,
            label=label,
        )

    def as_binary_vector(self, num_data_qubits: int) -> np.ndarray:
        """Return the support as a length-`num_data_qubits` binary vector."""
        vec = np.zeros(num_data_qubits, dtype=np.uint8)
        for i in self.data_support:
            vec[i] = 1
        return vec


# =============================================================================
# Residual distance (Strikis Definition 1)
# =============================================================================


def residual_distance(
    residual: ResidualError,
    h_commute: np.ndarray,
    h_stab: np.ndarray,
    *,
    k_guard: int = 20,
) -> int | float:
    r"""Compute `Δ(E)` for a residual `E` (Strikis Definition 1).

    `Δ(E)` is the minimum Hamming weight of a data-level completion
    `D` such that `(E + D)` is a *nontrivial* logical operator —
    plus one for the residual itself:

    .. math::

        \Delta(E) \;=\; 1 + \min_{D}\Bigl\{\, |D| \;:\;
            (E + D) \in \ker(H_{\mathrm{commute}}) \setminus
            \mathrm{span}(H_{\mathrm{stab}}) \,\Bigr\}.

    For the `X`-error sector we take `h_commute = HZ` (commutation
    with Z-stabilizers) and `h_stab = HX` (stabilizer triviality).
    For the `Z`-error sector the roles swap.

    The `+1` accounts for the original hook error that produced the
    residual: even if the residual is itself already a logical (`D = 0`),
    the total fault weight at the circuit level is one (the hook)
    plus the data-level image.

    Implementation is exhaustive and matches the brute-force distance
    computation used by :meth:`weave.codes.css_code.CSSCode.distance`:
    enumerate non-zero elements of `ker(h_commute)`, add `E` modulo 2,
    and take the minimum weight over elements that are not in
    `span(h_stab)`. Raises `ValueError` if the nullspace dimension
    exceeds `k_guard` to avoid accidental combinatorial explosions.

    Parameters
    ----------
    residual : ResidualError
        The residual `E`.
    h_commute : np.ndarray
        Parity-check matrix that the completed fault must commute
        with (`HZ` for X-sector residuals, `HX` for Z-sector).
    h_stab : np.ndarray
        Stabilizer parity-check matrix whose row space we quotient by.
    k_guard : int, optional
        Cap on `dim ker(h_commute)` to keep the enumeration tractable.
        Default is 20 (matches :mod:`weave.util.pcm`).

    Returns
    -------
    int or float
        The value `Δ(E)`. Returns `float('inf')` if no nontrivial
        completion exists (e.g. if the code has no logicals in this
        sector).

    Raises
    ------
    ValueError
        If the ambient data-qubit count is inconsistent, or if
        `dim ker(h_commute) > k_guard`.
    """
    n_data = h_commute.shape[1]
    if h_stab.shape[1] != n_data:
        raise ValueError(f"h_commute has {n_data} columns but h_stab has {h_stab.shape[1]}.")

    e_vec = residual.as_binary_vector(n_data)

    ker = pcm.nullspace(h_commute)
    stab_basis = pcm.row_basis(h_stab)
    dim_ker = ker.shape[0]

    if dim_ker == 0:
        return float("inf")
    if dim_ker > k_guard:
        raise ValueError(
            f"residual_distance enumeration infeasible for "
            f"{dim_ker}-dimensional nullspace (> k_guard={k_guard}). "
            f"Would require enumerating {2**dim_ker - 1} candidates."
        )

    stab_rank = stab_basis.shape[0]

    # We want min_{u in ker \ span(h_stab)} |u + E|, then add 1.
    # Enumerate every nonzero u in ker (all subsets of basis rows),
    # reject u ∈ span(h_stab), and track the minimum weight of u XOR E.
    best: int | float = float("inf")
    for r in range(1, dim_ker + 1):
        for combo in combinations(range(dim_ker), r):
            u = np.zeros(n_data, dtype=np.uint8)
            for idx in combo:
                u = (u + ker[idx]) % 2
            # Reject trivial (stabilizer) u.
            if stab_rank > 0:
                aug = np.vstack([stab_basis, u.reshape(1, -1)])
                if pcm.row_echelon(aug)[1] == stab_rank:
                    continue  # u ∈ span(h_stab), skip.
            # Compute |u + E|.
            total = (u + e_vec) % 2
            weight = int(total.sum())
            if weight < best:
                best = weight

    # Include the residual itself as the E = u case (u = 0) only if
    # the residual is already a nontrivial logical on its own.
    if np.any(e_vec):
        if stab_rank == 0:
            # No stabilizers — the residual itself (u = 0, D = E) is
            # a candidate; its weight is |E|.
            if residual.weight < best:
                best = residual.weight
        else:
            # E is a nontrivial logical iff augmenting the stabilizer
            # basis with E increases the rank.
            aug = np.vstack([stab_basis, e_vec.reshape(1, -1)])
            if pcm.row_echelon(aug)[1] > stab_rank and residual.weight < best:
                best = residual.weight

    if best == float("inf"):
        return float("inf")
    return int(best) + 1


# =============================================================================
# Effective-distance upper bound
# =============================================================================


def effective_distance_upper_bound(
    residuals: list[ResidualError],
    h_commute: np.ndarray,
    h_stab: np.ndarray,
    *,
    k_guard: int = 20,
) -> int | float:
    r"""Upper bound on the circuit distance from a residual set.

    Computes

    .. math::

        \min_{E \in \mathcal{R}} \Delta(E)

    as defined in :func:`residual_distance`. By Strikis et al.
    Theorem 1 this is `d_circ` for non-interleaved SECs, and an upper
    bound in general.

    Parameters
    ----------
    residuals : list[ResidualError]
        The residual set `R`.
    h_commute, h_stab : np.ndarray
        Same as :func:`residual_distance`.
    k_guard : int, optional
        Nullspace-dimension cap passed through to
        :func:`residual_distance`.

    Returns
    -------
    int or float
        The minimum `Δ(E)` over `residuals`, or `float('inf')` if
        `residuals` is empty or every residual has
        `Δ(E) = ∞`.
    """
    if not residuals:
        return float("inf")
    best: int | float = float("inf")
    for r in residuals:
        delta = residual_distance(r, h_commute, h_stab, k_guard=k_guard)
        if delta < best:
            best = delta
    return best


# =============================================================================
# Hook-residual enumeration for a single ZZ-check
# =============================================================================


def enumerate_hook_residuals_z_sector(
    ancilla_cnot_targets: list[int],
    *,
    label_prefix: str = "hook",
) -> list[ResidualError]:
    r"""Enumerate `E_ℓ` hook residuals for a single sequential `ZZ`-check.

    Direct implementation of Strikis Definition 1 for the `Z`-check
    case. Given a sequence of data-qubit CNOT targets
    `[i_1, i_2, ..., i_w]` hit by a single ancilla in order, a hook
    `X`-error on the ancilla AFTER the `(ℓ-1)`-th CNOT propagates
    forward through every subsequent CNOT, leaving a data-level
    `Z`-error on `{i_ℓ, i_{ℓ+1}, ..., i_w}`. The residual set is

    .. math::

        \mathcal{E}_{Z} = \{\, E_\ell := \{i_\ell, \dots, i_w\} \,:\, \ell = 2, \dots, w \,\}.

    Note that `ℓ = 1` is excluded: a hook before the first CNOT
    propagates to all `w` data qubits, but that corresponds to an
    error *before* any gate in the check, which is indistinguishable
    from a data-qubit error that the cycle was about to measure
    anyway. Strikis et al. start their enumeration at `ℓ = 2`.

    Parameters
    ----------
    ancilla_cnot_targets : list[int]
        The data qubit indices hit by successive CNOTs, in CNOT
        order.
    label_prefix : str, optional
        Prefix for the residuals' `label` field.

    Returns
    -------
    list[ResidualError]
        The hook residuals, one per `ℓ` from 2 to `w`.
    """
    w = len(ancilla_cnot_targets)
    residuals: list[ResidualError] = []
    for ell in range(2, w + 1):
        support = tuple(sorted(ancilla_cnot_targets[ell - 1 :]))
        residuals.append(
            ResidualError(
                data_support=support,
                weight=len(support),
                label=f"{label_prefix}_{ell}",
            )
        )
    return residuals

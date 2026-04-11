r"""Swap-descent optimizer for exposure-based embedding objectives.

This module implements a randomized first-improvement swap descent
over qubit positions. The canonical use case is minimizing
:math:`J_\kappa` on a
:class:`~weave.ir.MonomialColumnEmbedding` of a bivariate bicycle
code, starting from the canonical monomial layout and searching
within qubit-class-preserving swaps (L-data ↔ L-data, Z-anc ↔ Z-anc,
etc.). The optimizer is generic, though: it only needs a
`position_array_from(embedding)` map and a scalar objective
callable.

Algorithm
---------
At each iteration:

1. Draw `sample_size` random unordered pairs `(q_a, q_b)` from the
   configured swap classes.
2. For each pair, swap the two positions in the `NumPy`-backed
   position array and evaluate the scalar objective.
3. Apply the best single-swap improvement among the sample (if any
   improvement exists). If no sample swap improves the objective,
   terminate.
4. Repeat up to `max_iterations` iterations.

This is *random-best-improvement* descent — deterministic given a
seed, simple to reason about, and parallel-friendly (the objective
calls within a single iteration are independent and could be
farmed out to threads/processes in a later PR).

The optimizer tracks the history of accepted objective values so
tests and benchmarks can assert monotone decrease and compute
relative reduction ratios.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..ir import ColumnEmbedding

__all__ = [
    "SwapDescentResult",
    "swap_descent",
]


# =============================================================================
# Result type
# =============================================================================


@dataclass(frozen=True)
class SwapDescentResult:
    """Summary of a swap-descent optimization run.

    Parameters
    ----------
    initial_value : float
        `J_\\kappa` (or whichever objective) at the starting
        embedding.
    final_value : float
        Objective value at the returned (optimized) embedding.
    optimized_positions : np.ndarray
        Shape `(n_qubits, 3)` positions array at the final state.
        The caller typically converts this back to an
        :class:`Embedding` via
        :func:`apply_positions_to_column_embedding`.
    history : tuple[float, ...]
        Sequence of accepted objective values, one per improving
        iteration. Always monotonically non-increasing and always
        starts with `initial_value`.
    accepted_swaps : tuple[tuple[int, int], ...]
        The sequence of accepted `(q_a, q_b)` swaps, parallel to
        `history` (same length — 1 entry fewer).
    n_evaluations : int
        Total number of objective calls made during the run.
    stopped_early : bool
        True iff the descent stopped because a whole iteration
        found no improvement (a local minimum). False if the
        iteration cap was reached instead.
    """

    initial_value: float
    final_value: float
    optimized_positions: np.ndarray
    history: tuple[float, ...] = field(default_factory=tuple)
    accepted_swaps: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    n_evaluations: int = 0
    stopped_early: bool = False

    @property
    def reduction_ratio(self) -> float:
        """`1 - final/initial`, clipped to `[0, 1]`. 0 for no-op runs."""
        if self.initial_value <= 0:
            return 0.0
        return max(0.0, 1.0 - self.final_value / self.initial_value)


# =============================================================================
# Core descent loop
# =============================================================================


def swap_descent(
    initial_positions: np.ndarray,
    objective: Callable[[np.ndarray], float],
    swap_classes: Sequence[Sequence[int]],
    *,
    max_iterations: int = 50,
    sample_size: int = 40,
    rng: np.random.Generator | None = None,
    tol: float = 1e-12,
) -> SwapDescentResult:
    r"""Run randomized first-improvement swap descent.

    Parameters
    ----------
    initial_positions : np.ndarray
        Shape `(n_qubits, 3)` starting positions. A copy is made
        so the caller's array is left untouched.
    objective : Callable[[np.ndarray], float]
        Scalar objective to minimize. Called with a `(n_qubits, 3)`
        positions array; typically a closure over a
        :class:`~weave.optimize.NumpyExposureTemplate` plus a
        :class:`~weave.ir.Kernel` (see
        :func:`~weave.optimize.objectives.j_kappa_numpy`).
    swap_classes : Sequence[Sequence[int]]
        One sequence of qubit indices per class (e.g. L-data,
        R-data, Z-ancilla, X-ancilla). Swaps are only proposed
        within a class — different classes never exchange positions.
        Singletons and empty classes are silently skipped.
    max_iterations : int, optional
        Maximum outer-loop iterations. Default 50. Each iteration
        samples `sample_size` candidate swaps, so the total
        objective budget is at most `max_iterations * sample_size`.
    sample_size : int, optional
        Number of random swap candidates per iteration. Default 40.
        Larger values find better minima at higher cost.
    rng : numpy.random.Generator, optional
        Random source for swap sampling. Defaults to a fresh
        default RNG — pass a seeded generator for reproducibility.
    tol : float, optional
        Minimum objective improvement to accept a swap. Default
        1e-12.

    Returns
    -------
    SwapDescentResult
        The optimized positions plus history and diagnostics.
    """
    if rng is None:
        rng = np.random.default_rng()

    positions = initial_positions.copy()
    current_value = float(objective(positions))
    initial_value = current_value
    history: list[float] = [current_value]
    accepted_swaps: list[tuple[int, int]] = []

    # Pre-materialize swap-class index arrays and the size-2 pair lists.
    class_arrays: list[np.ndarray] = []
    class_sizes: list[int] = []
    for cls in swap_classes:
        arr = np.asarray(list(cls), dtype=np.int64)
        if arr.size >= 2:
            class_arrays.append(arr)
            class_sizes.append(arr.size)
    if not class_arrays:
        return SwapDescentResult(
            initial_value=initial_value,
            final_value=current_value,
            optimized_positions=positions,
            history=tuple(history),
            accepted_swaps=(),
            n_evaluations=1,
            stopped_early=True,
        )

    total_class_weight = np.asarray(class_sizes, dtype=float)
    total_class_weight /= total_class_weight.sum()

    n_evaluations = 1
    stopped_early = False

    for _ in range(max_iterations):
        # Sample candidate swaps: draw class by size-weighted choice,
        # then two distinct indices within that class.
        best_delta = 0.0
        best_swap: tuple[int, int] | None = None
        best_value = current_value
        for _ in range(sample_size):
            cls_idx = int(rng.choice(len(class_arrays), p=total_class_weight))
            cls = class_arrays[cls_idx]
            i, j = rng.choice(cls, size=2, replace=False)
            i = int(i)
            j = int(j)
            # Apply the swap in place, evaluate, then revert.
            positions[[i, j]] = positions[[j, i]]
            trial_value = float(objective(positions))
            positions[[i, j]] = positions[[j, i]]
            n_evaluations += 1
            delta = trial_value - current_value
            if delta < best_delta - tol:
                best_delta = delta
                best_swap = (i, j)
                best_value = trial_value

        if best_swap is None:
            stopped_early = True
            break
        # Apply the best improving swap and continue.
        i, j = best_swap
        positions[[i, j]] = positions[[j, i]]
        current_value = best_value
        history.append(current_value)
        accepted_swaps.append(best_swap)

    return SwapDescentResult(
        initial_value=initial_value,
        final_value=current_value,
        optimized_positions=positions,
        history=tuple(history),
        accepted_swaps=tuple(accepted_swaps),
        n_evaluations=n_evaluations,
        stopped_early=stopped_early,
    )


# =============================================================================
# Convenience helper: apply an optimized positions array back to an embedding
# =============================================================================


def apply_positions_to_column_embedding(
    embedding: ColumnEmbedding, positions_array: np.ndarray
) -> ColumnEmbedding:
    """Return a new :class:`ColumnEmbedding` with updated positions.

    Use after :func:`swap_descent` to turn the optimized NumPy
    positions array back into a frozen, JSON-round-trippable
    embedding whose layout matches the optimizer's output.

    Parameters
    ----------
    embedding : ColumnEmbedding
        The embedding whose class, metadata, and `name` are
        preserved. Only `positions` is replaced.
    positions_array : np.ndarray
        Shape `(n_qubits, 3)`. Length must match
        `len(embedding.positions)`.
    """
    from dataclasses import replace

    if positions_array.shape != (len(embedding.positions), 3):
        raise ValueError(
            f"positions_array shape {positions_array.shape} does not match "
            f"embedding.positions length {len(embedding.positions)}"
        )
    new_positions = tuple((float(p[0]), float(p[1]), float(p[2])) for p in positions_array)
    return replace(embedding, positions=new_positions)

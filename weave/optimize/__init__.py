r"""Embedding optimizer for the `J_\kappa` exposure objective.

The `weave.optimize` package turns a
:class:`~weave.ir.Embedding` — typically the canonical
:class:`~weave.ir.MonomialColumnEmbedding` of a BB code — into an
embedding that minimizes the per-reference-support exposure
:math:`J_\kappa` of the geometry-induced retained channel. The
entry points are:

- :func:`swap_descent` — randomized first-improvement swap descent
  over a `(n_qubits, 3)` positions array, constrained to swaps
  within user-specified qubit classes (so e.g. Z-ancillas never
  exchange positions with data qubits).
- :func:`apply_positions_to_column_embedding` — turns the NumPy
  output of :func:`swap_descent` back into a frozen, JSON-round-
  trippable :class:`~weave.ir.ColumnEmbedding`.
- :class:`NumpyExposureTemplate` — a vectorized view of an
  :class:`ExposureTemplate` that the optimizer's inner loop uses
  to evaluate :math:`J_\kappa` in ~1 ms on BB72.
- :func:`j_kappa` / :func:`j_kappa_numpy` / :func:`j_cross` — the
  scalar objective functions.
- :func:`compute_bb_ibm_event_template` — the BB-specific
  analytical-shortcut template builder (see the module docstring
  of :mod:`weave.optimize.objectives` for the derivation).
- :func:`prepare_exposure_template` — filter a raw template by a
  reference family and precompute the event→support map.

See the module docstring of :mod:`weave.optimize.objectives` for
the mathematical derivation of :math:`J_\kappa` and the
analytical template shortcut, and the module docstring of
:mod:`weave.optimize.swap_descent` for the descent algorithm.
"""

from __future__ import annotations

from .objectives import (
    ExposureTemplate,
    NumpyExposureTemplate,
    PairEventTemplate,
    compute_bb_ibm_event_template,
    compute_event_template_generic,
    j_cross,
    j_kappa,
    j_kappa_numpy,
    prepare_exposure_template,
)
from .swap_descent import (
    SwapDescentResult,
    apply_positions_to_column_embedding,
    swap_descent,
)

__all__ = [
    "ExposureTemplate",
    "NumpyExposureTemplate",
    "PairEventTemplate",
    "SwapDescentResult",
    "apply_positions_to_column_embedding",
    "compute_bb_ibm_event_template",
    "compute_event_template_generic",
    "j_cross",
    "j_kappa",
    "j_kappa_numpy",
    "prepare_exposure_template",
    "swap_descent",
]

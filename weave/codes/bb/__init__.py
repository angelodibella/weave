"""Bivariate bicycle (BB) quantum LDPC codes.

This package implements the BB code family of Bravyi, Cross, Gambetta,
Maslov, Rall, and Yoder, *High-threshold and low-overhead fault-tolerant
quantum memory*, Nature **627**, 778 (2024), arXiv:2308.07915. BB codes
are CSS codes built from two commuting polynomials `A(x, y)` and
`B(x, y)` in the group algebra `F_2[Z_l × Z_m]`, with parity-check
matrices

.. math::

    H_X = [\\,A \\mid B\\,], \\qquad H_Z = [\\,B^\\top \\mid A^\\top\\,].

The commutation `A B = B A` (which holds automatically for polynomials
in commuting variables) is equivalent to `H_X H_Z^\\top = 0` over
`F_2`, so every pair `(A, B)` of monomial sums yields a valid CSS
code.

Public API
----------
- :class:`~weave.codes.bb.BivariateBicycleCode` — the code class.
- :func:`~weave.codes.bb.build_bb72` — [[72, 12, 6]] (Bravyi Table I, row 1).
- :func:`~weave.codes.bb.build_bb90` — [[90, 8, 10]] (row 2).
- :func:`~weave.codes.bb.build_bb108` — [[108, 8, 10]] (row 3).
- :func:`~weave.codes.bb.build_bb144` — [[144, 12, 12]] (row 4).
- :func:`~weave.codes.bb.algebra.enumerate_pure_L_minwt_logicals` —
  enumerate the minimum-weight pure-L quotient logical representatives.

See also
--------
- Bravyi, Cross, Gambetta, Maslov, Rall, Yoder, 2024,
  arXiv:2308.07915. Table I lists the parameters used below.
- Di Bella, *Geometry-induced correlated noise in qLDPC syndrome
  extraction* (PRX Quantum, under review, 2026). §II.C defines the
  pure-L quotient family used by the exposure-optimizer objective.
"""

from __future__ import annotations

from .algebra import Sector, enumerate_pure_L_minwt_logicals
from .bb_code import (
    BivariateBicycleCode,
    Monomial,
    build_bb72,
    build_bb90,
    build_bb108,
    build_bb144,
)
from .schedule import ibm_schedule

__all__ = [
    "BivariateBicycleCode",
    "Monomial",
    "Sector",
    "build_bb72",
    "build_bb90",
    "build_bb108",
    "build_bb144",
    "enumerate_pure_L_minwt_logicals",
    "ibm_schedule",
]

"""Retained-channel pair probabilities derived from microscopic kernels.

These helpers compose a proximity kernel with the physical tick duration
`τ` and coupling scale `J₀` to produce the pair-fault probabilities
consumed by the geometry-aware compiler. The exact form comes from the
same-tick Pauli twirl applied to a single-pair block Hamiltonian
`J₀·κ(d)·P̂_e⊗P̂_{e'}`:

.. math::

    p(d) \\;=\\; \\sin^{2}\\bigl(\\tau J_{0} \\kappa(d)\\bigr).

The weak-coupling limit `τ J₀ κ(d) ≪ 1` replaces the sine with its
argument, giving the quadratic approximation `(τ J₀ κ(d))²`.

See :mod:`weave.geometry.kernels` for the kernel protocol and concrete
implementations.
"""

from __future__ import annotations

import math

from ..ir.kernel import Kernel


def pair_amplitude(d: float, J0: float, kernel: Kernel) -> float:
    """Microscopic pair-coupling amplitude `J₀·κ(d)` at separation `d`.

    This is the coefficient of the single-pair block Hamiltonian before
    the twirl; it has units of inverse time when `J₀` does.
    """
    return J0 * kernel(d)


def pair_location_strength(d: float, tau: float, J0: float, kernel: Kernel) -> float:
    """Dimensionless location strength `τ·J₀·κ(d)`.

    This is the argument of the sine in :func:`exact_twirled_pair_probability`
    and appears directly in the AKP-style smallness criterion.
    """
    return tau * J0 * kernel(d)


def exact_twirled_pair_probability(d: float, tau: float, J0: float, kernel: Kernel) -> float:
    """Retained pair-fault probability from the same-tick Pauli twirl.

    For a single simultaneously active pair at routed separation `d`, the
    Pauli-twirled block channel applies the correlated operator
    `P̂_e⊗P̂_{e'}` with probability

    .. math::

        p(d) = \\sin^{2}\\bigl(\\tau J_{0} \\kappa(d)\\bigr).

    This is exact at leading order in the same-tick factorization;
    multi-pair interference corrections enter at `O(Θ⁴)` where
    `Θ² = Σ_a (τ J₀ κ_a)²`.

    Parameters
    ----------
    d : float
        Routed separation between the two gate blocks.
    tau : float
        Tick duration (positive).
    J0 : float
        Microscopic coupling scale (nonnegative).
    kernel : Kernel
        Proximity kernel.

    Returns
    -------
    float
        The twirled pair-fault probability, in `[0, 1]`.
    """
    return math.sin(tau * J0 * kernel(d)) ** 2


def weak_pair_probability(d: float, tau: float, J0: float, kernel: Kernel) -> float:
    """Weak-coupling approximation to the twirled pair probability.

    In the regime `τ J₀ κ(d) ≪ 1`, `sin²(x) ≈ x²`, so

    .. math::

        p_{\\text{weak}}(d) = \\bigl(\\tau J_{0} \\kappa(d)\\bigr)^{2}.

    Faster to compute than the exact form and often sufficient for
    analytical bounds; agrees with :func:`exact_twirled_pair_probability`
    to fourth order in the location strength.

    Parameters
    ----------
    d : float
        Routed separation between the two gate blocks.
    tau : float
        Tick duration (positive).
    J0 : float
        Microscopic coupling scale (nonnegative).
    kernel : Kernel
        Proximity kernel.

    Returns
    -------
    float
        The weak-limit pair-fault probability.
    """
    x = tau * J0 * kernel(d)
    return x * x

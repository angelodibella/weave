"""Proximity kernels and the `Kernel` protocol.

A *proximity kernel* is a dimensionless, nonnegative, monotonically
nonincreasing function `κ(d) ∈ [0, 1]` with `κ(0) = 1` that maps a routed
separation `d` to a coupling strength. Kernels compose with physical
parameters `(τ, J₀)` in :mod:`weave.geometry.pair` to produce the
retained-channel pair probabilities of the geometry-aware compiler.

Three concrete kernels are shipped:

* :class:`CrossingKernel` — combinatorial diagnostic (1 at `d=0`, 0 elsewhere).
* :class:`RegularizedPowerLawKernel` — `(1 + d/r₀)^(-α)`.
* :class:`ExponentialKernel` — `exp(-d/ξ)`.

The :class:`Kernel` protocol is intentionally minimal so that users can
plug in any callable with a `name`, a `params` dict, and a `__call__`
method. This also makes kernels JSON-serializable without ever committing
to a concrete base class.

.. note::
    In PR 3 of the implementation plan the :class:`Kernel` protocol and
    its concrete subclasses move to ``weave/ir/kernel.py`` (the IR home).
    Until then, they live here for colocation with the geometry
    primitives they interact with.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class Kernel(Protocol):
    """Structural type for a proximity kernel.

    Any object with `name`, `params`, and a `__call__(d) -> float` satisfies
    this protocol. Concrete implementations in weave are frozen dataclasses,
    but user code can pass arbitrary objects (including plain functions
    wrapped in a thin adapter) that match this shape.
    """

    @property
    def name(self) -> str:
        """Stable identifier for serialization (e.g. ``"crossing"``)."""
        ...

    @property
    def params(self) -> dict[str, float]:
        """Parameter dictionary. Empty for parameterless kernels."""
        ...

    def __call__(self, d: float) -> float:
        """Evaluate κ at separation `d`."""
        ...


@dataclass(frozen=True)
class CrossingKernel:
    """Combinatorial crossing indicator: 1 at `d = 0`, 0 elsewhere.

    This is the kernel used for crossing-kernel diagnostics in the paper;
    it is not a physical crosstalk law but lets theorems about matching
    number and support-level effective distance become direct
    combinatorial statements.
    """

    @property
    def name(self) -> str:
        return "crossing"

    @property
    def params(self) -> dict[str, float]:
        return {}

    def __call__(self, d: float) -> float:
        return 1.0 if abs(d) < 1e-12 else 0.0


@dataclass(frozen=True)
class RegularizedPowerLawKernel:
    """Regularized algebraic kernel `κ(d) = (1 + d/r₀)^(-α)`.

    The main distance-decay kernel studied in the paper. Both `α` and `r₀`
    must be strictly positive. The decay exponent `α` governs two-dimensional
    summability: `α > 2` is the planar threshold for aggregate pair coupling
    to remain finite (AKP-inspired compatibility).

    Parameters
    ----------
    alpha : float
        Decay exponent; must be strictly positive.
    r0 : float
        Regularization length; must be strictly positive.
    """

    alpha: float
    r0: float

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}.")
        if self.r0 <= 0:
            raise ValueError(f"r0 must be positive, got {self.r0}.")

    @property
    def name(self) -> str:
        return "regularized_power_law"

    @property
    def params(self) -> dict[str, float]:
        return {"alpha": self.alpha, "r0": self.r0}

    def __call__(self, d: float) -> float:
        return float((1.0 + d / self.r0) ** (-self.alpha))


@dataclass(frozen=True)
class ExponentialKernel:
    """Exponential kernel `κ(d) = exp(-d/ξ)`.

    Phenomenological short-range crosstalk law with characteristic decay
    length `ξ > 0`. Strictly positive for every finite `d`, so it saturates
    support graphs in the sense of Proposition 3 of the paper.

    Parameters
    ----------
    xi : float
        Decay length; must be strictly positive.
    """

    xi: float

    def __post_init__(self) -> None:
        if self.xi <= 0:
            raise ValueError(f"xi must be positive, got {self.xi}.")

    @property
    def name(self) -> str:
        return "exponential"

    @property
    def params(self) -> dict[str, float]:
        return {"xi": self.xi}

    def __call__(self, d: float) -> float:
        return math.exp(-d / self.xi)

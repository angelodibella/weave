"""Proximity kernels (IR): `Kernel` protocol and concrete implementations.

A *proximity kernel* is a dimensionless, nonnegative, monotonically
nonincreasing function `κ(d) ∈ [0, 1]` with `κ(0) = 1` that maps a routed
separation `d` to a coupling strength. Kernels compose with physical
parameters `(τ, J₀)` (see :mod:`weave.geometry.pair`) to produce the
retained-channel pair probabilities consumed by the compiler.

This module is the IR home for kernels. It defines:

* :class:`Kernel` — a runtime-checkable structural protocol with
  `name`, `params`, `__call__(d)`, and `to_json()`.
* :class:`CrossingKernel`, :class:`RegularizedPowerLawKernel`,
  :class:`ExponentialKernel` — frozen-dataclass concrete kernels with
  full JSON round-trip and schema versioning.
* :func:`load_kernel` — a dispatch helper that reconstructs any known
  kernel type from its JSON dict.

The module :mod:`weave.geometry.kernels` is now a thin re-export shim;
user code can continue to import ``CrossingKernel`` from either location.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable


@runtime_checkable
class Kernel(Protocol):
    """Structural type for a proximity kernel.

    Any object with `name`, `params`, `__call__(d) -> float`, and
    `to_json() -> dict` satisfies this protocol. Concrete implementations
    shipped with weave are frozen dataclasses, but user code can pass
    arbitrary objects matching this shape.

    Note
    ----
    `from_json` is intentionally *not* part of the protocol: it is a
    classmethod, which protocols cannot express cleanly. Use
    :func:`load_kernel` for polymorphic deserialization.
    """

    @property
    def name(self) -> str:
        """Stable type identifier used for JSON dispatch."""
        ...

    @property
    def params(self) -> dict[str, float]:
        """Parameter dictionary. Empty for parameterless kernels."""
        ...

    def __call__(self, d: float) -> float:
        """Evaluate κ at separation `d`."""
        ...

    def to_json(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        ...


@dataclass(frozen=True)
class CrossingKernel:
    """Combinatorial crossing indicator: 1 at `d = 0`, 0 elsewhere.

    This is the kernel used for crossing-kernel diagnostics in the paper;
    it is not a physical crosstalk law but lets theorems about matching
    number and support-level effective distance become direct
    combinatorial statements.
    """

    SCHEMA_VERSION: ClassVar[int] = 1

    @property
    def name(self) -> str:
        return "crossing"

    @property
    def params(self) -> dict[str, float]:
        return {}

    def __call__(self, d: float) -> float:
        return 1.0 if abs(d) < 1e-12 else 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "type": self.name,
            "params": self.params,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> CrossingKernel:
        _validate_json(data, expected_type="crossing", schema_version=cls.SCHEMA_VERSION)
        return cls()


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

    SCHEMA_VERSION: ClassVar[int] = 1

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

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "type": self.name,
            "params": self.params,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> RegularizedPowerLawKernel:
        _validate_json(
            data, expected_type="regularized_power_law", schema_version=cls.SCHEMA_VERSION
        )
        params = data.get("params", {})
        if "alpha" not in params or "r0" not in params:
            raise ValueError(
                f"regularized_power_law kernel params must include 'alpha' and 'r0'; "
                f"got {sorted(params)}."
            )
        return cls(alpha=float(params["alpha"]), r0=float(params["r0"]))


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

    SCHEMA_VERSION: ClassVar[int] = 1

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

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "type": self.name,
            "params": self.params,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ExponentialKernel:
        _validate_json(data, expected_type="exponential", schema_version=cls.SCHEMA_VERSION)
        params = data.get("params", {})
        if "xi" not in params:
            raise ValueError(f"exponential kernel params must include 'xi'; got {sorted(params)}.")
        return cls(xi=float(params["xi"]))


def load_kernel(data: dict[str, Any]) -> Kernel:
    """Reconstruct any registered kernel type from its JSON dict.

    Dispatches on the ``type`` field to the appropriate concrete class's
    `from_json` method.

    Parameters
    ----------
    data : dict
        A dict produced by some `Kernel.to_json()` call.

    Returns
    -------
    Kernel
        The reconstructed kernel.

    Raises
    ------
    ValueError
        If ``type`` is missing or unrecognized.
    """
    kernel_type = data.get("type")
    if kernel_type == "crossing":
        return CrossingKernel.from_json(data)
    if kernel_type == "regularized_power_law":
        return RegularizedPowerLawKernel.from_json(data)
    if kernel_type == "exponential":
        return ExponentialKernel.from_json(data)
    raise ValueError(
        f"Unknown kernel type {kernel_type!r}; expected one of "
        f"'crossing', 'regularized_power_law', 'exponential'."
    )


def _validate_json(data: dict[str, Any], *, expected_type: str, schema_version: int) -> None:
    """Shared type/version validator for kernel `from_json` methods."""
    actual_type = data.get("type")
    if actual_type != expected_type:
        raise ValueError(f"Expected type={expected_type!r}, got {actual_type!r}.")
    actual_version = data.get("schema_version")
    if actual_version != schema_version:
        raise ValueError(f"Unsupported schema_version {actual_version}; expected {schema_version}.")

"""Fault analysis utilities for weave schedules.

This package provides the Pauli-propagation engine, residual-error
enumeration, and weight-$\\le 2$ validation used by the geometry pass
and the effective-distance diagnostics. The API is intentionally
schedule-agnostic: every function here takes a
:class:`~weave.ir.Schedule` and its relevant parity-check matrices,
with no code-family-specific case logic.

Public surface
--------------
Pauli primitive (see :mod:`.pauli`)
    :class:`Pauli`,
    :func:`propagate_cnot`, :func:`propagate_h`, :func:`propagate_s`,
    :func:`propagate_x`, :func:`propagate_y`, :func:`propagate_z`,
    :func:`propagate_i`,
    :func:`measure_x`, :func:`measure_z`.
Schedule walker (see :mod:`.propagation`)
    :class:`AncillaFlip`, :class:`FaultLocation`,
    :class:`PropagationResult`,
    :func:`propagate_fault`,
    :func:`build_single_pair_fault`,
    :func:`propagate_single_pair_event`.
Residual errors (see :mod:`.residual`)
    :class:`ResidualError`,
    :func:`residual_distance`,
    :func:`effective_distance_upper_bound`,
    :func:`enumerate_hook_residuals_z_sector`.
Validation (see :mod:`.validation`)
    :class:`PairEventResult`, :class:`ValidationReport`,
    :func:`verify_weight_le_2_assumption`.

See the individual module docstrings for mathematical definitions and
citations (Gottesman 1997 for the symplectic Pauli representation;
Strikis, Browne, Beverland 2026 for residual-error formalism;
Di Bella 2026 PRX-Quantum-under-review for the correlated-noise
weight-$\\le 2$ assumption).
"""

from __future__ import annotations

from .pauli import (
    Pauli,
    measure_x,
    measure_z,
    propagate_cnot,
    propagate_h,
    propagate_i,
    propagate_s,
    propagate_x,
    propagate_y,
    propagate_z,
)
from .propagation import (
    AncillaFlip,
    FaultLocation,
    PropagationResult,
    build_single_pair_fault,
    propagate_fault,
    propagate_single_pair_event,
)
from .residual import (
    ResidualError,
    effective_distance_upper_bound,
    enumerate_hook_residuals_z_sector,
    residual_distance,
)
from .validation import (
    PairEventResult,
    ValidationReport,
    verify_weight_le_2_assumption,
)

__all__ = [
    "AncillaFlip",
    "FaultLocation",
    "PairEventResult",
    "Pauli",
    "PropagationResult",
    "ResidualError",
    "ValidationReport",
    "build_single_pair_fault",
    "effective_distance_upper_bound",
    "enumerate_hook_residuals_z_sector",
    "measure_x",
    "measure_z",
    "propagate_cnot",
    "propagate_fault",
    "propagate_h",
    "propagate_i",
    "propagate_s",
    "propagate_single_pair_event",
    "propagate_x",
    "propagate_y",
    "propagate_z",
    "residual_distance",
    "verify_weight_le_2_assumption",
]

"""Polyline distance, proximity kernels, and retained-channel pair probabilities.

This module is the geometry substrate consumed by the weave compiler
(:mod:`weave.compiler`, PR 5 onward). It ports the reference
implementation from the paper *Geometry-induced correlated noise in qLDPC
syndrome extraction* (Di Bella 2026) into a code-agnostic layer with
full type annotations, a structural :class:`Kernel` protocol, and
explicit input validation.

Public API
----------
Distance
    :data:`Point3`, :data:`Polyline`,
    :func:`segment_distance`, :func:`polyline_distance`
Kernels
    :class:`Kernel` (protocol),
    :class:`CrossingKernel`,
    :class:`RegularizedPowerLawKernel`,
    :class:`ExponentialKernel`
Pair probabilities
    :func:`pair_amplitude`,
    :func:`pair_location_strength`,
    :func:`exact_twirled_pair_probability`,
    :func:`weak_pair_probability`
"""

from .distance import Point3, Polyline, polyline_distance, segment_distance
from .kernels import (
    CrossingKernel,
    ExponentialKernel,
    Kernel,
    RegularizedPowerLawKernel,
)
from .pair import (
    exact_twirled_pair_probability,
    pair_amplitude,
    pair_location_strength,
    weak_pair_probability,
)

__all__ = [
    "CrossingKernel",
    "ExponentialKernel",
    "Kernel",
    "Point3",
    "Polyline",
    "RegularizedPowerLawKernel",
    "exact_twirled_pair_probability",
    "pair_amplitude",
    "pair_location_strength",
    "polyline_distance",
    "segment_distance",
    "weak_pair_probability",
]

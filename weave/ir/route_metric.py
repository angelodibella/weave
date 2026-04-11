"""`RoutePairMetric` — reduces a route pair to a scalar summary.

The geometry pipeline is split into two stages:

1. **Route-pair reduction.** Given two routed polylines, return a
   scalar summary of their geometric relationship. The default (and
   bbstim-faithful) reduction is minimum polyline distance; richer
   reductions such as overlap length, angle-weighted distance, or
   shared-layer traversal can be plugged in as future implementations
   of the same protocol.

2. **Coupling law.** Given the scalar summary, compute a coupling
   strength via :class:`~weave.ir.kernel.Kernel` and finally a pair
   probability via :func:`~weave.geometry.pair.exact_twirled_pair_probability`.

Splitting these stages means the compiler's geometry pass does

.. code-block:: python

    d = route_metric(poly_a, poly_b)       # scalar summary
    p = sin²(τ · J₀ · kernel(d))           # coupling law → pair prob

rather than hard-wiring `polyline_distance → κ(d)`. This makes the IR
ready for the richer routed-geometry work pointed at by HAL and
directional-qLDPC papers without an API break.

The protocol in this module is intentionally minimal: `name`, `params`,
`__call__`, and `to_json`. A registered dispatch via
:func:`load_route_metric` handles polymorphic JSON deserialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

from ..geometry import Polyline, polyline_distance


@runtime_checkable
class RoutePairMetric(Protocol):
    """Structural type for a route-pair → scalar reduction.

    Any object with `name`, `params`, `__call__(poly_a, poly_b) ->
    float`, and `to_json() -> dict` satisfies this protocol. Concrete
    implementations shipped with weave are frozen dataclasses, but
    user code can plug in any callable matching this shape.

    The canonical interpretation of the returned scalar is "a
    separation in the same units the kernel expects", but richer
    metrics may return other summaries (overlap length, angle, etc.)
    and pair with matching kernels. The protocol is deliberately
    agnostic.
    """

    @property
    def name(self) -> str:
        """Stable type identifier used for JSON dispatch."""
        ...

    @property
    def params(self) -> dict[str, float]:
        """Parameter dictionary. Empty for parameterless metrics."""
        ...

    def __call__(self, poly_a: Polyline, poly_b: Polyline) -> float:
        """Return a scalar summary of the route pair."""
        ...

    def to_json(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        ...


@dataclass(frozen=True)
class MinDistanceMetric:
    """Default route-pair metric: minimum polyline distance.

    Returns :func:`~weave.geometry.polyline_distance` between the two
    input polylines. This is the reduction used by the paper and by
    bbstim; pairing it with the crossing kernel reproduces the
    crossing diagnostic, and pairing with a regularized power-law or
    exponential kernel reproduces the distance-decay results.

    The parameterless design means all serialized specs look the same:
    `{"schema_version": 1, "type": "min_distance", "params": {}}`.
    """

    SCHEMA_VERSION: ClassVar[int] = 1

    @property
    def name(self) -> str:
        return "min_distance"

    @property
    def params(self) -> dict[str, float]:
        return {}

    def __call__(self, poly_a: Polyline, poly_b: Polyline) -> float:
        return polyline_distance(poly_a, poly_b)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "type": self.name,
            "params": self.params,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> MinDistanceMetric:
        _validate_metric_json(data, expected_type="min_distance", schema_version=cls.SCHEMA_VERSION)
        return cls()


def load_route_metric(data: dict[str, Any]) -> RoutePairMetric:
    """Reconstruct any registered route metric from its JSON dict.

    Dispatches on the ``type`` field to the appropriate concrete
    class's `from_json` method.

    Parameters
    ----------
    data : dict
        A dict produced by some `RoutePairMetric.to_json()` call.

    Returns
    -------
    RoutePairMetric
        The reconstructed metric.

    Raises
    ------
    ValueError
        If ``type`` is missing or unrecognized.
    """
    metric_type = data.get("type")
    if metric_type == "min_distance":
        return MinDistanceMetric.from_json(data)
    raise ValueError(f"Unknown route metric type {metric_type!r}; expected one of 'min_distance'.")


def _validate_metric_json(data: dict[str, Any], *, expected_type: str, schema_version: int) -> None:
    """Shared type/version validator for route metric `from_json` methods."""
    actual_type = data.get("type")
    if actual_type != expected_type:
        raise ValueError(f"Expected type={expected_type!r}, got {actual_type!r}.")
    actual_version = data.get("schema_version")
    if actual_version != schema_version:
        raise ValueError(f"Unsupported schema_version {actual_version}; expected {schema_version}.")

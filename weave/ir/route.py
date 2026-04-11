"""`RouteID` — stable identifier for a routed Tanner-graph edge.

In PR 2, `RoutingGeometry.edges` was keyed by `(source, target)` tuples.
That representation is ambiguous when a Tanner-graph edge appears in
multiple schedule ticks (different routing per tick), when imported
schedules disambiguate routes by algebraic term name, or when a single
`(source, target)` pair has multiple parallel routes in the same tick
(lanes in a biplanar embedding).

`RouteID` is the structured replacement: `(source, target, step_tick,
term_name, instance)`. It preserves backward compatibility via a
`from_tuple` factory and a `to_tuple` accessor; the updated
`RoutingGeometry` and embedding protocol accept both in PR 4 and
beyond.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RouteID:
    """Stable identifier for a routed Tanner-graph edge.

    Parameters
    ----------
    source : int
        Source qubit index (typically a data qubit).
    target : int
        Target qubit index (typically a check ancilla).
    step_tick : int, optional
        Tick index within the schedule block where this route is
        active. Defaults to 0.
    term_name : str or None, optional
        Algebraic or schedule-level label for provenance (e.g. ``"B1"``
        for a BB-code `B_1` round). Defaults to ``None``.
    instance : int, optional
        Disambiguator for parallel routes between the same endpoints at
        the same tick (e.g. different lanes in a biplanar embedding).
        Defaults to 0.
    """

    source: int
    target: int
    step_tick: int = 0
    term_name: str | None = None
    instance: int = 0

    def to_tuple(self) -> tuple[int, int]:
        """Return the legacy `(source, target)` view of this route.

        Discards `step_tick`, `term_name`, and `instance`. Useful when
        interoperating with code that predates `RouteID`.
        """
        return (self.source, self.target)

    @classmethod
    def from_tuple(
        cls,
        edge: tuple[int, int],
        *,
        step_tick: int = 0,
        term_name: str | None = None,
        instance: int = 0,
    ) -> RouteID:
        """Lift a legacy `(source, target)` tuple to a full `RouteID`.

        This is the canonical backward-compatibility path. A tuple with
        default metadata maps to `RouteID(source, target, 0, None, 0)`.
        """
        return cls(
            source=edge[0],
            target=edge[1],
            step_tick=step_tick,
            term_name=term_name,
            instance=instance,
        )


def route_id_sort_key(rid: RouteID) -> tuple[int, int, int, int, str]:
    """Deterministic sort key for `RouteID`.

    Frozen dataclasses don't implement ordering by default (and
    generating it via `order=True` breaks on the optional `term_name`
    field). This helper gives a stable total order for `sorted(...)`
    calls in JSON serialization and deterministic iteration.
    """
    return (
        rid.source,
        rid.target,
        rid.step_tick,
        rid.instance,
        rid.term_name if rid.term_name is not None else "",
    )

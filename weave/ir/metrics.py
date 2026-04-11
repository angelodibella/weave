r"""Exposure metrics, correlation edges, and support records.

This module defines the pure-data tables that the compiler emits
alongside the Stim circuit when geometry-induced noise is active:

* :class:`CorrelationEdgeRecord` — one edge of the data-qubit
  correlation graph, weighted by the total pair probability that the
  retained channel assigns to that qubit pair under union bound.
* :class:`SupportExposureRecord` — the "exposure" of a single logical
  support, computed as the total pair probability of retained events
  whose data-level image is contained entirely inside that support.
* :class:`ExposureMetrics` — the four-way decomposition of total
  exposure over `per_support`, `per_tick`, `per_route_pair`, and
  `per_data_pair` tables, with `total()`, `by_logical()`, and
  `max_over_family()` queries.

All three classes are frozen dataclasses with native JSON round-trip
and schema versioning. They form the third slice of
:class:`~weave.ir.CompiledExtraction` (alongside PR 8's
:class:`~weave.ir.ProvenanceRecord`) and are the object the
benchmarks and the optimizer query.

Exposure semantics (pinned here so downstream code can rely on it)
-------------------------------------------------------------------
Given a tuple of :class:`~weave.ir.ProvenanceRecord` and a set
:math:`L \subseteq \{0, \dots, n-1\}`, the *exposure of L* is

.. math::

    \mathcal{E}(L) \;:=\; \sum_{\substack{\mathrm{rec} \\ \mathrm{rec.data\_support} \subseteq L}}
        \mathrm{rec.pair\_probability}.

For a list of representative logical supports `family = [L_1, …, L_m]`,
the kernel exposure scale is

.. math::

    J_\kappa \;:=\; \max_{L \in \mathcal{F}} \mathcal{E}(L).

This is the object that :meth:`ExposureMetrics.max_over_family` returns.

References
----------
- Di Bella, *Geometry-induced correlated noise in qLDPC syndrome
  extraction* (PRX Quantum, under review, 2026). §III, exposure and
  `J_\kappa` scale.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from .compiled import ProvenanceRecord
from .route import RouteID

__all__ = [
    "CorrelationEdgeRecord",
    "ExposureMetrics",
    "SupportExposureRecord",
    "build_correlation_edges",
    "build_exposure_metrics",
]


InteractionSector = Literal["X", "Z"]


# =============================================================================
# Row types
# =============================================================================


@dataclass(frozen=True)
class SupportExposureRecord:
    """Exposure of one logical-representative support.

    Parameters
    ----------
    logical_index : int
        Index of the logical operator this support represents (as
        returned by :meth:`~weave.codes.CSSCode.find_logicals`).
    support : tuple[int, ...]
        Sorted data qubit indices of the logical's representative.
    exposure : float
        Sum of `pair_probability` over all `ProvenanceRecord`s whose
        `data_support` is a subset of `support`.
    """

    logical_index: int
    support: tuple[int, ...]
    exposure: float

    def __post_init__(self) -> None:
        if tuple(sorted(self.support)) != tuple(self.support):
            object.__setattr__(self, "support", tuple(sorted(self.support)))
        if self.exposure < 0:
            raise ValueError(f"exposure must be nonnegative, got {self.exposure}")

    def to_json(self) -> dict[str, Any]:
        return {
            "logical_index": self.logical_index,
            "support": list(self.support),
            "exposure": self.exposure,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SupportExposureRecord:
        return cls(
            logical_index=int(data["logical_index"]),
            support=tuple(int(q) for q in data["support"]),
            exposure=float(data["exposure"]),
        )


@dataclass(frozen=True)
class CorrelationEdgeRecord:
    """One weighted edge of the data-qubit correlation graph.

    Parameters
    ----------
    qubit_a, qubit_b : int
        The two data qubits touched by the pair fault. Canonicalized
        so that `qubit_a < qubit_b`.
    weight : float
        Total pair probability union-bounded over all events in this
        sector whose propagated data support is exactly
        `{qubit_a, qubit_b}`.
    sector : {"X", "Z"}
        CSS sector the pair fault lives in. A given qubit pair may
        appear twice in the table (once per sector) when both sectors
        produce events on it.
    """

    qubit_a: int
    qubit_b: int
    weight: float
    sector: InteractionSector

    def __post_init__(self) -> None:
        if self.qubit_a == self.qubit_b:
            raise ValueError(
                f"CorrelationEdgeRecord requires distinct qubits, both were {self.qubit_a}"
            )
        if self.qubit_a > self.qubit_b:
            # Swap via temporary so neither assignment depends on the
            # other's already-overwritten value.
            lo, hi = self.qubit_b, self.qubit_a
            object.__setattr__(self, "qubit_a", lo)
            object.__setattr__(self, "qubit_b", hi)
        if self.weight < 0:
            raise ValueError(f"weight must be nonnegative, got {self.weight}")
        if self.sector not in ("X", "Z"):
            raise ValueError(f"sector must be 'X' or 'Z', got {self.sector!r}")

    def to_json(self) -> dict[str, Any]:
        return {
            "qubit_a": self.qubit_a,
            "qubit_b": self.qubit_b,
            "weight": self.weight,
            "sector": self.sector,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> CorrelationEdgeRecord:
        return cls(
            qubit_a=int(data["qubit_a"]),
            qubit_b=int(data["qubit_b"]),
            weight=float(data["weight"]),
            sector=data["sector"],
        )


# =============================================================================
# ExposureMetrics
# =============================================================================


@dataclass(frozen=True)
class ExposureMetrics:
    r"""Four-way decomposition of total exposure over the retained channel.

    Every nontrivial-weight :class:`~weave.ir.ProvenanceRecord` is
    counted in at least `per_tick` and `per_route_pair`; it is counted
    in `per_data_pair` when its propagated data image has weight
    exactly 2 (the typical retained-channel case); and it is counted
    in `per_support` entries whose support contains the image.

    Parameters
    ----------
    per_support : tuple[SupportExposureRecord, ...]
        One record per reference logical support. Typically built
        from :meth:`~weave.codes.CSSCode.find_logicals`. Subset
        semantics: `rec.exposure = sum over records with data_support ⊆ rec.support`.
    per_tick : tuple[tuple[int, float], ...]
        `(tick_index, exposure)` sorted by `tick_index`. The exposure
        at a tick is the sum of `pair_probability` over every record
        injected at that tick, regardless of data weight.
    per_route_pair : tuple[tuple[RouteID, RouteID, float], ...]
        `(route_a, route_b, exposure)` sorted lexicographically by
        route pair. Typically one-to-one with provenance records,
        since each event is attributed to exactly one route pair.
    per_data_pair : tuple[tuple[int, int, float], ...]
        `(qubit_a, qubit_b, exposure)` sorted by `(qubit_a, qubit_b)`.
        Aggregates over weight-2 records only — the aggregate
        matches `total()` when every record has weight 2 (the
        retained-channel regime).
    """

    SCHEMA_VERSION: ClassVar[int] = 1

    per_support: tuple[SupportExposureRecord, ...] = ()
    per_tick: tuple[tuple[int, float], ...] = ()
    per_route_pair: tuple[tuple[RouteID, RouteID, float], ...] = ()
    per_data_pair: tuple[tuple[int, int, float], ...] = ()

    def __post_init__(self) -> None:
        # Coerce iterables to tuples for hashability.
        if not isinstance(self.per_support, tuple):
            object.__setattr__(self, "per_support", tuple(self.per_support))
        if not isinstance(self.per_tick, tuple):
            object.__setattr__(self, "per_tick", tuple(self.per_tick))
        if not isinstance(self.per_route_pair, tuple):
            object.__setattr__(self, "per_route_pair", tuple(self.per_route_pair))
        if not isinstance(self.per_data_pair, tuple):
            object.__setattr__(self, "per_data_pair", tuple(self.per_data_pair))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def total(self) -> float:
        r"""Return :math:`\sum_{\mathrm{events}} p_{\mathrm{event}}`.

        Computed as the sum of the `per_tick` column so it captures
        *every* provenance record, regardless of data weight. For
        schedules where every record is weight-2, this matches the
        sum of `per_data_pair` and `per_route_pair` to machine
        precision.
        """
        return float(sum(exposure for _, exposure in self.per_tick))

    def by_logical(self, logical_index: int) -> float:
        """Exposure of a specific logical, looked up from `per_support`.

        Parameters
        ----------
        logical_index : int
            The index into :meth:`~weave.codes.CSSCode.find_logicals`.

        Returns
        -------
        float
            The exposure of the logical's representative support, or
            `0.0` if no `SupportExposureRecord` with this index
            exists.
        """
        for rec in self.per_support:
            if rec.logical_index == logical_index:
                return rec.exposure
        return 0.0

    def max_over_family(self, family: Sequence[Iterable[int]]) -> float:
        r"""Compute :math:`J_\kappa = \max_{L \in \mathcal{F}} \mathcal{E}(L)`.

        Recomputes the exposure of each support in `family` from
        `per_data_pair` (which contains every weight-2 record),
        taking the maximum over the family. Returns `0.0` for an
        empty family.

        Parameters
        ----------
        family : Sequence[Iterable[int]]
            Each element is a logical support — the sorted or
            unsorted set of data qubit indices where a logical
            representative acts nontrivially.

        Returns
        -------
        float
            The maximum exposure over the family.

        Notes
        -----
        The implementation scans `per_data_pair` because it holds
        the canonical pair-event aggregation. Events with
        `data_weight != 2` are by construction excluded from
        `per_data_pair`; if a caller needs to include them, they
        should call :func:`build_exposure_metrics` with a family
        covering the non-weight-2 events and query `per_support`.
        """
        best = 0.0
        for raw_support in family:
            support_set = frozenset(int(q) for q in raw_support)
            exposure = 0.0
            for qa, qb, weight in self.per_data_pair:
                if qa in support_set and qb in support_set:
                    exposure += weight
            if exposure > best:
                best = exposure
        return best

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @property
    def schema_version(self) -> int:
        return self.SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "type": "exposure_metrics",
            "per_support": [r.to_json() for r in self.per_support],
            "per_tick": [[int(t), float(e)] for t, e in self.per_tick],
            "per_route_pair": [
                [ra.to_json(), rb.to_json(), float(e)] for ra, rb, e in self.per_route_pair
            ],
            "per_data_pair": [[int(a), int(b), float(e)] for a, b, e in self.per_data_pair],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ExposureMetrics:
        if data.get("type") != "exposure_metrics":
            raise ValueError(f"Expected type='exposure_metrics', got {data.get('type')!r}.")
        version = data.get("schema_version")
        if version != cls.SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported ExposureMetrics schema_version {version}; "
                f"expected {cls.SCHEMA_VERSION}."
            )
        per_support = tuple(SupportExposureRecord.from_json(r) for r in data.get("per_support", []))
        per_tick = tuple((int(t), float(e)) for t, e in data.get("per_tick", []))
        per_route_pair = tuple(
            (RouteID.from_json(ra), RouteID.from_json(rb), float(e))
            for ra, rb, e in data.get("per_route_pair", [])
        )
        per_data_pair = tuple(
            (int(a), int(b), float(e)) for a, b, e in data.get("per_data_pair", [])
        )
        return cls(
            per_support=per_support,
            per_tick=per_tick,
            per_route_pair=per_route_pair,
            per_data_pair=per_data_pair,
        )


# =============================================================================
# Builders — aggregate a ProvenanceRecord list into tables
# =============================================================================


def build_correlation_edges(
    provenance: Sequence[ProvenanceRecord],
) -> tuple[CorrelationEdgeRecord, ...]:
    """Aggregate provenance records into a sorted correlation-edge table.

    Only weight-2 records contribute; smaller or larger data supports
    do not define a single qubit-pair and are silently dropped. For
    each `(sector, qubit pair)` the weight is the union-bound sum of
    pair probabilities across records, so

    .. code-block:: text

        sum(e.weight for e in edges) == sum(rec.pair_probability
            for rec in provenance if rec.data_weight == 2)

    holds to machine precision.

    The returned tuple is sorted by `(sector, qubit_a, qubit_b)`.
    """
    aggregated: dict[tuple[InteractionSector, int, int], float] = defaultdict(float)
    for rec in provenance:
        if rec.data_weight != 2:
            continue
        qa, qb = rec.data_support
        key: tuple[InteractionSector, int, int] = (rec.sector, int(qa), int(qb))
        aggregated[key] += rec.pair_probability

    edges = tuple(
        CorrelationEdgeRecord(qubit_a=qa, qubit_b=qb, weight=weight, sector=sector)
        for (sector, qa, qb), weight in sorted(aggregated.items())
    )
    return edges


def build_exposure_metrics(
    provenance: Sequence[ProvenanceRecord],
    *,
    logical_supports: Sequence[tuple[int, ...]] = (),
) -> ExposureMetrics:
    """Compute the four-way exposure decomposition from provenance.

    Parameters
    ----------
    provenance : Sequence[ProvenanceRecord]
        The records to aggregate. Typically comes from
        :func:`~weave.compiler.geometry_pass.compute_provenance`.
    logical_supports : Sequence[tuple[int, ...]], optional
        One support per logical representative, e.g. from
        :meth:`~weave.codes.CSSCode.find_logicals`. Each is used to
        build a :class:`SupportExposureRecord`. Defaults to empty:
        `per_support` is then `()`.

    Returns
    -------
    ExposureMetrics
        Populated with all four decompositions.
    """
    # per_tick.
    per_tick_accum: dict[int, float] = defaultdict(float)
    for rec in provenance:
        per_tick_accum[rec.tick_index] += rec.pair_probability
    per_tick = tuple(sorted(per_tick_accum.items()))

    # per_route_pair: one entry per record. Lift (control, target)
    # tuples into RouteIDs so downstream tools that use RouteID
    # directly don't have to re-lift.
    per_route_pair_items: list[tuple[RouteID, RouteID, float]] = []
    for rec in provenance:
        route_a = RouteID(
            source=rec.edge_a[0],
            target=rec.edge_a[1],
            step_tick=rec.tick_index,
        )
        route_b = RouteID(
            source=rec.edge_b[0],
            target=rec.edge_b[1],
            step_tick=rec.tick_index,
        )
        per_route_pair_items.append((route_a, route_b, rec.pair_probability))
    per_route_pair = tuple(sorted(per_route_pair_items, key=_route_pair_sort_key))

    # per_data_pair: over weight-2 records, summed per data pair and
    # sector-merged. This stays aligned with build_correlation_edges
    # modulo the sector column.
    pair_accum: dict[tuple[int, int], float] = defaultdict(float)
    for rec in provenance:
        if rec.data_weight != 2:
            continue
        qa, qb = rec.data_support
        pair_accum[(int(qa), int(qb))] += rec.pair_probability
    per_data_pair = tuple(sorted((a, b, w) for (a, b), w in pair_accum.items()))

    # per_support.
    per_support: list[SupportExposureRecord] = []
    for i, raw_support in enumerate(logical_supports):
        support_set = frozenset(int(q) for q in raw_support)
        exposure = 0.0
        for rec in provenance:
            if set(rec.data_support).issubset(support_set):
                exposure += rec.pair_probability
        per_support.append(
            SupportExposureRecord(
                logical_index=i,
                support=tuple(sorted(support_set)),
                exposure=exposure,
            )
        )

    return ExposureMetrics(
        per_support=tuple(per_support),
        per_tick=per_tick,
        per_route_pair=per_route_pair,
        per_data_pair=per_data_pair,
    )


def _route_pair_sort_key(
    item: tuple[RouteID, RouteID, float],
) -> tuple[int, int, int, int, int, int, int, int]:
    ra, rb, _ = item
    return (
        ra.source,
        ra.target,
        ra.step_tick,
        ra.instance,
        rb.source,
        rb.target,
        rb.step_tick,
        rb.instance,
    )

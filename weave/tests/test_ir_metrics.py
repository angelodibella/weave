"""Tests for `weave.ir.metrics`: exposure tables and correlation edges.

Every numeric expectation is hand-verified against tiny fixture
ProvenanceRecord lists. The builder functions are unit-tested in
isolation so that PR 8's geometry pass and PR 9's compiler wiring
share a single aggregation ground truth.
"""

from __future__ import annotations

import pytest

from weave.ir import (
    CorrelationEdgeRecord,
    ExposureMetrics,
    ProvenanceRecord,
    RouteID,
    SupportExposureRecord,
    build_correlation_edges,
    build_exposure_metrics,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _record(
    *,
    tick: int,
    edge_a: tuple[int, int],
    edge_b: tuple[int, int],
    sector: str = "X",
    distance: float = 1.0,
    pair_prob: float,
    data_support: tuple[int, ...],
    pauli: str = "X",
) -> ProvenanceRecord:
    return ProvenanceRecord(
        tick_index=tick,
        edge_a=edge_a,
        edge_b=edge_b,
        sector=sector,
        routed_distance=distance,
        pair_probability=pair_prob,
        data_support=data_support,
        data_pauli_symbols=tuple(pauli for _ in data_support),
    )


# ---------------------------------------------------------------------------
# SupportExposureRecord
# ---------------------------------------------------------------------------


class TestSupportExposureRecord:
    def test_basic_construction(self):
        rec = SupportExposureRecord(logical_index=0, support=(1, 3, 5), exposure=0.25)
        assert rec.logical_index == 0
        assert rec.support == (1, 3, 5)
        assert rec.exposure == 0.25

    def test_support_is_sorted(self):
        rec = SupportExposureRecord(logical_index=0, support=(5, 1, 3), exposure=0.1)
        assert rec.support == (1, 3, 5)

    def test_negative_exposure_rejected(self):
        with pytest.raises(ValueError, match="nonnegative"):
            SupportExposureRecord(logical_index=0, support=(0, 1), exposure=-0.1)

    def test_round_trip(self):
        rec = SupportExposureRecord(logical_index=2, support=(0, 2, 4), exposure=0.375)
        assert SupportExposureRecord.from_json(rec.to_json()) == rec


# ---------------------------------------------------------------------------
# CorrelationEdgeRecord
# ---------------------------------------------------------------------------


class TestCorrelationEdgeRecord:
    def test_basic_construction(self):
        edge = CorrelationEdgeRecord(qubit_a=1, qubit_b=3, weight=0.1, sector="X")
        assert edge.qubit_a == 1
        assert edge.qubit_b == 3
        assert edge.weight == 0.1
        assert edge.sector == "X"

    def test_canonical_qubit_order(self):
        """Passing the qubits out of order is auto-canonicalized."""
        edge = CorrelationEdgeRecord(qubit_a=5, qubit_b=2, weight=0.1, sector="Z")
        assert edge.qubit_a == 2
        assert edge.qubit_b == 5

    def test_equal_qubits_rejected(self):
        with pytest.raises(ValueError, match="distinct"):
            CorrelationEdgeRecord(qubit_a=3, qubit_b=3, weight=0.1, sector="X")

    def test_negative_weight_rejected(self):
        with pytest.raises(ValueError, match="nonnegative"):
            CorrelationEdgeRecord(qubit_a=0, qubit_b=1, weight=-0.01, sector="X")

    def test_bad_sector_rejected(self):
        with pytest.raises(ValueError, match="sector"):
            CorrelationEdgeRecord(qubit_a=0, qubit_b=1, weight=0.1, sector="Y")  # type: ignore[arg-type]

    def test_round_trip(self):
        edge = CorrelationEdgeRecord(qubit_a=0, qubit_b=4, weight=0.2, sector="Z")
        assert CorrelationEdgeRecord.from_json(edge.to_json()) == edge


# ---------------------------------------------------------------------------
# build_correlation_edges
# ---------------------------------------------------------------------------


class TestBuildCorrelationEdges:
    def test_empty_provenance(self):
        assert build_correlation_edges([]) == ()

    def test_single_weight_2_record(self):
        rec = _record(
            tick=0,
            edge_a=(0, 4),
            edge_b=(2, 5),
            pair_prob=0.1,
            data_support=(0, 2),
        )
        edges = build_correlation_edges([rec])
        assert edges == (CorrelationEdgeRecord(qubit_a=0, qubit_b=2, weight=0.1, sector="X"),)

    def test_multiple_records_same_pair_union_bound(self):
        """Two records on the same pair sum their probabilities."""
        recs = [
            _record(tick=0, edge_a=(0, 4), edge_b=(2, 5), pair_prob=0.1, data_support=(0, 2)),
            _record(tick=1, edge_a=(0, 6), edge_b=(2, 7), pair_prob=0.05, data_support=(0, 2)),
        ]
        edges = build_correlation_edges(recs)
        assert len(edges) == 1
        assert edges[0].weight == pytest.approx(0.15, abs=1e-12)

    def test_different_sectors_produce_separate_records(self):
        recs = [
            _record(
                tick=0, edge_a=(0, 4), edge_b=(2, 5), sector="X", pair_prob=0.1, data_support=(0, 2)
            ),
            _record(
                tick=0,
                edge_a=(0, 4),
                edge_b=(2, 5),
                sector="Z",
                pair_prob=0.2,
                data_support=(0, 2),
                pauli="Z",
            ),
        ]
        edges = build_correlation_edges(recs)
        assert len(edges) == 2
        sectors = {e.sector for e in edges}
        assert sectors == {"X", "Z"}

    def test_non_weight_2_records_skipped(self):
        recs = [
            _record(tick=0, edge_a=(0, 4), edge_b=(2, 5), pair_prob=0.1, data_support=(0, 2)),
            _record(
                tick=0, edge_a=(0, 4), edge_b=(2, 5), pair_prob=0.3, data_support=(0, 1, 2)
            ),  # weight 3
        ]
        edges = build_correlation_edges(recs)
        assert len(edges) == 1
        assert edges[0].weight == pytest.approx(0.1, abs=1e-12)

    def test_sorted_output(self):
        """Edges come back sorted by (sector, qubit_a, qubit_b)."""
        recs = [
            _record(tick=0, edge_a=(5, 10), edge_b=(3, 11), pair_prob=0.1, data_support=(3, 5)),
            _record(tick=0, edge_a=(0, 10), edge_b=(1, 11), pair_prob=0.2, data_support=(0, 1)),
        ]
        edges = build_correlation_edges(recs)
        assert [(e.qubit_a, e.qubit_b) for e in edges] == [(0, 1), (3, 5)]


# ---------------------------------------------------------------------------
# build_exposure_metrics
# ---------------------------------------------------------------------------


class TestBuildExposureMetrics:
    def test_empty_provenance_yields_empty_tables(self):
        m = build_exposure_metrics([])
        assert m.per_tick == ()
        assert m.per_route_pair == ()
        assert m.per_data_pair == ()
        assert m.per_support == ()
        assert m.total() == 0.0

    def test_single_event(self):
        rec = _record(
            tick=3,
            edge_a=(0, 4),
            edge_b=(2, 5),
            pair_prob=0.1,
            data_support=(0, 2),
        )
        m = build_exposure_metrics([rec])
        assert m.per_tick == ((3, 0.1),)
        assert m.per_data_pair == ((0, 2, 0.1),)
        assert len(m.per_route_pair) == 1
        ra, rb, weight = m.per_route_pair[0]
        assert isinstance(ra, RouteID)
        assert ra.step_tick == 3
        assert weight == 0.1
        assert m.total() == pytest.approx(0.1, abs=1e-12)

    def test_total_equals_sum_of_probabilities_to_1e_12(self):
        """Plan acceptance test #2: total() == sum(pair_probability)
        over provenance to 1e-12."""
        recs = [
            _record(
                tick=0, edge_a=(0, 4), edge_b=(2, 5), pair_prob=0.123456789, data_support=(0, 2)
            ),
            _record(tick=1, edge_a=(0, 4), edge_b=(2, 5), pair_prob=0.000123, data_support=(0, 2)),
            _record(
                tick=2, edge_a=(1, 5), edge_b=(3, 6), pair_prob=0.0987654321, data_support=(1, 3)
            ),
        ]
        expected = sum(r.pair_probability for r in recs)
        m = build_exposure_metrics(recs)
        assert m.total() == pytest.approx(expected, abs=1e-12)
        # sum(per_data_pair) also equals the total when every event is weight-2.
        assert sum(w for _, _, w in m.per_data_pair) == pytest.approx(expected, abs=1e-12)

    def test_per_tick_groups_by_tick(self):
        recs = [
            _record(tick=0, edge_a=(0, 4), edge_b=(2, 5), pair_prob=0.1, data_support=(0, 2)),
            _record(tick=0, edge_a=(1, 4), edge_b=(3, 5), pair_prob=0.2, data_support=(1, 3)),
            _record(tick=5, edge_a=(0, 4), edge_b=(1, 5), pair_prob=0.05, data_support=(0, 1)),
        ]
        m = build_exposure_metrics(recs)
        assert m.per_tick == (
            (0, pytest.approx(0.3, abs=1e-12)),
            (5, pytest.approx(0.05, abs=1e-12)),
        )

    def test_per_support_uses_subset_semantics(self):
        """A logical support receives a record iff the record's data
        support is a subset of it."""
        recs = [
            _record(
                tick=0, edge_a=(0, 4), edge_b=(2, 5), pair_prob=0.1, data_support=(0, 2)
            ),  # ⊆ L_0
            _record(
                tick=0, edge_a=(1, 4), edge_b=(3, 5), pair_prob=0.2, data_support=(1, 3)
            ),  # ⊆ L_1
            _record(
                tick=0, edge_a=(0, 4), edge_b=(3, 5), pair_prob=0.3, data_support=(0, 3)
            ),  # ⊄ either
        ]
        L0 = (0, 2, 4, 6)
        L1 = (1, 3, 5, 7)
        m = build_exposure_metrics(recs, logical_supports=(L0, L1))
        assert len(m.per_support) == 2
        assert m.by_logical(0) == pytest.approx(0.1, abs=1e-12)
        assert m.by_logical(1) == pytest.approx(0.2, abs=1e-12)

    def test_by_logical_missing_returns_zero(self):
        m = build_exposure_metrics([], logical_supports=((0, 1),))
        assert m.by_logical(99) == 0.0


# ---------------------------------------------------------------------------
# ExposureMetrics.max_over_family
# ---------------------------------------------------------------------------


class TestMaxOverFamily:
    def test_empty_family_returns_zero(self):
        m = build_exposure_metrics(
            [_record(tick=0, edge_a=(0, 4), edge_b=(2, 5), pair_prob=0.1, data_support=(0, 2))]
        )
        assert m.max_over_family([]) == 0.0

    def test_max_picks_highest_exposure(self):
        """Plan acceptance test #3: max_over_family returns J_κ."""
        recs = [
            _record(tick=0, edge_a=(0, 4), edge_b=(2, 5), pair_prob=0.1, data_support=(0, 2)),
            _record(tick=0, edge_a=(1, 4), edge_b=(3, 5), pair_prob=0.25, data_support=(1, 3)),
            _record(tick=0, edge_a=(5, 4), edge_b=(7, 5), pair_prob=0.15, data_support=(5, 7)),
        ]
        m = build_exposure_metrics(recs)
        # Family of three single-pair supports, each containing exactly
        # one of the events.
        family = [(0, 2), (1, 3), (5, 7)]
        assert m.max_over_family(family) == pytest.approx(0.25, abs=1e-12)

    def test_supports_containing_multiple_events(self):
        """A support that contains multiple events gets their sum."""
        recs = [
            _record(tick=0, edge_a=(0, 4), edge_b=(2, 5), pair_prob=0.1, data_support=(0, 2)),
            _record(tick=0, edge_a=(0, 4), edge_b=(3, 5), pair_prob=0.15, data_support=(0, 3)),
        ]
        m = build_exposure_metrics(recs)
        # The support {0, 2, 3} catches both pairs.
        assert m.max_over_family([(0, 2, 3)]) == pytest.approx(0.25, abs=1e-12)


# ---------------------------------------------------------------------------
# ExposureMetrics JSON round-trip
# ---------------------------------------------------------------------------


class TestExposureMetricsRoundTrip:
    def test_empty_round_trip(self):
        m = ExposureMetrics()
        assert ExposureMetrics.from_json(m.to_json()) == m

    def test_populated_round_trip(self):
        recs = [
            _record(tick=0, edge_a=(0, 4), edge_b=(2, 5), pair_prob=0.1, data_support=(0, 2)),
            _record(tick=1, edge_a=(1, 4), edge_b=(3, 5), pair_prob=0.2, data_support=(1, 3)),
        ]
        m = build_exposure_metrics(recs, logical_supports=((0, 2),))
        reconstructed = ExposureMetrics.from_json(m.to_json())
        assert reconstructed == m

    def test_rejects_wrong_type(self):
        with pytest.raises(ValueError, match="type"):
            ExposureMetrics.from_json({"type": "wrong", "schema_version": 1})

    def test_rejects_wrong_schema_version(self):
        with pytest.raises(ValueError, match="schema_version"):
            ExposureMetrics.from_json({"type": "exposure_metrics", "schema_version": 999})

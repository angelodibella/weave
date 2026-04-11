"""Tests for `weave.ir.route_metric`: `RoutePairMetric` protocol and metrics."""

from __future__ import annotations

import json

import pytest

from weave.geometry import polyline_distance
from weave.ir import MinDistanceMetric, RoutePairMetric, load_route_metric

# =============================================================================
# MinDistanceMetric: behavior
# =============================================================================


class TestMinDistanceMetricBehavior:
    def test_parallel_segments(self):
        m = MinDistanceMetric()
        d = m(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)), ((0.0, 5.0, 0.0), (10.0, 5.0, 0.0)))
        assert abs(d - 5.0) < 1e-12

    def test_crossing_segments_zero(self):
        m = MinDistanceMetric()
        d = m(
            ((1.0, 0.0, 0.0), (3.0, 10.0, 0.0)),
            ((1.0, 10.0, 0.0), (3.0, 0.0, 0.0)),
        )
        assert d < 1e-10

    def test_agrees_with_polyline_distance(self):
        """Plan acceptance test: `MinDistanceMetric()` agrees with
        `polyline_distance` on 4 test polyline pairs to 1e-10.
        """
        m = MinDistanceMetric()
        cases = [
            # Crossing
            (
                ((1.0, 0.0, 0.0), (3.0, 10.0, 0.0)),
                ((1.0, 10.0, 0.0), (3.0, 0.0, 0.0)),
            ),
            # Parallel
            (
                ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)),
                ((0.0, 5.0, 0.0), (10.0, 5.0, 0.0)),
            ),
            # 3D offset
            (
                ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)),
                ((0.0, 0.0, 3.0), (10.0, 0.0, 3.0)),
            ),
            # Multi-segment biplanar surrogate
            (
                (
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0),
                    (10.0, 0.0, 1.0),
                    (10.0, 0.0, 0.0),
                ),
                (
                    (0.0, 5.0, 0.0),
                    (0.0, 5.0, 1.0),
                    (10.0, 5.0, 1.0),
                    (10.0, 5.0, 0.0),
                ),
            ),
        ]
        for poly_a, poly_b in cases:
            d_metric = m(poly_a, poly_b)
            d_direct = polyline_distance(poly_a, poly_b)
            assert abs(d_metric - d_direct) < 1e-10, (
                f"MinDistanceMetric disagrees with polyline_distance: {d_metric} vs {d_direct}"
            )


# =============================================================================
# MinDistanceMetric: metadata and serialization
# =============================================================================


class TestMinDistanceMetricMetadata:
    def test_name(self):
        assert MinDistanceMetric().name == "min_distance"

    def test_params_empty(self):
        assert MinDistanceMetric().params == {}

    def test_schema_version(self):
        assert MinDistanceMetric.SCHEMA_VERSION == 1

    def test_equality_and_hashability(self):
        assert MinDistanceMetric() == MinDistanceMetric()
        assert hash(MinDistanceMetric()) == hash(MinDistanceMetric())

    def test_frozen(self):
        from dataclasses import FrozenInstanceError

        m = MinDistanceMetric()
        with pytest.raises((FrozenInstanceError, AttributeError)):
            m.name = "other"  # type: ignore[misc]


class TestMinDistanceMetricJson:
    def test_roundtrip(self):
        m = MinDistanceMetric()
        data = m.to_json()
        restored = MinDistanceMetric.from_json(data)
        assert restored == m

    def test_to_json_keys(self):
        data = MinDistanceMetric().to_json()
        assert data["schema_version"] == 1
        assert data["type"] == "min_distance"
        assert data["params"] == {}

    def test_roundtrip_through_json_dumps(self):
        m = MinDistanceMetric()
        data = json.loads(json.dumps(m.to_json()))
        restored = MinDistanceMetric.from_json(data)
        assert restored == m

    def test_from_json_rejects_wrong_type(self):
        with pytest.raises(ValueError, match="type='min_distance'"):
            MinDistanceMetric.from_json({"schema_version": 1, "type": "exponential", "params": {}})

    def test_from_json_rejects_wrong_schema_version(self):
        with pytest.raises(ValueError, match="schema_version"):
            MinDistanceMetric.from_json(
                {"schema_version": 999, "type": "min_distance", "params": {}}
            )


# =============================================================================
# RoutePairMetric protocol
# =============================================================================


class TestRoutePairMetricProtocol:
    def test_min_distance_satisfies_protocol(self):
        assert isinstance(MinDistanceMetric(), RoutePairMetric)

    def test_protocol_includes_to_json(self):
        m = MinDistanceMetric()
        assert callable(m.to_json)


# =============================================================================
# load_route_metric dispatch
# =============================================================================


class TestLoadRouteMetric:
    def test_dispatches_to_min_distance(self):
        m = MinDistanceMetric()
        loaded = load_route_metric(m.to_json())
        assert isinstance(loaded, MinDistanceMetric)
        assert loaded == m

    def test_rejects_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown route metric type"):
            load_route_metric({"schema_version": 1, "type": "mystery", "params": {}})

    def test_rejects_missing_type(self):
        with pytest.raises(ValueError, match="Unknown route metric type"):
            load_route_metric({"schema_version": 1, "params": {}})

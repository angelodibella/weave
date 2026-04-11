"""Tests for `weave.ir.route`: `RouteID` and `route_id_sort_key`."""

from __future__ import annotations

import pytest

from weave.ir import RouteID, route_id_sort_key


class TestRouteIDBasics:
    def test_construction_minimal(self):
        r = RouteID(source=0, target=7)
        assert r.source == 0
        assert r.target == 7
        assert r.step_tick == 0
        assert r.term_name is None
        assert r.instance == 0

    def test_construction_full(self):
        r = RouteID(source=3, target=12, step_tick=5, term_name="B1", instance=2)
        assert r.source == 3
        assert r.target == 12
        assert r.step_tick == 5
        assert r.term_name == "B1"
        assert r.instance == 2

    def test_frozen(self):
        r = RouteID(source=0, target=1)
        from dataclasses import FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            r.source = 5  # type: ignore[misc]

    def test_hashable(self):
        r = RouteID(source=0, target=1)
        assert hash(r) == hash(RouteID(source=0, target=1))
        # Can be used as a dict key.
        d = {r: "value"}
        assert d[RouteID(source=0, target=1)] == "value"

    def test_equality_depends_on_all_fields(self):
        base = RouteID(source=0, target=1)
        assert base == RouteID(source=0, target=1)
        assert base != RouteID(source=0, target=2)
        assert base != RouteID(source=1, target=1)
        assert base != RouteID(source=0, target=1, step_tick=1)
        assert base != RouteID(source=0, target=1, term_name="B1")
        assert base != RouteID(source=0, target=1, instance=1)


class TestRouteIDTupleInterop:
    def test_to_tuple(self):
        r = RouteID(source=3, target=12, step_tick=5, term_name="B1", instance=2)
        # to_tuple drops extra metadata.
        assert r.to_tuple() == (3, 12)

    def test_from_tuple_default_metadata(self):
        r = RouteID.from_tuple((4, 9))
        assert r == RouteID(source=4, target=9)
        assert r.step_tick == 0
        assert r.term_name is None
        assert r.instance == 0

    def test_from_tuple_with_metadata(self):
        r = RouteID.from_tuple((4, 9), step_tick=3, term_name="A2", instance=1)
        assert r.source == 4
        assert r.target == 9
        assert r.step_tick == 3
        assert r.term_name == "A2"
        assert r.instance == 1

    def test_roundtrip_default(self):
        r = RouteID.from_tuple((0, 1))
        assert r.to_tuple() == (0, 1)


class TestRouteIDSortKey:
    def test_sort_by_source_first(self):
        routes = [
            RouteID(source=2, target=0),
            RouteID(source=0, target=5),
            RouteID(source=1, target=3),
        ]
        sorted_routes = sorted(routes, key=route_id_sort_key)
        assert [r.source for r in sorted_routes] == [0, 1, 2]

    def test_sort_by_target_when_source_equal(self):
        routes = [
            RouteID(source=0, target=5),
            RouteID(source=0, target=2),
            RouteID(source=0, target=10),
        ]
        sorted_routes = sorted(routes, key=route_id_sort_key)
        assert [r.target for r in sorted_routes] == [2, 5, 10]

    def test_sort_by_step_tick_when_endpoints_equal(self):
        routes = [
            RouteID(source=0, target=1, step_tick=5),
            RouteID(source=0, target=1, step_tick=0),
            RouteID(source=0, target=1, step_tick=3),
        ]
        sorted_routes = sorted(routes, key=route_id_sort_key)
        assert [r.step_tick for r in sorted_routes] == [0, 3, 5]

    def test_none_term_name_sorts_before_string(self):
        """None term_name should compare as empty string (sorts first)."""
        routes = [
            RouteID(source=0, target=1, term_name="B1"),
            RouteID(source=0, target=1, term_name=None),
            RouteID(source=0, target=1, term_name="A1"),
        ]
        sorted_routes = sorted(routes, key=route_id_sort_key)
        # None → "", which sorts before "A1".
        assert sorted_routes[0].term_name is None
        assert sorted_routes[1].term_name == "A1"
        assert sorted_routes[2].term_name == "B1"

    def test_sort_mixed_routes(self):
        """A realistic BB-style route sort: stable and deterministic."""
        routes = [
            RouteID(source=3, target=12, step_tick=0, term_name="B2"),
            RouteID(source=3, target=12, step_tick=0, term_name="B1"),
            RouteID(source=2, target=12, step_tick=0, term_name="B1"),
            RouteID(source=3, target=12, step_tick=1, term_name="B1"),
        ]
        sorted_routes = sorted(routes, key=route_id_sort_key)
        # Order: (2,12,B1,0) < (3,12,B1,0) < (3,12,B2,0) < (3,12,B1,1)
        assert sorted_routes[0] == RouteID(source=2, target=12, step_tick=0, term_name="B1")
        assert sorted_routes[1] == RouteID(source=3, target=12, step_tick=0, term_name="B1")
        assert sorted_routes[-1].step_tick == 1

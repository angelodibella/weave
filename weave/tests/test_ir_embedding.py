"""Tests for `weave.ir.embedding` and `weave.ir.embeddings`.

Covers the `Embedding` protocol, `RoutingGeometry` invariants, both
concrete embeddings (`StraightLineEmbedding`, `JsonPolylineEmbedding`),
schema-versioned JSON round-trip, and the Steane acceptance test from
PR 2 of `private/plan.md`.
"""

from __future__ import annotations

import json

import pytest

from weave.codes.css_code import CSSCode
from weave.ir import (
    Embedding,
    IREdge,
    JsonPolylineEmbedding,
    RouteID,
    RoutingGeometry,
    StraightLineEmbedding,
    load_embedding,
)
from weave.util import pcm

# =============================================================================
# RoutingGeometry
# =============================================================================


class TestRoutingGeometry:
    def test_empty_is_valid(self):
        rg = RoutingGeometry(edges={})
        assert len(rg) == 0

    def test_single_edge(self):
        rg = RoutingGeometry(edges={(0, 1): ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))})
        assert len(rg) == 1
        assert (0, 1) in rg
        assert rg[(0, 1)] == ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))

    def test_rejects_polyline_with_one_point(self):
        with pytest.raises(ValueError, match="at least 2 points"):
            RoutingGeometry(edges={(0, 1): ((0.0, 0.0, 0.0),)})

    def test_rejects_empty_polyline(self):
        with pytest.raises(ValueError, match="at least 2 points"):
            RoutingGeometry(edges={(0, 1): ()})

    def test_multi_segment_polyline(self):
        """4-point polyline (biplanar surrogate shape) is valid."""
        poly = ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (5.0, 0.0, 1.0), (5.0, 0.0, 0.0))
        rg = RoutingGeometry(edges={(0, 1): poly})
        assert rg[(0, 1)] == poly

    def test_name_default_and_explicit(self):
        assert RoutingGeometry(edges={}).name == ""
        assert RoutingGeometry(edges={}, name="step_3").name == "step_3"


# =============================================================================
# StraightLineEmbedding
# =============================================================================


class TestStraightLineEmbedding:
    def test_from_positions_2d_lifts_to_z_zero(self):
        emb = StraightLineEmbedding.from_positions([(0, 0), (1, 2), (3, 4)])
        assert emb.positions == (
            (0.0, 0.0, 0.0),
            (1.0, 2.0, 0.0),
            (3.0, 4.0, 0.0),
        )

    def test_from_positions_3d_pass_through(self):
        emb = StraightLineEmbedding.from_positions([(0, 0, 0), (1, 2, 3), (4, 5, 6)])
        assert emb.positions == (
            (0.0, 0.0, 0.0),
            (1.0, 2.0, 3.0),
            (4.0, 5.0, 6.0),
        )

    def test_from_positions_mixed_rejected(self):
        """Mixing 2D and 3D points is fine: each is lifted independently."""
        emb = StraightLineEmbedding.from_positions([(0, 0), (1, 2, 3)])
        assert emb.positions == ((0.0, 0.0, 0.0), (1.0, 2.0, 3.0))

    def test_from_positions_unknown_dim_rejected(self):
        with pytest.raises(ValueError, match="2D or 3D point"):
            StraightLineEmbedding.from_positions([(1,), (2,)])

    def test_direct_init_rejects_2d_positions(self):
        """Direct __init__ is strict; use from_positions for 2D inputs."""
        with pytest.raises(ValueError, match="length 2, expected 3"):
            StraightLineEmbedding(positions=((0.0, 0.0), (1.0, 1.0)))  # type: ignore[arg-type]

    def test_list_input_coerced_to_tuple(self):
        """Lists are accepted and converted to tuples in __post_init__."""
        emb = StraightLineEmbedding(
            positions=[(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]  # type: ignore[arg-type]
        )
        assert isinstance(emb.positions, tuple)

    def test_schema_version(self):
        emb = StraightLineEmbedding.from_positions([(0, 0)])
        assert emb.schema_version == StraightLineEmbedding.SCHEMA_VERSION == 1

    def test_surface_name(self):
        emb = StraightLineEmbedding.from_positions([(0, 0)])
        assert emb.surface_name == "plane"

    def test_node_position(self):
        emb = StraightLineEmbedding.from_positions([(0, 0), (5, 3), (7, 2)])
        assert emb.node_position(0) == (0.0, 0.0, 0.0)
        assert emb.node_position(1) == (5.0, 3.0, 0.0)
        assert emb.node_position(2) == (7.0, 2.0, 0.0)

    def test_node_position_out_of_range(self):
        emb = StraightLineEmbedding.from_positions([(0, 0), (1, 1)])
        with pytest.raises(IndexError, match="out of range"):
            emb.node_position(5)
        with pytest.raises(IndexError, match="out of range"):
            emb.node_position(-1)

    def test_routing_geometry_basic(self):
        emb = StraightLineEmbedding.from_positions([(0, 0), (1, 0), (2, 0), (3, 0)])
        rg = emb.routing_geometry([(0, 1), (2, 3)])
        assert len(rg) == 2
        assert rg[(0, 1)] == ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        assert rg[(2, 3)] == ((2.0, 0.0, 0.0), (3.0, 0.0, 0.0))

    def test_routing_geometry_empty_edges(self):
        emb = StraightLineEmbedding.from_positions([(0, 0), (1, 1)])
        rg = emb.routing_geometry([])
        assert len(rg) == 0

    def test_routing_geometry_propagates_name(self):
        emb = StraightLineEmbedding.from_positions([(0, 0), (1, 1)], name="test")
        rg = emb.routing_geometry([(0, 1)])
        assert rg.name == "test"

    def test_routing_geometry_rejects_out_of_range_source(self):
        emb = StraightLineEmbedding.from_positions([(0, 0), (1, 1)])
        with pytest.raises(IndexError, match="Edge source"):
            emb.routing_geometry([(5, 0)])

    def test_routing_geometry_rejects_out_of_range_target(self):
        emb = StraightLineEmbedding.from_positions([(0, 0), (1, 1)])
        with pytest.raises(IndexError, match="Edge target"):
            emb.routing_geometry([(0, 5)])

    def test_to_from_json_roundtrip(self):
        emb = StraightLineEmbedding.from_positions([(0, 0), (1, 2), (3, 4, 5)], name="test_emb")
        data = emb.to_json()
        restored = StraightLineEmbedding.from_json(data)
        assert restored == emb
        assert restored.name == "test_emb"

    def test_to_json_is_json_serializable(self):
        """to_json produces a dict that survives json.dumps/loads."""
        emb = StraightLineEmbedding.from_positions([(0, 0), (1, 1)])
        data = emb.to_json()
        roundtrip = json.loads(json.dumps(data))
        restored = StraightLineEmbedding.from_json(roundtrip)
        assert restored == emb

    def test_from_json_rejects_wrong_type(self):
        with pytest.raises(ValueError, match="type='straight_line'"):
            StraightLineEmbedding.from_json(
                {"schema_version": 1, "type": "json_polyline", "positions": []}
            )

    def test_from_json_rejects_wrong_schema_version(self):
        with pytest.raises(ValueError, match="schema_version"):
            StraightLineEmbedding.from_json(
                {"schema_version": 999, "type": "straight_line", "positions": []}
            )


# =============================================================================
# JsonPolylineEmbedding
# =============================================================================


class TestJsonPolylineEmbedding:
    def _make_simple(self) -> JsonPolylineEmbedding:
        """A small embedding with one multi-segment edge."""
        return JsonPolylineEmbedding(
            positions={
                0: (0.0, 0.0, 0.0),
                1: (5.0, 0.0, 0.0),
                2: (0.0, 5.0, 0.0),
            },
            edge_polylines={
                (0, 1): ((0.0, 0.0, 0.0), (2.5, 0.0, 1.0), (5.0, 0.0, 0.0)),
                (0, 2): ((0.0, 0.0, 0.0), (0.0, 5.0, 0.0)),
            },
            name="simple",
        )

    def test_schema_version(self):
        assert JsonPolylineEmbedding.SCHEMA_VERSION == 1
        assert self._make_simple().schema_version == 1

    def test_surface_name(self):
        assert self._make_simple().surface_name == "json_polyline"

    def test_node_position_lookup(self):
        emb = self._make_simple()
        assert emb.node_position(0) == (0.0, 0.0, 0.0)
        assert emb.node_position(1) == (5.0, 0.0, 0.0)
        assert emb.node_position(2) == (0.0, 5.0, 0.0)

    def test_node_position_missing_raises(self):
        emb = self._make_simple()
        with pytest.raises(IndexError, match="No position defined"):
            emb.node_position(99)

    def test_routing_geometry_returns_stored_polylines(self):
        emb = self._make_simple()
        rg = emb.routing_geometry([(0, 1)])
        assert rg[(0, 1)] == (
            (0.0, 0.0, 0.0),
            (2.5, 0.0, 1.0),
            (5.0, 0.0, 0.0),
        )

    def test_routing_geometry_multiple_edges(self):
        emb = self._make_simple()
        rg = emb.routing_geometry([(0, 1), (0, 2)])
        assert len(rg) == 2
        assert (0, 1) in rg
        assert (0, 2) in rg

    def test_routing_geometry_unknown_edge_raises(self):
        emb = self._make_simple()
        with pytest.raises(KeyError, match="No polyline defined"):
            emb.routing_geometry([(1, 2)])

    def test_rejects_single_point_polyline(self):
        with pytest.raises(ValueError, match="at least 2 points"):
            JsonPolylineEmbedding(
                positions={0: (0.0, 0.0, 0.0), 1: (1.0, 0.0, 0.0)},
                edge_polylines={(0, 1): ((0.0, 0.0, 0.0),)},
            )

    def test_to_from_json_roundtrip(self):
        emb = self._make_simple()
        data = emb.to_json()
        restored = JsonPolylineEmbedding.from_json(data)
        assert restored.positions == emb.positions
        assert restored.edge_polylines == emb.edge_polylines
        assert restored.name == emb.name

    def test_from_file(self, tmp_path):
        """Loading from disk reproduces the in-memory embedding."""
        emb = self._make_simple()
        path = tmp_path / "embedding.json"
        path.write_text(json.dumps(emb.to_json()))
        loaded = JsonPolylineEmbedding.from_file(path)
        assert loaded.positions == emb.positions
        assert loaded.edge_polylines == emb.edge_polylines

    def test_from_file_string_path(self, tmp_path):
        """from_file accepts str paths as well as Path objects."""
        emb = self._make_simple()
        path = tmp_path / "embedding.json"
        path.write_text(json.dumps(emb.to_json()))
        loaded = JsonPolylineEmbedding.from_file(str(path))
        assert loaded.positions == emb.positions

    def test_from_json_rejects_wrong_type(self):
        with pytest.raises(ValueError, match="type='json_polyline'"):
            JsonPolylineEmbedding.from_json(
                {"schema_version": 1, "type": "straight_line", "positions": {}, "edges": []}
            )

    def test_from_json_rejects_wrong_schema_version(self):
        with pytest.raises(ValueError, match="schema_version"):
            JsonPolylineEmbedding.from_json(
                {"schema_version": 999, "type": "json_polyline", "positions": {}, "edges": []}
            )

    def test_from_json_canonicalizes_int_keys(self):
        """JSON serializes dict keys as strings; from_json must coerce back to int."""
        emb = self._make_simple()
        data = emb.to_json()
        # The positions dict in data has str keys after to_json.
        assert all(isinstance(k, str) for k in data["positions"])
        restored = JsonPolylineEmbedding.from_json(data)
        assert all(isinstance(k, int) for k in restored.positions)


# =============================================================================
# Embedding protocol satisfaction
# =============================================================================


class TestEmbeddingProtocol:
    def test_straight_line_satisfies_protocol(self):
        emb = StraightLineEmbedding.from_positions([(0, 0), (1, 1)])
        assert isinstance(emb, Embedding)

    def test_json_polyline_satisfies_protocol(self):
        emb = JsonPolylineEmbedding(
            positions={0: (0.0, 0.0, 0.0), 1: (1.0, 0.0, 0.0)},
            edge_polylines={(0, 1): ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))},
        )
        assert isinstance(emb, Embedding)


# =============================================================================
# load_embedding dispatch
# =============================================================================


class TestLoadEmbedding:
    def test_dispatches_to_straight_line(self):
        emb = StraightLineEmbedding.from_positions([(0, 0), (1, 1)])
        loaded = load_embedding(emb.to_json())
        assert isinstance(loaded, StraightLineEmbedding)
        assert loaded == emb

    def test_dispatches_to_json_polyline(self):
        emb = JsonPolylineEmbedding(
            positions={0: (0.0, 0.0, 0.0), 1: (1.0, 0.0, 0.0)},
            edge_polylines={(0, 1): ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))},
        )
        loaded = load_embedding(emb.to_json())
        assert isinstance(loaded, JsonPolylineEmbedding)

    def test_rejects_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown embedding type"):
            load_embedding({"type": "mystery_embedding", "schema_version": 1})

    def test_rejects_missing_type(self):
        with pytest.raises(ValueError, match="Unknown embedding type"):
            load_embedding({"schema_version": 1})


# =============================================================================
# Acceptance test (plan §2 PR 2)
# =============================================================================


class TestSteaneAcceptance:
    """The PR 2 acceptance tests from `private/plan.md`:

    1. A `StraightLineEmbedding` built from a Steane code's positions,
       round-tripped through `to_json` / `from_json`, produces bit-for-bit
       identical polylines.
    2. `JsonPolylineEmbedding` loaded from a canned JSON file produces a
       `RoutingGeometry` with the exact polyline coordinates from the file.
    """

    def _steane_edges(self) -> list[IREdge]:
        """All 24 Tanner-graph edges in a Steane code (data → check qubits)."""
        H = pcm.hamming(7)
        num_data = 7
        edges: list[IREdge] = []
        # Z-checks: data qubits at indices 7..9 (Steane labels HZ row k as qubit 7+k)
        for k in range(H.shape[0]):
            check_qubit = num_data + k
            for i in range(H.shape[1]):
                if H[k, i]:
                    edges.append((i, check_qubit))
        # X-checks: data qubits at indices 10..12
        for k in range(H.shape[0]):
            check_qubit = num_data + H.shape[0] + k
            for i in range(H.shape[1]):
                if H[k, i]:
                    edges.append((i, check_qubit))
        return edges

    def test_steane_straight_line_roundtrip_bit_for_bit(self):
        """PR 2 Acceptance Test 1: Steane StraightLineEmbedding JSON round-trip."""
        H = pcm.hamming(7)
        code = CSSCode(HX=H, HZ=H, rounds=1)
        code.embed("spring", seed=42)
        assert code.pos is not None

        # Wrap the code's 2D positions in a StraightLineEmbedding.
        original = StraightLineEmbedding.from_positions(code.pos, name="steane_spring")

        # Round-trip through JSON.
        data = original.to_json()
        serialized = json.dumps(data)
        restored = StraightLineEmbedding.from_json(json.loads(serialized))

        # Dataclass equality already checks bit-for-bit.
        assert restored == original

        # And the routing_geometry output is bit-for-bit identical on every
        # one of Steane's 24 Tanner-graph edges.
        edges = self._steane_edges()
        assert len(edges) == 24
        rg_original = original.routing_geometry(edges)
        rg_restored = restored.routing_geometry(edges)
        for e in edges:
            assert rg_original[e] == rg_restored[e]

    def test_json_polyline_from_canned_file_exact_coordinates(self, tmp_path):
        """PR 2 Acceptance Test 2: loaded polylines exactly match the file."""
        # A canned JSON with precisely specified floats.
        canned = {
            "schema_version": 1,
            "type": "json_polyline",
            "name": "canned",
            "positions": {
                "0": [0.0, 0.0, 0.0],
                "1": [3.0, 4.0, 0.0],
                "2": [6.0, 0.0, 0.0],
            },
            "edges": [
                {
                    "source": 0,
                    "target": 1,
                    "polyline": [[0.0, 0.0, 0.0], [1.5, 2.0, 0.5], [3.0, 4.0, 0.0]],
                },
                {
                    "source": 1,
                    "target": 2,
                    "polyline": [[3.0, 4.0, 0.0], [6.0, 0.0, 0.0]],
                },
            ],
        }
        path = tmp_path / "canned.json"
        path.write_text(json.dumps(canned))

        emb = JsonPolylineEmbedding.from_file(path)
        rg = emb.routing_geometry([(0, 1), (1, 2)])

        assert rg[(0, 1)] == (
            (0.0, 0.0, 0.0),
            (1.5, 2.0, 0.5),
            (3.0, 4.0, 0.0),
        )
        assert rg[(1, 2)] == (
            (3.0, 4.0, 0.0),
            (6.0, 0.0, 0.0),
        )


# =============================================================================
# PR 4 backward-compat: RouteID upgrade
# =============================================================================


class TestRouteIDBackwardCompat:
    """Verify the PR 4 `RouteID` upgrade preserves PR 2's tuple-keyed API."""

    def test_routing_geometry_stores_route_ids_internally(self):
        """Tuple input is auto-lifted to RouteID in __post_init__."""
        rg = RoutingGeometry(edges={(0, 1): ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))})
        # After lift, the dict is keyed by RouteID.
        assert list(rg.edges.keys())[0] == RouteID(source=0, target=1)

    def test_routing_geometry_accepts_route_id_input(self):
        """Native RouteID input flows through without lifting."""
        rid = RouteID(source=0, target=1, step_tick=3, term_name="B1")
        rg = RoutingGeometry(edges={rid: ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))})
        # The RouteID is preserved with its metadata.
        stored = list(rg.edges.keys())[0]
        assert stored == rid
        assert stored.step_tick == 3
        assert stored.term_name == "B1"

    def test_routing_geometry_tuple_contains_works(self):
        rg = RoutingGeometry(edges={(0, 1): ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))})
        assert (0, 1) in rg
        assert (0, 2) not in rg

    def test_routing_geometry_route_id_contains_works(self):
        rg = RoutingGeometry(
            edges={RouteID(source=0, target=1): ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))}
        )
        assert RouteID(source=0, target=1) in rg
        assert RouteID(source=0, target=1, step_tick=5) not in rg

    def test_routing_geometry_tuple_getitem_works(self):
        rg = RoutingGeometry(edges={(0, 1): ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))})
        assert rg[(0, 1)] == ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))

    def test_routing_geometry_route_id_getitem_works(self):
        rid = RouteID(source=0, target=1, step_tick=3)
        rg = RoutingGeometry(edges={rid: ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))})
        assert rg[rid] == ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))

    def test_routing_geometry_rejects_bad_key_type(self):
        with pytest.raises(TypeError, match="must be RouteID"):
            RoutingGeometry(
                edges={"bad": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))}  # type: ignore[dict-item]
            )

    def test_straight_line_accepts_tuple_input(self):
        """PR 4 acceptance test 5a: StraightLineEmbedding accepts tuple input."""
        emb = StraightLineEmbedding.from_positions([(0, 0), (1, 0), (2, 0), (3, 0)])
        rg = emb.routing_geometry([(0, 1), (2, 3)])
        assert len(rg) == 2
        # Result is keyed by RouteID with default metadata.
        assert RouteID(source=0, target=1) in rg.edges
        assert RouteID(source=2, target=3) in rg.edges

    def test_straight_line_accepts_route_id_input(self):
        """PR 4 acceptance test 5b: StraightLineEmbedding accepts RouteID input."""
        emb = StraightLineEmbedding.from_positions([(0, 0), (1, 0), (2, 0), (3, 0)])
        rg = emb.routing_geometry([RouteID(source=0, target=1), RouteID(source=2, target=3)])
        assert len(rg) == 2

    def test_straight_line_tuple_and_route_id_equivalent(self):
        """PR 4 acceptance test 5c: tuple and RouteID inputs give equivalent geometries."""
        emb = StraightLineEmbedding.from_positions([(0, 0), (1, 0), (2, 0), (3, 0)])
        rg_tuple = emb.routing_geometry([(0, 1), (2, 3)])
        rg_route = emb.routing_geometry([RouteID(source=0, target=1), RouteID(source=2, target=3)])
        # Both should have the same set of RouteID keys (since tuples lift to
        # default-metadata RouteID).
        assert set(rg_tuple.edges.keys()) == set(rg_route.edges.keys())
        # And the polylines are identical.
        for rid in rg_tuple.edges:
            assert rg_tuple.edges[rid] == rg_route.edges[rid]

    def test_json_polyline_accepts_route_id_storage(self):
        """JsonPolylineEmbedding accepts RouteID keys in edge_polylines."""
        emb = JsonPolylineEmbedding(
            positions={0: (0.0, 0.0, 0.0), 1: (1.0, 0.0, 0.0)},
            edge_polylines={
                RouteID(source=0, target=1, step_tick=3, term_name="B1"): (
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                ),
            },
        )
        rid = RouteID(source=0, target=1, step_tick=3, term_name="B1")
        rg = emb.routing_geometry([rid])
        assert rid in rg.edges

    def test_json_polyline_roundtrip_preserves_route_id_metadata(self):
        """JsonPolylineEmbedding JSON round-trip preserves step_tick / term_name / instance."""
        emb = JsonPolylineEmbedding(
            positions={0: (0.0, 0.0, 0.0), 1: (1.0, 0.0, 0.0)},
            edge_polylines={
                RouteID(source=0, target=1, step_tick=3, term_name="B1", instance=2): (
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                ),
            },
        )
        restored = JsonPolylineEmbedding.from_json(emb.to_json())
        stored = list(restored.edge_polylines.keys())[0]
        assert stored.source == 0
        assert stored.target == 1
        assert stored.step_tick == 3
        assert stored.term_name == "B1"
        assert stored.instance == 2

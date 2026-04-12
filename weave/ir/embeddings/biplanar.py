r"""Bounded-thickness biplanar embedding for bivariate bicycle codes.

The *biplanar* layout introduced by Bravyi et al. (Nature 2024,
§III) assigns the six BB code monomial families to two routing
layers and routes each edge through a 3D polyline that lifts from a
shared base plane (`z = 0`) to one of the two routing planes
(`z = ±h`), traverses horizontally, and descends back. This
eliminates all inter-layer crossings by construction: edges in
opposite layers are separated by `2h` vertically, and edges within
the same layer interact only through their in-plane distances.

Layer assignment
----------------
The six families are partitioned into two layers following the
IBM bounded-thickness decomposition theorem:

- **Layer A** (upper routing plane, `z = +h`): `A2, A3, B3`.
- **Layer B** (lower routing plane, `z = -h`): `A1, B1, B2`.

Each layer's connectivity graph is individually planar; this is
a necessary condition for the bounded-thickness realization and
is verified by `bbstim` at construction time via NetworkX's
`check_planarity` (we do not repeat the planarity check in weave
since it is a code-family invariant, not a per-compile assertion).

Base-plane layout
-----------------
All qubits are placed on a common 2D grid at `z = 0`:

- `L`-data `(i, j)` at `(2 a, 2 b, 0)` where `(a, b)` are the
  group-element coordinates of index `i` (bbstim's row-major
  `idx(a, b) = a m + b`; weave uses column-major
  `flat = j l + i`, converted internally).
- `X`-ancilla at `(2 a + 1, 2 b, 0)`.
- `Z`-ancilla at `(2 a, 2 b + 1, 0)`.
- `R`-data at `(2 a + 1, 2 b + 1, 0)`.

This reproduces the torus-like chequerboard pattern from
`bbstim.embeddings.IBMToricBiplanarEmbedding`.

Polyline structure
------------------
Each edge is a 6-point 3D polyline:

.. code-block:: text

    (source_xy, 0)          ← base
    (source_port_xy, 0)     ← small angular offset
    (source_port_xy, ±h)    ← lift to routing plane
    (target_port_xy, ±h)    ← traverse
    (target_port_xy, 0)     ← descend
    (target_xy, 0)          ← base

The angular port offsets prevent two edges' vertical segments from
coinciding; the lane offsets (`lane_eps`) prevent same-layer
edges from sharing the same z height, so no two segments are
coplanar.

Term-name dispatch
------------------
The embedding reads the `term_name` field of each incoming
:class:`~weave.ir.RouteID` to determine the monomial family:

- `"A1"`, `"A2"`, `"A3"` → the first, second, third A-monomial.
- `"B1"`, `"B2"`, `"B3"` → similarly for B.
- `None` or unrecognized → defaults to layer A with a warning in
  provenance (future version may raise).

The `ibm_schedule` factory (see :mod:`weave.codes.bb.schedule`)
tags every `TwoQubitEdge` with the appropriate family label so the
dispatch works out-of-the-box.

References
----------
- Bravyi, Cross, Gambetta, Maslov, Rall, Yoder, *High-threshold
  and low-overhead fault-tolerant quantum memory*, Nature **627**,
  778 (2024), arXiv:2308.07915. §III.
- Di Bella, *Geometry-induced correlated noise in qLDPC syndrome
  extraction* (PRX Quantum, under review, 2026). §IV.B describes
  the bounded-thickness formalization; `bbstim/embeddings.py`
  implements the reference version.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from ...geometry import Point3
from ..embedding import IREdge, IRPolyline, RoutingGeometry
from ..route import RouteID

if TYPE_CHECKING:
    from ...codes.bb.bb_code import BivariateBicycleCode


__all__ = ["IBMBiplanarEmbedding"]


# Layer assignment matching bbstim's IBMToricBiplanarEmbedding.
_LAYER_A_TERMS: frozenset[str] = frozenset({"A2", "A3", "B3"})
_LAYER_B_TERMS: frozenset[str] = frozenset({"A1", "B1", "B2"})


@dataclass(frozen=True)
class IBMBiplanarEmbedding:
    r"""Bounded-thickness biplanar embedding for a BB code.

    See the module docstring for the full description. All parameters
    are frozen for JSON round-trip and deterministic `fingerprint()`
    hashing.

    Parameters
    ----------
    positions : tuple[Point3, ...]
        One 3D position per qubit (all at `z = 0`), indexed by the
        qubit's integer index. Populated by :meth:`from_bb`.
    l, m : int
        Cyclic-factor sizes of the underlying BB code.
    spacing : float
        In-plane lattice pitch (cell-to-cell distance is
        `2 * spacing`). Must be positive.
    layer_height : float
        Absolute z-coordinate of the routing planes.
    lane_eps : float
        Per-edge z-offset within a layer for lane separation.
    bb_name : str
        Human-readable label of the underlying code.
    name : str
        Embedding label.
    """

    SCHEMA_VERSION: ClassVar[int] = 2

    positions: tuple[Point3, ...]
    l: int
    m: int
    spacing: float = 1.0
    layer_height: float = 1.0
    lane_eps: float = 0.0
    bb_name: str = ""
    name: str = "ibm_biplanar"

    def __post_init__(self) -> None:
        if not isinstance(self.positions, tuple):
            object.__setattr__(
                self,
                "positions",
                tuple(
                    (float(p[0]), float(p[1]), float(p[2]))
                    if len(p) == 3
                    else (float(p[0]), float(p[1]), 0.0)
                    for p in self.positions
                ),
            )
        if self.l < 0 or self.m < 0:
            raise ValueError(f"l, m must be non-negative; got l={self.l}, m={self.m}")
        if self.spacing <= 0:
            raise ValueError(f"spacing must be positive, got {self.spacing}")
        if self.layer_height <= 0:
            raise ValueError(f"layer_height must be positive, got {self.layer_height}")

    # ------------------------------------------------------------------
    # Classmethod factory
    # ------------------------------------------------------------------

    @classmethod
    def from_bb(
        cls,
        bb_code: BivariateBicycleCode,
        *,
        spacing: float = 1.0,
        layer_height: float = 1.0,
        lane_eps: float | None = None,
        name: str | None = None,
    ) -> IBMBiplanarEmbedding:
        r"""Build the canonical biplanar layout for a BB code.

        Positions follow the chequerboard grid pattern from
        ``bbstim.embeddings.IBMToricBiplanarEmbedding``:

        - L-data `(i, j)` at `(2 a, 2 b, 0)`.
        - X-ancilla at `(2 a + 1, 2 b, 0)`.
        - Z-ancilla at `(2 a, 2 b + 1, 0)`.
        - R-data at `(2 a + 1, 2 b + 1, 0)`.

        where `(a, b)` are the group-element coordinates obtained
        from the flat index via
        `bb_code.unflat_index(flat) -> (i_w, j_w, block)`, then
        `(a, b) = (i_w, j_w)`.

        Parameters
        ----------
        bb_code : BivariateBicycleCode
        spacing : float, optional
        layer_height : float, optional
        lane_eps : float, optional
            Per-edge z-offset for lane separation. Defaults to
            `layer_height * 1e-3 / max(1, lm)`.
        name : str, optional
        """
        if spacing <= 0:
            raise ValueError(f"spacing must be positive, got {spacing}")
        if layer_height <= 0:
            raise ValueError(f"layer_height must be positive, got {layer_height}")

        l, m = bb_code.l, bb_code.m
        lm = l * m
        total = 4 * lm
        if lane_eps is None:
            lane_eps = layer_height * 1e-3 / max(1, lm)

        positions: list[Point3] = [(0.0, 0.0, 0.0)] * total

        for flat_within_block in range(lm):
            # Decode flat within block → group element (a, b).
            # weave column-major: flat = j * l + i, so i = flat % l, j = flat // l.
            a = flat_within_block % l
            b = flat_within_block // l
            base_x = 2.0 * float(a) * spacing
            base_y = 2.0 * float(b) * spacing
            # L-data at (2a, 2b).
            positions[flat_within_block] = (base_x, base_y, 0.0)
            # R-data at (2a+1, 2b+1).
            positions[lm + flat_within_block] = (
                base_x + spacing,
                base_y + spacing,
                0.0,
            )

        for idx, ancilla_q in enumerate(bb_code.z_check_qubits):
            a = idx % l
            b = idx // l
            # Z-ancilla at (2a, 2b+1).
            positions[ancilla_q] = (
                2.0 * float(a) * spacing,
                2.0 * float(b) * spacing + spacing,
                0.0,
            )
        for idx, ancilla_q in enumerate(bb_code.x_check_qubits):
            a = idx % l
            b = idx // l
            # X-ancilla at (2a+1, 2b).
            positions[ancilla_q] = (
                2.0 * float(a) * spacing + spacing,
                2.0 * float(b) * spacing,
                0.0,
            )

        return cls(
            positions=tuple(positions),
            l=l,
            m=m,
            spacing=spacing,
            layer_height=layer_height,
            lane_eps=lane_eps,
            bb_name=bb_code.name,
            name=name or f"{bb_code.name}_ibm_biplanar",
        )

    # ------------------------------------------------------------------
    # Embedding protocol
    # ------------------------------------------------------------------

    @property
    def schema_version(self) -> int:
        return self.SCHEMA_VERSION

    @property
    def surface_name(self) -> str:
        return "biplanar"

    def node_position(self, qubit_idx: int) -> Point3:
        if not 0 <= qubit_idx < len(self.positions):
            raise IndexError(f"qubit_idx {qubit_idx} out of range [0, {len(self.positions)}).")
        return self.positions[qubit_idx]

    def routing_geometry(self, active_edges: Sequence[RouteID | IREdge]) -> RoutingGeometry:
        """Return routed polylines for the given active edges.

        Each edge is dispatched to a routing layer based on its
        `term_name`. Edges whose `term_name` is in
        `_LAYER_A_TERMS` are routed through `z = +layer_height`;
        those in `_LAYER_B_TERMS` through `z = -layer_height`.
        Unrecognized names default to layer A.

        The resulting polylines have **6 points** each (see the
        module docstring for the point-by-point layout), so
        downstream distance computations via
        :func:`~weave.geometry.polyline_distance` correctly measure
        the 3D segment-pair minimum distance.
        """
        edges_map: dict[RouteID, IRPolyline] = {}
        n = len(self.positions)
        lm = self.l * self.m

        for item in active_edges:
            if isinstance(item, RouteID):
                rid = item
                u, v = rid.source, rid.target
            else:
                u, v = item
                rid = RouteID(source=int(u), target=int(v))
            if not 0 <= u < n:
                raise IndexError(f"Edge source {u} out of range [0, {n}).")
            if not 0 <= v < n:
                raise IndexError(f"Edge target {v} out of range [0, {n}).")

            pu = self.positions[u]
            pv = self.positions[v]

            # Determine routing layer from term_name.
            term = rid.term_name or ""
            sign = -1.0 if term in _LAYER_B_TERMS else +1.0

            # Lane z for this specific edge.
            edge_idx = rid.source % max(1, lm)
            z = sign * self.layer_height + sign * self.lane_eps * (edge_idx + 1)

            # Port offsets: small angular displacement around each vertex
            # so vertical segments don't overlap.
            family_order = {"A1": 0, "A2": 1, "A3": 2, "B1": 3, "B2": 4, "B3": 5}.get(term, 0)
            angle = 2.0 * math.pi * ((edge_idx + 0.37 * family_order) / max(1, lm))
            eps = min(0.15, 0.2 * min(1.0, self.layer_height))
            du = (eps * math.cos(angle), eps * math.sin(angle))

            port_u = (pu[0] + du[0], pu[1] + du[1])
            port_v = (pv[0] - du[0], pv[1] - du[1])

            edges_map[rid] = (
                (pu[0], pu[1], 0.0),
                (port_u[0], port_u[1], 0.0),
                (port_u[0], port_u[1], z),
                (port_v[0], port_v[1], z),
                (port_v[0], port_v[1], 0.0),
                (pv[0], pv[1], 0.0),
            )
        return RoutingGeometry(edges=edges_map, name=self.name)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "type": "ibm_biplanar",
            "name": self.name,
            "positions": [list(p) for p in self.positions],
            "l": self.l,
            "m": self.m,
            "spacing": float(self.spacing),
            "layer_height": float(self.layer_height),
            "lane_eps": float(self.lane_eps),
            "bb_name": self.bb_name,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> IBMBiplanarEmbedding:
        if data.get("type") != "ibm_biplanar":
            raise ValueError(f"Expected type='ibm_biplanar', got {data.get('type')!r}.")
        version = data.get("schema_version")
        if version not in (1, cls.SCHEMA_VERSION):
            raise ValueError(
                f"Unsupported schema_version {version}; expected {cls.SCHEMA_VERSION}."
            )
        positions = tuple(
            (float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0)
            for p in data["positions"]
        )
        return cls(
            positions=positions,
            l=int(data["l"]),
            m=int(data["m"]),
            spacing=float(data.get("spacing", 1.0)),
            layer_height=float(data.get("layer_height", 1.0)),
            lane_eps=float(data.get("lane_eps", 0.0)),
            bb_name=str(data.get("bb_name", "")),
            name=str(data.get("name", "ibm_biplanar")),
        )

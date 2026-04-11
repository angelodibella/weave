r"""Column embeddings for CSS codes on regular grid surfaces.

A *column embedding* places qubits on an integer grid whose points
are labelled by `(column, row)` pairs. It is the natural layout for
any CSS code whose qubits (data and ancillas) can be organised into
a bipartite column structure — most importantly the bivariate
bicycle (BB) codes of Bravyi et al. (Nature 2024), whose four qubit
classes (`L`-data, Z-ancilla, X-ancilla, `R`-data) fit into a
`(4 l) \times m` grid indexed by the `\mathbb{Z}_l \times \mathbb{Z}_m`
group action.

This module ships two concrete classes:

- :class:`ColumnEmbedding` — the general column-grid embedding. Holds
  a tuple of 3D positions (one per qubit index) and implements the
  :class:`~weave.ir.Embedding` protocol with straight-line
  `routing_geometry` and JSON round-trip. Suitable for any code
  whose layout is describable as a grid of columns.
- :class:`MonomialColumnEmbedding` — a :class:`ColumnEmbedding`
  specialization carrying BB-specific metadata (`l`, `m`, the
  canonical "four layers per column" lattice). Its
  :meth:`~MonomialColumnEmbedding.from_bb` factory produces a
  mathematically canonical layout in which each `\mathbb{Z}_l \times
  \mathbb{Z}_m` orbit is laid out as one (i, j)-cell containing the
  four qubits of that cell in a fixed layer order.

The factories take a spacing parameter and an optional `name` so
downstream benchmarks can carry layout provenance through the
compiled output's `embedding_spec` field.

References
----------
- Bravyi, Cross, Gambetta, Maslov, Rall, Yoder, *High-threshold and
  low-overhead fault-tolerant quantum memory*, Nature **627**, 778
  (2024), arXiv:2308.07915. §III describes the two-plane layout and
  the "long" (column) variant used as a contrast to the biplanar
  short-range connectivity.
- Di Bella, *Geometry-induced correlated noise in qLDPC syndrome
  extraction* (PRX Quantum, under review, 2026). §IV.A uses the
  monomial column embedding as the baseline for the `J_\kappa`
  optimizer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from ...geometry import Point3
from ..embedding import IREdge, IRPolyline, RoutingGeometry
from ..route import RouteID

if TYPE_CHECKING:
    from ...codes.bb.bb_code import BivariateBicycleCode


__all__ = [
    "ColumnEmbedding",
    "MonomialColumnEmbedding",
]


# =============================================================================
# Internal helpers
# =============================================================================


def _to_point3(p: Sequence[float]) -> Point3:
    """Lift a 2D or 3D numeric sequence into a :data:`Point3`."""
    if len(p) == 2:
        return (float(p[0]), float(p[1]), 0.0)
    if len(p) == 3:
        return (float(p[0]), float(p[1]), float(p[2]))
    raise ValueError(f"Expected a 2D or 3D point, got length {len(p)}.")


def _straight_line_routing(
    positions: tuple[Point3, ...],
    active_edges: Sequence[RouteID | IREdge],
    *,
    name: str,
) -> RoutingGeometry:
    """Shared straight-line routing body for all column-style embeddings.

    Every edge is a two-point polyline between its endpoint positions.
    Factored out so :class:`ColumnEmbedding`, :class:`IBMBiplanarEmbedding`,
    and :class:`FixedPermutationColumnEmbedding` can share one
    implementation without inheritance gymnastics.
    """
    edges_map: dict[RouteID, IRPolyline] = {}
    n = len(positions)
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
        edges_map[rid] = (positions[u], positions[v])
    return RoutingGeometry(edges=edges_map, name=name)


# =============================================================================
# ColumnEmbedding — generic grid layout
# =============================================================================


@dataclass(frozen=True)
class ColumnEmbedding:
    r"""Embedding of a CSS code on a regular column grid.

    A column embedding places each qubit at an explicit 3D position
    and routes every edge as a straight polyline between its two
    endpoints (matching :class:`~weave.ir.StraightLineEmbedding`'s
    default routing).

    The class adds three pieces of metadata that generic
    straight-line embeddings lack: the number of columns and rows of
    the logical grid, and the number of "layers" per column (how
    many qubits sit in a single `(column, row)` cell). Downstream
    consumers read these to produce diagrams, to serialize
    benchmark-quality layout provenance, and to drive the monomial
    routing in :class:`MonomialColumnEmbedding`.

    Parameters
    ----------
    positions : tuple[Point3, ...]
        One 3D position per qubit, indexed by the qubit's integer
        index. Coerced to a tuple of triples in `__post_init__`.
    num_columns : int
        Number of columns in the logical grid (`l` for BB codes).
        Used for diagrams and for :class:`MonomialColumnEmbedding`'s
        position arithmetic.
    num_rows : int
        Number of rows in the logical grid (`m` for BB codes).
    layers_per_cell : int
        Number of qubits that share a `(column, row)` cell. For BB
        codes this is 4 (`L`-data, Z-ancilla, X-ancilla, `R`-data).
    name : str, optional
        Human-readable label for provenance.
    """

    SCHEMA_VERSION: ClassVar[int] = 1

    positions: tuple[Point3, ...]
    num_columns: int = 0
    num_rows: int = 0
    layers_per_cell: int = 1
    name: str = "column"

    def __post_init__(self) -> None:
        # Coerce list inputs to tuples of Point3 for hashability.
        if not isinstance(self.positions, tuple):
            object.__setattr__(self, "positions", tuple(_to_point3(p) for p in self.positions))
        else:
            coerced: list[Point3] = []
            needs_coerce = False
            for p in self.positions:
                triple = _to_point3(p)
                if triple != p:
                    needs_coerce = True
                coerced.append(triple)
            if needs_coerce:
                object.__setattr__(self, "positions", tuple(coerced))

        # Validate metadata consistency if a grid is declared.
        if self.num_columns < 0 or self.num_rows < 0 or self.layers_per_cell < 0:
            raise ValueError("num_columns, num_rows, and layers_per_cell must be non-negative")
        declared = self.num_columns * self.num_rows * self.layers_per_cell
        if declared and declared != len(self.positions):
            raise ValueError(
                f"num_columns * num_rows * layers_per_cell = {declared} "
                f"does not match len(positions) = {len(self.positions)}"
            )

    # ------------------------------------------------------------------
    # Classmethod factory
    # ------------------------------------------------------------------

    @classmethod
    def from_positions(
        cls,
        positions: Sequence[Sequence[float]],
        *,
        num_columns: int = 0,
        num_rows: int = 0,
        layers_per_cell: int = 1,
        name: str = "column",
    ) -> ColumnEmbedding:
        """Build a :class:`ColumnEmbedding` from an iterable of 2D/3D points.

        2D inputs are lifted to `z = 0`. The grid metadata parameters
        are optional; they default to zero (meaning "no declared
        grid"), in which case the `positions` tuple is accepted
        as-is. If any of them is nonzero, the product
        `num_columns * num_rows * layers_per_cell` must equal
        `len(positions)`.
        """
        pts = tuple(_to_point3(p) for p in positions)
        return cls(
            positions=pts,
            num_columns=num_columns,
            num_rows=num_rows,
            layers_per_cell=layers_per_cell,
            name=name,
        )

    # ------------------------------------------------------------------
    # Embedding protocol
    # ------------------------------------------------------------------

    @property
    def schema_version(self) -> int:
        return self.SCHEMA_VERSION

    @property
    def surface_name(self) -> str:
        return "plane"

    def node_position(self, qubit_idx: int) -> Point3:
        if not 0 <= qubit_idx < len(self.positions):
            raise IndexError(f"qubit_idx {qubit_idx} out of range [0, {len(self.positions)}).")
        return self.positions[qubit_idx]

    def routing_geometry(self, active_edges: Sequence[RouteID | IREdge]) -> RoutingGeometry:
        return _straight_line_routing(self.positions, active_edges, name=self.name)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "type": "column",
            "name": self.name,
            "positions": [list(p) for p in self.positions],
            "num_columns": self.num_columns,
            "num_rows": self.num_rows,
            "layers_per_cell": self.layers_per_cell,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ColumnEmbedding:
        if data.get("type") != "column":
            raise ValueError(f"Expected type='column', got {data.get('type')!r}.")
        version = data.get("schema_version")
        if version != cls.SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version {version}; expected {cls.SCHEMA_VERSION}."
            )
        positions = tuple(_to_point3(p) for p in data["positions"])
        return cls(
            positions=positions,
            num_columns=int(data.get("num_columns", 0)),
            num_rows=int(data.get("num_rows", 0)),
            layers_per_cell=int(data.get("layers_per_cell", 1)),
            name=data.get("name", "column"),
        )


# =============================================================================
# MonomialColumnEmbedding — BB-specific specialization
# =============================================================================


@dataclass(frozen=True)
class MonomialColumnEmbedding(ColumnEmbedding):
    r"""Canonical BB-code column embedding.

    Lays out a bivariate bicycle code on a regular
    `(4 l) \times m` grid. The four qubit classes occupy four
    parallel sub-columns per `(i, j) \in \mathbb{Z}_l \times \mathbb{Z}_m`
    cell, in the fixed order

    .. code-block:: text

        L-data    Z-ancilla    X-ancilla    R-data
          ↑          ↑            ↑           ↑
          0          1            2           3     (sub-column offset)

    so that for every cell `(i, j)` the four qubits sit at

    - `L(i, j)` at position `(4 i + 0, j, 0) \cdot \mathrm{spacing}`
    - `z(i, j)` at position `(4 i + 1, j, 0) \cdot \mathrm{spacing}`
    - `x(i, j)` at position `(4 i + 2, j, 0) \cdot \mathrm{spacing}`
    - `R(i, j)` at position `(4 i + 3, j, 0) \cdot \mathrm{spacing}`

    The resulting layout makes every Z-check CNOT (data → Z-ancilla)
    a nearest-column segment when the monomial is the identity, and
    a longer straight line when it is a nontrivial group shift. This
    is the *canonical "monomial" routing* used as a baseline in the
    BB exposure optimizer and as a worst-case contrast against the
    biplanar layout of :class:`~weave.ir.IBMBiplanarEmbedding`.

    The class inherits from :class:`ColumnEmbedding` and adds only
    the BB metadata (`l`, `m`, `spacing`). Every Embedding protocol
    method is inherited unchanged; only :meth:`to_json` /
    :meth:`from_json` are overridden to carry the BB fields.

    Parameters
    ----------
    positions : tuple[Point3, ...]
    l : int
        First cyclic factor size.
    m : int
        Second cyclic factor size.
    spacing : float
        Common lattice pitch. Defaults to 1.0. Not used by the
        :meth:`from_bb` factory once the positions have been
        materialized; kept in the class so JSON round-trips preserve it.
    bb_name : str
        Human-readable label of the underlying BB code, e.g.
        ``"BB72"``. Used for provenance on the compiled output.
    """

    SCHEMA_VERSION: ClassVar[int] = 1

    l: int = 0
    m: int = 0
    spacing: float = 1.0
    bb_name: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.l < 0 or self.m < 0 or self.spacing <= 0:
            raise ValueError(
                f"l, m must be non-negative and spacing positive; "
                f"got l={self.l}, m={self.m}, spacing={self.spacing}"
            )
        if self.l and self.m:
            expected_n = 4 * self.l * self.m
            if len(self.positions) != expected_n:
                raise ValueError(
                    f"MonomialColumnEmbedding expects 4 * l * m = {expected_n} "
                    f"qubits, got {len(self.positions)}"
                )

    @classmethod
    def from_bb(
        cls,
        bb_code: BivariateBicycleCode,
        *,
        spacing: float = 1.0,
        name: str | None = None,
    ) -> MonomialColumnEmbedding:
        r"""Build the canonical monomial-column layout for a BB code.

        Every data qubit in block `B \in \{L, R\}` and every ancilla
        in class `T \in \{\mathrm{Z}, \mathrm{X}\}` is placed at the
        sub-column determined by `B` or `T`, with cell coordinates
        `(i, j)` matching the `\mathbb{Z}_l \times \mathbb{Z}_m`
        group index of the qubit. The resulting positions are
        deterministic and satisfy the qubit-index ordering used by
        :class:`~weave.codes.bb.BivariateBicycleCode`:
        L-data at `[0, lm)`, R-data at `[lm, 2lm)`,
        Z-ancilla at `[2lm, 3lm)`, X-ancilla at `[3lm, 4lm)`.

        Parameters
        ----------
        bb_code : BivariateBicycleCode
            The code to lay out. The embedding only reads
            `bb_code.l`, `bb_code.m`, and the data/ancilla index
            ranges; it does not consult `HX`/`HZ`.
        spacing : float, optional
            Common lattice pitch. Defaults to 1.0.
        name : str, optional
            Human-readable label. Defaults to
            `"<bb_name>_monomial_column"`.

        Returns
        -------
        MonomialColumnEmbedding
        """
        if spacing <= 0:
            raise ValueError(f"spacing must be positive, got {spacing}")

        l, m = bb_code.l, bb_code.m
        total = 4 * l * m
        positions: list[Point3] = [(0.0, 0.0, 0.0)] * total

        # Data qubits: L at sub-column 0, R at sub-column 3.
        for j in range(m):
            for i in range(l):
                cell_x = float(4 * i) * spacing
                cell_y = float(j) * spacing
                # BB column-major index: flat = j * l + i
                flat_within_block = j * l + i
                positions[flat_within_block] = (cell_x + 0.0 * spacing, cell_y, 0.0)
                positions[l * m + flat_within_block] = (cell_x + 3.0 * spacing, cell_y, 0.0)

        # Ancillas: Z at sub-column 1, X at sub-column 2.
        for idx, ancilla_q in enumerate(bb_code.z_check_qubits):
            j, i = divmod(idx, l)
            cell_x = float(4 * i) * spacing
            cell_y = float(j) * spacing
            positions[ancilla_q] = (cell_x + 1.0 * spacing, cell_y, 0.0)
        for idx, ancilla_q in enumerate(bb_code.x_check_qubits):
            j, i = divmod(idx, l)
            cell_x = float(4 * i) * spacing
            cell_y = float(j) * spacing
            positions[ancilla_q] = (cell_x + 2.0 * spacing, cell_y, 0.0)

        return cls(
            positions=tuple(positions),
            num_columns=4 * l,
            num_rows=m,
            layers_per_cell=1,
            l=l,
            m=m,
            spacing=spacing,
            bb_name=bb_code.name,
            name=name or f"{bb_code.name}_monomial_column",
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        base = super().to_json()
        base["type"] = "monomial_column"
        base["l"] = self.l
        base["m"] = self.m
        base["spacing"] = float(self.spacing)
        base["bb_name"] = self.bb_name
        return base

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> MonomialColumnEmbedding:
        if data.get("type") != "monomial_column":
            raise ValueError(f"Expected type='monomial_column', got {data.get('type')!r}.")
        version = data.get("schema_version")
        if version != cls.SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version {version}; expected {cls.SCHEMA_VERSION}."
            )
        positions = tuple(_to_point3(p) for p in data["positions"])
        return cls(
            positions=positions,
            num_columns=int(data.get("num_columns", 0)),
            num_rows=int(data.get("num_rows", 0)),
            layers_per_cell=int(data.get("layers_per_cell", 1)),
            l=int(data.get("l", 0)),
            m=int(data.get("m", 0)),
            spacing=float(data.get("spacing", 1.0)),
            bb_name=str(data.get("bb_name", "")),
            name=data.get("name", "monomial_column"),
        )


# Silence the unused-import warning for `field`, which is reserved for
# future ColumnEmbedding subclasses that will store optional metadata
# dicts. Keeping the import stable avoids churn in consumer code.
_ = field

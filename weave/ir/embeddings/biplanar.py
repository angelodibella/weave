r"""Biplanar embedding for bivariate bicycle codes.

The *biplanar* layout is the two-plane stacking introduced by
Bravyi et al. (Nature 2024, §III) for BB codes. The `n = 2 l m` data
qubits split into an `L`-block and an `R`-block; the two blocks
occupy two parallel planes at `z = +h` and `z = -h` respectively,
with the `z = 0` mid-plane reserved for the `2 l m` ancillas. Every
Z-check and X-check polyline crosses the mid-plane exactly once,
so `L`-block and `R`-block edges are topologically distinguished
by their sign-of-`z` excursion — a fact the exposure optimizer
relies on and the PR 11 acceptance test pins.

Layout convention
-----------------
For a BB code with parameters `(l, m)`:

- `L`-data qubit at cell `(i, j)` occupies position
  `(i, j, +\text{layer\_height}) \cdot \text{spacing}`.
- `R`-data qubit at cell `(i, j)` occupies position
  `(i, j, -\text{layer\_height}) \cdot \text{spacing}`.
- Z-ancilla at cell `(i, j)` occupies position
  `(i + 0.5, j, 0) \cdot \text{spacing}` (shifted to sit between
  four surrounding data qubits).
- X-ancilla at cell `(i, j)` occupies position
  `(i, j + 0.5, 0) \cdot \text{spacing}`.

This places every ancilla on the mid-plane `z = 0`, with Z-ancillas
on a horizontal half-lattice and X-ancillas on a vertical half-lattice
so the two never collide. The polyline of an `L`-block data → ancilla
edge has an average `z > 0` (one endpoint is the `L` data at `+h`,
the other is the ancilla at `0`), and symmetrically for `R`-block
edges.

References
----------
- Bravyi, Cross, Gambetta, Maslov, Rall, Yoder, *High-threshold and
  low-overhead fault-tolerant quantum memory*, Nature **627**, 778
  (2024), arXiv:2308.07915. §III describes the two-plane layout.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from ...geometry import Point3
from ..embedding import IREdge, RoutingGeometry
from ..route import RouteID
from .column import _straight_line_routing, _to_point3

if TYPE_CHECKING:
    from ...codes.bb.bb_code import BivariateBicycleCode


__all__ = ["IBMBiplanarEmbedding"]


@dataclass(frozen=True)
class IBMBiplanarEmbedding:
    r"""Two-plane embedding for a BB code.

    Parameters
    ----------
    positions : tuple[Point3, ...]
        One 3D position per qubit, indexed by qubit index. Obtained
        from :meth:`from_bb` or supplied manually for custom layouts.
    l, m : int
        Cyclic-factor sizes of the underlying BB code. Used for grid
        metadata and JSON round-trip.
    spacing : float
        Common lattice pitch in the `xy` plane. Must be positive.
    layer_height : float
        Absolute `z`-coordinate of the `L`-block plane (`+layer_height`)
        and `R`-block plane (`-layer_height`). Ancillas sit at `z = 0`.
    bb_name : str
        Human-readable label of the underlying code, e.g. ``"BB72"``.
    name : str, optional
        Embedding label (used for provenance).

    Notes
    -----
    This class implements the :class:`~weave.ir.Embedding` protocol
    directly. It is not a subclass of :class:`ColumnEmbedding`
    because its natural grid is not flat: the two data planes carry
    non-zero `z` coordinates, so there is no single "column index"
    that captures both L and R data.
    """

    SCHEMA_VERSION: ClassVar[int] = 1

    positions: tuple[Point3, ...]
    l: int
    m: int
    spacing: float = 1.0
    layer_height: float = 1.0
    bb_name: str = ""
    name: str = "ibm_biplanar"

    def __post_init__(self) -> None:
        if not isinstance(self.positions, tuple):
            object.__setattr__(self, "positions", tuple(_to_point3(p) for p in self.positions))
        else:
            coerced = tuple(_to_point3(p) for p in self.positions)
            if coerced != self.positions:
                object.__setattr__(self, "positions", coerced)
        if self.l < 0 or self.m < 0:
            raise ValueError(f"l, m must be non-negative; got l={self.l}, m={self.m}")
        if self.spacing <= 0:
            raise ValueError(f"spacing must be positive, got {self.spacing}")
        if self.layer_height <= 0:
            raise ValueError(f"layer_height must be positive, got {self.layer_height}")
        if self.l and self.m:
            expected_n = 4 * self.l * self.m
            if len(self.positions) != expected_n:
                raise ValueError(
                    f"IBMBiplanarEmbedding expects 4 * l * m = {expected_n} "
                    f"qubits, got {len(self.positions)}"
                )

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
        name: str | None = None,
    ) -> IBMBiplanarEmbedding:
        r"""Build the canonical biplanar layout for a BB code.

        Parameters
        ----------
        bb_code : BivariateBicycleCode
            The BB code whose qubits we are laying out. Consulted for
            `l`, `m`, and the data/ancilla qubit index ranges.
        spacing : float, optional
            In-plane lattice pitch. Defaults to 1.0.
        layer_height : float, optional
            Out-of-plane separation between the mid-plane and each
            data plane. Must be strictly positive so that the
            `L`-block and `R`-block average `z` coordinates are
            nonzero (the acceptance test relies on this).
        name : str, optional
            Embedding label. Defaults to `"<bb_name>_ibm_biplanar"`.

        Returns
        -------
        IBMBiplanarEmbedding
        """
        if spacing <= 0:
            raise ValueError(f"spacing must be positive, got {spacing}")
        if layer_height <= 0:
            raise ValueError(f"layer_height must be positive, got {layer_height}")

        l, m = bb_code.l, bb_code.m
        total = 4 * l * m
        positions: list[Point3] = [(0.0, 0.0, 0.0)] * total

        for j in range(m):
            for i in range(l):
                cell_x = float(i) * spacing
                cell_y = float(j) * spacing
                flat_within_block = j * l + i
                # L-data at +layer_height, R-data at -layer_height.
                positions[flat_within_block] = (cell_x, cell_y, +layer_height)
                positions[l * m + flat_within_block] = (cell_x, cell_y, -layer_height)

        # Z-ancilla offset by (+0.5, 0) on the mid-plane.
        for idx, ancilla_q in enumerate(bb_code.z_check_qubits):
            j, i = divmod(idx, l)
            positions[ancilla_q] = (
                (float(i) + 0.5) * spacing,
                float(j) * spacing,
                0.0,
            )
        # X-ancilla offset by (0, +0.5) on the mid-plane.
        for idx, ancilla_q in enumerate(bb_code.x_check_qubits):
            j, i = divmod(idx, l)
            positions[ancilla_q] = (
                float(i) * spacing,
                (float(j) + 0.5) * spacing,
                0.0,
            )

        return cls(
            positions=tuple(positions),
            l=l,
            m=m,
            spacing=spacing,
            layer_height=layer_height,
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
        return _straight_line_routing(self.positions, active_edges, name=self.name)

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
            "bb_name": self.bb_name,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> IBMBiplanarEmbedding:
        if data.get("type") != "ibm_biplanar":
            raise ValueError(f"Expected type='ibm_biplanar', got {data.get('type')!r}.")
        version = data.get("schema_version")
        if version != cls.SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version {version}; expected {cls.SCHEMA_VERSION}."
            )
        positions = tuple(_to_point3(p) for p in data["positions"])
        return cls(
            positions=positions,
            l=int(data["l"]),
            m=int(data["m"]),
            spacing=float(data.get("spacing", 1.0)),
            layer_height=float(data.get("layer_height", 1.0)),
            bb_name=str(data.get("bb_name", "")),
            name=str(data.get("name", "ibm_biplanar")),
        )

r"""Fixed-permutation column embedding for frozen workbook layouts.

`FixedPermutationColumnEmbedding` holds a :class:`ColumnEmbedding`
layout produced externally — typically a `bbstim` workbook run, a
physicist's hand-designed qubit pattern, or the output of
:mod:`weave.optimize`'s swap-descent — and treats it as a read-only
fixture. The only difference between this class and a plain
:class:`ColumnEmbedding` is that it stores an explicit
`source_description` and `permutation` metadata field identifying
how the layout was derived, so downstream benchmarks and the PR 13
regression fixture can assert that the loaded positions match the
expected source.

Typical use
-----------
.. code-block:: python

    from weave.ir.embeddings import FixedPermutationColumnEmbedding

    emb = FixedPermutationColumnEmbedding.from_json_file(
        "benchmarks/fixtures/bb72_workbook.json"
    )
    compiled = compile_extraction(
        code=bb72,
        embedding=emb,
        schedule=ibm_schedule(bb72),
        ...
    )

The JSON format used by :meth:`from_json_file` is the one produced
by :meth:`to_json`: the same shape as :class:`ColumnEmbedding` with
two extra fields (`source_description`, `permutation`).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from ...geometry import Point3
from ..embedding import IREdge, RoutingGeometry
from ..route import RouteID
from .column import _straight_line_routing, _to_point3

__all__ = ["FixedPermutationColumnEmbedding"]


@dataclass(frozen=True)
class FixedPermutationColumnEmbedding:
    r"""A column-style embedding loaded from a frozen source.

    Parameters
    ----------
    positions : tuple[Point3, ...]
        One 3D position per qubit. Validated for shape only; no
        grid-consistency check is performed (the user committed to
        this layout, we trust it).
    num_columns, num_rows, layers_per_cell : int
        Optional grid metadata, matching :class:`ColumnEmbedding`.
    permutation : tuple[int, ...]
        Optional permutation mapping "canonical" qubit indices (e.g.
        the integers that :class:`ColumnEmbedding.from_bb_monomial`
        would produce) onto the stored `positions` order. Empty if
        the stored layout was not derived from a canonical one.
    source_description : str
        Free-form provenance string (e.g. ``"bbstim workbook @ commit
        a1b2c3"``). Stored verbatim in the serialized JSON.
    name : str
        Embedding label.
    """

    SCHEMA_VERSION: ClassVar[int] = 1

    positions: tuple[Point3, ...]
    num_columns: int = 0
    num_rows: int = 0
    layers_per_cell: int = 1
    permutation: tuple[int, ...] = field(default_factory=tuple)
    source_description: str = ""
    name: str = "fixed_permutation_column"

    def __post_init__(self) -> None:
        if not isinstance(self.positions, tuple):
            object.__setattr__(self, "positions", tuple(_to_point3(p) for p in self.positions))
        if not isinstance(self.permutation, tuple):
            object.__setattr__(self, "permutation", tuple(int(x) for x in self.permutation))
        if self.num_columns < 0 or self.num_rows < 0 or self.layers_per_cell < 0:
            raise ValueError("num_columns, num_rows, layers_per_cell must be non-negative")
        # Validate permutation when provided.
        if self.permutation:
            n = len(self.positions)
            if len(self.permutation) != n:
                raise ValueError(
                    f"permutation length {len(self.permutation)} "
                    f"does not match len(positions) = {n}"
                )
            if sorted(self.permutation) != list(range(n)):
                raise ValueError("permutation must be a bijection on range(len(positions))")

    # ------------------------------------------------------------------
    # Classmethod factories
    # ------------------------------------------------------------------

    @classmethod
    def from_json_file(cls, path: str | Path) -> FixedPermutationColumnEmbedding:
        """Load a :class:`FixedPermutationColumnEmbedding` from a JSON file.

        Parameters
        ----------
        path : str or Path
            Filesystem path to the JSON document.

        Returns
        -------
        FixedPermutationColumnEmbedding
            The parsed embedding. Raises :class:`ValueError` for
            malformed input.
        """
        with open(path) as f:
            data = json.load(f)
        return cls.from_json(data)

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
            "type": "fixed_permutation_column",
            "name": self.name,
            "positions": [list(p) for p in self.positions],
            "num_columns": self.num_columns,
            "num_rows": self.num_rows,
            "layers_per_cell": self.layers_per_cell,
            "permutation": list(self.permutation),
            "source_description": self.source_description,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> FixedPermutationColumnEmbedding:
        if data.get("type") != "fixed_permutation_column":
            raise ValueError(f"Expected type='fixed_permutation_column', got {data.get('type')!r}.")
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
            permutation=tuple(int(x) for x in data.get("permutation", [])),
            source_description=str(data.get("source_description", "")),
            name=str(data.get("name", "fixed_permutation_column")),
        )

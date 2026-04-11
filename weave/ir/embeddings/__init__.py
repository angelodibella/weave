"""Concrete :class:`~weave.ir.embedding.Embedding` implementations.

The base :class:`~weave.ir.Embedding` is a structural protocol; the
classes in this package are concrete implementations used by the
compiler and the benchmarks.

- :class:`StraightLineEmbedding` — every edge is a straight segment
  between its endpoints. The default for non-grid CSS codes.
- :class:`JsonPolylineEmbedding` — arbitrary precomputed polylines
  loaded from a JSON document; the path used by externally routed
  inputs.
- :class:`ColumnEmbedding` — regular column-grid layout for CSS
  codes with a bipartite-lattice structure.
- :class:`MonomialColumnEmbedding` — the canonical
  `(4 l) \\times m` BB code layout (L-data, Z-ancilla, X-ancilla,
  R-data in four parallel sub-columns).
- :class:`IBMBiplanarEmbedding` — two-plane BB code layout with
  L-block data at `z > 0`, R-block data at `z < 0`, and ancillas on
  the mid-plane.
- :class:`FixedPermutationColumnEmbedding` — load a frozen BB layout
  from a JSON file (e.g. a workbook run).
"""

from .biplanar import IBMBiplanarEmbedding
from .column import ColumnEmbedding, MonomialColumnEmbedding
from .fixed_permutation import FixedPermutationColumnEmbedding
from .json_polyline import JsonPolylineEmbedding
from .straight_line import StraightLineEmbedding

__all__ = [
    "ColumnEmbedding",
    "FixedPermutationColumnEmbedding",
    "IBMBiplanarEmbedding",
    "JsonPolylineEmbedding",
    "MonomialColumnEmbedding",
    "StraightLineEmbedding",
]

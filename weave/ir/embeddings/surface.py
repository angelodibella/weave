r"""Surface embedding: geodesic polylines on an arbitrary 2D manifold.

A :class:`SurfaceEmbedding` places code qubits on a
:class:`~weave.surface.Surface` (the abstract 2D manifold ABC) and
routes each edge as a geodesic sampled at `num_samples` points,
embedded into 3D via the surface's :meth:`get_3d_embedding`. The
downstream :func:`~weave.geometry.polyline_distance` then computes
the minimum 3D chord distance between two such sampled arcs,
which is a conservative proxy for the geodesic separation (chord ≤
geodesic for convex surfaces, with equality only for flat
manifolds).

Primary use case: a distance-3 or distance-5 CSS code laid out on
a :class:`~weave.surface.Torus`, where the geodesic polylines wrap
through the periodic boundary and `polyline_distance` correctly
measures the 3D proximity of routed edges on the torus surface.

Design
------
The class does NOT subclass any existing embedding; it implements
the :class:`~weave.ir.Embedding` protocol directly so it can hold
a live `Surface` reference (which is mutable in general — it
stores node coordinates). The protocol requires:

- ``schema_version`` / ``surface_name`` properties.
- ``node_position(qubit_idx) → Point3``.
- ``routing_geometry(active_edges) → RoutingGeometry``.
- ``to_json() → dict``.

For JSON round-trip the class serializes the intrinsic coordinates
and the surface type + parameters, so the surface can be
reconstructed on deserialization.

References
----------
- Di Bella, *Geometry-induced correlated noise in qLDPC syndrome
  extraction* (PRX Quantum, under review, 2026). §V.C describes
  the torus novelty demonstration.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ...geometry import Point3
from ...surface.base import Surface
from ...surface.torus import Torus
from ..embedding import IREdge, IRPolyline, RoutingGeometry
from ..route import RouteID

__all__ = ["SurfaceEmbedding"]


class SurfaceEmbedding:
    """Embedding of a CSS code on an arbitrary 2D surface.

    Qubits are placed at intrinsic coordinates `(u, v)` on the
    surface and routed via geodesics sampled into 3D polylines.

    Parameters
    ----------
    surface : Surface
        The 2D manifold hosting the embedding. Must support
        :meth:`~weave.surface.Surface.get_shortest_path`,
        :meth:`~weave.surface.Surface.get_3d_embedding`, and
        :meth:`~weave.surface.Surface.path_length`.
    node_coords : dict[int, tuple[float, float]]
        Map from qubit index to intrinsic `(u, v)` coordinates on
        the surface.
    num_samples : int, optional
        Number of points to sample along each geodesic polyline.
        Higher values give more accurate 3D chord-distance
        estimates at the cost of longer polylines. Default 20.
    name : str, optional
        Embedding label. Defaults to
        ``"surface_<surface_name>"``.

    Notes
    -----
    This class is **not** a frozen dataclass: the underlying
    `Surface` is mutable. It implements the `Embedding` protocol
    structurally (duck typing) rather than inheriting from any
    base.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        surface: Surface,
        node_coords: dict[int, tuple[float, float]],
        *,
        num_samples: int = 20,
        name: str | None = None,
    ) -> None:
        if num_samples < 2:
            raise ValueError(f"num_samples must be ≥ 2, got {num_samples}")
        self.surface = surface
        self.node_coords = dict(node_coords)
        self.num_samples = num_samples
        self.name = name or f"surface_{self._surface_type_name()}"

        # Cache 3D positions once.
        coords_array = np.array([node_coords[q] for q in sorted(node_coords)], dtype=float)
        embedded_3d = surface.get_3d_embedding(coords_array)
        self._positions: dict[int, Point3] = {}
        for i, q in enumerate(sorted(node_coords)):
            p = embedded_3d[i]
            self._positions[q] = (float(p[0]), float(p[1]), float(p[2]))

    def _surface_type_name(self) -> str:
        return type(self.surface).__name__.lower()

    # ------------------------------------------------------------------
    # Embedding protocol
    # ------------------------------------------------------------------

    @property
    def schema_version(self) -> int:
        return self.SCHEMA_VERSION

    @property
    def surface_name(self) -> str:
        return self._surface_type_name()

    def node_position(self, qubit_idx: int) -> Point3:
        if qubit_idx not in self._positions:
            raise IndexError(
                f"qubit_idx {qubit_idx} not in node_coords (valid: {sorted(self._positions)[:5]}…)"
            )
        return self._positions[qubit_idx]

    def routing_geometry(self, active_edges: Sequence[RouteID | IREdge]) -> RoutingGeometry:
        """Route each edge as a geodesic sampled on the surface.

        For each edge `(source, target)`, this:

        1. Looks up intrinsic coords `(u_s, v_s)` and `(u_t, v_t)`.
        2. Computes the shortest path on the surface via
           :meth:`Surface.get_shortest_path`.
        3. Samples `num_samples` uniformly-spaced points along the
           straight line in covering-space coordinates from
           `(u_s, v_s)` to the unwrapped target
           `(u_s + du, v_s + dv)`.
        4. Maps each sample to 3D via
           :meth:`Surface.get_3d_embedding`.
        5. Returns the tuple of 3D points as the polyline.

        The resulting polylines approximate the 3D arc of the
        geodesic. :func:`~weave.geometry.polyline_distance` then
        computes min chord distance between polyline segment pairs.
        """
        edges_map: dict[RouteID, IRPolyline] = {}
        for item in active_edges:
            if isinstance(item, RouteID):
                rid = item
                u, v = rid.source, rid.target
            else:
                u, v = item
                rid = RouteID(source=int(u), target=int(v))
            if u not in self.node_coords:
                raise IndexError(f"Edge source {u} not in node_coords.")
            if v not in self.node_coords:
                raise IndexError(f"Edge target {v} not in node_coords.")

            coord_u = self.node_coords[u]
            coord_v = self.node_coords[v]
            path = self.surface.get_shortest_path(coord_u, coord_v)

            # Sample along the covering-space geodesic.
            start = np.array(path["start"], dtype=float)
            end_unw = np.array(path["end_unwrapped"], dtype=float)
            ts = np.linspace(0.0, 1.0, self.num_samples)
            samples_2d = start[None, :] + ts[:, None] * (end_unw - start)[None, :]

            # Embed in 3D.
            samples_3d = self.surface.get_3d_embedding(samples_2d)
            poly: list[Point3] = [(float(p[0]), float(p[1]), float(p[2])) for p in samples_3d]
            edges_map[rid] = tuple(poly)
        return RoutingGeometry(edges=edges_map, name=self.name)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        surface_data: dict[str, Any] = {
            "type": self._surface_type_name(),
        }
        if isinstance(self.surface, Torus):
            surface_data["Lx"] = self.surface.Lx
            surface_data["Ly"] = self.surface.Ly
        return {
            "schema_version": self.SCHEMA_VERSION,
            "type": "surface",
            "name": self.name,
            "surface": surface_data,
            "node_coords": {str(q): list(c) for q, c in sorted(self.node_coords.items())},
            "num_samples": self.num_samples,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SurfaceEmbedding:
        if data.get("type") != "surface":
            raise ValueError(f"Expected type='surface', got {data.get('type')!r}.")
        surface_info = data["surface"]
        if surface_info["type"] == "torus":
            surface: Surface = Torus(
                Lx=float(surface_info["Lx"]),
                Ly=float(surface_info["Ly"]),
            )
        else:
            raise ValueError(f"Unknown surface type {surface_info['type']!r}; expected 'torus'.")
        node_coords = {int(q): (float(c[0]), float(c[1])) for q, c in data["node_coords"].items()}
        return cls(
            surface=surface,
            node_coords=node_coords,
            num_samples=int(data.get("num_samples", 20)),
            name=data.get("name", "surface"),
        )

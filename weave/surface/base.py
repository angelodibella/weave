# weave/surface/base.py
"""Abstract base class for geometric surfaces used for embedding QEC codes."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np


class Surface(ABC):
    """
    Abstract base class representing a 2D manifold for embedding Tanner graphs.

    Provides an interface for coordinate systems, path definitions, distance
    calculations, intersection checks, and 2D projections, all specific to the
    surface's geometry.

    Nodes are typically identified by integer indices. Positions are represented
    by 2D coordinates within the surface's intrinsic coordinate system (e.g., (u, v)).
    """

    @abstractmethod
    def get_intrinsic_coords(self, node_indices: Sequence[int]) -> np.ndarray:
        """
        Retrieves the intrinsic 2D coordinates for given node indices.

        Parameters
        ----------
        node_indices : Sequence[int]
            A sequence of node indices.

        Returns
        -------
        np.ndarray
            A NumPy array of shape (N, 2), where N is the number of indices,
            containing the (u, v) coordinates for each node on the surface.
        """
        pass

    @abstractmethod
    def set_intrinsic_coords(self, coords: dict[int, tuple[float, float]]) -> None:
        """
        Sets the intrinsic 2D coordinates for nodes.

        Parameters
        ----------
        coords : Dict[int, Tuple[float, float]]
            A dictionary mapping node indices to their (u, v) coordinates.
        """
        pass

    @abstractmethod
    def get_shortest_path(
        self, coord1: tuple[float, float], coord2: tuple[float, float]
    ) -> Any:
        """
        Represents the shortest path (geodesic) between two points on the surface.

        The exact return type depends on the surface implementation. It should
        contain enough information to draw the path and check intersections.
        For simple surfaces, this might just be the start and end coordinates,
        implying a straight line in the parameter space (possibly wrapped).

        Parameters
        ----------
        coord1 : Tuple[float, float]
            The intrinsic (u, v) coordinates of the starting point.
        coord2 : Tuple[float, float]
            The intrinsic (u, v) coordinates of the ending point.

        Returns
        -------
        Any
            An object representing the shortest path.
        """
        pass

    @abstractmethod
    def path_length(self, path: Any) -> float:
        """
        Calculates the length of a path on the surface.

        Parameters
        ----------
        path : Any
            The path object returned by `get_shortest_path`.

        Returns
        -------
        float
            The length of the path.
        """
        pass

    @abstractmethod
    def check_intersection(
        self, path1: Any, path2: Any, return_points: bool = False
    ) -> bool | list[tuple[float, float]]:
        """
        Checks if two shortest paths intersect, optionally returning intrinsic (u, v) points.
        ... (rest of docstring) ...
        """
        pass

    @abstractmethod
    def project_to_2d(self, coords: np.ndarray) -> np.ndarray:
        """Projects intrinsic (u, v) coordinates to a 2D plane (x, y)."""
        pass

    @abstractmethod
    def get_boundary_info(self) -> dict[str, Any] | None:
        """Provides info for drawing 2D projection boundaries."""
        pass

    @abstractmethod
    def draw_path_2d(self, ax: Any, path: Any, **kwargs) -> None:
        """Draws a path representation on a 2D Matplotlib axes."""
        pass

    # --- 3D Methods ---

    @abstractmethod
    def get_3d_embedding(self, coords: np.ndarray) -> np.ndarray:
        """Maps intrinsic (u, v) coordinates to 3D Euclidean space (x, y, z)."""
        pass

    def get_3d_mesh(
        self, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Generates coordinate arrays (X, Y, Z) for plotting the surface mesh in 3D."""
        return None  # Default: no mesh

    def get_3d_intersection_points(
        self, path1: Any, path2: Any
    ) -> list[tuple[float, float, float]]:
        """
        Calculates intersection points of two paths in 3D embedding space.

        Default implementation finds intrinsic intersection points using
        `check_intersection` and maps them to 3D using `get_3d_embedding`.
        Subclasses can override for more direct calculation if needed.

        Parameters
        ----------
        path1 : Any
            The first path object.
        path2 : Any
            The second path object.

        Returns
        -------
        List[Tuple[float, float, float]]
            A list of (x, y, z) coordinates where the paths intersect in 3D.
        """
        intrinsic_points_uv = self.check_intersection(path1, path2, return_points=True)
        if not intrinsic_points_uv:
            return []
        
        # Map intrinsic points to 3D.
        points_uv_array = np.array(intrinsic_points_uv)
        points_xyz_array = self.get_3d_embedding(points_uv_array)
        return [tuple(p) for p in points_xyz_array]

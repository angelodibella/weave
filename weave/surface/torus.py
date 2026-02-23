# weave/surface/torus.py
"""Implementation of a flat Torus surface for embedding QEC Tanner graphs."""

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
from matplotlib import patches
from matplotlib.collections import LineCollection

from .base import Surface


# Helper function for line clipping (Liang-Barsky algorithm).
def _liang_barsky_clip(
    p1: tuple[float, float],
    p2: tuple[float, float],
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Clips a line segment p1-p2 to a rectangle [xmin, ymin] x [xmax, ymax]."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    p = [-dx, dx, -dy, dy]
    q = [x1 - xmin, xmax - x1, y1 - ymin, ymax - y1]
    t0, t1 = 0.0, 1.0

    for i in range(4):
        if abs(p[i]) < 1e-9:  # parallel
            if q[i] < -1e-9:
                return None  # outside
        else:
            r = q[i] / p[i]
            if p[i] < 0:  # entering
                if r > t1:
                    return None
                if r > t0:
                    t0 = r
            else:  # leaving
                if r < t0:
                    return None
                if r < t1:
                    t1 = r
    if t0 > t1:
        return None

    return ((x1 + t0 * dx, y1 + t0 * dy), (x1 + t1 * dx, y1 + t1 * dy))


class Torus(Surface):
    """
    Represents a flat 2D Torus surface defined by identifications on a rectangle [0, Lx) x [0, Ly).
    Provides methods for coordinate handling, path calculations, intersection checks, and
    generating 2D/3D visualizations.

    Parameters
    ----------
    Lx : float
        Width of the fundamental domain (u-dimension).
    Ly : float
        Height of the fundamental domain (v-dimension).
    node_coords : Optional[Dict[int, Tuple[float, float]]], optional
        Initial mapping from node indices to (u, v) coordinates.
    major_radius : float, optional
        Major radius (R) for the 3D torus embedding. Defaults dynamically.
    minor_radius : float, optional
        Minor radius (r) for the 3D torus embedding. Defaults dynamically.
    """

    def __init__(
        self,
        Lx: float,
        Ly: float,
        node_coords: dict[int, tuple[float, float]] | None = None,
        major_radius: float | None = None,
        minor_radius: float | None = None,
    ) -> None:
        if Lx <= 0 or Ly <= 0:
            raise ValueError("Torus dimensions Lx and Ly must be positive.")
        self.Lx = float(Lx)
        self.Ly = float(Ly)
        self.coords: dict[int, tuple[float, float]] = {}
        if node_coords:
            self.set_intrinsic_coords(node_coords)

        # Default 3D parameters aiming for a reasonable aspect ratio.
        scale_factor = max(self.Lx, self.Ly) / (2 * np.pi)
        self.R = major_radius if major_radius is not None else scale_factor * 2.0
        self.r = minor_radius if minor_radius is not None else scale_factor * 0.5
        if self.R <= self.r:
            warnings.warn(
                f"Torus R={self.R:.2f} <= r={self.r:.2f}. 3D embedding may self-intersect.",
                stacklevel=2,
            )

    def get_intrinsic_coords(self, node_indices: Sequence[int]) -> np.ndarray:
        """Retrieves the (u, v) coordinates for given node indices."""
        missing = [idx for idx in node_indices if idx not in self.coords]
        if missing:
            raise ValueError(f"Coordinates not set for node index(es): {missing}")
        return np.array([self.coords[idx] for idx in node_indices])

    def set_intrinsic_coords(self, coords: dict[int, tuple[float, float]]) -> None:
        """Sets the (u, v) coordinates, wrapping them into [0, Lx) x [0, Ly)."""
        for idx, (u, v) in coords.items():
            wu = np.fmod(u, self.Lx)
            wv = np.fmod(v, self.Ly)
            self.coords[idx] = (
                wu + self.Lx if wu < 0 else wu,
                wv + self.Ly if wv < 0 else wv,
            )

    def _get_displacement_vector(
        self, coord1: tuple[float, float], coord2: tuple[float, float]
    ) -> tuple[float, float]:
        """Calculates the shortest displacement vector (du, dv) on the torus."""
        u1, v1 = coord1
        u2, v2 = coord2
        du = u2 - u1
        dv = v2 - v1
        du = (du + self.Lx / 2) % self.Lx - self.Lx / 2
        dv = (dv + self.Ly / 2) % self.Ly - self.Ly / 2
        return du, dv

    def get_shortest_path(
        self, coord1: tuple[float, float], coord2: tuple[float, float]
    ) -> dict[str, Any]:
        """Represents the shortest path (straight line segment in covering space)."""
        u1, v1 = coord1
        du, dv = self._get_displacement_vector(coord1, coord2)
        return {
            "start": tuple(coord1),
            "end": tuple(coord2),
            "end_unwrapped": (u1 + du, v1 + dv),
        }

    def path_length(self, path: dict[str, Any]) -> float:
        """Calculates the Euclidean length of the shortest path displacement."""
        du = path["end_unwrapped"][0] - path["start"][0]
        dv = path["end_unwrapped"][1] - path["start"][1]
        return np.sqrt(du**2 + dv**2)

    def check_intersection(
        self, path1: dict[str, Any], path2: dict[str, Any], return_points: bool = False
    ) -> bool | list[tuple[float, float]]:
        """Checks if two shortest paths intersect on the torus."""
        intersections = []
        p1_s, p1_e_unw = path1["start"], path1["end_unwrapped"]
        p2_s_orig, p2_e_orig = path2["start"], path2["end"]
        du2, dv2 = self._get_displacement_vector(p2_s_orig, p2_e_orig)

        for dx_shift in [-self.Lx, 0, self.Lx]:
            for dy_shift in [-self.Ly, 0, self.Ly]:
                p2_s_shift = (p2_s_orig[0] + dx_shift, p2_s_orig[1] + dy_shift)
                p2_e_shift_unw = (p2_s_shift[0] + du2, p2_s_shift[1] + dv2)
                intersect_pt_unw = self._segment_intersection_point_2d(
                    p1_s, p1_e_unw, p2_s_shift, p2_e_shift_unw
                )
                if intersect_pt_unw:
                    if not return_points:
                        return True
                    u, v = intersect_pt_unw
                    wu = np.fmod(u, self.Lx)
                    wv = np.fmod(v, self.Ly)
                    intersections.append(
                        (wu + self.Lx if wu < 0 else wu, wv + self.Ly if wv < 0 else wv)
                    )

        if return_points and intersections:
            unique_intersections = []
            seen = set()
            tol = 1e-8
            for pt in intersections:
                key = (round(pt[0] / tol), round(pt[1] / tol))
                if key not in seen:
                    unique_intersections.append(pt)
                    seen.add(key)
            return unique_intersections
        return intersections if return_points else False

    @staticmethod
    def _segment_intersection_point_2d(p1, q1, p2, q2) -> tuple[float, float] | None:
        """Find intersection point of 2D segments (p1,q1) and (p2,q2), excluding endpoints."""
        x1, y1 = p1
        x2, y2 = q1
        x3, y3 = p2
        x4, y4 = q2
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-9:
            return None  # Parallel/collinear
        t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
        u_num = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3))
        t, u = t_num / den, u_num / den
        eps = 1e-9
        if eps < t < 1.0 - eps and eps < u < 1.0 - eps:  # Strict interior intersection
            return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
        return None

    def project_to_2d(self, coords: np.ndarray) -> np.ndarray:
        """Projects intrinsic torus (u, v) coordinates directly to (x, y)."""
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("Input coords must be shape (N, 2) or (2,).")
        return coords.copy()

    def get_boundary_info(self) -> dict[str, Any] | None:
        """Returns boundary lines and labels for the rectangular projection."""
        return {"type": "rectangle", "xlim": (0.0, self.Lx), "ylim": (0.0, self.Ly)}

    def draw_path_2d(self, ax: Any, path: dict[str, Any], **kwargs: Any) -> None:
        """Draws the shortest path on 2D axes, handling torus wrapping via clipping."""
        u1, v1 = path["start"]
        u2_unw, v2_unw = path["end_unwrapped"]
        xmin, ymin, xmax, ymax = 0.0, 0.0, self.Lx, self.Ly
        segments = []
        for dx_shift in [-self.Lx, 0, self.Lx]:
            for dy_shift in [-self.Ly, 0, self.Ly]:
                p1_s = (u1 + dx_shift, v1 + dy_shift)
                p2_s = (u2_unw + dx_shift, v2_unw + dy_shift)
                clipped = _liang_barsky_clip(p1_s, p2_s, xmin, ymin, xmax, ymax)
                if (
                    clipped
                    and np.linalg.norm(np.array(clipped[1]) - np.array(clipped[0]))
                    > 1e-9
                ):
                    segments.append(clipped)
        if segments:
            lc_kwargs = {
                k: v for k, v in kwargs.items() if k != "segments"
            }  # Avoid duplicate kwarg
            lc = LineCollection(segments, **lc_kwargs)
            ax.add_collection(lc)

    # --- 3D Methods: EXPERIMENTAL ---
    def get_3d_embedding(self, coords: np.ndarray) -> np.ndarray:
        """Maps intrinsic (u, v) coordinates to 3D torus embedding (x, y, z)."""
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)
        if coords.shape[1] != 2:
            raise ValueError("Input coords must be shape (N, 2).")

        u, v = coords[:, 0], coords[:, 1]
        phi = (u / self.Lx) * 2 * np.pi
        theta = (v / self.Ly) * 2 * np.pi

        x = (self.R + self.r * np.cos(theta)) * np.cos(phi)
        y = (self.R + self.r * np.cos(theta)) * np.sin(phi)
        z = self.r * np.sin(theta)

        return np.stack([x, y, z], axis=-1)

    def get_3d_mesh(
        self, u_res: int = 60, v_res: int = 40
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Generates coordinate arrays for plotting the torus mesh in 3D."""
        u = np.linspace(0, self.Lx, u_res)
        v = np.linspace(0, self.Ly, v_res)
        u_grid, v_grid = np.meshgrid(u, v)

        coords_uv = np.stack([u_grid.ravel(), v_grid.ravel()], axis=-1)
        coords_xyz = self.get_3d_embedding(coords_uv)

        shape = (v_res, u_res)
        return (
            coords_xyz[:, 0].reshape(shape),
            coords_xyz[:, 1].reshape(shape),
            coords_xyz[:, 2].reshape(shape),
        )

"""Polyline distance primitives.

This module provides the geometric primitives used by the geometry-aware
noise compiler: exact 3D segment-to-segment closest approach and its
reduction to polylines. Everything here is pure numpy; no dependency on
Stim, networkx, or any weave code.

Ported and generalized from the reference implementation in the paper
*Geometry-induced correlated noise in qLDPC syndrome extraction*
(Di Bella 2026).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

Point3 = tuple[float, float, float]
"""A 3D point `(x, y, z)`. Flat 2D embeddings use `z = 0.0`."""

Polyline = Sequence[Point3]
"""A sequence of 3D points defining a routed edge as a chain of straight
segments. Must contain at least 2 points."""


def _point_segment_distance(point: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray) -> float:
    """Distance from a point to a 3D segment."""
    u = seg_b - seg_a
    w = point - seg_a
    c1 = float(np.dot(u, w))
    if c1 <= 0.0:
        return float(np.linalg.norm(point - seg_a))
    c2 = float(np.dot(u, u))
    if c1 >= c2:
        return float(np.linalg.norm(point - seg_b))
    t = c1 / c2
    foot = seg_a + t * u
    return float(np.linalg.norm(point - foot))


def segment_distance(p1: Point3, p2: Point3, q1: Point3, q2: Point3) -> float:
    """Minimum Euclidean distance between two 3D line segments.

    Correctly handles every case: non-parallel with clamped interior
    closest approach, parallel-non-collinear (constant perpendicular
    distance), collinear overlapping (zero), and collinear disjoint
    (the endpoint gap). Algorithm: compute the non-parallel closest
    approach when the segment directions are not degenerate, then
    additionally minimize over the four endpoint-to-other-segment
    distances to catch the parallel and collinear cases robustly.

    Parameters
    ----------
    p1, p2 : Point3
        Endpoints of the first segment.
    q1, q2 : Point3
        Endpoints of the second segment.

    Returns
    -------
    float
        The minimum Euclidean distance between any point on segment
        `p1`-`p2` and any point on segment `q1`-`q2`. Returns `0.0`
        (up to floating-point tolerance) when the segments intersect.
    """
    pa = np.asarray(p1, dtype=float)
    pb = np.asarray(p2, dtype=float)
    qa = np.asarray(q1, dtype=float)
    qb = np.asarray(q2, dtype=float)

    u = pb - pa
    v = qb - qa
    w0 = pa - qa

    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d_prod = float(np.dot(u, w0))
    e = float(np.dot(v, w0))
    denom = a * c - b * b
    eps = 1e-12

    best = math.inf

    # Primary: non-parallel closest approach, clamped to segment interiors.
    # Skipped when the segments are parallel or one is degenerate — in those
    # cases the endpoint-to-segment fallback below gives the correct answer.
    if denom > eps:
        s = (b * e - c * d_prod) / denom
        t = (a * e - b * d_prod) / denom
        s = min(1.0, max(0.0, s))
        t = min(1.0, max(0.0, t))
        closest_p = pa + s * u
        closest_q = qa + t * v
        best = float(np.linalg.norm(closest_p - closest_q))

    # Fallback / completion: the minimum is always ≤ the minimum of the
    # four endpoint-to-other-segment distances. This alone correctly handles
    # parallel and collinear cases (overlapping and disjoint).
    best = min(
        best,
        _point_segment_distance(pa, qa, qb),
        _point_segment_distance(pb, qa, qb),
        _point_segment_distance(qa, pa, pb),
        _point_segment_distance(qb, pa, pb),
    )
    return best


def polyline_distance(poly1: Polyline, poly2: Polyline) -> float:
    """Minimum distance between two polylines.

    Computes the minimum of `segment_distance` over all pairs of
    segments drawn from the two polylines.

    Parameters
    ----------
    poly1, poly2 : Polyline
        Sequences of 3D points. Each polyline is interpreted as the
        consecutive segments connecting neighboring points. Both must
        contain at least 2 points.

    Returns
    -------
    float
        The minimum Euclidean distance between any point on `poly1` and
        any point on `poly2`.

    Raises
    ------
    ValueError
        If either polyline has fewer than 2 points.
    """
    if len(poly1) < 2 or len(poly2) < 2:
        raise ValueError(
            f"Polylines must contain at least 2 points; got lengths {len(poly1)} and {len(poly2)}."
        )

    best = math.inf
    for i in range(len(poly1) - 1):
        for j in range(len(poly2) - 1):
            dist = segment_distance(poly1[i], poly1[i + 1], poly2[j], poly2[j + 1])
            if dist < best:
                best = dist
    return best

"""Concrete :class:`~weave.ir.embedding.Embedding` implementations."""

from .json_polyline import JsonPolylineEmbedding
from .straight_line import StraightLineEmbedding

__all__ = ["JsonPolylineEmbedding", "StraightLineEmbedding"]

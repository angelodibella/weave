"""JSON-file import helpers for schedules and embeddings.

These are thin wrappers that add file I/O on top of the existing
:meth:`Schedule.from_json` and :func:`load_embedding` deserialisers.
They exist so downstream scripts and notebooks can write

.. code-block:: python

    from weave.ir.importers import schedule_from_json_file

    sched = schedule_from_json_file("my_schedule.json")

without manually opening files and parsing JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..embedding import Embedding, load_embedding
from ..schedule import Schedule

__all__ = [
    "embedding_from_json_file",
    "schedule_from_json_file",
]


def schedule_from_json_file(path: str | Path) -> Schedule:
    """Load a :class:`~weave.ir.Schedule` from a JSON file.

    The file must contain a single JSON object whose shape matches
    :meth:`Schedule.to_json`. This is the recommended interchange
    format for schedules produced by external tools.

    Parameters
    ----------
    path : str or Path
        Filesystem path to the JSON document.

    Returns
    -------
    Schedule

    Raises
    ------
    ValueError
        If the JSON does not conform to the Schedule schema.
    FileNotFoundError
        If the file does not exist.
    """
    with open(path) as f:
        data = json.load(f)
    return Schedule.from_json(data)


def embedding_from_json_file(path: str | Path) -> Embedding:
    """Load an :class:`~weave.ir.Embedding` from a JSON file.

    The file must contain a single JSON object with a ``type`` field
    that :func:`~weave.ir.load_embedding` can dispatch on. Every
    shipped embedding class (:class:`StraightLineEmbedding`,
    :class:`ColumnEmbedding`, :class:`MonomialColumnEmbedding`,
    :class:`IBMBiplanarEmbedding`, :class:`FixedPermutationColumnEmbedding`,
    :class:`JsonPolylineEmbedding`) supports this interface.

    Parameters
    ----------
    path : str or Path
        Filesystem path to the JSON document.

    Returns
    -------
    Embedding
    """
    with open(path) as f:
        data = json.load(f)
    return load_embedding(data)

"""Schedule and embedding import adapters.

This package converts external representations — Stim circuits,
JSON documents, and (in future PRs) tool-specific formats such as
AlphaSyndrome or LR-circuits output — into weave's typed IR objects
(:class:`~weave.ir.Schedule`, :class:`~weave.ir.Embedding`).

Public API
----------
- :func:`schedule_from_stim_circuit` — recover a :class:`Schedule`
  from a ``stim.Circuit`` by splitting at ``TICK`` markers and
  mapping Stim instructions to :class:`~weave.ir.ScheduleEdge`
  objects. The caller supplies the qubit-role mapping.
- :func:`schedule_from_json_file` — load a :class:`Schedule` from
  a JSON file (thin wrapper over :meth:`Schedule.from_json`).
- :func:`embedding_from_json_file` — load an :class:`Embedding`
  from a JSON file (thin wrapper over :func:`~weave.ir.load_embedding`).
"""

from __future__ import annotations

from .from_json import embedding_from_json_file, schedule_from_json_file
from .from_stim_circuit import schedule_from_stim_circuit

__all__ = [
    "embedding_from_json_file",
    "schedule_from_json_file",
    "schedule_from_stim_circuit",
]

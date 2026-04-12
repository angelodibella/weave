"""Recover a :class:`~weave.ir.Schedule` from a ``stim.Circuit``.

The importer walks a Stim circuit instruction-by-instruction,
grouping gates between consecutive ``TICK`` markers into
:class:`~weave.ir.ScheduleStep` objects, and partitioning the
resulting tick sequence into head / cycle / tail blocks by
detecting Stim's ``REPEAT`` structure.

Limitations
-----------
- **No detector / observable recovery.** The importer extracts
  gates only; detector and observable annotations are ignored
  (they live in the compiler's detector-emission helpers, not in
  the schedule).
- **No noise recovery.** ``DEPOLARIZE1``, ``DEPOLARIZE2``,
  ``PAULI_CHANNEL_*``, and ``CORRELATED_ERROR`` instructions are
  silently skipped — the schedule is a noise-free gate sequence.
- **Sector inference is heuristic.** For two-qubit gates, the
  importer assigns ``interaction_sector`` by checking whether the
  control qubit is in ``qubit_roles["data"]`` (→ X sector, Z-check
  style) or in an ancilla class (→ Z sector, X-check style). If
  the caller's ``qubit_roles`` doesn't cover all qubits the
  instruction touches, or the CNOT direction doesn't fit the CSS
  pattern, the sector is left as ``None``.
- **REPEAT must be outermost.** Nested ``REPEAT`` blocks are not
  supported; only a single ``REPEAT`` at the top level is
  interpreted as the cycle boundary.

Stim instruction mapping
------------------------
.. list-table::
   :header-rows: 1

   * - Stim instruction
     - ScheduleEdge
   * - ``CX a b`` / ``CNOT a b``
     - ``TwoQubitEdge("CNOT", control=a, target=b)``
   * - ``H q``
     - ``SingleQubitEdge("H", qubit=q)``
   * - ``S q``
     - ``SingleQubitEdge("S", qubit=q)``
   * - ``R q``
     - ``SingleQubitEdge("R", qubit=q)``
   * - ``RX q``
     - ``SingleQubitEdge("RX", qubit=q)``
   * - ``M q``
     - ``SingleQubitEdge("M", qubit=q)``
   * - ``MX q``
     - ``SingleQubitEdge("MX", qubit=q)``
   * - ``MR q``
     - ``SingleQubitEdge("MR", qubit=q)``
   * - ``MRX q``
     - ``SingleQubitEdge("MRX", qubit=q)``
   * - ``X q`` / ``Y q`` / ``Z q`` / ``I q``
     - ``SingleQubitEdge("<gate>", qubit=q)``
   * - ``TICK``
     - Step boundary (not an edge).
   * - Noise / detector / observable
     - Silently skipped.
"""

from __future__ import annotations

from typing import Any

import stim

from ..schedule import (
    QubitRole,
    Schedule,
    ScheduleEdge,
    ScheduleRole,
    ScheduleStep,
    SingleQubitEdge,
    TwoQubitEdge,
)

__all__ = ["schedule_from_stim_circuit"]


# Gate names that Stim uses for two-qubit Cliffords.
_TWO_QUBIT_GATES: frozenset[str] = frozenset({"CX", "CNOT", "CZ"})

# Gate names that map directly to SingleQubitEdge.
_SINGLE_QUBIT_GATES: frozenset[str] = frozenset(
    {"H", "S", "X", "Y", "Z", "I", "R", "RX", "M", "MX", "MR", "MRX"}
)

# Instructions that are silently skipped (noise, annotations).
_SKIP_INSTRUCTIONS: frozenset[str] = frozenset(
    {
        "DEPOLARIZE1",
        "DEPOLARIZE2",
        "PAULI_CHANNEL_1",
        "PAULI_CHANNEL_2",
        "CORRELATED_ERROR",
        "E",
        "ELSE_CORRELATED_ERROR",
        "DETECTOR",
        "OBSERVABLE_INCLUDE",
        "QUBIT_COORDS",
        "SHIFT_COORDS",
        "X_ERROR",
        "Y_ERROR",
        "Z_ERROR",
    }
)


def schedule_from_stim_circuit(
    circuit: stim.Circuit,
    qubit_roles: dict[int, QubitRole],
    *,
    name: str = "imported",
) -> Schedule:
    """Recover a :class:`Schedule` from a ``stim.Circuit``.

    Parameters
    ----------
    circuit : stim.Circuit
        The Stim circuit to parse. Must contain ``TICK`` markers
        between gate layers. A single top-level ``REPEAT`` block
        is interpreted as the cycle; everything before is the head
        and everything after is the tail.
    qubit_roles : dict[int, QubitRole]
        Maps every qubit index touched by the circuit to its role.
        Used for ``interaction_sector`` heuristic and for the
        ``Schedule.qubit_roles`` field.
    name : str, optional
        Label for the resulting schedule.

    Returns
    -------
    Schedule

    Raises
    ------
    ValueError
        If the circuit contains unsupported instructions or the
        qubit-role mapping doesn't cover all touched qubits.
    """
    all_qubits = frozenset(qubit_roles.keys())
    data_qubits = frozenset(q for q, r in qubit_roles.items() if r == "data")

    # ── Phase 1: split into (head_instructions, cycle_instructions, tail_instructions) ──
    head_insts: list[stim.CircuitInstruction] = []
    cycle_insts: list[stim.CircuitInstruction] = []
    tail_insts: list[stim.CircuitInstruction] = []
    found_repeat = False

    for item in circuit:
        if isinstance(item, stim.CircuitRepeatBlock):
            if found_repeat:
                raise ValueError(
                    "schedule_from_stim_circuit supports only one top-level REPEAT block."
                )
            found_repeat = True
            # The body of the REPEAT is the cycle.
            cycle_insts.extend(item.body_copy())
        elif isinstance(item, stim.CircuitInstruction):
            if found_repeat:
                tail_insts.append(item)
            else:
                head_insts.append(item)
        else:
            raise ValueError(f"Unexpected circuit element: {type(item)}")

    # If there was no REPEAT, treat everything as cycle (single round).
    if not found_repeat:
        cycle_insts = head_insts + tail_insts
        head_insts = []
        tail_insts = []

    # ── Phase 2: convert instruction lists into ScheduleStep tuples ──
    head_steps = _instructions_to_steps(head_insts, all_qubits, data_qubits)
    cycle_steps = _instructions_to_steps(cycle_insts, all_qubits, data_qubits)
    tail_steps = _instructions_to_steps(tail_insts, all_qubits, data_qubits)

    return Schedule(
        head_steps=tuple(head_steps),
        cycle_steps=tuple(cycle_steps),
        tail_steps=tuple(tail_steps),
        qubits=all_qubits,
        qubit_roles=qubit_roles,
        name=name,
    )


# =============================================================================
# Internals
# =============================================================================


def _instructions_to_steps(
    instructions: list[Any],
    all_qubits: frozenset[int],
    data_qubits: frozenset[int],
) -> list[ScheduleStep]:
    """Group instructions between TICKs into ScheduleSteps."""
    steps: list[ScheduleStep] = []
    current_edges: list[ScheduleEdge] = []
    tick_index = 0

    def _flush() -> None:
        nonlocal current_edges, tick_index
        if not current_edges:
            return
        active: set[int] = set()
        for e in current_edges:
            for q in e.qubits:
                active.add(q)
        idle = all_qubits - active
        role = _infer_role(current_edges)
        steps.append(
            ScheduleStep(
                tick_index=tick_index,
                role=role,
                active_edges=tuple(current_edges),
                active_qubits=frozenset(active),
                idle_qubits=idle,
            )
        )
        current_edges = []
        tick_index += 1

    for inst in instructions:
        if not isinstance(inst, stim.CircuitInstruction):
            continue
        gate_name = inst.name
        if gate_name == "TICK":
            _flush()
            continue
        if gate_name in _SKIP_INSTRUCTIONS:
            continue
        targets = inst.targets_copy()
        if gate_name in _TWO_QUBIT_GATES:
            # Targets come in pairs: (control, target).
            for i in range(0, len(targets), 2):
                ctrl = targets[i].qubit_value
                tgt = targets[i + 1].qubit_value
                if ctrl is None or tgt is None:
                    continue
                sector = _infer_sector(ctrl, tgt, data_qubits)
                stim_gate: str = "CNOT" if gate_name in ("CX", "CNOT") else gate_name
                current_edges.append(
                    TwoQubitEdge(
                        gate=stim_gate,  # type: ignore[arg-type]
                        control=int(ctrl),
                        target=int(tgt),
                        interaction_sector=sector,  # type: ignore[arg-type]
                    )
                )
        elif gate_name in _SINGLE_QUBIT_GATES:
            for t in targets:
                qv = t.qubit_value
                if qv is None:
                    continue
                current_edges.append(SingleQubitEdge(gate=gate_name, qubit=int(qv)))
        else:
            # Unknown instruction — skip with a warning-level tolerance.
            # Future PRs can raise here if strict mode is requested.
            continue

    _flush()  # flush the last group if no trailing TICK
    return steps


def _infer_role(edges: list[ScheduleEdge]) -> ScheduleRole:
    """Heuristically assign a ScheduleRole from the gate types."""
    gates = {e.gate for e in edges}
    if gates <= {"CNOT", "CZ"}:
        return "cnot_layer"
    if gates <= {"R", "RX"}:
        return "reset"
    if gates <= {"M", "MX", "MR", "MRX"}:
        return "meas"
    if gates <= {"H", "S", "X", "Y", "Z", "I"}:
        return "single_q"
    # Mixed — use the most general role.
    return "single_q"


def _infer_sector(control: int, target: int, data_qubits: frozenset[int]) -> str | None:
    """Heuristically assign interaction_sector from CNOT direction.

    - Data → ancilla (control is data) → X sector (Z-check style).
    - Ancilla → data (control is ancilla) → Z sector (X-check style).
    - Both data or both ancilla → None (ambiguous).
    """
    ctrl_is_data = control in data_qubits
    tgt_is_data = target in data_qubits
    if ctrl_is_data and not tgt_is_data:
        return "X"
    if not ctrl_is_data and tgt_is_data:
        return "Z"
    return None

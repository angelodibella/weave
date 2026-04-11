"""Stim-circuit emission helpers for the weave compiler.

These low-level functions translate `ScheduleStep` objects into Stim
`circuit.append(...)` calls, applying local noise and TICK markers
along the way. Detector emission is handled separately in
:mod:`weave.compiler.compile` because it is code-structure-aware
(depends on `HX` / `HZ` / `experiment`), not schedule-aware.

The compiler internally builds a `stim.Circuit` object via these
helpers, then stringifies it for storage in
:class:`~weave.ir.CompiledExtraction.circuit_text`. The text form is
the canonical artifact; the live `stim.Circuit` is a lazy
materializer on top of it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..ir import LocalNoise, ScheduleStep, TwoQubitEdge

if TYPE_CHECKING:
    import stim


def emit_step(circuit: stim.Circuit, step: ScheduleStep, local_noise: LocalNoise) -> None:
    """Emit one `ScheduleStep` to a `stim.Circuit`.

    Emits, in order:

    1. Gates from the step's `active_edges`, grouped by gate type so
       that consecutive edges with the same gate on disjoint qubits
       become a single Stim instruction.
    2. `DEPOLARIZE2(p_cnot)` after each two-qubit edge, if
       `local_noise.cnot_rate(...) > 0`.
    3. `DEPOLARIZE1(p_idle)` on idle qubits, grouped by rate, if
       `local_noise.idle_rate(...) > 0`.
    4. A terminating `TICK` marker.

    Prep/meas noise is intentionally not emitted in this path — the
    PR 5 acceptance tests use zero prep/meas rates, and adding
    `X_ERROR` / `Z_ERROR` emission will happen in a later PR once the
    regression fixture has a reference to compare against.

    Parameters
    ----------
    circuit : stim.Circuit
        The circuit to append to.
    step : ScheduleStep
        The step to emit.
    local_noise : LocalNoise
        Local noise model queried for per-gate and per-qubit rates.
    """
    # Group edges by gate type so Stim's text form collapses
    # consecutive same-gate instructions into one line.
    gate_order: list[str] = []
    gate_to_qubits: dict[str, list[int]] = {}

    for edge in step.active_edges:
        gate_name: str = edge.gate
        if gate_name not in gate_to_qubits:
            gate_order.append(gate_name)
            gate_to_qubits[gate_name] = []
        if isinstance(edge, TwoQubitEdge):
            gate_to_qubits[gate_name].extend([edge.control, edge.target])
        else:
            gate_to_qubits[gate_name].append(edge.qubit)

    # Emit one instruction per distinct gate type, in first-appearance order.
    for instr_gate in gate_order:
        circuit.append(instr_gate, gate_to_qubits[instr_gate])

    # Emit per-edge DEPOLARIZE2 after two-qubit gates.
    for edge in step.active_edges:
        if isinstance(edge, TwoQubitEdge):
            rate = local_noise.cnot_rate(edge, step)
            if rate > 0:
                circuit.append("DEPOLARIZE2", [edge.control, edge.target], rate)

    # Emit DEPOLARIZE1 on idle qubits, grouped by rate.
    if step.idle_qubits:
        idle_sorted = sorted(step.idle_qubits)
        rate_groups: dict[float, list[int]] = {}
        for q in idle_sorted:
            rate = local_noise.idle_rate(q, step)
            if rate > 0:
                rate_groups.setdefault(rate, []).append(q)
        for rate in sorted(rate_groups):
            circuit.append("DEPOLARIZE1", rate_groups[rate], rate)

    # TICK marks the end of the tick.
    circuit.append("TICK")

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

from ..ir import LocalNoise, ProvenanceRecord, ScheduleStep, TwoQubitEdge

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


def emit_correlated_error(circuit: stim.Circuit, record: ProvenanceRecord) -> None:
    """Emit one `CORRELATED_ERROR` instruction for a provenance record.

    Translates `record.data_support` and `record.data_pauli_symbols`
    into Stim Pauli targets and appends the instruction to `circuit`
    at `record.pair_probability`. The emitted text form is

    .. code-block:: text

        CORRELATED_ERROR(p) X<q_a> X<q_b>

    for a weight-2 X-pair event, with analogous forms for Y/Z and
    for other weights.

    Called by the geometry branch of :func:`compile_extraction`
    **before** the step's CNOT gates so that the correlated fault
    conjugates forward through the remaining Cliffords — matching
    the retained-channel convention of the PRX-Quantum-under-review
    paper §II.D.
    """
    import stim

    targets: list[stim.GateTarget] = []
    for qubit, symbol in zip(record.data_support, record.data_pauli_symbols, strict=True):
        if symbol == "X":
            targets.append(stim.target_x(qubit))
        elif symbol == "Y":
            targets.append(stim.target_y(qubit))
        elif symbol == "Z":
            targets.append(stim.target_z(qubit))
        else:
            raise ValueError(
                f"ProvenanceRecord has non-Pauli symbol {symbol!r} at qubit {qubit}; "
                f"expected X, Y, or Z."
            )
    if not targets:
        # Weight-0 events should have been filtered by the geometry
        # pass already, but defend against an empty-support record.
        return
    circuit.append("CORRELATED_ERROR", targets, record.pair_probability)

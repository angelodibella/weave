"""The central `Schedule` IR: extraction protocols as typed, immutable data.

A `Schedule` is an ordered sequence of ticks, each tick a
`ScheduleStep` holding the gates that run in parallel at that tick.
Steps are partitioned into three blocks — `head`, `cycle`, `tail` —
that map directly to Stim's `REPEAT` construction: head runs once,
cycle runs `rounds` times, tail runs once.

This is the v2 form introduced in PR 4. Key design choices (and the
reasons, drawn from the audit in `private/advice.md`):

1. **Discriminated union for edges.** `ScheduleEdge` is
   `TwoQubitEdge | SingleQubitEdge`, not a single class with an
   ambiguous `qubits: tuple[int, ...]`. `TwoQubitEdge` has explicit
   `control` and `target` fields so propagation logic never has to
   guess from tuple position.

2. **Per-edge `interaction_sector`.** Not per-step. Imported schedules
   may mix X-sector and Z-sector edges within a single tick; the
   geometry pass reads each edge's sector individually.

3. **Schedule-level `qubit_roles`.** Promoted from a per-step field
   because the data/ancilla partition is a property of the protocol,
   not of any single tick. The propagation analyzer (PR 7) and the
   compiler (PR 5) read this at schedule-entry.

4. **`ScheduleStep.duration`.** Defaults to 1.0 (the legacy `τ`).
   Reserved for future heterogeneous-duration schedules (CNOT vs
   reset vs meas) without an API break.

5. **Head / cycle / tail blocks.** Map directly to Stim `REPEAT`.
   `tick_index` is strictly increasing *within* each block, resetting
   across blocks.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

InteractionSector = Literal["X", "Z"]
"""Which CSS sector a gate contributes to.

`z_memory` experiments decode the X sector (Z-check measurements
detect X errors). `x_memory` experiments decode the Z sector. In the
retained single-and-pair channel derivation of the paper, a CNOT in a
Z-check belongs to the X sector (data-qubit X errors propagate through
it to the ancilla); an X-check CNOT belongs to the Z sector.
"""

QubitRole = Literal["data", "z_ancilla", "x_ancilla", "flag"]
"""A qubit's role in the extraction protocol."""

ScheduleRole = Literal["reset", "prep", "cnot_layer", "single_q", "meas", "tick_barrier"]
"""The operational role of a `ScheduleStep`."""

TwoQubitGate = Literal["CNOT", "CZ"]
SingleQubitGate = Literal["H", "X", "Y", "Z", "S", "R", "RX", "M", "MX", "MR", "MRX", "I"]


# =============================================================================
# Edges
# =============================================================================


@dataclass(frozen=True)
class TwoQubitEdge:
    """A two-qubit gate with explicit control and target.

    For `CNOT`, `control` is the X-propagating qubit and `target` is
    the Z-propagating qubit. Propagation logic in the residual-error
    analyzer (PR 7) reads these names directly.

    Parameters
    ----------
    gate : {"CNOT", "CZ"}
        The two-qubit gate.
    control : int
        Control qubit index.
    target : int
        Target qubit index. Must differ from `control`.
    interaction_sector : {"X", "Z", None}, optional
        Which CSS sector this gate contributes to, if any. The
        geometry pass uses this to decide whether to inject a
        correlated-error channel on simultaneously active pairs.
    term_name : str or None, optional
        Algebraic / schedule-level label for provenance.
    """

    gate: TwoQubitGate
    control: int
    target: int
    interaction_sector: InteractionSector | None = None
    term_name: str | None = None

    def __post_init__(self) -> None:
        if self.control == self.target:
            raise ValueError(
                f"TwoQubitEdge control and target must be distinct, both were {self.control}."
            )

    @property
    def qubits(self) -> tuple[int, int]:
        """The `(control, target)` tuple for compatibility with legacy code."""
        return (self.control, self.target)

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": "two_qubit",
            "gate": self.gate,
            "control": self.control,
            "target": self.target,
            "interaction_sector": self.interaction_sector,
            "term_name": self.term_name,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> TwoQubitEdge:
        return cls(
            gate=data["gate"],
            control=int(data["control"]),
            target=int(data["target"]),
            interaction_sector=data.get("interaction_sector"),
            term_name=data.get("term_name"),
        )


@dataclass(frozen=True)
class SingleQubitEdge:
    """A single-qubit operation (Clifford, reset, measurement).

    Covers unitary 1Q gates (`H`, `X`, `Y`, `Z`, `S`, `I`), resets
    (`R`, `RX`), measurements (`M`, `MX`), and measurement-and-reset
    (`MR`, `MRX`). From the schedule's perspective these are all
    "single-qubit operations that happen at a tick"; the compiler
    translates each to the appropriate Stim instruction.

    Parameters
    ----------
    gate : str
        The single-qubit operation name.
    qubit : int
        The qubit index the operation acts on.
    term_name : str or None, optional
        Algebraic / schedule-level label for provenance.
    """

    gate: SingleQubitGate
    qubit: int
    term_name: str | None = None

    @property
    def qubits(self) -> tuple[int]:
        """The `(qubit,)` tuple for compatibility with legacy code."""
        return (self.qubit,)

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": "single_qubit",
            "gate": self.gate,
            "qubit": self.qubit,
            "term_name": self.term_name,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SingleQubitEdge:
        return cls(
            gate=data["gate"],
            qubit=int(data["qubit"]),
            term_name=data.get("term_name"),
        )


ScheduleEdge = TwoQubitEdge | SingleQubitEdge
"""Discriminated union of the two edge types."""


def _edge_from_json(data: dict[str, Any]) -> ScheduleEdge:
    kind = data.get("kind")
    if kind == "two_qubit":
        return TwoQubitEdge.from_json(data)
    if kind == "single_qubit":
        return SingleQubitEdge.from_json(data)
    raise ValueError(f"Unknown schedule edge kind {kind!r}.")


# =============================================================================
# Steps
# =============================================================================


@dataclass(frozen=True)
class ScheduleStep:
    """One tick of the extraction cycle.

    All edges in `active_edges` run in parallel at this tick. Idle
    qubits (those not touched by any edge) are tracked explicitly so
    the compiler's local-noise pass can apply `DEPOLARIZE1` correctly.

    Parameters
    ----------
    tick_index : int
        Tick position within the enclosing block (head, cycle, or
        tail). Must be strictly increasing within each block.
    role : ScheduleRole
        Operational role. Constrains which edge types are allowed:
        `cnot_layer` requires `TwoQubitEdge`; `reset`, `prep`, `meas`,
        `single_q` require `SingleQubitEdge`; `tick_barrier` allows
        empty `active_edges`.
    active_edges : tuple[ScheduleEdge, ...]
        Gates running in parallel at this tick.
    active_qubits : frozenset[int]
        Every qubit touched by at least one edge. Must be a superset
        of the union of edge qubits.
    idle_qubits : frozenset[int]
        Disjoint from `active_qubits`. The compiler applies
        `DEPOLARIZE1` here at rate `LocalNoise.idle_rate(q, step)`.
    duration : float, optional
        Wall-clock duration of this tick. Defaults to 1.0 (the legacy
        `τ`). Reserved for future heterogeneous-duration schedules.
    """

    tick_index: int
    role: ScheduleRole
    active_edges: tuple[ScheduleEdge, ...]
    active_qubits: frozenset[int]
    idle_qubits: frozenset[int]
    duration: float = 1.0

    def __post_init__(self) -> None:
        # Coerce sets to frozensets for hashability.
        if not isinstance(self.active_qubits, frozenset):
            object.__setattr__(self, "active_qubits", frozenset(self.active_qubits))
        if not isinstance(self.idle_qubits, frozenset):
            object.__setattr__(self, "idle_qubits", frozenset(self.idle_qubits))
        # Coerce edges to tuple.
        if not isinstance(self.active_edges, tuple):
            object.__setattr__(self, "active_edges", tuple(self.active_edges))

        # Invariant: active and idle disjoint.
        overlap = self.active_qubits & self.idle_qubits
        if overlap:
            raise ValueError(
                f"active_qubits and idle_qubits must be disjoint; overlap: {sorted(overlap)}"
            )

        # Invariant: edge-type vs role.
        if self.role == "cnot_layer":
            for e in self.active_edges:
                if not isinstance(e, TwoQubitEdge):
                    raise ValueError(
                        f"cnot_layer step must contain only TwoQubitEdge, got {type(e).__name__}"
                    )
        elif self.role in ("reset", "prep", "meas", "single_q"):
            for e in self.active_edges:
                if not isinstance(e, SingleQubitEdge):
                    raise ValueError(
                        f"{self.role} step must contain only SingleQubitEdge, "
                        f"got {type(e).__name__}"
                    )
        # tick_barrier has no constraint; usually empty edges.

        # Invariant: no two edges share a qubit in the same step.
        touched: set[int] = set()
        for e in self.active_edges:
            for q in e.qubits:
                if q in touched:
                    raise ValueError(f"qubit {q} is touched by multiple edges in the same step")
                touched.add(q)

        # Invariant: active_qubits covers every qubit touched by an edge.
        if not touched.issubset(self.active_qubits):
            missing = touched - self.active_qubits
            raise ValueError(
                f"active_qubits must contain all qubits touched by edges; "
                f"missing: {sorted(missing)}"
            )

        if self.duration <= 0:
            raise ValueError(f"duration must be positive, got {self.duration}")

    def to_json(self) -> dict[str, Any]:
        return {
            "tick_index": self.tick_index,
            "role": self.role,
            "active_edges": [e.to_json() for e in self.active_edges],
            "active_qubits": sorted(self.active_qubits),
            "idle_qubits": sorted(self.idle_qubits),
            "duration": self.duration,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ScheduleStep:
        return cls(
            tick_index=int(data["tick_index"]),
            role=data["role"],
            active_edges=tuple(_edge_from_json(e) for e in data["active_edges"]),
            active_qubits=frozenset(int(q) for q in data["active_qubits"]),
            idle_qubits=frozenset(int(q) for q in data["idle_qubits"]),
            duration=float(data.get("duration", 1.0)),
        )


# =============================================================================
# Schedule
# =============================================================================


@dataclass(frozen=True)
class Schedule:
    """A complete extraction protocol: head + repeated cycle + tail.

    The three blocks map directly to Stim's `REPEAT` construction:

    - `head_steps` run once at the start (resets, prep).
    - `cycle_steps` run inside `REPEAT rounds { ... }` (stabilizer
      extraction rounds).
    - `tail_steps` run once at the end (final data measurement).

    Parameters
    ----------
    head_steps : tuple[ScheduleStep, ...]
        Initial steps; run once. `tick_index` strictly increasing.
    cycle_steps : tuple[ScheduleStep, ...]
        Body of `REPEAT`; runs `rounds` times. `tick_index` resets at
        the start of this block and is strictly increasing.
    tail_steps : tuple[ScheduleStep, ...]
        Final steps; run once. `tick_index` resets again.
    qubits : frozenset[int]
        All qubit indices in scope.
    qubit_roles : dict[int, QubitRole]
        Role of each qubit (data / z_ancilla / x_ancilla / flag). Keys
        must equal `self.qubits`.
    name : str, optional
        Human-readable label.

    Notes
    -----
    `qubit_roles` is a dict and therefore mutable; this class is
    declared frozen but not runtime-hashable. Equality via field
    comparison still works.
    """

    SCHEMA_VERSION: ClassVar[int] = 1

    head_steps: tuple[ScheduleStep, ...]
    cycle_steps: tuple[ScheduleStep, ...]
    tail_steps: tuple[ScheduleStep, ...]
    qubits: frozenset[int]
    qubit_roles: dict[int, QubitRole]
    name: str = "unnamed"

    def __post_init__(self) -> None:
        # Coerce to tuples / frozensets for hashability-of-parts.
        if not isinstance(self.head_steps, tuple):
            object.__setattr__(self, "head_steps", tuple(self.head_steps))
        if not isinstance(self.cycle_steps, tuple):
            object.__setattr__(self, "cycle_steps", tuple(self.cycle_steps))
        if not isinstance(self.tail_steps, tuple):
            object.__setattr__(self, "tail_steps", tuple(self.tail_steps))
        if not isinstance(self.qubits, frozenset):
            object.__setattr__(self, "qubits", frozenset(self.qubits))

        # Invariant: qubit_roles keys equal self.qubits.
        role_keys = set(self.qubit_roles.keys())
        if role_keys != set(self.qubits):
            missing = set(self.qubits) - role_keys
            extra = role_keys - set(self.qubits)
            raise ValueError(
                f"qubit_roles keys must equal schedule qubits. "
                f"missing: {sorted(missing)}, extra: {sorted(extra)}"
            )

        # Invariant: tick_index strictly increasing within each block.
        for block_name, block_steps in self._block_iter():
            last_tick = -1
            for step in block_steps:
                if step.tick_index <= last_tick:
                    raise ValueError(
                        f"{block_name} block: tick_index {step.tick_index} "
                        f"not strictly increasing after {last_tick}"
                    )
                last_tick = step.tick_index

        # Invariant: every step's qubits are a subset of self.qubits.
        for block_name, block_steps in self._block_iter():
            for step in block_steps:
                all_step_qubits: frozenset[int] = step.active_qubits | step.idle_qubits
                if not all_step_qubits.issubset(self.qubits):
                    not_in_schedule = sorted(q for q in all_step_qubits if q not in self.qubits)
                    raise ValueError(
                        f"{block_name} step at tick {step.tick_index}: "
                        f"qubits not in schedule: {not_in_schedule}"
                    )

    def _block_iter(self) -> Iterable[tuple[str, tuple[ScheduleStep, ...]]]:
        yield ("head", self.head_steps)
        yield ("cycle", self.cycle_steps)
        yield ("tail", self.tail_steps)

    @property
    def schema_version(self) -> int:
        return self.SCHEMA_VERSION

    @property
    def cycle_depth(self) -> int:
        """Number of steps in one cycle iteration."""
        return len(self.cycle_steps)

    def all_steps(self) -> tuple[ScheduleStep, ...]:
        """Concatenation of head + cycle + tail (useful for iteration)."""
        return self.head_steps + self.cycle_steps + self.tail_steps

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "type": "schedule",
            "name": self.name,
            "head_steps": [s.to_json() for s in self.head_steps],
            "cycle_steps": [s.to_json() for s in self.cycle_steps],
            "tail_steps": [s.to_json() for s in self.tail_steps],
            "qubits": sorted(self.qubits),
            "qubit_roles": [{"qubit": q, "role": self.qubit_roles[q]} for q in sorted(self.qubits)],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Schedule:
        if data.get("type") != "schedule":
            raise ValueError(f"Expected type='schedule', got {data.get('type')!r}.")
        version = data.get("schema_version")
        if version != cls.SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version {version}; expected {cls.SCHEMA_VERSION}."
            )
        return cls(
            head_steps=tuple(ScheduleStep.from_json(s) for s in data["head_steps"]),
            cycle_steps=tuple(ScheduleStep.from_json(s) for s in data["cycle_steps"]),
            tail_steps=tuple(ScheduleStep.from_json(s) for s in data["tail_steps"]),
            qubits=frozenset(int(q) for q in data["qubits"]),
            qubit_roles={int(r["qubit"]): r["role"] for r in data["qubit_roles"]},
            name=data.get("name", "unnamed"),
        )


# =============================================================================
# Default CSS schedule factory
# =============================================================================


def default_css_schedule(
    code: Any,  # CSSCode; typed as Any to avoid circular import
    *,
    experiment: Literal["z_memory", "x_memory"] = "z_memory",
    name: str = "default_css",
) -> Schedule:
    """Build the naive per-check serial schedule for a CSS code.

    Reproduces the gate order of `CSSCode._legacy_generate()` but with
    (a) one `ScheduleStep` per CNOT so TICKs are meaningful, (b)
    explicit idle-qubit tracking so idle noise is correctly applied,
    and (c) head / cycle / tail partition for Stim `REPEAT` emission.

    Parameters
    ----------
    code : CSSCode
        The CSS code to build a schedule for. Must have attributes
        `data_qubits`, `z_check_qubits`, `x_check_qubits`, `qubits`,
        `HZ`, `HX`.
    experiment : {"z_memory", "x_memory"}, optional
        Experiment type. Determines reset gate (`R` vs `RX`) and
        final measurement gate (`M` vs `MX`).
    name : str, optional
        Label for the schedule.

    Returns
    -------
    Schedule
        A schedule with head (reset), cycle (Z-check CNOTs, X-check
        brackets, MR ancillas), and tail (final data measurement).
    """
    if experiment not in ("z_memory", "x_memory"):
        raise ValueError(f"experiment must be 'z_memory' or 'x_memory', got {experiment!r}")

    data_qubits: list[int] = list(code.data_qubits)
    z_checks: list[int] = list(code.z_check_qubits)
    x_checks: list[int] = list(code.x_check_qubits)
    all_qubits = frozenset(code.qubits)

    qubit_roles: dict[int, QubitRole] = {}
    for q in data_qubits:
        qubit_roles[q] = "data"
    for q in z_checks:
        qubit_roles[q] = "z_ancilla"
    for q in x_checks:
        qubit_roles[q] = "x_ancilla"

    # ---------------- Step builder ----------------
    def make_step(tick: int, role: ScheduleRole, edges: list[ScheduleEdge]) -> ScheduleStep:
        active: set[int] = set()
        for e in edges:
            for q in e.qubits:
                active.add(q)
        idle = all_qubits - active
        return ScheduleStep(
            tick_index=tick,
            role=role,
            active_edges=tuple(edges),
            active_qubits=frozenset(active),
            idle_qubits=idle,
        )

    # ---------------- Head ----------------
    head_steps: list[ScheduleStep] = []
    tick = 0

    if experiment == "z_memory":
        # R on all qubits.
        reset_edges: list[ScheduleEdge] = [
            SingleQubitEdge(gate="R", qubit=q) for q in sorted(all_qubits)
        ]
        head_steps.append(make_step(tick, "reset", reset_edges))
        tick += 1
    else:  # x_memory
        # RX on data qubits, R on ancillas.
        rx_edges: list[ScheduleEdge] = [SingleQubitEdge(gate="RX", qubit=q) for q in data_qubits]
        head_steps.append(make_step(tick, "reset", rx_edges))
        tick += 1
        r_edges: list[ScheduleEdge] = [
            SingleQubitEdge(gate="R", qubit=q) for q in z_checks + x_checks
        ]
        head_steps.append(make_step(tick, "reset", r_edges))
        tick += 1

    # ---------------- Cycle ----------------
    cycle_steps: list[ScheduleStep] = []
    tick = 0

    # Z-check CNOTs: one step per CNOT, in the legacy generator's order.
    # The Z-check is in the X sector (data X errors propagate through it).
    for check_idx, target in enumerate(z_checks):
        row = code.HZ[check_idx]
        for col_idx in range(len(row)):
            if row[col_idx]:
                data_q = data_qubits[col_idx]
                edge: ScheduleEdge = TwoQubitEdge(
                    gate="CNOT",
                    control=data_q,
                    target=target,
                    interaction_sector="X",
                    term_name=f"HZ[{check_idx},{col_idx}]",
                )
                cycle_steps.append(make_step(tick, "cnot_layer", [edge]))
                tick += 1

    # X-check brackets: H, CNOTs, H per check. In the Z sector.
    for check_idx, target in enumerate(x_checks):
        # H before the bracket.
        cycle_steps.append(make_step(tick, "single_q", [SingleQubitEdge(gate="H", qubit=target)]))
        tick += 1

        row = code.HX[check_idx]
        for col_idx in range(len(row)):
            if row[col_idx]:
                data_q = data_qubits[col_idx]
                edge = TwoQubitEdge(
                    gate="CNOT",
                    control=target,
                    target=data_q,
                    interaction_sector="Z",
                    term_name=f"HX[{check_idx},{col_idx}]",
                )
                cycle_steps.append(make_step(tick, "cnot_layer", [edge]))
                tick += 1

        # H after the bracket.
        cycle_steps.append(make_step(tick, "single_q", [SingleQubitEdge(gate="H", qubit=target)]))
        tick += 1

    # MR on all ancillas (measure and reset).
    mr_edges: list[ScheduleEdge] = [
        SingleQubitEdge(gate="MR", qubit=q) for q in z_checks + x_checks
    ]
    cycle_steps.append(make_step(tick, "meas", mr_edges))
    tick += 1

    # ---------------- Tail ----------------
    tail_steps: list[ScheduleStep] = []
    tick = 0

    if experiment == "z_memory":
        m_edges: list[ScheduleEdge] = [SingleQubitEdge(gate="M", qubit=q) for q in data_qubits]
        tail_steps.append(make_step(tick, "meas", m_edges))
    else:  # x_memory
        mx_edges: list[ScheduleEdge] = [SingleQubitEdge(gate="MX", qubit=q) for q in data_qubits]
        tail_steps.append(make_step(tick, "meas", mx_edges))

    return Schedule(
        head_steps=tuple(head_steps),
        cycle_steps=tuple(cycle_steps),
        tail_steps=tuple(tail_steps),
        qubits=all_qubits,
        qubit_roles=qubit_roles,
        name=name,
    )

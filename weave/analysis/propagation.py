r"""Schedule-aware fault propagation through CSS extraction cycles.

Given a weave :class:`~weave.ir.Schedule` and a Pauli fault injected at
a specific tick, this module walks the remaining schedule steps,
applying Clifford conjugation to the fault at each gate and recording
any ancilla-measurement flips the fault induces. At the end of
propagation, the fault is decomposed into its data-qubit residual
(the "data-level image" of the fault) and its set of flipped ancilla
measurement outcomes.

This is the engine behind:

* **Verifying the weight-≤2 propagation assumption** of Theorem 1 in
  the PRX-Quantum-under-review paper *Geometry-induced correlated
  noise in qLDPC syndrome extraction* (Di Bella 2026): a single-pair
  geometry event in a sector-relevant CNOT layer should propagate to
  a data-level Pauli of weight ≤ 2 after ancilla elimination. The
  check is implemented in :mod:`weave.analysis.validation`.
* **Residual-error enumeration** in the sense of Strikis, Browne,
  Beverland (2026), arXiv:2603.05481, §IV.1, Definition 1. See
  :mod:`weave.analysis.residual`.
* **Generic effective-distance diagnostics** that work for any
  :class:`~weave.ir.Schedule`, not just the BB IBM depth-8 cycle.

Terminology
-----------
We use "Pauli propagation" in its original Gottesman-stabilizer sense:
Heisenberg-picture conjugation of a single Pauli operator through a
Clifford circuit. This is distinct from the recent "Pauli Propagation
framework" of Rudolph, Jones, Teng, Angrisani, Holmes (arXiv:2505.21606,
2025) and the "Majorana Propagation" line (Facelli–Fawzi
arXiv:2503.18939, Rudolph et al. arXiv:2602.04878), which are
classical-simulation techniques for evolving full observables. Those
tools track coefficient magnitudes of multi-Pauli decompositions and
apply truncations. Here we track a single Pauli fault exactly, which
is enough for the residual-model analysis we need.

References
----------
- D. Gottesman, PhD thesis, Caltech 1997, arXiv:quant-ph/9705052.
- A. Strikis, D. E. Browne, M. E. Beverland, *High-performance
  syndrome extraction circuits for quantum codes*, arXiv:2603.05481
  (2026).
- Di Bella, *Geometry-induced correlated noise in qLDPC syndrome
  extraction* (PRX Quantum, under review, 2026). Theorem 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..ir import (
    InteractionSector,
    Schedule,
    ScheduleStep,
    SingleQubitEdge,
    TwoQubitEdge,
)
from .pauli import (
    Pauli,
    measure_x,
    measure_z,
    propagate_cnot,
    propagate_h,
    propagate_i,
    propagate_s,
    propagate_x,
    propagate_y,
    propagate_z,
)

__all__ = [
    "AncillaFlip",
    "FaultLocation",
    "PropagationResult",
    "build_single_pair_fault",
    "propagate_fault",
    "propagate_single_pair_event",
]


ScheduleBlock = Literal["head", "cycle", "tail"]


# =============================================================================
# Result types
# =============================================================================


@dataclass(frozen=True)
class AncillaFlip:
    """One ancilla measurement flipped by the propagating fault.

    Records enough context to reconstruct which syndrome bit was
    affected: the block and tick where the measurement occurred, the
    qubit index, and the measurement basis.

    Parameters
    ----------
    block : {"head", "cycle", "tail"}
        Which schedule block contained the measurement.
    tick_index : int
        `ScheduleStep.tick_index` of the measurement's step.
    qubit : int
        Measured qubit.
    basis : {"X", "Z"}
        Measurement basis (`"Z"` for `M` / `MR`, `"X"` for `MX` / `MRX`).
    """

    block: ScheduleBlock
    tick_index: int
    qubit: int
    basis: Literal["X", "Z"]


@dataclass(frozen=True)
class FaultLocation:
    """Where a fault is injected into the schedule.

    Parameters
    ----------
    block : {"head", "cycle", "tail"}
        Which block the injection point belongs to.
    tick_index : int
        Tick within that block. The fault is applied *after* the gates
        at this tick execute but *before* the next tick begins, i.e.
        it replaces the noise channel that would otherwise sit between
        the two.
    """

    block: ScheduleBlock
    tick_index: int


@dataclass(frozen=True)
class PropagationResult:
    """Outcome of propagating a fault through a schedule.

    Parameters
    ----------
    initial_fault : Pauli
        The fault as injected at `injection_location`, on all schedule
        qubits (not just data).
    injection_location : FaultLocation
        Where the fault was injected.
    data_pauli : Pauli
        The residual Pauli on the code's *data qubits* after all
        subsequent gates have been applied and all ancillas have been
        measured and reset. Its support and weight are the quantities
        used by downstream residual-error analysis.
    full_pauli : Pauli
        The fault on *all* schedule qubits (data + ancillas) at the
        end of propagation. On a well-formed schedule this should
        have zero weight on ancilla qubits, because every ancilla is
        measured and cleared during the cycle/tail. Kept as a debug
        witness so tests can catch schedule bugs (e.g., a missing
        measurement leaving a residual on an ancilla).
    ancilla_flips : tuple[AncillaFlip, ...]
        The ancilla measurements this fault flipped, in the order they
        were encountered during propagation.
    steps_walked : int
        How many schedule steps the walker executed.
    """

    initial_fault: Pauli
    injection_location: FaultLocation
    data_pauli: Pauli
    full_pauli: Pauli
    ancilla_flips: tuple[AncillaFlip, ...]
    steps_walked: int

    @property
    def data_weight(self) -> int:
        """Weight of the residual fault on data qubits."""
        return self.data_pauli.weight

    @property
    def flipped_ancilla_ticks(self) -> frozenset[tuple[int, int]]:
        """`(tick_index, qubit)` pairs for the flipped ancilla measurements."""
        return frozenset((f.tick_index, f.qubit) for f in self.ancilla_flips)


# =============================================================================
# Core propagator
# =============================================================================


def propagate_fault(
    schedule: Schedule,
    initial_fault: Pauli,
    injection_location: FaultLocation,
    data_qubits: frozenset[int],
    *,
    end_block: ScheduleBlock = "tail",
) -> PropagationResult:
    """Propagate a fault through the remainder of a schedule.

    Starts at `injection_location` and walks every subsequent step in
    the `(head, cycle, tail)` order defined by the `Schedule`, up to
    and including `end_block`. At each step:

    - Clifford gates are applied in-order via the symplectic
      propagation rules (:func:`~weave.analysis.pauli.propagate_cnot`,
      :func:`~weave.analysis.pauli.propagate_h`, etc.).
    - Measurements (`M`, `MX`, `MR`, `MRX`) are checked for
      anticommutation with the current fault. Every measurement that
      anticommutes produces an :class:`AncillaFlip`; the corresponding
      qubit entries are then cleared in the fault (simulating either
      a reset or the end-of-circuit absorption of the measurement
      outcome).
    - Resets (`R`, `RX`) do nothing to a fault that has already been
      injected: the reset state is a stabilizer of the reset
      operation, not an operator that conjugates through. We skip
      them.
    - Identity / Pauli-type labels (`I`, `X`, `Y`, `Z`) are no-ops in
      the phase-free representation.

    Parameters
    ----------
    schedule : Schedule
        The schedule to walk.
    initial_fault : Pauli
        The fault as injected. Must have
        `num_qubits == len(schedule.qubits)`.
    injection_location : FaultLocation
        Block and tick at which the fault is injected. Ticks strictly
        less than this location (in the block order head → cycle →
        tail) are skipped, as are ticks in the same block with smaller
        `tick_index`.
    data_qubits : frozenset[int]
        The code's data qubit indices. Used to split `full_pauli`
        into `data_pauli`.
    end_block : {"head", "cycle", "tail"}, optional
        The last block to walk, inclusive. Defaults to `"tail"`
        (walk the whole remainder of the schedule). Downstream callers
        that only care about the state at the end of the cycle (e.g.
        the weight-$\\le 2$ validator, which excludes the terminal
        data measurement) pass `end_block="cycle"` to avoid walking
        into the tail. Must not precede `injection_location.block`.

    Returns
    -------
    PropagationResult
        The structured outcome.

    Raises
    ------
    ValueError
        If `initial_fault.num_qubits` does not match the schedule's
        qubit count, if `injection_location` refers to a block/tick
        that does not exist, or if `end_block` precedes
        `injection_location.block`.
    NotImplementedError
        If the walker encounters a gate it does not know how to
        propagate through. The set of supported gates is `{CNOT,
        H, S, X, Y, Z, I, R, RX, M, MX, MR, MRX}`; extend
        :func:`_apply_edge` to add more.
    """
    num_qubits = _schedule_num_qubits(schedule)
    if initial_fault.num_qubits != num_qubits:
        raise ValueError(
            f"initial_fault has {initial_fault.num_qubits} qubits; "
            f"schedule has {num_qubits} qubits."
        )

    blocks = _block_order(schedule)
    injection_block_index = _block_index(injection_location.block)
    end_block_index = _block_index(end_block)
    if end_block_index < injection_block_index:
        raise ValueError(
            f"end_block {end_block!r} precedes injection block {injection_location.block!r}."
        )

    current = initial_fault
    flips: list[AncillaFlip] = []
    steps_walked = 0

    for block_index, (block_name, steps) in enumerate(blocks):
        if block_index < injection_block_index:
            continue
        if block_index > end_block_index:
            break
        for step in steps:
            if (
                block_index == injection_block_index
                and step.tick_index < injection_location.tick_index
            ):
                continue
            current, new_flips = _apply_step(current, step, block_name)
            flips.extend(new_flips)
            steps_walked += 1

    # Split into data-level residual.
    data_pauli = _restrict_to(current, data_qubits)

    return PropagationResult(
        initial_fault=initial_fault,
        injection_location=injection_location,
        data_pauli=data_pauli,
        full_pauli=current,
        ancilla_flips=tuple(flips),
        steps_walked=steps_walked,
    )


# =============================================================================
# Single-pair geometry event helper
# =============================================================================


def build_single_pair_fault(
    edge_a: TwoQubitEdge,
    edge_b: TwoQubitEdge,
    sector: InteractionSector,
    num_qubits: int,
) -> Pauli:
    r"""Build the `P_e ⊗ P_{e'}` fault for a single-pair geometry event.

    Per Assumption 2 and Section II.D of the PRX-Quantum-under-review
    paper, the retained pair fault for two simultaneously active gate
    blocks `e, e'` in the X sector acts as `X ⊗ X` on the CNOT control
    qubits (the `q(L)` data qubits in the BB schedule), and in the
    Z sector as `Z ⊗ Z` on the CNOT target qubits (the `q(R)` data
    qubits in BB). These identifications come from propagating the
    microscopic block-Pauli `P̂_e = X_{q(L)} ⊗ I_{q(Z)}` (X sector) or
    `P̂_e = Z_{q(X)} ⊗ Z_{q(R)}` (Z sector) through the specific CSS
    CNOT direction.

    Here we implement the natural generalization to any CSS schedule:

    - **X sector.** Each edge's `control` qubit picks up an `X`.
    - **Z sector.** Each edge's `target` qubit picks up a `Z`.

    Both edges must have `interaction_sector == sector` or have it
    unset. The two edges must act on disjoint qubits (which is always
    the case within a single `ScheduleStep` because of the no-shared-
    qubit invariant).

    Parameters
    ----------
    edge_a, edge_b : TwoQubitEdge
        The two simultaneously-active gate blocks.
    sector : {"X", "Z"}
        The interaction sector.
    num_qubits : int
        Total number of qubits in the enclosing schedule.

    Returns
    -------
    Pauli
        The `P_e ⊗ P_{e'}` fault on the full-schedule Hilbert space.

    Raises
    ------
    ValueError
        If the two edges share a qubit, or if either edge's declared
        sector is incompatible with the requested one.
    """
    shared = set(edge_a.qubits) & set(edge_b.qubits)
    if shared:
        raise ValueError(
            f"single-pair fault requires disjoint edges; edges share qubits {sorted(shared)}."
        )
    for name, edge in (("edge_a", edge_a), ("edge_b", edge_b)):
        if edge.interaction_sector is not None and edge.interaction_sector != sector:
            raise ValueError(
                f"{name}.interaction_sector is {edge.interaction_sector!r} "
                f"but requested sector is {sector!r}."
            )

    if sector == "X":
        # X on each CNOT's control qubit.
        result = Pauli.single_x(edge_a.control, num_qubits) * Pauli.single_x(
            edge_b.control, num_qubits
        )
    elif sector == "Z":
        # Z on each CNOT's target qubit.
        result = Pauli.single_z(edge_a.target, num_qubits) * Pauli.single_z(
            edge_b.target, num_qubits
        )
    else:
        raise ValueError(f"sector must be 'X' or 'Z', got {sector!r}.")
    return result


def propagate_single_pair_event(
    schedule: Schedule,
    step_tick: int,
    edge_a: TwoQubitEdge,
    edge_b: TwoQubitEdge,
    sector: InteractionSector,
    data_qubits: frozenset[int],
    *,
    end_block: ScheduleBlock = "cycle",
) -> PropagationResult:
    """Propagate a `P_e ⊗ P_{e'}` fault through the rest of the cycle.

    Convenience wrapper that builds the initial fault via
    :func:`build_single_pair_fault` and dispatches to
    :func:`propagate_fault` with the injection located in the cycle
    block at `step_tick`. By default it stops at the end of the cycle
    (`end_block="cycle"`), which is the boundary used by the
    weight-$\\le 2$ validator: include all ancilla measurements, but
    not the terminal data measurement.

    Parameters
    ----------
    schedule : Schedule
        The schedule whose cycle contains the simultaneously-active
        edges.
    step_tick : int
        Tick in `schedule.cycle_steps` at which the edges are active.
    edge_a, edge_b : TwoQubitEdge
        The two edges forming the pair.
    sector : {"X", "Z"}
        The sector whose retained channel we are analysing.
    data_qubits : frozenset[int]
        Code's data qubit indices (used to split the residual).
    end_block : {"head", "cycle", "tail"}, optional
        Where to stop walking. Defaults to `"cycle"` (do not walk
        into the tail).

    Returns
    -------
    PropagationResult
    """
    num_qubits = _schedule_num_qubits(schedule)
    initial = build_single_pair_fault(edge_a, edge_b, sector, num_qubits)
    location = FaultLocation(block="cycle", tick_index=step_tick)
    return propagate_fault(
        schedule=schedule,
        initial_fault=initial,
        injection_location=location,
        data_qubits=data_qubits,
        end_block=end_block,
    )


# =============================================================================
# Internals
# =============================================================================


def _block_order(schedule: Schedule) -> list[tuple[ScheduleBlock, tuple[ScheduleStep, ...]]]:
    return [
        ("head", schedule.head_steps),
        ("cycle", schedule.cycle_steps),
        ("tail", schedule.tail_steps),
    ]


def _block_index(block: ScheduleBlock) -> int:
    return {"head": 0, "cycle": 1, "tail": 2}[block]


def _schedule_num_qubits(schedule: Schedule) -> int:
    """Qubits in the schedule. We use max index + 1 so qubit gaps are allowed."""
    if not schedule.qubits:
        return 0
    return max(schedule.qubits) + 1


def _apply_step(
    pauli: Pauli, step: ScheduleStep, block: ScheduleBlock
) -> tuple[Pauli, list[AncillaFlip]]:
    """Apply all gates in a single `ScheduleStep` to `pauli`.

    Returns the updated fault and the list of ancilla flips generated
    at this step.
    """
    current = pauli
    flips: list[AncillaFlip] = []
    for edge in step.active_edges:
        current, edge_flips = _apply_edge(current, edge, step, block)
        flips.extend(edge_flips)
    return current, flips


def _apply_edge(
    pauli: Pauli,
    edge: TwoQubitEdge | SingleQubitEdge,
    step: ScheduleStep,
    block: ScheduleBlock,
) -> tuple[Pauli, list[AncillaFlip]]:
    """Apply one edge (a single gate) to the running fault."""
    if isinstance(edge, TwoQubitEdge):
        if edge.gate == "CNOT":
            return propagate_cnot(pauli, edge.control, edge.target), []
        raise NotImplementedError(
            f"Gate {edge.gate!r} not supported by propagator. "
            f"Extend weave.analysis.propagation._apply_edge to add it."
        )

    # SingleQubitEdge
    q = edge.qubit
    gate = edge.gate
    if gate == "H":
        return propagate_h(pauli, q), []
    if gate == "S":
        return propagate_s(pauli, q), []
    if gate == "X":
        return propagate_x(pauli, q), []
    if gate == "Y":
        return propagate_y(pauli, q), []
    if gate == "Z":
        return propagate_z(pauli, q), []
    if gate == "I":
        return propagate_i(pauli, q), []
    if gate in ("R", "RX"):
        # Reset operations prepare a fresh |0⟩ or |+⟩ ancilla. For a
        # fault injected BEFORE the reset, the reset projects onto the
        # post-reset state and erases the fault on that qubit. For a
        # fault injected AFTER the reset, the reset is in the past and
        # does nothing. Since we never inject a fault *before* its
        # injection_location, and we never walk steps before
        # injection_location in this module, any reset encountered
        # during propagation is BEFORE any possible fault on that
        # qubit — so it has no effect. (We still consume the edge so
        # the caller can account for it.)
        return pauli, []
    if gate == "M":
        flipped, reduced = measure_z(pauli, q)
        flips = (
            [AncillaFlip(block=block, tick_index=step.tick_index, qubit=q, basis="Z")]
            if flipped
            else []
        )
        return reduced, flips
    if gate == "MX":
        flipped, reduced = measure_x(pauli, q)
        flips = (
            [AncillaFlip(block=block, tick_index=step.tick_index, qubit=q, basis="X")]
            if flipped
            else []
        )
        return reduced, flips
    if gate == "MR":
        # Measure-and-reset in Z basis: measure, clear, reset.
        # Clearing qubit's x/z entries inside `measure_z` already
        # models the combined operation.
        flipped, reduced = measure_z(pauli, q)
        flips = (
            [AncillaFlip(block=block, tick_index=step.tick_index, qubit=q, basis="Z")]
            if flipped
            else []
        )
        return reduced, flips
    if gate == "MRX":
        flipped, reduced = measure_x(pauli, q)
        flips = (
            [AncillaFlip(block=block, tick_index=step.tick_index, qubit=q, basis="X")]
            if flipped
            else []
        )
        return reduced, flips

    raise NotImplementedError(
        f"Single-qubit gate {gate!r} not supported by propagator. "
        f"Extend weave.analysis.propagation._apply_edge to add it."
    )


def _restrict_to(pauli: Pauli, qubit_set: frozenset[int]) -> Pauli:
    """Return a Pauli with only the components on qubits in `qubit_set`.

    Non-`qubit_set` entries are zeroed out. The total qubit count is
    preserved.
    """
    new_x = tuple(xi if i in qubit_set else False for i, xi in enumerate(pauli.x))
    new_z = tuple(zi if i in qubit_set else False for i, zi in enumerate(pauli.z))
    return Pauli(x=new_x, z=new_z)

r"""Syndrome extraction schedules for bivariate bicycle codes.

This module implements the *monomial-parallel* syndrome extraction
schedule for any :class:`~weave.codes.bb.BivariateBicycleCode`. The
schedule fires one CNOT layer per monomial appearing in the code's
polynomial data, so the cycle has exactly `|A| + |B|` Z-check CNOT
ticks (data → Z-ancilla) followed by the same number of X-check
CNOT ticks (X-ancilla → data), with Hadamard bracketing on the
X-ancillas and a final measure-and-reset on every ancilla.

Why monomial-parallel?
----------------------
For a BB code with `A = \sum_k x^{a_k^{(1)}} y^{a_k^{(2)}}` and
`B = \sum_k x^{b_k^{(1)}} y^{b_k^{(2)}}`, every Z-check at row `i`
(`\in \mathbb{Z}_l \times \mathbb{Z}_m`) has CNOTs

- from `L`-data at cell `i - b_k` for each monomial `b_k` of `B`, and
- from `R`-data at cell `i - a_k` for each monomial `a_k` of `A`.

If we fix a monomial and iterate over all `lm` Z-check ancillas in
parallel, the resulting CNOT layer is a permutation: each data
qubit is control of exactly one CNOT and each Z-ancilla is target
of exactly one CNOT. This parallelism comes entirely from the
abelian-group structure of the BB code — no further scheduling
magic is needed.

The symmetric picture holds for the X-checks via `H_X = [A \mid B]`:
each row `i` has an X-ancilla as control and `L`-data at cell
`i - a_k` or `R`-data at cell `i - b_k` as target.

Depth and relation to the Bravyi et al. depth-8 schedule
--------------------------------------------------------
The schedule produced here has cycle depth

.. code-block:: text

    1    (H bracket open for X-ancillas)
  + |A|  (Z-check CNOT layers for A monomials, from R-block data)
  + |B|  (Z-check CNOT layers for B monomials, from L-block data)
  + |A|  (X-check CNOT layers for A monomials, into L-block data)
  + |B|  (X-check CNOT layers for B monomials, into R-block data)
  + 1    (H bracket close)
  + 1    (MR on every ancilla)
    ────
    3 + 2 (|A| + |B|)        ticks per round.

For a Bravyi-Cross-Gambetta-Maslov-Rall-Yoder BB code (`|A| = |B| = 3`)
this is `3 + 12 = 15` ticks per round. The original paper's
`depth-8` schedule achieves a cycle of `8` ticks by *also*
interleaving Z-check and X-check CNOTs that act on disjoint qubit
sets in the same tick — a nontrivial data-qubit scheduling
optimization that is out of scope for the initial weave BB factory.
Shipping the depth-8 interleave as a separate
`bravyi_depth8_schedule(bb)` factory is left to a future PR; the
monomial-parallel schedule here is mathematically equivalent as a
syndrome extraction protocol (same stabilizers, same logical action)
and gives the geometry pass and the PR 10 reference-family
enumeration exactly the right sector-tagged edges to operate on.

The `name` kwarg of :func:`ibm_schedule` defaults to
``"<bb_name>_ibm_schedule"`` so downstream provenance tracks which
factory produced the schedule.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ...ir import (
    QubitRole,
    Schedule,
    ScheduleEdge,
    ScheduleStep,
    SingleQubitEdge,
    TwoQubitEdge,
)

if TYPE_CHECKING:
    from .bb_code import BivariateBicycleCode, Monomial


__all__ = ["ibm_schedule"]


def ibm_schedule(
    bb_code: BivariateBicycleCode,
    *,
    experiment: Literal["z_memory", "x_memory"] = "z_memory",
    name: str | None = None,
) -> Schedule:
    r"""Build the monomial-parallel syndrome extraction schedule.

    Parameters
    ----------
    bb_code : BivariateBicycleCode
        The code whose syndrome extraction we are describing.
    experiment : {"z_memory", "x_memory"}, optional
        Memory experiment type. Determines the head reset gates
        (`R` on ancillas, plus `RX` on data qubits for `x_memory`).
    name : str, optional
        Schedule label. Defaults to
        ``"<bb_code.name>_ibm_schedule"``.

    Returns
    -------
    Schedule
        A pure-data schedule whose `cycle_steps` cover one full
        syndrome extraction round. The `head_steps` perform the
        initial reset and (for `x_memory`) the data-qubit X-basis
        prep; `tail_steps` fire the final data measurement.
    """
    if experiment not in ("z_memory", "x_memory"):
        raise ValueError(f"experiment must be 'z_memory' or 'x_memory', got {experiment!r}")

    l = bb_code.l
    m = bb_code.m
    lm = l * m
    all_qubits: frozenset[int] = frozenset(bb_code.qubits)

    data_qubits: list[int] = list(bb_code.data_qubits)
    z_ancillas: list[int] = list(bb_code.z_check_qubits)
    x_ancillas: list[int] = list(bb_code.x_check_qubits)

    qubit_roles: dict[int, QubitRole] = {}
    for q in data_qubits:
        qubit_roles[q] = "data"
    for q in z_ancillas:
        qubit_roles[q] = "z_ancilla"
    for q in x_ancillas:
        qubit_roles[q] = "x_ancilla"

    # ---------------- Step builder ----------------

    def _make_step(
        tick: int,
        role: str,
        edges: list[ScheduleEdge],
    ) -> ScheduleStep:
        active: set[int] = set()
        for e in edges:
            for q in e.qubits:
                active.add(q)
        idle = all_qubits - active
        return ScheduleStep(
            tick_index=tick,
            role=role,  # type: ignore[arg-type]
            active_edges=tuple(edges),
            active_qubits=frozenset(active),
            idle_qubits=idle,
        )

    # ---------------- Head ----------------

    head: list[ScheduleStep] = []
    if experiment == "z_memory":
        head.append(
            _make_step(
                tick=0,
                role="reset",
                edges=[SingleQubitEdge(gate="R", qubit=q) for q in sorted(all_qubits)],
            )
        )
    else:  # x_memory
        head.append(
            _make_step(
                tick=0,
                role="reset",
                edges=[SingleQubitEdge(gate="RX", qubit=q) for q in data_qubits],
            )
        )
        head.append(
            _make_step(
                tick=1,
                role="reset",
                edges=[SingleQubitEdge(gate="R", qubit=q) for q in z_ancillas + x_ancillas],
            )
        )

    # ---------------- Cycle ----------------

    cycle: list[ScheduleStep] = []
    tick = 0

    # Open the X-check H bracket on every X-ancilla. This tick fires
    # `lm` Hadamards in parallel.
    cycle.append(
        _make_step(
            tick=tick,
            role="single_q",
            edges=[SingleQubitEdge(gate="H", qubit=q) for q in x_ancillas],
        )
    )
    tick += 1

    # --- Z-check CNOT layers ---
    # Z-check `i = (i1, i2)` is at z_ancillas[j*l + i1] where
    # j = i2. For monomial b_k = (b1, b2) in B, the CNOT is
    # CNOT(L[j - b2, i1 - b1], z[j, i1]). In flat column-major index:
    #   L_flat = (j - b2) * l + (i1 - b1)
    #   z_flat =  j * l + i1
    # where all subtractions are modulo (l, m).
    #
    # For each monomial, iterate over every (j, i1) pair — every
    # data qubit is hit exactly once per layer, so the CNOTs are
    # fully parallel.

    for k, b_mon in enumerate(bb_code.B_monomials):
        cycle.append(
            _z_check_layer(
                tick=tick,
                monomial=b_mon,
                block="L",
                family_label=f"B{k + 1}",
                l=l,
                m=m,
                lm=lm,
                data_qubits=data_qubits,
                z_ancillas=z_ancillas,
                all_qubits=all_qubits,
            )
        )
        tick += 1

    for k, a_mon in enumerate(bb_code.A_monomials):
        cycle.append(
            _z_check_layer(
                tick=tick,
                monomial=a_mon,
                block="R",
                family_label=f"A{k + 1}",
                l=l,
                m=m,
                lm=lm,
                data_qubits=data_qubits,
                z_ancillas=z_ancillas,
                all_qubits=all_qubits,
            )
        )
        tick += 1

    # --- X-check CNOT layers ---
    # X-check `i` has A-monomial CNOTs to L-data and B-monomial
    # CNOTs to R-data. CNOT direction is x-ancilla → data
    # (x-ancilla is control, data is target).

    for k, a_mon in enumerate(bb_code.A_monomials):
        cycle.append(
            _x_check_layer(
                tick=tick,
                monomial=a_mon,
                block="L",
                family_label=f"A{k + 1}",
                l=l,
                m=m,
                lm=lm,
                data_qubits=data_qubits,
                x_ancillas=x_ancillas,
                all_qubits=all_qubits,
            )
        )
        tick += 1

    for k, b_mon in enumerate(bb_code.B_monomials):
        cycle.append(
            _x_check_layer(
                tick=tick,
                monomial=b_mon,
                block="R",
                family_label=f"B{k + 1}",
                l=l,
                m=m,
                lm=lm,
                data_qubits=data_qubits,
                x_ancillas=x_ancillas,
                all_qubits=all_qubits,
            )
        )
        tick += 1

    # Close the X-check H bracket.
    cycle.append(
        _make_step(
            tick=tick,
            role="single_q",
            edges=[SingleQubitEdge(gate="H", qubit=q) for q in x_ancillas],
        )
    )
    tick += 1

    # Measure and reset every ancilla.
    cycle.append(
        _make_step(
            tick=tick,
            role="meas",
            edges=[SingleQubitEdge(gate="MR", qubit=q) for q in z_ancillas + x_ancillas],
        )
    )
    tick += 1

    # ---------------- Tail ----------------

    tail: list[ScheduleStep] = []
    if experiment == "z_memory":
        tail.append(
            _make_step(
                tick=0,
                role="meas",
                edges=[SingleQubitEdge(gate="M", qubit=q) for q in data_qubits],
            )
        )
    else:  # x_memory
        tail.append(
            _make_step(
                tick=0,
                role="meas",
                edges=[SingleQubitEdge(gate="MX", qubit=q) for q in data_qubits],
            )
        )

    return Schedule(
        head_steps=tuple(head),
        cycle_steps=tuple(cycle),
        tail_steps=tuple(tail),
        qubits=all_qubits,
        qubit_roles=qubit_roles,
        name=name or f"{bb_code.name}_ibm_schedule",
    )


# =============================================================================
# Internals: single monomial CNOT layer builders
# =============================================================================


def _z_check_layer(
    *,
    tick: int,
    monomial: Monomial,
    block: Literal["L", "R"],
    family_label: str,
    l: int,
    m: int,
    lm: int,
    data_qubits: list[int],
    z_ancillas: list[int],
    all_qubits: frozenset[int],
) -> ScheduleStep:
    r"""Build a full-parallel Z-check CNOT layer for one monomial.

    `HZ[i, c] = B[c, i] = 1` (for L-block `c`) or `A[c, i] = 1`
    (for R-block `c`) iff `c = i + (d_1, d_2)` for some monomial
    `(d_1, d_2)`. Therefore the Z-check at group element
    `i = (i_1, i_2)` receives a CNOT from the data qubit at
    `(i_1 + d_1, i_2 + d_2) \bmod (l, m)` for each monomial. Iterating
    over every `i` at fixed monomial gives a permutation CNOT layer
    with `lm` parallel CNOTs. Sector tag is ``"X"`` because a
    data-qubit X error propagates to the Z-ancilla.
    """
    d1, d2 = monomial
    edges: list[ScheduleEdge] = []
    block_base = 0 if block == "L" else lm
    for j in range(m):
        for i1 in range(l):
            z_anc = z_ancillas[j * l + i1]
            src_j = (j + d2) % m
            src_i1 = (i1 + d1) % l
            data_q = data_qubits[block_base + src_j * l + src_i1]
            edges.append(
                TwoQubitEdge(
                    gate="CNOT",
                    control=data_q,
                    target=z_anc,
                    interaction_sector="X",
                    term_name=family_label,
                )
            )
    active: set[int] = set()
    for e in edges:
        for q in e.qubits:
            active.add(q)
    return ScheduleStep(
        tick_index=tick,
        role="cnot_layer",
        active_edges=tuple(edges),
        active_qubits=frozenset(active),
        idle_qubits=all_qubits - active,
    )


def _x_check_layer(
    *,
    tick: int,
    monomial: Monomial,
    block: Literal["L", "R"],
    family_label: str,
    l: int,
    m: int,
    lm: int,
    data_qubits: list[int],
    x_ancillas: list[int],
    all_qubits: frozenset[int],
) -> ScheduleStep:
    r"""Build a full-parallel X-check CNOT layer for one monomial.

    `HX[i, c] = A[i, c] = 1` (for L-block `c`) or `B[i, c] = 1` (for
    R-block `c`) iff `i = c + (d_1, d_2)` for some monomial
    `(d_1, d_2)`, so `c = i - (d_1, d_2)`. Therefore the X-check at
    group element `i = (i_1, i_2)` drives a CNOT into the data qubit
    at `(i_1 - d_1, i_2 - d_2) \bmod (l, m)` for each monomial. The
    `x`-ancilla is control (post-Hadamard it's in the `|+\rangle`
    basis, so `CNOT` propagates its `|+\rangle` state into the data
    targets) and the sector tag is ``"Z"`` because a data-qubit Z
    error propagates back to the X-ancilla.
    """
    d1, d2 = monomial
    edges: list[ScheduleEdge] = []
    block_base = 0 if block == "L" else lm
    for j in range(m):
        for i1 in range(l):
            x_anc = x_ancillas[j * l + i1]
            src_j = (j - d2) % m
            src_i1 = (i1 - d1) % l
            data_q = data_qubits[block_base + src_j * l + src_i1]
            edges.append(
                TwoQubitEdge(
                    gate="CNOT",
                    control=x_anc,
                    target=data_q,
                    interaction_sector="Z",
                    term_name=family_label,
                )
            )
    active: set[int] = set()
    for e in edges:
        for q in e.qubits:
            active.add(q)
    return ScheduleStep(
        tick_index=tick,
        role="cnot_layer",
        active_edges=tuple(edges),
        active_qubits=frozenset(active),
        idle_qubits=all_qubits - active,
    )

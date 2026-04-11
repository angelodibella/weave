r"""Phase-free symplectic Pauli operators and Clifford gate propagation.

A Pauli operator on `n` qubits is represented by a *symplectic vector*
`(x, z) ∈ F₂ⁿ × F₂ⁿ` where

.. math::

    P \;=\; \prod_{i=0}^{n-1} X_i^{x_i} Z_i^{z_i}

(up to a global sign, which we do not track — see the note below). The
symplectic representation identifies the single-qubit Pauli group
modulo phases with `F₂²`:

    (0,0) = I,    (1,0) = X,    (0,1) = Z,    (1,1) = Y.

**Why phase-free.** For fault-propagation analysis the relevant datum
is *which* Pauli acts on each qubit, not its sign. The downstream
noise model (the retained single-and-pair channel in the paper) is a
convex mixture of Paulis with nonnegative probabilities; signs factor
out on both sides of the conjugation and never affect the Pauli
support. We therefore represent Paulis as elements of the symplectic
quotient `P_n / {±1, ±i} ≅ F₂^{2n}`.

**Commutation.** Two phase-free Paulis `P, P'` commute iff their
symplectic inner product vanishes:

.. math::

    \omega(P, P') \;=\; \sum_{i=0}^{n-1}\bigl(x_i z'_i + x'_i z_i\bigr) \pmod{2}.

They anticommute iff `ω = 1`. This is a bilinear form over `F₂` and
underlies all propagation and measurement bookkeeping in this module.

**Clifford propagation.** For each supported Clifford gate `C`, the
Heisenberg-picture conjugation `P ↦ C† P C` is an `F₂`-linear map on
the symplectic vector. This module provides one propagation function
per gate — :func:`propagate_cnot`, :func:`propagate_h`,
:func:`propagate_s`, :func:`propagate_x`, :func:`propagate_y`,
:func:`propagate_z`, and :func:`propagate_i`. Identity-gate variants
are no-ops. The rules are standard; see Gottesman (1997) Table 3.1
and Nielsen & Chuang §10.5.1.

**Measurements.** A projective measurement of `Z_q` on a fault `P`
flips the measurement outcome iff `P` anticommutes with `Z_q`, i.e.
iff `P` has an `X` or `Y` component on qubit `q`. After the
measurement, the qubit's `x_q` and `z_q` bits are cleared — this
correctly models a measure-and-reset (`MR`) operation in the stabilizer
extraction cycle, because the ancilla is freshly reset to `|0⟩` after
the measurement and any subsequent fault component on it is absorbed
by the reset. :func:`measure_z` and :func:`measure_x` return
`(flipped, reduced_pauli)`.

Notes on terminology
--------------------
"Pauli propagation" has a second, more recent meaning in the quantum
information literature — see Rudolph, Jones, Teng, Angrisani, Holmes,
*Pauli Propagation: A Computational Framework for Simulating Quantum
Systems*, arXiv:2505.21606 (2025), and the Majorana Propagation work of
Facelli–Fawzi (arXiv:2503.18939) and Rudolph et al. (arXiv:2602.04878)
— where it refers to a *classical simulation* technique that evolves a
full observable in the Heisenberg picture as a sum of Paulis with
tracked coefficients. That is **not** what this module does. Here we
track a single (one- or two-qubit) Pauli fault through a Clifford
circuit to determine its data-level image and which ancilla
measurements it flips. The underlying algebra (Clifford conjugation
of Paulis) is the same, but the usage context and data structures
differ substantially.

References
----------
- D. Gottesman, *Stabilizer Codes and Quantum Error Correction*, PhD
  thesis, Caltech (1997), arXiv:quant-ph/9705052. §III.1 for the
  symplectic representation; Table 3.1 for Clifford propagation rules.
- S. Aaronson, D. Gottesman, *Improved Simulation of Stabilizer
  Circuits*, Phys. Rev. A 70, 052328 (2004),
  arXiv:quant-ph/0406196. §III for binary-matrix stabilizer updates.
- M. A. Nielsen, I. L. Chuang, *Quantum Computation and Quantum
  Information*, 10th Anniversary Edition (Cambridge, 2010), §10.5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

__all__ = [
    "Pauli",
    "measure_x",
    "measure_z",
    "propagate_cnot",
    "propagate_h",
    "propagate_i",
    "propagate_s",
    "propagate_x",
    "propagate_y",
    "propagate_z",
]


@dataclass(frozen=True)
class Pauli:
    """Phase-free Pauli operator in symplectic representation.

    Two length-`n` tuples of booleans `x` and `z` encode the operator

    .. math:: P = \\prod_{i=0}^{n-1} X_i^{x_i} Z_i^{z_i}

    modulo the phase subgroup `{±1, ±i}`. The class is frozen and
    hashable; two `Pauli` instances compare equal iff their symplectic
    vectors are identical.

    Parameters
    ----------
    x : tuple[bool, ...]
        The X (and Y) support indicator.
    z : tuple[bool, ...]
        The Z (and Y) support indicator.

    Raises
    ------
    ValueError
        If `len(x) != len(z)`.
    """

    x: tuple[bool, ...]
    z: tuple[bool, ...]

    _SINGLE_SYMBOL: ClassVar[dict[tuple[bool, bool], str]] = {
        (False, False): "I",
        (True, False): "X",
        (False, True): "Z",
        (True, True): "Y",
    }

    def __post_init__(self) -> None:
        # Coerce to tuples of bools for consistency and hashability.
        if not isinstance(self.x, tuple):
            object.__setattr__(self, "x", tuple(bool(v) for v in self.x))
        else:
            object.__setattr__(self, "x", tuple(bool(v) for v in self.x))
        if not isinstance(self.z, tuple):
            object.__setattr__(self, "z", tuple(bool(v) for v in self.z))
        else:
            object.__setattr__(self, "z", tuple(bool(v) for v in self.z))
        if len(self.x) != len(self.z):
            raise ValueError(
                f"Pauli symplectic vectors must have equal length, "
                f"got len(x)={len(self.x)}, len(z)={len(self.z)}."
            )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def identity(cls, num_qubits: int) -> Pauli:
        """The identity Pauli on `num_qubits` qubits."""
        zeros = (False,) * num_qubits
        return cls(x=zeros, z=zeros)

    @classmethod
    def single_x(cls, qubit: int, num_qubits: int) -> Pauli:
        """`X_{qubit}` on `num_qubits` qubits."""
        x = [False] * num_qubits
        x[qubit] = True
        return cls(x=tuple(x), z=(False,) * num_qubits)

    @classmethod
    def single_y(cls, qubit: int, num_qubits: int) -> Pauli:
        """`Y_{qubit}` on `num_qubits` qubits."""
        x = [False] * num_qubits
        z = [False] * num_qubits
        x[qubit] = z[qubit] = True
        return cls(x=tuple(x), z=tuple(z))

    @classmethod
    def single_z(cls, qubit: int, num_qubits: int) -> Pauli:
        """`Z_{qubit}` on `num_qubits` qubits."""
        z = [False] * num_qubits
        z[qubit] = True
        return cls(x=(False,) * num_qubits, z=tuple(z))

    @classmethod
    def from_string(cls, s: str) -> Pauli:
        """Parse a Pauli string like ``"IXZYI"`` into a `Pauli`.

        Each character must be one of ``"IXYZ"``. The resulting Pauli
        has `num_qubits == len(s)` qubits.
        """
        x = []
        z = []
        for i, ch in enumerate(s):
            if ch == "I":
                x.append(False)
                z.append(False)
            elif ch == "X":
                x.append(True)
                z.append(False)
            elif ch == "Z":
                x.append(False)
                z.append(True)
            elif ch == "Y":
                x.append(True)
                z.append(True)
            else:
                raise ValueError(
                    f"Unknown Pauli symbol {ch!r} at position {i}; expected one of I, X, Y, Z."
                )
        return cls(x=tuple(x), z=tuple(z))

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    @property
    def num_qubits(self) -> int:
        """Number of qubits the operator acts on."""
        return len(self.x)

    @property
    def weight(self) -> int:
        """Hamming weight: number of qubits acted on nontrivially.

        Counts each qubit `i` for which the single-qubit operator is
        not identity (that is, `x_i ∨ z_i = 1`).
        """
        return sum(1 for xi, zi in zip(self.x, self.z, strict=True) if xi or zi)

    @property
    def support(self) -> frozenset[int]:
        """Set of qubits on which the operator acts nontrivially."""
        return frozenset(
            i for i, (xi, zi) in enumerate(zip(self.x, self.z, strict=True)) if xi or zi
        )

    @property
    def x_support(self) -> frozenset[int]:
        """Set of qubits with an `X` or `Y` component."""
        return frozenset(i for i, xi in enumerate(self.x) if xi)

    @property
    def z_support(self) -> frozenset[int]:
        """Set of qubits with a `Z` or `Y` component."""
        return frozenset(i for i, zi in enumerate(self.z) if zi)

    def is_identity(self) -> bool:
        """True iff the operator is the `n`-qubit identity."""
        return not any(self.x) and not any(self.z)

    def pauli_on(self, qubit: int) -> str:
        """Return the single-qubit Pauli symbol (`"I"`, `"X"`, `"Y"`, or `"Z"`)."""
        return self._SINGLE_SYMBOL[(self.x[qubit], self.z[qubit])]

    # ------------------------------------------------------------------
    # Algebra
    # ------------------------------------------------------------------

    def __mul__(self, other: Pauli) -> Pauli:
        """Phase-free Pauli product: XOR the symplectic vectors.

        Ignores the `±1, ±i` phase that may arise from noncommuting
        factors; see the module docstring for why that is the right
        convention for fault-propagation analysis.
        """
        if self.num_qubits != other.num_qubits:
            raise ValueError(
                f"Cannot multiply Paulis on different qubit counts: "
                f"{self.num_qubits} vs {other.num_qubits}."
            )
        new_x = tuple(a ^ b for a, b in zip(self.x, other.x, strict=True))
        new_z = tuple(a ^ b for a, b in zip(self.z, other.z, strict=True))
        return Pauli(x=new_x, z=new_z)

    def anticommutes_with(self, other: Pauli) -> bool:
        """True iff `self` and `other` anticommute.

        Computes the symplectic inner product over `F₂`:

        .. math:: \\omega(P, P') = \\sum_i (x_i z'_i + x'_i z_i) \\pmod{2}.

        Returns `True` when `ω = 1`.
        """
        if self.num_qubits != other.num_qubits:
            raise ValueError(
                f"Cannot compare Paulis on different qubit counts: "
                f"{self.num_qubits} vs {other.num_qubits}."
            )
        total = 0
        for i in range(self.num_qubits):
            if self.x[i] and other.z[i]:
                total ^= 1
            if other.x[i] and self.z[i]:
                total ^= 1
        return total == 1

    def commutes_with(self, other: Pauli) -> bool:
        """True iff `self` and `other` commute (symplectic inner product = 0)."""
        return not self.anticommutes_with(other)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self.num_qubits == 0:
            return "Pauli('')"
        s = "".join(self.pauli_on(i) for i in range(self.num_qubits))
        return f"Pauli('{s}')"


# =============================================================================
# Clifford gate propagation (Heisenberg picture)
# =============================================================================
#
# For each gate `U`, the conjugation `P ↦ U† P U` is an F₂-linear map on the
# symplectic vector. The rules below are standard and taken directly from
# Gottesman (1997) Table 3.1. We state each rule both as a Pauli equation
# and as its symplectic-vector update, so reviewers can verify by eye.


def propagate_i(pauli: Pauli, qubit: int) -> Pauli:  # noqa: ARG001
    """Identity gate: no propagation. Returns `pauli` unchanged.

    Used by the schedule walker to handle `"I"` marker gates
    explicitly so that "do nothing" is a typed operation, not an
    implicit skip.
    """
    return pauli


def propagate_x(pauli: Pauli, qubit: int) -> Pauli:  # noqa: ARG001
    """Pauli-X gate: no propagation in the phase-free representation.

    `X` commutes with `X` and `Y`; it anticommutes with `Z` and `I`, but
    anticommutation only flips the *phase* of the Heisenberg-picture
    image, which we do not track. Therefore `X† P X = ±P` and the
    phase-free symplectic representation returns `P` unchanged.

    Included as a typed operation so the schedule walker can handle
    `X`-gate schedule steps (e.g., in schedules imported from external
    tools) without a special case.
    """
    return pauli


def propagate_y(pauli: Pauli, qubit: int) -> Pauli:  # noqa: ARG001
    """Pauli-Y gate: no propagation in the phase-free representation."""
    return pauli


def propagate_z(pauli: Pauli, qubit: int) -> Pauli:  # noqa: ARG001
    """Pauli-Z gate: no propagation in the phase-free representation."""
    return pauli


def propagate_h(pauli: Pauli, qubit: int) -> Pauli:
    """Hadamard conjugation: swaps the `X` and `Z` components on `qubit`.

    Pauli rules (Gottesman 1997, Table 3.1):

    .. math::

        H X H = Z, \\quad H Z H = X, \\quad H Y H = -Y.

    Symplectic update (sign discarded):

    .. math:: (x_q, z_q) \\;\\mapsto\\; (z_q, x_q).
    """
    _check_qubit(pauli, qubit)
    new_x = list(pauli.x)
    new_z = list(pauli.z)
    new_x[qubit], new_z[qubit] = pauli.z[qubit], pauli.x[qubit]
    return Pauli(x=tuple(new_x), z=tuple(new_z))


def propagate_s(pauli: Pauli, qubit: int) -> Pauli:
    """Phase gate (`S`) conjugation:

    .. math::

        S X S^\\dagger = Y, \\quad S Z S^\\dagger = Z.

    Symplectic update (sign discarded):

    .. math:: (x_q, z_q) \\;\\mapsto\\; (x_q,\\; x_q \\oplus z_q).

    The `X` component is unchanged; the `Z` component is XORed with the
    `X` component, turning `X` into `Y` and leaving `Z` fixed.
    """
    _check_qubit(pauli, qubit)
    new_z = list(pauli.z)
    new_z[qubit] = pauli.x[qubit] ^ pauli.z[qubit]
    return Pauli(x=pauli.x, z=tuple(new_z))


def propagate_cnot(pauli: Pauli, control: int, target: int) -> Pauli:
    """CNOT (controlled-`X`) conjugation.

    Pauli rules (Gottesman 1997, Table 3.1):

    .. math::

        X_c \\;\\mapsto\\; X_c X_t, \\quad X_t \\;\\mapsto\\; X_t,
        \\quad Z_c \\;\\mapsto\\; Z_c, \\quad Z_t \\;\\mapsto\\; Z_c Z_t.

    In words: `X` on the control spreads to the target; `Z` on the
    target spreads to the control. Symplectic update:

    .. math::

        x_t &\\;\\mapsto\\; x_t \\oplus x_c, \\\\
        z_c &\\;\\mapsto\\; z_c \\oplus z_t,

    with `x_c` and `z_t` unchanged. This is the single most important
    propagation rule in stabilizer fault analysis.

    Raises
    ------
    ValueError
        If `control == target`.
    """
    if control == target:
        raise ValueError(f"CNOT control and target must differ, both were {control}.")
    _check_qubit(pauli, control)
    _check_qubit(pauli, target)
    new_x = list(pauli.x)
    new_z = list(pauli.z)
    new_x[target] = pauli.x[target] ^ pauli.x[control]
    new_z[control] = pauli.z[control] ^ pauli.z[target]
    return Pauli(x=tuple(new_x), z=tuple(new_z))


# =============================================================================
# Measurements and ancilla elimination
# =============================================================================


def measure_z(pauli: Pauli, qubit: int) -> tuple[bool, Pauli]:
    """Measure `Z_{qubit}`, returning `(flipped, reduced)`.

    The measurement outcome is flipped iff the fault `P` anticommutes
    with `Z_q`, which happens exactly when `P` has an `X` or `Y`
    component on qubit `q` (i.e. `x_q = 1`).

    The returned `reduced` Pauli has `x_q` and `z_q` both cleared.
    This models the measure-and-reset semantics used in the stabilizer
    extraction cycle: after `MR`, the ancilla is freshly reset to
    `|0⟩`, so any residual fault component on it is absorbed by the
    reset and has no effect on subsequent propagation.

    Parameters
    ----------
    pauli : Pauli
        The fault immediately before the measurement.
    qubit : int
        The qubit being measured.

    Returns
    -------
    flipped : bool
        `True` iff the measurement record is flipped by this fault.
    reduced : Pauli
        The fault after ancilla elimination.
    """
    _check_qubit(pauli, qubit)
    flipped = pauli.x[qubit]
    new_x = list(pauli.x)
    new_z = list(pauli.z)
    new_x[qubit] = False
    new_z[qubit] = False
    return flipped, Pauli(x=tuple(new_x), z=tuple(new_z))


def measure_x(pauli: Pauli, qubit: int) -> tuple[bool, Pauli]:
    """Measure `X_{qubit}`, returning `(flipped, reduced)`.

    Symmetric to :func:`measure_z`: the outcome is flipped iff the
    fault anticommutes with `X_q`, which happens when `P` has a `Z`
    or `Y` component (`z_q = 1`).
    """
    _check_qubit(pauli, qubit)
    flipped = pauli.z[qubit]
    new_x = list(pauli.x)
    new_z = list(pauli.z)
    new_x[qubit] = False
    new_z[qubit] = False
    return flipped, Pauli(x=tuple(new_x), z=tuple(new_z))


def _check_qubit(pauli: Pauli, qubit: int) -> None:
    if not 0 <= qubit < pauli.num_qubits:
        raise IndexError(f"qubit {qubit} out of range [0, {pauli.num_qubits}).")

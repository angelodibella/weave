"""Noise configuration objects for the geometry-aware compiler.

Two configs, separated by mechanism:

* :class:`LocalNoiseConfig` — the local depolarizing rates applied by the
  compiler to gates and idle qubits (`DEPOLARIZE2` after each CNOT,
  `DEPOLARIZE1` on idle qubits, prep/meas single-Pauli errors).
* :class:`GeometryNoiseConfig` — the physical parameters `(J₀, τ)` and
  configuration flags governing the retained single-and-pair channel
  derived from the routed embedding and proximity kernel.

Both are frozen dataclasses with validation in `__post_init__`,
class-level `SCHEMA_VERSION`, and full JSON round-trip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .schedule import ScheduleStep, TwoQubitEdge


@runtime_checkable
class LocalNoise(Protocol):
    """Structural type for a local-noise model queried by the compiler.

    Any object with `cnot_rate`, `idle_rate`, `prep_rate`, `meas_rate`,
    and `to_json` methods satisfies this protocol.
    :class:`LocalNoiseConfig` is the scalar-uniform implementation
    shipped with v1; richer per-qubit, per-edge, or duration-aware
    implementations can be plugged in later without changing the
    compiler's public API.

    Notes
    -----
    The `edge` and `step` parameters give implementations the full
    context of the gate being queried, but uniform implementations are
    free to ignore them. The compiler queries this protocol once per
    gate / idle qubit / prep / meas operation during the schedule walk.
    """

    def cnot_rate(self, edge: TwoQubitEdge, step: ScheduleStep) -> float:
        """Two-qubit depolarizing rate for a CNOT at this tick."""
        ...

    def idle_rate(self, qubit: int, step: ScheduleStep) -> float:
        """Single-qubit depolarizing rate for an idle qubit at this tick."""
        ...

    def prep_rate(self, qubit: int, step: ScheduleStep) -> float:
        """Prep-error probability for a qubit being reset at this tick."""
        ...

    def meas_rate(self, qubit: int, step: ScheduleStep) -> float:
        """Measurement-error probability for a qubit being measured at this tick."""
        ...

    def to_json(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        ...


GeometryScope = Literal["theory_reduced", "full_cycle"]
"""Sector scope for geometry-induced noise insertion.

``"theory_reduced"`` inserts correlated channels only on the
sector-relevant `B`-rounds (matches the bbstim default and keeps the
retained-channel derivation applicable line-for-line).

``"full_cycle"`` inserts channels on every simultaneously active block
pair in the whole cycle; useful for audits and for testing whether the
theory-reduced scope is capturing all of the physics.
"""


@dataclass(frozen=True)
class LocalNoiseConfig:
    """Local depolarizing rates applied uniformly across the extraction cycle.

    The compiler emits `DEPOLARIZE2` after each CNOT at rate `p_cnot`,
    `DEPOLARIZE1` on each idle qubit per tick at rate `p_idle`, and
    single-Pauli errors on preparation and measurement at `p_prep` and
    `p_meas` respectively.

    Every rate is a probability in `[0, 1]` and may be zero; construction
    with a negative or >1 value raises `ValueError`.

    Parameters
    ----------
    p_cnot : float
        Two-qubit depolarizing rate applied after each CNOT gate.
    p_idle : float
        Single-qubit depolarizing rate applied to idle qubits per tick.
    p_prep : float
        Probability of a single-Pauli error at the preparation step.
    p_meas : float
        Probability of a single-Pauli error at the measurement step.
    """

    SCHEMA_VERSION: ClassVar[int] = 1

    p_cnot: float = 0.0
    p_idle: float = 0.0
    p_prep: float = 0.0
    p_meas: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("p_cnot", self.p_cnot),
            ("p_idle", self.p_idle),
            ("p_prep", self.p_prep),
            ("p_meas", self.p_meas),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")

    @property
    def schema_version(self) -> int:
        return self.SCHEMA_VERSION

    # ------------------------------------------------------------------
    # LocalNoise protocol implementation
    # ------------------------------------------------------------------
    # Scalar-uniform: every gate of the same kind has the same rate,
    # regardless of which edge / qubit / step is being queried.

    def cnot_rate(self, edge: TwoQubitEdge, step: ScheduleStep) -> float:
        return self.p_cnot

    def idle_rate(self, qubit: int, step: ScheduleStep) -> float:
        return self.p_idle

    def prep_rate(self, qubit: int, step: ScheduleStep) -> float:
        return self.p_prep

    def meas_rate(self, qubit: int, step: ScheduleStep) -> float:
        return self.p_meas

    # ------------------------------------------------------------------
    # JSON serialization
    # ------------------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "type": "local_noise",
            "p_cnot": self.p_cnot,
            "p_idle": self.p_idle,
            "p_prep": self.p_prep,
            "p_meas": self.p_meas,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> LocalNoiseConfig:
        _validate_config_json(data, expected_type="local_noise", schema_version=cls.SCHEMA_VERSION)
        return cls(
            p_cnot=float(data.get("p_cnot", 0.0)),
            p_idle=float(data.get("p_idle", 0.0)),
            p_prep=float(data.get("p_prep", 0.0)),
            p_meas=float(data.get("p_meas", 0.0)),
        )


@dataclass(frozen=True)
class GeometryNoiseConfig:
    """Physical parameters for the geometry-induced retained channel.

    The retained pair probability at routed separation `d` is

    .. math::

        p(d) = \\sin^2(\\tau \\cdot J_0 \\cdot \\kappa(d))

    (or its weak-coupling quadratic approximation if `use_weak_limit` is
    set). This config carries the physical inputs to that formula; the
    kernel `κ` is a separate IR object passed to the compiler alongside
    this config.

    Parameters
    ----------
    J0 : float
        Microscopic coupling scale. Must be nonnegative. Zero disables
        geometry-induced noise regardless of the kernel.
    tau : float
        Tick duration. Must be strictly positive.
    use_weak_limit : bool
        If True, replace `sin²(τ J₀ κ(d))` with `(τ J₀ κ(d))²`. Faster
        and matches the weak-coupling bound used for the AKP-style
        compatibility argument.
    geometry_scope : {"theory_reduced", "full_cycle"}
        Whether to insert correlated channels only on the sector-relevant
        `B`-rounds (``"theory_reduced"``) or on every simultaneously
        active block pair in the whole cycle (``"full_cycle"``).
    """

    SCHEMA_VERSION: ClassVar[int] = 1

    J0: float = 0.0
    tau: float = 1.0
    use_weak_limit: bool = False
    geometry_scope: GeometryScope = "full_cycle"

    def __post_init__(self) -> None:
        if self.J0 < 0:
            raise ValueError(f"J0 must be non-negative, got {self.J0}.")
        if self.tau <= 0:
            raise ValueError(f"tau must be positive, got {self.tau}.")
        if self.geometry_scope not in ("theory_reduced", "full_cycle"):
            raise ValueError(
                f"geometry_scope must be 'theory_reduced' or 'full_cycle', "
                f"got {self.geometry_scope!r}."
            )

    @property
    def enabled(self) -> bool:
        """True iff geometry-induced noise would produce nonzero pair channels."""
        return self.J0 > 0.0

    @property
    def schema_version(self) -> int:
        return self.SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "type": "geometry_noise",
            "J0": self.J0,
            "tau": self.tau,
            "use_weak_limit": self.use_weak_limit,
            "geometry_scope": self.geometry_scope,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> GeometryNoiseConfig:
        _validate_config_json(
            data, expected_type="geometry_noise", schema_version=cls.SCHEMA_VERSION
        )
        scope = data.get("geometry_scope", "full_cycle")
        if scope not in ("theory_reduced", "full_cycle"):
            raise ValueError(
                f"geometry_scope must be 'theory_reduced' or 'full_cycle', got {scope!r}."
            )
        return cls(
            J0=float(data.get("J0", 0.0)),
            tau=float(data.get("tau", 1.0)),
            use_weak_limit=bool(data.get("use_weak_limit", False)),
            geometry_scope=scope,
        )


def _validate_config_json(data: dict[str, Any], *, expected_type: str, schema_version: int) -> None:
    """Shared type/version validator for noise config `from_json` methods."""
    actual_type = data.get("type")
    if actual_type != expected_type:
        raise ValueError(f"Expected type={expected_type!r}, got {actual_type!r}.")
    actual_version = data.get("schema_version")
    if actual_version != schema_version:
        raise ValueError(f"Unsupported schema_version {actual_version}; expected {schema_version}.")

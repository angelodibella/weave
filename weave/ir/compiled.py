"""Canonical pure-data output from the weave compiler.

`CompiledExtraction` is the output type of
:func:`weave.compiler.compile_extraction`. It holds Stim artifacts in
their canonical *text* form (`circuit_text`, `dem_text`) plus input
specification fingerprints, and exposes lazy materializers that parse
the text back into live `stim.Circuit` / `stim.DetectorErrorModel`
objects on demand.

Why pure data
-------------
* **JSON round-trip is native.** No special handling for live Stim
  objects. `to_json` / `from_json` just work.
* **Deterministic equality.** Two `CompiledExtraction` instances are
  equal iff their text and fingerprint fields are equal — enabling
  byte-for-byte regression diffs in CI.
* **Cheap fingerprints.** `fingerprint()` returns the SHA256 of the
  canonical JSON, suitable for committing to benchmark fixtures.
* **Lazy materialization.** The `circuit` / `dem` properties parse the
  stored text on first access and cache the result in a private dict.

Current shape (PR 8)
--------------------
Schema v2 adds the `provenance` field: a tuple of
:class:`ProvenanceRecord`, one per pair event emitted as a Stim
`CORRELATED_ERROR` by the geometry pass. Older v1 artifacts still
load via a default-empty `provenance` fallback in `from_json`. PR 9
will add `correlation_edges`, `exposure_metrics`, and
`decoder_artifact` and bump to v3.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

if TYPE_CHECKING:
    import stim  # noqa: F401


# =============================================================================
# ProvenanceRecord
# =============================================================================


InteractionSector = Literal["X", "Z"]


@dataclass(frozen=True)
class ProvenanceRecord:
    """Record of one emitted pair-fault event from the geometry pass.

    Each record corresponds 1:1 with a `CORRELATED_ERROR` instruction
    emitted into `CompiledExtraction.circuit_text` by a single round
    of the compiled extraction cycle. (When `rounds > 1`, the same
    record matches multiple circuit lines — one per round — because
    the geometry pass collapses per-round duplication.)

    Parameters
    ----------
    tick_index : int
        `ScheduleStep.tick_index` of the CNOT layer where the pair
        fault was injected (i.e. the tick immediately before the
        CORRELATED_ERROR fires).
    edge_a, edge_b : tuple[int, int]
        `(control, target)` pairs of the two simultaneously active
        CNOT edges forming the pair. Sorted canonically so
        `edge_a < edge_b` as tuples.
    sector : {"X", "Z"}
        Sector convention used to build the initial pair fault — see
        :func:`weave.analysis.build_single_pair_fault`. In `"X"`, the
        fault is `X_{control_a} X_{control_b}`; in `"Z"`, it is
        `Z_{target_a} Z_{target_b}`.
    routed_distance : float
        Scalar summary of the route pair returned by the configured
        :class:`~weave.ir.RoutePairMetric` (default: minimum polyline
        distance).
    pair_probability : float
        The exact or weak-limit pair probability computed from the
        kernel and the `(J₀, τ)` parameters in
        :class:`~weave.ir.GeometryNoiseConfig`.
    data_support : tuple[int, ...]
        Sorted data qubit indices on which the data-level image of
        the propagated pair fault acts nontrivially. Typically of
        length 2 under the retained-channel theory; any other length
        indicates an edge case (length 0: the fault cancelled; length
        1: only one data qubit is touched; length ≥3: Assumption 2
        of the PRX-Quantum-under-review paper is violated).
    data_pauli_symbols : tuple[str, ...]
        Parallel to `data_support`: the single-qubit Pauli symbol
        (`"X"`, `"Y"`, or `"Z"`) acting on each data qubit.

    Notes
    -----
    The class is frozen and hashable (all fields are immutable),
    supporting its use as a dict key or in a set. `to_json` is a
    dict of primitives suitable for direct JSON serialization.
    """

    tick_index: int
    edge_a: tuple[int, int]
    edge_b: tuple[int, int]
    sector: InteractionSector
    routed_distance: float
    pair_probability: float
    data_support: tuple[int, ...]
    data_pauli_symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.data_support) != len(self.data_pauli_symbols):
            raise ValueError(
                f"data_support length {len(self.data_support)} does not match "
                f"data_pauli_symbols length {len(self.data_pauli_symbols)}"
            )
        for sym in self.data_pauli_symbols:
            if sym not in ("X", "Y", "Z"):
                raise ValueError(
                    f"data_pauli_symbols entries must be 'X', 'Y', or 'Z'; got {sym!r}"
                )

    @property
    def data_weight(self) -> int:
        """Hamming weight of the data-level image of this pair event."""
        return len(self.data_support)

    @property
    def data_qubit_a(self) -> int:
        """First data qubit in the pair's support (weight ≥ 1 required)."""
        if not self.data_support:
            raise IndexError("ProvenanceRecord has empty data_support")
        return self.data_support[0]

    @property
    def data_qubit_b(self) -> int:
        """Second data qubit in the pair's support (weight ≥ 2 required)."""
        if len(self.data_support) < 2:
            raise IndexError(
                f"ProvenanceRecord has data_weight={self.data_weight}; data_qubit_b is undefined"
            )
        return self.data_support[1]

    def to_json(self) -> dict[str, Any]:
        return {
            "tick_index": self.tick_index,
            "edge_a": list(self.edge_a),
            "edge_b": list(self.edge_b),
            "sector": self.sector,
            "routed_distance": self.routed_distance,
            "pair_probability": self.pair_probability,
            "data_support": list(self.data_support),
            "data_pauli_symbols": list(self.data_pauli_symbols),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ProvenanceRecord:
        return cls(
            tick_index=int(data["tick_index"]),
            edge_a=(int(data["edge_a"][0]), int(data["edge_a"][1])),
            edge_b=(int(data["edge_b"][0]), int(data["edge_b"][1])),
            sector=data["sector"],
            routed_distance=float(data["routed_distance"]),
            pair_probability=float(data["pair_probability"]),
            data_support=tuple(int(q) for q in data["data_support"]),
            data_pauli_symbols=tuple(str(s) for s in data["data_pauli_symbols"]),
        )


@dataclass(frozen=True)
class CompiledExtraction:
    """The canonical output bundle of `compile_extraction`.

    Parameters
    ----------
    circuit_text : str
        The compiled Stim circuit in canonical text form. This is the
        source of truth — `circuit` is a lazy materializer that parses
        this text on first access.
    dem_text : str
        The detector error model in canonical text form.
    code_fingerprint : str
        SHA256 over the canonical byte representation of the code's
        `HX` and `HZ` parity-check matrices.
    embedding_spec : dict
        Output of `Embedding.to_json()` for the embedding used to
        compile.
    schedule_spec : dict
        Output of `Schedule.to_json()`.
    kernel_spec : dict
        Output of `Kernel.to_json()`.
    route_metric_spec : dict
        Output of `RoutePairMetric.to_json()`.
    local_noise_spec : dict
        Output of `LocalNoise.to_json()`.
    geometry_noise_spec : dict
        Output of `GeometryNoiseConfig.to_json()`.

    Notes
    -----
    The class is declared frozen, but it is **not hashable**: the
    private `_cache` field is a mutable dict holding lazily
    materialized Stim objects. Equality still works, because `_cache`
    is excluded from `compare` and `repr`.
    """

    SCHEMA_VERSION: ClassVar[int] = 2

    circuit_text: str
    dem_text: str

    code_fingerprint: str
    embedding_spec: dict[str, Any]
    schedule_spec: dict[str, Any]
    kernel_spec: dict[str, Any]
    route_metric_spec: dict[str, Any]
    local_noise_spec: dict[str, Any]
    geometry_noise_spec: dict[str, Any]

    provenance: tuple[ProvenanceRecord, ...] = ()

    # Private lazy cache for live Stim objects. Excluded from compare
    # and repr so equality and printing work as pure data.
    _cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def schema_version(self) -> int:
        return self.SCHEMA_VERSION

    # ------------------------------------------------------------------
    # Lazy materializers
    # ------------------------------------------------------------------

    @property
    def circuit(self) -> stim.Circuit:
        """Lazily parse `circuit_text` into a live `stim.Circuit`."""
        if "circuit" not in self._cache:
            import stim

            self._cache["circuit"] = stim.Circuit(self.circuit_text)
        return self._cache["circuit"]

    @property
    def dem(self) -> stim.DetectorErrorModel:
        """Lazily parse `dem_text` into a live `stim.DetectorErrorModel`."""
        if "dem" not in self._cache:
            import stim

            self._cache["dem"] = stim.DetectorErrorModel(self.dem_text)
        return self._cache["dem"]

    # ------------------------------------------------------------------
    # Fingerprint
    # ------------------------------------------------------------------

    def fingerprint(self) -> str:
        """Deterministic SHA256 hash over the canonical JSON form.

        Suitable for committing to benchmark fixtures, detecting
        regression drift in CI, and caching compiler outputs.
        """
        canonical = json.dumps(self.to_json(), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "type": "compiled_extraction",
            "circuit_text": self.circuit_text,
            "dem_text": self.dem_text,
            "code_fingerprint": self.code_fingerprint,
            "embedding_spec": self.embedding_spec,
            "schedule_spec": self.schedule_spec,
            "kernel_spec": self.kernel_spec,
            "route_metric_spec": self.route_metric_spec,
            "local_noise_spec": self.local_noise_spec,
            "geometry_noise_spec": self.geometry_noise_spec,
            "provenance": [rec.to_json() for rec in self.provenance],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> CompiledExtraction:
        if data.get("type") != "compiled_extraction":
            raise ValueError(f"Expected type='compiled_extraction', got {data.get('type')!r}.")
        version = data.get("schema_version")
        if version not in (1, cls.SCHEMA_VERSION):
            raise ValueError(
                f"Unsupported schema_version {version}; expected 1 or {cls.SCHEMA_VERSION}."
            )
        # v1 had no provenance field; default to empty. v2 round-trips it.
        raw_provenance = data.get("provenance", [])
        provenance = tuple(ProvenanceRecord.from_json(r) for r in raw_provenance)
        return cls(
            circuit_text=data["circuit_text"],
            dem_text=data["dem_text"],
            code_fingerprint=data["code_fingerprint"],
            embedding_spec=data["embedding_spec"],
            schedule_spec=data["schedule_spec"],
            kernel_spec=data["kernel_spec"],
            route_metric_spec=data["route_metric_spec"],
            local_noise_spec=data["local_noise_spec"],
            geometry_noise_spec=data["geometry_noise_spec"],
            provenance=provenance,
        )

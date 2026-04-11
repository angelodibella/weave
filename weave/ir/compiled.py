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

PR 5 minimal form
-----------------
This is the PR 5 shape: circuit + DEM + fingerprints. PR 8 will add
`provenance`, PR 9 will add `correlation_edges`, `exposure_metrics`,
and `decoder_artifact`. Each of those additions is additive with a
default-empty sentinel and may bump `SCHEMA_VERSION`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    import stim  # noqa: F401


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

    SCHEMA_VERSION: ClassVar[int] = 1

    circuit_text: str
    dem_text: str

    code_fingerprint: str
    embedding_spec: dict[str, Any]
    schedule_spec: dict[str, Any]
    kernel_spec: dict[str, Any]
    route_metric_spec: dict[str, Any]
    local_noise_spec: dict[str, Any]
    geometry_noise_spec: dict[str, Any]

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
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> CompiledExtraction:
        if data.get("type") != "compiled_extraction":
            raise ValueError(f"Expected type='compiled_extraction', got {data.get('type')!r}.")
        version = data.get("schema_version")
        if version != cls.SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version {version}; expected {cls.SCHEMA_VERSION}."
            )
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
        )

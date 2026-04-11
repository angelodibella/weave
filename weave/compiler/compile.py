"""`compile_extraction` — the central function of the weave compiler.

Takes a code, embedding, schedule, kernel, and noise configuration,
and produces a :class:`~weave.ir.CompiledExtraction` bundle whose
`circuit_text` is a canonical Stim circuit and whose `dem_text` is
its detector error model.

PR 5 shipped the local-noise-only path; PR 8 added the geometry
branch, which — when `geometry_noise.J0 > 0` — walks the routed
embedding, computes per-pair coefficients via the
`route_metric → kernel → sin²` pipeline, calls the
:mod:`weave.analysis` propagator to determine each pair fault's
data-level image, and injects `CORRELATED_ERROR` instructions into
each round of the cycle.

PR 9 finalises the `CompiledExtraction` output bundle: the
provenance list feeds :mod:`weave.ir.metrics` to produce
correlation-edge records, exposure-metric tables, and a decoder
artifact shell, all attached to the returned object. The compiler
is now a single pass: inputs in → compiled text + tables + lazy
materializers out.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import TYPE_CHECKING, Literal

import numpy as np

from ..ir import (
    CompiledExtraction,
    CrossingKernel,
    Embedding,
    GeometryNoiseConfig,
    Kernel,
    LocalNoise,
    LocalNoiseConfig,
    MinDistanceMetric,
    ProvenanceRecord,
    RoutePairMetric,
    Schedule,
    build_correlation_edges,
    build_decoder_artifact,
    build_exposure_metrics,
)
from .circuit_emit import emit_correlated_error, emit_step
from .geometry_pass import compute_provenance

if TYPE_CHECKING:
    import stim

    from ..codes.css_code import CSSCode


def compile_extraction(
    code: CSSCode,
    embedding: Embedding,
    schedule: Schedule,
    kernel: Kernel | None = None,
    *,
    route_metric: RoutePairMetric | None = None,
    local_noise: LocalNoise | None = None,
    geometry_noise: GeometryNoiseConfig | None = None,
    rounds: int = 3,
    experiment: Literal["z_memory", "x_memory"] = "z_memory",
    logical: int | list[int] | None = None,
) -> CompiledExtraction:
    """Compile a routed CSS extraction design into a `CompiledExtraction`.

    Parameters
    ----------
    code : CSSCode
        The CSS code. Must expose `HX`, `HZ`, `data_qubits`,
        `z_check_qubits`, `x_check_qubits`, and `find_logicals()`.
    embedding : Embedding
        The routed embedding. Its `routing_geometry` output will be
        consumed by PR 8's geometry pass; for PR 5 it is passed
        through as a fingerprint only.
    schedule : Schedule
        The extraction schedule produced by a factory like
        :func:`~weave.ir.default_css_schedule` or imported from an
        external tool.
    kernel : Kernel, optional
        Proximity kernel. Defaults to :class:`CrossingKernel` when
        omitted; unused in PR 5 (geometry pass is in PR 8) but
        recorded in the fingerprint for reproducibility.
    route_metric : RoutePairMetric, optional
        Route-pair reducer. Defaults to :class:`MinDistanceMetric`.
    local_noise : LocalNoise, optional
        Local noise model. Defaults to a zero-rate
        :class:`LocalNoiseConfig`.
    geometry_noise : GeometryNoiseConfig, optional
        Geometry noise parameters. Defaults to zero `J0`. When zero,
        no `CORRELATED_ERROR` is emitted.
    rounds : int, optional
        Number of extraction rounds. Must be ≥ 1.
    experiment : {"z_memory", "x_memory"}, optional
        Memory experiment type.
    logical : int or list[int] or None, optional
        Which logical operators to mark as observables. `None` means
        all.

    Returns
    -------
    CompiledExtraction
        The compiled output bundle (pure-data form).

    Raises
    ------
    ValueError
        If `rounds < 1` or `experiment` is not recognized.
    """
    import stim

    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}.")
    if experiment not in ("z_memory", "x_memory"):
        raise ValueError(f"experiment must be 'z_memory' or 'x_memory', got {experiment!r}.")

    # Defaults for optional args.
    if kernel is None:
        kernel = CrossingKernel()
    if route_metric is None:
        route_metric = MinDistanceMetric()
    if local_noise is None:
        local_noise = LocalNoiseConfig()
    if geometry_noise is None:
        geometry_noise = GeometryNoiseConfig()

    # ------------------------------------------------------------------
    # Geometry pass: one-shot provenance for the whole cycle. Each
    # record represents a pair-fault event that fires in every round.
    # ------------------------------------------------------------------
    provenance: tuple[ProvenanceRecord, ...] = tuple(
        compute_provenance(
            schedule=schedule,
            embedding=embedding,
            kernel=kernel,
            route_metric=route_metric,
            geometry_noise=geometry_noise,
        )
    )
    events_by_tick: dict[int, list[ProvenanceRecord]] = defaultdict(list)
    for rec in provenance:
        events_by_tick[rec.tick_index].append(rec)

    # ------------------------------------------------------------------
    # Build the circuit.
    # ------------------------------------------------------------------
    circuit = stim.Circuit()

    # Head.
    for step in schedule.head_steps:
        emit_step(circuit, step, local_noise)

    # Round 1 body.
    for step in schedule.cycle_steps:
        for rec in events_by_tick.get(step.tick_index, ()):
            emit_correlated_error(circuit, rec)
        emit_step(circuit, step, local_noise)

    # First-round detectors reference only the just-measured ancillas.
    _emit_first_round_detectors(circuit, code, experiment)

    # Rounds 2..N: body + comparison detectors.
    for _ in range(rounds - 1):
        for step in schedule.cycle_steps:
            for rec in events_by_tick.get(step.tick_index, ()):
                emit_correlated_error(circuit, rec)
            emit_step(circuit, step, local_noise)
        _emit_comparison_detectors(circuit, code)

    # Tail: final data measurement step(s).
    for step in schedule.tail_steps:
        emit_step(circuit, step, local_noise)

    # Tail detectors link the final data measurement to the last ancilla
    # measurement of the final cycle.
    _emit_tail_detectors(circuit, code, experiment)

    # Observables for the decoded sector's logical operators.
    _emit_observables(circuit, code, experiment, logical)

    # ------------------------------------------------------------------
    # Compute the DEM.
    # ------------------------------------------------------------------
    # `decompose_errors=True` produces matching-friendly DEMs for
    # PyMatching. Correlated two-qubit errors injected by the
    # geometry pass typically flip three or more detectors and
    # defeat the graphlike decomposition, so we fall back to the
    # undecomposed BP+OSD-friendly form whenever provenance is
    # non-empty. Pure local-noise compiles still take the matching
    # path to preserve the PR 5 faithfulness tests.
    decompose = len(provenance) == 0
    dem = circuit.detector_error_model(decompose_errors=decompose, approximate_disjoint_errors=True)

    # ------------------------------------------------------------------
    # Build the PR 9 pure-data tables from the provenance list.
    # ------------------------------------------------------------------
    correlation_edges = build_correlation_edges(provenance)
    logical_supports = _code_logical_supports(code, experiment)
    exposure_metrics = build_exposure_metrics(provenance, logical_supports=logical_supports)
    decoder_artifact = build_decoder_artifact(provenance, num_data_qubits=len(code.data_qubits))

    # ------------------------------------------------------------------
    # Package the output as pure data.
    # ------------------------------------------------------------------
    result = CompiledExtraction(
        circuit_text=str(circuit),
        dem_text=str(dem),
        code_fingerprint=_code_fingerprint(code),
        embedding_spec=embedding.to_json(),
        schedule_spec=schedule.to_json(),
        kernel_spec=kernel.to_json(),
        route_metric_spec=route_metric.to_json(),
        local_noise_spec=local_noise.to_json(),
        geometry_noise_spec=geometry_noise.to_json(),
        provenance=provenance,
        correlation_edges=correlation_edges,
        exposure_metrics=exposure_metrics,
        decoder_artifact=decoder_artifact,
    )
    # Stim's text form rounds floating-point arguments to ~7 digits
    # for readability, which loses precision on e.g. a pair probability
    # like `sin²(τJ₀κ)`. We pre-populate `_cache` with the exact
    # in-memory objects so downstream consumers that access
    # `result.circuit` / `result.dem` see full double precision.
    # JSON round-trips still go through the lossy text form.
    result._cache["circuit"] = circuit
    result._cache["dem"] = dem
    return result


# ============================================================================
# Detector-emission helpers
# ============================================================================


def _emit_first_round_detectors(circuit: stim.Circuit, code: CSSCode, experiment: str) -> None:
    """First-round detectors: reference the ancilla measurements we just wrote.

    Reproduces the legacy detector-indexing logic from
    `CSSCode._legacy_generate()` verbatim. In `z_memory`, only the
    Z-check ancillas' first-round values are deterministically zero
    (from `R`-prepared data), so we detect only those. The symmetric
    case for `x_memory` detects only the X-check ancillas.
    """
    import stim

    num_x_check = len(code.x_check_qubits)
    num_z_check = len(code.z_check_qubits)

    if experiment == "z_memory":
        for k in range(num_z_check):
            circuit.append(
                "DETECTOR",
                [stim.target_rec(-1 - k - num_x_check)],
            )
    else:  # x_memory
        for k in range(num_x_check):
            circuit.append("DETECTOR", [stim.target_rec(-1 - k)])


def _emit_comparison_detectors(circuit: stim.Circuit, code: CSSCode) -> None:
    """Comparison detectors: current-round ancilla XOR previous-round ancilla.

    One detector per ancilla, pairing the most recent measurement with
    the corresponding measurement one cycle earlier. Emitted inside
    rounds 2..N, so the legacy path's `round_circuit.append("DETECTOR",
    ...)` in-place mutation is reproduced by calling this helper once
    per subsequent round.
    """
    import stim

    num_ancillas = len(code.z_check_qubits) + len(code.x_check_qubits)
    for k in range(num_ancillas):
        circuit.append(
            "DETECTOR",
            [
                stim.target_rec(-1 - k),
                stim.target_rec(-1 - k - num_ancillas),
            ],
        )


def _emit_tail_detectors(circuit: stim.Circuit, code: CSSCode, experiment: str) -> None:
    """Tail detectors link final data measurements to the last ancilla reads.

    After all rounds complete and the data qubits have been measured,
    we emit one detector per check linking the last ancilla
    measurement to the parity of the data-qubit measurements in that
    check's support.
    """
    import stim

    num_data = len(code.data_qubits)
    num_x_check = len(code.x_check_qubits)

    if experiment == "z_memory":
        for k in range(len(code.z_check_qubits)):
            row = code.HZ[-1 - k]
            idxs = [i for i, v in enumerate(row) if v]
            recs = [stim.target_rec(-1 - k - num_data - num_x_check)]
            for idx in idxs:
                recs.append(stim.target_rec(idx - num_data))
            circuit.append("DETECTOR", recs)
    else:  # x_memory
        for k in range(num_x_check):
            row = code.HX[-1 - k]
            idxs = [i for i, v in enumerate(row) if v]
            recs = [stim.target_rec(-1 - k - num_data)]
            for idx in idxs:
                recs.append(stim.target_rec(idx - num_data))
            circuit.append("DETECTOR", recs)


def _emit_observables(
    circuit: stim.Circuit,
    code: CSSCode,
    experiment: str,
    logical: int | list[int] | None,
) -> None:
    """Emit `OBSERVABLE_INCLUDE` for each logical operator."""
    import stim

    x_logicals, z_logicals = code.find_logicals()
    x_logical_qubits = [np.where(log == 1)[0] for log in x_logicals]
    z_logical_qubits = [np.where(log == 1)[0] for log in z_logicals]
    logicals_to_emit: list[np.ndarray] = (
        z_logical_qubits if experiment == "z_memory" else x_logical_qubits
    )

    if logical is not None:
        indices = [logical] if isinstance(logical, int) else list(logical)
        logicals_to_emit = [logicals_to_emit[i] for i in indices]

    num_data = len(code.data_qubits)
    for i, lq in enumerate(logicals_to_emit):
        recs = [stim.target_rec(int(q) - num_data) for q in lq]
        circuit.append("OBSERVABLE_INCLUDE", recs, i)


# ============================================================================
# PR 9 support: logical-representative supports for exposure metrics
# ============================================================================


def _code_logical_supports(code: CSSCode, experiment: str) -> tuple[tuple[int, ...], ...]:
    """Return the decoded-sector logical supports as sorted qubit tuples.

    Mirrors the sector convention of :func:`_emit_observables`:
    `z_memory` decodes X errors and uses the Z-logical supports;
    `x_memory` decodes Z errors and uses the X-logical supports.
    Each support is the set of data-qubit indices where the
    representative logical acts nontrivially, sorted ascending.
    Used by the PR 9 exposure-metric builder to produce
    `per_support` records.
    """
    x_logicals, z_logicals = code.find_logicals()
    logicals = z_logicals if experiment == "z_memory" else x_logicals
    supports: list[tuple[int, ...]] = []
    for row in logicals:
        support = tuple(int(i) for i in np.where(row == 1)[0])
        supports.append(support)
    return tuple(supports)


# ============================================================================
# Fingerprint
# ============================================================================


def _code_fingerprint(code: CSSCode) -> str:
    """SHA256 over a canonical byte representation of (HX, HZ).

    Incorporates shape as text to avoid ambiguous reshapes producing
    the same byte sequence.
    """
    shape_str = f"HX{code.HX.shape}|HZ{code.HZ.shape}".encode()
    hx_bytes = np.ascontiguousarray(code.HX, dtype=np.uint8).tobytes()
    hz_bytes = np.ascontiguousarray(code.HZ, dtype=np.uint8).tobytes()
    return hashlib.sha256(shape_str + hx_bytes + hz_bytes).hexdigest()


# Re-export for convenience (used by tests and downstream modules).
__all__: list[str] = ["compile_extraction"]

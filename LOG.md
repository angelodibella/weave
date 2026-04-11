# weave — PR log

Running log of the geometry-aware compiler roadmap (see `private/plan.md`
for the full specification). Each entry is a brief retrospective of
what shipped, in the order the PRs were merged. Test counts are
cumulative over `uv run pytest weave/tests/`.

## PR 1 — Geometry engine

Tanner-graph geometry primitives: layout computation, edge
intersection checks, and the scaffolding that later PRs build on.
Sits under `weave/geometry/`. No circuit emission yet.

## PR 2 — `weave/ir/` and the `Embedding` protocol

Introduced the immutable IR layer: `Embedding`, `RoutingGeometry`,
`IREdge`, `IRPolyline`, `StraightLineEmbedding`, `JsonPolylineEmbedding`,
`load_embedding`. Decouples geometric layout from the code classes so
the compiler can consume either directly.

## PR 3 — Kernel and noise IR

`Kernel` protocol and the concrete `CrossingKernel`,
`RegularizedPowerLawKernel`, `ExponentialKernel` implementations, plus
`LocalNoiseConfig` and `GeometryNoiseConfig` (with `LocalNoise` protocol
for the compiler's noise queries).

## PR 4 — `RouteID`, `RoutePairMetric`, `Schedule` IR (upgraded)

The central `Schedule` IR landed in its v2 form. Key design decisions:

- Discriminated union `ScheduleEdge = TwoQubitEdge | SingleQubitEdge`
  with explicit `control`/`target` on the two-qubit variant so
  propagation logic never guesses from tuple position.
- Per-edge `interaction_sector` (not per-step) so imported schedules
  can mix sectors in a single tick.
- Schedule-level `qubit_roles`.
- Head / cycle / tail blocks mapping directly to Stim `REPEAT`.
- `default_css_schedule(code)` factory that reproduces the legacy gate
  order as a proper schedule.

`RouteID` + `RoutePairMetric` (with `MinDistanceMetric`) support the
geometry pass's pair-distance queries.

## PR 5 — Schedule-aware compiler (local-noise-only path)

First real `compile_extraction(...)` call site with `geometry_noise.J0
== 0`. Produces Stim text (not a live circuit object yet) with TICK
markers, `DEPOLARIZE1` idle noise, and `DEPOLARIZE2` circuit noise.
Lives in `weave/compiler/` (`compile.py`, `circuit_emit.py`). Sits
alongside the legacy `CSSCode._legacy_generate` and doesn't replace it.

Phase-I roadmap items closed: TICK markers, CNOT scheduling, idle noise.

## PR 5.5 — Reference-code test fixtures (Stage A)

Added hand-verified reference codes (Shor, Steane via HGP, small
Tillich–Zémor HGPs) with their published parameters as direct test
fixtures, so downstream faithfulness checks have stable ground truth.

## PR 6 — `CSSCode.circuit` delegates to the compiler

`CSSCode.circuit` becomes a thin wrapper that dispatches:

- **Noiseless codes** → `compile_extraction(...)` via the compiler
  path (emits TICK markers, lazy materialization).
- **Noisy codes** → fall back to `_legacy_generate()` (the renamed
  legacy method, kept as a private helper until PR 20).

The dispatch is a pragmatic interpretation: the legacy
`PAULI_CHANNEL_1`/`PAULI_CHANNEL_2` noise channels cannot losslessly
translate to the new `LocalNoise` protocol, so we route them to the
legacy generator until the IR noise model catches up. Tests in
`TestCircuitDispatchPR6` pin the dispatch via TICK / PAULI_CHANNEL
fingerprints.

## PR 7 — Pauli propagation and residual-error analyzer

New `weave/analysis/` package for schedule-agnostic fault analysis.

**`pauli.py`** — phase-free symplectic Pauli primitive. Frozen
`Pauli(x, z)` dataclass, Clifford propagation functions for CNOT, H,
S, X/Y/Z/I, measurements (`measure_x`/`measure_z`). Rules verbatim
from Gottesman 1997 Table 3.1. Module docstring explicitly
disambiguates from the modern Rudolph et al. 2025 "Pauli Propagation"
framework and the Facelli–Fawzi "Majorana Propagation" line — both
are classical-simulation techniques, not stabilizer fault tracking.

**`propagation.py`** — schedule walker.
`propagate_fault(schedule, initial_fault, injection_location,
data_qubits, end_block)` walks head → cycle → tail (clipped at
`end_block`) and returns a `PropagationResult` with `data_pauli`,
`full_pauli`, and the list of `AncillaFlip`s.
`build_single_pair_fault` implements the §II.D sector convention
(X-sector → X on controls, Z-sector → Z on targets).
`propagate_single_pair_event` is a wrapper defaulting to
`end_block="cycle"`.

**`residual.py`** — Strikis, Browne, Beverland 2026 (arXiv:2603.05481)
formalism. `ResidualError` dataclass, `residual_distance(E, h_commute,
h_stab)` computing `Δ(E) = 1 + min_D{|D| : (E+D) ∈ ker(h_commute) \
span(h_stab)}` by exhaustive `k_guard`-capped nullspace enumeration,
`effective_distance_upper_bound` for the Theorem 1 target, and
`enumerate_hook_residuals_z_sector` for the direct `E_ℓ` definition.

**`validation.py`** — `verify_weight_le_2_assumption(schedule, sector)
→ ValidationReport` checking the PRX Quantum paper's Assumption 2.
Iterates parallel sector-relevant CNOT pairs, propagates each pair
fault with `end_block="cycle"`, and reports `PairEventResult`s with a
pass/fail verdict on `data_weight ≤ 2`.

**Tests** — 71 new (553 total): `test_analysis_pauli.py` (30),
`test_analysis_propagation.py` (12), `test_analysis_residual.py` (20),
`test_analysis_validation.py` (9).

**Config** — `weave/analysis` added to `[tool.mypy] files` in
`pyproject.toml` and to the mypy hook pattern in
`.pre-commit-config.yaml`.

## PR 8 — Geometry-aware compiler path

The second half of `compile_extraction`: when `geometry_noise.enabled`
(i.e. `J0 > 0`), the geometry pass walks the schedule, computes
route-pair coefficients via `route_metric → kernel → sin²`, calls the
PR 7 propagation analyzer to determine each pair's data-level image,
and emits `CORRELATED_ERROR` instructions into every round of the
cycle.

**New files**

- `weave/compiler/geometry_pass.py` — `compute_provenance(schedule,
  embedding, kernel, route_metric, geometry_noise) → list[ProvenanceRecord]`.
  For each `cnot_layer` step and each sector `S ∈ {X, Z}`, filters
  sector-relevant edges (respecting `geometry_scope`), queries the
  embedding for polylines, enumerates unordered pairs, computes
  distance → pair probability, propagates the pair fault via
  `propagate_single_pair_event(..., end_block="cycle")`, and builds
  one `ProvenanceRecord` per surviving event. Skips zero-probability
  events and weight-0 data residuals. Returns a deterministically
  sorted list.
- `weave/tests/test_compiler_geometry.py` — 14 tests covering the
  unit behavior, numerics (hand-computed `sin²(0.25)` to 1e-12),
  end-to-end `compile_extraction` integration, JSON round-trip with
  provenance, determinism, and v1 backward compatibility.

**Touched**

- `weave/ir/compiled.py` — bumped `SCHEMA_VERSION` from 1 to 2, added
  the `ProvenanceRecord` dataclass, added the `provenance` field on
  `CompiledExtraction` with default-empty sentinel, extended
  `to_json`/`from_json` to round-trip provenance, and left v1 JSON
  loadable with an empty provenance fallback.
- `weave/ir/__init__.py` — exported `ProvenanceRecord`.
- `weave/compiler/circuit_emit.py` — added `emit_correlated_error`
  helper that translates a `ProvenanceRecord` into a
  `CORRELATED_ERROR` instruction with proper Stim Pauli targets.
- `weave/compiler/compile.py` — computes provenance up front, indexes
  by cycle tick, and emits the correlated-error channels *before*
  each cycle step's gates in every round. Pre-populates `_cache` on
  the returned `CompiledExtraction` with the exact `stim.Circuit`
  and `DetectorErrorModel` objects, so downstream consumers that
  access `.circuit` or `.dem` see full double-precision probabilities
  (the serialized `circuit_text` is still Stim's lossy text form by
  necessity).
- `weave/tests/test_ir_compiled.py` — updated the two `schema_version
  == 1` pins to the new `== 2`.

**Design notes**

- `ProvenanceRecord` stores the sorted data support and parallel
  Pauli symbols (`X`/`Y`/`Z`), so the record captures the actual
  propagated fault structure rather than assuming weight-2. The
  `data_qubit_a` / `data_qubit_b` properties give the clean weight-2
  accessor the plan's acceptance test expects, while still raising
  on unexpected weights.
- The pass is sector-symmetric: both `X` and `Z` are evaluated at
  every tick, so mixed-sector schedules get both channels. In
  `geometry_scope == "theory_reduced"` only explicitly-tagged edges
  count; in `"full_cycle"` untagged edges are treated as
  sector-agnostic and included in both sectors.
- Weight-0 propagated residuals are silently skipped (the pair
  fault cancels itself). Higher-weight residuals (indicating the
  `verify_weight_le_2_assumption` check would fail) are still
  emitted so that users can audit them post-compile; the canonical
  way to reject a violating schedule is to call the PR 7 validator
  explicitly before compiling.

**Tests** — 14 new (567 total): unit-tests of `compute_provenance`
for empty, serial, parallel, numerics, weak-limit, and propagation
integration; end-to-end `compile_extraction` tests for instruction
count, multi-round scaling, exact probability preservation,
determinism, and zero-`J0` fallback; plus JSON round-trip tests for
both `ProvenanceRecord` and the new schema-v2 `CompiledExtraction`
(with v1 backward-compat).

## PR 9 — `ExposureMetrics`, correlation edges, `DecoderArtifact`, `CompiledExtraction` finalization

Finished the `CompiledExtraction` output bundle as pure-data tables.
Every provenance record produced by PR 8 now feeds three aggregate
tables that the optimizer, decoders, and benchmarks consume.

**New files**

- `weave/ir/metrics.py` — `SupportExposureRecord`, `CorrelationEdgeRecord`,
  and `ExposureMetrics` with the four-way decomposition
  (`per_support`, `per_tick`, `per_route_pair`, `per_data_pair`) plus
  `total()`, `by_logical(i)`, and `max_over_family(family)` queries.
  Also the two builder functions `build_correlation_edges` and
  `build_exposure_metrics` that aggregate a provenance list into the
  canonical tables. Exposure semantics pinned in the module docstring:
  `ℰ(L) = Σ rec.pair_probability for rec with data_support ⊆ L`.
- `weave/ir/decoder_artifact.py` — `DecoderArtifact` shell with
  `pair_edges`, `single_prior`, `decoder_hint`; `build_decoder_artifact`
  sums weight-2 pair events sector-merged. The adapter methods
  (`to_pymatching_hint`, `to_bposd_dem`) are deferred to PR 17.
- `weave/tests/test_ir_metrics.py` — 29 tests covering construction,
  aggregation, the `max_over_family` J_κ query, and JSON round-trip.
- `weave/tests/test_ir_decoder_artifact.py` — 16 tests covering
  validation, sector-merged aggregation, and round-trip.

**Touched**

- `weave/ir/compiled.py` — bumped `SCHEMA_VERSION` 2 → 3. Added
  `correlation_edges`, `exposure_metrics`, `decoder_artifact` fields
  with backward-compat defaults. Added a lazy `correlation_graph`
  NetworkX materializer cached alongside `circuit`/`dem`. Extended
  `to_json`/`from_json` to round-trip the new tables and fall back
  gracefully for schema v1 and v2 artifacts.
- `weave/ir/route.py` — added `RouteID.to_json` / `from_json` (needed
  by `ExposureMetrics.per_route_pair` serialization).
- `weave/ir/__init__.py` — exported `CorrelationEdgeRecord`,
  `DecoderArtifact`, `ExposureMetrics`, `SupportExposureRecord`, and
  the three `build_*` helpers.
- `weave/compiler/compile.py` — after the geometry pass, the
  compiler now calls `build_correlation_edges`, `build_exposure_metrics`
  (with `_code_logical_supports(code, experiment)`), and
  `build_decoder_artifact`, and attaches all three to the returned
  `CompiledExtraction`. Switched `detector_error_model(...)` to the
  undecomposed BP+OSD-friendly form when `provenance` is non-empty —
  correlated multi-qubit errors typically flip ≥3 detectors and
  defeat graphlike decomposition, so PR 9 routes them through BP+OSD
  while leaving the PR 5 pure-local-noise path on the matching DEM.
- `weave/tests/test_ir_compiled.py` — updated `schema_version == 2`
  pins to `== 3`.
- `weave/tests/test_compiler_geometry.py` — added `TestCompiledExtractionPR9`
  (10 tests) covering correlation-edge population, exposure total
  matching sum of provenance probabilities to 1e-12, per-data-pair
  alignment with correlation edges, per-support population,
  decoder-artifact population, fingerprint determinism, deep JSON
  round-trip of all tables, fingerprint stability across round-trip,
  lazy `circuit`/`dem` text equality, and the NetworkX correlation
  graph. Also added `TestSteaneCrossingPR9` (2 tests) with a
  hand-crafted Steane schedule that forces a single parallel X-sector
  pair between data qubits 0 and 3, verifying plan acceptance test #1
  (exactly one correlation edge with the expected `sin²(τJ₀κ)` weight).
- `weave/tests/test_compiler_geometry.py` — added a v2
  backward-compat test (`test_v2_compiled_extraction_loads_with_empty_correlation_fields`)
  that mirrors the existing v1 test.

**Design notes**

- *Exposure semantics.* `per_support[L].exposure = Σ_{rec : data_support ⊆ L} pair_probability`.
  The tight "subset" interpretation (vs "intersects") matches the
  retained-channel §III exposure scale in the paper: a pair event
  contributes to a logical only when its full 2-qubit image can be
  absorbed inside the support.
- *`total()` uses `per_tick`,* which is over *all* records regardless
  of weight. `per_data_pair` is restricted to weight-2 records (the
  ones that define a canonical qubit pair); it matches `total()` when
  every event is weight-2, which is the normal retained-channel
  regime. Non-weight-2 events still appear in `per_tick` and
  `per_route_pair` for full provenance.
- *`fingerprint()` now covers the new tables* because `to_json`
  includes them. Two compiles of the same inputs produce
  byte-identical JSON and therefore byte-identical SHA256.
- *`correlation_graph` is NetworkX, undirected, sector-merged.*
  Callers that need per-sector weights iterate
  `correlation_edges` directly.
- *Decoder artifact is a shell.* `single_prior` is populated to
  `(0.0,) * len(code.data_qubits)`; PR 17 will fill it with local-noise
  priors and add the adapter methods. The `decoder_hint` defaults to
  empty string.

**Tests** — 58 new (625 total, up from 567): 29 in
`test_ir_metrics.py`, 16 in `test_ir_decoder_artifact.py`, 12 in
`test_compiler_geometry.py::TestCompiledExtractionPR9` and
`TestSteaneCrossingPR9`, plus the v2 backward-compat test.

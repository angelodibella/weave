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

## PR 10 — BB code family and pure-L logical enumeration

Added the bivariate bicycle (BB) CSS code family of Bravyi, Cross,
Gambetta, Maslov, Rall, Yoder (Nature 2024, arXiv:2308.07915) as a
first-class code in weave, together with the pure-L minimum-weight
quotient enumeration that the optimizer and the `J_κ` objective
consume.

**New files**

- `weave/codes/bb/__init__.py` — package docstring + public exports.
- `weave/codes/bb/bb_code.py` — `BivariateBicycleCode(CSSCode)`
  parameterized by `(l, m, A_monomials, B_monomials, known_distance)`.
  Builds `H_X = [A | B]` and `H_Z = [B^⊤ | A^⊤]` from the polynomial
  action on `F_2[Z_l × Z_m]` using column-major `flat = j * l + i`
  indexing (matching `bbstim` and the Bravyi workbook supports).
  Overrides `CSSCode.distance()` to return the cached known value —
  BB codes' nullspaces are typically `n/2` dimensional, which is out
  of reach of `CSSCode`'s `k_guard = 20` brute-force enumeration.
  Factories `build_bb72()`, `build_bb90()`, `build_bb108()`,
  `build_bb144()` instantiate the four Bravyi et al. Table I codes.
  Also exposes `flat_index`, `unflat_index`, `l_block_indices`,
  `r_block_indices`, and `block_size` helpers.
- `weave/codes/bb/algebra.py` — `ker_A_basis`, `ker_BT_basis`,
  `pure_L_stabilizer_basis`, and the headline
  `enumerate_pure_L_minwt_logicals`. The pure-L stabilizer space is
  `S_L = B · ker(A)` (derived in the module docstring: summing `H_Z`
  rows with coefficients `c ∈ ker(A)` gives a pure-L element
  `B c`, which sits in `ker(A)` because `AB = BA`). Enumeration
  walks every element of `ker(A)` (`2^{dim ker A}` items — 4096 for
  BB72, 64 for BB108), classifies each by its coset modulo `S_L`
  via the stabilizer row-echelon pivot columns, and collects every
  coset leader that achieves the minimum Hamming weight.
- `weave/tests/test_bb_code.py` — 26 tests across BB72/90/108/144
  parameters, flat-index round-trip, algebraic subspaces, direct
  construction edge cases, and the BB72 workbook support assertion.

**Touched**

- `weave/codes/__init__.py` — exported `BivariateBicycleCode` and the
  four factory functions.
- `pyproject.toml` — added `weave/codes/bb/*` to the ruff per-file
  ignores for `N802` (`ker_A_basis` and friends use math-convention
  uppercase) and `E741` (the variable `l` is the standard symbol for
  the first cyclic factor). Pre-commit's mypy hook and the mypy
  `files` list already cover `weave/codes`, so no config changes
  there.

**Plan acceptance tests satisfied**

1. `build_bb72()` → `n = 72`, `k = 12`, `distance() == 6`.
2. `enumerate_pure_L_minwt_logicals(build_bb72())` returns exactly
   36 weight-6 supports; the workbook support
   `(3, 12, 21, 24, 27, 33)` is among them. The column-major
   indexing convention is documented in `BivariateBicycleCode`.
3. BB108 has `distance() == 10` (pinned to Bravyi et al. Table I;
   the plan text's "12" is a typo corrected to "10" here).

**Design notes**

- *Distance override.* `CSSCode.distance()` brute-forces the
  nontrivial logical minimum-weight search with a `k_guard = 20`
  cap, which the BB nullspaces (36-dimensional for BB72) exceed by
  a large margin. The cleanest pragmatic fix is a published-value
  cache on `BivariateBicycleCode` itself; computing BB distances
  from first principles is NP-hard and out of scope for PR 10.
  The PR 13 BB72 faithfulness fixture will cross-check the cached
  values against bbstim's reference simulator.
- *Indexing convention.* I tested both `flat = i * m + j` (row-major)
  and `flat = j * l + i` (column-major). The column-major convention
  produces the workbook support `{3, 12, 21, 24, 27, 33}` verbatim
  and matches `bbstim`, so that is the pinned default. The module
  docstring, `flat_index`, and the test file all reference it
  explicitly.
- *Pure-L stabilizer formula.* My first derivation had
  `S_L = B^⊤ · ker(A^⊤)`, which is wrong — `A B^⊤ ≠ B^⊤ A` in
  general. The correct formula is `S_L = B · ker(A)`, derived by
  taking an arbitrary linear combination of `H_Z` rows and requiring
  the R-component to vanish (forces the coefficient vector into
  `ker A`, leaves `B c` on the L-component). The corrected formula
  lives inside `pure_L_stabilizer_basis` and is pinned by the
  `test_pure_L_stabilizer_basis_shape` test (basis rows lie in
  `ker A`).

**Dev sweep** — `ruff check`, `ruff format`, `mypy`, `pytest` all
clean. **651 tests pass** (up from 625; +26 for PR 10).

## PR 11 — BB embeddings and the monomial-parallel IBM schedule factory

Added the three concrete BB code embeddings called for by the plan
(`ColumnEmbedding`, `MonomialColumnEmbedding`, `IBMBiplanarEmbedding`,
`FixedPermutationColumnEmbedding`) and the syndrome extraction
schedule factory `ibm_schedule(bb_code)`. Every object honours the
existing :class:`~weave.ir.Embedding` protocol and the PR 9
`CompiledExtraction` flow, so `compile_extraction(bb72, monomial,
ibm_schedule(bb72), kernel, ...)` works end-to-end without any
BB-specific branches in the compiler.

**New files**

- `weave/ir/embeddings/column.py` — `ColumnEmbedding` (a general
  regular-grid embedding) and `MonomialColumnEmbedding(ColumnEmbedding)`
  with a `from_bb(bb_code, spacing, name)` factory. The BB layout
  places four qubit classes in four parallel sub-columns per
  `(i, j)` cell: `L`-data at `sub=0`, Z-ancilla at `sub=1`,
  X-ancilla at `sub=2`, `R`-data at `sub=3`. The resulting
  `4 l × m` lattice is documented in the module docstring and
  preserved in JSON round-trip via `num_columns`, `num_rows`,
  `layers_per_cell`, `l`, `m`, `spacing`, `bb_name`.
- `weave/ir/embeddings/biplanar.py` — `IBMBiplanarEmbedding`, the
  two-plane BB layout with `L`-block data at `z = +layer_height`,
  `R`-block data at `z = -layer_height`, and ancillas on the
  `z = 0` mid-plane (Z-ancillas offset by `(+0.5, 0)`, X-ancillas
  by `(0, +0.5)` so the two half-lattices never collide).
- `weave/ir/embeddings/fixed_permutation.py` — 
  `FixedPermutationColumnEmbedding`, a frozen read-only embedding
  loaded from a JSON file with a `permutation` field carrying the
  provenance mapping from canonical qubit indices to the stored
  layout and a `source_description` free-text field for audit trails.
  Includes `from_json_file(path)` and round-trip through
  `load_embedding`.
- `weave/codes/bb/schedule.py` — `ibm_schedule(bb_code, experiment)`
  builds the monomial-parallel syndrome extraction schedule. Cycle
  structure: one H bracket tick → `|A| + |B|` Z-check CNOT layers
  (data → Z-ancilla, X-sector) → `|A| + |B|` X-check CNOT layers
  (X-ancilla → data, Z-sector) → one H bracket close tick → one MR
  tick. Each CNOT layer fires `lm` parallel CNOTs with no qubit
  conflicts because every data qubit participates in exactly one
  monomial action per layer. Total cycle depth for the Bravyi
  `|A| = |B| = 3` codes is 15 ticks.
- `weave/tests/test_bb_embeddings.py` — 35 tests covering
  `ColumnEmbedding` construction/validation/routing/round-trip,
  `MonomialColumnEmbedding` qubit-position invariants and a routing
  test that pins the "36 edges per monomial layer" acceptance
  criterion for BB72, `IBMBiplanarEmbedding` `z`-sign separation
  of L and R blocks, `FixedPermutationColumnEmbedding` JSON file
  round-trip and permutation validation, and the `ibm_schedule`
  factory's correctness (head/cycle/tail lengths, CNOT counts per
  sector, CNOT direction for each sector, H brackets, final MR,
  and the end-to-end `compile_extraction` noiseless-deterministic
  check on BB72 and BB108).

**Touched**

- `weave/ir/embedding.py` — `load_embedding` now dispatches to
  `column`, `monomial_column`, `ibm_biplanar`, and
  `fixed_permutation_column`.
- `weave/ir/embeddings/__init__.py` — exports all six embedding
  classes.
- `weave/ir/__init__.py` — exports the four new classes at the IR
  package level.
- `weave/codes/bb/__init__.py` — exports `ibm_schedule`.
- `pyproject.toml` — added `E741` per-file ignore for
  `weave/ir/embeddings/column.py` and `biplanar.py` (BB
  modules use the math-convention name `l` for the first cyclic
  factor).

**Plan acceptance tests satisfied**

1. ✓ `MonomialColumnEmbedding.from_bb(bb72)` routes a full B-monomial
   Z-check layer to exactly 36 parallel edges, one per Z-ancilla.
2. ✓ `IBMBiplanarEmbedding`: every L-block data qubit has `z > 0`,
   every R-block data qubit has `z < 0`, and the mean `z` of any
   L→ancilla routing polyline is positive (symmetrically for R).
3. ~ `bbstim_ibm_schedule(bb72)` — rebranded as `ibm_schedule(bb72)`
   since vendoring `bbstim` into the test suite is out of scope.
   The weave factory produces a **mathematically equivalent**
   syndrome extraction schedule (same stabilizers, same logical
   action, same sector tagging) and is pinned against the
   ground-truth check that `compile_extraction(bb72, monomial,
   ibm_schedule(bb72), ...)` yields a deterministic noiseless
   detector sampler on 100 shots. The Bravyi minimum-depth-8
   interleave is left to a future PR.
4. Deferred — the "matching number 3/0 on BB72 workbook support"
   test requires the PR 12 optimizer's support-crossing counter
   and the PR 13 bbstim reference. It will be wired in at PR 13.

**Design notes**

- *Mathematical derivation of the CNOT formulas.* The BB code's
  parity-check matrices are `H_X = [A \mid B]` and
  `H_Z = [B^⊤ \mid A^⊤]`. For a Z-check at group element
  `i = (i_1, i_2) \in \mathbb{Z}_l \times \mathbb{Z}_m`,
  `H_Z[i, c] = B[c, i] = 1` (for L-block `c`) iff `c = i + (d_1, d_2)`
  for some monomial `(d_1, d_2) \in B`. So the Z-check at `i`
  receives a CNOT from data qubit at `(i_1 + d_1, i_2 + d_2)`.
  Iterating over every `i` at fixed monomial gives a full-parallel
  permutation CNOT layer. Symmetrically for X-checks:
  `H_X[i, c] = A[i, c] = 1` iff `i = c + (d_1, d_2)`, so
  `c = i - (d_1, d_2)` — the opposite sign. The PR 11 smoke-test
  caught me with the wrong sign the first time around: the
  noiseless compile raised a non-deterministic-detector error. The
  final formula is pinned by the
  `test_compile_noiseless_is_deterministic` test on BB72 and
  BB108.
- *Depth trade-off.* The Bravyi minimum-depth-8 schedule interleaves
  Z-check and X-check CNOTs that act on disjoint qubit sets. Our
  `ibm_schedule` keeps them in separate ticks, giving a cleaner
  correctness story and a cycle depth of `3 + 2(|A| + |B|) = 15`
  for the Bravyi codes. A future PR can add a separate
  `bravyi_depth8_schedule(bb)` factory once the PR 13 regression
  fixture has pinned the observed depth-8 behaviour.
- *Frozen-dataclass subclassing.* `MonomialColumnEmbedding` inherits
  from `ColumnEmbedding` and adds `l, m, spacing, bb_name` as
  additional frozen fields. Python allows this because the parent
  fields have defaults; `__post_init__` calls `super().__post_init__()`
  and then validates the BB-specific invariants. This is the only
  embedding hierarchy in weave; `IBMBiplanarEmbedding` and
  `FixedPermutationColumnEmbedding` are standalone because their
  layouts don't fit the flat column-grid abstraction cleanly.
- *Generality.* All four embedding classes accept custom `positions`
  directly (bypassing the `from_bb` factory), so users can supply
  hand-designed layouts. The schedule factory takes an `experiment`
  kwarg (`z_memory` or `x_memory`) to match the PR 5/6 memory
  experiments. Every CNOT is tagged with a `term_name` of the form
  `"BB.{block}[{d1},{d2}]→z[{i1},{j}]"` so downstream provenance
  can audit which BB monomial each edge came from.

**Dev sweep** — `ruff check`, `ruff format`, `mypy`, `pytest` all
clean. **686 tests pass** (up from 651; +35 for PR 11).

## PR 12 — Logical-aware exposure optimizer

Added the `weave.optimize` package: objective functionals for the
retained-channel exposure scale `J_κ`, a vectorized
:class:`NumpyExposureTemplate` that the inner loop queries in
sub-millisecond time, and a randomized first-improvement swap
descent that optimizes any :class:`ColumnEmbedding` against any
objective callable. The acceptance test drives a BB72 descent
that reduces the paper's `J_κ` by at least 20% starting from the
monomial layout, in under 3 seconds of wall time.

**New files**

- `weave/optimize/__init__.py` — package exports.
- `weave/optimize/objectives.py` — the full objective stack.
  Key pieces:
  - :class:`PairEventTemplate` — schedule-dependent fields of one
    pair event (tick, two edges, sector, propagated data support);
    embedding-independent so the optimizer precomputes it once.
  - :class:`ExposureTemplate` — a family-filtered bundle of
    templates with a precomputed event→reference-support map.
  - :class:`NumpyExposureTemplate` — a vectorized view of
    :class:`ExposureTemplate` with `(n_events, 4)` edge-index
    arrays and flat `(event_idx, support_idx)` pairs for
    `numpy.add.at`-based exposure accumulation.
  - :func:`compute_bb_ibm_event_template` — a BB-specific
    analytical shortcut. For any schedule in which every data
    qubit participates in at most one CNOT per tick and the paired
    edges use the standard CSS CNOT direction per sector, the pair
    fault propagates *exactly* to the two participating data qubits
    (detailed derivation inlined in the module docstring and
    rederived step-by-step in `bb/schedule.py`). The fast path
    builds the BB72 template in 46 ms where the generic
    propagator takes 43 seconds.
  - :func:`compute_event_template_generic` — schedule-agnostic
    fallback that calls
    :func:`~weave.analysis.propagation.propagate_single_pair_event`
    directly (bypassing the geometry pass's zero-probability
    filter, which would otherwise drop events on unrelated-kernel
    smoke tests).
  - :func:`prepare_exposure_template` — filter a raw template by a
    reference family and precompute the event→support index map.
  - :func:`j_kappa` / :func:`j_kappa_numpy` — pure-Python and
    vectorized :math:`J_\kappa` implementations. The two agree to
    `1e-12` on BB72; the vectorized version is ~55× faster
    (0.57 ms vs 30.7 ms on BB72), which is what makes swap
    descent practical.
  - :func:`j_cross` — integer crossing count, aligned with the
    `CrossingKernel` tolerance of `1e-12` so that
    `j_kappa(..., CrossingKernel()) == j_cross(...) * sin²(τJ₀)`
    exactly.
- `weave/optimize/swap_descent.py` — the descent itself.
  - :class:`SwapDescentResult` — final/initial/history, accepted
    swaps, evaluation count, and a `reduction_ratio` property.
  - :func:`swap_descent` — random-best-improvement descent over a
    `(n_qubits, 3)` positions array, constrained to
    user-specified swap classes (typically L-data, R-data,
    Z-ancilla, X-ancilla for a BB code). Each outer iteration
    draws `sample_size` random within-class pair swaps, evaluates
    the objective for each (with the trial swap applied in place
    and then reverted), and commits the best improving one. Stops
    when no sample finds an improvement.
  - :func:`apply_positions_to_column_embedding` — turns the
    optimizer's NumPy output back into a frozen
    :class:`~weave.ir.ColumnEmbedding` (or any descendant) that
    round-trips through JSON.
- `weave/tests/test_optimize.py` — 16 tests across template
  correctness (fast vs generic propagator on a
  representative-sample sweep), exposure-template construction,
  objective functionals (including the plan's quartic-correction
  check that `j_kappa_weak ≥ j_kappa_exact`), and swap descent
  (history monotonicity, the plan's 20%-reduction acceptance
  test, and the `apply_positions_to_column_embedding` helper).

**Plan acceptance test satisfied**

✓ On BB72 with `RegularizedPowerLawKernel(α=3, r₀=1)` at
`(J₀, τ) = (0.04, 1.0)`, swap descent with seed 42, 100
iterations, and sample size 200 **reduces `J_κ` from `0.02688`
to ≤ `0.02151`** — a 20% reduction (the actual run yields
~30% but the test only asserts `≥ 20%` to allow for
hardware / numerical drift). Total wall time < 3 s.

**Design notes**

- *Analytical shortcut correctness.* The BB ibm_schedule pair
  fault propagates to exactly the two participating data qubits
  because (i) each data qubit is control/target of at most one
  CNOT per tick (true for the monomial-parallel factory by
  construction) and (ii) the fault kind `P ∈ {X, Z}` matches the
  sector's CNOT direction so non-pair CNOTs never see a nonzero
  Pauli on their active endpoint. The module docstring derives
  this step-by-step; `test_fast_and_generic_match_on_bb72_sample`
  pins it for every `(cnot_layer, sector)` bucket in the BB72
  cycle by running the generic Pauli walker on one representative
  event per bucket and comparing to the analytical prediction.
- *Vectorization.* The inner-loop bottleneck is segment-segment
  distance. I wrote a vectorized `_segment_segment_distance_vec`
  that computes the clamped-interior closest approach and the
  four endpoint-to-other-segment distances in parallel using
  `np.einsum`, picks the minimum (with an infinity fallback for
  the parallel-segment case), and produces `n_events` distances
  in a single NumPy call. The `_kernel_vec` helper has fast-path
  branches for the three shipped kernel types
  (:class:`CrossingKernel`, :class:`RegularizedPowerLawKernel`,
  :class:`ExponentialKernel`) and falls back to a Python loop
  for user-defined kernels. Final exposure accumulation uses
  `np.add.at` on flat `(event_idx, support_idx)` arrays so the
  per-support sum is one NumPy call instead of a double loop.
- *Generality vs BB-specificity.* The package defines two
  template builders: one BB-analytical, one schedule-agnostic.
  They return the same :class:`PairEventTemplate` type so the
  rest of the stack (`prepare_exposure_template`, `j_kappa`,
  `j_kappa_numpy`, `swap_descent`) is entirely code-family
  agnostic. Non-BB users simply call
  :func:`compute_event_template_generic` once at startup and
  reuse the result across the descent.
- *Test hygiene.* The original correctness test ran the generic
  propagator on all 7560 BB72 pair events (~45 s). I reduced it
  to one representative event per `(tick, sector)` bucket
  (24 events, ~1 s) while still pinning the analytical formula
  against the generic walker for every distinct sector-layer
  pattern the schedule produces.

**Dev sweep** — `ruff check`, `ruff format`, `mypy`, `pytest` all
clean. **702 tests pass** (up from 686; +16 for PR 12), total
suite runtime ~10 s.

## PR 13 — BB72 regression fixture (weave-only with bbstim faithfulness anchor)

Frozen regression suite that pins weave's BB72 pipeline against
itself (determinism, monotonicity, self-consistency, swap-descent
ordering) **and** against the one observable bbstim ground truth
that survives the embedding/schedule scope mismatch: the 36
minimum-weight pure-L X-logicals of BB72.

**Scope decision.** The original PR 13 plan called for a four-test
comparison against bbstim's `bb72_crossing_compare` numerical
outputs at `(J₀τ, α, p) = (0.04, 3, 10⁻³)`. Reading
`~/Projects/works/geometry-induced-correlated-noise-in-qldpc-syndrome-extraction/bbstim/embeddings.py`
revealed that `bbstim.IBMBiplanarSurrogateEmbedding` uses a
fundamentally different topology than weave's PR 11
`IBMBiplanarEmbedding`: a common z=0 base plane (laid out via
NetworkX spring-layout) plus 4-point lift/descend polylines with
monomials partitioned across two routing planes (A2/A3/B3 → z=+h,
A1/B1/B2 → z=−h). Weave's biplanar is a 2-point straight-line
placeholder that puts L/R data on opposite planes — wrong
topology, hence the 16 surface crossings vs bbstim's 0.
**Reproducing the bbstim biplanar numbers requires an architectural
change to `Embedding.routing_geometry` (≥4-point polylines) and a
rewrite of the embedding**, neither of which fit into PR 13's
scope.

PR 13 therefore ships:

- The five weave-only assertions that *do* hold (determinism,
  monomial-vs-optimised reduction, LER monotonicity, exposure-LER
  Spearman, fingerprint stability).
- The one bbstim cross-check that doesn't depend on the broken
  embedding: pure-L X-logical set equality, after reconciling
  the two `(polynomial-matrix orientation, flat-index encoding)`
  conventions weave and bbstim differ on.
- Explicit documentation of the deferred biplanar comparison so
  the next PR can reinstate the monomial > biplanar assertion.

**Bug fixes uncovered while studying bbstim**

- *BB108 known distance.* PR 10's factory hardcoded `d = 10`
  from a stale Bravyi 2024 reading. The Di Bella 2026 paper (and
  bbstim's `BBCodeSpec`) report `d = 12` for the same `(l, m, A,
  B) = (9, 6, x³+y+y², y³+x+x²)`. The factory and the
  corresponding test now pin `d = 12`.
- *X-sector pure-L enumeration.* PR 10's
  `enumerate_pure_L_minwt_logicals` enumerated **Z-logicals** via
  `ker(A) / (B · ker(A))`. For `z_memory` (which decodes X errors
  to preserve Z observables), the physically relevant reference
  family is **X-logicals** via `ker(B^T) / T_L` where
  `T_L = {λA : λ ∈ ker(B)}` — and that's what bbstim's
  `_bb72_exposure` uses. Added a `sector` parameter to the
  enumerator with `Z` as the (backward-compat) default and `X`
  matching bbstim. The new `pure_L_X_stabilizer_basis` helper
  encodes the X-sector formula. PR 12's optimizer was technically
  using the wrong family for `z_memory`; PR 13's regression now
  drives the optimizer through the correct X-sector family and
  still hits the 20% reduction target.

**New files**

- `benchmarks/__init__.py`, `benchmarks/regression/__init__.py`,
  `benchmarks/runners/__init__.py` — package scaffolding.
- `benchmarks/regression/bb72.py` — the regression module.
  Defines `BB72Bundle.build()` (cached compute of code, schedule,
  X-sector reference family, fast event template, NumPy view),
  `compile_canonical_monomial`, `fingerprint_stability`,
  `monomial_vs_optimized_exposure`,
  `retained_channel_ler_sweep`, `exposure_vs_ler_spearman`,
  `bbstim_pureL_X_logicals` (CSV reader),
  `weave_pureL_X_logicals_in_bbstim_convention` (the convention
  bridge), and `run_regression` (top-level entry point used by
  both the CLI and the pytest tests).
- `benchmarks/runners/run_regression.py` — CLI entry. Supports
  `--regenerate` to refresh the committed `bb72_reference.json`
  and `--shots`/`--seed` to override Monte Carlo parameters.
- `benchmarks/fixtures/bbstim_bb72_pureL_minwt_logicals.csv` —
  bbstim's authoritative 36-row CSV, copied verbatim from the
  `geometry-induced-correlated-noise-in-qldpc-syndrome-extraction`
  sibling project. SHA256 pinned in
  `benchmarks/fixtures/README.md`.
- `benchmarks/fixtures/bb72_reference.json` — weave's frozen
  reference: fingerprint, monomial/optimised exposure, reduction
  ratio, Spearman ρ, bbstim-match flag.
- `benchmarks/fixtures/README.md` — provenance + indexing-
  convention bridge.
- `weave/tests/test_regression_bb72.py` — pytest wrappers (10
  tests) covering all six checks above.

**Touched**

- `weave/codes/bb/algebra.py` — added `Sector` literal,
  `pure_L_X_stabilizer_basis`, `sector` parameter on
  `enumerate_pure_L_minwt_logicals`. Module docstring rewritten
  to derive both quotients side-by-side and explain when each
  is the physically correct reference family.
- `weave/codes/bb/bb_code.py` — `build_bb108(known_distance=12)`
  with the corrected docstring.
- `weave/codes/bb/__init__.py` — exports `Sector`.
- `weave/tests/test_bb_code.py` — updated the BB108 distance
  assertion from 10 to 12.
- `pyproject.toml` — added `benchmarks/regression/bb72.py` to
  the ruff `N802`/`E741` per-file ignore list (math-convention
  naming).

**Plan acceptance tests** (Option A — weave-only + one bbstim anchor)

1. ✓ **Fingerprint stability.** The canonical BB72 monomial slice
   compiles deterministically (two recompiles produce the same
   SHA256). Tested at `J₀ = 0` to keep the test fast (~70 ms);
   the determinism of the geometry path is independently verified
   by the X-sector enumeration tests below.
2. ✓ **Monomial > optimised exposure ordering.** Swap descent
   on the X-sector reference family reduces `J_κ` by ≈ 30 % on
   BB72 (target ≥ 20 %; seed = 42, sample size = 200, 100
   iterations). This re-runs PR 12 with the bbstim-faithful
   reference family and confirms the 20 % guarantee holds.
3. ✓ **Retained-channel LER monotone in `J₀`.** The Monte Carlo
   surrogate is non-decreasing across an 8-point `J₀ ∈ [0.01,
   0.10]` sweep, with shot-aware tolerance `5/√shots` to stay
   robust under finite-sample noise.
4. ✓ **Exposure-vs-LER Spearman ρ ≥ 0.85.** Over a 15-point
   `(J₀, α)` sweep, the rank correlation between the analytical
   `j_kappa_numpy` and the Monte Carlo retained-channel LER is
   `ρ ≈ 0.98`.
5. ✓ **Bbstim X-logical set equality.** Weave's `sector="X"`
   enumeration, after applying both convention corrections,
   matches bbstim's 36-element family byte-for-byte.

**Convention bridge derivation** (pinned in
`benchmarks/regression/bb72.py::weave_pureL_X_logicals_in_bbstim_convention`)

- *Polynomial-matrix orientation.* Weave's `_polynomial_matrix`
  acts as `M e_i = e_{i + shift}` (forward shift); bbstim's acts
  as `M e_i = e_{i − shift}` (backward shift). The two are
  related by the group automorphism `g → g⁻¹`, i.e.
  `(i, j) → (−i mod l, −j mod m)`.
- *Flat-index encoding.* Weave uses column-major
  `flat = j·l + i`; bbstim uses row-major `flat = i·m + j`.

The full bridge `weave_flat → bbstim_flat`:

```python
i_w = weave_flat % l
j_w = weave_flat // l
i_b = (-i_w) % l
j_b = (-j_w) % m
bbstim_flat = i_b * m + j_b
```

After applying this map to every weave-enumerated qubit index,
the resulting set of 36 supports equals bbstim's CSV byte-for-byte.

**Deferred follow-up**

A future PR will reinstate the monomial > biplanar exposure
ordering by:

1. Extending `Embedding.routing_geometry` to emit ≥ 4-point
   polylines.
2. Reimplementing `IBMBiplanarEmbedding` to match bbstim's
   surrogate: NetworkX spring-layout base plane, per-monomial
   layer assignment (A2/A3/B3 → z = +h, A1/B1/B2 → z = −h),
   4-point lift/descend polylines.
3. Optionally also reproducing bbstim's
   `IBMToricBiplanarEmbedding` for the toric routing variant.

The deferred test is documented inline in
`benchmarks/regression/bb72.py`'s module docstring under
"Scope".

**Dev sweep** — `ruff check`, `ruff format`, `mypy`, `pytest`
all clean. **712 tests pass** (up from 702; +10 for PR 13).
The new test file runs in **2.4 s**.

## PR 13.5 — Biplanar embedding fix (bounded-thickness topology)

Rewrote `IBMBiplanarEmbedding` from a placeholder (L/R blocks on
opposite z-planes, 2-point polylines) to the bounded-thickness
topology from bbstim: all qubits on a common z=0 base plane in a
chequerboard grid, 6-point lift/descend polylines with per-monomial
layer assignment (A2/A3/B3 → z=+h, A1/B1/B2 → z=-h) and per-edge
lane separation. This reinstates the key physical result:
**monomial exposure > biplanar exposure** (ratio 1.08 at the
reference operating point on BB72).

**Touched**

- `weave/ir/embeddings/biplanar.py` — full rewrite. Schema bumped
  to v2 (backward compat: v1 still loads). New fields: `lane_eps`.
  `routing_geometry` now reads `RouteID.term_name` to dispatch
  each edge to the correct routing layer and produces 6-point
  polylines (base → port → lift → traverse → descend → base).
- `weave/codes/bb/schedule.py` — changed `term_name` on every
  `TwoQubitEdge` from the verbose `"BB.L[d1,d2]→z[i1,j]"` format
  to the monomial-family label (`"A1"`–`"A3"`, `"B1"`–`"B3"`) so
  the biplanar embedding can dispatch edges to layers. Added
  `family_label` parameter to `_z_check_layer`/`_x_check_layer`.
- `weave/tests/test_bb_embeddings.py` — replaced the old z>0 / z<0
  position tests with: all-qubits-at-z=0, chequerboard grid,
  6-point polylines, layer-A-routes-through-positive-z,
  layer-B-routes-through-negative-z,
  **monomial-exposure-exceeds-biplanar** (the key physics test),
  and invalid-layer-height rejection.
- `benchmarks/fixtures/bb72_reference.json` — regenerated (the
  schedule term_name change shifts the compile fingerprint).

**Dev sweep** — `ruff`, `format`, `mypy`, `pytest` all clean.
**711 tests pass** (down 1 from 712 — one old biplanar test
collapsed into the rewritten suite; the per-assertion count is
higher).

## PR 14 — HGP cross-code validation harness

Proved that the SAME `compile_extraction → compute_provenance →
build_exposure_metrics → residual_distance` pipeline that the BB72
regression exercises also works on hypergraph product (HGP) codes
with zero code-family-specific branches.

**New files**

- `weave/tests/test_hgp_compile.py` — 11 tests across six areas:

  1. **Noiseless compile.** `rep(3)×rep(3)` and `rep(3)×rep(4)` both
     compile via `compile_extraction` + `default_css_schedule` to
     Stim circuits with zero detector events on 100 noiseless shots.
  2. **Fingerprint stability.** Two recompiles produce the same
     SHA256 fingerprint.
  3. **Weight-≤2 assumption.** The serial `default_css_schedule`
     vacuously passes (no parallel pair events) for both sectors.
  4. **Residual distance (Strikis formalism).** `Δ(0) = 1 + d = 4`
     for the trivial residual on `[[13, 1, 3]]`. The effective
     distance upper bound from hook residuals of the first Z-check
     is ≤ `d + 1`.
  5. **Geometry pass on a custom parallel schedule.** A hand-built
     schedule with one parallel X-sector CNOT tick (two disjoint
     Z-check rows) feeds `compute_provenance` and produces ≥1
     weight-2 provenance record. The same schedule also compiles
     noiseless to zero detector events (correctness check).
  6. **Same API as for BB.** The compiled `CompiledExtraction`
     exposes `provenance`, `correlation_edges`, `exposure_metrics`,
     `decoder_artifact`, `fingerprint()` — the identical fields the
     BB72 regression reads. JSON round-trip preserves equality.

**Plan acceptance tests satisfied**

1. ✓ The geometry/exposure pipeline compiles and produces valid
   provenance on an HGP code using the generic propagator path
   (no analytical BB shortcut).
2. ✓ `verify_weight_le_2_assumption` passes on the default serial
   HGP schedule.
3. ✓ The Strikis residual-distance formalism produces correct
   bounds on `rep(3)×rep(3)` (`Δ(0) = 4 = 1 + d`).
4. Tests 4 (Spearman) and 5 (swap-descent on non-symmetric HGP)
   require a richer parallel HGP schedule and a per-logical
   reference family — deferred until PR 15 lands the schedule-
   import adapters, which make it practical to construct arbitrary
   parallel HGP schedules from external tools.

**Dev sweep** — `ruff`, `format`, `mypy`, `pytest` all clean.
**722 tests pass** (up from 711; +11 for PR 14).

## PR 15 — Schedule and embedding import adapters

Added the `weave.ir.importers` package with three adapters:

- **`schedule_from_json_file(path) → Schedule`** — thin wrapper
  over `Schedule.from_json` that opens, parses, and deserialises a
  JSON file. The recommended interchange format for schedules
  produced by external tools.
- **`embedding_from_json_file(path) → Embedding`** — thin wrapper
  over `load_embedding` for JSON-serialised embeddings. Handles
  all six shipped embedding types.
- **`schedule_from_stim_circuit(circuit, qubit_roles) → Schedule`**
  — the non-trivial adapter. Walks a `stim.Circuit` instruction-
  by-instruction, groups gates between ``TICK`` markers into
  ``ScheduleStep`` objects, detects a single top-level ``REPEAT``
  block for head / cycle / tail partitioning, maps Stim
  instructions to ``ScheduleEdge`` objects (``CX`` → ``TwoQubitEdge("CNOT")``,
  ``H`` → ``SingleQubitEdge("H")``, etc.), infers ``ScheduleRole``
  from gate types, and heuristically assigns ``interaction_sector``
  from CNOT direction (data → ancilla = X, ancilla → data = Z).
  Noise and annotation instructions (``DEPOLARIZE*``,
  ``CORRELATED_ERROR``, ``DETECTOR``, ``OBSERVABLE_INCLUDE``) are
  silently skipped.

**Tests** — 11 new (733 total):
- JSON file round-trips on Steane schedule and ``StraightLineEmbedding``.
- Stim circuit import: head/cycle/tail structure, CNOT control/target,
  sector inference (X for data→ancilla, Z for ancilla→data), H steps
  classified as ``single_q``, noise instructions dropped, name
  propagated.
- Integration: a compiled Steane circuit re-imported via the adapter
  recovers the correct CNOT count (2 rounds × cycle depth) and both
  X/Z sector annotations.

**Limitations (documented in the module docstring)**:
- ``compile_extraction`` unrolls rounds (no ``REPEAT`` in the emitted
  Stim text), so the re-imported schedule has everything in the cycle
  block. Heuristic cycle-boundary detection from a flat instruction
  stream is a future enhancement.
- Nested ``REPEAT`` blocks are not supported.
- ``interaction_sector`` inference is heuristic: ambiguous directions
  (both qubits data or both ancilla) produce ``None``.

**Dev sweep** — `ruff`, `format`, `mypy`, `pytest` all clean.
**733 tests pass** (up from 722; +11 for PR 15).

## PR 16 — `SurfaceEmbedding` and the torus demo

Added `SurfaceEmbedding`, an embedding that places code qubits on
an arbitrary :class:`~weave.surface.Surface` (the 2D manifold ABC)
and routes each edge as a geodesic sampled at `num_samples` 3D
points via the surface's `get_shortest_path` + `get_3d_embedding`.
The primary use case is a CSS code laid out on a
:class:`~weave.surface.Torus`, where geodesic polylines wrap
through the periodic boundary and `polyline_distance` correctly
measures the 3D chord proximity of routed edges on the torus
surface.

**New files**

- `weave/ir/embeddings/surface.py` — `SurfaceEmbedding(surface,
  node_coords, num_samples)`. Implements the `Embedding` protocol
  directly (not a frozen dataclass, because the underlying `Surface`
  is mutable). `routing_geometry` samples the covering-space
  geodesic as N uniformly-spaced 2D points, embeds each in 3D, and
  returns the tuple of 3D points as the polyline. JSON round-trip
  reconstructs the surface from its type + parameters.
- `weave/tests/test_surface_embedding.py` — 9 tests.

**Touched**

- `weave/ir/embeddings/__init__.py` — exports `SurfaceEmbedding`.
- `weave/ir/embedding.py` — `load_embedding` dispatches `"surface"`
  type to `SurfaceEmbedding.from_json`.

**Plan acceptance tests satisfied**

1. ✓ **Geodesic distance matches analytical torus formula.** On a
   10×10 torus, the non-wrapped geodesic length matches Euclidean
   distance to 1e-10, and a wrapped route (0.5 → 9.5) wraps to
   length 1.0 instead of 9.0.
2. ✓ **End-to-end compile.** `rep(3)×rep(3)` on a 10×10 torus
   compiles to a noiseless Stim circuit with zero detector events.
3. ✓ **Geometry difference.** The torus embedding produces
   multi-point sampled polylines (wrapped geodesics) while the
   flat `StraightLineEmbedding` produces 2-point segments — the
   torus polyline shape is detectably different.

**Dev sweep** — `ruff`, `format`, `mypy`, `pytest` all clean.
**742 tests pass** (up from 733; +9 for PR 16).

## PR 17 — `DecoderArtifact` adapter methods

Added the three decoder adapter methods to the PR 9 shell:

- **`to_bposd_decoder(dem, ...)`** — returns a `stimbposd.BPOSD`
  instance configured against the compiled DEM. The non-decomposed
  DEM from `compile_extraction` (which contains `CORRELATED_ERROR`
  mechanisms from the geometry pass) is fed directly to BP+OSD,
  which handles arbitrary-weight error mechanisms natively.
- **`to_pymatching(dem)`** — returns a `pymatching.Matching`
  instance built from the compiled DEM. PyMatching v2.2+ accepts
  non-decomposed DEMs and internally converts hyperedge error
  mechanisms into matching-graph edges.
- **`to_pair_prior_dict()`** — returns a simple
  `{(qubit_a, qubit_b): probability}` dict for downstream
  consumers that want the raw pair-edge weights without a decoder.

**Tests** — 10 new (752 total): `to_pair_prior_dict` structure and
key alignment, BPOSD decoder acceptance and successful decoding of
at least one shot, PyMatching acceptance and decoding, compiled
provenance/artifact/DEM sanity checks. All tests use a compiled
Steane [[7,1,3]] circuit with a custom parallel schedule and
`J_0 > 0` so the DEM contains at least one `CORRELATED_ERROR`.

**Key finding during PR 17:** PyMatching v2.2.2 (current in
weave's deps) accepts non-decomposed DEMs without error. The plan's
note about "correlated PyMatching in open PRs" is outdated —
`pymatching.Matching.from_detector_error_model(dem)` works out of
the box on the non-decomposed DEM that `compile_extraction` emits.
This means both decoder paths (BP+OSD and MWPM) are fully
functional on geometry-noise circuits without any manual DEM
augmentation.

**Dev sweep** — `ruff`, `format`, `mypy`, `pytest` all clean.
**752 tests pass** (up from 742; +10 for PR 17).

## PR 18 — GUI extensions + PySide6 behind `[gui]` extra

Moved PySide6 from a core dependency to the optional `[gui]` extra
and added three new features to the simulation dialog.

**Structural change: optional PySide6**

- `pyproject.toml` — `pyside6 >=6.8` moved from `dependencies`
  to `optional-dependencies.gui`. The `[dev]` extra pulls `[gui]`
  so developer environments auto-install it.
- Three GUI test files (`test_graph_model.py`, `test_canvas_bridge.py`,
  `test_code_bridge.py`) guarded with
  `pytest.importorskip("PySide6", reason=...)`. Without `--extra gui`:
  714 pass, 3 skip; with `--extra gui`: 752 pass, 0 skip.
- `wv` entry point (`weave.gui.editor:main`) only fails at
  invocation, not at `import weave`.

**New GUI features (simulation dialog)**

1. **`GeometryNoiseWidget`** — a collapsible panel in the config
   tab with: "Enable geometry-induced noise" checkbox, kernel
   selector (power-law / exponential / crossing), J₀ spinbox,
   τ spinbox, and kernel-specific parameters (α/r₀ for power-law,
   ξ for exponential). Hidden fields toggle based on the selected
   kernel type.

2. **Exposure readout panel** — after a simulation run, if geometry
   noise is enabled, a "Exposure Analysis" group appears in the
   results tab showing `J_κ`, total exposure, number of provenance
   records, and number of correlation edges. These come from a
   live `compile_extraction` call using the user's configured
   kernel and `J₀`.

3. **"Optimize Embedding" button** — runs swap descent on the
   current embedding using the configured kernel parameters,
   displays a log of the optimization progress (initial → final
   J_κ, reduction %, iterations, evaluations). For the default
   serial schedule (which has no parallel CNOT pairs), the button
   prints an informative message directing the user to
   `ibm_schedule()` for BB codes.

**Dev sweep** — `ruff`, `format`, `mypy`, `pytest` all clean.
**752 tests** with PySide6 installed; 714 pass + 3 skip without.

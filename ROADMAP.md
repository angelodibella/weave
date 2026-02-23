# Roadmap

This roadmap organizes the development of Weave around three auditing perspectives, followed by the integration work needed to unify the GUI editor with the simulation backend.

---

## I. Physical Correctness

The highest priority. Every circuit Weave generates must faithfully implement the intended QEC protocol.

### Completed (Feb 2026)

- ~~**X-stabilizer gate sequence**~~: Verified algebraically correct (H·H = I cancels between consecutive CNOTs). Optimized from per-CNOT `H-CNOT-H` wrapping to single `H-CNOT-...-CNOT-H` bracket per check.
- ~~**Initialization and measurement bases**~~: Verified `R`/`M` for z_memory and `RX`/`MX` for x_memory match the intended bases.
- ~~**Detector record indexing**~~: All `stim.target_rec` offsets audited and verified correct for both experiments — first-round, comparison, and final detectors.
- ~~**`OBSERVABLE_INCLUDE` targets**~~: Fixed critical bug where logical operators were not in a symplectic basis for k > 1 codes. Added symplectic Gram-Schmidt pairing.
- ~~**`find_logicals()` sparse matrix crash**~~: Fixed incompatibility with ldpc 2.x sparse returns.
- ~~**Crossing noise**~~: Verified that qubit pairs in each crossing correctly correspond to Tanner graph edge endpoints.
- ~~**Circuit noise placement**~~: Verified that post-CNOT `PAULI_CHANNEL_2` matches standard Stim conventions.

### Remaining

- **TICK markers**: The circuit does not use Stim `TICK` instructions to delineate time steps. Without TICKs, Stim cannot distinguish parallel from sequential gates, which affects error analysis, circuit visualization, and the detector error model structure. Adding TICKs requires defining a gate scheduling strategy.
- **CNOT scheduling**: The current circuit applies all Z-check CNOTs sequentially, then all X-check CNOTs sequentially. For HP codes, gate ordering can introduce hook errors that reduce the effective distance. Investigate layer-by-layer or greedy scheduling to parallelize gates while preserving code distance.
- **Idle noise**: Qubits not participating in a gate during a given time step should still experience decoherence (`DEPOLARIZE1` or `PAULI_CHANNEL_1`). Currently, single-qubit noise is applied once per round after all gates, regardless of how many time steps a qubit was idle. Requires TICK-based scheduling first.

---

## II. Bugs, Modernization, and Optimization

### Completed (Feb 2026)

- ~~**Eager circuit generation**~~: `CSSCode.circuit` is now a lazy property. Circuit is only generated on first access; `embed()` invalidates the cache and triggers regeneration on next access. No more double-build.
- ~~**Stale docstrings**~~: `HypergraphProductCode` docstring updated to reflect `H1`/`H2` matrix parameters.
- ~~**`_reorder_matrix` assumption**~~: Fixed to handle `r1 > n1` by iterating up to `max(n1, r1)` instead of only `n1`.
- ~~**`NoiseModel` type check**~~: Replaced `np.issubdtype(type(param), np.number)` with `isinstance(param, (int, float))`. Changed from `assert` to `raise ValueError` for proper error handling.
- ~~**GUI BFS queue**~~: Replaced `list.pop(0)` with `collections.deque.popleft()` in `Canvas._detect_connected_component`.
- ~~**Type annotations**~~: Migrated all source files from `typing.List`, `typing.Tuple`, `typing.Optional`, `typing.Union` to built-in generics (`list`, `tuple`, `X | None`, `X | Y`).
- ~~**Build system**~~: Migrated from `poetry-core` to `hatchling` as build backend. Moved test dependencies to `[project.optional-dependencies]`.
- ~~**Testing coverage**~~: Expanded from 30 to 50 tests. New coverage includes:
  - Steane code [[7,1,3]] circuit correctness (noiseless detectors, noisy DEM, logical validity).
  - Lazy circuit generation and invalidation behavior.
  - CSS condition violation detection.
  - NoiseModel edge cases (zero, int input, wrong lengths for all channels).
  - Code distance computation, asymmetric HP codes, reordering preservation of CSS condition.
- ~~**Torus warning**~~: Replaced `print()` warning with `warnings.warn()`.

### Remaining

- **Sparse matrices**: `pcm.hypergraph_product` uses dense NumPy arrays and `np.kron`. For larger codes, switch to `scipy.sparse` for the Kronecker products and parity-check storage.

### Performance

- **Crossing detection**: `find_edge_crossings` is O(E²). For large Tanner graphs this becomes a bottleneck. Consider a sweep-line algorithm or spatial indexing (R-tree).
- **Circuit construction**: Building Stim circuits by appending instruction-by-instruction is slow for large codes. Consider building the circuit from a string template or using Stim's `CircuitRepeatBlock`.

---

## III. Novelty and Extensibility

### Crossing Error Model (Current Novelty)

The treatment of Tanner graph edge crossings as a source of correlated noise is the distinguishing feature of Weave. Directions to strengthen this:

- **Parameterized crossing models**: Allow different noise channels per crossing (e.g., weighted by geometric distance between the crossing qubits, or by the angle of intersection).
- **Crossing-aware decoding**: Feed crossing information into the decoder's error model so that belief propagation or MWPM can account for correlated errors.
- **Embedding optimization**: Implement heuristics (e.g., simulated annealing on node positions) to minimize the crossing number, directly reducing the code's effective noise floor.

### Embedding-Based Noise Models

- **Distance-based crosstalk**: Noise between qubit pairs as a continuous function of their Euclidean distance in the embedding. Closer qubits experience stronger correlated noise regardless of whether their edges cross. Could use an inverse-power-law or exponential decay model.
- **Edge-length amplitude damping**: Noise on each edge proportional to its geometric length in the embedding, modeling signal attenuation over longer physical connections. Longer wires in the layout incur more decoherence.

### Beyond Pauli Noise

- **Coherent errors**: Stim is a Clifford simulator and cannot natively model coherent rotations. To support coherent errors, integrate with a density-matrix or state-vector simulator (e.g., Qiskit Aer, Cirq) for small codes, or develop a Pauli twirling approximation layer.
- **Leakage**: Model leakage as an additional qubit state. This requires extending the noise model and potentially the circuit representation beyond what Stim supports natively. One approach: a hybrid simulator that tracks leaked qubits separately.
- **Non-Pauli correlated noise**: Extend `PAULI_CHANNEL_2` to arbitrary two-qubit channels via process matrices, then twirl to Pauli for Stim compatibility.

### Topological Generalization

- **Surface embeddings in the GUI**: The `Surface` ABC and `Torus` implementation already exist in `weave/surface/` but are not connected to the GUI. Steps:
  1. Add a surface selector to the GUI (plane, torus, Klein bottle, genus-g surface).
  2. Replace the flat ℝ² canvas with a surface-aware coordinate system — for the torus, this means periodic boundary conditions on the canvas edges.
  3. For non-flat surfaces, add a 3D viewport (using Qt3D or VTK) alongside the 2D projection.
- **Biplanar / multi-layer layouts**: Some code families (e.g., bivariate bicycle codes) are naturally embedded on two planes connected by inter-layer edges. Support this as a special case of a product surface.
- **Arbitrary parametric surfaces**: Allow users to define a surface via a parametric map `(u, v) → (x, y, z)` and metric tensor, then use geodesic calculations for edge routing and crossing detection on the surface.

### Chain Complexes and Fiber Bundles

- **Chain complex framework**: Generalize the code hierarchy from "two parity-check matrices HX, HZ" to a chain complex `C_n → C_{n-1} → ... → C_0` with boundary maps. CSS codes are the `n=2` case. This enables:
  - Higher-dimensional codes (e.g., 3D toric codes, fracton models).
  - Homological product codes as a generalization of hypergraph products.
  - A unified interface: `ChainComplex` with methods `boundary(i)`, `coboundary(i)`, `homology(i)`.
- **Fiber bundles**: Twisted products where the fiber varies over the base — models codes like the Hastings-Haah fiber bundle codes. Requires:
  - A `FiberBundle` class that takes a base complex and a fiber complex with a twist (automorphism of the fiber over each edge of the base).
  - Circuit generation that respects the bundle structure.

### GUI ↔ Simulation Bridge

Currently the GUI (`weave/gui/`) and the simulation logic (`weave/codes/`, `weave/util/`) are completely decoupled. The canvas stores nodes and edges as plain dicts; the simulation expects NumPy parity-check matrices. To unify them:

1. **Canvas → Code extraction**: Implement `Canvas.to_css_code()` that:
   - Reads the quantum nodes (qubits, Z-stabilizers, X-stabilizers) and edges.
   - Constructs HX and HZ matrices from the Tanner graph adjacency.
   - Extracts node positions as the embedding for crossing detection.
   - Returns a `CSSCode` instance ready for simulation.

2. **Code → Canvas loading**: Implement `Canvas.from_css_code(code)` that:
   - Takes a `CSSCode` (or `HypergraphProductCode`) and populates the canvas with its Tanner graph.
   - Uses the code's embedding positions (from `embed()`) to place nodes.

3. **In-GUI simulation**: Add a "Simulate" action to the GUI menu that:
   - Extracts the code from the canvas.
   - Runs a Sinter/Stim simulation with user-specified noise parameters.
   - Displays results (logical error rate, decoding success) in a panel or dialog.

4. **Live feedback**: As the user drags nodes, update the crossing count in real time (already partially implemented) and optionally show the estimated impact on logical error rate.

---

## Execution Order

We tackle these in dependency order:

1. **Physical correctness audit** (Phase I) — must come first; everything else builds on a correct foundation.
2. **Bug fixes and modernization** (Phase II) — clean up the codebase so extensions are built on solid ground.
3. **GUI ↔ Simulation bridge** (Phase III, item 4) — the most immediate user-facing improvement.
4. **Topological generalization** (Phase III) — surface embeddings in the GUI, then 3D.
5. **Chain complexes / fiber bundles** (Phase III) — the deep algebraic generalization.
6. **Beyond Pauli noise** (Phase III) — requires external simulator integration, lowest priority.

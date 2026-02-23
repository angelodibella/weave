# Roadmap

This roadmap organizes the development of Weave around three auditing perspectives, followed by the integration work needed to unify the GUI editor with the simulation backend.

---

## I. Physical Correctness

The highest priority. Every circuit Weave generates must faithfully implement the intended QEC protocol.

### Stabilizer Measurement Circuit

- **X-stabilizer gate sequence**: `CSSCode.generate()` currently implements X-checks as `H → CNOT(check, data) → H` per data qubit. Verify this is equivalent to measuring the X-type stabilizer generators (it should be, but the Hadamards should bracket the *entire* CNOT fan-out for a given check, not each individual CNOT — the current per-CNOT wrapping is physically incorrect if multiple data qubits participate in one X-check, since intermediate Hadamards would interfere).
- **CNOT ordering and parallelism**: The current circuit applies CNOTs sequentially within each check. For HP codes the gate ordering can introduce hook errors that change the effective distance. Investigate whether a specific CNOT scheduling (e.g., the "greedy" or "layer-by-layer" approach) is needed to preserve distance.
- **Initialization and measurement bases**: Verify that `z_memory` initializes in |0⟩ and measures in Z, while `x_memory` initializes in |+⟩ and measures in X. The current code does `R` vs `RX` and `M` vs `MX` — confirm these Stim instructions match the intended bases.

### Detector and Observable Correctness

- **Detector record indexing**: The first-round detectors, mid-round detectors, and final-round detectors each use different `stim.target_rec` offset calculations. Audit these carefully — off-by-one errors here silently produce wrong decoding graphs without raising exceptions.
- **Final-round detectors iterate in reverse** (`HZ[-1-k]`, `HX[-1-k]`) — verify this matches the measurement record ordering.
- **`OBSERVABLE_INCLUDE` targets**: Logical operators are extracted via `find_logicals()` which uses GF(2) linear algebra from `ldpc.mod2`. Verify that the returned representatives are valid logical operators (i.e., they commute with all stabilizers but are not themselves stabilizers).

### Noise Model Physicality

- **Crossing noise**: Applied as `PAULI_CHANNEL_2` on pairs of qubits whose Tanner graph edges cross. Document the physical justification — this models correlated Pauli noise from qubit cross-talk during parallel gate execution. Verify that the qubit pairs in each crossing are correct (the current code uses edge endpoint indices).
- **Circuit noise placement**: Two-qubit `PAULI_CHANNEL_2` noise is applied *after* each CNOT. Verify this matches the convention used by standard Stim circuit noise models (before vs. after matters for error propagation).
- **Single-qubit noise timing**: `PAULI_CHANNEL_1` on data, Z-check, and X-check qubits is applied once per round after all gates. Consider whether idle noise should also be applied during time steps when a qubit is not involved in a gate.

### Logical Operator Extraction

- **`find_logicals()`**: Uses `nullspace(HX)` to get the kernel, then `row_basis(HZ)` to get the stabilizer span, stacks them, and extracts independent rows via pivot selection. Verify this correctly produces `k` independent logical Z (resp. X) operators. Edge case: codes with `k = 0` should be handled gracefully.

---

## II. Bugs, Modernization, and Optimization

### Known Issues

- **Eager circuit generation**: `CSSCode.__init__()` calls `self.generate()` at construction time, then `embed()` clears and regenerates the circuit. This means every code object builds its circuit twice if embedded. Consider making generation lazy or removing the `generate()` call from `__init__`.
- **Stale docstrings**: `HypergraphProductCode`'s docstring still references `clist1`/`clist2` parameters, but the constructor now takes `H1`/`H2` matrices directly.
- **`_reorder_matrix` assumption**: The interleaving loop in `pcm._reorder_matrix` assumes `n1 >= r1`. If `r1 > n1`, some right-split blocks are silently dropped. Add a guard or handle the asymmetric case.
- **`NoiseModel` type check**: `np.issubdtype(type(param), np.number)` may not reliably detect plain Python `float`/`int` across all NumPy versions. Consider `isinstance(param, (int, float))` as a more robust check.
- **GUI BFS queue**: `Canvas._detect_connected_component` uses `list.pop(0)` which is O(n). Use `collections.deque` for O(1) popleft.

### Modernization

- **Type annotations**: Migrate from `typing.List`, `typing.Tuple`, etc. to built-in generics (`list`, `tuple`) — available since Python 3.10, which is our minimum.
- **Sparse matrices**: `pcm.hypergraph_product` uses dense NumPy arrays and `np.kron`. For larger codes, switch to `scipy.sparse` for the Kronecker products and parity-check storage.
- **Testing coverage**: Expand `pytest` tests to cover:
  - Circuit correctness: compare detector error models against known-good references for small codes (e.g., [[7,1,3]] Steane code from `hamming(7)`).
  - Logical operator validity: check `HX @ lz.T % 2 == 0` and `HZ @ lx.T % 2 == 0`.
  - Noise model edge cases: list vs. scalar input, mismatched lengths.
  - Round-trip `to_clist` / `to_matrix` fidelity.

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

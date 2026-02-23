# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Weave is a Python framework for constructing, visualizing, and simulating quantum error-correcting codes (QECs), with a focus on CSS codes and hypergraph product codes. It uses Stim for circuit simulation and supports modeling qubit cross-talk via Tanner graph edge crossings.

## Build & Development

Uses **uv** for dependency management (Python >=3.10, <3.14). Build backend is **hatchling**.

```bash
uv sync                                            # Install dependencies
uv run pytest                                      # Run all tests (50 tests)
uv run pytest weave/tests/test_pcm.py              # Run a single test file
uv run pytest weave/tests/test_pcm.py::test_name   # Run a single test
uv run wv                                          # Launch the PySide6 GUI editor
```

## Architecture

### Code Hierarchy (`weave/codes/`)
- `base.py` — `NoiseModel` (noise parameter container), `ClassicalCode` (parity-check matrix wrapper), `QuantumCode` (ABC requiring `generate()` → `stim.Circuit`)
- `css_code.py` — `CSSCode(QuantumCode)`: takes HX/HZ parity-check matrices, builds Stim circuits with stabilizer rounds, detectors, and logical observables. The `circuit` property is **lazy** — generated on first access, invalidated by `embed()`. Supports `z_memory` and `x_memory` experiments. `embed()` computes a Tanner graph layout and finds edge crossings for crossing noise.
- `hypergraph_product_code.py` — `HypergraphProductCode(CSSCode)`: takes two classical parity-check matrices H1, H2, computes the hypergraph product (HX, HZ), and delegates to CSSCode.

### Utilities (`weave/util/`)
- `pcm.py` — Parity-check matrix operations: `repetition()`, `hamming()`, `hypergraph_product()`, `to_clist()`/`to_matrix()` conversions, `distance()`. Includes our own GF(2) linear algebra (`row_echelon`, `nullspace`, `row_basis`, `row_reduce`, `find_pivot_columns`) — no external dependency for this.
- `graph.py` — Tanner graph visualization: layout computation (`compute_layout`), edge crossing detection (`find_edge_crossings`), and matplotlib drawing (`draw`).

### Surfaces (`weave/surface/`)
- `base.py` — `Surface` ABC defining the interface for 2D manifolds used to embed Tanner graphs (intrinsic coords, geodesics, intersection checks, 2D/3D projections).
- `torus.py` — `Torus(Surface)`: flat torus with periodic boundary conditions. Handles path wrapping, segment intersection via shifted copies, Liang-Barsky clipping for 2D drawing, and 3D torus embedding.

### GUI (`weave/gui/`)
PySide6-based visual editor launched via `wv` command. Entry point: `editor.py:main()`.

## Key Concepts

- **Crossings**: When a Tanner graph is embedded in 2D, edges that cross represent potential qubit cross-talk. `CSSCode.embed()` detects these and injects `PAULI_CHANNEL_2` noise into the Stim circuit.
- **Stim circuits**: Generated in `CSSCode.generate()` with a head (initialization) → repeated rounds (stabilizers + noise + measurement) → tail (final measurement + detectors + `OBSERVABLE_INCLUDE` for logical operators).
- **Symplectic pairing**: For k>1 codes, logical operators are paired via `_symplectic_gram_schmidt()` so that X_Li anticommutes only with Z_Li, required for correct `OBSERVABLE_INCLUDE` assignment.

## Code Style

- Type annotations use built-in generics (`list`, `tuple`, `X | None`) — no `typing.List`/`typing.Tuple`/`typing.Optional`.
- `from __future__ import annotations` is used in `css_code.py` for forward references.
- Errors use `raise ValueError(...)` not `assert` for input validation.

## Dependencies

Core: numpy, stim, sinter, stimbposd, sympy, networkx, matplotlib, PySide6, notebook.

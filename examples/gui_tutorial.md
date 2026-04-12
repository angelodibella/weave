# Weave GUI Tutorial

This tutorial walks you through the weave visual editor from first
launch to running a simulation with geometry-induced correlated noise.
It takes about 10 minutes.

## 1. Launch the editor

```bash
uv sync --extra gui
uv run wv
```

The editor opens with an empty canvas. The **hamburger menu** (☰) in the
top-right corner gives access to all features. The **status bar** at the
bottom shows live node/edge/graph counts.

## 2. Load a code from the template library

1. Click the **hamburger menu** (☰).
2. Click **New Code from Template...**
3. In the dialog, select **Steane [[7, 1, 3]]** from the dropdown.
4. Click **Load onto Canvas**.

The Steane code's Tanner graph appears: 7 grey circles (data qubits),
3 blue rounded squares (Z-stabilizers), and 3 pink rounded squares
(X-stabilizers), connected by edges. The spring-force layout arranges
them automatically.

> **Status bar** now shows: `Nodes: 13 | Edges: 24`

## 3. Navigate the canvas

| Action | How |
|---|---|
| **Pan** | Click and drag empty space |
| **Zoom** | Mouse wheel, or `Ctrl+=` / `Ctrl+-` |
| **Reset view** | `Ctrl+0` |

Try zooming in to see individual nodes and edges clearly.

## 4. Select and rearrange nodes

- **Click** a node to select it (blue highlight).
- **Drag** a selected node to reposition it.
- **Shift+drag** on empty space to draw a selection rectangle.
- **Ctrl+A** selects all nodes.
- **Escape** deselects everything.
- **Delete** or **Backspace** removes selected nodes and their edges.

Rearrange the Steane code into a layout you like. Toggle grid snapping
with **G** for cleaner alignment.

## 5. Detect a graph and view code parameters

1. **Right-click** any node.
2. Click **Detect** in the context menu.
3. A coloured border appears around the connected component.

> **Status bar** now shows: `Nodes: 13 | Edges: 24 | Graphs: 1 | [[7, 1]]`
>
> The `[[7, 1]]` indicates the Steane code has 7 data qubits and 1
> logical qubit.

## 6. Run a noiseless simulation

1. Open the **hamburger menu** → **Simulate...**
2. In the **Configuration** tab:
   - Leave all noise channels at their defaults.
   - Leave the experiment as `z_memory`.
3. Switch to the **Simulation** tab:
   - Set **Rounds** to 3.
   - Set **Shots** to 10000.
   - Select **bposd** as the decoder.
4. Click **Run Simulation**.

After a few seconds the results appear:

- **Logical error rate**: should be very close to 0 (noiseless code).
- **Shots**: 10000.
- **Errors**: 0 or very few.

## 7. Add circuit noise

1. Go back to the **Configuration** tab.
2. Set **Circuit (2Q gate) noise** to `0.001`.
3. Set **Data qubit noise** to `0.001`.
4. Switch to **Simulation** → click **Run Simulation** again.

Now the error rate is nonzero. This is the physical error rate
of the Steane code under depolarizing noise.

## 8. Enable geometry-induced noise

1. In the **Configuration** tab, scroll to **Geometry-Induced Noise**.
2. Check **Enable geometry-induced noise**.
3. Set:
   - **Kernel**: `power_law`
   - **J₀**: `0.04`
   - **τ**: `1.0`
   - **α**: `3.0`
   - **r₀**: `1.0`
4. Switch to **Simulation** → click **Run Simulation**.

After the simulation finishes, a new **Exposure Analysis** panel appears:

- **J_κ**: the maximum per-support exposure from the retained channel.
- **Total exposure**: the sum of all pair probabilities.
- **Pair events**: number of `CORRELATED_ERROR` instructions emitted.
- **Correlation edges**: number of distinct data-qubit pairs affected.

> **Note**: The default serial schedule (one CNOT per tick) produces
> zero pair events because no two CNOTs fire in parallel. For
> nontrivial geometry noise, use the Python API with `ibm_schedule()`
> on BB codes.

## 9. Try a larger code

1. **Hamburger menu** → **New Code from Template...**
2. Select **BB72 [[72, 12, 6]]** → **Load onto Canvas**.

The BB72 Tanner graph is large (144 nodes, ~432 edges). Zoom out with
the scroll wheel to see the full layout.

> **Tip**: BB72 is the primary benchmark code in the paper. Its
> exposure ordering (monomial > biplanar) and the swap-descent
> optimizer are the main results.

## 10. Keyboard shortcut reference

Press **Help & Shortcuts...** in the hamburger menu (or read the
summary below):

| Shortcut | Action |
|---|---|
| `Ctrl+0` | Reset zoom |
| `Ctrl+A` | Select all |
| `Ctrl+C` / `Ctrl+V` | Copy / Paste |
| `Ctrl+O` / `Ctrl+S` | Open / Save |
| `Ctrl+=` / `Ctrl+-` | Zoom in / out |
| `G` | Toggle grid snap |
| `Escape` | Deselect all |
| `Delete` | Delete selected |
| Right-click | Context menu |

## 11. Next steps

- **Python API**: For advanced workflows (BB embeddings, swap-descent
  optimization, custom schedules, torus embeddings), use the Python
  API directly. See `examples/Tutorial.ipynb`.
- **Export**: Save your canvas to JSON via `Ctrl+S` for later editing,
  or export the parity-check matrices to CSV via the hamburger menu.
- **Reference**: The full API documentation is in the module
  docstrings. Start with `weave.compiler.compile_extraction` and
  `weave.codes.bb`.

---

*Weave — a geometry-aware compiler for CSS syndrome extraction.*

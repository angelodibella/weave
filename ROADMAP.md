# Roadmap

This roadmap outlines the steps to verify the current functionality, improve code quality and efficiency, and then extend Weave with additional features.

---

## Phase 1: Verification of Functionality

**Objective:**  
Ensure that the Hypergraph Product (HP) codes are being constructed correctly and that the calculations reflect the underlying theory as accurately as possible.

- **Validate HP Code Construction:**
  - Run existing examples and notebooks (e.g., `Analysis.ipynb` and `Debug.ipynb`) to verify that HP codes are generated according to theory.
  - Compare generated Tanner graphs, crossing calculations, and stim circuits against theoretical predictions.
  - Identify and document any discrepancies for further investigation.

- **Review Stim Circuit Generation:**
  - Examine the circuit construction (especially in `hypergraph_product_code.py`) to verify that the sequence of operations (initialization, CNOTs, Pauli channels, measurements, detectors, etc.) aligns with the expected simulation protocol.
  - Ensure that both `z_memory` and `x_memory` experiments produce logically consistent outcomes.

---

## Phase 2: Code Quality and Efficiency Improvements

**Objective:**  
Raise the standard of the codebase by adhering to best Python practices and optimizing performance where possible.

- **Code Standards and Refactoring:**
  - Review and update docstrings and inline comments for clarity.
  - Refactor duplicated code between `CSSCode` and `HypergraphProductCode` to prepare for future inheritance or modularization.
  - Ensure the project adheres to PEP8 style guidelines (using tools like `flake8` or `black`).

- **Performance Profiling:**
  - Profile key parts of the code (e.g., hypergraph product computation, crossing detection, circuit construction) to identify bottlenecks.
  - Explore optimization strategies (vectorization, efficient use of NumPy, etc.) within Python.
  - Document performance improvements and any remaining challenges.

- **Testing Framework:**
  - Begin writing unit tests (using `pytest`) for critical components:
    - NoiseModel parameter validation.
    - Correct construction of parity-check matrices from `clist`.
    - Graph generation and crossing detection.
  - Set up a preliminary CI/CD pipeline (e.g., GitHub Actions) to run tests on commits and pull requests.

---

## Phase 3: Extended Functionality and Documentation

**Objective:**  
After verifying functionality and improving code quality, focus on extending the framework and enriching documentation.

- **Enhanced Documentation:**
  - Write detailed guides on:
    - The HP code construction and its theoretical background.
    - How Stim simulates QEC circuits.
    - The differences between `z_memory` and `x_memory` experiments.
    - Explanation of key instructions like `OBSERVABLE_INCLUDE` and how they relate to the Logical Error Rate (LER).
  - Create additional Jupyter notebooks or tutorials to illustrate advanced usage.

- **Interactive Visualization:**
  - Explore interactive visualization libraries (Plotly, Bokeh) to complement the current matplotlib/networkx plots.
  - Prototype interactive Tanner graph visualizations with features like zoom, pan, and tooltip details.

- **Advanced QEC Features:**
  - Begin implementing the `ClassicalCode` class for canonical linear bit-check sequences.
  - Plan for the eventual integration of a generalized `CSSCode` class that can support both X- and Z-type stabilizers.

---

## Phase 4: Long-Term Goals

**Objective:**  
Lay the groundwork for performance enhancements and broader adoption of Weave.

- **Performance Enhancements:**
  - Identify and integrate C/C++ modules using pybind11 for performance-critical sections, especially those related to the hypergraph product and noise simulation loops.
  - Investigate potential improvements or alternatives to Stim for faster QEC simulations.

- **Web-Based and Interactive Tools:**
  - Develop a web interface or Jupyter widget for configuring simulations, visualizing graphs interactively, and real-time result monitoring.
  - Enhance community engagement with extensive tutorials and example pipelines.

- **Broader Release Preparation:**
  - Finalize documentation, tutorials, and testing frameworks.
  - Package the project for distribution (PyPI, conda, etc.).
  - Establish CI/CD pipelines and ensure comprehensive test coverage.


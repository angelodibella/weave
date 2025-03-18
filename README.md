![Canvas example](docs/images/surface_code_canvas.png)

# Weave

A quantum error correction framework that can implement and visualize qubit cross-talk.

## Structure

Weave is structured as a clean hybrid C++/Python codebase with Python at the top level and C++ integrated within:

- `weave/`: All code (both Python and C++)
  - `weave/__init__.py`: Package exports and imports from C++ module
  - `weave/__main__.py`: Entry point for CLI
  - `weave/gui/`: GUI implementation using PySide6
  - `weave/simulator.py`: Python implementation of simulation tools
  
  - `weave/_core/`: C++ implementation (internal)
    - `weave/_core/include/`: Header files
      - `weave/_core/include/weave/`: Core library headers
      - `weave/_core/include/bindings/`: Python binding headers (`.pybind.hpp` files)
    - `weave/_core/src/`: Implementation files
      - `weave/_core/src/codes/`: Implementation of quantum error correction codes
      - `weave/_core/src/util/`: Utility functions and classes
      - `weave/_core/src/bindings/`: Python binding implementation

## Installation

```bash
# Development installation
pip install -e .
```

## Usage

```python
import weave

# Create a hypergraph product code
code = weave.HypergraphProductCode()

# Set up some matrices
import numpy as np
pcx = [[1, 0, 1], [0, 1, 1]]
pcz = [[1, 1, 0], [1, 0, 1]]

# Generate the code
code.generate(pcx, pcz)

# Get code parameters
n, k, d = code.get_parameters()
print(f"Code parameters: [[{n}, {k}, {d}]]")

# Get stabilizers
stabilizers = code.get_stabilizers()
print("Stabilizers:", stabilizers)
```

## GUI

To launch the GUI:

```bash
python -m weave
```

## Development

### Prerequisites

- Python 3.10+
- C++ compiler supporting C++20
- CMake 3.15+
- pybind11

### Building from source

```bash
# Clone the repository
git clone https://github.com/yourusername/weave.git
cd weave

# Install development dependencies
pip install -e ".[dev]"

# Build the project
pip install -e .
```

### Running tests

```bash
pytest
```

### Developer's Guide

#### Adding new C++ components

1. **Create header file**
   
   Add a new header in `weave/_core/include/weave/` or an appropriate subdirectory:
   
   ```cpp
   // weave/_core/include/weave/your_component.hpp
   #pragma once
   
   namespace weave {
   
   class YourComponent {
   public:
       // Public methods
       void doSomething();
       
   private:
       // Private members
   };
   
   } // namespace weave
   ```

2. **Implement the component**
   
   Add implementation in `weave/_core/src/` using the same structure:
   
   ```cpp
   // weave/_core/src/your_component.cpp
   #include "weave/your_component.hpp"
   
   namespace weave {
   
   void YourComponent::doSomething() {
       // Implementation
   }
   
   } // namespace weave
   ```

3. **Create Python binding**
   
   Add binding header:
   
   ```cpp
   // weave/_core/include/bindings/your_component.pybind.hpp
   #pragma once
   
   #include <pybind11/pybind11.h>
   #include "weave/your_component.hpp"
   
   namespace py = pybind11;
   
   namespace weave {
   namespace bindings {
   
   inline void bind_your_component(py::module& m) {
       py::class_<YourComponent>(m, "YourComponent")
           .def(py::init<>())
           .def("do_something", &YourComponent::doSomething);
   }
   
   } // namespace bindings
   } // namespace weave
   ```

4. **Register the binding**
   
   Update the main binding file:
   
   ```cpp
   // weave/_core/src/bindings/weave.pybind.cpp
   // Add include
   #include "bindings/your_component.pybind.hpp"
   
   PYBIND11_MODULE(_core, m) {
       // ...
       // Add your new binding
       weave::bindings::bind_your_component(m);
   }
   ```

5. **Update CMakeLists.txt**
   
   Add your new source file:
   
   ```cmake
   set(WEAVE_CORE_SOURCES
       # ...
       weave/_core/src/your_component.cpp
   )
   ```

6. **Expose in Python**
   
   Update `weave/__init__.py`:
   
   ```python
   from ._core import (
       # ...
       YourComponent,
   )
   
   __all__ = [
       # ...
       "YourComponent",
   ]
   ```

#### Adding Python components

1. **Create Python module**
   
   Add a new Python file in `weave/` or a subdirectory:
   
   ```python
   # weave/your_module.py
   
   class YourPythonComponent:
       def __init__(self):
           # Initialize
           pass
           
       def do_something(self):
           # Implementation
           pass
   ```

2. **Expose in the package**
   
   Update `weave/__init__.py`:
   
   ```python
   from .your_module import YourPythonComponent
   
   __all__ = [
       # ...
       "YourPythonComponent",
   ]
   ```

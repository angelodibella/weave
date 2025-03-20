#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bindings/codes.pybind.hpp"
#include "bindings/util.pybind.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "Weave quantum error correction library core C++ bindings";

    // Bind utility functions.
    weave::bindings::bind_util(m);

    // Bind codes.
    weave::bindings::bind_codes(m);
}

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace weave {
namespace bindings {

/**
 * Create Python bindings for all utility functions.
 *
 * This function creates bindings for all utility functions defined in the weave/util/ directory.
 *
 * @param module The pybind11 module to add the bindings to.
 */
void bind_util(py::module& module);

/**
 * Create Python bindings for the PCM (parity-check matrix) utility functions.
 *
 * @param module The pybind11 module to add the bindings to.
 */
void bind_pcm(py::module& module);

/**
 * Create Python bindings for the graph utility functions.
 *
 * @param module The pybind11 module to add the bindings to.
 */
void bind_graph(py::module& module);

}  // namespace bindings
}  // namespace weave

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace weave {
namespace bindings {

/**
 * Create Python bindings for the module containing codes.
 *
 * @param module The pybind11 module to add the bindings to.
 */
void bind_codes(py::module& module);

/**
 * Create Python bindings for the NoiseModel class.
 *
 * @param module The pybind11 module to add the bindings to.
 */
void bind_noise_model(py::module& module);

}  // namespace bindings
}  // namespace weave

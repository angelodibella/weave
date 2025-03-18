#pragma once

#include <pybind11/pybind11.h>
#include "weave/noise_model.hpp"

namespace py = pybind11;

namespace weave {
namespace bindings {

inline void bind_noise_model(py::module& m) {
    py::class_<NoiseModel>(m, "NoiseModel")
        .def("set_error_rate", &NoiseModel::setErrorRate)
        .def("get_error_rate", &NoiseModel::getErrorRate);
}

} // namespace bindings
} // namespace weave
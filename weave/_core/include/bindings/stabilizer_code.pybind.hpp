#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "weave/codes/stabilizer_code.hpp"

namespace py = pybind11;

namespace weave {
namespace bindings {

inline void bind_stabilizer_code(py::module& m) {
    py::class_<StabilizerCode>(m, "StabilizerCode")
        .def(py::init<>())
        .def("initialize_from_stabilizers", &StabilizerCode::initializeFromStabilizers)
        .def("get_parameters", &StabilizerCode::getParameters)
        .def("get_stabilizers", &StabilizerCode::getStabilizers)
        .def("get_logical_operators", &StabilizerCode::getLogicalOperators);
}

} // namespace bindings
} // namespace weave
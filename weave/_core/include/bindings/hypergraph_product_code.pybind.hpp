#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "weave/hypergraph_product_code.hpp"

namespace py = pybind11;

namespace weave {
namespace bindings {

inline void bind_hypergraph_product_code(py::module& m) {
    py::class_<HypergraphProductCode>(m, "HypergraphProductCode")
        .def(py::init<>())
        .def(py::init<const std::vector<std::vector<int>>&>())
        .def("generate", &HypergraphProductCode::generate)
        .def("get_parameters", &HypergraphProductCode::getParameters)
        .def("get_stabilizers", &HypergraphProductCode::getStabilizers);
}

} // namespace bindings
} // namespace weave
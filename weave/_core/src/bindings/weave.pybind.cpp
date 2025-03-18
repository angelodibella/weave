#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bindings/noise_model.pybind.hpp"
#include "bindings/hypergraph_product_code.pybind.hpp"
#include "bindings/stabilizer_code.pybind.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "Weave: A quantum error correction framework.";

    // Register all the bindings
    weave::bindings::bind_noise_model(m);
    weave::bindings::bind_hypergraph_product_code(m);
    weave::bindings::bind_stabilizer_code(m);
}
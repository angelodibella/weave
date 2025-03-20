#include "bindings/util.pybind.hpp"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "weave/util/pcm.hpp"
#include "weave/util/graph.hpp"

namespace py = pybind11;

namespace weave {
namespace bindings {

void bind_pcm(py::module& util_module) {
    // Create a submodule for PCM utilities.
    auto pcm = util_module.def_submodule("pcm", "Parity-check matrix utility functions");

    // Bind the repetition code PCM generation function.
    pcm.def("repetition", &weave::util::repetition, py::arg("n"),
            "Construct the parity-check matrix for a repetition code.\n\n"
            "Args:\n"
            "    n: Length of the repetition code.\n"
            "Returns:\n"
            "    An (n-1) x n parity-check matrix.");
}

void bind_graph(py::module& util_module) {
    // Create a submodule for graph utilities.
    auto graph = util_module.def_submodule("graph", "Graph utility functions");

    // Bind the find_edge_crossings function.
    graph.def("find_edge_crossings", &weave::util::find_edge_crossings, py::arg("pos"), py::arg("edges"),
              "Find all edge crossings in a graph.\n\n"
              "Args:\n"
              "    pos: The positions of the nodes in the graph.\n"
              "    edges: The edges in the graph.\n"
              "Returns:\n"
              "    A set of sets of pairs of edges that cross each other.");

    // Bind the line_intersection function.
    graph.def("line_intersection", &weave::util::line_intersection, py::arg("a"), py::arg("b"), py::arg("c"), py::arg("d"),
              "Find the intersection of two lines.\n\n"
              "Args:\n"
              "    a: The first point of the first line.\n"
              "    b: The second point of the first line.\n"
              "    c: The first point of the second line.\n"
              "    d: The second point of the second line.\n"
              "Returns:\n"
              "    The intersection point if it exists.");
}

void bind_util(py::module& module) {
    // Create a submodule for utility functions.
    auto util = module.def_submodule("util", "Utility functions for quantum error correction");

    // Bind PCM utilities.
    bind_pcm(util);
    
    // Bind graph utilities.
    bind_graph(util);
}

}  // namespace bindings
}  // namespace weave

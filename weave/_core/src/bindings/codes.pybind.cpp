#include "bindings/codes.pybind.hpp"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "weave/codes/noise_model.hpp"

namespace py = pybind11;

namespace weave {
namespace bindings {

void bind_noise_model(py::module& codes_module) {
    py::class_<NoiseModel>(codes_module, "NoiseModel")
        .def(py::init<double, double, double, double, double>(), py::arg("data") = 0.0,
             py::arg("z_check") = 0.0, py::arg("x_check") = 0.0, py::arg("circuit") = 0.0,
             py::arg("crossing") = 0.0,
             "Creates a NoiseModel with specified noise levels.\n\n"
             "Args:\n"
             "    data: Noise level for data qubits.\n"
             "    z_check: Noise level for Z-check qubits.\n"
             "    x_check: Noise level for X-check qubits.\n"
             "    circuit: Noise level for two-qubit circuit operations.\n"
             "    crossing: Noise level for crossing edges (cross-talk).")
        .def("set_data_noise", py::overload_cast<double>(&NoiseModel::set_data_noise),
             py::arg("value"), "Sets uniform noise for data qubits.")
        .def("set_data_noise",
             py::overload_cast<const std::vector<double>&>(&NoiseModel::set_data_noise),
             py::arg("params"), "Sets specific noise parameters for data qubits.")
        .def("set_z_check_noise", py::overload_cast<double>(&NoiseModel::set_z_check_noise),
             py::arg("value"), "Sets uniform noise for Z-check qubits.")
        .def("set_z_check_noise",
             py::overload_cast<const std::vector<double>&>(&NoiseModel::set_z_check_noise),
             py::arg("params"), "Sets specific noise parameters for Z-check qubits.")
        .def("set_x_check_noise", py::overload_cast<double>(&NoiseModel::set_x_check_noise),
             py::arg("value"), "Sets uniform noise for X-check qubits.")
        .def("set_x_check_noise",
             py::overload_cast<const std::vector<double>&>(&NoiseModel::set_x_check_noise),
             py::arg("params"), "Sets specific noise parameters for X-check qubits.")
        .def("set_circuit_noise", py::overload_cast<double>(&NoiseModel::set_circuit_noise),
             py::arg("value"), "Sets uniform noise for circuit operations.")
        .def("set_circuit_noise",
             py::overload_cast<const std::vector<double>&>(&NoiseModel::set_circuit_noise),
             py::arg("params"), "Sets specific noise parameters for circuit operations.")
        .def("set_crossing_noise", py::overload_cast<double>(&NoiseModel::set_crossing_noise),
             py::arg("value"), "Sets uniform noise for crossing edges.")
        .def("set_crossing_noise",
             py::overload_cast<const std::vector<double>&>(&NoiseModel::set_crossing_noise),
             py::arg("params"), "Sets specific noise parameters for crossing edges.");
}

void bind_codes(py::module& module) {
    // Create a submodule codes.
    auto codes_module = module.def_submodule("codes", "Quantum error-correcting codes");

    // Bind the noise model.
    bind_noise_model(codes_module);

    // TODO: Other code bindings to be added here.
}

}  // namespace bindings
}  // namespace weave

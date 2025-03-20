#include "weave/codes/noise_model.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace weave {

// TODO: Refactor. This is temporary.

NoiseModel::NoiseModel(double data, double z_check, double x_check, double circuit, double crossing) {
    set_data_noise(data);
    set_z_check_noise(z_check);
    set_x_check_noise(x_check);
    set_circuit_noise(circuit);
    set_crossing_noise(crossing);
}

void NoiseModel::set_data_noise(double value) {
    m_data = process_noise(value, 3, 3.0);
}

void NoiseModel::set_data_noise(const std::vector<double>& params) {
    m_data = process_noise(params, 3);
}

void NoiseModel::set_z_check_noise(double value) {
    m_z_check = process_noise(value, 3, 3.0);
}

void NoiseModel::set_z_check_noise(const std::vector<double>& params) {
    m_z_check = process_noise(params, 3);
}

void NoiseModel::set_x_check_noise(double value) {
    m_x_check = process_noise(value, 3, 3.0);
}

void NoiseModel::set_x_check_noise(const std::vector<double>& params) {
    m_x_check = process_noise(params, 3);
}

void NoiseModel::set_circuit_noise(double value) {
    m_circuit = process_noise(value, 15, 15.0);
}

void NoiseModel::set_circuit_noise(const std::vector<double>& params) {
    m_circuit = process_noise(params, 15);
}

void NoiseModel::set_crossing_noise(double value) {
    m_crossing = process_noise(value, 15, 15.0);
}

void NoiseModel::set_crossing_noise(const std::vector<double>& params) {
    m_crossing = process_noise(params, 15);
}

std::vector<double> NoiseModel::process_noise(double value, size_t expected, double divisor) const {
    std::vector<double> result(expected, value / divisor);
    return result;
}

std::vector<double> NoiseModel::process_noise(const std::vector<double>& params, size_t expected) const {
    if (params.size() != expected) {
        throw std::invalid_argument("Noise parameter vector has incorrect size. Expected " + 
                                   std::to_string(expected) + ", got " + 
                                   std::to_string(params.size()));
    }
    return params;
}

}  // namespace weave
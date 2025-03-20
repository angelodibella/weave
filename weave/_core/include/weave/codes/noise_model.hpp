#pragma once

#include <string>
#include <vector>

namespace weave {

/**
 * Noise model for quantum error-correcting codes.
 *
 * @param data Noise level(s) for data qubits. If a single float is provided, it is uniformly divided among 3 error
 * types.
 * @param z_check Noise level(s) for Z-check qubits. If a single float is provided, it is uniformly divided among 3
 * error types.
 * @param x_check Noise level(s) for X-check qubits. If a single float is provided, it is uniformly divided among 3
 * error types.
 * @param circuit Noise level(s) for two-qubit circuit operations. Expected to be 15 values. If a single float is
 * provided, it is uniformly divided among 15 values.
 * @param crossing Noise level(s) for crossing edges (cross-talk) in the Tanner graph. Expected to be 15 values. If a
 * single float is provided, it is uniformly divided among 15 values.
 *
 * @throws std::invalid_argument If any noise parameter provided as a list does not have the expected length.
 */
class NoiseModel {
public:
    NoiseModel(double data = 0.0, double z_check = 0.0, double x_check = 0.0, double circuit = 0.0,
               double crossing = 0.0);
    // TODO: Add constructor that takes std::vector<double> for each parameter.

    void set_data_noise(double value);
    void set_data_noise(const std::vector<double>& params);

    void set_z_check_noise(double value);
    void set_z_check_noise(const std::vector<double>& params);

    void set_x_check_noise(double value);
    void set_x_check_noise(const std::vector<double>& params);

    void set_circuit_noise(double value);
    void set_circuit_noise(const std::vector<double>& params);

    void set_crossing_noise(double value);
    void set_crossing_noise(const std::vector<double>& params);

private:
    std::vector<double> process_noise(double value, size_t expected, double divisor) const;
    std::vector<double> process_noise(const std::vector<double>& params, size_t expected) const;

    std::vector<double> m_data;
    std::vector<double> m_z_check;
    std::vector<double> m_x_check;
    std::vector<double> m_circuit;
    std::vector<double> m_crossing;
};

}  // namespace weave
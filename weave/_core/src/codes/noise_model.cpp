#include "weave/noise_model.hpp"

namespace weave {

void NoiseModel::setErrorRate(double rate) {
    m_errorRate = rate;
}

double NoiseModel::getErrorRate() const {
    return m_errorRate;
}

} // namespace weave
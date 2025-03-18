#pragma once

namespace weave {

class NoiseModel {
public:
    NoiseModel() = default;
    virtual ~NoiseModel() = default;
    
    // Virtual method to apply noise based on model parameters
    virtual void applyNoise() = 0;
    
    // Example method to set error rate
    void setErrorRate(double rate);
    
    // Example method to get error rate
    double getErrorRate() const;
    
private:
    double m_errorRate = 0.0;
};

} // namespace weave
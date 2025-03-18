#pragma once

#include <vector>
#include <string>

namespace weave {

class HypergraphProductCode {
public:
    HypergraphProductCode() = default;
    explicit HypergraphProductCode(const std::vector<std::vector<int>>& parityCheckMatrix);
    
    // Generate the code from parity check matrices
    void generate(const std::vector<std::vector<int>>& pcMatrixX, 
                 const std::vector<std::vector<int>>& pcMatrixZ);
    
    // Get code parameters [n, k, d]
    std::vector<int> getParameters() const;
    
    // Get the stabilizers
    std::vector<std::string> getStabilizers() const;
    
private:
    std::vector<std::vector<int>> m_parityCheckMatrixX;
    std::vector<std::vector<int>> m_parityCheckMatrixZ;
    int m_numQubits = 0;
    int m_numLogicalQubits = 0;
    int m_distance = 0;
};

} // namespace weave
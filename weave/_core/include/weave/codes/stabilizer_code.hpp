#pragma once

#include <vector>
#include <string>

namespace weave {

/**
 * StabilizerCode class
 * 
 * Base class for stabilizer quantum error correction codes
 */
class StabilizerCode {
public:
    StabilizerCode() = default;
    virtual ~StabilizerCode() = default;
    
    // Initialize from stabilizer generators
    virtual void initializeFromStabilizers(const std::vector<std::string>& stabilizers);
    
    // Get code parameters [n, k, d]
    virtual std::vector<int> getParameters() const;
    
    // Get the stabilizer generators
    virtual std::vector<std::string> getStabilizers() const;
    
    // Get logical operators
    virtual std::vector<std::string> getLogicalOperators() const;
    
protected:
    std::vector<std::string> m_stabilizers;
    std::vector<std::string> m_logicalOperators;
    int m_numQubits = 0;
    int m_numLogicalQubits = 0;
    int m_distance = 0;
};

} // namespace weave
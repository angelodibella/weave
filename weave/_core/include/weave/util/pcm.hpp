#pragma once

#include <vector>
#include <string>

namespace weave {
namespace util {

/**
 * Parity Check Matrix (PCM) utility class
 * Provides operations for working with parity check matrices
 */
class PCM {
public:
    PCM() = default;
    explicit PCM(const std::vector<std::vector<int>>& matrix);
    
    // Basic operations
    void setMatrix(const std::vector<std::vector<int>>& matrix);
    std::vector<std::vector<int>> getMatrix() const;
    
    // Dimensions
    size_t rows() const;
    size_t cols() const;
    
    // Matrix operations
    PCM transpose() const;
    
    // Generate Tanner graph representation
    // (In real implementation, this would return a Graph object)
    void generateTannerGraph() const;
    
private:
    std::vector<std::vector<int>> m_matrix;
};

} // namespace util
} // namespace weave
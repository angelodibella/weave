#pragma once

#include <Eigen/Dense>

namespace weave {
namespace util {

/**
 * Construct the parity-check matrix for a repetition code.
 *
 * @param n Length of the repetition code.
 * @return An (n-1) x n parity-check matrix.
 */
Eigen::MatrixXi repetition(int n);

}  // namespace util
}  // namespace weave
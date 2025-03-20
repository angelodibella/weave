#include "weave/util/pcm.hpp"

#include <Eigen/Dense>

namespace weave {
namespace util {

Eigen::MatrixXi repetition(int n) {
    Eigen::MatrixXi H = Eigen::MatrixXi::Zero(n - 1, n);
    for (int i = 0; i < n - 1; i++) H(i, i) = 1;
    H.col(n - 1).setOnes();
    return H;
}

}  // namespace util
}  // namespace weave
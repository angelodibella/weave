#include "weave/util/graph.hpp"

#include <optional>
#include <set>
#include <vector>

namespace weave {
namespace util {

std::set<std::set<std::pair<int, int>>> find_edge_crossings(const std::vector<std::pair<float, float>>& pos,
                                                            const std::vector<std::pair<int, int>>& edges) {
    std::set<std::set<std::pair<int, int>>> crossings;
    for (size_t i = 0; i < edges.size(); i++) {
        for (size_t j = i + 1; j < edges.size(); j++) {
            const auto& e1 = edges[i];
            const auto& e2 = edges[j];
            if (e1.first == e2.first || e1.first == e2.second || e1.second == e2.first || e1.second == e2.second) {
                continue;
            }

            const auto& pos1 = std::make_pair(pos[e1.first], pos[e1.second]);
            const auto& pos2 = std::make_pair(pos[e2.first], pos[e2.second]);

            const auto ccw = [](const auto& A, const auto& B, const auto& C) {
                return (C.second - A.second) * (B.first - A.first) > (B.second - A.second) * (C.first - A.first);
            };
            const auto intersect = [&ccw](const auto& A, const auto& B, const auto& C, const auto& D) {
                return ccw(A, C, D) != ccw(B, C, D) && ccw(A, B, C) != ccw(A, B, D);
            };

            if (intersect(pos1.first, pos1.second, pos2.first, pos2.second)) crossings.insert({e1, e2});
        }
    }
    return crossings;
}

std::optional<std::pair<float, float>> line_intersection(const std::pair<float, float>& a,
                                                         const std::pair<float, float>& b,
                                                         const std::pair<float, float>& c,
                                                         const std::pair<float, float>& d) {
    const auto x1 = a.first;
    const auto y1 = a.second;
    const auto x2 = b.first;
    const auto y2 = b.second;
    const auto x3 = c.first;
    const auto y3 = c.second;
    const auto x4 = d.first;
    const auto y4 = d.second;

    const auto denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    if (denom == 0) return std::nullopt;

    const auto x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom;
    const auto y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom;
    return std::make_pair(x, y);
}

}  // namespace util
}  // namespace weave

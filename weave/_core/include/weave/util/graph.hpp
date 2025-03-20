#pragma once

#include <optional>
#include <set>
#include <vector>

namespace weave {
namespace util {

/**
 * Find all edge crossings in a graph.
 *
 * @param pos The positions of the nodes in the graph.
 * @param edges The edges in the graph.
 * @return A set of sets of pairs of edges that cross each other.
 */
std::set<std::set<std::pair<int, int>>> find_edge_crossings(const std::vector<std::pair<float, float>>& pos,
                                                            const std::vector<std::pair<int, int>>& edges);

/**
 * Find the intersection of two lines.
 *
 * @param a The first point of the first line.
 * @param b The second point of the first line.
 * @param c The first point of the second line.
 * @param d The second point of the second line.
 * @return The intersection point if it exists.
 */
std::optional<std::pair<float, float>> line_intersection(const std::pair<float, float>& a,
                                                         const std::pair<float, float>& b,
                                                         const std::pair<float, float>& c,
                                                         const std::pair<float, float>& d);

}  // namespace util
}  // namespace weave

#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <memory>

namespace weave {
namespace util {

class Node;
class Edge;

/**
 * Graph utility class
 * Represents a graph structure for quantum code representations
 */
class Graph {
public:
    Graph() = default;
    
    // Node operations
    void addNode(int id, const std::string& type = "data");
    void removeNode(int id);
    std::shared_ptr<Node> getNode(int id) const;
    std::vector<std::shared_ptr<Node>> getNodes() const;
    
    // Edge operations
    void addEdge(int sourceId, int targetId, const std::string& type = "");
    void removeEdge(int sourceId, int targetId);
    std::vector<std::shared_ptr<Edge>> getEdges() const;
    
    // Graph operations
    void clear();
    size_t nodeCount() const;
    size_t edgeCount() const;
    
    // Serialization
    std::string toJson() const;
    static Graph fromJson(const std::string& json);
    
private:
    std::unordered_map<int, std::shared_ptr<Node>> m_nodes;
    std::vector<std::shared_ptr<Edge>> m_edges;
};

} // namespace util
} // namespace weave
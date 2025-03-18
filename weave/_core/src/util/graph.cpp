#include "weave/util/graph.hpp"

// Node and Edge classes can be defined in the implementation file
// since they're not part of the public interface

namespace weave {
namespace util {

class Node {
public:
    Node(int id, const std::string& type) : m_id(id), m_type(type) {}
    
    int getId() const { return m_id; }
    std::string getType() const { return m_type; }
    void setType(const std::string& type) { m_type = type; }
    
private:
    int m_id;
    std::string m_type;
};

class Edge {
public:
    Edge(int sourceId, int targetId, const std::string& type = "")
        : m_sourceId(sourceId), m_targetId(targetId), m_type(type) {}
    
    int getSourceId() const { return m_sourceId; }
    int getTargetId() const { return m_targetId; }
    std::string getType() const { return m_type; }
    
private:
    int m_sourceId;
    int m_targetId;
    std::string m_type;
};

// Graph implementation
void Graph::addNode(int id, const std::string& type) {
    m_nodes[id] = std::make_shared<Node>(id, type);
}

void Graph::removeNode(int id) {
    m_nodes.erase(id);
    
    // Also remove any edges connected to this node
    auto it = m_edges.begin();
    while (it != m_edges.end()) {
        if ((*it)->getSourceId() == id || (*it)->getTargetId() == id) {
            it = m_edges.erase(it);
        } else {
            ++it;
        }
    }
}

std::shared_ptr<Node> Graph::getNode(int id) const {
    auto it = m_nodes.find(id);
    if (it != m_nodes.end()) {
        return it->second;
    }
    return nullptr;
}

std::vector<std::shared_ptr<Node>> Graph::getNodes() const {
    std::vector<std::shared_ptr<Node>> result;
    for (const auto& pair : m_nodes) {
        result.push_back(pair.second);
    }
    return result;
}

void Graph::addEdge(int sourceId, int targetId, const std::string& type) {
    // Ensure both nodes exist
    if (m_nodes.find(sourceId) == m_nodes.end() || 
        m_nodes.find(targetId) == m_nodes.end()) {
        return;
    }
    
    m_edges.push_back(std::make_shared<Edge>(sourceId, targetId, type));
}

void Graph::removeEdge(int sourceId, int targetId) {
    auto it = m_edges.begin();
    while (it != m_edges.end()) {
        if ((*it)->getSourceId() == sourceId && (*it)->getTargetId() == targetId) {
            it = m_edges.erase(it);
        } else {
            ++it;
        }
    }
}

std::vector<std::shared_ptr<Edge>> Graph::getEdges() const {
    return m_edges;
}

void Graph::clear() {
    m_nodes.clear();
    m_edges.clear();
}

size_t Graph::nodeCount() const {
    return m_nodes.size();
}

size_t Graph::edgeCount() const {
    return m_edges.size();
}

std::string Graph::toJson() const {
    // Placeholder for JSON serialization
    return "{}";
}

Graph Graph::fromJson(const std::string& json) {
    // Placeholder for JSON deserialization
    return Graph();
}

} // namespace util
} // namespace weave
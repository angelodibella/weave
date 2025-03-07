import networkx as nx
import numpy as np
from matplotlib import pyplot as plt


def compute_layout(graph: nx.Graph, pos_spec: str | list[tuple[float, float]], index_key: str = "index") -> list[
    tuple[float, float]]:
    """
    Compute node positions for a graph based on a layout specification.

    If pos_spec is a string, it selects a networkx layout; if a list, it is assumed
    to be the positions in order. The resulting list is ordered by each node's attribute
    given by index_key.

    Parameters
    ----------
    graph : nx.Graph
        The graph for which to compute positions.
    pos_spec : str or list of tuple of float
        Layout specification. Supported strings are "random", "spring", "bipartite", and "tripartite".
        If a list, positions are assumed to be provided.
    index_key : str, optional
        The node attribute key that holds the original index (default is "index").

    Returns
    -------
    list of tuple of float
        A list of positions ordered by the node's index.
    """
    if isinstance(pos_spec, str):
        if pos_spec in ("random", None):
            pos_dict = nx.random_layout(graph)
        elif pos_spec == "spring":
            pos_dict = nx.spring_layout(graph, iterations=1000)
        elif pos_spec == "bipartite":
            # For bipartite layout, assume nodes have a "type" attribute; otherwise, use all nodes.
            nodes = list(graph.nodes)
            pos_dict = nx.bipartite_layout(graph, nodes)
        elif pos_spec == "tripartite":
            pos_dict = nx.multipartite_layout(graph, subset_key="layer")
        else:
            raise ValueError(f"Layout '{pos_spec}' not recognized.")
    elif isinstance(pos_spec, list):
        # Assume pos_spec is already a list of positions.
        return pos_spec
    else:
        raise ValueError("pos_spec must be either a string or a list of positions.")

    # Build a list of positions ordered by node attribute given by index_key.
    num_nodes = graph.number_of_nodes()
    pos_list = [None] * num_nodes
    for node, pos in pos_dict.items():
        idx = graph.nodes[node].get(index_key, None)
        if idx is None:
            raise ValueError(f"Node {node} missing attribute '{index_key}'.")
        pos_list[idx] = pos
    return pos_list


def find_edge_crossings(pos: list[tuple[float, float]], edges: list[tuple[int, int]]) -> set[
    frozenset[tuple[int, int]]]:
    """
    Identify intersections among a set of edges based on node positions.

    Parameters
    ----------
    pos : list of tuple of float
        List of positions, where each position corresponds to a node index.
    edges : list of tuple of int
        List of edges, each represented as a tuple of two node indices.

    Returns
    -------
    set of frozenset
        A set of frozensets, each containing two edges (as tuples) that cross.
    """

    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def intersect(A, B, C, D):
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    crossings = set()
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            e1, e2 = edges[i], edges[j]
            if e1[0] in e2 or e1[1] in e2:
                continue
            pos1 = (pos[e1[0]], pos[e1[1]])
            pos2 = (pos[e2[0]], pos[e2[1]])
            if intersect(pos1[0], pos1[1], pos2[0], pos2[1]):
                crossings.add(frozenset({e1, e2}))
    return crossings


def line_intersection(a: float, b: float, c: float, d: float) -> None | tuple[float]:
    """
    Compute the intersection point of lines ab and cd.
    Each parameter is a tuple (x, y) in world coordinates.
    Returns (x, y) if lines intersect, else None.
    """
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    x4, y4 = d
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None
    x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return (x, y)


def draw(
        graph: nx.Graph,
        pos: list[tuple[float, float]],
        with_labels: bool = False,
        crossings: bool = True,
        connection_rad: float = 0.0,
        **kwargs,
) -> None:
    """
    Draw a Tanner graph with default styling.

    Parameters
    ----------
    graph : nx.Graph
        The graph to be drawn.
    pos : list of tuple of float
        List of node positions ordered by node index.
    with_labels : bool, optional
        Whether to display node labels (default is False).
    crossings : bool, optional
        Whether to highlight edge crossings (default is True).
    connection_rad : float, optional
        Curvature radius for edges (default is 0.0).
    **kwargs
        Additional keyword arguments for node drawing.
    """
    # Default styles based on node type.
    colors = {"q": "#D3D3D3", "X": "#FFC0CB", "Z": "#ADD8E6"}
    shapes = {"q": "o", "X": "s", "Z": "s"}
    sizes = {"q": 300, "X": 230, "Z": 230}

    # Build a dictionary mapping node to its position.
    layout = {node: pos[graph.nodes[node]["index"]] for node in graph.nodes()}

    # Draw nodes by type.
    for ntype, shape in shapes.items():
        nodes = [node for node, attr in graph.nodes.items() if attr.get("type") == ntype]
        nx.draw_networkx_nodes(
            graph,
            layout,
            nodelist=nodes,
            node_color=[colors.get(ntype, "#FFFFFF") for _ in nodes],
            node_shape=shape,
            node_size=[sizes.get(ntype, 300) for _ in nodes],
            **kwargs,
        )

    nx.draw_networkx_edges(
        graph, layout, width=0.7, arrows=True, connectionstyle=f"arc3,rad={connection_rad}"
    )
    if with_labels:
        labels = {node: node for node in graph.nodes()}
        nx.draw_networkx_labels(graph, layout, labels)

    if crossings:
        # Compute edges using the "index" attribute.
        edges = [tuple(graph.nodes[node]["index"] for node in edge) for edge in graph.edges()]
        cross_set = find_edge_crossings(pos, edges)
        for crossing in cross_set:
            e1, e2 = list(crossing)
            pt = line_intersection(pos[e1[0]], pos[e1[1]], pos[e2[0]], pos[e2[1]])
            if pt is not None:
                plt.scatter(pt[0], pt[1], color="black", s=15, marker="D")

    plt.axis("off")

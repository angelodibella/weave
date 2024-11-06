import numpy as np
import stim
import networkx as nx
import matplotlib.pyplot as plt


class NoiseModel:
    """Class encapsulating noise models for quantum error-correcting codes.
    
    Attributes
    ----------
    data : float or list of float
        Noise level(s) for data qubits. This is the list of probabilities of applying, respectively, 'X', 'Y' and 'Z'
        Pauli gates to data qubits. If a single number is passed, then this value represents the probability of applying
        any of these errors uniformly. Defaults to zero.
    z_check : float or list of float
        Noise level(s) for Z-check qubits. Similar to `data`. Defaults to zero.
    x_check : float or list of float
        Noise level(s) for X-check qubits. Similar to `data`. Defaults to zero.
    circuit : float or list of float
        Noise level(s) for the circuit operations. Respectively, the probability of assigning, to two-qubit gates,
        combination pairs of Pauli I, X, Y and Z operators in the following order:

        [IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ]

        except the operator 'II', which is determined by subtracting the sum of this list of probabilities from 1. If a
        single number is passed, then this value represents the probability of applying any of these non-trivial errors
        uniformly. Defaults to zero.
    crossing : float or list of float
        Noise level(s) for crossing edges of the Tanner graph of the code. Behaves exactly the same as `circuit`, where
        the two-qubit errors are applied on qubits that cross-talk. Defaults to zero.

    Raises
    ------
    AssertionError
        If the provided noise parameter lists do not have the expected lengths.
    """

    def __init__(
            self,
            data: float | list[float] = 0.0,
            z_check: float | list[float] = 0.0,
            x_check: float | list[float] = 0.0,
            circuit: float | list[float] = 0.0,
            crossing: float | list[float] = 0.0,
    ) -> None:
        if np.issubdtype(type(circuit), np.number):
            self.circuit = [circuit / 15 for _ in range(15)]
        else:
            assert (
                    len(circuit) == 15
            ), f"Stabilizer measurement noise takes 15 parameters, given {len(circuit)}."
            self.circuit = circuit

        if np.issubdtype(type(data), np.number):
            self.data = [data / 3 for _ in range(3)]
        else:
            assert (
                    len(data) == 3
            ), f"Data qubit noise takes 3 parameters, given {len(data)}."
            self.data = data

        if np.issubdtype(type(crossing), np.number):
            self.crossing = [crossing / 15 for _ in range(15)]
        else:
            assert (
                    len(crossing) == 15
            ), f"Crossing noise takes 15 parameters, given {len(crossing)}."
            self.circuit = crossing

        if np.issubdtype(type(z_check), np.number):
            self.z_check = [z_check / 3 for _ in range(3)]
        else:
            assert (
                    len(z_check) == 3
            ), f"Z check qubit noise takes 3 parameters, given {len(z_check)}."
            self.z_check = z_check

        if np.issubdtype(type(x_check), np.number):
            self.x_check = [x_check / 3 for _ in range(3)]
        else:
            assert (
                    len(x_check) == 3
            ), f"Data qubit noise takes 3 parameters, given {len(z_check)}."
            self.x_check = x_check


class StabilizerCode:
    """Base class for quantum error-correcting stabilizer codes.

    Attributes
    ----------
    circuit : stim.Circuit
        Quantum circuit to be prepended to the circuit constructed by this class.
    rounds : int
        Number of rounds of stabilizer measurements.
    pos : str or list of tuple of int, optional
        Specifies the embedding of the Tanner graph of the code in a plane. Specified as a string for a built-in
        algorithmic embedding, or a list of `(x, y)` tuples following the order of qubits and stabilizers as specified
        in `qubits`. Defaults to a random embedding.
    experiment : str
        Type of experiment to perform, options are `"z_memory"` or `"x_memory"`. Defaults to `"z_memory"`.
    logical : int or list of int, optional
        Index or indices of the logicals to use out of the list of all logicals for the code. Defaults to using all
        logicals.
    graph : networkx.Graph or None
        The Tanner graph of the code.
    crossings : set of frozenset of tuple or None
        The crossing set of the embedding of the Tanner graph of the code. Note: using this assumes straight-line
        topological graph isomorphism conjecture.
    """

    def __init__(
            self,
            circuit: stim.Circuit = None,
            rounds: int = 3,
            pos: str | list[tuple[int, int]] = None,
            noise: NoiseModel = NoiseModel(),
            experiment: str = "z_memory",
            logical: int | list[int] = None,
    ) -> None:
        self.circuit = stim.Circuit() if circuit is None else circuit
        self.rounds = rounds

        self.pos = pos
        self.noise = noise

        self.experiment = experiment

        self.qubits = []
        self.data_qubits = []
        self.z_check_qubits = []
        self.x_check_qubits = []
        self.logical = logical

        self.graph = None
        self.crossings = None

    def pos_from_str(self):
        z_check_nodes = [
            node for node, attributes in self.graph.nodes.items() if attributes["type"] == "Z"
        ]
        x_check_nodes = [
            node for node, attributes in self.graph.nodes.items() if attributes["type"] == "X"
        ]
        match self.pos:
            case "random" | None:
                self.pos = nx.random_layout(self.graph)
            case "spring":
                self.pos = nx.spring_layout(self.graph, iterations=1000)
            case "bipartite":
                self.pos = nx.bipartite_layout(self.graph, z_check_nodes + x_check_nodes)
            case "tripartite":
                self.pos = nx.multipartite_layout(self.graph, subset_key="layer")
            case _:
                raise ValueError(f"Qubit layout not recognized: '{self.pos}'.")

        pos_list = [(0, 0)] * len(self.qubits)
        for node, pos in self.pos.items():
            pos_list[self.graph.nodes[node]["index"]] = pos

        self.pos = pos_list

    def reset_data_qubits(self) -> None:
        self.circuit.append("R", self.data_qubits)

    def find_logicals(self):
        pass

    def find_crossings(self) -> set[frozenset[tuple]]:
        # TODO: Put these methods into a math module in util.
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

        crossings = set()
        qubit_edges = [
            tuple(self.graph.nodes[node]["index"] for node in edge) for edge in self.graph.edges
        ]
        for i in range(len(qubit_edges)):
            for j in range(i + 1, len(qubit_edges)):
                qubit_edge_1, qubit_edge_2 = qubit_edges[i], qubit_edges[j]
                if qubit_edge_1[0] in qubit_edge_2 or qubit_edge_1[1] in qubit_edge_2:
                    continue

                pos1 = (self.pos[qubit_edge_1[0]], self.pos[qubit_edge_1[1]])
                pos2 = (self.pos[qubit_edge_2[0]], self.pos[qubit_edge_2[1]])
                if intersect(pos1[0], pos1[1], pos2[0], pos2[1]):
                    crossings.add(frozenset({qubit_edge_1, qubit_edge_2}))
        return crossings

    def crossing_number(self) -> int:
        return len(self.crossings)

    def construct_code(self):
        pass

    def construct_graph(self):
        pass

    def draw(
            self,
            with_labels: bool = False,
            crossings: bool = True,
            connection_rad: float = 0.0,
            **kwargs,
    ) -> None:
        """Colors nodes based on their type ('q', 'x', 'z') and plots the graph."""

        colors = {"q": "#D3D3D3", "X": "#FFC0CB", "Z": "#ADD8E6"}
        shapes = {"q": "o", "X": "s", "Z": "s"}
        sizes = {"q": 300, "X": 230, "Z": 230}

        layout = {}
        for node, attributes in self.graph.nodes.items():
            layout[node] = self.pos[attributes["index"]]

        # TODO: Highlight logicals.
        for node_type, shape in shapes.items():
            filtered_nodes = [
                node
                for node, attributes in self.graph.nodes.items()
                if attributes["type"] == node_type
            ]
            nx.draw_networkx_nodes(
                self.graph,
                layout,
                nodelist=filtered_nodes,
                node_color=[colors[node_type] for _ in filtered_nodes],
                node_shape=shape,
                node_size=[sizes[node_type] for _ in filtered_nodes],
                **kwargs,
            )

        nx.draw_networkx_edges(
            self.graph, layout, width=0.7, arrows=True, connectionstyle=f"arc3,rad={connection_rad}"
        )
        if with_labels:
            labels = {node: node for node in self.graph.nodes()}
            nx.draw_networkx_labels(self.graph, layout, labels)

        if crossings:
            for crossing in self.crossings:
                qubit_edge_1, qubit_edge_2 = list(crossing)
                line1 = np.array([self.pos[qubit_edge_1[0]], self.pos[qubit_edge_1[1]]])
                line2 = np.array([self.pos[qubit_edge_2[0]], self.pos[qubit_edge_2[1]]])

                # Line 1 equation: A1x + B1y = C1
                A1 = line1[1, 1] - line1[0, 1]
                B1 = line1[0, 0] - line1[1, 0]
                C1 = A1 * line1[0, 0] + B1 * line1[0, 1]

                # Line 2 equation: A2x + B2y = C2
                A2 = line2[1, 1] - line2[0, 1]
                B2 = line2[0, 0] - line2[1, 0]
                C2 = A2 * line2[0, 0] + B2 * line2[0, 1]

                determinant = A1 * B2 - A2 * B1

                if determinant != 0:
                    x = (C1 * B2 - C2 * B1) / determinant
                    y = (A1 * C2 - A2 * C1) / determinant
                    plt.scatter(x, y, color="black", s=15, marker="D")

        plt.axis("off")


# TODO: Implement a classical code class, and make code classes acquire this instead of clists.
class ClassicalCode:
    def __init__(self):
        pass

# TODO: Perhaps add a CSS code class between the `StabilizerCode` and `HypergraphProductCode` classes.

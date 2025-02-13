import networkx as nx
import numpy as np
import stim
from matplotlib import pyplot as plt
from ldpc import mod2

# from .css_code import CSSCode
from .base import NoiseModel
from ..util import pcm


class HypergraphProductCode:
    def __init__(
            self,
            clist1: list,
            clist2: list,
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

        # -------------------------------------

        self.H1: np.ndarray = pcm.to_matrix(clist1)
        self.H2: np.ndarray = pcm.to_matrix(clist2)

        self.qubits = np.arange(sum(self.H1.shape) * sum(self.H2.shape))

        z_check_order = [
            "Q" if s == "B" else "Z"
            for s in clist2
            if not np.issubdtype(type(s), np.number)
        ]

        x_check_order = [
            "X" if s == "B" else "Q"
            for s in clist2
            if not np.issubdtype(type(s), np.number)
        ]

        check_order = np.array(
            [
                z_check_order if s == "B" else x_check_order
                for s in clist1
                if not np.issubdtype(type(s), np.number)
            ]
        ).flatten()

        self.data_qubits = [q for q, s in zip(self.qubits, check_order) if s == "Q"]
        self.z_check_qubits = [q for q, s in zip(self.qubits, check_order) if s == "Z"]
        self.x_check_qubits = [q for q, s in zip(self.qubits, check_order) if s == "X"]

        self.HX, self.HZ = pcm.hypergraph_product(self.H1, self.H2)
        self.graph = self.construct_graph()

        if self.pos is None or type(self.pos) == str:
            self.pos_from_str()

        self.crossings = self.find_crossings()

        self.construct_code()

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
        def compute_lz(HX, HZ):
            ker_HX = mod2.nullspace(HX)
            im_HZ_T = mod2.row_basis(HZ)

            logicals_stack = np.vstack([im_HZ_T, ker_HX])
            pivots = mod2.row_echelon(logicals_stack.T)[3]

            logical_operators_indices = [
                i for i in range(im_HZ_T.shape[0], logicals_stack.shape[0]) if i in pivots
            ]

            return logicals_stack[logical_operators_indices]

        x_logicals = compute_lz(self.HZ, self.HX)
        z_logicals = compute_lz(self.HX, self.HZ)

        return x_logicals, z_logicals

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

    def construct_graph(self) -> nx.Graph:
        """Assumes `HX` and `HZ` are reordered as generated by the `hypergraph_pcm` function."""

        num_data_qubits = self.HZ.shape[1]
        num_z_checks = self.HZ.shape[0]
        num_x_checks = self.HX.shape[0]

        adjacency_matrix = np.block(
            [
                [np.zeros((num_data_qubits, num_data_qubits)), self.HZ.T, self.HX.T],
                [
                    self.HZ,
                    np.zeros((num_z_checks, num_z_checks)),
                    np.zeros((num_z_checks, num_x_checks)),
                ],
                [
                    self.HX,
                    np.zeros((num_x_checks, num_z_checks)),
                    np.zeros((num_x_checks, num_x_checks)),
                ],
            ]
        ).astype(int)

        G = nx.from_numpy_array(adjacency_matrix)
        labels = {}
        for i in range(num_data_qubits):
            G.nodes[i]["type"] = "q"
            G.nodes[i]["index"] = self.data_qubits[i]
            labels[i] = f"$q_{{{i + 1}}}$"

            G.nodes[i]["layer"] = 1
        for i in range(num_data_qubits, num_data_qubits + num_z_checks):
            G.nodes[i]["type"] = "Z"
            G.nodes[i]["index"] = self.z_check_qubits[i - num_data_qubits]
            labels[i] = f"$Z_{{{i - num_data_qubits + 1}}}$"

            G.nodes[i]["layer"] = 0
        for i in range(
                num_data_qubits + num_z_checks, num_data_qubits + num_z_checks + num_x_checks
        ):
            G.nodes[i]["type"] = "X"
            G.nodes[i]["index"] = self.x_check_qubits[i - (num_data_qubits + num_z_checks)]
            labels[i] = f"$X_{{{i - (num_data_qubits + num_z_checks) + 1}}}$"

            G.nodes[i]["layer"] = 2

        G = nx.relabel_nodes(G, labels)

        return G

    def construct_code(self) -> None:
        self.circuit.append("R", self.qubits)
        self.circuit.append("M", self.z_check_qubits + self.x_check_qubits)

        circuit = stim.Circuit()
        for target_qubit, z_pcm_row in zip(self.z_check_qubits, self.HZ):
            qubits = [self.data_qubits[i] for i, v in enumerate(z_pcm_row) if v]
            for z_logical_qubit in qubits:
                circuit.append("CNOT", [z_logical_qubit, target_qubit])
                circuit.append(
                    "PAULI_CHANNEL_2", [z_logical_qubit, target_qubit], self.noise.circuit
                )

        for target_qubit, x_pcm_row in zip(self.x_check_qubits, self.HX):
            qubits = [self.data_qubits[i] for i, v in enumerate(x_pcm_row) if v]
            for x_logical_qubit in qubits:
                # TODO: Experiment with taking Hadamard gates out of this loop, and/or using CZ gates.
                circuit.append("H", [target_qubit])
                circuit.append("CNOT", [target_qubit, x_logical_qubit])
                circuit.append("H", [target_qubit])
                circuit.append(
                    "PAULI_CHANNEL_2", [x_logical_qubit, target_qubit], self.noise.circuit
                )

        for crossing in self.crossings:
            for edge in crossing:
                circuit.append("PAULI_CHANNEL_2", [edge[0], edge[1]], self.noise.crossing)

        circuit.append("PAULI_CHANNEL_1", self.data_qubits, self.noise.data)
        circuit.append("PAULI_CHANNEL_1", self.z_check_qubits, self.noise.z_check)
        circuit.append("PAULI_CHANNEL_1", self.x_check_qubits, self.noise.x_check)

        circuit.append("MR", self.z_check_qubits)
        if self.experiment == "z_memory":
            for k in range(len(self.z_check_qubits)):
                circuit.append(
                    "DETECTOR",
                    [
                        stim.target_rec(-1 - k),
                        stim.target_rec(-1 - k - len(self.x_check_qubits + self.z_check_qubits)),
                    ],
                )
            circuit.append("MR", self.x_check_qubits)
        elif self.experiment == "x_memory":
            circuit.append("MR", self.x_check_qubits)
            for k in range(len(self.x_check_qubits)):
                circuit.append(
                    "DETECTOR",
                    [
                        stim.target_rec(-1 - k),
                        stim.target_rec(-1 - k - len(self.x_check_qubits + self.z_check_qubits)),
                    ],
                )
        else:
            raise ValueError(f"Experiment not recognized: '{self.experiment}'.")

        self.circuit += circuit * self.rounds
        self.circuit.append("M", self.data_qubits)

        if self.experiment == "z_memory":
            for k in range(len(self.z_check_qubits)):
                z_pcm_row = self.HZ[-1 - k]
                idx_qubits = [i for i, v in enumerate(z_pcm_row) if v]
                lookback_records = [
                    stim.target_rec(-1 - k - len(self.data_qubits + self.x_check_qubits))
                ]
                for idx_qubit in idx_qubits:
                    lookback_records.append(stim.target_rec(idx_qubit - len(self.data_qubits)))
                self.circuit.append("DETECTOR", lookback_records)
        elif self.experiment == "x_memory":
            for k in range(len(self.x_check_qubits)):
                x_pcm_row = self.HX[-1 - k]
                idx_qubits = [i for i, v in enumerate(x_pcm_row) if v]
                lookback_records = [stim.target_rec(-1 - k - len(self.data_qubits))]
                for idx_qubit in idx_qubits:
                    lookback_records.append(stim.target_rec(idx_qubit - len(self.data_qubits)))
                self.circuit.append("DETECTOR", lookback_records)

        x_logicals, z_logicals = self.find_logicals()
        x_logicals_qubits = [np.where(logical == 1)[0] for logical in x_logicals]
        z_logicals_qubits = [np.where(logical == 1)[0] for logical in z_logicals]

        logicals_qubits = z_logicals_qubits if self.experiment == "z_memory" else x_logicals_qubits
        original_logicals_length = len(logicals_qubits)
        if self.logical is not None:
            if type(self.logical) == int:
                self.logical = [self.logical]
            logicals_qubits = [logicals_qubits[i] for i in self.logical]

        print(f"[{'Z' if self.experiment == 'z_memory' else 'X'} Logical Operators]")
        print(f"Using {len(logicals_qubits)}/{original_logicals_length} logical operators:")
        print(logicals_qubits, "\n")

        for i, logical_qubits in enumerate(logicals_qubits):
            observable_lookback_indices = []
            for logical_qubit in logical_qubits:
                observable_lookback_indices.append(
                    stim.target_rec(logical_qubit - len(self.data_qubits))
                )
            self.circuit.append("OBSERVABLE_INCLUDE", observable_lookback_indices, i)

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



import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import stim
from ldpc import mod2

import weave as wv


class Code:
    def __init__(
            self,
            code: str,
            circuit: stim.Circuit = None,
            rounds: int = 3,
            noise_circuit: float | list[float] = 0.0,
            noise_crossing: float | list[float] = 0.0,
            noise_data: float | list[float] = 0.0,
            noise_z_check: float | list[float] = 0.0,
            noise_x_check: float | list[float] = 0.0,
            logical: int | list[int] = None,
            **kwargs,
    ) -> None:
        self.circuit = stim.Circuit() if circuit is None else circuit
        self.rounds = rounds

        self.qubits = []
        self.data_qubits = []
        self.z_check_qubits = []
        self.x_check_qubits = []

        # Currently only supported for HP codes.
        self.graph = None

        if np.issubdtype(type(noise_circuit), np.number):
            self.noise_circuit = [noise_circuit / 15 for _ in range(15)]
        else:
            assert (
                    len(noise_circuit) == 15
            ), f"Stabilizer measurement noise takes 15 parameters, given {len(noise_circuit)}."
            self.noise_circuit = noise_circuit

        if np.issubdtype(type(noise_data), np.number):
            self.noise_data = [noise_data / 3 for _ in range(3)]
        else:
            assert (
                    len(noise_data) == 3
            ), f"Data qubit noise takes 3 parameters, given {len(noise_data)}."
            self.noise_data = noise_data

        if np.issubdtype(type(noise_crossing), np.number):
            self.noise_crossing = [noise_crossing / 15 for _ in range(15)]
        else:
            assert (
                    len(noise_crossing) == 15
            ), f"Crossing noise takes 15 parameters, given {len(noise_crossing)}."
            self.noise_circuit = noise_crossing

        if np.issubdtype(type(noise_z_check), np.number):
            self.noise_z_check = [noise_z_check / 3 for _ in range(3)]
        else:
            assert (
                    len(noise_z_check) == 3
            ), f"Z check qubit noise takes 3 parameters, given {len(noise_z_check)}."
            self.noise_z_check = noise_z_check

        if np.issubdtype(type(noise_x_check), np.number):
            self.noise_x_check = [noise_x_check / 3 for _ in range(3)]
        else:
            assert (
                    len(noise_x_check) == 3
            ), f"Data qubit noise takes 3 parameters, given {len(noise_z_check)}."
            self.noise_x_check = noise_x_check

        self.logical = logical

        self.code_params: dict = kwargs
        main_code, experiment = code.split(":") if ":" in code else (code, None)
        match main_code:
            case "repetition_code":
                self.distance = self.code_params["distance"]

                self.qubits = np.arange(2 * self.distance + 1)
                self.data_qubits = self.qubits[::2]
                self.z_check_qubits = self.qubits[1::2]
                self._repetition_code()
            case "surface_code":
                self.scale = self.code_params["scale"]
                assert (
                        self.scale[0] % 2 != 0 and self.scale[1] % 2 != 0
                ), "Scale of the surface code must be odd."

                self.qubits = np.arange(self.scale[0] * self.scale[1])
                self.data_qubits = []
                self.x_check_qubits = []
                self.z_check_qubits = []
                for row in range(self.scale[0]):
                    for col in range(self.scale[1]):
                        curr_qubit = row * self.scale[1] + col
                        self.circuit.append("QUBIT_COORDS", [curr_qubit], [row, col])
                        if row % 2 == 0:
                            if col % 2 == 0:
                                self.data_qubits.append(curr_qubit)
                            else:
                                self.z_check_qubits.append(curr_qubit)
                        elif row % 2 != 0:
                            if col % 2 != 0:
                                self.data_qubits.append(curr_qubit)
                            else:
                                self.x_check_qubits.append(curr_qubit)
                self._surface_code(experiment)
            case "hypergraph_product_code":
                self.pos: list[tuple[int, int]] = (
                    None if "pos" not in self.code_params.keys() else self.code_params["pos"]
                )

                clist1 = self.code_params["clist1"]
                clist2 = self.code_params["clist2"]

                self.H1: np.ndarray = wv.pcm.to_matrix(clist1)
                self.H2: np.ndarray = wv.pcm.to_matrix(clist2)

                num_qubits = sum(self.H1.shape) * sum(self.H2.shape)
                self.qubits = np.arange(num_qubits)

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

                self.HX, self.HZ = wv.pcm.hypergraph(self.H1, self.H2)
                self.graph = self.construct_graph()
                if self.pos is None or type(self.pos) == str:
                    self.pos_from_str()

                self._hypergraph_product_code(experiment)
            case _:
                raise ValueError(f"Code not recognized: '{main_code}'.")

        # TODO: Implement X and Z error propagation analysis.
        # TODO: Implement single-gate noise, like for Hadamard gates in X stabilizer syndrome extraction.

    # ------------------------------------ Setters and Getters ------------------------------------

    def reset_data_qubits(self) -> None:
        self.circuit.append("R", self.data_qubits)

    def crossing_number(self) -> int:
        return len(self.crossings)

    def logicals(self):
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

    # TODO: Create distance getter.

    # TODO: Create logicals getter.

    # -------------------------------------- Display Methods --------------------------------------

    def display_samples(self, shots: int = 1) -> None:
        samples = self.circuit.compile_sampler().sample(shots)
        for i, sample in enumerate(samples):
            round_list = []
            for j, outcome in enumerate(sample):
                round_list.append("o" if outcome else "_")

                # Line formatting.
                if ((j + 1) % len(self.z_check_qubits) == 0) and ((j + 1) != len(sample) - 1):
                    round_list[j] += "\n"
                if (j + 1) == (1 + self.rounds) * len(self.z_check_qubits):
                    round_list[j] += "\n"
            print(f"Shot {i + 1}:\n" + "".join(round_list))
        print()

    def display_detector_samples(self, shots: int = 1) -> None:
        samples = self.circuit.compile_detector_sampler().sample(shots, append_observables=True)
        for i, sample in enumerate(samples):
            round_list = []
            for j, outcome in enumerate(sample):
                round_list.append("x" if outcome else "_")

                # Line formatting.
                if (j + 1) % len(self.z_check_qubits) == 0:
                    round_list[j] += "\n"
                if (j + 1) == self.rounds * len(self.z_check_qubits):
                    round_list[j] += "\n"
            print(f"Shot {i + 1}:\n" + "".join(round_list))
        print()

    def print(self) -> None:
        print(self.circuit, "\n")

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
            crossing_sets = self.crossings()
            for crossing in crossing_sets:
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

    # --------------------------------------- Graph Methods ---------------------------------------

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

    def crossings(self) -> set[frozenset[tuple[int, int]]]:
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

    # ------------------------------------------- Codes -------------------------------------------

    def _hypergraph_product_code(self, experiment: str | None) -> None:
        self.circuit.append("R", self.qubits)
        self.circuit.append("M", self.z_check_qubits + self.x_check_qubits)

        circuit = stim.Circuit()
        for target_qubit, z_pcm_row in zip(self.z_check_qubits, self.HZ):
            qubits = [self.data_qubits[i] for i, v in enumerate(z_pcm_row) if v]
            for z_logical_qubit in qubits:
                circuit.append("CNOT", [z_logical_qubit, target_qubit])
                circuit.append(
                    "PAULI_CHANNEL_2", [z_logical_qubit, target_qubit], self.noise_circuit
                )

        for target_qubit, x_pcm_row in zip(self.x_check_qubits, self.HX):
            qubits = [self.data_qubits[i] for i, v in enumerate(x_pcm_row) if v]
            for x_logical_qubit in qubits:
                # TODO: Experiment with taking Hadamard gates out of this loop, and/or using CZ gates.
                circuit.append("H", [target_qubit])
                circuit.append("CNOT", [target_qubit, x_logical_qubit])
                circuit.append("H", [target_qubit])
                circuit.append(
                    "PAULI_CHANNEL_2", [x_logical_qubit, target_qubit], self.noise_circuit
                )

        crossings = self.crossings()
        for crossing in crossings:
            for edge in crossing:
                circuit.append("PAULI_CHANNEL_2", [edge[0], edge[1]], self.noise_crossing)

        circuit.append("PAULI_CHANNEL_1", self.data_qubits, self.noise_data)
        circuit.append("PAULI_CHANNEL_1", self.z_check_qubits, self.noise_z_check)
        circuit.append("PAULI_CHANNEL_1", self.x_check_qubits, self.noise_x_check)

        circuit.append("MR", self.z_check_qubits)
        if experiment == "z_memory":
            for k in range(len(self.z_check_qubits)):
                circuit.append(
                    "DETECTOR",
                    [
                        stim.target_rec(-1 - k),
                        stim.target_rec(-1 - k - len(self.x_check_qubits + self.z_check_qubits)),
                    ],
                )
            circuit.append("MR", self.x_check_qubits)
        elif experiment == "x_memory":
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
            raise ValueError(f"Experiment not recognized: '{experiment}'.")

        self.circuit += circuit * self.rounds
        self.circuit.append("M", self.data_qubits)

        if experiment == "z_memory":
            for k in range(len(self.z_check_qubits)):
                z_pcm_row = self.HZ[-1 - k]
                idx_qubits = [i for i, v in enumerate(z_pcm_row) if v]
                lookback_records = [
                    stim.target_rec(-1 - k - len(self.data_qubits + self.x_check_qubits))
                ]
                for idx_qubit in idx_qubits:
                    lookback_records.append(stim.target_rec(idx_qubit - len(self.data_qubits)))
                self.circuit.append("DETECTOR", lookback_records)
        elif experiment == "x_memory":
            for k in range(len(self.x_check_qubits)):
                x_pcm_row = self.HX[-1 - k]
                idx_qubits = [i for i, v in enumerate(x_pcm_row) if v]
                lookback_records = [stim.target_rec(-1 - k - len(self.data_qubits))]
                for idx_qubit in idx_qubits:
                    lookback_records.append(stim.target_rec(idx_qubit - len(self.data_qubits)))
                self.circuit.append("DETECTOR", lookback_records)

        x_logicals, z_logicals = self.logicals()
        x_logicals_qubits = [np.where(logical == 1)[0] for logical in x_logicals]
        z_logicals_qubits = [np.where(logical == 1)[0] for logical in z_logicals]

        logicals_qubits = z_logicals_qubits if experiment == "z_memory" else x_logicals_qubits
        original_logicals_length = len(logicals_qubits)
        if self.logical is not None:
            if type(self.logical) == int:
                self.logical = [self.logical]
            logicals_qubits = [logicals_qubits[i] for i in self.logical]

        print(f"[{'Z' if experiment == 'z_memory' else 'X'} Logical Operators]")
        print(f"Using {len(logicals_qubits)}/{original_logicals_length} logical operators:")
        print(logicals_qubits, "\n")

        for i, logical_qubits in enumerate(logicals_qubits):
            observable_lookback_indices = []
            for logical_qubit in logical_qubits:
                observable_lookback_indices.append(
                    stim.target_rec(logical_qubit - len(self.data_qubits))
                )
            self.circuit.append("OBSERVABLE_INCLUDE", observable_lookback_indices, i)

    def _surface_code(self, experiment: str | None) -> None:
        self.circuit.append("R", self.qubits)
        self.circuit.append("M", self.z_check_qubits + self.x_check_qubits)  # Maybe...

        circuit = stim.Circuit()
        for row in range(self.scale[0]):
            for col in range(self.scale[1]):
                curr_qubit = row * self.scale[1] + col
                if row % 2 == 0 and col % 2 != 0:
                    # Z check and boundary conditions.
                    circuit.append("CNOT", [curr_qubit - 1, curr_qubit])
                    circuit.append("CNOT", [curr_qubit + 1, curr_qubit])

                    if row == 0:
                        circuit.append("CNOT", [curr_qubit + self.scale[1], curr_qubit])
                    elif row == self.scale[0] - 1:
                        circuit.append("CNOT", [curr_qubit - self.scale[1], curr_qubit])
                    else:
                        circuit.append("CNOT", [curr_qubit - self.scale[1], curr_qubit])
                        circuit.append("CNOT", [curr_qubit + self.scale[1], curr_qubit])

                    # Gate noise.
                    if self.noise_circuit is not None:
                        circuit.append(
                            "PAULI_CHANNEL_2", [curr_qubit - 1, curr_qubit], self.noise_circuit
                        )
                        circuit.append(
                            "PAULI_CHANNEL_2", [curr_qubit + 1, curr_qubit], self.noise_circuit
                        )
                        if row == 0:
                            circuit.append(
                                "PAULI_CHANNEL_2",
                                [curr_qubit + self.scale[1], curr_qubit],
                                self.noise_circuit,
                            )
                        elif row == self.scale[0] - 1:
                            circuit.append(
                                "PAULI_CHANNEL_2",
                                [curr_qubit - self.scale[1], curr_qubit],
                                self.noise_circuit,
                            )
                        else:
                            circuit.append(
                                "PAULI_CHANNEL_2",
                                [curr_qubit - self.scale[1], curr_qubit],
                                self.noise_circuit,
                            )
                            circuit.append(
                                "PAULI_CHANNEL_2",
                                [curr_qubit + self.scale[1], curr_qubit],
                                self.noise_circuit,
                            )

                elif row % 2 != 0 and col % 2 == 0:
                    # X check and boundary conditions.
                    circuit.append("H", [curr_qubit])

                    circuit.append("CNOT", [curr_qubit, curr_qubit - self.scale[1]])
                    circuit.append("CNOT", [curr_qubit, curr_qubit + self.scale[1]])
                    if col == 0:
                        circuit.append("CNOT", [curr_qubit, curr_qubit + 1])
                    elif col == self.scale[1] - 1:
                        circuit.append("CNOT", [curr_qubit, curr_qubit - 1])
                    else:
                        circuit.append("CNOT", [curr_qubit, curr_qubit - 1])
                        circuit.append("CNOT", [curr_qubit, curr_qubit + 1])

                    circuit.append("H", [curr_qubit])

                    # Gate noise.
                    if self.noise_circuit is not None:
                        circuit.append(
                            "PAULI_CHANNEL_2",
                            [curr_qubit - self.scale[1], curr_qubit],
                            self.noise_circuit,
                        )
                        circuit.append(
                            "PAULI_CHANNEL_2",
                            [curr_qubit + self.scale[1], curr_qubit],
                            self.noise_circuit,
                        )
                        if col == 0:
                            circuit.append(
                                "PAULI_CHANNEL_2", [curr_qubit + 1, curr_qubit], self.noise_circuit
                            )
                        elif col == self.scale[1] - 1:
                            circuit.append(
                                "PAULI_CHANNEL_2", [curr_qubit - 1, curr_qubit], self.noise_circuit
                            )
                        else:
                            circuit.append(
                                "PAULI_CHANNEL_2", [curr_qubit - 1, curr_qubit], self.noise_circuit
                            )
                            circuit.append(
                                "PAULI_CHANNEL_2", [curr_qubit + 1, curr_qubit], self.noise_circuit
                            )

        if self.noise_data is not None:
            circuit.append("PAULI_CHANNEL_1", self.data_qubits, self.noise_data)
        if self.noise_z_check is not None:
            circuit.append("PAULI_CHANNEL_1", self.z_check_qubits, self.noise_z_check)
        if self.noise_x_check is not None:
            circuit.append("PAULI_CHANNEL_1", self.x_check_qubits, self.noise_x_check)

        circuit.append("MR", self.z_check_qubits)
        if experiment == "z_memory":
            for k in range(len(self.z_check_qubits)):
                circuit.append(
                    "DETECTOR",
                    [
                        stim.target_rec(-1 - k),
                        stim.target_rec(-1 - k - len(self.x_check_qubits + self.z_check_qubits)),
                    ],
                )
            circuit.append("MR", self.x_check_qubits)
        elif experiment == "x_memory":
            circuit.append("MR", self.x_check_qubits)
            for k in range(len(self.x_check_qubits)):
                circuit.append(
                    "DETECTOR",
                    [
                        stim.target_rec(-1 - k),
                        stim.target_rec(-1 - k - len(self.x_check_qubits + self.z_check_qubits)),
                    ],
                )
        elif experiment is None:
            for k in range(len(self.z_check_qubits)):
                circuit.append(
                    "DETECTOR",
                    [
                        stim.target_rec(-1 - k),
                        stim.target_rec(-1 - k - len(self.x_check_qubits + self.z_check_qubits)),
                    ],
                )

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
            raise ValueError(f"Experiment not recognized: '{experiment}'.")

        self.circuit += circuit * self.rounds
        self.circuit.append("M", self.data_qubits)

        num_data_qubits_z = self.scale[1] // 2 + 1
        num_data_qubits_x = self.scale[1] // 2
        row = 0
        skip = 0
        if experiment == "z_memory":
            for k in range(len(self.z_check_qubits)):
                if k % num_data_qubits_x == 0 and k != 0:
                    row += 2
                    skip += num_data_qubits_z
                lookback_records = [
                    stim.target_rec(-1 - k - len(self.data_qubits + self.x_check_qubits)),
                    stim.target_rec(-1 - k - skip),
                    stim.target_rec(-2 - k - skip),
                ]
                if row == 0:
                    lookback_records.append(stim.target_rec(-2 - k - skip - num_data_qubits_x))
                elif row == self.scale[0] - 1:
                    lookback_records.append(stim.target_rec(-1 - k - skip + num_data_qubits_x))
                else:
                    lookback_records.append(stim.target_rec(-1 - k - skip + num_data_qubits_x))
                    lookback_records.append(stim.target_rec(-2 - k - skip - num_data_qubits_x))
                self.circuit.append("DETECTOR", lookback_records)
        elif experiment == "x_memory":
            for k in range(len(self.x_check_qubits)):
                lookback_records = []
                if k % num_data_qubits_z == 0:
                    if k != 0:
                        skip += num_data_qubits_x
                    lookback_records.append(stim.target_rec(-2 - k - skip - num_data_qubits_x))
                elif k % num_data_qubits_z == num_data_qubits_z - 1:
                    lookback_records.append(stim.target_rec(-1 - k - skip - num_data_qubits_x))
                else:
                    lookback_records.append(stim.target_rec(-2 - k - skip - num_data_qubits_x))
                    lookback_records.append(stim.target_rec(-1 - k - skip - num_data_qubits_x))

                lookback_records += [
                    stim.target_rec(-1 - k - len(self.data_qubits + self.x_check_qubits)),
                    stim.target_rec(-1 - k - skip),
                    stim.target_rec(-2 - k - skip),
                ]
                self.circuit.append("DETECTOR", lookback_records)
        elif experiment is None:
            # TODO: Start here for x and z experiment combined. See why this does not work.
            for k in range(len(self.z_check_qubits)):
                if k % num_data_qubits_x == 0 and k != 0:
                    row += 2
                    skip += num_data_qubits_z
                lookback_records = [
                    stim.target_rec(-1 - k - len(self.data_qubits + self.x_check_qubits)),
                    stim.target_rec(-1 - k - skip),
                    stim.target_rec(-2 - k - skip),
                ]
                if row == 0:
                    lookback_records.append(stim.target_rec(-2 - k - skip - num_data_qubits_x))
                elif row == self.scale[0] - 1:
                    lookback_records.append(stim.target_rec(-1 - k - skip + num_data_qubits_x))
                else:
                    lookback_records.append(stim.target_rec(-1 - k - skip + num_data_qubits_x))
                    lookback_records.append(stim.target_rec(-2 - k - skip - num_data_qubits_x))
                self.circuit.append("DETECTOR", lookback_records)

            for k in range(len(self.x_check_qubits)):
                lookback_records = []
                if k % num_data_qubits_z == 0:
                    if k != 0:
                        skip += num_data_qubits_x
                    lookback_records.append(stim.target_rec(-2 - k - skip - num_data_qubits_x))
                elif k % num_data_qubits_z == num_data_qubits_z - 1:
                    lookback_records.append(stim.target_rec(-1 - k - skip - num_data_qubits_x))
                else:
                    lookback_records.append(stim.target_rec(-2 - k - skip - num_data_qubits_x))
                    lookback_records.append(stim.target_rec(-1 - k - skip - num_data_qubits_x))

                lookback_records += [
                    stim.target_rec(-1 - k - len(self.data_qubits + self.x_check_qubits)),
                    stim.target_rec(-1 - k - skip),
                    stim.target_rec(-2 - k - skip),
                ]
                self.circuit.append("DETECTOR", lookback_records)

        observable_lookback_indices = []
        if experiment == "z_memory":
            for k in range(self.scale[0] // 2 + 1):
                observable_lookback_indices.append(
                    stim.target_rec(
                        -k * (num_data_qubits_z + num_data_qubits_x) - num_data_qubits_z
                    )
                )
            self.circuit.append("OBSERVABLE_INCLUDE", observable_lookback_indices, 0)
        elif experiment == "x_memory":
            observable_lookback_indices = [
                stim.target_rec(-1 - k) for k in range(num_data_qubits_z)
            ]
            self.circuit.append("OBSERVABLE_INCLUDE", observable_lookback_indices, 0)
        elif experiment is None:
            z_observable_lookback_indices = []
            for k in range(self.scale[0] // 2 + 1):
                z_observable_lookback_indices.append(
                    stim.target_rec(
                        -k * (num_data_qubits_z + num_data_qubits_x) - num_data_qubits_z
                    )
                )
            x_observable_lookback_indices = [
                stim.target_rec(-1 - k) for k in range(num_data_qubits_z)
            ]
            self.circuit.append("OBSERVABLE_INCLUDE", z_observable_lookback_indices, 0)
            self.circuit.append("OBSERVABLE_INCLUDE", x_observable_lookback_indices, 1)

    def _repetition_code(self) -> None:
        # We have to add initial dummy measurements for the detector to detect change in the first
        # set of qubit measurements.
        self.circuit.append("M", self.z_check_qubits)

        circuit = stim.Circuit()

        # Stabilizer measurements.
        for m in self.z_check_qubits:
            circuit.append("CNOT", [m - 1, m])
            circuit.append("CNOT", [m + 1, m])
            if self.noise_circuit is not None:
                circuit.append("PAULI_CHANNEL_2", [m - 1, m], self.noise_circuit)
                circuit.append("PAULI_CHANNEL_2", [m + 1, m], self.noise_circuit)

        # Apply random errors on qubits.
        if self.noise_data is not None:
            circuit.append("PAULI_CHANNEL_1", self.data_qubits, self.noise_data)
        if self.noise_z_check is not None:
            circuit.append("PAULI_CHANNEL_1", self.z_check_qubits, self.noise_z_check)

        # This measures and resets (to zero) the check qubits.
        circuit.append("MR", self.z_check_qubits)

        # Compare the last measurement result to the one previous to that of the same qubit.
        for k in range(len(self.z_check_qubits)):
            circuit.append(
                "DETECTOR", [stim.target_rec(-1 - k), stim.target_rec(-1 - k - self.distance)]
            )

        # Concatenate the circuits.
        self.circuit += circuit * self.rounds

        # Measure data qubits at the end.
        self.circuit.append("M", self.data_qubits)
        for k in range(len(self.z_check_qubits)):
            self.circuit.append(
                "DETECTOR",
                [
                    stim.target_rec(-1 - k),
                    stim.target_rec(-2 - k),
                    stim.target_rec(-2 - k - self.distance),
                ],
            )

        # Add observable.
        self.circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)

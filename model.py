import numpy as np
import networkx as nx
import stim
import matplotlib.pyplot as plt

from itertools import combinations

import util


class StabilizerModel:
    def __init__(
        self,
        code: str,
        circuit: stim.Circuit = None,
        rounds: int = 3,
        noise_circuit: float | list[float] = 0.0,
        noise_data: float | list[float] = 0.0,
        noise_z_check: float | list[float] = 0.0,
        noise_x_check: float | list[float] = 0.0,
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
                self.pos = None if "pos" not in self.code_params.keys() else self.code_params["pos"]
                self.clist1 = self.code_params["clist1"]
                self.clist2 = self.code_params["clist2"]

                H1 = util.classical_pcm(self.clist1)
                H2 = util.classical_pcm(self.clist2)

                num_qubits = sum(H1.shape) * sum(H2.shape)
                self.qubits = np.arange(num_qubits)

                z_check_order = [
                    "Q" if s == "B" else "Z"
                    for s in self.clist2
                    if not np.issubdtype(type(s), np.number)
                ]

                x_check_order = [
                    "X" if s == "B" else "Q"
                    for s in self.clist2
                    if not np.issubdtype(type(s), np.number)
                ]

                check_order = np.array(
                    [
                        z_check_order if s == "B" else x_check_order
                        for s in self.clist1
                        if not np.issubdtype(type(s), np.number)
                    ]
                ).flatten()

                self.data_qubits = [q for q, s in zip(self.qubits, check_order) if s == "Q"]
                self.z_check_qubits = [q for q, s in zip(self.qubits, check_order) if s == "Z"]
                self.x_check_qubits = [q for q, s in zip(self.qubits, check_order) if s == "X"]

                self._hypergraph_product_code(experiment)
            case _:
                raise ValueError(f"Code not recognized: '{main_code}'.")

        # TODO: Implement X and Z error propagation analysis.
        # TODO: Implement single-gate noise, like for Hadamard gates in X stabilizer syndrome extraction.

    # ------------------------------------ Setters and Getters ------------------------------------

    def reset_data_qubits(self) -> None:
        # TODO: Change to `set_data_qubits` and have it set the data qubits generally.
        self.circuit.append("R", self.data_qubits)

    # -------------------------------------- Utility Methods --------------------------------------

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

    def shot(self, detector: bool = True) -> None:
        # TODO: Update.
        sample = (
            self.circuit.compile_detector_sampler() if detector else self.circuit.compile_sampler()
        ).sample(1)[0]

        # Account for dummy measurement when using detectors.
        effective_rounds = self.rounds if detector else self.rounds + 1

        marker = "x" if detector else "o"
        round_list = [marker if outcome else "_" for outcome in sample]
        return np.reshape(round_list, (effective_rounds, len(sample) // effective_rounds))

    def print(self) -> None:
        print(self.circuit, "\n")

    def node_to_qubit(self, node: str) -> int:
        def strip(s: str) -> int:
            s_int = ""
            for c in s:
                if c.isdigit():
                    s_int += c

            return int(s_int)

        if node[1] == "q":
            return self.data_qubits[strip(node) - 1]
        elif node[1] == "X":
            return self.x_check_qubits[strip(node) - 1]
        elif node[1] == "Z":
            return self.z_check_qubits[strip(node) - 1]

    def crossings(self) -> set[frozenset[tuple[int, int]]]:
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

        crossings = set()
        qubit_edges = [tuple(map(self.node_to_qubit, edge)) for edge in self.graph.edges]
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

    def draw(
        self,
        with_labels: bool = False,
        layout: str = None,
        crossings: bool = True,
        connection_rad: float = 0.0,
        **kwargs,
    ) -> None:
        """Colors nodes based on their type ('q', 'x', 'z') and plots the graph."""

        colors = {"q": "#D3D3D3", "X": "#FFC0CB", "Z": "#ADD8E6"}
        shapes = {"q": "o", "X": "s", "Z": "s"}
        sizes = {"q": 300, "X": 230, "Z": 230}

        if layout == "spring":
            layout = nx.spring_layout(self.graph, iterations=5000)
        elif layout == "random":
            layout = nx.random_layout(self.graph)
        elif layout is None:
            layout = {}
            for node in self.graph.nodes():
                qubit = self.node_to_qubit(node)
                layout[node] = self.pos[qubit]
        else:
            raise ValueError(f"Qubit layout not recognized: `{layout}`.")

        for node_type, shape in shapes.items():
            filtered_nodes = [node for node in self.graph.nodes() if node[1] == node_type]
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

    # ------------------------------------------- Codes -------------------------------------------

    def _hypergraph_product_code(self, experiment: str | None) -> None:
        # TODO: Add crossings with `self.pos`.

        self.circuit.append("R", self.qubits)
        self.circuit.append("M", self.z_check_qubits + self.x_check_qubits)  # Maybe...

        H1 = util.classical_pcm(self.clist1)
        H2 = util.classical_pcm(self.clist2)
        HX, HZ = util.hypergraph_pcm(H1, H2)

        self.graph = util.construct_graph(HX, HZ)

        circuit = stim.Circuit()
        for target_qubit, z_pcm_row in zip(self.z_check_qubits, HZ):
            qubits = [self.data_qubits[i] for i, v in enumerate(z_pcm_row) if v]
            for qubit in qubits:
                circuit.append("CNOT", [qubit, target_qubit])
                if self.noise_circuit is not None:
                    circuit.append("PAULI_CHANNEL_2", [qubit, target_qubit], self.noise_circuit)

        for target_qubit, x_pcm_row in zip(self.x_check_qubits, HX):
            qubits = [self.data_qubits[i] for i, v in enumerate(x_pcm_row) if v]
            for qubit in qubits:
                circuit.append("H", [target_qubit])
                circuit.append("CNOT", [qubit, target_qubit])
                circuit.append("H", [target_qubit])
                if self.noise_circuit is not None:
                    circuit.append("PAULI_CHANNEL_2", [qubit, target_qubit], self.noise_circuit)

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
            # TODO: Add detectors.
            pass
        else:
            raise ValueError(f"Experiment not recognized: '{experiment}'.")

        self.circuit += circuit * self.rounds
        self.circuit.append("M", self.data_qubits)

        if experiment == "z_memory":
            for k in range(len(self.z_check_qubits)):
                z_pcm_row = HZ[-1 - k]
                idx_qubits = [i for i, v in enumerate(z_pcm_row) if v]
                lookback_records = [
                    stim.target_rec(-1 - k - len(self.data_qubits + self.x_check_qubits))
                ]
                for idx_qubit in idx_qubits:
                    lookback_records.append(stim.target_rec(idx_qubit - len(self.data_qubits)))
                self.circuit.append("DETECTOR", lookback_records)
        elif experiment == "x_memory":
            for k in range(len(self.x_check_qubits)):
                x_pcm_row = HX[-1 - k]
                idx_qubits = [i for i, v in enumerate(x_pcm_row) if v]
                lookback_records = [stim.target_rec(-1 - k - len(self.data_qubits))]
                for idx_qubit in idx_qubits:
                    lookback_records.append(stim.target_rec(idx_qubit - len(self.data_qubits)))
                self.circuit.append("DETECTOR", lookback_records)
        elif experiment is None:
            # TODO: Add detectors.
            pass

        observable_lookback_indices = []
        if experiment == "z_memory":
            for k in range(H1.shape[1]):
                observable_lookback_indices.append(
                    stim.target_rec(-k * (H2.shape[0] + H2.shape[1]) - H2.shape[1])
                )
            self.circuit.append("OBSERVABLE_INCLUDE", observable_lookback_indices, 0)
        elif experiment == "x_memory":
            observable_lookback_indices = [stim.target_rec(-1 - k) for k in range(H2.shape[1])]
            self.circuit.append("OBSERVABLE_INCLUDE", observable_lookback_indices, 0)
        elif experiment is None:
            # TODO: Add observable.
            pass

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

                    circuit.append("CNOT", [curr_qubit - self.scale[1], curr_qubit])
                    circuit.append("CNOT", [curr_qubit + self.scale[1], curr_qubit])
                    if col == 0:
                        circuit.append("CNOT", [curr_qubit + 1, curr_qubit])
                    elif col == self.scale[1] - 1:
                        circuit.append("CNOT", [curr_qubit - 1, curr_qubit])
                    else:
                        circuit.append("CNOT", [curr_qubit - 1, curr_qubit])
                        circuit.append("CNOT", [curr_qubit + 1, curr_qubit])

                    circuit.append("H", [curr_qubit])
                    # TODO: Apply noise after Hadamard gates.

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

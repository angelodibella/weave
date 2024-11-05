import numpy as np
import stim
import networkx as nx


# TODO: Make a noise model container, possibly a class, to contain all parameters passed to code classes.
class NoiseModel:
    pass


class Code:
    def __init__(
            self,
            circuit: stim.Circuit = None,
            rounds: int = 3,
            pos: str | list[tuple[int, int]] = None,
            noise_circuit: float | list[float] = 0.0,
            noise_crossing: float | list[float] = 0.0,
            noise_data: float | list[float] = 0.0,
            noise_z_check: float | list[float] = 0.0,
            noise_x_check: float | list[float] = 0.0,
            experiment: str = "z_memory",
            logical: int | list[int] = None,
    ) -> None:
        self.circuit = stim.Circuit() if circuit is None else circuit
        self.rounds = rounds

        self.pos = pos

        self.qubits = []
        self.data_qubits = []
        self.z_check_qubits = []
        self.x_check_qubits = []

        self.graph = None

        self.logical = logical
        self.experiment = experiment

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

    def construct_code(self):
        pass

    def construct_graph(self):
        pass


# TODO: Implement a classical code class, and make code classes acquire this instead of clists.
class ClassicalCode:
    def __init__(self):
        pass

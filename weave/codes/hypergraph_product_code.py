import networkx as nx
import numpy as np
import stim
from ldpc import mod2
from matplotlib import pyplot as plt

from .base import NoiseModel
from ..util import pcm
from ..util.graph import compute_layout, find_edge_crossings, draw_graph


class HypergraphProductCode:
    """
    Hypergraph product code based on two classical codes.

    Constructs the code from classical list representations (clist1 and clist2),
    generates the corresponding Tanner graph, and builds the Stim circuit for simulation.

    Parameters
    ----------
    clist1 : list
        Classical list representation of the first code.
    clist2 : list
        Classical list representation of the second code.
    circuit : stim.Circuit, optional
        A Stim circuit to which the code circuit will be appended. If None, a new circuit is created.
    rounds : int, optional
        Number of measurement rounds. Default is 3.
    pos : str or list of tuple of int, optional
        Layout specification for graph embedding. Can be a layout keyword ("random", "spring",
        "bipartite", "tripartite") or a custom list of positions. If None, defaults to "random".
    noise : NoiseModel, optional
        Noise model for circuit operations. Default is a zero-noise model.
    experiment : str, optional
        Experiment type ("z_memory" or "x_memory"). Default is "z_memory".
    logical : int or list of int, optional
        Index (or indices) of logical operators to use. Default is None (use all).

    Raises
    ------
    ValueError
        If an unrecognized layout or experiment type is provided.
    """

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
        self.logical = logical

        # Convert classical lists to parity-check matrices.
        self.H1: np.ndarray = pcm.to_matrix(clist1)
        self.H2: np.ndarray = pcm.to_matrix(clist2)

        # Define qubit indices.
        self.qubits = np.arange(sum(self.H1.shape) * sum(self.H2.shape))

        # Determine ordering for check qubits.
        z_order = ["Q" if s == "B" else "Z" for s in clist2 if not np.issubdtype(type(s), np.number)]
        x_order = ["X" if s == "B" else "Q" for s in clist2 if not np.issubdtype(type(s), np.number)]
        check_order = np.array(
            [z_order if s == "B" else x_order for s in clist1 if not np.issubdtype(type(s), np.number)]).flatten()

        self.data_qubits = [q for q, t in zip(self.qubits, check_order) if t == "Q"]
        self.z_check_qubits = [q for q, t in zip(self.qubits, check_order) if t == "Z"]
        self.x_check_qubits = [q for q, t in zip(self.qubits, check_order) if t == "X"]

        # Compute hypergraph product matrices.
        self.HX, self.HZ = pcm.hypergraph_product(self.H1, self.H2)
        self.graph = self.construct_graph()

        # Compute node positions using the general layout utility.
        if self.pos is None or isinstance(self.pos, str):
            self.pos = compute_layout(self.graph, self.pos or "random", index_key="index")

        # Compute crossings using the utility function.
        edges = [tuple(self.graph.nodes[node]["index"] for node in edge) for edge in self.graph.edges]
        self.crossings = find_edge_crossings(self.pos, edges)

        self.construct_code()

    def reset_data_qubits(self) -> None:
        """Append a reset operation for data qubits to the Stim circuit."""
        self.circuit.append("R", self.data_qubits)

    def find_logicals(self):
        """
        Compute logical operators from the parity-check matrices.

        Returns
        -------
        tuple
            (x_logicals, z_logicals) logical operator matrices.
        """

        def compute_lz(HX, HZ):
            ker = mod2.nullspace(HX)
            basis = mod2.row_basis(HZ)
            logicals = np.vstack([basis, ker])
            pivots = mod2.row_echelon(logicals.T)[3]
            indices = [i for i in range(basis.shape[0], logicals.shape[0]) if i in pivots]
            return logicals[indices]

        x_logicals = compute_lz(self.HZ, self.HX)
        z_logicals = compute_lz(self.HX, self.HZ)
        return x_logicals, z_logicals

    def construct_graph(self) -> nx.Graph:
        """
        Construct the Tanner graph from HX and HZ matrices.

        Returns
        -------
        nx.Graph
            The Tanner graph with node types and layers assigned.
        """
        num_data = self.HZ.shape[1]
        num_z = self.HZ.shape[0]
        num_x = self.HX.shape[0]

        A = np.block([
            [np.zeros((num_data, num_data)), self.HZ.T, self.HX.T],
            [self.HZ, np.zeros((num_z, num_z)), np.zeros((num_z, num_x))],
            [self.HX, np.zeros((num_x, num_z)), np.zeros((num_x, num_x))]
        ]).astype(int)

        G = nx.from_numpy_array(A)
        labels = {}
        for i in range(num_data):
            G.nodes[i]["type"] = "q"
            G.nodes[i]["index"] = self.data_qubits[i]
            labels[i] = f"$q_{{{i + 1}}}$"
            G.nodes[i]["layer"] = 1
        for i in range(num_data, num_data + num_z):
            G.nodes[i]["type"] = "Z"
            G.nodes[i]["index"] = self.z_check_qubits[i - num_data]
            labels[i] = f"$Z_{{{i - num_data + 1}}}$"
            G.nodes[i]["layer"] = 0
        for i in range(num_data + num_z, num_data + num_z + num_x):
            G.nodes[i]["type"] = "X"
            G.nodes[i]["index"] = self.x_check_qubits[i - (num_data + num_z)]
            labels[i] = f"$X_{{{i - (num_data + num_z) + 1}}}$"
            G.nodes[i]["layer"] = 2

        return nx.relabel_nodes(G, labels)

    def crossing_number(self) -> int:
        return len(self.crossings)

    def construct_code(self) -> None:
        """
        Build the Stim circuit for the code, including stabilizer operations, noise channels,
        measurement rounds, detectors, and observable inclusions.
        """
        self.circuit.append("R", self.qubits)
        self.circuit.append("M", self.z_check_qubits + self.x_check_qubits)

        circuit = stim.Circuit()

        # Z-check stabilizer operations.
        for target, row in zip(self.z_check_qubits, self.HZ):
            data_idxs = [self.data_qubits[i] for i, v in enumerate(row) if v]
            for dq in data_idxs:
                circuit.append("CNOT", [dq, target])
                circuit.append("PAULI_CHANNEL_2", [dq, target], self.noise.circuit)

        # X-check stabilizer operations.
        for target, row in zip(self.x_check_qubits, self.HX):
            data_idxs = [self.data_qubits[i] for i, v in enumerate(row) if v]
            for dq in data_idxs:
                circuit.append("H", [target])
                circuit.append("CNOT", [target, dq])
                circuit.append("H", [target])
                circuit.append("PAULI_CHANNEL_2", [dq, target], self.noise.circuit)

        # Noise for crossing edges.
        for crossing in self.crossings:
            for edge in crossing:
                circuit.append("PAULI_CHANNEL_2", [edge[0], edge[1]], self.noise.crossing)

        # Apply single-qubit noise.
        circuit.append("PAULI_CHANNEL_1", self.data_qubits, self.noise.data)
        circuit.append("PAULI_CHANNEL_1", self.z_check_qubits, self.noise.z_check)
        circuit.append("PAULI_CHANNEL_1", self.x_check_qubits, self.noise.x_check)

        # Measurement rounds and detectors.
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

        # Additional detectors.
        if self.experiment == "z_memory":
            for k in range(len(self.z_check_qubits)):
                row = self.HZ[-1 - k]
                idxs = [i for i, v in enumerate(row) if v]
                recs = [stim.target_rec(-1 - k - len(self.data_qubits + self.x_check_qubits))]
                for idx in idxs:
                    recs.append(stim.target_rec(idx - len(self.data_qubits)))
                self.circuit.append("DETECTOR", recs)
        elif self.experiment == "x_memory":
            for k in range(len(self.x_check_qubits)):
                row = self.HX[-1 - k]
                idxs = [i for i, v in enumerate(row) if v]
                recs = [stim.target_rec(-1 - k - len(self.data_qubits))]
                for idx in idxs:
                    recs.append(stim.target_rec(idx - len(self.data_qubits)))
                self.circuit.append("DETECTOR", recs)

        # Logical operator extraction and inclusion.
        x_logicals, z_logicals = self.find_logicals()
        x_logical_qubits = [np.where(log == 1)[0] for log in x_logicals]
        z_logical_qubits = [np.where(log == 1)[0] for log in z_logicals]
        logicals = z_logical_qubits if self.experiment == "z_memory" else x_logical_qubits
        orig_count = len(logicals)
        if self.logical is not None:
            if isinstance(self.logical, int):
                self.logical = [self.logical]
            logicals = [logicals[i] for i in self.logical]
        print(f"[{'Z' if self.experiment == 'z_memory' else 'X'} Logical Operators]")
        print(f"Using {len(logicals)}/{orig_count} logical operators:")
        print(logicals, "\n")
        for i, lq in enumerate(logicals):
            recs = [stim.target_rec(q - len(self.data_qubits)) for q in lq]
            self.circuit.append("OBSERVABLE_INCLUDE", recs, i)

    def draw(
            self,
            with_labels: bool = False,
            crossings: bool = True,
            connection_rad: float = 0.0,
            **kwargs,
    ) -> None:
        """
        Draw the Tanner graph using the general draw function.

        Parameters
        ----------
        with_labels : bool, optional
            Whether to display node labels.
        crossings : bool, optional
            Whether to highlight edge crossings.
        connection_rad : float, optional
            Edge curvature.
        **kwargs
            Additional keyword arguments for drawing.
        """
        draw_graph(self.graph, self.pos, with_labels=with_labels, crossings=crossings, connection_rad=connection_rad,
                   **kwargs)

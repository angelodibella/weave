"""Implementation of CSS (Calderbank-Shor-Steane) quantum error-correcting codes."""

import numpy as np
import stim
import networkx as nx
from typing import Optional, Union, List, Tuple, Any

from .base import NoiseModel, QuantumCode
from ..util import pcm, graph


class CSSCode(QuantumCode):
    """
    Base class for Calderbank-Shor-Steane (CSS) quantum error-correcting codes.

    CSS codes are a special class of stabilizer codes constructed from two
    classical linear codes. They have the property that X and Z errors can
    be corrected independently.

    Parameters
    ----------
    HX : np.ndarray
        X-type parity check matrix.
    HZ : np.ndarray
        Z-type parity check matrix.
    circuit : Optional[stim.Circuit]
        A Stim circuit to which the code circuit will be appended. If None, a new circuit is created.
    rounds : int
        Number of measurement rounds. Default is 3.
    noise : NoiseModel
        Noise model for circuit operations. Default is a zero-noise model.
    experiment : str
        Experiment type ("z_memory" or "x_memory"). Default is "z_memory".
    logical : Optional[Union[int, List[int]]]
        Index (or indices) of logical operators to use. Default is None (use all).
    """

    def __init__(
        self,
        HX: np.ndarray,
        HZ: np.ndarray,
        rounds: int = 3,
        noise: NoiseModel = NoiseModel(),
        experiment: str = "z_memory",
        logical: Optional[Union[int, List[int]]] = None,
    ) -> None:
        # Verify that HX and HZ satisfy the CSS condition.
        if not np.all(np.mod(np.dot(HX, HZ.T), 2) == 0):
            raise ValueError("CSS condition not satisfied.")

        # Calculate logical qubits.
        k = (
            HZ.shape[1]
            - np.linalg.matrix_rank(HZ.astype(float))
            - np.linalg.matrix_rank(HX.astype(float))
        )
        n = HZ.shape[1] + HZ.shape[0] + HX.shape[0]

        # Initialize the parent class.
        super().__init__(n=n, k=k)

        # Code properties.
        self.HX = HX
        self.HZ = HZ
        self.qubits = []
        self.data_qubits = []
        self.z_check_qubits = []
        self.x_check_qubits = []
        self.logical = logical

        # Error correction parameters.
        self.experiment = experiment
        self.noise = noise
        self.rounds = rounds
        self.circuit = stim.Circuit()

        # Embedding properties.
        self.pos = None
        self.graph: nx.Graph = None
        self.crossings = []

        # Derived properties.
        self._compute_qubit_indices()

        # Generate the error-correcting circuit.
        self.generate()

    def generate(self) -> stim.Circuit:
        """
        Build the Stim circuit for the code, including stabilizer operations, noise channels,
        measurement rounds, detectors, and observable inclusions.
        """

        # ---------------- Circuit Head ----------------

        if self.experiment == "z_memory":
            self.circuit.append("R", self.qubits)
        elif self.experiment == "x_memory":
            self.circuit.append("RX", self.data_qubits)
            self.circuit.append("R", self.z_check_qubits + self.x_check_qubits)
        else:
            raise ValueError(f"Experiment not recognized: '{self.experiment}'.")

        # ---------------- Round Initialization ----------------

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
                circuit.append(
                    "PAULI_CHANNEL_2", [edge[0], edge[1]], self.noise.crossing
                )

        # Apply single-qubit noise.
        circuit.append("PAULI_CHANNEL_1", self.data_qubits, self.noise.data)
        circuit.append("PAULI_CHANNEL_1", self.z_check_qubits, self.noise.z_check)
        circuit.append("PAULI_CHANNEL_1", self.x_check_qubits, self.noise.x_check)

        # Measure and reset before the rest of stabilizer rounds.
        circuit.append("MR", self.z_check_qubits + self.x_check_qubits)

        # Add first round to the circuit.
        self.circuit += circuit

        # Add initial detectors.
        if self.experiment == "z_memory":
            for k in range(len(self.z_check_qubits)):
                self.circuit.append(
                    "DETECTOR", [stim.target_rec(-1 - k - len(self.x_check_qubits))]
                )
        elif self.experiment == "x_memory":
            for k in range(len(self.x_check_qubits)):
                self.circuit.append("DETECTOR", [stim.target_rec(-1 - k)])

        # ---------------- Round Operations ----------------

        # Add the detectors for the rest of the rounds on all stabilizers.
        for k in range(len(self.z_check_qubits + self.x_check_qubits)):
            circuit.append(
                "DETECTOR",
                [
                    stim.target_rec(-1 - k),
                    stim.target_rec(
                        -1 - k - len(self.z_check_qubits + self.x_check_qubits)
                    ),
                ],
            )

        self.circuit += circuit * (self.rounds - 1)

        # ---------------- Circuit Tail ----------------

        if self.experiment == "z_memory":
            self.circuit.append("M", self.data_qubits)
            for k in range(len(self.z_check_qubits)):
                row = self.HZ[-1 - k]
                idxs = [i for i, v in enumerate(row) if v]
                recs = [
                    stim.target_rec(
                        -1 - k - len(self.data_qubits + self.x_check_qubits)
                    )
                ]
                for idx in idxs:
                    recs.append(stim.target_rec(idx - len(self.data_qubits)))
                self.circuit.append("DETECTOR", recs)
        elif self.experiment == "x_memory":
            self.circuit.append("MX", self.data_qubits)
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
        logicals = (
            z_logical_qubits if self.experiment == "z_memory" else x_logical_qubits
        )

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

        return self.circuit

    def embed(self, pos: Optional[Union[str, List[Tuple[int, int]]]] = None) -> None:
        self.graph = self._construct_graph()

        # Compute node positions using the general layout utility.
        if pos is None or isinstance(pos, str):
            self.pos = graph.compute_layout(
                self.graph, pos or "random", index_key="index"
            )
        else:
            self.pos = pos

        # Compute crossings using the utility function.
        edges = [
            tuple(self.graph.nodes[node]["index"] for node in edge)
            for edge in self.graph.edges
        ]
        self.crossings = graph.find_edge_crossings(self.pos, edges)

        # Clear the circuit and regenerate it with the new embedding.
        self.circuit = stim.Circuit()
        self.generate()

    def find_logicals(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute logical operators from the parity-check matrices.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (x_logicals, z_logicals) logical operator matrices.
        """

        def compute_lz(HX_mat, HZ_mat):
            ker = pcm.nullspace(HX_mat)
            basis = pcm.mod2.row_basis(HZ_mat)
            if ker.shape[0] == 0 or basis.shape[0] == 0:
                return np.zeros((0, HX_mat.shape[1]), dtype=int)

            logicals = np.vstack([basis, ker])
            pivots = pcm.mod2.row_echelon(logicals.T)[3]
            indices = [
                i for i in range(basis.shape[0], logicals.shape[0]) if i in pivots
            ]
            return logicals[indices]

        x_logicals = compute_lz(self.HZ, self.HX)
        z_logicals = compute_lz(self.HX, self.HZ)
        return x_logicals, z_logicals

    def draw(
        self,
        with_labels: bool = False,
        crossings: bool = True,
        connection_rad: float = 0.0,
        **kwargs,
    ) -> None:
        """
        Draw the Tanner graph using matplotlib.

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
        # # Need to compute positions for the graph
        # pos = graph.compute_layout(self.graph, "tripartite", index_key="index")

        if self.pos is None:
            raise RuntimeError(
                "Embedding not specified. Use the embed() method to generate node positions."
            )

        # Call the general draw function
        graph.draw(
            self.graph,
            self.pos,
            with_labels=with_labels,
            crossings=crossings,
            connection_rad=connection_rad,
            **kwargs,
        )

    def _compute_qubit_indices(self) -> None:
        """Compute qubit indices for data and check qubits."""
        num_data = self.HZ.shape[1]
        self.data_qubits = list(range(num_data))

        # Assign indices for check qubits after data qubits.
        self.z_check_qubits = list(range(num_data, num_data + self.HZ.shape[0]))
        self.x_check_qubits = list(
            range(
                num_data + self.HZ.shape[0],
                num_data + self.HZ.shape[0] + self.HX.shape[0],
            )
        )

        # All qubits.
        self.qubits = self.data_qubits + self.z_check_qubits + self.x_check_qubits

    def _construct_graph(self) -> nx.Graph:
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

        # Construct adjacency matrix for the Tanner graph.
        A = np.block(
            [
                [np.zeros((num_data, num_data)), self.HZ.T, self.HX.T],
                [self.HZ, np.zeros((num_z, num_z)), np.zeros((num_z, num_x))],
                [self.HX, np.zeros((num_x, num_z)), np.zeros((num_x, num_x))],
            ]
        ).astype(int)

        G = nx.from_numpy_array(A)
        labels = {}

        # Assign node attributes.
        for i in self.data_qubits:
            G.nodes[i]["type"] = "q"
            G.nodes[i]["index"] = i
            labels[i] = f"$q_{{{i + 1}}}$"
            G.nodes[i]["layer"] = 1

        for i in self.z_check_qubits:
            G.nodes[i]["type"] = "Z"
            G.nodes[i]["index"] = i
            labels[i] = f"$Z_{{{i - num_data + 1}}}$"
            G.nodes[i]["layer"] = 0

        for i in self.x_check_qubits:
            G.nodes[i]["type"] = "X"
            G.nodes[i]["index"] = i
            labels[i] = f"$X_{{{i - (num_data + num_z) + 1}}}$"
            G.nodes[i]["layer"] = 2

        return nx.relabel_nodes(G, labels)

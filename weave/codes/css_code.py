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
        circuit: Optional[stim.Circuit] = None,
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

        self.HX = HX
        self.HZ = HZ
        self.circuit = stim.Circuit() if circuit is None else circuit
        self.rounds = rounds
        self.noise = noise
        self.experiment = experiment
        self.logical = logical

        self.qubits = []
        self.data_qubits = []
        self.z_check_qubits = []
        self.x_check_qubits = []

        # Derived properties.
        self._compute_qubit_indices()
        self.graph = self._construct_graph()

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

    def logical_operators(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the logical X and Z operators of the code.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing the logical X and Z operators.
        """
        return self.find_logicals()

    def encode_circuit(self) -> stim.Circuit:
        """
        Generate a circuit to encode logical states into the code.

        This is a simple implementation that resets all qubits and applies the
        necessary CNOT gates to create the code state.

        Returns
        -------
        stim.Circuit
            The encoding circuit.
        """
        circuit = stim.Circuit()

        # Initialize all qubits to |0⟩
        circuit.append("R", self.qubits)

        # Apply Hadamard to create |+⟩ states for X-basis qubits if needed
        if self.experiment == "x_memory":
            circuit.append("H", self.data_qubits)

        # Apply CNOT gates based on the parity check matrices
        # This is a simplified encoding - more efficient encoders could be implemented

        return circuit

    def syndrome_circuit(self) -> stim.Circuit:
        """
        Generate a circuit to measure the syndrome of the code.

        Returns
        -------
        stim.Circuit
            A circuit for syndrome measurement.
        """
        circuit = stim.Circuit()

        # Reset ancilla qubits
        circuit.append("R", self.z_check_qubits + self.x_check_qubits)

        # Z-check stabilizer operations
        for target, row in zip(self.z_check_qubits, self.HZ):
            data_idxs = [self.data_qubits[i] for i, v in enumerate(row) if v]
            for dq in data_idxs:
                circuit.append("CNOT", [dq, target])

        # X-check stabilizer operations
        for target, row in zip(self.x_check_qubits, self.HX):
            data_idxs = [self.data_qubits[i] for i, v in enumerate(row) if v]
            for dq in data_idxs:
                circuit.append("H", [target])
                circuit.append("CNOT", [target, dq])
                circuit.append("H", [target])

        # Measure syndrome bits
        circuit.append("M", self.z_check_qubits + self.x_check_qubits)

        return circuit

    def construct_circuit(self) -> None:
        """
        Build the Stim circuit for the code, including stabilizer operations, noise channels,
        measurement rounds, detectors, and observable inclusions.
        """
        # Initialize all qubits
        self.circuit.append("R", self.qubits)
        self.circuit.append("M", self.z_check_qubits + self.x_check_qubits)

        circuit = stim.Circuit()

        # Z-check stabilizer operations
        for target, row in zip(self.z_check_qubits, self.HZ):
            data_idxs = [self.data_qubits[i] for i, v in enumerate(row) if v]
            for dq in data_idxs:
                circuit.append("CNOT", [dq, target])
                circuit.append("PAULI_CHANNEL_2", [dq, target], self.noise.circuit)

        # X-check stabilizer operations
        for target, row in zip(self.x_check_qubits, self.HX):
            data_idxs = [self.data_qubits[i] for i, v in enumerate(row) if v]
            for dq in data_idxs:
                circuit.append("H", [target])
                circuit.append("CNOT", [target, dq])
                circuit.append("H", [target])
                circuit.append("PAULI_CHANNEL_2", [dq, target], self.noise.circuit)

        # Apply single-qubit noise
        circuit.append("PAULI_CHANNEL_1", self.data_qubits, self.noise.data)
        circuit.append("PAULI_CHANNEL_1", self.z_check_qubits, self.noise.z_check)
        circuit.append("PAULI_CHANNEL_1", self.x_check_qubits, self.noise.x_check)

        # Measurement rounds and detectors
        circuit.append("MR", self.z_check_qubits)
        if self.experiment == "z_memory":
            for k in range(len(self.z_check_qubits)):
                circuit.append(
                    "DETECTOR",
                    [
                        stim.target_rec(-1 - k),
                        stim.target_rec(
                            -1 - k - len(self.x_check_qubits + self.z_check_qubits)
                        ),
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
                        stim.target_rec(
                            -1 - k - len(self.x_check_qubits + self.z_check_qubits)
                        ),
                    ],
                )
        else:
            raise ValueError(f"Experiment not recognized: '{self.experiment}'.")

        self.circuit += circuit * self.rounds
        self.circuit.append("M", self.data_qubits)

        # Add logical operators as observables
        self._add_logical_observables()

    def _add_logical_observables(self) -> None:
        """Add logical operators as observables to the circuit."""
        x_logicals, z_logicals = self.find_logicals()

        # Select which logical operators to use based on experiment type
        logicals = z_logicals if self.experiment == "z_memory" else x_logicals
        orig_count = len(logicals)

        # Filter logicals if specific indices are provided
        if self.logical is not None:
            if isinstance(self.logical, int):
                self.logical = [self.logical]
            logicals = [logicals[i] for i in self.logical]

        print(f"[{'Z' if self.experiment == 'z_memory' else 'X'} Logical Operators]")
        print(f"Using {len(logicals)}/{orig_count} logical operators:")
        print(logicals, "\n")

        # Add observables to the circuit
        for i, log in enumerate(logicals):
            qubit_indices = np.where(log == 1)[0]
            recs = [stim.target_rec(q - len(self.data_qubits)) for q in qubit_indices]
            self.circuit.append("OBSERVABLE_INCLUDE", recs, i)

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
        # Need to compute positions for the graph
        pos = graph.compute_layout(self.graph, "tripartite", index_key="index")

        # Call the general draw function
        graph.draw(
            self.graph,
            pos,
            with_labels=with_labels,
            crossings=crossings,
            connection_rad=connection_rad,
            **kwargs,
        )

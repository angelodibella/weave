"""Implementation of hypergraph product codes."""

import networkx as nx
import numpy as np
import stim
from typing import Optional, Union, List, Tuple, Any

from .base import NoiseModel
from .css_code import CSSCode
from ..util import pcm, graph


class HypergraphProductCode(CSSCode):
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
    circuit : Optional[stim.Circuit]
        A Stim circuit to which the code circuit will be appended. If None, a new circuit is created.
    rounds : int
        Number of measurement rounds. Default is 3.
    pos : Optional[Union[str, List[Tuple[int, int]]]]
        Layout specification for graph embedding. Can be a layout keyword ("random", "spring",
        "bipartite", "tripartite") or a custom list of positions. If None, defaults to "random".
    noise : NoiseModel
        Noise model for circuit operations. Default is a zero-noise model.
    experiment : str
        Experiment type ("z_memory" or "x_memory"). Default is "z_memory".
    logical : Optional[Union[int, List[int]]]
        Index (or indices) of logical operators to use. Default is None (use all).

    Raises
    ------
    ValueError
        If an unrecognized layout or experiment type is provided.
    """

    def __init__(
        self,
        clist1: List[Any],
        clist2: List[Any],
        circuit: Optional[stim.Circuit] = None,
        rounds: int = 3,
        pos: Optional[Union[str, List[Tuple[int, int]]]] = None,
        noise: NoiseModel = NoiseModel(),
        experiment: str = "z_memory",
        logical: Optional[Union[int, List[int]]] = None,
    ) -> None:
        # Convert classical lists to parity-check matrices.
        self.H1: np.ndarray = pcm.to_matrix(clist1)
        self.H2: np.ndarray = pcm.to_matrix(clist2)

        # Compute the hypergraph product matrices.
        HX, HZ = pcm.hypergraph_product(self.H1, self.H2)

        # Determine ordering for check qubits
        z_order = [
            "Q" if s == "B" else "Z"
            for s in clist2
            if not np.issubdtype(type(s), np.number)
        ]
        x_order = [
            "X" if s == "B" else "Q"
            for s in clist2
            if not np.issubdtype(type(s), np.number)
        ]
        self.check_order = np.array(
            [
                z_order if s == "B" else x_order
                for s in clist1
                if not np.issubdtype(type(s), np.number)
            ]
        ).flatten()

        # Store position information
        self.pos = pos

        # Initialize the parent class with computed HX and HZ
        super().__init__(
            HX=HX,
            HZ=HZ,
            circuit=circuit,
            rounds=rounds,
            noise=noise,
            experiment=experiment,
            logical=logical,
        )

        # Recompute qubit indices based on check_order
        self._compute_qubit_indices_from_check_order()

        # Compute node positions and crossings
        self._compute_positions_and_crossings()

        # Build the circuit
        self.construct_code()

    def _compute_qubit_indices_from_check_order(self) -> None:
        """Compute qubit indices based on check_order."""
        # Define qubit indices
        self.qubits = np.arange(sum(self.H1.shape) * sum(self.H2.shape))

        # Assign qubits based on check_order
        self.data_qubits = [
            q for q, t in zip(self.qubits, self.check_order) if t == "Q"
        ]
        self.z_check_qubits = [
            q for q, t in zip(self.qubits, self.check_order) if t == "Z"
        ]
        self.x_check_qubits = [
            q for q, t in zip(self.qubits, self.check_order) if t == "X"
        ]

    def _compute_positions_and_crossings(self) -> None:
        """Compute node positions and identify crossing edges."""
        # Compute node positions using the general layout utility.
        if self.pos is None or isinstance(self.pos, str):
            self.pos = graph.compute_layout(
                self.graph, self.pos or "random", index_key="index"
            )

        # Compute crossings using the utility function
        edges = [
            tuple(self.graph.nodes[node]["index"] for node in edge)
            for edge in self.graph.edges
        ]
        self.crossings = graph.find_edge_crossings(self.pos, edges)

    def construct_code(self) -> None:
        """
        Build the Stim circuit for the code, including stabilizer operations, noise channels,
        measurement rounds, detectors, and observable inclusions.
        """
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

        # --------------------------------------------------

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

        # Logical operator extraction and inclusion
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

    def crossing_number(self) -> int:
        """Return the number of edge crossings in the Tanner graph."""
        return len(self.crossings)

    def reset_data_qubits(self) -> None:
        """Append a reset operation for data qubits to the Stim circuit."""
        self.circuit.append("R", self.data_qubits)

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
        graph.draw(
            self.graph,
            self.pos,
            with_labels=with_labels,
            crossings=crossings,
            connection_rad=connection_rad,
            **kwargs,
        )

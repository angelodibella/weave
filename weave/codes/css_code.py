"""Implementation of CSS (Calderbank-Shor-Steane) quantum error-correcting codes."""

from __future__ import annotations

import numpy as np
import stim
import networkx as nx

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
        X-type parity check matrix (binary, 2D).
    HZ : np.ndarray
        Z-type parity check matrix (binary, 2D).
    rounds : int
        Number of measurement rounds (must be >= 1). Default is 3.
    noise : NoiseModel, optional
        Noise model for circuit operations. Default is a zero-noise model.
    experiment : str
        Experiment type ("z_memory" or "x_memory"). Default is "z_memory".
    logical : int or list of int, optional
        Index (or indices) of logical operators to use. Default is None (use all).

    Raises
    ------
    ValueError
        If HX/HZ are not binary 2D arrays, CSS condition is not met,
        rounds < 1, or experiment is not recognized.
    """

    def __init__(
        self,
        HX: np.ndarray,
        HZ: np.ndarray,
        rounds: int = 3,
        noise: NoiseModel | None = None,
        experiment: str = "z_memory",
        logical: int | list[int] | None = None,
    ) -> None:
        if noise is None:
            noise = NoiseModel()

        # Validate inputs.
        _validate_binary_matrix(HX, "HX")
        _validate_binary_matrix(HZ, "HZ")

        if rounds < 1:
            raise ValueError(f"rounds must be >= 1, got {rounds}.")

        if experiment not in ("z_memory", "x_memory"):
            raise ValueError(
                f"Experiment must be 'z_memory' or 'x_memory', got '{experiment}'."
            )

        # Verify that HX and HZ satisfy the CSS condition.
        if not np.all(np.mod(np.dot(HX, HZ.T), 2) == 0):
            raise ValueError("CSS condition not satisfied.")

        # Calculate logical qubits using exact GF(2) rank.
        k = (
            HZ.shape[1]
            - pcm.row_echelon(HZ)[1]
            - pcm.row_echelon(HX)[1]
        )
        n = HZ.shape[1] + HZ.shape[0] + HX.shape[0]

        # Initialize the parent class.
        super().__init__(n=n, k=k)

        # Code properties.
        self.HX = HX
        self.HZ = HZ
        self.qubits: list[int] = []
        self.data_qubits: list[int] = []
        self.z_check_qubits: list[int] = []
        self.x_check_qubits: list[int] = []
        self.logical = logical

        # Error correction parameters.
        self.experiment = experiment
        self.noise = noise
        self.rounds = rounds
        self._circuit: stim.Circuit | None = None

        # Embedding properties.
        self.pos: list[tuple[float, float]] | None = None
        self.graph: nx.Graph | None = None
        self.crossings: set[frozenset[tuple[int, int]]] = set()

        # Derived properties.
        self._compute_qubit_indices()

    @property
    def circuit(self) -> stim.Circuit:
        """Lazily generate and return the Stim circuit."""
        if self._circuit is None:
            self._circuit = self._generate()
        return self._circuit

    def _generate(self) -> stim.Circuit:
        """
        Build a fresh Stim circuit for the code, including stabilizer operations,
        noise channels, measurement rounds, detectors, and observable inclusions.

        Returns
        -------
        stim.Circuit
            The complete Stim circuit.
        """
        c = stim.Circuit()

        # ---------------- Circuit Head ----------------

        if self.experiment == "z_memory":
            c.append("R", self.qubits)
        elif self.experiment == "x_memory":
            c.append("RX", self.data_qubits)
            c.append("R", self.z_check_qubits + self.x_check_qubits)

        # ---------------- Round Initialization ----------------

        round_circuit = stim.Circuit()

        # Z-check stabilizer operations.
        for target, row in zip(self.z_check_qubits, self.HZ):
            data_idxs = [self.data_qubits[i] for i, v in enumerate(row) if v]
            for dq in data_idxs:
                round_circuit.append("CNOT", [dq, target])
                round_circuit.append("PAULI_CHANNEL_2", [dq, target], self.noise.circuit)

        # X-check stabilizer operations.
        # Uses H-CNOT-...-CNOT-H bracket per check (equivalent to measuring X stabilizers).
        for target, row in zip(self.x_check_qubits, self.HX):
            data_idxs = [self.data_qubits[i] for i, v in enumerate(row) if v]
            round_circuit.append("H", [target])
            for dq in data_idxs:
                round_circuit.append("CNOT", [target, dq])
                round_circuit.append("PAULI_CHANNEL_2", [dq, target], self.noise.circuit)
            round_circuit.append("H", [target])

        # Noise for crossing edges.
        # Each crossing is a frozenset of two edges. A crossing represents two
        # wires that are physically close in the embedding, so the noise should
        # target the physically meaningful qubit pair. For bipartite Tanner graphs
        # (data-check edges), this means the two data-qubit endpoints.
        data_set = set(self.data_qubits)
        for crossing in self.crossings:
            edges = list(crossing)
            e1, e2 = edges[0], edges[1]
            d1, d2 = _crossing_qubit_pair(e1, e2, data_set)
            round_circuit.append(
                "PAULI_CHANNEL_2", [d1, d2], self.noise.crossing
            )

        # Apply single-qubit noise.
        round_circuit.append("PAULI_CHANNEL_1", self.data_qubits, self.noise.data)
        round_circuit.append("PAULI_CHANNEL_1", self.z_check_qubits, self.noise.z_check)
        round_circuit.append("PAULI_CHANNEL_1", self.x_check_qubits, self.noise.x_check)

        # Measure and reset before the rest of stabilizer rounds.
        round_circuit.append("MR", self.z_check_qubits + self.x_check_qubits)

        # Add first round to the circuit.
        c += round_circuit

        # Add initial detectors.
        if self.experiment == "z_memory":
            for k in range(len(self.z_check_qubits)):
                c.append(
                    "DETECTOR", [stim.target_rec(-1 - k - len(self.x_check_qubits))]
                )
        elif self.experiment == "x_memory":
            for k in range(len(self.x_check_qubits)):
                c.append("DETECTOR", [stim.target_rec(-1 - k)])

        # ---------------- Round Operations ----------------

        # Add the detectors for the rest of the rounds on all stabilizers.
        for k in range(len(self.z_check_qubits + self.x_check_qubits)):
            round_circuit.append(
                "DETECTOR",
                [
                    stim.target_rec(-1 - k),
                    stim.target_rec(
                        -1 - k - len(self.z_check_qubits + self.x_check_qubits)
                    ),
                ],
            )

        c += round_circuit * (self.rounds - 1)

        # ---------------- Circuit Tail ----------------

        if self.experiment == "z_memory":
            c.append("M", self.data_qubits)
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
                c.append("DETECTOR", recs)
        elif self.experiment == "x_memory":
            c.append("MX", self.data_qubits)
            for k in range(len(self.x_check_qubits)):
                row = self.HX[-1 - k]
                idxs = [i for i, v in enumerate(row) if v]
                recs = [stim.target_rec(-1 - k - len(self.data_qubits))]
                for idx in idxs:
                    recs.append(stim.target_rec(idx - len(self.data_qubits)))
                c.append("DETECTOR", recs)

        # Logical operator extraction and inclusion.
        x_logicals, z_logicals = self.find_logicals()
        x_logical_qubits = [np.where(log == 1)[0] for log in x_logicals]
        z_logical_qubits = [np.where(log == 1)[0] for log in z_logicals]
        logicals = (
            z_logical_qubits if self.experiment == "z_memory" else x_logical_qubits
        )

        if self.logical is not None:
            if isinstance(self.logical, int):
                self.logical = [self.logical]
            logicals = [logicals[i] for i in self.logical]

        for i, lq in enumerate(logicals):
            recs = [stim.target_rec(q - len(self.data_qubits)) for q in lq]
            c.append("OBSERVABLE_INCLUDE", recs, i)

        return c

    def embed(
        self,
        pos: str | list[tuple[float, float]] | None = None,
        seed: int | None = None,
    ) -> CSSCode:
        """
        Embed the Tanner graph and compute edge crossings.

        Parameters
        ----------
        pos : str or list of position tuples, optional
            Layout specification. "random" is non-deterministic by default
            unless a seed is provided.
        seed : int, optional
            Random seed for reproducible "random" layouts.

        Returns
        -------
        CSSCode
            self, for method chaining.
        """
        self.graph = self._construct_graph()

        # Compute node positions using the general layout utility.
        if pos is None or isinstance(pos, str):
            self.pos = graph.compute_layout(
                self.graph, pos or "random", index_key="index", seed=seed
            )
        else:
            self.pos = pos

        # Compute crossings using the utility function.
        edges = [
            tuple(self.graph.nodes[node]["index"] for node in edge)
            for edge in self.graph.edges
        ]
        self.crossings = graph.find_edge_crossings(self.pos, edges)

        # Invalidate the circuit so it regenerates with the new embedding.
        self._circuit = None

        return self

    def find_logicals(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute logical operators from the parity-check matrices.

        Extracts independent X and Z logical operators and pairs them into
        a symplectic basis where X_Li anticommutes only with Z_Li.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (x_logicals, z_logicals) logical operator matrices in symplectic pairing.
        """

        def _extract_logicals(HX_mat, HZ_mat):
            """Extract independent logical operators from ker(HX) modulo rowspace(HZ)."""
            ker = pcm.nullspace(HX_mat)
            basis = pcm.row_basis(HZ_mat)
            if ker.shape[0] == 0 or basis.shape[0] == 0:
                return np.zeros((0, HX_mat.shape[1]), dtype=int)

            logicals = np.vstack([basis, ker])
            pivots = pcm.row_echelon(logicals.T)[3]
            indices = [
                i for i in range(basis.shape[0], logicals.shape[0]) if i in pivots
            ]
            return logicals[indices]

        x_logicals = _extract_logicals(self.HZ, self.HX)
        z_logicals = _extract_logicals(self.HX, self.HZ)

        # Symplectic Gram-Schmidt: pair X_Li with Z_Li so they anticommute,
        # and all cross-pairs commute. Required for correct OBSERVABLE_INCLUDE
        # assignment when k > 1.
        x_logicals, z_logicals = self._symplectic_gram_schmidt(
            x_logicals, z_logicals
        )

        return x_logicals, z_logicals

    @staticmethod
    def _symplectic_gram_schmidt(
        x_logicals: np.ndarray, z_logicals: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Pair X and Z logical operators into a symplectic basis.

        After this procedure, x[i] anticommutes with z[i] and commutes with z[j]
        for all j != i (and vice versa).

        Parameters
        ----------
        x_logicals : np.ndarray
            X logical operators, shape (k, n).
        z_logicals : np.ndarray
            Z logical operators, shape (k, n).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Symplectically paired (x_logicals, z_logicals).
        """
        k = x_logicals.shape[0]
        if k <= 1:
            return x_logicals, z_logicals

        x = x_logicals.copy()
        z = z_logicals.copy()

        for i in range(k):
            # Find j >= i such that x[i] anticommutes with z[j].
            for j in range(i, k):
                if np.dot(x[i], z[j]) % 2 == 1:
                    if j != i:
                        z[[i, j]] = z[[j, i]]
                    break

            # Make all other operators commute with x[i] and z[i].
            for j in range(k):
                if j == i:
                    continue
                if np.dot(x[j], z[i]) % 2 == 1:
                    x[j] = (x[j] + x[i]) % 2
                if np.dot(z[j], x[i]) % 2 == 1:
                    z[j] = (z[j] + z[i]) % 2

        return x, z

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


def _validate_binary_matrix(matrix: np.ndarray, name: str) -> None:
    """Validate that a matrix is a binary 2D numpy array."""
    if not isinstance(matrix, np.ndarray):
        raise ValueError(f"{name} must be a numpy array.")
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {matrix.ndim}D.")
    if not np.all(np.isin(matrix, [0, 1])):
        raise ValueError(f"{name} must be binary (contain only 0s and 1s).")


def _crossing_qubit_pair(
    e1: tuple[int, int],
    e2: tuple[int, int],
    data_qubits: set[int],
) -> tuple[int, int]:
    """
    Determine the physically meaningful qubit pair for a crossing.

    For a bipartite Tanner graph where edges connect data qubits to check
    qubits, a crossing means two data qubits' wires are near each other.
    Returns the two data-qubit endpoints. If both edges share a data endpoint,
    falls back to check-qubit endpoints.

    Parameters
    ----------
    e1 : tuple of int
        First crossing edge (node_a, node_b).
    e2 : tuple of int
        Second crossing edge (node_a, node_b).
    data_qubits : set of int
        Set of data qubit indices.

    Returns
    -------
    tuple of int
        The pair of qubits that should receive crossing noise.
    """
    d1 = [q for q in e1 if q in data_qubits]
    d2 = [q for q in e2 if q in data_qubits]

    # Normal case: each edge has one data qubit endpoint.
    if len(d1) == 1 and len(d2) == 1 and d1[0] != d2[0]:
        return d1[0], d2[0]

    # Fallback: use the check-qubit endpoints instead.
    c1 = [q for q in e1 if q not in data_qubits]
    c2 = [q for q in e2 if q not in data_qubits]
    if len(c1) == 1 and len(c2) == 1 and c1[0] != c2[0]:
        return c1[0], c2[0]

    # Last resort: return first endpoint of each edge.
    return e1[0], e2[0]

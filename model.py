import numpy as np
import stim


class StabilizerModel:
    def __init__(self, circuit: stim.Circuit = None, rounds: int = 3):
        self.circuit = stim.Circuit() if circuit is None else circuit
        self.rounds = rounds

        self.noise_data: list[float] = None
        self.noise_check: list[float] = None
        self.noise_circuit: list[float] = None

        self.code: str = None
        self.code_params: dict = None

        self.qubits = []
        self.data_qubits = []
        self.check_qubits = []

        # TODO: Implement X and Z error propagation analysis

        """
        TODO: Embedding vs performance: 
            - Hypergraph to graph, from parity check matrix (tripartite graph).
            - Random or specified embedding given a surface in 3-space.
            - Add specified error to crossings -> analyze propagation: some errors are less
              problematic than others!
                - If optimal embedding of seed code in 1D, does that mean anything in the
                  optimal embedding of the quantum code?
                - Analyze classical one-dimensional embeddings of seed codes... Start here!
            - Crossings with longer-ranged connections worse?
            
        
        Extra: How do we CONSTRUCT good embedding?
        """

    # ------------------------------------ Setters and Getters ------------------------------------

    def set_noise_circuit(self, p: float, noise_circuit: list[float] = None) -> None:
        if noise_circuit is None:
            self.noise_circuit = [p / 15 for _ in range(15)]
        else:
            assert (
                len(noise_circuit) == 15
            ), f"Stabilizer measurement noise takes 15 parameters, given {len(noise_circuit)}."
            assert (
                sum(noise_circuit) == p
            ), f"Stabilizer measurement noise has to sum to p = {p}, but sums to {sum(noise_circuit)}"
            self.noise_circuit = noise_circuit

    def set_noise_qubits(self, p: float, noise_main: list[float] = None) -> None:
        if noise_main is None:
            self.noise_data = [p / 3 for _ in range(3)]
            self.noise_check = [p / 3 for _ in range(3)]
        else:
            assert len(noise_main) == 3, f"Qubit noise takes 3 parameters, given {len(noise_main)}."
            assert (
                sum(noise_main) == p
            ), f"Qubit noise has to sum to p = {p}, but sums to {sum(noise_main)}"
            self.noise_data = noise_main
            self.noise_check = noise_main

    def set_noise_data(self, p: float, noise_data: list[float] = None) -> None:
        if noise_data is None:
            self.noise_data = [p / 3 for _ in range(3)]
        else:
            assert (
                len(noise_data) == 3
            ), f"Data qubit noise takes 3 parameters, given {len(noise_data)}."
            assert (
                sum(noise_data) == p
            ), f"Data qubit noise has to sum to p = {p}, but sums to {sum(noise_data)}"
            self.noise_data = noise_data

    def set_noise_check(self, p: float, noise_check: list[float] = None) -> None:
        if noise_check is None:
            self.noise_data = [p / 3 for _ in range(3)]
        else:
            assert (
                len(noise_check) == 3
            ), f"Data qubit noise takes 3 parameters, given {len(noise_check)}."
            assert (
                sum(noise_check) == p
            ), f"Data qubit noise has to sum to p = {p}, but sums to {sum(noise_check)}"
            self.noise_check = noise_check

    # -------------------------------------- Utility Methods --------------------------------------

    def display_samples(self, shots: int = 1) -> None:
        samples = self.circuit.compile_sampler().sample(shots)
        for i, sample in enumerate(samples):
            round_list = []
            for j, outcome in enumerate(sample):
                round_list.append("o" if outcome else "_")

                # Line formatting.
                if ((j + 1) % len(self.check_qubits) == 0) and ((j + 1) != len(sample) - 1):
                    round_list[j] += "\n"
                if (j + 1) == (1 + self.rounds) * len(self.check_qubits):
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
                if (j + 1) % len(self.check_qubits) == 0:
                    round_list[j] += "\n"
                if (j + 1) == self.rounds * len(self.check_qubits):
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

    def print(self):
        print(self.circuit, "\n")

    # TODO: Decode with BP-OSD, and see if this accounts for the FULL error model.

    # ------------------------------------------- Codes -------------------------------------------

    def repetition_code(self, distance: int) -> None:
        self.qubits = np.arange(2 * distance + 1)
        self.data_qubits = self.qubits[::2]
        self.check_qubits = self.qubits[1::2]

        self.code = "repetition_code"
        self.code_params = {"distance": distance}

    # ----------------------------------------- Measure -------------------------------------------

    def measure(self) -> None:
        match self.code:
            case "repetition_code":
                self._repetition_measure(**self.code_params)
            case _:
                raise ValueError("Attempting to measure stabilizers without a code specified.")

    def _repetition_measure(self, distance: int) -> None:
        # We have to add initial dummy measurements for the detector to detect change in the first
        # set of qubit measurements.
        self.circuit.append("M", self.check_qubits)

        circuit = stim.Circuit()

        # Stabilizer measurements.
        for m in self.check_qubits:
            circuit.append("CNOT", [m - 1, m])
            circuit.append("CNOT", [m + 1, m])
            if self.noise_circuit is not None:
                circuit.append("PAULI_CHANNEL_2", [m - 1, m], self.noise_circuit)
                circuit.append("PAULI_CHANNEL_2", [m + 1, m], self.noise_circuit)

        # Apply random errors on qubits.
        if self.noise_data is not None:
            circuit.append("PAULI_CHANNEL_1", self.data_qubits, self.noise_data)
        if self.noise_check is not None:
            circuit.append("PAULI_CHANNEL_1", self.check_qubits, self.noise_check)

        # This measures and resets (to zero) the check qubits.
        circuit.append("MR", self.check_qubits)

        # Compare the last measurement result to the one previous to that of the same qubit.
        for k in range(len(self.check_qubits)):
            circuit.append(
                "DETECTOR", [stim.target_rec(-1 - k), stim.target_rec(-1 - k - distance)]
            )

        # Concatenate the circuits.
        self.circuit += circuit * self.rounds

        # Measure data qubits at the end.
        self.circuit.append("M", self.data_qubits)
        for k in range(len(self.check_qubits)):
            self.circuit.append(
                "DETECTOR",
                [
                    stim.target_rec(-1 - k),
                    stim.target_rec(-2 - k),
                    stim.target_rec(-2 - k - distance),
                ],
            )

        # Measure observable (?)
        self.circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)

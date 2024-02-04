import numpy as np
import stim


class StabilizerModel:
    def __init__(
        self,
        circuit: stim.Circuit = None,
        rounds: int = 1,
        noise_data: list[float] = None,
        noise_check: list[float] = None,
        noise_main: list[float] = None,
        noise_phenom: list[float] = None,
    ):
        self.circuit = stim.Circuit() if circuit is None else circuit

        self.rounds = rounds
        if noise_main is not None:
            self.noise_data = self.noise_check = noise_main
        else:
            self.noise_data = noise_data
            self.noise_check = noise_check
        self.noise_phenom = noise_phenom

    # ------------------------------------ Setters and Getters ------------------------------------

    # -------------------------------------- Utility Methods --------------------------------------

    def display_samples(self, shots: int = 1) -> None:
        sampler = self.circuit.compile_sampler().sample(shots)
        for i, shot in enumerate(sampler, start=1):
            round_list = []
            for j, outcome in enumerate(shot):
                round_list.append("x" if outcome else "_")
                if (j + 1) % (len(shot) / self.rounds) == 0:
                    round_list[j] += "\n"
            print(f"Shot {i}:\n" + "".join(round_list))
        print("\n")

    def print(self):
        print(self.circuit, "\n")

    # ------------------------------------------- Codes -------------------------------------------

    def repetition_code(self, distance: int) -> stim.Circuit:
        """Creates a repetition code, with number of repetition equal to the distance.

        Args:
            distance: Number of data qubits.

        Returns:
            Stabilizer circuit.
        """

        # Define qubits.
        qubits = np.arange(2 * distance + 1)
        data_qubits = qubits[::2]
        check_qubits = qubits[1::2]

        # Apply random errors on qubits BEFORE stabilizer measurements, simulating a noisy channel.
        if self.noise_data is not None:
            self.circuit.append("PAULI_CHANNEL_1", data_qubits, self.noise_data)
        if self.noise_check is not None:
            self.circuit.append("PAULI_CHANNEL_1", check_qubits, self.noise_check)

        # Stabilizer measurements.
        for m in check_qubits:
            self.circuit.append("CNOT", [m - 1, m])
            self.circuit.append("CNOT", [m + 1, m])

            if self.noise_phenom is not None:
                self.circuit.append("PAULI_CHANNEL_2", [m - 1, m], self.noise_phenom)
                self.circuit.append("PAULI_CHANNEL_2", [m + 1, m], self.noise_phenom)

        # This measures and resets (to zero) the check qubits.
        self.circuit.append("MR", check_qubits)

        # Stim supports multiplication of circuits, so we essentially extend what we have so far.
        self.circuit *= self.rounds
        return self.circuit

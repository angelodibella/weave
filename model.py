import numpy as np
import stim


class StabilizerModel:
    def __init__(
        self,
        circuit: stim.Circuit = None,
        rounds: int = 1,
        noise_data: list[float] = [0, 0, 0],
        noise_check: list[float] = [0, 0, 0],
        noise_main: list[float] = None,
    ):
        self.circuit = stim.Circuit() if circuit is None else circuit

        self.rounds = rounds
        if noise_main is not None:
            self.noise_data = self.noise_check = noise_main
        else:
            self.noise_data = noise_data
            self.noise_check = noise_check

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

        # Apply random errors on data qubits BEFORE stabilizer measurements.
        error_types = ["X", "Y", "Z"]
        for error_type, probability in zip(error_types, self.noise_data):
            self.circuit.append(f"{error_type}_ERROR", data_qubits, probability)

        # Apply random errors on measure qubits BEFORE stabilizer measurements.
        for error_type, probability in zip(error_types, self.noise_check):
            self.circuit.append(f"{error_type}_ERROR", check_qubits, probability)

        # Stabilizer measurements.
        for m in check_qubits:
            self.circuit.append("CNOT", [m - 1, m])
            self.circuit.append("CNOT", [m + 1, m])

        # This measures and resets (to zero) the check qubits.
        self.circuit.append("MR", check_qubits)

        # Stim supports multiplication of circuits, so we essentially extend what we have so far.
        return self.circuit * self.rounds

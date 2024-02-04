import stim
import numpy as np


def simple_circuit() -> stim.Circuit:
    """Creates a Bell state.

    Returns:
        Final circuit.
    """
    
    circuit = stim.Circuit()

    # Initialize a Bell pair
    circuit.append("H", [0])
    circuit.append("CNOT", [0, 1])

    # Add random X errors on each qubit independently
    circuit.append("X_ERROR", [0, 1], 0.2)

    # Measure both qubits of the Bell pair in the Z basis
    circuit.append("M", [0, 1])

    return circuit


def repetition_code(distance: int, rounds: int, noise_data: list[float]=[0, 0]) -> stim.Circuit:
    """Creates a repetition code, with number of repetition equal to the distance.

    Args:
        distance: Number of data qubits.
        rounds: Number of rounds to check the stabilizer measurement.
        noise_data: Proportion of X and Z errors on data qubits.

    Returns:
        Stabilizer circuit.
    """

    circuit = stim.Circuit()

    # Define qubits.
    qubits = np.arange(2 * distance + 1)
    data_qubits = qubits[::2]
    measure_qubits = qubits[1::2]
    
    # Random errors on data qubits BEFORE stabilizer measurements.
    circuit.append("X_ERROR", data_qubits, noise_data[0])
    # circuit.append("Z_ERROR", data_qubits, noise_data[1])
    
    # Stabilizer measurements.
    for m in measure_qubits:
        circuit.append("CNOT", [m - 1, m])
        circuit.append("CNOT", [m + 1, m])

    # This measures and resets (to zero) the measure qubits.
    circuit.append("MR", measure_qubits)

    # Stim supports multiplication of circuits, so we essentially extend what we have so far.
    return circuit * rounds

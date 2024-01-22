import stim
import numpy as np

def simple_circuit() -> stim.Circuit:
    circuit = stim.Circuit()

    # Initialize a Bell pair
    circuit.append("H", [0])
    circuit.append("CNOT", [0, 1])

    # Measure both qubits of the Bell pair in the Z basis
    circuit.append("M", [0, 1])

    return circuit

def shor_encode(circuit: stim.Circuit=None) -> stim.Circuit:
    if circuit is None:
        circuit = stim.Circuit()

    

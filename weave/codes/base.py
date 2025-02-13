import numpy as np


class NoiseModel:
    """Class encapsulating noise models for quantum error-correcting codes.
    
    Attributes
    ----------
    data : float or list of float
        Noise level(s) for data qubits. This is the list of probabilities of applying, respectively, 'X', 'Y' and 'Z'
        Pauli gates to data qubits. If a single number is passed, then this value represents the probability of applying
        any of these errors uniformly. Defaults to zero.
    z_check : float or list of float
        Noise level(s) for Z-check qubits. Similar to `data`. Defaults to zero.
    x_check : float or list of float
        Noise level(s) for X-check qubits. Similar to `data`. Defaults to zero.
    circuit : float or list of float
        Noise level(s) for the circuit operations. Respectively, the probability of assigning, to two-qubit gates,
        combination pairs of Pauli I, X, Y and Z operators in the following order:

        [IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ]

        except the operator 'II', which is determined by subtracting the sum of this list of probabilities from 1. If a
        single number is passed, then this value represents the probability of applying any of these non-trivial errors
        uniformly. Defaults to zero.
    crossing : float or list of float
        Noise level(s) for crossing edges of the Tanner graph of the code. Behaves exactly the same as `circuit`, where
        the two-qubit errors are applied on qubits that cross-talk. Defaults to zero.

    Raises
    ------
    AssertionError
        If the provided noise parameter lists do not have the expected lengths.
    """

    def __init__(
            self,
            data: float | list[float] = 0.0,
            z_check: float | list[float] = 0.0,
            x_check: float | list[float] = 0.0,
            circuit: float | list[float] = 0.0,
            crossing: float | list[float] = 0.0,
    ) -> None:
        if np.issubdtype(type(circuit), np.number):
            self.circuit = [circuit / 15 for _ in range(15)]
        else:
            assert (
                    len(circuit) == 15
            ), f"Stabilizer measurement noise takes 15 parameters, given {len(circuit)}."
            self.circuit = circuit

        if np.issubdtype(type(data), np.number):
            self.data = [data / 3 for _ in range(3)]
        else:
            assert (
                    len(data) == 3
            ), f"Data qubit noise takes 3 parameters, given {len(data)}."
            self.data = data

        if np.issubdtype(type(crossing), np.number):
            self.crossing = [crossing / 15 for _ in range(15)]
        else:
            assert (
                    len(crossing) == 15
            ), f"Crossing noise takes 15 parameters, given {len(crossing)}."
            self.crossing = crossing

        if np.issubdtype(type(z_check), np.number):
            self.z_check = [z_check / 3 for _ in range(3)]
        else:
            assert (
                    len(z_check) == 3
            ), f"Z check qubit noise takes 3 parameters, given {len(z_check)}."
            self.z_check = z_check

        if np.issubdtype(type(x_check), np.number):
            self.x_check = [x_check / 3 for _ in range(3)]
        else:
            assert (
                    len(x_check) == 3
            ), f"Data qubit noise takes 3 parameters, given {len(z_check)}."
            self.x_check = x_check

# TODO: Implement a classical code class, and make code classes acquire this instead of clists.
class ClassicalCode:
    def __init__(self):
        pass
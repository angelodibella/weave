import numpy as np


class NoiseModel:
    """
    Noise model for quantum error-correcting codes.

    Parameters
    ----------
    data : float or list of float, optional
        Noise level(s) for data qubits. If a single float is provided, it is uniformly divided among 3 error types.
        Default is 0.0.
    z_check : float or list of float, optional
        Noise level(s) for Z-check qubits. If a single float is provided, it is uniformly divided among 3 error types.
        Default is 0.0.
    x_check : float or list of float, optional
        Noise level(s) for X-check qubits. If a single float is provided, it is uniformly divided among 3 error types.
        Default is 0.0.
    circuit : float or list of float, optional
        Noise level(s) for two-qubit circuit operations. Expected to be 15 values. If a single float is provided,
        it is uniformly divided among 15 values. Default is 0.0.
    crossing : float or list of float, optional
        Noise level(s) for crossing edges (cross-talk) in the Tanner graph. Expected to be 15 values. If a single float
        is provided, it is uniformly divided among 15 values. Default is 0.0.

    Raises
    ------
    AssertionError
        If any noise parameter provided as a list does not have the expected length.
    """
    def __init__(
        self,
        data: float | list[float] = 0.0,
        z_check: float | list[float] = 0.0,
        x_check: float | list[float] = 0.0,
        circuit: float | list[float] = 0.0,
        crossing: float | list[float] = 0.0,
    ) -> None:
        self.circuit = self._process_noise(circuit, expected=15, name="Circuit", divisor=15)
        self.data = self._process_noise(data, expected=3, name="Data qubit", divisor=3)
        self.crossing = self._process_noise(crossing, expected=15, name="Crossing", divisor=15)
        self.z_check = self._process_noise(z_check, expected=3, name="Z-check", divisor=3)
        self.x_check = self._process_noise(x_check, expected=3, name="X-check", divisor=3)

    @staticmethod
    def _process_noise(param: float | list[float], expected: int, name: str, divisor: float) -> list[float]:
        """
        Process a noise parameter by ensuring it is a list of the expected length.

        If a single number is provided, it is uniformly divided by the divisor and repeated.

        Parameters
        ----------
        param : float or list of float
            The noise parameter.
        expected : int
            The expected number of elements.
        name : str
            The name of the noise parameter (for error messages).
        divisor : float
            The divisor used when a single number is provided.

        Returns
        -------
        list of float
            The processed noise parameter.
        """
        if np.issubdtype(type(param), np.number):
            return [param / divisor for _ in range(expected)]
        else:
            assert len(param) == expected, f"{name} noise takes {expected} parameters, given {len(param)}."
            return param


class ClassicalCode:
    """
    Placeholder for a classical code representation.
    """
    def __init__(self):
        pass

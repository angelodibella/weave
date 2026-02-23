"""Base classes and abstractions for quantum error correction codes."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import stim

from ..util import pcm


class NoiseModel:
    """
    Noise model for quantum error-correcting codes.

    Provides a consistent interface for specifying noise parameters across different code types.

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
    ValueError
        If any noise parameter has the wrong length, contains negative values,
        or exceeds Stim's probability constraints (PAULI_CHANNEL_1 params sum > 1,
        PAULI_CHANNEL_2 params sum > 1).
    """

    def __init__(
        self,
        data: float | list[float] = 0.0,
        z_check: float | list[float] = 0.0,
        x_check: float | list[float] = 0.0,
        circuit: float | list[float] = 0.0,
        crossing: float | list[float] = 0.0,
    ) -> None:
        self.circuit = self._process_noise(
            circuit, expected=15, name="Circuit", divisor=15
        )
        self.data = self._process_noise(data, expected=3, name="Data qubit", divisor=3)
        self.crossing = self._process_noise(
            crossing, expected=15, name="Crossing", divisor=15
        )
        self.z_check = self._process_noise(
            z_check, expected=3, name="Z-check", divisor=3
        )
        self.x_check = self._process_noise(
            x_check, expected=3, name="X-check", divisor=3
        )

    @staticmethod
    def _process_noise(
        param: float | list[float], expected: int, name: str, divisor: float
    ) -> tuple[float, ...]:
        """
        Process a noise parameter into an immutable tuple of the expected length.

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
        tuple of float
            The processed noise parameter (immutable).

        Raises
        ------
        ValueError
            If values are negative, wrong length, or sum exceeds 1.
        """
        if isinstance(param, (int, float)):
            if param < 0:
                raise ValueError(
                    f"{name} noise parameter must be non-negative, got {param}."
                )
            values = [param / divisor for _ in range(expected)]
        else:
            if len(param) != expected:
                raise ValueError(
                    f"{name} noise takes {expected} parameters, given {len(param)}."
                )
            for i, v in enumerate(param):
                if v < 0:
                    raise ValueError(
                        f"{name} noise parameter[{i}] must be non-negative, got {v}."
                    )
            values = list(param)

        total = sum(values)
        if total > 1:
            raise ValueError(
                f"{name} noise parameters sum to {total}, which exceeds 1. "
                f"Stim requires PAULI_CHANNEL parameters to sum to <= 1."
            )

        return tuple(values)


class ClassicalCode:
    """
    Base class for classical linear codes.

    Classical codes are represented by their parity-check matrices and provide
    foundation for building quantum codes.

    Parameters
    ----------
    parity_check_matrix : np.ndarray
        The parity-check matrix H defining the code.
    """

    def __init__(self, parity_check_matrix: np.ndarray) -> None:
        self.H = parity_check_matrix
        self._validate_matrix()

    def _validate_matrix(self) -> None:
        """Validate that the parity check matrix is binary."""
        if not np.all(np.logical_or(self.H == 0, self.H == 1)):
            raise ValueError(
                "Parity check matrix must be binary (contain only 0s and 1s)"
            )

    @property
    def n(self) -> int:
        """The block length of the code."""
        return self.H.shape[1]

    @property
    def k(self) -> int:
        """The dimension of the code."""
        return self.n - pcm.row_echelon(self.H)[1]

    @property
    def m(self) -> int:
        """The number of parity checks."""
        return self.H.shape[0]


class QuantumCode(ABC):
    """
    Abstract base class for quantum error-correcting codes.

    This class defines a common interface for all quantum error-correcting code implementations.

    Parameters
    ----------
    n : int
        The number of physical qubits in the code.
    k : int
        The number of logical qubits encoded.
    """

    def __init__(self, n: int, k: int) -> None:
        self.n = n
        self.k = k

    @abstractmethod
    def _generate(self) -> stim.Circuit:
        """
        Generate a circuit to measure the syndrome of the code.

        Returns
        -------
        stim.Circuit
            A circuit representation for the syndrome measurement.
        """
        pass

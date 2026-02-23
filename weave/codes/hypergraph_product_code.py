"""Implementation of hypergraph product codes."""

import numpy as np
from typing import Optional, Union, List, Tuple

from .base import NoiseModel
from .css_code import CSSCode
from ..util import pcm


class HypergraphProductCode(CSSCode):
    """
    Hypergraph product code based on two classical parity-check matrices.

    Given H1 (r1 x n1) and H2 (r2 x n2), computes the hypergraph product
    to obtain CSS parity-check matrices HX and HZ, then builds the
    corresponding Stim circuit for simulation.

    Parameters
    ----------
    H1 : np.ndarray
        Parity-check matrix of the first classical code, shape (r1, n1).
    H2 : np.ndarray
        Parity-check matrix of the second classical code, shape (r2, n2).
    rounds : int
        Number of measurement rounds. Default is 3.
    noise : NoiseModel, optional
        Noise model for circuit operations. Default is a zero-noise model.
    experiment : str
        Experiment type ("z_memory" or "x_memory"). Default is "z_memory".
    logical : int or list of int, optional
        Index (or indices) of logical operators to use. Default is None (use all).
    """

    def __init__(
        self,
        H1: np.ndarray,
        H2: np.ndarray,
        rounds: int = 3,
        noise: Optional[NoiseModel] = None,
        experiment: str = "z_memory",
        logical: Optional[Union[int, List[int]]] = None,
    ) -> None:
        self.H1: np.ndarray = H1
        self.H2: np.ndarray = H2

        # Compute the hypergraph product matrices.
        HX, HZ = pcm.hypergraph_product(self.H1, self.H2)

        super().__init__(
            HX=HX,
            HZ=HZ,
            rounds=rounds,
            noise=noise,
            experiment=experiment,
            logical=logical,
        )

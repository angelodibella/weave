"""Implementation of hypergraph product codes."""

import networkx as nx
import numpy as np
import stim
from typing import Optional, Union, List, Tuple, Any

from .base import NoiseModel
from .css_code import CSSCode
from ..util import pcm, graph


class HypergraphProductCode(CSSCode):
    """
    Hypergraph product code based on two classical codes.

    Constructs the code from classical list representations (clist1 and clist2),
    generates the corresponding Tanner graph, and builds the Stim circuit for simulation.

    Parameters
    ----------
    clist1 : list
        Classical list representation of the first code.
    clist2 : list
        Classical list representation of the second code.
    circuit : Optional[stim.Circuit]
        A Stim circuit to which the code circuit will be appended. If None, a new circuit is created.
    rounds : int
        Number of measurement rounds. Default is 3.
    pos : Optional[Union[str, List[Tuple[int, int]]]]
        Layout specification for graph embedding. Can be a layout keyword ("random", "spring",
        "bipartite", "tripartite") or a custom list of positions. If None, defaults to "random".
    noise : NoiseModel
        Noise model for circuit operations. Default is a zero-noise model.
    experiment : str
        Experiment type ("z_memory" or "x_memory"). Default is "z_memory".
    logical : Optional[Union[int, List[int]]]
        Index (or indices) of logical operators to use. Default is None (use all).

    Raises
    ------
    ValueError
        If an unrecognized layout or experiment type is provided.
    """

    def __init__(
        self,
        H1: np.ndarray,
        H2: np.ndarray,
        rounds: int = 3,
        noise: NoiseModel = NoiseModel(),
        experiment: str = "z_memory",
        logical: Optional[Union[int, List[int]]] = None,
    ) -> None:
        # Convert classical lists to parity-check matrices.
        self.H1: np.ndarray = H1
        self.H2: np.ndarray = H2

        # Compute the hypergraph product matrices.
        HX, HZ = pcm.hypergraph_product(self.H1, self.H2)

        # Initialize the parent class with computed HX and HZ.
        super().__init__(
            HX=HX,
            HZ=HZ,
            rounds=rounds,
            noise=noise,
            experiment=experiment,
            logical=logical,
        )

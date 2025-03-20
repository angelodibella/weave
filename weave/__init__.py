"""
Weave: A quantum error correction framework
"""

# Import NoiseModel directly from _core.codes.
from ._core.codes import NoiseModel

# Import utility submodules to expose at the top level.
from ._core.util import pcm, graph

__all__ = [
    # C++ components at top level.
    "NoiseModel",
    
    # Utility submodules.
    "pcm",
    "graph",
    
    # Pure Python components (to be added)
]

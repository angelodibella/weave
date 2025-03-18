"""
Python interface to Weave quantum error correction codes
"""

import numpy as np
from typing import List, Union, Optional, Tuple

from ._core import (
    HypergraphProductCode,
    StabilizerCode,
)

class SurfaceCode:
    """
    Surface code implementation built on top of the core C++ functionality.
    This demonstrates how to create Python components that build on the C++ core.
    """
    
    def __init__(self, distance: int):
        """
        Initialize a surface code with the given distance.
        
        Args:
            distance: Code distance (must be odd)
        """
        if distance % 2 == 0:
            raise ValueError("Surface code distance must be odd")
        
        self.distance = distance
        self._code = HypergraphProductCode()
        self._initialize()
        
    def _initialize(self):
        """Initialize the surface code structure."""
        # In a real implementation, this would construct the appropriate
        # parity check matrices for a surface code of the given distance
        size = self.distance
        
        # This is just a simplified implementation
        x_checks = np.zeros((size*size//2, size*size), dtype=int)
        z_checks = np.zeros((size*size//2, size*size), dtype=int)
        
        # Set up a toy example
        for i in range(size*size//2):
            for j in range(4):  # 4 qubits per check
                x_checks[i, (i + j) % (size*size)] = 1
                z_checks[i, (i + j*2) % (size*size)] = 1
                
        # Convert to list format for the C++ interface
        self._code.generate(x_checks.tolist(), z_checks.tolist())
        
    @property
    def parameters(self) -> Tuple[int, int, int]:
        """
        Get the code parameters [n, k, d].
        
        Returns:
            Tuple containing (n, k, d) - number of physical qubits, 
            number of logical qubits, and code distance
        """
        params = self._code.get_parameters()
        return tuple(params)
    
    @property
    def stabilizers(self) -> List[str]:
        """
        Get the stabilizer generators.
        
        Returns:
            List of stabilizer strings in the format "IXZIY..." 
        """
        return self._code.get_stabilizers()
    
    def logical_operators(self) -> List[str]:
        """
        Get the logical Pauli operators.
        
        Returns:
            List of logical operator strings
        """
        # In a real implementation, we would compute these based on the code structure
        # This is just a placeholder
        n = self.parameters[0]
        return [
            "X" * n,  # X logical operator
            "Z" * n,  # Z logical operator
        ]
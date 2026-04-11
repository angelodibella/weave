"""Thin re-export shim for proximity kernels.

The canonical home for `Kernel`, `CrossingKernel`,
`RegularizedPowerLawKernel`, and `ExponentialKernel` is
:mod:`weave.ir.kernel`. They live there because they are IR objects —
immutable, JSON-round-trippable, schema-versioned.

This module re-exports them so that the ergonomic
``from weave.geometry import CrossingKernel`` import keeps working.
Prefer importing from :mod:`weave.ir` directly in new code:

.. code-block:: python

    from weave.ir import CrossingKernel, RegularizedPowerLawKernel
"""

from ..ir.kernel import (
    CrossingKernel,
    ExponentialKernel,
    Kernel,
    RegularizedPowerLawKernel,
)

__all__ = [
    "CrossingKernel",
    "ExponentialKernel",
    "Kernel",
    "RegularizedPowerLawKernel",
]

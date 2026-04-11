"""Schedule-aware compiler for CSS extraction circuits.

The weave compiler consumes a code, an embedding, a schedule, a
kernel, and a noise configuration, and produces a
:class:`~weave.ir.CompiledExtraction` artifact bundle in canonical
pure-data form (circuit text + DEM text + input specs + fingerprint).

PR 5 ships the local-noise-only path. PR 7 adds the propagation /
residual-error analyzer. PR 8 extends `compile_extraction` with the
geometry pass that walks routed polylines, applies the kernel, and
emits `CORRELATED_ERROR` channels with provenance. PR 9 adds the
correlation graph, exposure metrics, and decoder artifact to the
output bundle.
"""

from .compile import compile_extraction

__all__ = ["compile_extraction"]

r"""`DecoderArtifact` — pure-data bundle and decoder adapters.

Decoders that consume weave output (PyMatching for matching codes,
`stimbposd` / BP-OSD for general CSS codes) need a compact
description of the retained pair channel in a form that is decoupled
from the full Stim DEM. This module holds that description as a
frozen, JSON-round-trippable dataclass and provides adapter methods
that turn the artifact into ready-to-use decoder instances:

- :meth:`DecoderArtifact.to_bposd_decoder` — returns a
  ``stimbposd.BPOSD`` instance configured against the compiled DEM.
- :meth:`DecoderArtifact.to_pymatching` — returns a
  ``pymatching.Matching`` instance built from the compiled DEM.
- :meth:`DecoderArtifact.to_pair_prior_dict` — returns a simple
  ``{(qubit_a, qubit_b): probability}`` dict for downstream
  consumers that want the raw pair-edge weights.

Both decoders accept the non-decomposed DEM that
:func:`~weave.compiler.compile_extraction` produces when geometry
noise is active (``decompose_errors=False``):

- **stimbposd** (BP+OSD) handles arbitrary-weight error mechanisms
  natively; no decomposition needed.
- **PyMatching** v2.2+ accepts non-decomposed DEMs by converting
  hyperedge error mechanisms into equivalent matching-graph edges
  via the method of Higgott & Gidney (2023, arXiv:2309.15354).

References
----------
- Di Bella, *Geometry-induced correlated noise in qLDPC syndrome
  extraction* (PRX Quantum, under review, 2026). §V, decoder
  augmentation with retained pair priors.
- Roffe, White, Burton, Campbell, *Decoding across the quantum
  low-density parity-check code landscape*, Phys. Rev. Research 2,
  043423 (2020), arXiv:2005.07016. BP+OSD reference.
- Higgott, Gidney, *Sparse Blossom: correcting a million errors
  per core second with minimum-weight matching*, arXiv:2303.15933
  (2023). PyMatching v2 reference.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from .compiled import ProvenanceRecord

__all__ = [
    "DecoderArtifact",
    "build_decoder_artifact",
]


@dataclass(frozen=True)
class DecoderArtifact:
    """Pure-data bundle handed to correlation-aware decoders.

    Parameters
    ----------
    pair_edges : tuple[tuple[int, int, float], ...]
        Sorted list of `(qubit_a, qubit_b, weight)` triples. Each
        entry is the union-bound pair probability that the retained
        channel assigns to the data pair `(qubit_a, qubit_b)`,
        summed across sectors. A decoder that wants sector-specific
        priors should consult the full
        :class:`~weave.ir.CorrelationEdgeRecord` table on the
        enclosing :class:`~weave.ir.CompiledExtraction`.
    single_prior : tuple[float, ...]
        Per-data-qubit single-error prior, one float per data qubit
        index. Typically zero when only geometry-induced noise is
        active; populated by the local-noise channels in later PRs.
    decoder_hint : str
        Free-form tag describing the intended decoding strategy.
        Consumed by the PR 17 adapters (`"pymatching_correlated"`,
        `"bposd_augmented"`, etc.). Empty string is the default.

    Notes
    -----
    The class is frozen and hashable. Its `to_json` / `from_json`
    methods are fully round-trippable and participate in the
    :meth:`~weave.ir.CompiledExtraction.fingerprint` SHA256.
    """

    SCHEMA_VERSION: ClassVar[int] = 1

    pair_edges: tuple[tuple[int, int, float], ...] = ()
    single_prior: tuple[float, ...] = ()
    decoder_hint: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.pair_edges, tuple):
            object.__setattr__(self, "pair_edges", tuple(self.pair_edges))
        if not isinstance(self.single_prior, tuple):
            object.__setattr__(self, "single_prior", tuple(self.single_prior))

        # Validate pair edges.
        for edge in self.pair_edges:
            if len(edge) != 3:
                raise ValueError(f"pair_edges entry must be (i, j, w), got {edge!r}")
            i, j, w = edge
            if i == j:
                raise ValueError(f"pair_edges entry has equal qubits: {edge!r}")
            if w < 0:
                raise ValueError(f"pair_edges entry has negative weight: {edge!r}")

        # Single prior: nonnegative reals.
        for i, p in enumerate(self.single_prior):
            if not 0.0 <= float(p) <= 1.0:
                raise ValueError(f"single_prior[{i}] must be in [0, 1], got {p}")

    @property
    def schema_version(self) -> int:
        return self.SCHEMA_VERSION

    @property
    def num_pair_edges(self) -> int:
        """Convenience: number of distinct pair edges."""
        return len(self.pair_edges)

    # ------------------------------------------------------------------
    # Decoder adapters (PR 17)
    # ------------------------------------------------------------------

    def to_pair_prior_dict(self) -> dict[tuple[int, int], float]:
        """Return the pair-edge weights as a simple dict.

        Returns
        -------
        dict[tuple[int, int], float]
            Maps `(qubit_a, qubit_b)` → `pair_probability` for every
            weight-2 pair edge in the artifact. Empty dict when no
            pair edges exist.

        Examples
        --------
        >>> artifact.to_pair_prior_dict()
        {(0, 2): 0.061, (1, 5): 0.033}
        """
        return {(int(a), int(b)): float(w) for a, b, w in self.pair_edges}

    def to_bposd_decoder(
        self,
        dem: Any,
        *,
        max_bp_iters: int = 30,
        osd_order: int = 60,
        bp_method: str = "product_sum",
        osd_method: str = "osd_cs",
        **bposd_kwargs: Any,
    ) -> Any:
        """Build a ``stimbposd.BPOSD`` decoder from the compiled DEM.

        The compiled DEM already contains the ``CORRELATED_ERROR``
        mechanisms injected by the geometry pass (in non-decomposed
        form), so BP+OSD sees the full retained-channel noise model
        without any additional augmentation. This method is a thin
        convenience wrapper that imports ``stimbposd`` and
        instantiates the decoder with the provided hyperparameters.

        Parameters
        ----------
        dem : stim.DetectorErrorModel
            The DEM from :attr:`CompiledExtraction.dem`. Must be the
            non-decomposed form emitted by ``compile_extraction``
            when ``provenance`` is non-empty.
        max_bp_iters, osd_order, bp_method, osd_method :
            BP+OSD hyperparameters; forwarded to
            ``stimbposd.BPOSD.__init__``.
        **bposd_kwargs :
            Additional keyword arguments for ``stimbposd.BPOSD``.

        Returns
        -------
        stimbposd.BPOSD
            A ready-to-use decoder instance. Call
            ``.decode(syndrome_vector)`` or ``.decode_batch(shots)``
            to decode.

        Raises
        ------
        ImportError
            If ``stimbposd`` is not installed.
        """
        import stimbposd

        return stimbposd.BPOSD(
            dem,
            max_bp_iters=max_bp_iters,
            osd_order=osd_order,
            bp_method=bp_method,
            osd_method=osd_method,
            **bposd_kwargs,
        )

    def to_pymatching(self, dem: Any) -> Any:
        """Build a ``pymatching.Matching`` from the compiled DEM.

        PyMatching v2.2+ accepts non-decomposed DEMs and internally
        converts hyperedge error mechanisms into matching-graph
        edges. The returned ``Matching`` object is ready for
        ``.decode(syndrome_vector)`` or ``.decode_batch(shots)``.

        Parameters
        ----------
        dem : stim.DetectorErrorModel
            The DEM from :attr:`CompiledExtraction.dem`.

        Returns
        -------
        pymatching.Matching
            A ready-to-use MWPM decoder instance.

        Raises
        ------
        ImportError
            If ``pymatching`` is not installed.
        """
        import pymatching

        return pymatching.Matching.from_detector_error_model(dem)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "type": "decoder_artifact",
            "pair_edges": [[int(i), int(j), float(w)] for i, j, w in self.pair_edges],
            "single_prior": [float(p) for p in self.single_prior],
            "decoder_hint": self.decoder_hint,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> DecoderArtifact:
        if data.get("type") != "decoder_artifact":
            raise ValueError(f"Expected type='decoder_artifact', got {data.get('type')!r}.")
        version = data.get("schema_version")
        if version != cls.SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported DecoderArtifact schema_version {version}; "
                f"expected {cls.SCHEMA_VERSION}."
            )
        return cls(
            pair_edges=tuple(
                (int(e[0]), int(e[1]), float(e[2])) for e in data.get("pair_edges", [])
            ),
            single_prior=tuple(float(p) for p in data.get("single_prior", [])),
            decoder_hint=str(data.get("decoder_hint", "")),
        )


# =============================================================================
# Builder
# =============================================================================


def build_decoder_artifact(
    provenance: Sequence[ProvenanceRecord],
    *,
    num_data_qubits: int,
    decoder_hint: str = "",
) -> DecoderArtifact:
    """Aggregate weight-2 pair events into a canonical decoder bundle.

    For every weight-2 record, contribute its `pair_probability` to
    the `(qubit_a, qubit_b)` bin (union-bound sum across sectors).
    The result is sorted canonically by `(qubit_a, qubit_b)`.

    `single_prior` is initialised to all zeros of length
    `num_data_qubits` — PR 9's geometry-only scope does not populate
    it. Later PRs that carry local-noise priors into the decoder
    artifact will override this field.

    Parameters
    ----------
    provenance : Sequence[ProvenanceRecord]
        Provenance from :func:`~weave.compiler.compile_extraction`.
    num_data_qubits : int
        Length of the `single_prior` vector; typically
        `len(code.data_qubits)`.
    decoder_hint : str, optional
        Free-form strategy tag. Defaults to the empty string.

    Returns
    -------
    DecoderArtifact
    """
    pair_accum: dict[tuple[int, int], float] = {}
    for rec in provenance:
        if rec.data_weight != 2:
            continue
        qa, qb = rec.data_support
        key = (int(qa), int(qb))
        pair_accum[key] = pair_accum.get(key, 0.0) + float(rec.pair_probability)
    pair_edges = tuple((a, b, w) for (a, b), w in sorted(pair_accum.items()))
    single_prior = (0.0,) * int(num_data_qubits)
    return DecoderArtifact(
        pair_edges=pair_edges,
        single_prior=single_prior,
        decoder_hint=decoder_hint,
    )

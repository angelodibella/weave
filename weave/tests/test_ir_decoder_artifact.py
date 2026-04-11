"""Tests for `weave.ir.decoder_artifact.DecoderArtifact` and its builder.

PR 9 ships the shell form of `DecoderArtifact` — a sorted pair-edge
table, a per-data-qubit single prior, and a free-form `decoder_hint`.
PR 17 will add the `to_pymatching_hint()` / `to_bposd_dem()` adapter
methods; those are out of scope here.

The builder's semantics are pinned against small synthetic
provenance lists so the aggregation logic is unambiguous.
"""

from __future__ import annotations

import pytest

from weave.ir import DecoderArtifact, ProvenanceRecord, build_decoder_artifact


def _record(
    *,
    tick: int,
    pair_prob: float,
    data_support: tuple[int, ...],
    sector: str = "X",
) -> ProvenanceRecord:
    return ProvenanceRecord(
        tick_index=tick,
        edge_a=(0, 4),
        edge_b=(2, 5),
        sector=sector,
        routed_distance=1.0,
        pair_probability=pair_prob,
        data_support=data_support,
        data_pauli_symbols=tuple("X" for _ in data_support),
    )


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------


class TestDecoderArtifactConstruction:
    def test_defaults(self):
        artifact = DecoderArtifact()
        assert artifact.pair_edges == ()
        assert artifact.single_prior == ()
        assert artifact.decoder_hint == ""
        assert artifact.num_pair_edges == 0

    def test_populated(self):
        artifact = DecoderArtifact(
            pair_edges=((0, 1, 0.1), (1, 3, 0.2)),
            single_prior=(0.001, 0.002, 0.0),
            decoder_hint="bposd_augmented",
        )
        assert artifact.pair_edges == ((0, 1, 0.1), (1, 3, 0.2))
        assert artifact.single_prior == (0.001, 0.002, 0.0)
        assert artifact.decoder_hint == "bposd_augmented"
        assert artifact.num_pair_edges == 2

    def test_equal_qubits_rejected(self):
        with pytest.raises(ValueError, match="equal qubits"):
            DecoderArtifact(pair_edges=((3, 3, 0.1),))

    def test_negative_weight_rejected(self):
        with pytest.raises(ValueError, match="negative weight"):
            DecoderArtifact(pair_edges=((0, 1, -0.01),))

    def test_malformed_pair_entry_rejected(self):
        with pytest.raises(ValueError, match=r"\(i, j, w\)"):
            DecoderArtifact(pair_edges=((0, 1),))  # type: ignore[arg-type]

    def test_out_of_range_prior_rejected(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            DecoderArtifact(single_prior=(1.5,))


# ---------------------------------------------------------------------------
# build_decoder_artifact
# ---------------------------------------------------------------------------


class TestBuildDecoderArtifact:
    def test_empty_provenance(self):
        artifact = build_decoder_artifact([], num_data_qubits=4)
        assert artifact.pair_edges == ()
        assert artifact.single_prior == (0.0, 0.0, 0.0, 0.0)

    def test_single_weight_2_record(self):
        rec = _record(tick=0, pair_prob=0.1, data_support=(0, 2))
        artifact = build_decoder_artifact([rec], num_data_qubits=4)
        assert artifact.pair_edges == ((0, 2, 0.1),)

    def test_records_across_sectors_aggregate(self):
        """`DecoderArtifact.pair_edges` is sector-merged: an event on
        `(0, 2)` in the X sector and another on `(0, 2)` in the Z sector
        contribute to the same pair edge weight."""
        recs = [
            _record(tick=0, pair_prob=0.1, data_support=(0, 2), sector="X"),
            _record(tick=0, pair_prob=0.2, data_support=(0, 2), sector="Z"),
        ]
        artifact = build_decoder_artifact(recs, num_data_qubits=4)
        assert artifact.pair_edges == ((0, 2, pytest.approx(0.3, abs=1e-12)),)

    def test_non_weight_2_records_skipped(self):
        recs = [
            _record(tick=0, pair_prob=0.1, data_support=(0, 2)),
            _record(tick=0, pair_prob=0.3, data_support=(0, 1, 2)),  # weight 3
            _record(tick=0, pair_prob=0.05, data_support=()),  # weight 0
        ]
        artifact = build_decoder_artifact(recs, num_data_qubits=4)
        assert artifact.pair_edges == ((0, 2, 0.1),)

    def test_sorted_output(self):
        recs = [
            _record(tick=0, pair_prob=0.1, data_support=(3, 5)),
            _record(tick=0, pair_prob=0.2, data_support=(0, 1)),
            _record(tick=0, pair_prob=0.15, data_support=(0, 5)),
        ]
        artifact = build_decoder_artifact(recs, num_data_qubits=6)
        assert [e[:2] for e in artifact.pair_edges] == [(0, 1), (0, 5), (3, 5)]

    def test_decoder_hint_passthrough(self):
        artifact = build_decoder_artifact(
            [], num_data_qubits=1, decoder_hint="pymatching_correlated"
        )
        assert artifact.decoder_hint == "pymatching_correlated"


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


class TestDecoderArtifactRoundTrip:
    def test_empty_round_trip(self):
        artifact = DecoderArtifact()
        assert DecoderArtifact.from_json(artifact.to_json()) == artifact

    def test_populated_round_trip(self):
        artifact = DecoderArtifact(
            pair_edges=((0, 2, 0.1), (1, 3, 0.2)),
            single_prior=(0.001, 0.0, 0.002, 0.0),
            decoder_hint="bposd_augmented",
        )
        assert DecoderArtifact.from_json(artifact.to_json()) == artifact

    def test_rejects_wrong_type(self):
        with pytest.raises(ValueError, match="type"):
            DecoderArtifact.from_json({"type": "wrong", "schema_version": 1})

    def test_rejects_wrong_schema_version(self):
        with pytest.raises(ValueError, match="schema_version"):
            DecoderArtifact.from_json({"type": "decoder_artifact", "schema_version": 999})

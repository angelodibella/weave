"""Tests for `weave.ir.compiled.CompiledExtraction`."""

from __future__ import annotations

import json

import pytest

from weave.ir import CompiledExtraction


def _minimal_spec() -> dict[str, dict]:
    """Minimal valid spec dicts for constructing a CompiledExtraction."""
    return {
        "embedding_spec": {"schema_version": 1, "type": "straight_line", "positions": []},
        "schedule_spec": {
            "schema_version": 1,
            "type": "schedule",
            "name": "empty",
            "head_steps": [],
            "cycle_steps": [],
            "tail_steps": [],
            "qubits": [],
            "qubit_roles": [],
        },
        "kernel_spec": {"schema_version": 1, "type": "crossing", "params": {}},
        "route_metric_spec": {
            "schema_version": 1,
            "type": "min_distance",
            "params": {},
        },
        "local_noise_spec": {
            "schema_version": 1,
            "type": "local_noise",
            "p_cnot": 0.0,
            "p_idle": 0.0,
            "p_prep": 0.0,
            "p_meas": 0.0,
        },
        "geometry_noise_spec": {
            "schema_version": 1,
            "type": "geometry_noise",
            "J0": 0.0,
            "tau": 1.0,
            "use_weak_limit": False,
            "geometry_scope": "full_cycle",
        },
    }


def _minimal_compiled() -> CompiledExtraction:
    return CompiledExtraction(
        circuit_text="",
        dem_text="",
        code_fingerprint="a" * 64,
        **_minimal_spec(),
    )


class TestCompiledExtractionBasics:
    def test_construction(self):
        ce = _minimal_compiled()
        assert ce.circuit_text == ""
        assert ce.dem_text == ""
        assert ce.code_fingerprint == "a" * 64

    def test_schema_version(self):
        ce = _minimal_compiled()
        assert ce.schema_version == 1
        assert CompiledExtraction.SCHEMA_VERSION == 1

    def test_equality(self):
        a = _minimal_compiled()
        b = _minimal_compiled()
        assert a == b

    def test_inequality_on_circuit_text(self):
        a = _minimal_compiled()
        b = CompiledExtraction(
            circuit_text="H 0\n",
            dem_text="",
            code_fingerprint="a" * 64,
            **_minimal_spec(),
        )
        assert a != b

    def test_cache_is_excluded_from_equality(self):
        """Two instances with the same fields should compare equal even if
        one has its lazy cache populated."""
        a = _minimal_compiled()
        b = CompiledExtraction(
            circuit_text="",
            dem_text="",
            code_fingerprint="a" * 64,
            **_minimal_spec(),
        )
        # Don't actually call circuit/dem (empty text isn't parsable).
        # Just verify the cache field is present and mutable.
        a._cache["fake"] = "value"
        assert a == b  # still equal — _cache is excluded from compare


class TestCompiledExtractionLazyMaterializers:
    def _nontrivial(self) -> CompiledExtraction:
        """A minimally parseable circuit text."""
        return CompiledExtraction(
            circuit_text="H 0\nM 0\n",
            dem_text="",
            code_fingerprint="a" * 64,
            **_minimal_spec(),
        )

    def test_circuit_property_returns_stim_circuit(self):
        import stim

        ce = self._nontrivial()
        assert isinstance(ce.circuit, stim.Circuit)
        assert len(ce.circuit) > 0

    def test_circuit_property_is_cached(self):
        """Accessing circuit twice returns the same object (identity)."""
        ce = self._nontrivial()
        c1 = ce.circuit
        c2 = ce.circuit
        assert c1 is c2

    def test_circuit_text_roundtrip(self):
        """Parsing circuit_text and restringifying gives the same text."""
        ce = self._nontrivial()
        import stim

        parsed = stim.Circuit(ce.circuit_text)
        assert (
            str(parsed) == ce.circuit_text.rstrip("\n") + "\n"
            or str(parsed) + "\n" == ce.circuit_text
        )


class TestCompiledExtractionFingerprint:
    def test_fingerprint_is_sha256(self):
        ce = _minimal_compiled()
        fp = ce.fingerprint()
        assert len(fp) == 64  # SHA256 hex
        assert all(c in "0123456789abcdef" for c in fp)

    def test_fingerprint_stable(self):
        """Same inputs → same fingerprint across calls."""
        a = _minimal_compiled()
        b = _minimal_compiled()
        assert a.fingerprint() == b.fingerprint()

    def test_fingerprint_changes_with_circuit_text(self):
        a = _minimal_compiled()
        b = CompiledExtraction(
            circuit_text="H 0\n",
            dem_text="",
            code_fingerprint="a" * 64,
            **_minimal_spec(),
        )
        assert a.fingerprint() != b.fingerprint()

    def test_fingerprint_changes_with_code(self):
        a = _minimal_compiled()
        b = CompiledExtraction(
            circuit_text="",
            dem_text="",
            code_fingerprint="b" * 64,
            **_minimal_spec(),
        )
        assert a.fingerprint() != b.fingerprint()


class TestCompiledExtractionJson:
    def test_roundtrip(self):
        ce = _minimal_compiled()
        restored = CompiledExtraction.from_json(ce.to_json())
        assert restored == ce

    def test_roundtrip_through_json_dumps(self):
        ce = _minimal_compiled()
        data = json.loads(json.dumps(ce.to_json()))
        restored = CompiledExtraction.from_json(data)
        assert restored == ce

    def test_to_json_has_type_discriminator(self):
        data = _minimal_compiled().to_json()
        assert data["type"] == "compiled_extraction"
        assert data["schema_version"] == 1

    def test_from_json_rejects_wrong_type(self):
        with pytest.raises(ValueError, match="type='compiled_extraction'"):
            CompiledExtraction.from_json({"type": "other", "schema_version": 1})

    def test_from_json_rejects_wrong_schema_version(self):
        data = _minimal_compiled().to_json()
        data["schema_version"] = 999
        with pytest.raises(ValueError, match="schema_version"):
            CompiledExtraction.from_json(data)

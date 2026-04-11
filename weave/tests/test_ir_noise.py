"""Tests for `weave.ir.noise`: `LocalNoiseConfig` and `GeometryNoiseConfig`.

Covers validation (the PR 3 acceptance test: "LocalNoiseConfig rejects
negative probabilities with ValueError"), default construction, JSON
round-trip with schema versioning, and the `GeometryNoiseConfig.enabled`
convenience flag.
"""

from __future__ import annotations

import json

import pytest

from weave.ir import GeometryNoiseConfig, LocalNoiseConfig

# =============================================================================
# LocalNoiseConfig
# =============================================================================


class TestLocalNoiseConfigDefaults:
    def test_default_is_zero_noise(self):
        cfg = LocalNoiseConfig()
        assert cfg.p_cnot == 0.0
        assert cfg.p_idle == 0.0
        assert cfg.p_prep == 0.0
        assert cfg.p_meas == 0.0

    def test_schema_version(self):
        assert LocalNoiseConfig.SCHEMA_VERSION == 1
        assert LocalNoiseConfig().schema_version == 1

    def test_explicit_construction(self):
        cfg = LocalNoiseConfig(p_cnot=0.001, p_idle=0.002, p_prep=0.003, p_meas=0.004)
        assert cfg.p_cnot == 0.001
        assert cfg.p_idle == 0.002
        assert cfg.p_prep == 0.003
        assert cfg.p_meas == 0.004

    def test_frozen(self):
        """Frozen dataclass: field reassignment raises FrozenInstanceError."""
        cfg = LocalNoiseConfig(p_cnot=0.001)
        from dataclasses import FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            cfg.p_cnot = 0.5  # type: ignore[misc]

    def test_equality_and_hashability(self):
        a = LocalNoiseConfig(p_cnot=0.001)
        b = LocalNoiseConfig(p_cnot=0.001)
        assert a == b
        assert hash(a) == hash(b)


class TestLocalNoiseConfigValidation:
    """The PR 3 acceptance test: reject negative probabilities with ValueError."""

    @pytest.mark.parametrize("field", ["p_cnot", "p_idle", "p_prep", "p_meas"])
    def test_rejects_negative(self, field):
        with pytest.raises(ValueError, match=f"{field} must be in"):
            LocalNoiseConfig(**{field: -0.1})

    @pytest.mark.parametrize("field", ["p_cnot", "p_idle", "p_prep", "p_meas"])
    def test_rejects_above_one(self, field):
        with pytest.raises(ValueError, match=f"{field} must be in"):
            LocalNoiseConfig(**{field: 1.5})

    def test_rejects_slightly_above_one(self):
        with pytest.raises(ValueError):
            LocalNoiseConfig(p_cnot=1.0000001)

    def test_accepts_boundary_zero(self):
        LocalNoiseConfig(p_cnot=0.0, p_idle=0.0, p_prep=0.0, p_meas=0.0)

    def test_accepts_boundary_one(self):
        """p = 1.0 is physically degenerate but a valid probability."""
        LocalNoiseConfig(p_cnot=1.0, p_idle=1.0, p_prep=1.0, p_meas=1.0)

    def test_rejects_negative_idle_specifically(self):
        """Common use case: p_idle is the new idle-noise knob closed by PR 5."""
        with pytest.raises(ValueError, match="p_idle"):
            LocalNoiseConfig(p_idle=-1e-6)


class TestLocalNoiseConfigJson:
    def test_roundtrip(self):
        cfg = LocalNoiseConfig(p_cnot=0.001, p_idle=0.002, p_prep=0.003, p_meas=0.004)
        restored = LocalNoiseConfig.from_json(cfg.to_json())
        assert restored == cfg

    def test_roundtrip_defaults(self):
        cfg = LocalNoiseConfig()
        restored = LocalNoiseConfig.from_json(cfg.to_json())
        assert restored == cfg

    def test_roundtrip_through_json_dumps(self):
        cfg = LocalNoiseConfig(p_cnot=0.001, p_prep=0.002)
        data = json.loads(json.dumps(cfg.to_json()))
        restored = LocalNoiseConfig.from_json(data)
        assert restored == cfg

    def test_to_json_keys(self):
        data = LocalNoiseConfig(p_cnot=0.1).to_json()
        assert data["schema_version"] == 1
        assert data["type"] == "local_noise"
        assert data["p_cnot"] == 0.1
        assert "p_idle" in data
        assert "p_prep" in data
        assert "p_meas" in data

    def test_from_json_rejects_wrong_type(self):
        with pytest.raises(ValueError, match="type='local_noise'"):
            LocalNoiseConfig.from_json({"schema_version": 1, "type": "geometry_noise"})

    def test_from_json_rejects_wrong_schema_version(self):
        with pytest.raises(ValueError, match="schema_version"):
            LocalNoiseConfig.from_json({"schema_version": 999, "type": "local_noise"})

    def test_from_json_missing_fields_use_defaults(self):
        """Unknown or missing probability fields default to 0."""
        cfg = LocalNoiseConfig.from_json(
            {"schema_version": 1, "type": "local_noise", "p_cnot": 0.5}
        )
        assert cfg.p_cnot == 0.5
        assert cfg.p_idle == 0.0
        assert cfg.p_prep == 0.0
        assert cfg.p_meas == 0.0

    def test_from_json_propagates_validation(self):
        with pytest.raises(ValueError, match="p_cnot must be in"):
            LocalNoiseConfig.from_json({"schema_version": 1, "type": "local_noise", "p_cnot": -0.1})


# =============================================================================
# GeometryNoiseConfig
# =============================================================================


class TestGeometryNoiseConfigDefaults:
    def test_default_is_disabled(self):
        cfg = GeometryNoiseConfig()
        assert cfg.J0 == 0.0
        assert cfg.tau == 1.0
        assert cfg.use_weak_limit is False
        assert cfg.geometry_scope == "full_cycle"
        assert cfg.enabled is False

    def test_schema_version(self):
        assert GeometryNoiseConfig.SCHEMA_VERSION == 1
        assert GeometryNoiseConfig().schema_version == 1

    def test_explicit_construction(self):
        cfg = GeometryNoiseConfig(
            J0=0.08,
            tau=1.5,
            use_weak_limit=True,
            geometry_scope="theory_reduced",
        )
        assert cfg.J0 == 0.08
        assert cfg.tau == 1.5
        assert cfg.use_weak_limit is True
        assert cfg.geometry_scope == "theory_reduced"


class TestGeometryNoiseConfigEnabled:
    def test_enabled_false_when_J0_zero(self):
        assert GeometryNoiseConfig(J0=0.0).enabled is False

    def test_enabled_true_when_J0_positive(self):
        assert GeometryNoiseConfig(J0=0.01).enabled is True

    def test_enabled_true_for_tiny_positive_J0(self):
        assert GeometryNoiseConfig(J0=1e-12).enabled is True


class TestGeometryNoiseConfigValidation:
    def test_rejects_negative_J0(self):
        with pytest.raises(ValueError, match="J0 must be non-negative"):
            GeometryNoiseConfig(J0=-0.01)

    def test_accepts_zero_J0(self):
        """J0 = 0 disables geometry noise but is a valid config."""
        GeometryNoiseConfig(J0=0.0)

    def test_rejects_zero_tau(self):
        with pytest.raises(ValueError, match="tau must be positive"):
            GeometryNoiseConfig(tau=0.0)

    def test_rejects_negative_tau(self):
        with pytest.raises(ValueError, match="tau must be positive"):
            GeometryNoiseConfig(tau=-1.0)

    def test_rejects_invalid_geometry_scope(self):
        with pytest.raises(ValueError, match="geometry_scope must be"):
            GeometryNoiseConfig(geometry_scope="nonsense")  # type: ignore[arg-type]

    def test_accepts_theory_reduced(self):
        cfg = GeometryNoiseConfig(geometry_scope="theory_reduced")
        assert cfg.geometry_scope == "theory_reduced"

    def test_accepts_full_cycle(self):
        cfg = GeometryNoiseConfig(geometry_scope="full_cycle")
        assert cfg.geometry_scope == "full_cycle"


class TestGeometryNoiseConfigJson:
    def test_roundtrip(self):
        cfg = GeometryNoiseConfig(
            J0=0.04, tau=1.0, use_weak_limit=True, geometry_scope="theory_reduced"
        )
        restored = GeometryNoiseConfig.from_json(cfg.to_json())
        assert restored == cfg

    def test_roundtrip_defaults(self):
        cfg = GeometryNoiseConfig()
        restored = GeometryNoiseConfig.from_json(cfg.to_json())
        assert restored == cfg

    def test_roundtrip_through_json_dumps(self):
        cfg = GeometryNoiseConfig(J0=0.08, tau=1.5)
        data = json.loads(json.dumps(cfg.to_json()))
        restored = GeometryNoiseConfig.from_json(data)
        assert restored == cfg

    def test_to_json_keys(self):
        data = GeometryNoiseConfig(J0=0.04).to_json()
        assert data["schema_version"] == 1
        assert data["type"] == "geometry_noise"
        assert data["J0"] == 0.04
        assert "tau" in data
        assert "use_weak_limit" in data
        assert "geometry_scope" in data

    def test_from_json_rejects_wrong_type(self):
        with pytest.raises(ValueError, match="type='geometry_noise'"):
            GeometryNoiseConfig.from_json({"schema_version": 1, "type": "local_noise"})

    def test_from_json_rejects_wrong_schema_version(self):
        with pytest.raises(ValueError, match="schema_version"):
            GeometryNoiseConfig.from_json({"schema_version": 999, "type": "geometry_noise"})

    def test_from_json_rejects_invalid_scope(self):
        with pytest.raises(ValueError, match="geometry_scope"):
            GeometryNoiseConfig.from_json(
                {
                    "schema_version": 1,
                    "type": "geometry_noise",
                    "geometry_scope": "bogus",
                }
            )

    def test_from_json_defaults_missing_fields(self):
        cfg = GeometryNoiseConfig.from_json(
            {"schema_version": 1, "type": "geometry_noise", "J0": 0.08}
        )
        assert cfg.J0 == 0.08
        assert cfg.tau == 1.0
        assert cfg.use_weak_limit is False
        assert cfg.geometry_scope == "full_cycle"

    def test_from_json_propagates_validation(self):
        with pytest.raises(ValueError, match="J0 must be non-negative"):
            GeometryNoiseConfig.from_json(
                {"schema_version": 1, "type": "geometry_noise", "J0": -0.01}
            )

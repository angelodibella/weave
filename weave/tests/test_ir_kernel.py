"""Tests for the IR-home `Kernel` protocol and JSON round-trip.

Mathematical behavior (κ values, validation) is tested in
`test_geometry.py`; this file focuses on the new PR 3 concerns: JSON
serialization, schema versioning, polymorphic dispatch via
`load_kernel`, and protocol satisfaction through the IR import path.
"""

from __future__ import annotations

import json

import pytest

from weave.ir import (
    CrossingKernel,
    ExponentialKernel,
    Kernel,
    RegularizedPowerLawKernel,
    load_kernel,
)

ALL_KERNELS: list[Kernel] = [
    CrossingKernel(),
    RegularizedPowerLawKernel(alpha=3.0, r0=1.0),
    RegularizedPowerLawKernel(alpha=1.5, r0=0.5),
    ExponentialKernel(xi=1.0),
    ExponentialKernel(xi=2.5),
]


# =============================================================================
# to_json / from_json round-trip (the PR 3 flagship acceptance test)
# =============================================================================


class TestKernelJsonRoundtrip:
    """Plan §2 PR 3 acceptance test: `Kernel.from_json(k.to_json()) == k`
    for every kernel type."""

    @pytest.mark.parametrize("kernel", ALL_KERNELS)
    def test_roundtrip_equality(self, kernel):
        """The flagship: k.to_json → from_json reconstructs an equal object."""
        data = kernel.to_json()
        restored = type(kernel).from_json(data)
        assert restored == kernel

    @pytest.mark.parametrize("kernel", ALL_KERNELS)
    def test_roundtrip_evaluates_identically(self, kernel):
        """Behavioral check: the restored kernel evaluates to the same κ(d)."""
        restored = type(kernel).from_json(kernel.to_json())
        for d in [0.0, 0.5, 1.0, 2.0, 5.0]:
            assert restored(d) == kernel(d)

    @pytest.mark.parametrize("kernel", ALL_KERNELS)
    def test_roundtrip_through_json_dumps(self, kernel):
        """The emitted dict survives `json.dumps` / `json.loads` unchanged."""
        data = kernel.to_json()
        restored = type(kernel).from_json(json.loads(json.dumps(data)))
        assert restored == kernel

    @pytest.mark.parametrize("kernel", ALL_KERNELS)
    def test_to_json_has_required_keys(self, kernel):
        data = kernel.to_json()
        assert "schema_version" in data
        assert "type" in data
        assert "params" in data
        assert data["schema_version"] == 1
        assert data["type"] == kernel.name
        assert data["params"] == kernel.params


# =============================================================================
# Individual from_json behavior
# =============================================================================


class TestCrossingKernelFromJson:
    def test_basic(self):
        k = CrossingKernel.from_json({"schema_version": 1, "type": "crossing", "params": {}})
        assert k == CrossingKernel()

    def test_rejects_wrong_type(self):
        with pytest.raises(ValueError, match="type='crossing'"):
            CrossingKernel.from_json({"schema_version": 1, "type": "exponential", "params": {}})

    def test_rejects_wrong_schema_version(self):
        with pytest.raises(ValueError, match="schema_version"):
            CrossingKernel.from_json({"schema_version": 999, "type": "crossing", "params": {}})


class TestPowerLawFromJson:
    def test_basic(self):
        k = RegularizedPowerLawKernel.from_json(
            {
                "schema_version": 1,
                "type": "regularized_power_law",
                "params": {"alpha": 3.0, "r0": 1.0},
            }
        )
        assert k == RegularizedPowerLawKernel(alpha=3.0, r0=1.0)

    def test_rejects_wrong_type(self):
        with pytest.raises(ValueError, match="type='regularized_power_law'"):
            RegularizedPowerLawKernel.from_json(
                {
                    "schema_version": 1,
                    "type": "crossing",
                    "params": {"alpha": 3.0, "r0": 1.0},
                }
            )

    def test_rejects_missing_alpha(self):
        with pytest.raises(ValueError, match="'alpha'"):
            RegularizedPowerLawKernel.from_json(
                {
                    "schema_version": 1,
                    "type": "regularized_power_law",
                    "params": {"r0": 1.0},
                }
            )

    def test_rejects_missing_r0(self):
        with pytest.raises(ValueError, match="'r0'"):
            RegularizedPowerLawKernel.from_json(
                {
                    "schema_version": 1,
                    "type": "regularized_power_law",
                    "params": {"alpha": 3.0},
                }
            )

    def test_propagates_validation_from_post_init(self):
        """A negative alpha in JSON is rejected by __post_init__."""
        with pytest.raises(ValueError, match="alpha must be positive"):
            RegularizedPowerLawKernel.from_json(
                {
                    "schema_version": 1,
                    "type": "regularized_power_law",
                    "params": {"alpha": -1.0, "r0": 1.0},
                }
            )


class TestExponentialFromJson:
    def test_basic(self):
        k = ExponentialKernel.from_json(
            {
                "schema_version": 1,
                "type": "exponential",
                "params": {"xi": 2.0},
            }
        )
        assert k == ExponentialKernel(xi=2.0)

    def test_rejects_missing_xi(self):
        with pytest.raises(ValueError, match="'xi'"):
            ExponentialKernel.from_json({"schema_version": 1, "type": "exponential", "params": {}})

    def test_propagates_validation_from_post_init(self):
        with pytest.raises(ValueError, match="xi must be positive"):
            ExponentialKernel.from_json(
                {
                    "schema_version": 1,
                    "type": "exponential",
                    "params": {"xi": -1.0},
                }
            )


# =============================================================================
# load_kernel dispatch
# =============================================================================


class TestLoadKernel:
    @pytest.mark.parametrize("kernel", ALL_KERNELS)
    def test_dispatches_to_each_type(self, kernel):
        loaded = load_kernel(kernel.to_json())
        assert isinstance(loaded, type(kernel))
        assert loaded == kernel

    def test_rejects_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown kernel type"):
            load_kernel({"type": "mystery_kernel", "schema_version": 1, "params": {}})

    def test_rejects_missing_type(self):
        with pytest.raises(ValueError, match="Unknown kernel type"):
            load_kernel({"schema_version": 1, "params": {}})

    def test_roundtrip_via_load(self):
        """load_kernel(k.to_json()) is the polymorphic version of from_json."""
        k = RegularizedPowerLawKernel(alpha=2.0, r0=1.5)
        loaded = load_kernel(k.to_json())
        assert loaded == k


# =============================================================================
# Protocol satisfaction via the IR import path
# =============================================================================


class TestKernelProtocolIR:
    """Re-verify protocol satisfaction now that `Kernel` lives in `weave.ir`."""

    def test_crossing_satisfies_protocol(self):
        assert isinstance(CrossingKernel(), Kernel)

    def test_power_law_satisfies_protocol(self):
        assert isinstance(RegularizedPowerLawKernel(alpha=3, r0=1), Kernel)

    def test_exponential_satisfies_protocol(self):
        assert isinstance(ExponentialKernel(xi=1.0), Kernel)

    def test_has_to_json(self):
        """The new PR 3 protocol addition: `to_json` is part of Kernel."""
        for k in ALL_KERNELS:
            assert callable(k.to_json)

    def test_shim_reexports_same_classes(self):
        """`weave.geometry` re-exports resolve to the exact IR classes."""
        from weave.geometry import CrossingKernel as GeoCrossingKernel
        from weave.geometry import ExponentialKernel as GeoExponentialKernel
        from weave.geometry import Kernel as GeoKernel
        from weave.geometry import RegularizedPowerLawKernel as GeoPowerLawKernel

        assert GeoKernel is Kernel
        assert GeoCrossingKernel is CrossingKernel
        assert GeoPowerLawKernel is RegularizedPowerLawKernel
        assert GeoExponentialKernel is ExponentialKernel

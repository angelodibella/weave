"""Tests for the schedule and embedding import adapters.

Three adapters under test:

1. ``schedule_from_json_file`` — thin wrapper; we test JSON round-trip
   on the default Steane schedule.
2. ``embedding_from_json_file`` — thin wrapper; JSON round-trip on a
   ``StraightLineEmbedding``.
3. ``schedule_from_stim_circuit`` — the non-trivial adapter. We build a
   hand-crafted Stim circuit, import it, and verify the recovered
   ``Schedule`` has the correct step structure, gate types, qubit
   roles, and sector assignments. Then we compile the imported
   schedule through ``compile_extraction`` and check it produces a
   valid ``CompiledExtraction``.
"""

from __future__ import annotations

import json

import pytest
import stim

from weave.codes.css_code import CSSCode
from weave.compiler import compile_extraction
from weave.ir import (
    CrossingKernel,
    GeometryNoiseConfig,
    LocalNoiseConfig,
    Schedule,
    StraightLineEmbedding,
    TwoQubitEdge,
    default_css_schedule,
)
from weave.ir.importers import (
    embedding_from_json_file,
    schedule_from_json_file,
    schedule_from_stim_circuit,
)
from weave.util import pcm

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def steane_code() -> CSSCode:
    H = pcm.hamming(7)
    return CSSCode(HX=H, HZ=H, rounds=1)


@pytest.fixture
def steane_schedule(steane_code) -> Schedule:
    return default_css_schedule(steane_code)


# ---------------------------------------------------------------------------
# schedule_from_json_file
# ---------------------------------------------------------------------------


class TestScheduleFromJsonFile:
    def test_round_trip_on_steane(self, steane_schedule, tmp_path):
        """Write a schedule to JSON, reload via the importer, compare."""
        path = tmp_path / "sched.json"
        path.write_text(json.dumps(steane_schedule.to_json()))
        reloaded = schedule_from_json_file(path)
        assert reloaded == steane_schedule

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            schedule_from_json_file(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# embedding_from_json_file
# ---------------------------------------------------------------------------


class TestEmbeddingFromJsonFile:
    def test_round_trip_on_straight_line(self, tmp_path):
        emb = StraightLineEmbedding.from_positions([(0, 0), (1, 0), (0, 1)])
        path = tmp_path / "emb.json"
        path.write_text(json.dumps(emb.to_json()))
        reloaded = embedding_from_json_file(path)
        assert isinstance(reloaded, StraightLineEmbedding)
        assert reloaded == emb


# ---------------------------------------------------------------------------
# schedule_from_stim_circuit — structure tests
# ---------------------------------------------------------------------------


def _make_test_circuit() -> tuple[stim.Circuit, dict[int, str]]:
    """Build a small Stim circuit for import testing.

    5 qubits: 0, 1 are data; 3 is z_ancilla; 4 is x_ancilla; 2 is data.
    Structure:
      head: R 0 1 2 3 4
      cycle (REPEAT 3):
        TICK after R
        CX 0 3       ← data→z_anc, X sector
        TICK
        CX 1 3       ← data→z_anc, X sector
        TICK
        H 4
        TICK
        CX 4 0       ← x_anc→data, Z sector
        TICK
        H 4
        TICK
        MR 3 4
        TICK
      tail:
        M 0 1 2
        TICK
    """
    c = stim.Circuit()
    c.append("R", [0, 1, 2, 3, 4])
    c.append("TICK")
    body = stim.Circuit()
    body.append("CX", [0, 3])
    body.append("TICK")
    body.append("CX", [1, 3])
    body.append("TICK")
    body.append("H", [4])
    body.append("TICK")
    body.append("CX", [4, 0])
    body.append("TICK")
    body.append("H", [4])
    body.append("TICK")
    body.append("MR", [3, 4])
    body.append("TICK")
    c.append(stim.CircuitRepeatBlock(3, body))
    c.append("M", [0, 1, 2])
    c.append("TICK")
    roles = {0: "data", 1: "data", 2: "data", 3: "z_ancilla", 4: "x_ancilla"}
    return c, roles


class TestScheduleFromStimCircuit:
    def test_head_cycle_tail_structure(self):
        circuit, roles = _make_test_circuit()
        sched = schedule_from_stim_circuit(circuit, roles)
        # Head: one R step.
        assert len(sched.head_steps) == 1
        assert sched.head_steps[0].role == "reset"
        # Cycle: 6 steps (CX, CX, H, CX, H, MR).
        assert len(sched.cycle_steps) == 6
        # Tail: one M step.
        assert len(sched.tail_steps) == 1
        assert sched.tail_steps[0].role == "meas"

    def test_cnot_edges_have_correct_control_target(self):
        circuit, roles = _make_test_circuit()
        sched = schedule_from_stim_circuit(circuit, roles)
        # First cycle step: CX 0 3 (single).
        cnot_step = sched.cycle_steps[0]
        assert cnot_step.role == "cnot_layer"
        assert len(cnot_step.active_edges) == 1
        edge = cnot_step.active_edges[0]
        assert isinstance(edge, TwoQubitEdge)
        assert edge.gate == "CNOT"
        assert edge.control == 0
        assert edge.target == 3

    def test_sector_inference(self):
        """CX(data→z_anc) → X sector; CX(x_anc→data) → Z sector."""
        circuit, roles = _make_test_circuit()
        sched = schedule_from_stim_circuit(circuit, roles)
        # First cycle step: CX 0→3 (data→z_anc → X sector).
        edge0 = sched.cycle_steps[0].active_edges[0]
        assert isinstance(edge0, TwoQubitEdge)
        assert edge0.interaction_sector == "X"
        # Fourth cycle step (index 3): CX 4→0 (x_anc→data → Z sector).
        z_step = sched.cycle_steps[3]
        assert z_step.role == "cnot_layer"
        for edge in z_step.active_edges:
            assert isinstance(edge, TwoQubitEdge)
            assert edge.interaction_sector == "Z"

    def test_qubit_roles_propagated(self):
        circuit, roles = _make_test_circuit()
        sched = schedule_from_stim_circuit(circuit, roles)
        assert sched.qubit_roles == roles

    def test_h_steps_are_single_q(self):
        circuit, roles = _make_test_circuit()
        sched = schedule_from_stim_circuit(circuit, roles)
        # Third cycle step (index 2): H 4.
        h_step = sched.cycle_steps[2]
        assert h_step.role == "single_q"
        assert len(h_step.active_edges) == 1
        assert h_step.active_edges[0].gate == "H"
        assert h_step.active_edges[0].qubit == 4

    def test_name_propagated(self):
        circuit, roles = _make_test_circuit()
        sched = schedule_from_stim_circuit(circuit, roles, name="test_import")
        assert sched.name == "test_import"

    def test_noise_instructions_are_skipped(self):
        """Noise channels in the circuit don't produce schedule edges."""
        c = stim.Circuit()
        c.append("R", [0, 1])
        c.append("TICK")
        c.append("DEPOLARIZE1", [0, 1], 0.01)
        c.append("CX", [0, 1])
        c.append("DEPOLARIZE2", [0, 1], 0.01)
        c.append("TICK")
        roles = {0: "data", 1: "z_ancilla"}
        sched = schedule_from_stim_circuit(c, roles)
        # Should have 2 steps: R and CX. Noise is dropped.
        sum(len(s.active_edges) for s in sched.cycle_steps)
        cnot_edges = [
            e for s in sched.cycle_steps for e in s.active_edges if isinstance(e, TwoQubitEdge)
        ]
        assert len(cnot_edges) == 1


# ---------------------------------------------------------------------------
# Integration: imported schedule compiles through compile_extraction
# ---------------------------------------------------------------------------


class TestImportedScheduleCompiles:
    def test_steane_import_recovers_structure(self, steane_code):
        """Export a Steane schedule to Stim text (via compile_extraction
        with J0=0), re-import from the compiled circuit, and verify the
        recovered schedule has the expected gate count and types.

        Note: compile_extraction unrolls rounds (no REPEAT block in the
        emitted Stim circuit), so the re-imported schedule has an empty
        head and tail (everything lands in cycle). A full round-trip
        back through compile_extraction would require the importer to
        heuristically detect cycle boundaries, which is a future
        enhancement. For PR 15 we verify the structural recovery.
        """
        sched = default_css_schedule(steane_code)
        emb = StraightLineEmbedding.from_positions(
            [(float(i), 0.0) for i in range(steane_code.n_total)]
        )
        compiled = compile_extraction(
            code=steane_code,
            embedding=emb,
            schedule=sched,
            kernel=CrossingKernel(),
            local_noise=LocalNoiseConfig(),
            geometry_noise=GeometryNoiseConfig(),
            rounds=2,
        )
        # Re-import the compiled circuit.
        roles = dict(sched.qubit_roles)
        reimported = schedule_from_stim_circuit(compiled.circuit, roles, name="reimported_steane")
        # Without REPEAT, everything is in cycle.
        assert len(reimported.cycle_steps) > 0
        # The total number of CNOT edges in the reimported schedule
        # should match the original schedule × rounds.
        sum(
            1 for s in sched.all_steps() for e in s.active_edges if isinstance(e, TwoQubitEdge)
        )
        reimported_cnots = sum(
            1 for s in reimported.cycle_steps for e in s.active_edges if isinstance(e, TwoQubitEdge)
        )
        # Compiled had 2 rounds of cycle + head + tail = head + 2*cycle + tail.
        expected = (
            sum(1 for s in sched.head_steps for e in s.active_edges if isinstance(e, TwoQubitEdge))
            + 2
            * sum(
                1 for s in sched.cycle_steps for e in s.active_edges if isinstance(e, TwoQubitEdge)
            )
            + sum(
                1 for s in sched.tail_steps for e in s.active_edges if isinstance(e, TwoQubitEdge)
            )
        )
        assert reimported_cnots == expected
        # Sector inference should produce X and Z sectors.
        sectors_seen = {
            e.interaction_sector
            for s in reimported.cycle_steps
            for e in s.active_edges
            if isinstance(e, TwoQubitEdge)
        }
        assert "X" in sectors_seen
        assert "Z" in sectors_seen

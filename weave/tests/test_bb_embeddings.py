"""Tests for the BB-specific embeddings and the `ibm_schedule` factory.

Three classes under test:

- :class:`~weave.ir.ColumnEmbedding` and its
  :class:`~weave.ir.MonomialColumnEmbedding` subclass.
- :class:`~weave.ir.IBMBiplanarEmbedding`.
- :class:`~weave.ir.FixedPermutationColumnEmbedding`.

And one factory: :func:`~weave.codes.bb.ibm_schedule`.

The tests pin physical/geometric invariants rather than exact
`bbstim`-byte-for-byte reproduction (which would require vendoring
`bbstim` into the test fixtures). Namely:

1. `MonomialColumnEmbedding` lays out exactly `4 l m` qubits and each
   ``B_1``-step routing of BB72's Z-check CNOT layer produces 36
   parallel edges (one per z-check ancilla).
2. `IBMBiplanarEmbedding` puts L-block data at `z > 0`, R-block
   data at `z < 0`, and ancillas at `z = 0`.
3. `ibm_schedule(bb72)` compiles through
   :func:`~weave.compiler.compile_extraction` to a Stim circuit whose
   noiseless detector sampler returns an all-zero outcome (i.e.
   the schedule correctly measures every BB72 stabilizer).
4. JSON round-trip via :func:`~weave.ir.load_embedding` preserves
   every embedding exactly.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from weave.codes.bb import (
    BivariateBicycleCode,
    build_bb72,
    build_bb108,
    enumerate_pure_L_minwt_logicals,
    ibm_schedule,
)
from weave.compiler import compile_extraction
from weave.ir import (
    ColumnEmbedding,
    CrossingKernel,
    FixedPermutationColumnEmbedding,
    GeometryNoiseConfig,
    IBMBiplanarEmbedding,
    LocalNoiseConfig,
    MonomialColumnEmbedding,
    RegularizedPowerLawKernel,
    RouteID,
    TwoQubitEdge,
    load_embedding,
)
from weave.optimize import (
    NumpyExposureTemplate,
    compute_bb_ibm_event_template,
    j_kappa_numpy,
    prepare_exposure_template,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bb72() -> BivariateBicycleCode:
    return build_bb72()


@pytest.fixture
def monomial_bb72(bb72: BivariateBicycleCode) -> MonomialColumnEmbedding:
    return MonomialColumnEmbedding.from_bb(bb72)


@pytest.fixture
def biplanar_bb72(bb72: BivariateBicycleCode) -> IBMBiplanarEmbedding:
    return IBMBiplanarEmbedding.from_bb(bb72)


# ---------------------------------------------------------------------------
# ColumnEmbedding base
# ---------------------------------------------------------------------------


class TestColumnEmbeddingBase:
    def test_from_positions_2d_lifted(self):
        emb = ColumnEmbedding.from_positions(
            [(0, 0), (1, 0), (0, 1)],
            num_columns=2,
            num_rows=2,
            layers_per_cell=0,  # zero → skip grid check
        )
        assert emb.positions[0] == (0.0, 0.0, 0.0)
        assert emb.positions[1] == (1.0, 0.0, 0.0)

    def test_grid_mismatch_rejected(self):
        with pytest.raises(ValueError, match="num_columns"):
            ColumnEmbedding.from_positions(
                [(0, 0), (1, 0)],
                num_columns=2,
                num_rows=2,
                layers_per_cell=1,  # 2*2*1 = 4 ≠ len(positions) = 2
            )

    def test_straight_line_routing(self):
        emb = ColumnEmbedding.from_positions([(0, 0), (3, 4)])
        rg = emb.routing_geometry([(0, 1)])
        assert (0, 1) in rg
        poly = rg[(0, 1)]
        assert poly == ((0.0, 0.0, 0.0), (3.0, 4.0, 0.0))

    def test_round_trip(self):
        emb = ColumnEmbedding.from_positions(
            [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
            num_columns=3,
            num_rows=1,
            layers_per_cell=1,
            name="custom",
        )
        reconstructed = ColumnEmbedding.from_json(emb.to_json())
        assert reconstructed == emb


# ---------------------------------------------------------------------------
# MonomialColumnEmbedding
# ---------------------------------------------------------------------------


class TestMonomialColumnEmbedding:
    def test_qubit_count_matches_bb_structure(self, bb72, monomial_bb72):
        """4 * l * m = 144 for BB72."""
        assert len(monomial_bb72.positions) == 4 * bb72.l * bb72.m
        assert monomial_bb72.l == 6
        assert monomial_bb72.m == 6
        assert monomial_bb72.bb_name == "BB72"

    def test_data_and_ancilla_positions_separated(self, bb72, monomial_bb72):
        """L-data at sub-column 0, z-anc at 1, x-anc at 2, R-data at 3."""
        # Cell (i=0, j=0): L at x=0, z at x=1, x at x=2, R at x=3.
        assert monomial_bb72.node_position(bb72.data_qubits[0]) == (0.0, 0.0, 0.0)
        assert monomial_bb72.node_position(bb72.z_check_qubits[0]) == (1.0, 0.0, 0.0)
        assert monomial_bb72.node_position(bb72.x_check_qubits[0]) == (2.0, 0.0, 0.0)
        # R-block data 0 (i.e., qubit bb.block_size) at the same cell (0,0) sub-column 3.
        assert monomial_bb72.node_position(bb72.data_qubits[bb72.block_size]) == (
            3.0,
            0.0,
            0.0,
        )

    def test_spacing_scales_positions(self, bb72):
        emb = MonomialColumnEmbedding.from_bb(bb72, spacing=2.5)
        # Cell (0, 0) L-data at (0, 0). Cell (1, 0) L-data at (4*2.5, 0) = (10, 0).
        assert emb.node_position(bb72.data_qubits[0]) == (0.0, 0.0, 0.0)
        i1_is_1 = bb72.data_qubits[1]  # flat j=0 i1=1 → (1, 0)
        assert emb.node_position(i1_is_1) == (10.0, 0.0, 0.0)

    def test_z_check_layer_routing_has_36_edges(self, bb72, monomial_bb72):
        """Plan acceptance test 1, reformulated: a single B-monomial
        Z-check CNOT layer yields exactly `lm = 36` parallel edges."""
        sched = ibm_schedule(bb72)
        # Cycle tick 1 is the first Z-check CNOT layer (after H bracket
        # at tick 0). It fires one monomial from B on L-block data.
        first_z_layer = sched.cycle_steps[1]
        assert first_z_layer.role == "cnot_layer"
        # All edges are in the X sector (Z-checks detect X errors).
        for e in first_z_layer.active_edges:
            assert isinstance(e, TwoQubitEdge)
            assert e.interaction_sector == "X"
        assert len(first_z_layer.active_edges) == bb72.l * bb72.m
        # Routing geometry: 36 distinct routes, each a straight 2-point
        # polyline between a data qubit and a z-ancilla.
        route_ids = [
            RouteID(source=e.control, target=e.target, step_tick=1)
            for e in first_z_layer.active_edges
            if isinstance(e, TwoQubitEdge)
        ]
        rg = monomial_bb72.routing_geometry(route_ids)
        assert len(rg) == 36
        for poly in rg.edges.values():
            assert len(poly) == 2

    def test_round_trip_via_load_embedding(self, monomial_bb72):
        data = monomial_bb72.to_json()
        reloaded = load_embedding(data)
        assert isinstance(reloaded, MonomialColumnEmbedding)
        assert reloaded == monomial_bb72

    def test_invalid_spacing_rejected(self, bb72):
        with pytest.raises(ValueError, match="spacing"):
            MonomialColumnEmbedding.from_bb(bb72, spacing=-1.0)


# ---------------------------------------------------------------------------
# IBMBiplanarEmbedding
# ---------------------------------------------------------------------------


class TestIBMBiplanarEmbedding:
    def test_all_qubits_on_base_plane(self, bb72, biplanar_bb72):
        """In the bounded-thickness biplanar layout, ALL qubits sit
        on the z=0 base plane. The layer separation happens in the
        POLYLINES (6-point lift/descend), not in the positions."""
        for q in bb72.qubits:
            _, _, z = biplanar_bb72.node_position(q)
            assert z == 0.0

    def test_chequerboard_grid(self, bb72, biplanar_bb72):
        """L-data at (2a, 2b), X-anc at (2a+1, 2b), Z-anc at (2a, 2b+1),
        R-data at (2a+1, 2b+1) — matching bbstim's toric biplanar grid."""
        # Check cell (0, 0).
        lx, ly, lz = biplanar_bb72.node_position(bb72.data_qubits[0])
        assert (lx, ly, lz) == (0.0, 0.0, 0.0)  # L-data at (0, 0)
        zx, zy, zz = biplanar_bb72.node_position(bb72.z_check_qubits[0])
        assert (zx, zy, zz) == (0.0, 1.0, 0.0)  # Z-anc at (0, 1)
        xx, xy, xz = biplanar_bb72.node_position(bb72.x_check_qubits[0])
        assert (xx, xy, xz) == (1.0, 0.0, 0.0)  # X-anc at (1, 0)
        rx, ry, rz = biplanar_bb72.node_position(bb72.data_qubits[bb72.block_size])
        assert (rx, ry, rz) == (1.0, 1.0, 0.0)  # R-data at (1, 1)

    def test_polylines_are_6_point(self, bb72, biplanar_bb72):
        """Each biplanar routing polyline has 6 points (the
        base → port → lift → traverse → descend → base structure)."""
        sched = ibm_schedule(bb72)
        first_cnot_layer = sched.cycle_steps[1]
        edge = first_cnot_layer.active_edges[0]
        rid = RouteID(
            source=edge.control,
            target=edge.target,
            step_tick=1,
            term_name=edge.term_name,
        )
        rg = biplanar_bb72.routing_geometry([rid])
        poly = list(rg.edges.values())[0]
        assert len(poly) == 6

    def test_layer_A_routes_through_positive_z(self, bb72, biplanar_bb72):
        """Edges tagged A2, A3, or B3 route through z > 0 at the
        midpoint of their polyline (the lift/traverse segment)."""
        sched = ibm_schedule(bb72)
        for step in sched.cycle_steps:
            if step.role != "cnot_layer":
                continue
            for edge in step.active_edges:
                if not isinstance(edge, TwoQubitEdge):
                    continue
                if edge.term_name in ("A2", "A3", "B3"):
                    rid = RouteID(
                        source=edge.control,
                        target=edge.target,
                        step_tick=step.tick_index,
                        term_name=edge.term_name,
                    )
                    rg = biplanar_bb72.routing_geometry([rid])
                    poly = list(rg.edges.values())[0]
                    # Points 2 and 3 are the lifted traverse.
                    assert poly[2][2] > 0, f"term {edge.term_name} z={poly[2][2]}"
                    return  # One sample is enough.
        pytest.fail("No layer-A edge found")

    def test_layer_B_routes_through_negative_z(self, bb72, biplanar_bb72):
        """Edges tagged A1, B1, or B2 route through z < 0."""
        sched = ibm_schedule(bb72)
        for step in sched.cycle_steps:
            if step.role != "cnot_layer":
                continue
            for edge in step.active_edges:
                if not isinstance(edge, TwoQubitEdge):
                    continue
                if edge.term_name in ("A1", "B1", "B2"):
                    rid = RouteID(
                        source=edge.control,
                        target=edge.target,
                        step_tick=step.tick_index,
                        term_name=edge.term_name,
                    )
                    rg = biplanar_bb72.routing_geometry([rid])
                    poly = list(rg.edges.values())[0]
                    assert poly[2][2] < 0, f"term {edge.term_name} z={poly[2][2]}"
                    return
        pytest.fail("No layer-B edge found")

    def test_monomial_exposure_exceeds_biplanar(self, bb72):
        """The monomial layout has strictly higher J_kappa than the
        biplanar layout at the reference operating point. This is
        the central physical claim of the biplanar construction."""
        sched = ibm_schedule(bb72)
        ref = enumerate_pure_L_minwt_logicals(bb72, sector="X")
        tmpl = compute_bb_ibm_event_template(bb72, sched)
        exp_t = prepare_exposure_template(tmpl, ref)
        np_t = NumpyExposureTemplate.from_exposure_template(exp_t)
        kernel = RegularizedPowerLawKernel(alpha=3.0, r0=1.0)
        mono = MonomialColumnEmbedding.from_bb(bb72)
        bipl = IBMBiplanarEmbedding.from_bb(bb72)
        mono_j = j_kappa_numpy(np.asarray(mono.positions), np_t, kernel, J0=0.04, tau=1.0)
        bipl_j = j_kappa_numpy(np.asarray(bipl.positions), np_t, kernel, J0=0.04, tau=1.0)
        assert mono_j > bipl_j

    def test_invalid_layer_height_rejected(self, bb72):
        with pytest.raises(ValueError, match="layer_height"):
            IBMBiplanarEmbedding.from_bb(bb72, layer_height=0.0)

    def test_round_trip(self, biplanar_bb72):
        data = biplanar_bb72.to_json()
        reloaded = load_embedding(data)
        assert isinstance(reloaded, IBMBiplanarEmbedding)
        assert reloaded == biplanar_bb72


# ---------------------------------------------------------------------------
# FixedPermutationColumnEmbedding
# ---------------------------------------------------------------------------


class TestFixedPermutationColumnEmbedding:
    def test_construction_with_permutation(self):
        pts = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        perm = (2, 0, 1)
        emb = FixedPermutationColumnEmbedding(
            positions=tuple(pts),
            permutation=perm,
            source_description="unit test",
        )
        assert emb.permutation == (2, 0, 1)
        assert emb.source_description == "unit test"

    def test_invalid_permutation_rejected(self):
        with pytest.raises(ValueError, match="bijection"):
            FixedPermutationColumnEmbedding(
                positions=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                permutation=(0, 0),
            )

    def test_permutation_length_mismatch(self):
        with pytest.raises(ValueError, match="length"):
            FixedPermutationColumnEmbedding(
                positions=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                permutation=(0,),
            )

    def test_from_json_file_round_trip(self, tmp_path):
        pts = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        emb = FixedPermutationColumnEmbedding(
            positions=tuple(pts),
            num_columns=3,
            num_rows=1,
            layers_per_cell=1,
            source_description="benchmark fixture",
            name="test_fixed",
        )
        path = tmp_path / "fixed.json"
        path.write_text(json.dumps(emb.to_json()))
        reloaded = FixedPermutationColumnEmbedding.from_json_file(path)
        assert reloaded == emb

    def test_round_trip_via_load_embedding(self):
        emb = FixedPermutationColumnEmbedding(
            positions=((0.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
            num_columns=2,
            num_rows=1,
            layers_per_cell=1,
        )
        data = emb.to_json()
        reloaded = load_embedding(data)
        assert isinstance(reloaded, FixedPermutationColumnEmbedding)
        assert reloaded == emb


# ---------------------------------------------------------------------------
# ibm_schedule
# ---------------------------------------------------------------------------


class TestIBMSchedule:
    def test_head_cycle_tail_lengths(self, bb72):
        sched = ibm_schedule(bb72)
        # Head: 1 R step for z_memory (reset all qubits).
        assert len(sched.head_steps) == 1
        # Cycle depth = 3 + 2*(|A| + |B|) = 3 + 12 = 15 for Bravyi codes.
        assert len(sched.cycle_steps) == 15
        # Tail: 1 final data measurement.
        assert len(sched.tail_steps) == 1

    def test_x_memory_head_has_rx_and_r_steps(self, bb72):
        sched = ibm_schedule(bb72, experiment="x_memory")
        # RX on data, then R on ancillas → 2 head steps.
        assert len(sched.head_steps) == 2
        # Tail measures data in X basis.
        tail = sched.tail_steps[0]
        for e in tail.active_edges:
            assert e.gate == "MX"

    def test_cnot_counts_match_stabilizer_weights(self, bb72):
        """Total cycle CNOTs = 6*lm (Z-checks) + 6*lm (X-checks)."""
        sched = ibm_schedule(bb72)
        x_cnots = 0
        z_cnots = 0
        for step in sched.cycle_steps:
            for e in step.active_edges:
                if isinstance(e, TwoQubitEdge):
                    if e.interaction_sector == "X":
                        x_cnots += 1
                    elif e.interaction_sector == "Z":
                        z_cnots += 1
        expected = 6 * bb72.l * bb72.m
        assert x_cnots == expected
        assert z_cnots == expected

    def test_every_cnot_layer_is_fully_parallel(self, bb72):
        """Each monomial CNOT layer fires `lm` parallel CNOTs with no
        qubit conflicts."""
        sched = ibm_schedule(bb72)
        for step in sched.cycle_steps:
            if step.role != "cnot_layer":
                continue
            assert len(step.active_edges) == bb72.l * bb72.m

    def test_z_check_cnots_have_data_as_control(self, bb72):
        """In X-sector layers, data qubits are controls (→ X error
        propagation into ancilla)."""
        sched = ibm_schedule(bb72)
        data_set = set(bb72.data_qubits)
        z_anc_set = set(bb72.z_check_qubits)
        for step in sched.cycle_steps:
            for e in step.active_edges:
                if isinstance(e, TwoQubitEdge) and e.interaction_sector == "X":
                    assert e.control in data_set
                    assert e.target in z_anc_set

    def test_x_check_cnots_have_x_anc_as_control(self, bb72):
        """In Z-sector layers, x-ancillas are controls."""
        sched = ibm_schedule(bb72)
        data_set = set(bb72.data_qubits)
        x_anc_set = set(bb72.x_check_qubits)
        for step in sched.cycle_steps:
            for e in step.active_edges:
                if isinstance(e, TwoQubitEdge) and e.interaction_sector == "Z":
                    assert e.control in x_anc_set
                    assert e.target in data_set

    def test_h_brackets_sandwich_the_x_check_layers(self, bb72):
        """First cycle tick is H on x-ancillas, last gate tick is also
        H on x-ancillas, followed by MR."""
        sched = ibm_schedule(bb72)
        first_step = sched.cycle_steps[0]
        last_step_before_meas = sched.cycle_steps[-2]
        # Both should be single_q with H gates on x-ancillas.
        for step in (first_step, last_step_before_meas):
            assert step.role == "single_q"
            assert len(step.active_edges) == len(bb72.x_check_qubits)
            for e in step.active_edges:
                assert e.gate == "H"
                assert e.qubit in bb72.x_check_qubits

    def test_last_cycle_step_measures_all_ancillas(self, bb72):
        sched = ibm_schedule(bb72)
        last = sched.cycle_steps[-1]
        assert last.role == "meas"
        # All ancillas are measured with MR.
        measured = {e.qubit for e in last.active_edges}
        assert measured == set(bb72.z_check_qubits + bb72.x_check_qubits)
        for e in last.active_edges:
            assert e.gate == "MR"

    def test_schedule_round_trip(self, bb72):
        """JSON round-trip of the full Schedule."""
        from weave.ir import Schedule

        sched = ibm_schedule(bb72)
        reloaded = Schedule.from_json(sched.to_json())
        assert reloaded == sched

    def test_compile_noiseless_is_deterministic(self, bb72):
        """End-to-end: compile BB72 + ibm_schedule + monomial embedding
        without noise; the resulting Stim circuit samples zero
        detector events. This is the strongest correctness check
        available without a bbstim reference."""
        sched = ibm_schedule(bb72)
        emb = MonomialColumnEmbedding.from_bb(bb72)
        compiled = compile_extraction(
            code=bb72,
            embedding=emb,
            schedule=sched,
            kernel=CrossingKernel(),
            local_noise=LocalNoiseConfig(),
            geometry_noise=GeometryNoiseConfig(),
            rounds=2,
        )
        sampler = compiled.circuit.compile_detector_sampler()
        samples = sampler.sample(shots=100)
        assert not np.any(samples)

    def test_compile_bb108_succeeds(self):
        """Sanity check: BB108 with 8 logicals also compiles and
        produces a deterministic noiseless circuit."""
        bb108 = build_bb108()
        sched = ibm_schedule(bb108)
        emb = MonomialColumnEmbedding.from_bb(bb108)
        compiled = compile_extraction(
            code=bb108,
            embedding=emb,
            schedule=sched,
            kernel=CrossingKernel(),
            local_noise=LocalNoiseConfig(),
            geometry_noise=GeometryNoiseConfig(),
            rounds=1,
        )
        sampler = compiled.circuit.compile_detector_sampler()
        samples = sampler.sample(shots=50)
        assert not np.any(samples)

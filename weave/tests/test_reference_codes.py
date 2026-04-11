"""Reference-code regression tests: pinned parameters for well-known CSS codes.

Every code in this file has its parameters (n, k, d when computable)
verified against a published source AND cross-checked against weave's
own computation. This file is a **physics-level regression guard** for:

- Parity-check matrix construction (HX / HZ shapes, CSS condition).
- GF(2) rank computation (k = n - rank(HX) - rank(HZ) for CSS codes).
- Logical operator extraction (`find_logicals` returns k independent
  logicals, with symplectic pairing `X_i Z_j ≡ δ_ij (mod 2)`).
- Quantum distance computation (`CSSCode.distance()` on small codes
  where exhaustive enumeration is tractable).
- Legacy circuit generation (noiseless detector sampling is clean).

Pinned values here should only change when the underlying
construction or algorithm changes. A failing test in this file is a
physics-level regression and must be investigated before being
changed.

Sources
-------
- Shor, "Scheme for reducing decoherence in quantum computer memory",
  Phys. Rev. A 52, R2493 (1995). Original Shor code [[9, 1, 3]].
- Steane, "Multiple particle interference and quantum error
  correction", Proc. R. Soc. Lond. A 452, 2551 (1996).
  Independently: Calderbank & Shor, "Good quantum error-correcting
  codes exist", Phys. Rev. A 54, 1098 (1996). Steane code [[7, 1, 3]].
- Tillich & Zémor, "Quantum LDPC codes with positive rate and
  minimum distance proportional to √n", IEEE Trans. Inf. Theory 60,
  1193 (2014), arXiv:0903.0566. Hypergraph product construction and
  the k-formula `k_q = k1·k2 + k1^T·k2^T`.

Each test class cites the specific source for the code it tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from weave.codes.css_code import CSSCode
from weave.codes.hypergraph_product_code import HypergraphProductCode
from weave.util import pcm

# =============================================================================
# Shared assertion helpers
# =============================================================================


def assert_css_and_logicals(code: CSSCode) -> None:
    """Verify the core CSS / logical-basis invariants common to every
    reference code:

    1. `HX · HZ^T ≡ 0 (mod 2)` (CSS condition).
    2. `find_logicals()` returns `k` X-logicals and `k` Z-logicals.
    3. Logicals are in the correct kernels: `HZ · L_X ≡ 0`, `HX · L_Z ≡ 0`.
    4. Symplectic pairing: `L_X · L_Z^T ≡ I (mod 2)`.
    """
    css_product = (code.HX @ code.HZ.T) % 2
    assert np.all(css_product == 0), "CSS condition violated"

    x_log, z_log = code.find_logicals()
    assert x_log.shape[0] == code.k, f"|X logicals| = {x_log.shape[0]}, expected {code.k}"
    assert z_log.shape[0] == code.k, f"|Z logicals| = {z_log.shape[0]}, expected {code.k}"

    # X logicals must be in ker(HZ).
    for lx in x_log:
        assert np.all((code.HZ @ lx) % 2 == 0), "X logical not in ker(HZ)"

    # Z logicals must be in ker(HX).
    for lz in z_log:
        assert np.all((code.HX @ lz) % 2 == 0), "Z logical not in ker(HX)"

    # Symplectic pairing.
    if code.k > 0:
        comm = (x_log @ z_log.T) % 2
        expected_identity = np.eye(code.k, dtype=int)
        np.testing.assert_array_equal(
            comm, expected_identity, err_msg="logicals are not in a symplectic basis"
        )


def assert_noiseless_samples_clean(code: CSSCode, shots: int = 200) -> None:
    """A noiseless compiled circuit must produce zero detector events."""
    sampler = code.circuit.compile_detector_sampler()
    samples = sampler.sample(shots=shots)
    assert not np.any(samples), (
        f"noiseless detector sampler produced {samples.sum()} events in {shots} shots"
    )


# =============================================================================
# [[7, 1, 3]] Steane code
# =============================================================================


class TestSteaneCode:
    """The [[7, 1, 3]] Steane code.

    Defined as the CSS code with `HX = HZ = H_Hamming(7)`, where the
    Hamming(7, 4, 3) classical code has parity-check matrix whose columns
    are the binary representations of 1..7.

    Reference
    ---------
    Steane, "Multiple particle interference and quantum error correction"
    (1996); Calderbank & Shor, "Good quantum error-correcting codes exist"
    (1996). Parameters [[7, 1, 3]] are universally documented; see e.g.
    Nielsen & Chuang, *Quantum Computation and Quantum Information*
    (Cambridge, 2010), §10.4.2.
    """

    @pytest.fixture
    def code(self) -> CSSCode:
        H = pcm.hamming(7)
        return CSSCode(HX=H, HZ=H, rounds=1)

    def test_parameters(self, code):
        """Pinned: n = 7, k = 1."""
        assert code.n == 7
        assert code.k == 1

    def test_distance(self, code):
        """Pinned: quantum distance 3 (Steane 1996)."""
        assert code.distance() == 3

    def test_num_data_and_ancilla_qubits(self, code):
        """Steane has 7 data + 3 Z-ancillas + 3 X-ancillas = 13 total."""
        assert len(code.data_qubits) == 7
        assert len(code.z_check_qubits) == 3
        assert len(code.x_check_qubits) == 3
        assert code.n_total == 13

    def test_css_and_logicals(self, code):
        assert_css_and_logicals(code)

    def test_noiseless_detector_sampling(self, code):
        assert_noiseless_samples_clean(code)


# =============================================================================
# [[9, 1, 3]] Shor code
# =============================================================================


# The Shor code's parity-check matrices, written out explicitly from
# Shor's 1995 paper (Section IV). The CSS form uses:
#   Z-type stabilizers: pairs of adjacent Z operators within each
#     3-qubit block (detecting X errors).
#   X-type stabilizers: two "block-pair" X products spanning qubits
#     1-6 and 4-9 (detecting Z errors).
#
# Reference: Shor, "Scheme for reducing decoherence in quantum computer
# memory", Phys. Rev. A 52, R2493 (1995). Also Nielsen & Chuang §10.5.6.

_SHOR_HZ = np.array(
    [
        [1, 1, 0, 0, 0, 0, 0, 0, 0],  # Z1 Z2 (block 1)
        [0, 1, 1, 0, 0, 0, 0, 0, 0],  # Z2 Z3 (block 1)
        [0, 0, 0, 1, 1, 0, 0, 0, 0],  # Z4 Z5 (block 2)
        [0, 0, 0, 0, 1, 1, 0, 0, 0],  # Z5 Z6 (block 2)
        [0, 0, 0, 0, 0, 0, 1, 1, 0],  # Z7 Z8 (block 3)
        [0, 0, 0, 0, 0, 0, 0, 1, 1],  # Z8 Z9 (block 3)
    ],
    dtype=int,
)

_SHOR_HX = np.array(
    [
        [1, 1, 1, 1, 1, 1, 0, 0, 0],  # X1 X2 X3 X4 X5 X6
        [0, 0, 0, 1, 1, 1, 1, 1, 1],  # X4 X5 X6 X7 X8 X9
    ],
    dtype=int,
)


class TestShorCode:
    """The [[9, 1, 3]] Shor code — the first quantum error-correcting code.

    Reference
    ---------
    Shor, "Scheme for reducing decoherence in quantum computer memory",
    Phys. Rev. A 52, R2493 (1995). Also Nielsen & Chuang §10.5.6
    (where `[[9, 1, 3]]` parameters and the stabilizer structure are
    tabulated).
    """

    @pytest.fixture
    def code(self) -> CSSCode:
        return CSSCode(HX=_SHOR_HX, HZ=_SHOR_HZ, rounds=1)

    def test_parameters(self, code):
        """Pinned: n = 9, k = 1."""
        assert code.n == 9
        assert code.k == 1

    def test_distance(self, code):
        """Pinned: quantum distance 3 (Shor 1995)."""
        assert code.distance() == 3

    def test_stabilizer_counts(self, code):
        """Shor has 6 Z-type stabilizers and 2 X-type stabilizers."""
        assert code.HZ.shape == (6, 9)
        assert code.HX.shape == (2, 9)
        assert len(code.z_check_qubits) == 6
        assert len(code.x_check_qubits) == 2
        assert code.n_total == 17  # 9 data + 6 + 2

    def test_css_and_logicals(self, code):
        assert_css_and_logicals(code)

    def test_noiseless_detector_sampling(self, code):
        assert_noiseless_samples_clean(code)


# =============================================================================
# [[13, 1, 3]] HP(rep(3), rep(3)) — smallest hypergraph product code
# =============================================================================


class TestHPRep3Rep3:
    """The [[13, 1, 3]] hypergraph product of two rep(3) repetition codes.

    The smallest nontrivial hypergraph product code. Parameters follow
    directly from the Tillich-Zémor 2014 construction:

        n_q = n1·n2 + r1·r2 = 3·3 + 2·2 = 13
        k_q = k1·k2 + k1^T · k2^T = 1·1 + 0·0 = 1
        d_q = 3 (verified below; weave computes it exactly)

    where `rep(3)` has classical parameters `(n=3, k=1, d=3)` and
    transpose code dimension `k^T = 0` (the repetition parity-check
    has full row rank, so its transpose kernel is trivial).

    Reference
    ---------
    Tillich & Zémor, "Quantum LDPC codes with positive rate and
    minimum distance proportional to √n", IEEE Trans. Inf. Theory
    60, 1193 (2014), arXiv:0903.0566. The [[13, 1, 3]] code is the
    smallest HP instance and appears in every subsequent HP-code
    review.
    """

    @pytest.fixture
    def code(self) -> HypergraphProductCode:
        return HypergraphProductCode(pcm.repetition(3), pcm.repetition(3), rounds=1)

    def test_parameters(self, code):
        """Pinned: n = 13, k = 1."""
        assert code.n == 13
        assert code.k == 1

    def test_distance(self, code):
        """Pinned: quantum distance 3."""
        assert code.distance() == 3

    def test_stabilizer_shapes(self, code):
        """HZ: (n1·r2, n_data) = (6, 13). HX: (r1·n2, n_data) = (6, 13)."""
        assert code.HZ.shape == (6, 13)
        assert code.HX.shape == (6, 13)

    def test_css_and_logicals(self, code):
        assert_css_and_logicals(code)

    def test_noiseless_detector_sampling(self, code):
        assert_noiseless_samples_clean(code)


# =============================================================================
# [[25, 1, 4]] HP(rep(4), rep(4))
# =============================================================================


class TestHPRep4Rep4:
    """The [[25, 1, 4]] hypergraph product of two rep(4) codes.

    Parameters follow the Tillich-Zémor formula:

        n_q = 4·4 + 3·3 = 25
        k_q = 1·1 + 0·0 = 1
        d_q = 4 (verified below)

    This is the next-smallest symmetric HP code after rep(3)×rep(3).

    Reference
    ---------
    Tillich & Zémor 2014 (see `TestHPRep3Rep3` for full cite).
    """

    @pytest.fixture
    def code(self) -> HypergraphProductCode:
        return HypergraphProductCode(pcm.repetition(4), pcm.repetition(4), rounds=1)

    def test_parameters(self, code):
        """Pinned: n = 25, k = 1."""
        assert code.n == 25
        assert code.k == 1

    def test_distance(self, code):
        """Pinned: quantum distance 4 (matches rep(n)×rep(n) distance = n)."""
        assert code.distance() == 4

    def test_stabilizer_shapes(self, code):
        """HZ: (n1·r2, n_data) = (12, 25). HX: (r1·n2, n_data) = (12, 25)."""
        assert code.HZ.shape == (12, 25)
        assert code.HX.shape == (12, 25)

    def test_css_and_logicals(self, code):
        assert_css_and_logicals(code)

    def test_noiseless_detector_sampling(self, code):
        assert_noiseless_samples_clean(code)


# =============================================================================
# [[27, 4, 3]] HP(rep(3), Hamming(7))
# =============================================================================


class TestHPRep3Hamming7:
    """The [[27, 4, 3]] hypergraph product of rep(3) and Hamming(7).

    An asymmetric hypergraph product:

        n_q = 3·7 + 2·3 = 21 + 6 = 27
        k_q = 1·4 + 0·0 = 4 (four logical qubits)
        d_q = 3 (verified below)

    Useful test case because `k > 1` exercises the symplectic
    Gram-Schmidt pairing code path more thoroughly than `k = 1`.

    Reference
    ---------
    Tillich & Zémor 2014 (HP construction).
    """

    @pytest.fixture
    def code(self) -> HypergraphProductCode:
        return HypergraphProductCode(pcm.repetition(3), pcm.hamming(7), rounds=1)

    def test_parameters(self, code):
        """Pinned: n = 27, k = 4."""
        assert code.n == 27
        assert code.k == 4

    def test_distance(self, code):
        """Pinned: quantum distance 3."""
        assert code.distance() == 3

    def test_stabilizer_shapes(self, code):
        """HZ: (n1·r2, n_data) = (9, 27). HX: (r1·n2, n_data) = (14, 27)."""
        assert code.HZ.shape == (9, 27)
        assert code.HX.shape == (14, 27)

    def test_css_and_logicals(self, code):
        """For k = 4, the symplectic pairing test is nontrivial: the
        commutation matrix must be the 4×4 identity."""
        assert_css_and_logicals(code)

    def test_noiseless_detector_sampling(self, code):
        assert_noiseless_samples_clean(code)


# =============================================================================
# [[58, 16, ?]] HP(Hamming(7), Hamming(7)) — distance beyond brute-force cap
# =============================================================================


class TestHPHamming7Hamming7:
    """The [[58, 16, ?]] hypergraph product of two Hamming(7) codes.

    Parameters from the Tillich-Zémor formula:

        n_q = 7·7 + 3·3 = 49 + 9 = 58
        k_q = 4·4 + 0·0 = 16

    Distance is known in the literature to be bounded above by
    `min(d1, d2) = 3` (both Hamming codes have `d = 3`), but weave's
    brute-force `distance()` enumeration is infeasible here: the
    Z-sector nullspace has dimension 37, requiring `2^37 − 1 ≈ 10^11`
    codeword enumerations. We therefore pin **only** `(n, k)` and the
    non-distance invariants, and document that quantum distance
    verification awaits PR 8/9 when the propagation analyzer gives us
    a tighter bound without exhaustive enumeration.

    This code is also the `k = 16` stress test for the symplectic
    Gram-Schmidt routine.

    Reference
    ---------
    Tillich & Zémor 2014 (HP construction).
    """

    @pytest.fixture
    def code(self) -> HypergraphProductCode:
        return HypergraphProductCode(pcm.hamming(7), pcm.hamming(7), rounds=1)

    def test_parameters(self, code):
        """Pinned: n = 58, k = 16. Distance left unverified."""
        assert code.n == 58
        assert code.k == 16

    def test_distance_beyond_brute_force_cap(self, code):
        """`distance()` must raise ValueError for this code under weave's
        current k ≤ 20 enumeration cap. If this test ever stops raising,
        it means weave has a faster distance algorithm and we should
        add a real distance assertion here."""
        with pytest.raises(ValueError, match="infeasible"):
            code.distance()

    def test_stabilizer_shapes(self, code):
        """HZ: (n1·r2, n_data) = (21, 58). HX: (r1·n2, n_data) = (21, 58)."""
        assert code.HZ.shape == (21, 58)
        assert code.HX.shape == (21, 58)

    def test_css_and_logicals(self, code):
        """Stress test for k = 16 symplectic Gram-Schmidt."""
        assert_css_and_logicals(code)

    def test_noiseless_detector_sampling(self, code):
        """Even with 58 data qubits and 16 logicals, the noiseless circuit
        should sample zero detector events. Use fewer shots for speed."""
        assert_noiseless_samples_clean(code, shots=50)


# =============================================================================
# Tillich-Zémor k-formula cross-check
# =============================================================================


class TestTillichZemorFormula:
    """Direct verification that the Tillich-Zémor k-formula

        k_q = k1 · k2 + k1^T · k2^T

    where `k^T = dim ker(H^T)`, matches weave's computed `code.k`
    for every HP code in this file.

    Reference
    ---------
    Tillich & Zémor, IEEE Trans. Inf. Theory 60, 1193 (2014),
    arXiv:0903.0566, Theorem 1 / Section II.A.
    """

    @staticmethod
    def _k_and_kT(H: np.ndarray) -> tuple[int, int]:
        """Return `(k, k_T)` for a classical parity-check matrix `H`."""
        rows, cols = H.shape
        rank_H = pcm.row_echelon(H)[1]
        return (cols - rank_H, rows - rank_H)

    @pytest.mark.parametrize(
        "name,H1_factory,H2_factory",
        [
            ("rep3_rep3", lambda: pcm.repetition(3), lambda: pcm.repetition(3)),
            ("rep4_rep4", lambda: pcm.repetition(4), lambda: pcm.repetition(4)),
            ("rep3_ham7", lambda: pcm.repetition(3), lambda: pcm.hamming(7)),
            ("ham7_ham7", lambda: pcm.hamming(7), lambda: pcm.hamming(7)),
        ],
    )
    def test_k_formula_matches_weave(self, name, H1_factory, H2_factory):
        H1 = H1_factory()
        H2 = H2_factory()
        k1, k1_T = self._k_and_kT(H1)
        k2, k2_T = self._k_and_kT(H2)
        k_formula = k1 * k2 + k1_T * k2_T

        code = HypergraphProductCode(H1, H2, rounds=1)
        assert code.k == k_formula, (
            f"{name}: Tillich-Zemor formula gives k = {k_formula}, weave computed k = {code.k}"
        )

    def test_rep_n_has_trivial_transpose(self):
        """rep(n) has `k^T = 0`: its parity-check has full row rank."""
        for n in (3, 4, 5, 7):
            _, k_T = self._k_and_kT(pcm.repetition(n))
            assert k_T == 0, f"rep({n}) transpose code should be trivial"

    def test_hamming_7_has_trivial_transpose(self):
        """Hamming(7) has `k^T = 0`: its 3 rows are linearly independent
        so the transpose kernel in F^3 is trivial."""
        _, k_T = self._k_and_kT(pcm.hamming(7))
        assert k_T == 0

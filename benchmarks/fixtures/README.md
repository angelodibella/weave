# benchmarks/fixtures/

Frozen reference data that pins weave's numerical outputs against an
external authority or a previous weave run. Each fixture carries
**explicit provenance**: where it came from, which commit/version
produced it, and what weave test it backs.

## Fixtures in this directory

### `bbstim_bb72_pureL_minwt_logicals.csv`

The 36 minimum-weight pure-L X-logicals of BB72, as enumerated by the
reference implementation `bbstim` from

> A. Di Bella, *Geometry-induced correlated noise in qLDPC syndrome
> extraction* (2026),
> <https://github.com/angelodibella/geometry-induced-correlated-noise-in-qldpc-syndrome-extraction>

**Provenance.** Copied verbatim from
`~/Projects/works/geometry-induced-correlated-noise-in-qldpc-syndrome-extraction/results/bb72_pureL_minwt_logicals.csv`.
SHA256: `d5bdfa5c18548c998e20fa1ab1ea5f7a004e73b618ce219cabae535b6117dab4`.

**Format.** CSV with columns `index,support_indices,support_monomials`.
- `index` — bbstim's internal enumeration order.
- `support_indices` — a Python-literal list of six integers in
  `[0, 36)`, indexing into bbstim's **row-major** flat numbering
  `flat = i*m + j`.
- `support_monomials` — the equivalent monomial labels `x^a y^b`.

**Used by.** `weave/tests/test_regression_bb72.py::test_bbstim_pureL_X_logicals_match`
pins weave's `enumerate_pure_L_minwt_logicals(build_bb72(), sector="X")`
output against this fixture. The comparison requires two conversions
to reconcile the two conventions:

1. **Polynomial-matrix orientation.** bbstim's `polynomial_matrix`
   acts as `M e_i = e_{i - \text{shift}}` (a right/backward action);
   weave's acts as `M e_i = e_{i + \text{shift}}` (a forward action).
   The two conventions are related by the group automorphism
   `g \mapsto g^{-1}`, i.e. `(i, j) \mapsto (-i \bmod l,
   -j \bmod m)`. Every support gets this inversion applied before
   the comparison.
2. **Flat-index encoding.** weave uses column-major `flat = j*l + i`;
   bbstim uses row-major `flat = i*m + j`. The translation is
   `weave_flat -> (flat % l, flat // l) -> invert -> i*m + j`.

The translation is deterministic and documented in the test. The
post-translation set of 36 supports matches bbstim byte-for-byte.

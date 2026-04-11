# Benchmarks

This directory holds the regression fixtures, benchmark specs, and frozen
reference data that back weave's published claims. Everything here is
reproducible from a pinned commit hash, a pinned seed, and a pinned
`(code, embedding, schedule, kernel, local_noise)` tuple.

## Why this directory exists

Reviewers judging an infrastructure paper need to verify two things:

1. **Fidelity.** Does weave reproduce the numbers from the published
   reference implementation (the PRX-Quantum-under-review *"Geometry-induced
   correlated noise in qLDPC syndrome extraction"* / `bbstim`)?
2. **Generalization.** Does the same code path work on code families
   other than bivariate bicycle codes?

This directory answers both by shipping fixtures that can be rerun on
any branch and compared against committed reference outputs.

## Planned regression targets

### 1. BB72 faithful reproduction
Reproduce the bbstim BB72 numbers through weave's compiler, pinning:

- Spearman rank correlation between weighted exposure and logical error
  rate at fixed $p = 10^{-3}$ (bbstim reports $\rho_S \approx 0.965$ on
  59 baseline points).
- Monomial-vs-biplanar LER ratio at the reference operating point
  $(J_0\tau, \alpha, p) = (0.04, 3, 10^{-3})$ (bbstim reports
  $\approx 5\times$).
- 26% worst-case-exposure reduction on the pure-$q(L)$ reference family
  from logical-aware optimization.

### 2. BB144 scaling sanity
Smaller, coarser version of the BB72 suite to confirm the embedding
hierarchy persists at the next benchmark size.

### 3. Hypergraph product sanity
Reproduce the matching-number effective-distance bound and the
exposure-vs-LER ordering on Hamming(7) × Hamming(7) or similar. This is
the "works on non-BB" test.

### 4. Surface code on a flat torus
The novelty demonstration: route a surface code on a `Torus` surface
with geodesic polylines wrapping through the periodic boundary.
Compare the geodesic-distance crossing pattern against a Euclidean-layout
baseline at the same $p$, $J_0$, $\kappa$.

## Layout (planned)

```
benchmarks/
├── README.md               # this file
├── fixtures/               # pinned reference data (JSON / CSV)
│   ├── bb72_reference.json
│   ├── bb144_reference.json
│   └── ...
├── specs/                  # benchmark specifications (code + embedding + schedule + kernel + seeds)
│   ├── bb72_exposure_vs_ler.toml
│   └── ...
└── runners/                # scripts that execute specs and write results
    └── ...
```

## Provenance rules

Every fixture file carries, in its metadata header:

- Weave git commit hash used to generate the reference.
- `uv.lock` hash at generation time.
- Python version.
- Stim, stimbposd, sinter, numpy versions.
- Random seeds for every Monte Carlo stage.
- A SHA-256 of the canonicalized spec.

The CI regression job computes the same values on the current branch and
fails on drift beyond a tolerance (2-sigma Monte Carlo for LERs; exact
equality for deterministic quantities like exposure and matching number).

## Status

**Not yet populated.** This directory is scaffolding for the regression
work planned in the next phase of development. See `private/vision.md`
(gitignored) for the paper-submission acceptance criteria that these
benchmarks will satisfy.

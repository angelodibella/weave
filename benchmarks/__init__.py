"""Weave benchmarks, regression fixtures, and CI runners.

The `benchmarks/` package is not part of the installed `weave`
distribution — it lives alongside the tests so reviewers and CI
can run frozen regression checks against the library on any
commit. The three subpackages here are:

- :mod:`benchmarks.regression` — pure-Python regression functions
  that exercise a canonical slice of the compiler, produce
  deterministic numerical outputs, and can be compared to
  committed reference values in `benchmarks/fixtures/`.
- :mod:`benchmarks.runners` — CLI entry points that run the
  regression suite and (optionally) regenerate fixture files.
- `benchmarks/fixtures/` — committed reference JSON files, one
  per regression target.
"""

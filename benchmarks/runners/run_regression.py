r"""CLI runner for weave's BB72 regression suite.

Usage:

.. code-block:: bash

    uv run python -m benchmarks.runners.run_regression
    uv run python -m benchmarks.runners.run_regression --regenerate

The first form runs all PR 13 regression checks and prints a pass/
fail summary. The second form additionally writes the observed
numerical values to
`benchmarks/fixtures/bb72_reference.json` so downstream CI can
compare against them.

Exit codes
----------
- ``0`` — every assertion in :func:`benchmarks.regression.bb72.run_regression`
  passes.
- ``1`` — one or more assertions failed. Details are printed to
  stdout before exit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmarks.regression.bb72 import BB72Bundle, run_regression


def _fixture_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "fixtures"


def _write_reference_json(result: dict, path: Path) -> None:
    """Write the observed regression values to a JSON fixture.

    We omit the raw `spearman_points` and `ler_sweep` arrays from
    the committed file so casual diffs stay small — a future PR
    can re-add them as a separate `*_detailed.json` if needed.
    """
    payload = {
        "source": "weave PR 13 regression",
        "note": (
            "Weave-only regression fixture. Numerical cross-check "
            "against bbstim is limited to the pure-L X-logical set "
            "equality (see bbstim_bb72_pureL_minwt_logicals.csv)."
        ),
        "fingerprint": result["fingerprint_a"],
        "j_kappa_monomial": result["j_kappa_monomial"],
        "j_kappa_optimized": result["j_kappa_optimized"],
        "reduction_ratio": result["reduction_ratio"],
        "spearman_rho": result["spearman_rho"],
        "bbstim_match": result["bbstim_match"],
        "all_pass": result["all_pass"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run weave's BB72 regression suite (PR 13).")
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="After running, overwrite bb72_reference.json with the observed values.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=500,
        help="Monte Carlo shots per operating point. Default 500.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed base. Default 0.",
    )
    args = parser.parse_args(argv)

    bundle = BB72Bundle.build()
    fixture_dir = _fixture_dir()
    result = run_regression(
        bundle=bundle,
        fixture_dir=fixture_dir,
        shots_per_point=args.shots,
        seed=args.seed,
    )

    if args.regenerate:
        out_path = fixture_dir / "bb72_reference.json"
        _write_reference_json(result, out_path)
        print(f"[PR 13] wrote reference fixture → {out_path}")

    if result["all_pass"]:
        print("[PR 13] ALL CHECKS PASS")
        return 0
    print("[PR 13] REGRESSION FAILED")
    for key in (
        "fingerprint_stable",
        "reduction_ratio",
        "ler_monotone",
        "spearman_rho",
        "bbstim_match",
    ):
        print(f"  {key}: {result[key]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

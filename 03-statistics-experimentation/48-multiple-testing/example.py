"""Seeded synthetic multiple-testing and finite-look peeking experiments."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.stats import norm

from from_scratch import METHODS, reject_hypotheses, valid_level, valid_pvalues


def estimate(values):
    """Monte Carlo mean and sample-standard-deviation/sqrt(B) standard error."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("values must be a finite 1D array with at least two repetitions")
    return {"estimate": float(values.mean()),
            "mc_se": float(values.std(ddof=1) / np.sqrt(values.size))}


def family_metrics(rejected, true_null):
    """Estimate FWER, mean FDP (FDR), and mean non-null rejection fraction.

    Ground truth exists only because these are simulations. Empty discovery
    sets contribute FDP=0; FDR is not a pooled ratio across repetitions.
    """
    rejected = np.asarray(rejected)
    true_null = np.asarray(true_null)
    if (rejected.ndim != 2 or rejected.shape[0] < 2 or rejected.shape[1] == 0
            or rejected.dtype != bool or true_null.dtype != bool
            or true_null.shape != (rejected.shape[1],)):
        raise ValueError("expected a boolean trials-by-tests matrix and matching null mask")
    discoveries = rejected.sum(axis=1)
    false = rejected[:, true_null].sum(axis=1)
    return {
        "fwer": estimate(false > 0),
        "fdr": estimate(false / np.maximum(discoveries, 1)),
        "power": estimate(rejected[:, ~true_null].mean(axis=1)) if (~true_null).any() else None,
        "mean_rejections": estimate(discoveries),
    }


def normal_look_pvalues(block_sums, look_sizes):
    """Two-sided known-variance Z p-values on nested standard-normal samples.

    Each column is the SUM of the new observations since the previous look.
    Under H0, independent block sums have variance equal to the block size.
    """
    looks = np.asarray(look_sizes)
    sums = np.asarray(block_sums)
    if (looks.ndim != 1 or looks.size == 0 or looks.dtype.kind not in "iu"
            or looks[0] <= 0 or np.any(np.diff(looks.astype(float)) <= 0)):
        raise ValueError("look_sizes must be positive, strictly increasing integers")
    if (sums.ndim != 2 or sums.shape[0] == 0 or sums.shape[1] != looks.size
            or np.iscomplexobj(sums) or sums.dtype.kind not in "fiu"
            or not np.isfinite(sums).all()):
        raise ValueError("block_sums must be a finite real matrix matching look_sizes")
    z = np.cumsum(sums, axis=1) / np.sqrt(looks)
    return 2 * norm.sf(np.abs(z))


def peeking_decisions(p_values, level=0.05):
    """Compare policies on the SAME trajectories; last column is final horizon.

    A crossing at any look equals stopping and rejecting at the first crossing.
    Bonferroni is valid only for the supplied finite, prespecified look schedule.
    """
    p = valid_pvalues(p_values, ndim=2)
    level = valid_level(level)
    return {
        "fixed_horizon": p[:, -1] <= level,
        "naive_peeking": np.any(p <= level, axis=1),
        "bonferroni_looks": np.any(p <= level / p.shape[1], axis=1),
    }


def simulate_study(trials=10000, seed=48):
    """Run independent families and a correlated-in-time global-null process."""
    if isinstance(trials, bool) or not isinstance(trials, (int, np.integer)) or trials < 2:
        raise ValueError("trials must be an integer >= 2")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    m, signals, shift, level = 20, 4, 3.0, 0.05
    looks = np.arange(100, 1001, 100)
    families = {}
    for label, nonnull, rng in zip(("global_null", "mixed"), (0, signals), streams[:2]):
        z = rng.normal(size=(trials, m))
        z[:, :nonnull] += shift
        p = 2 * norm.sf(np.abs(z))
        true_null = np.arange(m) >= nonnull
        families[label] = {
            method: family_metrics(
                np.stack([reject_hypotheses(row, level, method) for row in p]), true_null
            ) for method in METHODS
        }

    block_sizes = np.diff(np.r_[0, looks])
    block_sums = streams[2].normal(size=(trials, looks.size)) * np.sqrt(block_sizes)
    p_looks = normal_look_pvalues(block_sums, looks)
    decisions = peeking_decisions(p_looks, level)
    return {
        "synthetic": True,
        "interpretation_status": "author review required",
        "configuration": {
            "trials": int(trials), "seed": int(seed), "tests_per_family": m,
            "mixed_nonnull": signals, "nonnull_z_shift": shift, "level": level,
            "look_sizes": looks.tolist(), "rng": "PCG64; SeedSequence.spawn(3)",
            "test": "two-sided normal; known variance; independent hypotheses",
        },
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__},
        "analytic_uncorrected_global_null_fwer": float(1 - (1 - level) ** m),
        "families": families,
        "peeking_null_rejection_rates": {name: estimate(mask) for name, mask in decisions.items()},
        "peeking_paired_differences": {
            "naive_minus_fixed": estimate(
                decisions["naive_peeking"].astype(float) - decisions["fixed_horizon"]),
            "naive_minus_bonferroni": estimate(
                decisions["naive_peeking"].astype(float) - decisions["bonferroni_looks"]),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=48)
    parser.add_argument("--output", type=Path, help="Optional JSON; use an ignored outputs/ directory")
    args = parser.parse_args(argv)
    try:
        report = simulate_study(args.trials, args.seed)
    except ValueError as error:
        parser.error(str(error))
    report["executed_at_utc"] = datetime.now(timezone.utc).isoformat()
    serialized = json.dumps(report, indent=2, allow_nan=False)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    main()

"""Seeded synthetic assignment study; prints evidence without writing artifacts."""

import argparse
import platform
import warnings

import numpy as np
import scipy
from scipy.special import expit
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from from_scratch import (
    effective_sample_size, ipw_weights, match_propensity, matching_weights,
    standardized_mean_difference, weighted_effect,
)


def simulate(n_samples=4000, seed=50, assignment_strength=1.0, heterogeneity=0.0):
    """Baseline common causes precede treatment; both potential outcomes known."""
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)) or n_samples < 4:
        raise ValueError("n_samples must be an integer >= 4.")
    if not np.isfinite(assignment_strength) or assignment_strength <= 0:
        raise ValueError("assignment_strength must be finite and positive.")
    if not np.isfinite(heterogeneity):
        raise ValueError("heterogeneity must be finite.")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_samples, 2))
    true_scores = expit(-0.4 + assignment_strength * (1.1 * x[:, 0] + 0.8 * x[:, 1]))
    # Common uniforms and noise make scenario comparisons paired by seed.
    treatment = (rng.random(n_samples) < true_scores).astype(int)
    y0 = 3 + 2 * x[:, 0] + 1.2 * x[:, 1] + rng.normal(size=n_samples)
    effect = 2 + heterogeneity * x[:, 0]
    outcome = y0 + effect * treatment
    return x, treatment, outcome, effect, true_scores


def analyze(x, treatment, outcome, effect, caliper=0.2):
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, tol=1e-10)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x, treatment)
    scores = model.predict_proba(x)[:, 1]
    full = match_propensity(scores, treatment)
    restricted = match_propensity(scores, treatment, caliper)
    if restricted.treated.size == 0:
        raise ValueError("No caliper matches; broaden the design or reconsider the population.")
    weights = {
        "naive": np.ones(treatment.size),
        "match_att": matching_weights(full, treatment),
        "caliper_att": matching_weights(restricted, treatment),
        "ipw_ate": ipw_weights(scores, treatment, "ate"),
        "ipw_att": ipw_weights(scores, treatment, "att"),
    }
    truth = {
        "naive": float(effect.mean()),  # ATE benchmark, not an identification claim.
        "match_att": float(effect[treatment == 1].mean()),
        "caliper_att": float(effect[restricted.treated].mean()),
        "ipw_ate": float(effect.mean()),
        "ipw_att": float(effect[treatment == 1].mean()),
    }
    methods = {}
    for name, w in weights.items():
        estimate = weighted_effect(outcome, treatment, w)
        methods[name] = {
            "estimate": estimate, "truth": truth[name], "error": estimate - truth[name],
            "smd": standardized_mean_difference(x, treatment, w),
            "ess": [effective_sample_size(w[treatment == arm]) for arm in (0, 1)],
            "max_weight": float(w.max()),
        }
    ranges = [(scores[treatment == arm].min(), scores[treatment == arm].max()) for arm in (0, 1)]
    lower, upper = max(r[0] for r in ranges), min(r[1] for r in ranges)
    return {
        "methods": methods, "counts": np.bincount(treatment).tolist(),
        "score_quantiles": [np.quantile(scores[treatment == arm], [0, .01, .5, .99, 1]) for arm in (0, 1)],
        "outside_common_range": float(np.mean((scores < lower) | (scores > upper))),
        "match_max_distance": float(full.distance.max()),
        "unique_controls": int(np.unique(full.control).size),
        "retained": int(restricted.treated.size), "unmatched": int(restricted.unmatched.size),
        "caliper_truth": truth["caliper_att"],
    }


def run_study(n_samples=4000, seed=50, repetitions=10, caliper=0.2, heterogeneity=0.0):
    if isinstance(repetitions, bool) or not isinstance(repetitions, (int, np.integer)) or repetitions < 2:
        raise ValueError("repetitions must be an integer >= 2 for Monte Carlo summaries.")
    scenarios = {}
    for strength in (1.0, 3.0):
        runs = []
        for offset in range(repetitions):
            x, t, y, effect, _ = simulate(n_samples, seed + offset, strength, heterogeneity)
            runs.append(analyze(x, t, y, effect, caliper))
        scenarios[strength] = runs
    return scenarios


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--caliper", type=float, default=0.2)
    parser.add_argument("--heterogeneity", type=float, default=0.0)
    args = parser.parse_args(argv)
    try:
        scenarios = run_study(**vars(args))
    except ValueError as exc:
        parser.error(str(exc))
    print("SYNTHETIC ONLY; point estimates; interpretation candidates require author review.")
    print(f"Python={platform.python_version()} NumPy={np.__version__} SciPy={scipy.__version__} sklearn={sklearn.__version__}")
    print(f"n={args.n_samples} seeds={args.seed}..{args.seed + args.repetitions - 1} caliper={args.caliper} heterogeneity={args.heterogeneity}")
    print("All outcome units are arbitrary; L2 logistic C=1, full-sample fit; no clipping.")
    for strength, runs in scenarios.items():
        first = runs[0]
        print(f"\nassignment_strength={strength}; first-seed controls/treated={first['counts']}")
        for arm, quantiles in enumerate(first["score_quantiles"]):
            formatted = ", ".join(f"{value:.12g}" for value in quantiles)
            print(f"score arm={arm} [min,p01,p50,p99,max]=[{formatted}]")
        print(f"outside empirical common score range={first['outside_common_range']:.4f}; not a positivity test")
        print(f"matching max logit distance={first['match_max_distance']:.4f}; unique controls={first['unique_controls']}")
        print(f"caliper retained={first['retained']} unmatched={first['unmatched']} target truth={first['caliper_truth']:.4f}")
        print("method first_estimate first_truth SMD_X0 SMD_X1 ESS_control ESS_treated max_weight")
        for name, result in first["methods"].items():
            print(f"{name:12} {result['estimate']:.4f} {result['truth']:.4f} "
                  f"{result['smd'][0]:.4f} {result['smd'][1]:.4f} "
                  f"{result['ess'][0]:.1f} {result['ess'][1]:.1f} {result['max_weight']:.2f}")
        print("repeated errors versus sample-specific target: mean_error SD_error MCSE_mean_error RMSE")
        for name in first["methods"]:
            errors = np.array([r["methods"][name]["error"] for r in runs])
            sd = errors.std(ddof=1)
            print(f"{name:12} {errors.mean():.4f} {sd:.4f} {sd / np.sqrt(args.repetitions):.4f} {np.sqrt(np.mean(errors ** 2)):.4f}")
    print("\nMCSE describes simulation precision; no observational-data confidence intervals.")
    return scenarios


if __name__ == "__main__":
    main()

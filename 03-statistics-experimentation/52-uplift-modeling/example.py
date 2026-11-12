"""Synthetic randomized T-Learner study; console/JSON output, no visual exports."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np
import sklearn
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from from_scratch import (
    binary, cost_aware_policy, fit_binned_uplift, ipw_effect, nonnegative,
    policy_value, ranking_metrics, same_length, top_fraction,
)


def simulate(n=12000, seed=52, propensity=0.5, scenario="heterogeneous"):
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 100:
        raise ValueError("n must be an integer of at least 100")
    propensity = nonnegative(propensity, "propensity")
    if not 0 < propensity < 1:
        raise ValueError("propensity must be strictly between 0 and 1")
    if scenario not in ("heterogeneous", "homogeneous"):
        raise ValueError("unknown scenario")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    p0 = expit(-1.0 + 0.8 * x[:, 0] - 0.5 * x[:, 1])
    if scenario == "heterogeneous":
        p1 = np.clip(p0 + 0.16 * np.tanh(x[:, 0]) - 0.10 * np.tanh(x[:, 1]), 0.01, 0.99)
    else:
        # Leave room for exactly +0.05, avoiding clipping-induced heterogeneity.
        p0 = np.minimum(p0, 0.94)
        p1 = p0 + 0.05
    treatment = rng.binomial(1, propensity, size=n)
    outcome = rng.binomial(1, np.where(treatment == 1, p1, p0))
    return x, treatment, outcome, p1 - p0


def features(values):
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or min(result.shape) == 0 or not np.isfinite(result).all():
        raise ValueError("features must be a nonempty finite two-dimensional matrix")
    return result


def fit_t_learner(x, treatment, outcome, seed=52):
    x = features(x)
    t, y = binary(treatment, "treatment"), binary(outcome, "outcome")
    same_length(x, t, y)
    models = []
    for arm in (0, 1):
        mask = t == arm
        if not mask.any():
            raise ValueError("T-Learner requires both treatment arms")
        model = RandomForestClassifier(
            n_estimators=160, max_depth=6, min_samples_leaf=60,
            random_state=seed, n_jobs=1,
        )
        models.append(model.fit(x[mask], y[mask]))
    return tuple(models)


def positive_probability(model, x):
    x = features(x)
    # A small arm may have only one observed outcome class.
    position = np.flatnonzero(model.classes_ == 1)
    return model.predict_proba(x)[:, position[0]] if position.size else np.zeros(len(x))


def run_once(n=12000, seed=52, propensity=0.5, scenario="heterogeneous", value=100.0, cost=5.0):
    value, cost = nonnegative(value, "value"), nonnegative(cost, "cost")
    x, t, y, tau = simulate(n, seed, propensity, scenario)
    train, test = train_test_split(np.arange(n), test_size=0.4, random_state=seed)
    control, treated = fit_t_learner(x[train], t[train], y[train], seed)
    baseline = positive_probability(control, x[test])
    uplift = positive_probability(treated, x[test]) - baseline
    binned = fit_binned_uplift(x[train, 0], t[train], y[train])
    bin_scores = binned.predict(x[test, 0])
    # Oracle CATE stays outside fitting, ranking and policy construction.
    z = ipw_effect(y[test], t[test], propensity)
    scores = {"t_learner": uplift, "baseline_response": baseline, "binned": bin_scores}
    policies = {
        "uplift_top30": top_fraction(uplift, 0.3),
        "response_top30": top_fraction(baseline, 0.3),
        "uplift_cost_capacity": cost_aware_policy(uplift, value, cost, 0.3),
        "treat_all": np.ones(len(test), dtype=bool),
        "treat_none": np.zeros(len(test), dtype=bool),
    }
    result = {
        "seed": int(seed), "train_n": len(train), "test_n": len(test),
        "train_treated": int(t[train].sum()), "test_treated": int(t[test].sum()),
        "oracle_test_ate": float(tau[test].mean()), "ipw_test_ate": float(z.mean()),
        "cate_rmse_synthetic_only": float(np.sqrt(np.mean((uplift - tau[test]) ** 2))),
        "binned_rmse_synthetic_only": float(np.sqrt(np.mean((bin_scores - tau[test]) ** 2))),
        "ranking": {name: ranking_metrics(score, z) for name, score in scores.items()},
        "policies": {},
    }
    for name, selected in policies.items():
        result["policies"][name] = {
            "treated_fraction": float(selected.mean()),
            "ipw_net_per_eligible": policy_value(selected, z, value, cost),
            "oracle_net_per_eligible_synthetic_only": policy_value(selected, tau[test], value, cost),
        }
    result["policies"]["random_expected30"] = {
        "treated_fraction": 0.3,
        "ipw_net_per_eligible": float(0.3 * (value * z.mean() - cost)),
        "oracle_net_per_eligible_synthetic_only": float(0.3 * (value * tau[test].mean() - cost)),
    }
    return result


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "sd_across_runs": float(values.std(ddof=1)) if values.size > 1 else None,
        "mcse_of_mean": float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else None,
    }


def run_study(n=12000, seed=52, propensity=0.5, scenario="heterogeneous", repetitions=10, value=100.0, cost=5.0):
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    runs = [run_once(n, seed + r, propensity, scenario, value, cost) for r in range(repetitions)]
    summary = {
        "ipw_test_ate": summarize([run["ipw_test_ate"] for run in runs]),
        "oracle_test_ate": summarize([run["oracle_test_ate"] for run in runs]),
        "cate_rmse_synthetic_only": summarize([run["cate_rmse_synthetic_only"] for run in runs]),
        "qini_area": {name: summarize([run["ranking"][name]["qini_area"] for run in runs])
                      for name in runs[0]["ranking"]},
        "policy_ipw_net_per_eligible": {
            name: summarize([run["policies"][name]["ipw_net_per_eligible"] for run in runs])
            for name in runs[0]["policies"]
        },
        "paired_uplift_minus_response_net": summarize([
            run["policies"]["uplift_top30"]["ipw_net_per_eligible"]
            - run["policies"]["response_top30"]["ipw_net_per_eligible"] for run in runs
        ]),
    }
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data": "synthetic randomized binary outcome; no public dataset",
        "interpretation_status": "author review required",
        "config": dict(n=n, first_seed=seed, repetitions=repetitions, scenario=scenario,
                       propensity=propensity, test_fraction=0.4, capacity=0.3, value=value, cost=cost,
                       forest=dict(n_estimators=160, max_depth=6, min_samples_leaf=60, n_jobs=1)),
        "versions": dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__),
        "summary": summary, "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--propensity", type=float, default=0.5)
    parser.add_argument("--scenario", choices=["heterogeneous", "homogeneous"], default="heterogeneous")
    parser.add_argument("--value", type=float, default=100.0)
    parser.add_argument("--cost", type=float, default=5.0)
    parser.add_argument("--output", type=Path, help="Optional JSON; use an ignored outputs/ path and a new run name")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("output exists; choose a new run name to preserve historical records")
    try:
        result = run_study(args.n, args.seed, args.propensity, args.scenario,
                           args.repetitions, args.value, args.cost)
    except ValueError as error:
        parser.error(str(error))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps({"config": result["config"], "versions": result["versions"],
                      "summary": result["summary"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

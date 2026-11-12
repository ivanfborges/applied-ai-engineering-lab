"""Seeded synthetic adjustment study; no real users, downloads or visual exports."""

from __future__ import annotations

import argparse
import platform

import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression

from from_scratch import DAG, fwl_coefficient, standardized_difference


TOTAL_EFFECT = 2.0
DIRECT_EFFECT = 1.3
POPULATION_SLOPES = {"Naive": 2.9, "C": 2.0, "C + M": 1.3, "C + S": 1.04}
SPECIFICATIONS = {"Naive": (), "C": ("C",), "C + M": ("C", "M"),
                  "C + S": ("C", "S")}


def study_graph() -> DAG:
    return DAG("CAMUSY", [("C", "A"), ("C", "Y"), ("A", "Y"), ("A", "M"),
                          ("M", "Y"), ("A", "S"), ("U", "S"), ("U", "Y")])


def simulate_scm(n_samples: int = 50_000, seed: int = 49) -> dict[str, np.ndarray]:
    """Independent disturbances; U is retained only as simulator-only oracle data."""
    if isinstance(n_samples, (bool, np.bool_)) or not isinstance(n_samples, (int, np.integer)) or n_samples < 100:
        raise ValueError("n_samples must be an integer >= 100")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    c = rng.binomial(1, 0.5, n_samples)
    a = rng.binomial(1, 0.2 + 0.6 * c)
    u = rng.normal(size=n_samples)
    noise_m = rng.normal(size=n_samples)
    noise_s = rng.normal(scale=0.5, size=n_samples)
    noise_y = rng.normal(size=n_samples)
    m = 0.7 * a + noise_m
    s = a + u + noise_s
    y = 1.3 * a + m + 1.5 * c + 1.2 * u + noise_y
    # Interventions reuse each unit's disturbances and regenerate the mediator.
    y0 = noise_m + 1.5 * c + 1.2 * u + noise_y
    y1 = 1.3 + (0.7 + noise_m) + 1.5 * c + 1.2 * u + noise_y
    return {"A": a, "C": c, "M": m, "S": s, "U": u, "Y": y,
            "Y0": y0, "Y1": y1}


def estimate_specifications(data: dict[str, np.ndarray]) -> dict[str, float]:
    """Fit declared specifications; no adjustment set is selected by prediction."""
    estimates = {}
    for name, controls in SPECIFICATIONS.items():
        design = np.column_stack([data[key] for key in ("A", *controls)])
        model = LinearRegression().fit(design, data["Y"])
        estimates[name] = float(model.coef_[0])
        c = np.column_stack([data[key] for key in controls]) if controls else np.empty((len(design), 0))
        np.testing.assert_allclose(fwl_coefficient(data["A"], data["Y"], c),
                                   estimates[name], atol=1e-10, rtol=1e-10)
    estimates["Standardized C"] = standardized_difference(data["A"], data["Y"], data["C"])
    return estimates


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args(argv)
    if args.n_samples < 100 or args.seed < 0 or args.repetitions < 2:
        parser.error("require n-samples >= 100, seed >= 0 and repetitions >= 2")

    print(f"Python {platform.python_version()}; NumPy {np.__version__}; scikit-learn {sklearn.__version__}")
    print(f"SYNTHETIC SCM: n={args.n_samples}; seeds={args.seed}..{args.seed + args.repetitions - 1}")
    print(f"Known total ATE={TOTAL_EFFECT:.2f}; direct path={DIRECT_EFFECT:.2f}")
    graph = study_graph()
    for z in [(), ("C",), ("C", "M"), ("C", "S")]:
        print(f"Original backdoor criterion {z}: {graph.satisfies_backdoor('A', 'Y', z)}")

    runs = []
    for index in range(args.repetitions):
        data = simulate_scm(args.n_samples, args.seed + index)
        estimates = estimate_specifications(data)
        np.testing.assert_allclose(data["Y1"] - data["Y0"], TOTAL_EFFECT, atol=1e-12)
        runs.append(estimates)
        if index == 0:
            print("First seed: treatment counts [untreated, treated] by C")
            for c in (0, 1):
                counts = [int(np.sum((data["C"] == c) & (data["A"] == a))) for a in (0, 1)]
                print(f"  C={c}: {counts}")
            for name, value in estimates.items():
                print(f"  {name}: {value:.6f}")
            print(f"  Oracle paired intervention ATE: {np.mean(data['Y1'] - data['Y0']):.6f}")

    print("Repeated samples: mean | SD | MCSE(mean) | mean minus total ATE")
    for name in runs[0]:
        values = np.array([run[name] for run in runs])
        sd = values.std(ddof=1)
        print(f"  {name}: {values.mean():.6f} | {sd:.6f} | "
              f"{sd / np.sqrt(args.repetitions):.6f} | {values.mean() - TOTAL_EFFECT:+.6f}")
    print("FWL/sklearn agreement checked for every fit. Author review required for interpretations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""A seeded, synthetic product scorecard; no services, plots or release action."""

import argparse
from datetime import datetime, timezone
import json
from numbers import Integral
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from metrics import conversion_lift, scorecard, zero_incident_upper_bound


CONFIG = {
    "control": {"served": 0.990, "conversion_given_served": 0.12, "retention": 0.38,
                "churn": 0.06, "unsafe_given_served": 0.002,
                "median_latency_s": 1.4, "latency_sigma": 0.50, "mean_cost": 0.04},
    "treatment": {"served": 0.985, "conversion_given_served": 0.15, "retention": 0.39,
                  "churn": 0.07, "unsafe_given_served": 0.008,
                  "median_latency_s": 2.2, "latency_sigma": 0.65, "mean_cost": 0.10},
}

# Chosen before the run for teaching; no business policy or safety standard.
ILLUSTRATIVE_MAXIMA = {
    "p95_latency_s": 5.0, "cost_per_request": 0.08,
    "unsafe_per_request": 0.005, "error_rate": 0.02,
}


def generate_users(users_per_arm: int = 3000, seed: int = 47) -> pd.DataFrame:
    """Balanced synthetic assignment; one eligible first request per user."""
    if (isinstance(users_per_arm, bool) or not isinstance(users_per_arm, Integral)
            or users_per_arm < 1):
        raise ValueError("users_per_arm must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    n = users_per_arm
    arms = rng.permutation(np.repeat(["control", "treatment"], n))
    # Entry timing and subscriber status precede treatment outcomes.
    users = pd.DataFrame({
        "user_id": [f"synthetic-{i}" for i in range(2 * n)],
        "arm": arms, "followup_days": rng.integers(1, 15, 2 * n),
        "subscriber_at_start": rng.random(2 * n) < 0.6,
    })
    frames = []
    for arm, cfg in CONFIG.items():
        group = users.loc[users["arm"] == arm].copy()
        served = rng.random(n) < cfg["served"]
        converted = served & (rng.random(n) < cfg["conversion_given_served"])
        retained = pd.Series(rng.random(n) < cfg["retention"], index=group.index, dtype="boolean")
        churned = pd.Series(rng.random(n) < cfg["churn"], index=group.index, dtype="boolean")
        group["served_ok"] = served
        group["converted"] = converted
        group["returned_d7"] = retained.where(group["followup_days"] >= 8)
        group["churned_7d"] = churned.where(
            (group["followup_days"] >= 7) & group["subscriber_at_start"]
        )
        group["latency_s"] = rng.lognormal(np.log(cfg["median_latency_s"]), cfg["latency_sigma"], n)
        group["cost"] = rng.gamma(shape=4, scale=cfg["mean_cost"] / 4, size=n)
        group["revenue"] = converted.astype(float) * 2.0
        group["unsafe"] = served & (rng.random(n) < cfg["unsafe_given_served"])
        frames.append(group)
    return pd.concat(frames).sort_index().reset_index(drop=True)


def run_study(users_per_arm: int = 3000, seed: int = 47) -> dict:
    summary = scorecard(generate_users(users_per_arm, seed))
    control, treatment = summary.loc["control"], summary.loc["treatment"]
    lift = conversion_lift(
        int(control["converted_users"]), int(control["assigned_users"]),
        int(treatment["converted_users"]), int(treatment["assigned_users"]),
    )
    checks = {
        metric: {"observed": float(treatment[metric]), "illustrative_maximum": maximum,
                 "within_point_boundary": bool(treatment[metric] <= maximum)}
        for metric, maximum in ILLUSTRATIVE_MAXIMA.items()
    }
    return {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "data": "synthetic independent users; one first request per assigned user",
        "hypothesis": "Higher conversion can coexist with worse latency, cost and safety incidence.",
        "configuration": {"seed": seed, "users_per_arm": users_per_arm, "arms": CONFIG,
                          "followup_days_inclusive": [1, 14], "subscriber_probability": 0.6,
                          "revenue_per_conversion": 2.0, "cost_gamma_shape": 4},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__},
        "scorecard": {arm: {k: None if pd.isna(v) else v for k, v in row.items()}
                      for arm, row in summary.to_dict(orient="index").items()},
        "conversion_lift_95pct_wald": lift,
        "treatment_point_checks": checks,
        "zero_incident_exercise": {
            "incidents": 0, "independent_trials": 1000,
            "one_sided_95pct_upper": zero_incident_upper_bound(1000),
            "scope": "Separate analytical exercise, not the observed arm safety rates.",
        },
        "interpretation_status": "author review required",
        "interpretation_candidate": (
            "Compare the conversion interval with every descriptive guardrail and contribution; "
            "a primary-metric gain alone does not establish acceptable product value."
        ),
        "limitation": (
            "Constructed generator, not production evidence. Only conversion has a comparative "
            "interval; guardrail point checks establish neither non-inferiority nor release approval. "
            "Independent retention/churn labels, perfect safety detection, incomplete costs."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users-per-arm", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--output-dir", type=Path, help="Optional ignored outputs/ directory")
    args = parser.parse_args()
    try:
        record = run_study(args.users_per_arm, args.seed)
    except ValueError as error:
        parser.error(str(error))
    print("SYNTHETIC PRODUCT SCORECARD — author review required")
    print(pd.DataFrame(record["scorecard"]).to_string(float_format=lambda x: f"{x:.6f}"))
    print("\nConversion lift (fractions, not percentage points):", record["conversion_lift_95pct_wald"])
    print("\nIllustrative treatment point checks (no guardrail inference):")
    for name, check in record["treatment_point_checks"].items():
        print(f"  {name}: {check}")
    print("\nSeparate zero-incident exercise:", record["zero_incident_exercise"])
    print("\n" + record["limitation"])
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = args.output_dir / f"scorecard-{stamp}.json"
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(f"\nRun record: {destination}")


if __name__ == "__main__":
    main()

"""Educational, balanced-panel DiD: one adoption date and never-treated controls."""

from __future__ import annotations

import platform
from importlib.metadata import version

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def difference_in_differences(treated_pre, treated_post, control_pre, control_post):
    """Compute an additive contrast of four finite scalar cell means."""
    values = np.asarray([treated_pre, treated_post, control_pre, control_post], dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("Cell means must be four finite scalars.")
    return float((values[1] - values[0]) - (values[3] - values[2]))


def validate_panel(data: pd.DataFrame, cutoff: int = 0) -> pd.DataFrame:
    """Reject ambiguous panels; derive post from the declared common cutoff."""
    columns = ["unit", "period", "treated", "outcome"]
    if not isinstance(data, pd.DataFrame) or not set(columns).issubset(data.columns):
        raise ValueError(f"Panel must contain {columns}.")
    panel = data[columns].copy()
    if panel.empty or panel.isna().any().any():
        raise ValueError("Panel must be nonempty and contain no missing values.")
    if not np.isfinite(cutoff) or int(cutoff) != cutoff:
        raise ValueError("cutoff must be a finite integer period.")
    for column in ["period", "treated", "outcome"]:
        try:
            panel[column] = pd.to_numeric(panel[column], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{column} must be numeric.") from exc
        if not np.isfinite(panel[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must be finite.")
    if not (panel["period"] == np.floor(panel["period"])).all():
        raise ValueError("period must contain integers.")
    if set(panel["treated"].unique()) != {0, 1}:
        raise ValueError("treated must contain both binary groups 0 and 1.")
    if panel.duplicated(["unit", "period"]).any():
        raise ValueError("Each unit-period pair must be unique.")
    if panel.groupby("unit")["treated"].nunique().ne(1).any():
        raise ValueError("Group membership must be constant within each unit.")
    n_periods = panel["period"].nunique()
    if panel.groupby("unit").size().ne(n_periods).any():
        raise ValueError("Balanced panel required: every unit needs every period.")
    panel["post"] = (panel["period"] >= cutoff).astype(int)
    if panel["post"].nunique() != 2:
        raise ValueError("At least one pre and one post period are required.")
    return panel


def manual_did(data: pd.DataFrame, cutoff: int = 0) -> float:
    """Equal-unit/equal-period mean contrast, for a complete balanced panel."""
    panel = validate_panel(data, cutoff)
    means = panel.groupby(["treated", "post"])["outcome"].mean()
    return difference_in_differences(
        means.loc[1, 0], means.loc[1, 1], means.loc[0, 0], means.loc[0, 1]
    )


def fit_twfe(data: pd.DataFrame, cutoff: int = 0):
    """Fit unit/time effects with CRV1 covariance and a cluster-df t reference."""
    panel = validate_panel(data, cutoff)
    counts = panel.drop_duplicates("unit").groupby("treated").size()
    if counts.min() < 2:
        raise ValueError("Inference requires at least two units in each group.")
    return smf.ols("outcome ~ C(unit) + C(period) + treated:post", data=panel).fit(
        cov_type="cluster",
        cov_kwds={"groups": panel["unit"], "use_correction": True, "df_correction": True},
        use_t=True,
    )


def simulate_panel(seed: int = 51, n_units: int = 100, effect: float = 4.0,
                   differential_trend: float = 0.0) -> pd.DataFrame:
    """Synthetic productivity points; treatment is assigned at the team level."""
    if isinstance(n_units, (bool, np.bool_)) or not isinstance(n_units, (int, np.integer)) or n_units < 4 or n_units % 2:
        raise ValueError("n_units must be an even integer of at least four.")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if not np.isfinite([effect, differential_trend]).all():
        raise ValueError("effect and differential_trend must be finite.")
    rng = np.random.default_rng(seed)
    periods = np.arange(-3, 3)
    panel = pd.MultiIndex.from_product(
        [range(n_units), periods], names=["unit", "period"]
    ).to_frame(index=False)
    treated = (np.arange(n_units) < n_units // 2).astype(int)
    baseline = rng.normal(0, 3, n_units)
    # Stationary AR(1) noise makes within-team dependence explicit.
    noise = rng.normal(0, 1.5, (n_units, len(periods)))
    for t in range(1, len(periods)):
        noise[:, t] = 0.6 * noise[:, t - 1] + np.sqrt(1 - 0.6**2) * noise[:, t]
    untreated = (
        50 + baseline[:, None] + 2 * treated[:, None]
        + np.array([-2.0, -1.1, 0.0, 1.0, 1.8, 2.2])[None, :]
        + differential_trend * treated[:, None] * periods[None, :] + noise
    )
    observed = untreated + effect * treated[:, None] * (periods >= 0)[None, :]
    panel["treated"] = np.repeat(treated, len(periods))
    panel["outcome"] = observed.ravel()
    return panel


def run_study() -> pd.DataFrame:
    """Run paired-seed baseline/violation and genuine-pre-period placebo fits."""
    rows = []
    for scenario, slope in [("parallel", 0.0), ("differential_trend", 0.8)]:
        panel = simulate_panel(differential_trend=slope)
        for comparison, data, cutoff, truth in [
            ("rollout", panel, 0, 4.0),
            ("pre_placebo", panel.loc[panel["period"] < 0], -1, 0.0),
        ]:
            model = fit_twfe(data, cutoff)
            estimate = manual_did(data, cutoff)
            if not np.isclose(estimate, model.params["treated:post"], atol=1e-10, rtol=0):
                raise RuntimeError("Manual DiD and balanced-panel TWFE disagree.")
            means = validate_panel(data, cutoff).groupby(["treated", "post"])["outcome"].mean()
            interval = model.conf_int(alpha=0.05).loc["treated:post"]
            periods = data["period"].unique()
            gap = periods[periods >= cutoff].mean() - periods[periods < cutoff].mean()
            rows.append({
                "scenario": scenario, "comparison": comparison,
                "true_effect": truth, "treated_change": means.loc[1, 1] - means.loc[1, 0],
                "control_change": means.loc[0, 1] - means.loc[0, 0],
                "manual_did": estimate, "twfe": model.params["treated:post"],
                "cluster_se": model.bse["treated:post"],
                "ci_low": interval.iloc[0], "ci_high": interval.iloc[1],
                "expected_trend_bias": slope * gap,
            })
    return pd.DataFrame(rows)


def main() -> None:
    print("Synthetic productivity points; 100 teams, 50 per group, seed=51.")
    print("Rollout at 0; periods -3..2; AR(1) rho=0.6, stationary noise SD=1.5.")
    print("Placebo uses only -3,-2,-1, with fake cutoff=-1; no actual treatment.")
    print("95% CRV1 intervals: cluster=team, correction=True, t reference df=99.")
    print("Python", platform.python_version(),
          {name: version(name) for name in ["numpy", "pandas", "scipy", "statsmodels"]})
    print(run_study().to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("Interpretation candidates require author review; no production effect established.")


if __name__ == "__main__":
    main()

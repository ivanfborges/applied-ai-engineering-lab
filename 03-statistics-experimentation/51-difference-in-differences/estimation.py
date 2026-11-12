"""Balanced single-rollout estimators; synthetic truth is never a fitted input."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t

from example import fit_twfe, manual_did, validate_panel
from simulation import simulate


def trajectories(panel: pd.DataFrame) -> pd.DataFrame:
    """Counterfactual anchored to the pre-window average gap, matching pooled DiD."""
    observed = panel.groupby(["period", "treated"]).outcome.mean().unstack()
    result = pd.DataFrame({"control": observed[0], "treated": observed[1]})
    pre = result.index < 0
    gap = (result.loc[pre, "treated"] - result.loc[pre, "control"]).mean()
    result["estimated_cf"] = result.control + gap
    result["true_y0"] = panel.loc[panel.treated == 1].groupby("period").y0.mean()
    result["estimated_gap"] = result.treated - result.estimated_cf
    return result


def oracle_decomposition(panel: pd.DataFrame) -> dict[str, float]:
    """Decompose a realized contrast using unobservable synthetic-only columns."""
    post_treated = panel.loc[(panel.treated == 1) & (panel.period >= 0)]
    truth = float((post_treated.y1 - post_treated.y0).mean())
    untreated = panel.copy()
    untreated["outcome"] = untreated.y0
    mean_untreated = panel.copy()
    mean_untreated["outcome"] = mean_untreated.mean_y0
    # Remove the noise contribution using the exact same finite set of teams.
    noise_contrast = manual_did(untreated) - manual_did(mean_untreated)
    deterministic = panel.copy()
    deterministic["outcome"] = panel.outcome - panel.y0 + panel.mean_y0
    bias = manual_did(deterministic) - truth
    estimate = manual_did(panel)
    return {"truth": truth, "estimate": estimate, "error": estimate - truth,
            "identification_bias": bias, "noise_error": noise_contrast}


def change_interval(panel: pd.DataFrame, cutoff: int = 0) -> dict[str, float]:
    """Fast CRV1 covariance algebraically equivalent to full balanced-panel TWFE.

    Statsmodels' full dummy regression is used separately for cross-checking.
    The full design parameter count is retained in the finite-sample correction.
    """
    data = validate_panel(panel, cutoff)
    changes = data.groupby(["unit", "post"]).outcome.mean().unstack()
    changes = changes[1] - changes[0]
    groups = data.drop_duplicates("unit").set_index("unit").treated
    g = len(groups)
    if groups.value_counts().min() < 2:
        raise ValueError("At least two team clusters per group are required.")
    n = len(data)
    k = g + data.period.nunique()  # intercept, G-1 unit, T-1 time, interaction
    if n <= k:
        raise ValueError("Insufficient residual degrees of freedom.")
    meat = 0.0
    for group in [0, 1]:
        values = changes.loc[groups == group].to_numpy()
        meat += np.square(values - values.mean()).sum() / len(values)**2
    correction = g / (g - 1) * (n - 1) / (n - k)
    se = float(np.sqrt(correction * meat))
    estimate = manual_did(data, cutoff)
    critical = t.ppf(0.975, g - 1)
    return {"estimate": estimate, "se": se, "low": estimate - critical * se,
            "high": estimate + critical * se, "df": g - 1}


def regression_summary(panel: pd.DataFrame) -> dict[str, float]:
    """Actual statsmodels unit/time dummy regression, CRV1, t(G-1) intervals."""
    result = fit_twfe(panel)
    ci = result.conf_int().loc["treated:post"]
    return {"manual": manual_did(panel), "twfe": float(result.params["treated:post"]),
            "se": float(result.bse["treated:post"]), "low": float(ci.iloc[0]),
            "high": float(ci.iloc[1]), "df": float(result.df_resid_inference)}


def event_study(panel: pd.DataFrame, reference: int = -1) -> pd.DataFrame:
    """Group-time contrasts relative to -1 with saturated TWFE CRV1 intervals.

    Pointwise intervals, not simultaneous bands. The reference is normalized,
    not estimated. Each team's contrast with the reference retains covariance.
    """
    data = validate_panel(panel)
    wide = data.pivot(index="unit", columns="period", values="outcome")
    if reference not in wide or reference >= 0:
        raise ValueError("reference must be an observed genuine pre-treatment period.")
    groups = data.drop_duplicates("unit").set_index("unit").treated
    if groups.value_counts().min() < 2:
        raise ValueError("At least two clusters in each group are required.")
    g, periods = wide.shape
    n, k = g * periods, g + 2 * periods - 2
    correction = g / (g - 1) * (n - 1) / (n - k)
    critical = t.ppf(0.975, g - 1)
    rows = []
    for period in wide.columns:
        changes = wide[period] - wide[reference]
        arms = [changes.loc[groups == group].to_numpy() for group in [0, 1]]
        estimate = float(arms[1].mean() - arms[0].mean())
        meat = sum(np.square(a - a.mean()).sum() / len(a)**2 for a in arms)
        se = float(np.sqrt(correction * meat))
        rows.append({"period": period, "estimate": estimate, "se": se,
                     "low": estimate - critical * se, "high": estimate + critical * se,
                     "reference": period == reference})
    return pd.DataFrame(rows)


def placebo(panel: pd.DataFrame, cutoff: int = -3) -> dict[str, float]:
    """Only genuine pre-periods: the six periods -6..-1, fake start at -3."""
    if cutoff >= 0:
        raise ValueError("Placebo cutoff must be before actual adoption at 0.")
    genuine_pre = panel.loc[panel.period < 0]
    result = change_interval(genuine_pre, cutoff)
    result["max_period_used"] = int(genuine_pre.period.max())
    return result


def sampling_distribution(draws: int = 200, seed: int = 5100, **config) -> pd.DataFrame:
    """Independently seeded datasets, each with a full pooled DiD estimate/CI."""
    if not isinstance(draws, int) or not 20 <= draws <= 1000:
        raise ValueError("draws must be an integer in [20, 1000].")
    if "seed" in config:
        raise ValueError("Use the root seed argument, not config['seed'].")
    children = np.random.SeedSequence(seed).spawn(draws)
    rows = []
    for child in children:
        panel = simulate(seed=int(child.generate_state(1)[0]), **config)
        interval = change_interval(panel)
        treated = panel.loc[(panel.treated == 1) & (panel.period >= 0)]
        truth = float((treated.y1 - treated.y0).mean())
        rows.append({**interval, "truth": truth,
                     "covered": interval["low"] <= truth <= interval["high"]})
    return pd.DataFrame(rows)

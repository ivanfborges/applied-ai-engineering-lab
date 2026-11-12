"""Synthetic AI-assistant rollout with explicit no-rollout potential outcomes."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate(
    seed: int = 51,
    effect: float = 4.0,
    baseline_gap: float = 5.0,
    noise: float = 1.5,
    slope: float = 0.0,
    spillover: float = 0.0,
    anticipation: float = 0.0,
    n_teams: int = 120,
) -> pd.DataFrame:
    """Balanced 12-period panel; first half adopts at 0, others never adopt.

    y0 is the no-rollout world, including the differential untreated slope.
    y1 adds own-treatment effects (including the defined anticipation window).
    Control spillovers are a separate exposure, not an own-treatment effect.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if isinstance(n_teams, bool) or not isinstance(n_teams, (int, np.integer)) or n_teams < 4 or n_teams % 2:
        raise ValueError("n_teams must be an even integer >= 4.")
    values = np.asarray([effect, baseline_gap, noise, slope, spillover, anticipation], dtype=float)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("Simulation parameters must be finite scalars.")
    if noise < 0 or not 0 <= spillover <= 1 or anticipation < 0:
        raise ValueError("noise/anticipation must be nonnegative; spillover must be in [0, 1].")
    rng = np.random.default_rng(seed)
    periods = np.arange(-6, 6)
    team_effects = rng.normal(0, 3, n_teams)
    # Shared shocks are fixed within a seed and identical for both groups.
    time_effects = 0.6 * periods + rng.normal(0, 0.3, len(periods))
    errors = rng.normal(0, 1, (n_teams, len(periods)))
    for k in range(1, len(periods)):
        errors[:, k] = 0.6 * errors[:, k - 1] + 0.8 * errors[:, k]
    treated = (np.arange(n_teams) < n_teams // 2).astype(int)
    post = periods >= 0
    anticipating = (periods >= -2) & (periods < 0)
    mean_y0 = (
        50 + team_effects[:, None] + baseline_gap * treated[:, None]
        + time_effects[None, :] + slope * treated[:, None] * periods[None, :]
    )
    y0 = mean_y0 + noise * errors
    own_effect = effect * post + anticipation * anticipating
    y1 = y0 + own_effect[None, :]
    spill = spillover * effect * (1 - treated[:, None]) * post[None, :]
    observed = y0 + treated[:, None] * own_effect[None, :] + spill
    panel = pd.MultiIndex.from_product(
        [range(n_teams), periods], names=["unit", "period"]
    ).to_frame(index=False)
    panel["treated"] = np.repeat(treated, len(periods))
    panel["post"] = np.tile(post.astype(int), n_teams)
    panel["exposed"] = panel.treated * panel.post
    panel["outcome"] = observed.ravel()
    panel["y0"] = y0.ravel()
    panel["y1"] = y1.ravel()
    panel["mean_y0"] = mean_y0.ravel()
    panel["spillover_effect"] = spill.ravel()
    panel["anticipation_effect"] = (
        treated[:, None] * anticipation * anticipating[None, :]
    ).ravel()
    return panel


def expected_bias(effect: float = 4.0, slope: float = 0.0,
                  spillover: float = 0.0, anticipation: float = 0.0) -> float:
    """E[DiD]-post ATT for the fixed six-pre/six-post, two-anticipation design."""
    return 6 * slope - spillover * effect - anticipation / 3


def adoption_schedule() -> pd.DataFrame:
    """Conceptual staggered schedule only; deliberately no pooled ATT estimate."""
    periods = np.arange(-6, 6)
    return pd.DataFrame(
        [periods >= 0, periods >= 3, np.zeros(12, dtype=bool)],
        index=["Early adopters", "Late adopters", "Never treated"], columns=periods,
    ).astype(int)

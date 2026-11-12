"""Educational product measurement with explicit populations and missingness."""

from math import ceil, expm1, isfinite, log, sqrt
from numbers import Integral
from statistics import NormalDist

import numpy as np
import pandas as pd
from scipy.stats import beta


def _count(value: int, name: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def rate(events: int, eligible: int) -> float:
    """Reject an empty denominator; do not turn missing evidence into zero."""
    _count(eligible, "eligible", 1)
    _count(events, "events")
    if events > eligible:
        raise ValueError("events cannot exceed eligible")
    return events / eligible


def nearest_rank(values, probability: float = 0.95) -> float:
    """Inverse empirical CDF, including the minimum at probability zero."""
    if not isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("probability must be finite and in [0, 1]")
    observations = np.asarray(values)
    if observations.ndim != 1 or observations.size == 0:
        raise ValueError("values must be a nonempty one-dimensional numeric sequence")
    if observations.dtype.kind not in "iuf" or not np.isfinite(observations).all():
        raise ValueError("values must be finite numbers")
    ordered = np.sort(observations)
    return float(ordered[max(0, ceil(probability * len(ordered)) - 1)])


def zero_incident_upper_bound(eligible: int, alpha: float = 0.05) -> float:
    """Exact one-sided binomial upper bound ONLY when zero incidents occur."""
    _count(eligible, "eligible", 1)
    if not isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and in (0, 1)")
    return -expm1(log(alpha) / eligible)


def conversion_lift(
    control_events: int,
    control_n: int,
    treatment_events: int,
    treatment_n: int,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Independent-user, unpooled Wald interval; not a rare-event method."""
    pc = rate(control_events, control_n)
    pt = rate(treatment_events, treatment_n)
    if not isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and in (0, 1)")
    if min(control_events, control_n - control_events,
           treatment_events, treatment_n - treatment_events) < 5:
        raise ValueError("Wald illustration requires >= 5 successes and failures per arm")
    se = sqrt(pc * (1 - pc) / control_n + pt * (1 - pt) / treatment_n)
    z = NormalDist().inv_cdf(1 - alpha / 2)
    delta = pt - pc
    return {"absolute_lift": delta, "relative_lift": delta / pc,
            "standard_error": se, "ci_low": delta - z * se, "ci_high": delta + z * se}


def validate_users(users: pd.DataFrame) -> None:
    """Validate a complete assignment ledger, one first request per user.

    This can detect malformed rows, not assignments absent from the supplied ledger.
    """
    required = {
        "user_id", "arm", "followup_days", "served_ok", "converted", "returned_d7",
        "subscriber_at_start", "churned_7d", "latency_s", "revenue", "cost", "unsafe",
    }
    missing = required - set(users.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if users.empty or not users.columns.is_unique:
        raise ValueError("Provide nonempty data with unique column names")
    ids = users["user_id"]
    if ids.isna().any() or not ids.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError("user_id must be a nonempty string")
    if ids.duplicated().any():
        raise ValueError("Duplicate user_id: aggregate/deduplicate events before measurement")
    if users["arm"].isna().any() or set(users["arm"]) != {"control", "treatment"}:
        raise ValueError("Both control and treatment assignments are required")
    for column in ("followup_days", "latency_s", "revenue", "cost"):
        series = users[column]
        if series.dtype.kind not in "iuf" or not np.isfinite(series.to_numpy()).all():
            raise ValueError(f"{column} must contain finite numeric telemetry")
        if (series < 0).any():
            raise ValueError(f"{column} must be nonnegative")
    age = users["followup_days"]
    if (age < 1).any() or (age % 1 != 0).any():
        raise ValueError("followup_days must be whole days >= 1 (24-hour outcome mature)")
    for column in ("served_ok", "converted", "subscriber_at_start", "unsafe"):
        if users[column].isna().any() or not users[column].isin([False, True]).all():
            raise ValueError(f"{column} must be observed binary 0/1, with no missing values")
    mature_d7 = age >= 8
    mature_subscribers = (age >= 7) & users["subscriber_at_start"].astype(bool)
    for column, eligible in (("returned_d7", mature_d7), ("churned_7d", mature_subscribers)):
        series = users[column]
        if not series.notna().equals(eligible):
            raise ValueError(f"{column}: observe exactly the mature eligible population")
        if not series.loc[eligible].isin([False, True]).all():
            raise ValueError(f"{column} must contain binary outcomes where eligible")
    failed = ~users["served_ok"].astype(bool)
    if (failed & (users["converted"].astype(bool) | users["unsafe"].astype(bool))).any():
        raise ValueError("This contract allows conversion/unsafe output only on served requests")


def scorecard(users: pd.DataFrame) -> pd.DataFrame:
    """Descriptive arm summaries; expose each changing cohort denominator."""
    validate_users(users)
    rows = []
    for arm, group in users.groupby("arm", sort=True):
        n = len(group)
        successes = int(group["converted"].sum())
        mature = group["followup_days"] >= 8
        subscribers = (group["followup_days"] >= 7) & group["subscriber_at_start"].astype(bool)
        retention_n, churn_n = int(mature.sum()), int(subscribers.sum())
        rows.append({
            "arm": arm, "assigned_users": n, "converted_users": successes,
            "conversion": rate(successes, n),
            "d7_eligible": retention_n,
            "d7_retention": rate(int(group.loc[mature, "returned_d7"].sum()), retention_n)
            if retention_n else np.nan,
            "churn_eligible": churn_n,
            "subscription_churn_7d": rate(int(group.loc[subscribers, "churned_7d"].sum()), churn_n)
            if churn_n else np.nan,
            "error_rate": rate(int((~group["served_ok"].astype(bool)).sum()), n),
            "p50_latency_s": nearest_rank(group["latency_s"].to_numpy(), 0.5),
            "p95_latency_s": nearest_rank(group["latency_s"].to_numpy(), 0.95),
            "revenue_per_user": float(group["revenue"].sum() / n),
            "cost_per_request": float(group["cost"].sum() / n),
            "contribution_per_user": float((group["revenue"] - group["cost"]).sum() / n),
            # Failed and non-converting attempts still incur serving costs.
            "cost_per_conversion": float(group["cost"].sum() / successes) if successes else np.nan,
            "unsafe_incidents": int(group["unsafe"].sum()),
            "unsafe_per_request": rate(int(group["unsafe"].sum()), n),
        })
    return pd.DataFrame(rows).set_index("arm")


def binomial_interval(events: int, n: int, confidence: float = .95) -> tuple[float, float]:
    """Two-sided Clopper-Pearson interval, including zero/all event cases."""
    rate(events, n)
    if not isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    tail = (1 - confidence) / 2
    return (float(beta.ppf(tail, events, n - events + 1)) if events else 0.,
            float(beta.isf(tail, events + 1, n - events)) if events < n else 1.)


def difference_interval(kc: int, nc: int, kt: int, nt: int,
                        confidence: float = .95) -> tuple[float, float]:
    """Conservative risk-difference bounds: union of two exact arm intervals."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    lc, uc = binomial_interval(kc, nc, 1 - (1 - confidence) / 2)
    lt, ut = binomial_interval(kt, nt, 1 - (1 - confidence) / 2)
    return lt - uc, ut - lc


def bounded_mean_interval(values, lower: float, upper: float,
                          confidence: float = .95) -> tuple[float, float]:
    """Hoeffding bound for independent samples with KNOWN support, not sample range."""
    values = np.asarray(values, dtype=float)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or not isfinite(lower) or not isfinite(upper) or lower > upper
            or (values < lower).any() or (values > upper).any() or not 0 < confidence < 1):
        raise ValueError("Provide finite independent observations within known bounds and valid confidence")
    radius = (upper - lower) * sqrt(log(2 / (1 - confidence)) / (2 * len(values)))
    return max(lower, float(values.mean()) - radius), min(upper, float(values.mean()) + radius)


def cost_per_outcome(total_cost: float, outcomes: int) -> float:
    _count(outcomes, "outcomes")
    if not isfinite(total_cost) or total_cost < 0:
        raise ValueError("total_cost must be finite and nonnegative")
    return total_cost / outcomes if outcomes else np.nan


def classify_bound(low: float, high: float, threshold: float, higher: bool = False) -> str:
    """Three-state evidence; a wide interval cannot pass by nonsignificance."""
    if np.isnan(low) or np.isnan(high):
        return "INCONCLUSIVE"
    if low > high or not isfinite(threshold):
        raise ValueError("Invalid interval or threshold")
    if higher:
        return "PASS" if low > threshold else "FAIL" if high <= threshold else "INCONCLUSIVE"
    return "PASS" if high <= threshold else "FAIL" if low > threshold else "INCONCLUSIVE"


def pareto_mask(objectives) -> np.ndarray:
    """Nondominated rows; ALL supplied objectives are minimized. Ties survive."""
    values = np.asarray(objectives, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("objectives must be a finite 2D matrix")
    return np.array([not np.any(np.all(values <= row, axis=1) & np.any(values < row, axis=1))
                     for row in values], dtype=bool)


def rollout_policy(users: pd.DataFrame, settings: dict, treatment_cost_mean: float,
                   detection: float = 1., confirmed_critical: bool = False) -> pd.DataFrame:
    """Illustrative fixed-horizon policy, Bonferroni across five statistical checks.

    Visual simulation uses Uniform[0.5*mean,1.5*mean] costs. This known support
    is essential to the economic bound; do not apply it to the historical Gamma study.
    """
    validate_users(users)
    if not 0 < settings["confidence"] < 1 or not 0 <= detection <= 1:
        raise ValueError("Invalid confidence or detection probability")
    for name in ("latency", "cost_conversion", "safety_delta", "error", "practical"):
        if not isfinite(settings[name]) or settings[name] < 0:
            raise ValueError(f"Invalid policy threshold: {name}")
    if settings["error"] > 1 or settings["safety_delta"] > 1:
        raise ValueError("Rate thresholds cannot exceed one")
    confidence = 1 - (1 - settings["confidence"]) / 5
    c, t = [users.loc[users.arm == arm] for arm in ("control", "treatment")]
    nc, nt = len(c), len(t)
    kc, kt = int(c.converted.sum()), int(t.converted.sum())
    rows = []

    def add(name, observed, bounds, threshold, higher=False, scope="Confidence bound"):
        lo, hi = bounds
        rows.append({"check": name, "observed": observed, "lower": lo, "upper": hi,
                     "boundary": threshold, "status": classify_bound(lo, hi, threshold, higher),
                     "evidence": scope})

    add("Primary conversion lift", kt / nt - kc / nc,
        difference_interval(kc, nc, kt, nt, confidence), settings["practical"], True)
    exceed = int((t.latency_s > settings["latency"]).sum())
    add("Latency: P(duration > maximum) <= 5%", exceed / nt,
        binomial_interval(exceed, nt, confidence), .05)
    joint_confidence = 1 - (1 - confidence) / 2
    cost_lo, cost_hi = bounded_mean_interval(t.cost, .5 * treatment_cost_mean,
                                            1.5 * treatment_cost_mean, joint_confidence)
    p_lo, p_hi = binomial_interval(kt, nt, joint_confidence)
    add("Cost per conversion", cost_per_outcome(float(t.cost.sum()), kt),
        (cost_lo / p_hi, cost_hi / p_lo if p_lo else np.inf), settings["cost_conversion"],
        scope="Exact conversion + known bounded-cost Hoeffding; union bound")
    errors = int((~t.served_ok).sum())
    add("Request error rate", errors / nt, binomial_interval(errors, nt, confidence), settings["error"])
    detected = "unsafe_detected" if "unsafe_detected" in users else "unsafe"
    sc, st = int(c[detected].sum()), int(t[detected].sum())
    add("Safety incidence degradation", st / nt - sc / nc,
        difference_interval(sc, nc, st, nt, confidence), settings["safety_delta"])
    if detection < 1:
        rows[-1]["status"] = "INCONCLUSIVE"
        rows[-1]["evidence"] = "Detected incidence only; latent safety not established with imperfect detection"
    rows.append({"check": "Confirmed critical security/privacy incident", "observed": int(confirmed_critical),
                 "lower": np.nan, "upper": np.nan, "boundary": 0,
                 "status": "FAIL" if confirmed_critical else "PASS",
                 "evidence": "Hard stop if detected; absence is not proof of absence"})
    return pd.DataFrame(rows)


def policy_recommendation(checks: pd.DataFrame) -> str:
    if (checks.status == "FAIL").any():
        return "Candidate violates one or more guardrails"
    if (checks.status == "INCONCLUSIVE").any():
        return "Insufficient evidence for a deployment decision"
    return "Candidate meets the simulated rollout criteria"

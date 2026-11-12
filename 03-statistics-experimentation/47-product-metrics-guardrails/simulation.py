"""Bounded, deterministic synthetic RAG experiments and separate teaching probes."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd
from scipy.stats import norm

from metrics import difference_interval, pareto_mask


@dataclass(frozen=True)
class System:
    conversion: float = .12  # Conditional on a successful first request.
    retention_d7: float = .38  # Marginal exact-day activity at day seven.
    success: float = .99
    median: float = 1.4
    sigma: float = .5
    cost: float = .04
    unsafe: float = .002  # Conditional on a served response.
    churn: float = .02  # Constant daily subscription-cancellation hazard.

    def __post_init__(self):
        for name in ("conversion", "success", "unsafe"):
            if not np.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not np.isfinite(self.retention_d7) or not 0 < self.retention_d7 < 1:
            raise ValueError("retention_d7 must be in (0, 1)")
        for name in ("median", "sigma", "cost", "churn"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")


CONTROL = System()
TREATMENT = System(.15, .39, .985, 2.2, .65, .10, .008, .025)


def _rng(seed):
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return np.random.default_rng(seed)


def simulate(control=CONTROL, treatment=TREATMENT, n=3000, seed=47, revenue=2.,
             snapshot=35, detection=1., critical_fraction=0.) -> tuple[pd.DataFrame, dict]:
    if isinstance(n, bool) or not isinstance(n, Integral) or not 10 <= n <= 50000:
        raise ValueError("n must be an integer between 10 and 50000 per arm")
    if (not np.isfinite(revenue) or revenue < 0 or not 28 <= snapshot <= 90
            or not 0 <= detection <= 1 or not 0 <= critical_fraction <= 1):
        raise ValueError("Invalid revenue, observation horizon or detector setting")
    rng = _rng(seed)
    arms = rng.permutation(np.repeat(["control", "treatment"], n))
    entry = rng.integers(0, 28, 2 * n)
    frames, histories = [], {}
    days = np.arange(31)
    # Arm-specific streams keep treatment changes from redrawing control data.
    streams = np.random.SeedSequence(seed).spawn(2)
    for arm, cfg, stream in zip(("control", "treatment"), (control, treatment), streams):
        r = np.random.default_rng(stream)
        ids = np.flatnonzero(arms == arm)
        age = snapshot - entry[ids]
        served = r.random(n) < cfg.success
        converted = served & (r.random(n) < cfg.conversion)
        latency = cfg.median * np.exp(cfg.sigma * r.standard_normal(n))
        cost = cfg.cost * r.uniform(.5, 1.5, n)
        incident = served & (r.random(n) < cfg.unsafe)
        detected = incident & (r.random(n) < detection)
        critical = detected & (r.random(n) < critical_fraction)
        # Shared exit time makes activity trajectories coherent; users may skip a day and return.
        activity_probability = (1 + cfg.retention_d7) / 2
        engagement_hazard = -np.log(cfg.retention_d7 / activity_probability) / 7
        exit_day = r.exponential(1 / engagement_hazard, n)
        active = (r.random((n, 31)) < activity_probability) & (exit_day[:, None] > days)
        active[:, 0] = True
        rolling = np.maximum.accumulate(active[:, ::-1], axis=1)[:, ::-1]
        # Subscription loss is separate: free-tier use after cancellation is allowed.
        cancellation = r.exponential(1 / cfg.churn, n)
        subscriber = r.random(n) < .6
        returned = pd.array(active[:, 7], dtype="boolean")
        returned[age < 8] = pd.NA
        churned = pd.array(cancellation < 7, dtype="boolean")
        churned[~(subscriber & (age >= 7))] = pd.NA
        frames.append(pd.DataFrame({
            "user_id": [f"synthetic-{i}" for i in ids], "arm": arm, "entry_day": entry[ids],
            "followup_days": age, "served_ok": served, "converted": converted,
            "returned_d7": returned, "subscriber_at_start": subscriber, "churned_7d": churned,
            "latency_s": latency, "cost": cost, "revenue": converted * revenue,
            "unsafe": incident, "unsafe_detected": detected, "critical_detected": critical,
        }))
        histories[arm] = {"activity": active, "rolling": rolling, "age": age,
                          "entry": entry[ids], "cancel_time": cancellation,
                          "subscriber": subscriber, "engagement_exit": exit_day}
    return pd.concat(frames, ignore_index=True), histories


def retention_table(histories: dict) -> pd.DataFrame:
    rows = []
    for arm, h in histories.items():
        subscriber = h["subscriber"]
        observed = np.minimum(h["cancel_time"][subscriber], h["age"][subscriber])
        events = h["cancel_time"][subscriber] <= h["age"][subscriber]
        km = kaplan_meier(observed, events, np.arange(31))
        for day in range(31):
            mature = h["age"] >= day + 1
            # Finite rolling return requires the entire remaining [day,31) window.
            rolling_eligible = h["age"] >= 31
            subscribers = h["subscriber"] & (h["age"] >= day)
            survival = h["cancel_time"] > day
            rows.append({"arm": arm, "day": day, "eligible": int(mature.sum()),
                         "exact_activity": float(h["activity"][mature, day].mean()) if mature.any() else np.nan,
                         "rolling_to_d30": float(h["rolling"][rolling_eligible, day].mean())
                         if rolling_eligible.any() else np.nan,
                         "subscription_survival": km[day],
                         "subscription_churn": 1 - km[day],
                         "oracle_alive": int(survival[h["subscriber"]].sum())})
    return pd.DataFrame(rows)


def kaplan_meier(observed, events, grid):
    observed, events, grid = np.asarray(observed, float), np.asarray(events, bool), np.asarray(grid, float)
    if (observed.ndim != 1 or events.shape != observed.shape or not np.isfinite(observed).all()
            or (observed < 0).any() or grid.ndim != 1 or not np.isfinite(grid).all()):
        raise ValueError("Invalid survival observations")
    if not len(observed):
        return np.full(len(grid), np.nan)
    times, event_counts = np.unique(observed[events], return_counts=True)
    risk = len(observed) - np.searchsorted(np.sort(observed), times, side="left")
    curve = np.cumprod(1 - event_counts / risk)
    result = np.ones(len(grid))
    for i, day in enumerate(grid):
        previous = np.searchsorted(times, day, side="right") - 1
        if day > observed.max():
            result[i] = np.nan  # No extrapolation beyond follow-up support.
        elif previous >= 0:
            result[i] = curve[previous]
    return result


def cohort_heatmap(history: dict) -> pd.DataFrame:
    rows = {}
    for week in range(4):
        in_cohort = history["entry"] // 7 == week
        rows[f"Entry week {week + 1}"] = [
            float(history["activity"][in_cohort & (history["age"] >= d + 1), d].mean())
            if (in_cohort & (history["age"] >= d + 1)).any() else np.nan
            for d in (1, 7, 14, 30)]
    return pd.DataFrame(rows, index=["D1", "D7", "D14", "D30"]).T


def repeated_experiments(control, treatment, n, seed, repeats=80):
    r = _rng(seed)
    kc = r.binomial(n, control.success * control.conversion, repeats)
    kt = r.binomial(n, treatment.success * treatment.conversion, repeats)
    intervals = [difference_interval(int(a), n, int(b), n) for a, b in zip(kc, kt)]
    return pd.DataFrame({"experiment": np.arange(1, repeats + 1), "lift": (kt - kc) / n,
                         "low": [x[0] for x in intervals], "high": [x[1] for x in intervals]})


def uncertainty_path(control, treatment, maximum, seed, frames=24):
    r = _rng(seed)
    c = np.cumsum(r.random(maximum) < control.success * control.conversion)
    t = np.cumsum(r.random(maximum) < treatment.success * treatment.conversion)
    ns = np.unique(np.geomspace(30, maximum, frames).astype(int))
    intervals = [difference_interval(int(c[n-1]), n, int(t[n-1]), n) for n in ns]
    return pd.DataFrame({"n": ns, "lift": [(t[n-1]-c[n-1])/n for n in ns],
                         "low": [x[0] for x in intervals], "high": [x[1] for x in intervals]})


def configurations(control, treatment, revenue, settings, seed, count=100):
    r = _rng(seed)
    configs = [System(conversion=r.uniform(.08, .25), success=r.uniform(.96, .999),
                      median=r.uniform(.7, 3.), sigma=r.uniform(.2, .95), cost=r.uniform(.015, .18),
                      unsafe=r.uniform(0, .015)) for _ in range(count)] + [control, treatment]
    rows = []
    for i, cfg in enumerate(configs):
        p = cfg.conversion * cfg.success
        safety = cfg.unsafe * cfg.success
        p95 = cfg.median * np.exp(norm.ppf(.95) * cfg.sigma)
        cost_conversion = cfg.cost / p if p else np.inf
        reasons = []
        for bad, reason in ((p95 > settings["latency"], "latency"),
                            (cost_conversion > settings["cost_conversion"], "cost/conversion"),
                            (safety - control.unsafe * control.success > settings["safety_delta"], "safety"),
                            (1-cfg.success > settings["error"], "reliability")):
            if bad:
                reasons.append(reason)
        rows.append({"name": "control" if i == count else "treatment" if i == count+1 else f"model-{i:03}",
                     "conversion": p, "p95": p95, "cost": cfg.cost, "unsafe": safety,
                     "contribution": p * revenue - cfg.cost, "feasible": not reasons,
                     "reason": ", ".join(reasons) or "Within population guardrails"})
    table = pd.DataFrame(rows)
    table["pareto"] = False
    feasible = table.feasible
    objectives = table.loc[feasible, ["conversion", "p95", "cost", "unsafe"]].to_numpy(copy=True)
    objectives[:, 0] *= -1
    table.loc[feasible, "pareto"] = pareto_mask(objectives)
    span = float(table.contribution.max() - table.contribution.min())
    table["size"] = 8 + 15 * (table.contribution - table.contribution.min()) / (span or 1)
    return table


def simpson_counts(easy_share_control=.2, easy_share_treatment=.8):
    rows = []
    for arm, share, rates in (("control", easy_share_control, (.9, .3)),
                               ("treatment", easy_share_treatment, (.8, .2))):
        easy = round(1000 * share)
        for segment, n, p in zip(("easy", "hard"), (easy, 1000-easy), rates):
            rows.append({"arm": arm, "segment": segment, "eligible": n,
                         "converted": round(n*p), "rate": p})
    return pd.DataFrame(rows)

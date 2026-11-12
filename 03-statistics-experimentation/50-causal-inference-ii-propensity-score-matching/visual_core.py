"""Synthetic support-team study and numerical evidence for the visual lab.

One independent unit represents a team's ticket-resolution summary. All
covariates are measured before enablement; negative effects save minutes.
The original score-outcome study in example.py is a separate experiment.
"""

from dataclasses import asdict, dataclass
import warnings

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from from_scratch import (
    effective_sample_size, ipw_weights, match_propensity, matching_weights,
    standardized_mean_difference, treatment_vector, weighted_effect,
)


@dataclass(frozen=True)
class Design:
    separation: float = 1.0
    effect: float = -8.0
    heterogeneity: float = 0.0
    hidden: float = 0.0
    structural: bool = False

    def __post_init__(self):
        values = (self.separation, self.effect, self.heterogeneity, self.hidden)
        if not np.isfinite(values).all():
            raise ValueError("Design parameters must be finite.")
        if not 0 <= self.separation <= 4:
            raise ValueError("separation must be between 0 and 4 for this bounded lab.")
        if not -14 <= self.effect <= 0 or not 0 <= self.heterogeneity <= 3:
            raise ValueError("effect must be in [-14, 0] and heterogeneity in [0, 3].")
        if not 0 <= self.hidden <= 2:
            raise ValueError("hidden must be between 0 and 2.")


def assignment(z, design, hidden=0.0):
    scores = expit(-0.3 + design.separation * (0.9*z[..., 0] + 0.65*z[..., 1])
                   + design.hidden*hidden)
    if design.structural:
        scores = np.where(z[..., 0] > 1.25, 1.0, scores)
    return scores


def simulate_teams(n=600, seed=50, design=Design()):
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 20:
        raise ValueError("n must be an integer >= 20.")
    rng = np.random.default_rng(seed)
    z = rng.uniform(-2, 2, size=(n, 2))
    u = rng.uniform(-1, 1, size=n)
    uniform = rng.random(n)
    noise = rng.uniform(-3, 3, size=n)
    true_scores = assignment(z, design, u)
    t = (uniform < true_scores).astype(int)
    treatment_vector(t)
    if min(np.bincount(t)) < 2:
        raise ValueError("This sample needs at least two observations per arm.")
    x = np.column_stack([5 + 2*z[:, 0], 30 + 10*z[:, 1]])
    y0 = 60 + 10*z[:, 0] + 6*z[:, 1] + 8*design.hidden*u + noise
    tau = design.effect - design.heterogeneity*z[:, 0]
    y1 = y0 + tau
    return {
        "x": x, "z": z, "t": t, "y0": y0, "y1": y1,
        "y": y0 + t*tau, "tau": tau, "true_scores": true_scores,
        "hidden_u": u, "design": design, "seed": int(seed),
    }


def population_truth(design, order=48):
    """Deterministic Gauss-Legendre integration over the known uniform population."""
    if not isinstance(order, int) or order < 4:
        raise ValueError("quadrature order must be an integer >= 4.")
    nodes, weights = leggauss(order)
    x0, x1 = np.meshgrid(2*nodes, 2*nodes, indexing="ij")
    z = np.stack([x0, x1], axis=-1)
    mass = np.outer(weights, weights)/4
    if design.hidden:
        scores = sum(w*assignment(z, design, u) for u, w in zip(nodes, weights))/2
    else:
        scores = assignment(z, design)
    tau = design.effect - design.heterogeneity*x0
    return {
        "ate": float(np.sum(mass*tau)),
        "att": float(np.sum(mass*scores*tau)/np.sum(mass*scores)),
    }


def fit_scores(data):
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, tol=1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(data["z"], data["t"])
    scores = model.predict_proba(data["z"])[:, 1]
    if np.any((scores <= 0) | (scores >= 1)):
        raise ValueError("Estimated scores reached a probability boundary; no silent clipping.")
    return model, scores


def diagnostics(data, weights):
    t = data["t"]
    return {
        "estimate": weighted_effect(data["y"], t, weights),
        "smd": standardized_mean_difference(data["x"], t, weights),
        "ess": np.array([effective_sample_size(weights[t == arm]) for arm in (0, 1)]),
        "max_weight": float(np.max(weights)),
    }


def overlap(scores, treatment, bins=16):
    """Histogram intersection and empirical range; neither tests positivity."""
    edges = np.linspace(0, 1, bins+1)
    masses = [np.histogram(scores[treatment == arm], bins=edges)[0] for arm in (0, 1)]
    masses = [m/m.sum() for m in masses]
    lower = max(scores[treatment == arm].min() for arm in (0, 1))
    upper = min(scores[treatment == arm].max() for arm in (0, 1))
    return {
        "intersection": float(np.minimum(*masses).sum()),
        "range": (float(lower), float(upper)),
        "outside_range": float(np.mean((scores < lower) | (scores > upper))),
    }


def analyze_teams(data, caliper=0.2):
    if data["design"].structural:
        raise ValueError("Structural nonpositivity: do not estimate full-population effects here.")
    model, scores = fit_scores(data)
    t = data["t"]
    full = match_propensity(scores, t)
    restricted = match_propensity(scores, t, caliper)
    weights = {
        "Naive": np.ones(t.size), "Matching ATT": matching_weights(full, t),
        "IPW ATE": ipw_weights(scores, t), "IPW ATT": ipw_weights(scores, t, "att"),
    }
    if restricted.treated.size:
        weights["Caliper ATT"] = matching_weights(restricted, t)
    summaries = {name: diagnostics(data, w) for name, w in weights.items()}
    return {
        "data": data, "model": model, "scores": scores,
        "full_matches": full, "matches": restricted,
        "weights": weights, "summaries": summaries, "overlap": overlap(scores, t),
        "sample_ate": float(data["tau"].mean()),
        "sample_att": float(data["tau"][t == 1].mean()),
        "retained_truth": float(data["tau"][restricted.treated].mean()) if restricted.treated.size else None,
    }


def separation_sweep(n=600, seed=50, frames=18, effect=-8.0):
    if not isinstance(frames, int) or frames < 3:
        raise ValueError("frames must be an integer >= 3.")
    # The same exogenous tape isolates assignment changes; the path is not MC evidence.
    return [analyze_teams(simulate_teams(n, seed, Design(float(a), effect)))
            for a in np.linspace(0, 4, frames)]


def repeated_study(n=600, seed=50, repetitions=24):
    if not isinstance(repetitions, int) or repetitions < 2:
        raise ValueError("repetitions must be an integer >= 2.")
    rows = []
    streams = np.random.SeedSequence(seed).spawn(2*repetitions)
    for scenario, h in (("Constant effect", 0.0), ("Heterogeneous effect", 3.0)):
        design = Design(heterogeneity=h)
        truth = population_truth(design)
        for _ in range(repetitions):
            run_seed = int(streams.pop().generate_state(1)[0])
            result = analyze_teams(simulate_teams(n, run_seed, design))
            for method in ("Naive", "Matching ATT", "IPW ATE"):
                target = truth["att"] if method == "Matching ATT" else truth["ate"]
                estimate = result["summaries"][method]["estimate"]
                rows.append({
                    "scenario": scenario, "seed": run_seed, "method": method,
                    "estimate": estimate, "target": target, "error": estimate-target,
                    "sample_ate": result["sample_ate"], "sample_att": result["sample_att"],
                })
    return rows


def repeated_summary(rows):
    summaries = []
    for scenario in dict.fromkeys(row["scenario"] for row in rows):
        for method in ("Naive", "Matching ATT", "IPW ATE"):
            selected = [row for row in rows if row["scenario"] == scenario and row["method"] == method]
            errors = np.array([row["error"] for row in selected])
            summaries.append({
                "scenario": scenario, "method": method, "repetitions": len(errors),
                "target": selected[0]["target"], "mean_error": float(errors.mean()),
                "error_sd": float(errors.std(ddof=1)),
                "mcse_mean_error": float(errors.std(ddof=1)/np.sqrt(errors.size)),
                "rmse": float(np.sqrt(np.mean(errors**2))),
            })
    return summaries


def record_analysis(result):
    """JSON-safe first-sample record; raw observations remain regenerable."""
    return {
        "design": asdict(result["data"]["design"]), "seed": result["data"]["seed"],
        "n": int(result["data"]["t"].size), "overlap": result["overlap"],
        "sample_ate": result["sample_ate"], "sample_att": result["sample_att"],
        "retained_truth": result["retained_truth"],
        "matched": int(result["matches"].treated.size),
        "unmatched": int(result["matches"].unmatched.size),
        "max_logit_distance": float(result["full_matches"].distance.max()),
        "methods": {name: {k: v.tolist() if isinstance(v, np.ndarray) else v
                           for k, v in summary.items()}
                    for name, summary in result["summaries"].items()},
    }

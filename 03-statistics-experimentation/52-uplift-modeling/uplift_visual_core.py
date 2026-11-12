"""Numerical mechanisms for the offline visual lab; all data are synthetic."""

import numpy as np
from scipy.special import expit
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier

from from_scratch import gain_curve, ipw_effect, ranking_metrics, vector, nonnegative


SEGMENTS = ["Already loyal", "Persuadable", "Unaffected", "Negatively affected"]
SEGMENT_P0 = np.array([0.85, 0.25, 0.50, 0.65])
SEGMENT_P1 = np.array([0.87, 0.55, 0.50, 0.55])


def response_surfaces(x, strength=1.0):
    """Logit shift gives bounded probabilities and signed nonlinear CATE."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all():
        raise ValueError("x must be a finite (n, 2) feature matrix")
    if not np.isscalar(strength) or not np.isfinite(strength) or not 0 <= strength <= 2:
        raise ValueError("strength must be in [0, 2]")
    a, b = x.T
    logit0 = -0.3 + 0.85 * a - 0.65 * b + 0.25 * np.sin(a * b)
    shift = 0.9 * np.tanh(b) - 0.7 * np.tanh(a) + 0.5 * np.sin(a * b)
    p0, p1 = expit(logit0), expit(logit0 + strength * shift)
    return p0, p1, p1 - p0


def assignment_probability(x, regime="randomized"):
    x = np.asarray(x, dtype=float)
    response_surfaces(x)  # Shared feature validation.
    if regime == "randomized":
        return np.full(len(x), 0.5)
    if regime not in ("observational", "poor_overlap"):
        raise ValueError("unknown assignment regime")
    scale = 0.8 if regime == "observational" else 3.5
    # This bound belongs to the assignment mechanism, never the IPW estimator.
    return np.clip(expit(scale * (x[:, 0] - 1.2 * x[:, 1])), 0.001, 0.999)


def randomized_data(n, seed, regime="randomized"):
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 20:
        raise ValueError("n must be an integer >= 20")
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.5, 2.5, size=(n, 2))
    p0, p1, tau = response_surfaces(x)
    e = assignment_probability(x, regime)
    t = (rng.random(n) < e).astype(int)
    y = (rng.random(n) < np.where(t == 1, p1, p0)).astype(int)
    return dict(x=x, p0=p0, p1=p1, tau=tau, e=e, t=t, y=y)


def probability(model, x):
    position = np.flatnonzero(model.classes_ == 1)
    return model.predict_proba(x)[:, position[0]] if position.size else np.zeros(len(x))


def fit_learners(data, seed=520):
    """One S model and two T models; equal base-learner settings, same RCT."""
    x, t, y = data["x"], data["t"], data["y"]
    models = {}
    for name in ("s", "t0", "t1"):
        forest = RandomForestClassifier(
            n_estimators=100, max_depth=7, min_samples_leaf=25,
            random_state=seed, n_jobs=1,
        )
        if name == "s":
            forest.fit(np.column_stack((x, t)), y)
        else:
            arm = int(name[-1])
            mask = t == arm
            if not mask.any():
                raise ValueError("training requires both treatment arms")
            forest.fit(x[mask], y[mask])
        models[name] = forest
    return models


def predict_learners(models, x):
    t0, t1 = probability(models["t0"], x), probability(models["t1"], x)
    s0 = probability(models["s"], np.column_stack((x, np.zeros(len(x)))))
    s1 = probability(models["s"], np.column_stack((x, np.ones(len(x)))))
    return dict(t0=t0, t1=t1, t=t1 - t0, s0=s0, s1=s1, s=s1 - s0)


def policy_selection(scores, value=100, cost=5, capacity_percent=30):
    scores = vector(scores, "scores")
    value, cost = nonnegative(value, "value"), nonnegative(cost, "cost")
    if not np.isscalar(capacity_percent) or not np.isfinite(capacity_percent) or not 0 <= capacity_percent <= 100:
        raise ValueError("capacity_percent must be in [0, 100]")
    count = int(len(scores) * capacity_percent / 100)
    order = np.argsort(-scores, kind="stable")
    mask = np.zeros(len(scores), dtype=bool)
    mask[order[:count]] = True
    return mask & (value * scores > cost)


def policy_summary(test, value=100, cost=5, capacity_percent=30):
    mask = policy_selection(test["score"], value, cost, capacity_percent)
    return dict(selected=int(mask.sum()),
                predicted_conversions=float(np.sum(test["score"][mask])),
                true_conversions=float(np.sum(test["tau"][mask])),
                ipw_conversions=float(np.sum(test["z"][mask])),
                contact_cost=float(cost * mask.sum()),
                predicted_net=float(np.sum(value * test["score"][mask] - cost)),
                true_net=float(np.sum(value * test["tau"][mask] - cost)),
                ipw_net=float(np.sum(value * test["z"][mask] - cost)))


def wilson(successes, n, confidence=0.95):
    if n <= 0 or not 0 <= successes <= n or not 0 < confidence < 1:
        raise ValueError("Wilson interval requires valid counts and confidence")
    z = norm.ppf((1 + confidence) / 2)
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - half, center + half


def newcombe_difference(s1, n1, s0, n0):
    """Approximate 95% Newcombe independent-proportions score interval."""
    l1, u1 = wilson(s1, n1)
    l0, u0 = wilson(s0, n0)
    p1, p0 = s1 / n1, s0 / n0
    difference = p1 - p0
    lower = difference - np.hypot(p1 - l1, u0 - p0)
    upper = difference + np.hypot(u1 - p1, p0 - l0)
    return float(difference), float(lower), float(upper)


def sample_size_study(repetitions=30, seed=525):
    result = {}
    for n in (200, 1000, 5000, 20000):
        records = []
        for r in range(repetitions):
            rng = np.random.default_rng(np.random.SeedSequence([seed, n, r]))
            group = rng.integers(0, 4, n)
            t = rng.binomial(1, 0.5, n)
            y = rng.binomial(1, np.where(t, SEGMENT_P1[group], SEGMENT_P0[group]))
            row = []
            for segment in range(4):
                a, b = (group == segment) & (t == 1), (group == segment) & (t == 0)
                if not a.any() or not b.any():
                    raise ValueError("empty arm in sample-size demonstration; use a different seed")
                estimate, low, high = newcombe_difference(int(y[a].sum()), int(a.sum()),
                                                        int(y[b].sum()), int(b.sum()))
                row.append(dict(estimate=estimate, low=low, high=high,
                                n1=int(a.sum()), n0=int(b.sum())))
            records.append(row)
        result[str(n)] = records
    return result


def overlap_study(repetitions=30, n=2000, seed=528):
    result = {}
    for regime in ("randomized", "observational", "poor_overlap"):
        records = []
        for r in range(repetitions):
            data = randomized_data(n, seed + r, regime)
            t, y, e = data["t"], data["y"], data["e"]
            naive = float(y[t == 1].mean() - y[t == 0].mean())
            adjusted = float(ipw_effect(y, t, e).mean())
            weights = np.where(t == 1, 1 / e, 1 / (1 - e))
            ess = [float(weights[t == arm].sum() ** 2 / np.sum(weights[t == arm] ** 2)) for arm in (0, 1)]
            records.append(dict(naive=naive, ipw=adjusted, truth=float(data["tau"].mean()),
                                ess=ess, max_weight=float(weights.max())))
        first = randomized_data(n, seed, regime)
        result[regime] = dict(
            x=first["x"], t=first["t"], e=first["e"], y=first["y"],
            weights=np.where(first["t"] == 1, 1 / first["e"], 1 / (1 - first["e"])),
            records=records,
        )
    return result


def build_payload(seed=520, n_train=6000, n_test=3000, repetitions=30, grid_size=31):
    train = randomized_data(n_train, seed)
    test = randomized_data(n_test, seed + 1)
    models = fit_learners(train, seed)
    predictions = predict_learners(models, test["x"])
    axis = np.linspace(-2.5, 2.5, grid_size)
    a, b = np.meshgrid(axis, axis)
    grid_x = np.column_stack((a.ravel(), b.ravel()))
    p0, p1, tau = response_surfaces(grid_x)
    grid = {name: values.reshape(grid_size, grid_size) for name, values in
            dict(p0=p0, p1=p1, tau=tau, **predict_learners(models, grid_x)).items()}
    z = ipw_effect(test["y"], test["t"], test["e"])
    curves = {}
    for name, scores in (("uplift", predictions["t"]), ("response", predictions["t0"])):
        q, gains = gain_curve(scores, z)
        curves[name] = dict(q=q, gain=gains, **ranking_metrics(scores, z))
    # A synthetic potential-outcome pair is a simulation device, not RCT data.
    rng = np.random.default_rng(seed + 2)
    individual = dict(x=test["x"][0], p0=float(test["p0"][0]), p1=float(test["p1"][0]),
                      y0=int(rng.random() < test["p0"][0]), y1=int(rng.random() < test["p1"][0]),
                      t=int(rng.random() < 0.5))
    metrics = {
        "n_train": n_train, "n_test": n_test, "seed": seed,
        "train_treated": int(train["t"].sum()), "test_treated": int(test["t"].sum()),
        "ipw_ate": float(z.mean()), "synthetic_test_ate": float(test["tau"].mean()),
        "s_rmse": float(np.sqrt(np.mean((predictions["s"] - test["tau"]) ** 2))),
        "t_rmse": float(np.sqrt(np.mean((predictions["t"] - test["tau"]) ** 2))),
        "uplift_qini": curves["uplift"]["qini_area"], "response_qini": curves["response"]["qini_area"],
    }
    return dict(
        axis=axis, grid=grid, curves=curves, individual=individual, metrics=metrics,
        test=dict(x=test["x"], t=test["t"], y=test["y"], e=test["e"], tau=test["tau"],
                  z=z, score=predictions["t"], baseline=predictions["t0"]),
        segments=dict(names=SEGMENTS, p0=SEGMENT_P0, p1=SEGMENT_P1),
        sample_sizes=sample_size_study(repetitions, seed + 5),
        overlap=overlap_study(repetitions, seed=seed + 8),
        config=dict(seed=seed, n_train=n_train, n_test=n_test, repetitions=repetitions,
                    grid_size=grid_size, forest=dict(n_estimators=100, max_depth=7, min_samples_leaf=25, n_jobs=1)),
    )

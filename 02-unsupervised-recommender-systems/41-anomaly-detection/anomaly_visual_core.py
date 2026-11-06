"""Deterministic numerical scenarios for the Day 41 visual lab (no rendering)."""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

from example import anomaly_scores, evaluate_policy
from from_scratch import (
    EducationalIsolationForest, RobustMAD, average_path_length, calibrate_threshold,
    feature_matrix, path_length,
)


SEED = 141
SCALE = 1.482602218505602
PROFILES = {
    "quick": dict(train=240, calibration=100, normal_test=100, relationship=12,
                  extreme=8, grid=35, trees=50, robust_frames=12, monitoring_frames=18),
    "full": dict(train=600, calibration=240, normal_test=220, relationship=25,
                 extreme=15, grid=65, trees=160, robust_frames=24, monitoring_frames=40),
}


def robust_summary(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("values must be a finite vector with at least two entries")
    median = float(np.median(values))
    raw_mad = float(np.median(np.abs(values - median)))
    if raw_mad <= 0:
        raise ValueError("zero MAD: use an explicit domain rule for constant/discrete signals")
    return dict(mean=float(values.mean()), std=float(values.std(ddof=1)), median=median,
                raw_mad=raw_mad, robust_scale=SCALE * raw_mad)


def contamination_states(n_frames=24):
    base = np.random.default_rng(SEED).normal(250, 20, 100)
    states = []
    for strength in np.linspace(0, 1, n_frames):
        # Append a growing number of high-latency observations; keep the baseline fixed.
        n = int(round(10 * strength))
        injected = np.linspace(300 + 700 * strength, 320 + 1100 * strength, n)
        values = np.r_[base, injected]
        states.append(dict(values=values, n_outliers=n, **robust_summary(values)))
    return states


def telemetry_data(profile="full"):
    cfg = PROFILES[profile]
    rng = np.random.default_rng(SEED)

    def normal(n):
        z = rng.normal(size=(n, 2))
        return np.c_[250 + 35 * z[:, 0], 40 + 8 * (0.8 * z[:, 0] + 0.6 * z[:, 1])]

    train, cal, normal_test = normal(cfg["train"]), normal(cfg["calibration"]), normal(cfg["normal_test"])
    relation = rng.normal([285, 25], [12, 4], size=(cfg["relationship"], 2))
    extreme = rng.normal([375, 70], [15, 6], size=(cfg["extreme"], 2))
    test = np.vstack((normal_test, relation, extreme))
    groups = np.array(["normal"] * len(normal_test) + ["relationship"] * len(relation) + ["extreme"] * len(extreme))
    return train, cal, test, groups


def mesh_for(X, size=65, padding=0.12):
    X = feature_matrix(X)
    span = np.ptp(X, axis=0)
    if X.shape[1] != 2 or np.any(span <= 0) or size < 3:
        raise ValueError("mesh needs two varying features and size >= 3")
    low, high = X.min(axis=0) - padding * span, X.max(axis=0) + padding * span
    x, y = np.linspace(low[0], high[0], size), np.linspace(low[1], high[1], size)
    xx, yy = np.meshgrid(x, y)
    return x, y, np.c_[xx.ravel(), yy.ravel()]


def fit_visual_detectors(train, profile="full"):
    train = feature_matrix(train)
    return {
        "MAD": RobustMAD().fit(train),
        "IsolationForest": IsolationForest(n_estimators=PROFILES[profile]["trees"],
            max_samples=min(128, len(train)), random_state=SEED, n_jobs=1).fit(train),
        "OneClassSVM": make_pipeline(RobustScaler(), OneClassSVM(nu=0.05, gamma="scale")).fit(train),
    }


def covariance_distance(X, train):
    """Squared Mahalanobis distance; covariance and mean come only from train."""
    X, train = feature_matrix(X), feature_matrix(train)
    if train.shape[1] != X.shape[1] or len(train) <= train.shape[1]:
        raise ValueError("need matching dimensions and more training rows than features")
    covariance = np.atleast_2d(np.cov(train, rowvar=False, ddof=1))
    if np.linalg.matrix_rank(covariance) < train.shape[1]:
        raise ValueError("normal reference covariance is singular")
    delta = X - train.mean(axis=0)
    return np.sum(delta * np.linalg.solve(covariance, delta.T).T, axis=1)


def isolation_data():
    rng = np.random.default_rng(SEED)
    X = np.vstack((rng.normal([-1, 0], [0.38, 0.5], (22, 2)),
                   rng.normal([1.2, 1], [0.3, 0.4], (16, 2)), [[-3, 2.8], [3.2, -2.1]]))
    typical = int(np.argmin(np.linalg.norm(X - [-1, 0], axis=1)))
    return X, (typical, len(X) - 1)


def tree_trace(X, random_state=SEED):
    """Replay actual educational-tree nodes in recursive build order with clipped bounds."""
    X = feature_matrix(X)
    model = EducationalIsolationForest(n_estimators=1, max_samples=len(X), random_state=random_state).fit(X)
    x, y, _ = mesh_for(X, 3)
    events = []

    def walk(node, indices, bounds, depth):
        if node.feature is None:
            return
        left = X[indices, node.feature] <= node.threshold
        event = dict(feature=node.feature, threshold=node.threshold, bounds=list(bounds),
                     indices=indices.copy(), left=indices[left], right=indices[~left], depth=depth)
        events.append(event)
        left_bounds, right_bounds = list(bounds), list(bounds)
        # bounds are xmin,xmax,ymin,ymax; split endpoints stay in the parent cell.
        left_bounds[2 * node.feature + 1] = node.threshold
        right_bounds[2 * node.feature] = node.threshold
        walk(node.left, indices[left], left_bounds, depth + 1)
        walk(node.right, indices[~left], right_bounds, depth + 1)

    walk(model.trees_[0], np.arange(len(X)), [x[0], x[-1], y[0], y[-1]], 0)
    return model, events


def ensemble_paths(X, selected, n_trees=80):
    model = EducationalIsolationForest(n_estimators=n_trees, max_samples=len(X), random_state=SEED).fit(X)
    paths = np.array([[path_length(X[i], t) for i in selected] for t in model.trees_])
    running = np.cumsum(paths, axis=0) / np.arange(1, n_trees + 1)[:, None]
    scores = 2 ** (-running / average_path_length(model.sample_size_))
    return paths, running, scores


def ring_data(quick=False):
    rng = np.random.default_rng(SEED + 1)
    n = 120 if quick else 260
    angle = rng.uniform(0, 2 * np.pi, n)
    radius = rng.normal(1, 0.08, n)
    train = np.c_[250 + 80 * radius * np.cos(angle), 40 + 4 * radius * np.sin(angle)]
    # Both center-hole points and points beyond the ring are held out from fitting.
    anomaly = np.array([[250, 40], [265, 40.5], [370, 40], [250, 47], [130, 36]])
    return train, anomaly


def svm_configurations(train):
    configurations = []
    for scaled in (True, False):
        for gamma in (0.03, 0.3, 3.0):
            for nu in (0.01, 0.05, 0.15):
                svc = OneClassSVM(kernel="rbf", gamma=gamma, nu=nu)
                model = make_pipeline(RobustScaler(), svc) if scaled else svc
                model.fit(train)
                configurations.append(dict(model=model, support=train[svc.support_],
                    label=f"{'scaled' if scaled else 'raw units'} | gamma={gamma:g} | nu={nu:g}"))
    return configurations


def threshold_policies(models, train, calibration, test, groups, profile="full"):
    policies = []
    for name, model in models.items():
        cal_scores, scores = anomaly_scores(name, model, calibration), anomaly_scores(name, model, test)
        for alpha in (0.05, 0.02, 0.005):
            t = calibrate_threshold(cal_scores, alpha)
            policies.append(dict(label=f"{name} | calibration alpha={alpha:g}", name=name,
                scores=scores, cal_scores=cal_scores, alpha=alpha, **evaluate_policy(groups, scores, t)))
    # Same randomized partitions, different built-in offsets: identical ranking, different policies.
    for contamination in (0.01, 0.10):
        model = IsolationForest(n_estimators=PROFILES[profile]["trees"], max_samples=min(128, len(train)),
                                random_state=SEED, contamination=contamination, n_jobs=1).fit(train)
        scores, cal_scores = -model.decision_function(test), -model.decision_function(calibration)
        policies.append(dict(label=f"IF built-in contamination={contamination:g}", name="IsolationForest built-in",
            scores=scores, cal_scores=cal_scores, alpha=None, **evaluate_policy(groups, scores, 0.0)))
    return policies


def monitoring_data():
    rng = np.random.default_rng(SEED + 2)
    time = np.arange(360)
    mean = 250 + 75 * np.clip((time - 240) / 90, 0, 1)
    latency = mean + rng.normal(0, 12, len(time))
    incident = (time >= 180) & (time < 188)
    latency[incident] += 160
    return time, latency, incident, mean


APPROVED_UPDATES = (
    dict(activate=300, fit_start=240, fit_end=272, cal_start=272, cal_end=288),
    dict(activate=330, fit_start=270, fit_end=302, cal_start=302, cal_end=318),
)


def monitoring_policies(latency, updates=APPROVED_UPDATES):
    """Scripted operator-approved windows; no anomaly-label filtering or future values."""
    X = np.asarray(latency, dtype=float).reshape(-1, 1)
    if len(X) < 120 or not np.isfinite(X).all():
        raise ValueError("need at least 120 finite chronological observations")
    historical = RobustMAD().fit(X[:80])
    fixed_scores = historical.anomaly_score(X)
    fixed_threshold = calibrate_threshold(historical.anomaly_score(X[80:120]), 0.02)
    adaptive_scores = fixed_scores.copy()
    adaptive_threshold = np.full(len(X), fixed_threshold)
    records = []
    previous = 120
    for update in updates:
        t, a, b, c, d = (update[k] for k in ("activate", "fit_start", "fit_end", "cal_start", "cal_end"))
        if not 0 <= a < b <= c < d <= t or t <= previous:
            raise ValueError("updates need ordered disjoint fit/calibration windows ending before activation")
        previous = t
        if t >= len(X):
            continue
        model = RobustMAD().fit(X[a:b])
        threshold = calibrate_threshold(model.anomaly_score(X[c:d]), 0.02)
        adaptive_scores[t:] = model.anomaly_score(X[t:])
        adaptive_threshold[t:] = threshold
        records.append({**update, "center": float(model.center_[0]), "scale": float(model.scale_[0]),
                        "threshold": threshold, "approval": "illustrative operator-approved, 12-step delay"})
    return dict(fixed_scores=fixed_scores, fixed_threshold=np.full(len(X), fixed_threshold),
                adaptive_scores=adaptive_scores, adaptive_threshold=adaptive_threshold, updates=records)

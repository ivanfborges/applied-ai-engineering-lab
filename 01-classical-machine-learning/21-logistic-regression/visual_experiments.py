"""Small deterministic experiments shared by the Day 21 renderers and tests."""

from dataclasses import dataclass
import warnings

import numpy as np
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.datasets import make_classification, make_moons
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from from_scratch import binary_cross_entropy, classify, logit, loss_and_gradient, make_synthetic_data, sigmoid

SEED = 42


@dataclass
class ClassificationCase:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    model: object


def logistic_model(C=1.0, l1=False):
    """Keep the repository's sklearn >=1.4 range without deprecated arguments."""
    options = dict(C=C, solver="saga" if l1 else "lbfgs",
                   max_iter=8000, tol=1e-7, random_state=SEED)
    if l1:
        version = tuple(int(part) for part in sklearn.__version__.split(".")[:2])
        options.update({"l1_ratio": 1.0} if version >= (1, 8) else {"penalty": "l1"})
    return LogisticRegression(**options)


def fit_checked(model, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        return model.fit(X, y)


def classification_case():
    X, y = make_classification(n_samples=450, n_features=2, n_informative=2,
        n_redundant=0, n_clusters_per_class=1, class_sep=1.2,
        flip_y=0.08, random_state=SEED)
    train, test, yt, yv = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    model = fit_checked(make_pipeline(StandardScaler(), logistic_model()), train, yt)
    return ClassificationCase(train, test, yt, yv, model)


def raw_parameters(model):
    scale, classifier = model.steps[-2][1], model.steps[-1][1]
    coef = classifier.coef_[0] / scale.scale_
    return coef, float(classifier.intercept_[0] - coef @ scale.mean_)


def feature_grid(X, resolution=150):
    if resolution < 2:
        raise ValueError("resolution must be >=2")
    low, high = X.min(axis=0) - 0.6, X.max(axis=0) + 0.6
    xx, yy = np.meshgrid(np.linspace(low[0], high[0], resolution),
                         np.linspace(low[1], high[1], resolution))
    return xx, yy, np.column_stack((xx.ravel(), yy.ravel()))


def boundary_segment(coef, intercept, threshold, bounds):
    """Clip w.x+b=logit(t) to a rectangle, including vertical boundaries."""
    coef = np.asarray(coef, dtype=float)
    if coef.shape != (2,) or not np.isfinite(coef).all() or not np.isfinite(intercept):
        raise ValueError("expected two finite weights and a finite intercept")
    target = float(logit(threshold)) - intercept
    xmin, xmax, ymin, ymax = bounds
    points = []
    if abs(coef[1]) > 1e-12:
        for x in (xmin, xmax):
            y = (target - coef[0] * x) / coef[1]
            if ymin - 1e-10 <= y <= ymax + 1e-10:
                points.append((x, y))
    if abs(coef[0]) > 1e-12:
        for y in (ymin, ymax):
            x = (target - coef[1] * y) / coef[0]
            if xmin - 1e-10 <= x <= xmax + 1e-10:
                points.append((x, y))
    if not points:
        return np.empty((0, 2))
    points = np.unique(np.round(points, 12), axis=0)
    return points[[0, -1]]


def decision_counts(y, probabilities, threshold):
    predicted = classify(probabilities, threshold)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return dict(TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp),
                positives=int(predicted.sum()),
                precision=float(tp / (tp + fp)) if tp + fp else 0.0,
                recall=float(tp / (tp + fn)) if tp + fn else 0.0)


def gradient_trace(X, y, initial, *, steps=220, learning_rate=0.3, frames=36):
    """Unpenalized batch GD; retain selected parameters and the scalar losses."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    theta = np.asarray(initial, dtype=float).copy()
    if X.ndim != 2 or theta.shape != (X.shape[1] + 1,):
        raise ValueError("initial must contain one weight per feature and a bias")
    if (not isinstance(steps, int) or isinstance(steps, bool) or steps < 1
            or not isinstance(frames, int) or frames < 2
            or not np.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("positive integer steps, >=2 frames, and positive finite rate required")
    indices = np.unique(np.r_[0, np.geomspace(1, steps, frames - 1).astype(int), steps])
    snapshots, losses = [], []
    for step in range(steps + 1):
        loss, dw, db = loss_and_gradient(X, y, theta[:-1], theta[-1])
        losses.append(loss)
        if step in indices:
            snapshots.append(theta.copy())
        if step < steps:
            theta -= learning_rate * np.r_[dw, db]
    return dict(iterations=indices, parameters=np.array(snapshots),
                losses=np.array(losses), learning_rate=learning_rate)


def regularization_experiment():
    X, y = make_classification(n_samples=700, n_features=8, n_informative=3,
        n_redundant=2, n_clusters_per_class=2, flip_y=0.06, shuffle=False, random_state=SEED)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    X_train = StandardScaler().fit_transform(X_train)
    strengths = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
    paths = {}
    for name, l1 in (("L2", False), ("L1", True)):
        paths[name] = np.array([
            fit_checked(logistic_model(C=float(C), l1=l1), X_train, y_train).coef_[0]
            for C in strengths
        ])
    return strengths, paths


def nonlinear_experiment():
    X, y = make_moons(n_samples=600, noise=0.23, random_state=SEED)
    train, test, yt, yv = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    models = {
        "Raw features": make_pipeline(StandardScaler(), logistic_model(C=1)),
        "Degree-3 features": make_pipeline(PolynomialFeatures(3, include_bias=False),
                                           StandardScaler(), logistic_model(C=1)),
    }
    metrics = {}
    for name, model in models.items():
        fit_checked(model, train, yt)
        p = model.predict_proba(test)[:, 1]
        metrics[name] = dict(accuracy=float(accuracy_score(yv, p >= 0.5)),
                             log_loss=float(log_loss(yv, p)))
    return ClassificationCase(train, test, yt, yv, models["Raw features"]), models, metrics


def calibration_experiment():
    X, y = make_synthetic_data(2400, SEED)
    train, test, yt, yv = train_test_split(X, y, test_size=0.5, stratify=y, random_state=SEED)
    model = fit_checked(make_pipeline(StandardScaler(), logistic_model()), train, yt)
    scores = model.decision_function(test)
    series = {}
    for name, multiplier in (("Fitted logistic", 1.0), ("Same ranking, 2.5x logits", 2.5)):
        p = sigmoid(scores * multiplier)
        fraction, mean = calibration_curve(yv, p, n_bins=10, strategy="uniform")
        counts, _ = np.histogram(p, bins=np.linspace(0, 1, 11))
        series[name] = dict(probabilities=p, mean=mean, fraction=fraction, counts=counts,
            auc=float(roc_auc_score(yv, p)), brier=float(brier_score_loss(yv, p)),
            log_loss=float(log_loss(yv, p)))
    return series


def loss_landscape():
    rng = np.random.default_rng(SEED)
    X = np.repeat(np.linspace(-2.5, 2.5, 30), 4)[:, None]
    y = rng.binomial(1, sigmoid(0.9 * X[:, 0] - 0.35))
    trace = gradient_trace(X, y, [-1.3, 1.4], steps=160, learning_rate=0.3)
    weights, biases = np.meshgrid(np.linspace(-2, 3, 65), np.linspace(-2.5, 2.5, 65))
    logits = X[:, 0, None, None] * weights + biases
    loss = np.logaddexp(0, (1 - 2 * y[:, None, None]) * logits).mean(axis=0)
    return weights, biases, loss, trace


def finite_json(value):
    """Convert experiment arrays and reject nonfinite results before publication."""
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (np.ndarray, list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError("nonfinite experiment result")
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value
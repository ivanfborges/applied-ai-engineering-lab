"""Deterministic data and inspectable PCA states for the Day 39 visual lab."""

from numbers import Integral

import numpy as np
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def geometry_data():
    angle = np.deg2rad(33)
    basis = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    return (np.random.default_rng(42).normal(size=(100, 2)) * [2.5, .65]) @ basis + [2, 1]


def plane_data():
    basis, _ = np.linalg.qr(np.array([[1., -1., .2], [1., 1., -.5], [.7, .3, 1.]]))
    return (np.random.default_rng(43).normal(size=(100, 3)) * [3., 1.5, .22]) @ basis.T + [2, -1, 1]


def spectrum_data():
    rng = np.random.default_rng(44)
    latent = rng.normal(size=(180, 3)) * [3., 1.4, .7]
    mixing = np.array([[1, .8, .7, 0, .2, .3], [0, .2, .1, 1, .7, .8], [.1, 0, .3, .2, .5, -.4]])
    return latent @ mixing + rng.normal(0, .18, (180, 6))


def fit_view(X, standardize=False):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or min(X.shape) < 2 or not np.isfinite(X).all():
        raise ValueError("X must be finite 2D data with at least two rows and columns")
    scaler = StandardScaler().fit(X) if standardize else None
    working = scaler.transform(X) if scaler else X.copy()
    if np.sum(np.var(working, axis=0)) <= 0:
        raise ValueError("PCA needs positive total variance")
    model = PCA(svd_solver="full").fit(working)
    # Fix signs for legibility, without asserting they are mathematically unique.
    pivots = np.argmax(np.abs(model.components_), axis=1)
    model.components_ *= np.sign(model.components_[np.arange(len(pivots)), pivots])[:, None]
    return {"X": X, "working": working, "scaler": scaler, "model": model,
            "scores": model.transform(working)}


def reconstruct(view, k):
    model = view["model"]
    if (isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral)
            or not 1 <= k <= len(model.components_)):
        raise ValueError("k must be an integer in the fitted component range")
    scores = view["scores"][:, :k]
    working = scores @ model.components_[:k] + model.mean_
    original = view["scaler"].inverse_transform(working) if view["scaler"] else working
    return {"k": int(k), "Xhat": original, "working_hat": working,
            "evr": float(model.explained_variance_ratio_[:k].sum()),
            "mse": float(np.mean((view["working"] - working) ** 2)),
            "raw_mse": float(np.mean((view["X"] - original) ** 2))}


def threshold_count(view, threshold=.95):
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("threshold must be finite and strictly between zero and one")
    cumulative = np.cumsum(view["model"].explained_variance_ratio_)
    return min(int(np.searchsorted(cumulative, threshold, side="right")) + 1, len(cumulative))


def direction_variance(X, angle):
    v = np.array([np.cos(angle), np.sin(angle)])
    return float(np.var((X - X.mean(axis=0)) @ v, ddof=1))


def covariance_ellipse(view, radius=2.):
    model = view["model"]
    theta = np.linspace(0, 2 * np.pi, 120)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    return (circle * radius * np.sqrt(model.explained_variance_)) @ model.components_ + model.mean_


def scaling_views():
    rng = np.random.default_rng(45)
    latent = rng.normal(size=(130, 2))
    X = np.column_stack([100 * latent[:, 0], .6 * latent[:, 0] + .8 * latent[:, 1]])
    return fit_view(X), fit_view(X, standardize=True)


def predictive_experiment():
    """New visual-phase run; fixed models, independent of the study's 800 rows."""
    rng = np.random.default_rng(46)
    X = np.column_stack([rng.normal(0, 20, 500), rng.normal(0, .12, 500)])
    y = (X[:, 1] > 0).astype(int)
    train, test = train_test_split(np.arange(len(X)), test_size=.3, stratify=y, random_state=46)
    models = {
        "Original": make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000)),
        "Raw PCA: k=1": make_pipeline(PCA(1, svd_solver="full"), StandardScaler(),
                                     LogisticRegression(C=1, max_iter=2000)),
        "Raw PCA: k=2": make_pipeline(PCA(2, svd_solver="full"), StandardScaler(),
                                     LogisticRegression(C=1, max_iter=2000)),
    }
    results = {}
    for name, model in models.items():
        model.fit(X[train], y[train])
        results[name] = float(balanced_accuracy_score(y[test], model.predict(X[test])))
    return {"X": X, "y": y, "train": train, "test": test,
            "view": fit_view(X[train]), "models": models, "balanced_accuracy": results}


def outlier_states():
    clean = geometry_data()
    # Actual augmented datasets at each step; no interpolation of fitted axes.
    outliers = np.array([[3., 18.], [2., 22.], [1., 25.], [-1., 28.]])
    return [(fit_view(np.vstack([clean, outliers[:k]])), k) for k in range(5)]


def wine_weights():
    wine = load_wine()
    train, _ = train_test_split(np.arange(len(wine.data)), test_size=.25,
                               stratify=wine.target, random_state=42)
    view = fit_view(wine.data[train], standardize=True)
    return view, list(wine.feature_names)

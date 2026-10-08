"""Synthetic data and inspectable squared-error boosting history for Day 29."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


@dataclass(frozen=True)
class RegressionData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    x_grid: np.ndarray
    truth_grid: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    reserved_indices: np.ndarray


@dataclass(frozen=True)
class BoostingHistory:
    """Rows 0..M are ensemble states; correction row m-1 belongs to update m."""

    predictions_train: np.ndarray
    predictions_validation: np.ndarray
    predictions_grid: np.ndarray
    residuals_train: np.ndarray
    residuals_validation: np.ndarray
    corrections_train: np.ndarray
    corrections_grid: np.ndarray
    train_rmse: np.ndarray
    validation_rmse: np.ndarray
    trees: tuple
    learning_rate: float

    @property
    def best_stage(self) -> int:
        return int(np.argmin(self.validation_rmse))


def true_function(x):
    return 2.0 * np.sin(x) + 0.4 * x ** 2


def make_regression_data(seed: int = 42) -> RegressionData:
    """Same generator/split as example.py; reserved test rows are never scored here."""
    x = np.linspace(-3.0, 3.0, 300).reshape(-1, 1)
    y = true_function(x[:, 0]) + np.random.default_rng(seed).normal(0, 0.35, len(x))
    development, reserved = train_test_split(np.arange(len(x)), test_size=0.2, random_state=seed)
    train, validation = train_test_split(development, test_size=0.25, random_state=seed)
    grid = np.linspace(-3.0, 3.0, 240).reshape(-1, 1)
    return RegressionData(
        x[train], y[train], x[validation], y[validation],
        grid, true_function(grid[:, 0]), train, validation, reserved,
    )


def _positive_integer(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _validate(data, n_estimators, learning_rate, max_depth, min_samples_leaf):
    for name, value in (
        ("n_estimators", n_estimators), ("max_depth", max_depth),
        ("min_samples_leaf", min_samples_leaf),
    ):
        _positive_integer(name, value)
    if (
        isinstance(learning_rate, (bool, np.bool_))
        or not isinstance(learning_rate, Real)
        or not np.isfinite(learning_rate)
        or not 0 < learning_rate <= 1
    ):
        raise ValueError("learning_rate must be finite and in (0, 1].")
    features = (data.x_train, data.x_validation, data.x_grid)
    for values in features:
        if values.ndim != 2 or min(values.shape) == 0 or not np.isfinite(values).all():
            raise ValueError("Features must be nonempty finite 2D arrays.")
    if len({values.shape[1] for values in features}) != 1:
        raise ValueError("Feature counts must agree across training, validation, and grid.")
    for x, y in ((data.x_train, data.y_train), (data.x_validation, data.y_validation)):
        if y.shape != (len(x),) or not np.isfinite(y).all():
            raise ValueError("Targets must be finite 1D arrays aligned with observations.")


def fit_history(
    data: RegressionData,
    n_estimators: int = 100,
    learning_rate: float = 0.2,
    max_depth: int = 2,
    min_samples_leaf: int = 3,
    seed: int = 42,
) -> BoostingHistory:
    """Implement the boosting loop; delegate only the weak-tree fit to sklearn."""
    _validate(data, n_estimators, learning_rate, max_depth, min_samples_leaf)
    initial = float(data.y_train.mean())
    train = np.full(len(data.y_train), initial)
    validation = np.full(len(data.y_validation), initial)
    grid = np.full(len(data.x_grid), initial)
    train_states, validation_states, grid_states = [train.copy()], [validation.copy()], [grid.copy()]
    train_corrections, grid_corrections, trees = [], [], []
    for _ in range(n_estimators):
        residuals = data.y_train - train
        weak_learner = DecisionTreeRegressor(
            max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=seed
        ).fit(data.x_train, residuals)
        correction = weak_learner.predict(data.x_train)
        grid_correction = weak_learner.predict(data.x_grid)
        train += learning_rate * correction
        validation += learning_rate * weak_learner.predict(data.x_validation)
        grid += learning_rate * grid_correction
        train_corrections.append(correction)
        grid_corrections.append(grid_correction)
        train_states.append(train.copy())
        validation_states.append(validation.copy())
        grid_states.append(grid.copy())
        trees.append(weak_learner)
    predictions_train = np.stack(train_states)
    predictions_validation = np.stack(validation_states)
    residuals_train = data.y_train - predictions_train
    residuals_validation = data.y_validation - predictions_validation
    return BoostingHistory(
        predictions_train, predictions_validation, np.stack(grid_states),
        residuals_train, residuals_validation, np.stack(train_corrections),
        np.stack(grid_corrections),
        np.sqrt(np.mean(residuals_train ** 2, axis=1)),
        np.sqrt(np.mean(residuals_validation ** 2, axis=1)),
        tuple(trees), float(learning_rate),
    )


def fit_library_path(data, learning_rate=0.1, max_depth=2, n_estimators=300, min_samples_leaf=3):
    """Return actual library predictions/metrics with a manually prepended baseline."""
    _validate(data, n_estimators, learning_rate, max_depth, min_samples_leaf)
    model = GradientBoostingRegressor(
        learning_rate=learning_rate, max_depth=max_depth,
        n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
        subsample=1.0, loss="squared_error", random_state=42, n_iter_no_change=None,
    ).fit(data.x_train, data.y_train)
    initial = float(data.y_train.mean())
    curves = []
    for x, y in ((data.x_train, data.y_train), (data.x_validation, data.y_validation)):
        predictions = np.vstack([
            np.full((1, len(x)), initial),
            np.stack(list(model.staged_predict(x))),
        ])
        curves.append(np.sqrt(np.mean((y - predictions) ** 2, axis=1)))
    return {
        "model": model, "train_rmse": curves[0], "validation_rmse": curves[1],
        "best_stage": int(np.argmin(curves[1])),
        "learning_rate": learning_rate, "max_depth": max_depth,
        "n_estimators": n_estimators, "min_samples_leaf": min_samples_leaf,
    }


def squared_loss(y, prediction):
    return 0.5 * (np.asarray(y) - np.asarray(prediction)) ** 2


def logistic_loss(y, logit):
    """Binary cross-entropy in stable logit form."""
    y, logit = np.asarray(y, dtype=float), np.asarray(logit, dtype=float)
    if y.shape != logit.shape or not np.isin(y, [0, 1]).all() or not np.isfinite(logit).all():
        raise ValueError("Provide aligned binary labels and finite logits.")
    return np.logaddexp(0.0, logit) - y * logit


def logistic_negative_gradient(y, logit):
    y, logit = np.asarray(y, dtype=float), np.asarray(logit, dtype=float)
    logistic_loss(y, logit)
    probability = np.exp(-np.logaddexp(0.0, -logit))
    return y - probability


def numerical_negative_gradient(loss, y, prediction, step=1e-5):
    """Central finite difference with respect to each prediction."""
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step must be finite and positive.")
    prediction = np.asarray(prediction, dtype=float)
    return -(loss(y, prediction + step) - loss(y, prediction - step)) / (2.0 * step)


def path_summary(path):
    """JSON-ready measured curves and configuration, excluding the fitted estimator."""
    stage = path["best_stage"]
    return {
        "configuration": {key: path[key] for key in (
            "learning_rate", "max_depth", "n_estimators", "min_samples_leaf"
        )},
        "best_stage": stage,
        "train_rmse_at_best": float(path["train_rmse"][stage]),
        "validation_rmse_at_best": float(path["validation_rmse"][stage]),
        "train_rmse_at_final": float(path["train_rmse"][-1]),
        "validation_rmse_at_final": float(path["validation_rmse"][-1]),
        "train_rmse": path["train_rmse"].tolist(),
        "validation_rmse": path["validation_rmse"].tolist(),
    }


def save_history(path, data, history):
    """Persist inspectable numeric arrays without pickling trained estimators."""
    np.savez_compressed(
        path, x_train=data.x_train, y_train=data.y_train,
        x_validation=data.x_validation, y_validation=data.y_validation,
        x_grid=data.x_grid, truth_grid=data.truth_grid,
        train_indices=data.train_indices, validation_indices=data.validation_indices,
        reserved_indices=data.reserved_indices,
        predictions_train=history.predictions_train,
        predictions_validation=history.predictions_validation,
        predictions_grid=history.predictions_grid,
        residuals_train=history.residuals_train,
        residuals_validation=history.residuals_validation,
        corrections_train=history.corrections_train,
        corrections_grid=history.corrections_grid,
        train_rmse=history.train_rmse, validation_rmse=history.validation_rmse,
        learning_rate=history.learning_rate,
    )

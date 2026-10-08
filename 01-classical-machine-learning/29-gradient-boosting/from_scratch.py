"""Educational squared-error boosting with exhaustive one-dimensional stumps."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _feature(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("x must be a nonempty finite array of shape (n,) or (n, 1).")
    return x


def _target(y, n: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.shape != (n,) or not np.isfinite(y).all():
        raise ValueError("y must be a finite 1D array with one value per observation.")
    return y


class DecisionStumpRegressor:
    """Choose a single split by squared error; ties keep the first improvement.

    Leaf predictions are target means. A constant leaf is retained when no
    valid split improves on the mean, including constant features or targets.
    """

    def __init__(self, min_samples_leaf: int = 1):
        _positive_integer("min_samples_leaf", min_samples_leaf)
        self.min_samples_leaf = min_samples_leaf

    def fit(self, x, y) -> DecisionStumpRegressor:
        x = _feature(x)
        y = _target(y, len(x))
        mean = float(y.mean())
        best_sse = float(np.sum((y - mean) ** 2))
        threshold, left_value, right_value = None, mean, mean
        values = np.unique(x)
        for low, high in zip(values[:-1], values[1:]):
            # Half-sums avoid overflowing low + high; rounding must not empty a leaf.
            candidate = low / 2.0 + high / 2.0
            if candidate >= high:
                candidate = low
            left = x <= candidate
            if min(int(left.sum()), int((~left).sum())) < self.min_samples_leaf:
                continue
            left_mean, right_mean = float(y[left].mean()), float(y[~left].mean())
            sse = float(
                np.sum((y[left] - left_mean) ** 2)
                + np.sum((y[~left] - right_mean) ** 2)
            )
            if sse < best_sse:
                best_sse = sse
                threshold, left_value, right_value = float(candidate), left_mean, right_mean
        self.threshold_ = threshold
        self.left_value_, self.right_value_ = left_value, right_value
        return self

    def predict(self, x) -> np.ndarray:
        if not hasattr(self, "threshold_"):
            raise RuntimeError("Fit the stump before prediction.")
        x = _feature(x)
        if self.threshold_ is None:
            return np.full(len(x), self.left_value_)
        return np.where(x <= self.threshold_, self.left_value_, self.right_value_)


class SimpleGradientBoostingRegressor:
    """Full-data, unweighted squared-error boosting on one numeric feature.

    The boosting loop and stump search are implemented with NumPy. This teaching
    model omits interactions, other losses, sampling, and optimized tree growth.
    """

    def __init__(
        self,
        n_estimators: int = 20,
        learning_rate: float = 0.1,
        min_samples_leaf: int = 1,
    ):
        _positive_integer("n_estimators", n_estimators)
        _positive_integer("min_samples_leaf", min_samples_leaf)
        if (
            isinstance(learning_rate, (bool, np.bool_))
            or not isinstance(learning_rate, Real)
            or not np.isfinite(learning_rate)
            or not 0 < learning_rate <= 1
        ):
            raise ValueError("learning_rate must be finite and in (0, 1].")
        self.n_estimators = n_estimators
        self.learning_rate = float(learning_rate)
        self.min_samples_leaf = min_samples_leaf

    def fit(self, x, y) -> SimpleGradientBoostingRegressor:
        x = _feature(x)
        y = _target(y, len(x))
        initial = float(y.mean())
        prediction = np.full(len(x), initial)
        estimators = []
        train_mse = [float(np.mean((y - prediction) ** 2))]
        for _ in range(self.n_estimators):
            # For L = (y - F)^2 / 2, the negative prediction gradient is y - F.
            residual = y - prediction
            stump = DecisionStumpRegressor(self.min_samples_leaf).fit(x, residual)
            prediction += self.learning_rate * stump.predict(x)
            estimators.append(stump)
            train_mse.append(float(np.mean((y - prediction) ** 2)))
        self.initial_prediction_ = initial
        self.estimators_ = estimators
        self.train_mse_ = np.asarray(train_mse)
        return self

    def staged_predict(self, x):
        """Yield independent arrays for stage 0 (mean), then stages 1 through M.

        Unlike sklearn.staged_predict, this includes the constant baseline.
        """
        if not hasattr(self, "estimators_"):
            raise RuntimeError("Fit the model before prediction.")
        x = _feature(x)
        prediction = np.full(len(x), self.initial_prediction_)
        yield prediction.copy()
        for stump in self.estimators_:
            prediction += self.learning_rate * stump.predict(x)
            yield prediction.copy()

    def predict(self, x) -> np.ndarray:
        for prediction in self.staged_predict(x):
            pass
        return prediction


def main() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 1.0, 5.0, 5.0])
    model = SimpleGradientBoostingRegressor(n_estimators=3, learning_rate=0.5).fit(x, y)
    print("Synthetic four-row example; training-only mechanics, not generalization evidence.")
    for stage, prediction in enumerate(model.staged_predict(x)):
        print(f"Stage {stage}: predictions={prediction.tolist()}, MSE={model.train_mse_[stage]:.6f}")


if __name__ == "__main__":
    main()
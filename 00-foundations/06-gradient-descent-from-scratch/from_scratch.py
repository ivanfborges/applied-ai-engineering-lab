"""Educational linear regression optimized with batch gradient descent."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


class LinearRegressionGD:
    """Fit linear regression with full-batch gradient descent.

    The optimized objective is half mean squared error:

        J(w, b) = (1 / 2n) * sum((Xw + b - y) ** 2)

    The implementation records the state before the first update and after
    every update. It is intended for study, not as a production estimator.
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        max_iterations: int = 2_000,
        tolerance: float = 1e-8,
        fit_intercept: bool = True,
    ) -> None:
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be a finite positive number.")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be greater than zero.")
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be a finite non-negative number.")

        self.learning_rate = float(learning_rate)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.fit_intercept = bool(fit_intercept)

        self.coef_: FloatArray | None = None
        self.intercept_: float | None = None
        self.n_features_in_: int | None = None
        self.n_iterations_: int = 0
        self.converged_: bool = False

        self.loss_history_: list[float] = []
        self.gradient_norm_history_: list[float] = []
        self.coefficient_history_: list[FloatArray] = []
        self.intercept_history_: list[float] = []

    @staticmethod
    def _prepare_features(x: ArrayLike) -> FloatArray:
        features = np.asarray(x, dtype=np.float64)

        if features.ndim == 1:
            features = features.reshape(-1, 1)
        if features.ndim != 2:
            raise ValueError("x must be a one- or two-dimensional array.")
        if features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError("x must contain at least one row and one feature.")
        if not np.all(np.isfinite(features)):
            raise ValueError("x contains NaN or infinite values.")

        return features

    @staticmethod
    def _prepare_target(y: ArrayLike, expected_rows: int) -> FloatArray:
        target = np.asarray(y, dtype=np.float64)

        if target.ndim == 2 and target.shape[1] == 1:
            target = target[:, 0]
        if target.ndim != 1:
            raise ValueError("y must be one-dimensional.")
        if target.shape[0] != expected_rows:
            raise ValueError("x and y must contain the same number of rows.")
        if not np.all(np.isfinite(target)):
            raise ValueError("y contains NaN or infinite values.")

        return target

    def fit(self, x: ArrayLike, y: ArrayLike) -> "LinearRegressionGD":
        """Estimate coefficients and retain convergence diagnostics."""
        features = self._prepare_features(x)
        target = self._prepare_target(y, features.shape[0])
        n_samples, n_features = features.shape

        coefficients = np.zeros(n_features, dtype=np.float64)
        intercept = 0.0

        self.n_features_in_ = n_features
        self.n_iterations_ = 0
        self.converged_ = False
        self.loss_history_ = []
        self.gradient_norm_history_ = []
        self.coefficient_history_ = []
        self.intercept_history_ = []

        # max_iterations counts parameter updates. The extra evaluation records
        # the state produced by the final permitted update.
        for iteration in range(self.max_iterations + 1):
            predictions = features @ coefficients + intercept
            residuals = predictions - target

            loss = float(np.dot(residuals, residuals) / (2.0 * n_samples))
            coefficient_gradient = features.T @ residuals / n_samples
            intercept_gradient = (
                float(np.mean(residuals)) if self.fit_intercept else 0.0
            )
            gradient_norm = float(
                np.sqrt(
                    np.dot(coefficient_gradient, coefficient_gradient)
                    + intercept_gradient**2
                )
            )

            if not (
                np.isfinite(loss)
                and np.all(np.isfinite(coefficient_gradient))
                and np.isfinite(intercept_gradient)
                and np.isfinite(gradient_norm)
            ):
                raise FloatingPointError(
                    "Optimization became non-finite. Reduce the learning rate "
                    "or scale the features."
                )

            self.loss_history_.append(loss)
            self.gradient_norm_history_.append(gradient_norm)
            self.coefficient_history_.append(coefficients.copy())
            self.intercept_history_.append(intercept)

            if gradient_norm <= self.tolerance:
                self.converged_ = True
                break
            if iteration == self.max_iterations:
                break

            # Both gradients come from the same parameter state.
            coefficients -= self.learning_rate * coefficient_gradient
            intercept -= self.learning_rate * intercept_gradient
            self.n_iterations_ = iteration + 1

        self.coef_ = coefficients
        self.intercept_ = intercept
        return self

    def predict(self, x: ArrayLike) -> FloatArray:
        """Predict targets using fitted coefficients."""
        if (
            self.coef_ is None
            or self.intercept_ is None
            or self.n_features_in_ is None
        ):
            raise RuntimeError("The model must be fitted before prediction.")

        features = self._prepare_features(x)
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, "
                f"received {features.shape[1]}."
            )

        return features @ self.coef_ + self.intercept_


def _smoke_test() -> None:
    """Fit an exact line and verify the implementation's basic behavior."""
    x = np.linspace(-2.0, 2.0, 25)
    y = 3.0 * x - 1.0

    model = LinearRegressionGD(
        learning_rate=0.1,
        max_iterations=2_000,
        tolerance=1e-10,
    ).fit(x, y)

    if model.coef_ is None or model.intercept_ is None:
        raise RuntimeError("Fit did not create model parameters.")
    if not np.allclose(model.coef_, [3.0], atol=1e-6):
        raise RuntimeError("Coefficient smoke test failed.")
    if not np.isclose(model.intercept_, -1.0, atol=1e-6):
        raise RuntimeError("Intercept smoke test failed.")

    print("Smoke test passed.")
    print(f"Coefficient: {model.coef_[0]:.6f}")
    print(f"Intercept:   {model.intercept_:.6f}")
    print(f"Updates:     {model.n_iterations_}")
    print(f"Converged:   {model.converged_}")


if __name__ == "__main__":
    _smoke_test()


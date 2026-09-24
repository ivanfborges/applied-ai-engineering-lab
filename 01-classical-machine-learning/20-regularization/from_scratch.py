"""Educational dense, single-target ElasticNet via proximal gradient descent."""

from dataclasses import dataclass

import numpy as np


def _matrix(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or min(X.shape) == 0 or not np.isfinite(X).all():
        raise ValueError("X must be a nonempty finite 2D matrix.")
    return X


def _nonnegative(value, name):
    if np.ndim(value) != 0 or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative scalar.")


def soft_threshold(values, threshold):
    """Proximal operator for threshold * ||w||_1, including exact zeros."""
    values = np.asarray(values, dtype=float)
    _nonnegative(threshold, "threshold")
    if not np.isfinite(values).all():
        raise ValueError("values must be finite.")
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


@dataclass
class ProximalFit:
    coef: np.ndarray
    intercept: float
    objective_history: np.ndarray
    n_iter: int
    converged: bool
    kkt_residual: float

    def predict(self, X):
        X = _matrix(X)
        if X.shape[1] != self.coef.size:
            raise ValueError("X has the wrong number of features.")
        return X @ self.coef + self.intercept


def fit_elastic_net(X, y, *, alpha=0.1, l1_ratio=0.5, tol=1e-8, max_iter=50_000):
    """Minimize SSE/(2n) + alpha*rho*L1 + alpha*(1-rho)*L2_squared/2.

    Center internally to leave the intercept unpenalized. Feature units are
    preserved: callers must fit any scaling on training data only. Stopping
    uses an absolute KKT residual, not a small parameter update. Exhausting
    max_iter returns converged=False, which callers must inspect.
    """
    X = _matrix(X)
    y = np.asarray(y, dtype=float)
    if y.shape != (len(X),) or not np.isfinite(y).all():
        raise ValueError("y must be a finite 1D vector with one value per row.")
    _nonnegative(alpha, "alpha")
    _nonnegative(l1_ratio, "l1_ratio")
    if l1_ratio > 1:
        raise ValueError("l1_ratio must be between 0 and 1.")
    _nonnegative(tol, "tol")
    if tol == 0:
        raise ValueError("tol must be positive.")
    if (
        isinstance(max_iter, (bool, np.bool_))
        or not isinstance(max_iter, (int, np.integer))
        or max_iter < 1
    ):
        raise ValueError("max_iter must be a positive integer.")

    x_mean, y_mean = X.mean(axis=0), float(y.mean())
    centered_X, centered_y = X - x_mean, y - y_mean
    n = len(X)
    l1, l2 = alpha * l1_ratio, alpha * (1 - l1_ratio)
    # The spectral bound gives a safe step for the smooth part of the loss.
    lipschitz = np.linalg.norm(centered_X, ord=2) ** 2 / n + l2
    step = 1.0 / lipschitz if lipschitz > 0 else 1.0
    weights = np.zeros(X.shape[1])
    history = []

    for iteration in range(max_iter + 1):
        residual = centered_X @ weights - centered_y
        objective = (
            residual @ residual / (2 * n)
            + l1 * np.abs(weights).sum()
            + l2 * (weights @ weights) / 2
        )
        gradient = centered_X.T @ residual / n + l2 * weights
        violation = np.where(
            weights != 0,
            np.abs(gradient + l1 * np.sign(weights)),
            np.maximum(np.abs(gradient) - l1, 0),
        )
        kkt_residual = float(violation.max())
        if not np.isfinite(objective) or not np.isfinite(kkt_residual):
            raise FloatingPointError("Nonfinite optimization state; rescale the data.")
        history.append(float(objective))
        converged = kkt_residual <= tol
        if converged or iteration == max_iter:
            break
        weights = soft_threshold(weights - step * gradient, step * l1)

    return ProximalFit(
        coef=weights,
        intercept=float(y_mean - x_mean @ weights),
        objective_history=np.asarray(history),
        n_iter=iteration,
        converged=converged,
        kkt_residual=kkt_residual,
    )


def main():
    rng = np.random.default_rng(20)
    X = rng.normal(size=(300, 4))
    y = 3 + X @ np.array([4.0, 0.0, -2.0, 0.0]) + rng.normal(size=300)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    fitted = fit_elastic_net(X, y, alpha=0.1, l1_ratio=0.8)
    if not fitted.converged:
        raise RuntimeError("The demonstration did not converge.")
    print("Synthetic optimization demonstration; no generalization evaluation.")
    print("Coefficients:", np.round(fitted.coef, 6))
    print(f"Intercept: {fitted.intercept:.6f}")
    print(f"Iterations: {fitted.n_iter}; KKT residual: {fitted.kkt_residual:.3e}")
    print(f"Objective: {fitted.objective_history[0]:.6f} -> {fitted.objective_history[-1]:.6f}")


if __name__ == "__main__":
    main()

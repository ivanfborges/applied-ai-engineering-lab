"""Educational dense, single-target OLS and batch GD; callers handle scaling."""

from dataclasses import dataclass
import numpy as np


def _features(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or min(X.shape) == 0 or not np.isfinite(X).all():
        raise ValueError("X must be a nonempty finite matrix of shape (n, p).")
    return X


def _training_data(X, y):
    X = _features(X)
    y = np.asarray(y, dtype=float)
    if y.shape != (len(X),) or not np.isfinite(y).all():
        raise ValueError("y must be a finite vector of shape (n,).")
    return np.column_stack((np.ones(len(X)), X)), y


def _objective(A, y, beta):
    with np.errstate(over="raise", invalid="raise"):
        error = A @ beta - y
        loss = float(error @ error / (2 * len(y)))
        gradient = A.T @ error / len(y)
    if not np.isfinite(loss) or not np.isfinite(gradient).all():
        raise FloatingPointError("Nonfinite objective; rescale data or lower the step size.")
    return loss, gradient


def loss_gradient(X, y, beta):
    """Return half-MSE and its gradient; beta includes the intercept first."""
    A, y = _training_data(X, y)
    beta = np.asarray(beta, dtype=float)
    if beta.shape != (A.shape[1],) or not np.isfinite(beta).all():
        raise ValueError("beta must be a finite vector of length p + 1.")
    return _objective(A, y, beta)


@dataclass
class LinearFit:
    """Parameters and optional solver diagnostics; no sklearn estimator protocol."""

    beta: np.ndarray
    rank: int | None = None
    loss_history: np.ndarray | None = None
    n_iter: int = 0
    converged: bool | None = None

    @property
    def intercept(self):
        return float(self.beta[0])

    @property
    def coef(self):
        return self.beta[1:]

    def predict(self, X):
        X = _features(X)
        if X.shape[1] != len(self.coef):
            raise ValueError("Prediction features must match the training feature count.")
        with np.errstate(over="raise", invalid="raise"):
            return self.intercept + X @ self.coef


def fit_ols(X, y):
    """Solve OLS without forming X.T @ X; report augmented design rank."""
    A, y = _training_data(X, y)
    beta, _, rank, _ = np.linalg.lstsq(A, y, rcond=None)
    if not np.isfinite(beta).all():
        raise FloatingPointError("Nonfinite coefficients; rescale the inputs.")
    return LinearFit(beta=beta, rank=int(rank))


def fit_gradient_descent(X, y, *, learning_rate=0.1, max_iter=10000, tol=1e-8):
    """Minimize half-MSE from zero using full-batch gradients.

    Stop when the gradient infinity norm is at most tol. History includes
    the initial loss and every update, including the returned iterate.
    Budget exhaustion returns converged=False; it is not success.
    """
    A, y = _training_data(X, y)
    for name, value in (("learning_rate", learning_rate), ("tol", tol)):
        if not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive scalar.")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")
    beta = np.zeros(A.shape[1])
    history = []
    for iteration in range(max_iter + 1):
        loss, gradient = _objective(A, y, beta)
        history.append(loss)
        converged = bool(np.linalg.norm(gradient, ord=np.inf) <= tol)
        if converged or iteration == max_iter:
            break
        with np.errstate(over="raise", invalid="raise"):
            beta = beta - learning_rate * gradient
    return LinearFit(beta, loss_history=np.array(history), n_iter=iteration, converged=converged)

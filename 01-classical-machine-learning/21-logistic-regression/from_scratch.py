"""Educational binary logistic regression on small, finite, dense arrays."""

from dataclasses import dataclass

import numpy as np


def _finite(values, name):
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _matrix(X):
    X = _finite(X, "X")
    if X.ndim != 2 or min(X.shape) == 0:
        raise ValueError("X must be a nonempty 2D matrix")
    return X


def _targets(y, n):
    y = _finite(y, "y")
    if y.shape != (n,) or not np.all((y == 0) | (y == 1)):
        raise ValueError("y must be a vector of 0/1 labels matching X")
    return y


def _scalar(value, name, strict=False):
    value = _finite(value, name)
    if value.ndim != 0 or value < 0 or (strict and value == 0):
        raise ValueError(f"{name} must be a nonnegative scalar (positive if required)")
    return float(value)


def sigmoid(z):
    """Evaluate sigmoid without exponentiating large positive numbers."""
    z = _finite(z, "z")
    decay = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + decay), decay / (1.0 + decay))


def logit(p):
    """Map probabilities strictly inside (0, 1) to log-odds."""
    p = _finite(p, "p")
    if np.any((p <= 0) | (p >= 1)):
        raise ValueError("p must be strictly between 0 and 1")
    return np.log(p) - np.log1p(-p)


def classify(probabilities, threshold=0.5):
    """Apply a fixed threshold; equality belongs to the positive class."""
    probabilities = _finite(probabilities, "probabilities")
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("probabilities must be a nonempty vector")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be in [0, 1]")
    threshold = _scalar(threshold, "threshold", strict=True)
    if threshold >= 1:
        raise ValueError("threshold must be strictly between 0 and 1")
    return (probabilities >= threshold).astype(int)


def binary_cross_entropy(y, logits):
    """Mean Bernoulli negative log-likelihood, evaluated directly from logits."""
    logits = _finite(logits, "logits")
    if logits.ndim != 1 or logits.size == 0:
        raise ValueError("logits must be a nonempty vector")
    y = _targets(y, len(logits))
    # Signed margins avoid log(0) and subtracting two large equal terms.
    return float(np.mean(np.logaddexp(0.0, (1.0 - 2.0 * y) * logits)))


def loss_and_gradient(X, y, coef, intercept, l2=0.0):
    """Return mean BCE + l2/2 * ||coef||^2 and its two gradients."""
    X = _matrix(X)
    y = _targets(y, len(X))
    coef = _finite(coef, "coef")
    intercept = _finite(intercept, "intercept")
    l2 = _scalar(l2, "l2")
    if coef.shape != (X.shape[1],) or intercept.ndim != 0:
        raise ValueError("coef must match X columns and intercept must be scalar")
    with np.errstate(over="raise", invalid="raise"):
        logits = X @ coef + intercept
        error = sigmoid(logits) - y
        loss = binary_cross_entropy(y, logits) + 0.5 * l2 * (coef @ coef)
        return float(loss), X.T @ error / len(X) + l2 * coef, float(error.mean())


@dataclass
class LogisticFit:
    coef: np.ndarray
    intercept: float
    objective_history: np.ndarray
    n_iter: int
    converged: bool
    gradient_norm: float
    learning_rate: float

    def decision_function(self, X):
        X = _matrix(X)
        if X.shape[1] != len(self.coef):
            raise ValueError("X has the wrong number of features")
        with np.errstate(over="raise", invalid="raise"):
            return X @ self.coef + self.intercept

    def predict_proba(self, X):
        """Return a 1D vector P(y=1), unlike sklearn's two-column output."""
        return sigmoid(self.decision_function(X))

    def predict(self, X, threshold=0.5):
        return classify(self.predict_proba(X), threshold)


def fit_logistic(X, y, *, l2=0.02, max_iter=10_000, tol=1e-8):
    """Batch gradient descent with an unpenalized intercept and a safe step.

    Requires both classes. Features are not scaled internally. Convergence is
    measured by the infinity norm of the full gradient, not classification.
    """
    X = _matrix(X)
    y = _targets(y, len(X))
    if np.unique(y).size != 2:
        raise ValueError("training requires both binary classes")
    l2 = _scalar(l2, "l2")
    tol = _scalar(tol, "tol", strict=True)
    if (isinstance(max_iter, (bool, np.bool_))
            or not isinstance(max_iter, (int, np.integer)) or max_iter < 1):
        raise ValueError("max_iter must be a positive integer")
    augmented = np.column_stack((X, np.ones(len(X))))
    # p(1-p) <= 1/4 bounds the Hessian, including the intercept coordinate.
    with np.errstate(over="raise", invalid="raise"):
        bound = np.linalg.norm(augmented, ord=2) ** 2 / (4 * len(X)) + l2
    learning_rate = 1.0 / bound
    coef, intercept = np.zeros(X.shape[1]), 0.0
    history = []
    for iteration in range(max_iter + 1):
        loss, dw, db = loss_and_gradient(X, y, coef, intercept, l2)
        history.append(loss)
        gradient_norm = float(max(np.max(np.abs(dw)), abs(db)))
        converged = gradient_norm <= tol
        if converged or iteration == max_iter:
            break
        coef -= learning_rate * dw
        intercept -= learning_rate * db
    return LogisticFit(coef, intercept, np.asarray(history), iteration,
                       converged, gradient_norm, learning_rate)


def make_synthetic_data(n_samples=1200, seed=21):
    """Gaussian features and stochastic Bernoulli labels, with arbitrary units.

    X = latent * [2, 10] + [3, -4].
    The generating logit is -0.4 + 1.4*latent_1 - 1.1*latent_2.
    """
    if (isinstance(n_samples, (bool, np.bool_))
            or not isinstance(n_samples, (int, np.integer)) or n_samples < 2):
        raise ValueError("n_samples must be an integer >= 2")
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n_samples, 2))
    probabilities = sigmoid(-0.4 + latent @ np.array([1.4, -1.1]))
    y = rng.binomial(1, probabilities)
    return latent * [2.0, 10.0] + [3.0, -4.0], y


def main():
    X, y = make_synthetic_data()
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    fitted = fit_logistic(X, y)
    print("Synthetic optimization demonstration: all 1200 rows used for training.")
    print("This is a training-objective check, not a generalization estimate.")
    print(f"Objective: {fitted.objective_history[0]:.8f} -> {fitted.objective_history[-1]:.8f}")
    print(f"Iterations: {fitted.n_iter}; converged: {fitted.converged}")
    print(f"Gradient infinity norm: {fitted.gradient_norm:.3e}")
    print(f"Step: {fitted.learning_rate:.6f}")
    if not fitted.converged:
        raise RuntimeError("Optimization did not converge; inspect scaling and budget")


if __name__ == "__main__":
    main()
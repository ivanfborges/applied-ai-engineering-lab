"""Educational NumPy implementation of logistic-regression MLE and MAP."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LogisticFit:
    """Store fitted weights and optimizer diagnostics."""

    weights: np.ndarray
    iterations: int
    objective: float


def sigmoid(values: float | np.ndarray) -> float | np.ndarray:
    """Compute a numerically stable logistic sigmoid."""
    array = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(array)):
        raise ValueError("values must be finite.")
    result = np.empty_like(array)
    nonnegative = array >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-array[nonnegative]))
    exponential = np.exp(array[~nonnegative])
    result[~nonnegative] = exponential / (1.0 + exponential)
    return float(result) if result.ndim == 0 else result


def _validate_data(
    features: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if features.ndim != 2 or features.shape[0] < 2 or features.shape[1] < 1:
        raise ValueError("features must be a 2D array with at least two rows.")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("labels must be a 1D array aligned with features.")
    if np.any(~np.isfinite(features)) or np.any(~np.isfinite(labels)):
        raise ValueError("features and labels must contain only finite values.")
    if np.any((labels != 0.0) & (labels != 1.0)):
        raise ValueError("labels must contain only 0 and 1.")
    return features, labels


def _validate_weights(weights: np.ndarray, feature_count: int) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (feature_count,):
        raise ValueError("weights must have one value per feature column.")
    if np.any(~np.isfinite(weights)):
        raise ValueError("weights must contain only finite values.")
    return weights


def _validate_prior_std(prior_std: float | None) -> float | None:
    if prior_std is None:
        return None
    prior_std = float(prior_std)
    if not math.isfinite(prior_std) or prior_std <= 0.0:
        raise ValueError("prior_std must be finite and positive when provided.")
    return prior_std


def negative_log_posterior(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    prior_std: float | None = None,
    reduction: str = "sum",
) -> float:
    """Return Bernoulli NLL plus a Gaussian-prior penalty.

    The first weight is treated as an intercept and is not assigned the prior.
    With ``reduction='mean'``, the complete summed posterior objective is
    divided by the sample size. This preserves the same MAP optimum.
    """
    features, labels = _validate_data(features, labels)
    weights = _validate_weights(weights, features.shape[1])
    prior_std = _validate_prior_std(prior_std)
    if reduction not in {"sum", "mean"}:
        raise ValueError("reduction must be 'sum' or 'mean'.")

    logits = features @ weights
    objective = float(np.sum(np.logaddexp(0.0, logits) - labels * logits))
    if prior_std is not None:
        objective += float(np.dot(weights[1:], weights[1:])) / (2.0 * prior_std**2)
    if reduction == "mean":
        objective /= features.shape[0]
    return objective


def negative_log_posterior_gradient(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    prior_std: float | None = None,
    reduction: str = "sum",
) -> np.ndarray:
    """Return the analytical gradient of ``negative_log_posterior``."""
    features, labels = _validate_data(features, labels)
    weights = _validate_weights(weights, features.shape[1])
    prior_std = _validate_prior_std(prior_std)
    if reduction not in {"sum", "mean"}:
        raise ValueError("reduction must be 'sum' or 'mean'.")

    gradient = features.T @ (sigmoid(features @ weights) - labels)
    if prior_std is not None:
        gradient[1:] += weights[1:] / prior_std**2
    if reduction == "mean":
        gradient /= features.shape[0]
    return gradient


def fit_logistic_regression(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    prior_std: float | None = None,
    learning_rate: float = 0.2,
    max_iterations: int = 10_000,
    tolerance: float = 1e-10,
) -> LogisticFit:
    """Fit logistic regression with batch gradient descent.

    This implementation is intentionally educational. Production work should
    use a mature optimizer with convergence diagnostics and tested solvers.
    """
    features, labels = _validate_data(features, labels)
    prior_std = _validate_prior_std(prior_std)
    learning_rate = float(learning_rate)
    tolerance = float(tolerance)
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive.")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive.")

    weights = np.zeros(features.shape[1], dtype=float)
    for iteration in range(1, max_iterations + 1):
        gradient = negative_log_posterior_gradient(
            features,
            labels,
            weights,
            prior_std=prior_std,
            reduction="mean",
        )
        updated = weights - learning_rate * gradient
        if np.linalg.norm(updated - weights, ord=2) <= tolerance:
            weights = updated
            break
        weights = updated

    return LogisticFit(
        weights=weights.copy(),
        iterations=iteration,
        objective=negative_log_posterior(
            features,
            labels,
            weights,
            prior_std=prior_std,
            reduction="sum",
        ),
    )


def generate_synthetic_classification(
    *, seed: int = 14, sample_size: int = 600
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate deterministic synthetic Bernoulli outcomes with three features."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    if isinstance(sample_size, bool) or not isinstance(sample_size, int):
        raise TypeError("sample_size must be an integer.")
    if sample_size < 20:
        raise ValueError("sample_size must be at least 20.")

    rng = np.random.default_rng(seed)
    raw_features = rng.normal(size=(sample_size, 3))
    features = np.column_stack((np.ones(sample_size), raw_features))
    true_weights = np.array([-0.35, 1.4, -1.1, 0.65])
    probabilities = sigmoid(features @ true_weights)
    labels = rng.binomial(1, probabilities).astype(float)
    return features, labels, true_weights


def classification_metrics(
    features: np.ndarray, labels: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    """Return mean NLL and threshold-0.5 accuracy."""
    features, labels = _validate_data(features, labels)
    weights = _validate_weights(weights, features.shape[1])
    mean_nll = negative_log_posterior(
        features, labels, weights, reduction="mean"
    )
    predictions = (sigmoid(features @ weights) >= 0.5).astype(float)
    return mean_nll, float(np.mean(predictions == labels))


def main() -> None:
    """Compare MLE with two Gaussian-prior MAP estimates."""
    features, labels, true_weights = generate_synthetic_classification()
    train_features, validation_features = features[:450], features[450:]
    train_labels, validation_labels = labels[:450], labels[450:]
    configurations = (("MLE", None), ("MAP sigma=2.0", 2.0), ("MAP sigma=0.5", 0.5))

    print("Synthetic logistic regression (450 train, 150 validation, seed=14)")
    print(f"Generator weights: {np.array2string(true_weights, precision=4)}")
    print("The following are outcomes of this constructed dataset, not benchmarks.")
    for label, prior_std in configurations:
        fit = fit_logistic_regression(
            train_features, train_labels, prior_std=prior_std
        )
        train_nll, train_accuracy = classification_metrics(
            train_features, train_labels, fit.weights
        )
        validation_nll, validation_accuracy = classification_metrics(
            validation_features, validation_labels, fit.weights
        )
        slope_norm = float(np.linalg.norm(fit.weights[1:]))
        print()
        print(label)
        print(f"  weights: {np.array2string(fit.weights, precision=4)}")
        print(f"  slope L2 norm: {slope_norm:.6f}")
        print(f"  train mean NLL / accuracy: {train_nll:.6f} / {train_accuracy:.4f}")
        print(
            "  validation mean NLL / accuracy: "
            f"{validation_nll:.6f} / {validation_accuracy:.4f}"
        )
        print(f"  optimizer iterations: {fit.iterations}")


if __name__ == "__main__":
    main()

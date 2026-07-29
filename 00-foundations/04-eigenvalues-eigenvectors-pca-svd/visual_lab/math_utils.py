"""Explicit linear-algebra utilities shared by every visualization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PCAResult:
    """Values needed to inspect and reuse a fitted PCA transformation."""

    mean: np.ndarray
    components: np.ndarray
    scores: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray


def _as_finite_matrix(matrix: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validate_n_components(
    n_components: int,
    n_samples: int,
    n_features: int,
) -> None:
    if not isinstance(n_components, int) or isinstance(n_components, bool):
        raise TypeError("n_components must be an integer.")
    if not 1 <= n_components <= min(n_samples, n_features):
        raise ValueError(
            "n_components must be between 1 and min(n_samples, n_features)."
        )


def center_data(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a feature-centered copy of data and its original feature means."""
    matrix = _as_finite_matrix(data, "data")
    mean = matrix.mean(axis=0)
    return matrix - mean, mean


def covariance_matrix(data: np.ndarray, *, assume_centered: bool = False) -> np.ndarray:
    """Compute the sample covariance matrix with denominator n - 1."""
    matrix = _as_finite_matrix(data, "data")
    if matrix.shape[0] < 2:
        raise ValueError("At least two samples are required for sample covariance.")
    centered = matrix if assume_centered else center_data(matrix)[0]
    return centered.T @ centered / (centered.shape[0] - 1)


def eigendecomposition_pca(
    data: np.ndarray,
    n_components: int,
) -> PCAResult:
    """Fit PCA through symmetric covariance eigendecomposition."""
    matrix = _as_finite_matrix(data, "data")
    _validate_n_components(n_components, *matrix.shape)
    centered, mean = center_data(matrix)
    covariance = covariance_matrix(centered, assume_centered=True)

    # eigh exploits covariance symmetry and returns eigenvalues in ascending order.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    total_variance = float(eigenvalues.sum())
    if np.isclose(total_variance, 0.0):
        raise ValueError("PCA requires at least one feature with nonzero variance.")

    components = eigenvectors[:, :n_components].T
    scores = project_data(centered, components)
    return PCAResult(
        mean=mean,
        components=components,
        scores=scores,
        explained_variance=eigenvalues[:n_components],
        explained_variance_ratio=eigenvalues[:n_components] / total_variance,
    )


def svd_pca(data: np.ndarray, n_components: int) -> PCAResult:
    """Fit PCA through direct SVD of the centered data matrix."""
    matrix = _as_finite_matrix(data, "data")
    _validate_n_components(n_components, *matrix.shape)
    centered, mean = center_data(matrix)
    _, singular_values, right_singular_vectors_t = np.linalg.svd(
        centered,
        full_matrices=False,
    )
    explained_variance = explained_variance_from_singular_values(
        singular_values,
        matrix.shape[0],
    )
    total_variance = float(explained_variance.sum())
    if np.isclose(total_variance, 0.0):
        raise ValueError("PCA requires at least one feature with nonzero variance.")

    components = right_singular_vectors_t[:n_components]
    return PCAResult(
        mean=mean,
        components=components,
        scores=project_data(centered, components),
        explained_variance=explained_variance[:n_components],
        explained_variance_ratio=(
            explained_variance[:n_components] / total_variance
        ),
    )


def project_data(data: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Project rows of data onto PCA components stored as row vectors."""
    matrix = _as_finite_matrix(data, "data")
    basis = _as_finite_matrix(components, "components")
    if matrix.shape[1] != basis.shape[1]:
        raise ValueError("data and components must share the feature dimension.")
    return matrix @ basis.T


def reconstruct_data(
    scores: np.ndarray,
    components: np.ndarray,
    mean: np.ndarray,
) -> np.ndarray:
    """Reconstruct observations from component scores and an original mean."""
    score_matrix = _as_finite_matrix(scores, "scores")
    basis = _as_finite_matrix(components, "components")
    mean_vector = np.asarray(mean, dtype=float)
    if mean_vector.ndim != 1 or not np.all(np.isfinite(mean_vector)):
        raise ValueError("mean must be a finite one-dimensional vector.")
    if score_matrix.shape[1] != basis.shape[0]:
        raise ValueError("scores columns must match the number of components.")
    if basis.shape[1] != mean_vector.shape[0]:
        raise ValueError("components and mean must share the feature dimension.")
    return score_matrix @ basis + mean_vector


def explained_variance_from_singular_values(
    singular_values: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """Convert singular values of centered data into sample variances.

    Cov(X) = X.T @ X / (n - 1). If X = U @ Sigma @ V.T, then
    Cov(X) = V @ Sigma**2 @ V.T / (n - 1), so lambda_i = sigma_i**2 / (n - 1).
    """
    values = np.asarray(singular_values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("singular_values must be a nonempty one-dimensional array.")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("singular_values must be finite and nonnegative.")
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2.")
    return values**2 / (n_samples - 1)


def low_rank_approximation(
    matrix: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the rank-k truncated-SVD reconstruction and singular values."""
    array = _as_finite_matrix(matrix, "matrix")
    max_rank = min(array.shape)
    if not isinstance(rank, int) or isinstance(rank, bool):
        raise TypeError("rank must be an integer.")
    if not 1 <= rank <= max_rank:
        raise ValueError(f"rank must be between 1 and {max_rank}.")

    left, singular_values, right_t = np.linalg.svd(array, full_matrices=False)
    approximation = (
        left[:, :rank] * singular_values[:rank]
    ) @ right_t[:rank, :]
    return approximation, singular_values


def align_component_signs(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> np.ndarray:
    """Flip candidate rows to align them with corresponding reference rows.

    Eigenvectors and singular vectors are defined only up to sign. Opposite
    signs represent the same one-dimensional subspace and are not an error.
    """
    reference_matrix = _as_finite_matrix(reference, "reference")
    candidate_matrix = _as_finite_matrix(candidate, "candidate")
    if reference_matrix.shape != candidate_matrix.shape:
        raise ValueError("reference and candidate must have the same shape.")

    aligned = candidate_matrix.copy()
    row_dots = np.sum(reference_matrix * aligned, axis=1)
    aligned[row_dots < 0] *= -1.0
    return aligned


def approximate_svd_compression_ratio(
    shape: tuple[int, int],
    rank: int,
) -> float:
    """Estimate dense-value compression from storing U_k, sigma_k, and V_k.T."""
    rows, columns = shape
    if not 1 <= rank <= min(shape):
        raise ValueError("rank is outside the matrix dimensions.")
    original_values = rows * columns
    truncated_values = rows * rank + rank + rank * columns
    return original_values / truncated_values

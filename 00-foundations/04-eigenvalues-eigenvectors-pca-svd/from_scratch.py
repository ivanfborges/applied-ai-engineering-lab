"""Educational PCA using covariance eigendecomposition with NumPy."""

from __future__ import annotations

import numpy as np


class PCAFromScratch:
    """A small PCA implementation for learning, not production use."""

    def __init__(self, n_components: int) -> None:
        if not isinstance(n_components, int) or isinstance(n_components, bool):
            raise TypeError("n_components must be an integer.")
        if n_components <= 0:
            raise ValueError("n_components must be greater than zero.")

        self.n_components = n_components
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    def fit(self, data: np.ndarray) -> "PCAFromScratch":
        """Fit principal directions to a dense numeric matrix."""
        matrix = self._validate_matrix(data, name="data")
        n_samples, n_features = matrix.shape

        if n_samples < 2:
            raise ValueError("At least two observations are required.")
        if self.n_components > n_features:
            raise ValueError("n_components cannot exceed the feature count.")

        self.n_features_in_ = n_features
        self.mean_ = matrix.mean(axis=0)
        centered = matrix - self.mean_
        covariance = centered.T @ centered / (n_samples - 1)

        # eigh is specialized for symmetric matrices and returns ascending values.
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        descending_order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[descending_order]
        eigenvectors = eigenvectors[:, descending_order]

        # A covariance matrix is positive semidefinite. Clipping only removes
        # tiny negative values that may appear from floating-point roundoff.
        eigenvalues = np.maximum(eigenvalues, 0.0)
        total_variance = float(eigenvalues.sum())
        if np.isclose(total_variance, 0.0):
            raise ValueError("PCA is undefined when every feature has zero variance.")

        self.components_ = eigenvectors[:, : self.n_components].T
        self.explained_variance_ = eigenvalues[: self.n_components]
        self.explained_variance_ratio_ = (
            self.explained_variance_ / total_variance
        )
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Project observations into the fitted principal-component space."""
        self._check_is_fitted()
        matrix = self._validate_matrix(data, name="data")
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("data has a different feature count from the fit data.")

        return (matrix - self.mean_) @ self.components_.T

    def inverse_transform(self, scores: np.ndarray) -> np.ndarray:
        """Map component scores back to the original feature coordinates."""
        self._check_is_fitted()
        matrix = self._validate_matrix(scores, name="scores")
        if matrix.shape[1] != self.n_components:
            raise ValueError("scores must have n_components columns.")

        return matrix @ self.components_ + self.mean_

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit the model and return component scores."""
        return self.fit(data).transform(data)

    @staticmethod
    def _validate_matrix(data: np.ndarray, name: str) -> np.ndarray:
        matrix = np.asarray(data, dtype=float)
        if matrix.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional matrix.")
        if matrix.shape[1] == 0:
            raise ValueError(f"{name} must contain at least one feature.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} must contain only finite values.")
        return matrix

    def _check_is_fitted(self) -> None:
        if (
            self.mean_ is None
            or self.components_ is None
            or self.explained_variance_ is None
            or self.n_features_in_ is None
        ):
            raise RuntimeError("The PCA instance has not been fitted.")


def generate_synthetic_data(
    n_samples: int = 300,
    random_state: int = 42,
) -> np.ndarray:
    """Create a synthetic matrix with one dominant shared direction."""
    rng = np.random.default_rng(random_state)
    shared_factor = rng.normal(size=n_samples)
    return np.column_stack(
        [
            shared_factor + rng.normal(scale=0.1, size=n_samples),
            2.0 * shared_factor + rng.normal(scale=0.2, size=n_samples),
            -0.5 * shared_factor + rng.normal(scale=0.1, size=n_samples),
        ]
    )


def main() -> None:
    data = generate_synthetic_data()
    pca = PCAFromScratch(n_components=2)
    scores = pca.fit_transform(data)
    reconstructed = pca.inverse_transform(scores)

    centered = data - data.mean(axis=0)
    _, singular_values, right_singular_vectors_t = np.linalg.svd(
        centered,
        full_matrices=False,
    )
    variance_from_svd = singular_values**2 / (data.shape[0] - 1)
    direction_alignment = np.abs(
        pca.components_ @ right_singular_vectors_t[:2].T
    )

    reconstruction_squared_error = float(
        np.linalg.norm(data - reconstructed, ord="fro") ** 2
    )
    discarded_squared_singular_values = float(
        np.sum(singular_values[2:] ** 2)
    )

    print("Synthetic data shape:", data.shape)
    print("Reduced shape:", scores.shape)
    print("\nComponents (rows):\n", np.round(pca.components_, 6))
    print("\nExplained variance:", np.round(pca.explained_variance_, 6))
    print(
        "Explained variance ratio:",
        np.round(pca.explained_variance_ratio_, 6),
    )
    print("\nAbsolute SVD direction alignment:\n", np.round(direction_alignment, 6))
    print(
        "Squared reconstruction error:",
        round(reconstruction_squared_error, 6),
    )
    print(
        "Sum of discarded squared singular values:",
        round(discarded_squared_singular_values, 6),
    )

    assert np.allclose(pca.explained_variance_, variance_from_svd[:2])
    assert np.allclose(direction_alignment, np.eye(2), atol=1e-10)
    assert np.isclose(
        reconstruction_squared_error,
        discarded_squared_singular_values,
    )


if __name__ == "__main__":
    main()

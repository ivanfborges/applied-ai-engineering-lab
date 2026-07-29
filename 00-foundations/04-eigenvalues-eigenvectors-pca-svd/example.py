"""Demonstrate PCA and verify its relationship with direct SVD."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def generate_synthetic_data(
    n_samples: int = 500,
    random_state: int = 42,
) -> np.ndarray:
    """Return three correlated synthetic features driven by two latent factors."""
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2.")

    rng = np.random.default_rng(random_state)
    latent_1 = rng.normal(size=n_samples)
    latent_2 = rng.normal(scale=0.5, size=n_samples)
    noise = rng.normal(scale=0.15, size=(n_samples, 3))

    features = np.column_stack(
        [
            2.0 * latent_1 + 0.2 * latent_2,
            1.7 * latent_1 + 0.5 * latent_2,
            -0.6 * latent_1 + 1.2 * latent_2,
        ]
    )
    return features + noise


def main() -> None:
    data = generate_synthetic_data()

    # The data is synthetic. Scaling is explicit here so each feature contributes
    # in standard-deviation units; this is a modeling choice, not a PCA rule.
    scaled_data = StandardScaler().fit_transform(data)
    pca = PCA(n_components=2)
    scores = pca.fit_transform(scaled_data)
    reconstructed = pca.inverse_transform(scores)

    # PCA centers internally. The scaled data is already centered up to rounding,
    # but centering again makes the mathematical comparison explicit.
    centered_data = scaled_data - scaled_data.mean(axis=0)
    _, singular_values, right_singular_vectors_t = np.linalg.svd(
        centered_data,
        full_matrices=False,
    )
    variance_from_svd = singular_values**2 / (centered_data.shape[0] - 1)

    # Signs are arbitrary, so absolute dot products compare component directions.
    direction_alignment = np.abs(
        pca.components_ @ right_singular_vectors_t[:2].T
    )
    reconstruction_mse = float(np.mean((scaled_data - reconstructed) ** 2))

    print("Synthetic data shape:", data.shape)
    print("Reduced shape:", scores.shape)
    print("\nPCA directions (rows):\n", np.round(pca.components_, 6))
    print("\nExplained variance:", np.round(pca.explained_variance_, 6))
    print(
        "Explained variance from SVD:",
        np.round(variance_from_svd[:2], 6),
    )
    print(
        "Explained variance ratio:",
        np.round(pca.explained_variance_ratio_, 6),
    )
    print(
        "Cumulative explained variance:",
        round(float(pca.explained_variance_ratio_.sum()), 6),
    )
    print("\nAbsolute direction alignment:\n", np.round(direction_alignment, 6))
    print("Two-component reconstruction MSE:", round(reconstruction_mse, 6))

    # These checks encode the key PCA-SVD identities.
    assert np.allclose(pca.explained_variance_, variance_from_svd[:2])
    assert np.allclose(direction_alignment, np.eye(2), atol=1e-10)
    assert np.allclose(pca.singular_values_, singular_values[:2])


if __name__ == "__main__":
    main()

"""Deterministic synthetic datasets used by the visual laboratory."""

from __future__ import annotations

import numpy as np


def correlated_2d(
    n_samples: int = 320,
    correlation: float = 0.88,
    noise: float = 0.35,
    seed: int = 42,
) -> np.ndarray:
    """Create a rotated, correlated two-dimensional point cloud."""
    if n_samples < 10:
        raise ValueError("n_samples must be at least 10.")
    if not -0.99 <= correlation <= 0.99:
        raise ValueError("correlation must be between -0.99 and 0.99.")
    if noise < 0:
        raise ValueError("noise must be nonnegative.")

    rng = np.random.default_rng(seed)
    covariance = np.array([[1.0, correlation], [correlation, 1.0]])
    latent = rng.multivariate_normal(np.zeros(2), covariance, size=n_samples)
    transform = np.array([[1.8, 0.35], [0.25, 0.85]])
    return latent @ transform.T + rng.normal(scale=noise, size=(n_samples, 2))


def correlated_3d(
    n_samples: int = 260,
    noise: float = 0.16,
    seed: int = 7,
) -> np.ndarray:
    """Create 3D data driven by two latent variables plus isotropic noise."""
    if n_samples < 10:
        raise ValueError("n_samples must be at least 10.")
    if noise < 0:
        raise ValueError("noise must be nonnegative.")

    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n_samples, 2))
    latent[:, 1] *= 0.55
    mixing = np.array(
        [
            [1.9, 0.25],
            [1.4, -0.75],
            [0.55, 1.35],
        ]
    )
    offset = np.array([0.7, -0.4, 0.9])
    return latent @ mixing.T + offset + rng.normal(
        scale=noise,
        size=(n_samples, 3),
    )


def correlated_features(
    n_samples: int = 500,
    n_features: int = 8,
    n_latent: int = 3,
    noise: float = 0.12,
    seed: int = 21,
) -> np.ndarray:
    """Create a dense feature matrix from a small latent-factor model."""
    if n_samples < 10:
        raise ValueError("n_samples must be at least 10.")
    if n_features < 2:
        raise ValueError("n_features must be at least 2.")
    if not 1 <= n_latent <= n_features:
        raise ValueError("n_latent must be between 1 and n_features.")
    if noise < 0:
        raise ValueError("noise must be nonnegative.")

    rng = np.random.default_rng(seed)
    latent_scales = np.linspace(2.0, 0.65, n_latent)
    latent = rng.normal(size=(n_samples, n_latent)) * latent_scales
    mixing = rng.normal(size=(n_latent, n_features))
    mixing /= np.linalg.norm(mixing, axis=0, keepdims=True)
    return latent @ mixing + rng.normal(
        scale=noise,
        size=(n_samples, n_features),
    )


def synthetic_grayscale_image(size: int = 96) -> np.ndarray:
    """Build a reproducible grayscale image from smooth geometric patterns."""
    if size < 32:
        raise ValueError("size must be at least 32.")

    y, x = np.mgrid[0:size, 0:size]
    x_norm = x / (size - 1)
    y_norm = y / (size - 1)

    image = 0.14 + 0.32 * x_norm + 0.16 * y_norm
    image += 0.18 * np.sin(4.0 * np.pi * (x_norm + 0.35 * y_norm))

    circle = (x_norm - 0.27) ** 2 + (y_norm - 0.30) ** 2 <= 0.14**2
    image[circle] += 0.42

    rectangle = (
        (x_norm > 0.55)
        & (x_norm < 0.88)
        & (y_norm > 0.16)
        & (y_norm < 0.36)
    )
    image[rectangle] -= 0.32

    # Three bars provide text-like high-contrast structure.
    for start in (0.18, 0.28, 0.38):
        bar = (
            (x_norm > start)
            & (x_norm < start + 0.055)
            & (y_norm > 0.62)
            & (y_norm < 0.86)
        )
        image[bar] += 0.28

    diagonal = np.abs(y_norm - (0.92 - 0.55 * x_norm)) < 0.025
    image[diagonal] += 0.30
    return np.clip(image, 0.0, 1.0)

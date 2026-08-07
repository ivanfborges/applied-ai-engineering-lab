"""Lightweight numerical and artifact tests for the visual laboratory."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

TOPIC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC_ROOT))

from visual_lab.datasets import correlated_3d, correlated_features
from visual_lab.generate_visuals import (
    ensure_output_directories,
    expected_output_paths,
)
from visual_lab.math_utils import (
    align_component_signs,
    center_data,
    covariance_matrix,
    eigendecomposition_pca,
    explained_variance_from_singular_values,
    low_rank_approximation,
    reconstruct_data,
)


def test_reconstructed_svd_matrix_matches_original() -> None:
    matrix = correlated_features(n_samples=80, n_features=8, seed=101)
    approximation, _ = low_rank_approximation(matrix, rank=min(matrix.shape))
    assert np.allclose(approximation, matrix, atol=1e-10)


def test_covariance_eigenvalues_match_svd_variances() -> None:
    data = correlated_features(seed=102)
    centered, _ = center_data(data)
    eigenvalues = np.linalg.eigvalsh(
        covariance_matrix(centered, assume_centered=True)
    )[::-1]
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    svd_variances = explained_variance_from_singular_values(
        singular_values,
        data.shape[0],
    )
    assert np.allclose(eigenvalues, svd_variances, atol=1e-10)


def test_pca_components_are_orthonormal() -> None:
    result = eigendecomposition_pca(correlated_3d(), n_components=3)
    identity = np.eye(result.components.shape[0])
    assert np.allclose(result.components @ result.components.T, identity)


def test_component_score_covariance_is_diagonal() -> None:
    result = eigendecomposition_pca(
        correlated_features(seed=103),
        n_components=8,
    )
    score_covariance = covariance_matrix(result.scores, assume_centered=True)
    off_diagonal = score_covariance - np.diag(np.diag(score_covariance))
    assert np.allclose(off_diagonal, 0.0, atol=1e-10)


def test_reconstruction_error_is_monotonic_with_rank() -> None:
    matrix = correlated_features(n_samples=120, seed=104)
    errors = []
    for rank in range(1, min(matrix.shape) + 1):
        approximation, _ = low_rank_approximation(matrix, rank)
        errors.append(float(np.mean((matrix - approximation) ** 2)))
    assert np.all(np.diff(errors) <= 1e-12)


def test_from_scratch_pca_subspace_matches_sklearn() -> None:
    data = correlated_features(seed=105)
    n_components = 3
    scratch = eigendecomposition_pca(data, n_components)
    sklearn_model = PCA(n_components=n_components).fit(data)
    aligned = align_component_signs(scratch.components, sklearn_model.components_)

    assert np.allclose(scratch.components, aligned, atol=1e-10)
    scratch_projection = scratch.components.T @ scratch.components
    sklearn_projection = sklearn_model.components_.T @ sklearn_model.components_
    assert np.allclose(scratch_projection, sklearn_projection, atol=1e-10)

    centered, _ = center_data(data)
    scratch_reconstruction = reconstruct_data(
        centered @ scratch.components.T,
        scratch.components,
        scratch.mean,
    )
    sklearn_reconstruction = sklearn_model.inverse_transform(
        sklearn_model.transform(data)
    )
    assert np.allclose(scratch_reconstruction, sklearn_reconstruction, atol=1e-10)


def test_output_manifest_is_bounded_and_directories_are_created(tmp_path) -> None:
    output_root = tmp_path / "outputs"
    directories = ensure_output_directories(output_root)
    assert all(directory.is_dir() for directory in directories)

    expected = expected_output_paths(output_root)
    assert len(expected) == 13
    assert len(expected) == len(set(expected))
    assert all(path.is_relative_to(output_root) for path in expected)

"""Shared numerical and data-generation utilities for the visualization lab."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

DEFAULT_RANDOM_SEED = 42
FLOAT_ARRAY = NDArray[np.float64]


def _as_vector(vector: Iterable[float] | np.ndarray, name: str) -> FLOAT_ARRAY:
    """Convert input to a finite, one-dimensional floating-point vector."""
    array = np.asarray(vector, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _as_matrix(matrix: Iterable[Iterable[float]] | np.ndarray, name: str) -> FLOAT_ARRAY:
    """Convert input to a finite, non-empty floating-point matrix."""
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional; got shape {array.shape}.")
    if 0 in array.shape:
        raise ValueError(f"{name} must have at least one row and one column.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def normalize_vector(vector: Iterable[float] | np.ndarray) -> FLOAT_ARRAY:
    """Return a unit-length copy of a nonzero vector."""
    array = _as_vector(vector, "vector")
    norm = float(np.linalg.norm(array))
    if np.isclose(norm, 0.0):
        raise ValueError("Cannot normalize a zero vector.")
    return array / norm


def calculate_cosine_similarity(
    vector_a: Iterable[float] | np.ndarray,
    vector_b: Iterable[float] | np.ndarray,
) -> float:
    """Calculate cosine similarity for two nonzero vectors of equal shape."""
    a = _as_vector(vector_a, "vector_a")
    b = _as_vector(vector_b, "vector_b")
    if a.shape != b.shape:
        raise ValueError(f"Vectors must have equal shape; got {a.shape} and {b.shape}.")
    return float(np.dot(normalize_vector(a), normalize_vector(b)))


def project_onto_vector(
    vector: Iterable[float] | np.ndarray,
    direction: Iterable[float] | np.ndarray,
) -> FLOAT_ARRAY:
    """Orthogonally project a vector onto a nonzero direction."""
    x = _as_vector(vector, "vector")
    u = _as_vector(direction, "direction")
    if x.shape != u.shape:
        raise ValueError(f"Inputs must have equal shape; got {x.shape} and {u.shape}.")
    denominator = float(np.dot(u, u))
    if np.isclose(denominator, 0.0):
        raise ValueError("Projection direction must be nonzero.")
    return (float(np.dot(u, x)) / denominator) * u


def gram_schmidt(
    vectors: Iterable[Iterable[float]] | np.ndarray,
    tolerance: float = 1e-10,
) -> FLOAT_ARRAY:
    """Return an orthonormal basis for independent input columns.

    This is classical Gram-Schmidt for educational use. Householder QR or SVD
    is preferable for production numerical work with ill-conditioned inputs.
    """
    matrix = _as_matrix(vectors, "vectors")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    if matrix.shape[1] > matrix.shape[0]:
        raise ValueError("More columns than ambient dimensions cannot be independent.")

    orthonormal_columns: list[FLOAT_ARRAY] = []
    for column_index in range(matrix.shape[1]):
        original = matrix[:, column_index].copy()
        orthogonal = original.copy()
        for accepted in orthonormal_columns:
            orthogonal -= float(np.dot(accepted, original)) * accepted
        norm = float(np.linalg.norm(orthogonal))
        if norm <= tolerance:
            raise ValueError(
                "Input columns are linearly dependent or numerically indistinguishable."
            )
        orthonormal_columns.append(orthogonal / norm)

    result = np.column_stack(orthonormal_columns)
    if not np.allclose(result.T @ result, np.eye(result.shape[1]), atol=1e-8):
        raise ArithmeticError("Gram-Schmidt result failed the orthonormality check.")
    return result


def orthonormalize_basis(
    basis: Iterable[Iterable[float]] | np.ndarray,
    tolerance: float = 1e-10,
) -> FLOAT_ARRAY:
    """Create an orthonormal basis for a full-column-rank matrix using QR."""
    matrix = _as_matrix(basis, "basis")
    if matrix.shape[1] > matrix.shape[0]:
        raise ValueError("Basis cannot contain more columns than ambient dimensions.")
    if np.linalg.matrix_rank(matrix, tol=tolerance) != matrix.shape[1]:
        raise ValueError("Basis columns must be linearly independent.")
    orthonormal, _ = np.linalg.qr(matrix, mode="reduced")
    return orthonormal


def projection_matrix_from_basis(
    basis: Iterable[Iterable[float]] | np.ndarray,
) -> FLOAT_ARRAY:
    """Return the symmetric, idempotent projector onto a basis column space."""
    orthonormal = orthonormalize_basis(basis)
    projection_matrix = orthonormal @ orthonormal.T
    if not np.allclose(projection_matrix.T, projection_matrix, atol=1e-10):
        raise ArithmeticError("Projection matrix is not symmetric.")
    if not np.allclose(
        projection_matrix @ projection_matrix, projection_matrix, atol=1e-10
    ):
        raise ArithmeticError("Projection matrix is not idempotent.")
    return projection_matrix


def project_onto_subspace(
    vector: Iterable[float] | np.ndarray,
    basis: Iterable[Iterable[float]] | np.ndarray,
) -> FLOAT_ARRAY:
    """Project a vector onto the column space of a full-rank basis.

    ``numpy.linalg.lstsq`` avoids explicitly computing the inverse in the
    algebraic expression ``A(A.T A)^-1 A.T x``.
    """
    x = _as_vector(vector, "vector")
    matrix = _as_matrix(basis, "basis")
    if matrix.shape[0] != x.shape[0]:
        raise ValueError(
            "Basis row count must equal vector dimension; "
            f"got {matrix.shape[0]} and {x.shape[0]}."
        )
    if np.linalg.matrix_rank(matrix) != matrix.shape[1]:
        raise ValueError("Basis columns must be linearly independent.")
    coordinates, _, _, _ = np.linalg.lstsq(matrix, x, rcond=None)
    projected = matrix @ coordinates
    residual = x - projected
    if not np.allclose(matrix.T @ residual, 0.0, atol=1e-8):
        raise ArithmeticError("Projection residual is not orthogonal to the subspace.")
    return projected


def create_synthetic_embedding_clusters(
    samples_per_cluster: int = 30,
    dimensions: int = 16,
    cluster_spread: float = 0.12,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[FLOAT_ARRAY, NDArray[np.str_], FLOAT_ARRAY, list[str]]:
    """Create normalized synthetic embeddings, labels, and a query embedding.

    The three groups are intentionally controlled clusters. They are not
    generated by an embedding API and do not measure real semantic quality.
    """
    if samples_per_cluster < 3:
        raise ValueError("samples_per_cluster must be at least 3.")
    if dimensions < 3:
        raise ValueError("dimensions must be at least 3.")
    if cluster_spread <= 0:
        raise ValueError("cluster_spread must be positive.")

    group_names = [
        "Machine learning",
        "Cloud infrastructure",
        "Legal documents",
    ]
    rng = np.random.default_rng(random_seed)
    raw_centers = rng.normal(size=(len(group_names), dimensions))
    centers = raw_centers / np.linalg.norm(raw_centers, axis=1, keepdims=True)

    clusters = []
    labels: list[str] = []
    point_ids: list[str] = []
    for group_index, group_name in enumerate(group_names):
        cluster = centers[group_index] + rng.normal(
            scale=cluster_spread, size=(samples_per_cluster, dimensions)
        )
        cluster /= np.linalg.norm(cluster, axis=1, keepdims=True)
        clusters.append(cluster)
        labels.extend([group_name] * samples_per_cluster)
        slug = group_name.lower().replace(" ", "-")
        point_ids.extend(
            f"{slug}-{point_index:02d}" for point_index in range(samples_per_cluster)
        )

    embeddings = np.vstack(clusters)
    query = centers[0] + rng.normal(scale=cluster_spread / 2, size=dimensions)
    query = normalize_vector(query)
    return embeddings, np.asarray(labels), query, point_ids


def nearest_neighbors_by_cosine(
    embeddings: np.ndarray,
    query: np.ndarray,
    count: int = 5,
) -> tuple[NDArray[np.int64], FLOAT_ARRAY]:
    """Return indices and cosine scores of the nearest normalized embeddings."""
    matrix = _as_matrix(embeddings, "embeddings")
    query_vector = _as_vector(query, "query")
    if matrix.shape[1] != query_vector.shape[0]:
        raise ValueError("Embedding width must equal query dimension.")
    if not 1 <= count <= matrix.shape[0]:
        raise ValueError("count must be between 1 and the number of embeddings.")

    matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(np.isclose(matrix_norms, 0.0)):
        raise ValueError("Embeddings must not contain zero vectors.")
    normalized_matrix = matrix / matrix_norms
    scores = normalized_matrix @ normalize_vector(query_vector)
    indices = np.argsort(-scores)[:count]
    return indices.astype(np.int64), scores


def ensure_output_directories(
    base_directory: str | Path | None = None,
) -> dict[str, Path]:
    """Create and return the image, animation, and interactive output paths."""
    base = (
        Path(base_directory)
        if base_directory is not None
        else Path(__file__).resolve().parent / "outputs"
    )
    directories = {
        "base": base,
        "images": base / "images",
        "animations": base / "animations",
        "interactive": base / "interactive",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


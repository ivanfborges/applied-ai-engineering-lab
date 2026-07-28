"""Pure NumPy utilities used by the linear algebra visual explorer."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
EPSILON = 1e-12


def as_vector(values: ArrayLike) -> FloatArray:
    """Return a finite one-dimensional float vector."""
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError("Expected a one-dimensional vector.")
    if not np.all(np.isfinite(vector)):
        raise ValueError("Vector values must be finite.")
    return vector


def require_same_shape(a: FloatArray, b: FloatArray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"Vector shapes must match: {a.shape} != {b.shape}.")


def dot_product(a: ArrayLike, b: ArrayLike) -> float:
    first, second = as_vector(a), as_vector(b)
    require_same_shape(first, second)
    return float(first @ second)


def vector_norm(vector: ArrayLike, order: float = 2) -> float:
    values = as_vector(vector)
    if order not in (1, 2, np.inf):
        raise ValueError("This explorer supports only L1, L2, and infinity norms.")
    return float(np.linalg.norm(values, ord=order))


def normalize(vector: ArrayLike) -> FloatArray:
    values = as_vector(vector)
    norm = vector_norm(values, 2)
    if norm <= EPSILON:
        raise ValueError("A zero vector cannot be normalized.")
    return values / norm


def cosine_similarity(a: ArrayLike, b: ArrayLike) -> float:
    first, second = as_vector(a), as_vector(b)
    require_same_shape(first, second)
    denominator = vector_norm(first, 2) * vector_norm(second, 2)
    if denominator <= EPSILON:
        raise ValueError("Cosine similarity is undefined for zero vectors.")
    return float(np.clip((first @ second) / denominator, -1.0, 1.0))


def angle_degrees(a: ArrayLike, b: ArrayLike) -> float:
    return float(np.degrees(np.arccos(cosine_similarity(a, b))))


def projection(vector: ArrayLike, onto: ArrayLike) -> FloatArray:
    source, direction = as_vector(vector), as_vector(onto)
    require_same_shape(source, direction)
    denominator = float(direction @ direction)
    if denominator <= EPSILON:
        raise ValueError("Cannot project onto a zero vector.")
    return ((source @ direction) / denominator) * direction


def distance(a: ArrayLike, b: ArrayLike, order: float = 2) -> float:
    first, second = as_vector(a), as_vector(b)
    require_same_shape(first, second)
    return vector_norm(first - second, order)


def scaling_matrix(scale_x: float, scale_y: float) -> FloatArray:
    return np.array([[scale_x, 0.0], [0.0, scale_y]], dtype=float)


def rotation_matrix(angle_degrees_value: float) -> FloatArray:
    angle = np.deg2rad(angle_degrees_value)
    return np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=float,
    )


def reflection_matrix(axis: str) -> FloatArray:
    matrices = {
        "none": np.eye(2),
        "x-axis": np.array([[1.0, 0.0], [0.0, -1.0]]),
        "y-axis": np.array([[-1.0, 0.0], [0.0, 1.0]]),
        "origin": np.array([[-1.0, 0.0], [0.0, -1.0]]),
        "y=x": np.array([[0.0, 1.0], [1.0, 0.0]]),
    }
    try:
        return matrices[axis].copy()
    except KeyError as error:
        raise ValueError(f"Unknown reflection axis: {axis}.") from error


def shear_matrix(shear_x: float, shear_y: float) -> FloatArray:
    return np.array([[1.0, shear_x], [shear_y, 1.0]], dtype=float)


def compose_transformations(matrices: Iterable[ArrayLike]) -> FloatArray:
    """Compose matrices listed in the order they are applied to a vector."""
    result = np.eye(2)
    found_matrix = False
    for matrix in matrices:
        values = np.asarray(matrix, dtype=float)
        if values.shape != (2, 2):
            raise ValueError("Every transformation must have shape (2, 2).")
        result = values @ result
        found_matrix = True
    if not found_matrix:
        raise ValueError("At least one transformation is required.")
    return result


def apply_transformation(points: ArrayLike, matrix: ArrayLike) -> FloatArray:
    values = np.asarray(points, dtype=float)
    transformation = np.asarray(matrix, dtype=float)
    if transformation.shape != (2, 2):
        raise ValueError("The transformation matrix must have shape (2, 2).")
    if values.shape[-1] != 2:
        raise ValueError("Points must have two coordinates.")
    return values @ transformation.T


def standardize_features(features: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
    values = np.asarray(features, dtype=float)
    if values.ndim != 2:
        raise ValueError("Features must be a two-dimensional matrix.")
    mean = values.mean(axis=0)
    standard_deviation = values.std(axis=0)
    if np.any(standard_deviation <= EPSILON):
        raise ValueError("Every feature must have nonzero variance.")
    return (values - mean) / standard_deviation, mean, standard_deviation


def pairwise_distances_from_query(
    points: ArrayLike, query: ArrayLike, order: float = 2
) -> FloatArray:
    matrix = np.asarray(points, dtype=float)
    vector = as_vector(query)
    if matrix.ndim != 2 or matrix.shape[1] != vector.size:
        raise ValueError("Point rows and query must have the same dimension.")
    if order not in (1, 2, np.inf):
        raise ValueError("Unsupported distance norm.")
    return np.linalg.norm(matrix - vector, ord=order, axis=1)


def embedding_scores(
    embeddings: ArrayLike, query: ArrayLike, metric: str
) -> FloatArray:
    matrix = np.asarray(embeddings, dtype=float)
    vector = as_vector(query)
    if matrix.ndim != 2 or matrix.shape[1] != vector.size:
        raise ValueError("Embedding rows and query must have the same dimension.")

    if metric == "Dot product":
        return matrix @ vector
    if metric == "Cosine similarity":
        row_norms = np.linalg.norm(matrix, axis=1)
        query_norm = np.linalg.norm(vector)
        if query_norm <= EPSILON or np.any(row_norms <= EPSILON):
            raise ValueError("Cosine similarity is undefined for zero embeddings.")
        return (matrix @ vector) / (row_norms * query_norm)
    if metric == "Euclidean distance":
        return np.linalg.norm(matrix - vector, axis=1)
    raise ValueError(f"Unknown embedding metric: {metric}.")


def rank_embedding_labels(
    labels: Sequence[str],
    embeddings: ArrayLike,
    query: ArrayLike,
    metric: str,
) -> list[tuple[str, float]]:
    scores = embedding_scores(embeddings, query, metric)
    if len(labels) != len(scores):
        raise ValueError("Labels and embeddings must contain the same number of rows.")
    order = np.argsort(scores if metric == "Euclidean distance" else -scores)
    return [(labels[index], float(scores[index])) for index in order]


def distance_concentration(
    dimensions: Sequence[int],
    sample_count: int = 400,
    seed: int = 42,
) -> dict[str, FloatArray]:
    """Measure how Gaussian point-to-query distances change with dimension."""
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2.")
    if any(dimension < 1 for dimension in dimensions):
        raise ValueError("Dimensions must be positive.")

    rng = np.random.default_rng(seed)
    means, coefficients, relative_contrasts = [], [], []
    for dimension in dimensions:
        points = rng.normal(size=(sample_count, dimension))
        query = rng.normal(size=dimension)
        distances = np.linalg.norm(points - query, axis=1)
        mean = float(distances.mean())
        means.append(mean)
        coefficients.append(float(distances.std() / mean))
        relative_contrasts.append(
            float((distances.max() - distances.min()) / distances.min())
        )
    return {
        "dimensions": np.asarray(dimensions, dtype=float),
        "mean_distance": np.asarray(means),
        "coefficient_of_variation": np.asarray(coefficients),
        "relative_contrast": np.asarray(relative_contrasts),
    }

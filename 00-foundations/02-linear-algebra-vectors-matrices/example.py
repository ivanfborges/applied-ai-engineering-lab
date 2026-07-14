"""Practical NumPy examples for vector similarity and matrix transformations."""

from __future__ import annotations

import numpy as np


def cosine_similarity(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Return cosine similarity between every matrix row and one vector."""
    if matrix.ndim != 2 or vector.ndim != 1:
        raise ValueError("Expected a 2D matrix and a 1D vector.")
    if matrix.shape[1] != vector.shape[0]:
        raise ValueError("Matrix rows and vector must have the same dimension.")

    row_norms = np.linalg.norm(matrix, axis=1)
    vector_norm = np.linalg.norm(vector)
    if vector_norm == 0 or np.any(row_norms == 0):
        raise ValueError("Cosine similarity is undefined for zero vectors.")

    return (matrix @ vector) / (row_norms * vector_norm)


def rank_documents(documents: np.ndarray, query: np.ndarray) -> None:
    """Compare rankings produced by three common vector metrics."""
    dot_scores = documents @ query
    cosine_scores = cosine_similarity(documents, query)
    euclidean_distances = np.linalg.norm(documents - query, axis=1)

    print("Document matrix shape:", documents.shape)
    print("Query shape:", query.shape)
    print("Dot-product scores:", np.round(dot_scores, 4))
    print("Cosine similarities:", np.round(cosine_scores, 4))
    print("Euclidean distances:", np.round(euclidean_distances, 4))
    print("Dot-product ranking:", np.argsort(-dot_scores))
    print("Cosine ranking:", np.argsort(-cosine_scores))
    print("Euclidean ranking:", np.argsort(euclidean_distances))

    normalized_documents = documents / np.linalg.norm(
        documents, axis=1, keepdims=True
    )
    normalized_query = query / np.linalg.norm(query)
    normalized_dot_scores = normalized_documents @ normalized_query
    print(
        "Normalized dot product equals cosine similarity:",
        np.allclose(normalized_dot_scores, cosine_scores),
    )


def compare_transformation_order(points: np.ndarray) -> None:
    """Show that scaling and rotation generally do not commute."""
    scaling = np.array([[2.0, 0.0], [0.0, 0.5]])
    angle = np.deg2rad(30)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )

    # In A @ B @ x, B acts first. Points are rows, so transpose the map.
    scale_then_rotate = points @ (rotation @ scaling).T
    rotate_then_scale = points @ (scaling @ rotation).T

    print("\nFirst point after scale then rotate:", np.round(scale_then_rotate[0], 4))
    print("First point after rotate then scale:", np.round(rotate_then_scale[0], 4))
    print("Transformation orders are equal:", np.allclose(scale_then_rotate, rotate_then_scale))


def main() -> None:
    # Synthetic two-dimensional document embeddings for a compact demonstration.
    documents = np.array(
        [[0.9, 0.8], [0.2, 1.0], [-0.8, -0.6], [1.5, 1.2]],
        dtype=float,
    )
    query = np.array([1.0, 0.9], dtype=float)

    rank_documents(documents, query)
    compare_transformation_order(documents)


if __name__ == "__main__":
    main()

"""Project a vector onto a subspace with stable NumPy routines."""

from __future__ import annotations

import numpy as np


def projection_with_least_squares(
    basis: np.ndarray, vector: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Return coefficients, projection, rank, and singular values."""
    if basis.ndim != 2 or vector.ndim != 1:
        raise ValueError("Expected a 2D basis matrix and a 1D vector.")
    if basis.shape[0] != vector.shape[0]:
        raise ValueError("Basis columns and vector must share the ambient dimension.")

    # lstsq uses a stable solver and also handles rank-deficient inputs.
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        basis, vector, rcond=None
    )
    projected = basis @ coefficients
    return coefficients, projected, rank, singular_values


def projection_with_qr(basis: np.ndarray, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the orthogonal projection matrix and projected vector."""
    if np.linalg.matrix_rank(basis) != basis.shape[1]:
        raise ValueError("QR example requires linearly independent basis columns.")

    # Reduced QR gives one orthonormal column for each basis direction.
    orthonormal_basis, _ = np.linalg.qr(basis, mode="reduced")
    projection_matrix = orthonormal_basis @ orthonormal_basis.T
    return projection_matrix, projection_matrix @ vector


def main() -> None:
    # Synthetic columns spanning a two-dimensional subspace of R^3.
    basis = np.array(
        [
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    # This vector does not lie in the plane, so its residual is nonzero.
    vector = np.array([3.0, 1.0, 1.0])

    coefficients, projected_lstsq, rank, singular_values = (
        projection_with_least_squares(basis, vector)
    )
    projection_matrix, projected_qr = projection_with_qr(basis, vector)
    residual = vector - projected_lstsq

    print("Synthetic basis (columns):\n", basis)
    print("\nOriginal vector:", vector)
    print("Coordinates in the non-orthogonal basis:", np.round(coefficients, 6))
    print("Projected vector:", np.round(projected_lstsq, 6))
    print("Residual:", np.round(residual, 6))
    print("Basis.T @ residual:", np.round(basis.T @ residual, 12))
    print("Rank:", rank)
    print("Singular values:", np.round(singular_values, 6))
    print("\nProjection matrix:\n", np.round(projection_matrix, 6))
    print(
        "Symmetric:",
        np.allclose(projection_matrix.T, projection_matrix),
    )
    print(
        "Idempotent:",
        np.allclose(projection_matrix @ projection_matrix, projection_matrix),
    )
    print("Least-squares and QR projections agree:", np.allclose(projected_lstsq, projected_qr))

    # Assertions turn the mathematical invariants into executable checks.
    assert np.allclose(basis.T @ residual, 0.0)
    assert np.allclose(projection_matrix.T, projection_matrix)
    assert np.allclose(projection_matrix @ projection_matrix, projection_matrix)
    assert np.allclose(projected_lstsq, projected_qr)


if __name__ == "__main__":
    main()

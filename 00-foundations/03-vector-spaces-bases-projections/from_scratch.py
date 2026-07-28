"""Educational classical Gram-Schmidt and orthogonal projection."""

from __future__ import annotations

import numpy as np


def gram_schmidt(vectors: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
    """Convert independent column vectors into an orthonormal basis.

    This is classical Gram-Schmidt for teaching. Numerical libraries generally
    use more stable QR or SVD algorithms for difficult inputs.
    """
    if vectors.ndim != 2:
        raise ValueError("vectors must be a two-dimensional matrix.")
    if vectors.shape[1] == 0:
        raise ValueError("vectors must contain at least one column.")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")

    orthonormal_vectors: list[np.ndarray] = []

    for column_index in range(vectors.shape[1]):
        original = vectors[:, column_index].astype(float).copy()
        orthogonal_component = original.copy()

        # Remove the component of the original vector along every accepted q.
        for basis_vector in orthonormal_vectors:
            coefficient = float(np.dot(basis_vector, original))
            orthogonal_component -= coefficient * basis_vector

        component_norm = float(np.linalg.norm(orthogonal_component))
        if component_norm <= tolerance:
            raise ValueError(
                "Input columns are linearly dependent or numerically indistinguishable."
            )

        orthonormal_vectors.append(orthogonal_component / component_norm)

    return np.column_stack(orthonormal_vectors)


def project_onto_subspace(
    vector: np.ndarray, orthonormal_basis: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return subspace coordinates and the reconstructed projection."""
    if vector.ndim != 1 or orthonormal_basis.ndim != 2:
        raise ValueError("Expected a 1D vector and a 2D orthonormal basis.")
    if orthonormal_basis.shape[0] != vector.shape[0]:
        raise ValueError("Vector and basis must share the ambient dimension.")
    if not np.allclose(
        orthonormal_basis.T @ orthonormal_basis,
        np.eye(orthonormal_basis.shape[1]),
    ):
        raise ValueError("Basis columns must be orthonormal.")

    coordinates = orthonormal_basis.T @ vector
    projected = orthonormal_basis @ coordinates
    return coordinates, projected


def main() -> None:
    # Synthetic, non-orthogonal basis for a plane in R^3.
    basis = np.array(
        [
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    # This vector does not lie in the plane, so projection discards a component.
    vector = np.array([3.0, 1.0, 1.0])

    orthonormal_basis = gram_schmidt(basis)
    coordinates, projected = project_onto_subspace(vector, orthonormal_basis)
    residual = vector - projected

    print("Synthetic input basis:\n", basis)
    print("\nOrthonormal basis Q:\n", np.round(orthonormal_basis, 6))
    print("\nQ.T @ Q:\n", np.round(orthonormal_basis.T @ orthonormal_basis, 12))
    print("Coordinates in Q:", np.round(coordinates, 6))
    print("Projected vector:", np.round(projected, 6))
    print("Residual:", np.round(residual, 6))
    print("Q.T @ residual:", np.round(orthonormal_basis.T @ residual, 12))

    assert np.allclose(
        orthonormal_basis.T @ orthonormal_basis,
        np.eye(orthonormal_basis.shape[1]),
    )
    assert np.allclose(orthonormal_basis.T @ residual, 0.0)

    redundant_basis = np.column_stack((basis, basis[:, 0] + basis[:, 1]))
    try:
        gram_schmidt(redundant_basis)
    except ValueError as error:
        print("\nExpected redundant-column check:", error)


if __name__ == "__main__":
    main()

"""Educational vector and matrix operations implemented without NumPy."""

from __future__ import annotations

from math import sqrt
from typing import Sequence

Vector = Sequence[float]
Matrix = Sequence[Sequence[float]]


def _validate_same_dimension(a: Vector, b: Vector) -> None:
    if len(a) != len(b):
        raise ValueError("Vectors must have the same dimension.")


def dot_product(a: Vector, b: Vector) -> float:
    _validate_same_dimension(a, b)
    return sum(x * y for x, y in zip(a, b))


def l1_norm(vector: Vector) -> float:
    return sum(abs(value) for value in vector)


def l2_norm(vector: Vector) -> float:
    return sqrt(dot_product(vector, vector))


def euclidean_distance(a: Vector, b: Vector) -> float:
    _validate_same_dimension(a, b)
    return l2_norm([x - y for x, y in zip(a, b)])


def cosine_similarity(a: Vector, b: Vector) -> float:
    norm_a = l2_norm(a)
    norm_b = l2_norm(b)
    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cosine similarity is undefined for zero vectors.")
    return dot_product(a, b) / (norm_a * norm_b)


def _validate_matrix(matrix: Matrix, name: str) -> int:
    if not matrix or not matrix[0]:
        raise ValueError(f"Matrix {name} must not be empty.")
    columns = len(matrix[0])
    if any(len(row) != columns for row in matrix):
        raise ValueError(f"Matrix {name} has inconsistent row lengths.")
    return columns


def transpose(matrix: Matrix) -> list[list[float]]:
    columns = _validate_matrix(matrix, "input")
    return [
        [matrix[row][column] for row in range(len(matrix))]
        for column in range(columns)
    ]


def matrix_multiply(a: Matrix, b: Matrix) -> list[list[float]]:
    a_columns = _validate_matrix(a, "A")
    _validate_matrix(b, "B")
    if a_columns != len(b):
        raise ValueError("The number of columns in A must equal the rows in B.")

    # Turning B's columns into rows lets us reuse the dot product directly.
    b_transposed = transpose(b)
    return [
        [dot_product(a_row, b_column) for b_column in b_transposed]
        for a_row in a
    ]


def main() -> None:
    x = [1.0, 2.0, 3.0]
    y = [4.0, 5.0, 6.0]
    print("Dot product:", dot_product(x, y))
    print("L1 norm of x:", l1_norm(x))
    print("L2 norm of x:", l2_norm(x))
    print("Euclidean distance:", euclidean_distance(x, y))
    print("Cosine similarity:", cosine_similarity(x, y))

    a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    b = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
    print("Matrix multiplication:", matrix_multiply(a, b))


if __name__ == "__main__":
    main()

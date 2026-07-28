"""Streamlit application for visually exploring core linear algebra concepts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import components
import math_utils

ASSET_DIR = APP_DIR / "assets"

st.set_page_config(
    page_title="Linear Algebra Visual Explorer",
    page_icon="📐",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.8rem; padding-bottom: 3rem;}
    div[data-testid="stMetric"] {
        border: 1px solid #e2e8f0;
        border-radius: 0.75rem;
        padding: 0.75rem;
        background: #ffffff;
    }
    .insight {
        border-left: 4px solid #2563eb;
        padding: 0.8rem 1rem;
        background: #eff6ff;
        border-radius: 0.35rem;
        margin: 0.6rem 0 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def vector_inputs(
    prefix: str,
    defaults: tuple[float, float],
    label: str,
) -> np.ndarray:
    st.markdown(f"**{label}**")
    first, second = st.columns(2)
    x = first.number_input(
        f"{label} x",
        min_value=-5.0,
        max_value=5.0,
        value=defaults[0],
        step=0.25,
        key=f"{prefix}_x",
    )
    y = second.number_input(
        f"{label} y",
        min_value=-5.0,
        max_value=5.0,
        value=defaults[1],
        step=0.25,
        key=f"{prefix}_y",
    )
    return np.array([x, y], dtype=float)


def show_gif(filename: str, caption: str) -> None:
    path = ASSET_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")


def vectors_page() -> None:
    st.header("1 · Vectors and vector arithmetic")
    st.markdown(
        '<div class="insight">A vector can represent a direction, a point, '
        "or a feature representation. Its coordinates depend on the chosen basis; "
        "its geometric length does not.</div>",
        unsafe_allow_html=True,
    )
    controls, visualization = st.columns([0.34, 0.66])
    with controls:
        operation = st.selectbox(
            "Operation",
            (
                "Addition: a + b",
                "Subtraction: a - b",
                "Scalar multiplication: ca",
            ),
        )
        a = vector_inputs("arithmetic_a", (2.0, 1.0), "Vector a")
        b = vector_inputs("arithmetic_b", (-1.0, 2.0), "Vector b")
        scalar = st.slider("Scalar c", -3.0, 3.0, 1.5, 0.1)

    figure, result = components.vector_arithmetic_figure(a, b, operation, scalar)
    with visualization:
        st.plotly_chart(figure, width="stretch")

    first, second, third = st.columns(3)
    first.metric("Result", np.array2string(result, precision=2))
    second.metric("Magnitude ‖a‖₂", f"{math_utils.vector_norm(a):.3f}")
    third.metric("Direction change", "reversed" if scalar < 0 else "preserved")

    st.subheader("Magnitude and normalization")
    left, right = st.columns([0.55, 0.45])
    with left:
        st.latex(r"\lVert a\rVert_2=\sqrt{a_1^2+a_2^2}")
        st.write(
            "L2 normalization divides a nonzero vector by its magnitude. "
            "The direction remains unchanged and the new magnitude becomes one."
        )
        if math_utils.vector_norm(a) > math_utils.EPSILON:
            normalized = math_utils.normalize(a)
            st.code(
                f"a          = {np.array2string(a, precision=3)}\n"
                f"a / ||a||  = {np.array2string(normalized, precision=3)}\n"
                f"magnitude  = {math_utils.vector_norm(normalized):.3f}"
            )
        else:
            st.warning("The zero vector has no direction and cannot be normalized.")
    with right:
        st.plotly_chart(
            components.normalization_figure(a),
            width="stretch",
        )
    with st.expander("Animation: vector addition"):
        show_gif(
            "vector_addition.gif",
            "Vector addition follows the parallelogram rule.",
        )

    st.info(
        "Applied connection: feature rows, gradients, embeddings, and model "
        "parameters are all vectors. Addition appears in residual connections and "
        "gradient updates; scalar multiplication appears in learning-rate updates."
    )


def similarity_page() -> None:
    st.header("2 · Dot product, angle, cosine, and projection")
    controls, visualization = st.columns([0.34, 0.66])
    with controls:
        a = vector_inputs("similarity_a", (3.0, 2.0), "Vector a")
        b = vector_inputs("similarity_b", (2.0, -1.0), "Vector b")
        st.latex(r"a\cdot b=\lVert a\rVert_2\lVert b\rVert_2\cos(\theta)")
        st.caption(
            "Move the vectors until the dot product reaches zero to see orthogonality."
        )

    figure, metrics = components.similarity_projection_figure(a, b)
    with visualization:
        st.plotly_chart(figure, width="stretch")

    columns = st.columns(4)
    columns[0].metric("Dot product", f"{metrics['dot']:.3f}")
    columns[1].metric(
        "Angle",
        f"{metrics['angle']:.2f}°" if "angle" in metrics else "undefined",
    )
    columns[2].metric(
        "Cosine similarity",
        f"{metrics['cosine']:.3f}" if "cosine" in metrics else "undefined",
    )
    columns[3].metric(
        "Orthogonal?",
        "yes" if metrics["orthogonal"] else "no",
    )

    left, right = st.columns([0.6, 0.4])
    with left:
        st.subheader("Projection")
        st.latex(
            r"\operatorname{proj}_b(a)=\frac{a\cdot b}{b\cdot b}b"
        )
        if "projection" in metrics:
            projection = np.asarray(metrics["projection"])
            st.write(
                "The green vector is the component of **a** explained by the "
                "direction of **b**. The dotted residual is perpendicular to **b**."
            )
            st.code(f"projection = {np.array2string(projection, precision=3)}")
        else:
            st.warning("Projection and angle require nonzero vectors.")
    with right:
        show_gif(
            "vector_projection.gif",
            "A projection decomposes a vector into parallel and orthogonal parts.",
        )

    st.info(
        "Applied connection: dot products score query-key compatibility in "
        "Transformer attention. Cosine similarity is common in semantic retrieval "
        "when direction matters more than embedding magnitude."
    )


def norms_page() -> None:
    st.header("3 · Norms and distances")
    st.markdown(
        '<div class="insight">A norm defines what “size” means. Applying a norm '
        "to the difference between two points defines a distance.</div>",
        unsafe_allow_html=True,
    )
    controls, visualization = st.columns([0.34, 0.66])
    with controls:
        p = vector_inputs("distance_p", (-2.0, -1.0), "Point p")
        q = vector_inputs("distance_q", (3.0, 2.0), "Point q")
        st.latex(r"d_p(x,y)=\lVert x-y\rVert_p")
        st.write(
            "The unit balls reveal each norm's geometry: diamond for L1, circle "
            "for L2, and square for L∞."
        )

    figure, values = components.norms_and_distances_figure(p, q)
    with visualization:
        st.plotly_chart(figure, width="stretch")

    first, second, third = st.columns(3)
    first.metric("Manhattan distance · L1", f"{values['l1']:.3f}")
    second.metric("Euclidean distance · L2", f"{values['l2']:.3f}")
    third.metric("Chebyshev distance · L∞", f"{values['linf']:.3f}")

    st.markdown(
        """
        - **L1** sums coordinate-wise movement and is connected to sparse regularization.
        - **L2** measures straight-line distance and penalizes large deviations more strongly.
        - **L∞** is determined by the largest coordinate difference.
        """
    )
    st.warning(
        "Distance-based methods such as KNN and K-means are sensitive to feature "
        "scale. A numerically large feature may dominate without being more useful."
    )


def transformations_page() -> None:
    st.header("4 · Matrices as transformations")
    st.markdown(
        '<div class="insight">A 2×2 matrix transforms the basis vectors. Every '
        "other vector follows from the same linear combination.</div>",
        unsafe_allow_html=True,
    )
    controls, visualization = st.columns([0.34, 0.66])
    with controls:
        scale_x = st.slider("Scale x", -2.0, 2.0, 1.4, 0.1)
        scale_y = st.slider("Scale y", -2.0, 2.0, 0.8, 0.1)
        rotation = st.slider("Rotation (degrees)", -180, 180, 30, 5)
        shear_x = st.slider("Horizontal shear", -2.0, 2.0, 0.3, 0.1)
        shear_y = st.slider("Vertical shear", -2.0, 2.0, 0.0, 0.1)
        reflection = st.selectbox(
            "Reflection",
            ("none", "x-axis", "y-axis", "origin", "y=x"),
        )
        order_name = st.selectbox(
            "Application order",
            (
                "scale → shear → reflection → rotate",
                "rotate → reflection → shear → scale",
            ),
        )

    matrices = {
        "scale": math_utils.scaling_matrix(scale_x, scale_y),
        "shear": math_utils.shear_matrix(shear_x, shear_y),
        "reflection": math_utils.reflection_matrix(reflection),
        "rotate": math_utils.rotation_matrix(rotation),
    }
    operation_names = (
        ["scale", "shear", "reflection", "rotate"]
        if order_name.startswith("scale")
        else ["rotate", "reflection", "shear", "scale"]
    )
    combined = math_utils.compose_transformations(
        matrices[name] for name in operation_names
    )
    with visualization:
        st.plotly_chart(
            components.transformation_figure(
                combined,
                f"Combined transformation · {order_name}",
            ),
            width="stretch",
        )

    first, second, third = st.columns(3)
    first.metric("Determinant", f"{np.linalg.det(combined):.3f}")
    second.metric("Area scale", f"{abs(np.linalg.det(combined)):.3f}×")
    third.metric("Orientation", "reversed" if np.linalg.det(combined) < 0 else "preserved")
    st.code(
        "Combined matrix T =\n"
        + np.array2string(combined, precision=3, suppress_small=True)
    )

    st.subheader("Matrix multiplication and operation order")
    st.write(
        "If a vector is scaled and then rotated, the combined matrix is "
        "`R @ S`. The rightmost matrix acts first under the column-vector convention."
    )
    scale = math_utils.scaling_matrix(scale_x, scale_y)
    rotate = math_utils.rotation_matrix(rotation)
    scale_then_rotate = math_utils.compose_transformations([scale, rotate])
    rotate_then_scale = math_utils.compose_transformations([rotate, scale])
    st.plotly_chart(
        components.transformation_order_figure(
            scale_then_rotate,
            rotate_then_scale,
            ("Scale then rotate · R @ S", "Rotate then scale · S @ R"),
        ),
        width="stretch",
    )
    st.code(
        f"R @ S =\n{np.array2string(scale_then_rotate, precision=3)}\n\n"
        f"S @ R =\n{np.array2string(rotate_then_scale, precision=3)}\n\n"
        f"Equal? {np.allclose(scale_then_rotate, rotate_then_scale)}"
    )
    show_gif(
        "transformation_order.gif",
        "Scaling and rotation usually produce different results when reordered.",
    )
    st.info(
        "Applied connection: dense layers compute matrix products over entire "
        "batches. Shape reasoning explains why (n×d) @ (d×k) produces (n×k)."
    )


def applied_ai_page() -> None:
    st.header("5 · Applied ML, embeddings, and high-dimensional geometry")
    feature_tab, retrieval_tab, dimension_tab = st.tabs(
        ("Feature scaling", "Embedding retrieval", "Distance concentration")
    )

    with feature_tab:
        st.subheader("Feature scaling changes distance")
        st.caption("Synthetic observations; no external dataset is used.")
        raw_points = np.array(
            [[1.0, 100.0], [2.0, 950.0], [4.0, 400.0], [7.0, 720.0], [8.0, 250.0]]
        )
        raw_query = np.array([3.0, 800.0])
        scaled_points, mean, std = math_utils.standardize_features(raw_points)
        scaled_query = (raw_query - mean) / std
        st.plotly_chart(
            components.feature_scaling_figure(
                raw_points,
                raw_query,
                scaled_points,
                scaled_query,
            ),
            width="stretch",
        )
        raw_distances = math_utils.pairwise_distances_from_query(raw_points, raw_query)
        scaled_distances = math_utils.pairwise_distances_from_query(
            scaled_points, scaled_query
        )
        left, right = st.columns(2)
        left.metric("Nearest before scaling", f"P{np.argmin(raw_distances)}")
        right.metric("Nearest after scaling", f"P{np.argmin(scaled_distances)}")
        st.warning(
            "Fit scaling parameters only on training data. Recomputing them with "
            "validation, test, or serving observations causes leakage or skew."
        )

    with retrieval_tab:
        st.subheader("Synthetic semantic retrieval")
        st.write(
            "The vectors below are handcrafted synthetic embeddings. They illustrate "
            "metric behavior; they are not outputs from an embedding model."
        )
        labels = (
            "vector databases and semantic search",
            "linear algebra for machine learning",
            "cooking pasta at home",
            "nearest-neighbor retrieval",
            "large-magnitude generic document",
        )
        embeddings = np.array(
            [
                [0.95, 0.80, 0.05],
                [0.70, 0.65, 0.20],
                [0.02, 0.10, 0.98],
                [0.88, 0.75, 0.10],
                [3.00, 2.20, 2.50],
            ]
        )
        queries = {
            "How does embedding search work?": np.array([1.0, 0.85, 0.05]),
            "Explain ML mathematics": np.array([0.65, 0.80, 0.15]),
            "What should I cook?": np.array([0.05, 0.05, 1.0]),
        }
        query_name = st.selectbox("Synthetic query", tuple(queries))
        metric = st.radio(
            "Retrieval metric",
            ("Cosine similarity", "Dot product", "Euclidean distance"),
            horizontal=True,
        )
        ranking = math_utils.rank_embedding_labels(
            labels, embeddings, queries[query_name], metric
        )
        st.plotly_chart(
            components.embedding_ranking_figure(ranking, metric),
            width="stretch",
        )
        st.write("**Ranking:**")
        for position, (label, score) in enumerate(ranking, start=1):
            st.write(f"{position}. {label} — `{score:.3f}`")
        st.info(
            "Cosine removes magnitude; dot product retains it. In a production RAG "
            "system, use the metric expected by the embedding model and validate "
            "retrieval recall and end-task quality."
        )

    with dimension_tab:
        st.subheader("High-dimensional distance concentration")
        sample_count = st.slider("Synthetic points per dimension", 100, 1_000, 400, 100)
        seed = st.number_input("Random seed", 0, 10_000, 42)
        dimensions = [2, 5, 10, 25, 50, 100, 250, 500, 1_000]
        results = math_utils.distance_concentration(
            dimensions,
            sample_count=sample_count,
            seed=int(seed),
        )
        st.plotly_chart(
            components.concentration_figure(results),
            width="stretch",
        )
        st.write(
            "Mean distance grows with dimension, but its relative variability tends "
            "to shrink for this synthetic Gaussian experiment. Near and far points "
            "become less distinguishable relative to the overall scale."
        )
        st.warning(
            "This is an illustration, not a universal benchmark. Concentration "
            "depends on the data distribution, metric, intrinsic dimension, and preprocessing."
        )


def main() -> None:
    st.title("📐 Linear Algebra Visual Explorer")
    st.write(
        "Interactive geometric intuition for vectors, metrics, projections, matrix "
        "transformations, and their role in modern AI systems."
    )
    st.caption(
        "Day 2 · Applied AI Engineering Lab · All examples use local synthetic data"
    )

    page = st.sidebar.radio(
        "Choose a laboratory",
        (
            "Vectors and arithmetic",
            "Similarity and projection",
            "Norms and distances",
            "Matrix transformations",
            "Applied AI geometry",
        ),
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Learning loop**\n\n"
        "1. Change one input.\n"
        "2. Predict the visual result.\n"
        "3. Inspect the metric.\n"
        "4. Explain the behavior aloud."
    )

    pages = {
        "Vectors and arithmetic": vectors_page,
        "Similarity and projection": similarity_page,
        "Norms and distances": norms_page,
        "Matrix transformations": transformations_page,
        "Applied AI geometry": applied_ai_page,
    }
    pages[page]()


if __name__ == "__main__":
    main()

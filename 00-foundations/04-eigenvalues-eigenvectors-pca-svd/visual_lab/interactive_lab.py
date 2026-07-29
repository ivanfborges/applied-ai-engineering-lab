"""Streamlit study environment for eigenvectors, PCA, and SVD."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Streamlit executes the entrypoint as a standalone script. When the command is
# launched from the repository root, Python does not automatically add the Day
# 4 topic folder (the parent of visual_lab) to sys.path. Add that stable,
# file-relative location so imports work from either documented working directory.
TOPIC_ROOT = Path(__file__).resolve().parents[1]
if str(TOPIC_ROOT) not in sys.path:
    sys.path.insert(0, str(TOPIC_ROOT))

from visual_lab.datasets import (
    correlated_2d,
    correlated_3d,
    correlated_features,
    synthetic_grayscale_image,
)
from visual_lab.math_utils import (
    align_component_signs,
    approximate_svd_compression_ratio,
    center_data,
    eigendecomposition_pca,
    low_rank_approximation,
    reconstruct_data,
    svd_pca,
)
from visual_lab.plotting import (
    COLORS,
    pca_3d_figure,
    projection_3d_animation_figure,
)

st.set_page_config(
    page_title="Eigenvectors, PCA & SVD Visual Lab",
    page_icon="📐",
    layout="wide",
)


def what_to_observe(items: list[str]) -> None:
    """Render a consistent observation block."""
    st.markdown("#### What to observe")
    st.markdown("\n".join(f"- {item}" for item in items))


def transformed_circle_figure(matrix: np.ndarray, progress: float) -> go.Figure:
    """Show a continuous 2D transform with its eigenvectors."""
    angles = np.linspace(0, 2 * np.pi, 180)
    circle = np.column_stack([np.cos(angles), np.sin(angles)])
    transform = (1 - progress) * np.eye(2) + progress * matrix
    transformed = circle @ transform.T
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=circle[:, 0],
            y=circle[:, 1],
            mode="lines",
            line={"color": COLORS["gray"], "dash": "dash"},
            name="Original unit circle",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=transformed[:, 0],
            y=transformed[:, 1],
            mode="lines",
            line={"color": COLORS["blue"], "width": 4},
            name="Transformed circle",
        )
    )
    for index, (value, vector, color) in enumerate(
        zip(
            eigenvalues,
            eigenvectors.T,
            (COLORS["orange"], COLORS["red"]),
            strict=True,
        ),
        start=1,
    ):
        endpoint = transform @ vector
        figure.add_trace(
            go.Scatter(
                x=[-endpoint[0], endpoint[0]],
                y=[-endpoint[1], endpoint[1]],
                mode="lines+markers",
                line={"color": color, "width": 5},
                name=f"v{index}: λ={value:.3f}",
            )
        )
    figure.update_layout(
        title=f"A(t) = (1 − t)I + tA, t={progress:.2f}",
        template="plotly_white",
        xaxis={"range": [-4.5, 4.5], "scaleanchor": "y", "title": "x"},
        yaxis={"range": [-4.5, 4.5], "title": "y"},
        height=600,
    )
    return figure


def pca_2d_interactive_figure(
    data: np.ndarray,
    *,
    center: bool,
    standardize: bool,
) -> tuple[go.Figure, np.ndarray, np.ndarray]:
    """Build a 2D PCA view while allowing deliberate preprocessing mistakes."""
    working = data.copy()
    if standardize:
        scale = working.std(axis=0, ddof=1)
        working = working / np.where(scale == 0, 1.0, scale)
    origin = working.mean(axis=0) if center else np.zeros(2)
    analyzed = working - origin if center else working
    second_moment = analyzed.T @ analyzed / (analyzed.shape[0] - 1)
    values, vectors = np.linalg.eigh(second_moment)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    ratios = values / values.sum()

    figure = go.Figure(
        go.Scatter(
            x=working[:, 0],
            y=working[:, 1],
            mode="markers",
            marker={"size": 6, "color": COLORS["blue"], "opacity": 0.52},
            name="Synthetic observations",
            hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
        )
    )
    for index, color in enumerate((COLORS["orange"], COLORS["red"])):
        length = 2.2 * np.sqrt(max(values[index], 0))
        endpoint = vectors[:, index] * length
        figure.add_trace(
            go.Scatter(
                x=[origin[0] - endpoint[0], origin[0] + endpoint[0]],
                y=[origin[1] - endpoint[1], origin[1] + endpoint[1]],
                mode="lines+markers",
                line={"color": color, "width": 5},
                name=f"PC{index + 1}: {ratios[index]:.1%}",
            )
        )
    figure.add_trace(
        go.Scatter(
            x=[origin[0]],
            y=[origin[1]],
            mode="markers",
            marker={"symbol": "x", "size": 13, "color": COLORS["navy"]},
            name="Analysis origin",
        )
    )
    figure.update_layout(
        title="PCA axes from the selected preprocessing",
        template="plotly_white",
        xaxis={"title": "feature 1", "scaleanchor": "y"},
        yaxis={"title": "feature 2"},
        height=590,
    )
    return figure, values, vectors


def svd_geometry_figure(matrix: np.ndarray) -> tuple[go.Figure, np.ndarray]:
    """Show the unit circle after each exact SVD stage."""
    left, singular_values, right_t = np.linalg.svd(matrix)
    sigma = np.diag(singular_values)
    angles = np.linspace(0, 2 * np.pi, 180)
    circle = np.column_stack([np.cos(angles), np.sin(angles)])
    transforms = [
        ("Original", np.eye(2)),
        ("Apply Vᵀ", right_t),
        ("Apply Σ", sigma @ right_t),
        ("Apply U", left @ sigma @ right_t),
    ]
    figure = make_subplots(
        rows=1,
        cols=4,
        subplot_titles=[stage[0] for stage in transforms],
        horizontal_spacing=0.04,
    )
    for column, (_, transform) in enumerate(transforms, start=1):
        points = circle @ transform.T
        figure.add_trace(
            go.Scatter(
                x=points[:, 0],
                y=points[:, 1],
                mode="lines",
                line={"color": COLORS["blue"], "width": 4},
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        figure.update_xaxes(range=[-3.3, 3.3], scaleanchor=f"y{column}", row=1, col=column)
        figure.update_yaxes(range=[-3.3, 3.3], row=1, col=column)
    figure.update_layout(
        title="X = UΣVᵀ: input reorientation, scaling, output reorientation",
        template="plotly_white",
        height=480,
        margin={"l": 20, "r": 20, "t": 80, "b": 20},
    )
    return figure, singular_values


def pitfall_comparison_figure(
    scale_multiplier: float,
    outlier_intensity: float,
    seed: int,
) -> go.Figure:
    """Build four Plotly panels for common PCA preprocessing pitfalls."""
    base = correlated_2d(n_samples=220, noise=0.25, seed=seed)
    scenarios = [
        ("Centered", base, True),
        ("Offset, not centered", base + [8.0, -6.0], False),
        ("Scale mismatch", base * [scale_multiplier, 1.0], True),
        (
            "Influential outliers",
            np.vstack(
                [
                    base,
                    [
                        [outlier_intensity, -outlier_intensity * 0.8],
                        [outlier_intensity * 1.1, -outlier_intensity],
                    ],
                ]
            ),
            True,
        ),
    ]
    figure = make_subplots(rows=2, cols=2, subplot_titles=[x[0] for x in scenarios])
    for index, (_, data, should_center) in enumerate(scenarios):
        row, column = index // 2 + 1, index % 2 + 1
        origin = data.mean(axis=0) if should_center else np.zeros(2)
        analyzed = data - origin if should_center else data
        values, vectors = np.linalg.eigh(
            analyzed.T @ analyzed / (analyzed.shape[0] - 1)
        )
        direction = vectors[:, np.argmax(values)]
        length = np.quantile(np.linalg.norm(analyzed, axis=1), 0.9)
        endpoints = np.vstack(
            [origin - direction * length, origin + direction * length]
        )
        figure.add_trace(
            go.Scatter(
                x=data[:, 0],
                y=data[:, 1],
                mode="markers",
                marker={"size": 5, "color": COLORS["blue"], "opacity": 0.45},
                showlegend=False,
            ),
            row=row,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=endpoints[:, 0],
                y=endpoints[:, 1],
                mode="lines",
                line={"color": COLORS["red"], "width": 4},
                name="First direction",
                showlegend=index == 0,
            ),
            row=row,
            col=column,
        )
    figure.update_layout(
        title="Centering, scale, and outliers change the fitted PCA direction",
        template="plotly_white",
        height=720,
    )
    return figure


st.title("Eigenvectors, PCA, and SVD — Visual Laboratory")
st.caption(
    "Every dataset is synthetic and generated locally with a fixed or user-selected seed."
)

tabs = st.tabs(
    [
        "1. Eigenvectors",
        "2. PCA in 2D",
        "3. PCA in 3D",
        "4. Explained variance",
        "5. SVD geometry",
        "6. Low-rank approximation",
        "7. PCA pitfalls",
        "8. PCA–SVD equivalence",
    ]
)

with tabs[0]:
    st.latex(r"Av=\lambda v")
    control_1, control_2 = st.columns([1, 2])
    with control_1:
        st.markdown("Use a symmetric matrix so its eigenvectors are real and orthogonal.")
        a11 = st.number_input("A₁₁", -4.0, 4.0, 3.0, 0.1)
        a12 = st.number_input("A₁₂ = A₂₁", -3.0, 3.0, 1.0, 0.1)
        a22 = st.number_input("A₂₂", -4.0, 4.0, 2.0, 0.1)
        progress = st.slider("Transformation progress t", 0.0, 1.0, 1.0, 0.02)
        matrix = np.array([[a11, a12], [a12, a22]])
        eigenvalues = np.linalg.eigvalsh(matrix)[::-1]
        st.metric("Largest eigenvalue", f"{eigenvalues[0]:.3f}")
        st.metric("Second eigenvalue", f"{eigenvalues[1]:.3f}")
    with control_2:
        st.plotly_chart(
            transformed_circle_figure(matrix, progress),
            width="stretch",
        )
    what_to_observe(
        [
            "The ellipse changes continuously as A(t) moves from I to A.",
            "Eigenvector lines do not rotate away from themselves.",
            "A negative eigenvalue reverses orientation along the same line.",
        ]
    )

with tabs[1]:
    st.latex(r"\operatorname{Var}(X_cv)=v^\top C v")
    controls, plot = st.columns([1, 2])
    with controls:
        samples_2d = st.slider("Number of samples", 50, 900, 320, 10, key="n2d")
        correlation = st.slider("Correlation strength", -0.95, 0.95, 0.88, 0.01)
        noise_2d = st.slider("Noise level", 0.0, 1.0, 0.30, 0.02, key="noise2d")
        seed_2d = st.number_input("Random seed", 0, 10000, 42, key="seed2d")
        center_2d = st.checkbox("Center data", True, key="center2d")
        standardize_2d = st.checkbox("Standardize features", False, key="scale2d")
        data_2d = correlated_2d(
            samples_2d,
            correlation,
            noise_2d,
            int(seed_2d),
        )
        figure_2d, values_2d, _ = pca_2d_interactive_figure(
            data_2d,
            center=center_2d,
            standardize=standardize_2d,
        )
        st.write(
            {
                "eigenvalues": np.round(values_2d, 4).tolist(),
                "explained_variance_ratio": np.round(
                    values_2d / values_2d.sum(),
                    4,
                ).tolist(),
            }
        )
    with plot:
        st.plotly_chart(figure_2d, width="stretch")
    what_to_observe(
        [
            "PC1 follows the direction that maximizes projected variance.",
            "PC2 is orthogonal to PC1 for a symmetric covariance matrix.",
            "Centering changes the reference from the origin to the data mean.",
            "Standardization changes the question to variance in standard-deviation units.",
        ]
    )

with tabs[2]:
    st.latex(r"Z=X_cV,\qquad \hat X=ZV^\top+\mu")
    controls, plot = st.columns([1, 3])
    with controls:
        samples_3d = st.slider("Number of samples", 50, 700, 260, 10, key="n3d")
        noise_3d = st.slider("Noise level", 0.0, 0.8, 0.16, 0.02, key="noise3d")
        seed_3d = st.number_input("Random seed", 0, 10000, 7, key="seed3d")
        show_projection = st.checkbox("Animate projection onto PC1–PC2", False)
        data_3d = correlated_3d(samples_3d, noise_3d, int(seed_3d))
        result_3d = eigendecomposition_pca(data_3d, 3)
        st.write(
            {
                f"PC{i + 1}": f"{ratio:.2%}"
                for i, ratio in enumerate(result_3d.explained_variance_ratio)
            }
        )
    with plot:
        figure_3d = (
            projection_3d_animation_figure(data_3d)
            if show_projection
            else pca_3d_figure(data_3d)
        )
        st.plotly_chart(figure_3d, width="stretch")
    what_to_observe(
        [
            "PCA creates a new orthogonal coordinate system, not a subset of columns.",
            "PC1 and PC2 span the plane that minimizes squared projection error.",
            "Projection removes only the residual along the discarded PC3 direction.",
        ]
    )

with tabs[3]:
    st.latex(r"r_i=\lambda_i/\sum_j\lambda_j")
    samples_var = st.slider("Number of samples", 80, 1000, 500, 20, key="nvar")
    noise_var = st.slider("Noise level", 0.0, 0.7, 0.12, 0.01, key="noisevar")
    selected_components = st.slider("Retained PCA components", 1, 8, 3)
    seed_var = st.number_input("Random seed", 0, 10000, 21, key="seedvar")
    feature_data = correlated_features(
        samples_var,
        n_features=8,
        n_latent=3,
        noise=noise_var,
        seed=int(seed_var),
    )
    pca_all = PCA().fit(feature_data)
    counts = np.arange(1, 9)
    cumulative = np.cumsum(pca_all.explained_variance_ratio_)
    errors = []
    for count in counts:
        scores = pca_all.transform(feature_data)[:, :count]
        reconstruction = scores @ pca_all.components_[:count] + pca_all.mean_
        errors.append(float(np.mean((feature_data - reconstruction) ** 2)))
    variance_figure = make_subplots(specs=[[{"secondary_y": True}]])
    variance_figure.add_bar(
        x=counts,
        y=pca_all.explained_variance_ratio_,
        name="Individual explained variance",
        marker_color=COLORS["blue"],
        secondary_y=False,
    )
    variance_figure.add_scatter(
        x=counts,
        y=cumulative,
        name="Cumulative explained variance",
        line={"color": COLORS["orange"], "width": 3},
        mode="lines+markers",
        secondary_y=False,
    )
    variance_figure.add_scatter(
        x=counts,
        y=errors,
        name="Reconstruction MSE",
        line={"color": COLORS["purple"], "width": 3},
        mode="lines+markers",
        secondary_y=True,
    )
    variance_figure.add_vline(
        x=selected_components,
        line_dash="dash",
        line_color=COLORS["red"],
    )
    variance_figure.update_yaxes(title_text="Variance ratio", secondary_y=False)
    variance_figure.update_yaxes(title_text="Reconstruction MSE", secondary_y=True)
    variance_figure.update_layout(
        title="More components retain variance and reduce reconstruction error",
        template="plotly_white",
        height=590,
    )
    st.plotly_chart(variance_figure, width="stretch")
    st.metric(
        "Variance retained at selected rank",
        f"{cumulative[selected_components - 1]:.2%}",
    )
    what_to_observe(
        [
            "Covariance eigenvalues become the explained-variance bars.",
            "Cumulative explained variance can only increase with retained rank.",
            "Reconstruction MSE can only decrease or remain equal.",
            "A variance threshold is a heuristic, not a task-quality guarantee.",
        ]
    )

with tabs[4]:
    st.latex(r"X=U\Sigma V^\top")
    controls, plot = st.columns([1, 3])
    with controls:
        x11 = st.number_input("X₁₁", -3.0, 3.0, 1.8, 0.1)
        x12 = st.number_input("X₁₂", -3.0, 3.0, 0.8, 0.1)
        x21 = st.number_input("X₂₁", -3.0, 3.0, -0.4, 0.1)
        x22 = st.number_input("X₂₂", -3.0, 3.0, 1.2, 0.1)
        svd_matrix = np.array([[x11, x12], [x21, x22]])
        svd_figure, singular_values = svd_geometry_figure(svd_matrix)
        st.write("Singular values", np.round(singular_values, 4))
        st.write("Direct reconstruction", np.round(np.linalg.svd(svd_matrix)[0] @ np.diag(singular_values) @ np.linalg.svd(svd_matrix)[2], 4))
    with plot:
        st.plotly_chart(svd_figure, width="stretch")
    what_to_observe(
        [
            "Vᵀ chooses orthogonal input directions.",
            "Σ stretches those directions by the singular values.",
            "U places the scaled directions in the output space.",
            "The final ellipse equals direct application of X.",
        ]
    )

with tabs[5]:
    st.latex(r"X_k=U_k\Sigma_kV_k^\top")
    image = synthetic_grayscale_image()
    selected_rank = st.slider("Selected SVD rank", 1, min(image.shape), 10)
    reconstruction, singular_values = low_rank_approximation(image, selected_rank)
    energy = np.sum(singular_values[:selected_rank] ** 2) / np.sum(
        singular_values**2
    )
    mse = np.mean((image - reconstruction) ** 2)
    ratio = approximate_svd_compression_ratio(image.shape, selected_rank)
    metrics = st.columns(3)
    metrics[0].metric("Singular-value energy", f"{energy:.2%}")
    metrics[1].metric("Reconstruction MSE", f"{mse:.6f}")
    metrics[2].metric("Approx. dense-value ratio", f"{ratio:.2f}:1")
    rank_figure = go.Figure(
        go.Heatmap(
            z=reconstruction,
            colorscale="Gray",
            zmin=0,
            zmax=1,
            showscale=False,
            hovertemplate=(
                "row=%{y}<br>column=%{x}<br>intensity=%{z:.3f}<extra></extra>"
            ),
        )
    )
    rank_figure.update_layout(
        title=f"Rank {selected_rank} reconstruction",
        template="plotly_white",
        xaxis={"visible": False, "scaleanchor": "y"},
        yaxis={"visible": False, "autorange": "reversed"},
        height=650,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    st.plotly_chart(rank_figure, width="stretch")
    st.caption(
        "The ratio counts values in Uₖ, Σₖ, and Vₖᵀ. It is an educational "
        "matrix-storage estimate, not a universal image-compression benchmark."
    )
    what_to_observe(
        [
            "Large singular values recover broad image structure first.",
            "Higher rank restores progressively finer detail.",
            "The best rank depends on quality and storage requirements.",
        ]
    )

with tabs[6]:
    scale_multiplier = st.slider("Feature scale multiplier", 1.0, 10.0, 5.0, 0.25)
    outlier_intensity = st.slider("Outlier intensity", 2.0, 12.0, 7.0, 0.25)
    seed_pitfalls = st.number_input("Random seed", 0, 10000, 42, key="seedpit")
    st.plotly_chart(
        pitfall_comparison_figure(
            scale_multiplier,
            outlier_intensity,
            int(seed_pitfalls),
        ),
        width="stretch",
    )
    what_to_observe(
        [
            "Without centering, the dominant direction can point toward the mean offset.",
            "Feature scale can dominate covariance even without greater domain importance.",
            "Outliers have high leverage because PCA optimizes squared error.",
            "Standardization and outlier handling are modeling decisions.",
        ]
    )

with tabs[7]:
    st.latex(r"\lambda_i=\sigma_i^2/(n-1)")
    equivalence_seed = st.number_input("Random seed", 0, 10000, 21, key="seedeq")
    equivalence_components = st.slider("Number of components", 1, 8, 5, key="keq")
    equivalence_data = correlated_features(seed=int(equivalence_seed))
    eigen_result = eigendecomposition_pca(
        equivalence_data,
        equivalence_components,
    )
    svd_result = svd_pca(equivalence_data, equivalence_components)
    aligned_svd = align_component_signs(
        eigen_result.components,
        svd_result.components,
    )
    sklearn_pca = PCA(n_components=equivalence_components).fit(equivalence_data)
    aligned_sklearn = align_component_signs(
        eigen_result.components,
        sklearn_pca.components_,
    )
    comparison = {
        "component": np.arange(1, equivalence_components + 1),
        "covariance eigenvalue": eigen_result.explained_variance,
        "SVD variance": svd_result.explained_variance,
        "|cos(eigh, SVD)|": np.abs(
            np.sum(eigen_result.components * aligned_svd, axis=1)
        ),
        "|cos(eigh, sklearn)|": np.abs(
            np.sum(eigen_result.components * aligned_sklearn, axis=1)
        ),
    }
    st.dataframe(comparison, width="stretch", hide_index=True)
    centered_equivalence, _ = center_data(equivalence_data)
    eig_reconstructed = reconstruct_data(
        centered_equivalence @ eigen_result.components.T,
        eigen_result.components,
        eigen_result.mean,
    )
    svd_reconstructed = reconstruct_data(
        centered_equivalence @ aligned_svd.T,
        aligned_svd,
        svd_result.mean,
    )
    st.metric(
        "Maximum reconstruction difference",
        f"{np.max(np.abs(eig_reconstructed - svd_reconstructed)):.3e}",
    )
    what_to_observe(
        [
            "Eigenvalues match squared singular values divided by n − 1.",
            "Corresponding directions have absolute cosine similarity near one.",
            "Sign flips are valid and leave scores/reconstruction equivalent after alignment.",
        ]
    )

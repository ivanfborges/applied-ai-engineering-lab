"""Write a standalone 3D view of exact KNN geometry on synthetic data."""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import make_classification


RANDOM_STATE = 42
ASSETS = Path(__file__).resolve().parent / "assets"


def make_three_feature_data():
    X, y = make_classification(
        n_samples=120, n_features=3, n_informative=3, n_redundant=0,
        n_repeated=0, n_clusters_per_class=1, class_sep=1.3,
        random_state=RANDOM_STATE,
    )
    query = X.mean(axis=0) + np.array([0.3, -0.2, 0.15])
    return X, y, query


def build_figure(X, y, query, k=5):
    if X.ndim != 2 or X.shape[1] != 3 or query.shape != (3,):
        raise ValueError("This view requires three-dimensional reference data and query")
    if not np.isfinite(X).all() or not np.isfinite(query).all():
        raise ValueError("Coordinates must be finite")
    if not 1 <= k <= len(X):
        raise ValueError("k must be between 1 and the reference count")
    distances = np.linalg.norm(X - query, axis=1)
    selected = np.argsort(distances, kind="stable")[:k]
    fig = go.Figure()
    colors = ("#2878a0", "#d66a35")
    for label in (0, 1):
        ids = np.flatnonzero(y == label)
        fig.add_trace(go.Scatter3d(
            x=X[ids, 0], y=X[ids, 1], z=X[ids, 2],
            mode="markers", name=f"Class {label}",
            marker=dict(size=4, color=colors[label], opacity=0.73),
            customdata=np.column_stack((ids, np.full(len(ids), label), distances[ids])),
            hovertemplate="Sample %{customdata[0]:.0f}<br>Class %{customdata[1]:.0f}"
                          "<br>Distance %{customdata[2]:.3f}<extra></extra>",
        ))
    for i, index in enumerate(selected):
        fig.add_trace(go.Scatter3d(
            x=[query[0], X[index, 0]], y=[query[1], X[index, 1]],
            z=[query[2], X[index, 2]], mode="lines",
            line=dict(color="#263238", width=5), showlegend=False,
            hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter3d(
        x=X[selected, 0], y=X[selected, 1], z=X[selected, 2],
        mode="markers", name=f"{k} nearest", marker=dict(
            size=8, color="#f2ca4d", symbol="circle",
            line=dict(color="#263238", width=2)),
        customdata=np.column_stack((selected, y[selected], distances[selected])),
        hovertemplate="Neighbor %{customdata[0]:.0f}<br>Class %{customdata[1]:.0f}"
                      "<br>Distance %{customdata[2]:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter3d(
        x=[query[0]], y=[query[1]], z=[query[2]],
        mode="markers", name="Query",
        marker=dict(size=10, color="#17242b", symbol="diamond"),
        hovertemplate="Query (not a training row)<extra></extra>",
    ))
    fig.update_layout(
        title="KNN in three informative features (synthetic)<br><sup>Rotate this actual 3D space; it is not a high-dimensional projection.</sup>",
        scene=dict(xaxis_title="Feature 1", yaxis_title="Feature 2",
                   zaxis_title="Feature 3", aspectmode="data"),
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=0, r=0, t=90, b=0),
    )
    return fig, selected, distances[selected]


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    X, y, query = make_three_feature_data()
    fig, selected, distances = build_figure(X, y, query)
    path = ASSETS / "knn_3d.html"
    fig.write_html(str(path), include_plotlyjs=True, full_html=True)
    print(f"Generated {path.name} (standalone HTML, synthetic 3-feature data)")
    print(f"Neighbor indices: {selected.tolist()}")
    print(f"Distances: {np.round(distances, 3).tolist()}")


if __name__ == "__main__":
    main()

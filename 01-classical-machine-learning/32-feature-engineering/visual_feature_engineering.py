"""Generate the Day 32 visual lab locally; no GPU, network or ffmpeg."""
from __future__ import annotations

import argparse
import inspect
import json
import platform
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import FancyBboxPatch
    from PIL import Image
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.tree import DecisionTreeClassifier
except ImportError as error:
    raise SystemExit(
        f"Missing required dependency: {error.name}. Install the shared dependencies "
        'from the repository root with: python -m pip install -e ".[dev]"'
    ) from error

SEED = 42
BLUE, ORANGE, TEAL, RED = "#2563a6", "#d9822b", "#15857b", "#c34243"
GRAY, INK = "#8a96a4", "#23344a"
CLASS_COLORS = [BLUE, ORANGE]
ARTIFACT_NAMES = (
    "01_scaling_before_after.png", "02_scaling_distance.gif",
    "03_categorical_encoding.png", "04_binning.png", "05_binning_resolution.gif",
    "06_interaction_2d.png", "07_interaction_3d.html", "08_polynomial_features.png",
    "09_aggregation_timeline.png", "10_point_in_time_leakage.gif",
    "11_model_inductive_bias.png", "12_feature_engineering_pipeline.png",
    "13_preprocessing_leakage.png",
)


def configure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": GRAY, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "figure.facecolor": "white",
        "axes.facecolor": "#fafbfd", "legend.frameon": False,
    })


def finish(fig, title, note, bottom=0.18):
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.97)
    fig.text(0.5, 0.04, note, ha="center", va="bottom", fontsize=10, linespacing=1.5)
    footer_space = 0.16 + len(note.splitlines()) * 15 / 72 / fig.get_figheight()
    fig.subplots_adjust(top=0.81, bottom=max(bottom, footer_space), wspace=0.32, hspace=0.65)


def save_png(fig, directory, name):
    path = directory / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def save_gif(fig, update, states, directory, name):
    path = directory / name
    animation = FuncAnimation(fig, update, frames=states, interval=500,
                              repeat=True, blit=False, cache_frame_data=False)
    # Repeated states provide readable pauses; Pillow merges identical frames.
    animation.save(path, writer=PillowWriter(fps=2), dpi=100)
    plt.close(fig)
    return path


def nearest_indices(values, reference, k=4):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 2 or values.size == 0 or not np.isfinite(values).all()
            or isinstance(reference, bool) or not isinstance(reference, (int, np.integer))
            or not 0 <= reference < len(values)
            or isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= k < len(values)):
        raise ValueError("Expected finite rows, a valid reference index and 1 <= k < n.")
    distances = np.linalg.norm(values - values[reference], axis=1)
    distances[reference] = np.inf
    return np.argsort(distances, kind="stable")[:k]


def scaling_data():
    rng = np.random.default_rng(SEED)
    values = np.column_stack([rng.uniform(18, 70, 100), rng.uniform(20000, 200000, 100)])
    # Candidate rows expose a unit-driven versus balanced-distance contrast.
    values[:7] = [[44, 100000], [19, 100020], [69, 100050], [22, 100090],
                  [45, 112000], [42, 90000], [46, 106000]]
    train = np.arange(70)
    scaler = StandardScaler().fit(values[train])
    return values, scaler.transform(values), scaler, train


def fit_bins(train, n_bins, strategy):
    values = np.asarray(train, dtype=float)
    if (values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all()
            or np.ptp(values) == 0):
        raise ValueError("Bins require a finite, nonconstant training vector.")
    if (not isinstance(n_bins, (int, np.integer)) or isinstance(n_bins, bool)
            or n_bins < 2 or strategy not in ("uniform", "quantile")):
        raise ValueError("Use at least two bins and uniform or quantile strategy.")
    kwargs = dict(n_bins=n_bins, strategy=strategy, encode="ordinal", subsample=None)
    if "quantile_method" in inspect.signature(KBinsDiscretizer).parameters:
        kwargs["quantile_method"] = "linear"
    return KBinsDiscretizer(**kwargs).fit(values[:, None])


def binning_data():
    rng = np.random.default_rng(SEED + 1)
    train = 18 + 52 * rng.beta(2, 5, 320)
    holdout = 18 + 52 * rng.beta(2, 5, 80)
    return train, holdout


def interaction_data():
    rng = np.random.default_rng(SEED + 2)
    values = rng.uniform(-2, 2, (420, 2))
    labels = (values[:, 0] * values[:, 1] + rng.normal(0, 0.3, len(values)) > 0).astype(int)
    train, holdout = train_test_split(np.arange(len(values)), test_size=0.25,
                                     stratify=labels, random_state=SEED)
    return values, labels, train, holdout


def train_boundary_models(values, labels, train):
    models = [
        make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1500)),
        make_pipeline(PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
                      StandardScaler(), LogisticRegression(C=1, max_iter=1500)),
        DecisionTreeClassifier(max_depth=4, min_samples_leaf=8, random_state=SEED),
    ]
    for model in models:
        model.fit(values[train], labels[train])
    return models


def interaction_plane(model):
    """Return score coefficients in raw (x1, x2, x1*x2) coordinates."""
    scaler, classifier = model.steps[1][1], model.steps[2][1]
    weight = classifier.coef_[0] / scaler.scale_
    intercept = classifier.intercept_[0] - np.dot(weight, scaler.mean_)
    return weight, intercept


def transaction_data():
    cutoff = pd.Timestamp("2026-09-01", tz="UTC")
    offsets = np.array([-42, -30, -25, -18, -12, -6, -3, -1, 0, 2, 6])
    events = pd.DataFrame({
        "customer_id": 101,
        "timestamp": cutoff + pd.to_timedelta(offsets, unit="D"),
        "amount": [80., 45., 120., 65., 180., 95., 150., 70., 200., 210., 110.],
        "merchant": ["A", "A", "B", "C", "A", "B", "D", "C", "A", "D", "B"],
    })
    events["available_at"] = events["timestamp"]
    # The -3 day event occurred in the past but was not yet available at T.
    events.loc[6, "available_at"] = cutoff + pd.Timedelta(days=1)
    return events, cutoff


def eligible_history(events, customer_id, cutoff, days=30):
    required = {"customer_id", "timestamp", "available_at", "amount", "merchant"}
    if not required <= set(events):
        raise ValueError("Missing event columns.")
    if not isinstance(days, (int, np.integer)) or isinstance(days, bool) or days <= 0:
        raise ValueError("days must be a positive integer.")
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:
        raise ValueError("Use a timezone-aware prediction cutoff.")
    events = events.copy()
    for column in ("timestamp", "available_at"):
        events[column] = pd.to_datetime(events[column], utc=True)
    if events[list(required)].isna().any().any():
        raise ValueError("Event fields cannot be missing.")
    amounts = pd.to_numeric(events["amount"], errors="raise")
    if (not np.isfinite(amounts.to_numpy(dtype=float)).all()
            or (events["available_at"] < events["timestamp"]).any()):
        raise ValueError("Require finite amounts and availability at/after event time.")
    events["amount"] = amounts
    mask = ((events["customer_id"] == customer_id)
            & (events["timestamp"] >= cutoff - pd.Timedelta(days=days))
            & (events["timestamp"] < cutoff) & (events["available_at"] < cutoff))
    return events.loc[mask].copy()


def history_features(events, customer_id, cutoff):
    history = eligible_history(events, customer_id, cutoff)
    recent = eligible_history(events, customer_id, cutoff, days=7)
    return {
        "transactions_last_7d": len(recent),
        "transactions_last_30d": len(history),
        "average_amount_30d": float(history["amount"].mean()) if len(history) else 0.,
        "max_amount_30d": float(history["amount"].max()) if len(history) else 0.,
        "distinct_merchants_30d": int(history["merchant"].nunique()),
        "days_since_last_transaction": (
            float((cutoff - history["timestamp"].max()) / pd.Timedelta(days=1))
            if len(history) else None),
    }


def plot_classes(ax, values, labels, title):
    for label, color in enumerate(CLASS_COLORS):
        rows = labels == label
        ax.scatter(values[rows, 0], values[rows, 1], c=color, s=18, alpha=0.75,
                   edgecolors="white", linewidths=0.3, label=f"Class {label}")
    ax.set(xlabel="$x_1$", ylabel="$x_2$", title=title, xlim=(-2.1, 2.1), ylim=(-2.1, 2.1))
    ax.set_aspect("equal")
    ax.axhline(0, color=GRAY, lw=0.6)
    ax.axvline(0, color=GRAY, lw=0.6)


def timeline(ax, events, cutoff, title, show_future=True, show_cutoff=True):
    offset = (events["timestamp"] - cutoff) / pd.Timedelta(days=1)
    available = (offset < 0) & (events["available_at"] < cutoff)
    groups = [(available, TEAL, "Available past"),
              ((offset < 0) & ~available, ORANGE, "Past, arrives after T")]
    if show_future:
        groups.append((offset >= 0, RED, "At/after T: excluded"))
    for mask, color, label in groups:
        ax.vlines(offset[mask], 0, events.loc[mask, "amount"], color=color, alpha=0.35)
        ax.scatter(offset[mask], events.loc[mask, "amount"], color=color, s=65, label=label, zorder=3)
    ax.axvspan(-30, 0, color=TEAL, alpha=0.07)
    ax.axvline(-30, color=GRAY, ls=":", lw=1)
    if show_cutoff:
        ax.axvline(0, color=INK, lw=1.7)
        ax.text(-1, 242, "Prediction T", ha="right", fontsize=10, color=INK)
    ax.set(xlabel="Event time relative to T (days)", ylabel="Transaction amount ($)",
           title=title, xlim=(-45, 9), ylim=(0, 265))
    ax.legend(loc="upper left", fontsize=8)

def create_scaling_visualization(directory):
    values, scaled, scaler, train = scaling_data()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.7))
    axes[0].scatter(values[:, 0], values[:, 1] / 1000, s=16, color=BLUE, alpha=0.7)
    axes[0].set(xlabel="Age (years)", ylabel="Income ($ thousands)",
                title="BEFORE: readable, independent axes")
    axes[1].scatter(values[:, 0], values[:, 1], s=16, color=BLUE, alpha=0.7)
    axes[1].set(xlabel="Age (native numeric units)", ylabel="Income (native numeric units)",
                title="BEFORE: equal numeric units",
                xlim=(-100000, 100000), ylim=(10000, 210000))
    axes[1].set_aspect("equal")
    axes[1].ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    axes[1].text(0.5, 0.08, "Age variation collapses\nrelative to income variation.",
                 ha="center", transform=axes[1].transAxes, fontsize=9)
    axes[2].scatter(scaled[:, 0], scaled[:, 1], s=16, color=TEAL, alpha=0.7)
    axes[2].set(xlabel="Age (training standard deviations)",
                ylabel="Income (training standard deviations)", title="AFTER: StandardScaler")
    axes[2].set_aspect("equal", adjustable="box")
    finish(fig, "Scaling changes the numerical geometry, not the observations",
           "Scaler fitted on 70 training rows; all 100 rows shown. Equal units mean numerical, not physical equality.\n"
           "KNN / K-Means / SVM / PCA depend on scale; trees usually need much less scaling. No universal gain.")
    paths = [save_png(fig, directory, ARTIFACT_NAMES[0])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.1))
    status = fig.text(0.5, 0.89, "", ha="center", fontsize=12, fontweight="bold")
    raw_ids, scaled_ids = nearest_indices(values, 0), nearest_indices(scaled, 0)

    def update(state):
        chosen = raw_ids if state == 0 else scaled_ids
        status.set_text(("RAW Euclidean neighbors" if state == 0 else "STANDARDIZED Euclidean neighbors")
                        + ": " + ", ".join(map(str, chosen)))
        for ax, coords, title in zip(axes, [values, scaled], ["Original units", "Training-standardized units"]):
            ax.clear()
            ax.scatter(coords[:, 0], coords[:, 1], s=16, c=GRAY, alpha=0.35, label="Other rows")
            ax.scatter(coords[chosen, 0], coords[chosen, 1], c=BLUE if state == 0 else TEAL,
                       s=85, edgecolors="white", label="Chosen 4 neighbors", zorder=3)
            ax.scatter(*coords[0], marker="*", c=ORANGE, s=210, edgecolors=INK,
                       linewidths=0.6, label="Reference row 0", zorder=4)
            for row in chosen:
                ax.plot(coords[[0, row], 0], coords[[0, row], 1], color=INK, lw=1, alpha=0.5)
            ax.set_title(title)
            ax.set_xlabel("Age (years)" if title == "Original units" else "Age (standard deviations)")
            ax.set_ylabel("Income ($)" if title == "Original units" else "Income (standard deviations)")
            if title != "Original units":
                ax.set_aspect("equal", adjustable="box")
            ax.legend(loc="upper right", fontsize=8)
    finish(fig, "The same reference row can have different nearest neighbors",
           "Selected row IDs are highlighted in BOTH spaces. Distances are computed in the active units.\n"
           "Raw axes are zoomed independently for readability; scaling is a metric choice, not a guaranteed score improvement.")
    fig.subplots_adjust(top=0.78)
    paths.append(save_gif(fig, update, [0] * 6 + [1] * 8 + [0] * 2,
                          directory, ARTIFACT_NAMES[1]))
    return paths


def create_encoding_visualization(directory):
    categories = ["north", "south", "southeast", "center"]
    rows = np.array(categories + ["south", "north"], dtype=object)[:, None]
    encoder = OneHotEncoder(categories=[categories], sparse_output=False,
                            handle_unknown="ignore").fit(rows[:4])
    encoded = encoder.transform(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), gridspec_kw={"width_ratios": [1, 1.35]})
    ax = axes[0]
    ax.scatter(np.arange(1, 5), np.zeros(4), c=BLUE, s=160, zorder=3)
    for code, name in enumerate(categories, 1):
        ax.text(code, 0.2, name, ha="center", fontsize=10, rotation=20)
        ax.text(code, -0.2, str(code), ha="center", fontsize=11)
    ax.plot([1, 4], [0, 0], c=GRAY, lw=1)
    ax.annotate("", (4, -0.5), (1, -0.5), arrowprops={"arrowstyle": "->", "color": RED})
    ax.text(2.5, -0.7, "Artificial order and unequal distances", ha="center", fontsize=10, color=RED)
    ax.set(xlabel="Arbitrary numeric category code", ylabel="No measured magnitude",
           title="BEFORE: north=1, ..., center=4", xlim=(0.5, 4.5), ylim=(-1, 0.65),
           xticks=[1, 2, 3, 4], yticks=[])
    ax = axes[1]
    ax.imshow(encoded, cmap=ListedColormap(["#eef3f8", BLUE]), vmin=0, vmax=1, aspect="auto")
    for row in range(len(rows)):
        for col in range(4):
            ax.text(col, row, str(int(encoded[row, col])), ha="center", va="center",
                    color="white" if encoded[row, col] else INK)
    ax.set(xticks=np.arange(4), xticklabels=[f"region_{c}" for c in categories],
           yticks=np.arange(len(rows)), yticklabels=rows.ravel(),
           xlabel="Encoded feature columns", ylabel="Original row / region",
           title="AFTER: one-hot indicator matrix")
    ax.tick_params(axis="x", rotation=25)
    finish(fig, "Encoding defines the assumptions and geometry a model sees",
           "One-hot: every distinct pair has Euclidean distance sqrt(2); no ordinal spacing.\n"
           "4 categories: 4 columns. 100,000 categories: potentially costly width, even with sparse storage.",
           bottom=0.28)
    return [save_png(fig, directory, ARTIFACT_NAMES[2])]


def draw_bins(ax, train, holdout, n_bins, strategy, representation=False):
    discretizer = fit_bins(train, n_bins, strategy)
    edges = discretizer.bin_edges_[0]
    codes = discretizer.transform(holdout[:, None]).ravel().astype(int)
    if representation:
        ax.scatter(holdout, codes, s=14, c=codes, cmap="viridis", alpha=0.75)
        ax.set(xlabel="Original age (years)", ylabel="Discrete bin ID",
               title=f"AFTER: {strategy}, {n_bins} bins", yticks=np.arange(len(edges) - 1))
    else:
        ax.hist(train, bins=26, color=BLUE, alpha=0.35, label="Training ages")
        ax.set(xlabel="Age (years)", ylabel="Training observations",
               title=f"{strategy.title()} boundaries: {n_bins} bins")
        counts = np.bincount(discretizer.transform(train[:, None]).ravel().astype(int),
                             minlength=len(edges) - 1)
        for index, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
            ax.axvspan(start, stop, color=plt.get_cmap("viridis")(index / max(1, n_bins - 1)), alpha=0.1)
            ax.text((start + stop) / 2, ax.get_ylim()[1] * (0.92 - 0.13 * (index % 3)), str(counts[index]),
                    ha="center", fontsize=8, color=INK)
    for edge in edges[1:-1]:
        ax.axvline(edge, color=INK, ls=":", lw=1)
    ax.set_xlim(18, 70)


def create_binning_visualization(directory):
    train, holdout = binning_data()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes[0, 0].hist(train, bins=26, color=BLUE, alpha=0.7)
    axes[0, 0].set(title="BEFORE: continuous training ages", xlabel="Age (years)", ylabel="Observations")
    axes[1, 0].scatter(holdout, np.zeros(len(holdout)), color=GRAY, s=18, alpha=0.7)
    axes[1, 0].set(title="Held-out ages retain exact values", xlabel="Age (years)",
                   ylabel="Continuous observations", yticks=[], xlim=(18, 70), ylim=(-1, 1))
    for column, bins in [(1, 4), (2, 8)]:
        draw_bins(axes[0, column], train, holdout, bins, "quantile")
        draw_bins(axes[1, column], train, holdout, bins, "quantile", representation=True)
    finish(fig, "Discretization replaces precise values with interval membership",
           "Numbers above intervals are training counts; quantiles aim for similar counts (ties can prevent equality).\n"
           "Edges fit on 320 training rows; 80 held-out rows shown below. Bin IDs are labels: one-hot them for flexible linear effects.",
           bottom=0.2)
    paths = [save_png(fig, directory, ARTIFACT_NAMES[3])]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7))
    status = fig.text(0.5, 0.86, "", ha="center", fontweight="bold", fontsize=12)

    def update(bins):
        status.set_text(f"{bins} bins: width versus occupancy")
        for col, strategy in enumerate(["uniform", "quantile"]):
            for row in range(2):
                axes[row, col].clear()
                draw_bins(axes[row, col], train, holdout, bins, strategy, representation=bool(row))
    finish(fig, "Binning trades resolution for a simpler representation",
           "Fewer bins: fewer distinct levels, more lost resolution. More bins: finer detail, more parameters if one-hot.\n"
           "Uniform gives equal widths; quantile aims for equal counts. More bins can be less stable on new data.",
           bottom=0.19)
    fig.subplots_adjust(top=0.76, hspace=0.75)
    states = [b for b in [2, 4, 6, 8, 12, 8, 6, 4, 2] for _ in range(2)]
    paths.append(save_gif(fig, update, states, directory, ARTIFACT_NAMES[4]))
    return paths


def create_interaction_visualization(directory):
    values, labels, train, holdout = interaction_data()
    fig, ax = plt.subplots(figsize=(7.2, 6))
    plot_classes(ax, values[holdout], labels[holdout], "Held-out observations: noisy product-dependent labels")
    ax.legend(loc="upper right")
    finish(fig, "A single straight line struggles with opposite-quadrant structure",
           "Synthetic label: class 1 when x1*x2 + Gaussian noise > 0. Noise can put either class near the other.\n"
           "Adding z=x1*x2 exposes the relationship; a plane in (x1,x2,z) pulls back to a curve in (x1,x2).")
    paths = [save_png(fig, directory, ARTIFACT_NAMES[5])]
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Skipped 07_interaction_3d.html: optional Plotly is unavailable.", file=sys.stderr)
        return paths
    model = train_boundary_models(values, labels, train)[1]
    weight, intercept = interaction_plane(model)
    lifted = np.column_stack([values, values[:, 0] * values[:, 1]])
    fig3d = go.Figure()
    for label, color in enumerate(CLASS_COLORS):
        ids = holdout[labels[holdout] == label]
        fig3d.add_trace(go.Scatter3d(
            x=lifted[ids, 0], y=lifted[ids, 1], z=lifted[ids, 2],
            mode="markers", name=f"Class {label} (held out)",
            marker=dict(size=4, color=color, opacity=0.85), customdata=ids,
            hovertemplate="Row %{customdata}<br>x1=%{x:.3f}<br>x2=%{y:.3f}<br>x1*x2=%{z:.3f}<extra>%{fullData.name}</extra>",
        ))
    if abs(weight[2]) > 1e-10:
        grid = np.linspace(-2, 2, 24)
        gx, gy = np.meshgrid(grid, grid)
        gz = -(intercept + weight[0] * gx + weight[1] * gy) / weight[2]
        fig3d.add_trace(go.Surface(x=gx, y=gy, z=gz, name="Fitted p=0.5 plane",
                                  showscale=False, showlegend=True, opacity=0.35,
                                  colorscale=[[0, TEAL], [1, TEAL]],
                                  hovertemplate="Fitted decision plane (score=0)<extra></extra>"))
    fig3d.update_layout(
        title=dict(text="Interaction lift: z = x1*x2<br><sup>Rotate / zoom / hover. A fitted plane; noisy classes are not perfectly separable.</sup>"),
        scene=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="x1*x2",
                   aspectmode="cube", zaxis=dict(range=[-4, 4])),
        margin=dict(l=10, r=10, b=110, t=100), height=800, template="plotly_white",
        annotations=[dict(text="Model and scaler fit only 315 training rows; 105 held-out rows shown. No benchmark or accuracy claim.",
                          x=0.5, y=-0.12, xref="paper", yref="paper", showarrow=False)],
        legend=dict(x=0, y=1),
    )
    path = directory / ARTIFACT_NAMES[6]
    try:
        fig3d.write_html(str(path), include_plotlyjs=True, full_html=True,
                         config={"scrollZoom": True, "displaylogo": False}, auto_open=False)
    except (OSError, ValueError) as error:
        print(f"Skipped optional 07_interaction_3d.html: {error}", file=sys.stderr)
        return paths
    paths.append(path)
    return paths


def create_polynomial_visualization(directory):
    x = np.linspace(-2, 2, 100)
    y = 0.5 + 0.4 * x + 0.8 * x**2
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5))
    axes[0].plot(x, y, color=BLUE, lw=2)
    axes[0].set(title="Original coordinates: a curved response",
                xlabel="Original x", ylabel="y = 0.5 + 0.4x + 0.8x²")
    points = axes[1].scatter(x, x**2, c=y, cmap="viridis", s=20)
    fig.colorbar(points, ax=axes[1], label="Response y", shrink=0.75)
    axes[1].set(title="Feature lift: (x, x²)", xlabel="Feature 1: x", ylabel="Feature 2: x²")
    axes[1].text(0.5, 0.96, "y = β0 + β1·x + β2·x²\nA linear score in these coordinates.",
                 transform=axes[1].transAxes, va="top", ha="center", fontsize=9)
    finish(fig, "Nonlinear features, coefficients that still enter linearly",
           "y = β0 + β1·x1 + β2·x2 + β3·(x1·x2)\n"
           "Linear in β; not necessarily linear in the original feature space. Color indicates y; this is a formula illustration.")
    return [save_png(fig, directory, ARTIFACT_NAMES[7])]

def create_aggregation_visualization(directory):
    events, cutoff = transaction_data()
    features = history_features(events, 101, cutoff)
    fig = plt.figure(figsize=(12, 7))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.3, 1], width_ratios=[1, 1.3])
    ax = fig.add_subplot(grid[0, :])
    timeline(ax, events, cutoff, "RAW EVENTS: customer 101 (merchant and amount per transaction)")
    offsets = (events["timestamp"] - cutoff) / pd.Timedelta(days=1)
    for index, row in events.iterrows():
        ax.annotate(row["merchant"], (offsets.iloc[index], row["amount"]),
                    xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9)
    ax = fig.add_subplot(grid[1, 0])
    ax.axis("off")
    ax.text(0.03, 0.85, "Variable-length event history", fontsize=13, fontweight="bold")
    ax.text(0.03, 0.61, "↓  customer + window + availability cutoff", fontsize=11)
    ax.text(0.03, 0.35, "Fixed-size vector: six features per customer", fontsize=12, fontweight="bold", color=TEAL)
    ax.text(0.03, 0.09, "Count accompanies the mean; more events do not\nchange the number of feature columns.", fontsize=10)
    ax = fig.add_subplot(grid[1, 1])
    ax.axis("off")
    table = ax.table(
        cellText=[[name, f"{value:.2f}" if isinstance(value, float) else str(value)]
                  for name, value in features.items()],
        colLabels=["Feature at T", "Value"], colWidths=[0.8, 0.2],
        cellLoc="left", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#dae3ed")
        cell.set_facecolor("#e8f3f1" if row == 0 else "white")
    finish(fig, "Aggregation turns behavioral history into a fixed-size model input",
           "30-day window = [T−30d, T); events must also be available before T. Orange event arrives late and is excluded.\n"
           "Recency here uses eligible events in that 30-day window; absent history uses count=0, mean/max=0, recency=missing.",
           bottom=0.17)
    fig.subplots_adjust(hspace=0.85)
    return [save_png(fig, directory, ARTIFACT_NAMES[8])]


def create_leakage_visualization(directory):
    events, cutoff = transaction_data()
    correct = history_features(events, 101, cutoff)["transactions_last_30d"]
    leaking = int((events["timestamp"] >= cutoff - pd.Timedelta(days=30)).sum())
    stages = [
        ("1 / 5   Freeze the prediction timestamp", "T is the boundary of what could be known.", ""),
        ("2 / 5   Select eligible past events", "Use [T−30d,T), and check when each event became available.", ""),
        ("3 / 5   Compute the feature available at T", f"CORRECT transactions_last_30d = {correct}", TEAL),
        ("4 / 5   Reveal events occurring at / after T", "These events were not available at the prediction timestamp.", ""),
        ("5 / 5   Deliberately include unavailable events", f"LEAKING count = {leaking}   vs   CORRECT count = {correct}", RED),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    headline = fig.text(0.5, 0.86, "", ha="center", fontsize=12, fontweight="bold")
    message = fig.text(0.5, 0.14, "", ha="center", fontsize=11, fontweight="bold")

    def update(state):
        ax.clear()
        subset = events.iloc[:0] if state == 0 else (
            events.loc[events["timestamp"] < cutoff] if state < 3 else events)
        timeline(ax, subset, cutoff, "Transaction history around the prediction cutoff",
                 show_future=state >= 3)
        headline.set_text(stages[state][0])
        message.set_text(stages[state][1])
        message.set_color(stages[state][2] or INK)
        if state == 4:
            ax.axvspan(0, 9, color=RED, alpha=0.12)
            ax.text(4, 238, "Unavailable\nat T", ha="center", fontsize=10, color=RED)
            ax.text(-18, 225, "Wrong: no upper-time or\navailability filter", color=RED, fontsize=9)
    finish(fig, "Point-in-time correctness: future knowledge is not a feature",
           "At prediction time T, the model can only use information available at or before T.\n"
           "This lab uses a stricter <T contract. The final state is a labeled invalid negative control; no score comparison.",
           bottom=0.28)
    states = [state for state in range(5) for _ in range(4)] + [0, 0]
    return [save_gif(fig, update, states, directory, ARTIFACT_NAMES[9])]


def create_model_comparison_visualization(directory):
    values, labels, train, holdout = interaction_data()
    models = train_boundary_models(values, labels, train)
    axis = np.linspace(-2.1, 2.1, 160)
    gx, gy = np.meshgrid(axis, axis)
    points = np.column_stack([gx.ravel(), gy.ravel()])
    titles = ["Logistic: original features", "Logistic: + x1*x2", "Decision tree: original features"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    cmap = ListedColormap(["#dce8f7", "#fae8d6"])
    for ax, model, title in zip(axes, models, titles):
        probability = model.predict_proba(points)[:, 1].reshape(gx.shape)
        ax.contourf(gx, gy, probability >= 0.5, levels=[-0.5, 0.5, 1.5], cmap=cmap, alpha=0.8)
        if probability.min() < 0.5 < probability.max():
            ax.contour(gx, gy, probability, levels=[0.5], colors=INK, linewidths=1.2)
        plot_classes(ax, values[holdout], labels[holdout], title)
    axes[0].legend(loc="upper right", fontsize=8)
    finish(fig, "The feature map and the model family both shape the boundary",
           "Same 315 training rows; 105 held-out points shown. Fixed C=1; tree depth≤4, leaf size≥8. No ranking or benchmark.\n"
           "Explicit products let a linear score bend in raw coordinates; trees learn axis-aligned combinations.\n"
           "Aggregation and domain features can still help both; scaling is model-dependent.",
           bottom=0.22)
    return [save_png(fig, directory, ARTIFACT_NAMES[10])]


def diagram_box(ax, x, y, width, height, text, color="#e9eff7", size=10):
    box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.012",
                         facecolor=color, edgecolor="#b9c8d9", linewidth=1)
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=size)
    return box


def diagram_arrow(ax, start, stop, color=GRAY):
    ax.annotate("", xy=stop, xytext=start,
                arrowprops={"arrowstyle": "->", "color": color, "lw": 1.5})


def create_pipeline_overview(directory):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.7), gridspec_kw={"width_ratios": [1.65, 1]})
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
    ax = axes[0]
    ax.set_title("RAW DATA → TRANSFORMATION → FEATURE MATRIX", pad=15)
    items = [
        ("Numerical values", "Scaling"),
        ("Nominal categories", "Encoding"),
        ("Continuous values", "Binning"),
        ("Feature combinations", "Interactions"),
        ("Event history", "As-of aggregations"),
    ]
    for index, (source, transform) in enumerate(items):
        y = 0.79 - index * 0.145
        diagram_box(ax, 0.015, y, 0.3, 0.08, source)
        diagram_box(ax, 0.40, y, 0.29, 0.08, transform, color="#e6f3ef")
        diagram_arrow(ax, (0.32, y + 0.04), (0.39, y + 0.04))
        diagram_arrow(ax, (0.705, y + 0.04), (0.81, y + 0.04))
    diagram_box(ax, 0.82, 0.19, 0.16, 0.68, "FEATURE\nMATRIX\n\nOne row\nper\nprediction", size=10)
    diagram_box(ax, 0.82, 0.025, 0.16, 0.09, "MODEL", color="#fff0dc")
    diagram_arrow(ax, (0.90, 0.18), (0.90, 0.12))
    ax = axes[1]
    ax.set_title("FITTED PREPROCESSING", pad=15)
    diagram_box(ax, 0.10, 0.73, 0.8, 0.13, "TRAIN\nfit means, scales, bins, vocabulary", color="#e6f3ef")
    diagram_box(ax, 0.10, 0.44, 0.8, 0.14, "SAVE FITTED TRANSFORMS\nReuse the same feature definitions")
    diagram_box(ax, 0.10, 0.14, 0.8, 0.15, "VALIDATION / TEST / PRODUCTION\ntransform with training parameters", color="#fff0dc")
    diagram_arrow(ax, (0.5, 0.72), (0.5, 0.59))
    diagram_arrow(ax, (0.5, 0.43), (0.5, 0.30))
    finish(fig, "A feature pipeline needs both training isolation and valid information",
           "Never learn preprocessing parameters from validation/test data. Refit inside each training fold.\n"
           "Products use predictors only. Historical aggregation additionally checks each row's prediction-time availability.",
           bottom=0.17)
    return [save_png(fig, directory, ARTIFACT_NAMES[11])]


def create_preprocessing_leakage_visualization(directory):
    rng = np.random.default_rng(SEED + 3)
    train = rng.normal(40, 7, 90)
    holdout = rng.normal(62, 5, 30)
    all_values = np.r_[train, holdout]
    correct = StandardScaler().fit(train[:, None])
    wrong = StandardScaler().fit(all_values[:, None])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, scaler, color, title in [
        (axes[0], wrong, RED, "WRONG: fit all rows → split"),
        (axes[1], correct, TEAL, "CORRECT: split → fit train → transform both"),
    ]:
        ax.hist(train, bins=np.linspace(15, 85, 18), color=BLUE, alpha=0.55, label="Training rows")
        ax.hist(holdout, bins=np.linspace(15, 85, 18), color=ORANGE, alpha=0.55, label="Held-out rows")
        ax.axvline(scaler.mean_[0], color=color, lw=2, label="Fitted mean")
        ax.text(0.04, 0.93, f"Fitted mean = {scaler.mean_[0]:.2f}\nFitted std = {scaler.scale_[0]:.2f}",
                transform=ax.transAxes, va="top", color=color, fontsize=11)
        ax.set(xlabel="Synthetic age (years)", ylabel="Observations", title=title)
        ax.legend(fontsize=9, loc="upper right")
    finish(fig, "Preprocessing leakage exposes held-out distribution information",
           "Both panels show the same data; the red mean uses held-out rows. Holdout shift is deliberate, not a benchmark.\n"
           "Correct preprocessing reuses training parameters even if new values have a different distribution.")
    return [save_png(fig, directory, ARTIFACT_NAMES[12])]


def write_run_record(paths, directory):
    values, scaled, scaler, train = scaling_data()
    events, cutoff = transaction_data()
    record = {
        "hypothesis": "Feature maps change geometry and representable boundaries; as-of filtering changes admissible history.",
        "configuration": {
            "seed": SEED, "scaling_rows": 100, "scaler_training_rows": len(train),
            "bin_training_rows": 320, "bin_holdout_rows": 80,
            "interaction_rows": 420, "interaction_training_rows": 315,
            "interaction_holdout_rows": 105, "label_noise_std": 0.3,
            "models": "Logistic C=1, lbfgs; interaction-only degree 2; tree depth=4, leaf size=8",
            "cutoff": str(cutoff), "event_window": "[T-30d,T), available_at<T",
            "gif_writer": "FuncAnimation + PillowWriter, 2 fps, 100 dpi",
        },
        "results": {
            "artifacts_generated": [path.name for path in paths],
            "raw_neighbor_ids": nearest_indices(values, 0).tolist(),
            "scaled_neighbor_ids": nearest_indices(scaled, 0).tolist(),
            "scaler_mean": scaler.mean_.tolist(),
            "historical_features": history_features(events, 101, cutoff),
        },
        "interpretation_candidate": "Inspect the generated geometry and decision boundaries; no predictive improvement is estimated.",
        "limitations": ["Synthetic constructions and fixed model settings.",
                        "No benchmark, calibration or generalization measurement.",
                        "The leaking feature and all-data scaler are explicitly invalid negative controls.",
                        "Recency is limited to eligible events in the 30-day history window."],
        "review_status": "Visual interpretations pending author review.",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
                     "scikit_learn": sklearn.__version__},
    }
    record_path = directory.parent / "outputs" / "visual_lab_record.json"
    record_path.parent.mkdir(exist_ok=True, parents=True)
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "visuals",
                        help="Default: visuals/ beside this script, independent of working directory.")
    args = parser.parse_args(argv)
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    configure_style()
    paths = []
    generators = [
        create_scaling_visualization, create_encoding_visualization,
        create_binning_visualization, create_interaction_visualization,
        create_polynomial_visualization, create_aggregation_visualization,
        create_leakage_visualization, create_model_comparison_visualization,
        create_pipeline_overview, create_preprocessing_leakage_visualization,
    ]
    for generator in generators:
        print(f"Generating {generator.__name__} ...", flush=True)
        paths.extend(generator(directory))
    write_run_record(paths, directory)
    print("\nVisual Feature Engineering Lab generated successfully.")
    print("\nArtifacts:")
    for path in paths:
        if path.is_file():
            print(path)
    print("\nInspect 01 (scaling), 06 then 07 (interaction lift), and 09 then 10 (history / leakage) first.")
    if directory / ARTIFACT_NAMES[6] in paths:
        print("Open 07_interaction_3d.html in a browser to rotate, zoom and inspect points offline.")
    print("Rerun: python visual_feature_engineering.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

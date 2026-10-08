"""Deterministic SVM visual laboratory on synthetic data, not benchmarks.

Run ``python visualize_svm.py``; outputs are written beside this script.
"""
from __future__ import annotations

import argparse
import json
import platform
import webbrowser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import sklearn
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
C_VALUES = (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100)
GAMMA_VALUES = C_VALUES
DPI = 150
BLUE, ORANGE, GREEN, PURPLE = "#2864B6", "#D77930", "#16734D", "#783CB4"
PALE = ("#E7EFF9", "#FCEDDF")


def configure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12,
        "axes.labelsize": 11, "legend.fontsize": 9, "figure.titlesize": 16,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def signed_margins(y, scores):
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if y.ndim != 1 or y.shape != scores.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("y must be binary {0, 1} labels matching the score vector")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    return (2 * y - 1) * scores


def margin_status(margins):
    margins = np.asarray(margins, dtype=float)
    if margins.ndim != 1 or not np.isfinite(margins).all():
        raise ValueError("margins must be a finite vector")
    return np.where(margins >= 1, 0, np.where(margins > 0, 1, 2))


def rbf_similarity(distance, gamma):
    distance = np.asarray(distance, dtype=float)
    if not np.isfinite(distance).all() or np.any(distance < 0):
        raise ValueError("distance must be finite and nonnegative")
    if not np.isscalar(gamma) or not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be a finite positive scalar")
    return np.exp(-gamma * distance**2)


def lifted_coordinates(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2 or len(X) == 0 or not np.isfinite(X).all():
        raise ValueError("X must be a nonempty finite matrix with two features")
    return np.column_stack([X, np.sum(X**2, axis=1)])


def fitted_svc(model):
    return model if isinstance(model, SVC) else model.steps[-1][1]


def kernel_contributions(model, X):
    """Binary dual terms, in the fitted model's preprocessing coordinates."""
    svm = fitted_svc(model)
    if len(svm.classes_) != 2:
        raise ValueError("dual reconstruction here supports binary models only")
    transformed = X if isinstance(model, SVC) else model[:-1].transform(X)
    if svm.kernel == "rbf":
        # _gamma is libsvm's fitted value, including the gamma='scale' case.
        kernels = rbf_kernel(transformed, svm.support_vectors_, gamma=svm._gamma)
    elif svm.kernel == "linear":
        kernels = linear_kernel(transformed, svm.support_vectors_)
    else:
        raise ValueError("dual reconstruction here supports linear and RBF kernels")
    return kernels * svm.dual_coef_[0], float(svm.intercept_[0])


def make_data():
    separable = make_blobs(n_samples=60, centers=[[-0.2, -2], [0.2, 2]],
                          cluster_std=0.42, random_state=RANDOM_STATE)
    overlapping = make_blobs(n_samples=100, centers=[[-1, -0.5], [1, 0.5]],
                            cluster_std=1.05, random_state=RANDOM_STATE)
    X, y = make_moons(n_samples=180, noise=0.24, random_state=RANDOM_STATE)
    train, validation = train_test_split(np.arange(len(X)), test_size=1 / 3,
                                        stratify=y, random_state=RANDOM_STATE)
    circles = make_circles(n_samples=120, factor=0.35, noise=0.025,
                          random_state=RANDOM_STATE)
    return separable, overlapping, (X[train], y[train], X[validation], y[validation]), circles


def fit_rbf(X, y, C=10, gamma=1):
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=C, gamma=gamma,
                                              tol=1e-6)).fit(X, y)


def mesh_for(X, resolution=160, square=False):
    lower, upper = np.min(X, axis=0), np.max(X, axis=0)
    padding = np.maximum((upper - lower) * 0.12, 0.1)
    if square:
        center = (lower + upper) / 2
        half_width = np.max((upper - lower) / 2 + padding)
        lower, upper = center - half_width, center + half_width
        padding = np.zeros(2)
    xx, yy = np.meshgrid(np.linspace(lower[0] - padding[0], upper[0] + padding[0], resolution),
                         np.linspace(lower[1] - padding[1], upper[1] + padding[1], resolution))
    return xx, yy, np.column_stack([xx.ravel(), yy.ravel()])


def scatter_classes(ax, X, y, *, alpha=1, size=30):
    for label, color, marker in [(0, BLUE, "x"), (1, ORANGE, "o")]:
        mask = y == label
        ax.scatter(X[mask, 0], X[mask, 1], c=color, marker=marker, s=size,
                   alpha=alpha, label=f"Class {2 * label - 1:+d}", zorder=3)


def draw_classifier(ax, X, y, model, *, mesh=None, margins=True,
                    supports=True, fill=True, aspect="equal", points=True):
    xx, yy, grid = mesh if mesh is not None else mesh_for(X)
    score = model.decision_function(grid).reshape(xx.shape)
    if fill:
        ax.contourf(xx, yy, (score > 0).astype(int), levels=[-0.5, 0.5, 1.5], colors=PALE)
    for level in ([-1, 0, 1] if margins else [0]):
        if score.min() < level < score.max():
            ax.contour(xx, yy, score, levels=[level], colors=[GREEN if level == 0 else "#5D6370"],
                       linewidths=2 if level == 0 else 1.2,
                       linestyles="solid" if level == 0 else "dashed")
    if points:
        scatter_classes(ax, X, y)
    if supports:
        indices = fitted_svc(model).support_
        ax.scatter(X[indices, 0], X[indices, 1], s=105, facecolors="none",
                   edgecolors=PURPLE, linewidths=1.4, label="Support vector", zorder=4)
    ax.set(xlim=(xx.min(), xx.max()), ylim=(yy.min(), yy.max()), xlabel="$x_1$", ylabel="$x_2$")
    ax.set_aspect(aspect)
    return score


def classifier_legend(ax, margins=True, loc="best"):
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=GREEN, lw=2, label="$f(x)=0$"))
    if margins:
        handles.append(Line2D([0], [0], color="#5D6370", ls="--", label="$f(x)=\\pm1$"))
    ax.legend(handles=handles, loc=loc, framealpha=0.95)


def save_figure(fig, out, name, footer="Synthetic data; interpretations require author review."):
    fig.text(0.5, 0.018, footer, ha="center", va="bottom", fontsize=10, color="#414754")
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    path = out / name
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def separating_candidates(X, y, model):
    """Construct genuinely separating lines; verify margins rather than guessing."""
    normal = model.coef_[0] / np.linalg.norm(model.coef_[0])
    candidates = []
    for angle, position in [(-0.30, 0.25), (0.30, 0.75), (0.0, 0.20)]:
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        w = rotation @ normal
        projection = X @ w
        low, high = projection[y == 0].max(), projection[y == 1].min()
        if high <= low:
            raise ValueError("candidate construction needs a separating projection gap")
        b = -(low + position * (high - low))
        margin = float(signed_margins(y, X @ w + b).min() / np.linalg.norm(w))
        candidates.append((w, b, margin))
    return candidates


def plot_maximum_margin(X, y, model, out):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    fig.suptitle("Which separating boundary leaves the widest margin?")
    draw_classifier(ax, X, y, model, mesh=mesh_for(X, square=True))
    w, b = model.coef_[0], model.intercept_[0]
    norm = np.linalg.norm(w)
    anchor = -b * w / norm**2
    delta = w / norm**2
    ax.annotate("", xy=anchor + delta, xytext=anchor - delta,
                arrowprops={"arrowstyle": "<->", "color": "#243044", "lw": 1.8})
    ax.annotate(f"Full margin = 2 / ||w||\n= {2 / norm:.3f} feature units",
                xy=anchor, xytext=(0.04, 0.45), textcoords="axes fraction",
                arrowprops={"arrowstyle": "->", "color": "#243044"},
                bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": "#D2D7DF"})
    ax.annotate("Support vectors anchor the margin", xy=X[model.support_[0]], xytext=(0.02, 0.91),
                textcoords="axes fraction", arrowprops={"arrowstyle": "->", "color": PURPLE},
                color=PURPLE, fontsize=10)
    ax.set_title("Solid green: decision boundary | dashed: score ±1 | rings: support vectors")
    classifier_legend(ax, loc="upper right")
    return save_figure(fig, out, "01_maximum_margin.png",
                       "Linear SVC on separable data; high C approximates hard margin (checked numerically).")


def plot_candidate_hyperplanes(X, y, model, out):
    candidates = separating_candidates(X, y, model)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    fig.suptitle("Why isn't zero training error enough?")
    xx, yy, grid = mesh_for(X, square=True)
    scatter_classes(ax, X, y)
    for i, (w, b, margin) in enumerate(candidates, 1):
        ax.contour(xx, yy, (grid @ w + b).reshape(xx.shape), levels=[0],
                   colors=["#8C929D"], linestyles="--", linewidths=1.2)
        ax.plot([], [], "--", color="#8C929D", label=f"Separator {i}: nearest distance {margin:.3f}")
    ax.contour(xx, yy, model.decision_function(grid).reshape(xx.shape), levels=[0], colors=[GREEN], linewidths=3)
    margin = signed_margins(y, model.decision_function(X)).min() / np.linalg.norm(model.coef_[0])
    ax.plot([], [], color=GREEN, lw=3, label=f"SVM: nearest distance {margin:.3f}")
    ax.set(xlabel="$x_1$", ylabel="$x_2$", xlim=(xx.min(), xx.max()), ylim=(yy.min(), yy.max()),
           title="Every shown line separates all training rows; geometric margins differ")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=9)
    return save_figure(fig, out, "02_possible_hyperplanes.png"), candidates


def plot_support_vectors(X, y, model, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle("Which observations directly contribute to the fitted score?")
    mesh = mesh_for(X, square=True)
    draw_classifier(axes[0], X, y, model, mesh=mesh)
    draw_classifier(axes[1], X, y, model, mesh=mesh, points=False, supports=False)
    support = np.zeros(len(X), dtype=bool)
    support[model.support_] = True
    axes[1].scatter(X[~support, 0], X[~support, 1], color="#A6ADBA", s=20, alpha=0.4,
                    label="α = 0 (no direct dual term)")
    scatter_classes(axes[1], X[support], y[support], size=50)
    axes[1].scatter(X[support, 0], X[support, 1], s=120, facecolors="none", edgecolors=PURPLE)
    axes[0].set_title(f"Complete fitted training set: {len(X)} rows")
    axes[1].set_title(f"Same fitted solution: {support.sum()} nonzero dual terms")
    classifier_legend(axes[0], loc="upper left")
    axes[1].legend(handles=[
        Line2D([0], [0], marker="o", color="none", markeredgecolor=PURPLE, label=r"$\alpha > 0$: support"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#A6ADBA", label=r"$\alpha = 0$: no direct term"),
    ], loc="upper left", framealpha=0.95)
    terms, intercept = kernel_contributions(model, mesh[2])
    error = float(np.max(np.abs(terms.sum(axis=1) + intercept - model.decision_function(mesh[2]))))
    print(f"Number of training samples: {len(X)}\nNumber of support vectors: {support.sum()}\n"
          f"Support-vector fraction: {support.mean():.4f}")
    path = save_figure(fig, out, "03_support_vectors.png",
                       "No refitting or point removal: α = 0 means no direct contribution to this fixed dual score.")
    return path, error


def plot_hinge_loss(out):
    margin = np.linspace(-2.5, 3.5, 400)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.suptitle("When does a correctly classified example stop contributing hinge loss?")
    for low, high, color in [(-2.5, 0, "#FBE8E6"), (0, 1, "#FFF1CC"), (1, 3.5, "#E7F3EC")]:
        ax.axvspan(low, high, color=color)
    ax.plot(margin, np.maximum(0, 1 - margin), color=PURPLE, lw=3, label="Hinge: max(0, 1 − m)")
    ax.plot(margin, np.logaddexp(0, -margin), color=BLUE, lw=2, ls="--", label="Logistic: log(1 + exp(−m))")
    for value in [0, 1]:
        ax.axvline(value, color="#687080", lw=1, ls=":")
    ax.text(-2.25, 3.15, "m < 0\nMisclassified", fontsize=11)
    ax.text(0.10, 2.2, "0 < m < 1\nCorrect, but\ninside margin", fontsize=10)
    ax.text(1.25, 1.5, "m ≥ 1\nCorrect, at/beyond margin\nHinge loss = 0", color=GREEN)
    ax.set(xlabel="Signed functional margin m = y f(x)", ylabel="Loss per observation",
           xlim=(-2.5, 3.5), ylim=(-0.12, 3.65))
    ax.legend(loc="upper right")
    return save_figure(fig, out, "04_hinge_loss.png", "A decision score is not a calibrated probability.")


def configuration_metrics(model, X, y, X_validation=None, y_validation=None):
    svm = fitted_svc(model)
    margins = signed_margins(y, model.decision_function(X))
    record = {"support_vectors": len(svm.support_), "support_fraction": len(svm.support_) / len(X),
              "train_accuracy": float(model.score(X, y)), "mean_hinge": float(np.maximum(0, 1 - margins).mean()),
              "margin_status_counts": np.bincount(margin_status(margins), minlength=3).tolist()}
    if svm.kernel == "linear" and isinstance(model, SVC):
        record["full_margin_width"] = float(2 / np.linalg.norm(model.coef_[0]))
    if X_validation is not None:
        record["validation_accuracy"] = float(model.score(X_validation, y_validation))
    return record


def save_parameter_animation(X, y, models, values, parameter, out, name,
                             X_validation=None, y_validation=None):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    mesh = mesh_for(X)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.80, bottom=0.16)
    if parameter == "C":
        fig.suptitle("Increasing C penalizes margin violations more strongly", y=0.96)
        footer = "Same overlapping data; linear SVM. Margin width and support count need not vary monotonically."
    else:
        fig.suptitle("Increasing gamma makes RBF similarity more local", y=0.96)
        footer = "Fixed C = 10; train-fitted scaling. Validation is descriptive; no gamma is universally optimal."
    fig.text(0.5, 0.025, footer, ha="center", fontsize=9)
    rows = [{parameter: float(value), **configuration_metrics(model, X, y, X_validation, y_validation)}
            for value, model in zip(values, models)]

    def draw(index):
        ax.clear()
        model, value, row = models[index], values[index], rows[index]
        draw_classifier(ax, X, y, model, mesh=mesh, margins=parameter == "C")
        if parameter == "C":
            subtitle = f"C = {value:g} | support vectors = {row['support_vectors']}\nFull linear margin = {row['full_margin_width']:.3f}"
        else:
            role = "Broad influence; underfitting risk" if value < 0.1 else (
                "Nonlinear capacity; inspect validation" if value <= 3 else "Very local influence; overfitting risk")
            subtitle = (f"gamma = {value:g} | support vectors = {row['support_vectors']}\n"
                        f"Train = {row['train_accuracy']:.3f} | validation = {row['validation_accuracy']:.3f}\n{role}")
        ax.set_title(subtitle, fontsize=12)
        classifier_legend(ax, margins=parameter == "C")
        return []

    animation = FuncAnimation(fig, draw, frames=len(values), interval=1100, repeat=True,
                              blit=False, cache_frame_data=False)
    path = out / name
    animation.save(path, writer=PillowWriter(fps=1), dpi=DPI)
    plt.close(fig)
    return path, rows


def animate_C_effect(X, y, models, out):
    return save_parameter_animation(X, y, models, C_VALUES, "C", out, "05_C_effect.gif")


def plot_soft_margin_violations(X, y, model, out):
    fig, ax = plt.subplots(figsize=(9, 6.5))
    fig.suptitle("Which observations violate the soft margin?")
    draw_classifier(ax, X, y, model, points=False, supports=False)
    margins = signed_margins(y, model.decision_function(X))
    status = margin_status(margins)
    descriptions = ["m ≥ 1: at/beyond margin", "0 < m < 1: correct, inside margin", "m ≤ 0: wrong side / boundary"]
    for code, color, marker in [(0, GREEN, "o"), (1, "#C28A16", "s"), (2, "#BA3D42", "X")]:
        mask = status == code
        ax.scatter(X[mask, 0], X[mask, 1], c=color, marker=marker, s=42,
                   label=f"{descriptions[code]} (n={mask.sum()})", zorder=3)
    for code, target in [(0, 1.5), (1, 0.5), (2, -0.7)]:
        indices = np.flatnonzero(status == code)
        if len(indices):
            index = indices[np.argmin(np.abs(margins[indices] - target))]
            ax.annotate(f"m = {margins[index]:.2f}\nhinge = {max(0, 1 - margins[index]):.2f}",
                        xy=X[index], xytext={0: (0.76, 0.52), 1: (0.02, 0.48), 2: (0.02, 0.15)}[code],
                        textcoords="axes fraction", fontsize=9,
                        bbox={"fc": "white", "ec": "#D2D7DF", "alpha": 0.95},
                        arrowprops={"arrowstyle": "->", "color": "#5D6370"})
    ax.set_title(f"Same linear C-sweep data, C = {model.C:g}; color now encodes margin status")
    classifier_legend(ax)
    return save_figure(fig, out, "06_soft_margin_violations.png",
                       "At m = 0 the point is on the decision boundary; predicted class depends on the tie rule.")

def circle_lift_models(X, y):
    lifted = lifted_coordinates(X)
    linear_2d = SVC(kernel="linear", C=10000, tol=1e-6).fit(X, y)
    # This finite quadratic lift is an analogy, NOT the RBF feature mapping.
    radial = SVC(kernel="linear", C=10000, tol=1e-8).fit(lifted[:, 2:3], y)
    threshold = float(-radial.intercept_[0] / radial.coef_[0, 0])
    return lifted, linear_2d, radial, threshold


def write_plotly(fig, out, name):
    fig.update_layout(template="plotly_white", font={"family": "Arial", "size": 13},
                      height=730, margin={"t": 110, "b": 170, "l": 35, "r": 30},
                      legend={"orientation": "h", "y": -0.10, "x": 0.5, "xanchor": "center", "yanchor": "top"})
    path = out / name
    # Inline Plotly.js allows rotation/zoom offline; large HTML stays local.
    fig.write_html(path, include_plotlyjs=True, full_html=True, div_id=path.stem,
                   config={"scrollZoom": True, "displaylogo": False, "responsive": True})
    return path


def create_kernel_3d_visualization(X, y, out):
    lifted, linear, radial, threshold = circle_lift_models(X, y)
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "scene"}]],
                        subplot_titles=(f"2D linear fit: training accuracy {linear.score(X, y):.3f}",
                                        f"Explicit lift: radial separator accuracy {radial.score(lifted[:, 2:3], y):.3f}"))
    for label, color in [(0, BLUE), (1, ORANGE)]:
        mask = y == label
        fig.add_trace(go.Scatter(x=X[mask, 0], y=X[mask, 1], mode="markers",
                                marker={"color": color, "size": 6}, name=f"Class {2 * label - 1:+d}"), row=1, col=1)
        fig.add_trace(go.Scatter3d(x=X[mask, 0], y=X[mask, 1], z=lifted[mask, 2], mode="markers",
                                  marker={"color": color, "size": 4}, showlegend=False,
                                  hovertemplate="x1=%{x:.3f}<br>x2=%{y:.3f}<br>z=x1²+x2²=%{z:.3f}<extra></extra>"), row=1, col=2)
    xx, yy, grid = mesh_for(X, 70)
    scores = linear.decision_function(grid).reshape(xx.shape)
    fig.add_trace(go.Contour(x=xx[0], y=yy[:, 0], z=scores,
                            contours={"start": 0, "end": 0, "size": 1, "coloring": "lines"},
                            line={"color": GREEN, "width": 2}, colorscale=[[0, GREEN], [1, GREEN]],
                            showscale=False, showlegend=False,
                            hoverinfo="skip"), row=1, col=1)
    plane_x, plane_y = np.meshgrid([-1.2, 1.2], [-1.2, 1.2])
    fig.add_trace(go.Surface(x=plane_x, y=plane_y, z=np.full((2, 2), threshold),
                            colorscale=[[0, GREEN], [1, GREEN]], opacity=0.35, showscale=False,
                            name="Separating plane", hovertemplate=f"z threshold = {threshold:.4f}<extra></extra>"), row=1, col=2)
    fig.update_xaxes(title_text="x1", scaleanchor="y", row=1, col=1)
    fig.update_yaxes(title_text="x2", row=1, col=1)
    fig.update_layout(title="How can a richer feature space make circles separable?",
                      scene={"xaxis_title": "x1", "yaxis_title": "x2", "zaxis_title": "z = x1² + x2²",
                             "aspectmode": "cube", "camera": {"eye": {"x": 1.4, "y": -1.5, "z": 1.0}}})
    fig.add_annotation(text="Educational finite lift, not an RBF mapping. Kernels compute feature-space inner products implicitly.<br>Rotate / zoom the 3D panel; hover to inspect coordinates. Synthetic circles only.",
                       x=0.5, y=-0.30, xref="paper", yref="paper", showarrow=False)
    html = write_plotly(fig, out, "07_kernel_feature_space.html")
    static = plt.figure(figsize=(12, 5.5))
    static.suptitle("A curved 2D separation can become a plane after an explicit lift")
    ax = static.add_subplot(121)
    draw_classifier(ax, X, y, linear, margins=False, supports=False, fill=False)
    ax.set_title(f"2D linear fit accuracy: {linear.score(X, y):.3f}")
    ax3 = static.add_subplot(122, projection="3d")
    for label, color in [(0, BLUE), (1, ORANGE)]:
        mask = y == label
        ax3.scatter(lifted[mask, 0], lifted[mask, 1], lifted[mask, 2], c=color, s=16)
    ax3.plot_surface(plane_x, plane_y, np.full((2, 2), threshold), color=GREEN, alpha=0.28)
    ax3.set(xlabel="x1", ylabel="x2", zlabel="z = x1² + x2²",
            title=f"Radial separator accuracy: {radial.score(lifted[:, 2:3], y):.3f}")
    png = save_figure(static, out, "07_kernel_feature_space.png",
                      "Explicit quadratic lift is an educational analogy; an RBF kernel does not construct these three coordinates.")
    return [html, png], {"linear_2d_train_accuracy": float(linear.score(X, y)),
                         "radial_lift_train_accuracy": float(radial.score(lifted[:, 2:3], y)),
                         "z_threshold": threshold, "outer_min_z": float(lifted[y == 0, 2].min()),
                         "inner_max_z": float(lifted[y == 1, 2].max())}


def plot_rbf_similarity(out):
    distance = np.linspace(0, 6, 400)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    fig.suptitle("How far does one reference point influence RBF similarity?")
    for gamma, color in zip([0.1, 0.5, 1, 5], [BLUE, GREEN, ORANGE, PURPLE]):
        ax.plot(distance, rbf_similarity(distance, gamma), lw=2, color=color, label=f"gamma = {gamma:g}")
    ax.axhline(np.exp(-1), color="#79818D", ls=":", label="Similarity e⁻¹ at distance 1 / √gamma")
    ax.annotate("Small gamma: broad influence", xy=(3, rbf_similarity(3, 0.1)), xytext=(1.3, 0.82),
                arrowprops={"arrowstyle": "->", "color": BLUE}, color=BLUE)
    ax.annotate("Large gamma: local influence", xy=(0.45, rbf_similarity(0.45, 5)), xytext=(1.8, 0.55),
                arrowprops={"arrowstyle": "->", "color": PURPLE}, color=PURPLE)
    ax.set(xlabel="Distance |x − z| from the fixed reference z = 0", ylabel="RBF similarity exp(−gamma × distance²)",
           xlim=(0, 6), ylim=(-0.03, 1.03))
    ax.legend(loc="upper right")
    return save_figure(fig, out, "08_rbf_similarity.png", "Kernel similarity is not a class probability.")


def animate_gamma_effect(X, y, X_validation, y_validation, models, out):
    return save_parameter_animation(X, y, models, GAMMA_VALUES, "gamma", out,
                                    "09_gamma_effect.gif", X_validation, y_validation)


def plot_C_gamma_grid(X, y, out):
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    fig.suptitle("How do violation cost and kernel locality interact?")
    mesh = mesh_for(X)
    fig.legend(handles=[
        Line2D([0], [0], marker="x", color=BLUE, ls="none", label="Class -1"),
        Line2D([0], [0], marker="o", color=ORANGE, ls="none", label="Class +1"),
        Line2D([0], [0], color=GREEN, lw=2, label="Decision boundary"),
    ], loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=3)
    for row, C in enumerate([0.1, 1, 100]):
        for col, gamma in enumerate([0.1, 1, 10]):
            model = fit_rbf(X, y, C, gamma)
            draw_classifier(axes[row, col], X, y, model, mesh=mesh, margins=False, supports=False)
            axes[row, col].set_title(f"C = {C:g} | gamma = {gamma:g}")
            if row < 2:
                axes[row, col].set_xlabel("")
                axes[row, col].tick_params(labelbottom=False)
            if col > 0:
                axes[row, col].set_ylabel("")
    return save_figure(fig, out, "10_C_gamma_grid.png",
                       "Rows increase C; columns increase gamma. Same training rows and scaler rule in every cell.")


def cv_scores(X, y, C_values, gamma_values, folds=5):
    cv = StratifiedKFold(folds, shuffle=True, random_state=RANDOM_STATE)
    means = np.empty((len(C_values), len(gamma_values)))
    rows = []
    for i, C in enumerate(C_values):
        for j, gamma in enumerate(gamma_values):
            pipeline = make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma, tol=1e-6))
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=1, error_score="raise")
            means[i, j] = scores.mean()
            rows.append({"C": float(C), "gamma": float(gamma), "fold_scores": scores.tolist(),
                         "mean_accuracy": float(scores.mean()), "fold_std": float(scores.std())})
    return means, rows


def plot_C_gamma_heatmap(X, y, out):
    values = [0.01, 0.1, 1, 10, 100]
    means, rows = cv_scores(X, y, values, values)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    fig.suptitle("Which C / gamma combinations validate on this synthetic task?")
    image = ax.imshow(means, cmap="YlGnBu", vmin=0.5, vmax=1.0, origin="lower")
    for i in range(len(values)):
        for j in range(len(values)):
            ax.text(j, i, f"{means[i, j]:.3f}", ha="center", va="center",
                    color="white" if means[i, j] > 0.83 else "#162234", fontsize=12)
    best = np.unravel_index(np.argmax(means), means.shape)
    ax.add_patch(Rectangle((best[1] - 0.5, best[0] - 0.5), 1, 1, fill=False, ec=ORANGE, lw=3))
    ax.set(xticks=range(5), yticks=range(5), xticklabels=values, yticklabels=values,
           xlabel="gamma (kernel locality)", ylabel="C (violation penalty)",
           title="5-fold training-only CV; StandardScaler fitted inside each fold")
    fig.colorbar(image, ax=ax, label="Mean validation accuracy")
    path = save_figure(fig, out, "11_C_gamma_cv_heatmap.png",
                       "Synthetic moons only; highlighted cell is the first maximum. Fold variation is not a confidence interval.")
    return path, rows


def plot_scaling_effect(X, y, X_validation, y_validation, out):
    factors = np.array([1, 1000])
    raw, validation = X * factors, X_validation * factors
    # Keep the gamma='scale' rule in both models; fitted gamma is recorded.
    unscaled = SVC(C=10, gamma="scale", tol=1e-6).fit(raw, y)
    scaled = fit_rbf(raw, y, C=10, gamma="scale")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle("What happens when one feature dominates squared distance?")
    mesh = mesh_for(raw)
    rows = []
    for ax, model, title in zip(axes, [unscaled, scaled], ["No feature scaling", "Training-fitted StandardScaler + SVC"]):
        draw_classifier(ax, raw, y, model, mesh=mesh, margins=False, supports=False, aspect="auto")
        metrics = configuration_metrics(model, raw, y, validation, y_validation)
        metrics["effective_gamma"] = float(fitted_svc(model)._gamma)
        rows.append(metrics)
        ax.set_title(f"{title}\nTrain {metrics['train_accuracy']:.3f} | validation {metrics['validation_accuracy']:.3f}")
        ax.set_ylabel("x2 × 1000 (raw input units)")
    ranges = np.column_stack([raw.min(axis=0), raw.max(axis=0)])
    footer = (f"Training ranges: x1 [{ranges[0, 0]:.2f}, {ranges[0, 1]:.2f}], x2×1000 [{ranges[1, 0]:.1f}, {ranges[1, 1]:.1f}].\n"
              "Same raw display axes; C=10, gamma='scale' in both. The fitted gamma also changes after scaling.")
    return save_figure(fig, out, "12_scaling_effect.png", footer), {"raw_feature_ranges": ranges.tolist(),
                                                                      "unscaled": rows[0], "scaled": rows[1]}


def create_decision_surface_3d(X, y, model, out):
    xx, yy, grid = mesh_for(X, 65)
    scores = model.decision_function(grid).reshape(xx.shape)
    fig = go.Figure()
    fig.add_trace(go.Surface(x=xx, y=yy, z=scores, colorscale=[[0, BLUE], [0.5, "#F4F5F7"], [1, ORANGE]],
                            cmid=0, opacity=0.8, colorbar={"title": "Decision score"}, name="f(x1,x2)",
                            hovertemplate="x1=%{x:.3f}<br>x2=%{y:.3f}<br>f(x)=%{z:.3f}<extra></extra>"))
    fig.add_trace(go.Surface(x=xx, y=yy, z=np.zeros_like(scores), colorscale=[[0, GREEN], [1, GREEN]],
                            opacity=0.25, showscale=False, name="f = 0 plane", hoverinfo="skip"))
    observations = model.decision_function(X)
    for label, color in [(0, BLUE), (1, ORANGE)]:
        mask = y == label
        fig.add_trace(go.Scatter3d(x=X[mask, 0], y=X[mask, 1], z=observations[mask],
                                  mode="markers", marker={"color": color, "size": 4, "line": {"color": "white", "width": 1}},
                                  name=f"Class {2 * label - 1:+d} at its score",
                                  hovertemplate="x1=%{x:.3f}<br>x2=%{y:.3f}<br>f(x)=%{z:.3f}<extra></extra>"))
    # Project the actual zero contour onto z=0 to connect the surface to 2D regions.
    scratch_fig, ax = plt.subplots()
    contour = ax.contour(xx, yy, scores, levels=[0])
    for i, segment in enumerate(contour.allsegs[0]):
        if len(segment):
            fig.add_trace(go.Scatter3d(x=segment[:, 0], y=segment[:, 1], z=np.zeros(len(segment)),
                                      mode="lines", line={"color": GREEN, "width": 7}, name="Decision boundary f=0",
                                      showlegend=i == 0, hoverinfo="skip"))
    plt.close(scratch_fig)
    fig.update_layout(title="How does a numerical score surface define decision regions?",
                      scene={"xaxis_title": "x1", "yaxis_title": "x2", "zaxis_title": "decision_function(x)",
                             "camera": {"eye": {"x": 1.5, "y": -1.6, "z": 1.0}}})
    fig.add_annotation(text="Positive score → class +1; negative score → class −1. Intersection with the green zero plane is the boundary.<br>These are decision scores, not probabilities or distances in input space. Fixed C=10, gamma=1.",
                       x=0.5, y=-0.30, xref="paper", yref="paper", showarrow=False)
    return write_plotly(fig, out, "13_decision_function_3d.html")


def plot_support_vector_count(models, out):
    counts = np.array([len(fitted_svc(model).support_) for model in models])
    total = int(models[0].named_steps["standardscaler"].n_samples_seen_)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    fig.suptitle("How many training observations remain in kernel inference?")
    ax.semilogx(GAMMA_VALUES, counts, color=PURPLE, marker="o", lw=2)
    ax.axhline(total, color="#838B98", ls="--", label=f"All {total} training rows")
    for gamma, count in zip(GAMMA_VALUES, counts):
        ax.annotate(str(count), (gamma, count), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=10)
    ax.set(xlabel="gamma (log scale), fixed C = 10", ylabel="Number of support vectors",
           ylim=(0, total * 1.16), title="Same synthetic moons and preprocessing as the gamma animation")
    ax.legend(loc="lower right")
    return save_figure(fig, out, "14_support_vector_count.png",
                       "Dense RBF prediction evaluates kernels against retained vectors; this is a count, not a latency benchmark.")


def plot_prediction_process(X_query, model, out):
    terms, intercept = kernel_contributions(model, X_query)
    score = float(terms.sum() + intercept)
    predicted = int(model.predict(X_query)[0])
    order = np.argsort(np.abs(terms[0]))[::-1]
    top = order[:3]
    rest = float(terms[0].sum() - terms[0, top].sum())
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle("How does a kernel SVM predict one new observation?")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    boxes = [
        (0.13, 0.78, f"New x = ({X_query[0, 0]:.2f}, {X_query[0, 1]:.2f})\nApply the fitted scaler"),
        (0.50, 0.78, "Compare with ALL support vectors\nK(SVᵢ, x), then multiply by αᵢyᵢ"),
        (0.87, 0.78, f"Sum {terms.shape[1]} weighted terms\nand add b = {intercept:.4f}"),
        (0.50, 0.27, f"Decision score f(x) = {score:.4f}\nPredicted class = {2 * predicted - 1:+d}"),
    ]
    for x, yy, text in boxes:
        ax.text(x, yy, text, ha="center", va="center", fontsize=10,
                bbox={"boxstyle": "round,pad=0.65", "fc": "#F0F3F8", "ec": "#B7C3D5"})
    for start, end in [((0.28, 0.78), (0.33, 0.78)), ((0.68, 0.78), (0.73, 0.78)), ((0.87, 0.64), (0.56, 0.38))]:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 2, "color": GREEN})
    example_terms = "\n".join(f"SV index {fitted_svc(model).support_[i]}: αᵢyᵢ K = {terms[0, i]:+.4f}" for i in top)
    ax.text(0.13, 0.38, f"Largest |contributions|:\n{example_terms}\nOther terms sum = {rest:+.4f}",
            ha="left", va="center", fontsize=10)
    error = abs(score - float(model.decision_function(X_query)[0]))
    path = save_figure(fig, out, "15_kernel_prediction_process.png",
                       "All terms are included in the computed score; shown contributions are examples. Scores are not probabilities.")
    return path, {"query": X_query[0].tolist(), "dual_score": score,
                  "library_score": float(model.decision_function(X_query)[0]),
                  "reconstruction_error": error, "support_vectors": terms.shape[1]}


def experiment(hypothesis, configuration, result, interpretation, limitation):
    return {"hypothesis": hypothesis, "configuration": configuration, "result": result,
            "interpretation_candidate": interpretation, "limitation": limitation,
            "review_status": "pending author review"}


def run_lab(out=OUTPUT_DIR, skip_gifs=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    configure_style()
    separable, overlap, moons, circles = make_data()
    X_sep, y_sep = separable
    X_overlap, y_overlap = overlap
    X, y, X_validation, y_validation = moons
    hard = SVC(kernel="linear", C=10000, tol=1e-9).fit(X_sep, y_sep)
    hard_margins = signed_margins(y_sep, hard.decision_function(X_sep))
    if hard_margins.min() < 1 - 1e-5 or np.max(np.abs(hard.dual_coef_)) >= hard.C:
        raise RuntimeError("canonical data did not achieve a hard-margin-equivalent solution")
    linear_models = [SVC(kernel="linear", C=C, tol=1e-6).fit(X_overlap, y_overlap) for C in C_VALUES]
    gamma_models = [fit_rbf(X, y, C=10, gamma=gamma) for gamma in GAMMA_VALUES]
    assets = [plot_maximum_margin(X_sep, y_sep, hard, out)]
    path, candidates = plot_candidate_hyperplanes(X_sep, y_sep, hard, out)
    assets.append(path)
    path, dual_error = plot_support_vectors(X_sep, y_sep, hard, out)
    assets += [path, plot_hinge_loss(out)]
    linear_rows = [{"C": C, **configuration_metrics(model, X_overlap, y_overlap)}
                   for C, model in zip(C_VALUES, linear_models)]
    gamma_rows = [{"gamma": gamma, **configuration_metrics(model, X, y, X_validation, y_validation)}
                  for gamma, model in zip(GAMMA_VALUES, gamma_models)]
    if not skip_gifs:
        assets.append(animate_C_effect(X_overlap, y_overlap, linear_models, out)[0])
    assets.append(plot_soft_margin_violations(X_overlap, y_overlap, linear_models[4], out))
    paths, lift_result = create_kernel_3d_visualization(*circles, out)
    assets.extend(paths)
    assets.append(plot_rbf_similarity(out))
    if not skip_gifs:
        assets.append(animate_gamma_effect(X, y, X_validation, y_validation, gamma_models, out)[0])
    assets.append(plot_C_gamma_grid(X, y, out))
    path, cv_rows = plot_C_gamma_heatmap(X, y, out)
    assets.append(path)
    path, scaling_result = plot_scaling_effect(X, y, X_validation, y_validation, out)
    assets.append(path)
    model = gamma_models[4]
    assets.extend([create_decision_surface_3d(X, y, model, out), plot_support_vector_count(gamma_models, out)])
    path, prediction_result = plot_prediction_process(X_validation[:1], model, out)
    assets.append(path)
    report = {
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
                        "plotly": plotly.__version__, "pillow": Image.__version__},
        "data": {
            "random_state": RANDOM_STATE,
            "separable_blobs": {"n": 60, "centers": [[-0.2, -2], [0.2, 2]], "std": 0.42},
            "overlapping_blobs": {"n": 100, "centers": [[-1, -0.5], [1, 0.5]], "std": 1.05},
            "moons": {"n": 180, "noise": 0.24, "train_rows": len(X), "validation_rows": len(X_validation),
                      "stratified_validation_fraction": 1 / 3, "split_seed": RANDOM_STATE},
            "circles": {"n": 120, "factor": 0.35, "noise": 0.025},
            "provenance": "Synthetic sklearn generators only; no downloaded observational data.",
        },
        "rendering": {"dpi": DPI, "decision_grid": 160, "surface_grid": 65,
                      "gif_frames": 0 if skip_gifs else 9, "gif_fps": 1,
                      "html": "self-contained, embedded Plotly.js; offline; deterministic div IDs"},
        "experiments": {
            "maximum_margin": experiment("Separable data admit different zero-error lines with different geometric margins.",
                {"kernel": "linear", "C": hard.C, "tol": hard.tol},
                {**configuration_metrics(hard, X_sep, y_sep), "minimum_functional_margin": float(hard_margins.min()),
                 "candidate_nearest_distances": [row[2] for row in candidates], "dual_reconstruction_max_error": dual_error},
                "Compare computed distances and inspect which points anchor this fixed fitted solution.",
                "High-C approximation checked on one separable generator; solver tolerance applies."),
            "C_sweep": experiment("Changing C changes the penalty on violations, rather than directly setting margin width.",
                {"values": list(C_VALUES), "kernel": "linear", "tol": 1e-6, "scaling": "none; comparable original feature units"},
                linear_rows, "Inspect width, violation counts and support counts as conditional behavior.",
                "One overlapping generator; training geometry only; no monotonicity or generalization guarantee."),
            "circle_lift": experiment("An explicit squared-radius coordinate permits linear separation of concentric circles.",
                {"lift": "(x1,x2,x1²+x2²)", "radial_separator_C": 10000, "radial_tol": 1e-8, "linear_2d_tol": 1e-6},
                lift_result, "Inspect radial separation relative to the measured inner/outer radius gap.",
                "Finite educational lift; not the RBF feature mapping; training accuracy only."),
            "gamma_sweep": experiment("RBF locality changes complexity and held-out behavior at fixed C.",
                {"C": 10, "gamma_values": list(GAMMA_VALUES), "tol": 1e-6, "scaling": "training-fitted StandardScaler"},
                gamma_rows, "Compare broad and local boundaries alongside measured training/validation scores.",
                "One seed and descriptive validation sweep; not an independent final test or universal optimum."),
            "CV_grid": experiment("C and gamma should be considered jointly under fold-local preprocessing.",
                {"values": [0.01, 0.1, 1, 10, 100], "folds": 5, "shuffle": True, "seed": RANDOM_STATE,
                 "tol": 1e-6, "scoring": "accuracy", "training_rows_only": True},
                cv_rows, "Inspect the grid as validation evidence for this synthetic sample only.",
                "Small training sample; overlapping folds; best CV score has selection optimism; no final test."),
            "scaling": experiment("A feature multiplied by 1000 can dominate RBF distances without preprocessing.",
                {"factors": [1, 1000], "C": 10, "gamma": "scale", "tol": 1e-6, "scaler_fit": "training rows only"},
                scaling_result, "Compare boundaries and validation metrics after changing the feature-space geometry.",
                "gamma='scale' also changes numerical gamma; a pipeline comparison, not a fixed-gamma ablation."),
            "dual_prediction": experiment("The binary library score equals all weighted support-kernel terms plus the intercept.",
                {"C": 10, "gamma": 1, "query": "first validation row", "scaler": "training-fitted"},
                prediction_result, "Use reconstruction error as a numerical check of the prediction diagram.",
                "One binary query; multiclass coefficient layout differs; no timing or calibration evidence."),
        },
        "artifacts": [{"name": path.name, "bytes": path.stat().st_size, "classification": "regenerable artifact"} for path in assets],
        "review_status": "pending author review",
    }
    record_path = out / "visual_lab_results.json"
    record_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    assets.append(record_path)
    print("\nSVM Visual Lab complete.\n\nGenerated:")
    for path in assets:
        print(f"outputs/{path.name} ({path.stat().st_size / 1024:.1f} KiB)")
    print("\nAll outputs are regenerable and ignored. Interpretations require author review.")
    return assets, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-gifs", action="store_true", help="Generate static/interactive assets without animations")
    parser.add_argument("--show", action="store_true", help="Open the two interactive HTML files and canonical PNG after generation")
    args = parser.parse_args()
    assets, _ = run_lab(skip_gifs=args.skip_gifs)
    if args.show:
        for path in assets:
            if path.suffix == ".html" or path.name == "01_maximum_margin.png":
                webbrowser.open(path.as_uri())


if __name__ == "__main__":
    main()

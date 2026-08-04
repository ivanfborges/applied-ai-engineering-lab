"""Generate a visual learning suite for gradient descent on linear regression."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from example import (
    TRUE_INTERCEPT,
    TRUE_WEIGHT,
    create_synthetic_dataset,
    mean_squared_error,
)
from from_scratch import LinearRegressionGD


if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Mode = Literal["core", "extended", "all"]
RANDOM_SEED = 42
GRID_POINTS = 120
ANIMATION_FRAMES = 96
IMAGE_DPI = 160


@dataclass(frozen=True)
class SafeOptimizationTrace:
    """Finite optimization history plus its termination status."""

    losses: FloatArray
    coefficients: FloatArray
    intercepts: FloatArray
    gradient_norms: FloatArray
    status: Literal["converged", "stopped", "diverged"]

    @property
    def updates(self) -> int:
        """Return the number of completed parameter updates."""
        return max(0, self.losses.size - 1)


@dataclass(frozen=True)
class CoreExperiment:
    """Shared data and fitted models for the core visualizations."""

    x_train: FloatArray
    x_test: FloatArray
    y_train: FloatArray
    y_test: FloatArray
    gd_model: LinearRegressionGD
    sklearn_model: LinearRegression


def require_fitted_model(model: LinearRegressionGD) -> tuple[FloatArray, float]:
    """Return fitted parameters or raise a clear error."""
    if model.coef_ is None or model.intercept_ is None:
        raise RuntimeError("LinearRegressionGD must be fitted first.")
    return model.coef_, model.intercept_


def half_mean_squared_error(
    x: FloatArray,
    y: FloatArray,
    coefficients: FloatArray,
    intercept: float,
) -> float:
    """Evaluate the half-MSE objective used by LinearRegressionGD."""
    residuals = x @ coefficients + intercept - y
    return float(np.dot(residuals, residuals) / (2.0 * y.size))


def fit_core_experiment() -> CoreExperiment:
    """Create deterministic synthetic data and fit GD and OLS references."""
    x, y = create_synthetic_dataset(seed=RANDOM_SEED)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=RANDOM_SEED,
    )

    gd_model = LinearRegressionGD(
        learning_rate=0.05,
        max_iterations=2_000,
        tolerance=1e-10,
    ).fit(x_train, y_train)
    sklearn_model = LinearRegression().fit(x_train, y_train)

    return CoreExperiment(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        gd_model=gd_model,
        sklearn_model=sklearn_model,
    )


def report_reference_comparison(experiment: CoreExperiment) -> None:
    """Print GD and numerical least-squares metrics without conflating solvers."""
    gd_coef, gd_intercept = require_fitted_model(experiment.gd_model)
    sklearn_coef = np.asarray(experiment.sklearn_model.coef_, dtype=np.float64)
    sklearn_intercept = float(experiment.sklearn_model.intercept_)

    gd_train = experiment.gd_model.predict(experiment.x_train)
    gd_test = experiment.gd_model.predict(experiment.x_test)
    ols_train = experiment.sklearn_model.predict(experiment.x_train)
    ols_test = experiment.sklearn_model.predict(experiment.x_test)

    print("\nReference comparison on deterministic synthetic data")
    print("-" * 68)
    print(
        f"{'Model':<24}{'Coefficient':>13}{'Intercept':>12}"
        f"{'Train MSE':>11}{'Test MSE':>10}"
    )
    print(
        f"{'From-scratch batch GD':<24}{gd_coef[0]:>13.6f}"
        f"{gd_intercept:>12.6f}"
        f"{mean_squared_error(experiment.y_train, gd_train):>11.6f}"
        f"{mean_squared_error(experiment.y_test, gd_test):>10.6f}"
    )
    print(
        f"{'scikit-learn OLS':<24}{sklearn_coef[0]:>13.6f}"
        f"{sklearn_intercept:>12.6f}"
        f"{mean_squared_error(experiment.y_train, ols_train):>11.6f}"
        f"{mean_squared_error(experiment.y_test, ols_test):>10.6f}"
    )
    print(
        f"Absolute coefficient difference: "
        f"{abs(gd_coef[0] - sklearn_coef[0]):.3e}"
    )
    print(
        f"Absolute intercept difference:   "
        f"{abs(gd_intercept - sklearn_intercept):.3e}"
    )
    print(
        "Note: scikit-learn LinearRegression solves ordinary least squares "
        "with numerical linear algebra; it does not run this custom GD loop."
    )


def save_figure(
    figure: Figure,
    output_path: Path,
    show: bool,
    displayed_figures: list[Figure],
) -> None:
    """Save a static figure and either retain or close it."""
    figure.savefig(output_path, dpi=IMAGE_DPI, bbox_inches="tight")
    if show:
        displayed_figures.append(figure)
    else:
        plt.close(figure)


def sampled_indices(length: int, maximum_points: int) -> IntArray:
    """Return evenly spaced unique integer indices."""
    if length <= 0:
        raise ValueError("length must be positive.")
    count = min(length, maximum_points)
    return np.unique(np.linspace(0, length - 1, count, dtype=int))


def create_regression_animation(
    experiment: CoreExperiment,
    output_path: Path,
    show: bool,
    displayed_figures: list[Figure],
    animations: list[FuncAnimation],
) -> None:
    """Animate the fitted line along the recorded parameter trajectory."""
    model = experiment.gd_model
    coefficient_history = np.vstack(model.coefficient_history_)[:, 0]
    intercept_history = np.asarray(model.intercept_history_)
    losses = np.asarray(model.loss_history_)
    frame_indices = sampled_indices(losses.size, ANIMATION_FRAMES)

    x_line = np.linspace(
        float(experiment.x_train.min()),
        float(experiment.x_train.max()),
        250,
    )
    true_line = TRUE_WEIGHT * x_line + TRUE_INTERCEPT

    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.scatter(
        experiment.x_train[:, 0],
        experiment.y_train,
        alpha=0.45,
        s=22,
        color="tab:blue",
        label="Synthetic training observations",
    )
    axis.plot(
        x_line,
        true_line,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Data-generating line",
    )
    current_line, = axis.plot(
        [],
        [],
        color="tab:orange",
        linewidth=3,
        label="Current GD line",
    )
    status_text = axis.text(
        0.02,
        0.97,
        "",
        transform=axis.transAxes,
        va="top",
        family="monospace",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.75"},
    )
    axis.set(
        title="Regression Fitting During Batch Gradient Descent",
        xlabel="Feature x",
        ylabel="Target y",
    )
    axis.set_xlim(float(x_line.min()), float(x_line.max()))
    y_margin = 0.08 * float(np.ptp(experiment.y_train))
    axis.set_ylim(
        float(experiment.y_train.min() - y_margin),
        float(experiment.y_train.max() + y_margin),
    )
    axis.grid(alpha=0.2)
    axis.legend(loc="lower right")
    figure.tight_layout()

    def update(frame_number: int) -> tuple[object, ...]:
        history_index = int(frame_indices[frame_number])
        coefficient = coefficient_history[history_index]
        intercept = intercept_history[history_index]
        current_line.set_data(x_line, coefficient * x_line + intercept)
        status_text.set_text(
            f"iteration:   {history_index:4d}\n"
            f"coefficient: {coefficient:8.4f}\n"
            f"intercept:   {intercept:8.4f}\n"
            f"half-MSE:    {losses[history_index]:8.5f}"
        )
        return current_line, status_text

    animation = FuncAnimation(
        figure,
        update,
        frames=frame_indices.size,
        interval=80,
        blit=True,
        repeat=True,
    )
    animation.save(output_path, writer=PillowWriter(fps=12), dpi=115)

    if show:
        displayed_figures.append(figure)
        animations.append(animation)
    else:
        plt.close(figure)


def create_loss_convergence(model: LinearRegressionGD) -> Figure:
    """Plot objective history and the actual stopping update."""
    losses = np.asarray(model.loss_history_)
    iterations = np.arange(losses.size)
    stop_label = (
        "gradient tolerance reached"
        if model.converged_
        else "maximum updates reached"
    )

    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.semilogy(iterations, losses, linewidth=2.3, color="tab:blue")
    axis.scatter([0], [losses[0]], color="tab:orange", zorder=3)
    axis.scatter([iterations[-1]], [losses[-1]], color="tab:green", zorder=3)
    axis.axvline(
        iterations[-1],
        color="tab:red",
        linestyle="--",
        alpha=0.8,
        label=f"Stop: iteration {iterations[-1]} ({stop_label})",
    )
    axis.annotate(
        f"initial = {losses[0]:.4f}",
        (0, losses[0]),
        xytext=(35, -5),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->"},
    )
    axis.annotate(
        f"final = {losses[-1]:.4f}",
        (iterations[-1], losses[-1]),
        xytext=(-105, 35),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->"},
    )
    axis.set(
        title="Training Objective Convergence",
        xlabel="Parameter updates",
        ylabel="Half mean squared error (log scale)",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.text(
        0.5,
        0.01,
        "Convergence demonstrates optimization success on training data, "
        "not necessarily generalization.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    return figure


def create_gradient_norm_plot(model: LinearRegressionGD) -> Figure:
    """Plot the first-order convergence diagnostic."""
    norms = np.asarray(model.gradient_norm_history_)
    iterations = np.arange(norms.size)

    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.semilogy(iterations, norms, linewidth=2.3, color="tab:purple")
    axis.axhline(
        model.tolerance,
        color="black",
        linestyle="--",
        label=f"Stopping tolerance = {model.tolerance:.0e}",
    )
    axis.scatter(
        [iterations[-1]],
        [norms[-1]],
        color="tab:green",
        zorder=3,
        label=f"Final norm = {norms[-1]:.2e}",
    )
    axis.set(
        title="Gradient Norm Approaches a Stationary Point",
        xlabel="Parameter updates",
        ylabel="Euclidean gradient norm (log scale)",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    return figure


def create_parameter_convergence(
    values: FloatArray,
    parameter_name: str,
    true_value: float,
    sklearn_value: float,
) -> Figure:
    """Plot one parameter history with generating and OLS references."""
    iterations = np.arange(values.size)
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(
        iterations,
        values,
        linewidth=2.3,
        label=f"GD {parameter_name.lower()}",
    )
    axis.axhline(
        true_value,
        color="black",
        linestyle=":",
        linewidth=2,
        label=f"Data-generating value = {true_value:.4f}",
    )
    axis.axhline(
        sklearn_value,
        color="tab:red",
        linestyle="--",
        linewidth=1.8,
        label=f"scikit-learn OLS = {sklearn_value:.4f}",
    )
    axis.scatter(
        [iterations[-1]],
        [values[-1]],
        color="tab:green",
        zorder=3,
        label=f"Final GD = {values[-1]:.4f}",
    )
    axis.set(
        title=f"{parameter_name} Convergence",
        xlabel="Parameter updates",
        ylabel=parameter_name,
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    return figure


def parameter_grid(
    experiment: CoreExperiment,
    points: int = GRID_POINTS,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Build a compact loss grid containing the path and OLS optimum."""
    if points < 20:
        raise ValueError("points must be at least 20.")

    coefficient_history = np.vstack(
        experiment.gd_model.coefficient_history_
    )[:, 0]
    intercept_history = np.asarray(experiment.gd_model.intercept_history_)
    coefficient_values = np.append(
        coefficient_history,
        float(experiment.sklearn_model.coef_[0]),
    )
    intercept_values = np.append(
        intercept_history,
        float(experiment.sklearn_model.intercept_),
    )

    coefficient_span = max(float(np.ptp(coefficient_values)), 1.0)
    intercept_span = max(float(np.ptp(intercept_values)), 1.0)
    coefficients = np.linspace(
        float(coefficient_values.min() - 0.15 * coefficient_span),
        float(coefficient_values.max() + 0.15 * coefficient_span),
        points,
    )
    intercepts = np.linspace(
        float(intercept_values.min() - 0.15 * intercept_span),
        float(intercept_values.max() + 0.15 * intercept_span),
        points,
    )
    coefficient_grid, intercept_grid = np.meshgrid(coefficients, intercepts)
    losses = univariate_loss_grid(
        experiment.x_train[:, 0],
        experiment.y_train,
        coefficient_grid,
        intercept_grid,
    )
    return coefficient_grid, intercept_grid, losses


def univariate_loss_grid(
    x: FloatArray,
    y: FloatArray,
    coefficient_grid: FloatArray,
    intercept_grid: FloatArray,
) -> FloatArray:
    """Evaluate half-MSE on a coefficient/intercept grid without 3D residuals."""
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("x and y must be aligned one-dimensional arrays.")
    if coefficient_grid.shape != intercept_grid.shape:
        raise ValueError("coefficient and intercept grids must have one shape.")

    mean_x2 = float(np.mean(x**2))
    mean_y2 = float(np.mean(y**2))
    mean_xy = float(np.mean(x * y))
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    loss = 0.5 * (
        coefficient_grid**2 * mean_x2
        + intercept_grid**2
        + mean_y2
        + 2.0 * coefficient_grid * intercept_grid * mean_x
        - 2.0 * coefficient_grid * mean_xy
        - 2.0 * intercept_grid * mean_y
    )
    return np.maximum(loss, np.finfo(np.float64).eps)


def contour_levels(loss_grid: FloatArray, count: int = 28) -> FloatArray:
    """Create logarithmically spaced levels for a positive loss grid."""
    minimum = max(float(loss_grid.min()), np.finfo(np.float64).eps)
    maximum = float(loss_grid.max())
    if np.isclose(minimum, maximum):
        maximum = minimum * 1.01
    return np.geomspace(minimum, maximum, count)


def create_loss_contour(
    experiment: CoreExperiment,
    coefficient_grid: FloatArray,
    intercept_grid: FloatArray,
    loss_grid: FloatArray,
) -> Figure:
    """Plot actual parameter history over logarithmic loss contours."""
    coefficients = np.vstack(experiment.gd_model.coefficient_history_)[:, 0]
    intercepts = np.asarray(experiment.gd_model.intercept_history_)
    direction_indices = sampled_indices(coefficients.size, 24).astype(int)
    levels = contour_levels(loss_grid)

    figure, axis = plt.subplots(figsize=(9, 6.5))
    filled = axis.contourf(
        coefficient_grid,
        intercept_grid,
        loss_grid,
        levels=levels,
        norm=LogNorm(vmin=levels[0], vmax=levels[-1]),
        cmap="viridis",
    )
    axis.contour(
        coefficient_grid,
        intercept_grid,
        loss_grid,
        levels=levels[::2],
        colors="white",
        linewidths=0.45,
        alpha=0.45,
    )
    axis.plot(
        coefficients,
        intercepts,
        color="white",
        linewidth=2,
        alpha=0.9,
        label="Recorded GD trajectory",
    )

    starts = direction_indices[:-1]
    ends = direction_indices[1:]
    axis.quiver(
        coefficients[starts],
        intercepts[starts],
        coefficients[ends] - coefficients[starts],
        intercepts[ends] - intercepts[starts],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="tab:orange",
        width=0.005,
        zorder=4,
        label="Optimization direction",
    )
    axis.scatter(
        coefficients[0],
        intercepts[0],
        s=90,
        color="tab:orange",
        edgecolor="black",
        label="Initial parameters",
        zorder=5,
    )
    axis.scatter(
        coefficients[-1],
        intercepts[-1],
        s=230,
        marker="*",
        color="lime",
        edgecolor="black",
        label="Final GD parameters",
        zorder=5,
    )
    axis.scatter(
        float(experiment.sklearn_model.coef_[0]),
        float(experiment.sklearn_model.intercept_),
        s=70,
        marker="X",
        color="red",
        edgecolor="white",
        label="scikit-learn OLS",
        zorder=5,
    )
    figure.colorbar(filled, ax=axis, label="Half mean squared error")
    axis.set(
        title="Gradient Descent Across the Loss Contours",
        xlabel="Coefficient",
        ylabel="Intercept",
    )
    axis.legend(loc="upper right")
    figure.tight_layout()
    return figure


def create_static_loss_surface(
    experiment: CoreExperiment,
    coefficient_grid: FloatArray,
    intercept_grid: FloatArray,
    loss_grid: FloatArray,
) -> Figure:
    """Create a Matplotlib 3D loss surface with the recorded trajectory."""
    coefficients = np.vstack(experiment.gd_model.coefficient_history_)[:, 0]
    intercepts = np.asarray(experiment.gd_model.intercept_history_)
    losses = np.asarray(experiment.gd_model.loss_history_)
    lift = 0.012 * float(loss_grid.max() - loss_grid.min())
    ols_coef = np.asarray([float(experiment.sklearn_model.coef_[0])])
    ols_intercept = float(experiment.sklearn_model.intercept_)
    ols_loss = half_mean_squared_error(
        experiment.x_train,
        experiment.y_train,
        ols_coef,
        ols_intercept,
    )

    figure = plt.figure(figsize=(10, 7))
    axis = figure.add_subplot(111, projection="3d")
    surface = axis.plot_surface(
        coefficient_grid,
        intercept_grid,
        loss_grid,
        cmap="viridis",
        alpha=0.72,
        linewidth=0,
        antialiased=True,
    )
    axis.plot(
        coefficients,
        intercepts,
        losses + lift,
        color="tab:red",
        linewidth=3,
        label="GD trajectory (slightly lifted)",
    )
    axis.scatter(
        coefficients[0],
        intercepts[0],
        losses[0] + lift,
        color="black",
        s=55,
        label="Initial point",
    )
    axis.scatter(
        coefficients[-1],
        intercepts[-1],
        losses[-1] + lift,
        color="lime",
        edgecolor="black",
        s=150,
        label="Final point",
    )
    axis.scatter(
        ols_coef[0],
        ols_intercept,
        ols_loss + lift,
        color="orange",
        marker="X",
        s=55,
        label="scikit-learn OLS",
    )
    axis.set(
        title="Linear Regression Loss Surface and Optimization Path",
        xlabel="Coefficient",
        ylabel="Intercept",
        zlabel="Half mean squared error",
    )
    axis.view_init(elev=32, azim=-128)
    axis.legend(loc="upper left")
    figure.colorbar(surface, ax=axis, shrink=0.62, pad=0.10, label="Half-MSE")
    figure.text(
        0.5,
        0.02,
        "The trajectory is lifted slightly above the surface for visibility; "
        "the stored losses are unchanged.",
        ha="center",
        fontsize=9,
    )
    return figure


def create_interactive_loss_surface(
    experiment: CoreExperiment,
    coefficient_grid: FloatArray,
    intercept_grid: FloatArray,
    loss_grid: FloatArray,
    output_path: Path,
) -> None:
    """Write a browser-openable Plotly loss surface and trajectory."""
    import plotly.graph_objects as go

    coefficients = np.vstack(experiment.gd_model.coefficient_history_)[:, 0]
    intercepts = np.asarray(experiment.gd_model.intercept_history_)
    losses = np.asarray(experiment.gd_model.loss_history_)
    iterations = np.arange(losses.size)
    ols_coef = float(experiment.sklearn_model.coef_[0])
    ols_intercept = float(experiment.sklearn_model.intercept_)
    ols_loss = half_mean_squared_error(
        experiment.x_train,
        experiment.y_train,
        np.asarray([ols_coef]),
        ols_intercept,
    )

    figure = go.Figure()
    figure.add_trace(
        go.Surface(
            x=coefficient_grid,
            y=intercept_grid,
            z=loss_grid,
            colorscale="Viridis",
            opacity=0.82,
            colorbar={"title": "Half-MSE"},
            name="Loss surface",
            hovertemplate=(
                "coefficient=%{x:.4f}<br>intercept=%{y:.4f}"
                "<br>loss=%{z:.5f}<extra>surface</extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=coefficients,
            y=intercepts,
            z=losses,
            mode="lines+markers",
            marker={"size": 2.5, "color": iterations, "colorscale": "Plasma"},
            line={"width": 6, "color": "red"},
            customdata=iterations,
            name="Actual GD trajectory",
            hovertemplate=(
                "iteration=%{customdata}<br>coefficient=%{x:.6f}"
                "<br>intercept=%{y:.6f}<br>loss=%{z:.6f}"
                "<extra>GD trajectory</extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=[coefficients[0], coefficients[-1], ols_coef],
            y=[intercepts[0], intercepts[-1], ols_intercept],
            z=[losses[0], losses[-1], ols_loss],
            mode="markers",
            marker={
                "size": [7, 9, 8],
                "color": ["black", "lime", "orange"],
                "symbol": ["circle", "diamond", "x"],
            },
            text=["Initial point", "Final GD point", "scikit-learn OLS"],
            customdata=[0, int(iterations[-1]), -1],
            name="Reference points",
            hovertemplate=(
                "%{text}<br>iteration=%{customdata}<br>coefficient=%{x:.6f}"
                "<br>intercept=%{y:.6f}<br>loss=%{z:.6f}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title="Interactive Linear Regression Loss Surface",
        scene={
            "xaxis_title": "Coefficient",
            "yaxis_title": "Intercept",
            "zaxis_title": "Half mean squared error",
            "camera": {"eye": {"x": 1.55, "y": -1.55, "z": 1.15}},
        },
        legend={"x": 0.01, "y": 0.99},
        margin={"l": 0, "r": 0, "b": 0, "t": 55},
    )
    figure.write_html(
        output_path,
        include_plotlyjs="cdn",
        full_html=True,
    )


def create_single_gradient_step(
    experiment: CoreExperiment,
) -> Figure:
    """Illustrate one update in coefficient/intercept parameter space."""
    model = experiment.gd_model
    coefficients = np.vstack(model.coefficient_history_)[:, 0]
    intercepts = np.asarray(model.intercept_history_)
    step_index = min(8, coefficients.size - 2)
    current = np.asarray(
        [coefficients[step_index], intercepts[step_index]],
        dtype=np.float64,
    )
    next_point = np.asarray(
        [coefficients[step_index + 1], intercepts[step_index + 1]],
        dtype=np.float64,
    )
    positive_point = current + (current - next_point)

    all_points = np.vstack([current, next_point, positive_point])
    spans = np.maximum(np.ptp(all_points, axis=0), 0.25)
    coefficient_values = np.linspace(
        float(all_points[:, 0].min() - 0.8 * spans[0]),
        float(all_points[:, 0].max() + 0.8 * spans[0]),
        100,
    )
    intercept_values = np.linspace(
        float(all_points[:, 1].min() - 0.8 * spans[1]),
        float(all_points[:, 1].max() + 0.8 * spans[1]),
        100,
    )
    coefficient_grid, intercept_grid = np.meshgrid(
        coefficient_values,
        intercept_values,
    )
    local_losses = univariate_loss_grid(
        experiment.x_train[:, 0],
        experiment.y_train,
        coefficient_grid,
        intercept_grid,
    )

    figure, axis = plt.subplots(figsize=(9, 6.5))
    axis.contour(
        coefficient_grid,
        intercept_grid,
        local_losses,
        levels=18,
        cmap="viridis",
        linewidths=1,
    )
    axis.quiver(
        current[0],
        current[1],
        positive_point[0] - current[0],
        positive_point[1] - current[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="tab:red",
        width=0.008,
        label=r"Scaled gradient direction $+\eta\nabla J$",
    )
    axis.quiver(
        current[0],
        current[1],
        next_point[0] - current[0],
        next_point[1] - current[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="tab:green",
        width=0.008,
        label=r"Update direction $-\eta\nabla J$",
    )
    axis.scatter(
        current[0],
        current[1],
        color="black",
        s=80,
        zorder=4,
        label=r"Current parameters $\theta_t$",
    )
    axis.scatter(
        next_point[0],
        next_point[1],
        color="tab:green",
        s=90,
        marker="*",
        zorder=4,
        label=r"Next parameters $\theta_{t+1}$",
    )
    axis.annotate(
        f"learning rate η = {model.learning_rate:g}",
        xy=((current[0] + next_point[0]) / 2, (current[1] + next_point[1]) / 2),
        xytext=(10, -35),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "tab:green"},
    )
    axis.set(
        title="One Gradient Descent Parameter Update",
        xlabel="Coefficient",
        ylabel="Intercept",
    )
    axis.text(
        0.5,
        0.02,
        r"$\theta_{t+1} = \theta_t - \eta\nabla J(\theta_t)$",
        transform=axis.transAxes,
        ha="center",
        fontsize=13,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )
    axis.grid(alpha=0.15)
    axis.legend(loc="best")
    figure.tight_layout()
    return figure


def safe_batch_gradient_descent(
    x: FloatArray,
    y: FloatArray,
    learning_rate: float,
    max_updates: int,
    tolerance: float = 1e-8,
    loss_limit: float = 1e6,
    parameter_limit: float = 1e5,
) -> SafeOptimizationTrace:
    """Run full-batch GD with explicit guards for comparison experiments."""
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.size:
        raise ValueError("x and y must have aligned two- and one-dimensional shapes.")
    if learning_rate <= 0 or max_updates <= 0:
        raise ValueError("learning_rate and max_updates must be positive.")

    coefficients = np.zeros(x.shape[1], dtype=np.float64)
    intercept = 0.0
    losses: list[float] = []
    coefficient_history: list[FloatArray] = []
    intercept_history: list[float] = []
    gradient_norms: list[float] = []
    status: Literal["converged", "stopped", "diverged"] = "stopped"

    with np.errstate(over="ignore", invalid="ignore"):
        for update in range(max_updates + 1):
            residuals = x @ coefficients + intercept - y
            loss = float(np.dot(residuals, residuals) / (2.0 * y.size))
            coefficient_gradient = x.T @ residuals / y.size
            intercept_gradient = float(np.mean(residuals))
            gradient_norm = float(
                np.sqrt(
                    np.dot(coefficient_gradient, coefficient_gradient)
                    + intercept_gradient**2
                )
            )

            is_finite = (
                np.isfinite(loss)
                and np.all(np.isfinite(coefficients))
                and np.isfinite(intercept)
                and np.all(np.isfinite(coefficient_gradient))
                and np.isfinite(intercept_gradient)
            )
            too_large = (
                loss > loss_limit
                or np.max(np.abs(coefficients), initial=0.0) > parameter_limit
                or abs(intercept) > parameter_limit
            )
            if not is_finite:
                status = "diverged"
                break

            losses.append(loss)
            coefficient_history.append(coefficients.copy())
            intercept_history.append(intercept)
            gradient_norms.append(gradient_norm)

            if too_large:
                status = "diverged"
                break
            if gradient_norm <= tolerance:
                status = "converged"
                break
            if update == max_updates:
                status = "stopped"
                break

            coefficients -= learning_rate * coefficient_gradient
            intercept -= learning_rate * intercept_gradient

    if not losses:
        losses = [loss_limit]
        coefficient_history = [coefficients.copy()]
        intercept_history = [intercept]
        gradient_norms = [np.inf]

    return SafeOptimizationTrace(
        losses=np.asarray(losses, dtype=np.float64),
        coefficients=np.vstack(coefficient_history),
        intercepts=np.asarray(intercept_history, dtype=np.float64),
        gradient_norms=np.asarray(gradient_norms, dtype=np.float64),
        status=status,
    )


def create_learning_rate_comparison(
    experiment: CoreExperiment,
) -> Figure:
    """Compare slow, stable, fast, and divergent learning-rate behavior."""
    learning_rates = [0.0001, 0.001, 0.01, 0.05, 0.2, 1.0, 1.5]
    traces = {
        rate: safe_batch_gradient_descent(
            experiment.x_train,
            experiment.y_train,
            learning_rate=rate,
            max_updates=2_000,
            tolerance=1e-8,
        )
        for rate in learning_rates
    }
    display_ceiling = float(traces[0.05].losses[0] * 100.0)

    figure, axis = plt.subplots(figsize=(10, 6))
    for rate, trace in traces.items():
        clipped_losses = np.minimum(trace.losses, display_ceiling)
        label = f"η={rate:g} ({trace.status})"
        line, = axis.semilogy(
            np.arange(clipped_losses.size),
            clipped_losses,
            linewidth=2,
            label=label,
        )
        if trace.status == "diverged":
            axis.scatter(
                trace.updates,
                clipped_losses[-1],
                color=line.get_color(),
                marker="X",
                s=70,
                zorder=4,
            )
            axis.annotate(
                "diverged",
                (trace.updates, clipped_losses[-1]),
                xytext=(5, -15),
                textcoords="offset points",
                color=line.get_color(),
            )

    axis.set_ylim(bottom=max(float(traces[0.05].losses[-1]) * 0.8, 1e-4))
    axis.set(
        title="Learning Rate Controls Speed and Stability",
        xlabel="Parameter updates",
        ylabel="Half mean squared error (log scale, divergent curves clipped)",
    )
    axis.grid(alpha=0.2)
    axis.legend(ncol=2)

    oscillatory_trace = traces[1.0]
    oscillatory_error = (
        oscillatory_trace.coefficients[:, 0]
        - float(experiment.sklearn_model.coef_[0])
    )
    inset = axis.inset_axes([0.58, 0.20, 0.37, 0.25])
    inset.plot(
        np.arange(min(45, oscillatory_error.size)),
        oscillatory_error[:45],
        color="tab:brown",
        linewidth=1.5,
    )
    inset.axhline(0.0, color="black", linestyle=":", linewidth=1)
    inset.set(
        title="η=1 coefficient error alternates",
        xlabel="Updates",
        ylabel="w − w*",
    )
    inset.tick_params(labelsize=7)
    inset.title.set_fontsize(8)
    inset.xaxis.label.set_fontsize(7)
    inset.yaxis.label.set_fontsize(7)
    inset.grid(alpha=0.15)
    figure.tight_layout()

    print("\nLearning-rate comparison")
    print("-" * 58)
    print(f"{'Learning rate':>14}{'Final loss':>16}{'Updates':>12}{'Status':>14}")
    for rate, trace in traces.items():
        print(
            f"{rate:>14g}{trace.losses[-1]:>16.6g}"
            f"{trace.updates:>12d}{trace.status:>14}"
        )
    return figure


def create_feature_scaling_comparison(seed: int = RANDOM_SEED) -> Figure:
    """Contrast ill-conditioned raw features with training-only scaling."""
    rng = np.random.default_rng(seed)
    n_samples = 320
    x_small = rng.normal(0.0, 1.0, n_samples)
    x_large = rng.normal(0.0, 100_000.0, n_samples)
    x = np.column_stack([x_small, x_large])
    y = 3.0 * x_small + 0.00004 * x_large + 1.25
    y += rng.normal(0.0, 0.8, n_samples)

    x_train, _, y_train, _ = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=seed,
    )
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    raw_design = np.column_stack([x_train, np.ones(x_train.shape[0])])
    scaled_design = np.column_stack(
        [x_train_scaled, np.ones(x_train_scaled.shape[0])]
    )
    raw_hessian = raw_design.T @ raw_design / raw_design.shape[0]
    scaled_hessian = scaled_design.T @ scaled_design / scaled_design.shape[0]
    raw_condition = float(np.linalg.cond(raw_hessian))
    scaled_condition = float(np.linalg.cond(scaled_hessian))
    largest_raw_eigenvalue = float(np.linalg.eigvalsh(raw_hessian).max())
    stable_raw_rate = 0.9 / largest_raw_eigenvalue
    shared_rate = 0.05

    raw_shared = safe_batch_gradient_descent(
        x_train,
        y_train,
        learning_rate=shared_rate,
        max_updates=1_200,
    )
    scaled_shared = safe_batch_gradient_descent(
        x_train_scaled,
        y_train,
        learning_rate=shared_rate,
        max_updates=1_200,
    )
    raw_tuned = safe_batch_gradient_descent(
        x_train,
        y_train,
        learning_rate=stable_raw_rate,
        max_updates=1_200,
    )

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ceiling = max(float(raw_shared.losses[0] * 100.0), 100.0)
    axes[0].semilogy(
        np.minimum(raw_shared.losses, ceiling),
        linewidth=2,
        label=f"Raw features ({raw_shared.status})",
    )
    axes[0].semilogy(
        scaled_shared.losses,
        linewidth=2,
        label=f"Standardized ({scaled_shared.status})",
    )
    axes[0].set(
        title=f"Same Learning Rate η={shared_rate:g}",
        xlabel="Parameter updates",
        ylabel="Half-MSE (divergent values clipped)",
    )
    axes[0].legend()

    axes[1].semilogy(
        raw_tuned.losses,
        linewidth=2,
        label=f"Raw, η={stable_raw_rate:.1e} ({raw_tuned.status})",
    )
    axes[1].semilogy(
        scaled_shared.losses,
        linewidth=2,
        label=f"Standardized, η={shared_rate:g} ({scaled_shared.status})",
    )
    axes[1].set(
        title="Stable Rates Still Reveal Conditioning",
        xlabel="Parameter updates",
        ylabel="Half mean squared error",
    )
    axes[1].legend()

    for axis in axes:
        axis.grid(alpha=0.2)

    figure.suptitle(
        "Feature Scaling Changes Optimization Geometry",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.01,
        "StandardScaler was fitted on training features only. Scaling changes "
        "conditioning, not the underlying linear relationship.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))

    print("\nFeature-scaling experiment")
    print("-" * 68)
    print(f"Raw Hessian condition number:        {raw_condition:.3e}")
    print(f"Standardized Hessian condition number:{scaled_condition:.3e}")
    print(f"Same-rate raw status:                {raw_shared.status}")
    print(f"Same-rate standardized status:       {scaled_shared.status}")
    print(
        "Scaling changes the curvature and conditioning of the objective. "
        "It does not fundamentally change the regression problem, and the "
        "scaler was fitted using training data only."
    )
    return figure


def minibatch_optimization(
    x: FloatArray,
    y: FloatArray,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    seed: int,
) -> FloatArray:
    """Train with shuffled batches and record full-data loss after each update."""
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.size:
        raise ValueError("x and y must have aligned shapes.")
    if not 1 <= batch_size <= x.shape[0]:
        raise ValueError("batch_size must be between 1 and the sample count.")
    if learning_rate <= 0 or epochs <= 0:
        raise ValueError("learning_rate and epochs must be positive.")

    rng = np.random.default_rng(seed)
    coefficients = np.zeros(x.shape[1], dtype=np.float64)
    intercept = 0.0
    losses = [half_mean_squared_error(x, y, coefficients, intercept)]

    for _ in range(epochs):
        shuffled_indices = rng.permutation(x.shape[0])
        for start in range(0, x.shape[0], batch_size):
            indices = shuffled_indices[start : start + batch_size]
            x_batch = x[indices]
            y_batch = y[indices]
            residuals = x_batch @ coefficients + intercept - y_batch
            coefficient_gradient = x_batch.T @ residuals / indices.size
            intercept_gradient = float(np.mean(residuals))
            coefficients -= learning_rate * coefficient_gradient
            intercept -= learning_rate * intercept_gradient
            losses.append(half_mean_squared_error(x, y, coefficients, intercept))

    return np.asarray(losses, dtype=np.float64)


def create_batch_strategy_comparison(
    experiment: CoreExperiment,
) -> Figure:
    """Compare loss noise and update counts across three batch strategies."""
    n_samples = experiment.x_train.shape[0]
    configurations = [
        ("Full batch", n_samples, 0.05),
        ("Mini-batch (32)", 32, 0.02),
        ("Stochastic (1)", 1, 0.003),
    ]
    histories = {
        name: minibatch_optimization(
            experiment.x_train,
            experiment.y_train,
            batch_size=batch_size,
            learning_rate=rate,
            epochs=80,
            seed=RANDOM_SEED,
        )
        for name, batch_size, rate in configurations
    }

    figure, axis = plt.subplots(figsize=(10, 6))
    for name, batch_size, rate in configurations:
        history = histories[name]
        axis.semilogy(
            np.arange(history.size),
            history,
            linewidth=1.7,
            label=(
                f"{name}: {history.size - 1:,} updates, "
                f"η={rate:g}"
            ),
        )
    axis.set_xscale("symlog", linthresh=10)
    axis.set(
        title="Batch Size Trades Gradient Stability for Update Frequency",
        xlabel="Parameter updates (symlog scale)",
        ylabel="Full-training half-MSE (log scale)",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.text(
        0.5,
        0.01,
        "One epoch is one data pass; the number of parameter updates per epoch "
        "depends on batch size. This does not establish generalization quality.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))

    print("\nBatch-strategy comparison (80 epochs each)")
    print("-" * 64)
    for name, _, _ in configurations:
        history = histories[name]
        print(
            f"{name:<20} updates={history.size - 1:>6,}  "
            f"final half-MSE={history[-1]:.6f}"
        )
    return figure


def create_outlier_sensitivity(seed: int = RANDOM_SEED) -> Figure:
    """Show how extreme target values influence a squared-error fit."""
    x, y = create_synthetic_dataset(n_samples=140, seed=seed)
    y_with_outliers = y.copy()
    sorted_indices = np.argsort(x[:, 0])
    outlier_indices = sorted_indices[[18, 70, 121]]
    y_with_outliers[outlier_indices] += np.asarray([20.0, -24.0, 22.0])

    clean_model = LinearRegression().fit(x, y)
    outlier_model = LinearRegression().fit(x, y_with_outliers)
    x_line = np.linspace(float(x.min()), float(x.max()), 250).reshape(-1, 1)

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.scatter(
        x[:, 0],
        y,
        alpha=0.45,
        s=25,
        color="tab:blue",
        label="Original synthetic observations",
    )
    axis.scatter(
        x[outlier_indices, 0],
        y_with_outliers[outlier_indices],
        s=80,
        marker="X",
        color="tab:red",
        label="Injected target outliers",
        zorder=4,
    )
    axis.plot(
        x_line[:, 0],
        clean_model.predict(x_line),
        linewidth=2.5,
        color="tab:blue",
        label="MSE fit without outliers",
    )
    axis.plot(
        x_line[:, 0],
        outlier_model.predict(x_line),
        linewidth=2.5,
        linestyle="--",
        color="tab:red",
        label="MSE fit after outliers",
    )
    axis.set(
        title="Squared Residuals Give Extreme Errors High Influence",
        xlabel="Feature x",
        ylabel="Target y",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    return figure


def generate_core_visualizations(
    experiment: CoreExperiment,
    output_dir: Path,
    show: bool,
    displayed_figures: list[Figure],
    animations: list[FuncAnimation],
) -> list[Path]:
    """Generate the essential geometry and convergence visualizations."""
    model = experiment.gd_model
    coefficients = np.vstack(model.coefficient_history_)[:, 0]
    intercepts = np.asarray(model.intercept_history_)
    coefficient_grid, intercept_grid, loss_grid = parameter_grid(experiment)
    generated: list[Path] = []

    animation_path = output_dir / "regression_fitting.gif"
    create_regression_animation(
        experiment,
        animation_path,
        show,
        displayed_figures,
        animations,
    )
    generated.append(animation_path)

    static_figures = [
        (
            create_loss_convergence(model),
            output_dir / "loss_convergence.png",
        ),
        (
            create_gradient_norm_plot(model),
            output_dir / "gradient_norm.png",
        ),
        (
            create_parameter_convergence(
                coefficients,
                "Coefficient",
                TRUE_WEIGHT,
                float(experiment.sklearn_model.coef_[0]),
            ),
            output_dir / "coefficient_convergence.png",
        ),
        (
            create_parameter_convergence(
                intercepts,
                "Intercept",
                TRUE_INTERCEPT,
                float(experiment.sklearn_model.intercept_),
            ),
            output_dir / "intercept_convergence.png",
        ),
        (
            create_loss_contour(
                experiment,
                coefficient_grid,
                intercept_grid,
                loss_grid,
            ),
            output_dir / "loss_contour.png",
        ),
        (
            create_static_loss_surface(
                experiment,
                coefficient_grid,
                intercept_grid,
                loss_grid,
            ),
            output_dir / "loss_surface_3d.png",
        ),
        (
            create_single_gradient_step(experiment),
            output_dir / "single_gradient_step.png",
        ),
    ]
    for figure, path in static_figures:
        save_figure(figure, path, show, displayed_figures)
        generated.append(path)

    interactive_path = output_dir / "loss_surface_interactive.html"
    create_interactive_loss_surface(
        experiment,
        coefficient_grid,
        intercept_grid,
        loss_grid,
        interactive_path,
    )
    generated.append(interactive_path)
    return generated


def generate_extended_visualizations(
    experiment: CoreExperiment,
    output_dir: Path,
    show: bool,
    displayed_figures: list[Figure],
) -> list[Path]:
    """Generate learning-rate, scaling, batching, and outlier experiments."""
    figures = [
        (
            create_learning_rate_comparison(experiment),
            output_dir / "learning_rate_comparison.png",
        ),
        (
            create_feature_scaling_comparison(),
            output_dir / "feature_scaling_comparison.png",
        ),
        (
            create_batch_strategy_comparison(experiment),
            output_dir / "batch_strategy_comparison.png",
        ),
        (
            create_outlier_sensitivity(),
            output_dir / "outlier_sensitivity.png",
        ),
    ]
    generated: list[Path] = []
    for figure, path in figures:
        save_figure(figure, path, show, displayed_figures)
        generated.append(path)
    return generated


def parse_args() -> argparse.Namespace:
    """Parse visualization mode, display behavior, and output location."""
    parser = argparse.ArgumentParser(
        description="Generate visual explanations of gradient descent.",
    )
    parser.add_argument(
        "--mode",
        choices=("core", "extended", "all"),
        default="core",
        help="core geometry, extended experiments, or every visualization.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display Matplotlib figures after saving them.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for generated artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    """Fit the shared experiment and generate the requested visualization set."""
    args = parse_args()
    mode: Mode = args.mode
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment = fit_core_experiment()
    report_reference_comparison(experiment)

    displayed_figures: list[Figure] = []
    animations: list[FuncAnimation] = []
    generated: list[Path] = []

    if mode in {"core", "all"}:
        generated.extend(
            generate_core_visualizations(
                experiment,
                output_dir,
                args.show,
                displayed_figures,
                animations,
            )
        )
    if mode in {"extended", "all"}:
        generated.extend(
            generate_extended_visualizations(
                experiment,
                output_dir,
                args.show,
                displayed_figures,
            )
        )

    print(f"\nGenerated {len(generated)} artifact(s) in {output_dir}:")
    for path in generated:
        print(f"  - {path.name}")

    if args.show:
        print("\nClose the figure windows to finish the process.")
        plt.show()
        for figure in displayed_figures:
            plt.close(figure)


if __name__ == "__main__":
    main()

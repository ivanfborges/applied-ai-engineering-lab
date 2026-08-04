"""Train linear regression on synthetic data and visualize convergence."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from from_scratch import LinearRegressionGD


TRUE_WEIGHT = 4.2
TRUE_INTERCEPT = -1.5


def create_synthetic_dataset(
    n_samples: int = 250,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic observations from y = 4.2x - 1.5 + noise."""
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=0.0, scale=1.5, size=(n_samples, 1))
    noise = rng.normal(loc=0.0, scale=1.0, size=n_samples)
    y = TRUE_WEIGHT * x[:, 0] + TRUE_INTERCEPT + noise
    return x, y


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return mean squared prediction error."""
    return float(np.mean((y_pred - y_true) ** 2))


def save_regression_plot(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    gd_model: LinearRegressionGD,
    sklearn_model: LinearRegression,
    output_path: Path,
) -> Figure:
    """Save the synthetic observations and both fitted regression lines."""
    import matplotlib.pyplot as plt

    x_line = np.linspace(
        min(float(x_train.min()), float(x_test.min())),
        max(float(x_train.max()), float(x_test.max())),
        250,
    ).reshape(-1, 1)

    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.scatter(
        x_train[:, 0],
        y_train,
        alpha=0.5,
        s=24,
        label="Synthetic training data",
    )
    axis.scatter(
        x_test[:, 0],
        y_test,
        alpha=0.75,
        s=30,
        marker="x",
        label="Synthetic test data",
    )
    axis.plot(
        x_line[:, 0],
        gd_model.predict(x_line),
        linewidth=2.5,
        label="Batch gradient descent",
    )
    axis.plot(
        x_line[:, 0],
        sklearn_model.predict(x_line),
        linestyle="--",
        linewidth=2,
        label="scikit-learn OLS",
    )
    axis.set(
        title="Linear Regression on Synthetic Data",
        xlabel="Feature x",
        ylabel="Target y",
    )
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    return figure


def save_convergence_plot(
    model: LinearRegressionGD,
    output_path: Path,
) -> Figure:
    """Save loss, gradient-norm, and parameter histories."""
    import matplotlib.pyplot as plt

    losses = np.asarray(model.loss_history_)
    gradient_norms = np.asarray(model.gradient_norm_history_)
    coefficient_history = np.vstack(model.coefficient_history_)[:, 0]
    intercept_history = np.asarray(model.intercept_history_)
    iterations = np.arange(losses.size)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].semilogy(iterations, losses, color="tab:blue")
    axes[0].set(
        title="Objective Convergence",
        xlabel="Parameter updates",
        ylabel="Half mean squared error",
    )

    axes[1].semilogy(iterations, gradient_norms, color="tab:orange")
    axes[1].axhline(
        model.tolerance,
        color="black",
        linestyle=":",
        linewidth=1.5,
        label="Tolerance",
    )
    axes[1].set(
        title="First-Order Convergence",
        xlabel="Parameter updates",
        ylabel="Gradient L2 norm",
    )
    axes[1].legend()

    axes[2].plot(iterations, coefficient_history, label="Estimated coefficient")
    axes[2].plot(iterations, intercept_history, label="Estimated intercept")
    axes[2].axhline(
        TRUE_WEIGHT,
        color="tab:blue",
        linestyle=":",
        linewidth=1.5,
        label="Generating coefficient",
    )
    axes[2].axhline(
        TRUE_INTERCEPT,
        color="tab:orange",
        linestyle=":",
        linewidth=1.5,
        label="Generating intercept",
    )
    axes[2].set(
        title="Parameter Trajectory",
        xlabel="Parameter updates",
        ylabel="Parameter value",
    )
    axes[2].legend(fontsize=8)

    for axis in axes:
        axis.grid(alpha=0.2)

    figure.suptitle("Batch Gradient Descent Diagnostics", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    return figure


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    default_output = Path(__file__).resolve().parent / "outputs"
    parser = argparse.ArgumentParser(
        description="Fit a synthetic regression problem with gradient descent."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory for generated PNG files.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures after saving them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Saving is the default, so avoid requiring a GUI backend in CI or shells
    # without a working Tk installation. --show keeps Matplotlib's configured
    # interactive backend.
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, y = create_synthetic_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
    )

    gd_model = LinearRegressionGD(
        learning_rate=0.05,
        max_iterations=2_000,
        tolerance=1e-10,
    ).fit(x_train, y_train)

    sklearn_model = LinearRegression().fit(x_train, y_train)

    gd_predictions = gd_model.predict(x_test)
    sklearn_predictions = sklearn_model.predict(x_test)

    if gd_model.coef_ is None or gd_model.intercept_ is None:
        raise RuntimeError("Gradient-descent model did not produce parameters.")

    regression_path = args.output_dir / "regression_fit.png"
    convergence_path = args.output_dir / "convergence.png"
    figures = [
        save_regression_plot(
            x_train,
            y_train,
            x_test,
            y_test,
            gd_model,
            sklearn_model,
            regression_path,
        ),
        save_convergence_plot(gd_model, convergence_path),
    ]

    print("Dataset: synthetic linear observations (fixed random seed)")
    print(
        f"Data-generating relationship: y = {TRUE_WEIGHT:.1f}x "
        f"{TRUE_INTERCEPT:+.1f} + noise"
    )
    print("\nBatch gradient descent")
    print(f"  Coefficient: {gd_model.coef_[0]:.6f}")
    print(f"  Intercept:   {gd_model.intercept_:.6f}")
    print(f"  Test MSE:    {mean_squared_error(y_test, gd_predictions):.6f}")
    print(f"  Updates:     {gd_model.n_iterations_}")
    print(f"  Converged:   {gd_model.converged_}")
    print(f"  Final |grad|:{gd_model.gradient_norm_history_[-1]:.3e}")

    print("\nscikit-learn ordinary least squares")
    print(f"  Coefficient: {sklearn_model.coef_[0]:.6f}")
    print(f"  Intercept:   {sklearn_model.intercept_:.6f}")
    print(
        f"  Test MSE:    "
        f"{mean_squared_error(y_test, sklearn_predictions):.6f}"
    )

    parameter_difference = np.linalg.norm(
        np.array([gd_model.coef_[0], gd_model.intercept_])
        - np.array([sklearn_model.coef_[0], sklearn_model.intercept_])
    )
    print(f"\nParameter-vector difference: {parameter_difference:.3e}")
    print(f"Saved: {regression_path}")
    print(f"Saved: {convergence_path}")

    if args.show:
        plt.show()
    else:
        for figure in figures:
            plt.close(figure)


if __name__ == "__main__":
    main()

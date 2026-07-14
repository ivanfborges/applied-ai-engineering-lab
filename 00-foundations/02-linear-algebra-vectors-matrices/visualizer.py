"""Interactive visualizations for fundamental vector and matrix operations."""

from __future__ import annotations

import argparse
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.widgets import Slider


LIMIT = 6.0


def configure_plane(ax: Axes, title: str, limit: float = LIMIT) -> None:
    """Configure a Cartesian plane with equal scale on both axes."""
    ax.clear()
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(alpha=0.25)


def draw_vector(
    ax: Axes,
    vector: np.ndarray,
    color: str,
    label: str,
    origin: np.ndarray | None = None,
    linestyle: str = "-",
) -> None:
    """Draw a two-dimensional vector and label its endpoint."""
    start = np.zeros(2) if origin is None else origin
    ax.quiver(
        start[0],
        start[1],
        vector[0],
        vector[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color=color,
        linestyle=linestyle,
        width=0.008,
    )
    endpoint = start + vector
    ax.text(endpoint[0] + 0.12, endpoint[1] + 0.12, label, color=color)


def add_slider(
    figure: plt.Figure,
    position: list[float],
    label: str,
    initial: float,
    minimum: float = -4.0,
    maximum: float = 4.0,
) -> Slider:
    """Create a consistently styled slider."""
    slider_axis = figure.add_axes(position)
    return Slider(slider_axis, label, minimum, maximum, valinit=initial, valstep=0.1)


def show_vector_addition() -> None:
    """Visualize a + b using arrows and the parallelogram rule."""
    figure, ax = plt.subplots(figsize=(8, 8))
    figure.subplots_adjust(bottom=0.28)
    sliders = {
        "ax": add_slider(figure, [0.15, 0.18, 0.3, 0.03], "aₓ", 2.0),
        "ay": add_slider(figure, [0.58, 0.18, 0.3, 0.03], "aᵧ", 1.0),
        "bx": add_slider(figure, [0.15, 0.11, 0.3, 0.03], "bₓ", -1.0),
        "by": add_slider(figure, [0.58, 0.11, 0.3, 0.03], "bᵧ", 2.0),
    }

    def update(_: float | None = None) -> None:
        a = np.array([sliders["ax"].val, sliders["ay"].val])
        b = np.array([sliders["bx"].val, sliders["by"].val])
        result = a + b
        configure_plane(ax, f"Vector addition: a + b = {np.round(result, 2)}")
        draw_vector(ax, a, "tab:blue", "a")
        draw_vector(ax, b, "tab:orange", "b")
        draw_vector(ax, result, "tab:green", "a + b")
        draw_vector(ax, b, "tab:orange", "b translated", origin=a, linestyle="--")
        draw_vector(ax, a, "tab:blue", "a translated", origin=b, linestyle="--")
        figure.canvas.draw_idle()

    for slider in sliders.values():
        slider.on_changed(update)
    update()
    plt.show()


def show_dot_product_and_projection() -> None:
    """Visualize alignment, angle, dot product, and projection."""
    figure, ax = plt.subplots(figsize=(8, 8))
    figure.subplots_adjust(bottom=0.28)
    sliders = {
        "ax": add_slider(figure, [0.15, 0.18, 0.3, 0.03], "aₓ", 3.0),
        "ay": add_slider(figure, [0.58, 0.18, 0.3, 0.03], "aᵧ", 2.0),
        "bx": add_slider(figure, [0.15, 0.11, 0.3, 0.03], "bₓ", 2.0),
        "by": add_slider(figure, [0.58, 0.11, 0.3, 0.03], "bᵧ", -1.0),
    }

    def update(_: float | None = None) -> None:
        a = np.array([sliders["ax"].val, sliders["ay"].val])
        b = np.array([sliders["bx"].val, sliders["by"].val])
        dot = float(a @ b)
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        configure_plane(ax, "Dot product and projection")
        draw_vector(ax, a, "tab:blue", "a")
        draw_vector(ax, b, "tab:orange", "b")

        if norm_b > 0:
            projection = (dot / (b @ b)) * b
            draw_vector(ax, projection, "tab:green", "proj_b(a)")
            ax.plot(
                [a[0], projection[0]],
                [a[1], projection[1]],
                color="tab:green",
                linestyle="--",
            )

        if norm_a > 0 and norm_b > 0:
            cosine = float(np.clip(dot / (norm_a * norm_b), -1.0, 1.0))
            angle = float(np.degrees(np.arccos(cosine)))
            detail = f"a · b = {dot:.2f}\ncos(θ) = {cosine:.3f}\nθ = {angle:.1f}°"
        else:
            detail = f"a · b = {dot:.2f}\nAngle undefined for a zero vector"

        ax.text(
            0.02,
            0.98,
            detail,
            transform=ax.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.85},
        )
        figure.canvas.draw_idle()

    for slider in sliders.values():
        slider.on_changed(update)
    update()
    plt.show()


def show_norms_and_distances() -> None:
    """Compare L1 and L2 distances between two adjustable points."""
    figure, ax = plt.subplots(figsize=(8, 8))
    figure.subplots_adjust(bottom=0.28)
    sliders = {
        "px": add_slider(figure, [0.15, 0.18, 0.3, 0.03], "pₓ", -2.0),
        "py": add_slider(figure, [0.58, 0.18, 0.3, 0.03], "pᵧ", -1.0),
        "qx": add_slider(figure, [0.15, 0.11, 0.3, 0.03], "qₓ", 3.0),
        "qy": add_slider(figure, [0.58, 0.11, 0.3, 0.03], "qᵧ", 2.0),
    }

    def update(_: float | None = None) -> None:
        p = np.array([sliders["px"].val, sliders["py"].val])
        q = np.array([sliders["qx"].val, sliders["qy"].val])
        delta = q - p
        l1 = float(np.linalg.norm(delta, ord=1))
        l2 = float(np.linalg.norm(delta, ord=2))
        linf = float(np.linalg.norm(delta, ord=np.inf))

        configure_plane(ax, "Distances between points p and q")
        ax.scatter(*p, color="tab:blue", s=70, label="p")
        ax.scatter(*q, color="tab:orange", s=70, label="q")
        ax.text(p[0] + 0.12, p[1] + 0.12, "p", color="tab:blue")
        ax.text(q[0] + 0.12, q[1] + 0.12, "q", color="tab:orange")
        ax.plot([p[0], q[0]], [p[1], q[1]], color="tab:green", label="L2: straight line")
        ax.plot(
            [p[0], q[0], q[0]],
            [p[1], p[1], q[1]],
            color="tab:red",
            linestyle="--",
            label="One L1 path",
        )
        ax.text(
            0.02,
            0.98,
            f"L1 = {l1:.2f}\nL2 = {l2:.2f}\nL∞ = {linf:.2f}",
            transform=ax.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.85},
        )
        ax.legend(loc="lower right")
        figure.canvas.draw_idle()

    for slider in sliders.values():
        slider.on_changed(update)
    update()
    plt.show()


def show_matrix_transformation() -> None:
    """Apply scaling, rotation, and shear to a grid and unit square."""
    figure, ax = plt.subplots(figsize=(8, 8))
    figure.subplots_adjust(bottom=0.31)
    sliders = {
        "sx": add_slider(figure, [0.15, 0.21, 0.3, 0.03], "scale x", 1.5, -2.0, 2.0),
        "sy": add_slider(figure, [0.58, 0.21, 0.3, 0.03], "scale y", 0.8, -2.0, 2.0),
        "angle": add_slider(figure, [0.15, 0.14, 0.3, 0.03], "angle", 30.0, -180.0, 180.0),
        "shear": add_slider(figure, [0.58, 0.14, 0.3, 0.03], "shear", 0.0, -2.0, 2.0),
    }
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
    grid_values = np.arange(-3.0, 4.0)

    def update(_: float | None = None) -> None:
        angle = np.deg2rad(sliders["angle"].val)
        scaling = np.array([[sliders["sx"].val, 0.0], [0.0, sliders["sy"].val]])
        shear = np.array([[1.0, sliders["shear"].val], [0.0, 1.0]])
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        transformation = rotation @ shear @ scaling
        configure_plane(ax, "Matrix transformation: rotation @ shear @ scaling", limit=7.0)

        for value in grid_values:
            horizontal = np.array([[-3.0, value], [3.0, value]])
            vertical = np.array([[value, -3.0], [value, 3.0]])
            for line in (horizontal, vertical):
                transformed_line = line @ transformation.T
                ax.plot(
                    transformed_line[:, 0],
                    transformed_line[:, 1],
                    color="lightsteelblue",
                    linewidth=0.8,
                )

        transformed_square = square @ transformation.T
        ax.plot(square[:, 0], square[:, 1], "--", color="gray", label="Original square")
        ax.fill(
            transformed_square[:, 0],
            transformed_square[:, 1],
            color="tab:blue",
            alpha=0.3,
            label="Transformed square",
        )
        draw_vector(ax, transformation @ np.array([1.0, 0.0]), "tab:red", "T(e₁)")
        draw_vector(ax, transformation @ np.array([0.0, 1.0]), "tab:green", "T(e₂)")
        ax.text(
            0.02,
            0.98,
            f"T =\n{np.array2string(transformation, precision=2)}\ndet(T) = {np.linalg.det(transformation):.2f}",
            transform=ax.transAxes,
            va="top",
            family="monospace",
            bbox={"facecolor": "white", "alpha": 0.85},
        )
        ax.legend(loc="lower right")
        figure.canvas.draw_idle()

    for slider in sliders.values():
        slider.on_changed(update)
    update()
    plt.show()


DEMOS: dict[str, tuple[str, Callable[[], None]]] = {
    "1": ("Vector addition", show_vector_addition),
    "2": ("Dot product and projection", show_dot_product_and_projection),
    "3": ("Norms and distances", show_norms_and_distances),
    "4": ("Matrix transformations", show_matrix_transformation),
}


def run_menu() -> None:
    """Display a terminal menu and open the selected interactive plot."""
    while True:
        print("\nLinear Algebra Visualizer")
        for key, (name, _) in DEMOS.items():
            print(f"  {key}. {name}")
        print("  0. Exit")
        selection = input("Choose an operation: ").strip()
        if selection == "0":
            return
        demo = DEMOS.get(selection)
        if demo is None:
            print("Invalid option. Choose a number from 0 to 4.")
            continue
        demo[1]()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo",
        choices=DEMOS,
        help="Open one demo directly (1-4) instead of showing the terminal menu.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.demo:
        DEMOS[args.demo][1]()
    else:
        run_menu()


if __name__ == "__main__":
    main()

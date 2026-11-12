"""Four bounded Matplotlib/Pillow teaching animations, with PNG fallbacks."""

from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from uplift_visual_core import policy_selection


EFFECT_COLORS = ["#b23a48", "#f5f2e8", "#167d9a"]
EFFECT_MAP = LinearSegmentedColormap.from_list("uplift", EFFECT_COLORS)


def export_animation(fig, update, path, frames=24, fps=8):
    """Keep the other modules usable if Pillow fails; inspect fallback in report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, repeat=True)
        fig.canvas.draw()
        animation.save(path, writer=PillowWriter(fps=fps), dpi=85)
        return dict(file=path.name, kind="gif", frames=frames)
    except (OSError, RuntimeError, ValueError, MemoryError) as error:
        update(frames - 1)
        fallback = path.with_suffix(".png")
        fig.savefig(fallback, dpi=100, bbox_inches="tight")
        warnings.warn(f"GIF {path.name} failed ({error}); wrote {fallback.name}", RuntimeWarning)
        return dict(file=fallback.name, kind="png_fallback", reason=str(error))
    finally:
        plt.close(fig)


def potential_outcomes(data, path, frames):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    d = data["individual"]

    def update(frame):
        ax.clear()
        stage = min(4, frame * 5 // frames)
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        ax.text(.5, .94, "One decision, one observed outcome", ha="center", fontsize=17, weight="bold")
        ax.text(.5, .79, f"Pre-treatment X = ({d['x'][0]:+.2f}, {d['x'][1]:+.2f})", ha="center")
        for arm, xpos in ((0, .25), (1, .75)):
            assigned = arm == d["t"]
            color = "#167d9a" if stage >= 3 and assigned else "#687787"
            ax.annotate("", (xpos, .57), (.5, .73), arrowprops=dict(arrowstyle="->", color=color, lw=2))
            text = f"{'Control' if arm == 0 else 'Treatment'}: T={arm}\n"
            text += f"Known synthetic probability: {d['p' + str(arm)]:.3f}\n"
            if stage < 2:
                text += "Y(t): not yet revealed"
            elif stage >= 3 and assigned:
                text += f"Observed Y = Y({arm}) = {d['y' + str(arm)]}"
            else:
                text += "Y(t): HIDDEN counterfactual" if stage >= 3 else "Binary outcome still hidden"
            ax.text(xpos, .40, text, ha="center", va="center", linespacing=1.8,
                    bbox=dict(boxstyle="round,pad=.8", facecolor="#eff4f7", edgecolor=color))
        message = ["Define the person before treatment", "Two possible interventions",
                   "Randomize assignment, e(X)=0.5", f"Assigned T={d['t']}; reveal only that outcome",
                   "The other outcome stays missing. CATE is a probability contrast."][stage]
        ax.text(.5, .10, message, ha="center", fontsize=11, weight="bold")
        ax.text(.5, .02, "Synthetic demonstration; probabilities are not realized binary outcomes.", ha="center", fontsize=9)

    return export_animation(fig, update, path, frames)


def cate_rotation(data, path, frames):
    fig = plt.figure(figsize=(7.2, 5.5))
    ax = fig.add_subplot(projection="3d")
    a, b = np.meshgrid(data["axis"], data["axis"])
    ax.plot_surface(a, b, data["grid"]["tau"], cmap=EFFECT_MAP, vmin=-1, vmax=1,
                    linewidth=0, antialiased=True, alpha=.95)
    ax.set(xlabel="Engagement (standard units)", ylabel="Price sensitivity (standard units)",
           zlabel="CATE (probability points)", zlim=(-1, 1))
    ax.set_title("Known synthetic CATE = mu1 - mu0\nCamera rotation; no time evolution")
    fig.tight_layout()

    def update(frame):
        ax.view_init(elev=28, azim=25 + frame * 360 / frames)

    return export_animation(fig, update, path, frames)


def model_contrast(data, path, frames):
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    grid = data["grid"]

    def update(frame):
        alpha = frame / (frames - 1)
        for ax, surface, label in zip(axes,
                (grid["t0"], grid["t1"], grid["t1"] - alpha * grid["t0"]),
                ("Predicted control mu0", "Predicted treatment mu1", f"mu1 - {alpha:.2f} x mu0")):
            ax.clear()
            ax.imshow(surface, origin="lower", extent=(-2.5, 2.5, -2.5, 2.5),
                      cmap=EFFECT_MAP, vmin=-1, vmax=1)
            ax.set(title=label, xlabel="Engagement", ylabel="Price sensitivity")
        fig.suptitle("T-Learner: subtract predictions, not observed individual labels\n"
                     "Algebraic animation; final frame is estimated CATE", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .84))

    return export_animation(fig, update, path, frames)


def policy_threshold(data, path, frames):
    fig, ax = plt.subplots(figsize=(7, 5.2))
    scores = np.asarray(data["test"]["score"])
    grid = np.asarray(data["grid"]["t"])

    def update(frame):
        cost = 30 * frame / (frames - 1)
        selected = policy_selection(scores, 100, cost, 30)
        region = np.zeros_like(grid, dtype=float)
        if selected.any():
            region = ((grid >= scores[selected].min()) & (100 * grid > cost)).astype(float)
        ax.clear()
        ax.imshow(region, origin="lower", extent=(-2.5, 2.5, -2.5, 2.5),
                  cmap=LinearSegmentedColormap.from_list("policy", ["#edf1f4", "#167d9a"]), vmin=0, vmax=1)
        ax.set(xlabel="Engagement (standard units)", ylabel="Price sensitivity (standard units)",
               title=f"Treat if 100 x estimated CATE > {cost:.1f}; capacity 30%\n"
                     f"Selected holdout: {selected.sum()} / {len(scores)}")
        fig.text(.5, .02, "Grid boundary uses cohort score cutoff; equal-score ties remain cohort-specific.",
                 ha="center", fontsize=8)

    return export_animation(fig, update, path, frames)


def generate_animations(data, directory, frames=24):
    if not isinstance(frames, int) or not 4 <= frames <= 60:
        raise ValueError("frames must be an integer in [4, 60]")
    directory = Path(directory)
    return {
        name: function(data, directory / f"{name}.gif", frames)
        for name, function in (
            ("potential_outcomes", potential_outcomes), ("cate_surface_rotation", cate_rotation),
            ("model_comparison", model_contrast), ("policy_threshold", policy_threshold),
        )
    }

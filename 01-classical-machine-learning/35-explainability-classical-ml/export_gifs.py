"""Generate three CPU-only educational GIFs, PNGs and an ignored experiment record."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import shap
import sklearn
from PIL import Image

import visual_core as core
import visualizations as viz


def save_animation(fig, update, frames, path, fps):
    limits = [(ax.get_xlim(), ax.get_ylim()) for ax in fig.axes]

    def checked_update(frame):
        artists = update(frame)
        for ax, (xlim, ylim) in zip(fig.axes, limits):
            if ax.get_xlim() != xlim or ax.get_ylim() != ylim:
                raise RuntimeError("Animation axis limits changed across frames.")
        return artists

    animation = FuncAnimation(fig, checked_update, frames=frames, interval=1000/fps, blit=False, repeat=True)
    animation.save(str(path), writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
    with Image.open(path) as image:
        if not image.is_animated or image.n_frames < 2:
            raise RuntimeError(f"Export did not produce an animation: {path}")
        dimensions = image.size
        for index in range(image.n_frames):
            image.seek(index)
            if image.size != dimensions:
                raise RuntimeError("GIF dimensions changed across frames.")
        return {"frames": image.n_frames, "pixels": dimensions, "bytes": path.stat().st_size, "stable_axes": True}


def permutation_gif(lab, output, fps):
    path = core.permutation_path(lab, (0,), frames=17)
    X, y = lab["X_test"], lab["y_test"]
    baseline = core.metric_score(y, path["probabilities"][0], "ROC-AUC")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
    points = []
    for panel, ax in enumerate(axes):
        artists = []
        for label in (0, 1):
            mask = y == label
            artists.append(ax.scatter(X[mask, 0], path["probabilities"][0][mask],
                color=viz.CLASS_COLORS[label], marker=("o", "x")[label],
                s=20, alpha=0.7, label=f"Observed y={label}"))
        points.append(artists)
        ax.set(xlabel="signal value assigned to this row", ylabel="Model P(y=1)",
               xlim=(X[:, 0].min()-0.2, X[:, 0].max()+0.2), ylim=(-0.04, 1.04))
        ax.legend(loc="upper left", fontsize=9)
    axes[0].set_title(f"Original rows: ROC-AUC {baseline:.4f}")
    axes[1].set_title("Move values; targets stay fixed")
    header = fig.suptitle("Permutation importance: disrupt one input, freeze the forest", fontsize=14)
    footer = fig.supxlabel("Blue circles: class 0; orange crosses: class 1. Axes stay fixed.", fontsize=10)
    positions = list(range(len(path["states"]))) + [len(path["states"])-1]*6

    def update(frame):
        index = positions[frame]
        rows, probabilities = path["states"][index], path["probabilities"][index]
        for label in (0, 1):
            mask = y == label
            points[1][label].set_offsets(np.column_stack([rows[mask, 0], probabilities[mask]]))
        score = core.metric_score(y, probabilities, "ROC-AUC")
        axes[1].set_title(f"ROC-AUC {score:.4f}; baseline - current = {baseline-score:.4f}")
        phase = "Actual completed row shuffle" if index == len(path["states"])-1 else "Interpolated transport: NOT a permutation sample"
        footer.set_text(phase + "\nTargets remain on their original rows; only signal changes.")
        return header, footer

    metadata = save_animation(fig, update, len(positions), output/"permutation_importance.gif", fps)
    return metadata, {
        "baseline_auc": baseline,
        "shuffled_auc": core.metric_score(y, path["probabilities"][-1], "ROC-AUC"),
        "decrease": baseline-core.metric_score(y, path["probabilities"][-1], "ROC-AUC"),
    }


def ice_gif(lab, output, fps):
    response = core.response_curves(lab, rows=30, grid_size=25)
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    artists = [ax.plot(response["grid"], row, color=viz.COLORS[0], alpha=0.3,
                       linewidth=1, visible=False)[0] for row in response["ice"]]
    mean, = ax.plot(response["grid"], response["pdp"], color=viz.PDP_COLOR, linewidth=3,
                    label="PDP: mean of ALL 30 reference ICE", visible=False)
    ax.plot([], [], color=viz.COLORS[0], alpha=0.5, label="Individual response; other features fixed")
    ax.set(xlabel="Replacement signal value", ylabel="Model P(y=1)",
           xlim=(response["grid"].min(), response["grid"].max()), ylim=(0, 1))
    ax.legend(loc="upper left", fontsize=9)
    header = ax.set_title("Construct ICE, then take the matching mean")
    footer = fig.supxlabel("Probability-scale heterogeneity is not proof of a logit-scale interaction.", fontsize=10)
    # Introduce curves one by one, then trace their full-reference average.
    frames = 30 + 5 + 6

    def update(frame):
        count = min(frame+1, 30)
        for index, artist in enumerate(artists):
            artist.set_visible(index < count)
        mean.set_visible(frame >= 30)
        if frame >= 30:
            grid_count = min(len(response["grid"]), max(1, (frame-29)*5))
            mean.set_data(response["grid"][:grid_count], response["pdp"][:grid_count])
        header.set_text(f"{count}/30 individual curves" if frame < 30 else
                        "PDP emerges: average the SAME 30 curves at every grid value")
        return [*artists, mean, header, footer]

    metadata = save_animation(fig, update, frames, output/"pdp_ice_construction.gif", fps)
    return metadata, response


def shapley_gif(output, fps):
    game = core.shapley_product(2, 3)
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.set(xlim=(-0.6, 1.6), ylim=(-0.15, 1.1))
    ax.axis("off")
    circles = []
    for feature in (0, 1):
        circle = plt.Circle((feature, 0.65), 0.18, color=viz.COLORS[feature], alpha=0.25)
        ax.add_patch(circle)
        circles.append(circle)
        ax.text(feature, 0.65, f"x{feature+1} = {(2,3)[feature]}", ha="center", va="center",
                color="white", weight="bold", fontsize=15)
    state = ax.text(0.5, 0.27, "", ha="center", fontsize=14)
    allocation = ax.text(0.5, 0.03, "", ha="center", fontsize=13)
    title = ax.set_title("f(x1,x2)=x1*x2; observation (2,3); missing features use zero", fontsize=13)
    fig.supxlabel("A single feature ordering is NOT a Shapley attribution. Average both orders.", fontsize=11)
    stages = [(order, step) for order, path in enumerate(game["paths"]) for step in path["steps"]]
    frame_stages = [stage for stage in range(7) for _ in range(5)]

    def update(frame):
        stage = frame_stages[frame]
        if stage < 6:
            order, step = stages[stage]
            for feature, circle in enumerate(circles):
                circle.set_alpha(1 if feature in step["members"] else 0.25)
            member_text = ", ".join(f"x{j+1}" for j in step["members"]) or "empty"
            state.set_text(f"Order {order+1}: coalition {member_text}\nv(S)={step['value']:.0f}; "
                           f"joining contribution={step['marginal']:.0f}")
            allocation.set_text("Order 1 gives (0,6); order 2 gives (6,0).")
        else:
            for circle in circles:
                circle.set_alpha(1)
            state.set_text("Average both orders: Shapley values = (3, 3)")
            allocation.set_text("Verified additive reconstruction: 0 + 3 + 3 = f(2,3) = 6")
        return [*circles, state, allocation, title]

    metadata = save_animation(fig, update, len(frame_stages), output/"shap_contributions.gif", fps)
    return metadata, {"baseline": game["baseline"], "contributions": game["values"].tolist(),
                      "prediction": game["prediction"]}


def generate(output, config=core.DEFAULT_CONFIG, fps=6):
    core.integer(fps, 2, 12, "Frames per second")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lab = core.build_lab(config)
    permutation_meta, permutation_result = permutation_gif(lab, output, fps)
    ice_meta, ice_response = ice_gif(lab, output, fps)
    shapley_meta, shapley_result = shapley_gif(output, fps)
    importance = core.importance_results(lab, "ROC-AUC", repeats=8)
    explanation = core.forest_shap(lab, 50, 35, explained_count=min(80, len(lab["X_test"])))
    for name, fig in [
        ("importance_comparison.png", viz.matplotlib_importance(importance)),
        ("pdp_ice.png", viz.matplotlib_ice(ice_response)),
        ("shap_waterfall.png", viz.matplotlib_waterfall(explanation)),
    ]:
        fig.savefig(output/name, dpi=200)
        plt.close(fig)
    response_surface = core.surfaces(lab)
    # Self-contained offline HTML uses embedded Plotly, not a CDN.
    viz.surface_plot(response_surface, "pdp").write_html(
        str(output/"partial_dependence_3d.html"), include_plotlyjs=True
    )
    report = {
        "hypothesis": "Visual transport, matching ICE averaging and coalition-order averaging should "
                      "show different explanation definitions while satisfying their numerical identities.",
        "configuration": {
            "synthetic": True, "seed": core.SEED,
            "config_order": ["n", "rho", "interaction", "trees", "depth", "leaf"],
            "config": list(config), "split": "stratified 75/25",
            "gif_fps": fps, "probability_output": "P(y=1)",
            "permutation": "signal; 17 transport frames; only endpoint is a true shuffle; ROC-AUC",
            "importance_repeats": 8, "ice_rows": 30, "ice_grid_size": 25,
            "shap_background_rows": 50, "shap_background_seed": 35,
            "shap_explained_rows": len(explanation["rows"]),
            "fixed_model": lab["model"].get_params(),
        },
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "sklearn": sklearn.__version__, "shap": shap.__version__,
                     "matplotlib": matplotlib.__version__},
        "result": {
            "permutation": permutation_result,
            "pdp_equals_mean_ice_max_error": float(np.max(np.abs(ice_response["pdp"]-ice_response["ice"].mean(axis=0)))),
            "shapley_product": shapley_result,
            "probability_shap_max_error": explanation["error"],
            "probability_shap_baseline": float(explanation["baseline"][0]),
            "mdi": importance["mdi"].tolist(),
            "library_permutation_mean": importance["library"].importances_mean.tolist(),
            "scratch_group_permutation_mean": importance["scratch"]["importances_mean"].tolist(),
            "animations": {
                "permutation_importance.gif": permutation_meta,
                "pdp_ice_construction.gif": ice_meta,
                "shap_contributions.gif": shapley_meta,
            },
        },
        "interpretation_candidate": "The checked identities validate the stated arithmetic; apparent "
                                    "heterogeneity or reliance should be interpreted with support and reference assumptions.",
        "review_status": "Interpretation candidate pending author review.",
        "limitations": [
            "One synthetic configuration and seed; not a performance benchmark or causal estimate.",
            "Transport interpolation is illustrative, not a sequence of actual row permutations.",
            "The PDP averages exactly the displayed 30 ICE curves; interactive PDP uses up to 100.",
            "Probability geometry does not establish an interaction on another output scale.",
            "Fixed animation axes and dimensions are checked; editorial readability needs visual inspection.",
            "Interventional SHAP uses one training background; no stability claim is established.",
        ],
        "output_policy": "All generated files are regenerable, ignored artifacts; no public previews selected.",
    }
    (output/"visual_lab_results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"Created 3 GIFs, 3 PNGs, 1 self-contained 3D HTML, and 1 experiment record in {output}")
    print(f"Forest SHAP reconstruction error: {explanation['error']:.3e}")
    for name, meta in report["result"]["animations"].items():
        print(f"{name}: {meta['frames']} frames, {meta['bytes']/1024:.0f} KiB")
    print(report["review_status"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent/"outputs")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--samples", type=int, default=900)
    parser.add_argument("--rho", type=float, default=0.95)
    parser.add_argument("--interaction", type=float, default=1.6)
    args = parser.parse_args()
    config = (args.samples, args.rho, args.interaction, 60, 6, 5)
    core.validate_config(config)
    generate(args.output_dir, config, args.fps)


if __name__ == "__main__":
    main()

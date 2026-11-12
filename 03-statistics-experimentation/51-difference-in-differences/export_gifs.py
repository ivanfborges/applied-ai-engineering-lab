"""Bounded, headless Matplotlib/Pillow exports; kept separate from the dashboard."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from charts import COLORS
from estimation import oracle_decomposition, trajectories
from simulation import simulate

NAMES = ["did_counterfactual_animation.gif", "parallel_trends_violation.gif"]


def canvas(title: str, bounds: tuple[float, float], width: int):
    fig, ax = plt.subplots(figsize=(width/100, width/160), dpi=100)
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.26, top=0.82)
    fig.suptitle(title, x=0.09, ha="left", fontsize=17, fontweight="bold", color="#172b4d")
    ax.set(xlim=(-6.3, 5.6), ylim=bounds, xlabel="Period relative to rollout", ylabel="Synthetic productivity points")
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(alpha=0.15)
    return fig, ax


def export_counterfactual(path: Path, frames: int, fps: int, width: int):
    panel = simulate()
    data = trajectories(panel)
    levels = data[["control", "treated", "estimated_cf", "true_y0"]]
    bounds = (float(levels.min().min()-3), float(levels.max().max()+3))
    fig, ax = canvas("Difference-in-Differences · build the missing counterfactual", bounds, width)
    lines = {}
    for key, label, style in [
        ("control", "Control observed", "-"), ("treated", "Treated observed", "-"),
        ("cf", "Estimated untreated counterfactual", "--"),
    ]:
        lines[key], = ax.plot([], [], style, color=COLORS[key], linewidth=2.8, marker="o", markersize=4, label=label)
    effect_line, = ax.plot([], [], color=COLORS["effect"], linewidth=5, label="Estimated gap")
    boundary = ax.axvline(-0.5, color=COLORS["cf"], linestyle=":", alpha=0)
    ax.legend(loc="upper left", fontsize=9)
    status = fig.text(0.09, 0.90, "", fontsize=11, color="#172b4d")
    footer = fig.text(0.09, 0.09, "", fontsize=10)
    steps = ["1 / 6  Observe both groups before treatment",
             "2 / 6  The assistant rolls out at period 0",
             "3 / 6  Follow the control group's post-treatment change",
             "4 / 6  Extend the average pre-period group gap along control",
             "5 / 6  Reveal the treated group's observed post outcomes",
             "6 / 6  Compare observed treatment with the estimated counterfactual"]
    times = data.index.to_numpy()

    def draw(frame):
        position = frame / max(frames-1, 1) * 6
        step = min(int(position), 5)
        fraction = min(position-step, 1) if step < 5 else 1
        control_mask = times < 0
        treated_mask = times < 0
        if step == 2:
            control_mask = times <= int(fraction*6)-1
        elif step > 2:
            control_mask = np.ones(len(times), dtype=bool)
        if step == 4:
            treated_mask = times <= int(fraction*6)-1
        elif step > 4:
            treated_mask = np.ones(len(times), dtype=bool)
        lines["control"].set_data(times[control_mask], data.control.to_numpy()[control_mask])
        lines["treated"].set_data(times[treated_mask], data.treated.to_numpy()[treated_mask])
        lines["cf"].set_data(times[times >= 0], data.loc[0:, "estimated_cf"] if step >= 3 else [np.nan]*6)
        boundary.set_alpha(1 if step >= 1 else 0)
        effect_line.set_data([5, 5], [data.loc[5, "estimated_cf"], data.loc[5, "treated"]] if step >= 5 else [np.nan, np.nan])
        status.set_text(steps[step])
        estimate = oracle_decomposition(panel)["estimate"]
        footer.set_text(f"Seed 51 · 120 teams · true ATT +4 points · pooled DiD {estimate:+.2f}\n"
                        "Dashed path is assumption-based, not observed truth. Green segment is one period's gap.")
        return *lines.values(), effect_line, boundary, status, footer

    try:
        animation = FuncAnimation(fig, draw, frames=frames, interval=1000/fps, blit=False)
        animation.save(str(path), writer=PillowWriter(fps=fps))
    finally:
        plt.close(fig)
    return oracle_decomposition(panel)


def export_violation(path: Path, frames: int, fps: int, width: int):
    # Zero noise isolates identification bias from sampling error throughout this GIF.
    slopes = np.linspace(0, 1.5, frames)
    panels = [simulate(noise=0, slope=float(slope)) for slope in slopes]
    data = [trajectories(panel) for panel in panels]
    levels = [d[["control", "treated", "estimated_cf", "true_y0"]] for d in data]
    bounds = (min(float(d.min().min()) for d in levels)-3, max(float(d.max().max()) for d in levels)+3)
    fig, ax = canvas("Parallel trends · the same ATT, a changing DiD estimate", bounds, width)
    lines = {}
    for key, column, label, style in [
        ("control", "control", "Control observed", "-"),
        ("treated", "treated", "Treated observed", "-"),
        ("cf", "estimated_cf", "Estimated counterfactual", "--"),
        ("truth", "true_y0", "True untreated Y(0) — synthetic only", ":"),
    ]:
        lines[column], = ax.plot([], [], style, color=COLORS[key], linewidth=2.5, label=label)
    ax.axvline(-0.5, color=COLORS["cf"], linestyle=":")
    bias_line, = ax.plot([], [], color=COLORS["bias"], linewidth=5, label="Counterfactual error at period 5")
    ax.legend(loc="upper left", fontsize=9)
    status = fig.text(0.09, 0.90, "", fontsize=11)
    footer = fig.text(0.09, 0.09, "", fontsize=11, color=COLORS["bias"])

    def draw(frame):
        current = data[frame]
        for column, line in lines.items():
            subset = current.loc[0:] if column == "estimated_cf" else current
            line.set_data(subset.index, subset[column])
        bias_line.set_data([5, 5], [current.loc[5, "estimated_cf"], current.loc[5, "true_y0"]])
        values = oracle_decomposition(panels[frame])
        status.set_text(f"Differential untreated slope {slopes[frame]:+.2f} points / period · noise = 0")
        footer.set_text(f"True ATT {values['truth']:+.2f}     Estimated DiD {values['estimate']:+.2f}     "
                        f"Pooled error {values['error']:+.2f}\n"
                        "Pooled error = 6 × slope; red segment is period-specific. Purple truth is simulation-only.")
        return *lines.values(), bias_line, status, footer

    try:
        animation = FuncAnimation(fig, draw, frames=frames, interval=1000/fps, blit=False)
        animation.save(str(path), writer=PillowWriter(fps=fps))
    finally:
        plt.close(fig)
    return {"start": oracle_decomposition(panels[0]), "end": oracle_decomposition(panels[-1])}


def export_all(out: Path, frames: int = 48, fps: int = 6, width: int = 960, overwrite: bool = False) -> dict:
    if not 12 <= frames <= 120 or not 2 <= fps <= 15 or not 640 <= width <= 1280:
        raise ValueError("Require 12..120 frames, 2..15 fps and 640..1280 pixel width.")
    out = Path(out)
    targets = [out / name for name in NAMES] + [out / "visual_run.json"]
    if not overwrite and any(path.exists() for path in targets):
        raise FileExistsError("Exports already exist; use a new directory or --overwrite deliberately.")
    out.mkdir(parents=True, exist_ok=True)
    results = {
        "hypotheses": ["Control changes construct a counterfactual, not an observed untreated outcome.",
                       "A differential untreated slope adds six times its value to pooled DiD."],
        "configuration": {"seed": 51, "teams": 120, "periods": [-6, 5], "effect": 4,
                          "frames": frames, "fps": fps, "width": width,
                          "counterfactual_noise_sd": 1.5, "violation_noise_sd": 0,
                          "violation_slopes": [0, 1.5]},
        "versions": {"python": platform.python_version(), **{p: version(p) for p in
                     ["numpy", "pandas", "matplotlib", "Pillow", "plotly", "statsmodels", "streamlit"]}},
        "counterfactual": export_counterfactual(targets[0], frames, fps, width),
        "violation": export_violation(targets[1], frames, fps, width),
        "interpretation_status": "author review required",
        "limitation": "Synthetic constructed panels, not empirical AI productivity or repeated-sampling evidence.",
    }
    targets[2].write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "assets")
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    export_all(arguments.out, arguments.frames, arguments.fps, arguments.width, arguments.overwrite)
    print(f"Generated {', '.join(NAMES)} and visual_run.json in {arguments.out}")

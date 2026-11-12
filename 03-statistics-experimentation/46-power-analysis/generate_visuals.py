"""Bounded offline export: three GIFs, one PNG, interactive HTML and run records."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import PIL
import scipy
from scipy.stats import norm
import statsmodels
import plotly

from power_math import cluster_information, required_n, sampling_model, theoretical_power, validate_design
from simulation import simulate_experiments
from visual_charts import BLUE, TEAL, AMBER, ROSE, INK, sensitivity_data, surface_figure

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
    "axes.spines.right": False, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": "#BECBC6",
    "figure.facecolor": "#FAFAF7", "axes.facecolor": "#FAFAF7",
})


def frame_image(fig):
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert("RGB")


def save_animation(images, path):
    images[0].save(path, save_all=True, append_images=images[1:], duration=125,
                   loop=0, optimize=True, disposal=2)


def distribution_animation(path, baseline, effect, alpha, frames=36):
    ns = np.geomspace(1000, 60000, frames).astype(int)
    first = sampling_model(baseline, effect, int(ns[0]), alpha)
    last = sampling_model(baseline, effect, int(ns[-1]), alpha)
    x = np.linspace(min(-4.3 * first["se_null"], effect - 4.3 * first["se_alt"]),
                    max(4.3 * first["se_null"], effect + 4.3 * first["se_alt"]), 1800)
    ymax = max(norm.pdf(0, 0, last["se_null"]), norm.pdf(effect, effect, last["se_alt"])) / 100 * 1.18
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=100)
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.17, top=0.78)
    images = []
    for n in ns:
        ax.clear()
        model = sampling_model(baseline, effect, int(n), alpha)
        c = model["critical"]
        null = norm.pdf(x, 0, model["se_null"]) / 100
        alt = norm.pdf(x, effect, model["se_alt"]) / 100
        ax.plot(x * 100, null, color=BLUE, lw=2.5, label="H0: no effect")
        ax.plot(x * 100, alt, color=TEAL, lw=2.5, label="H1: assumed effect")
        reject = np.abs(x) >= c
        ax.fill_between(x * 100, 0, null, where=reject, color=ROSE, alpha=0.4, label="Type I error under H0")
        ax.fill_between(x * 100, 0, alt, where=~reject, color=AMBER, alpha=0.45, label="Type II error under H1")
        ax.fill_between(x * 100, 0, alt, where=reject, color=TEAL, alpha=0.35, label="Power under H1")
        for boundary in (-c, c):
            ax.axvline(boundary * 100, color=ROSE, ls="--", lw=1)
        ax.axvline(effect * 100, color=TEAL, ls=":", lw=1)
        ax.set(xlim=(x[0] * 100, x[-1] * 100), ylim=(0, ymax),
               xlabel="Estimated treatment − control difference (percentage points)",
               ylabel="Density per percentage point")
        ax.grid(alpha=0.15)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        ax.set_title(f"n = {n:,} per arm     α = {model['alpha']:.1%}     β = {model['beta']:.1%}     power = {model['power']:.1%}",
                     fontsize=12, loc="left", pad=15)
        if not images:
            fig.suptitle("More independent data separates signal from uncertainty", x=0.10, ha="left", fontsize=18, y=0.96)
            fig.text(0.10, 0.055, "Analytical normal approximation · fixed axes · probabilities use full tails · hypothetical rates", fontsize=9)
        images.append(frame_image(fig))
    plt.close(fig)
    save_animation(images, path)


def power_animation(path, baseline, effect, alpha, target, frames=36):
    needed = required_n(baseline, effect, alpha, target)
    maximum = min(max(needed * 2.4, 1000), 2_000_000)
    ns = np.linspace(100, maximum, 180).astype(int)
    values = np.array([theoretical_power(baseline, effect, int(n), alpha) for n in ns])
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=100)
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.17, top=0.79)
    ax.plot(ns, values, lw=3, color=TEAL)
    ax.fill_between(ns, 0, values, color=TEAL, alpha=0.08)
    for reference in sorted(set((0.8, 0.9, target))):
        ax.axhline(reference, color=AMBER, ls=":", lw=1)
        ax.text(maximum * 0.98, reference - 0.035, f"{reference:.0%}", ha="right", fontsize=9)
    ax.axvline(needed, color=BLUE, ls="--", label=f"Required n = {needed:,}/arm")
    ax.set(xlim=(0, maximum), ylim=(0, 1.04), xlabel="Independent users per arm", ylabel="Statistical power")
    ax.grid(alpha=0.15)
    ax.legend(loc="lower right", frameon=False)
    point, = ax.plot([], [], "o", color=ROSE, ms=10)
    label = ax.text(0.02, 0.91, "", transform=ax.transAxes, fontsize=14, color=INK)
    fig.suptitle("Power grows nonlinearly with sample size", x=0.10, ha="left", fontsize=19, y=0.96)
    fig.text(0.10, 0.055, f"Hypothetical rates: {baseline:.1%} → {baseline + effect:.1%} · two-sided α={alpha:g} · analytical approximation", fontsize=9)
    images = []
    for n in np.linspace(100, maximum, frames).astype(int):
        value = theoretical_power(baseline, effect, int(n), alpha)
        point.set_data([n], [value])
        label.set_text(f"{n:,} / arm   →   {value:.1%} power")
        images.append(frame_image(fig))
    plt.close(fig)
    save_animation(images, path)


def cluster_animation(path, frames=36):
    users, conversations = 1000, 10
    rhos = np.linspace(0, 1, 200)
    raw = users * conversations
    ess = raw / (1 + (conversations - 1) * rhos)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 5.8), dpi=100, gridspec_kw=dict(width_ratios=[1, 1.4]))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.18, top=0.78, wspace=0.35)
    bars = left.bar(["Raw\nconversations", "Equivalent\nindependent"], [raw, raw], color=[BLUE, TEAL], width=0.55)
    left.set_ylim(0, raw * 1.08)
    left.set_ylabel("Information count")
    right.plot(rhos, ess, color=TEAL, lw=3)
    right.axhline(raw, color=BLUE, ls=":", label="Raw count stays fixed")
    point, = right.plot([], [], "o", color=AMBER, ms=10)
    right.set(xlim=(0, 1), ylim=(0, raw * 1.08), xlabel="ICC", ylabel="Approximate effective sample size")
    right.grid(alpha=0.15)
    label = fig.text(0.09, 0.83, "", fontsize=13)
    fig.suptitle("Conversation count is not independent information", x=0.09, ha="left", fontsize=19, y=0.96)
    fig.text(0.09, 0.06, "1,000 users × 10 conversations · equal-sized-cluster heuristic · not a cluster-aware power analysis", fontsize=9)
    images = []
    for rho in np.linspace(0, 1, frames):
        _, de, effective = cluster_information(users, conversations, float(rho))
        bars[1].set_height(effective)
        point.set_data([rho], [effective])
        label.set_text(f"ICC = {rho:.2f}     design effect = {de:.2f}     effective N ≈ {effective:,.0f}")
        images.append(frame_image(fig))
    plt.close(fig)
    save_animation(images, path)


def effect_png(path, baseline, alpha, target):
    effects, ns = sensitivity_data(baseline, alpha, target)
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=170)
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.18, top=0.78)
    ax.plot(effects * 100, ns, color=TEAL, lw=3, label="Bernoulli normal-approximation sizing")
    ax.plot(effects * 100, ns[-1] * (effects[-1] / effects) ** 2,
            color=AMBER, ls="--", lw=2, label="Fixed-variance 1/Δ² reference (anchored at right)")
    ax.set(yscale="log", xlabel="Positive absolute improvement (percentage points)",
           ylabel="Required independent users per arm (log scale)")
    ax.grid(alpha=0.15, which="both")
    ax.legend(frameon=False, fontsize=10)
    fig.suptitle("Smaller effects make experiments expensive", x=0.12, ha="left", fontsize=20, y=0.96)
    fig.text(0.12, 0.85, f"Baseline {baseline:.1%}   |   Two-sided α={alpha:g}   |   Target power {target:.0%}", fontsize=12)
    fig.text(0.12, 0.055, "Calculated planning quantities · Bernoulli variance changes with treatment rate · hypothetical inputs", fontsize=9)
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_surface_html(path, baseline=0.10, alpha=0.05, target=0.80):
    """Embed Plotly and an empty data favicon for standalone, request-free rendering."""
    html = surface_figure(baseline, alpha, target).to_html(
        include_plotlyjs=True, include_mathjax=False, full_html=True,
        config=dict(displaylogo=False, scrollZoom=True),
    )
    html = html.replace("<head>", '<head><link rel="icon" href="data:,">', 1)
    Path(path).write_text(html, encoding="utf-8")


def generate(output_dir, baseline=0.10, effect=0.01, alpha=0.05, target=0.80,
             seed=46, frames=36, repetitions=20000):
    validate_design(baseline, effect, alpha)
    if not isinstance(frames, int) or isinstance(frames, bool) or not 3 <= frames <= 80:
        raise ValueError("Frames must be an integer between 3 and 80.")
    n = required_n(baseline, effect, alpha, target)
    # Validate all Monte Carlo inputs before exporting any files.
    runs = {mode: simulate_experiments(baseline, effect, n, alpha, repetitions, seed, mode)
            for mode in ("null", "alternative")}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    distribution_animation(output_dir / "sampling_distributions.gif", baseline, effect, alpha, frames)
    power_animation(output_dir / "power_growth.gif", baseline, effect, alpha, target, frames)
    cluster_animation(output_dir / "effective_sample_size.gif", frames)
    effect_png(output_dir / "effect_size_sample_size.png", baseline, alpha, target)
    write_surface_html(output_dir / "power_surface_3d.html", baseline, alpha, target)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    record = dict(
        timestamp_utc=timestamp,
        hypothesis="At the declared alternative, pooled Z rejection frequency should approximate planned power; under H0 it should approximate alpha.",
        config=dict(baseline=baseline, effect=effect, alpha=alpha, target=target,
                    n_per_arm=n, seed=seed, frames=frames, repetitions=repetitions),
        versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                      statsmodels=statsmodels.__version__, plotly=plotly.__version__,
                      matplotlib=matplotlib.__version__, pillow=PIL.__version__),
        results={mode: {key: run[key] for key in ("rejections", "rejection_rate", "ci_low", "ci_high", "analytical", "degenerate_draws")}
                 for mode, run in runs.items()},
        interpretation_candidate="Compare estimates with Monte Carlo intervals, allowing finite-sample approximation error. Author review required.",
        limitation="Synthetic independent Bernoulli units; no real product effect or production decision. Wilson intervals capture simulation error only.",
    )
    (output_dir / f"visual_run_{timestamp}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--baseline", type=float, default=0.10)
    parser.add_argument("--effect", type=float, default=0.01, help="Absolute probability difference, not percentage points")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--target", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--frames", type=int, default=36)
    parser.add_argument("--repetitions", type=int, default=20000)
    args = parser.parse_args()
    record = generate(**vars(args))
    print(json.dumps(record, indent=2))
    print(f"Exports saved in {args.output_dir.resolve()}")

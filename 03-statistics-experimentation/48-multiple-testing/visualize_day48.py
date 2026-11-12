"""Export the offline Day 48 statistical visualization lab (no FFmpeg required)."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image
import PIL
import plotly
import plotly.graph_objects as go
import scipy
from scipy.stats import norm
import statsmodels

from day48_visual_core import (LabConfig, METHODS, adjusted_matrix, dependence_study,
                               independent_fwer, normal_families, reference_check,
                               sequential_study, tradeoff_study)
from example import estimate


COLORS = {"uncorrected": "#64748b", "bonferroni": "#d97706", "holm": "#0f766e",
          "bh": "#7c3aed", "by": "#a16207"}
LABELS = {"uncorrected": "Uncorrected", "bonferroni": "Bonferroni", "holm": "Holm",
          "bh": "BH", "by": "BY"}
RED, BLUE, INK = "#dc2626", "#2563eb", "#172554"
EXPERIMENTS = ("null", "fwer", "corrections", "tradeoffs", "peeking", "dependence")
HERE = Path(__file__).resolve().parent


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.labelsize": 10,
                         "figure.facecolor": "#f8fafc", "axes.facecolor": "white",
                         "text.color": INK, "axes.labelcolor": INK,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .16, "legend.frameon": False,
                         "savefig.facecolor": "#f8fafc"})


def canvas(title, footer, shape=(1, 2), size=(11, 6.2)):
    fig, axes = plt.subplots(*shape, figsize=size, squeeze=False)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=.965)
    fig.text(.04, .025, footer, fontsize=9, color="#475569", va="bottom")
    fig.subplots_adjust(left=.08, right=.96, top=.84, bottom=.18, wspace=.32, hspace=.5)
    return fig, axes


def save_png(fig, output, name):
    path = output / name
    fig.savefig(path, dpi=180)
    return path.name


def save_gif(fig, update, output, name, frames):
    animation = FuncAnimation(fig, update, frames=frames, interval=200, blit=False,
                              repeat=True, cache_frame_data=False)
    animation.save(output / name, writer=PillowWriter(fps=5), dpi=90)
    return name


def null_visual(config, output):
    p = 2 * norm.sf(np.abs(config.rng(1).normal(size=config.null_tests)))
    false = p <= config.alpha
    counts = np.cumsum(false)
    rates = counts / np.arange(1, p.size + 1)
    checkpoints = np.unique(np.linspace(min(20, p.size), p.size, config.frames, dtype=int))
    fig, axes = canvas("A · Chance alone creates significant results",
                       "SYNTHETIC · Independent N(0,1) statistics; two-sided p-values. Every null is true.\n"
                       "Every highlighted rejection is therefore a false discovery; this is one simulated sequence.")
    left, right = axes[0]
    left.set(xlabel="Independent null test number", ylabel="Two-sided p-value", xlim=(0, p.size + 1), ylim=(0, 1))
    left.axhspan(0, config.alpha, color=RED, alpha=.1)
    left.axhline(config.alpha, color=RED, ls="--", label=f"p = {config.alpha:g}")
    dots = left.scatter([], [], s=12, alpha=.7)
    left.legend(loc="upper right")
    right.set(xlabel="Number of null tests observed", ylabel="Cumulative false-positive proportion",
              xlim=(0, p.size), ylim=(0, min(1, max(.15, rates[checkpoints[0]-1:].max() * 1.15))))
    right.axhline(config.alpha, color=RED, ls="--", label=f"Nominal per-test level = {config.alpha:g}")
    line, = right.plot([], [], color=BLUE, lw=2)
    label = fig.text(.08, .88, "", fontsize=11)
    right.legend(loc="upper right")

    def update(frame):
        n = checkpoints[frame]
        dots.set_offsets(np.c_[np.arange(1, n + 1), p[:n]])
        dots.set_color(np.where(false[:n], RED, BLUE))
        line.set_data(np.arange(1, n + 1), rates[:n])
        label.set_text(f"n = {n:,}  |  False positives = {counts[n-1]}  |  Observed proportion = {rates[n-1]:.3f}")

    gif = save_gif(fig, update, output, "01_false_positives.gif", len(checkpoints))
    update(len(checkpoints)-1)
    png = save_png(fig, output, "01_false_positives.png")
    plt.close(fig)
    return [gif, png], {"false_positives": int(counts[-1]), "false_positive_rate": estimate(false),
                        "p_min": float(p.min()), "p_max": float(p.max())}


def surface_figure():
    m = np.arange(1, 101)
    alpha = np.linspace(.001, .1, 60)
    z = independent_fwer(m[None, :], alpha[:, None])
    fig = go.Figure([
        go.Surface(x=m.tolist(), y=alpha.tolist(), z=z.tolist(), colorscale="Viridis", cmin=0, cmax=1,
                   colorbar={"title": "FWER"}, hovertemplate="Tests m=%{x:.0f}<br>Per-test alpha=%{y:.3f}<br>FWER=%{z:.4f}<extra>Independent true nulls</extra>"),
        go.Scatter3d(x=m.tolist(), y=[.05]*100, z=independent_fwer(m, .05).tolist(),
                     mode="lines", line={"color": "#ff9855", "width": 7}, name="Selected alpha slice"),
        go.Scatter3d(x=[20], y=[.05], z=[float(independent_fwer(20, .05))], mode="markers",
                     marker={"size": 7, "color": "#ff3344"}, name="Selected decision",
                     hovertemplate="m=%{x}<br>alpha=%{y:.3f}<br>FWER=%{z:.4f}<extra></extra>"),
    ])
    fig.update_layout(title="Independent true nulls: FWER = 1 - (1 - alpha)^m",
                      scene={"xaxis_title": "Number of tests m", "yaxis_title": "Per-test alpha",
                             "zaxis_title": "Probability of at least one false positive",
                             "zaxis": {"range": [0, 1]}, "camera": {"eye": {"x": 1.6, "y": -1.8, "z": 1.1}}},
                      margin={"l": 0, "r": 0, "b": 0, "t": 60}, height=650,
                      legend={"orientation": "h", "y": 0}, template="plotly_white")
    return fig


def fwer_visual(config, output):
    fig, axes = canvas("B · Independent errors accumulate across a family",
                       "THEORY · All nulls true; independent tests with exact per-test rejection probability alpha.\n"
                       "The formula is not a universal result for correlated tests.", shape=(1, 1))
    ax = axes[0, 0]
    m = np.arange(1, 101)
    for alpha, color in ((.01, "#0f766e"), (.05, BLUE), (.1, "#d97706")):
        ax.plot(m, independent_fwer(m, alpha), label=f"Per-test alpha = {alpha:.2f}", color=color, lw=2.3)
    value = float(independent_fwer(20, .05))
    ax.scatter([20], [value], color=RED, s=65, zorder=5)
    ax.annotate(f"20 tests at 5%\nFWER = {value:.2%}", xy=(20, value), xytext=(42, .45),
                arrowprops={"arrowstyle": "->", "color": RED}, color=RED)
    ax.set(xlabel="Number of independent true null hypotheses", ylabel="Probability of at least one false positive", ylim=(0, 1), xlim=(1, 100))
    ax.legend(loc="lower right")
    png = save_png(fig, output, "02_fwer_growth.png")
    plt.close(fig)
    body = surface_figure().to_html(full_html=False, include_plotlyjs=True, div_id="fwer-plot",
                                  config={"responsive": True, "displaylogo": False, "scrollZoom": True})
    controls = '''<h1>Multiple testing: independent-null FWER</h1>
<p>Theory, not Monte Carlo. Every null is true and each marginal rejection probability equals alpha.
The formula is exact under independence; it does not describe arbitrary correlated tests.</p>
<div class="controls"><label>Number of tests m <input id="m" type="range" min="1" max="100" value="20"></label>
<label>Per-test alpha <input id="alpha" type="range" min="1" max="100" value="50"></label>
<button id="reset">Reset view and parameters</button><output id="readout" aria-live="polite"></output></div>
<p>Drag the 3D surface to rotate; use the mouse wheel to zoom; hover for values.
The orange line shows the selected alpha slice; the red marker shows the selected family.</p>'''
    script = '''<script>
const plot=document.getElementById('fwer-plot'),mInput=document.getElementById('m'),aInput=document.getElementById('alpha');
function updateSelection(){const m=Number(mInput.value),a=Number(aInput.value)/1000;
 const f=-Math.expm1(m*Math.log1p(-a)),xs=Array.from({length:100},(_,i)=>i+1);
 Plotly.restyle(plot,{x:[xs],y:[xs.map(()=>a)],z:[xs.map(n=>-Math.expm1(n*Math.log1p(-a)))]},[1]);
 Plotly.restyle(plot,{x:[[m]],y:[[a]],z:[[f]]},[2]);
 document.getElementById('readout').textContent=`m = ${m}; per-test alpha = ${a.toFixed(3)}; FWER = ${(100*f).toFixed(2)}%`;
 plot.dataset.m=m;plot.dataset.alpha=a;plot.dataset.fwer=f;
}
mInput.addEventListener('input',updateSelection);aInput.addEventListener('input',updateSelection);
document.getElementById('reset').addEventListener('click',()=>{mInput.value=20;aInput.value=50;updateSelection();
 Plotly.relayout(plot,{'scene.camera':{eye:{x:1.6,y:-1.8,z:1.1}}});});
updateSelection();
</script>'''
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Day 48 · Independent-null FWER</title><style>body{font:17px system-ui;max-width:1200px;margin:30px auto;padding:0 24px;color:#172554;background:#f8fafc}.controls{display:flex;gap:24px;flex-wrap:wrap;align-items:center}label{display:grid;gap:8px}output{font-weight:700}button{padding:10px}</style><body>' + controls + body + script + '</body></html>'
    (output / "02_fwer_surface.html").write_text(page, encoding="utf-8")
    return [png, "02_fwer_surface.html"], {"independent_m20_alpha05_fwer": value, "type": "analytical; no simulation"}


def corrections_visual(config, output):
    _, matrix, truth = normal_families(config, 3)
    p = matrix[0]
    order = np.argsort(p, kind="stable")
    ordered = p[order]
    m = p.size
    ranks = np.arange(1, m + 1)
    thresholds = {"uncorrected": np.full(m, config.alpha), "bonferroni": np.full(m, config.alpha/m),
                  "holm": config.alpha / (m - ranks + 1), "bh": config.alpha * ranks/m}
    masks = {method: adjusted_matrix(p[None, :], method)[0] <= config.alpha for method in METHODS[:-1]}
    fig, axes = canvas("C · Same family, different error targets",
                       "SYNTHETIC · Independent two-sided N(mean,1) tests; triangles mark simulated signals, circles true nulls.\n"
                       "Filled colored marks = rejected. Holm stops at first failure; BH uses the largest qualifying rank.", (2, 2), (12, 8))
    for ax, method in zip(axes.flat, METHODS[:-1]):
        rejected = masks[method][order]
        ax.plot(ranks, thresholds[method], color=COLORS[method], ls="--", lw=2, label="Rank threshold")
        for null, marker in ((True, "o"), (False, "^")):
            selected = truth[order] == null
            ax.scatter(ranks[selected], ordered[selected], marker=marker, s=40,
                       facecolors=np.where(rejected[selected], COLORS[method], "white"),
                       edgecolors=COLORS[method], zorder=4)
        if method == "holm":
            failures = np.flatnonzero(ordered > thresholds[method])
            if failures.size:
                ax.axvline(ranks[failures[0]], color=RED, alpha=.65, label="First failure: stop")
        if method == "bh" and rejected.any():
            ax.axvline(ranks[rejected][-1], color=COLORS[method], alpha=.65, label="Largest qualifying rank")
        ax.set(yscale="log", xlabel="Ordered hypothesis rank", ylabel="Raw two-sided p-value",
               title=f"{LABELS[method]}: {rejected.sum()} rejected ({'FDR' if method == 'bh' else 'per-test' if method == 'uncorrected' else 'FWER'})")
        ax.legend(fontsize=8, loc="upper left")
    png = save_png(fig, output, "03_correction_comparison.png")
    plt.close(fig)
    fig, axes = canvas("C · Benjamini–Hochberg is a step-up procedure",
                       "SYNTHETIC · Same first simulated family, not selected for its rejection count.\n"
                       "BH targets FDR under independence / suitable PRDS. Bonferroni and Holm instead protect FWER.", (1, 1))
    ax = axes[0, 0]
    ax.set(xlim=(.4, m+.6), ylim=(max(np.finfo(float).tiny, p.min()*.45), 1.2), yscale="log",
           xlabel="Hypothesis position (input order → sorted rank)", ylabel="Two-sided p-value (log scale)")
    inverse = np.empty(m, dtype=int)
    inverse[order] = np.arange(m)
    dots = ax.scatter(ranks, p, s=65, c=np.where(truth, BLUE, "#d97706"), zorder=4)
    rule, = ax.plot([], [], color=COLORS["bh"], ls="--", lw=2, label="BH boundary: rank × q / m")
    star, = ax.plot([], [], "*", color=RED, ms=18, zorder=6)
    status = fig.text(.08, .88, "", fontsize=11)
    k = int(masks["bh"].sum())
    ax.legend(loc="upper left")

    def update(frame):
        progress = frame / (config.frames-1)
        tween = np.clip((progress-.15)/.2, 0, 1)
        positions = ranks*(1-tween) + (inverse+1)*tween
        dots.set_offsets(np.c_[positions, p])
        show_rule = progress >= .4
        rule.set_data(ranks if show_rule else [], thresholds["bh"] if show_rule else [])
        scan = int(np.clip((progress-.45)/.35, 0, 1)*m)
        passed = np.flatnonzero(ordered[:scan] <= thresholds["bh"][:scan])
        current_k = int(passed[-1]+1) if passed.size else 0
        star.set_data([current_k] if current_k else [], [ordered[current_k-1]] if current_k else [])
        finished = progress >= .82
        dots.set_facecolor(np.where(masks["bh"], COLORS["bh"], "#cbd5e1") if finished else np.where(truth, BLUE, "#d97706"))
        if finished:
            status.set_text(f"Largest qualifying rank k = {k} → reject ALL ranks 1…{k}. Total discoveries = {k}.")
        elif show_rule:
            status.set_text(f"Scan through rank {scan}; largest qualifying rank so far = {current_k}. Do not stop at the first failure.")
        elif tween:
            status.set_text("Sort the complete family by p-value; keep each hypothesis identity.")
        else:
            status.set_text(f"Unsorted p-values · m = {m}, q = {config.alpha:g} · blue = null; orange = simulated signal")

    gif = save_gif(fig, update, output, "03_bh_step_up.gif", config.frames)
    plt.close(fig)
    return [png, gif], {"p_values": p.tolist(), "true_null": truth.tolist(),
                        "rejections": {key: int(mask.sum()) for key, mask in masks.items()},
                        "reference_check": reference_check(matrix[:1], config.alpha)}


def errorbars(ax, x, records, metric, methods):
    for method in methods:
        cells = [record[method][metric] for record in records]
        y = np.array([cell["estimate"] if cell is not None else np.nan for cell in cells])
        se = np.array([cell["mc_se"] if cell is not None else np.nan for cell in cells])
        ax.plot(x, y, "o-", lw=1.7, ms=4, color=COLORS[method], label=LABELS[method])
        ax.fill_between(x, np.maximum(0, y-1.96*se), np.minimum(1, y+1.96*se), color=COLORS[method], alpha=.12)


def tradeoffs_visual(config, output):
    results, counts, references = tradeoff_study(config)
    x = [r["effect"] for r in results]
    records = [r["metrics"] for r in results]
    fig, axes = canvas("D · More discoveries and a different error budget",
                       f"SYNTHETIC · {config.repetitions:,} repetitions per effect; {config.signals}/{config.hypotheses} simulated non-nulls. Shared noise across effect sizes.\n"
                       "At zero effect all nulls are true; power is undefined. Shading: pointwise ±1.96 MCSE (not simultaneous intervals).", (2, 2), (12, 8))
    for ax, metric, title in zip(axes.flat[:3], ("fwer", "fdr", "power"),
                               ("FWER: P(any false discovery)", "FDR: mean FDP, including zero discoveries", "Power: mean fraction of non-nulls rejected")):
        errorbars(ax, x, records, metric, METHODS[:-1])
        ax.set(xlabel="Non-null mean shift in Z units", ylabel="Estimated probability / proportion", ylim=(-.015, 1.015), title=title)
        if metric != "power":
            ax.axhline(config.alpha, color=RED, ls="--", alpha=.8)
        elif not config.signals:
            ax.text(.5, .5, "Power undefined: no non-null hypotheses", transform=ax.transAxes, ha="center")
    ax = axes.flat[3]
    for method in METHODS[:-1]:
        for metric, ls in (("tp", "-"), ("fp", ":")):
            cells = [record[method][metric] for record in records]
            y = np.array([cell["estimate"] for cell in cells])
            se = np.array([cell["mc_se"] for cell in cells])
            ax.errorbar(x, y, yerr=1.96*se, color=COLORS[method], ls=ls, capsize=2)
    ax.set(xlabel="Non-null mean shift in Z units", ylabel="Mean count per family", title="True discoveries (solid), false (dotted)")
    axes[0, 0].legend(ncol=2, fontsize=8)
    png = save_png(fig, output, "04_error_control_tradeoffs.png")
    plt.close(fig)

    fig, axes = canvas("D · Realized FDP varies; FDR averages over experiments",
                       f"SYNTHETIC · Repeated independent families at Z shift {config.effect:g}. Empty discovery sets contribute FDP = 0.\n"
                       "The pale points show individual BH FDP values. Running averages are not sequentially valid significance tests.")
    left, right = axes[0]
    left.set(xlabel="Independent experiments accumulated", ylabel="Mean FDP so far (estimated FDR)", xlim=(1, config.repetitions), ylim=(0, 1))
    left.axhline(config.alpha, color=RED, ls="--", label="Target level")
    dots = left.scatter([], [], s=6, color=COLORS["bh"], alpha=.12)
    lines = {method: left.plot([], [], color=COLORS[method], label=LABELS[method])[0] for method in METHODS[:-1]}
    left.legend(ncol=2, fontsize=8, loc="upper right")
    positions = np.arange(4)
    tp_bars = right.bar(positions-.18, np.zeros(4), width=.35, color=[COLORS[m] for m in METHODS[:-1]])
    fp_bars = right.bar(positions+.18, np.zeros(4), width=.35, color=RED, alpha=.65)
    right.set(xticks=positions, xticklabels=[LABELS[m] for m in METHODS[:-1]],
              ylabel="Average count per family", ylim=(0, max(1, config.hypotheses)), title="True discoveries (method color) / false (red)")
    right.tick_params(axis="x", labelrotation=18)
    status = fig.text(.08, .88, "", fontsize=11)
    checkpoints = np.unique(np.linspace(10, config.repetitions, config.frames, dtype=int))
    cumulative = {method: {key: np.cumsum(counts[method][key])/np.arange(1, config.repetitions+1)
                           for key in ("fdp", "tp", "fp")} for method in METHODS[:-1]}

    def update(frame):
        n = checkpoints[frame]
        # Limit the FDP scatter to a deterministic sample; means still use ALL repetitions.
        sample = np.unique(np.linspace(0, n-1, min(n, 250), dtype=int))
        dots.set_offsets(np.c_[sample+1, counts["bh"]["fdp"][sample]])
        for i, method in enumerate(METHODS[:-1]):
            lines[method].set_data(np.arange(1, n+1), cumulative[method]["fdp"][:n])
            tp_bars[i].set_height(cumulative[method]["tp"][n-1])
            fp_bars[i].set_height(cumulative[method]["fp"][n-1])
        status.set_text(f"{n:,} experiments · latest BH FDP = {counts['bh']['fdp'][n-1]:.3f} · mean BH FDP = {cumulative['bh']['fdp'][n-1]:.3f}")

    gif = save_gif(fig, update, output, "04_discovery_simulation.gif", len(checkpoints))
    plt.close(fig)
    return [png, gif], {"effect_sweep": results, "reference_check": references,
                        "paired_noise_across_effects": True,
                        "signals_at_selected_effect": config.signals if config.effect > 0 else 0,
                        "candidate_signal_count": config.signals}


def peeking_visual(config, output):
    looks, z, p, decisions, details = sequential_study(config)
    selected = details["selected_index_zero_based"]
    zs, ps = z[selected], p[selected]
    stop = details["first_naive_crossing_index"]
    fig, axes = canvas("E · A valid p-value at each look is not a valid stopping rule",
                       f"SYNTHETIC NULL · Selected path {selected} (zero-based) from {config.repetitions:,} simulated paths: {details['selection']}.\n"
                       "Not a typical path. After the naive stop, the continued path is counterfactual; it is kept to compare policies.", (2, 1), (11.5, 7.4))
    top, bottom = axes[:, 0]
    critical = norm.isf(config.alpha/2)
    for value in (-critical, critical):
        top.axhline(value, color=RED, ls="--", lw=1)
    top.axhline(0, color="#94a3b8", lw=.8)
    top.set(xlabel="Cumulative observations at planned looks", ylabel="Cumulative Z statistic",
            xlim=(0, config.final_n*1.02), ylim=(min(-critical-1, zs.min()-.5), max(critical+1, zs.max()+.5)))
    bottom.set(xlabel="Cumulative observations at planned looks", ylabel="Two-sided p-value (log scale)",
               xlim=(0, config.final_n*1.02), ylim=(min(ps.min()*.6, config.alpha/config.looks*.6), 1.2), yscale="log")
    bottom.axhline(config.alpha, color=RED, ls="--", label=f"Naive threshold {config.alpha:g}")
    bottom.axhline(config.alpha/config.looks, color=COLORS["bonferroni"], ls=":", label=f"Finite-look threshold {config.alpha/config.looks:g}")
    bottom.legend(loc="upper right", fontsize=8)
    all_z, = top.plot([], [], "o-", color="#94a3b8", alpha=.6)
    observed_z, = top.plot([], [], "o-", color=BLUE, lw=2)
    all_p, = bottom.plot([], [], "o-", color="#94a3b8", alpha=.6)
    observed_p, = bottom.plot([], [], "o-", color=BLUE, lw=2)
    stop_z, = top.plot([], [], "*", color=RED, ms=17, zorder=5)
    stop_p, = bottom.plot([], [], "*", color=RED, ms=17, zorder=5)
    status = fig.text(.08, .88, "", fontsize=11)

    def update(frame):
        nlooks = int(frame/(config.frames-1)*config.looks)
        visible = min(nlooks, stop+1) if stop is not None else nlooks
        all_z.set_data(looks[:nlooks], zs[:nlooks])
        all_p.set_data(looks[:nlooks], ps[:nlooks])
        observed_z.set_data(looks[:visible], zs[:visible])
        observed_p.set_data(looks[:visible], ps[:visible])
        crossed = stop is not None and nlooks > stop
        stop_z.set_data([looks[stop]] if crossed else [], [zs[stop]] if crossed else [])
        stop_p.set_data([looks[stop]] if crossed else [], [ps[stop]] if crossed else [])
        final = f"Fixed final: {'REJECT' if decisions['fixed_horizon'][selected] else 'do not reject'}" if nlooks == config.looks else "Fixed final decision: wait for final horizon"
        if nlooks:
            status.set_text(f"Look {nlooks}/{config.looks} · n = {looks[nlooks-1]} · p = {ps[nlooks-1]:.4f} · Naive: {'STOP / reject' if crossed else 'continue'}\n{final}")
        else:
            status.set_text("Before the first planned look: no observations yet.\nFixed-horizon inference waits until the final sample size.")

    gif = save_gif(fig, update, output, "05_peeking_problem.gif", config.frames)
    plt.close(fig)
    fig, axes = canvas("E · Type I error belongs to the complete monitoring policy",
                       f"SYNTHETIC NULL · {config.repetitions:,} shared paths; {config.looks} nested planned looks; final n = {config.final_n:,}.\n"
                       "Bars: Monte Carlo estimates; whiskers: ±1.96 MCSE. Bonferroni over finite looks is not an optimized sequential design.", (1, 1))
    ax = axes[0, 0]
    names = ("fixed_horizon", "naive_peeking", "bonferroni_looks")
    cells = [details["rates"][name] for name in names]
    heights = [cell["estimate"] for cell in cells]
    ax.bar(np.arange(3), heights, color=[BLUE, RED, COLORS["bonferroni"]],
           yerr=[1.96*cell["mc_se"] for cell in cells], capsize=6, width=.6)
    for i, height in enumerate(heights):
        ax.text(i, height+1.96*cells[i]["mc_se"]+.008, f"{height:.3f}", ha="center", fontsize=12)
    ax.axhline(config.alpha, color=RED, ls="--", label=f"Nominal target {config.alpha:g}")
    ax.set(xticks=np.arange(3), xticklabels=["Final horizon only", "First ordinary p crossing", "Finite-look Bonferroni"],
           ylabel="Estimated null rejection probability", ylim=(0, min(1, max(heights)*1.3+.03)))
    ax.legend(loc="upper right")
    png = save_png(fig, output, "05_peeking_error_rates.png")
    plt.close(fig)
    return [gif, png], details


def dependence_visual(config, output):
    results = dependence_study(config)
    x = [r["rho"] for r in results]
    fig, axes = canvas("F · Dependence changes observed family-level error",
                       f"SYNTHETIC · PSD equicorrelated MVN, rho ≥ 0; two-sided tests; {config.repetitions:,} families per point. Shading: ±1.96 MCSE.\n"
                       "These observations do not establish a BH guarantee for arbitrary dependence or prove PRDS for two-sided tests.", (2, 2), (12, 8))
    settings = (("global_null", "fwer", "Global null: FWER = FDR"),
                ("mixed", "fwer", f"Mixed family: FWER ({config.signals if config.effect > 0 else 0} simulated signals)"),
                ("mixed", "fdr", "Mixed family: mean FDP = estimated FDR"),
                ("mixed", "power", "Mixed family: average non-null rejection fraction"))
    for ax, (scenario, metric, title) in zip(axes.flat, settings):
        errorbars(ax, x, [r[scenario] for r in results], metric, METHODS)
        ax.set(xlabel="Equicorrelation rho of Z statistics", ylabel="Estimated probability / proportion", ylim=(-.015, 1.015), title=title)
        if metric != "power":
            ax.axhline(config.alpha, color=RED, ls="--", lw=1)
    axes[0, 0].legend(ncol=3, fontsize=8)
    png = save_png(fig, output, "06_correlation_effects.png")
    plt.close(fig)
    return [png], {"correlation_sweep": results, "common_random_numbers_across_rho": True,
                   "scope": "nonnegative equicorrelation only; two-sided tests; no general BH dependence guarantee"}


GENERATORS = dict(zip(EXPERIMENTS, (null_visual, fwer_visual, corrections_visual,
                                   tradeoffs_visual, peeking_visual, dependence_visual)))


def validate_artifact(path):
    """Validate decoded content, not just file existence."""
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty artifact: {path}")
    result = {"file": path.name, "bytes": path.stat().st_size,
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if path.suffix in (".png", ".gif"):
        with Image.open(path) as im:
            result["size"] = list(im.size)
            result["frames"] = getattr(im, "n_frames", 1)
            if path.suffix == ".gif":
                hashes = set()
                for frame in range(im.n_frames):
                    im.seek(frame)
                    hashes.add(hashlib.sha256(im.convert("RGB").tobytes()).hexdigest())
                result["distinct_frames"] = len(hashes)
                if len(hashes) < 2:
                    raise ValueError(f"GIF needs multiple distinct frames: {path}")
            else:
                im.verify()
    return result


def write_index(output, records):
    articles = []
    for name, record in records.items():
        assets = []
        for asset in record["assets"]:
            escaped = html.escape(asset)
            if asset.endswith(".html"):
                assets.append(f'<p><a href="{escaped}">Open interactive 3D FWER surface and parameter controls</a></p>')
            else:
                assets.append(f'<figure><img src="{escaped}" alt="{escaped}" loading="lazy"><figcaption>{escaped}</figcaption></figure>')
        articles.append(f'<article id="{name}"><h2>{name.title()}</h2>' + ''.join(assets) + '</article>')
    links = ' · '.join(f'<a href="#{name}">{name.title()}</a>' for name in records)
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Day 48 statistical lab</title><style>body{font:17px system-ui;max-width:1200px;margin:30px auto;padding:0 20px;color:#172554;background:#f8fafc}img{max-width:100%;height:auto}figure{margin:20px 0}article{border-top:1px solid #cbd5e1;margin-top:40px}nav a{padding:8px}figcaption{color:#475569}</style><body><h1>Day 48 · Multiple testing and false positives</h1><p>Synthetic educational experiments. Interpretations require author review. Generated offline: no server or internet needed.</p><nav>' + links + '</nav>' + ''.join(articles) + '</body></html>'
    (output / "index.html").write_text(page, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--experiment", choices=("all", *EXPERIMENTS), default="all")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=48)
    parser.add_argument("--hypotheses", type=int, default=20)
    parser.add_argument("--nonnull-proportion", type=float, default=.2)
    parser.add_argument("--effect", type=float, default=3)
    parser.add_argument("--alpha", type=float, default=.05)
    parser.add_argument("--repetitions", type=int, help="Independent families per condition (100-50000)")
    parser.add_argument("--frames", type=int, help="Animation frames (8-60)")
    parser.add_argument("--null-tests", type=int)
    parser.add_argument("--looks", type=int, default=10)
    parser.add_argument("--final-n", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing generated run directory")
    args = parser.parse_args(argv)
    full = args.mode == "full"
    try:
        config = LabConfig(mode=args.mode, seed=args.seed, hypotheses=args.hypotheses,
                           nonnull_proportion=args.nonnull_proportion, effect=args.effect, alpha=args.alpha,
                           repetitions=args.repetitions if args.repetitions is not None else (20000 if full else 2000),
                           frames=args.frames if args.frames is not None else (48 if full else 24),
                           null_tests=args.null_tests if args.null_tests is not None else (5000 if full else 1000),
                           looks=args.looks, final_n=args.final_n)
    except ValueError as error:
        parser.error(str(error))
    output = (args.output_dir or HERE / "outputs" / args.mode).resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        parser.error(f"output directory is not empty: {output}; choose a new path or --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    style()
    selected = EXPERIMENTS if args.experiment == "all" else (args.experiment,)
    records = {}
    for name in selected:
        print(f"Generating {name} ...", flush=True)
        assets, statistics = GENERATORS[name](config, output)
        records[name] = {"assets": assets, "statistics": statistics,
                         "validation": [validate_artifact(output / asset) for asset in assets]}
        print(f"Validated {len(assets)} artifacts for {name}", flush=True)
    write_index(output, records)
    report = {"executed_at_utc": datetime.now(timezone.utc).isoformat(), "configuration": asdict(config),
              "actual_signals": config.signals if config.effect > 0 else 0, "synthetic": True, "interpretation_status": "author review required",
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                           "statsmodels": statsmodels.__version__, "matplotlib": matplotlib.__version__,
                           "pillow": PIL.__version__, "plotly": plotly.__version__}, "experiments": records}
    (output / "run.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Offline lab: {output / 'index.html'}", flush=True)
    return report


if __name__ == "__main__":
    main()

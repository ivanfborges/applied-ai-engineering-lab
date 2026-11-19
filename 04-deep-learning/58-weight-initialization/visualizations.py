"""Scientific figures, measured animations and self-contained Plotly documents."""

from io import BytesIO
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from experiments import COLORS, SCALES


plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.18, "figure.facecolor": "#f7f9fc",
                     "axes.facecolor": "white", "savefig.facecolor": "#f7f9fc"})


def positive_log(values):
    """No arbitrary floor: zero/non-finite values become gaps, counted by the UI."""
    a = np.array([np.nan if x is None else x for x in values], dtype=float)
    valid = np.isfinite(a) & (a > 0)
    result = np.full(a.shape, np.nan)
    result[valid] = np.log10(a[valid])
    return result


def save_png(fig, path):
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def frame_image(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=95)
    buffer.seek(0)
    with Image.open(buffer) as im:
        result = im.convert("RGB").copy()
    plt.close(fig)
    return result


def write_gif(frames, path, duration=160):
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration, loop=0, optimize=False)


def moment_plot(bundle, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for strategy, runs in bundle["repeats"].items():
        data = np.array([[r["q"] / run["layers"][0]["q"] for r in run["layers"]] for run in runs])
        depth = np.arange(data.shape[1])
        color = COLORS[strategy]
        for row in data:
            axes[0].plot(depth, positive_log(row), color=color, alpha=0.22, lw=1)
        axes[0].plot(depth, positive_log(np.median(data, axis=0)), color=color, lw=2.4, label=f"{strategy}: median")
        expected = -depth*np.log10(2) if strategy == "xavier" else np.zeros(len(depth))
        axes[0].plot(depth, expected, "--", color=color, label=f"{strategy}: idealized")
    axes[0].set(title=f"B1 | Paired ReLU realizations ({len(bundle['seeds'])} seeds)", xlabel="Layer depth",
                ylabel="log10(q_layer / q_input)")
    axes[0].legend(fontsize=8)
    relu = bundle["relu"]
    axes[1].bar([0, 1, 3, 4], [relu["q_before"], relu["q_after"], relu["var_before"], relu["var_after"]],
                color=["#53738c", COLORS["he"], "#53738c", COLORS["he"]])
    axes[1].set(xticks=[0, 1, 3, 4], xticklabels=["Z", "ReLU(Z)", "Z", "ReLU(Z)"],
                title="B3 | Symmetric measured sample", ylabel="Moment value")
    axes[1].text(0.17, -0.17, "Second moment", transform=axes[1].transAxes)
    axes[1].text(0.67, -0.17, "Centered variance", transform=axes[1].transAxes)
    fig.suptitle("Initialization preserves a moment under assumptions, not every direction", weight="bold")
    fig.text(.5, .015, "Solid/thin: actual networks. Dashed: ensemble approximation. No optimizer steps or accuracy claims.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, .94))
    save_png(fig, path)


def distributions_plot(bundle, path):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for row, strategy in enumerate(("xavier", "he")):
        result = bundle["baseline"][strategy]
        for ax, (layer, values) in zip(axes[row], result["samples"].items()):
            samples = [v for v in values if v is not None]
            measured = result["layers"][int(layer)]
            ax.hist(samples, bins=36, color=COLORS[strategy], alpha=.85)
            ax.set(title=f"{strategy} | layer {layer}", xlabel="Activation (panel-specific x range)", ylabel="Sample count")
            ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 3), useMathText=True)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.text(.98, .95, f"q={measured['q']:.2e}\nvar={measured['variance']:.2e}\nmean={measured['mean']:.2e}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
    fig.suptitle("A3 | ReLU histograms: mass at zero, positive mean, q = variance + mean²", weight="bold")
    fig.text(.5, .01, "Histograms use a deterministic subsample; annotated moments use the entire batch. Horizontal scales differ explicitly.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .94))
    save_png(fig, path)


def normalization_plot(bundle, path):
    n = bundle["normalization"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    values = [np.array(n[key]) for key in ("before", "batchnorm", "layernorm")]
    limit = max(abs(v).max() for v in values)
    for ax, key, v in zip(axes[0], ("Before", "BatchNorm: columns ↓", "LayerNorm: rows →"), values):
        image = ax.imshow(v, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        ax.set(title=key, xlabel="Feature", ylabel="Example")
    for key, color in (("before", "#53738c"), ("batchnorm", COLORS["he"]), ("layernorm", COLORS["he_layer"])):
        axes[1, 0].hist(np.array(n[key]).ravel(), bins=20, alpha=.4, label=key, color=color)
    axes[1, 0].set(title="E1 | Before/after distributions", xlabel="Activation", ylabel="Count")
    axes[1, 0].legend(fontsize=8)
    for feature in range(4):
        axes[1, 1].plot([r["batch"] for r in n["sensitivity"]], [r["feature_mean_sd"][feature] for r in n["sensitivity"]],
                        "o-", label=f"Feature {feature}")
    axes[1, 1].set(xscale="log", yscale="log", title="E3 | Batch-mean variability",
                   xlabel="Batch size", ylabel="SD of estimated feature mean")
    axes[1, 1].legend(fontsize=8)
    axes[1, 2].plot(n["anchor_bn_a"], "o-", label="BN: companions A", color=COLORS["he"])
    axes[1, 2].plot(n["anchor_bn_b"], "o--", label="BN: shifted companions B", color=COLORS["large"])
    axes[1, 2].plot(n["anchor_ln_a"], "o-", label="LN: A = B", color=COLORS["he_layer"])
    axes[1, 2].set(title="E3 | Same anchor, new companions", xlabel="Feature", ylabel="Normalized anchor value")
    axes[1, 2].legend(fontsize=8)
    fig.suptitle("Normalization changes a group of values and its derivatives", weight="bold")
    fig.text(.5, .015, f"Empirical synthetic batches; {n['repetitions']} repetitions per batch size. No inference or training-quality ranking follows.", ha="center", fontsize=9)
    fig.subplots_adjust(top=.90, bottom=.12, right=.89, hspace=.65, wspace=.6)
    for ax in axes.ravel():
        ax.title.set_fontsize(10)
    cax = fig.add_axes([.92, .61, .012, .25])
    fig.colorbar(image, cax=cax, label="Activation (shared scale)")
    save_png(fig, path)


def normalization_eval_plot(bundle, path):
    n = bundle["normalization"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, key, title in zip(axes, ("train_test", "eval_test"), ("Training: statistics of this batch", "Evaluation: populated running statistics")):
        ax.imshow(n[key], cmap="RdBu_r", aspect="auto", vmin=-4, vmax=4)
        ax.set(title=title, xlabel="Feature", ylabel="Example")
    fig.suptitle(f"E4 | BatchNorm after {n['training_batches_for_eval']} actual training-mode forward batches", weight="bold")
    fig.text(.5, .01, "Same held batch; shared color range ±4 (saturated colors are explicit). Running buffers include its training-mode pass.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .9))
    save_png(fig, path)


def propagation_gif(bundle, path, *, kind="forward", strategies=None):
    strategies = strategies or (["small", "he", "large"] if kind == "backward" else list(bundle["baseline"]))
    depth = bundle["config"]["depth"]
    steps = list(range(depth + 1))
    if kind == "backward":
        steps.reverse()
    metric = "probe_rms" if kind == "backward" else "q"
    logs = np.array([positive_log([r[metric] for r in bundle["baseline"][s]["layers"]]) for s in strategies])
    finite = logs[np.isfinite(logs)]
    norm = Normalize(float(finite.min())-0.2, float(finite.max())+0.2)
    cmap = matplotlib.colormaps["viridis"]
    frames = []
    for current in steps:
        fig, (network, chart) = plt.subplots(1, 2, figsize=(11, 5.3), gridspec_kw={"width_ratios": [1.25, 1]})
        for row, strategy in enumerate(strategies):
            values = logs[row]
            visible = np.arange(depth + 1) >= current if kind == "backward" else np.arange(depth + 1) <= current
            # Each node is a layer and its color is the measured layer statistic.
            network.plot([0, depth], [row, row], color="#d2d8df", lw=2, zorder=0)
            network.scatter(np.arange(depth + 1)[visible], np.full(visible.sum(), row),
                            c=values[visible], cmap=cmap, norm=norm, s=55, edgecolors="white", lw=.3)
            network.scatter([current], [row], s=120, facecolors="none", edgecolors=COLORS[strategy], lw=2)
            measured = bundle["baseline"][strategy]["layers"][current][metric]
            network.text(0, row+.27, f"{strategy}: {measured:.2e}" if measured is not None else f"{strategy}: non-finite", color=COLORS[strategy], fontsize=9)
            chart.plot(np.arange(depth+1), values, color=COLORS[strategy], lw=2, label=strategy)
            chart.scatter([current], [values[current]], color=COLORS[strategy], s=30)
        network.set(ylim=(-.5, len(strategies)-.2), xlabel="Layer index (nodes encode measured layer magnitude)", yticks=[])
        network.grid(False)
        chart.axvline(current, color="#334155", alpha=.4)
        chart.set(xlabel="Layer index (input = 0)", ylabel=f"log10({'activation-gradient RMS' if kind == 'backward' else 'activation second moment'})",
                  ylim=(norm.vmin, norm.vmax))
        chart.legend(fontsize=8)
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=network, orientation="horizontal", fraction=.09, pad=.19,
                     label=f"Shared fixed color scale: log10({metric})")
        direction = "← Backward fixed output adjoint" if kind == "backward" else "Forward signal →"
        fig.suptitle(f"{direction} | layer {current}/{depth}", weight="bold")
        fig.text(.5, .02, "Actual CPU measurements; common random draws. Grey = not visited. Gaps = zero/non-finite; no artificial floor.", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .06, 1, .94))
        frames.append(frame_image(fig))
    write_gif(frames, path)


def relu_gif(bundle, path):
    r = bundle["relu"]
    before, after = np.array(r["before"]), np.array(r["after"])
    frames = []
    bins = np.linspace(-4, 4, 49)
    for progress in np.linspace(0, 1, 21):
        transformed = np.where(before < 0, before*(1-progress), before)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        axes[0].hist(before, bins=bins, alpha=.4, color="#53738c", label="Original symmetric input")
        axes[0].hist(transformed, bins=bins, alpha=.6, color=COLORS["he"], label="Current transformed sample")
        axes[0].set(xlim=(-4, 4), ylim=(0, len(before)*.6), xlabel="Value (tails outside ±4 omitted from histogram)", ylabel="Count")
        axes[0].legend(fontsize=8)
        x = np.linspace(-4, 4)
        axes[1].plot(x, np.where(x < 0, (1-progress)*x, x), color=COLORS["he"], lw=3)
        axes[1].set(xlim=(-4, 4), ylim=(-4, 4), xlabel="Input Z", ylabel="Transformed value")
        fig.suptitle(f"B3 | Negative slope {1-progress:.2f} → ReLU at 0", weight="bold")
        fig.text(.5, .06, f"Full symmetric sample: q(Z)={r['q_before']:.4f}; q(ReLU(Z))={r['q_after']:.4f}; "
                 f"var(ReLU(Z))={r['var_after']:.4f}", ha="center", fontsize=9)
        fig.text(.5, .01, "Intermediate frames are an explicit leaky-rectifier interpolation. The endpoint is actual ReLU, not half the variance.", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .12, 1, .93))
        frames.append(frame_image(fig))
    write_gif(frames, path, duration=110)


def training_plot(bundle, path, *, animated=False):
    runs = bundle["training"]
    max_epoch = max(r["epochs_requested"] for r in runs)
    frames = []
    checkpoints = list(range(0, max_epoch+1, max(1, max_epoch//24))) if animated else [max_epoch]
    if checkpoints[-1] != max_epoch:
        checkpoints.append(max_epoch)
    for checkpoint in checkpoints:
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        for ax, lr in zip(axes[0], (0.01, 0.15)):
            detail = ax.inset_axes([.48, .30, .48, .37])
            for strategy in SCALES:
                selected = [r for r in runs if r["strategy"] == strategy and r["lr"] == lr]
                for index, run in enumerate(selected):
                    history = [h for h in run["history"] if h["epoch"] <= checkpoint]
                    x = [h["epoch"] for h in history]
                    ax.plot(x, positive_log([h["train_loss"] for h in history]), color=COLORS[strategy], alpha=.25, lw=1)
                    ax.plot(x, positive_log([h["validation_loss"] for h in history]), "--", color=COLORS[strategy], alpha=.75,
                            label=strategy if index == 0 else None)
                    detail.plot(x, positive_log([h["validation_loss"] for h in history]), "--", color=COLORS[strategy], alpha=.7, lw=.8)
                failed = [r["nonfinite"]["epoch"] for r in selected if r["nonfinite"]]
                if failed:
                    ax.text(.98, .96-list(SCALES).index(strategy)*.07, f"{strategy}: failures at {failed}", color=COLORS[strategy],
                            transform=ax.transAxes, ha="right", fontsize=8)
            ax.set(title=f"F | SGD lr={lr}; thin=train, dashed=validation", xlabel="Epoch (completed updates)",
                   ylabel="log10(mean squared error)", xlim=(0, max_epoch))
            ax.legend(fontsize=8, loc="lower left")
            detail.set(xlim=(0, max_epoch), ylim=(-2, 1), title="Validation detail: log10 MSE ∈ [−2, 1]")
            detail.title.set_fontsize(7)
            detail.tick_params(labelsize=6)
        for ax, metric, title in zip(axes[1], ("gradient_norm", "update_norm"), ("Pre-clipping global gradient norm", "Actual parameter update norm")):
            for strategy in SCALES:
                run = next(r for r in runs if r["strategy"] == strategy and r["lr"] == .15)
                history = [h for h in run["history"] if h["epoch"] <= checkpoint]
                ax.plot([h["epoch"] for h in history], positive_log([h[metric] for h in history]), color=COLORS[strategy], label=strategy)
            clipped = bundle["clipping"][0]
            history = [h for h in clipped["history"] if h["epoch"] <= checkpoint]
            ax.plot([h["epoch"] for h in history], positive_log([h[metric] for h in history]), "--", color=COLORS["he_layer"], label="large + clip=1")
            if metric == "gradient_norm":
                ax.plot([h["epoch"] for h in history], positive_log([h["clipped_gradient_norm"] for h in history]), ":", color="#222", label="after clipping")
            ax.set(title=title + " | first seed, lr=0.15", xlabel="Update from checkpoint epoch", ylabel="log10(norm)", xlim=(0, max_epoch))
            ax.legend(fontsize=7)
        fig.suptitle(f"Actual synthetic regression | {len(bundle['seeds'])} seeds | checkpoint {checkpoint}", weight="bold")
        fig.text(.5, .01, "Fixed 288/96 split, full-batch SGD, shared head and Gaussian hidden draws. Gaps mark zero/non-finite; failures terminate a run.", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .04, 1, .95))
        if animated:
            frames.append(frame_image(fig))
        else:
            save_png(fig, path)
    if animated:
        write_gif(frames, path, duration=170)


def jacobian_plot(bundle, path):
    j = bundle["jacobian"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for r in j["records"]:
        axes[0].plot(np.arange(1, 13), positive_log(r["singular_values"]), color=COLORS[r["strategy"]],
                     alpha=.45 if r["depth"] != 12 else 1, ls={1: ":", 5: "--", 12: "-"}[r["depth"]],
                     label=f"{r['strategy']} d={r['depth']} rank={r['numerical_rank']}")
    axes[0].set(title="G | Measured 12×12 input-output Jacobians", xlabel="Singular-value rank (descending)", ylabel="log10(singular value)")
    axes[0].legend(fontsize=6, ncol=2)
    axes[1].plot(j["controlled_spectra"]["identity"], "o-", label="Identity: 12 preserved directions")
    axes[1].plot(j["controlled_spectra"]["rank_one"], "s-", label="Rank-one: 11 zero directions")
    axes[1].set(title="Constructed linear control: same mean(s²)=1", xlabel="Singular-value index (0-based)", ylabel="Singular value (linear scale)")
    axes[1].legend(fontsize=8)
    fig.text(.5, .01, "Left: exact SVD of measured Jacobian; zeros are gaps, ranks use a recorded machine-precision threshold. Right: analytical control.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    save_png(fig, path)


def surface_figure(bundle):
    s = bundle["surface"]
    figure = go.Figure()
    for key in ("forward", "backward"):
        raw = np.array(s[key], dtype=float)
        z = np.array([positive_log(row) for row in s[key]])
        figure.add_trace(go.Surface(x=s["depths"], y=s["scales"], z=z, customdata=raw,
                                   visible=key == "forward", colorscale="RdBu_r", colorbar={"title": "log10(metric)"},
                                   hovertemplate="Depth %{x}<br>Scale/He %{y:.3f}<br>log10 %{z:.4f}<br>Measured %{customdata:.4e}<extra></extra>"))
        for name, scale in (("Xavier", 1/np.sqrt(2)), ("He", 1.)):
            row = s["scales"].index(scale)
            figure.add_trace(go.Scatter3d(x=s["depths"], y=[scale]*len(s["depths"]), z=z[row],
                                         mode="lines+markers", name=name+" reference", line={"color": COLORS[name.lower()], "width": 6},
                                         visible=key == "forward"))
    figure.update_layout(title={"text": "D | Measured signal propagation — not a loss landscape",
                                "x": .5, "y": .98, "xanchor": "center", "font": {"size": 18}},
                         scene={"xaxis_title": "Network depth", "yaxis_title": "Weight std / He std",
                                "zaxis_title": "log10(activation second moment)", "aspectmode": "cube"},
                         updatemenus=[{"buttons": [
                             {"label": "Forward second moment", "method": "update", "args": [{"visible": [True]*3+[False]*3}, {"scene.zaxis.title.text": "log10(activation second moment)"}]},
                             {"label": "Backward probe RMS", "method": "update", "args": [{"visible": [False]*3+[True]*3}, {"scene.zaxis.title.text": "log10(input activation-gradient RMS)"}]}
                         ], "x": 0, "y": 1.12, "yanchor": "top"}],
                         annotations=[{"text": f"CPU float64; seed {s['config']['seed']}, width {s['config']['width']}, batch 16. He scale=1 is an approximate balance under ReLU assumptions.",
                                       "xref": "paper", "yref": "paper", "x": .5, "y": -.17, "showarrow": False}],
                         legend={"orientation": "h", "x": .5, "xanchor": "center", "y": -.08},
                         margin={"t": 145, "b": 130, "r": 100, "l": 20}, height=850,
                         template="plotly_white", uirevision="surface")
    return figure


def dashboard_html(bundle, path):
    script = Path(__file__).with_name("dashboard.js").read_text(encoding="utf-8")
    data = json.dumps(bundle, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    style = """body{margin:0;background:#edf2f7;color:#142b43;font:16px system-ui,sans-serif}header,main{max-width:1280px;margin:auto;padding:24px}
    header{padding-bottom:4px}h1{font-size:32px;margin-bottom:8px}h2{font-size:23px}p{line-height:1.6}.tag{color:#52677b;font-size:13px}
    nav{display:flex;flex-wrap:wrap;gap:14px}a{color:#176e9e}section{background:white;border-radius:12px;padding:20px;margin:20px 0;box-shadow:0 2px 8px #172b4310}
    .controls{display:flex;flex-wrap:wrap;gap:16px;margin:18px 0}label{font-size:14px;display:flex;flex-direction:column;gap:5px}select{font:inherit;padding:8px;border:1px solid #9caebe;border-radius:6px;background:white}
    .plot{height:420px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:10px}.note{font-size:14px;color:#4b6175}.warning{padding:10px;background:#fff3d6}
    #surface{height:700px}details{margin:12px 0}summary{cursor:pointer}table{border-collapse:collapse}td,th{border:1px solid #d5dfe9;padding:7px;text-align:left}
    @media(max-width:760px){.pair{grid-template-columns:1fr}header,main{padding:12px}h1{font-size:25px}section{padding:12px}.plot{height:380px}}"""
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Day 58 | Signal, gradients and stability</title><style>{style}</style><script>{get_plotlyjs()}</script></head><body>
    <header><div class="tag">APPLIED AI ENGINEERING LAB · DAY 58 · EMPIRICAL CPU LAB</div><h1>Can a signal survive depth?</h1>
    <p>Explore initialization, activation, normalization and actual learning. Every curve below uses saved numerical measurements.
    Dashed theoretical predictions are labeled. Interpretation: <strong>author review required</strong>.</p>
    <nav><a href="#signals">Signals</a><a href="#theory">Xavier vs He</a><a href="#landscape">3D stability</a><a href="#normalization">Normalization</a><a href="#training">Training</a><a href="#conditioning">Conditioning</a></nav></header>
    <main><section id="signals"><h2>01 / Inspect a paired network</h2><p>Choose a measured configuration. Selectors choose a finite saved grid; they do not run Python or interpolate networks.</p>
    <div class="controls"><label>Depth<select id="depth"></select></label><label>Width<select id="width"></select></label><label>Activation<select id="activation"></select></label>
    <label>Normalization before activation<select id="normalization-choice"></select></label><label>Seed<select id="seed"></select></label><label>Initializer<select id="strategy"><option value="all">All matched initializers</option></select></label></div>
    <p class="tag" id="configuration"></p><div class="pair"><div id="forward" class="plot"></div><div id="backward" class="plot"></div></div>
    <div class="controls"><label>Backward objective<select id="objective"><option value="probe_rms">Fixed output adjoint (RMS=1)</option><option value="loss_rms">Squared-output mean loss</option></select></label></div>
    <p class="note">q = mean(a²), while variance = q − mean(a)². The common output adjoint measures Jᵀu; one direction does not measure the entire Jacobian.
    Parameter gradients below use L=mean(output²), independently of the probe selector. Layer 0 is input.</p>
    <div id="parameters" class="plot"></div><p id="status" class="warning"></p><div id="histograms" class="plot"></div>
    <p class="note">Histograms use deterministic subsamples of early/middle/final layers. Hover for full-batch moments; x ranges differ if scales differ. BN measurements use training mode.</p></section>
    <section id="theory"><h2>02 / Xavier, He and the ReLU gate</h2><div class="pair"><div id="paired" class="plot"></div><div id="relu" class="plot"></div></div>
    <p class="note">Thin lines are actual paired seeds; dashed lines are idealized q/q₀=(1/2)^depth and 1. ReLU retains half the second moment for a symmetric distribution, not half the centered variance.
    Finite width and accumulated correlations can produce drift. This does not compare convergence.</p></section>
    <section id="landscape"><h2>03 / The depth–scale stability surface</h2><label>Measured surface<select id="surface-metric"><option value="forward">Forward activation second moment</option><option value="backward">Backward input probe RMS</option></select></label>
    <div id="surface"></div><p class="note" id="surface-note"></p><p class="note">Rotate, zoom and hover. The He scale of 1 is an approximate theoretical balance for constant-width ReLU layers.
    Xavier is 1/√2 of He here. This is a propagation surface, not an optimization loss landscape. One seed and one output probe do not establish robustness or conditioning.</p></section>
    <section id="normalization"><h2>04 / Which values share statistics?</h2><div class="controls"><label>Normalization view<select id="norm-view"><option value="batchnorm">BatchNorm: across examples ↓</option><option value="layernorm">LayerNorm: across features →</option><option value="before">Before normalization</option></select></label></div>
    <div class="pair"><div id="heatmap" class="plot"></div><div id="norm-hist" class="plot"></div></div><div class="pair"><div id="batchsize" class="plot"></div><div id="anchor" class="plot"></div></div>
    <div class="controls"><label>BatchNorm mode on same held batch<select id="bn-mode"><option value="train_test">Training statistics</option><option value="eval_test">Populated running statistics</option></select></label></div>
    <div id="bn-eval" class="plot"></div><p class="note" id="bn-note"></p><p class="note">BatchNorm's batch companions change an anchor's output; LayerNorm's do not in this experiment.
    Small-batch mean variability is measured over repeated batches. Neither demonstration proves a training or generalization advantage.</p></section>
    <section id="training"><h2>05 / Do the diagnostics predict learning?</h2><div class="controls"><label>SGD learning rate<select id="lr"><option>0.01</option><option>0.15</option></select></label>
    <label>Training seed<select id="train-seed"></select></label><label>Gradient clipping comparison<select id="clipping"><option value="yes">Show large + clip=1</option><option value="no">Hide clipping comparison</option></select></label></div>
    <div class="pair"><div id="losses" class="plot"></div><div id="updates" class="plot"></div></div><p class="warning" id="training-status"></p>
    <p class="note">Actual full-batch SGD on a fixed synthetic regression split, 288 train / 96 validation, six hidden layers, common linear head and paired hidden Gaussian draws.
    Solid loss=train; dashed=validation. Gradients are recorded before/after clipping; update norms measure actual parameter changes. No test set, universal ranking or production claim.</p></section>
    <section id="conditioning"><h2>06 / Average scale does not preserve every direction</h2><div id="jacobian" class="plot"></div><p class="note" id="jacobian-note"></p></section>
    <section><h2>Protocol and numerical limits</h2><p id="protocol"></p><p>Exact zeros and non-finite values become explicit gaps on logarithmic plots; counts are shown above. No positive plotting floor is substituted.
    Subnormals are counted. Zeros can be genuine ReLU outputs or numerical underflow; this lab does not identify their cause from magnitude alone.
    Color scales are fixed within each comparison. Full configuration and raw values are in the sibling measurement record.</p></section></main>
    <script id="lab-data" type="application/json">{data}</script><script>{script}</script></body></html>"""
    Path(path).write_text(html, encoding="utf-8")


def export_all(bundle, output, *, interactive_only=False):
    output = Path(output)
    for sub in ("interactive", "static", "gifs"):
        (output/sub).mkdir(parents=True, exist_ok=True)
    dashboard_html(bundle, output/"interactive/dashboard.html")
    surface_figure(bundle).write_html(output/"interactive/stability_surface.html", include_plotlyjs=True, full_html=True,
                                     config={"displaylogo": False, "responsive": True, "scrollZoom": True})
    if not interactive_only:
        moment_plot(bundle, output/"static/initialization_moments.png")
        distributions_plot(bundle, output/"static/activation_distributions.png")
        normalization_plot(bundle, output/"static/normalization_axes.png")
        normalization_eval_plot(bundle, output/"static/batchnorm_modes.png")
        training_plot(bundle, output/"static/training_comparison.png")
        propagation_gif(bundle, output/"gifs/forward_propagation.gif")
        propagation_gif(bundle, output/"gifs/xavier_vs_he.gif", strategies=["xavier", "he"])
        propagation_gif(bundle, output/"gifs/backward_gradients.gif", kind="backward")
        relu_gif(bundle, output/"gifs/relu_gate.gif")
        training_plot(bundle, output/"gifs/training_trajectories.gif", animated=True)
        if bundle["jacobian"]:
            jacobian_plot(bundle, output/"static/jacobian_spectrum.png")
    return [path for path in output.rglob("*") if path.is_file()]

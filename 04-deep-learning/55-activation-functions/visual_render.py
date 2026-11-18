"""Matplotlib teaching figures, sliders and bounded Pillow animations."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.widgets import Slider
import numpy as np

from from_scratch import ELEMENTWISE, activation_and_derivative, softmax, softmax_jacobian
from visual_math import attention, dying_neuron, entropy, gradient_chains


COLORS = {"sigmoid": "#2878a5", "tanh": "#16877b", "relu": "#dc7835",
          "leaky_relu": "#8755a3", "gelu": "#ce4e72", "identity": "#596779"}
LABELS = {"sigmoid": "Sigmoid", "tanh": "Tanh", "relu": "ReLU",
          "leaky_relu": "Leaky ReLU", "gelu": "GELU", "identity": "Identity control"}
CLASS_COLORS = ["#2878a5", "#dc7835", "#8755a3"]


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .16, "axes.titleweight": "bold",
                         "figure.facecolor": "#fafbfd", "axes.facecolor": "white",
                         "savefig.facecolor": "#fafbfd", "legend.frameon": False})


def layout(title, ncols=1, nrows=1, size=None, footer=""):
    style()
    fig, axes = plt.subplots(nrows, ncols, figsize=size or (12, 4.8), squeeze=False)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=.97)
    fig.subplots_adjust(top=.79, bottom=.18, left=.07, right=.97, wspace=.32, hspace=.5)
    fig.text(.07, .04, footer, fontsize=9, color="#465369")
    return fig, axes.ravel()


def save_png(fig, path):
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return Path(path)


def save_gif(fig, draw, frames, path, fps=8):
    animation = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps, blit=False)
    animation.save(path, writer=PillowWriter(fps=fps), dpi=85)
    plt.close(fig)
    return Path(path)


def activation_explorer(interactive=True):
    fig, axes = layout("A | A local slope changes the backward signal", 3,
                       size=(12, 6.5) if interactive else None,
                       footer="Sigmoid: (0,1); tanh: (-1,1); ReLU: [0,infinity); Leaky ReLU: unbounded; GELU: lower bounded, no upper bound.\n"
                              "Slope contributes to the chain rule; a local derivative alone does not determine network training.")
    if interactive:
        fig.subplots_adjust(bottom=.35)
    domain = np.linspace(-6, 6, 601)
    forward, derivatives, dots = {}, {}, {}
    for ax in axes[:2]:
        ax.axvspan(-6, 0, color="#edf1f8")
        ax.axhline(0, color="#a5aab3", lw=.7)
        ax.axvline(0, color="#a5aab3", lw=.7)
        ax.set_xlim(-6, 6)
        ax.set_xlabel("Preactivation x")
    for name in ELEMENTWISE:
        y, d = activation_and_derivative(domain, name, .1)
        forward[name], = axes[0].plot(domain, y, color=COLORS[name], label=LABELS[name])
        derivatives[name], = axes[1].plot(domain, d, color=COLORS[name], label=LABELS[name])
        dots[name], = axes[0].plot([], [], "o", color=COLORS[name], ms=5)
    axes[0].set(title="Forward transformation", ylabel="f(x)", ylim=(-1.4, 6.3))
    axes[1].set(title="Local sensitivity", ylabel="Selected backward slope", ylim=(-.18, 1.25))
    axes[0].legend(fontsize=8, loc="upper left")
    axes[1].text(.03, .95, "Flat tails: sigmoid / tanh\nReLU negative side: slope 0", transform=axes[1].transAxes,
                 va="top", fontsize=8)
    axes[2].axis("off")
    axes[2].set_title("Inspect one input")
    readout = axes[2].text(0, .94, "", va="top", family="monospace", fontsize=10)
    markers = [ax.axvline(-2, color="#1e293b", ls="--", lw=1) for ax in axes[:2]]
    def update(value=-2, alpha=.1):
        y, d = activation_and_derivative(domain, "leaky_relu", alpha)
        forward["leaky_relu"].set_ydata(y)
        derivatives["leaky_relu"].set_ydata(d)
        rows = [f"x = {value:+.3f}", f"alpha = {alpha:.3f}", "", "Function     f(x)   slope"]
        for name in ELEMENTWISE:
            v, slope = activation_and_derivative(value, name, alpha)
            rows.append(f"{LABELS[name]:10} {float(v):+.3f} {float(slope):+.3f}")
            dots[name].set_data([value], [v])
        rows += ["", "At x=0: ReLU and Leaky", "ReLU are not differentiable.",
                 "Shown slopes 0 / alpha", "are backward conventions.", "", "Tails approach bounds;",
                 "finite precision may", "reach an endpoint."]
        readout.set_text("\n".join(rows))
        for marker in markers:
            marker.set_xdata([value, value])
        fig.canvas.draw_idle()
    update()
    controls = {}
    if interactive:
        controls["input"] = Slider(fig.add_axes((.13, .23, .72, .035)), "Input x", -6, 6, valinit=-2)
        controls["alpha"] = Slider(fig.add_axes((.13, .16, .72, .035)), "Leaky slope", .001, .3, valinit=.1)
        for slider in controls.values():
            slider.on_changed(lambda _: update(controls["input"].val, controls["alpha"].val))
    return {"figure": fig, "sliders": controls, "update": update, "readout": readout,
            "forward_lines": forward, "derivative_lines": derivatives}


def heatmap(ax, matrix, title, vmin, vmax, cmap="coolwarm", row_prefix="p", col_prefix="z"):
    ax.grid(False)
    image = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(matrix.shape[1]), [f"{col_prefix}{i+1}" for i in range(matrix.shape[1])])
    ax.set_yticks(range(matrix.shape[0]), [f"{row_prefix}{i+1}" for i in range(matrix.shape[0])])
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=9)
    return image


def softmax_explorer(interactive=True):
    fig, axes = layout("F / H | One logit changes every competing probability", 4, size=(14, 5.3),
                       footer="Entropy (nats) measures concentration, not factual correctness. Temperature is not fitted calibration.\n"
                              "Diagonal Jacobian terms are nonnegative; off-diagonal terms are nonpositive. At temperature T, J=(diag(p)-pp')/T.")
    if interactive:
        fig.subplots_adjust(bottom=.43)
    fig.subplots_adjust(top=.74)
    labels = ["Class 1", "Class 2", "Class 3"]
    logits_bars = axes[0].bar(labels, [2, 1, -1], color=CLASS_COLORS)
    probability_bars = axes[1].bar(labels, softmax([2, 1, -1]), color=CLASS_COLORS)
    axes[0].set(title="Raw logits (before common offset)", ylabel="Logit", ylim=(-5.5, 5.5))
    axes[1].set(title="Normalized probabilities", ylabel="Probability", ylim=(0, 1.12))
    baseline = softmax_jacobian([2, 1, -1])
    j_image = heatmap(axes[2], baseline, "Jacobian at current logits", -.5, .5)
    delta_image = heatmap(axes[3], baseline * 0, "Jacobian change from initial state", -.5, .5)
    for ax in axes[:2]:
        ax.tick_params(axis="x", labelsize=8)
    status = fig.text(.07, .83, "", fontsize=10)
    state = {}
    def update(logits=(2, 1, -1), temperature=1., offset=0.):
        z = np.asarray(logits, dtype=float)
        p = softmax(z + offset, temperature=temperature)
        j = softmax_jacobian(z + offset, temperature)
        delta = j - baseline
        for bar, v in zip(logits_bars, z):
            bar.set_height(v)
        for bar, v in zip(probability_bars, p):
            bar.set_height(v)
        for image, ax, matrix in ((j_image, axes[2], j), (delta_image, axes[3], delta)):
            image.set_data(matrix)
            limit = max(.25, np.max(np.abs(matrix)))
            image.set_clim(-limit, limit)
            for text, value in zip(ax.texts, matrix.ravel()):
                text.set_text(f"{value:.3f}")
        winners = np.flatnonzero(np.isclose(z, z.max(), atol=0, rtol=0)) + 1
        shift_error = np.max(np.abs(p - softmax(z, temperature=temperature)))
        status.set_text(f"T={temperature:.2f} | sum={p.sum():.9f} | largest logit: {winners.tolist()} | "
                        f"entropy={entropy(p):.4f} nats | common offset={offset:+.0f}; max probability change={shift_error:.1e}")
        state.update(probabilities=p, jacobian=j, entropy=float(entropy(p)))
        fig.canvas.draw_idle()
    controls = {}
    if interactive:
        for i in range(3):
            controls[f"logit{i}"] = Slider(fig.add_axes((.13, .32 - i * .045, .72, .022)),
                                           f"Logit {i+1}", -5, 5, valinit=[2, 1, -1][i])
        controls["temperature"] = Slider(fig.add_axes((.13, .185, .72, .022)), "Temperature", .1, 5, valinit=1)
        controls["offset"] = Slider(fig.add_axes((.13, .14, .72, .022)), "Offset", -10000, 10000, valinit=0)
        for slider in controls.values():
            slider.on_changed(lambda _: update([controls[f"logit{i}"].val for i in range(3)],
                                                controls["temperature"].val, controls["offset"].val))
    update()
    return {"figure": fig, "sliders": controls, "update": update, "state": state,
            "probability_bars": probability_bars}


class Renderer:
    def __init__(self, output, frames=32):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.frames = frames

    def activations(self):
        return save_png(activation_explorer(False)["figure"], self.output / "01_activation_derivatives.png")

    def transformation(self):
        fig, axes = layout("B | Follow a scalar from input to output", 2, 2, size=(10, 7),
                           footer="Arrows connect (input, top rail) to (activation output, bottom rail).\n"
                                  "All panels share the same scalar axis. Slopes shown at zero use the selected backward convention.")
        names = ("relu", "sigmoid", "tanh", "gelu")
        signal = np.linspace(-6, 6, self.frames)
        def draw(frame):
            x = signal[frame]
            for ax, name in zip(axes, names):
                ax.clear()
                y, slope = activation_and_derivative(x, name)
                ax.scatter(signal, np.ones_like(signal), c="#cbd4df", s=10)
                mapped = activation_and_derivative(signal, name)[0]
                ax.scatter(mapped, np.zeros_like(signal), color=COLORS[name], alpha=.18, s=12)
                ax.scatter([x, y], [1, 0], color=COLORS[name], s=65, zorder=3)
                ax.annotate("", xy=(y, .04), xytext=(x, .96),
                            arrowprops={"arrowstyle": "->", "lw": 2, "color": COLORS[name]})
                ax.axvline(0, color="#8c98a7", ls="--", lw=.7)
                ax.set(xlim=(-6.5, 6.5), ylim=(-.3, 1.4), yticks=[0, 1],
                       yticklabels=["Output", "Input"], xlabel="Scalar value",
                       title=f"{LABELS[name]} | x={x:+.2f} -> f(x)={y:+.3f} | slope={slope:+.3f}")
                ax.title.set_fontsize(10)
        return save_gif(fig, draw, self.frames, self.output / "02_signal_transformation.gif")

    def gradients(self, animate=False):
        chains = gradient_chains()
        fig, axes = layout("C | Backpropagation multiplies local sensitivities", 2,
                           footer="Controlled scalar chains, upstream gradient=1; constant local preactivations maintained by chosen biases.\n"
                                  "These are derivative products, not trained-network benchmarks. Weights can attenuate or amplify positive ReLU gradients.")
        ax, table = axes
        ax.set(xlim=(0, 20), ylim=(1e-60, 1e4), yscale="log", xlabel="Layers traversed backward",
               ylabel="Absolute cumulative gradient", title="Activation slope AND weight contribute")
        colors = [COLORS["sigmoid"], COLORS["tanh"], COLORS["relu"], "#596779", "#8755a3"]
        lines = [ax.plot([], [], color=color, label=c["label"])[0] for c, color in zip(chains, colors)]
        ax.legend(fontsize=8, loc="lower left")
        table.axis("off")
        text = table.text(0, 1, "", va="top", family="monospace", fontsize=10)
        def draw(frame):
            depth = round(frame * 20 / max(1, self.frames - 1)) if animate else 20
            rows = [f"Current backward depth: {depth}", "", "Case and weight    |gradient|", ""]
            for chain, line in zip(chains, lines):
                line.set_data(chain["depth"][:depth+1], chain["magnitude"][:depth+1])
                rows.append(f"{chain['activation']:8} w={chain['weight']:<3} {chain['magnitude'][depth]:.2e}")
            rows += ["", "Selected depths (sigmoid / ReLU w=1)"]
            for d in [1, 5, 10, 20]:
                rows.append(f"{d:2d}: {chains[0]['magnitude'][d]:.2e} / 1.00")
            rows += ["", "Sigmoid local slope: 0.017663", "Tanh local slope:    0.001341", "Positive ReLU slope: 1.000000"]
            text.set_text("\n".join(rows))
        if animate:
            return save_gif(fig, draw, self.frames, self.output / "03_gradient_chain.gif")
        draw(0)
        return save_png(fig, self.output / "03_gradient_chain.png")

    def dying(self, animate=False):
        data = dying_neuron()
        fig, axes = layout("D | Identical starts, different negative-side gradient paths", 3, size=(12, 5.4),
                           footer="41 synthetic inputs; target=0.6+0.2x; initial w=0.2, b=-1.2; half-MSE; full-batch GD, lr=0.5; Leaky slope=0.1.\n"
                                  "This constructed recovery is not guaranteed for other data, initialization, slope or optimizer settings.")
        def draw(frame):
            step = round(frame * 80 / max(1, self.frames - 1)) if animate else 80
            for ax in axes:
                ax.clear()
            axes[0].axvline(0, color="#26364b", ls="--")
            axes[1].plot(data["x"], data["target"], "k--", label="Target")
            for name in ("relu", "leaky_relu"):
                history = data["traces"][name]
                state = history[step]
                axes[0].hist(state["z"], bins=np.linspace(-1.5, 1.2, 20), alpha=.5,
                             color=COLORS[name], label=LABELS[name])
                axes[1].plot(data["x"], state["output"], color=COLORS[name], label=LABELS[name])
                axes[2].plot(range(step + 1), [h["loss"] for h in history[:step + 1]], color=COLORS[name], label=LABELS[name])
            relu = data["traces"]["relu"][step]
            leaky = data["traces"]["leaky_relu"][step]
            axes[0].set(title=f"Step {step}: preactivations relative to zero", xlabel="z = wx + b", ylabel="Input count", xlim=(-1.5, 1.2))
            axes[1].set(title="Same target, different output signal", xlabel="Input x", ylabel="Neuron output", ylim=(-.2, 1))
            axes[2].set(title="Loss and current parameter gradient", xlabel="GD updates", ylabel="Half-MSE (log scale)",
                        yscale="log", xlim=(0, 80), ylim=(1e-12, 1))
            axes[2].text(.03, .1, f"ReLU ||grad||={relu['gradient_norm']:.3g}\nLeaky ||grad||={leaky['gradient_norm']:.3g}\n"
                         f"Negative slope: 0 vs {data['alpha']}\nReLU b={relu['b']:+.2f}; Leaky b={leaky['b']:+.2f}",
                         transform=axes[2].transAxes, fontsize=9)
            axes[1].legend(fontsize=8)
        if animate:
            return save_gif(fig, draw, self.frames, self.output / "04_dying_relu.gif")
        draw(0)
        return save_png(fig, self.output / "04_dying_relu.png")

    def gelu(self):
        fig, axes = layout("E | GELU modulates smoothly; ReLU thresholds at zero", 3,
                           footer="Exact GELU = x Phi(x); deterministic modulation, not a probability output.\n"
                                  "A smooth function may have negative derivatives or derivatives above 1. Smoothness does not prove better predictive quality.")
        x = np.linspace(-3, 3, 601)
        for name in ("relu", "leaky_relu", "gelu"):
            values, derivatives = activation_and_derivative(x, name, .1)
            axes[0].plot(x, values, color=COLORS[name], label=LABELS[name])
            axes[1].plot(x, derivatives, color=COLORS[name], label=LABELS[name])
            axes[2].plot(x, values, color=COLORS[name], label=LABELS[name])
        for ax in axes:
            ax.axhline(0, color="#8994a0", lw=.8)
            ax.axvline(0, color="#8994a0", lw=.8)
            ax.set_xlabel("Preactivation x")
        axes[0].set(title="Forward functions", ylabel="f(x)")
        axes[1].set(title="Local derivatives", ylabel="Backward slope")
        axes[2].set(title="Zoom around zero", xlim=(-1, 1), ylim=(-.25, .9), ylabel="f(x)")
        axes[0].legend(fontsize=9)
        return save_png(fig, self.output / "05_gelu_geometry.png")

    def temperature(self, animate=False):
        view = softmax_explorer(False)
        fig = view["figure"]
        if animate:
            t = np.r_[np.linspace(.15, 5, self.frames // 2), np.linspace(5, .15, self.frames - self.frames // 2)]
            return save_gif(fig, lambda i: view["update"](temperature=t[i]), self.frames,
                            self.output / "06_softmax_temperature.gif")
        return save_png(fig, self.output / "06_softmax_jacobian.png")

    def jacobian(self):
        from from_scratch import finite_difference_jacobian
        fig, axes = layout("H | Off-diagonal derivatives reveal competition", 3,
                           footer="Rows: output probabilities; columns: input logits. T=1. A larger logit redistributes mass across classes.\n"
                                  "The difference of Jacobians is sensitivity change, not the probability change itself; finite differences use epsilon=1e-5.")
        z = np.array([2., 1., -1.])
        j = softmax_jacobian(z)
        changed = softmax_jacobian(z + [0, 2, 0])
        numerical = finite_difference_jacobian(softmax, z)
        heatmap(axes[0], j, "Initial J: logits [2, 1, -1]", -.25, .25)
        heatmap(axes[1], changed, "New J: logits [2, 3, -1]", -.25, .25)
        heatmap(axes[2], changed - j, "New J minus initial J", -.25, .25)
        fig.text(.07, .84, f"Maximum analytical / central-difference disagreement: {np.max(np.abs(j-numerical)):.2e}", fontsize=10)
        return save_png(fig, self.output / "08_jacobian_change.png")

    def attention(self):
        fig, axes = layout("J | Attention normalizes each query over the keys", 4, size=(14, 5),
                           footer="Synthetic Q (3x2), K (4x2); scores=Q K-transpose / sqrt(2). Softmax acts along keys. Weight panels share [0,1].\n"
                                  "Weights are normalized compatibility scores, not semantic importance or factual correctness. No pretrained model or mask is used.")
        data = attention()
        scores = data["scores"]
        image = heatmap(axes[0], scores, "Scaled scores (signed)", -1.5, 1.5, row_prefix="q", col_prefix="k")
        fig.colorbar(image, ax=axes[0], shrink=.65)
        for ax, t in zip(axes[1:], (.5, 1., 2.)):
            image = heatmap(ax, attention(t)["weights"], f"Attention weights | T={t:g}", 0, 1,
                            cmap="Blues", row_prefix="q", col_prefix="k")
        fig.colorbar(image, ax=list(axes[1:]), shrink=.65)
        return save_png(fig, self.output / "10_attention.png")

    def boundaries(self, study):
        fig, axes = layout("I | Same inputs and weights, different learned probability maps", 4, size=(14, 5.3),
                           footer="Synthetic noisy moons; training points only; colors=P(y=1); black contour=P=0.5. Features standardized on training rows.\n"
                                  "Single split/seed and matched budget; this is not an activation ranking. Identity hidden layers collapse to an affine logit.")
        for ax, (name, result) in zip(axes, study["models"].items()):
            p = result["probability"]
            image = ax.contourf(study["grid_x"], study["grid_y"], p, levels=np.linspace(0, 1, 21), cmap="RdBu_r", vmin=0, vmax=1)
            ax.contour(study["grid_x"], study["grid_y"], p, levels=[.5], colors="black", linewidths=1)
            indices = study["train_idx"]
            ax.scatter(study["x"][indices, 0], study["x"][indices, 1], c=study["y"][indices],
                       cmap="RdBu_r", vmin=0, vmax=1, s=9, edgecolors="white", linewidths=.2)
            ax.set(title=f"{LABELS[name]}\nvalidation acc={result['metrics']['validation_accuracy']:.2f}",
                   xlabel="Feature 1 (standardized)", ylabel="Feature 2 (standardized)")
            ax.title.set_fontsize(10)
        fig.colorbar(image, ax=list(axes), shrink=.7, label="Predicted positive-class probability")
        return save_png(fig, self.output / "09_decision_boundaries.png")

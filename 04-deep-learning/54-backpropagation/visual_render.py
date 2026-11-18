"""Headless Matplotlib/Pillow exports and a self-contained Plotly explorer."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
import numpy as np
import plotly.graph_objects as go

from visual_math import (
    accumulation, activations, central_difference, clip_global_norm,
    gradient_products, loss_gradient, loss_surface, scalar_graph,
)

BLUE, ORANGE, GREEN, RED = "#2563eb", "#d97706", "#15803d", "#b91c1c"
INK, MUTED, PAPER = "#17243b", "#63738b", "#f5f8fc"
RATE_COLORS = ("#2563eb", "#7c3aed", "#15803d", "#a16207")


class Renderer:
    def __init__(self, output, *, frames=60, grid=91, dpi=100):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.frames, self.grid, self.dpi = frames, grid, dpi
        plt.rcParams.update({
            "font.family": "DejaVu Sans", "font.size": 11,
            "axes.titlesize": 13, "axes.labelsize": 11,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.labelcolor": INK, "text.color": INK,
            "axes.edgecolor": "#cbd5e1", "grid.color": "#dbe3ef",
            "figure.facecolor": PAPER, "axes.facecolor": "white",
            "savefig.facecolor": PAPER,
        })

    def figure(self, title, caption, rows=1, cols=1, figsize=(11, 6)):
        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        fig.suptitle(title, x=0.06, y=0.96, ha="left", weight="bold", fontsize=17)
        fig.text(0.06, 0.035, caption, fontsize=10, color=MUTED, va="bottom")
        fig.subplots_adjust(left=0.08, right=0.96, bottom=0.19, top=0.84,
                            wspace=0.30, hspace=0.45)
        return fig, axes

    def png(self, fig, name):
        path = self.output / name
        try:
            fig.savefig(path, dpi=self.dpi)
        finally:
            plt.close(fig)
        return path

    def gif(self, fig, update, name, *, fps=8):
        path = self.output / name
        try:
            animation = FuncAnimation(fig, update, frames=self.frames, interval=1000/fps,
                                      repeat=True, blit=False)
            animation.save(path, writer=PillowWriter(fps=fps), dpi=self.dpi)
        finally:
            plt.close(fig)
        return path

    def computational_graph(self):
        s = scalar_graph()
        eta = 0.2
        updated = np.array([s["w1"], s["b"], s["w2"]]) - eta * s["gradient"]
        end = scalar_graph(w1=updated[0], b=updated[1], w2=updated[2])
        fig, axes = self.figure(
            "01 / Follow values, then gradients, then an update",
            "Blue: forward values   |   Orange: backward sensitivities   |   Green: optimizer\n"
            "One synthetic observation. Weights stay frozen during the entire backward pass.",
            rows=2, figsize=(11, 7))
        graph, detail = axes[:, 0]
        graph.set_position([0.04, 0.37, 0.92, 0.45])
        detail.set_position([0.08, 0.13, 0.86, 0.23])
        nodes = {"x": (0.06, .68), "z": (.28, .68), "h": (.49, .68),
                 "prediction": (.70, .68), "loss": (.92, .68),
                 "w1": (.18, .20), "b": (.37, .20), "w2": (.64, .20), "y": (.90, .20)}
        links = [("x", "z"), ("z", "h"), ("h", "prediction"), ("prediction", "loss")]
        formulas = [
            f"Input x = {s['x']:.3f}; target y = {s['y']:.3f}. Parameters are fixed.",
            f"z = w1*x + b = {s['w1']:.3f}*{s['x']:.3f} + ({s['b']:.3f}) = {s['z']:.6f}",
            f"h = tanh(z) = tanh({s['z']:.3f}) = {s['h']:.6f}",
            f"prediction = w2*h = {s['w2']:.3f}*{s['h']:.6f} = {s['prediction']:.6f}",
            f"L = 0.5*(prediction-y)^2 = {s['loss']:.8f}",
            f"dL/dprediction = prediction-y = {s['local'][0]:.6f}",
            f"dL/dh = (dL/dprediction)*w2 = {s['local'][0]:.6f}*{s['w2']:.3f} = {s['dL_dh']:.6f}",
            f"dL/dz = (dL/dh)*(1-h^2) = {s['dL_dh']:.6f}*{s['local'][2]:.6f} = {s['dL_dz']:.6f}",
            f"dL/dw2 = (prediction-y)*h = {s['local'][0]:.6f}*{s['h']:.6f} = {s['gradient'][2]:.6f}",
            f"dL/dw1 = (dL/dz)*x = {s['dL_dz']:.6f}*{s['x']:.3f} = {s['gradient'][0]:.6f}",
            f"dL/db = (dL/dz)*1 = {s['gradient'][1]:.6f}",
            "Backward complete: gradients exist, parameters are still unchanged.",
            f"Optimizer: theta_new = theta_old - {eta:.2f}*gradient (all three parameters together).",
            f"Updated [w1, b, w2] = [{updated[0]:.6f}, {updated[1]:.6f}, {updated[2]:.6f}]\n"
            f"Fresh forward evaluation: loss {s['loss']:.8f} -> {end['loss']:.8f}",
        ]

        def update(frame):
            stage = min(13, frame * 14 // self.frames)
            graph.clear(); detail.clear()
            graph.set(xlim=(-.02, 1.02), ylim=(-.03, 1.02)); graph.axis("off"); detail.axis("off")
            current = end if stage == 13 else s
            color = BLUE if stage < 5 else ORANGE if stage < 12 else GREEN
            label = "FORWARD" if stage < 5 else "BACKWARD / NO UPDATES" if stage < 12 else "OPTIMIZER STEP"
            graph.text(0.01, .96, label, color=color, weight="bold", fontsize=13)
            reveal = {"x": 0, "z": 1, "h": 2, "prediction": 3, "loss": 4}
            for name, (nx, ny) in nodes.items():
                value = f"{current[name]:.5f}" if stage >= reveal.get(name, 0) else "pending"
                text = f"{'y_pred' if name == 'prediction' else name}\n{value}"
                graph.text(nx, ny, text, ha="center", va="center", fontsize=11,
                           bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=color if stage == 13 else "#b6c5d9", lw=1.5))
            for i, (a, b) in enumerate(links):
                start, finish = nodes[a], nodes[b]
                reverse = 5 <= stage <= 10
                if reverse:
                    start, finish = finish, start
                active = (stage < 5 and stage == i+1) or (stage == 5 and i == 3) or (stage == 6 and i == 2) or (stage == 7 and i == 1)
                graph.annotate("", finish, start, arrowprops=dict(arrowstyle="->", color=color if active else "#cbd5e1",
                                                                 lw=3 if active else 1.4, shrinkA=30, shrinkB=31))
            for parameter, node, active_stage in (("w1", "z", 9), ("b", "z", 10), ("w2", "prediction", 8), ("y", "loss", -1)):
                active = stage == active_stage
                start, finish = (nodes[node], nodes[parameter]) if active else (nodes[parameter], nodes[node])
                graph.annotate("", finish, start, arrowprops=dict(arrowstyle="->", lw=2.4 if active else 1,
                                                                 color=ORANGE if active else "#cbd5e1", shrinkA=27, shrinkB=28))
            detail.text(0, .94, f"Phase {stage+1:02d} / 14", color=color, weight="bold", va="top")
            detail.text(0, .65, formulas[stage], fontsize=12, va="top", linespacing=1.8)
            if 5 <= stage <= 11:
                detail.text(0, .12, f"Frozen parameters: w1={s['w1']:.3f}, b={s['b']:.3f}, w2={s['w2']:.3f}", color=MUTED)
        return [self.gif(fig, update, "computational_graph.gif", fps=3)]

    def chain_rule(self):
        s = scalar_graph()
        local = s["local"]
        expressions = ["dL/dw1", "(dL/dprediction) * (dprediction/dw1)",
                       "(dL/dprediction) * (dprediction/dh) * (dh/dw1)",
                       "(dL/dprediction) * (dprediction/dh) * (dh/dz) * (dz/dw1)"]
        labels = ["dL/dprediction", "dprediction/dh", "dh/dz", "dz/dw1"]
        fig, axes = self.figure("02 / Multiply local sensitivities along a path",
            "The graph and parameter values match computational_graph.gif. Signed derivatives are shown.\n"
            "These are local sensitivities; the optimizer is a separate operation.", figsize=(11, 6))
        ax = axes[0, 0]

        def update(frame):
            phase = min(7, frame * 8 // self.frames)
            ax.clear(); ax.axis("off")
            ax.text(0, .97, expressions[min(phase, 3)], fontsize=13, color=ORANGE, va="top")
            ax.text(0, .78, f"x={s['x']}, w1={s['w1']}, b={s['b']}, w2={s['w2']}, y={s['y']}", color=MUTED)
            count = max(0, phase - 3)
            for j in range(4):
                left = .02 + j * .245
                ax.text(left, .55, labels[j], fontsize=12)
                ax.text(left, .39, f"{local[j]:.7f}" if j < count else "?", fontsize=18,
                        color=ORANGE if j < count else MUTED)
                if j < 3:
                    ax.text(left + .205, .39, "×", fontsize=18, color=MUTED)
            if count:
                ax.text(.02, .13, f"Product of {count} revealed factor(s): {np.prod(local[:count]):.8f}", fontsize=15, color=ORANGE)
            if phase == 7:
                ax.text(.02, -.02, f"dL/dw1 = {s['gradient'][0]:.8f}", fontsize=16, color=GREEN, weight="bold")
        outputs = [self.gif(fig, update, "chain_rule.gif", fps=3)]
        delta = np.linspace(-1.5, 1.5, 301)
        actual = np.array([scalar_graph(w1=s["w1"]+v)["loss"] for v in delta]) - s["loss"]
        linear = s["gradient"][0] * delta
        fig, axes = self.figure("02 / A derivative predicts a nearby change, not a distant one",
            "Same biased scalar model and fixed parameters. Only w1 is perturbed; no retraining.\n"
            "Delta L ≈ (dL/dw1) Delta w1 is first order. The local panel uses a different horizontal scale.", cols=2)
        for ax, limit, title in zip(axes[0], (1.5, .04), ("Wide perturbation", "Local zoom")):
            ax.plot(delta, actual, color=BLUE, label="Actual change in loss")
            ax.plot(delta, linear, color=ORANGE, ls="--", label="Local linear approximation")
            ax.axhline(0, lw=.8, color=MUTED); ax.axvline(0, lw=.8, color=MUTED)
            ax.set(xlim=(-limit, limit), xlabel="Change in w1", ylabel="Change in half squared error", title=title)
            if limit < .1:
                mask = abs(delta) <= .05
                ax.set_ylim(min(actual[mask].min(), linear[mask].min()) * 1.2,
                            max(actual[mask].max(), linear[mask].max()) * 1.2)
            ax.legend(fontsize=9); ax.grid(alpha=.6)
        outputs.append(self.png(fig, "local_gradient.png"))
        return outputs

    def _contours(self, ax, x, y, traces):
        points = np.concatenate([t["theta"] for t in traces])
        low = np.minimum(points.min(axis=0) - .4, [-.2, -.2])
        high = np.maximum(points.max(axis=0) + .4, [2.2, 2.0])
        a, b = np.meshgrid(np.linspace(low[0], high[0], self.grid), np.linspace(low[1], high[1], self.grid))
        z = loss_surface(a, b, x, y)
        contour = ax.contourf(a, b, z, levels=22, cmap="Blues", alpha=.8)
        ax.set(xlabel="w1", ylabel="w2")
        return contour

    def loss_landscape(self, x, y, traces):
        main = traces[2]
        points = np.concatenate([t["theta"] for t in traces])
        w1 = np.linspace(min(-2.5, points[:, 0].min()-.2), max(2.5, points[:, 0].max()+.2), self.grid)
        w2 = np.linspace(min(-2.5, points[:, 1].min()-.2), max(2.5, points[:, 1].max()+.2), self.grid)
        a, b = np.meshgrid(w1, w2)
        z = loss_surface(a, b, x, y)
        fig = go.Figure(go.Surface(x=w1.tolist(), y=w2.tolist(), z=z.tolist(), colorscale="Blues",
                                  opacity=.83, name="Actual half-MSE", colorbar=dict(title="Half-MSE"),
                                  hovertemplate="w1=%{x:.5f}<br>w2=%{y:.5f}<br>Loss=%{z:.7f}<extra>Surface</extra>"))
        for color, trace in zip(RATE_COLORS, traces):
            positions, losses = trace["theta"], trace["loss"]
            rate = trace["learning_rate"]
            fig.add_trace(go.Scatter3d(x=positions[:, 0].tolist(), y=positions[:, 1].tolist(), z=losses.tolist(),
                mode="lines+markers", line=dict(color=color, width=6), marker=dict(size=2, color=color),
                name=f"eta={rate:g}", customdata=np.arange(len(losses)).tolist(),
                hovertemplate="Update %{customdata}<br>w1=%{x:.6f}<br>w2=%{y:.6f}<br>Loss=%{z:.8f}<extra>%{fullData.name}</extra>"))
            for index, symbol, label in ((0, "diamond", "initial"), (-1, "circle", "final")):
                step = len(losses)-1 if index == -1 else 0
                fig.add_trace(go.Scatter3d(x=[float(positions[index, 0])], y=[float(positions[index, 1])], z=[float(losses[index])],
                    mode="markers", marker=dict(size=6, symbol=symbol, color=color), showlegend=False,
                    name=f"eta={rate:g}, {label}, update {step}",
                    hovertemplate="w1=%{x:.6f}<br>w2=%{y:.6f}<br>Loss=%{z:.8f}<extra>%{fullData.name}</extra>"))
        fig.update_layout(title="Actual neural loss surface · rotate, zoom and hover",
            template="plotly_white", height=720, margin=dict(l=5, r=5, t=70, b=10),
            scene=dict(xaxis_title="w1", yaxis_title="w2", zaxis_title="Half mean squared error",
                       camera=dict(eye=dict(x=1.55, y=1.55, z=.9))),
            legend=dict(x=.01, y=.99), uirevision="day54")
        # No external scripts, fonts, MathJax or map tiles; toolbar controls stay local.
        chart = fig.to_html(include_plotlyjs=True, full_html=False, div_id="loss-chart",
                            config={"responsive": True, "displaylogo": False, "scrollZoom": True,
                                    "toImageButtonOptions": {"filename": "day54_loss_landscape"}})
        page = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Day 54 — Backpropagation Lab</title>
<style>body{margin:0;background:#f5f8fc;color:#17243b;font:16px system-ui,sans-serif}main{max-width:1280px;margin:auto;padding:24px}h1{margin:8px 0}p{max-width:1000px;line-height:1.5}button{padding:10px 18px;border:1px solid #aabbd0;border-radius:6px;background:white;color:#17243b;cursor:pointer}#loss-chart{min-height:600px}footer{color:#63738b}</style></head><body><main>
<h1>Backpropagation / the geometry of an actual loss</h1>
<p>Synthetic data: 41 evenly spaced x in [-1,1]; y = sin(3x) + 0.2x.<br>
Model: prediction = w2 tanh(w1 x). Loss = mean(0.5 (prediction - y)^2). No bias; two trainable parameters.</p>
<p>Drag to rotate; scroll to zoom; hover each trajectory for update number, weights and loss.
Click legend entries to hide/show rates; double-click to isolate one rate. Trajectory heights are evaluated directly at their parameters, not interpolated from the surface grid.</p>
<button id="reset-camera" type="button">Reset 3D camera</button>
""" + chart + """<footer><p>Same initialization [0.5, 0.8], dataset and iteration budget for every learning rate.
Colors distinguish rates, not quality. All observations are synthetic; author interpretation review required.
This HTML embeds Plotly and works offline. GitHub displays source; download the file and open it locally.</p></footer></main>
<script>document.getElementById('reset-camera').addEventListener('click',()=>Plotly.relayout('loss-chart',{'scene.camera':{eye:{x:1.55,y:1.55,z:0.9},center:{x:0,y:0,z:0},up:{x:0,y:0,z:1}}}));window.labReady=true;</script></body></html>"""
        path = self.output / "loss_landscape_3d.html"
        path.write_text(page, encoding="utf-8")
        outputs = [path]
        fig, axes = self.figure("03 / Gradient descent follows the computed loss",
            "Actual two-parameter neural objective; same initialization [0.5, 0.8]. No bias.\n"
            "Contours show half-MSE, not a decorative landscape. Each path point is an actual update.")
        ax = axes[0, 0]
        contour = self._contours(ax, x, y, traces)
        for color, trace in zip(RATE_COLORS, traces):
            ax.plot(*trace["theta"].T, color=color, lw=2, label=f"eta={trace['learning_rate']:g}")
            ax.scatter(*trace["theta"][-1], color=color, marker="s", s=35, zorder=5)
        ax.scatter(*main["theta"][0], color=INK, marker="*", s=140, label="Shared start", zorder=6)
        ax.legend(); fig.colorbar(contour, ax=ax, label="Half-MSE")
        outputs.append(self.png(fig, "loss_contours.png"))
        fig, axes = self.figure("03 / A real optimization trajectory, one update at a time",
            "Green: evaluated parameter updates; orange: local negative-gradient direction (scaled for visibility).\n"
            "Same two-parameter objective, eta=0.1. GIF samples checkpoints; no synthetic interpolation.")
        ax = axes[0, 0]
        contour = self._contours(ax, x, y, [main]); fig.colorbar(contour, ax=ax, label="Half-MSE")
        line, = ax.plot([], [], color=GREEN, lw=2.5)
        dot, = ax.plot([], [], "o", color=GREEN, ms=8)
        arrow = ax.quiver([0], [0], [0], [0], color=ORANGE, angles="xy", scale_units="xy", scale=1)
        status = ax.text(.02, .98, "", transform=ax.transAxes, va="top", fontsize=11,
                         bbox=dict(fc="white", ec="none", alpha=.9))
        def update(frame):
            k = round(frame * (len(main["loss"])-1) / (self.frames-1))
            position, gradient = main["theta"][k], main["gradient"][k]
            line.set_data(*main["theta"][:k+1].T); dot.set_data([position[0]], [position[1]])
            arrow.set_offsets([position]); arrow.set_UVC(*(-gradient * .7))
            status.set_text(f"Completed updates: {k}\n[w1,w2]=[{position[0]:.4f},{position[1]:.4f}]\nLoss={main['loss'][k]:.7f}")
        outputs.append(self.gif(fig, update, "gradient_descent.gif"))
        return outputs

    def gradient_flow(self):
        factors = (.5, 1., 1.5)
        colors = (ORANGE, BLUE, RED)
        curves = [gradient_products(f) for f in factors]
        caption = "Controlled scalar chain u_l = factor * u_(l-1), upstream derivative = 1. No trained MLP.\n" \
                  "Magnitude after k links = |upstream| * factor^k. Real network Jacobian products need not be monotone."
        def setup(title):
            fig, axes = self.figure(title, caption)
            ax = axes[0, 0]
            ax.set(xlim=(0, 16), ylim=(1e-5, 1e3), yscale="log",
                   xlabel="Number of links traversed backward (output toward input)",
                   ylabel="Absolute gradient magnitude (log scale)")
            ax.grid(which="both", alpha=.4)
            return fig, ax
        fig, ax = setup("04 / Repeated local derivatives can shrink or amplify a signal")
        for factor, curve, color in zip(factors, curves, colors):
            ax.plot(np.arange(17), curve, color=color, lw=2.5, label=f"factor={factor:g}; at input={curve[-1]:.6g}")
        ax.legend(); outputs = [self.png(fig, "gradient_magnitudes.png")]
        fig, ax = setup("04 / Watch the product grow across backward links")
        lines = [ax.plot([], [], color=c, lw=2.5, marker="o", ms=3, label=f"local derivative={f:g}")[0]
                 for f, c in zip(factors, colors)]
        ax.legend(loc="lower left")
        annotation = ax.text(.02, .96, "", transform=ax.transAxes, va="top", bbox=dict(fc="white", ec="none", alpha=.9))
        def update(frame):
            k = round(frame * 16 / (self.frames-1))
            for line, curve in zip(lines, curves):
                line.set_data(np.arange(k+1), curve[:k+1])
            annotation.set_text(f"Backward links traversed: {k}\n"
                f"0.5^{k} = {curves[0][k]:.6g}     1^{k} = 1     1.5^{k} = {curves[2][k]:.6g}")
        outputs.append(self.gif(fig, update, "gradient_flow.gif", fps=5))
        return outputs

    def activation_gradients(self):
        z = np.linspace(-7, 7, 501)
        values = activations(z)
        caption = "Shaded: |derivative| < 0.05 (an explanatory threshold, not an optimization guarantee).\n" \
                  "ReLU derivative at zero is defined as 0 here; no classical derivative exists at that point."
        def setup(title):
            fig, axes = self.figure(title, caption, rows=2, cols=3, figsize=(12, 7))
            for j, (name, (value, derivative)) in enumerate(values.items()):
                axes[0, j].plot(z, value, color=BLUE, lw=2)
                axes[1, j].plot(z, derivative, color=ORANGE, lw=2)
                for row in range(2):
                    ax = axes[row, j]
                    ax.fill_between(z, 0, 1, where=abs(derivative) < .05,
                                    transform=ax.get_xaxis_transform(), color=RED, alpha=.09)
                    ax.set(xlabel="Pre-activation z", ylabel="Activation" if row == 0 else "Local derivative",
                           title=name if row == 0 else "Derivative")
                    ax.grid(alpha=.5)
                if name == "ReLU":
                    axes[1, j].scatter([0], [0], c=ORANGE, s=35, zorder=4)
            return fig, axes
        fig, _ = setup("05 / Activation shape controls its local derivative")
        outputs = [self.png(fig, "activation_derivatives.png")]
        fig, axes = setup("05 / Move along the activation and inspect its sensitivity")
        dots = [[axes[row, col].plot([], [], "o", color=GREEN, ms=7)[0] for col in range(3)] for row in range(2)]
        labels = [axes[1, j].text(.03, .92, "", transform=axes[1, j].transAxes, va="top",
                                  bbox=dict(fc="white", ec="none", alpha=.9)) for j in range(3)]
        def update(frame):
            at = -7 + 14 * frame / (self.frames-1)
            current = activations(at)
            for col, name in enumerate(values):
                for row in range(2):
                    dots[row][col].set_data([at], [float(current[name][row])])
                labels[col].set_text(f"z={at:.2f}\nderivative={float(current[name][1]):.5f}")
        outputs.append(self.gif(fig, update, "activation_gradients.gif"))
        return outputs

    def learning_rates(self, x, y, traces):
        caption = "Same data, initial weights [0.5,0.8], objective and update budget for every rate.\n" \
                  "Colors identify rates, not stability. A correct gradient does not guarantee descent at a finite step."
        fig, axes = self.figure("06 / The learning rate scales the update, not the gradient", caption, cols=2)
        fig.subplots_adjust(left=.12)
        for color, trace in zip(RATE_COLORS, traces):
            losses = trace["loss"]
            sizes = np.linalg.norm(np.diff(trace["theta"], axis=0), axis=1)
            axes[0, 0].semilogy(np.arange(len(losses)), losses, color=color, lw=2, label=f"eta={trace['learning_rate']:g}")
            axes[0, 1].semilogy(np.arange(1, len(sizes)+1), np.maximum(sizes, 1e-16), color=color, lw=2)
        axes[0, 0].set(xlabel="Completed updates", ylabel="Half-MSE (log scale)")
        axes[0, 1].set(xlabel="Update number", ylabel="Euclidean parameter step norm (log scale)")
        axes[0, 0].legend(); outputs = [self.png(fig, "learning_rate_curves.png")]
        fig, axes = self.figure("06 / Same gradients at the start, different update sizes", caption, cols=2, figsize=(12, 6))
        fig.subplots_adjust(bottom=.28)
        left, right = axes[0]
        self._contours(left, x, y, traces)
        right.set(xlim=(0, traces[0]["requested_steps"]), ylim=(min(t["loss"].min() for t in traces)*.8,
                      max(t["loss"].max() for t in traces)*1.3), yscale="log",
                  xlabel="Completed updates", ylabel="Half-MSE (log scale)")
        lines = [left.plot([], [], color=c, lw=2, label=f"eta={t['learning_rate']:g}")[0] for c, t in zip(RATE_COLORS, traces)]
        curves = [right.plot([], [], color=c, lw=2)[0] for c in RATE_COLORS]
        left.legend(fontsize=9)
        caption_text = fig.text(.08, .14, "", fontsize=10, color=GREEN)
        def update(frame):
            step = round(frame * traces[0]["requested_steps"] / (self.frames-1))
            norms = []
            for line, curve, trace in zip(lines, curves, traces):
                k = min(step, len(trace["loss"])-1)
                line.set_data(*trace["theta"][:k+1].T)
                curve.set_data(np.arange(k+1), trace["loss"][:k+1])
                size = np.linalg.norm(trace["theta"][k] - trace["theta"][k-1]) if k else 0
                norms.append(f"eta={trace['learning_rate']:g}: {size:.4g}")
            caption_text.set_text(f"Completed updates: {step}   |   Last step norm: " + "   /   ".join(norms))
        outputs.append(self.gif(fig, update, "learning_rate_comparison.gif"))
        return outputs

    def gradient_check(self, x, y):
        theta = np.array([.7, 1.2])
        eps = np.logspace(-1, -12, 24)
        analytical = loss_gradient(theta, x, y)[1]
        numerical = np.array([central_difference(theta, x, y, e) for e in eps])
        absolute = abs(numerical - analytical)
        relative = absolute / np.maximum(abs(analytical), 1e-15)
        fig, axes = self.figure("07 / Smaller epsilon eventually stops helping",
            "Float64 central differences at [w1,w2]=[0.7,1.2]; both analytical gradients are nonzero.\n"
            "Relative error = absolute error / max(|analytical|,1e-15). Display floor: 1e-18; exact zeros marked there.",
            cols=3, figsize=(12, 6))
        for j, color in enumerate((BLUE, ORANGE)):
            axes[0, 0].semilogx(eps, numerical[:, j], "o-", ms=3, color=color, label=f"w{j+1}: numerical")
            axes[0, 0].axhline(analytical[j], color=color, ls="--", label=f"w{j+1}: analytical")
            axes[0, 1].loglog(eps, np.maximum(absolute[:, j], 1e-18), "o-", color=color, ms=3, label=f"w{j+1}")
            axes[0, 2].loglog(eps, np.maximum(relative[:, j], 1e-18), "o-", color=color, ms=3, label=f"w{j+1}")
        for ax, label in zip(axes[0], ("Gradient component", "Absolute gradient error", "Relative gradient error")):
            ax.set(xlabel="Perturbation epsilon (log scale)", ylabel=label)
            ax.invert_xaxis(); ax.legend(fontsize=8); ax.grid(alpha=.5)
        axes[0, 1].text(.04, .96, "Large epsilon:\ntruncation", transform=axes[0, 1].transAxes, va="top", fontsize=10)
        return [self.png(fig, "gradient_check.png")]

    def training_engineering(self, x, y):
        _, gradient = loss_gradient((.5, .8), x, y)
        norm = np.linalg.norm(gradient)
        threshold = .6 * norm
        clipped = clip_global_norm(gradient, threshold)
        batch = accumulation(x, y)
        fig, axes = self.figure("08 / Bound a gradient norm; combine batch contributions correctly",
            "Both panels use actual gradients of the same two-parameter model at [0.5,0.8].\n"
            "Clipping changes the update direction's magnitude. Accumulation freezes weights until the final update.",
            cols=2, figsize=(12, 7))
        fig.subplots_adjust(bottom=.28)
        left, right = axes[0]
        left.add_patch(Circle((0, 0), threshold, color=GREEN, fill=False, ls="--", lw=2))
        for vector, color, label in ((gradient, ORANGE, "Before clipping"), (clipped, GREEN, "After clipping")):
            left.quiver(0, 0, *vector, angles="xy", scale_units="xy", scale=1, color=color, label=label, width=.012)
        limit = norm * 1.25
        left.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal", xlabel="dL/dw1", ylabel="dL/dw2",
                 title="Global norm clipping preserves direction")
        left.axhline(0, color=MUTED, lw=.7); left.axvline(0, color=MUTED, lw=.7)
        left.text(.02, .98, f"Norm: {norm:.6f} -> {np.linalg.norm(clipped):.6f}\nThreshold: {threshold:.6f}",
                  transform=left.transAxes, va="top", fontsize=10)
        left.legend(loc="lower right", fontsize=9)
        values = np.array([*batch["pieces"], batch["sum"], batch["unweighted"], batch["weighted"], batch["full"]])
        positions = np.arange(len(values))
        right.bar(positions-.16, values[:, 0], .32, color=BLUE, label="dL/dw1")
        right.bar(positions+.16, values[:, 1], .32, color=ORANGE, label="dL/dw2")
        right.set_xticks(positions, ["B1\nn=13", "B2\nn=28", "Sum", "Equal\nmean", "Weighted\nmean", "Full\nn=41"], fontsize=9)
        right.set(ylabel="Mean-loss gradient component", title="Two unequal microbatches: effective N = 41")
        right.axhline(0, color=MUTED, lw=.7); right.legend(fontsize=9)
        right.text(0, -.20, "Correct: (13/41) g1 + (28/41) g2\nSum and equal mean generally change the gradient scale.",
                   transform=right.transAxes, fontsize=10, color=GREEN)
        return [self.png(fig, "training_engineering.png")]

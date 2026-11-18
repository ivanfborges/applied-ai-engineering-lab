"""Bounded Matplotlib/Pillow exports and self-contained Plotly teaching pages."""

from pathlib import Path
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
import numpy as np

from from_scratch import predict_proba
from visual_math import (
    ACTIVATIONS, XOR_X, XOR_Y, activation, boundary_segment, fixed_forward,
    grid_points, loss_slice, perceptron_trace, relu_approximation,
    relu_features, target_function,
)

BLUE, ORANGE, INK = "#167c80", "#d46c35", "#23364d"
CLASS_COLORS = (BLUE, ORANGE)
PROBABILITY_CMAP = LinearSegmentedColormap.from_list("xor_classes", [BLUE, "#f6f4ed", ORANGE])


class Renderer:
    def __init__(self, output, *, quick=False, seed=42):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.quick, self.seed = quick, seed
        self.size = 41 if quick else 71
        self.dpi = 90 if quick else 110
        self.frames = 20 if quick else 40
        self.files = []
        self.results = {}
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                             "axes.spines.top": False, "axes.spines.right": False,
                             "axes.titleweight": "bold", "axes.labelcolor": INK,
                             "text.color": INK, "figure.facecolor": "#fafbf9",
                             "axes.facecolor": "#fafbf9", "savefig.facecolor": "#fafbf9"})

    def path(self, name):
        path = self.output / name
        self.files.append(path)
        return path

    def png(self, fig, name):
        fig.savefig(self.path(name), dpi=self.dpi)

    def gif(self, fig, draw, states, name, *, fps=5):
        """Each frame is drawn from its supplied state; no parameter morphing."""
        writer = PillowWriter(fps=fps)
        with writer.saving(fig, self.path(name), dpi=self.dpi):
            for state in states:
                draw(state)
                writer.grab_frame()

    @staticmethod
    def points(ax, X, y, *, annotate=False):
        for label, marker in ((0, "o"), (1, "^")):
            subset = X[np.asarray(y) == label]
            ax.scatter(subset[:, 0], subset[:, 1], c=CLASS_COLORS[label], marker=marker,
                       s=65 if annotate else 25, edgecolor="white", linewidth=0.6,
                       label=f"class {label}", zorder=5)
        if annotate:
            for index, row in enumerate(X):
                ax.annotate(str(tuple(int(v) for v in row)), row, xytext=(5, 8), textcoords="offset points", fontsize=8)

    @staticmethod
    def axes2d(ax, title, *, low=-0.5, high=1.5, xlabel="$x_1$", ylabel="$x_2$"):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(low, high), ylim=(low, high))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.16)

    @staticmethod
    def probability(ax, xx, yy, scores, *, threshold=0.5):
        scores = np.asarray(scores).reshape(xx.shape)
        if not np.isfinite(scores).all():
            raise ValueError("non-finite plotted model output")
        artist = ax.pcolormesh(xx, yy, scores, cmap=PROBABILITY_CMAP, vmin=0, vmax=1,
                               shading="auto", alpha=0.85)
        if scores.min() < threshold < scores.max():
            ax.contour(xx, yy, scores, levels=[threshold], colors=INK, linewidths=1.5)
        return artist

    def html(self, fig, name, heading, explanation, *, controls="", script=""):
        try:
            import plotly.io as pio
        except ImportError as exc:
            raise RuntimeError('Plotly is required for HTML. From the repository root run: python -m pip install -e ".[dev]"') from exc
        fig.update_layout(template="plotly_white", font={"family": "Arial", "color": INK},
                          margin={"l": 30, "r": 30, "t": 65, "b": 45}, height=650,
                          paper_bgcolor="#fafbf9", uirevision="keep-camera")
        chart = pio.to_html(fig, full_html=False, include_plotlyjs=True, include_mathjax=False,
                            div_id="lab-chart", auto_play=False,
                            config={"responsive": True, "displaylogo": False},
                            post_script=script + "\nwindow.labReady=true;")
        page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{heading}</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;color:{INK};background:#fafbf9;margin:0 auto;padding:24px;max-width:1500px}}
h1{{font-size:27px;margin-bottom:10px}}p{{max-width:1100px}}.controls{{display:flex;flex-wrap:wrap;gap:18px;align-items:center;padding:14px;background:#edf2ef}}
label{{display:flex;gap:8px;align-items:center}}button,select{{font:inherit;padding:6px}}input{{width:130px}}output{{min-width:40px}}
.note{{font-size:14px}}#status{{font-variant-numeric:tabular-nums}}:focus-visible{{outline:3px solid #996500}}</style></head>
<body><h1>{heading}</h1><p>{explanation}</p><div class="controls">{controls}</div>{chart}
<p class="note">Synthetic educational demonstration. Drag a 3D panel to rotate; scroll to zoom. No server or network is needed. Author interpretation review required.</p>
</body></html>'''
        self.path(name).write_text(page, encoding="utf-8")

    @staticmethod
    def plotly():
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError as exc:
            raise RuntimeError('Install Plotly from the repository root: python -m pip install -e ".[dev]"') from exc
        return go, make_subplots

    def neuron(self):
        fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
        ax.set(xlim=(0, 11), ylim=(0, 4)); ax.axis("off")
        nodes = [(1, 3, "$x_1$"), (1, 1, "$x_2$"), (4, 2, r"$\Sigma$"),
                 (7, 2, r"$\phi(z)$"), (10, 2, "$a$")]
        for x, y, label in nodes:
            ax.text(x, y, label, ha="center", va="center", fontsize=19,
                    bbox={"boxstyle": "round,pad=0.5", "fc": "#e4efec", "ec": BLUE})
        for start, end, label in [((1.5, 3), (3.5, 2.15), "$w_1 x_1$"),
                                   ((1.5, 1), (3.5, 1.85), "$w_2 x_2$"),
                                   ((4.5, 2), (6.4, 2), "$z=w^Tx+b$"),
                                   ((7.7, 2), (9.5, 2), r"$a=\phi(z)$")]:
            ax.annotate("", end, start, arrowprops={"arrowstyle": "->", "color": INK, "lw": 2})
            ax.text((start[0]+end[0])/2, (start[1]+end[1])/2+0.35, label, ha="center")
        ax.annotate("bias b", (4, 2.4), (4, 3.5), ha="center", arrowprops={"arrowstyle": "->"})
        ax.set_title("01 | Inside a neuron: affine score, then activation")
        self.png(fig, "01_neuron_diagram.png"); plt.close(fig)

        go, subplots = self.plotly()
        xx, yy, _ = grid_points(-1.5, 1.5, self.size)
        z = xx + yy
        fig = subplots(rows=1, cols=3, specs=[[{"type": "surface"}, {"type": "surface"}, {"type": "xy"}]],
                       subplot_titles=("Affine pre-activation z", "Activation a = sigmoid(z)", "Input-space z = 0 boundary"))
        fig.add_trace(go.Surface(x=xx, y=yy, z=z, colorscale="Tealrose", cmin=-8, cmax=8, showscale=False), row=1, col=1)
        fig.add_trace(go.Surface(x=xx, y=yy, z=activation("sigmoid", z), colorscale="Tealrose", cmin=-8, cmax=8, showscale=False), row=1, col=2)
        fig.add_trace(go.Contour(x=xx[0], y=yy[:, 0], z=z, colorscale="Tealrose", zmin=-8, zmax=8,
                                 contours={"coloring": "heatmap", "showlines": False}, showscale=False), row=1, col=3)
        segment = boundary_segment([1, 1], 0, -1.5, 1.5)
        fig.add_trace(go.Scatter(x=segment[:, 0], y=segment[:, 1], mode="lines", line={"color": INK, "width": 4}, name="z = 0"), row=1, col=3)
        for scene in ("scene", "scene2"):
            fig.update_layout(**{scene: {"xaxis": {"title": "x1", "range": [-1.5, 1.5]},
                                        "yaxis": {"title": "x2", "range": [-1.5, 1.5]},
                                        "zaxis": {"title": "z" if scene == "scene" else "a", "range": [-8, 8]},
                                        "aspectmode": "cube"}})
        fig.update_xaxes(title="x1", range=[-1.5, 1.5], constrain="domain", row=1, col=3)
        fig.update_yaxes(title="x2", range=[-1.5, 1.5], scaleanchor="x", constrain="domain", row=1, col=3)
        controls = '''<label for="preset">Preset <select id="preset"><option value="1,1,0">w=(1,1), b=0</option><option value="2,-1,0">w=(2,-1), b=0</option><option value="1,1,-1">w=(1,1), b=-1</option></select></label>
<label for="w1">w1 <input id="w1" type="range" min="-2" max="2" step="0.1" value="1"><output id="o-w1">1</output></label>
<label for="w2">w2 <input id="w2" type="range" min="-2" max="2" step="0.1" value="1"><output id="o-w2">1</output></label>
<label for="bias">b <input id="bias" type="range" min="-2" max="2" step="0.1" value="0"><output id="o-bias">0</output></label>
<label for="activation">Activation <select id="activation"><option>sigmoid</option><option>tanh</option><option>relu</option><option>identity</option></select></label><output id="status" aria-live="polite"></output>'''
        script = '''
const chart=document.getElementById('lab-chart');
const axis=chart.data[2].x;
function activate(z,kind){return kind==='sigmoid'?1/(1+Math.exp(-z)):kind==='tanh'?Math.tanh(z):kind==='relu'?Math.max(0,z):z;}
function updateNeuron(){
  const w1=+document.getElementById('w1').value,w2=+document.getElementById('w2').value,b=+document.getElementById('bias').value;
  const kind=document.getElementById('activation').value;
  const z=axis.map(y=>axis.map(x=>w1*x+w2*y+b)),a=z.map(row=>row.map(v=>activate(v,kind)));
  let pts=[]; if(Math.abs(w2)>1e-12){for(const x of [-1.5,1.5]){pts.push([x,-(w1*x+b)/w2]);}}
  if(Math.abs(w1)>1e-12){for(const y of [-1.5,1.5]){pts.push([-(w2*y+b)/w1,y]);}}
  pts=pts.filter(p=>p.every(v=>v>=-1.50000001&&v<=1.50000001)).slice(0,2);
  Plotly.restyle(chart,{z:[z,a,z]},[0,1,2]);
  Plotly.restyle(chart,{x:[pts.map(p=>p[0])],y:[pts.map(p=>p[1])]},[3]);
  const range=kind==='sigmoid'?[0,1]:kind==='tanh'?[-1,1]:kind==='relu'?[0,8]:[-8,8];
  Plotly.relayout(chart,{'scene2.zaxis.range':range,'annotations[1].text':'Activation a = '+kind+'(z)'});
  for(const id of ['w1','w2','bias'])document.getElementById('o-'+id).textContent=(+document.getElementById(id).value).toFixed(1);
  document.getElementById('status').textContent=(w1===0&&w2===0)?(b===0?'z=0 everywhere; no unique boundary':'constant score; boundary empty'):'Boundary: '+w1.toFixed(1)+' x1 + '+w2.toFixed(1)+' x2 + '+b.toFixed(1)+' = 0';
  window.neuronState={w1,w2,b,kind,z,a};
}
for(const id of ['w1','w2','bias','activation'])document.getElementById(id).addEventListener('input',updateNeuron);
document.getElementById('preset').addEventListener('change',e=>{e.target.value.split(',').forEach((v,i)=>document.getElementById(['w1','w2','bias'][i]).value=v);updateNeuron();});
updateNeuron();'''
        script = script.replace("const axis=chart.data[2].x;", f"const axis={json.dumps(xx[0].tolist())};")
        self.html(fig, "01_neuron_controls.html", "01 | Weights tilt; bias translates",
                  "Change weights, bias and activation. Both surfaces use the same affine score. The right panel shows its zero contour (the sigmoid 0.5 boundary). Other activations do not automatically define a classifier. A zero weight vector has no unique separating line. Fixed input ranges; activation height ranges follow their mathematical scale.",
                  controls=controls, script=script)

    def perceptron(self):
        trace = perceptron_trace(self.seed)
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), layout="constrained")
        def draw(state):
            for ax in axes: ax.clear()
            self.points(axes[0], trace["X"], (trace["y"]+1)//2)
            self.axes2d(axes[0], "Actual perceptron boundary", low=-1.1, high=1.1)
            segment = boundary_segment(state["weights"], state["bias"], -1.1, 1.1)
            if len(segment) == 2:
                axes[0].plot(*segment.T, color=INK, lw=2)
            else:
                axes[0].text(0.02, 0.02, "No unique visible boundary", transform=axes[0].transAxes)
            axes[0].legend(loc="upper left")
            axes[1].axis("off")
            w = state["weights"]
            axes[1].text(0, 0.94, "Mistake-driven learning, not backpropagation", fontsize=11, weight="bold", va="top", transform=axes[1].transAxes)
            axes[1].text(0, 0.8, f"epoch {state['epoch']} | cumulative update {state['update']}\nw = [{w[0]:.3f}, {w[1]:.3f}]   b = {state['bias']:.3f}\n\nCurrent training errors: {state['training_errors']} / {len(trace['X'])}\nNonpositive margins (including ties): {state['margin_violations']}", linespacing=1.7, va="top", transform=axes[1].transAxes)
            axes[1].text(0, 0.27, "$y(w^Tx+b)\\leq0:\\quad w\\leftarrow w+yx,\\ b\\leftarrow b+y$\nPrediction: +1 iff score > 0. Learning rate = 1.\nEach displayed state follows a real update; frames may be held.", linespacing=1.8, va="top", transform=axes[1].transAxes, fontsize=9)
            fig.suptitle("02 | Perceptron: moving a line after a margin violation", fontsize=14)
        states = trace["states"]
        displayed = [s for s in states for _ in range(2)] + [states[-1]] * 4
        self.gif(fig, draw, displayed, "02_perceptron_learning.gif", fps=4)
        draw(states[-1]); self.png(fig, "02_perceptron_final.png"); plt.close(fig)
        self.results["perceptron"] = {"samples": len(trace["X"]), "updates": states[-1]["update"],
                                      "final_errors": states[-1]["training_errors"], "zero_update_epoch": trace["certified"]}

    def xor(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from example import make_noisy_xor
        X, y = make_noisy_xor(seed=self.seed)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=self.seed)
        estimators = [LogisticRegression(max_iter=1000),
                      MLPClassifier(hidden_layer_sizes=(8, 8), activation="relu", solver="lbfgs",
                                    alpha=0.001, max_iter=1000, random_state=self.seed)]
        xx, yy, grid = grid_points(size=self.size)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.9), layout="constrained")
        results = []
        for ax, estimator, title in zip(axes[:2], estimators, ("Affine logit", "ReLU MLP (8,8)")):
            model = make_pipeline(StandardScaler(), estimator)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(Xtr, ytr)
            scores = model.predict_proba(grid)[:, 1]
            self.probability(ax, xx, yy, scores)
            self.points(ax, Xte, yte)
            self.points(ax, XOR_X, XOR_Y, annotate=True)
            accuracy = model.score(Xte, yte)
            self.axes2d(ax, f"{title}\nholdout accuracy = {accuracy:.3f}")
            results.append({"model": title, "test_accuracy": accuracy, "warnings": [str(w.message) for w in caught]})
        self.points(axes[2], XOR_X, XOR_Y, annotate=True)
        axes[2].plot([0, 1], [0, 1], color=BLUE, lw=2)
        axes[2].plot([0, 1], [1, 0], color=ORANGE, lw=2)
        axes[2].scatter([0.5], [0.5], marker="x", c=INK, s=100)
        self.axes2d(axes[2], "Both class convex hulls\ncontain the same midpoint")
        axes[2].text(0.5, -0.36, "Intersecting hulls prevent linear separation", ha="center", fontsize=9)
        colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=PROBABILITY_CMAP), ax=axes[:2], shrink=0.7)
        colorbar.set_label("Model class-1 score")
        fig.suptitle("03 | XOR: original corners are large markers; small markers are synthetic holdout samples", fontsize=12)
        self.png(fig, "03_xor_comparison.png"); plt.close(fig)
        self.results["xor"] = {"samples": 800, "noise_sd": 0.12, "train": 600, "test": 200, "models": results,
                               "limitation": "Scores between corners are model extensions; no intrinsic continuous XOR target is asserted."}

    def activations(self):
        z = np.linspace(-6, 6, 601)
        colors = [INK, "#8c619e", BLUE, "#a06938", ORANGE, "#5772b1"]
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
        def draw(value):
            for ax, name, color in zip(axes.ravel(), ACTIVATIONS, colors):
                ax.clear()
                deriv = activation(name, z, derivative=True)
                values = activation(name, z)
                if name == "step": values = np.where(z == 0, np.nan, values)
                ax.plot(z, values, c=color, lw=2, label="activation")
                ax.plot(z, deriv, c=INK, ls="--", lw=1.4, label="derivative")
                if name in ("sigmoid", "tanh"):
                    ax.axvspan(-6, -3, color=BLUE, alpha=0.08)
                    ax.axvspan(3, 6, color=BLUE, alpha=0.08)
                ax.axvline(value, c="gray", lw=0.8)
                a, d = float(activation(name, value)), float(activation(name, value, derivative=True))
                ax.scatter([value], [a], color=color, s=30, zorder=4)
                if np.isfinite(d): ax.scatter([value], [d], c=INK, marker="x", s=35)
                if name in ("relu", "step"):
                    ax.scatter([0, 0] if name == "relu" else [0], [0, 1] if name == "relu" else [0],
                               facecolors="#fafbf9", edgecolors=INK, zorder=6)
                    ax.text(0.02, 0.04, "Derivative undefined at z = 0", transform=ax.transAxes, fontsize=8)
                    if name == "step":
                        ax.scatter([0], [1], facecolors="#fafbf9", edgecolors=color, zorder=6)
                        ax.scatter([0], [0], c=color, s=15, zorder=7)
                ax.set(title=f"{name} | a={a:.2f}, derivative={'undefined' if not np.isfinite(d) else f'{d:.2f}'}",
                       xlabel="pre-activation z", ylabel="activation / derivative", xlim=(-6, 6),
                       ylim=(-6.5, 6.5) if name == "identity" else (-1.3, 6.5) if name in ("relu", "gelu") else (-1.3, 2.2))
                ax.grid(alpha=0.15); ax.legend(fontsize=8, loc="upper left")
            fig.suptitle("04 | Moving pre-activation; shaded sigmoid/tanh tails indicate small derivatives; each panel keeps fixed axes", fontsize=12)
        states = np.linspace(-6, 6, self.frames + 1)
        self.gif(fig, draw, states, "04_activation_flow.gif", fps=5)
        draw(0.0); self.png(fig, "04_activation_derivatives.png"); plt.close(fig)

    def features(self):
        hidden = relu_features(XOR_X)
        xx, yy, grid = grid_points(size=self.size)
        score = relu_features(grid) @ np.array([1, -2])
        fig, axes = plt.subplots(1, 3, figsize=(12, 5), layout="constrained")
        def draw(t):
            for ax in axes: ax.clear()
            self.points(axes[0], XOR_X, XOR_Y, annotate=True)
            self.axes2d(axes[0], "Original binary inputs", low=-0.4, high=2.4)
            transitional = (1-t)*XOR_X + t*hidden
            self.points(axes[1], transitional, XOR_Y, annotate=False)
            for source, destination in zip(XOR_X, hidden):
                axes[1].annotate("", destination, source, arrowprops={"arrowstyle": "->", "color": "gray", "alpha": 0.35})
            if t == 1:
                line = np.linspace(-0.4, 2.4, 100)
                axes[1].plot(line, (line-0.5)/2, c=INK, lw=2)
            axes[1].text(0.02, 0.02, f"display transition t={t:.2f}\nConceptual morph, not a learned layer", transform=axes[1].transAxes, fontsize=8)
            self.axes2d(axes[1], "Hidden features at t=1\n(h1,h2) = (ReLU(s),ReLU(s-1))", low=-0.4, high=2.4,
                        xlabel="h1 at t=1", ylabel="h2 at t=1")
            axes[2].pcolormesh(xx, yy, (score > 0.5).reshape(xx.shape), cmap=PROBABILITY_CMAP, vmin=0, vmax=1, shading="auto", alpha=0.6)
            axes[2].contour(xx, yy, score.reshape(xx.shape), levels=[0.5], colors=INK)
            self.points(axes[2], XOR_X, XOR_Y, annotate=True)
            self.axes2d(axes[2], "Pullback to input space\nh1 - 2 h2 > 0.5")
            fig.suptitle("05 | Nonlinear representation, then linear separation; only binary corners define XOR labels", fontsize=12)
        states = list(np.linspace(0, 1, self.frames)) + [1.0]*6
        self.gif(fig, draw, states, "05_hidden_representation.gif", fps=6)
        draw(1.0); self.png(fig, "05_hidden_representation.png"); plt.close(fig)

        go, _ = self.plotly()
        axis = np.linspace(0, 1, self.size)
        xx, yy = np.meshgrid(axis, axis)
        fig = go.Figure()
        fig.add_trace(go.Surface(x=xx, y=yy, z=xx*yy, colorscale="Greys", opacity=0.45, showscale=False, name="h=x1*x2"))
        fig.add_trace(go.Surface(x=xx, y=yy, z=(xx+yy-0.5)/2, colorscale=[[0, BLUE], [1, BLUE]], opacity=0.55,
                                 showscale=False, name="separating plane"))
        for label, symbol in ((0, "circle"), (1, "diamond")):
            points = XOR_X[XOR_Y == label]
            fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points.prod(axis=1), mode="markers+text",
                                      text=[str(tuple(int(v) for v in (*p, np.prod(p)))) for p in points], textposition="top center",
                                      marker={"color": CLASS_COLORS[label], "size": 7, "symbol": symbol}, name=f"class {label}"))
        fig.update_layout(scene={"xaxis": {"title": "x1", "range": [0, 1]}, "yaxis": {"title": "x2", "range": [0, 1]},
                                 "zaxis": {"title": "h = x1*x2", "range": [-0.3, 1]}, "aspectmode": "cube",
                                 "camera": {"eye": {"x": 1.8, "y": -1.8, "z": 1.2}}})
        controls = '<label for="plane">Separating plane <input id="plane" type="checkbox" checked></label><label for="lifting">Feature surface <input id="lifting" type="checkbox" checked></label>'
        script = "document.getElementById('plane').onchange=e=>Plotly.restyle('lab-chart',{visible:e.target.checked},[1]);document.getElementById('lifting').onchange=e=>Plotly.restyle('lab-chart',{visible:e.target.checked},[0]);"
        self.html(fig, "05_feature_lifting.html", "05 | Lift XOR into three dimensions",
                  "Gray surface: h=x1*x2. Teal plane: h=(x1+x2-0.5)/2, equivalent to x1+x2-2h=0.5. Class 1 lies below the plane at the four binary corners. This is a fixed feature construction, not a trained network or a target definition between corners.", controls=controls, script=script)

    def forward(self):
        model = fixed_forward(self.seed)
        fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.5, 1]}, layout="constrained")
        widths = (2, 4, 4, 1)
        positions = [np.column_stack((np.full(width, layer), np.linspace(-1, 1, width) if width > 1 else [0])) for layer, width in enumerate(widths)]
        def draw(active):
            for ax in axes: ax.clear()
            ax = axes[0]; ax.axis("off"); ax.set(xlim=(-0.4, 3.4), ylim=(-1.65, 1.6))
            for layer, w in enumerate(model["weights"]):
                for i in range(w.shape[0]):
                    for j in range(w.shape[1]):
                        p, q = positions[layer][i], positions[layer+1][j]
                        ax.plot([p[0], q[0]], [p[1], q[1]], color=BLUE if w[i, j] >= 0 else ORANGE,
                                lw=0.4+1.4*abs(w[i, j]), alpha=0.75 if layer+1 == active else 0.2, zorder=1)
            for layer, points in enumerate(positions):
                colors = {"c": model["values"][layer], "cmap": "coolwarm", "vmin": -1, "vmax": 1} if layer <= active else {"color": "#dde3e6"}
                ax.scatter(points[:, 0], points[:, 1], **colors, s=650, edgecolor=INK, zorder=3)
                for index, point in enumerate(points):
                    value = f"{model['values'][layer][index]:.2f}" if layer <= active else "?"
                    ax.text(*point, value, ha="center", va="center", fontsize=9, zorder=4)
                ax.text(layer, -1.4, "input" if layer == 0 else "sigmoid" if layer == 3 else "tanh", ha="center")
            ax.set_title(f"06 | Fixed, untrained 2 → 4 → 4 → 1 ({model['parameters']} parameters)")
            ax.legend(handles=[Line2D([0], [0], color=BLUE, label="positive weight"),
                               Line2D([0], [0], color=ORANGE, label="negative weight")], loc="upper left", fontsize=8)
            axes[1].axis("off")
            if active == 0:
                text = "Input x = [0, 1]\nWeights and biases are fixed random values.\nNothing is trained in this diagram."
            else:
                index = active-1
                text = (f"Layer {active}: a_prev @ W + b\nW shape: {model['weights'][index].shape}\n"
                        f"bias = {np.array2string(model['biases'][index], precision=2)}\n"
                        f"z = {np.array2string(model['logits'][index], precision=3)}\n"
                        f"a = {np.array2string(model['values'][active], precision=3)}")
            axes[1].text(0, 0.85, text, va="top", fontsize=11, linespacing=1.9)
            axes[1].text(0, 0.08, "Node color encodes activation (range -1 to 1).\nEdge width = 0.4 + 1.4 |weight|.\nEvery displayed bias contributes to z.\n\nWidth example: 2→4→1 has 17 parameters;\n2→8→1 has 33. More parameters ≠ better fit.", fontsize=10, linespacing=1.5)
        self.gif(fig, draw, [i for i in range(4) for _ in range(5)], "06_forward_propagation.gif", fps=4)
        draw(3); self.png(fig, "06_forward_propagation.png"); plt.close(fig)
        self.results["forward"] = {"parameters": model["parameters"], "output": model["values"][-1].tolist(),
                                   "status": "fixed random parameters, not trained"}

    def training(self, trace):
        xx, yy, grid = grid_points(size=self.size)
        fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), layout="constrained")
        losses = trace["losses"]
        def draw(state):
            for ax in axes: ax.clear()
            self.probability(axes[0], xx, yy, predict_proba(grid, state["parameters"]))
            self.points(axes[0], XOR_X, XOR_Y, annotate=True)
            self.axes2d(axes[0], "Actual model scores and p=0.5 contour")
            axes[0].legend(loc="lower right", fontsize=8)
            epoch = state["epoch"]
            axes[1].plot(np.arange(epoch+1)+1, losses[:epoch+1], c=BLUE)
            axes[1].scatter([epoch+1], [state["loss"]], color=BLUE, s=20)
            axes[1].set(xlabel="gradient-descent updates + 1 (log scale)", ylabel="mean training BCE (log scale)",
                        xscale="log", yscale="log", xlim=(1, len(losses)),
                        ylim=(max(losses.min()*0.7, 1e-8), max(losses.max()*1.3, 1)))
            axes[1].grid(alpha=0.15)
            axes[2].axis("off")
            p = state["probabilities"]
            accuracy = np.mean((p >= 0.5) == XOR_Y)
            weights = state["parameters"]
            axes[2].text(0, 0.95, f"Update: {epoch:,}\nActual BCE: {state['loss']:.6f}\nTraining accuracy: {accuracy:.2f}\n\nGradient norm: {state['gradient_norm']:.4f}\nPrevious step norm: {state['previous_step_norm']:.4f}\nLearning rate: {trace['learning_rate']}\nW2[0,0]={weights['W2'][0,0]:.3f}, b2={weights['b2'][0]:.3f}\n\np at (00,01,10,11):\n{np.array2string(p, precision=3)}", va="top", linespacing=1.6, fontsize=9)
            axes[2].text(0, 0.02, "Backprop computes gradients.\nGradient descent updates parameters.\nGeometrically spaced real checkpoints;\nplayback speed is not training time.\nAll four points are training data.", fontsize=9, linespacing=1.5)
            fig.suptitle("07 | NumPy 2 → 8 → 1: tanh hidden layer, sigmoid output, stable BCE", fontsize=13)
        self.gif(fig, draw, trace["states"]+[trace["states"][-1]]*5, "07_mlp_training.gif", fps=5)
        draw(trace["states"][-1]); self.png(fig, "07_mlp_training_final.png"); plt.close(fig)

    def landscape(self, trace):
        go, _ = self.plotly()
        parameters = trace["states"][-1]["parameters"]
        da, db, losses = loss_slice(parameters, size=self.size)
        fig = go.Figure(go.Surface(x=da, y=db, z=losses, colorscale="Viridis", opacity=0.85,
                                   colorbar={"title": "mean BCE"}, name="loss slice"))
        reference = trace["states"][-1]["loss"]
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[reference], mode="markers", name="trained reference",
                                  marker={"color": ORANGE, "size": 8, "symbol": "diamond", "line": {"color": "white", "width": 2}}))
        fig.update_layout(scene={"xaxis": {"title": "offset of W2[0,0]", "range": [-4, 4]},
                                 "yaxis": {"title": "offset of b2[0]", "range": [-4, 4]},
                                 "zaxis": {"title": "mean training BCE"}, "aspectmode": "cube",
                                 "annotations": [{"x": 0, "y": 0, "z": reference, "text": f"trained reference<br>BCE={reference:.6f}",
                                                  "showarrow": True, "arrowhead": 2, "ax": -80, "ay": -60}]})
        self.html(fig, "08_loss_slice.html", "08 | A real two-coordinate loss slice",
                  "Only W2[0,0] and output bias b2[0] vary around the final trained reference; the other 31 coordinates stay fixed. Each surface height is the actual four-point training BCE. This output-layer slice is convex with the hidden representation fixed; it does not reveal the full network's nonconvex landscape. The full training trajectory leaves this slice, so no trajectory is drawn. A low point in this view is not a global-minimum certificate.")
        self.results["landscape"] = {"grid": list(losses.shape), "reference_loss": reference,
                                      "slice_min": float(losses.min()), "slice_max": float(losses.max()),
                                      "trajectory": "omitted: other parameters change during training"}

    def approximation(self):
        x = np.linspace(0, 1, 1001)
        target = target_function(x)
        widths = (2, 4, 8, 16)
        errors = []
        for width in widths:
            residual = relu_approximation(x, width)-target
            errors.append({"width": width, "grid_rmse": float(np.sqrt(np.mean(residual**2))),
                           "grid_max_abs_error": float(np.max(np.abs(residual)))})
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), layout="constrained")
        def draw(width):
            for ax in axes: ax.clear()
            prediction = relu_approximation(x, int(width))
            axes[0].plot(x, target, c=INK, lw=2, label="target sin(2πx) + 0.3 x²")
            axes[0].plot(x, prediction, c=ORANGE, lw=2, label=f"construction: {width} ReLU units")
            knots = np.linspace(0, 1, int(width)+1)
            axes[0].scatter(knots, target_function(knots), c=ORANGE, s=18)
            axes[0].set(xlabel="x", ylabel="function value", xlim=(0, 1), ylim=(-1.2, 1.3), title="Piecewise linear interpolation, not training")
            axes[0].legend(fontsize=8)
            for w in widths:
                axes[1].plot(x, relu_approximation(x, w)-target, label=f"width {w}", alpha=1 if w == width else 0.3)
            axes[1].set(xlabel="x", ylabel="approximation minus target", xlim=(0, 1), ylim=(-1.1, 1.1), title="Errors on a fixed 1,001-point grid")
            axes[1].legend(fontsize=8)
            fig.suptitle(f"09 | {width} ReLU basis functions: capacity by construction; no optimizer or generalization test", fontsize=12)
        steps = [2, 4, 8, 16] if self.quick else list(range(2, 17))
        self.gif(fig, draw, [w for w in steps for _ in range(3)], "09_relu_approximation.gif", fps=3)
        plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
        for ax, width, error in zip(axes.ravel(), widths, errors):
            ax.plot(x, target, color=INK, label="target")
            ax.plot(x, relu_approximation(x, width), color=ORANGE, label="constructed approximation")
            ax.set(title=f"{width} ReLU units | grid RMSE {error['grid_rmse']:.4f}", xlabel="x", ylabel="value", xlim=(0, 1), ylim=(-1.2, 1.3))
            ax.legend(fontsize=8)
        fig.suptitle("09 | Increasing representation detail on [0,1]; this is not a proof of universal approximation")
        self.png(fig, "09_relu_approximation.png"); plt.close(fig)
        self.results["approximation"] = {"grid_points": len(x), "domain": [0, 1], "method": "constructive interpolation", "errors": errors}

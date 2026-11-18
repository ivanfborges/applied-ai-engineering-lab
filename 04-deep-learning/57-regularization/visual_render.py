"""Figures derived from recorded training or explicitly named mathematical simulations."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from visual_math import (CENTER, METHODS, batchnorm_trace, decay_paths, dropout_moments,
                         dropout_pass, quadratic, regularized, stopping_replay)


COLORS = ["#2463a6", "#d97706", "#008577", "#a14c96", "#53652a", "#bd3a45"]
LABELS = {"baseline": "Baseline", "dropout": "Dropout", "weight_decay": "Weight decay",
          "batch_norm": "BatchNorm", "early_stopping": "Early stopping", "combined": "Combined"}


STOPPING_CONTROLS_JS = """
const plot = document.getElementById('{plot_id}');
plot.on('plotly_buttonclicked', function(event) {
    const play = event.button.label === 'Play';
    const action = play
        ? Plotly.animate(plot, null, {fromcurrent: true, frame: {duration: 220, redraw: true}, transition: {duration: 0}})
        : Plotly.animate(plot, [null], {mode: 'immediate', frame: {duration: 0, redraw: false}});
    action.catch(function(reason) {
        // Plotly rejects an interrupted animation with undefined, including intentional Pause.
        // Preserve actual errors; handle only this observed local cancellation case.
        if (reason !== undefined) setTimeout(function() { throw reason; }, 0);
    });
});
"""


def style(figure, title, height=530):
    figure.update_layout(template="plotly_white", title=title, height=height,
                         margin=dict(l=55, r=25, t=90, b=75),
                         legend=dict(orientation="h", y=-0.15), uirevision="day57")
    return figure


def run_by_name(study, name):
    return next(run for run in study["runs"] if run["name"] == name)


def state_at(run, epoch):
    if epoch >= run["epochs_run"] and run["configuration"]["early_stopping"]:
        return run["selected"]
    return run["snapshots"][min(epoch, run["epochs_run"])]


def boundary_axes(ax, study, state, title):
    data = study["data"]
    ax.contourf(data["grid_x"], data["grid_y"], state["probability"],
                levels=np.linspace(0, 1, 15), cmap="coolwarm", vmin=0, vmax=1, alpha=.55)
    if state["probability"].min() < .5 < state["probability"].max():
        ax.contour(data["grid_x"], data["grid_y"], state["probability"], levels=[.5], colors="#263345", linewidths=1)
    for split, marker in (("train", "o"), ("validation", "x")):
        indices = data["ids"][split]
        ax.scatter(*data["x"][indices].T, c=data["y"][indices], cmap="coolwarm", vmin=0, vmax=1,
                   marker=marker, s=15 if split == "train" else 24, alpha=.8, label=split)
    ax.set(title=f"{title} | checkpoint {state['epoch']}\ntrain CE {state['train']['loss']:.3f} · val CE {state['validation']['loss']:.3f}",
           xlabel="Synthetic feature 1 (raw units)", ylabel="Synthetic feature 2 (raw units)")
    ax.legend(fontsize=8, loc="upper right")


def boundary_figure(study, epoch=None, candidate="combined", selected=False):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    epoch = study["configuration"]["epochs_max"] if epoch is None else epoch
    runs = [run_by_name(study, name) for name in ("baseline", candidate)]
    for ax, run in zip(axes[0], runs):
        state = run["selected"] if selected else state_at(run, epoch)
        boundary_axes(ax, study, state, LABELS[run["name"]])
    for color, run in zip(COLORS, runs):
        rows = run["history"][:min(epoch, run["epochs_run"])]
        x = [row["epoch"] for row in rows]
        train = [row["train"]["loss"] for row in rows]
        val = [row["validation"]["loss"] for row in rows]
        axes[1, 0].plot(x, train, color=color, ls="--", label=LABELS[run["name"]] + " train")
        axes[1, 0].plot(x, val, color=color, label=LABELS[run["name"]] + " validation")
        axes[1, 1].plot(x, np.asarray(val) - train, color=color, label=LABELS[run["name"]])
    axes[1, 0].set(xlabel="Executed epoch", ylabel="Mean cross-entropy", title="Both losses in evaluation mode")
    axes[1, 1].set(xlabel="Executed epoch", ylabel="Validation CE − train CE", title="Generalization gap (may be negative)")
    axes[1, 1].axhline(0, color="gray", lw=.7)
    all_losses = [row[split]["loss"] for run in runs for row in run["history"] for split in ("train", "validation")]
    all_gaps = [row["validation"]["loss"] - row["train"]["loss"] for run in runs for row in run["history"]]
    axes[1, 0].set_ylim(0, max(all_losses) * 1.05)
    axes[1, 1].set_ylim(min(0, min(all_gaps)) - .02, max(0, max(all_gaps)) + .02)
    for ax in axes[1]:
        ax.set_xlim(0, study["configuration"]["epochs_max"])
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle(f"Real CPU training · synthetic 2D moons · timeline {epoch}\n"
                 "Stopped models display their restored checkpoint; regularization need not win", fontsize=12)
    return fig


def boundary_plot(study, run, state):
    data = study["data"]
    figure = go.Figure(go.Contour(x=data["grid_x"][0], y=data["grid_y"][:, 0], z=state["probability"],
                                  zmin=0, zmax=1, colorscale="RdBu_r", opacity=.6,
                                  contours=dict(start=0, end=1, size=.1), colorbar=dict(title="P(class 1)")))
    for split, symbol in (("train", "circle"), ("validation", "x")):
        for label in (0, 1):
            ids = data["ids"][split]
            ids = ids[data["y"][ids] == label]
            figure.add_trace(go.Scatter(x=data["x"][ids, 0], y=data["x"][ids, 1], mode="markers",
                                       name=f"{split} · class {label}", marker=dict(size=6, symbol=symbol,
                                       color=COLORS[label], line=dict(width=.5, color="white"))))
    figure.update_xaxes(title="Feature 1 (raw units)")
    figure.update_yaxes(title="Feature 2 (raw units)")
    return style(figure, f"{LABELS[run['name']]} · actual checkpoint {state['epoch']}")


def learning_plot(study, metric="loss"):
    fig = go.Figure()
    for color, run in zip(COLORS, study["runs"]):
        x = [row["epoch"] for row in run["history"]]
        for split, dash in (("train", "dash"), ("validation", "solid")):
            y = [row[split][metric] for row in run["history"]]
            fig.add_trace(go.Scatter(x=x, y=y, name=f"{LABELS[run['name']]} · {split}",
                                    line=dict(color=color, dash=dash), legendgroup=run["name"]))
    fig.update_xaxes(title="Executed epoch")
    fig.update_yaxes(title="Mean cross-entropy" if metric == "loss" else "Accuracy", rangemode="tozero")
    return style(fig, "Real CPU training · evaluation-mode metrics", 600)


def comparison_figure(study):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for color, run in zip(COLORS, study["runs"]):
        x = [row["epoch"] for row in run["history"]]
        t = np.array([row["train"]["loss"] for row in run["history"]])
        v = np.array([row["validation"]["loss"] for row in run["history"]])
        for ax, values in zip(axes.flat, (t, v, v - t)):
            ax.plot(x, values, color=color, label=LABELS[run["name"]])
    for ax, title, ylabel in zip(axes.flat, ("Training", "Validation", "Gap"),
                                ("Mean CE", "Mean CE", "Validation CE − train CE")):
        ax.set(title=title, xlabel="Executed epoch", ylabel=ylabel)
        ax.grid(alpha=.2)
    ax = axes[1, 1]
    ax.bar(np.arange(6), [run["selected"]["validation"]["loss"] for run in study["runs"]], color=COLORS)
    ax.set_xticks(range(6), [LABELS[r["name"]] for r in study["runs"]], rotation=24, ha="right", fontsize=8)
    ax.set(title="Retained validation checkpoint", ylabel="Mean cross-entropy", ylim=(0, None))
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Executed synthetic comparison · different checkpoint policies and update budgets\n"
                 "Author interpretation review required", fontsize=12)
    return fig


def dropout_figure(p=.3, pass_index=0):
    result = dropout_pass(p, pass_index)
    moments = dropout_moments(p)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
    ax = axes[0, 0]
    positions = [np.column_stack((np.full(len(values), i), np.linspace(-1, 1, len(values))))
                 for i, values in enumerate(result["post"])]
    for layer in range(3):
        for i, start in enumerate(positions[layer]):
            for j, end in enumerate(positions[layer + 1]):
                active = result["active"][layer][i] and result["active"][layer + 1][j]
                ax.plot([start[0], end[0]], [start[1], end[1]], color="#abc0c9" if active else "#edf0f2", lw=.6, zorder=0)
    for i, (positions_i, values, mask) in enumerate(zip(positions, result["post"], result["active"])):
        colors = plt.cm.viridis(Normalize(0, 6, clip=True)(np.abs(values)))
        colors[~mask] = [.85, .87, .89, 1]
        ax.scatter(*positions_i.T, s=260, c=colors, edgecolors="#334155")
        for (x, y), value, keep in zip(positions_i, values, mask):
            ax.text(x, y, f"{value:.1f}" if keep else "×", fontsize=8, ha="center", va="center",
                    color="white" if keep else "#667085")
        ax.text(i, -1.4, ["Input", "Hidden 1", "Hidden 2", "P(class)"][i], ha="center", fontsize=8)
    ax.set(ylim=(-1.6, 1.4), title=f"Fixed untrained network · pass {pass_index}\nColor |activation|: 0–6 (clipped); × = dropped")
    ax.axis("off")
    ax = axes[0, 1]
    ax.hist(moments["samples"][:, 1], bins=np.linspace(-.1, 5.2, 35), color=COLORS[0], alpha=.7)
    ax.axvline(1, color=COLORS[1], ls="--", label="Inference activation = 1")
    ax.set(title="One fixed activation h=1 · repeated masks", xlabel="Training activation after dropout", ylabel="Count (2,000 passes)")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    ax.plot(moments["activation"], moments["mean"], "o-", label="Empirical training mean")
    ax.plot(moments["activation"], moments["activation"], "--", label="Expectation / inference")
    ax.set(xlabel="Original activation h", ylabel="Mean activation", title="Inverted scaling preserves conditional mean")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    probabilities = np.linspace(0, .8, 41)
    empirical = [dropout_moments(value)["variance"][1] for value in probabilities]
    ax.plot(probabilities, probabilities / (1 - probabilities), label="Theory: p/(1-p), h=1")
    ax.scatter(probabilities, empirical, s=10, label="Empirical variance")
    ax.axvline(p, color=COLORS[1], ls="--")
    ax.set(xlabel="Drop probability p", ylabel="Conditional variance", title="Activation variance, h=1\nMore noise need not improve generalization")
    ax.legend(fontsize=8)
    fig.suptitle(f"Dropout simulation · p={p:.2f} · retained fraction {moments['retained']:.3f}\n"
                 f"Training scales retained activations by 1/(1−p)={1/(1-p):.2f}; inference uses identity", fontsize=12)
    return fig


def dropout_network_plot(p, pass_index):
    result = dropout_pass(p, pass_index)
    fig = go.Figure()
    positions = [(np.full(len(values), i), np.linspace(-1, 1, len(values))) for i, values in enumerate(result["post"])]
    for layer in range(3):
        for i, (x, y) in enumerate(zip(*positions[layer])):
            for j, (nx, ny) in enumerate(zip(*positions[layer + 1])):
                active = result["active"][layer][i] and result["active"][layer + 1][j]
                fig.add_trace(go.Scatter(x=[x, nx], y=[y, ny], mode="lines", hoverinfo="skip",
                                        line=dict(color="#aabcca" if active else "#e7edf2", width=1), showlegend=False))
    for layer, ((x, y), values, mask) in enumerate(zip(positions, result["post"], result["active"])):
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers+text", name=["Input", "Hidden 1", "Hidden 2", "Output probabilities"][layer],
                                text=[f"{v:.2f}" if keep else "dropped" for v, keep in zip(values, mask)],
                                textposition="top center", marker=dict(size=23, color=np.abs(values), cmin=0, cmax=6,
                                colorscale="Viridis", symbol=["circle" if keep else "x" for keep in mask]),
                                customdata=np.column_stack((result["raw"][layer], values, mask)),
                                hovertemplate="Before mask %{customdata[0]:.3f}<br>After %{customdata[1]:.3f}<br>Kept %{customdata[2]}<extra></extra>"))
    fig.update_xaxes(tickvals=[0, 1, 2, 3], ticktext=["Input", "Hidden 1", "Hidden 2", "P(class)"], showgrid=False, zeroline=False)
    fig.update_yaxes(visible=False, range=[-1.5, 1.5])
    return style(fig, f"Fixed untrained network · independent forward mask {pass_index} · p={p:.2f}", 430)


def decay_surface(decay=.3, steps=80):
    study = decay_paths(decay, steps)
    gx, gy = np.meshgrid(np.linspace(-2.5, 2.5, 35), np.linspace(-2.5, 2.5, 35))
    points = np.stack((gx, gy), axis=-1)
    fig = go.Figure()
    fig.add_trace(go.Surface(x=gx, y=gy, z=quadratic(points), colorscale="Blues", opacity=.45,
                             showscale=False, name="Original L", showlegend=True))
    fig.add_trace(go.Surface(x=gx, y=gy, z=regularized(points, decay), colorscale="Oranges", opacity=.5,
                             showscale=False, name="L2 reference J", showlegend=True))
    for i, (name, run) in enumerate(study["paths"].items()):
        w = run["points"]
        fig.add_trace(go.Scatter3d(x=w[:, 0], y=w[:, 1], z=run["objective"], mode="lines+markers", name=name,
                                   marker=dict(size=2), line=dict(color=COLORS[i], width=5)))
    minima = np.stack((CENTER, study["minimum"]))
    heights = [float(quadratic(CENTER)), float(regularized(study["minimum"], decay))]
    fig.add_trace(go.Scatter3d(x=minima[:, 0], y=minima[:, 1], z=heights, mode="markers+text",
                               marker=dict(size=6, color=[COLORS[0], COLORS[1]]), text=["min L", "min J"], name="Analytic minima"))
    fig.update_layout(scene=dict(xaxis_title="w1", yaxis_title="w2", zaxis_title="Loss / reference objective height",
                                  aspectmode="cube", camera=dict(eye=dict(x=1.5, y=1.5, z=1.1))))
    fig.add_annotation(text="All paths lifted onto J for comparison. AdamW is not gradient descent on J.",
                       x=.5, y=-.05, xref="paper", yref="paper", showarrow=False)
    return style(fig, f"Shifted quadratic: L (blue), J=L+λ||w||²/2 (orange) · λ={decay:.2f}", 700)


def decay_contour(decay=.3, steps=80):
    study = decay_paths(decay, steps)
    x = y = np.linspace(-2.5, 2.5, 70)
    gx, gy = np.meshgrid(x, y)
    fig = make_subplots(rows=1, cols=3, subplot_titles=("L2 reference contours J", "Linear weight norm", "Reference objective J"))
    fig.add_trace(go.Contour(x=x, y=y, z=regularized(np.stack((gx, gy), axis=-1), decay),
                             colorscale="Blues", showscale=False, opacity=.5, contours=dict(coloring="lines")), row=1, col=1)
    for i, (name, run) in enumerate(study["paths"].items()):
        for col, xx, yy in ((1, run["points"][:, 0], run["points"][:, 1]),
                            (2, np.arange(steps + 1), run["norm"]), (3, np.arange(steps + 1), run["objective"])):
            fig.add_trace(go.Scatter(x=xx, y=yy, name=name, legendgroup=name, showlegend=col == 1,
                                    line=dict(color=COLORS[i])), row=1, col=col)
    for col, x_label, y_label in ((1, "w1", "w2"), (2, "Update", "||w||₂"), (3, "Update", "J (reference only for AdamW)")):
        fig.update_xaxes(title=x_label, row=1, col=col)
        fig.update_yaxes(title=y_label, row=1, col=col)
    return style(fig, "Mathematical simulation · SGD lr=.12; Adam/L2 and AdamW lr=.08", 450)


def trajectory_figure(decay=.3, step=80):
    study = decay_paths(decay, 80)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), layout="constrained")
    gx, gy = np.meshgrid(np.linspace(-2.5, 2.5, 70), np.linspace(-2.5, 2.5, 70))
    axes[0].contour(gx, gy, regularized(np.stack((gx, gy), axis=-1), decay), levels=18, colors="#ccd6df", linewidths=.7)
    for color, (name, run) in zip(COLORS, study["paths"].items()):
        values = run["points"][:step + 1]
        axes[0].plot(*values.T, color=color, label=name)
        axes[0].scatter(*values[-1], color=color, s=25)
        axes[1].plot(np.arange(step + 1), run["norm"][:step + 1], color=color)
        axes[2].plot(np.arange(step + 1), run["objective"][:step + 1], color=color)
    axes[0].scatter(*study["minimum"], marker="*", s=120, color="black", label="min J")
    axes[0].set(xlabel="w1", ylabel="w2", title="Trajectories on J contours", xlim=(-2.5, 2.5), ylim=(-2.5, 2.5))
    axes[0].legend(fontsize=7)
    axes[1].set(xlabel="Update", ylabel="||w||₂", title="Parameter norm", xlim=(0, 80), ylim=(0, 3))
    ceiling = max(run["objective"].max() for run in study["paths"].values()) * 1.05
    axes[2].set(xlabel="Update", ylabel="J (reference)", title="Not AdamW's optimized objective", xlim=(0, 80), ylim=(0, ceiling))
    fig.suptitle(f"Mathematical simulation · λ={decay:.2f} · update {step}\nSGD lr=.12; adaptive lr=.08, betas=(.9,.999), eps=1e-8", fontsize=11)
    return fig


def bn_figure(trace, batch=39):
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), layout="constrained")
    for ax, key, title in zip(axes[0], ("raw", "normalized", "affine"), ("Raw activations", "Normalized (before affine)", "After γ scale and β shift")):
        for j, color in enumerate(COLORS[:3]):
            values = trace[key][batch, :, j]
            ax.hist(values, bins=12, alpha=.5, color=color, label=f"Feature {j+1}: μ={values.mean():.2f}, v={values.var():.2f}")
        ax.set(title=title, xlabel="Activation", ylabel="Count")
        extent = trace[key]
        ax.set_xlim(extent.min() - .2, extent.max() + .2)
        ax.set_ylim(0, trace["batch_size"])
        ax.legend(fontsize=6)
    t = np.arange(batch + 1) + 1
    for j, color in enumerate(COLORS[:3]):
        axes[1, 0].plot(t, trace["batch_mean"][:batch+1, j], color=color, alpha=.3)
        axes[1, 0].plot(t, trace["running_mean"][:batch+1, j], color=color, label=f"Feature {j+1}")
        axes[1, 1].plot(t, trace["batch_var"][:batch+1, j], color=color, alpha=.3)
        axes[1, 1].plot(t, trace["running_var"][:batch+1, j], color=color)
    axes[1, 0].set(title="Batch vs running mean\nFaint = current batch", xlabel="Mini-batch", ylabel="Mean", xlim=(1, 40))
    axes[1, 1].set(title="Running variance uses\nunbiased batch estimates", xlabel="Mini-batch", ylabel="Variance", xlim=(1, 40))
    axes[1, 0].set_ylim(min(trace["batch_mean"].min(), trace["running_mean"].min()) - .3,
                       max(trace["batch_mean"].max(), trace["running_mean"].max()) + .3)
    axes[1, 1].set_ylim(0, max(trace["batch_var"].max(), trace["running_var"].max()) * 1.05)
    axes[1, 0].legend(fontsize=7)
    for key, name, color in (("affine", "Training", COLORS[0]), ("inference", "Inference", COLORS[1])):
        axes[1, 2].hist(trace[key][batch, :, 2], bins=12, alpha=.6, label=name, color=color)
    axes[1, 2].set(title="Feature 3: same batch\nTraining vs inference", xlabel="Activation after affine", ylabel="Count")
    modes = np.concatenate((trace["affine"][:, :, 2].ravel(), trace["inference"][:, :, 2].ravel()))
    axes[1, 2].set_xlim(modes.min() - .2, modes.max() + .2)
    axes[1, 2].set_ylim(0, trace["batch_size"])
    axes[1, 2].legend(fontsize=8)
    fig.suptitle(f"BatchNorm simulation · batch {batch+1}, n={trace['batch_size']} · γ={trace['gamma']:g}, β={trace['beta']:g}, ε={trace['eps']:g}\n"
                 "Stationary synthetic features; running statistics initialized at mean 0 and variance 1", fontsize=11)
    return fig


def bn_running_plot(trace, feature=2):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Mean", "Variance"))
    t = np.arange(len(trace["raw"])) + 1
    for col, suffix, truth in ((1, "mean", "population_mean"), (2, "var", "population_var")):
        for key, name, color in (("batch_", "Current batch (biased for variance)", COLORS[0]),
                                  ("running_", "Exponential running estimate", COLORS[1])):
            fig.add_trace(go.Scatter(x=t, y=trace[key + suffix][:, feature], name=name,
                                    line=dict(color=color), showlegend=col == 1), row=1, col=col)
        fig.add_trace(go.Scatter(x=t, y=np.full(len(t), trace[truth][feature]), name="Population truth",
                                line=dict(color=COLORS[2], dash="dash"), showlegend=col == 1), row=1, col=col)
        fig.update_xaxes(title="Mini-batch", row=1, col=col)
        fig.update_yaxes(title=suffix, row=1, col=col)
    return style(fig, f"Stationary BatchNorm simulation · feature {feature+1} · batch size {trace['batch_size']}", 430)


def stopping_plot(run, replay, current=None):
    current = replay["stop_epoch"] if current is None else min(current, replay["stop_epoch"])
    rows = run["history"][:current]
    record = replay["rows"][current - 1]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Real recorded losses; future excluded", "Patience used"))
    for split, color in (("train", COLORS[0]), ("validation", COLORS[1])):
        fig.add_trace(go.Scatter(x=[r["epoch"] for r in rows], y=[r[split]["loss"] for r in rows],
                                name=split, line=dict(color=color)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[record["best_epoch"]], y=[record["best_loss"]], mode="markers",
                            marker=dict(size=12, symbol="star", color=COLORS[2]), name="Best observed validation"), row=1, col=1)
    fig.add_trace(go.Bar(x=[r["epoch"] for r in replay["rows"][:current]],
                        y=[r["bad_checks"] for r in replay["rows"][:current]], marker_color=COLORS[0], name="Checks without progress"), row=2, col=1)
    fig.add_hline(y=replay["patience"], line_dash="dash", row=2, col=1)
    fig.update_yaxes(title="Mean cross-entropy", row=1, col=1)
    fig.update_yaxes(title="Patience counter", row=2, col=1)
    fig.update_xaxes(title="Recorded epoch", row=2, col=1)
    return style(fig, f"Stopping-rule replay · current {current} · best {record['best_epoch']} · counter {record['bad_checks']}/{replay['patience']}", 600)


def early_figure(run, replay, current=None):
    current = replay["stop_epoch"] if current is None else min(current, replay["stop_epoch"])
    record = replay["rows"][current - 1]
    rows = run["history"][:current]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for split, color in (("train", COLORS[0]), ("validation", COLORS[1])):
        axes[0].plot([r["epoch"] for r in rows], [r[split]["loss"] for r in rows], label=split, color=color)
    axes[0].scatter(record["best_epoch"], record["best_loss"], color=COLORS[2], s=80, marker="*", label="Retained best")
    losses = [row[split]["loss"] for row in run["history"] for split in ("train", "validation")]
    axes[0].set(xlabel="Recorded epoch", ylabel="Mean cross-entropy", xlim=(1, len(run["history"])),
                ylim=(0, max(losses) * 1.1), title="Validation chooses the checkpoint")
    axes[0].legend(fontsize=8)
    axes[1].bar([r["epoch"] for r in replay["rows"][:current]], [r["bad_checks"] for r in replay["rows"][:current]], color=COLORS[0])
    axes[1].axhline(replay["patience"], color=COLORS[1], ls="--", label="Patience budget")
    axes[1].set(xlabel="Recorded epoch", ylabel="Consecutive checks without progress", xlim=(1, len(run["history"])),
                ylim=(0, replay["patience"] + 2), title=f"Current {current} · best {record['best_epoch']} · counter {record['bad_checks']}")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Replay of actual CPU baseline training · min_delta={replay['min_delta']:g}\nOnly the observed prefix is eligible; no test feedback", fontsize=11)
    return fig


def checkpoint_figure(study, replay):
    baseline = run_by_name(study, "baseline")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    best = baseline["snapshots"][replay["best_epoch"]]
    last = baseline["snapshots"][replay["stop_epoch"]]
    boundary_axes(axes[0], study, best, "Best validation checkpoint")
    boundary_axes(axes[1], study, last, "Last at hypothetical stopping time")
    distance = np.linalg.norm(best["weights"] - last["weights"])
    fig.suptitle(f"Actual recorded weights · ||w_best − w_last||₂ = {distance:.4f}\nIncludes linear biases; retained does not imply deployment approval", fontsize=11)
    return fig


def save_animation(output, filename, values, draw, fps=5):
    """Render a bounded sequence using actual draw callbacks, preserving semantic frame states."""
    fig = draw(values[0])
    fig.set_layout_engine(None)
    def update(index):
        fig.clear()
        temporary = draw(values[index])
        # Re-render into the original canvas without moving Matplotlib axes between figures.
        temporary.canvas.draw()
        pixels = np.asarray(temporary.canvas.buffer_rgba()).copy()
        plt.close(temporary)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(pixels)
        ax.axis("off")
    animation = FuncAnimation(fig, update, frames=len(values), interval=1000 / fps, repeat=True)
    path = output / filename
    animation.save(path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
    return path


def lambda_html_figure(steps=80):
    lambdas = np.linspace(0, 1, 11)
    fig = decay_surface(0, steps)
    frames = []
    for value in lambdas:
        frame = decay_surface(float(value), steps)
        frames.append(go.Frame(name=f"{value:.1f}", data=list(frame.data), layout=dict(title=frame.layout.title)))
    fig.frames = frames
    fig.update_layout(sliders=[dict(active=0, currentvalue=dict(prefix="λ = "), y=-.02,
        steps=[dict(label=f"{v:.1f}", method="animate", args=[[f"{v:.1f}"],
                    dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))]) for v in lambdas])])
    fig.layout.annotations[0].y = -.28
    fig.layout.legend.y = -.35
    fig.update_layout(height=820, margin=dict(b=150))
    return fig


def bn_html_figure():
    sizes = [2, 4, 8, 16, 32, 64, 128]
    fig = bn_running_plot(batchnorm_trace(2))
    fig.frames = [go.Frame(name=str(n), data=list(bn_running_plot(batchnorm_trace(n)).data),
                           layout=dict(title=f"Stationary BatchNorm simulation · feature 3 · batch size {n}")) for n in sizes]
    fig.update_layout(sliders=[dict(currentvalue=dict(prefix="Batch size = "), y=-.1,
        steps=[dict(label=str(n), method="animate", args=[[str(n)], dict(mode="immediate",
                    frame=dict(duration=0, redraw=True), transition=dict(duration=0))]) for n in sizes])], height=550)
    fig.layout.legend.y = -.4
    return fig


def stopping_html_figure(run, replay):
    epochs = np.unique(np.linspace(1, replay["stop_epoch"], min(30, replay["stop_epoch"])).astype(int))
    fig = stopping_plot(run, replay, 1)
    fig.frames = [go.Frame(name=str(epoch), data=list(stopping_plot(run, replay, int(epoch)).data),
                           layout=dict(title=stopping_plot(run, replay, int(epoch)).layout.title)) for epoch in epochs]
    fig.update_layout(sliders=[dict(currentvalue=dict(prefix="Recorded epoch = "), y=-.12,
        steps=[dict(label=str(epoch), method="animate", args=[[str(epoch)], dict(mode="immediate",
                    frame=dict(duration=0, redraw=True), transition=dict(duration=0))]) for epoch in epochs])],
        updatemenus=[dict(type="buttons", x=1, xanchor="right", y=1.15, direction="left", buttons=[
            dict(label="Play", method="skip", args=[]),
            dict(label="Pause", method="skip", args=[])], showactive=False)],
        height=740, margin=dict(b=130))
    fig.layout.legend.y = -.35
    return fig

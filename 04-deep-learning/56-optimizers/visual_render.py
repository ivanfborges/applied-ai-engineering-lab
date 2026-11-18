"""Bounded Matplotlib/Pillow exports and standalone Plotly trajectory inspection."""

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import numpy as np

from visual_math import COLORS, LABELS, frame_steps, grid


QUALITY = {"quick": (18, 75, 110, 51), "standard": (36, 85, 150, 75),
           "high": (60, 110, 180, 101)}


def heading(fig, title, caption):
    fig.suptitle(title, fontsize=16, fontweight="bold", y=.98)
    fig.text(.055, .025, textwrap.fill(caption, 150), fontsize=9, va="bottom", color="#344054")
    fig.subplots_adjust(left=.075, right=.97, bottom=.19, top=.84, wspace=.32, hspace=.65)
    return fig.text(.055, .905, "", fontsize=10, color="#344054")


def decorate(ax, title, ylabel, xlabel="Optimizer update"):
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=.2)
    ax.tick_params(labelsize=8)


def loss_limits(ax, traces):
    ax.set_yscale("log")
    values = np.concatenate([trace.losses for trace in traces])
    ax.set_ylim(max(float(values.min())*.5, 1e-30), max(float(values.max())*2, 1e-20))


def contours(ax, objective, traces, resolution=75, fixed=False):
    x, y, z = grid(objective, traces, resolution, fixed=fixed)
    levels = np.geomspace(max(float(z.max())*1e-5, 1e-5), max(float(z.max()), .1), 18)
    ax.contour(x, y, z, levels=levels, colors="#a9b5c2", linewidths=.65)
    ax.plot(*objective.optimum, marker="*", color="#202b3c", markersize=12, label="Objective minimum", zorder=8)
    ax.plot(*traces[0].points[0], marker="o", markerfacecolor="white", markeredgecolor="black",
            markersize=6, label="Shared start", zorder=8)
    ax.set(xlim=(x.min(), x.max()), ylim=(y.min(), y.max()), xlabel="Parameter x", ylabel="Parameter y")
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=8)


def paths(ax, traces, *, labels=None, styles=None, focus_view=False):
    artists = []
    for i, trace in enumerate(traces):
        color = COLORS[trace.method]
        label = labels[i] if labels else f"{LABELS[trace.method]} | lr={trace.lr:g}"
        style = styles[i] if styles else "-"
        line, = ax.plot([], [], style, color=color, lw=1.8, label=label)
        point, = ax.plot([], [], "o", color=color, ms=5)
        arrow = ax.quiver([0], [0], [0], [0], color=color, angles="xy", scale_units="xy", scale=1, width=.005)
        artists.append((line, point, arrow))

    def update(step):
        for trace, (line, point, arrow) in zip(traces, artists):
            k = min(step, trace.steps)
            points = trace.points[:k+1]
            visible = np.ones(len(points), dtype=bool)
            if focus_view:
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()
                visible = ((points[:, 0] >= xmin) & (points[:, 0] <= xmax)
                           & (points[:, 1] >= ymin) & (points[:, 1] <= ymax))
            displayed = points.copy()
            if focus_view and np.any(~visible):
                # Retain the actual exiting segment, then stop the focus path.
                # Drawing later outside-to-outside segments obscures the stable paths.
                displayed = displayed[:np.flatnonzero(~visible)[0]+1]
            line.set_data(displayed[:, 0], displayed[:, 1])
            point.set_data([points[-1, 0]] if visible[-1] else [], [points[-1, 1]] if visible[-1] else [])
            previous = trace.points[max(k-1, 0)]
            delta = trace.points[k]-previous
            if focus_view and not np.all(visible[-2:]):
                delta = np.zeros(2)
            arrow.set_offsets(previous[None, :])
            arrow.set_UVC([delta[0]], [delta[1]])
    return update


def series(ax, traces, quantities, *, ylabel, title, logarithmic=False):
    """Each quantity: (trace index, values, label, linestyle); fixed full-run limits."""
    lines = []
    for index, values, label, style in quantities:
        values = np.asarray(values)
        line, = ax.plot([], [], style, color=COLORS[traces[index].method], lw=1.6, label=label)
        lines.append((line, values))
    decorate(ax, title, ylabel)
    ax.set_xlim(0, max(t.steps for t in traces) or 1)
    values = np.concatenate([v for _, v in lines])
    if logarithmic:
        ax.set_yscale("log")
        ax.set_ylim(max(values.min()*.5, 1e-30), max(values.max()*2, 1e-20))
    else:
        low, high = min(0, values.min()), max(0, values.max())
        padding = max((high-low)*.12, 1e-8)
        ax.set_ylim(low-padding, high+padding)
    cursor = ax.axvline(0, color="#64748b", lw=.8, alpha=.6)
    ax.legend(fontsize=8, loc="best")

    def update(step):
        for line, values in lines:
            n = min(step, len(values))
            line.set_data(np.arange(1, n+1), np.maximum(values[:n], 1e-30) if logarithmic else values[:n])
        cursor.set_xdata([step, step])
    return update


def export(fig, update, steps, output, *, png=None, gif=None, quality="standard"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frames, gif_dpi, png_dpi, _ = QUALITY[quality]
    files = []
    try:
        if png:
            update(steps)
            path = output/png
            fig.savefig(path, dpi=png_dpi, facecolor="white")
            files.append(path)
        if gif:
            path = output/gif
            writer = PillowWriter(fps=8)
            with writer.saving(fig, str(path), dpi=gif_dpi):
                for step in frame_steps(steps, frames):
                    update(int(step))
                    writer.grab_frame(facecolor="white")
            files.append(path)
    finally:
        plt.close(fig)
    return files


def trajectory_figure(objective, traces, resolution=75):
    fig, (ax, loss_ax) = plt.subplots(1, 2, figsize=(11.8, 6))
    status = heading(fig, "01 | Same landscape, different gradient-to-update rules",
                     f"{objective.title}. Noise sigma={traces[0].noise:g}: "
                     "0 means exact analytic gradients; positive sigma is synthetic Gaussian noise, not sampled data. "
                     "Arrows are actual previous-to-current parameter updates. Rates are illustrative, not tuned.")
    contours(ax, objective, traces, resolution)
    path_update = paths(ax, traces)
    ax.legend(fontsize=7, loc="best")
    loss_lines = []
    for trace in traces:
        line, = loss_ax.plot([], [], color=COLORS[trace.method], label=LABELS[trace.method])
        loss_lines.append(line)
    decorate(loss_ax, "Loss evaluated on the exact objective", "Objective f(x,y)")
    loss_ax.set_xlim(0, max(t.steps for t in traces))
    loss_limits(loss_ax, traces)
    loss_ax.legend(fontsize=8)

    def update(step):
        path_update(step)
        for trace, line in zip(traces, loss_lines):
            k = min(step, trace.steps)
            line.set_data(np.arange(k+1), np.maximum(trace.losses[:k+1], 1e-30))
        status.set_text(f"Recorded update {step} | same start; each path follows its own actual gradients")
    return fig, update


def momentum_figure(objective, traces, momentum=.9, resolution=75):
    fig, axes = plt.subplots(1, 3, figsize=(13, 6))
    ax, vector_ax, state_ax = axes
    status = heading(fig, "03 | Momentum retains a direction, not a second gradient",
                     f"v_t = {momentum:g} v_(t-1) + g_t; theta_new = theta_old - lr v_t. "
                     "Vector diagram: g, mu*v_prev and v share one display scale; black update arrow is unit direction. "
                     "Numeric magnitudes are shown above. No dampening or Nesterov.")
    contours(ax, objective, traces, resolution)
    path_update = paths(ax, traces)
    ax.legend(fontsize=7)
    decorate(vector_ax, "Current vector composition", "Scaled y", "Scaled x")
    vector_ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
    vector_ax.set_aspect("equal")
    vector_ax.axhline(0, color="#d0d5dd", lw=.7)
    vector_ax.axvline(0, color="#d0d5dd", lw=.7)
    arrows = [vector_ax.quiver([0], [0], [0], [0], color=color, angles="xy", scale_units="xy", scale=1,
                              label=label, width=.012)
              for color, label in [("#64748b", "g_t"), ("#8759b6", "mu * v_prev"),
                                   (COLORS["momentum"], "v_t (sum)"), ("#111827", "Update direction")]]
    vector_ax.legend(fontsize=7, loc="upper left")
    trace = traces[1]
    history = trace.history
    state_update = series(state_ax, [trace], [(0, history["first"][:, i], f"v_{'xy'[i]}", style)
                                              for i, style in enumerate(["-", "--"])],
                          title="Velocity components over actual steps", ylabel="Unnormalized accumulator")

    def update(step):
        path_update(step)
        state_update(step)
        if not step or not trace.steps:
            g = previous = velocity = delta = np.zeros(2)
        else:
            k = min(step, trace.steps)-1
            g, previous = history["gradient"][k], momentum*history["previous_first"][k]
            velocity, delta = history["first"][k], history["total_delta"][k]
        scale = max(np.linalg.norm(g), np.linalg.norm(previous), np.linalg.norm(velocity), 1e-30)
        vectors = [g/scale, previous/scale, velocity/scale, delta/max(np.linalg.norm(delta), 1e-30)*.8]
        for i, (arrow, vector) in enumerate(zip(arrows, vectors)):
            arrow.set_offsets((g/scale if i == 1 else np.zeros(2))[None, :])
            arrow.set_UVC([vector[0]], [vector[1]])
        status.set_text(f"Update {step} | |g|={np.linalg.norm(g):.3g}; |v|={np.linalg.norm(velocity):.3g}; "
                        f"|delta|={np.linalg.norm(delta):.3g} | synthetic noise sigma={trace.noise:g}")
    return fig, update


def diagnostic_figure(trace, kind, epsilon=1e-8, *, rho=.99, beta1=.9, beta2=.999):
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7))
    h = trace.history
    if kind == "rmsprop":
        title = "04 | RMSProp normalizes coordinates using squared-gradient history"
        panels = [("gradient", "Raw gradient", "g"), ("second", "Uncentered second moment", "s"),
                  ("scaling", "Adaptive gradient scaling factors", "lr / (sqrt(s) + epsilon)"),
                  ("total_delta", "Actual parameter updates", "Delta parameter")]
        caption = ("s_t = rho*s_prev + (1-rho)*g_t^2; delta = -lr*g_t/(sqrt(s_t)+epsilon). "
                   "Second raw moment is not necessarily variance or true curvature. "
                   "Scaling factors are not universally valid effective learning rates.")
    else:
        title = "05 | Adam: direction memory, magnitude memory, bias-corrected update"
        panels = [("gradient", "Raw gradient", "g"), ("first", "First moment (smoothed direction)", "m"),
                  ("second", "Second raw moment (not variance)", "v"),
                  ("total_delta", "Corrected vs uncorrected update", "Delta parameter")]
        caption = ("m = beta1*m_prev+(1-beta1)*g; v = beta2*v_prev+(1-beta2)*g^2. "
                   "Actual update uses m/(1-beta1^t) and v/(1-beta2^t). "
                   "Dotted curves omit correction using the SAME history; they are not separately trained trajectories.")
    caption += (f" rho={rho:g}." if kind == "rmsprop" else f" beta1={beta1:g}, beta2={beta2:g}.")
    caption += f" lr={trace.lr:g}; epsilon={epsilon:g}; synthetic noise sigma={trace.noise:g}; no weight decay."
    status = heading(fig, title, caption)
    callbacks = []
    for ax, (key, panel_title, ylabel) in zip(axes.flat, panels):
        quantities = [(0, h[key][:, i], f"{'xy'[i]}", style) for i, style in enumerate(["-", "--"])]
        if kind == "adam" and key == "total_delta":
            uncorrected = -trace.lr*h["first"]/(np.sqrt(h["second"])+epsilon)
            quantities += [(0, uncorrected[:, i], f"{'xy'[i]} without correction", ":") for i in range(2)]
        callbacks.append(series(ax, [trace], quantities, title=panel_title, ylabel=ylabel))

    def update(step):
        for callback in callbacks:
            callback(step)
        status.set_text(f"Recorded update {step} | solid: x; dashed: y | states advance per update, not per epoch")
    return fig, update


def bias_figure(history, beta1, beta2):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 6))
    heading(fig, "05 | Zero initialization and Adam bias correction: constant nonzero gradient",
            f"20 steps; beta1={beta1:g}, beta2={beta2:g}; g=(0.7,-1.2). "
            "Corrected first moments equal g; corrected second raw moments equal g^2. "
            "This controlled constant signal isolates initialization bias; nonstationary gradients need separate interpretation.")
    x = np.arange(1, len(history["first"])+1)
    for ax, raw, corrected, title, ylabel in zip(axes, ["first", "second"], ["first_hat", "second_hat"],
                                               ["First moment: raw vs corrected", "Second raw moment: raw vs corrected"], ["m", "v"]):
        for i, coord in enumerate("xy"):
            ax.plot(x, history[raw][:, i], "--" if i else "-", color=COLORS["adam"], label=f"Raw {coord}")
            ax.plot(x, history[corrected][:, i], ":" if i else "-.", color="#202b3c", label=f"Corrected {coord}")
        decorate(ax, title, ylabel)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.legend(fontsize=8)
    return fig


def decay_figure(objective, pair, control, resolution=75):
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.5))
    status = heading(fig, "06 | Adam puts L2 inside moments; AdamW shrinks parameters separately",
                     f"Matched lr={pair[0].lr:g}, lambda={pair[0].weight_decay:g}. "
                     "Adam: input g+lambda*theta; no independently additive decay delta. "
                     "AdamW: theta_new=(1-lr*lambda)*theta_old-adaptive_step. "
                     "Equal coefficients do not mean equal regularization strength; no validation-quality claim.")
    contours(axes[0, 0], objective, pair, resolution)
    axes[0, 0].plot(0, 0, "+", color="black", label="Decay origin")
    path_update = paths(axes[0, 0], pair)
    axes[0, 0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.03, 1))
    all_traces = pair+control
    norm_quantities = []
    for i, trace in enumerate(all_traces):
        norm_quantities.append((i, np.linalg.norm(trace.points[1:], axis=1),
                                f"{LABELS[trace.method]} lambda={trace.weight_decay:g}", "-" if i < 2 else "--"))
    norm_update = series(axes[0, 1], all_traces, norm_quantities, title="Parameter norms and lambda=0 control", ylabel="Euclidean norm")
    component_updates = []
    for coord, ax in enumerate(axes[1]):
        quantities = [(0, pair[0].history["adaptive_delta"][:, coord], "Adam: coupled adaptive delta", "-"),
                      (1, pair[1].history["adaptive_delta"][:, coord], "AdamW: adaptive delta", "-"),
                      (1, pair[1].history["decay_delta"][:, coord], "AdamW: separate decay delta", "--"),
                      (1, pair[1].history["total_delta"][:, coord], "AdamW: total delta", ":")]
        component_updates.append(series(ax, pair, quantities, title=f"Coordinate {'xy'[coord]}: actual update components", ylabel="Delta parameter"))
    difference = float(np.max(np.abs(control[0].points-control[1].points)))

    def update(step):
        path_update(step)
        norm_update(step)
        for callback in component_updates:
            callback(step)
        status.set_text(f"Recorded update {step} | zero-decay full-path maximum difference={difference:.3g} | "
                        "quadratic minimum shifted away from origin" if objective.name == "quadratic" else
                        f"Recorded update {step} | zero-decay maximum difference={difference:.3g} | Rosenbrock objective")
    return fig, update


def rate_figure(objective, traces, labels, resolution=75):
    fig, (ax, loss_ax) = plt.subplots(1, 2, figsize=(11.8, 6))
    styles = [":", "-", "--"]
    status = heading(fig, f"07 | Learning-rate sensitivity: {LABELS[traces[0].method]}",
                     f"{objective.title}. Exact gradients. Rates are objective/optimizer-specific. "
                     "The parameter view is fixed around the start and minimum: divergent paths may leave it. "
                     "Focus paths stop at their first viewport exit; full losses remain at right. No optimizer update is clipped.")
    contours(ax, objective, traces, resolution, fixed=True)
    path_update = paths(ax, traces, labels=labels, styles=styles, focus_view=True)
    ax.legend(fontsize=7)
    lines = [loss_ax.plot([], [], style, color=COLORS[t.method], label=label)[0]
             for t, label, style in zip(traces, labels, styles)]
    decorate(loss_ax, "Outcome is measured, not assigned by the preset name", "Exact objective (log scale)")
    loss_limits(loss_ax, traces)
    loss_ax.set_xlim(0, max(t.steps for t in traces))
    loss_ax.legend(fontsize=8)

    def update(step):
        path_update(step)
        for trace, line in zip(traces, lines):
            k = min(step, trace.steps)
            line.set_data(np.arange(k+1), np.maximum(trace.losses[:k+1], 1e-30))
        stopped = "; ".join(f"{label.split(':')[0]} stopped at {t.steps}" for label, t in zip(labels, traces)
                            if t.status != "completed" and step >= t.steps)
        status.set_text(f"Requested update {step}" + (f" | {stopped}" if stopped else " | all displayed points are actual updates"))
    return fig, update


def noise_figure(objective, groups, resolution=75):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.3))
    status = heading(fig, "08 | Controlled Gaussian gradient noise interacts with optimizer memory",
                     "Top: paths; bottom: loss on the exact objective, not a noisy loss estimate. "
                     "Same standardized Gaussian draws across methods/levels; sigma is per-coordinate standard deviation. "
                     "Circle: common start; star: minimum. Synthetic perturbation is not an actual mini-batch or generalization experiment.")
    callbacks = []
    for i, traces in enumerate(groups):
        ax, loss_ax = axes[0, i], axes[1, i]
        contours(ax, objective, traces, resolution)
        labels = [f"sigma={t.noise:g}" for t in traces]
        styles = ["-", "--", ":"]
        callbacks.append(paths(ax, traces, labels=labels, styles=styles))
        ax.set_title(f"{LABELS[traces[0].method]} | lr={traces[0].lr:g}", fontsize=10)
        handles, names = ax.get_legend_handles_labels()
        noise_items = [(handle, name) for handle, name in zip(handles, names) if name.startswith("sigma=")]
        ax.legend([item[0] for item in noise_items], [item[1] for item in noise_items], fontsize=7, loc="upper right")
        lines = [loss_ax.plot([], [], style, color=COLORS[t.method], label=label)[0]
                 for t, label, style in zip(traces, labels, styles)]
        decorate(loss_ax, "Exact objective along noisy path", "f(x,y)")
        loss_ax.set_xlim(0, max(t.steps for t in traces))
        loss_limits(loss_ax, traces)

        def update_loss(step, traces=traces, lines=lines):
            for trace, line in zip(traces, lines):
                k = min(step, trace.steps)
                line.set_data(np.arange(k+1), np.maximum(trace.losses[:k+1], 1e-30))
        callbacks.append(update_loss)

    def update(step):
        for callback in callbacks:
            callback(step)
        status.set_text(f"Recorded update {step} | solid: exact; dashed: low noise; dotted: high noise")
    return fig, update


def surface_figure(objective, traces, *, resolution=75, frames=36, log_surface=False):
    import plotly.graph_objects as go

    x, y, raw_z = grid(objective, traces, resolution)
    transform = np.log1p if log_surface else lambda value: value
    zlabel = "log(1 + loss)" if log_surface else "Loss f(x,y)"
    data = [go.Surface(x=x, y=y, z=transform(raw_z), customdata=raw_z, opacity=.65,
                       colorscale="Cividis", showscale=False, name="Exact objective",
                       hovertemplate="x=%{x:.4f}<br>y=%{y:.4f}<br>loss=%{customdata:.6g}<extra>Surface</extra>"),
            go.Scatter3d(x=[objective.optimum[0]], y=[objective.optimum[1]], z=[0], mode="markers",
                         marker={"size": 6, "color": "black", "symbol": "diamond"}, name="Known minimum"),
            go.Scatter3d(x=[traces[0].points[0, 0]], y=[traces[0].points[0, 1]], z=[transform(traces[0].losses[0])],
                         mode="markers", marker={"size": 6, "color": "black"}, name="Shared start")]

    def trace_data(trace, step, current=False):
        k = min(step, trace.steps)
        indices = np.array([k]) if current else np.arange(k+1)
        points, losses = trace.points[indices], trace.losses[indices]
        return dict(x=points[:, 0].tolist(), y=points[:, 1].tolist(), z=transform(losses).tolist(),
                    customdata=np.column_stack([indices, losses]).tolist())

    for trace in traces:
        color, label = COLORS[trace.method], LABELS[trace.method]
        hover = f"{label}<br>step=%{{customdata[0]}}<br>x=%{{x:.5f}}<br>y=%{{y:.5f}}<br>loss=%{{customdata[1]:.6g}}<extra></extra>"
        data.extend([
            go.Scatter3d(**trace_data(trace, 0), mode="lines", line={"color": color, "width": 5},
                         name=f"{label} | lr={trace.lr:g}", legendgroup=trace.method, hovertemplate=hover),
            go.Scatter3d(**trace_data(trace, 0, True), mode="markers", marker={"color": color, "size": 5},
                         name=f"{label}: current", showlegend=False, legendgroup=trace.method, hovertemplate=hover),
            go.Scatter3d(**trace_data(trace, trace.steps, True), mode="markers",
                         marker={"color": color, "size": 5, "symbol": "diamond", "opacity": .6},
                         name=f"{label}: precomputed endpoint", showlegend=False, legendgroup=trace.method, hovertemplate=hover)])
    figure = go.Figure(data)
    frame_indices = frame_steps(max(t.steps for t in traces), frames)
    frame_list = []
    for step in frame_indices:
        frame_data, indices = [], []
        for i, trace in enumerate(traces):
            frame_data.extend([go.Scatter3d(**trace_data(trace, int(step))), go.Scatter3d(**trace_data(trace, int(step), True))])
            indices.extend([3+3*i, 4+3*i])
        frame_list.append(go.Frame(name=str(step), data=frame_data, traces=indices))
    figure.frames = frame_list
    animation_options = {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}
    selector = [{"label": "Compare all", "method": "restyle", "args": ["visible", [True]*len(data)]}]
    for method in [t.method for t in traces]:
        visible = [True]*3 + [t.method == method for t in traces for _ in range(3)]
        selector.append({"label": LABELS[method], "method": "restyle", "args": ["visible", visible]})
    figure.update_layout(
        title={"text": "02 | Inspect real updates on the exact loss surface<br><sup>" + objective.title +
                        f" | noise sigma={traces[0].noise:g} | vertical axis: {zlabel}</sup>", "font": {"size": 20},
               "x": .03, "y": .985, "xanchor": "left", "yanchor": "top"},
        template="plotly_white", height=950, margin={"l": 25, "r": 25, "t": 155, "b": 280},
        scene={"xaxis_title": "Parameter x", "yaxis_title": "Parameter y", "zaxis_title": zlabel,
               "aspectmode": "auto", "uirevision": "optimizer-camera", "dragmode": "orbit"},
        uirevision="optimizer-lab", legend={"orientation": "h", "y": -.27, "yanchor": "top", "groupclick": "togglegroup"},
        updatemenus=[{"buttons": selector, "x": 0, "y": 1.10, "xanchor": "left", "yanchor": "top"},
                     {"type": "buttons", "direction": "left", "x": .4, "y": 1.10,
                      "buttons": [{"label": "Play", "method": "skip"},
                                  {"label": "Pause", "method": "skip"}]}],
        sliders=[{"active": 0, "currentvalue": {"prefix": "Recorded update: "}, "y": -.04,
                  "steps": [{"label": str(step), "method": "animate", "args": [[str(step)], animation_options]}
                            for step in frame_indices]}],
        annotations=[{"text": "Drag to rotate; modebar for pan/zoom. Circles: current; translucent diamonds: precomputed endpoints.<br>"
                              "Paths use actual updates. Surface samples and path heights use the SAME objective/vertical transform.<br>"
                              "Noise, when enabled, is synthetic Gaussian perturbation. No generalization or benchmark claim.",
                      "xref": "paper", "yref": "paper", "x": 0, "y": -.38,
                      "xanchor": "left", "yanchor": "top", "showarrow": False, "font": {"size": 11}}])
    return figure


def export_surface(figure, output):
    path = Path(output)/"02_interactive_loss_surface.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Restyle one recorded frame at a time: canceling Plotly's 3D animation queue
    # with a native pause can reject an in-flight promise in the tested runtime.
    playback = """
    const gd = document.getElementById('{plot_id}');
    let playbackToken = 0;
    async function playRecordedFrames(token) {
      const frames = gd._transitionData._frames;
      let index = gd._fullLayout.sliders[0].active;
      if (index === frames.length - 1) index = -1;
      while (++index < frames.length && token === playbackToken) {
        const frame = frames[index];
        const update = {};
        for (const key of ['x', 'y', 'z', 'customdata'])
          update[key] = frame.data.map(trace => trace[key]);
        await Plotly.restyle(gd, update, frame.traces);
        if (token !== playbackToken) return;
        await Plotly.relayout(gd, {'sliders[0].active': index});
        await new Promise(resolve => setTimeout(resolve, 100));
      }
    }
    gd.on('plotly_buttonclicked', event => {
      if (event.button.label === 'Pause') ++playbackToken;
      if (event.button.label === 'Play') {
        const token = ++playbackToken;
        playRecordedFrames(token).catch(error => { ++playbackToken; console.error(error); });
      }
    });
    gd.on('plotly_sliderstart', () => { ++playbackToken; });
    """
    figure.write_html(path, include_plotlyjs=True, full_html=True, auto_open=False, auto_play=False, post_script=playback,
                      config={"displaylogo": False, "responsive": True, "scrollZoom": True})
    return path

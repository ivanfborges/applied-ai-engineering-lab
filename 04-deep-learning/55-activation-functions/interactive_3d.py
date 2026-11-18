"""Offline Plotly simplex and real PyTorch probability/logit surfaces."""

from pathlib import Path

import numpy as np

from from_scratch import softmax
from visual_math import simplex_samples


def plotly_modules():
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as error:
        raise RuntimeError('Plotly is required for 3D. Run '
                           'python -m pip install -e ".[dev,deep-learning]" from the repository root.') from error
    return go, make_subplots


def write_html(figure, path):
    figure.write_html(path, include_plotlyjs=True, full_html=True, auto_open=False, auto_play=False,
                      config={"displaylogo": False, "responsive": True, "scrollZoom": True})
    return Path(path)


def simplex_figure():
    go, _ = plotly_modules()
    data = simplex_samples()
    p, path, temperatures = data["probabilities"], data["trajectory"], data["temperatures"]
    figure = go.Figure([
        go.Mesh3d(x=[1, 0, 0], y=[0, 1, 0], z=[0, 0, 1], i=[0], j=[1], k=[2],
                  color="#b9c7d8", opacity=.18, name="p1+p2+p3=1", hoverinfo="skip"),
        go.Scatter3d(x=[1, 0, 0, 1], y=[0, 1, 0, 0], z=[0, 0, 1, 0],
                     mode="lines", line={"color": "#4a5d78", "width": 4}, name="Simplex edges"),
        go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers", name="180 synthetic logit vectors",
                     marker={"size": 3, "color": np.argmax(p, axis=1), "colorscale": [[0, "#2878a5"], [.5, "#dc7835"], [1, "#8755a3"]],
                             "cmin": 0, "cmax": 2}, hovertemplate="p=(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>"),
        go.Scatter3d(x=path[:, 0], y=path[:, 1], z=path[:, 2], name="Fixed logits [2,1,-1]: temperature trajectory",
                     mode="lines+markers", line={"color": "#1e293b", "width": 4},
                     marker={"size": 3, "color": temperatures, "colorscale": "Viridis",
                             "colorbar": {"title": "Temperature", "x": 1.02}},
                     customdata=temperatures, hovertemplate="T=%{customdata:.2f}<br>p=(%{x:.3f},%{y:.3f},%{z:.3f})<extra></extra>"),
        go.Scatter3d(x=[path[0, 0]], y=[path[0, 1]], z=[path[0, 2]], mode="markers",
                     marker={"size": 8, "color": "#ce4e72"}, name="Current fixed-vector probability"),
        go.Scatter3d(x=[1/3], y=[1/3], z=[1/3], mode="markers", marker={"symbol": "diamond", "size": 5, "color": "black"},
                     name="Uniform center (limit T -> infinity)")
    ])
    frames = []
    for index in [*range(0, len(temperatures), 3), len(temperatures) - 1]:
        t = temperatures[index]
        cloud = softmax(data["logits"], temperature=t)
        fixed = path[index]
        frames.append(go.Frame(name=str(index), traces=[2, 4], data=[
            go.Scatter3d(x=cloud[:, 0], y=cloud[:, 1], z=cloud[:, 2]),
            go.Scatter3d(x=[fixed[0]], y=[fixed[1]], z=[fixed[2]])]))
    # Match the first slider position to the initial scatter state.
    cloud = softmax(data["logits"], temperature=temperatures[0])
    figure.data[2].update(x=cloud[:, 0], y=cloud[:, 1], z=cloud[:, 2])
    figure.frames = frames
    figure.update_layout(
        title={"text": "G | Softmax lives on a probability simplex<br><sup>Cloud blue/orange/purple: dominant class 1/2/3. Trajectory color: temperature. Rotate and zoom.</sup>", "font": {"size": 19}},
        scene={"xaxis": {"title": "p1", "range": [0, 1]}, "yaxis": {"title": "p2", "range": [0, 1]},
               "zaxis": {"title": "p3", "range": [0, 1]}, "aspectmode": "cube", "uirevision": "camera"},
        template="plotly_white", height=880, margin={"l": 20, "r": 160, "b": 220, "t": 95},
        legend={"orientation": "h", "y": -.22},
        sliders=[{"currentvalue": {"prefix": "Temperature: "}, "pad": {"t": 30}, "y": 0,
                  "steps": [{"label": f"{temperatures[int(f.name)]:.2f}", "method": "animate",
                             "args": [[f.name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}]}
                            for f in frames]}],
        annotations=[{"text": "All points satisfy p1+p2+p3=1. This is a 2D plane embedded in 3D, not a probability volume.<br>Fixed logits; no calibration fitting or semantic correctness claim.",
                      "x": 0, "y": -.34, "xref": "paper", "yref": "paper", "showarrow": False, "xanchor": "left"}])
    return figure


def surface_figure(study, probability=True):
    go, make_subplots = plotly_modules()
    from visual_render import LABELS
    names = list(study["models"])
    low, high = (0., 1.) if probability else (
        min(float(model["logits"].min()) for model in study["models"].values()),
        max(float(model["logits"].max()) for model in study["models"].values()))
    figure = make_subplots(rows=2, cols=2, specs=[[{"type": "surface"}]*2]*2,
                           subplot_titles=[LABELS[name] for name in names],
                           horizontal_spacing=.04, vertical_spacing=.12)
    surface_trace_indices = []
    grid_x, grid_y = study["grid_x"], study["grid_y"]
    indices = study["train_idx"]
    x, y = study["x"][indices], study["y"][indices]
    for i, name in enumerate(names):
        result = study["models"][name]
        z = result["snapshots"][0]["probability"] if probability else result["logits"]
        surface_trace_indices.append(len(figure.data))
        figure.add_trace(go.Surface(x=grid_x, y=grid_y, z=z, colorscale="RdBu_r" if probability else "Viridis",
                                    cmin=low, cmax=high,
                                    showscale=(i == 0), colorbar={"title": "P(y=1)" if probability else "Raw logit", "len": .4},
                                    name=LABELS[name], opacity=.86,
                                    hovertemplate="x1=%{x:.2f}<br>x2=%{y:.2f}<br>" + ("P(y=1)" if probability else "Raw logit") + "=%{z:.4f}<extra></extra>"),
                         row=i//2+1, col=i%2+1)
        if probability:
            figure.add_trace(go.Scatter3d(x=x[:, 0], y=x[:, 1], z=y, mode="markers", name="Observed training label (0/1)",
                                         showlegend=(i == 0), marker={"size": 2.5, "color": y, "colorscale": "RdBu_r", "cmin": 0, "cmax": 1,
                                                                 "line": {"width": .5, "color": "black"}},
                                         hovertemplate="Training observation<br>label=%{z}<extra></extra>"), row=i//2+1, col=i%2+1)
        scene_name = "scene" if i == 0 else f"scene{i+1}"
        figure.update_layout(**{scene_name: {"xaxis_title": "Feature 1 (standardized)", "yaxis_title": "Feature 2 (standardized)",
                                             "zaxis": {"title": "P(y=1)" if probability else "Raw binary logit",
                                                       "range": [low, high]},
                                             "uirevision": "camera", "camera": {"eye": {"x": 1.5, "y": -1.5, "z": 1.2}}}})
    if probability:
        snapshots = study["models"][names[0]]["snapshots"]
        figure.frames = [go.Frame(name=str(s["epoch"]), traces=surface_trace_indices,
                                  data=[go.Surface(z=study["models"][name]["snapshots"][j]["probability"]) for name in names])
                         for j, s in enumerate(snapshots)]
        figure.update_layout(sliders=[{"currentvalue": {"prefix": "Completed updates: "}, "y": -.09,
                                        "steps": [{"label": f.name, "method": "animate", "args": [[f.name], {
                                            "mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}]}
                                                  for f in figure.frames]}])
    config = study["configuration"]
    figure.update_layout(template="plotly_white", height=1150 if probability else 1000,
                         title={"text": f"I | {'Probability surfaces: inspect real training checkpoints' if probability else 'Raw-logit surfaces: final model, before sigmoid'}"
                                        f"<br><sup>Synthetic moons; identical initialization / split / Adam lr=0.02; {config['epochs']} updates; seed={config['seed']}. No activation ranking.</sup>", "font": {"size": 19}},
                         margin={"l": 20, "r": 70, "t": 120, "b": 280 if probability else 190},
                         legend={"orientation": "h", "y": -.22},
                         annotations=list(figure.layout.annotations) + [{"x": 0, "y": -.31 if probability else -.19,
                            "xref": "paper", "yref": "paper", "xanchor": "left",
                            "text": "Shared vertical and color scales across all four panels. Sigmoid maps raw logits to probabilities.<br>A curved probability surface alone does not prove a curved boundary; use the 2D p=0.5 contours. Validation points are not overlaid.", "showarrow": False}])
    return figure


def export_3d(output, study):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    return [write_html(simplex_figure(), output / "07_softmax_simplex.html"),
            write_html(surface_figure(study), output / "09_probability_surfaces.html"),
            write_html(surface_figure(study, False), output / "09_raw_logit_surfaces.html")]


if __name__ == "__main__":
    # Share validation, reports and resource policy with the main entrypoint.
    from visual_lab import main
    main(default_mode="3d")

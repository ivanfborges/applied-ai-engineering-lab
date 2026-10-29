"""Exact trace contracts, consistent scaling, offline exports and the public CLI."""

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from PIL import Image
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

matplotlib.use("Agg")
TOPIC = Path(__file__).resolve().parents[1]


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, TOPIC / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


previous = {name: sys.modules.get(name) for name in ("from_scratch", "kmeans_visual_core")}
try:
    scratch = load("day36_visual_scratch", "from_scratch.py")
    sys.modules["from_scratch"] = scratch
    core = load("day36_visual_core", "kmeans_visual_core.py")
    sys.modules["kmeans_visual_core"] = core
    visual = load("day36_visual_renderer", "visualize_kmeans.py")
finally:
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def test_trace_has_exact_assignments_updates_and_correct_state_objectives():
    X, initial = core.convergence_data(42)
    original_X, original_initial = X.copy(), initial.copy()
    states = core.lloyd_trace(X, initial)
    assert states[0].phase == "initial"
    assert states[-1].converged
    assert states[-1].iteration > 1
    assert np.all(np.diff([s.inertia for s in states]) <= 1e-9)
    for index, state in enumerate(states):
        assert state.labels.shape == (len(X),)
        assert state.centers.shape == (3, 2)
        assert state.labels.min() >= 0 and state.labels.max() < 3
        assert state.inertia == pytest.approx(np.sum((X - state.centers[state.labels]) ** 2))
        if state.phase in ("initial", "assignment"):
            np.testing.assert_array_equal(state.labels, scratch.predict_labels(X, state.centers))
        if state.phase == "update":
            np.testing.assert_array_equal(state.labels, states[index - 1].labels)
            for j in range(3):
                members = X[state.labels == j]
                if len(members):
                    np.testing.assert_allclose(state.centers[j], members.mean(axis=0))
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(initial, original_initial)
    actual = scratch.kmeans_numpy(X, 3, init=initial, tol=0)
    assert actual.inertia == pytest.approx(states[-1].inertia)
    assert adjusted_rand_score(actual.labels, states[-1].labels) == 1


def test_empty_centers_ties_and_capped_trace_are_explicit():
    states = core.lloyd_trace([[0], [0], [10]], [[0], [0], [10]])
    np.testing.assert_array_equal(states[-1].centers, [[0], [0], [10]])
    np.testing.assert_array_equal(states[-1].labels, [0, 0, 2])
    assert states[-1].converged
    assert states[-1].inertia == 0
    capped = core.lloyd_trace([[0], [1], [4], [8], [10]], [[0], [1]], max_iter=1)
    assert not capped[-1].converged
    assert capped[-1].inertia == pytest.approx(27.1875)
    np.testing.assert_array_equal(capped[-1].labels, [0, 0, 1, 1, 1])


@pytest.mark.parametrize("X,initial,kwargs", [
    ([], [[0]], {}), ([[0]], [], {}), ([[np.nan]], [[0]], {}),
    ([[0]], [[0, 1]], {}), ([[0]], [[0], [1]], {}),
    ([[0]], [[0]], {"max_iter": 0}), ([[0]], [[0]], {"max_iter": True}),
])
def test_trace_rejects_invalid_inputs(X, initial, kwargs):
    with pytest.raises(ValueError):
        core.lloyd_trace(X, initial, **kwargs)


def test_scaling_inverse_centers_and_distance_shares():
    with threadpool_limits(limits=1):
        X, _, scaler, raw, scaled, shares = core.scaling_experiment(42)
    np.testing.assert_allclose(scaler.mean_, X.mean(axis=0))
    np.testing.assert_allclose(scaler.scale_, X.std(axis=0))
    for j in range(3):
        np.testing.assert_allclose(scaler.inverse_transform(scaled.cluster_centers_)[j],
                                   X[scaled.labels_ == j].mean(axis=0))
    np.testing.assert_allclose(shares.sum(axis=1), 1)
    delta = np.array([60, 1])
    assert shares[0, 0] == pytest.approx(3600 / 3601)
    np.testing.assert_allclose(shares[1], (delta / scaler.scale_) ** 2 / np.sum((delta / scaler.scale_) ** 2))
    assert raw.inertia_ == pytest.approx(np.sum((X - raw.cluster_centers_[raw.labels_]) ** 2))


def test_silhouette_validity_and_k_sweep_population():
    assert core.valid_silhouette([[0], [1]], [0, 0]) is None
    assert core.valid_silhouette([[0], [1]], [0, 1]) is None
    with threadpool_limits(limits=1):
        X, rows = core.selection_experiment(42, 2, 4)
    assert len(X) == 360
    assert [r["k"] for r in rows] == [2, 3, 4]
    for row in rows:
        assert sum(row["sizes"]) == len(X)
        assert -1 <= row["silhouette"] <= 1
    with pytest.raises(ValueError):
        core.selection_experiment(42, 4, 2)


def test_3d_is_genuinely_three_dimensional_and_frames_preserve_exact_states():
    X, scaler, scaled, initial = core.three_dimensional_data(42)
    assert np.linalg.matrix_rank(X - X.mean(axis=0)) == 3
    states = [s for s in core.lloyd_trace(scaled, initial) if s.phase == "assignment"]
    fig, result = visual.three_d_figure(42)
    assert result["dimensions"] == 3
    assert len(fig.frames) == len(states) == len(fig.layout.sliders[0].steps)
    for frame, state in zip(fig.frames, states):
        assert frame.name == str(state.iteration)
        assert f"{state.inertia:.3f}" in frame.layout.title.text
        for j in range(3):
            points, center = frame.data[2 * j:2 * j + 2]
            actual_points = np.column_stack((points.x, points.y, points.z))
            np.testing.assert_allclose(actual_points, X[state.labels == j])
            actual_center = np.column_stack((center.x, center.y, center.z))[0]
            np.testing.assert_allclose(actual_center, scaler.inverse_transform(state.centers)[j])
            assert points.marker.color == visual.COLORS[j]
            assert "Cluster" in points.hovertemplate


def test_gif_and_offline_html_exports_in_temporary_directory(tmp_path):
    visual.style()
    gif_path = tmp_path / "trace.gif"
    fig, result = visual.convergence_scene(gif_path, 42)
    visual.plt.close(fig)
    with Image.open(gif_path) as image:
        assert image.format == "GIF"
        assert image.n_frames >= result["states"]
        first = np.asarray(image.convert("RGB"))
        image.seek(image.n_frames - 1)
        assert np.any(first != np.asarray(image.convert("RGB")))
    html_path = tmp_path / "lab.html"
    visual.three_d_scene(html_path, 42)
    html = html_path.read_text(encoding="utf-8")
    assert "Plotly.newPlot" in html and "Plotly.addFrames" in html
    assert "kmeans-3d-lab" in html
    assert not re.search(r"<script\b[^>]*\bsrc\s*=", html, flags=re.IGNORECASE)
    assert html_path.stat().st_size > 1_000_000


def test_cli_agg_png_and_report(tmp_path):
    result = subprocess.run(
        [sys.executable, str(TOPIC / "visualize_kmeans.py"), "--scene", "elbow", "--k-min", "2", "--k-max", "4",
         "--output-dir", str(tmp_path), "--seed", "17", "--no-show", "--save"],
        env={**os.environ, "MPLBACKEND": "Agg"}, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    with Image.open(tmp_path / visual.OUTPUT_NAMES["elbow"]) as image:
        assert image.format == "PNG"
        assert image.width > 1000
    report = json.loads((tmp_path / "visual_report_elbow.json").read_text())
    assert report["seed"] == 17
    assert report["scenes"]["elbow"]["k_max"] == 4
    assert report["interpretations"] == "pending author review"


def test_local_gallery_links_all_deliverables_without_network_dependencies(tmp_path):
    path = visual.write_gallery(tmp_path)
    html = path.read_text(encoding="utf-8")
    assert "What to observe:" in html and "Production implication:" in html
    for filename in visual.OUTPUT_NAMES.values():
        assert f'href="{filename}"' in html
    assert not re.search(r'(?:src|href)="https?://', html)


@pytest.mark.parametrize("argv", [["--seed", "-1"], ["--k-min", "1"], ["--k-min", "5", "--k-max", "3"], ["--all", "--scene", "3d"]])
def test_cli_rejects_invalid_configuration(argv):
    with pytest.raises(SystemExit) as error:
        visual.parse_args(argv)
    assert error.value.code == 2

"""Exact trace invariants and bounded rendering checks; exports use tmp_path."""

import importlib.util
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image
import pytest
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits


TOPIC = Path(__file__).resolve().parents[1]
ALIASES = ("from_scratch", "example", "dbscan_visual_core", "generate_visualizations")
previous = {name: sys.modules.get(name) for name in ALIASES}
loaded = {}
try:
    for name in ALIASES:
        spec = importlib.util.spec_from_file_location(name, TOPIC / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        loaded[name] = module
finally:
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module

scratch, core, render = loaded["from_scratch"], loaded["dbscan_visual_core"], loaded["generate_visualizations"]


def test_trace_records_real_noise_to_border_and_only_expands_core():
    result = core.expansion_experiment()
    steps = result["steps"]
    assert any(step.operation == "noise" and step.current == 0 for step in steps)
    assert any(step.operation == "noise_to_border" and step.current == 0 for step in steps)
    assert not result["core"][0] and result["labels"][0] >= 0
    for step in steps:
        assert np.all(~step.expanded | result["core"])
        if step.operation == "expand":
            assert result["core"][step.current]
        if step.current is not None:
            expected = np.flatnonzero(np.linalg.norm(result["X"] - result["X"][step.current], axis=1) <= 0.64)
            np.testing.assert_array_equal(step.neighborhood, expected)
            assert step.current in step.neighborhood
    np.testing.assert_array_equal(steps[-1].labels, result["labels"])
    assert np.all(steps[-1].visited)
    assert not np.any(steps[-1].labels == -2)
    assert len(steps) < 100


def test_snapshots_are_independent_and_engine_agrees_with_library():
    result = core.expansion_experiment()
    with threadpool_limits(limits=1):
        reference = DBSCAN(eps=0.64, min_samples=4).fit(result["X"])
    reference_core = np.zeros(len(result["X"]), dtype=bool)
    reference_core[reference.core_sample_indices_] = True
    np.testing.assert_array_equal(result["core"], reference_core)
    np.testing.assert_array_equal(result["labels"] == -1, reference.labels_ == -1)
    assert adjusted_rand_score(result["labels"][reference_core], reference.labels_[reference_core]) == 1
    labels, _ = scratch.dbscan_numpy(result["X"], eps=0.64, min_samples=4)
    assert adjusted_rand_score(labels, result["labels"]) == 1
    assert not np.shares_memory(result["steps"][0].labels, result["steps"][-1].labels)
    assert np.all(result["steps"][0].labels == -2)


def test_border_roles_have_a_core_neighbor_and_noise_has_none():
    result = core.expansion_experiment()
    X, labels, is_core = result["X"], result["labels"], result["core"]
    distance_to_core = np.linalg.norm(X[:, None] - X[is_core][None, :], axis=2).min(axis=1)
    assert ((~is_core) & (labels >= 0)).any()
    assert (labels == -1).any()
    assert np.all(distance_to_core[(~is_core) & (labels >= 0)] <= result["eps"])
    assert np.all(distance_to_core[labels == -1] > result["eps"])


def test_trace_shared_border_is_order_dependent_without_merging_cores():
    X = np.array([[-1.6], [-1.5], [-1.4], [-0.9], [0.], [0.9], [1.4], [1.5], [1.6]])
    labels, mask, _ = scratch.dbscan_trace(X, eps=1, min_samples=4)
    reverse, reverse_mask, _ = scratch.dbscan_trace(X[::-1], eps=1, min_samples=4)
    reverse = reverse[::-1]
    np.testing.assert_array_equal(mask, reverse_mask[::-1])
    assert adjusted_rand_score(labels[mask], reverse[mask]) == 1
    assert labels[4] == labels[0] and reverse[4] == reverse[-1]
    assert reverse[4] != reverse[0]


def test_trace_rejects_unbounded_animation_inputs():
    with pytest.raises(ValueError, match="200 points"):
        scratch.dbscan_trace(np.zeros((201, 2)))


def test_cli_list_and_single_png_export(tmp_path, capsys):
    assert render.main(["--list"]) == 0
    assert "Exact DBSCAN expansion" in capsys.readouterr().out
    assert render.main(["--visualization", "4", "--output-dir", str(tmp_path)]) == 0
    with Image.open(tmp_path / "04_kmeans_vs_dbscan.png") as picture:
        assert picture.width > 1000 and picture.height > 500
        picture.verify()
    assert (tmp_path / "visual_report_04.json").is_file()
    gallery = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert 'href="04_kmeans_vs_dbscan.png"' in gallery


def test_pillow_export_uses_real_epsilon_frames(tmp_path):
    render.style()
    result = render.render_epsilon(tmp_path)
    with Image.open(tmp_path / "01_epsilon_neighborhood.gif") as animation:
        assert animation.n_frames > 20
        assert sum(animation.seek(i) or animation.info["duration"] for i in range(animation.n_frames)) > 7000
    assert result["neighbor_count_range"][0] < 7 <= result["neighbor_count_range"][1]


def test_parameter_grid_and_all_frame_statistics_are_consistent():
    with threadpool_limits(limits=1):
        grid = core.parameter_grid()
        figure = render.sensitivity_figure(grid)
    assert grid["cluster_counts"].shape == (5, 6)
    assert len(figure.frames) == 30
    assert len(figure.layout.sliders[0].steps) == 6
    assert len(figure.layout.updatemenus[0].buttons) == 5
    assert figure.layout.meta["eps"] == 0.18 and figure.layout.meta["min_samples"] == 7
    for frame in figure.frames:
        mi, ei = map(int, frame.name.split("-"))
        result = grid["results"][mi][ei]
        assert sum(len(trace.x) for trace in frame.data[:3]) == len(grid["X"])
        assert len(frame.data[2].x) == result["noise_count"]
        assert frame.layout.meta["clusters"] == result["clusters"]
        assert sum((result[k] for k in ("core_count", "border_count", "noise_count"))) == len(grid["X"])
        assert frame.data[3].x == (ei,) and frame.data[3].y == (mi,)


def test_scaling_preserves_identities_and_uses_fitted_population_statistics():
    with threadpool_limits(limits=1):
        result = core.scaling_experiment()
    original, distorted, standardized = result["spaces"]
    np.testing.assert_allclose(distorted, original * [30, 1])
    np.testing.assert_allclose(standardized.mean(axis=0), 0, atol=1e-14)
    np.testing.assert_allclose(standardized.std(axis=0), 1)
    np.testing.assert_allclose(result["fitted_means"], distorted.mean(axis=0))
    for fit in result["results"]:
        assert len(fit["labels"]) == 300 and fit["eps"] == 0.3


def test_density_illustration_is_exact_counts_over_area_and_matches_core_rule():
    points = np.array([[0., 0], [1, 0], [3, 0]])
    counts, density = core.fixed_radius_density(points, [[0., 0], [2, 0]], 1)
    np.testing.assert_array_equal(counts, [2, 2])
    np.testing.assert_allclose(density, np.array([2, 2])/(3*np.pi))
    X, _ = core.make_data()
    counts, density = core.fixed_radius_density(X, X, 0.18)
    with threadpool_limits(limits=1):
        result = core.fit_density(X, 0.18, 7)
    np.testing.assert_array_equal(counts >= 7, result["core"])
    np.testing.assert_allclose(density, counts/(len(X)*np.pi*0.18**2))


@pytest.mark.parametrize("eps", [0, -1, np.nan, np.inf, True, "bad", 1j])
def test_density_rejects_invalid_radii(eps):
    with pytest.raises(ValueError, match="eps"):
        core.fixed_radius_density([[0, 0]], [[0, 0]], eps)


def test_density_rejects_non_2d_coordinates():
    with pytest.raises(ValueError, match="two-dimensional"):
        core.fixed_radius_density([[0, 0, 0]], [[0, 0, 0]], 1)


def test_three_feature_geometry_tooltips_and_standalone_export(tmp_path):
    X = core.three_dimensional_data()
    assert X.shape == (304, 3) and np.linalg.matrix_rank(X-X.mean(axis=0)) == 3
    with threadpool_limits(limits=1):
        result = core.fit_density(X, 0.26, 7)
    assert result["clusters"] >= 2 and result["noise_count"] > 0
    figure = render.three_dimensional_figure(X, result)
    assert figure.layout.meta["fit_dimensions"] == 3
    assert sum(len(trace.x) for trace in figure.data) == len(X)
    for trace in figure.data:
        for row, z in zip(trace.customdata, trace.z):
            index, label, kind, x, y, feature_z = row
            np.testing.assert_allclose([x, y, feature_z], X[index])
            assert z == feature_z and label == result["labels"][index]
            assert kind in ("core", "border", "noise")
    path = tmp_path / "local.html"
    render.save_html(figure, path, "test-dbscan")
    exported = path.read_text(encoding="utf-8")
    assert "plotly.js v" in exported and 'id="test-dbscan"' in exported
    assert not re.search(r'<script[^>]+src=["\']https?://', exported, re.I)


def test_landscape_height_is_not_a_third_dbscan_feature():
    with threadpool_limits(limits=1):
        figure, result = render.density_landscape_figure()
    assert figure.layout.meta["fit_dimensions"] == result["fit_dimensions"] == 2
    assert figure.data[0].type == "surface"
    assert np.asarray(figure.data[0].z).shape == (65, 75)
    assert sum(len(trace.x) for trace in figure.data[2:]) == result["n"]


def test_variable_density_and_concentration_are_reproducible_controls():
    with threadpool_limits(limits=1):
        variable = core.variable_density_experiment()
    assert variable["X"].shape == (255, 2)
    assert np.array_equal(variable["X"], core.variable_density_experiment()["X"])
    concentration = core.distance_concentration()
    assert concentration["dimensions"] == [2, 10, 50, 100]
    assert all(len(values) == 180*179//2 for values in concentration["relative_distances"])
    assert np.isfinite(concentration["coefficient_of_variation"]).all()
    assert all(0 < value <= 1 for value in concentration["median_nearest_farthest_ratio"])


def test_all_cli_dispatches_every_scene_and_writes_only_available_gallery_targets(tmp_path, monkeypatch):
    calls = []
    def fake(number):
        def generate(output):
            calls.append(number)
            for name in render.SCENES[number][1]:
                (output / name).write_text("test orchestration placeholder", encoding="utf-8")
            return {"checked": number}
        return generate
    monkeypatch.setattr(render, "RENDERERS", {i: fake(i) for i in render.SCENES})
    assert render.main(["--all", "--output-dir", str(tmp_path)]) == 0
    assert calls == sorted(render.SCENES)
    report = json.loads((tmp_path/"visual_report_all.json").read_text())
    assert len(report["scenes"]) == 11
    gallery = (tmp_path/"index.html").read_text()
    for target in re.findall(r'(?:href|src)="([^"]+)"', gallery):
        assert (tmp_path/target).is_file()

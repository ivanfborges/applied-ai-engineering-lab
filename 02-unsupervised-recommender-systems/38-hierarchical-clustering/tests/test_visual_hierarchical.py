"""Numerical invariants and bounded exports; rendering always uses tmp_path."""

import importlib.util
from pathlib import Path
import re

import numpy as np
from PIL import Image
import pytest
from scipy.cluster.hierarchy import cut_tree, dendrogram, fcluster, linkage
from sklearn.metrics import adjusted_rand_score


TOPIC = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("day38_visual", TOPIC / "visualize_hierarchical.py")
visual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visual)


def test_average_control_and_exact_partition_with_zero_heights():
    Z = linkage([[1.], [2.], [8.], [9.]], method="average")
    np.testing.assert_allclose(Z[:, 2], [1, 1, 7])
    assert visual.interior_height(Z, 3) is None
    assert visual.interior_height(Z, 2) == 4
    zeros = linkage(np.zeros((4, 2)), method="average")
    for k in range(1, 5):
        assert len(np.unique(visual.exact_partition(zeros, k))) == k


@pytest.mark.parametrize("method", ["single", "complete", "average", "ward"])
def test_every_animation_state_is_an_actual_tree_partition(method):
    X = visual.small_data()
    Z = linkage(X, method=method)
    states, members = visual.tree_states(Z), visual.node_members(Z)
    assert len(states) == len(X)
    for completed, active in enumerate(states):
        coverage = [member for node in active for member in members[node]]
        assert sorted(coverage) == list(range(len(X)))
        assert len(active) == len(X)-completed
        if 0 < completed < len(Z):
            expected = cut_tree(Z, n_clusters=[len(X)-completed]).ravel()
            assert adjusted_rand_score(visual.merge_labels(Z, completed), expected) == 1
        if completed:
            left, right = map(int, Z[completed-1, :2])
            assert left in states[completed-1] and right in states[completed-1]
            assert left not in active and right not in active


def test_constructed_branches_match_scipy_geometry_by_height():
    Z = linkage(visual.small_data()[:12], method="average")
    order, branches = visual.tree_geometry(Z)
    reference = dendrogram(Z, no_plot=True)
    assert order == reference["leaves"]
    reference_by_height = {row[1]: (x, row) for x, row in zip(reference["icoord"], reference["dcoord"])}
    for step, (xs, ys) in enumerate(branches):
        assert ys[1] == Z[step, 2]
        rx, ry = reference_by_height[ys[1]]
        np.testing.assert_allclose(np.asarray(xs)*10 + 5, rx)
        np.testing.assert_allclose(ys, ry)
    frames = visual.merge_schedule(len(Z)+1)
    assert len(frames) == 1 + 2*len(Z)
    assert [step for step, phase in frames if phase == "merge"] == list(range(len(Z)))


def test_chaining_geometry_guarantees_short_connecting_links():
    X = visual.bridge_data()
    single = linkage(X, method="single")
    complete = linkage(X, method="complete")
    average = linkage(X, method="average")
    assert single[-1, 2] < 1.1
    assert complete[-1, 2] > 8
    assert len(np.unique(fcluster(single, 1.1, criterion="distance"))) == 1
    assert len(np.unique(fcluster(complete, 1.1, criterion="distance"))) > 1
    assert len(np.unique(fcluster(average, 1.1, criterion="distance"))) > 1


def test_ward_actual_sse_change_size_effect_and_selected_merge():
    X, Z, active, candidates = visual.ward_control()
    members = visual.node_members(Z)
    step = len(X)-3
    delta, a, b = candidates[0]
    assert {a, b} == set(Z[step, :2].astype(int))
    assert sorted(len(members[node]) for node in active) == [2, 2, 8]
    nearest_centroids = min((np.linalg.norm(X[members[left]].mean(axis=0)-X[members[right]].mean(axis=0)), left, right)
                            for _, left, right in candidates)
    assert {a, b} != set(nearest_centroids[1:])

    def sse(ids):
        return np.square(X[ids]-X[ids].mean(axis=0)).sum()

    actual_delta = sse(members[a]+members[b])-sse(members[a])-sse(members[b])
    np.testing.assert_allclose(delta, actual_delta)
    np.testing.assert_allclose(Z[step, 2]**2, 2*delta)


def test_scaling_preserves_rows_and_changes_geometry_without_quality_claim():
    raw, standardized, alphas, points, trees = visual.scaling_control()
    np.testing.assert_array_equal(points[0], raw)
    np.testing.assert_allclose(points[-1], standardized)
    np.testing.assert_allclose(standardized.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(standardized.std(axis=0), 1)
    assert alphas[0] == 0 and alphas[-1] == 1
    assert adjusted_rand_score(visual.exact_partition(trees[0], 3), visual.exact_partition(trees[-1], 3)) < 1


def test_offline_cut_explorer_has_exact_partitions_and_no_false_tie_line():
    fig = visual.explorer_figure()
    states = fig.layout.meta["states"]
    assert len(fig.layout.sliders[0].steps) == 6
    for index, state in enumerate(states):
        labels, k, Z = state["labels"], state["k"], np.asarray(state["Z"])
        assert len(np.unique(labels)) == k
        np.testing.assert_array_equal(labels, visual.exact_partition(Z, k))
        if state["threshold"] is not None:
            assert adjusted_rand_score(fcluster(Z, state["threshold"], criterion="distance"), labels) == 1
        else:
            assert len(fig.data[index*3+1].x) == 0
            assert "no horizontal" in fig.layout.sliders[0].steps[index].args[1]["title.text"]


def test_genuine_three_dimensional_slider_uses_one_tree():
    fig = visual.three_dimensional_figure()
    meta = fig.layout.meta
    X, Z = np.asarray(meta["X"]), np.asarray(meta["Z"])
    assert X.shape == (48, 3) and meta["fit_dimensions"] == 3
    assert np.linalg.matrix_rank(X-X.mean(axis=0)) == 3
    assert len(meta["states"]) >= 4
    for state in meta["states"]:
        np.testing.assert_array_equal(state["labels"], visual.exact_partition(Z, state["k"]))
    assert fig.layout.scene.uirevision == "fixed-geometry"


def test_embedding_math_nonzero_vectors_and_average_linkage():
    settings = visual.embedding_control()
    for _, X, metric, Z, distances in settings:
        assert np.all(np.linalg.norm(X, axis=1) > 0)
        np.testing.assert_allclose(Z, linkage(X, method="average", metric=metric))
        np.testing.assert_allclose(distances, distances.T)
    cosine, squared = settings[3], settings[4]
    np.testing.assert_allclose(2*cosine[4], squared[4], atol=1e-14)
    np.testing.assert_allclose(2*cosine[3][:, 2], squared[3][:, 2], atol=1e-14)
    assert visual.embedding_figure().layout.meta["real_embedding_model"] is False


def test_pair_storage_is_exact_and_distinguishes_square_matrix():
    assert visual.pair_storage(50000) == (1249975000, 9999800000, 20000000000)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "3"])
def test_partition_input_errors(bad):
    with pytest.raises(ValueError):
        visual.exact_partition(linkage([[0.], [1.], [4.]], method="average"), bad)


def test_cli_fast_render_html_and_skip_does_not_overwrite(tmp_path):
    assert visual.main(["--scene", "4", "--fast", "--output-dir", str(tmp_path)]) == 0
    gif = tmp_path / "04_dendrogram_cuts.gif"
    before = gif.read_bytes()
    with Image.open(gif) as image:
        assert image.n_frames == 6
        for frame in range(image.n_frames):
            image.seek(frame)
            image.load()
    content = (tmp_path/"04_dendrogram_explorer.html").read_text(encoding="utf-8")
    assert not re.search(r"<script[^>]+src=", content)
    assert "Plotly.newPlot" in content and "no horizontal separator" in content
    assert visual.main(["--scene", "4", "--output-dir", str(tmp_path)]) == 0
    assert gif.read_bytes() == before
    assert len(list(tmp_path.glob("visual_run_*.json"))) == 2


def test_single_merge_animation_and_static_export_to_temporary_directory(tmp_path):
    visual.style()
    result = visual.render_process(tmp_path, fast=True)
    assert result["merges_shown"] == 14
    with Image.open(tmp_path / "01_agglomerative_process.gif") as image:
        assert image.n_frames == 29
        image.seek(image.n_frames-1)
        image.load()
    visual.render_complexity(tmp_path, fast=True)
    with Image.open(tmp_path / "10_complexity.png") as image:
        image.verify()

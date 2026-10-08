"""Numerical and temporary-artifact validation for the Day 26 visual lab."""

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest
from sklearn.tree import DecisionTreeClassifier

TOPIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC))

import decision_tree_visual_math as vm
import visualize_decision_trees as viz


def test_binary_impurity_formulas_and_endpoints():
    np.testing.assert_allclose(vm.gini_binary([0, 0.5, 1]), [0, 0.5, 0])
    np.testing.assert_allclose(vm.entropy_binary([0, 0.5, 1]), [0, 1, 0])
    assert float(vm.entropy_binary(0.5)) == pytest.approx(1)
    assert vm.class_impurity([0, 0, 0, 1]) == pytest.approx(0.375)
    for bad in ([-0.1], [1.1], [np.nan], [np.inf]):
        with pytest.raises(ValueError, match="probabilities"):
            vm.entropy_binary(bad)
    with pytest.raises(ValueError, match="binary"):
        vm.class_impurity([0, 2])


def test_weighted_threshold_scores_match_library_stumps():
    x, y = vm.toy_data()
    for criterion in ("gini", "entropy"):
        rows = vm.evaluate_thresholds(x, y, criterion)
        chosen = max(rows, key=lambda row: row["gain"])
        tree = DecisionTreeClassifier(criterion=criterion, max_depth=1, random_state=26).fit(x[:, None], y)
        assert chosen["threshold"] == pytest.approx(tree.tree_.threshold[0])
        assert all(row["n_left"] + row["n_right"] == len(x) for row in rows)
        assert all(row["n_left"] > 0 and row["n_right"] > 0 for row in rows)
    gini_best = max(vm.evaluate_thresholds(x, y), key=lambda row: row["gain"])
    assert gini_best["threshold"] == 4.5
    assert gini_best["parent_impurity"] == pytest.approx(0.46875)
    assert gini_best["weighted_impurity"] == pytest.approx(0.1875)
    assert gini_best["gain"] == pytest.approx(0.28125)
    with pytest.raises(ValueError, match="distinct"):
        vm.evaluate_thresholds([1, 1], [0, 1])


def test_dataset_reproducibility_and_training_only_pruning_path():
    data = vm.create_dataset()
    repeat = vm.create_dataset()
    for key in ("X_train", "X_val", "y_train", "y_val"):
        np.testing.assert_array_equal(data[key], repeat[key])
    assert len(data["y_train"]) == 315
    assert len(data["y_val"]) == 135
    original = vm.pruning_sweep(data, max_candidates=8)
    changed_validation = {**data, "y_val": 1 - data["y_val"]}
    changed = vm.pruning_sweep(changed_validation, max_candidates=8)
    np.testing.assert_allclose([r["ccp_alpha"] for r in original], [r["ccp_alpha"] for r in changed])
    leaves = [row["leaves"] for row in original]
    assert all(a >= b for a, b in zip(leaves[:-1], leaves[1:]))
    assert leaves[-1] == 1
    assert all(0 <= r["validation_accuracy"] <= 1 for r in original)
    with pytest.raises(ValueError, match="at least two"):
        vm.pruning_sweep(data, max_candidates=1)


def test_geometry_tiles_space_and_matches_leaf_predictions():
    data = vm.create_dataset()
    model = vm.fit_tree(data, max_depth=3)
    nodes = vm.node_geometry(model, data["bounds"])
    split_ids = {node["node"] for node in nodes if not node["leaf"]}
    root_frontier = vm.frontier_regions(model, data["bounds"], set())
    assert len(root_frontier) == 1
    assert root_frontier[0]["node"] == 0
    leaves = vm.frontier_regions(model, data["bounds"], split_ids)
    xmin, xmax, ymin, ymax = data["bounds"]
    area = sum((r["bounds"][1] - r["bounds"][0]) * (r["bounds"][3] - r["bounds"][2]) for r in leaves)
    assert area == pytest.approx((xmax - xmin) * (ymax - ymin))
    assert len(leaves) == model.get_n_leaves()
    for leaf in leaves:
        x0, x1, y0, y1 = leaf["bounds"]
        query = [[(x0 + x1) / 2, (y0 + y1) / 2]]
        assert leaf["probability"] == pytest.approx(model.predict_proba(query)[0, 1])
    for node in nodes:
        if not node["leaf"]:
            x0, x1, y0, y1 = node["bounds"]
            segment = np.asarray(node["segment"])
            assert np.all(segment[:, 0] >= x0) and np.all(segment[:, 0] <= x1)
            assert np.all(segment[:, 1] >= y0) and np.all(segment[:, 1] <= y1)


def test_all_core_views_generate_only_temporary_artifacts(tmp_path):
    data = vm.create_dataset()
    paths = []
    experiments = {}
    for name, render in viz.VIEWS.items():
        if name == "probability-3d":
            continue
        generated, experiment = render(data, tmp_path, False)
        paths.extend(generated)
        if experiment is not None:
            assert experiment["review_status"] == "pending author review"
            assert experiment["hypothesis"] and experiment["configuration"]
            assert experiment["result"] is not None
            assert experiment["limitation"]
            json.dumps(experiment, allow_nan=False)
            experiments[name] = experiment
    assert len(paths) == 21
    assert len({path.name for path in paths}) == 21
    for path in paths:
        assert path.is_file() and path.stat().st_size > 0
        with Image.open(path) as image:
            assert image.width > 300 and image.height > 200
            if path.suffix == ".gif":
                assert 2 <= image.n_frames <= 10
    assert plt.get_fignums() == []
    (tmp_path / "visual_validation_report.json").write_text(
        json.dumps({"configuration": data["configuration"], "experiments": experiments}, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Temporary visual validation directory: {tmp_path}")


def test_optional_flat_plateaus_html(tmp_path):
    pytest.importorskip("plotly")
    paths, report = viz.create_optional_3d_visualization(vm.create_dataset(), tmp_path, False)
    assert report is None
    text = paths[0].read_text(encoding="utf-8")
    assert "Plotly.newPlot" in text and '"type":"mesh3d"' in text
    assert '<script src="https://' not in text
    assert paths[0].name == "interactive_decision_tree_lab.html"


def test_cli_selected_views_report_and_no_show(tmp_path, capsys):
    assert viz.main(["--view", "gini", "candidate-splits", "--output-dir", str(tmp_path), "--no-show"]) == 0
    report = json.loads((tmp_path / "experiment_report.json").read_text(encoding="utf-8"))
    assert report["generated_files"] == ["01_gini_impurity.png", "04_candidate_splits.png"]
    assert set(report["experiments"]) == {"candidate-splits"}
    stdout = capsys.readouterr().out
    assert "Generated:" in stdout and "experiment_report.json" in stdout
    assert plt.get_fignums() == []


def test_clean_interaction_uses_conditional_splits():
    X, y = vm.interaction_dataset()
    model = DecisionTreeClassifier(max_depth=2, random_state=vm.SEED).fit(X, y)
    np.testing.assert_array_equal(model.predict(X), y)
    assert model.tree_.feature[0] == 0
    assert model.get_depth() == 2
    assert model.get_n_leaves() == 3
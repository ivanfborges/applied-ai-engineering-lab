"""Checks for visual-lab mathematics, data boundaries and actual rendering."""
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image, ImageStat
import pytest
from sklearn.svm import SVC

TOPIC = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("day31_visuals", TOPIC / "visualize_svm.py")
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


def test_margin_status_includes_exact_boundaries():
    margins = lab.signed_margins([0, 1, 1, 0], [2, 0.4, 1, -2])
    np.testing.assert_allclose(margins, [-2, 0.4, 1, 2])
    np.testing.assert_array_equal(lab.margin_status([-0.1, 0, 0.5, 1, 1.1]), [2, 2, 1, 0, 0])


def test_rbf_similarity_has_correct_distance_scale():
    for gamma in [0.1, 0.5, 1, 5]:
        assert lab.rbf_similarity(0, gamma) == 1
        assert lab.rbf_similarity(1 / np.sqrt(gamma), gamma) == pytest.approx(np.exp(-1))
    assert lab.rbf_similarity(2, 0.1) > lab.rbf_similarity(2, 5)


@pytest.mark.parametrize("distance,gamma", [(-1, 1), (np.inf, 1), (1, 0), (1, np.nan), (1, [1])])
def test_invalid_similarity_inputs(distance, gamma):
    with pytest.raises(ValueError):
        lab.rbf_similarity(distance, gamma)


def test_explicit_lift_separates_rings_with_a_measured_plane():
    X, y = lab.make_data()[3]
    lifted, _, radial, threshold = lab.circle_lift_models(X, y)
    np.testing.assert_allclose(lifted[:, 2], np.sum(X**2, axis=1))
    assert lifted[y == 1, 2].max() < threshold < lifted[y == 0, 2].min()
    np.testing.assert_array_equal(radial.predict(lifted[:, 2:3]), y)


@pytest.mark.parametrize("X", [[], [[1, 2, 3]], [[np.nan, 1]]])
def test_lift_rejects_invalid_feature_shapes(X):
    with pytest.raises(ValueError):
        lab.lifted_coordinates(X)


def test_candidate_hyperplanes_are_valid_but_have_smaller_margins():
    X, y = lab.make_data()[0]
    model = SVC(kernel="linear", C=10000, tol=1e-9).fit(X, y)
    svm_margin = lab.signed_margins(y, model.decision_function(X)).min() / np.linalg.norm(model.coef_[0])
    for w, b, distance in lab.separating_candidates(X, y, model):
        assert np.all(lab.signed_margins(y, X @ w + b) > 0)
        assert 0 < distance < svm_margin


@pytest.mark.parametrize("kernel", ["rbf", "linear"])
def test_dual_terms_reconstruct_fitted_binary_decision(kernel):
    X, y, validation, _ = lab.make_data()[2]
    if kernel == "rbf":
        model = lab.fit_rbf(X, y, gamma="scale")
    else:
        model = SVC(kernel="linear", C=1).fit(X, y)
    terms, intercept = lab.kernel_contributions(model, validation)
    assert terms.shape == (len(validation), len(lab.fitted_svc(model).support_))
    np.testing.assert_allclose(terms.sum(axis=1) + intercept, model.decision_function(validation), atol=1e-10)


def test_reconstruction_rejects_unimplemented_multiclass_layout():
    model = SVC().fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
    with pytest.raises(ValueError, match="binary"):
        lab.kernel_contributions(model, [[1, 1]])


def test_cv_scalers_see_only_fold_training_rows(monkeypatch):
    X, y, _, _ = lab.make_data()[2]
    seen = []
    original = lab.StandardScaler.fit

    def record_fit(self, values, *args, **kwargs):
        seen.append(np.asarray(values).copy())
        return original(self, values, *args, **kwargs)

    monkeypatch.setattr(lab.StandardScaler, "fit", record_fit)
    means, rows = lab.cv_scores(X, y, [0.1, 1], [1], folds=3)
    folds = lab.StratifiedKFold(3, shuffle=True, random_state=42)
    expected = [X[train] for train, _ in folds.split(X, y)] * 2
    assert len(seen) == len(expected)
    for actual, desired in zip(seen, expected):
        np.testing.assert_array_equal(actual, desired)
    np.testing.assert_allclose(means[:, 0], [np.mean(row["fold_scores"]) for row in rows])


def test_data_generation_and_validation_split_are_deterministic():
    first, second = lab.make_data(), lab.make_data()
    for a, b in zip(first, second):
        for aa, bb in zip(a, b):
            np.testing.assert_array_equal(aa, bb)
    X, y, validation, validation_y = first[2]
    assert len(X) == 120 and len(validation) == 60
    assert np.bincount(y).tolist() == [60, 60]
    assert np.bincount(validation_y).tolist() == [30, 30]
    assert not set(map(tuple, X)).intersection(map(tuple, validation))


def test_animation_saves_distinct_looping_frames(tmp_path):
    X, y = lab.make_data()[1]
    values = [0.1, 1, 10]
    models = [SVC(kernel="linear", C=C).fit(X, y) for C in values]
    lab.configure_style()
    path, rows = lab.save_parameter_animation(X, y, models, values, "C", tmp_path, "test.gif")
    with Image.open(path) as animation:
        assert animation.n_frames == 3
        assert animation.info["loop"] == 0
        frames = []
        for i in range(animation.n_frames):
            animation.seek(i)
            frames.append(animation.convert("RGB").tobytes())
        assert len(set(frames)) == 3
    assert [row["support_vectors"] for row in rows] == [len(model.support_) for model in models]


def test_generation_without_gifs_creates_nonempty_manifest_and_closes_figures(tmp_path):
    out = tmp_path / "outputs"
    paths, report = lab.run_lab(out, skip_gifs=True)
    assert out.is_dir()
    assert len(paths) == 15
    assert not list(out.glob("*.gif"))
    assert len(report["artifacts"]) == 14
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
    for path in paths:
        if path.suffix == ".png":
            with Image.open(path) as img:
                assert img.width >= 1000 and img.height >= 800
                assert max(ImageStat.Stat(img.convert("RGB")).stddev) > 5
        elif path.suffix == ".html":
            text = path.read_text(encoding="utf-8")
            assert "Plotly.newPlot" in text and path.stem in text
            assert '<script src="https://' not in text
    assert lab.plt.get_fignums() == []
    assert all(item["review_status"] == "pending author review" for item in report["experiments"].values())

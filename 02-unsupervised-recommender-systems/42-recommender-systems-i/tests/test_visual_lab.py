"""Visual-lab numerical contracts and temporary artifact smoke checks."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
from sklearn.metrics import ndcg_score


TOPIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC))


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, TOPIC / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


scratch = load("from_scratch", "from_scratch.py")
factor = load("factorization", "factorization.py")
core = load("recommender_visual_core", "recommender_visual_core.py")
visuals = load("recommender_visuals", "recommender_visuals.py")
cli = load("day42_visual_cli", "visualize_day42.py")


def test_synthetic_masks_and_event_semantics():
    a, b = core.make_data(), core.make_data()
    assert a["train"].shape == (12, 10)
    np.testing.assert_array_equal(a["truth"], b["truth"])
    assert not (a["train_mask"] & a["test_mask"]).any()
    np.testing.assert_array_equal(np.isfinite(a["train"]), a["train_mask"])
    assert ((a["truth"] >= 1) & (a["truth"] <= 5)).all()
    assert np.all(a["purchases"] <= a["clicks"])
    assert np.all(a["clicks"] <= a["views"])
    assert np.all(a["views"] <= a["impressions"])


def test_similarity_variants_have_correct_support():
    values = np.array([[1., 2, 3, np.nan], [2, 4, 5, 1], [np.nan, np.nan, np.nan, 4], [2, 2, 2, 2]])
    matrices = core.similarity_matrices(values)
    for key in ("raw_cosine", "centered_cosine", "pearson", "shrunk", "overlap"):
        np.testing.assert_allclose(matrices[key], matrices[key].T)
    assert matrices["pearson"][0, 2] == 0
    assert matrices["pearson"][0, 3] == 0
    assert matrices["shrunk"][0, 1] == pytest.approx(matrices["pearson"][0, 1] * 3 / 8)
    assert matrices["centered_cosine"][0, 1] != pytest.approx(matrices["pearson"][0, 1])
    assert abs(scratch.overlap_pearson([1, 2], [5, 1], 2, 0)) == pytest.approx(1)


def test_prediction_trace_reconstructs_weighted_score():
    model = scratch.UserNeighborhood(shrink=0).fit([[1, 2, 3, np.nan], [1, 2, 3, 4], [2, 3, 4, 5.]])
    trace = core.prediction_trace(model, 0, 3)
    assert trace["result"]["score"] == pytest.approx(3.5)
    assert trace["baseline"] + sum(r["contribution"] for r in trace["rows"]) == pytest.approx(3.5)
    assert trace["weight_sum"] == pytest.approx(2)


def test_implicit_targets_and_invalid_counts():
    p, c = core.implicit_targets([0, 1, 4], 3)
    np.testing.assert_array_equal(p, [0, 1, 1])
    np.testing.assert_array_equal(c, [1, 4, 13])
    for values, alpha in (([-1], 1), ([np.nan], 1), ([1], -1), ([1], "2")):
        with pytest.raises(ValueError):
            core.implicit_targets(values, alpha)


def test_factor_shapes_determinism_and_real_masked_loss():
    data = core.make_data()
    fitted = factor.fit_factorization(data["train"], epochs=80)
    repeat = factor.fit_factorization(data["train"], epochs=80)
    assert fitted["p"].shape == (12, 3)
    assert fitted["q"].shape == (10, 3)
    assert fitted["predictions"].shape == (12, 10)
    assert np.isfinite(fitted["predictions"]).all()
    np.testing.assert_array_equal(fitted["predictions"], repeat["predictions"])
    actual_mse = np.mean((fitted["predictions"][data["train_mask"]] - data["train"][data["train_mask"]]) ** 2)
    assert fitted["history"][-1]["mse"] == pytest.approx(actual_mse)
    assert fitted["history"][-1]["mse"] < fitted["history"][0]["mse"]
    np.testing.assert_allclose(fitted["predictions"], fitted["mean"] + fitted["bu"][:, None]
                               + fitted["bi"][None, :] + fitted["p"] @ fitted["q"].T)


def test_one_step_gradient_matches_finite_difference():
    ratings = np.array([[1., np.nan, 4.], [2., 5., 3.]])
    result = factor.fit_factorization(ratings, epochs=1, learning_rate=.01, regularization=.03, snapshots=2)
    before, after = result["states"]
    mask = np.isfinite(ratings)
    def loss(p, q, bu, bi):
        pred = result["mean"] + bu[:, None] + bi[None, :] + p @ q.T
        return np.mean((pred[mask] - ratings[mask]) ** 2) + .03 * sum(np.sum(v*v) for v in (p, q, bu, bi))
    for key, index in (("p", (0, 1)), ("q", (2, 0)), ("bu", (1,)), ("bi", (0,))):
        values = {k: before[k].copy() for k in ("p", "q", "bu", "bi")}
        values[key][index] += 1e-6
        high = loss(**values)
        values[key][index] -= 2e-6
        low = loss(**values)
        numerical_gradient = (high - low) / 2e-6
        update_gradient = (before[key][index] - after[key][index]) / .01
        assert numerical_gradient == pytest.approx(update_gradient, abs=1e-7)


def test_empty_entities_do_not_keep_random_factors():
    fitted = factor.fit_factorization([[1, 5, np.nan], [3, 4, np.nan], [np.nan]*3], epochs=20)
    np.testing.assert_array_equal(fitted["p"][-1], 0)
    np.testing.assert_array_equal(fitted["q"][-1], 0)
    assert np.isfinite(fitted["predictions"]).all()


def test_orthogonal_rotation_preserves_compatibility_and_distance_is_different():
    fitted = factor.fit_factorization(core.make_data()["train"], epochs=20)
    rotation, _ = np.linalg.qr(np.random.default_rng(42).normal(size=(3, 3)))
    np.testing.assert_allclose((fitted["p"] @ rotation) @ (fitted["q"] @ rotation).T,
                               fitted["p"] @ fitted["q"].T, atol=1e-12)
    user = np.array([1., 0, 0])
    items = np.array([[1., .1, 0], [3., 0, 0]])
    assert np.argmax(items @ user) != np.argmin(np.linalg.norm(items-user, axis=1))


@pytest.mark.parametrize("kwargs", [{"dimensions": 0}, {"epochs": True}, {"regularization": -1},
                                    {"learning_rate": 0}, {"snapshots": 0}])
def test_invalid_factor_controls(kwargs):
    with pytest.raises(ValueError):
        factor.fit_factorization([[1, 3]], **kwargs)


@pytest.mark.parametrize("ratings", [[[np.nan]], [[1, np.inf]], [[1j]], []])
def test_invalid_factor_ratings(ratings):
    with pytest.raises(ValueError):
        factor.fit_factorization(ratings)


def test_ranking_metrics_match_hand_calculation_and_sklearn():
    candidates, relevant = list("ABCDE"), set("AC")
    a = core.ranking_metrics(list("ACBDE"), relevant, 3, candidates)
    b = core.ranking_metrics(list("BCADE"), relevant, 3, candidates)
    assert a["precision"] == b["precision"] == pytest.approx(2/3)
    assert a["recall"] == b["recall"] == 1
    assert a["ndcg"] == 1
    assert b["ndcg"] == pytest.approx((1/np.log2(3)+.5)/(1+1/np.log2(3)))
    assert b["ndcg"] == pytest.approx(ndcg_score([[1, 0, 1, 0, 0]], [[3, 5, 4, 2, 1]], k=3))


@pytest.mark.parametrize("order,relevant,k,candidates", [
    ([0, 0], {0}, 1, [0, 1]), ([0], {1}, 1, [0]), ([0], set(), 1, [0]),
    ([0], {0}, 2, [0]), ([0], {0}, True, [0]),
])
def test_invalid_ranking_protocol(order, relevant, k, candidates):
    with pytest.raises(ValueError):
        core.ranking_metrics(order, relevant, k, candidates)


def test_heldout_changes_cannot_change_model_or_neighborhood():
    data = core.make_data()
    changed_truth = data["truth"].copy()
    changed_truth[~data["train_mask"]] = 1
    changed_train = np.where(data["train_mask"], changed_truth, np.nan)
    original = factor.fit_factorization(data["train"], epochs=30)
    changed = factor.fit_factorization(changed_train, epochs=30)
    np.testing.assert_array_equal(original["predictions"], changed["predictions"])
    a = scratch.UserNeighborhood().fit(data["train"])
    b = scratch.UserNeighborhood().fit(changed_train)
    np.testing.assert_array_equal(a.similarities_, b.similarities_)
    np.testing.assert_array_equal(a.user_means_, b.user_means_)


def test_cold_start_and_sparsity_are_computed():
    data = core.make_data()
    states = core.cold_start_states(data["train"])
    assert states[0]["positive_neighbors"] == states[0]["personalized"] == 0
    assert all(p["source"] == "popularity_count" for p in states[0]["predictions"])
    assert states[-1]["personalized"] > 0
    results = core.sparsity_sweep(data, repeats=3)
    assert results[0]["stability_sd"] == pytest.approx(0)
    assert results[-1]["sparsity"] > results[0]["sparsity"]
    assert all(0 <= row["coverage"] <= 1 for row in results)


def test_exposure_accounting():
    for history in core.exposure_simulation(rounds=5).values():
        assert history[-1]["impressions"].sum() == 5 * 80
        assert np.all(history[-1]["clicks"] <= history[-1]["impressions"])
        assert history[-1]["share"].sum() == pytest.approx(1)


def test_fast_preserves_factor_fit():
    a, b = core.build_context(False), core.build_context(True)
    np.testing.assert_array_equal(a["factor"]["predictions"], b["factor"]["predictions"])
    assert len(a["factor"]["states"]) > len(b["factor"]["states"])


def test_rendered_png_gif_and_offline_html_in_temporary_directory(tmp_path):
    ctx = core.build_context(True)
    for renderer in (visuals.scene1, visuals.scene4, visuals.scene5):
        record = renderer(ctx, tmp_path)
        for name in record["files"]:
            path = tmp_path / name
            assert path.stat().st_size > 1000
            if path.suffix in (".png", ".gif"):
                with Image.open(path) as image:
                    assert image.width >= 800 and image.height >= 400
                    if path.suffix == ".gif":
                        assert image.n_frames >= 6
                        for frame in range(image.n_frames):
                            image.seek(frame); image.load()
            else:
                html = path.read_text(encoding="utf-8")
                assert 'id="user-select"' in html and 'id="epoch-slider"' in html
                assert '<script src=' not in html
                assert 'cdn.plot.ly/' not in html.split('const lab=')[-1]


def test_cli_list_and_independent_scene(tmp_path):
    result = subprocess.run([sys.executable, "-X", "utf8", str(TOPIC / "visualize_day42.py"), "--list"],
                            capture_output=True, text=True, check=True, encoding="utf-8")
    assert len(result.stdout.strip().splitlines()) == 10
    assert cli.main(["--scene", "2", "--fast", "--output-dir", str(tmp_path)]) == 0
    catalog = json.loads((tmp_path / "catalog.json").read_text(encoding="utf-8"))
    assert list(catalog["scenes"]) == ["2"]
    assert (tmp_path / "index.html").is_file()
    assert len(list(tmp_path.glob("run-*.json"))) == 1

"""Numerical invariants and selective/offline entrypoint checks for the visual lab."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import visualize_day30 as lab
from from_scratch import split_gain


def test_ordered_statistics_hand_calculation_and_self_exclusion():
    stats, details = lab.ordered_statistics(["Rio", "SP", "Rio", "Rio"], [1, 0, 0, 1], [0, 1, 2, 3], prior=0.5, strength=2)
    np.testing.assert_allclose(stats, [0.5, 0.5, 2 / 3, 0.5])
    assert details[2]["previous_sum"] == 1
    assert details[2]["previous_count"] == 1
    changed, _ = lab.ordered_statistics(["Rio", "SP", "Rio", "Rio"], [1, 0, 1, 1], [0, 1, 2, 3], prior=0.5, strength=2)
    np.testing.assert_array_equal(stats[:3], changed[:3])
    assert stats[3] != changed[3]


def test_ordered_statistics_outputs_follow_original_rows_not_permutation_positions():
    stats, details = lab.ordered_statistics(["A", "A", "A"], [0, 1, 1], [2, 0, 1], prior=0.5, strength=2)
    np.testing.assert_allclose(stats, [2 / 3, 0.5, 0.5])
    assert [row["row"] for row in details] == [2, 0, 1]


@pytest.mark.parametrize("order, prior, strength", [([0, 0], 0.5, 2), ([0, 2], 0.5, 2),
    ([0.0, 1.0], 0.5, 2), ([0, 1], -1, 2), ([0, 1], 0.5, 0)])
def test_ordered_statistics_reject_invalid_permutation_or_prior(order, prior, strength):
    with pytest.raises(ValueError):
        lab.ordered_statistics(["A", "A"], [0, 1], order, prior, strength)


def test_goss_retains_largest_gradients_and_reweights_small_sample():
    gradients = np.linspace(-0.95, 0.8, 100)
    groups, weights = lab.goss_sample(gradients)
    expected_high = np.argsort(-abs(gradients), kind="stable")[:20]
    assert set(np.flatnonzero(groups == 2)) == set(expected_high)
    assert (groups == 1).sum() == 20
    assert (groups == 0).sum() == 60
    np.testing.assert_array_equal(weights[groups == 2], np.ones(20))
    np.testing.assert_array_equal(weights[groups == 1], np.full(20, 4))
    assert weights.sum() == 100
    other_groups, other_weights = lab.goss_sample(gradients)
    np.testing.assert_array_equal(groups, other_groups)
    np.testing.assert_array_equal(weights, other_weights)


@pytest.mark.parametrize("rates", [(0, 0.2), (0.9, 0.5), (0.2, -1), (np.nan, 0.2)])
def test_goss_rejects_invalid_sampling_fractions(rates):
    with pytest.raises(ValueError):
        lab.goss_sample(np.arange(100), *rates)


def test_histogram_conserves_totals_and_matches_raw_boundary_gain():
    x = np.array([0, 0.2, 0.5, 0.7, 1])
    g = np.array([0.5, 0.5, -0.5, 0.5, -0.5])
    h = np.full(5, 0.25)
    bins, gb, hb = lab.histogram_statistics(x, g, h, [0, 0.5, 1])
    np.testing.assert_array_equal(bins, [0, 0, 1, 1, 1])
    assert gb.sum() == g.sum()
    assert hb.sum() == h.sum()
    mask = x < 0.5
    assert split_gain(gb[0], hb[0], gb[1], hb[1]) == pytest.approx(
        split_gain(g[mask].sum(), h[mask].sum(), g[~mask].sum(), h[~mask].sum()))


@pytest.mark.parametrize("x,g,h,edges", [([0, 1], [0], [1, 1], [0, 1]),
    ([0, 1], [1, 1], [-1, 1], [0, 1]), ([0, 1], [1, 1], [1, 1], [1, 0]),
    ([-0.1, 1], [1, 1], [1, 1], [0, 1])])
def test_histogram_rejects_misaligned_arrays_or_invalid_bins(x, g, h, edges):
    with pytest.raises(ValueError):
        lab.histogram_statistics(x, g, h, edges)


def test_heading_reuses_caption_instead_of_accumulating_frame_text():
    fig, ax = plt.subplots()
    lab.heading(fig, "First", "Old caption")
    for i in range(5):
        lab.heading(fig, f"Step {i}", f"Caption {i}")
    assert len(fig.texts) == 2
    assert fig._day30_caption.get_text() == "Caption 4"
    plt.close(fig)


def test_missing_optional_packages_skip_only_dependent_view(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(lab, "OUTPUT_DIR", tmp_path / "visuals")
    monkeypatch.setitem(lab.OPTIONAL_AVAILABLE, "plotly", False)
    assert lab.main(["--only", "gradient_hessian", "decision_surface"]) == 0
    output = capsys.readouterr().out
    assert "[OK] 02_gradient_hessian_geometry.png" in output
    assert "[SKIP] 11_decision_surface_3d.html" in output
    assert "python -m pip install plotly" in output
    assert "![Gradient Hessian](visuals/02_gradient_hessian_geometry.png)" in output
    assert "[Open the offline" not in output
    assert not plt.get_fignums()
    with Image.open(lab.OUTPUT_DIR / "02_gradient_hessian_geometry.png") as picture:
        picture.verify()
    record = json.loads(next(lab.OUTPUT_DIR.glob("experiment_records_*.json")).read_text(encoding="utf-8"))
    assert record["records"]["gradient_hessian"]["review_status"] == "pending author review"


def test_unexpected_rendering_error_is_reported_and_returns_failure(monkeypatch, tmp_path, capsys):
    def broken(path):
        raise ValueError("deliberate rendering failure")
    monkeypatch.setattr(lab, "OUTPUT_DIR", tmp_path)
    monkeypatch.setitem(lab.VISUALIZATIONS, "gradient_hessian", ("02_gradient_hessian_geometry.png", broken, "test"))
    assert lab.main(["--only", "gradient_hessian"]) == 1
    assert "[ERROR] 02_gradient_hessian_geometry.png" in capsys.readouterr().out
    assert not plt.get_fignums()


def test_model_parameter_record_handles_nan_missing_sentinel():
    class Model:
        def get_params(self):
            return {"missing": float("nan"), "max_depth": 3}
    params = lab.documented_model_parameters(Model())
    assert params["missing"] == "NaN (library missing-value sentinel)"
    assert json.loads(json.dumps(params, allow_nan=False))["max_depth"] == 3

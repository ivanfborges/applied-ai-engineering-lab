"""Hand oracles, framework agreement, spatial counterexamples and CPU training."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import example
from example import SmallCNN, make_data, run_study
from from_scratch import correlate2d, output_size, pool2d, receptive_field


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_cross_correlation_hand_oracle_without_flipping():
    image = np.arange(1., 10.).reshape(3, 3)
    kernel = [[1., 2.], [3., 4.]]
    np.testing.assert_array_equal(correlate2d(image, kernel), [[37., 47.], [67., 77.]])
    step = np.tile([0., 0., 0., 1., 1.], (5, 1))
    edge = np.tile([-1., 0., 1.], (3, 1))
    np.testing.assert_array_equal(correlate2d(step, edge), np.tile([0., 3., 3.], (3, 1)))


@pytest.mark.parametrize("shape,kernel_shape,stride,padding,dilation", [
    ((5, 7), (2, 3), 1, 0, 1), ((7, 8), (3, 2), 2, 1, 1),
    ((8, 9), (2, 3), 2, 2, 2), ((3, 4), (3, 3), 1, 1, 1),
])
def test_numpy_matches_torch_with_rectangles_stride_padding_and_dilation(
        shape, kernel_shape, stride, padding, dilation):
    rng = np.random.default_rng(59)
    image = rng.normal(size=shape)
    kernel = rng.normal(size=kernel_shape)
    actual = correlate2d(image, kernel, stride=stride, padding=padding, dilation=dilation)
    expected = F.conv2d(torch.tensor(image)[None, None], torch.tensor(kernel)[None, None],
                        stride=stride, padding=padding, dilation=dilation)[0, 0].numpy()
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-14)


def test_pooling_hand_oracle_and_negative_values():
    image = np.array([[1., 3., -1., -2.], [2., 4., -3., -4.]])
    np.testing.assert_array_equal(pool2d(image), [[4., -1.]])
    np.testing.assert_array_equal(pool2d(image, mode="average"), [[2.5, -2.5]])


@pytest.mark.parametrize("mode", ["max", "average"])
def test_pooling_matches_torch_and_drops_trailing_pixels(mode):
    image = np.arange(-15., 20.).reshape(5, 7)
    function = F.max_pool2d if mode == "max" else F.avg_pool2d
    expected = function(torch.tensor(image)[None, None], 2, stride=2)[0, 0].numpy()
    np.testing.assert_allclose(pool2d(image, mode=mode), expected)


def test_output_size_and_receptive_field_hand_oracles():
    assert output_size(16, 3, padding=1) == 16
    assert output_size(16, 2, stride=2) == 8
    assert output_size(9, 3, stride=2, padding=1, dilation=2) == 4
    assert receptive_field([]) == (1, 1)
    assert receptive_field([(3, 1, 1), (2, 2, 1)]) == (4, 2)
    assert receptive_field([(3, 1, 1), (2, 2, 1), (3, 1, 1)]) == (8, 2)
    assert receptive_field([(3, 1, 2), (3, 2, 1)]) == (7, 2)


def test_receptive_field_span_agrees_with_autograd_on_linear_path():
    x = torch.ones(1, 1, 20, 20, dtype=torch.float64, requires_grad=True)
    kernel = torch.ones(1, 1, 3, 3, dtype=torch.float64)
    first = F.conv2d(x, kernel)
    pooled = F.avg_pool2d(first, 2)
    final = F.conv2d(pooled, kernel)
    final[0, 0, 1, 1].backward()
    locations = torch.nonzero(x.grad[0, 0])
    spans = locations.max(0).values - locations.min(0).values + 1
    assert spans.tolist() == [8, 8]
    assert len(locations) == 64


def test_pooling_is_not_invariant_to_a_one_pixel_shift():
    original = np.zeros((4, 4))
    shifted = original.copy()
    original[1, 1] = 1
    shifted[1, 2] = 1
    assert not np.array_equal(pool2d(original), pool2d(shifted))


@pytest.mark.parametrize("call", [
    lambda: output_size(0, 3), lambda: output_size(4, True),
    lambda: output_size(4, 2, stride=0), lambda: output_size(4, 2, padding=-1),
    lambda: output_size(4, 2, dilation=0), lambda: output_size(2, 5),
    lambda: output_size(4.5, 2), lambda: correlate2d([], [[1.]]),
    lambda: correlate2d([[np.nan]], [[1.]]),
    lambda: correlate2d([[1.]], [[np.inf]]),
    lambda: correlate2d([[1j]], [[1.]]), lambda: pool2d([[1.]], mode="median"),
    lambda: pool2d([[1.]], kernel=2),
    lambda: receptive_field([(3, -1, 1)]),
])
def test_invalid_numerical_inputs(call):
    with pytest.raises(ValueError):
        call()


def test_numerical_overflow_is_explicit():
    with pytest.raises(FloatingPointError):
        correlate2d([[1e308]], [[1e308]])


def test_data_geometry_balance_seeds_and_equal_signal_mass():
    x, y = make_data(128, 59, noise=0)
    repeat_x, repeat_y = make_data(128, 59, noise=0)
    torch.testing.assert_close(x, repeat_x, rtol=0, atol=0)
    torch.testing.assert_close(y, repeat_y, rtol=0, atol=0)
    assert x.shape == (128, 1, 16, 16) and x.dtype == torch.float32
    assert y.dtype == torch.int64 and torch.bincount(y).tolist() == [64, 64]
    assert torch.all(x.sum(dim=(1, 2, 3)) == 27)
    for image, label in zip(x[:, 0], y):
        occupied = torch.nonzero(image)
        dimensions = occupied.max(0).values - occupied.min(0).values + 1
        assert dimensions.tolist() == ([9, 3] if label == 0 else [3, 9])
    other_x, _ = make_data(128, 60, noise=0)
    assert not torch.equal(x, other_x)


def test_model_shapes_parameter_count_and_rng_isolation():
    rng_state = torch.get_rng_state().clone()
    model = SmallCNN()
    assert torch.equal(torch.get_rng_state(), rng_state)
    images, _ = make_data(4, 59)
    activated, pooled = model.feature_maps(images)
    assert activated.shape == (4, 8, 16, 16)
    assert pooled.shape == (4, 8, 8, 8)
    assert model(images).shape == (4, 2)
    assert sum(p.numel() for p in model.conv.parameters()) == 80
    assert sum(p.numel() for p in model.parameters()) == 98


@pytest.mark.parametrize("images", [
    torch.zeros(1, 16, 16), torch.zeros(1, 3, 16, 16),
    torch.zeros(1, 1, 15, 16), torch.zeros(0, 1, 16, 16),
    torch.zeros(1, 1, 16, 16, dtype=torch.float64), torch.full((1, 1, 16, 16), float("nan")),
])
def test_model_rejects_bad_batches(images):
    with pytest.raises(ValueError):
        SmallCNN()(images)


@pytest.mark.parametrize("kwargs", [
    {"epochs": 0}, {"batch_size": True}, {"train_size": 1},
    {"test_size": 0}, {"learning_rate": 0}, {"learning_rate": np.nan},
    {"learning_rate": "fast"}, {"learning_rate": np.bool_(True)},
    {"seed": -1}, {"seed": 2**32}, {"noise": -0.1}, {"noise": np.inf},
    {"noise": "0.1"}, {"noise": 1j},
])
def test_study_rejects_bad_configuration(kwargs):
    with pytest.raises(ValueError):
        run_study(**kwargs)


def test_largest_documented_study_seed_accepts_all_derived_seeds():
    report = run_study(seed=2**32 - 4, train_size=2, test_size=2, epochs=1, batch_size=2)
    assert report["configuration"]["shuffle_seed"] == 2**32 - 1
    assert np.isfinite(report["results"]["test_cross_entropy"])


@pytest.mark.integration
def test_cpu_training_updates_learn_and_repeat():
    config = dict(seed=5, train_size=64, test_size=32, epochs=12, batch_size=16)
    first = run_study(**config)
    second = run_study(**config)
    assert first["results"] == second["results"]
    result = first["results"]
    assert result["epoch_end_train_cross_entropy"][-1] < result["initial_train_cross_entropy"]
    assert result["final_train_accuracy"] > 0.8
    assert result["test_accuracy"] > result["train_majority_baseline_test_accuracy"]
    matrix = np.array(result["test_confusion_true_rows_predicted_columns"])
    assert matrix.sum() == 32
    assert matrix.trace() / 32 == pytest.approx(result["test_accuracy"])
    assert first["interpretation_status"] == "author review required"


@pytest.mark.integration
def test_test_labels_do_not_change_fitting(monkeypatch):
    config = dict(seed=5, train_size=32, test_size=16, epochs=2, batch_size=16)
    original = run_study(**config)

    def flipped_test_labels(n, seed, noise):
        images, labels = make_data(n, seed, noise)
        return images, 1 - labels if seed == 6 else labels

    monkeypatch.setattr(example, "make_data", flipped_test_labels)
    altered = run_study(**config)
    for key in ("initial_train_cross_entropy", "epoch_end_train_cross_entropy", "final_train_accuracy"):
        assert original["results"][key] == altered["results"][key]
    original_matrix = original["results"]["test_confusion_true_rows_predicted_columns"]
    altered_matrix = altered["results"]["test_confusion_true_rows_predicted_columns"]
    assert altered_matrix == original_matrix[::-1]

"""Gate oracles, framework agreement, sequence boundaries and selection logic."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from example import SequenceClassifier, evaluate, fit_model, make_data, run_study
from from_scratch import SimpleLSTMCell, sigmoid


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_sigmoid_large_logits_and_symmetry():
    with np.errstate(over="raise", invalid="raise"):
        actual = sigmoid([-1000, -2, 0, 2, 1000])
    np.testing.assert_allclose(actual + actual[::-1], 1)
    assert actual[0] == 0 and actual[-1] == 1 and actual[2] == 0.5
    with pytest.raises(ValueError):
        sigmoid([np.nan])


def test_zero_weight_cell_has_exact_retention_and_exposure():
    cell = SimpleLSTMCell(2, 2, forget_bias=0)
    cell.W[:] = 0
    h, c, gates = cell.step([1, -1], [0.8, -0.7], [2, -2])
    np.testing.assert_allclose(c, [1, -1])
    np.testing.assert_allclose(h, 0.5 * np.tanh([1, -1]))
    for gate in ("i", "f", "o"):
        np.testing.assert_array_equal(gates[gate], [0.5, 0.5])
    np.testing.assert_array_equal(gates["g"], [0, 0])


def test_forget_gate_direct_path_derivative_with_zero_weights():
    cell = SimpleLSTMCell(1, 1, forget_bias=2)
    cell.W[:] = 0
    epsilon = 1e-6
    plus = cell.step([0], [0], [0.4 + epsilon])[1]
    minus = cell.step([0], [0], [0.4 - epsilon])[1]
    np.testing.assert_allclose((plus - minus) / (2 * epsilon), sigmoid(2), atol=1e-10)


def test_numpy_lstm_matches_pytorch_over_multiple_steps_and_nonzero_states():
    cell = SimpleLSTMCell(2, 3)
    reference = torch.nn.LSTMCell(2, 3, dtype=torch.float64)
    with torch.no_grad():
        reference.weight_ih.copy_(torch.from_numpy(cell.W[:, :2]))
        reference.weight_hh.copy_(torch.from_numpy(cell.W[:, 2:]))
        # Test the sum of both bias vectors, not just one zero bias.
        reference.bias_ih.copy_(torch.from_numpy(cell.b / 3))
        reference.bias_hh.copy_(torch.from_numpy(2 * cell.b / 3))
    rng = np.random.default_rng(60)
    h, c = rng.normal(size=3), rng.normal(size=3)
    th, tc = torch.from_numpy(h), torch.from_numpy(c)
    with torch.no_grad():
        for x in rng.normal(size=(5, 2)):
            h, c, _ = cell.step(x, h, c)
            th, tc = reference(torch.from_numpy(x), (th, tc))
            np.testing.assert_allclose(h, th.numpy(), atol=2e-14, rtol=2e-14)
            np.testing.assert_allclose(c, tc.numpy(), atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize("x,h,c", [([0], [0, 0], [0, 0]),
                                    ([0, 0], [np.nan, 0], [0, 0]),
                                    ([0, 0], [0, 0], [[0, 0]])])
def test_numpy_rejects_invalid_vectors(x, h, c):
    with pytest.raises(ValueError):
        SimpleLSTMCell(2, 2).step(x, h, c)


@pytest.mark.parametrize("kwargs", [{"input_size": 0}, {"hidden_size": True},
                                    {"seed": -1}, {"forget_bias": np.inf}])
def test_numpy_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        SimpleLSTMCell(**kwargs)


def test_data_contract_and_independent_seed_streams():
    x, y = make_data(101, 7, 60)
    repeated_x, repeated_y = make_data(101, 7, 60)
    other_x, other_y = make_data(101, 7, 61)
    assert x.shape == (101, 7, 2) and y.shape == (101,)
    assert x.dtype == y.dtype == torch.float32
    torch.testing.assert_close(x, repeated_x, rtol=0, atol=0)
    torch.testing.assert_close(y, repeated_y, rtol=0, atol=0)
    assert not torch.equal(x, other_x) and not torch.equal(y, other_y)
    torch.testing.assert_close(x[:, 0, 0], 2 * y - 1)
    assert torch.all(x[:, 0, 1] == 1) and torch.all(x[:, 1:, 1] == 0)


@pytest.mark.parametrize("args", [(0, 10, 60), (10, 1, 60), (10, 10, -1),
                                 (10, 10, 2**32), (True, 10, 60)])
def test_data_rejects_invalid_configuration(args):
    with pytest.raises(ValueError):
        make_data(*args)


@pytest.mark.parametrize("kind,factor", [("rnn", 1), ("lstm", 4), ("gru", 3)])
def test_parameter_count_logits_and_chunked_state_equivalence(kind, factor):
    model = SequenceClassifier(kind, hidden=5)
    x, _ = make_data(4, 8, 60)
    assert sum(p.numel() for p in model.parameters()) == factor * 5 * (2 + 5 + 2) + 5 + 1
    model.eval()
    with torch.no_grad():
        full_output, full_state = model.encode(x)
        _, state = model.encode(x[:, :3])
        tail, chunk_state = model.encode(x[:, 3:], state)
        _, reset_state = model.encode(x[:, 3:])
        torch.testing.assert_close(tail, full_output[:, 3:], rtol=2e-6, atol=2e-7)
        states = zip(full_state, chunk_state) if kind == "lstm" else [(full_state, chunk_state)]
        for expected, actual in states:
            torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
        hidden = full_state[0] if kind == "lstm" else full_state
        reset_hidden = reset_state[0] if kind == "lstm" else reset_state
        assert not torch.equal(hidden, reset_hidden)
        torch.testing.assert_close(model(x), model.head(hidden[-1]).squeeze(-1))
        # An independent call starts from zeros instead of retaining earlier state.
        torch.testing.assert_close(model(x), model(x), rtol=0, atol=0)
    assert model(x).shape == (4,)
    logits = model(x)
    logits.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("x", [torch.zeros(2, 2), torch.zeros(2, 0, 2),
                              torch.zeros(2, 3, 2, dtype=torch.float64),
                              torch.full((2, 3, 2), float("nan"))])
def test_model_rejects_invalid_input(x):
    with pytest.raises(ValueError):
        SequenceClassifier("gru")(x)


def test_initialization_preserves_caller_rng():
    before = torch.random.get_rng_state().clone()
    SequenceClassifier("lstm")
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)


def test_checkpoint_is_validation_minimum_and_replay_restores_its_weights():
    train, validation = make_data(32, 4, 60), make_data(16, 4, 61)
    model, report = fit_model("gru", train, validation, hidden=4, epochs=4, batch_size=8)
    best = min(report["history"], key=lambda row: row["validation_bce"])
    assert report["selected_epoch"] == best["epoch"]
    assert evaluate(model, *validation)["bce"] == best["validation_bce"]
    replay, repeated = fit_model("gru", train, validation, hidden=4,
                                 epochs=best["epoch"], batch_size=8)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, replay.state_dict()[name], rtol=0, atol=0)
    assert repeated["selected_validation"] == report["selected_validation"]


def test_small_study_is_deterministic_and_reports_controls():
    config = dict(cells=("rnn", "lstm", "gru"), lengths=(3,), hidden=4,
                  epochs=2, train_size=32, validation_size=16, test_size=16, batch_size=8)
    first, second = run_study(**config), run_study(**config)
    assert first["results"] == second["results"]
    assert first["controls"] == second["controls"]
    assert first["controls"][0]["first_step_sign_test_accuracy"] == 1.0
    assert first["interpretation_status"] == "author review required"
    for row in first["results"]:
        assert 0 <= row["test"]["accuracy"] <= 1 and np.isfinite(row["test"]["bce"])


@pytest.mark.parametrize("kwargs", [{"cells": ()}, {"cells": ("gru", "gru")},
                                    {"cells": ("transformer",)}, {"lengths": ()},
                                    {"lengths": (2, 2)}, {"epochs": 0},
                                    {"seed": 2**32 - 4}, {"learning_rate": np.nan},
                                    {"clip_norm": 0}, {"batch_size": False}])
def test_study_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        run_study(**kwargs)

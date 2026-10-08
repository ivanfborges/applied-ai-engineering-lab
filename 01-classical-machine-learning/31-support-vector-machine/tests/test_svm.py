"""Objective, subgradient, analytic optimum and fold-local pipeline checks."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC


TOPIC = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, TOPIC / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scratch = load_module("day31_scratch", "from_scratch.py")
example = load_module("day31_example", "example.py")
objective = scratch.objective_and_subgradient


def test_subgradient_normalizes_by_all_rows():
    X, y, w = np.array([[-2.0], [0.5], [3.0]]), np.array([-1, 1, 1]), np.array([1.0])
    loss, grad_w, grad_b = objective(X, y, w, 0.0, C=6.0)
    assert loss == pytest.approx(1.5)
    np.testing.assert_allclose(grad_w, [0.0])
    assert grad_b == pytest.approx(-2.0)


def test_subgradient_matches_finite_differences_away_from_hinge_kinks():
    X = np.array([[-2, 0.4], [0.2, -0.7], [3, 1.0], [-0.3, 1.2]])
    y = np.array([-1, 1, 1, -1])
    theta = np.array([0.6, -0.2, 0.15])
    _, grad_w, grad_b = objective(X, y, theta[:2], theta[2], C=2.3)
    numerical = []
    for axis in range(3):
        delta = np.zeros(3)
        delta[axis] = 1e-6
        plus, minus = theta + delta, theta - delta
        numerical.append((objective(X, y, plus[:2], plus[2], 2.3)[0]
                          - objective(X, y, minus[:2], minus[2], 2.3)[0]) / 2e-6)
    np.testing.assert_allclose(np.r_[grad_w, grad_b], numerical, atol=1e-8)


def test_chosen_kink_subgradient_obeys_convex_supporting_inequality():
    X, y, w = np.array([[-1.0], [1.0]]), np.array([-1, 1]), np.array([1.0])
    loss, grad_w, grad_b = objective(X, y, w, 0.0, 2.0)
    for dw, db in [(-0.1, 0.0), (0.1, 0.0), (0.0, 0.2), (-0.1, -0.2)]:
        perturbed = objective(X, y, w + dw, db, 2.0)[0]
        assert perturbed >= loss + grad_w[0] * dw + grad_b * db - 1e-12


def test_inactive_rows_still_receive_regularization_gradient():
    loss, grad_w, grad_b = objective([[-2], [2]], [-1, 1], [1], 0, 3)
    assert loss == pytest.approx(0.5)
    np.testing.assert_array_equal(grad_w, [1])
    assert grad_b == 0


@pytest.mark.parametrize("C,expected_w", [(0.3, 0.3), (2.0, 1.0)])
def test_solver_approaches_closed_form_symmetric_optimum(C, expected_w):
    # J(w,0) = w^2/2 + C max(0,1-w), whose minimizer is min(C,1).
    model = scratch.LinearSVM(C=C, epochs=6000).fit([[-1], [1]], [-1, 1])
    np.testing.assert_allclose(model.coef_, [expected_w], atol=0.005)
    assert model.intercept_ == pytest.approx(0.0)
    expected_loss = 0.5 * expected_w**2 + C * max(0, 1 - expected_w)
    assert model.objective_ == pytest.approx(expected_loss, abs=0.005)
    np.testing.assert_array_equal(model.predict([[-1], [1]]), [-1, 1])


def test_mean_hinge_matches_svc_when_C_is_divided_by_n():
    X, y, C = np.array([[-2.0], [-1], [0.4], [1.5], [2]]), np.array([-1, -1, 1, 1, 1]), 2.0
    model = scratch.LinearSVM(C=C, epochs=10000).fit(X, y)
    reference = SVC(kernel="linear", C=C / len(X), tol=1e-9).fit(X, y)
    reference_loss = objective(X, y, reference.coef_[0], reference.intercept_[0], C)[0]
    assert abs(model.objective_ - reference_loss) < 0.01
    np.testing.assert_array_equal(model.predict(X), reference.predict(X))


def test_refitting_resets_parameters_and_history():
    model = scratch.LinearSVM(epochs=50).fit([[-1], [1]], [-1, 1])
    model.fit([[-1], [1]], [1, -1])
    fresh = scratch.LinearSVM(epochs=50).fit([[-1], [1]], [1, -1])
    np.testing.assert_array_equal(model.coef_, fresh.coef_)
    np.testing.assert_array_equal(model.loss_history_, fresh.loss_history_)
    assert model.intercept_ == fresh.intercept_


@pytest.mark.parametrize("kwargs", [
    {"C": 0}, {"C": np.nan}, {"C": True}, {"C": [1]},
    {"learning_rate": -1}, {"learning_rate": np.inf},
    {"epochs": 0}, {"epochs": 1.5}, {"epochs": True},
])
def test_invalid_solver_parameters(kwargs):
    with pytest.raises(ValueError):
        scratch.LinearSVM(**kwargs)


@pytest.mark.parametrize("X,y", [
    ([], []), ([1, 2], [-1, 1]), ([[np.nan], [1]], [-1, 1]),
    ([[1], [2]], [0, 1]), ([[1], [2]], [-1]),
    ([[1], [2]], [1, 1]), ([[1], [2]], [-1, np.inf]),
])
def test_invalid_training_inputs(X, y):
    with pytest.raises(ValueError):
        scratch.LinearSVM().fit(X, y)


def test_prediction_requires_fit_and_matching_finite_features():
    model = scratch.LinearSVM(epochs=10)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict([[1]])
    model.fit([[-1], [1]], [-1, 1])
    for X in ([[1, 2]], [[np.inf]], []):
        with pytest.raises(ValueError):
            model.predict(X)
    assert model.predict([[0]])[0] == 1


@pytest.mark.parametrize("w,b", [([1, 2], 0), ([np.nan], 0), ([1], np.inf), ([1], [0])])
def test_invalid_objective_parameters(w, b):
    with pytest.raises(ValueError):
        objective([[-1], [1]], [-1, 1], w, b)


def test_scaler_is_fitted_on_each_training_fold_and_full_training_refit(monkeypatch):
    X, y = example.make_moons(n_samples=40, noise=0.2, random_state=3)
    recorded = []
    original_fit = example.StandardScaler.fit

    def record_fit(self, values, *args, **kwargs):
        recorded.append(np.asarray(values).copy())
        return original_fit(self, values, *args, **kwargs)

    monkeypatch.setattr(example.StandardScaler, "fit", record_fit)
    search = example.tune_rbf(X, y, C_values=[1], gamma_values=[0.5], folds=2)
    folds = example.StratifiedKFold(2, shuffle=True, random_state=42)
    expected = [X[train] for train, _ in folds.split(X, y)] + [X]
    assert len(recorded) == len(expected)
    for actual, desired in zip(recorded, expected):
        np.testing.assert_array_equal(actual, desired)
    scaler = search.best_estimator_.named_steps["scaler"]
    np.testing.assert_allclose(scaler.mean_, X.mean(axis=0))


def test_rbf_dual_decision_uses_scaled_support_vectors():
    X, y = example.make_moons(n_samples=50, noise=0.2, random_state=7)
    model = example.tune_rbf(X, y, C_values=[1], gamma_values=[0.5], folds=2).best_estimator_
    scaler, svm = model.named_steps["scaler"], model.named_steps["svm"]
    np.testing.assert_allclose(svm.support_vectors_, scaler.transform(X)[svm.support_])
    probe = np.array([[0, 0], [1, 1], [-0.5, 0.6]])
    kernels = rbf_kernel(scaler.transform(probe), svm.support_vectors_, gamma=0.5)
    reconstructed = kernels @ svm.dual_coef_[0] + svm.intercept_[0]
    np.testing.assert_allclose(model.decision_function(probe), reconstructed, atol=1e-12)


def test_invalid_search_parameter_raises_instead_of_silently_scoring_nan():
    X, y = example.make_moons(n_samples=20, random_state=0)
    with pytest.raises(ValueError):
        example.tune_rbf(X, y, C_values=[0], gamma_values=[1], folds=2)

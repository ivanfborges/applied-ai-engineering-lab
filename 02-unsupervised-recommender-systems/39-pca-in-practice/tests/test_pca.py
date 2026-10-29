"""PCA identities, subspace comparisons, invalid inputs and fit boundaries."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_validate
from threadpoolctl import threadpool_limits


TOPIC = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, TOPIC / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scratch = load_module("day39_scratch", "from_scratch.py")
example = load_module("day39_example", "example.py")


def test_hand_calculated_covariance_and_variance():
    a, b = np.sqrt(3.75), np.sqrt(0.75)
    X = np.array([[a, a], [-a, -a], [b, -b], [-b, b]])
    pca = scratch.CovariancePCA(1).fit(X)
    np.testing.assert_allclose(np.cov(X, rowvar=False), [[3, 2], [2, 3]])
    np.testing.assert_allclose(pca.eigenvalues_, [5, 1])
    np.testing.assert_allclose(pca.explained_variance_ratio_, [5 / 6])
    np.testing.assert_allclose(pca.components_[0], [1 / np.sqrt(2)] * 2)
    reconstructed = pca.inverse_transform(pca.transform(X))
    assert np.sum((X - reconstructed) ** 2) == pytest.approx(3)


@pytest.mark.parametrize("shape,k", [((30, 5), 2), ((6, 12), 3), ((20, 1), 1)])
def test_subspace_variance_and_reconstruction_match_full_svd(shape, k):
    X = np.random.default_rng(7).normal(size=shape) + 11
    educational = scratch.CovariancePCA(k).fit(X)
    library = PCA(n_components=k, svd_solver="full").fit(X)
    A, B = educational.components_, library.components_
    # Projectors survive arbitrary eigenvector signs and within-subspace rotations.
    np.testing.assert_allclose(A.T @ A, B.T @ B, atol=1e-11)
    np.testing.assert_allclose(educational.explained_variance_, library.explained_variance_, atol=1e-11)
    np.testing.assert_allclose(educational.explained_variance_ratio_, library.explained_variance_ratio_, atol=1e-11)
    np.testing.assert_allclose(educational.inverse_transform(educational.transform(X)),
                               library.inverse_transform(library.transform(X)), atol=1e-11)


def test_training_covariance_and_tail_sse_identity():
    X = np.random.default_rng(19).normal(size=(40, 4)) * [4, 2, 1, 0.1] + 9
    pca = scratch.CovariancePCA(2).fit(X)
    scores = pca.transform(X)
    np.testing.assert_allclose(pca.components_ @ pca.components_.T, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(np.cov(scores, rowvar=False),
                               np.diag(pca.explained_variance_), atol=1e-12)
    sse = np.sum((X - pca.inverse_transform(scores)) ** 2)
    assert sse == pytest.approx((len(X) - 1) * pca.eigenvalues_[2:].sum())


def test_heldout_uses_training_mean_and_nested_projection_errors_decrease():
    rng = np.random.default_rng(6)
    train = rng.normal(size=(30, 3)) + [5, 7, -8]
    test = rng.normal(size=(10, 3)) + [15, 2, -10]
    errors = []
    for k in (1, 2, 3):
        pca = scratch.CovariancePCA(k).fit(train)
        stored_mean = pca.mean_.copy()
        scores = pca.transform(test)
        np.testing.assert_allclose(scores, (test - train.mean(axis=0)) @ pca.components_.T)
        np.testing.assert_array_equal(pca.mean_, stored_mean)
        errors.append(np.sum((test - pca.inverse_transform(scores)) ** 2))
    assert np.all(np.diff(errors) <= 1e-10)
    assert errors[-1] < 1e-24


def test_constant_column_and_rank_deficiency_are_allowed():
    X = np.array([[1., 2., 7.], [2., 4., 7.], [3., 6., 7.], [4., 8., 7.]])
    pca = scratch.CovariancePCA(1).fit(X)
    assert pca.explained_variance_ratio_.sum() == pytest.approx(1)
    np.testing.assert_allclose(pca.inverse_transform(pca.transform(X)), X, atol=1e-12)


def test_tied_eigenvalues_compare_complete_tied_subspace():
    X = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    model = scratch.CovariancePCA(2).fit(X)
    np.testing.assert_allclose(model.components_.T @ model.components_, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(model.inverse_transform(model.transform(X)), X, atol=1e-12)
    # A one-axis comparison would be undefined: every unit direction is optimal.


def test_exact_low_variance_label_direction_is_discarded():
    X = np.array([[-100., -1.], [-100., 1.], [100., -1.], [100., 1.]])
    model = scratch.CovariancePCA(1).fit(X)
    scores = model.transform(X)
    assert model.explained_variance_ratio_.sum() > 0.999
    np.testing.assert_allclose(model.components_, [[1, 0]])
    np.testing.assert_allclose(scores[0], scores[1])
    np.testing.assert_allclose(scores[2], scores[3])


def test_pipeline_preprocessing_is_fold_local_and_threshold_is_train_fitted():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 4)) * [4, 2, 1, 0.3] + np.arange(60)[:, None] / 10
    y = np.tile([0, 1], 30)
    folds = list(StratifiedKFold(3, shuffle=True, random_state=42).split(X, y))
    with threadpool_limits(limits=1):
        result = cross_validate(example.build_model(0.95), X, y, cv=folds,
                                return_estimator=True, error_score="raise")
    for (train, heldout), model in zip(folds, result["estimator"]):
        scaler, pca = model.named_steps["scaler"], model.named_steps["pca"]
        np.testing.assert_allclose(scaler.mean_, X[train].mean(axis=0))
        assert scaler.n_samples_seen_ == len(train)
        reference = PCA(0.95, svd_solver="full").fit(scaler.transform(X[train]))
        assert pca.n_components_ == reference.n_components_
        np.testing.assert_allclose(pca.components_, reference.components_)
        before = pca.components_.copy()
        model.predict(X[heldout] + 1000)
        np.testing.assert_array_equal(pca.components_, before)


@pytest.mark.parametrize("X", [[], [1, 2], [[1, 2]], np.empty((3, 0)),
                                   [[1, np.nan], [2, 3]], [[1, np.inf], [2, 3]],
                                   np.ones((3, 2)), [[1j], [2j]], [["a"], ["b"]],
                                   [[1], [2, 3]], [[1e200], [-1e200]]])
def test_invalid_fit_data_raise_explicit_errors(X):
    with pytest.raises(ValueError):
        scratch.CovariancePCA(1).fit(X)


@pytest.mark.parametrize("k", [0, -1, 1.5, True, np.bool_(False), 3, "1", None])
def test_invalid_component_counts(k):
    with pytest.raises(ValueError, match="n_components"):
        scratch.CovariancePCA(k).fit([[1, 2], [2, 3], [3, 5]])


def test_transform_contracts():
    pca = scratch.CovariancePCA(1)
    with pytest.raises(ValueError, match="fit PCA"):
        pca.transform([[1, 2]])
    with pytest.raises(ValueError, match="fit PCA"):
        pca.inverse_transform([[1]])
    pca.fit([[1, 2], [2, 3], [3, 5]])
    with pytest.raises(ValueError, match="feature count"):
        pca.transform([[1, 2, 3]])
    with pytest.raises(ValueError, match="retained component"):
        pca.inverse_transform([[1, 2]])
    with pytest.raises(ValueError, match="finite"):
        pca.transform([[1, np.inf]])

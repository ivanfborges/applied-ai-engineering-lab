'''Deterministic simulations for the Day 16 visual learning laboratory.'''

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification, make_moons
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler


SEED = 16
LIFECYCLE_STAGES = (
    'Business problem',
    'Data collection',
    'Feature engineering',
    'Train / validation / test',
    'Baseline',
    'Training',
    'Model selection',
    'Final evaluation',
    'Deployment',
    'Monitoring',
    'Retraining / feedback',
)


@dataclass(frozen=True)
class SplitComparison:
    '''Models, masks, and scores for random and temporal validation.'''

    data: pd.DataFrame
    random_train: np.ndarray
    random_test: np.ndarray
    temporal_train: np.ndarray
    temporal_test: np.ndarray
    random_model: LogisticRegression
    temporal_model: LogisticRegression
    random_auc: float
    temporal_auc: float


def _finite_float(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite.')
    return result


def sigmoid(values: np.ndarray | float) -> np.ndarray:
    '''Return a stable logistic transformation.'''
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError('values must be finite.')
    result = np.empty_like(array)
    positive = array >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exponential = np.exp(array[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def generate_temporal_drift(
    n_samples: int = 1_200,
    drift_magnitude: float = 1.3,
    seed: int = SEED,
) -> pd.DataFrame:
    '''Generate timestamped observations whose class boundary rotates over time.'''
    if n_samples < 300:
        raise ValueError('n_samples must be at least 300.')
    drift_magnitude = _finite_float(drift_magnitude, 'drift_magnitude')
    if not 0 <= drift_magnitude <= 2.5:
        raise ValueError('drift_magnitude must be in [0, 2.5].')
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, 1.0, n_samples)
    x1 = rng.normal(size=n_samples)
    x2 = rng.normal(size=n_samples)
    angle = drift_magnitude * time
    logits = 1.6 * (np.cos(angle) * x1 + np.sin(angle) * x2)
    target = rng.binomial(1, sigmoid(logits))
    return pd.DataFrame({'time': time, 'x1': x1, 'x2': x2, 'target': target})


def compare_random_and_temporal_splits(
    drift_magnitude: float = 1.3,
    seed: int = SEED,
) -> SplitComparison:
    '''Evaluate the same estimator under random and forward-looking boundaries.'''
    data = generate_temporal_drift(drift_magnitude=drift_magnitude, seed=seed)
    indices = np.arange(len(data))
    random_train, random_test = train_test_split(
        indices,
        test_size=0.25,
        stratify=data['target'],
        random_state=seed,
    )
    cutoff = int(0.75 * len(data))
    temporal_train = indices[:cutoff]
    temporal_test = indices[cutoff:]
    features = data[['x1', 'x2']].to_numpy()
    target = data['target'].to_numpy()

    random_model = LogisticRegression(random_state=seed).fit(
        features[random_train], target[random_train]
    )
    temporal_model = LogisticRegression(random_state=seed).fit(
        features[temporal_train], target[temporal_train]
    )
    return SplitComparison(
        data=data,
        random_train=random_train,
        random_test=random_test,
        temporal_train=temporal_train,
        temporal_test=temporal_test,
        random_model=random_model,
        temporal_model=temporal_model,
        random_auc=float(
            roc_auc_score(
                target[random_test],
                random_model.predict_proba(features[random_test])[:, 1],
            )
        ),
        temporal_auc=float(
            roc_auc_score(
                target[temporal_test],
                temporal_model.predict_proba(features[temporal_test])[:, 1],
            )
        ),
    )


def group_leakage_experiment(
    n_groups: int = 100,
    rows_per_group: int = 8,
    seed: int = SEED,
) -> dict[str, object]:
    '''Compare row-level and entity-level splitting with repeated observations.'''
    if n_groups < 20 or rows_per_group < 2:
        raise ValueError('Use at least 20 groups and two rows per group.')
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), rows_per_group)
    entity_effect = rng.normal(scale=1.5, size=n_groups)
    activity = rng.normal(size=len(groups))
    logits = entity_effect[groups] + 0.25 * activity
    target = rng.binomial(1, sigmoid(logits))
    features = pd.DataFrame(
        {
            'entity': [f'customer_{value:03d}' for value in groups],
            'activity': activity,
        }
    )
    preprocessing = ColumnTransformer(
        (
            ('entity', OneHotEncoder(handle_unknown='ignore'), ['entity']),
            ('activity', StandardScaler(), ['activity']),
        )
    )

    row_train, row_test = train_test_split(
        np.arange(len(features)),
        test_size=0.25,
        stratify=target,
        random_state=seed,
    )
    group_split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    group_train, group_test = next(group_split.split(features, target, groups))

    def fit_score(train: np.ndarray, test: np.ndarray) -> tuple[Pipeline, float]:
        model = Pipeline(
            (
                ('preprocessing', preprocessing),
                ('model', LogisticRegression(max_iter=1_000, random_state=seed)),
            )
        ).fit(features.iloc[train], target[train])
        score = roc_auc_score(
            target[test], model.predict_proba(features.iloc[test])[:, 1]
        )
        return model, float(score)

    row_model, row_auc = fit_score(row_train, row_test)
    group_model, group_auc = fit_score(group_train, group_test)
    return {
        'features': features,
        'target': target,
        'groups': groups,
        'row_train': row_train,
        'row_test': row_test,
        'group_train': group_train,
        'group_test': group_test,
        'row_auc': row_auc,
        'group_auc': group_auc,
        'row_overlap': len(set(groups[row_train]) & set(groups[row_test])),
        'group_overlap': len(set(groups[group_train]) & set(groups[group_test])),
        'row_model': row_model,
        'group_model': group_model,
    }


def leakage_experiment(seed: int = SEED) -> dict[str, object]:
    '''Compare valid features with a target-derived post-outcome feature.'''
    features, target = make_classification(
        n_samples=1_400,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        class_sep=0.8,
        flip_y=0.04,
        random_state=seed,
    )
    train, test = train_test_split(
        np.arange(len(target)),
        test_size=0.30,
        stratify=target,
        random_state=seed,
    )
    rng = np.random.default_rng(seed)
    leaky_feature = target + rng.normal(scale=0.12, size=len(target))

    def score(matrix: np.ndarray) -> float:
        model = Pipeline(
            (
                ('scale', StandardScaler()),
                ('model', LogisticRegression(max_iter=1_000, random_state=seed)),
            )
        ).fit(matrix[train], target[train])
        return float(
            roc_auc_score(target[test], model.predict_proba(matrix[test])[:, 1])
        )

    return {
        'valid_auc': score(features),
        'leaky_auc': score(np.column_stack((features, leaky_feature))),
        'leaky_feature': leaky_feature,
        'target': target,
        'test': test,
    }


def baseline_experiment(
    positive_fraction: float = 0.08,
    seed: int = SEED,
) -> pd.DataFrame:
    '''Evaluate baselines and learned classifiers on an imbalanced problem.'''
    positive_fraction = _finite_float(positive_fraction, 'positive_fraction')
    if not 0.02 <= positive_fraction <= 0.35:
        raise ValueError('positive_fraction must be in [0.02, 0.35].')
    features, target = make_classification(
        n_samples=1_600,
        n_features=8,
        n_informative=5,
        n_redundant=2,
        weights=[1.0 - positive_fraction, positive_fraction],
        class_sep=1.0,
        random_state=seed,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.30,
        stratify=target,
        random_state=seed,
    )
    models = {
        'Majority class': DummyClassifier(strategy='most_frequent'),
        'Class prior': DummyClassifier(strategy='prior'),
        'Logistic regression': Pipeline(
            (
                ('scale', StandardScaler()),
                ('model', LogisticRegression(max_iter=1_000, random_state=seed)),
            )
        ),
        'Random forest': RandomForestClassifier(
            n_estimators=120,
            max_depth=6,
            min_samples_leaf=4,
            random_state=seed,
            n_jobs=1,
        ),
    }
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = model.predict(X_test)
        rows.append(
            {
                'model': name,
                'accuracy': accuracy_score(y_test, predictions),
                'balanced_accuracy': balanced_accuracy_score(y_test, predictions),
                'precision': precision_score(
                    y_test, predictions, zero_division=0
                ),
                'recall': recall_score(y_test, predictions, zero_division=0),
                'f1': f1_score(y_test, predictions, zero_division=0),
                'roc_auc': roc_auc_score(y_test, probabilities),
                'pr_auc': average_precision_score(y_test, probabilities),
            }
        )
    return pd.DataFrame(rows)


def threshold_experiment(seed: int = SEED) -> dict[str, object]:
    '''Fit one probability model and return held-out scores for policy analysis.'''
    features, target = make_classification(
        n_samples=1_800,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        weights=[0.78, 0.22],
        class_sep=1.05,
        random_state=seed,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.35,
        stratify=target,
        random_state=seed,
    )
    model = Pipeline(
        (
            ('scale', StandardScaler()),
            ('model', LogisticRegression(max_iter=1_000, random_state=seed)),
        )
    ).fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_test': y_test,
        'probabilities': probabilities,
    }


def threshold_metrics(
    target: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 5.0,
) -> dict[str, float | int | np.ndarray]:
    '''Calculate confusion counts, classification metrics, and decision cost.'''
    threshold = _finite_float(threshold, 'threshold')
    false_positive_cost = _finite_float(
        false_positive_cost, 'false_positive_cost'
    )
    false_negative_cost = _finite_float(
        false_negative_cost, 'false_negative_cost'
    )
    if not 0 <= threshold <= 1:
        raise ValueError('threshold must be in [0, 1].')
    if false_positive_cost < 0 or false_negative_cost < 0:
        raise ValueError('costs must be nonnegative.')
    target = np.asarray(target, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if target.shape != probabilities.shape or target.ndim != 1:
        raise ValueError('target and probabilities must be aligned vectors.')
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(target, predictions, labels=(0, 1)).ravel()
    return {
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'precision': float(precision_score(target, predictions, zero_division=0)),
        'recall': float(recall_score(target, predictions, zero_division=0)),
        'f1': float(f1_score(target, predictions, zero_division=0)),
        'predicted_positives': int(predictions.sum()),
        'cost': float(fp * false_positive_cost + fn * false_negative_cost),
        'predictions': predictions,
    }


def threshold_curve(
    target: np.ndarray,
    probabilities: np.ndarray,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 5.0,
) -> pd.DataFrame:
    '''Evaluate decision behavior on a bounded threshold grid.'''
    rows = []
    for threshold in np.linspace(0.01, 0.99, 99):
        metrics = threshold_metrics(
            target,
            probabilities,
            threshold,
            false_positive_cost,
            false_negative_cost,
        )
        rows.append(
            {
                'threshold': threshold,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'predicted_positives': metrics['predicted_positives'],
                'cost': metrics['cost'],
            }
        )
    return pd.DataFrame(rows)


def complexity_experiment(seed: int = SEED) -> dict[str, object]:
    '''Fit polynomial logistic models from underfit to high variance.'''
    features, target = make_moons(
        n_samples=520,
        noise=0.28,
        random_state=seed,
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        features,
        target,
        test_size=0.40,
        stratify=target,
        random_state=seed,
    )
    degrees = (1, 2, 3, 5, 9, 12)
    models = {}
    rows = []
    for degree in degrees:
        model = Pipeline(
            (
                ('polynomial', PolynomialFeatures(degree=degree, include_bias=False)),
                ('scale', StandardScaler()),
                (
                    'model',
                    LogisticRegression(
                        C=100.0,
                        max_iter=4_000,
                        random_state=seed,
                    ),
                ),
            )
        ).fit(X_train, y_train)
        models[degree] = model
        rows.append(
            {
                'degree': degree,
                'train_accuracy': accuracy_score(y_train, model.predict(X_train)),
                'validation_accuracy': accuracy_score(
                    y_validation, model.predict(X_validation)
                ),
            }
        )
    return {
        'X_train': X_train,
        'X_validation': X_validation,
        'y_train': y_train,
        'y_validation': y_validation,
        'models': models,
        'scores': pd.DataFrame(rows),
    }


def validation_search_experiment(
    repetitions: int = 180,
    validation_size: int = 180,
    true_accuracy: float = 0.70,
    seed: int = SEED,
) -> pd.DataFrame:
    '''Simulate winner's curse when equal-quality candidates share a holdout.'''
    if repetitions < 20 or validation_size < 30:
        raise ValueError('Use at least 20 repetitions and 30 validation rows.')
    true_accuracy = _finite_float(true_accuracy, 'true_accuracy')
    if not 0 < true_accuracy < 1:
        raise ValueError('true_accuracy must be in (0, 1).')
    rng = np.random.default_rng(seed)
    rows = []
    for candidate_count in (5, 10, 50, 100, 500):
        validation = rng.binomial(
            validation_size,
            true_accuracy,
            size=(repetitions, candidate_count),
        ) / validation_size
        winners = np.argmax(validation, axis=1)
        best_validation = validation[np.arange(repetitions), winners]
        independent_test = rng.binomial(
            validation_size,
            true_accuracy,
            size=(repetitions, candidate_count),
        ) / validation_size
        selected_test = independent_test[np.arange(repetitions), winners]
        rows.append(
            {
                'candidates': candidate_count,
                'best_validation_mean': best_validation.mean(),
                'selected_test_mean': selected_test.mean(),
                'best_validation_std': best_validation.std(ddof=1),
                'selected_test_std': selected_test.std(ddof=1),
                'true_accuracy': true_accuracy,
            }
        )
    return pd.DataFrame(rows)


def drift_experiment(
    covariate_shift: float = 1.0,
    concept_rotation: float = 1.0,
    seed: int = SEED,
) -> dict[str, object]:
    '''Separate a change in P(X) from a change in P(Y|X).'''
    covariate_shift = _finite_float(covariate_shift, 'covariate_shift')
    concept_rotation = _finite_float(concept_rotation, 'concept_rotation')
    if not 0 <= covariate_shift <= 2 or not 0 <= concept_rotation <= 2:
        raise ValueError('Drift controls must be in [0, 2].')
    rng = np.random.default_rng(seed)
    n_samples = 1_000
    train_features = rng.normal(size=(n_samples, 2))
    covariate_features = rng.normal(
        loc=(covariate_shift, -0.45 * covariate_shift),
        size=(n_samples, 2),
    )
    concept_features = rng.normal(size=(n_samples, 2))
    base_weights = np.array([1.5, -0.8])
    rotated_weights = np.array(
        [
            1.5 * np.cos(concept_rotation) + 0.8 * np.sin(concept_rotation),
            1.5 * np.sin(concept_rotation) - 0.8 * np.cos(concept_rotation),
        ]
    )
    train_target = rng.binomial(1, sigmoid(train_features @ base_weights))
    covariate_target = rng.binomial(1, sigmoid(covariate_features @ base_weights))
    concept_target = rng.binomial(
        1, sigmoid(concept_features @ rotated_weights)
    )
    model = LogisticRegression(random_state=seed).fit(
        train_features, train_target
    )

    def auc(features: np.ndarray, target: np.ndarray) -> float:
        return float(roc_auc_score(target, model.predict_proba(features)[:, 1]))

    return {
        'train_features': train_features,
        'covariate_features': covariate_features,
        'concept_features': concept_features,
        'train_target': train_target,
        'covariate_target': covariate_target,
        'concept_target': concept_target,
        'base_weights': base_weights,
        'rotated_weights': rotated_weights,
        'model': model,
        'train_auc': auc(train_features, train_target),
        'covariate_auc': auc(covariate_features, covariate_target),
        'concept_auc': auc(concept_features, concept_target),
        'wasserstein_x1': float(
            wasserstein_distance(train_features[:, 0], covariate_features[:, 0])
        ),
        'concept_wasserstein_x1': float(
            wasserstein_distance(train_features[:, 0], concept_features[:, 0])
        ),
    }


def cross_validation_layout(
    n_samples: int = 60,
    n_folds: int = 5,
) -> np.ndarray:
    '''Return a fold-by-observation matrix: 0 training and 1 validation.'''
    if n_samples < 10 or n_folds < 2 or n_samples % n_folds:
        raise ValueError('n_samples must be divisible by n_folds and at least 10.')
    layout = np.zeros((n_folds, n_samples), dtype=int)
    fold_size = n_samples // n_folds
    for fold in range(n_folds):
        layout[fold, fold * fold_size : (fold + 1) * fold_size] = 1
    return layout


def serving_skew_example(
    coefficient: float = 0.006,
) -> dict[str, float]:
    '''Show one raw entity transformed with snapshot time and current time.'''
    coefficient = _finite_float(coefficient, 'coefficient')
    training_age_days = 365.0
    production_age_days = 730.0
    intercept = -2.4
    training_probability = float(
        sigmoid(np.array([intercept + coefficient * training_age_days]))[0]
    )
    skewed_probability = float(
        sigmoid(np.array([intercept + coefficient * production_age_days]))[0]
    )
    return {
        'training_age_days': training_age_days,
        'production_age_days': production_age_days,
        'training_probability': training_probability,
        'skewed_probability': skewed_probability,
    }


def monitoring_simulation(
    drift_magnitude: float = 1.0,
    concept_magnitude: float = 0.8,
    seed: int = SEED,
) -> pd.DataFrame:
    '''Simulate weekly service, input, prediction, and delayed-label signals.'''
    drift_magnitude = _finite_float(drift_magnitude, 'drift_magnitude')
    concept_magnitude = _finite_float(concept_magnitude, 'concept_magnitude')
    if not 0 <= drift_magnitude <= 2 or not 0 <= concept_magnitude <= 2:
        raise ValueError('Monitoring controls must be in [0, 2].')
    rng = np.random.default_rng(seed)
    weights = np.array([1.3, -0.9])
    train_features = rng.normal(size=(1_200, 2))
    train_target = rng.binomial(1, sigmoid(train_features @ weights))
    model = LogisticRegression(random_state=seed).fit(train_features, train_target)
    rows = []
    for week in range(1, 13):
        progress = (week - 1) / 11
        features = rng.normal(
            loc=(drift_magnitude * progress, 0.0),
            size=(420, 2),
        )
        angle = concept_magnitude * progress
        current_weights = np.array(
            [
                weights[0] * np.cos(angle) - weights[1] * np.sin(angle),
                weights[0] * np.sin(angle) + weights[1] * np.cos(angle),
            ]
        )
        target = rng.binomial(1, sigmoid(features @ current_weights))
        probabilities = model.predict_proba(features)[:, 1]
        rows.append(
            {
                'week': week,
                'feature_mean': features[:, 0].mean(),
                'missing_rate': max(
                    0.0, 0.015 + 0.004 * week + rng.normal(scale=0.002)
                ),
                'prediction_rate': np.mean(probabilities >= 0.5),
                'latency_ms': 42 + 1.8 * week + rng.normal(scale=2.0),
                'delayed_roc_auc': roc_auc_score(target, probabilities),
            }
        )
    return pd.DataFrame(rows)


def probability_surface(seed: int = SEED) -> dict[str, np.ndarray]:
    '''Fit a two-feature logistic model for a continuous probability surface.'''
    features, target = make_classification(
        n_samples=420,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        class_sep=1.2,
        random_state=seed,
    )
    model = LogisticRegression(random_state=seed).fit(features, target)
    x_grid = np.linspace(features[:, 0].min() - 0.5, features[:, 0].max() + 0.5, 55)
    y_grid = np.linspace(features[:, 1].min() - 0.5, features[:, 1].max() + 0.5, 55)
    xx, yy = np.meshgrid(x_grid, y_grid)
    probabilities = model.predict_proba(np.column_stack((xx.ravel(), yy.ravel())))[
        :, 1
    ].reshape(xx.shape)
    return {
        'features': features,
        'target': target,
        'xx': xx,
        'yy': yy,
        'probabilities': probabilities,
    }

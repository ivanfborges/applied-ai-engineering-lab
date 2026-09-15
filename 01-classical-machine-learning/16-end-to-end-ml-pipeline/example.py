'''A deterministic, end-to-end binary-classification workflow.

The data are synthetic and model a customer-retention decision. The example
keeps model selection and threshold selection away from the final test set,
then persists preprocessing, the estimator, and the decision threshold as one
inference bundle.
'''

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_SEED = 16
TARGET_COLUMN = 'churned_within_30_days'
NUMERIC_FEATURES = (
    'tenure_months',
    'monthly_usage_hours',
    'support_tickets_90d',
    'days_since_last_active',
)
CATEGORICAL_FEATURES = ('plan',)
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class DataSplits:
    '''Train, validation, and test partitions with their original row indices.'''

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


@dataclass(frozen=True)
class TrainingResult:
    '''Selected pipeline and the cross-validation evidence used to choose it.'''

    pipeline: Pipeline
    best_c: float
    mean_cv_roc_auc: float


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f'{name} must be an integer.')
    if value <= 0:
        raise ValueError(f'{name} must be positive.')


def _validate_binary_target(target: Sequence[int], name: str) -> np.ndarray:
    values = np.asarray(target)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f'{name} must be a non-empty one-dimensional target.')
    if not np.isin(values, (0, 1)).all():
        raise ValueError(f'{name} must contain only binary labels 0 and 1.')
    if np.unique(values).size != 2:
        raise ValueError(f'{name} must contain both classes.')
    return values.astype(int, copy=False)


def generate_synthetic_dataset(
    sample_size: int = 2_400,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    '''Create synthetic customer snapshots and a simulated 30-day churn label.'''
    _validate_positive_integer(sample_size, 'sample_size')
    if sample_size < 200:
        raise ValueError('sample_size must be at least 200 for stable data partitions.')
    _validate_positive_integer(seed, 'seed')

    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 121, size=sample_size)
    usage = np.clip(rng.normal(loc=18.0, scale=7.0, size=sample_size), 0.0, 60.0)
    tickets = rng.poisson(lam=1.4, size=sample_size)
    inactivity = np.clip(
        rng.gamma(shape=2.0, scale=5.0, size=sample_size), 0.0, 60.0
    )
    plan = rng.choice(
        np.array(['basic', 'plus', 'premium'], dtype=object),
        size=sample_size,
        p=(0.55, 0.30, 0.15),
    )

    plan_effect = np.select(
        [plan == 'basic', plan == 'premium'],
        [0.35, -0.30],
        default=0.0,
    )
    logits = (
        -1.45
        - 0.018 * (tenure - 36.0)
        - 0.050 * (usage - 18.0)
        + 0.22 * tickets
        + 0.055 * (inactivity - 10.0)
        + plan_effect
    )
    churn_probability = 1.0 / (1.0 + np.exp(-logits))
    target = rng.binomial(1, churn_probability)

    data = pd.DataFrame(
        {
            'tenure_months': tenure.astype(float),
            'monthly_usage_hours': usage,
            'support_tickets_90d': tickets.astype(float),
            'days_since_last_active': inactivity,
            'plan': plan,
            TARGET_COLUMN: target,
        }
    )

    # Missing values are injected after label generation so imputation is part
    # of the learning workflow rather than the synthetic outcome mechanism.
    data.loc[rng.random(sample_size) < 0.04, 'monthly_usage_hours'] = np.nan
    data.loc[rng.random(sample_size) < 0.02, 'plan'] = np.nan
    return data


def validate_features(features: pd.DataFrame) -> pd.DataFrame:
    '''Validate the inference contract and return ordered feature columns.'''
    if not isinstance(features, pd.DataFrame):
        raise TypeError('features must be a pandas DataFrame.')
    if features.empty:
        raise ValueError('features must contain at least one row.')

    missing = [column for column in FEATURE_COLUMNS if column not in features.columns]
    if missing:
        missing_text = ', '.join(missing)
        raise ValueError(f'Missing required feature columns: {missing_text}')

    ordered = features.loc[:, FEATURE_COLUMNS].copy()
    for column in NUMERIC_FEATURES:
        converted = pd.to_numeric(ordered[column], errors='coerce')
        invalid = ordered[column].notna() & converted.isna()
        if invalid.any():
            raise ValueError(f'Feature {column!r} contains a non-numeric value.')
        finite = converted.dropna().to_numpy(dtype=float)
        if not np.isfinite(finite).all():
            raise ValueError(f'Feature {column!r} contains an infinite value.')
        ordered[column] = converted

    invalid_plan = ordered['plan'].dropna().map(
        lambda value: not isinstance(value, str)
    )
    if invalid_plan.any():
        raise ValueError('Feature plan must contain strings or missing values.')
    return ordered


def split_dataset(data: pd.DataFrame, seed: int = RANDOM_SEED) -> DataSplits:
    '''Create 60/20/20 stratified train, validation, and test partitions.'''
    if not isinstance(data, pd.DataFrame):
        raise TypeError('data must be a pandas DataFrame.')
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f'data must contain the target column {TARGET_COLUMN!r}.')
    _validate_positive_integer(seed, 'seed')

    features = validate_features(data)
    target = pd.Series(
        _validate_binary_target(data[TARGET_COLUMN], TARGET_COLUMN),
        index=data.index,
        name=TARGET_COLUMN,
    )
    X_development, X_test, y_development, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        stratify=target,
        random_state=seed,
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=0.25,
        stratify=y_development,
        random_state=seed,
    )
    return DataSplits(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
    )


def build_pipeline(seed: int = RANDOM_SEED) -> Pipeline:
    '''Build preprocessing and logistic regression as one fitted unit.'''
    _validate_positive_integer(seed, 'seed')
    numeric_pipeline = Pipeline(
        steps=(
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        )
    )
    categorical_pipeline = Pipeline(
        steps=(
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore')),
        )
    )
    preprocessing = ColumnTransformer(
        transformers=(
            ('numeric', numeric_pipeline, list(NUMERIC_FEATURES)),
            ('categorical', categorical_pipeline, list(CATEGORICAL_FEATURES)),
        )
    )
    return Pipeline(
        steps=(
            ('preprocessing', preprocessing),
            (
                'model',
                LogisticRegression(max_iter=1_000, random_state=seed),
            ),
        )
    )


def train_model(
    X_train: pd.DataFrame,
    y_train: Sequence[int],
    *,
    c_values: Sequence[float] = (0.1, 1.0, 10.0),
    cv_splits: int = 5,
    seed: int = RANDOM_SEED,
) -> TrainingResult:
    '''Select logistic regularization with fold-local preprocessing.'''
    features = validate_features(X_train)
    target = _validate_binary_target(y_train, 'y_train')
    if len(features) != target.size:
        raise ValueError('X_train and y_train must have the same number of rows.')
    _validate_positive_integer(cv_splits, 'cv_splits')
    if cv_splits < 2:
        raise ValueError('cv_splits must be at least 2.')
    if min(np.bincount(target)) < cv_splits:
        raise ValueError('Each class must have at least cv_splits observations.')

    candidate_values = tuple(c_values)
    if not candidate_values:
        raise ValueError('c_values must not be empty.')
    try:
        valid_candidates = all(
            math.isfinite(value) and value > 0 for value in candidate_values
        )
    except TypeError as error:
        raise TypeError('c_values must contain numeric values.') from error
    if not valid_candidates:
        raise ValueError('c_values must contain finite positive values.')

    folds = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=seed,
    )
    search = GridSearchCV(
        estimator=build_pipeline(seed),
        param_grid={'model__C': list(candidate_values)},
        scoring='roc_auc',
        cv=folds,
        n_jobs=1,
        refit=True,
    )
    search.fit(features, target)
    return TrainingResult(
        pipeline=search.best_estimator_,
        best_c=float(search.best_params_['model__C']),
        mean_cv_roc_auc=float(search.best_score_),
    )


def select_threshold_for_recall(
    y_validation: Sequence[int],
    positive_probabilities: Sequence[float],
    *,
    target_recall: float = 0.75,
) -> float:
    '''Choose the highest validation threshold that reaches the recall target.'''
    target = _validate_binary_target(y_validation, 'y_validation')
    probabilities = np.asarray(positive_probabilities, dtype=float)
    if probabilities.shape != target.shape:
        raise ValueError('Probabilities and validation labels must have equal shape.')
    invalid_probability = (
        not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
    )
    if invalid_probability:
        raise ValueError('Probabilities must be finite values in [0, 1].')
    if not math.isfinite(target_recall) or not 0 < target_recall <= 1:
        raise ValueError('target_recall must be in (0, 1].')

    _, recalls, thresholds = precision_recall_curve(target, probabilities)
    eligible = thresholds[recalls[:-1] >= target_recall]
    if eligible.size == 0:
        raise ValueError('No threshold satisfies the requested recall target.')
    return float(np.max(eligible))


def evaluate_predictions(
    target: Sequence[int],
    positive_probabilities: Sequence[float],
    threshold: float,
) -> dict[str, float]:
    '''Return ranking and threshold-dependent classification metrics.'''
    labels = _validate_binary_target(target, 'target')
    probabilities = np.asarray(positive_probabilities, dtype=float)
    if probabilities.shape != labels.shape:
        raise ValueError('Probabilities and target must have equal shape.')
    invalid_probability = (
        not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
    )
    if invalid_probability:
        raise ValueError('Probabilities must be finite values in [0, 1].')
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('threshold must be in [0, 1].')

    predictions = (probabilities >= threshold).astype(int)
    return {
        'roc_auc': float(roc_auc_score(labels, probabilities)),
        'precision': float(
            precision_score(labels, predictions, zero_division=0)
        ),
        'recall': float(recall_score(labels, predictions, zero_division=0)),
        'f1': float(f1_score(labels, predictions, zero_division=0)),
        'positive_prediction_rate': float(np.mean(predictions)),
    }


def create_inference_bundle(
    model: Pipeline,
    threshold: float,
    *,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    '''Package the fitted transformation, estimator, and decision policy.'''
    if not isinstance(model, BaseEstimator) or not hasattr(model, 'predict_proba'):
        raise TypeError(
            'model must be a fitted probabilistic scikit-learn estimator.'
        )
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('threshold must be in [0, 1].')
    return {
        'model': model,
        'threshold': float(threshold),
        'feature_columns': FEATURE_COLUMNS,
        'metadata': {'data_source': 'synthetic', 'seed': seed},
    }


def save_inference_bundle(bundle: dict[str, Any], output_path: Path) -> None:
    '''Serialize an inference bundle to a local artifact.'''
    if not isinstance(output_path, Path):
        raise TypeError('output_path must be a pathlib.Path.')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('wb') as file_handle:
        pickle.dump(bundle, file_handle)


def load_inference_bundle(input_path: Path) -> dict[str, Any]:
    '''Load a trusted local bundle and validate its expected structure.'''
    if not isinstance(input_path, Path):
        raise TypeError('input_path must be a pathlib.Path.')
    if not input_path.is_file():
        raise FileNotFoundError(f'Inference bundle not found: {input_path}')
    with input_path.open('rb') as file_handle:
        bundle = pickle.load(file_handle)  # noqa: S301 - trusted artifact only
    required_keys = {'model', 'threshold', 'feature_columns', 'metadata'}
    if not isinstance(bundle, dict) or not required_keys.issubset(bundle):
        raise ValueError('The serialized object is not a valid inference bundle.')
    if tuple(bundle['feature_columns']) != FEATURE_COLUMNS:
        raise ValueError(
            'The bundle feature contract does not match this code version.'
        )
    return bundle


def predict_with_bundle(
    bundle: dict[str, Any],
    features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    '''Apply the saved preprocessing, estimator, and threshold to raw rows.'''
    required_keys = {'model', 'threshold'}
    if not isinstance(bundle, dict) or not required_keys.issubset(bundle):
        raise ValueError('bundle must contain a model and threshold.')
    ordered = validate_features(features)
    probabilities = np.asarray(bundle['model'].predict_proba(ordered)[:, 1])
    predictions = (probabilities >= float(bundle['threshold'])).astype(int)
    return probabilities, predictions


def run_experiment(output_path: Path) -> dict[str, Any]:
    '''Execute the workflow and persist its complete inference bundle.'''
    data = generate_synthetic_dataset()
    splits = split_dataset(data)

    baseline = DummyClassifier(strategy='prior')
    baseline.fit(splits.X_train, splits.y_train)

    training = train_model(splits.X_train, splits.y_train)
    validation_probabilities = training.pipeline.predict_proba(
        splits.X_validation
    )[:, 1]
    threshold = select_threshold_for_recall(
        splits.y_validation,
        validation_probabilities,
        target_recall=0.75,
    )
    validation_metrics = evaluate_predictions(
        splits.y_validation,
        validation_probabilities,
        threshold,
    )

    # The test partition is consulted only after model and threshold are fixed.
    baseline_probabilities = baseline.predict_proba(splits.X_test)[:, 1]
    test_probabilities = training.pipeline.predict_proba(splits.X_test)[:, 1]
    test_metrics = evaluate_predictions(
        splits.y_test,
        test_probabilities,
        threshold,
    )

    bundle = create_inference_bundle(training.pipeline, threshold)
    save_inference_bundle(bundle, output_path)
    loaded_bundle = load_inference_bundle(output_path)
    inference_row = pd.DataFrame(
        [
            {
                'tenure_months': 12,
                'monthly_usage_hours': np.nan,
                'support_tickets_90d': 2,
                'days_since_last_active': 14,
                'plan': 'enterprise',
            }
        ]
    )
    expected_probability = training.pipeline.predict_proba(
        validate_features(inference_row)
    )[:, 1]
    loaded_probability, _ = predict_with_bundle(loaded_bundle, inference_row)

    return {
        'sample_size': len(data),
        'positive_rate': float(data[TARGET_COLUMN].mean()),
        'split_sizes': (
            len(splits.X_train),
            len(splits.X_validation),
            len(splits.X_test),
        ),
        'best_c': training.best_c,
        'mean_cv_roc_auc': training.mean_cv_roc_auc,
        'threshold': threshold,
        'validation_metrics': validation_metrics,
        'baseline_test_roc_auc': float(
            roc_auc_score(splits.y_test, baseline_probabilities)
        ),
        'test_metrics': test_metrics,
        'round_trip_max_absolute_difference': float(
            np.max(np.abs(expected_probability - loaded_probability))
        ),
        'output_path': output_path,
    }


def main() -> None:
    '''Run the pipeline and print a compact, reproducible execution record.'''
    output_path = (
        Path(__file__).resolve().parent / 'outputs' / 'inference_bundle.pkl'
    )
    result = run_experiment(output_path)
    validation = result['validation_metrics']
    test = result['test_metrics']
    sample_size = result['sample_size']
    positive_rate = result['positive_rate']
    best_c = result['best_c']
    mean_cv_roc_auc = result['mean_cv_roc_auc']
    threshold = result['threshold']
    baseline_test_roc_auc = result['baseline_test_roc_auc']
    round_trip_difference = result['round_trip_max_absolute_difference']
    saved_path = result['output_path']
    validation_precision = validation['precision']
    validation_recall = validation['recall']
    test_roc_auc = test['roc_auc']
    test_precision = test['precision']
    test_recall = test['recall']
    test_f1 = test['f1']
    test_positive_rate = test['positive_prediction_rate']

    print('End-to-end ML pipeline (deterministic synthetic data)')
    print(f'Rows: {sample_size}')
    print(f'Observed positive rate: {positive_rate:.4f}')
    split_text = '/'.join(map(str, result['split_sizes']))
    print(f'Train/validation/test rows: {split_text}')
    print(f'Selected C: {best_c:.3g}')
    print(f'Mean training CV ROC-AUC: {mean_cv_roc_auc:.4f}')
    print(f'Validation-selected threshold: {threshold:.4f}')
    print(
        'Validation precision/recall: '
        f'{validation_precision:.4f}/{validation_recall:.4f}'
    )
    print(f'Baseline test ROC-AUC: {baseline_test_roc_auc:.4f}')
    print(
        'Model test ROC-AUC/precision/recall/F1: '
        f'{test_roc_auc:.4f}/{test_precision:.4f}/'
        f'{test_recall:.4f}/{test_f1:.4f}'
    )
    print(f'Test positive prediction rate: {test_positive_rate:.4f}')
    print(
        'Serialization round-trip max absolute difference: '
        f'{round_trip_difference:.3e}'
    )
    print(f'Saved ignored artifact: {saved_path}')


if __name__ == '__main__':
    main()

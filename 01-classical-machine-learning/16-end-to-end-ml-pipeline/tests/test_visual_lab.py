'''Numerical, rendering, and Streamlit smoke tests for the Day 16 visual lab.'''

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest


TOPIC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC_DIR))

from simulation import (  # noqa: E402
    baseline_experiment,
    compare_random_and_temporal_splits,
    cross_validation_layout,
    drift_experiment,
    group_leakage_experiment,
    leakage_experiment,
    monitoring_simulation,
    threshold_curve,
    threshold_experiment,
    threshold_metrics,
    validation_search_experiment,
)
from visualizations import (  # noqa: E402
    PUBLIC_PREVIEWS,
    plot_lifecycle,
    probability_surface_3d,
)


def test_temporal_drift_is_reproducible_and_changes_evaluation() -> None:
    first = compare_random_and_temporal_splits(1.3)
    second = compare_random_and_temporal_splits(1.3)
    assert first.random_auc == pytest.approx(second.random_auc)
    assert first.temporal_auc == pytest.approx(second.temporal_auc)
    assert first.data.equals(second.data)
    assert first.random_auc > first.temporal_auc


def test_group_split_removes_entity_overlap() -> None:
    result = group_leakage_experiment()
    assert result['row_overlap'] > 0
    assert result['group_overlap'] == 0
    assert result['row_auc'] > result['group_auc']


def test_target_derived_feature_creates_invalid_score_gain() -> None:
    result = leakage_experiment()
    assert result['leaky_auc'] > 0.98
    assert result['leaky_auc'] > result['valid_auc']


def test_majority_baseline_exposes_accuracy_limit() -> None:
    metrics = baseline_experiment(0.08).set_index('model')
    assert metrics.loc['Majority class', 'accuracy'] > 0.85
    assert metrics.loc['Majority class', 'recall'] == 0.0
    assert metrics.loc['Logistic regression', 'pr_auc'] > 0.08


def test_threshold_changes_decisions_and_cost_curve_is_finite() -> None:
    experiment = threshold_experiment()
    low = threshold_metrics(
        experiment['y_test'],
        experiment['probabilities'],
        0.2,
    )
    high = threshold_metrics(
        experiment['y_test'],
        experiment['probabilities'],
        0.7,
    )
    curve = threshold_curve(
        experiment['y_test'],
        experiment['probabilities'],
    )
    assert low['predicted_positives'] > high['predicted_positives']
    assert low['recall'] >= high['recall']
    assert np.isfinite(curve.to_numpy(dtype=float)).all()
    assert 0.01 <= curve.loc[curve['cost'].idxmin(), 'threshold'] <= 0.99


def test_validation_search_shows_selection_optimism_not_true_improvement() -> None:
    results = validation_search_experiment()
    assert results['best_validation_mean'].is_monotonic_increasing
    assert np.max(np.abs(results['selected_test_mean'] - 0.70)) < 0.02
    assert results['best_validation_mean'].iloc[-1] > results[
        'selected_test_mean'
    ].iloc[-1]


def test_drift_simulations_separate_input_and_conditional_change() -> None:
    result = drift_experiment(1.0, 1.0)
    assert result['wasserstein_x1'] > 0.8
    assert result['concept_wasserstein_x1'] < 0.15
    assert result['concept_auc'] < result['covariate_auc']


def test_cross_validation_layout_moves_one_complete_holdout() -> None:
    layout = cross_validation_layout(60, 5)
    assert layout.shape == (5, 60)
    assert np.all(layout.sum(axis=1) == 12)
    assert np.all(layout.sum(axis=0) == 1)


def test_monitoring_simulation_is_bounded_and_reproducible() -> None:
    first = monitoring_simulation()
    second = monitoring_simulation()
    assert first.equals(second)
    assert list(first['week']) == list(range(1, 13))
    assert first['delayed_roc_auc'].between(0, 1).all()


def test_static_preview_and_plotly_surface_are_renderable(tmp_path: Path) -> None:
    output = plot_lifecycle(tmp_path / 'lifecycle.png')
    assert output.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert len(PUBLIC_PREVIEWS) == 6
    assert len(set(PUBLIC_PREVIEWS)) == 6
    surface = probability_surface_3d()
    assert len(surface.data) == 2


def test_streamlit_default_and_threshold_sections_load() -> None:
    app = AppTest.from_file(TOPIC_DIR / 'streamlit_app.py')
    app.run(timeout=40)
    assert not app.exception
    assert app.title[0].value == 'End-to-end ML pipeline laboratory'
    app.selectbox[0].set_value('Thresholds & business cost').run(timeout=40)
    assert not app.exception
    assert len(app.slider) >= 1
    assert len(app.metric) >= 5

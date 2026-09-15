'''Visualizations and bounded asset generation for the Day 16 laboratory.'''

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import precision_recall_curve

from simulation import (
    LIFECYCLE_STAGES,
    baseline_experiment,
    compare_random_and_temporal_splits,
    complexity_experiment,
    cross_validation_layout,
    drift_experiment,
    group_leakage_experiment,
    leakage_experiment,
    monitoring_simulation,
    probability_surface,
    serving_skew_example,
    threshold_curve,
    threshold_experiment,
    threshold_metrics,
    validation_search_experiment,
)


TOPIC_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = TOPIC_DIR / 'outputs'
COLORS = {
    'blue': '#2563EB',
    'cyan': '#0891B2',
    'green': '#16A34A',
    'orange': '#EA580C',
    'red': '#DC2626',
    'purple': '#7C3AED',
    'gray': '#64748B',
    'light': '#E2E8F0',
    'dark': '#0F172A',
}
PUBLIC_PREVIEWS = (
    'ml_lifecycle.png',
    'random_vs_temporal_split.png',
    'data_leakage.png',
    'threshold_tradeoff.png',
    'covariate_vs_concept_drift.png',
    'concept_drift.gif',
)


def _style_axis(axis: plt.Axes, *, grid_axis: str = 'both') -> None:
    axis.set_facecolor('#F8FAFC')
    axis.grid(axis=grid_axis, alpha=0.22)
    axis.set_axisbelow(True)
    axis.spines[['top', 'right']].set_visible(False)


def _save_figure(figure: plt.Figure, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=145, bbox_inches='tight', facecolor='white')
    plt.close(figure)
    return output_path


def plot_lifecycle(output_path: Path | None = None) -> plt.Figure | Path:
    '''Show model fitting as one stage inside a monitored decision lifecycle.'''
    figure, axis = plt.subplots(figsize=(13, 6.2))
    axis.axis('off')
    top_positions = [(value, 0.65) for value in np.linspace(0.08, 0.92, 6)]
    bottom_positions = [(value, 0.30) for value in np.linspace(0.92, 0.24, 5)]
    positions = top_positions + bottom_positions
    width, height = 0.13, 0.14
    for index, (stage, (x, y)) in enumerate(
        zip(LIFECYCLE_STAGES, positions, strict=True)
    ):
        color = COLORS['orange'] if stage == 'Training' else COLORS['blue']
        box = FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle='round,pad=0.012,rounding_size=0.02',
            facecolor=color,
            edgecolor='white',
            linewidth=2,
        )
        axis.add_patch(box)
        axis.text(
            x,
            y,
            stage.replace(' / ', '\n/ '),
            ha='center',
            va='center',
            color='white',
            fontsize=8.8,
            fontweight='bold',
        )
        if index < len(LIFECYCLE_STAGES) - 1:
            next_x, next_y = positions[index + 1]
            if index == 5:
                start = (x, y - height / 2)
                end = (next_x, next_y + height / 2)
            else:
                direction = 1 if next_x > x else -1
                start = (x + direction * width / 2, y)
                end = (next_x - direction * width / 2, next_y)
            axis.annotate(
                '',
                xy=end,
                xytext=start,
                arrowprops={'arrowstyle': '->', 'color': COLORS['gray'], 'lw': 1.6},
            )
    axis.annotate(
        'Production evidence changes the next candidate',
        xy=(positions[-1][0], positions[-1][1] - height / 2),
        xytext=(0.72, 0.08),
        ha='center',
        color=COLORS['purple'],
        arrowprops={
            'arrowstyle': '->',
            'color': COLORS['purple'],
            'connectionstyle': 'arc3,rad=-0.24',
        },
    )
    axis.text(
        positions[5][0],
        0.90,
        'model.fit() is one stage, not the system',
        ha='center',
        fontsize=14,
        fontweight='bold',
        color=COLORS['dark'],
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    if output_path is not None:
        return _save_figure(figure, output_path)
    return figure


def plot_prediction_time_boundary() -> plt.Figure:
    '''Distinguish valid snapshot features from future outcome information.'''
    figure, axis = plt.subplots(figsize=(12, 5.2))
    axis.axvspan(0, 5, color='#DBEAFE', alpha=0.85)
    axis.axvspan(5, 8, color='#FEE2E2', alpha=0.75)
    axis.axvline(5, color=COLORS['dark'], linewidth=2.4)
    axis.text(2.5, 4.2, 'AVAILABLE HISTORY', ha='center', fontweight='bold')
    axis.text(6.5, 4.2, 'OUTCOME WINDOW', ha='center', fontweight='bold')
    valid = ('Transactions', 'Account balance', 'Support interactions')
    future = ('Default', 'Collections activity', 'Payment outcome')
    for index, label in enumerate(valid):
        axis.scatter(1.0 + index * 1.4, 2.7, s=150, color=COLORS['blue'])
        axis.text(1.0 + index * 1.4, 2.35, label, ha='center', fontsize=9)
    for index, label in enumerate(future):
        axis.scatter(5.7 + index * 0.8, 2.7, s=150, color=COLORS['red'])
        axis.text(5.7 + index * 0.8, 2.25, label, ha='center', fontsize=9)
    axis.annotate(
        'Prediction',
        xy=(5, 1.25),
        xytext=(5, 0.35),
        ha='center',
        fontweight='bold',
        arrowprops={'arrowstyle': '->', 'color': COLORS['dark'], 'lw': 2},
    )
    axis.annotate(
        'LEAKAGE if used as a feature',
        xy=(6.5, 2.7),
        xytext=(3.8, 3.45),
        color=COLORS['red'],
        fontweight='bold',
        arrowprops={'arrowstyle': '->', 'color': COLORS['red']},
    )
    axis.set(xlim=(0, 8), ylim=(0, 4.8), xlabel='Relative time')
    axis.set_yticks([])
    axis.set_title(
        'Feature validity is defined at prediction time',
        fontweight='bold',
    )
    _style_axis(axis, grid_axis='x')
    figure.tight_layout()
    return figure


def _draw_boundary(
    axis: plt.Axes,
    model: object,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> None:
    xx, yy = np.meshgrid(
        np.linspace(*x_limits, 120),
        np.linspace(*y_limits, 120),
    )
    probabilities = model.predict_proba(np.column_stack((xx.ravel(), yy.ravel())))[
        :, 1
    ].reshape(xx.shape)
    axis.contour(xx, yy, probabilities, levels=(0.5,), colors=COLORS['dark'])


def plot_split_comparison(
    drift_magnitude: float = 1.3,
    output_path: Path | None = None,
) -> plt.Figure | Path:
    '''Contrast random mixing with a future holdout under concept drift.'''
    result = compare_random_and_temporal_splits(drift_magnitude)
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    data = result.data
    features = data[['x1', 'x2']].to_numpy()
    x_limits = (features[:, 0].min() - 0.3, features[:, 0].max() + 0.3)
    y_limits = (features[:, 1].min() - 0.3, features[:, 1].max() + 0.3)
    configurations = (
        ('Random split', result.random_train, result.random_test, result.random_model),
        (
            'Temporal split',
            result.temporal_train,
            result.temporal_test,
            result.temporal_model,
        ),
    )
    for column, (title, train, test, model) in enumerate(configurations):
        axis = axes[0, column]
        axis.scatter(
            data.iloc[train]['time'],
            np.zeros(len(train)),
            s=11,
            color=COLORS['blue'],
            alpha=0.55,
        )
        axis.scatter(
            data.iloc[test]['time'],
            np.ones(len(test)),
            s=11,
            color=COLORS['orange'],
            alpha=0.70,
        )
        axis.set(
            title=title,
            xlabel='Time',
            yticks=(0, 1),
            yticklabels=('Train', 'Test'),
            ylim=(-0.5, 1.5),
        )
        _style_axis(axis)

        axis = axes[1, column]
        axis.scatter(
            features[test, 0],
            features[test, 1],
            c=data.iloc[test]['target'],
            cmap='coolwarm',
            alpha=0.65,
            s=18,
        )
        _draw_boundary(axis, model, x_limits, y_limits)
        auc = result.random_auc if column == 0 else result.temporal_auc
        axis.set(
            title=f'Locked boundary on test data — ROC-AUC {auc:.3f}',
            xlabel='Feature 1',
            ylabel='Feature 2',
            xlim=x_limits,
            ylim=y_limits,
        )
        _style_axis(axis)
    figure.suptitle(
        f'Random vs temporal evaluation under gradual drift ({drift_magnitude:.1f})',
        fontsize=15,
        fontweight='bold',
    )
    figure.tight_layout()
    if output_path is not None:
        return _save_figure(figure, output_path)
    return figure


def plot_group_leakage() -> plt.Figure:
    '''Show repeated entities crossing a row split but not a grouped split.'''
    result = group_leakage_experiment()
    groups = result['groups']
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for axis, title, train_key, test_key, overlap_key, auc_key in (
        (
            axes[0],
            'Random row split',
            'row_train',
            'row_test',
            'row_overlap',
            'row_auc',
        ),
        (
            axes[1],
            'Group-aware split',
            'group_train',
            'group_test',
            'group_overlap',
            'group_auc',
        ),
    ):
        train = result[train_key]
        test = result[test_key]
        overlap = result[overlap_key]
        auc = result[auc_key]
        shown = np.arange(min(22, len(np.unique(groups))))
        for group in shown:
            rows = np.flatnonzero(groups == group)
            axis.scatter(
                np.where(np.isin(rows, train), 0, 1),
                np.full(len(rows), group),
                c=np.where(np.isin(rows, train), COLORS['blue'], COLORS['orange']),
                s=30,
            )
        axis.set(
            title=f'{title}\nentity overlap={overlap}, ROC-AUC={auc:.3f}',
            xlabel='Partition',
            xticks=(0, 1),
            xticklabels=('Train', 'Test'),
        )
        _style_axis(axis, grid_axis='y')
    axes[0].set_ylabel('Entity (first 22 shown)')
    figure.suptitle('Repeated observations require an entity boundary', fontweight='bold')
    figure.tight_layout()
    return figure


def plot_data_leakage(output_path: Path | None = None) -> plt.Figure | Path:
    '''Show an invalid score gain and its target-derived source.'''
    result = leakage_experiment()
    test = result['test']
    feature = result['leaky_feature'][test]
    target = result['target'][test]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5))
    axes[0].bar(
        ('Valid features', 'Valid + future target proxy'),
        (result['valid_auc'], result['leaky_auc']),
        color=(COLORS['blue'], COLORS['red']),
    )
    axes[0].axhline(0.5, color=COLORS['gray'], linestyle='--')
    axes[0].set(ylim=(0.45, 1.02), ylabel='Test ROC-AUC')
    axes[0].text(
        0.5,
        0.08,
        'HIGHER SCORE ≠ VALID EVIDENCE',
        transform=axes[0].transAxes,
        ha='center',
        color=COLORS['red'],
        fontweight='bold',
        bbox={'facecolor': 'white', 'edgecolor': COLORS['red'], 'alpha': 0.9},
    )
    _style_axis(axes[0], grid_axis='y')
    axes[1].hist(
        feature[target == 0],
        bins=24,
        alpha=0.65,
        color=COLORS['blue'],
        label='Actual negative',
    )
    axes[1].hist(
        feature[target == 1],
        bins=24,
        alpha=0.65,
        color=COLORS['red'],
        label='Actual positive',
    )
    axes[1].set(
        xlabel='Post-outcome target proxy',
        ylabel='Count',
        title='The added feature nearly reveals the label',
    )
    axes[1].legend()
    _style_axis(axes[1], grid_axis='y')
    figure.suptitle('Suspiciously excellent performance demands investigation', fontweight='bold')
    figure.tight_layout()
    if output_path is not None:
        return _save_figure(figure, output_path)
    return figure


def plot_baseline_metrics(positive_fraction: float = 0.08) -> plt.Figure:
    '''Compare accuracy with imbalance-aware ranking and class metrics.'''
    metrics = baseline_experiment(positive_fraction)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    left = metrics.set_index('model')[['accuracy', 'balanced_accuracy', 'f1']]
    right = metrics.set_index('model')[['roc_auc', 'pr_auc']]
    left.plot.bar(ax=axes[0], color=(COLORS['gray'], COLORS['blue'], COLORS['orange']))
    right.plot.bar(ax=axes[1], color=(COLORS['purple'], COLORS['green']))
    axes[0].set(
        title='Thresholded metrics',
        ylabel='Score',
        xlabel='',
        ylim=(0, 1.02),
    )
    axes[1].set(
        title='Ranking metrics',
        ylabel='Score',
        xlabel='',
        ylim=(0, 1.02),
    )
    for axis in axes:
        axis.tick_params(axis='x', rotation=18)
        _style_axis(axis, grid_axis='y')
        axis.legend(loc='lower right')
    figure.suptitle(
        f'High accuracy can hide failure at {positive_fraction:.0%} prevalence',
        fontweight='bold',
    )
    figure.tight_layout()
    return figure


def plot_threshold_tradeoff(
    threshold: float = 0.35,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 5.0,
    output_path: Path | None = None,
) -> plt.Figure | Path:
    '''Connect score distributions, PR position, and threshold-dependent cost.'''
    experiment = threshold_experiment()
    target = experiment['y_test']
    probabilities = experiment['probabilities']
    metrics = threshold_metrics(
        target,
        probabilities,
        threshold,
        false_positive_cost,
        false_negative_cost,
    )
    curve = threshold_curve(
        target,
        probabilities,
        false_positive_cost,
        false_negative_cost,
    )
    precision, recall, pr_thresholds = precision_recall_curve(target, probabilities)
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].hist(
        probabilities[target == 0],
        bins=28,
        alpha=0.65,
        color=COLORS['blue'],
        label='Actual negative',
    )
    axes[0, 0].hist(
        probabilities[target == 1],
        bins=28,
        alpha=0.65,
        color=COLORS['orange'],
        label='Actual positive',
    )
    axes[0, 0].axvline(threshold, color=COLORS['red'], linewidth=2)
    axes[0, 0].set(
        title='Probability scores + decision policy',
        xlabel='Predicted probability',
        ylabel='Count',
    )
    axes[0, 0].legend()

    axes[0, 1].plot(recall, precision, color=COLORS['purple'], linewidth=2.2)
    index = int(np.argmin(np.abs(pr_thresholds - threshold)))
    axes[0, 1].scatter(
        recall[index],
        precision[index],
        s=90,
        color=COLORS['red'],
        zorder=3,
    )
    axes[0, 1].set(
        title=f'Precision–recall operating point (t={threshold:.2f})',
        xlabel='Recall',
        ylabel='Precision',
        xlim=(0, 1.02),
        ylim=(0, 1.02),
    )

    axes[1, 0].plot(curve['threshold'], curve['precision'], label='Precision')
    axes[1, 0].plot(curve['threshold'], curve['recall'], label='Recall')
    axes[1, 0].plot(curve['threshold'], curve['f1'], label='F1')
    axes[1, 0].axvline(threshold, color=COLORS['red'], linestyle='--')
    axes[1, 0].set(
        title='The threshold changes the error trade-off',
        xlabel='Decision threshold',
        ylabel='Metric',
        ylim=(0, 1.02),
    )
    axes[1, 0].legend()

    best_index = int(curve['cost'].idxmin())
    best_threshold = float(curve.loc[best_index, 'threshold'])
    axes[1, 1].plot(
        curve['threshold'],
        curve['cost'],
        color=COLORS['green'],
        linewidth=2.2,
    )
    axes[1, 1].axvline(best_threshold, color=COLORS['dark'], linestyle='--')
    axes[1, 1].scatter(
        best_threshold,
        curve.loc[best_index, 'cost'],
        color=COLORS['dark'],
        s=70,
    )
    axes[1, 1].set(
        title=f'Cost minimum on this grid: t={best_threshold:.2f}',
        xlabel='Decision threshold',
        ylabel='Synthetic decision cost',
    )
    for axis in axes.ravel():
        _style_axis(axis)
    figure.suptitle(
        'MODEL: estimate P(Y=1|X)    POLICY: act when probability ≥ threshold',
        fontsize=14,
        fontweight='bold',
    )
    tp = metrics['tp']
    fp = metrics['fp']
    fn = metrics['fn']
    tn = metrics['tn']
    predicted_positives = metrics['predicted_positives']
    figure.text(
        0.5,
        0.01,
        (
            f'At t={threshold:.2f}: TP={tp}, FP={fp}, '
            f'FN={fn}, TN={tn}, predicted positives={predicted_positives}'
        ),
        ha='center',
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.96))
    if output_path is not None:
        return _save_figure(figure, output_path)
    return figure


def probability_surface_3d() -> go.Figure:
    '''Return a rotatable logistic probability surface with observations.'''
    data = probability_surface()
    features = data['features']
    target = data['target']
    figure = go.Figure()
    figure.add_trace(
        go.Surface(
            x=data['xx'],
            y=data['yy'],
            z=data['probabilities'],
            colorscale='Viridis',
            opacity=0.76,
            colorbar={'title': 'P(Y=1)'},
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=features[:, 0],
            y=features[:, 1],
            z=target,
            mode='markers',
            marker={
                'size': 3,
                'color': target,
                'colorscale': ((0, COLORS['blue']), (1, COLORS['orange'])),
                'opacity': 0.65,
            },
            name='Observed class',
        )
    )
    figure.update_layout(
        title='A classifier estimates a continuous probability surface',
        scene={
            'xaxis_title': 'Feature 1',
            'yaxis_title': 'Feature 2',
            'zaxis_title': 'Probability / class',
        },
        height=620,
        margin={'l': 0, 'r': 0, 't': 55, 'b': 0},
    )
    return figure


def plot_complexity() -> plt.Figure:
    '''Show three boundaries and the train-validation complexity curve.'''
    result = complexity_experiment()
    figure, axes = plt.subplots(2, 2, figsize=(11, 9))
    X_validation = result['X_validation']
    y_validation = result['y_validation']
    x_limits = (
        result['X_train'][:, 0].min() - 0.4,
        result['X_train'][:, 0].max() + 0.4,
    )
    y_limits = (
        result['X_train'][:, 1].min() - 0.4,
        result['X_train'][:, 1].max() + 0.4,
    )
    for axis, degree, label in zip(
        axes.ravel()[:3],
        (1, 3, 12),
        ('Underfit', 'Reasonable fit', 'High complexity'),
        strict=True,
    ):
        model = result['models'][degree]
        xx, yy = np.meshgrid(
            np.linspace(*x_limits, 130),
            np.linspace(*y_limits, 130),
        )
        surface = model.predict(np.column_stack((xx.ravel(), yy.ravel()))).reshape(
            xx.shape
        )
        axis.contourf(xx, yy, surface, alpha=0.18, cmap='coolwarm')
        axis.scatter(
            X_validation[:, 0],
            X_validation[:, 1],
            c=y_validation,
            cmap='coolwarm',
            s=18,
            alpha=0.7,
        )
        score = result['scores'].set_index('degree').loc[degree]
        axis.set(
            title=(
                f'{label}: degree {degree}\n'
                f'train={score.train_accuracy:.3f}, validation={score.validation_accuracy:.3f}'
            ),
            xlim=x_limits,
            ylim=y_limits,
        )
        _style_axis(axis)
    scores = result['scores']
    axes[1, 1].plot(
        scores['degree'],
        scores['train_accuracy'],
        marker='o',
        label='Train',
        color=COLORS['blue'],
    )
    axes[1, 1].plot(
        scores['degree'],
        scores['validation_accuracy'],
        marker='o',
        label='Validation',
        color=COLORS['orange'],
    )
    axes[1, 1].set(
        title='Complexity can widen the generalization gap',
        xlabel='Polynomial degree',
        ylabel='Accuracy',
        ylim=(0.5, 1.02),
    )
    axes[1, 1].legend()
    _style_axis(axes[1, 1])
    figure.suptitle('Bias and variance are properties of the evaluated process', fontweight='bold')
    figure.tight_layout()
    return figure


def plot_validation_search() -> plt.Figure:
    '''Visualize selection optimism as more equal-quality candidates are tried.'''
    results = validation_search_experiment()
    figure, axis = plt.subplots(figsize=(9, 5.4))
    axis.plot(
        results['candidates'],
        results['best_validation_mean'],
        marker='o',
        linewidth=2.2,
        label='Best reused-validation score',
        color=COLORS['orange'],
    )
    axis.plot(
        results['candidates'],
        results['selected_test_mean'],
        marker='o',
        linewidth=2.2,
        label='Independent test score of winner',
        color=COLORS['blue'],
    )
    axis.axhline(
        results['true_accuracy'].iloc[0],
        linestyle='--',
        color=COLORS['dark'],
        label='Equal candidate truth in simulation',
    )
    axis.set_xscale('log')
    axis.set(
        xlabel='Candidate configurations evaluated',
        ylabel='Mean accuracy across repeated simulations',
        ylim=(0.66, 0.84),
        title='Searching harder can select validation noise',
    )
    axis.legend()
    _style_axis(axis)
    figure.tight_layout()
    return figure


def plot_cross_validation() -> plt.Figure:
    '''Show the moving validation fold and fold-local preprocessing contract.'''
    layout = cross_validation_layout()
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    axes[0].imshow(layout, aspect='auto', cmap='Blues', vmin=0, vmax=1)
    axes[0].set(
        xlabel='Observation block',
        ylabel='Fold',
        yticks=np.arange(5),
        yticklabels=[f'Fold {index}' for index in range(1, 6)],
        title='Dark block = validation; light blocks = training',
    )
    axes[1].axis('off')
    axes[1].text(
        0.05,
        0.80,
        'INCORRECT',
        color=COLORS['red'],
        fontweight='bold',
        fontsize=13,
    )
    axes[1].text(0.05, 0.68, 'fit scaler on ALL data\n↓\ncross-validation', fontsize=12)
    axes[1].text(
        0.55,
        0.80,
        'CORRECT',
        color=COLORS['green'],
        fontweight='bold',
        fontsize=13,
    )
    axes[1].text(
        0.55,
        0.60,
        'for each fold:\nfit preprocessing on train\ntransform train + validation',
        fontsize=12,
    )
    axes[1].set_title('Preprocessing is part of every fitted fold')
    figure.suptitle('K-fold cross-validation rotates development evidence', fontweight='bold')
    figure.tight_layout()
    return figure


def plot_drift_comparison(
    covariate_shift: float = 1.0,
    concept_rotation: float = 1.0,
    output_path: Path | None = None,
) -> plt.Figure | Path:
    '''Contrast changed inputs with changed conditional label relationships.'''
    result = drift_experiment(covariate_shift, concept_rotation)
    covariate_distance = result['wasserstein_x1']
    concept_distance = result['concept_wasserstein_x1']
    covariate_auc = result['covariate_auc']
    concept_auc = result['concept_auc']
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    axes[0, 0].hist(
        result['train_features'][:, 0],
        bins=30,
        density=True,
        alpha=0.6,
        color=COLORS['blue'],
        label='Training',
    )
    axes[0, 0].hist(
        result['covariate_features'][:, 0],
        bins=30,
        density=True,
        alpha=0.6,
        color=COLORS['orange'],
        label='Production',
    )
    axes[0, 0].set(
        title=f'Covariate drift: Wasserstein={covariate_distance:.2f}',
        xlabel='Feature 1',
        ylabel='Density',
    )
    axes[0, 0].legend()

    axes[0, 1].hist(
        result['train_features'][:, 0],
        bins=30,
        density=True,
        alpha=0.6,
        color=COLORS['blue'],
        label='Training',
    )
    axes[0, 1].hist(
        result['concept_features'][:, 0],
        bins=30,
        density=True,
        alpha=0.6,
        color=COLORS['green'],
        label='Production',
    )
    axes[0, 1].set(
        title=f'Concept drift: Wasserstein={concept_distance:.2f}',
        xlabel='Feature 1',
        ylabel='Density',
    )
    axes[0, 1].legend()

    for axis, features, target, title in (
        (
            axes[1, 0],
            result['covariate_features'],
            result['covariate_target'],
            f'Same P(Y|X), old model AUC={covariate_auc:.3f}',
        ),
        (
            axes[1, 1],
            result['concept_features'],
            result['concept_target'],
            f'Changed P(Y|X), old model AUC={concept_auc:.3f}',
        ),
    ):
        axis.scatter(
            features[:, 0],
            features[:, 1],
            c=target,
            cmap='coolwarm',
            s=12,
            alpha=0.5,
        )
        limits = (-4, 4)
        _draw_boundary(axis, result['model'], limits, limits)
        axis.set(title=title, xlim=limits, ylim=limits, xlabel='Feature 1', ylabel='Feature 2')
        _style_axis(axis)
    for axis in axes[0]:
        _style_axis(axis, grid_axis='y')
    figure.suptitle(
        'P(X) can change without P(Y|X), and P(Y|X) can change while P(X) looks stable',
        fontweight='bold',
    )
    figure.tight_layout()
    if output_path is not None:
        return _save_figure(figure, output_path)
    return figure


def plot_training_serving_skew() -> plt.Figure:
    '''Show one raw entity receiving inconsistent time-based transformations.'''
    result = serving_skew_example()
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].bar(
        ('Training snapshot', 'Production bug'),
        (result['training_age_days'], result['production_age_days']),
        color=(COLORS['blue'], COLORS['red']),
    )
    axes[0].set(
        ylabel='customer_age_days',
        title='Same raw customer, different reference date',
    )
    axes[1].bar(
        ('Shared pipeline', 'Skewed transform'),
        (result['training_probability'], result['skewed_probability']),
        color=(COLORS['green'], COLORS['red']),
    )
    axes[1].set(
        ylabel='Predicted probability',
        ylim=(0, 1),
        title='The transformation bug changes the decision input',
    )
    for axis in axes:
        _style_axis(axis, grid_axis='y')
    figure.suptitle(
        'Training-serving skew: feature semantics are part of the model contract',
        fontweight='bold',
    )
    figure.tight_layout()
    return figure


def plot_monitoring(
    drift_magnitude: float = 1.0,
    concept_magnitude: float = 0.8,
) -> plt.Figure:
    '''Plot synthetic service, input, prediction, and delayed-label signals.'''
    data = monitoring_simulation(drift_magnitude, concept_magnitude)
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8))
    panels = (
        ('feature_mean', 'Input feature mean', COLORS['blue']),
        ('prediction_rate', 'Positive prediction rate', COLORS['orange']),
        ('latency_ms', 'Service latency (ms)', COLORS['purple']),
        ('delayed_roc_auc', 'Delayed-label ROC-AUC', COLORS['red']),
    )
    for axis, (column, title, color) in zip(axes.ravel(), panels, strict=True):
        axis.plot(data['week'], data[column], marker='o', color=color, linewidth=2)
        axis.set(title=title, xlabel='Week')
        _style_axis(axis)
    figure.suptitle(
        'Synthetic monitoring simulation — offline score is only one signal',
        fontweight='bold',
    )
    figure.tight_layout()
    return figure


def plot_retraining_loop() -> plt.Figure:
    '''Show evaluation and promotion between production and a candidate model.'''
    stages = (
        'Production\ndata',
        'Monitoring',
        'Trigger?',
        'New training\ndataset',
        'Candidate\nmodel',
        'Evaluation',
        'Compare with\nincumbent',
        'Promotion\ndecision',
    )
    figure, axis = plt.subplots(figsize=(12, 4.5))
    axis.axis('off')
    positions = np.linspace(0.06, 0.94, len(stages))
    for index, (position, stage) in enumerate(zip(positions, stages, strict=True)):
        color = COLORS['green'] if stage == 'Promotion\ndecision' else COLORS['blue']
        axis.scatter(position, 0.55, s=1_600, color=color, edgecolor='white')
        axis.text(
            position,
            0.55,
            stage,
            ha='center',
            va='center',
            color='white',
            fontsize=8.5,
            fontweight='bold',
        )
        if index < len(stages) - 1:
            axis.annotate(
                '',
                xy=(positions[index + 1] - 0.03, 0.55),
                xytext=(position + 0.03, 0.55),
                arrowprops={'arrowstyle': '->', 'color': COLORS['gray']},
            )
    axis.annotate(
        'Reject / investigate / revise',
        xy=(positions[5], 0.48),
        xytext=(positions[3], 0.16),
        color=COLORS['red'],
        arrowprops={
            'arrowstyle': '->',
            'color': COLORS['red'],
            'connectionstyle': 'arc3,rad=0.2',
        },
    )
    axis.set_title(
        'New data does not imply automatic replacement',
        fontweight='bold',
        fontsize=14,
    )
    axis.set(xlim=(0, 1), ylim=(0, 1))
    return figure


def animate_concept_drift(output_path: Path) -> Path:
    '''Animate a rotating production boundary and old-model performance.'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    magnitudes = np.linspace(0.0, 1.7, 24)
    figure, axis = plt.subplots(figsize=(7.2, 5.8))

    def update(frame: int) -> None:
        axis.clear()
        result = drift_experiment(0.0, float(magnitudes[frame]))
        features = result['concept_features']
        target = result['concept_target']
        axis.scatter(
            features[:, 0],
            features[:, 1],
            c=target,
            cmap='coolwarm',
            s=12,
            alpha=0.5,
        )
        _draw_boundary(axis, result['model'], (-3.5, 3.5), (-3.5, 3.5))
        weights = result['rotated_weights']
        x_values = np.array([-3.5, 3.5])
        if abs(weights[1]) > 1e-6:
            axis.plot(
                x_values,
                -weights[0] / weights[1] * x_values,
                color=COLORS['green'],
                linewidth=2,
                label='Current production boundary',
            )
        auc = result['concept_auc']
        axis.set(
            xlim=(-3.5, 3.5),
            ylim=(-3.5, 3.5),
            xlabel='Feature 1',
            ylabel='Feature 2',
            title=f'Concept rotation={magnitudes[frame]:.2f} rad | old-model AUC={auc:.3f}',
        )
        axis.legend(loc='upper right')
        _style_axis(axis)

    animation = FuncAnimation(figure, update, frames=len(magnitudes), interval=130)
    animation.save(output_path, writer=PillowWriter(fps=8), dpi=95)
    plt.close(figure)
    return output_path


ASSET_GENERATORS = {
    'ml_lifecycle.png': lambda path: plot_lifecycle(path),
    'random_vs_temporal_split.png': lambda path: plot_split_comparison(1.3, path),
    'data_leakage.png': lambda path: plot_data_leakage(path),
    'threshold_tradeoff.png': lambda path: plot_threshold_tradeoff(0.35, 1.0, 5.0, path),
    'covariate_vs_concept_drift.png': lambda path: plot_drift_comparison(1.0, 1.0, path),
    'concept_drift.gif': animate_concept_drift,
}


def generate_assets(output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    '''Generate the bounded public-preview candidate set.'''
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        generator(output_dir / filename)
        for filename, generator in ASSET_GENERATORS.items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Destination for generated preview candidates.',
    )
    args = parser.parse_args()
    for path in generate_assets(args.output_dir):
        print(path)


if __name__ == '__main__':
    main()

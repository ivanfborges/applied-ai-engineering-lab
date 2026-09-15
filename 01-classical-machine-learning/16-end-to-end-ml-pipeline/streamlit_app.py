'''Interactive visual laboratory for end-to-end ML pipeline decisions.'''

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from simulation import (
    baseline_experiment,
    compare_random_and_temporal_splits,
    drift_experiment,
    group_leakage_experiment,
    leakage_experiment,
    monitoring_simulation,
    serving_skew_example,
    threshold_experiment,
    threshold_metrics,
    validation_search_experiment,
)
from visualizations import (
    plot_baseline_metrics,
    plot_complexity,
    plot_cross_validation,
    plot_data_leakage,
    plot_drift_comparison,
    plot_group_leakage,
    plot_lifecycle,
    plot_monitoring,
    plot_prediction_time_boundary,
    plot_retraining_loop,
    plot_split_comparison,
    plot_threshold_tradeoff,
    plot_training_serving_skew,
    plot_validation_search,
    probability_surface_3d,
)


st.set_page_config(
    page_title='End-to-end ML pipeline lab',
    page_icon=':material/account_tree:',
    layout='wide',
)


@st.cache_data(max_entries=24, show_spinner=False)
def cached_figure(name: str, parameters: tuple[float, ...] = ()) -> plt.Figure:
    '''Cache bounded deterministic figures used by interactive controls.'''
    generators = {
        'lifecycle': lambda: plot_lifecycle(),
        'prediction_time': plot_prediction_time_boundary,
        'splits': lambda: plot_split_comparison(parameters[0]),
        'groups': plot_group_leakage,
        'leakage': lambda: plot_data_leakage(),
        'baseline': lambda: plot_baseline_metrics(parameters[0]),
        'threshold': lambda: plot_threshold_tradeoff(*parameters),
        'complexity': plot_complexity,
        'validation_search': plot_validation_search,
        'cross_validation': plot_cross_validation,
        'drift': lambda: plot_drift_comparison(*parameters),
        'serving_skew': plot_training_serving_skew,
        'monitoring': lambda: plot_monitoring(*parameters),
        'retraining': plot_retraining_loop,
    }
    if name not in generators:
        raise ValueError(f'Unknown figure name: {name}')
    return generators[name]()


@st.cache_data(max_entries=24, show_spinner=False)
def cached_metrics(name: str, parameters: tuple[float, ...] = ()) -> object:
    '''Cache deterministic experiment results displayed in metric cards.'''
    generators = {
        'splits': lambda: compare_random_and_temporal_splits(parameters[0]),
        'groups': group_leakage_experiment,
        'leakage': leakage_experiment,
        'baseline': lambda: baseline_experiment(parameters[0]),
        'threshold': threshold_experiment,
        'validation_search': validation_search_experiment,
        'drift': lambda: drift_experiment(*parameters),
        'serving_skew': serving_skew_example,
        'monitoring': lambda: monitoring_simulation(*parameters),
    }
    if name not in generators:
        raise ValueError(f'Unknown metric name: {name}')
    return generators[name]()


def show_figure(figure: plt.Figure) -> None:
    st.pyplot(figure, width='stretch')


def explain(
    what: str,
    why: str,
    failure: str,
    takeaway: str,
) -> None:
    '''Render the four requested interpretation layers without visual clutter.'''
    with st.container(border=True):
        st.markdown(
            f'''**What you are seeing:** {what}

**Why it matters:** {why}

**What can go wrong:** {failure}

**Senior interview takeaway:** {takeaway}'''
        )


SECTIONS = (
    'Lifecycle & prediction time',
    'Validation boundaries',
    'Leakage & baselines',
    'Thresholds & business cost',
    'Complexity & selection',
    'Cross-validation',
    'Covariate & concept drift',
    'Serving, monitoring & retraining',
)

with st.sidebar:
    section = st.selectbox('Laboratory section', SECTIONS, key='lab_section')
    st.caption('All data and monitoring signals are deterministic synthetic demonstrations.')
    st.badge('Seed 16', icon=':material/replay:', color='blue')

st.title('End-to-end ML pipeline laboratory')
st.caption(
    'How choices before, during, and after training change whether a model '
    'looks successful or supports a reliable production decision.'
)


if section == 'Lifecycle & prediction time':
    st.header('The model is one component')
    show_figure(cached_figure('lifecycle'))
    explain(
        'A decision lifecycle with training highlighted as one stage.',
        'Failures in labels, validation, serving, or monitoring can dominate algorithm quality.',
        'Teams can optimize model.fit() while the surrounding decision system is invalid.',
        'Start from the action and information boundary, then design evidence and operations.',
    )

    st.subheader('Prediction-time boundary')
    show_figure(cached_figure('prediction_time'))
    explain(
        'Features on the left exist at prediction time; outcomes on the right arrive later.',
        'Temporal availability determines whether a feature is valid.',
        'A future-derived database column can make offline evaluation look excellent.',
        'Ask whether the exact value and transformation existed at the real decision moment.',
    )


elif section == 'Validation boundaries':
    st.header('A split is a deployment assumption')
    drift = st.slider(
        'Concept drift magnitude',
        min_value=0.0,
        max_value=2.2,
        value=1.3,
        step=0.1,
        key='split_drift',
    )
    result = cached_metrics('splits', (drift,))
    with st.container(horizontal=True):
        st.metric('Random-split ROC-AUC', result.random_auc, format='%.3f', border=True)
        st.metric('Temporal-split ROC-AUC', result.temporal_auc, format='%.3f', border=True)
        st.metric(
            'Apparent optimism',
            result.random_auc - result.temporal_auc,
            format='%.3f',
            border=True,
        )
    show_figure(cached_figure('splits', (drift,)))
    explain(
        'A random holdout mixes periods, while the temporal holdout contains only later rows.',
        'Future production requires evidence across a time boundary.',
        'Random mixing can let later regimes influence training and overstate future performance.',
        'Validation should reproduce the boundary the deployed system must cross.',
    )

    st.subheader('Repeated entities')
    group_result = cached_metrics('groups')
    with st.container(horizontal=True):
        st.metric('Row-split entity overlap', group_result['row_overlap'], border=True)
        st.metric('Group-split entity overlap', group_result['group_overlap'], border=True)
        st.metric('Row-split ROC-AUC', group_result['row_auc'], format='%.3f', border=True)
        st.metric('Group-split ROC-AUC', group_result['group_auc'], format='%.3f', border=True)
    show_figure(cached_figure('groups'))
    explain(
        'Rows from the same synthetic customer cross the random row split but not the group split.',
        'Entity identity can act like memorized information when observations repeat.',
        'A row-disjoint test set may still share customers, patients, devices, or documents.',
        'Choose the independent unit required in production, not merely an independent row.',
    )


elif section == 'Leakage & baselines':
    st.header('A better score can be invalid')
    leakage = cached_metrics('leakage')
    with st.container(horizontal=True):
        st.metric('Valid-feature ROC-AUC', leakage['valid_auc'], format='%.3f', border=True)
        st.metric('Leaky-feature ROC-AUC', leakage['leaky_auc'], format='%.3f', border=True)
    show_figure(cached_figure('leakage'))
    explain(
        'The same classifier gains a near-label feature derived after the outcome.',
        'Score improvements are meaningful only when the information contract is valid.',
        'Target leakage can produce a highly convincing but undeployable model.',
        'Suspiciously strong performance should trigger lineage and prediction-time review.',
    )

    st.subheader('Baselines and metric choice')
    positive_fraction = st.slider(
        'Positive-class fraction',
        min_value=0.02,
        max_value=0.30,
        value=0.08,
        step=0.01,
        format='%.2f',
        key='baseline_prevalence',
    )
    metrics = cached_metrics('baseline', (positive_fraction,))
    majority_accuracy = float(
        metrics.loc[metrics['model'] == 'Majority class', 'accuracy'].iloc[0]
    )
    learned_pr_auc = float(
        metrics.loc[metrics['model'] == 'Logistic regression', 'pr_auc'].iloc[0]
    )
    with st.container(horizontal=True):
        st.metric('Majority accuracy', majority_accuracy, format='%.3f', border=True)
        st.metric('Positive prevalence', positive_fraction, format='percent', border=True)
        st.metric('Logistic PR-AUC', learned_pr_auc, format='%.3f', border=True)
    show_figure(cached_figure('baseline', (positive_fraction,)))
    st.dataframe(metrics.round(3), hide_index=True)
    explain(
        'A majority classifier can show high accuracy while never identifying a positive case.',
        'Different metrics answer different ranking and decision questions.',
        'Optimizing accuracy under severe imbalance can reward a useless policy.',
        'Report prevalence, a relevant baseline, and metrics tied to error costs and capacity.',
    )


elif section == 'Thresholds & business cost':
    st.header('Probability model versus decision policy')
    with st.container(horizontal=True, vertical_alignment='bottom'):
        threshold = st.slider(
            'Decision threshold',
            min_value=0.05,
            max_value=0.95,
            value=0.35,
            step=0.05,
            key='decision_threshold',
        )
        false_positive_cost = st.number_input(
            'False-positive cost',
            min_value=0.0,
            max_value=50.0,
            value=1.0,
            step=0.5,
            key='fp_cost',
        )
        false_negative_cost = st.number_input(
            'False-negative cost',
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=0.5,
            key='fn_cost',
        )
    experiment = cached_metrics('threshold')
    metrics = threshold_metrics(
        experiment['y_test'],
        experiment['probabilities'],
        threshold,
        false_positive_cost,
        false_negative_cost,
    )
    with st.container(horizontal=True):
        st.metric('Precision', metrics['precision'], format='%.3f', border=True)
        st.metric('Recall', metrics['recall'], format='%.3f', border=True)
        st.metric('F1', metrics['f1'], format='%.3f', border=True)
        st.metric('Predicted positives', metrics['predicted_positives'], border=True)
        st.metric('Synthetic cost', metrics['cost'], format='%.1f', border=True)
    show_figure(
        cached_figure(
            'threshold',
            (threshold, false_positive_cost, false_negative_cost),
        )
    )
    confusion = [
        {'actual': 'Negative', 'predicted negative': metrics['tn'], 'predicted positive': metrics['fp']},
        {'actual': 'Positive', 'predicted negative': metrics['fn'], 'predicted positive': metrics['tp']},
    ]
    st.dataframe(confusion, hide_index=True)
    explain(
        'One probability distribution supports many thresholds, confusion matrices, and costs.',
        'The useful operating point depends on errors, intervention capacity, and calibration.',
        'Treating 0.5 as inherently optimal mixes estimation with policy.',
        'Lock threshold selection on development evidence before final testing.',
    )

    if st.toggle('Show rotatable probability surface', key='show_probability_surface'):
        st.plotly_chart(
            probability_surface_3d(),
            width='stretch',
            key='probability_surface',
            config={'scrollZoom': False},
        )
        st.caption(
            'The continuous surface is the model output; a horizontal threshold '
            'turns it into a binary action boundary.'
        )


elif section == 'Complexity & selection':
    st.header('Underfitting, reasonable fit, and high complexity')
    show_figure(cached_figure('complexity'))
    explain(
        'Polynomial logistic boundaries become increasingly flexible on the same noisy moons data.',
        'Training fit and validation generalization need not improve together.',
        'A complex model can absorb sample-specific noise and become unstable.',
        'Complexity is justified by reproducible development evidence, not training score.',
    )

    st.subheader('Repeated validation search')
    search = cached_metrics('validation_search')
    show_figure(cached_figure('validation_search'))
    st.dataframe(search.round(4), hide_index=True)
    explain(
        'All candidates have equal true accuracy, yet the best reused-validation score rises with search.',
        'Selection favors positive validation noise even without a genuinely better candidate.',
        'Repeated human or automated tuning can overfit a fixed holdout.',
        'Govern search, quantify uncertainty, and preserve independent final evidence.',
    )


elif section == 'Cross-validation':
    st.header('Cross-validation rotates evidence, not preprocessing leakage')
    show_figure(cached_figure('cross_validation'))
    explain(
        'Each fold takes one turn as validation while learned preprocessing stays fold-local.',
        'Cross-validation uses limited development data more fully.',
        'Fitting a scaler, encoder, or selector on all rows contaminates every validation fold.',
        'Put every learned transformation inside the cross-validated pipeline.',
    )


elif section == 'Covariate & concept drift':
    st.header('P(X) and P(Y|X) are different monitoring questions')
    with st.container(horizontal=True):
        covariate_shift = st.slider(
            'Covariate shift',
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.1,
            key='covariate_shift',
        )
        concept_rotation = st.slider(
            'Concept rotation',
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.1,
            key='concept_rotation',
        )
    result = cached_metrics(
        'drift',
        (covariate_shift, concept_rotation),
    )
    with st.container(horizontal=True):
        st.metric('Training ROC-AUC', result['train_auc'], format='%.3f', border=True)
        st.metric(
            'Covariate-shift ROC-AUC',
            result['covariate_auc'],
            format='%.3f',
            border=True,
        )
        st.metric(
            'Concept-drift ROC-AUC',
            result['concept_auc'],
            format='%.3f',
            border=True,
        )
        st.metric(
            'Covariate X1 distance',
            result['wasserstein_x1'],
            format='%.2f',
            border=True,
        )
    show_figure(
        cached_figure(
            'drift',
            (covariate_shift, concept_rotation),
        )
    )
    explain(
        'One panel changes feature marginals while keeping the label rule; another rotates the rule while feature marginals stay similar.',
        'Input monitoring and delayed-label performance diagnose different distribution changes.',
        'Stable P(X) can hide concept drift, while changed P(X) does not prove performance loss.',
        'Monitor contracts and inputs immediately, then connect delayed labels to model and business quality.',
    )


elif section == 'Serving, monitoring & retraining':
    st.header('Training-serving consistency')
    result = cached_metrics('serving_skew')
    with st.container(horizontal=True):
        st.metric(
            'Training age feature',
            result['training_age_days'],
            format='%.0f days',
            border=True,
        )
        st.metric(
            'Skewed production feature',
            result['production_age_days'],
            format='%.0f days',
            border=True,
        )
        st.metric(
            'Probability change',
            result['skewed_probability'] - result['training_probability'],
            format='%.3f',
            border=True,
        )
    show_figure(cached_figure('serving_skew'))
    explain(
        'A reference-date bug gives the same raw customer a different age feature in serving.',
        'Feature semantics and transformations are part of the deployed model.',
        'A healthy endpoint can return systematically wrong scores.',
        'Version and test the complete raw-input-to-decision contract.',
    )

    st.subheader('Production monitoring')
    with st.container(horizontal=True):
        monitoring_drift = st.slider(
            'Input drift over 12 weeks',
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.1,
            key='monitoring_drift',
        )
        monitoring_concept = st.slider(
            'Concept drift over 12 weeks',
            min_value=0.0,
            max_value=2.0,
            value=0.8,
            step=0.1,
            key='monitoring_concept',
        )
    monitoring = cached_metrics(
        'monitoring',
        (monitoring_drift, monitoring_concept),
    )
    with st.container(horizontal=True):
        st.metric(
            'Week 12 feature mean',
            monitoring['feature_mean'].iloc[-1],
            format='%.2f',
            border=True,
        )
        st.metric(
            'Week 12 positive rate',
            monitoring['prediction_rate'].iloc[-1],
            format='percent',
            border=True,
        )
        st.metric(
            'Week 12 delayed ROC-AUC',
            monitoring['delayed_roc_auc'].iloc[-1],
            format='%.3f',
            border=True,
        )
    show_figure(
        cached_figure(
            'monitoring',
            (monitoring_drift, monitoring_concept),
        )
    )
    explain(
        'Synthetic weekly panels cover input, prediction, latency, and delayed-label quality.',
        'An offline score cannot reveal serving failures or future degradation by itself.',
        'Monitoring only latency or only P(X) leaves important failure modes invisible.',
        'Define signals, label delay, response playbooks, owners, and rollback before launch.',
    )

    st.subheader('Retraining is a promotion process')
    show_figure(cached_figure('retraining'))
    explain(
        'New production data creates a candidate that must be evaluated against the incumbent.',
        'Retraining changes data, parameters, and potentially the operating policy.',
        'Automatic replacement can promote a worse or incompatible model.',
        'Trigger, train, evaluate, compare, approve, promote, monitor, and retain rollback.',
    )

st.caption(
    'Synthetic educational laboratory. Observed metrics describe the configured '
    'simulations only and are not production benchmarks.'
)

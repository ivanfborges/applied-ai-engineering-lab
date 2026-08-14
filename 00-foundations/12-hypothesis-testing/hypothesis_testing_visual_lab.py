"""Interactive visual laboratory for hypothesis testing with synthetic data."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

from example import generate_synthetic_scores
from statistical_utils import (
    confidence_test_connection,
    normal_error_rates,
    normal_mean_test,
    one_sample_t_power,
    paired_visual_summary,
    power_curve,
    power_surface,
    practical_significance_scenarios,
    sign_flip_distribution,
    simulate_confidence_intervals,
    simulate_multiple_testing,
    simulate_t_experiments,
)


st.set_page_config(
    page_title="Hypothesis testing visual lab",
    page_icon=":material/query_stats:",
    layout="wide",
)

BLUE = "#2563EB"
RED = "#DC2626"
ORANGE = "#F59E0B"
GREEN = "#16A34A"
PURPLE = "#7C3AED"
GRAY = "#64748B"
LIGHT_GRAY = "#E2E8F0"
PLOT_CONFIG = {"displaylogo": False, "scrollZoom": False}

SECTIONS = (
    "Test intuition",
    "Errors and power",
    "Repeated experiments",
    "Paired AI experiment",
    "Randomization test",
    "Multiple testing",
    "Confidence intervals",
    "Practical significance",
)


def style_figure(
    figure: go.Figure,
    *,
    title: str,
    x_title: str,
    y_title: str,
    height: int = 470,
) -> go.Figure:
    """Apply consistent, readable Plotly layout settings."""
    figure.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=height,
        margin=dict(l=40, r=30, t=70, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
        hovermode="x unified",
    )
    return figure


def add_filled_region(
    figure: go.Figure,
    x_values: np.ndarray,
    density: np.ndarray,
    mask: np.ndarray,
    *,
    name: str,
    color: str,
    opacity: float,
) -> None:
    """Add one explicitly labeled density region."""
    figure.add_trace(
        go.Scatter(
            x=x_values[mask],
            y=density[mask],
            mode="lines",
            line=dict(width=0),
            fill="tozeroy",
            fillcolor=color,
            opacity=opacity,
            name=name,
            hoverinfo="skip",
        )
    )


@st.cache_data(max_entries=32, show_spinner=False)
def cached_t_simulation(
    true_mean: float,
    sample_size: int,
    standard_deviation: float,
    simulations: int,
    alpha: float,
    alternative: str,
    seed: int,
) -> pd.DataFrame:
    """Cache bounded repeated-test simulations across reruns."""
    return simulate_t_experiments(
        true_mean=true_mean,
        sample_size=sample_size,
        standard_deviation=standard_deviation,
        simulations=simulations,
        alpha=alpha,
        alternative=alternative,
        seed=seed,
    )


@st.cache_data(max_entries=16, show_spinner=False)
def cached_power_surface(alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cache the noncentral-t surface because it is independent of random data."""
    sample_sizes = np.unique(np.linspace(5, 500, 70, dtype=int))
    effects = np.linspace(0.0, 1.5, 55)
    return sample_sizes, effects, power_surface(sample_sizes, effects, alpha)


@st.cache_data(max_entries=32, show_spinner=False)
def cached_paired_scores(
    sample_size: int,
    mean_improvement: float,
    difference_sd: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cache the deterministic synthetic RAG evaluation."""
    return generate_synthetic_scores(
        sample_size=sample_size,
        mean_improvement=mean_improvement,
        difference_sd=difference_sd,
        seed=seed,
    )


@st.cache_data(max_entries=32, show_spinner=False)
def cached_sign_flip(
    differences: tuple[float, ...], permutations: int, seed: int
):
    """Cache the bounded randomization distribution."""
    return sign_flip_distribution(differences, permutations=permutations, seed=seed)


@st.cache_data(max_entries=24, show_spinner=False)
def cached_multiple_testing(
    hypotheses: int, repetitions: int, alpha: float, seed: int
):
    """Cache repeated all-null multiple-testing families."""
    return simulate_multiple_testing(
        hypotheses=hypotheses,
        repetitions=repetitions,
        alpha=alpha,
        seed=seed,
    )


@st.cache_data(max_entries=24, show_spinner=False)
def cached_intervals(
    true_mean: float,
    sample_size: int,
    intervals: int,
    confidence: float,
    standard_deviation: float,
    seed: int,
) -> pd.DataFrame:
    """Cache repeated confidence-interval simulations."""
    return simulate_confidence_intervals(
        true_mean=true_mean,
        sample_size=sample_size,
        intervals=intervals,
        confidence=confidence,
        standard_deviation=standard_deviation,
        seed=seed,
    )


def render_test_intuition(
    effect: float,
    sample_size: int,
    standard_deviation: float,
    alpha: float,
    alternative: str,
) -> None:
    """Render the null distribution, rejection region, and p-value area."""
    st.header("Null distribution and p-value")
    st.caption(
        "Known-SD normal reference for H₀: μ = 0. The horizontal scale is the "
        "standardized test statistic under the null."
    )
    result = normal_mean_test(
        effect, sample_size, standard_deviation, alpha, alternative
    )
    x_values = np.linspace(-4.5, 4.5, 1_200)
    density = stats.norm.pdf(x_values)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=density,
            mode="lines",
            line=dict(color=BLUE, width=3),
            name="Null distribution",
        )
    )

    if alternative == "two-sided":
        critical = result.critical_upper
        rejection_mask = np.abs(x_values) >= critical
        p_value_mask = np.abs(x_values) >= abs(result.statistic)
        for boundary in (-critical, critical):
            figure.add_vline(
                x=boundary,
                line_dash="dash",
                line_color=RED,
                annotation_text="critical",
            )
    elif alternative == "greater":
        rejection_mask = x_values >= result.critical_lower
        p_value_mask = x_values >= result.statistic
        figure.add_vline(
            x=result.critical_lower,
            line_dash="dash",
            line_color=RED,
            annotation_text="critical",
        )
    else:
        rejection_mask = x_values <= result.critical_upper
        p_value_mask = x_values <= result.statistic
        figure.add_vline(
            x=result.critical_upper,
            line_dash="dash",
            line_color=RED,
            annotation_text="critical",
        )

    add_filled_region(
        figure,
        x_values,
        density,
        rejection_mask,
        name="Rejection region (α)",
        color=RED,
        opacity=0.22,
    )
    add_filled_region(
        figure,
        x_values,
        density,
        p_value_mask,
        name="At least as extreme as observed",
        color=ORANGE,
        opacity=0.42,
    )
    figure.add_vline(x=0.0, line_color=GRAY, annotation_text="H₀ center")
    figure.add_vline(
        x=result.statistic,
        line_color=PURPLE,
        line_width=3,
        annotation_text="observed statistic",
    )
    style_figure(
        figure,
        title="Evidence is measured relative to the null distribution",
        x_title="Test statistic z",
        y_title="Probability density",
    )
    st.plotly_chart(figure, width="stretch", config=PLOT_CONFIG, key="null_plot")

    with st.container(horizontal=True):
        st.metric("Observed effect", f"{effect:.4f}", border=True)
        st.metric("Standard error", f"{result.standard_error:.4f}", border=True)
        st.metric("Test statistic", f"{result.statistic:.3f}", border=True)
        st.metric("p-value", f"{result.p_value:.5f}", border=True)
        st.metric("Alpha", f"{alpha:.3f}", border=True)
        st.metric(
            "Decision",
            "Reject H₀" if result.reject else "Fail to reject H₀",
            border=True,
        )

    st.info(
        "The p-value is **not** P(H₀ | data). It is the probability, under the "
        "null model, of a statistic at least as extreme as the observed one.",
        icon=":material/info:",
    )


def render_error_geometry(
    effect: float, sample_size: int, standard_deviation: float, alpha: float
) -> None:
    """Render overlapping H0/H1 sampling distributions and error regions."""
    result = normal_error_rates(effect, sample_size, standard_deviation, alpha)
    spread = max(standard_deviation / math.sqrt(sample_size), 1e-6)
    x_min = min(-4.2 * spread, effect - 4.2 * spread)
    x_max = max(4.2 * spread, effect + 4.2 * spread)
    x_values = np.linspace(x_min, x_max, 1_200)
    h0_density = stats.norm.pdf(x_values, loc=0.0, scale=spread)
    h1_density = stats.norm.pdf(x_values, loc=effect, scale=spread)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=h0_density,
            line=dict(color=BLUE, width=3),
            name="Distribution under H₀",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=h1_density,
            line=dict(color=PURPLE, width=3),
            name="Distribution under H₁",
        )
    )
    rejection = x_values >= result.critical_mean
    non_rejection = ~rejection
    add_filled_region(
        figure,
        x_values,
        h0_density,
        rejection,
        name="Type I error α",
        color=RED,
        opacity=0.35,
    )
    add_filled_region(
        figure,
        x_values,
        h1_density,
        non_rejection,
        name="Type II error β",
        color=ORANGE,
        opacity=0.38,
    )
    add_filled_region(
        figure,
        x_values,
        h1_density,
        rejection,
        name="Power 1 − β",
        color=GREEN,
        opacity=0.34,
    )
    figure.add_vline(
        x=result.critical_mean,
        line_dash="dash",
        line_color=RED,
        annotation_text="rejection boundary",
    )
    style_figure(
        figure,
        title="One-sided superiority test: H₀ versus a specified H₁",
        x_title="Sample mean",
        y_title="Sampling density",
    )
    st.plotly_chart(figure, width="stretch", config=PLOT_CONFIG, key="errors_plot")
    with st.container(horizontal=True):
        st.metric("Type I error α", f"{result.alpha:.3f}", border=True)
        st.metric("Type II error β", f"{result.beta:.3f}", border=True)
        st.metric("Power 1 − β", f"{result.power:.3f}", border=True)
        st.metric("Rejection boundary", f"{result.critical_mean:.4f}", border=True)
    st.caption(
        "This panel uses a one-sided known-SD normal model so the shaded areas "
        "exactly match the calculations. Larger n or effect separates H₁ from H₀; "
        "more variance increases their overlap."
    )


def render_power_curve(
    effect: float, standard_deviation: float, alpha: float, alternative: str
) -> None:
    """Render noncentral-t power across sample sizes."""
    standardized_effect = abs(effect) / standard_deviation
    sample_sizes = np.arange(5, 501)
    powers = power_curve(sample_sizes, standardized_effect, alpha, alternative)
    figure = go.Figure(
        go.Scatter(
            x=sample_sizes,
            y=powers,
            mode="lines",
            line=dict(color=BLUE, width=4),
            name=f"d = {standardized_effect:.3f}",
        )
    )
    for target, color in ((0.8, ORANGE), (0.9, GREEN)):
        figure.add_hline(
            y=target,
            line_dash="dash",
            line_color=color,
            annotation_text=f"{target:.0%} power",
        )
    style_figure(
        figure,
        title="One-sample t-test power from the noncentral t distribution",
        x_title="Sample size",
        y_title="Statistical power",
    )
    figure.update_yaxes(range=[0.0, 1.02], tickformat=".0%")
    st.plotly_chart(figure, width="stretch", config=PLOT_CONFIG, key="power_curve")

    required = {}
    for target in (0.8, 0.9):
        indices = np.flatnonzero(powers >= target)
        required[target] = int(sample_sizes[indices[0]]) if indices.size else None
    with st.container(horizontal=True):
        st.metric("Standardized effect |d|", f"{standardized_effect:.3f}", border=True)
        st.metric(
            "First n reaching 80%",
            str(required[0.8]) if required[0.8] else "> 500",
            border=True,
        )
        st.metric(
            "First n reaching 90%",
            str(required[0.9]) if required[0.9] else "> 500",
            border=True,
        )
    st.caption(
        "Power is calculated for each integer n with the noncentral t distribution. "
        "The displayed sample-size thresholds are approximate planning values for "
        "this model, not universal requirements."
    )


def render_power_surface(alpha: float) -> None:
    """Render the interactive effect-size by sample-size power surface."""
    sample_sizes, effects, surface = cached_power_surface(alpha)
    figure = go.Figure(
        data=[
            go.Surface(
                x=sample_sizes,
                y=effects,
                z=surface,
                colorscale="Viridis",
                colorbar=dict(title="Power"),
                hovertemplate="n=%{x}<br>d=%{y:.2f}<br>power=%{z:.1%}<extra></extra>",
            )
        ]
    )
    figure.update_layout(
        title="Two-sided one-sample t-test power surface",
        height=650,
        margin=dict(l=20, r=20, t=70, b=20),
        scene=dict(
            xaxis_title="Sample size",
            yaxis_title="Standardized effect d",
            zaxis_title="Power",
            zaxis=dict(range=[0.0, 1.0], tickformat=".0%"),
            camera=dict(eye=dict(x=1.55, y=1.55, z=1.0)),
        ),
    )
    st.plotly_chart(
        figure,
        width="stretch",
        config={"displaylogo": False, "scrollZoom": True},
        key="power_surface",
    )
    st.caption(
        "Rotate and zoom the surface. Small effects sit on a long, shallow part of "
        "the surface: achieving high power there requires disproportionately more data."
    )


def render_errors_and_power(
    effect: float,
    sample_size: int,
    standard_deviation: float,
    alpha: float,
    alternative: str,
) -> None:
    """Render one of three complementary power views."""
    st.header("Type I error, Type II error, and statistical power")
    view = st.segmented_control(
        "Power view",
        ["Error regions", "Power curve", "3D power surface"],
        default="Error regions",
        required=True,
        key="power_view",
    )
    if view == "Error regions":
        render_error_geometry(effect, sample_size, standard_deviation, alpha)
    elif view == "Power curve":
        render_power_curve(effect, standard_deviation, alpha, alternative)
    else:
        render_power_surface(alpha)


def p_value_histogram(
    data: pd.DataFrame,
    *,
    alpha: float,
    title: str,
    key: str,
) -> None:
    """Render a p-value histogram with the rejection region marked."""
    p_values = data["p_value"].to_numpy()
    rejected = p_values[p_values < alpha]
    retained = p_values[p_values >= alpha]
    figure = go.Figure()
    figure.add_trace(
        go.Histogram(
            x=retained,
            xbins=dict(start=0.0, end=1.0, size=0.05),
            marker_color=BLUE,
            opacity=0.75,
            name="p ≥ α",
        )
    )
    figure.add_trace(
        go.Histogram(
            x=rejected,
            xbins=dict(start=0.0, end=1.0, size=0.05),
            marker_color=RED,
            opacity=0.85,
            name="p < α",
        )
    )
    figure.add_vline(
        x=alpha, line_dash="dash", line_color=RED, annotation_text="α"
    )
    figure.update_layout(barmode="stack")
    style_figure(
        figure,
        title=title,
        x_title="p-value",
        y_title="Experiments",
    )
    figure.update_xaxes(range=[0.0, 1.0])
    st.plotly_chart(figure, width="stretch", config=PLOT_CONFIG, key=key)


def render_repeated_experiments(
    effect: float,
    sample_size: int,
    standard_deviation: float,
    alpha: float,
    alternative: str,
    simulations: int,
    seed: int,
) -> None:
    """Render repeated-test sequences and p-value distributions."""
    st.header("Repeated experiments and p-value distributions")
    view = st.segmented_control(
        "Simulation view",
        ["False positives over time", "p-values under H₀", "p-values under H₁"],
        default="False positives over time",
        required=True,
        key="simulation_view",
    )
    true_mean = 0.0 if view != "p-values under H₁" else effect
    data = cached_t_simulation(
        true_mean,
        sample_size,
        standard_deviation,
        simulations,
        alpha,
        alternative,
        seed,
    )
    rejection_rate = float(data["reject"].mean())

    if view == "False positives over time":
        maximum_shown = min(simulations, 1_000)
        shown = st.slider(
            "Experiments shown",
            min_value=min(10, maximum_shown),
            max_value=maximum_shown,
            value=min(250, maximum_shown),
            step=10 if maximum_shown >= 10 else 1,
        )
        visible = data.iloc[:shown].copy()
        visible["cumulative_rate"] = visible["reject"].expanding().mean()
        colors = np.where(visible["reject"], RED, BLUE)
        figure = make_subplots(specs=[[{"secondary_y": True}]])
        figure.add_trace(
            go.Scatter(
                x=visible["experiment"],
                y=visible["p_value"],
                mode="markers",
                marker=dict(color=colors, size=7),
                name="Experiment p-value",
                hovertemplate="experiment=%{x}<br>p=%{y:.4f}<extra></extra>",
            ),
            secondary_y=False,
        )
        figure.add_trace(
            go.Scatter(
                x=visible["experiment"],
                y=visible["cumulative_rate"],
                line=dict(color=PURPLE, width=3),
                name="Cumulative false-positive rate",
            ),
            secondary_y=True,
        )
        figure.add_hline(y=alpha, line_dash="dash", line_color=RED)
        figure.update_layout(
            title="H₀ is true in every experiment; red points are false positives",
            height=500,
            margin=dict(l=40, r=40, t=70, b=45),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        figure.update_xaxes(title_text="Experiment sequence")
        figure.update_yaxes(title_text="p-value", range=[0.0, 1.0], secondary_y=False)
        figure.update_yaxes(
            title_text="Cumulative rejection rate",
            range=[0.0, max(0.2, alpha * 3.0)],
            tickformat=".0%",
            secondary_y=True,
        )
        st.plotly_chart(
            figure, width="stretch", config=PLOT_CONFIG, key="false_positive_sequence"
        )
        with st.container(horizontal=True):
            st.metric("Experiments", f"{simulations:,}", border=True)
            st.metric("False positives", int(data["reject"].sum()), border=True)
            st.metric("Empirical rate", f"{rejection_rate:.2%}", border=True)
            st.metric("Nominal alpha", f"{alpha:.2%}", border=True)
        st.caption(
            "The false-positive count is simulated, not hardcoded. Sampling variation "
            "keeps the realized rate from equaling alpha in every finite run."
        )
    elif view == "p-values under H₀":
        p_value_histogram(
            data,
            alpha=alpha,
            title="Under H₀, calibrated continuous-test p-values are approximately uniform",
            key="h0_p_values",
        )
        with st.container(horizontal=True):
            st.metric("Empirical false-positive rate", f"{rejection_rate:.2%}", border=True)
            st.metric("Expected rate", f"{alpha:.2%}", border=True)
            st.metric("Mean p-value", f"{data['p_value'].mean():.3f}", border=True)
    else:
        p_value_histogram(
            data,
            alpha=alpha,
            title="Under H₁, stronger information concentrates p-values near zero",
            key="h1_p_values",
        )
        theoretical = one_sample_t_power(
            sample_size, effect / standard_deviation, alpha, alternative
        )
        with st.container(horizontal=True):
            st.metric("Empirical power", f"{rejection_rate:.2%}", border=True)
            st.metric("Theoretical power", f"{theoretical:.2%}", border=True)
            st.metric("True synthetic effect", f"{effect:.4f}", border=True)
        st.caption(
            "The theoretical value uses the noncentral t distribution. Monte Carlo "
            "power is an estimate and is not expected to match it exactly."
        )


def paired_dataframe(
    sample_size: int, effect: float, standard_deviation: float, seed: int
) -> tuple[pd.DataFrame, object]:
    """Generate the paired synthetic RAG dataset and its statistical summary."""
    baseline, candidate = cached_paired_scores(
        sample_size, effect, standard_deviation, seed
    )
    frame = pd.DataFrame(
        {
            "query_id": np.arange(1, sample_size + 1),
            "baseline_score": baseline,
            "candidate_score": candidate,
            "difference": candidate - baseline,
        }
    )
    return frame, paired_visual_summary(baseline, candidate)


def render_paired_scores(frame: pd.DataFrame) -> None:
    """Render paired score movements for a readable subset of queries."""
    if len(frame) > 80:
        indices = np.linspace(0, len(frame) - 1, 80, dtype=int)
        visible = frame.iloc[indices]
    else:
        visible = frame
    x_lines: list[float | None] = []
    y_lines: list[float | None] = []
    for row in visible.itertuples(index=False):
        x_lines.extend([0.0, 1.0, None])
        y_lines.extend([row.baseline_score, row.candidate_score, None])
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x_lines,
            y=y_lines,
            mode="lines",
            line=dict(color=LIGHT_GRAY, width=1),
            name="Paired query",
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=np.zeros(len(visible)),
            y=visible["baseline_score"],
            mode="markers",
            marker=dict(color=BLUE, size=7, opacity=0.75),
            name="Baseline",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=np.ones(len(visible)),
            y=visible["candidate_score"],
            mode="markers",
            marker=dict(color=PURPLE, size=7, opacity=0.75),
            name="Candidate",
        )
    )
    style_figure(
        figure,
        title="Same queries scored by both RAG pipelines",
        x_title="System",
        y_title="Synthetic quality score",
    )
    figure.update_xaxes(tickvals=[0, 1], ticktext=["Baseline", "Candidate"])
    st.plotly_chart(figure, width="stretch", config=PLOT_CONFIG, key="paired_scores")
    if len(frame) > len(visible):
        st.caption(
            f"A deterministic subset of {len(visible)} pairs is drawn for readability; "
            f"all {len(frame)} pairs are used in every calculation."
        )


def render_difference_distribution(frame: pd.DataFrame, summary: object) -> None:
    """Render the within-query difference distribution."""
    figure = go.Figure(
        go.Histogram(
            x=frame["difference"],
            nbinsx=min(30, max(8, int(math.sqrt(len(frame)) * 2))),
            marker_color=BLUE,
            opacity=0.8,
            name="Paired differences",
        )
    )
    figure.add_vline(
        x=0.0, line_dash="dash", line_color=RED, annotation_text="no improvement"
    )
    figure.add_vline(
        x=summary.mean_difference,
        line_color=PURPLE,
        line_width=3,
        annotation_text="mean difference",
    )
    style_figure(
        figure,
        title="Candidate − baseline by query",
        x_title="Paired score difference",
        y_title="Queries",
    )
    st.plotly_chart(figure, width="stretch", config=PLOT_CONFIG, key="differences")


def render_effect_interval(summary: object, mpid: float, alpha: float) -> None:
    """Render effect uncertainty against a practical threshold."""
    lower, upper = summary.confidence_interval
    figure = go.Figure(
        go.Scatter(
            x=[summary.mean_difference],
            y=["Candidate − baseline"],
            mode="markers",
            marker=dict(color=PURPLE, size=14),
            error_x=dict(
                type="data",
                symmetric=False,
                array=[upper - summary.mean_difference],
                arrayminus=[summary.mean_difference - lower],
                color=PURPLE,
                thickness=4,
            ),
            name="Estimate and CI",
            hovertemplate="effect=%{x:.4f}<extra></extra>",
        )
    )
    figure.add_vline(x=0.0, line_dash="dash", line_color=RED, annotation_text="H₀")
    figure.add_vline(
        x=mpid,
        line_dash="dot",
        line_color=GREEN,
        annotation_text="MPID",
    )
    style_figure(
        figure,
        title=f"Estimated effect and {(1 - alpha):.0%} confidence interval",
        x_title="Candidate − baseline score",
        y_title="",
        height=360,
    )
    st.plotly_chart(figure, width="stretch", config=PLOT_CONFIG, key="paired_ci")
    with st.container(horizontal=True):
        st.metric("Mean difference", f"{summary.mean_difference:.4f}", border=True)
        st.metric("Confidence interval", f"[{lower:.4f}, {upper:.4f}]", border=True)
        st.metric("Paired t statistic", f"{summary.paired_t_statistic:.3f}", border=True)
        st.metric("Paired p-value", f"{summary.paired_p_value:.5f}", border=True)
        st.metric("Cohen's dz", f"{summary.cohens_dz:.3f}", border=True)
    st.warning(
        "Statistical significance is not practical significance. The MPID is an "
        "illustrative decision input; deployment also depends on latency, cost, "
        "reliability, safety, and evaluation validity.",
        icon=":material/warning:",
    )


def render_paired_vs_independent(frame: pd.DataFrame, summary: object) -> None:
    """Compare preserved pairing with an intentionally unpaired test."""
    variability = pd.DataFrame(
        {
            "quantity": ["Baseline scores", "Candidate scores", "Paired differences"],
            "sample_sd": [
                frame["baseline_score"].std(ddof=1),
                frame["candidate_score"].std(ddof=1),
                frame["difference"].std(ddof=1),
            ],
        }
    )
    figure = go.Figure(
        go.Bar(
            x=variability["quantity"],
            y=variability["sample_sd"],
            marker_color=[BLUE, PURPLE, GREEN],
            text=variability["sample_sd"].map(lambda value: f"{value:.4f}"),
            textposition="outside",
        )
    )
    style_figure(
        figure,
        title="Query difficulty cancels only when pairing is preserved",
        x_title="Analyzed quantity",
        y_title="Sample standard deviation",
    )
    st.plotly_chart(figure, width="stretch", config=PLOT_CONFIG, key="pairing_variance")
    with st.container(horizontal=True):
        st.metric("Paired SE", f"{summary.paired_standard_error:.4f}", border=True)
        st.metric("Independent SE", f"{summary.independent_standard_error:.4f}", border=True)
        st.metric("Paired p-value", f"{summary.paired_p_value:.5f}", border=True)
        st.metric("Welch p-value", f"{summary.independent_p_value:.5f}", border=True)
    st.caption(
        "The Welch result is shown as a deliberate design mismatch, not as a competing "
        "default. The app reports the generated outcome and does not guarantee that the "
        "paired p-value will always be smaller."
    )


def render_paired_experiment(
    effect: float,
    sample_size: int,
    standard_deviation: float,
    alpha: float,
    mpid: float,
    seed: int,
) -> None:
    """Render the paired RAG-system evaluation views."""
    st.header("Paired AI-system experiment")
    st.caption(
        "Synthetic data · baseline RAG pipeline versus the same queries scored by a "
        "candidate pipeline with reranking."
    )
    frame, summary = paired_dataframe(sample_size, effect, standard_deviation, seed)
    view = st.segmented_control(
        "Paired view",
        ["Score pairs", "Differences", "Effect and CI", "Paired vs independent"],
        default="Score pairs",
        required=True,
        key="paired_view",
    )
    if view == "Score pairs":
        render_paired_scores(frame)
    elif view == "Differences":
        render_difference_distribution(frame, summary)
    elif view == "Effect and CI":
        render_effect_interval(summary, mpid, alpha)
    else:
        render_paired_vs_independent(frame, summary)

    if st.toggle("Show synthetic query table", key="show_query_table"):
        st.dataframe(
            frame,
            hide_index=True,
            column_config={
                "query_id": st.column_config.NumberColumn("Query", format="%d"),
                "baseline_score": st.column_config.NumberColumn(
                    "Baseline score", format="%.4f"
                ),
                "candidate_score": st.column_config.NumberColumn(
                    "Candidate score", format="%.4f"
                ),
                "difference": st.column_config.NumberColumn("Difference", format="%.4f"),
            },
            key="paired_query_table",
        )


def render_randomization_test(
    effect: float,
    sample_size: int,
    standard_deviation: float,
    permutations: int,
    seed: int,
) -> None:
    """Render the paired sign-flip null distribution."""
    st.header("Paired t-test versus sign-flip randomization test")
    frame, summary = paired_dataframe(sample_size, effect, standard_deviation, seed)
    differences = tuple(float(value) for value in frame["difference"])
    result = cached_sign_flip(differences, permutations, seed + 1)
    null_values = result.null_statistics
    extreme = np.abs(null_values) >= abs(result.observed_mean) - 1e-12
    figure = go.Figure()
    figure.add_trace(
        go.Histogram(
            x=null_values[~extreme],
            nbinsx=60,
            marker_color=BLUE,
            opacity=0.75,
            name="Less extreme",
        )
    )
    figure.add_trace(
        go.Histogram(
            x=null_values[extreme],
            nbinsx=60,
            marker_color=RED,
            opacity=0.85,
            name="At least as extreme",
        )
    )
    figure.add_vline(
        x=result.observed_mean,
        line_color=PURPLE,
        line_width=3,
        annotation_text="observed mean",
    )
    figure.add_vline(
        x=-result.observed_mean,
        line_color=PURPLE,
        line_width=2,
        line_dash="dot",
        annotation_text="symmetric extreme",
    )
    figure.update_layout(barmode="overlay")
    style_figure(
        figure,
        title="Empirical null distribution from random sign assignments",
        x_title="Sign-flipped mean difference",
        y_title="Permutations",
    )
    st.plotly_chart(figure, width="stretch", config=PLOT_CONFIG, key="sign_flip")
    with st.container(horizontal=True):
        st.metric("Observed mean", f"{result.observed_mean:.4f}", border=True)
        st.metric("Paired t-test p-value", f"{summary.paired_p_value:.5f}", border=True)
        st.metric("Sign-flip p-value", f"{result.p_value:.5f}", border=True)
        st.metric("Permutations", f"{permutations:,}", border=True)
        st.metric("Extreme draws", f"{result.extreme_count:,}", border=True)
    st.caption(
        "The Monte Carlo p-value uses (extreme + 1) / (permutations + 1). The "
        "sign-flip test constructs a null distribution directly, but it still requires "
        "the sign transformation to be justified under the null."
    )


def render_multiple_testing(alpha: float, simulations: int, seed: int) -> None:
    """Render all-null multiplicity and correction behavior."""
    st.header("Multiple hypothesis testing")
    with st.container(horizontal=True, vertical_alignment="bottom"):
        hypotheses = st.slider("Hypotheses per family", 5, 200, 50, 5)
        repetitions = st.select_slider(
            "Repeated families",
            options=[100, 500, 1_000, 5_000, 10_000],
            value=min(simulations, 10_000),
        )
    summary = cached_multiple_testing(hypotheses, repetitions, alpha, seed)
    method = st.segmented_control(
        "Displayed correction",
        ["No correction", "Bonferroni", "Benjamini-Hochberg"],
        default="No correction",
        required=True,
        key="multiplicity_method",
    )
    decisions = {
        "No correction": summary.example_uncorrected,
        "Bonferroni": summary.example_bonferroni,
        "Benjamini-Hochberg": summary.example_bh,
    }[method]
    colors = np.where(decisions, RED, BLUE)
    figure = go.Figure(
        go.Scatter(
            x=np.arange(1, hypotheses + 1),
            y=summary.example_p_values,
            mode="markers",
            marker=dict(color=colors, size=9),
            name="Synthetic tests",
            hovertemplate="hypothesis=%{x}<br>p=%{y:.4f}<extra></extra>",
        )
    )
    figure.add_hline(
        y=alpha, line_dash="dash", line_color=ORANGE, annotation_text="nominal α"
    )
    figure.add_hline(
        y=alpha / hypotheses,
        line_dash="dot",
        line_color=RED,
        annotation_text="Bonferroni α/m",
    )
    style_figure(
        figure,
        title=f"One all-null family · red points rejected by {method}",
        x_title="Hypothesis",
        y_title="p-value",
    )
    figure.update_yaxes(range=[0.0, 1.0])
    st.plotly_chart(figure, width="stretch", config=PLOT_CONFIG, key="multiple_family")

    aggregate = pd.DataFrame(
        {
            "method": list(summary.familywise_rates),
            "familywise_false_positive_rate": list(summary.familywise_rates.values()),
            "mean_false_positives": [
                summary.mean_discoveries[name] for name in summary.familywise_rates
            ],
        }
    )
    aggregate_figure = go.Figure(
        go.Bar(
            x=aggregate["method"],
            y=aggregate["familywise_false_positive_rate"],
            marker_color=[RED, BLUE, PURPLE],
            text=aggregate["familywise_false_positive_rate"].map(
                lambda value: f"{value:.1%}"
            ),
            textposition="outside",
        )
    )
    aggregate_figure.add_hline(
        y=alpha, line_dash="dash", line_color=ORANGE, annotation_text="α"
    )
    style_figure(
        aggregate_figure,
        title="Probability of at least one false positive across the family",
        x_title="Procedure",
        y_title="Empirical family-wise rate",
        height=430,
    )
    aggregate_figure.update_yaxes(tickformat=".0%")
    st.plotly_chart(
        aggregate_figure, width="stretch", config=PLOT_CONFIG, key="multiple_rates"
    )
    with st.container(horizontal=True):
        for name in summary.mean_discoveries:
            st.metric(
                f"Mean false positives · {name}",
                f"{summary.mean_discoveries[name]:.3f}",
                border=True,
            )
    st.caption(
        "Every null is true in this simulation, so every rejection is a false positive. "
        "That statement does not extend to mixed families containing real effects. "
        "Benjamini-Hochberg targets false discovery rate, not family-wise error in general."
    )


def render_confidence_intervals(
    effect: float,
    sample_size: int,
    standard_deviation: float,
    seed: int,
) -> None:
    """Render matching test/interval decisions and repeated coverage."""
    st.header("Confidence intervals and hypothesis tests")
    view = st.segmented_control(
        "Interval view",
        ["CI and test equivalence", "Repeated confidence intervals"],
        default="CI and test equivalence",
        required=True,
        key="interval_view",
    )
    if view == "CI and test equivalence":
        with st.container(horizontal=True, vertical_alignment="bottom"):
            estimate = st.number_input(
                "Effect estimate", min_value=-1.0, max_value=1.0, value=float(effect), step=0.005
            )
            standard_error = st.number_input(
                "Standard error",
                min_value=0.001,
                max_value=1.0,
                value=max(standard_deviation / math.sqrt(sample_size), 0.001),
                step=0.005,
                format="%.3f",
            )
            confidence = st.select_slider(
                "Confidence level", options=[0.80, 0.90, 0.95, 0.99], value=0.95
            )
        result = confidence_test_connection(estimate, standard_error, confidence)
        lower, upper = result.confidence_interval
        figure = go.Figure(
            go.Scatter(
                x=[estimate],
                y=["Estimate"],
                mode="markers",
                marker=dict(color=PURPLE, size=14),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[upper - estimate],
                    arrayminus=[estimate - lower],
                    color=PURPLE,
                    thickness=4,
                ),
                name="Estimate and CI",
            )
        )
        figure.add_vline(
            x=0.0, line_dash="dash", line_color=RED, annotation_text="null value"
        )
        style_figure(
            figure,
            title="Matching two-sided normal procedures give the same decision",
            x_title="Effect",
            y_title="",
            height=360,
        )
        st.plotly_chart(figure, width="stretch", config=PLOT_CONFIG, key="ci_test")
        interval_excludes_zero = not (lower <= 0.0 <= upper)
        with st.container(horizontal=True):
            st.metric("Confidence interval", f"[{lower:.4f}, {upper:.4f}]", border=True)
            st.metric("p-value", f"{result.p_value:.5f}", border=True)
            st.metric(
                "Test decision",
                "Reject H₀" if result.reject else "Fail to reject H₀",
                border=True,
            )
            st.metric(
                "CI contains zero", "No" if interval_excludes_zero else "Yes", border=True
            )
        st.caption(
            "The equivalence holds because the interval and two-sided test use the same "
            "normal reference model and alpha = 1 − confidence."
        )
    else:
        confidence = st.select_slider(
            "Confidence level", options=[0.80, 0.90, 0.95, 0.99], value=0.95,
            key="repeated_confidence",
        )
        interval_data = cached_intervals(
            effect, sample_size, 40, confidence, standard_deviation, seed
        )
        colors = np.where(interval_data["covered"], BLUE, RED)
        figure = go.Figure()
        for row, color in zip(interval_data.itertuples(index=False), colors, strict=True):
            figure.add_trace(
                go.Scatter(
                    x=[row.lower, row.upper],
                    y=[row.interval, row.interval],
                    mode="lines",
                    line=dict(color=color, width=3),
                    showlegend=False,
                    hovertemplate=(
                        f"interval={row.interval}<br>[{row.lower:.3f}, {row.upper:.3f}]"
                        "<extra></extra>"
                    ),
                )
            )
        figure.add_vline(
            x=effect, line_dash="dash", line_color=GREEN, annotation_text="true mean"
        )
        style_figure(
            figure,
            title="Repeated Student-t intervals · red intervals miss the true mean",
            x_title="Mean parameter",
            y_title="Experiment",
            height=620,
        )
        st.plotly_chart(
            figure, width="stretch", config=PLOT_CONFIG, key="repeated_intervals"
        )
        st.metric(
            "Coverage in this finite run",
            f"{interval_data['covered'].mean():.1%}",
            border=True,
        )
        st.caption(
            "Confidence is a long-run property of the procedure. The finite set of 40 "
            "intervals need not match the nominal rate exactly."
        )


def render_practical_significance(mpid: float) -> None:
    """Render four synthetic combinations of evidence and effect magnitude."""
    st.header("Statistical significance versus practical significance")
    scenarios = practical_significance_scenarios(mpid)
    colors = np.where(scenarios["statistically_significant"], PURPLE, GRAY)
    figure = go.Figure(
        go.Scatter(
            x=scenarios["estimate"],
            y=scenarios["scenario"],
            mode="markers",
            marker=dict(color=colors, size=13),
            error_x=dict(
                type="data",
                symmetric=False,
                array=scenarios["upper"] - scenarios["estimate"],
                arrayminus=scenarios["estimate"] - scenarios["lower"],
                color=PURPLE,
                thickness=3,
            ),
            customdata=np.column_stack([scenarios["p_value"]]),
            hovertemplate="effect=%{x:.4f}<br>p=%{customdata[0]:.5f}<extra></extra>",
        )
    )
    figure.add_vline(x=0.0, line_dash="dash", line_color=RED, annotation_text="H₀")
    figure.add_vline(
        x=mpid, line_dash="dot", line_color=GREEN, annotation_text="MPID"
    )
    style_figure(
        figure,
        title="Four synthetic outcomes: interval evidence and decision magnitude",
        x_title="Effect estimate and 95% interval",
        y_title="Synthetic scenario",
        height=500,
    )
    st.plotly_chart(
        figure, width="stretch", config=PLOT_CONFIG, key="practical_scenarios"
    )
    display = scenarios[
        [
            "scenario",
            "estimate",
            "lower",
            "upper",
            "p_value",
            "statistically_significant",
            "point_estimate_exceeds_threshold",
        ]
    ]
    st.dataframe(
        display,
        hide_index=True,
        column_config={
            "scenario": "Synthetic scenario",
            "estimate": st.column_config.NumberColumn("Effect", format="%.4f"),
            "lower": st.column_config.NumberColumn("CI lower", format="%.4f"),
            "upper": st.column_config.NumberColumn("CI upper", format="%.4f"),
            "p_value": st.column_config.NumberColumn("p-value", format="%.6f"),
            "statistically_significant": "p < 0.05",
            "point_estimate_exceeds_threshold": "Point estimate ≥ MPID",
        },
        key="practical_table",
    )
    st.caption(
        "These are constructed normal-reference examples, not executed product "
        "experiments. A p-value and a practical threshold answer different questions."
    )


st.title("Hypothesis testing visual lab")
st.write(
    "Explore evidence under a null model, error trade-offs, power, paired AI "
    "evaluation, randomization, multiplicity, and confidence intervals."
)
st.caption(
    "All data are synthetic and reproducible. Displayed results are educational, "
    "not production benchmarks or deployment recommendations."
)

with st.sidebar:
    st.header("Lab controls")
    section = st.selectbox("Section", SECTIONS, key="lab_section")
    sample_size = st.slider("Sample size", 5, 500, 80, 5)
    effect = st.slider("Effect / mean difference", 0.0, 0.20, 0.02, 0.005)
    standard_deviation = st.slider("Standard deviation", 0.01, 0.30, 0.08, 0.01)
    alpha = st.select_slider(
        "Significance level α", options=[0.01, 0.025, 0.05, 0.10], value=0.05
    )
    direction_label = st.segmented_control(
        "Test direction",
        ["Two-sided", "Greater"],
        default="Two-sided",
        required=True,
        key="test_direction",
    )
    simulations = st.select_slider(
        "Monte Carlo experiments",
        options=[100, 1_000, 5_000, 10_000],
        value=5_000,
    )
    permutations = st.select_slider(
        "Randomization permutations",
        options=[1_000, 5_000, 20_000, 50_000],
        value=20_000,
    )
    mpid = st.select_slider(
        "Minimum practically important difference",
        options=[0.005, 0.01, 0.02, 0.05],
        value=0.02,
    )
    seed = st.number_input("Random seed", 0, 1_000_000, 42, 1)
    st.caption("Controls are bounded to keep local reruns responsive.")

alternative = "two-sided" if direction_label == "Two-sided" else "greater"

if section == "Test intuition":
    render_test_intuition(effect, sample_size, standard_deviation, alpha, alternative)
elif section == "Errors and power":
    render_errors_and_power(
        effect, sample_size, standard_deviation, alpha, alternative
    )
elif section == "Repeated experiments":
    render_repeated_experiments(
        effect,
        sample_size,
        standard_deviation,
        alpha,
        alternative,
        simulations,
        int(seed),
    )
elif section == "Paired AI experiment":
    render_paired_experiment(
        effect,
        sample_size,
        standard_deviation,
        alpha,
        mpid,
        int(seed),
    )
elif section == "Randomization test":
    render_randomization_test(
        effect,
        sample_size,
        standard_deviation,
        permutations,
        int(seed),
    )
elif section == "Multiple testing":
    render_multiple_testing(alpha, simulations, int(seed))
elif section == "Confidence intervals":
    render_confidence_intervals(
        effect, sample_size, standard_deviation, int(seed)
    )
else:
    render_practical_significance(mpid)

st.caption(
    "A statistically significant result is evidence, not an automatic deployment "
    "decision. Review design, measurement, effect size, uncertainty, cost, latency, "
    "reliability, safety, and distribution shift."
)

"""Interactive Streamlit and Plotly laboratory for probability distributions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

from distribution_utils import (
    COLORS,
    SEED,
    calculate_empirical_statistics,
    create_binomial_probability_surface,
    create_lognormal_surface,
    create_normal_mu_surface,
    create_normal_sigma_surface,
    create_poisson_probability_surface,
    simulate_poisson_process,
    total_absolute_probability_difference,
)


SAMPLE_OPTIONS = [500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000]
PLOTLY_CONFIG = {
    "displaylogo": False,
    "scrollZoom": False,
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


st.set_page_config(
    page_title="Probability distributions laboratory",
    page_icon=":material/query_stats:",
    layout="wide",
)


@st.cache_data(max_entries=128, show_spinner=False)
def sample_distribution(
    distribution: str,
    parameters: tuple[float, ...],
    sample_size: int,
    seed: int = SEED,
) -> np.ndarray:
    """Generate bounded, deterministic synthetic samples for dashboard reruns."""
    rng = np.random.default_rng(seed)
    if distribution == "bernoulli":
        return rng.binomial(1, parameters[0], sample_size)
    if distribution == "binomial":
        return rng.binomial(int(parameters[0]), parameters[1], sample_size)
    if distribution == "poisson":
        return rng.poisson(parameters[0], sample_size)
    if distribution == "exponential":
        return rng.exponential(1.0 / parameters[0], sample_size)
    if distribution == "normal":
        return rng.normal(parameters[0], parameters[1], sample_size)
    if distribution == "lognormal":
        return rng.lognormal(parameters[0], parameters[1], sample_size)
    if distribution == "uniform":
        return rng.uniform(0.0, 1.0, sample_size)
    raise ValueError(f"Unsupported distribution: {distribution}")


@st.cache_data(max_entries=48, show_spinner=False)
def repeated_sample_means(
    distribution: str,
    sample_size: int,
    repetitions: int,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw observations and repeated means for a CLT demonstration."""
    rng = np.random.default_rng(seed)
    raw_size = max(5_000, sample_size * 100)
    if distribution == "Uniform":
        raw = rng.uniform(0.0, 1.0, raw_size)
        matrix = rng.uniform(0.0, 1.0, (repetitions, sample_size))
    elif distribution == "Exponential":
        raw = rng.exponential(1.0, raw_size)
        matrix = rng.exponential(1.0, (repetitions, sample_size))
    elif distribution == "Bernoulli":
        raw = rng.binomial(1, 0.3, raw_size)
        matrix = rng.binomial(1, 0.3, (repetitions, sample_size))
    else:
        raise ValueError(f"Unsupported CLT distribution: {distribution}")
    return raw, matrix.mean(axis=1)


def show_metric_row(metrics: Sequence[tuple[str, str]]) -> None:
    """Render a responsive row of bordered metric cards."""
    with st.container(horizontal=True):
        for label, value in metrics:
            st.metric(label, value, border=True)


def show_context(
    explanation: str,
    assumptions: Sequence[str],
    example: str,
    warning: str,
) -> None:
    """Render concise interpretation cards after a chart."""
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("**Interpretation**")
            st.write(explanation)
        with st.container(border=True):
            st.markdown("**Assumptions**")
            for assumption in assumptions:
                st.markdown(f"- {assumption}")
    with right:
        with st.container(border=True):
            st.markdown("**Applied AI example**")
            st.write(example)
        st.warning(warning, icon=":material/warning:")


def percentile_frame(
    distribution: stats.rv_continuous | stats.rv_discrete,
    probabilities: Sequence[float] = (0.50, 0.90, 0.95, 0.99),
) -> pd.DataFrame:
    """Return theoretical percentiles in a compact table."""
    values = distribution.ppf(probabilities)
    return pd.DataFrame(
        {
            "Percentile": [f"p{int(p * 100)}" for p in probabilities],
            "Theoretical value": np.asarray(values, dtype=float),
        }
    )


def discrete_comparison_figure(
    support: np.ndarray,
    theoretical_mass: np.ndarray,
    samples: np.ndarray,
    title: str,
    x_title: str,
) -> go.Figure:
    """Create theoretical PMF, empirical mass, and CDF subplots."""
    empirical_counts = np.bincount(
        samples.astype(int), minlength=int(support.max()) + 1
    )
    empirical_mass = empirical_counts[support] / samples.size
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("PMF and synthetic sample", "Cumulative distribution"),
    )
    figure.add_trace(
        go.Bar(
            x=support,
            y=empirical_mass,
            name="Empirical probability",
            marker_color=COLORS["empirical"],
            opacity=0.62,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=support,
            y=theoretical_mass,
            mode="markers+lines",
            name="Theoretical PMF",
            marker_color=COLORS["theoretical"],
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=support,
            y=np.cumsum(theoretical_mass),
            mode="lines",
            name="Theoretical CDF",
            line={"color": COLORS["accent"], "width": 3},
        ),
        row=1,
        col=2,
    )
    figure.update_xaxes(title_text=x_title, row=1, col=1)
    figure.update_xaxes(title_text=x_title, row=1, col=2)
    figure.update_yaxes(title_text="Probability mass", row=1, col=1)
    figure.update_yaxes(
        title_text="P(X ≤ x)", range=[0, 1.02], row=1, col=2
    )
    figure.update_layout(
        title=title,
        template="plotly_white",
        height=470,
        barmode="overlay",
        legend={"orientation": "h", "y": -0.18},
        margin={"l": 40, "r": 20, "t": 90, "b": 80},
    )
    return figure


def continuous_comparison_figure(
    x: np.ndarray,
    pdf: np.ndarray,
    cdf: np.ndarray,
    samples: np.ndarray,
    title: str,
    x_title: str,
) -> go.Figure:
    """Create empirical histogram, theoretical PDF, and CDF subplots."""
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("PDF and synthetic sample", "Cumulative distribution"),
    )
    figure.add_trace(
        go.Histogram(
            x=samples,
            histnorm="probability density",
            nbinsx=70,
            name="Empirical density",
            marker_color=COLORS["empirical"],
            opacity=0.58,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=pdf,
            mode="lines",
            name="Theoretical PDF",
            line={"color": COLORS["theoretical"], "width": 3},
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=cdf,
            mode="lines",
            name="Theoretical CDF",
            line={"color": COLORS["accent"], "width": 3},
        ),
        row=1,
        col=2,
    )
    figure.update_xaxes(title_text=x_title, row=1, col=1)
    figure.update_xaxes(title_text=x_title, row=1, col=2)
    figure.update_yaxes(title_text="Probability density", row=1, col=1)
    figure.update_yaxes(
        title_text="P(X ≤ x)", range=[0, 1.02], row=1, col=2
    )
    figure.update_layout(
        title=title,
        template="plotly_white",
        height=470,
        barmode="overlay",
        legend={"orientation": "h", "y": -0.18},
        margin={"l": 40, "r": 20, "t": 90, "b": 80},
    )
    return figure


def render_bernoulli() -> None:
    """Render the Bernoulli laboratory."""
    with st.sidebar:
        st.subheader("Bernoulli controls")
        p = st.slider(
            "Success probability p",
            0.01,
            0.99,
            0.70,
            0.01,
            key="bernoulli_p",
        )
        sample_size = st.select_slider(
            "Simulated observations",
            SAMPLE_OPTIONS,
            value=10_000,
            key="bernoulli_n_samples",
        )

    samples = sample_distribution("bernoulli", (p,), sample_size)
    empirical = calculate_empirical_statistics(samples)
    support = np.array([0, 1])
    distribution = stats.bernoulli(p)
    figure = discrete_comparison_figure(
        support,
        distribution.pmf(support),
        samples,
        f"Bernoulli(p={p:.2f})",
        "Binary outcome",
    )
    show_metric_row(
        [
            ("Theoretical E[X]", f"{p:.4f}"),
            ("Empirical success rate", f"{empirical.mean:.4f}"),
            ("Theoretical variance", f"{p * (1 - p):.4f}"),
            ("Empirical variance", f"{empirical.variance:.4f}"),
        ]
    )
    st.plotly_chart(
        figure,
        width="stretch",
        key="bernoulli_main_chart",
        config=PLOTLY_CONFIG,
    )

    st.subheader("Bernoulli likelihood and binary cross-entropy")
    true_label = st.segmented_control(
        "True label y",
        [0, 1],
        default=1,
        key="bernoulli_true_label",
    )
    predicted = np.linspace(0.001, 0.999, 500)
    negative_log_likelihood = -(
        true_label * np.log(predicted)
        + (1 - true_label) * np.log(1 - predicted)
    )
    loss_figure = go.Figure(
        go.Scatter(
            x=predicted,
            y=negative_log_likelihood,
            mode="lines",
            line={"color": COLORS["warning"], "width": 3},
            name=f"NLL for y={true_label}",
        )
    )
    loss_figure.add_vline(
        x=p,
        line_dash="dash",
        annotation_text=f"Current p={p:.2f}",
    )
    loss_figure.update_layout(
        title="Negative log-likelihood penalizes confident wrong probabilities",
        xaxis_title="Predicted P(Y=1)",
        yaxis_title="−log P(y | p)",
        yaxis_range=[0, 7],
        template="plotly_white",
        height=390,
    )
    st.plotly_chart(
        loss_figure,
        width="stretch",
        key="bernoulli_loss_chart",
        config=PLOTLY_CONFIG,
    )
    st.dataframe(percentile_frame(distribution), hide_index=True)
    show_context(
        "The parameter p moves probability mass between failure (0) and success "
        "(1). Its expectation is the long-run success proportion.",
        ["Binary support", "A well-defined success event", "Independent observations when multiplying likelihoods"],
        "One RAG answer either passes or fails grounding validation.",
        "Accuracy discards probability quality. Check calibration and log loss when probabilities drive decisions.",
    )


def render_binomial() -> None:
    """Render the Binomial laboratory."""
    with st.sidebar:
        st.subheader("Binomial controls")
        n = st.slider("Trials per experiment n", 1, 200, 30, key="binomial_n")
        p = st.slider(
            "Success probability p",
            0.01,
            0.99,
            0.40,
            0.01,
            key="binomial_p",
        )
        experiments = st.select_slider(
            "Simulated experiments",
            SAMPLE_OPTIONS,
            value=5_000,
            key="binomial_experiments",
        )

    samples = sample_distribution("binomial", (float(n), p), experiments)
    empirical = calculate_empirical_statistics(samples)
    distribution = stats.binom(n, p)
    support = np.arange(n + 1)
    figure = discrete_comparison_figure(
        support,
        distribution.pmf(support),
        samples,
        f"Binomial(n={n}, p={p:.2f})",
        "Success count",
    )
    total_trials = n * experiments
    total_successes = int(np.sum(samples))
    observed_rate = total_successes / total_trials
    z = stats.norm.ppf(0.975)
    denominator = 1 + z**2 / total_trials
    center = (observed_rate + z**2 / (2 * total_trials)) / denominator
    margin = (
        z
        * np.sqrt(
            observed_rate * (1 - observed_rate) / total_trials
            + z**2 / (4 * total_trials**2)
        )
        / denominator
    )
    show_metric_row(
        [
            ("Theoretical E[X]", f"{n * p:.3f}"),
            ("Empirical mean", f"{empirical.mean:.3f}"),
            ("Theoretical variance", f"{n * p * (1 - p):.3f}"),
            ("Observed success rate", f"{observed_rate:.3%}"),
        ]
    )
    st.plotly_chart(
        figure,
        width="stretch",
        key="binomial_main_chart",
        config=PLOTLY_CONFIG,
    )
    st.caption(
        f"95% Wilson interval for the pooled synthetic success rate: "
        f"[{center - margin:.3%}, {center + margin:.3%}]"
    )
    st.dataframe(percentile_frame(distribution), hide_index=True)
    show_context(
        "A Binomial variable sums n Bernoulli trials. The center is np and "
        "variance is np(1-p); p near 0.5 produces the most symmetric shape.",
        ["Fixed n", "Binary trials", "Common p", "Independent trials"],
        "Count how many answers pass validation in a fixed evaluation batch.",
        "If p differs by customer or answers share a document, a simple Binomial model can underestimate variability.",
    )


def render_poisson() -> None:
    """Render the Poisson laboratory, including overdispersion."""
    with st.sidebar:
        st.subheader("Poisson controls")
        rate = st.slider(
            "Event rate λ per unit",
            0.1,
            30.0,
            5.0,
            0.1,
            key="poisson_rate",
        )
        exposure = st.slider(
            "Exposure duration",
            0.1,
            5.0,
            1.0,
            0.1,
            key="poisson_exposure",
        )
        sample_size = st.select_slider(
            "Simulated intervals",
            SAMPLE_OPTIONS,
            value=10_000,
            key="poisson_samples",
        )
        show_mixture = st.toggle(
            "Compare a rate mixture",
            value=False,
            key="poisson_show_mixture",
        )
        second_rate = st.slider(
            "Second mixture rate",
            0.1,
            40.0,
            15.0,
            0.1,
            key="poisson_second_rate",
            disabled=not show_mixture,
        )

    expected_count = rate * exposure
    samples = sample_distribution("poisson", (expected_count,), sample_size)
    empirical = calculate_empirical_statistics(samples)
    upper = max(
        12,
        int(stats.poisson.ppf(0.9995, expected_count)) + 2,
    )
    support = np.arange(upper + 1)
    distribution = stats.poisson(expected_count)
    figure = discrete_comparison_figure(
        support,
        distribution.pmf(support),
        samples,
        f"Poisson(rate × exposure = {expected_count:.2f})",
        "Event count",
    )
    show_metric_row(
        [
            ("Expected count", f"{expected_count:.3f}"),
            ("Empirical mean", f"{empirical.mean:.3f}"),
            ("Empirical variance", f"{empirical.variance:.3f}"),
            ("P(X=0)", f"{np.exp(-expected_count):.4f}"),
        ]
    )
    st.plotly_chart(
        figure,
        width="stretch",
        key="poisson_main_chart",
        config=PLOTLY_CONFIG,
    )

    if show_mixture:
        rng = np.random.default_rng(SEED + 10)
        mixture_rates = rng.choice(
            [expected_count, second_rate * exposure],
            size=sample_size,
        )
        mixture_samples = rng.poisson(mixture_rates)
        mixture_mean = float(np.mean(mixture_samples))
        mixture_variance = float(np.var(mixture_samples))
        comparison = go.Figure()
        comparison.add_trace(
            go.Histogram(
                x=samples,
                histnorm="probability",
                name="Single Poisson",
                opacity=0.60,
                marker_color=COLORS["empirical"],
            )
        )
        comparison.add_trace(
            go.Histogram(
                x=mixture_samples,
                histnorm="probability",
                name="Mixture of rates",
                opacity=0.60,
                marker_color=COLORS["warning"],
            )
        )
        comparison.update_layout(
            title="Hidden rate heterogeneity creates overdispersion",
            xaxis_title="Event count",
            yaxis_title="Empirical probability",
            barmode="overlay",
            template="plotly_white",
            height=420,
        )
        st.plotly_chart(
            comparison,
            width="stretch",
            key="poisson_mixture_chart",
            config=PLOTLY_CONFIG,
        )
        show_metric_row(
            [
                (
                    "Single variance / mean",
                    f"{empirical.variance / empirical.mean:.3f}",
                ),
                (
                    "Mixture variance / mean",
                    f"{mixture_variance / mixture_mean:.3f}",
                ),
            ]
        )
        st.caption(
            "A ratio materially above one is a diagnostic signal, not by itself "
            "proof of a particular alternative model."
        )

    st.dataframe(percentile_frame(distribution), hide_index=True)
    show_context(
        "Poisson models counts per exposure. Its baseline conditional mean and "
        "variance are equal, while exposure scales the expected count.",
        ["Independent events", "Stable conditional rate", "Known exposure", "Independent increments"],
        "Model API requests per minute or malformed documents per 10,000 inputs.",
        "Seasonality, bursts, excess zeros, or rate mixtures can make Poisson uncertainty too narrow.",
    )


def render_exponential() -> None:
    """Render Exponential functions and a memorylessness demonstration."""
    with st.sidebar:
        st.subheader("Exponential controls")
        rate = st.slider(
            "Event rate λ",
            0.05,
            3.0,
            0.50,
            0.05,
            key="exponential_rate",
        )
        sample_size = st.select_slider(
            "Simulated waiting times",
            SAMPLE_OPTIONS,
            value=10_000,
            key="exponential_samples",
        )
        threshold = st.slider(
            "Waiting-time threshold",
            0.0,
            20.0,
            5.0,
            0.25,
            key="exponential_threshold",
        )
        elapsed = st.slider(
            "Already waited s",
            0.0,
            10.0,
            3.0,
            0.25,
            key="exponential_elapsed",
        )
        extra = st.slider(
            "Additional wait t",
            0.25,
            10.0,
            2.0,
            0.25,
            key="exponential_extra",
        )

    samples = sample_distribution("exponential", (rate,), sample_size)
    empirical = calculate_empirical_statistics(samples)
    distribution = stats.expon(scale=1 / rate)
    upper = float(distribution.ppf(0.995))
    x = np.linspace(0, upper, 600)
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("PDF and empirical sample", "CDF and survival"),
    )
    figure.add_trace(
        go.Histogram(
            x=samples,
            histnorm="probability density",
            nbinsx=80,
            opacity=0.55,
            marker_color=COLORS["empirical"],
            name="Empirical density",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=distribution.pdf(x),
            line={"color": COLORS["theoretical"], "width": 3},
            name="PDF",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=distribution.cdf(x),
            line={"color": COLORS["accent"], "width": 3},
            name="CDF",
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=distribution.sf(x),
            line={"color": COLORS["warning"], "width": 3},
            name="Survival",
        ),
        row=1,
        col=2,
    )
    figure.add_vline(
        x=threshold,
        line_dash="dash",
        annotation_text="Threshold",
        row=1,
        col=2,
    )
    figure.update_xaxes(title_text="Waiting time", row=1, col=1)
    figure.update_xaxes(title_text="Waiting time", row=1, col=2)
    figure.update_yaxes(title_text="Density", row=1, col=1)
    figure.update_yaxes(title_text="Probability", row=1, col=2)
    figure.update_layout(
        title=f"Exponential(λ={rate:.2f}): constant hazard h(t)=λ",
        template="plotly_white",
        height=480,
        barmode="overlay",
        legend={"orientation": "h", "y": -0.18},
    )
    show_metric_row(
        [
            ("Expected waiting time", f"{1 / rate:.3f}"),
            ("Empirical mean", f"{empirical.mean:.3f}"),
            (
                f"P(T > {threshold:.2f})",
                f"{distribution.sf(threshold):.4f}",
            ),
            ("Constant hazard", f"{rate:.3f}"),
        ]
    )
    st.plotly_chart(
        figure,
        width="stretch",
        key="exponential_main_chart",
        config=PLOTLY_CONFIG,
    )

    theoretical_conditional = distribution.sf(elapsed + extra) / distribution.sf(
        elapsed
    )
    eligible = samples > elapsed
    empirical_conditional = (
        float(np.mean(samples[eligible] > elapsed + extra))
        if np.any(eligible)
        else float("nan")
    )
    st.subheader("Memorylessness")
    show_metric_row(
        [
            (
                "P(T>s+t | T>s), theory",
                f"{theoretical_conditional:.4f}",
            ),
            ("P(T>t), theory", f"{distribution.sf(extra):.4f}"),
            (
                "P(T>s+t | T>s), sample",
                f"{empirical_conditional:.4f}",
            ),
        ]
    )
    st.latex(r"P(T>s+t\mid T>s)=P(T>t)=e^{-\lambda t}")
    st.dataframe(percentile_frame(distribution), hide_index=True)
    show_context(
        "Exponential waiting times correspond to a homogeneous Poisson process. "
        "A larger λ means more frequent events and shorter expected waits.",
        ["Positive support", "Constant event rate", "Memorylessness", "Constant hazard"],
        "Use it as a baseline for time until the next API request.",
        "LLM service time often depends on tokens, batching, retries, and elapsed progress, violating memorylessness.",
    )


def render_normal() -> None:
    """Render the Normal laboratory with interval probabilities and 68–95–99.7."""
    with st.sidebar:
        st.subheader("Normal controls")
        mean = st.slider(
            "Mean μ",
            -10.0,
            10.0,
            0.0,
            0.25,
            key="normal_mean",
        )
        std = st.slider(
            "Standard deviation σ",
            0.1,
            5.0,
            1.0,
            0.1,
            key="normal_std",
        )
        sample_size = st.select_slider(
            "Simulated observations",
            SAMPLE_OPTIONS,
            value=10_000,
            key="normal_samples",
        )
        z_bounds = st.slider(
            "Interval bounds as z-scores",
            -4.0,
            4.0,
            (-1.0, 1.0),
            0.1,
            key="normal_z_bounds",
        )
        show_rule = st.toggle(
            "Show 68–95–99.7 bands",
            value=True,
            key="normal_show_rule",
        )

    lower = mean + z_bounds[0] * std
    upper = mean + z_bounds[1] * std
    samples = sample_distribution("normal", (mean, std), sample_size)
    empirical = calculate_empirical_statistics(samples)
    distribution = stats.norm(mean, std)
    x = np.linspace(mean - 4.5 * std, mean + 4.5 * std, 700)
    figure = continuous_comparison_figure(
        x,
        distribution.pdf(x),
        distribution.cdf(x),
        samples,
        f"Normal(μ={mean:.2f}, σ={std:.2f})",
        "Value",
    )
    interval_x = x[(x >= lower) & (x <= upper)]
    figure.add_trace(
        go.Scatter(
            x=np.concatenate([interval_x, interval_x[::-1]]),
            y=np.concatenate(
                [distribution.pdf(interval_x), np.zeros(interval_x.size)]
            ),
            fill="toself",
            fillcolor="rgba(84,162,75,0.28)",
            line={"color": "rgba(0,0,0,0)"},
            name="Selected interval",
        ),
        row=1,
        col=1,
    )
    interval_probability = float(
        distribution.cdf(upper) - distribution.cdf(lower)
    )
    show_metric_row(
        [
            ("Theoretical mean", f"{mean:.3f}"),
            ("Empirical mean", f"{empirical.mean:.3f}"),
            ("Theoretical variance", f"{std**2:.3f}"),
            ("P(lower ≤ X ≤ upper)", f"{interval_probability:.3%}"),
        ]
    )
    st.plotly_chart(
        figure,
        width="stretch",
        key="normal_main_chart",
        config=PLOTLY_CONFIG,
    )
    st.caption(
        f"Selected values: [{lower:.3f}, {upper:.3f}] correspond to "
        f"z-scores [{z_bounds[0]:.2f}, {z_bounds[1]:.2f}]."
    )
    if show_rule:
        rule_data = pd.DataFrame(
            {
                "Band": ["μ ± 1σ", "μ ± 2σ", "μ ± 3σ"],
                "Theoretical coverage": [
                    distribution.cdf(mean + k * std)
                    - distribution.cdf(mean - k * std)
                    for k in (1, 2, 3)
                ],
                "Empirical coverage": [
                    np.mean(
                        (samples >= mean - k * std)
                        & (samples <= mean + k * std)
                    )
                    for k in (1, 2, 3)
                ],
            }
        )
        st.dataframe(
            rule_data.style.format(
                {
                    "Theoretical coverage": "{:.3%}",
                    "Empirical coverage": "{:.3%}",
                }
            ),
            hide_index=True,
        )
    st.dataframe(percentile_frame(distribution), hide_index=True)
    show_context(
        "μ moves the curve and σ controls its spread. Standardization expresses "
        "values in standard-deviation units without changing their probability.",
        ["Real-valued support", "Symmetric conditional variation", "Finite constant variance for the simple model"],
        "Model additive measurement error or suitable regression residuals.",
        "MSE implies a conditional Gaussian residual model; it does not require every feature or the marginal target to be Normal.",
    )


def render_lognormal() -> None:
    """Render original and log-scale views of a Log-normal variable."""
    with st.sidebar:
        st.subheader("Log-normal controls")
        latency_mode = st.toggle(
            "Simulate production latency",
            value=False,
            key="lognormal_latency_mode",
        )
        log_mean = st.slider(
            "Log mean",
            -1.0,
            8.0,
            6.2 if latency_mode else 0.5,
            0.1,
            key=f"lognormal_mean_{'latency' if latency_mode else 'generic'}",
        )
        log_std = st.slider(
            "Log standard deviation",
            0.05,
            1.5,
            0.55,
            0.05,
            key="lognormal_std",
        )
        sample_size = st.select_slider(
            "Simulated observations",
            SAMPLE_OPTIONS,
            value=10_000,
            key="lognormal_samples",
        )

    samples = sample_distribution(
        "lognormal", (log_mean, log_std), sample_size
    )
    empirical = calculate_empirical_statistics(samples)
    distribution = stats.lognorm(s=log_std, scale=np.exp(log_mean))
    mean = float(distribution.mean())
    median = float(distribution.median())
    mode = float(np.exp(log_mean - log_std**2))
    upper = float(distribution.ppf(0.997))
    x = np.linspace(max(1e-6, distribution.ppf(0.0001)), upper, 700)
    log_samples = np.log(samples)
    x_title = "Latency in milliseconds" if latency_mode else "Positive value"

    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Original scale: right-skewed", "Log scale: symmetric"),
    )
    figure.add_trace(
        go.Histogram(
            x=samples,
            histnorm="probability density",
            nbinsx=80,
            opacity=0.55,
            marker_color=COLORS["empirical"],
            name="Synthetic original scale",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=distribution.pdf(x),
            line={"color": COLORS["theoretical"], "width": 3},
            name="Log-normal PDF",
        ),
        row=1,
        col=1,
    )
    log_x = np.linspace(log_mean - 4 * log_std, log_mean + 4 * log_std, 600)
    figure.add_trace(
        go.Histogram(
            x=log_samples,
            histnorm="probability density",
            nbinsx=70,
            opacity=0.55,
            marker_color=COLORS["accent"],
            name="log(samples)",
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Scatter(
            x=log_x,
            y=stats.norm.pdf(log_x, log_mean, log_std),
            line={"color": COLORS["warning"], "width": 3},
            name="Normal PDF in log space",
        ),
        row=1,
        col=2,
    )
    for value, label, color in [
        (mode, "Mode", COLORS["neutral"]),
        (median, "Median", COLORS["accent"]),
        (mean, "Mean", COLORS["warning"]),
    ]:
        figure.add_vline(
            x=value,
            line_color=color,
            line_dash="dash",
            annotation_text=label,
            row=1,
            col=1,
        )
    figure.update_xaxes(title_text=x_title, row=1, col=1)
    figure.update_xaxes(title_text="log(value)", row=1, col=2)
    figure.update_yaxes(title_text="Density", row=1, col=1)
    figure.update_yaxes(title_text="Density", row=1, col=2)
    figure.update_layout(
        title=f"Log-normal(log μ={log_mean:.2f}, log σ={log_std:.2f})",
        template="plotly_white",
        height=500,
        barmode="overlay",
        legend={"orientation": "h", "y": -0.20},
    )
    show_metric_row(
        [
            ("Mean", f"{mean:,.2f}"),
            ("Median", f"{median:,.2f}"),
            ("Mode", f"{mode:,.2f}"),
            ("Empirical skewness", f"{empirical.skewness:.3f}"),
        ]
    )
    st.plotly_chart(
        figure,
        width="stretch",
        key="lognormal_main_chart",
        config=PLOTLY_CONFIG,
    )
    percentile_data = percentile_frame(distribution)
    st.dataframe(
        percentile_data.style.format({"Theoretical value": "{:,.3f}"}),
        hide_index=True,
    )
    st.caption(
        f"The synthetic arithmetic mean ({mean:,.2f}) differs from the median "
        f"({median:,.2f}) because the right tail pulls the mean upward."
    )
    show_context(
        "Exponentiating a Normal variable creates positive support and right "
        "skew. Increasing log σ separates mode, median, mean, and tail percentiles.",
        ["Strictly positive support", "Approximately Gaussian log values", "A single homogeneous regime"],
        "Represent a synthetic candidate model for LLM request latency.",
        "Positive skew alone does not prove Log-normality; retries and cold starts can create mixtures or heavier tails.",
    )


def render_relationships() -> None:
    """Render interactive relationships and meaningful parameter surfaces."""
    relationship = st.selectbox(
        "Select a relationship",
        [
            "Bernoulli to Binomial",
            "Binomial to Poisson approximation",
            "Poisson counts and Exponential waits",
            "Normal to Log-normal",
            "Central Limit Theorem",
            "3D parameter surfaces",
        ],
        key="relationship_selector",
    )

    if relationship == "Bernoulli to Binomial":
        n = st.slider("Number of Bernoulli trials", 2, 60, 12, key="rel_bern_n")
        p = st.slider(
            "Success probability",
            0.05,
            0.95,
            0.40,
            0.05,
            key="rel_bern_p",
        )
        rng = np.random.default_rng(SEED)
        one_experiment = rng.binomial(1, p, n)
        experiments = rng.binomial(n, p, 8_000)
        st.write(
            pd.DataFrame(
                {
                    "Trial": np.arange(1, n + 1),
                    "Xi": one_experiment,
                }
            ).T
        )
        show_metric_row(
            [
                ("Observed sum", str(int(one_experiment.sum()))),
                ("Expected sum np", f"{n * p:.2f}"),
                ("Variance np(1-p)", f"{n * p * (1-p):.2f}"),
            ]
        )
        support = np.arange(n + 1)
        figure = discrete_comparison_figure(
            support,
            stats.binom.pmf(support, n, p),
            experiments,
            "Summing independent Bernoulli trials produces Binomial counts",
            "Sum X₁ + ⋯ + Xₙ",
        )
        st.plotly_chart(
            figure,
            width="stretch",
            key="rel_bernoulli_binomial_chart",
            config=PLOTLY_CONFIG,
        )
        st.latex(
            r"X_i\sim\operatorname{Bernoulli}(p),\quad "
            r"\sum_{i=1}^{n}X_i\sim\operatorname{Binomial}(n,p)"
        )

    elif relationship == "Binomial to Poisson approximation":
        n = st.slider("Number of trials n", 5, 2_000, 200, key="rel_bp_n")
        p = st.slider(
            "Success probability p",
            0.001,
            min(0.5, 20 / n),
            min(0.025, 5 / n),
            0.001,
            key="rel_bp_p",
        )
        rate = n * p
        upper = max(15, int(stats.poisson.ppf(0.9999, rate)) + 1)
        support = np.arange(upper + 1)
        binomial_mass = stats.binom.pmf(support, n, p)
        poisson_mass = stats.poisson.pmf(support, rate)
        distance = total_absolute_probability_difference(
            binomial_mass, poisson_mass
        )
        figure = go.Figure()
        figure.add_trace(
            go.Bar(
                x=support,
                y=binomial_mass,
                name="Binomial",
                opacity=0.60,
                marker_color=COLORS["empirical"],
            )
        )
        figure.add_trace(
            go.Scatter(
                x=support,
                y=poisson_mass,
                mode="markers+lines",
                name=f"Poisson(λ={rate:.3f})",
                line={"color": COLORS["theoretical"]},
            )
        )
        figure.update_layout(
            title="Rare-event approximation",
            xaxis_title="Count k",
            yaxis_title="Probability mass",
            barmode="overlay",
            template="plotly_white",
            height=470,
        )
        show_metric_row(
            [
                ("λ = np", f"{rate:.4f}"),
                ("L1 probability distance", f"{distance:.5f}"),
            ]
        )
        st.plotly_chart(
            figure,
            width="stretch",
            key="rel_binomial_poisson_chart",
            config=PLOTLY_CONFIG,
        )
        st.caption(
            "The approximation improves when n is large, p is small, and np "
            "stays moderate. L1 distance is zero only for identical PMFs."
        )

    elif relationship == "Poisson counts and Exponential waits":
        rate = st.slider(
            "Arrival rate per time unit",
            0.2,
            8.0,
            2.0,
            0.1,
            key="rel_process_rate",
        )
        duration = st.slider(
            "Timeline duration",
            5.0,
            30.0,
            15.0,
            1.0,
            key="rel_process_duration",
        )
        rng = np.random.default_rng(SEED)
        arrivals, waits = simulate_poisson_process(rate, duration, rng)
        interval_edges = np.arange(0, np.ceil(duration) + 1)
        counts, _ = np.histogram(arrivals, bins=interval_edges)
        figure = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=(
                "Individual arrivals on a continuous timeline",
                "Counts in fixed one-unit intervals",
            ),
            vertical_spacing=0.18,
        )
        figure.add_trace(
            go.Scatter(
                x=arrivals,
                y=np.zeros(arrivals.size),
                mode="markers",
                marker={"symbol": "line-ns", "size": 18, "color": COLORS["warning"]},
                name="Arrivals",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Bar(
                x=interval_edges[:-1],
                y=counts,
                marker_color=COLORS["empirical"],
                name="Interval counts",
            ),
            row=2,
            col=1,
        )
        figure.update_xaxes(title_text="Time", row=1, col=1)
        figure.update_xaxes(title_text="Interval start", row=2, col=1)
        figure.update_yaxes(visible=False, row=1, col=1)
        figure.update_yaxes(title_text="Event count", row=2, col=1)
        figure.update_layout(
            title="One homogeneous Poisson process, two views",
            template="plotly_white",
            height=610,
            showlegend=False,
        )
        show_metric_row(
            [
                ("Observed events", str(arrivals.size)),
                ("Expected events", f"{rate * duration:.2f}"),
                (
                    "Mean inter-arrival time",
                    f"{np.mean(waits):.3f}" if waits.size else "No arrivals",
                ),
                ("Theoretical E[wait]", f"{1 / rate:.3f}"),
            ]
        )
        st.plotly_chart(
            figure,
            width="stretch",
            key="rel_poisson_process_chart",
            config=PLOTLY_CONFIG,
        )
        st.caption(
            "Counts in equal intervals follow Poisson under the homogeneous "
            "process assumptions; inter-arrival times follow Exponential."
        )

    elif relationship == "Normal to Log-normal":
        mean = st.slider("Normal mean μ", -1.0, 2.0, 0.5, 0.1, key="rel_nl_mu")
        std = st.slider(
            "Normal standard deviation σ",
            0.1,
            1.2,
            0.6,
            0.05,
            key="rel_nl_sigma",
        )
        normal_samples = sample_distribution("normal", (mean, std), 15_000)
        transformed = np.exp(normal_samples)
        figure = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("X ~ Normal", "Y = exp(X) ~ Log-normal"),
        )
        figure.add_trace(
            go.Histogram(
                x=normal_samples,
                histnorm="probability density",
                nbinsx=70,
                marker_color=COLORS["empirical"],
                name="X",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Histogram(
                x=transformed,
                histnorm="probability density",
                nbinsx=85,
                marker_color=COLORS["theoretical"],
                name="exp(X)",
            ),
            row=1,
            col=2,
        )
        figure.update_xaxes(title_text="Real value", row=1, col=1)
        figure.update_xaxes(title_text="Positive value", row=1, col=2)
        figure.update_yaxes(title_text="Density", row=1, col=1)
        figure.update_yaxes(title_text="Density", row=1, col=2)
        figure.update_layout(
            title="Exponentiation converts additive log-space variation into positive skew",
            template="plotly_white",
            height=480,
            showlegend=False,
        )
        st.plotly_chart(
            figure,
            width="stretch",
            key="rel_normal_lognormal_chart",
            config=PLOTLY_CONFIG,
        )
        st.latex(
            r"X\sim\mathcal{N}(\mu,\sigma^2),\quad "
            r"Y=e^X,\quad \log Y=X"
        )

    elif relationship == "Central Limit Theorem":
        source = st.segmented_control(
            "Original distribution",
            ["Uniform", "Exponential", "Bernoulli"],
            default="Exponential",
            key="clt_source",
        )
        sample_size = st.slider(
            "Observations per sample",
            1,
            100,
            20,
            key="clt_sample_size",
        )
        repetitions = st.slider(
            "Repeated samples",
            500,
            10_000,
            4_000,
            500,
            key="clt_repetitions",
        )
        raw, means = repeated_sample_means(source, sample_size, repetitions)
        mean_of_means = float(np.mean(means))
        std_of_means = float(np.std(means))
        normal_x = np.linspace(
            mean_of_means - 4 * std_of_means,
            mean_of_means + 4 * std_of_means,
            500,
        )
        figure = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "Raw observations",
                "Distribution of repeated sample means",
            ),
        )
        figure.add_trace(
            go.Histogram(
                x=raw,
                histnorm="probability density",
                nbinsx=70,
                marker_color=COLORS["empirical"],
                name="Raw observations",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Histogram(
                x=means,
                histnorm="probability density",
                nbinsx=70,
                marker_color=COLORS["accent"],
                opacity=0.62,
                name="Sample means",
            ),
            row=1,
            col=2,
        )
        figure.add_trace(
            go.Scatter(
                x=normal_x,
                y=stats.norm.pdf(normal_x, mean_of_means, std_of_means),
                line={"color": COLORS["warning"], "width": 3},
                name="Matched Normal curve",
            ),
            row=1,
            col=2,
        )
        figure.update_xaxes(title_text="Raw value", row=1, col=1)
        figure.update_xaxes(title_text="Sample mean", row=1, col=2)
        figure.update_yaxes(title_text="Density", row=1, col=1)
        figure.update_yaxes(title_text="Density", row=1, col=2)
        figure.update_layout(
            title="The CLT concerns sample means, not raw observations",
            template="plotly_white",
            height=480,
            barmode="overlay",
            legend={"orientation": "h", "y": -0.20},
        )
        st.plotly_chart(
            figure,
            width="stretch",
            key="rel_clt_chart",
            config=PLOTLY_CONFIG,
        )
        st.caption(
            "Increase observations per sample. The raw distribution keeps its "
            "shape while the sampling distribution narrows and becomes more Gaussian."
        )

    else:
        surface_name = st.selectbox(
            "Select a parameter surface",
            [
                "Normal: value × standard deviation",
                "Normal: value × mean",
                "Log-normal: value × log standard deviation",
                "Binomial: success count × p",
                "Poisson: event count × λ",
            ],
            key="surface_selector",
        )
        if surface_name == "Normal: value × standard deviation":
            figure = create_normal_sigma_surface()
        elif surface_name == "Normal: value × mean":
            figure = create_normal_mu_surface()
        elif surface_name == "Log-normal: value × log standard deviation":
            figure = create_lognormal_surface()
        elif surface_name == "Binomial: success count × p":
            fixed_n = st.slider(
                "Fixed number of trials n",
                5,
                80,
                30,
                key="surface_binomial_n",
            )
            figure = create_binomial_probability_surface(fixed_n)
        else:
            figure = create_poisson_probability_surface()
        st.plotly_chart(
            figure,
            width="stretch",
            height=680,
            key="relationship_surface_chart",
            config={**PLOTLY_CONFIG, "scrollZoom": True},
        )
        st.caption(
            "Rotate and zoom to inspect how parameter changes reshape the full "
            "probability function. These surfaces encode a genuine third dimension."
        )


def render_production_examples() -> None:
    """Render clearly labeled synthetic production-oriented examples."""
    st.caption(
        "Every value and dataset in this section is synthetic. The examples are "
        "mechanistic illustrations, not measurements from a production system."
    )
    example = st.selectbox(
        "Select a production example",
        [
            "RAG validation",
            "API request arrivals",
            "Waiting for the next request",
            "LLM latency",
            "Gaussian residual assumption",
        ],
        key="production_example_selector",
    )

    if example == "RAG validation":
        batch_size = st.slider(
            "Evaluation batch size",
            10,
            1_000,
            100,
            10,
            key="prod_rag_batch",
        )
        pass_probability = st.slider(
            "Estimated grounding-pass probability",
            0.01,
            0.99,
            0.82,
            0.01,
            key="prod_rag_p",
        )
        rng = np.random.default_rng(SEED)
        outcomes = rng.binomial(1, pass_probability, batch_size)
        observed = int(outcomes.sum())
        support = np.arange(batch_size + 1)
        mass = stats.binom.pmf(support, batch_size, pass_probability)
        visible = mass > 1e-6
        figure = go.Figure(
            go.Bar(
                x=support[visible],
                y=mass[visible],
                marker_color=COLORS["empirical"],
            )
        )
        figure.add_vline(
            x=observed,
            line_dash="dash",
            line_color=COLORS["warning"],
            annotation_text=f"Synthetic observed={observed}",
        )
        figure.update_layout(
            title="Batch-level grounding passes under a Binomial model",
            xaxis_title="Pass count",
            yaxis_title="Probability mass",
            template="plotly_white",
            height=450,
        )
        show_metric_row(
            [
                ("One answer model", f"Bernoulli({pass_probability:.2f})"),
                ("Expected batch passes", f"{batch_size * pass_probability:.2f}"),
                ("Synthetic observed passes", str(observed)),
            ]
        )
        st.plotly_chart(
            figure,
            width="stretch",
            key="prod_rag_chart",
            config=PLOTLY_CONFIG,
        )

    elif example == "API request arrivals":
        rate = st.slider(
            "Average requests per minute",
            1.0,
            500.0,
            120.0,
            1.0,
            key="prod_api_rate",
        )
        capacity = st.slider(
            "Service capacity per minute",
            1,
            700,
            150,
            key="prod_api_capacity",
        )
        exceedance = float(stats.poisson.sf(capacity, rate))
        upper = max(capacity + 20, int(stats.poisson.ppf(0.999, rate)))
        support = np.arange(0, upper + 1)
        mass = stats.poisson.pmf(support, rate)
        figure = go.Figure()
        figure.add_trace(
            go.Bar(
                x=support,
                y=mass,
                marker_color=np.where(
                    support > capacity,
                    COLORS["warning"],
                    COLORS["empirical"],
                ),
                name="Poisson PMF",
            )
        )
        figure.add_vline(
            x=capacity,
            line_dash="dash",
            annotation_text="Capacity",
        )
        figure.update_layout(
            title="Synthetic requests per minute and capacity exceedance",
            xaxis_title="Requests in one minute",
            yaxis_title="Probability mass",
            template="plotly_white",
            height=450,
        )
        show_metric_row(
            [
                ("Expected requests", f"{rate:.1f}"),
                ("Configured capacity", str(capacity)),
                ("P(request count > capacity)", f"{exceedance:.4%}"),
            ]
        )
        st.plotly_chart(
            figure,
            width="stretch",
            key="prod_api_chart",
            config=PLOTLY_CONFIG,
        )

    elif example == "Waiting for the next request":
        rate = st.slider(
            "Requests per second",
            0.05,
            10.0,
            1.0,
            0.05,
            key="prod_wait_rate",
        )
        interval = st.slider(
            "Selected no-arrival interval (seconds)",
            0.0,
            30.0,
            5.0,
            0.25,
            key="prod_wait_interval",
        )
        distribution = stats.expon(scale=1 / rate)
        x = np.linspace(0, distribution.ppf(0.999), 600)
        figure = go.Figure(
            go.Scatter(
                x=x,
                y=distribution.sf(x),
                mode="lines",
                line={"color": COLORS["warning"], "width": 3},
                name="No-arrival probability",
            )
        )
        figure.add_vline(
            x=interval,
            line_dash="dash",
            annotation_text="Selected interval",
        )
        figure.update_layout(
            title="Probability that no request has arrived yet",
            xaxis_title="Seconds",
            yaxis_title="P(T > t)",
            template="plotly_white",
            height=430,
        )
        show_metric_row(
            [
                ("Expected waiting time", f"{1 / rate:.3f} seconds"),
                (
                    f"P(no request in {interval:.2f}s)",
                    f"{distribution.sf(interval):.4%}",
                ),
            ]
        )
        st.plotly_chart(
            figure,
            width="stretch",
            key="prod_wait_chart",
            config=PLOTLY_CONFIG,
        )

    elif example == "LLM latency":
        log_mean = st.slider(
            "Synthetic log-latency mean",
            4.0,
            8.0,
            6.2,
            0.05,
            key="prod_latency_mu",
        )
        log_std = st.slider(
            "Synthetic log-latency standard deviation",
            0.1,
            1.2,
            0.55,
            0.05,
            key="prod_latency_sigma",
        )
        timeout = st.slider(
            "Timeout threshold (milliseconds)",
            100.0,
            10_000.0,
            2_000.0,
            50.0,
            key="prod_latency_timeout",
        )
        distribution = stats.lognorm(s=log_std, scale=np.exp(log_mean))
        samples = sample_distribution(
            "lognormal",
            (log_mean, log_std),
            20_000,
            seed=SEED + 7,
        )
        upper = float(distribution.ppf(0.999))
        x = np.linspace(1, upper, 800)
        figure = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Synthetic latency density", "Tail survival"),
        )
        figure.add_trace(
            go.Histogram(
                x=samples,
                histnorm="probability density",
                nbinsx=90,
                marker_color=COLORS["empirical"],
                opacity=0.60,
                name="Synthetic latency",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=x,
                y=distribution.pdf(x),
                line={"color": COLORS["theoretical"], "width": 3},
                name="Log-normal PDF",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=x,
                y=distribution.sf(x),
                line={"color": COLORS["warning"], "width": 3},
                name="P(latency > x)",
            ),
            row=1,
            col=2,
        )
        figure.add_vline(
            x=timeout,
            line_dash="dash",
            annotation_text="Timeout",
            row=1,
            col=2,
        )
        figure.update_xaxes(title_text="Latency (ms)", row=1, col=1)
        figure.update_xaxes(title_text="Latency (ms)", row=1, col=2)
        figure.update_yaxes(title_text="Density", row=1, col=1)
        figure.update_yaxes(
            title_text="Survival probability",
            type="log",
            row=1,
            col=2,
        )
        figure.update_layout(
            title="Synthetic LLM latency: averages hide tail risk",
            template="plotly_white",
            height=490,
            barmode="overlay",
            legend={"orientation": "h", "y": -0.20},
        )
        show_metric_row(
            [
                ("Mean latency", f"{distribution.mean():,.0f} ms"),
                ("Median latency", f"{distribution.median():,.0f} ms"),
                ("p95", f"{distribution.ppf(0.95):,.0f} ms"),
                ("p99", f"{distribution.ppf(0.99):,.0f} ms"),
                ("P(latency > timeout)", f"{distribution.sf(timeout):.3%}"),
            ]
        )
        st.plotly_chart(
            figure,
            width="stretch",
            key="prod_latency_chart",
            config=PLOTLY_CONFIG,
        )

    else:
        rng = np.random.default_rng(SEED)
        feature = rng.uniform(-3, 3, 500)
        predictions = 2.0 + 1.8 * feature
        residuals = rng.normal(0, 1.0, feature.size)
        observed = predictions + residuals
        slope, intercept = np.polyfit(feature, observed, 1)
        fitted = intercept + slope * feature
        fitted_residuals = observed - fitted
        theoretical_q, ordered_residuals = stats.probplot(
            fitted_residuals, dist="norm", fit=False
        )
        qq_slope, qq_intercept = np.polyfit(
            theoretical_q, ordered_residuals, 1
        )
        figure = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=(
                "Synthetic regression",
                "Residual histogram",
                "Normal Q–Q plot",
            ),
        )
        figure.add_trace(
            go.Scatter(
                x=feature,
                y=observed,
                mode="markers",
                marker={"color": COLORS["empirical"], "opacity": 0.55},
                name="Observed",
            ),
            row=1,
            col=1,
        )
        order = np.argsort(feature)
        figure.add_trace(
            go.Scatter(
                x=feature[order],
                y=fitted[order],
                mode="lines",
                line={"color": COLORS["warning"], "width": 3},
                name="Fitted line",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Histogram(
                x=fitted_residuals,
                histnorm="probability density",
                nbinsx=35,
                marker_color=COLORS["accent"],
                name="Residuals",
            ),
            row=1,
            col=2,
        )
        residual_x = np.linspace(
            fitted_residuals.min(), fitted_residuals.max(), 400
        )
        figure.add_trace(
            go.Scatter(
                x=residual_x,
                y=stats.norm.pdf(
                    residual_x,
                    np.mean(fitted_residuals),
                    np.std(fitted_residuals),
                ),
                line={"color": COLORS["theoretical"], "width": 3},
                name="Fitted Normal",
            ),
            row=1,
            col=2,
        )
        figure.add_trace(
            go.Scatter(
                x=theoretical_q,
                y=ordered_residuals,
                mode="markers",
                marker_color=COLORS["empirical"],
                name="Q–Q points",
            ),
            row=1,
            col=3,
        )
        qq_x = np.asarray([min(theoretical_q), max(theoretical_q)])
        figure.add_trace(
            go.Scatter(
                x=qq_x,
                y=qq_intercept + qq_slope * qq_x,
                mode="lines",
                line={"color": COLORS["warning"], "dash": "dash"},
                name="Reference line",
            ),
            row=1,
            col=3,
        )
        figure.update_xaxes(title_text="Feature x", row=1, col=1)
        figure.update_yaxes(title_text="Target y", row=1, col=1)
        figure.update_xaxes(title_text="Residual", row=1, col=2)
        figure.update_yaxes(title_text="Density", row=1, col=2)
        figure.update_xaxes(title_text="Theoretical Normal quantile", row=1, col=3)
        figure.update_yaxes(title_text="Ordered residual", row=1, col=3)
        figure.update_layout(
            title="MSE and the conditional Gaussian residual assumption",
            template="plotly_white",
            height=480,
            showlegend=False,
        )
        st.plotly_chart(
            figure,
            width="stretch",
            key="prod_regression_chart",
            config=PLOTLY_CONFIG,
        )
        st.caption(
            "MSE corresponds to a Gaussian model for conditional residuals with "
            "constant variance—not to Normal input features."
        )


st.title("Probability distributions laboratory")
st.caption(
    "Day 8 · deterministic synthetic simulations · theory, approximation, "
    "tail behavior, and Applied AI interpretations"
)

with st.sidebar:
    st.markdown(":material/tune: **Laboratory navigation**")
    selected_section = st.selectbox(
        "Select a distribution or section",
        [
            "Bernoulli",
            "Binomial",
            "Poisson",
            "Exponential",
            "Normal",
            "Log-normal",
            "Distribution relationships",
            "Production examples",
        ],
        key="main_section_selector",
    )
    st.caption("All simulations use fixed seeds and synthetic data.")

if selected_section == "Bernoulli":
    render_bernoulli()
elif selected_section == "Binomial":
    render_binomial()
elif selected_section == "Poisson":
    render_poisson()
elif selected_section == "Exponential":
    render_exponential()
elif selected_section == "Normal":
    render_normal()
elif selected_section == "Log-normal":
    render_lognormal()
elif selected_section == "Distribution relationships":
    render_relationships()
else:
    render_production_examples()

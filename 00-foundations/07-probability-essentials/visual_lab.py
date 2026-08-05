"""Interactive Probability Essentials visual laboratory.

Run from this topic directory with:
    streamlit run visual_lab.py

Every dataset and scenario in this app is synthetic and generated locally.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from visualizations import (
    bayes_posterior,
    bernoulli_variance,
    die_monte_carlo,
    discrete_variance,
    expected_decision_threshold,
    expected_value,
    fraud_outcome_counts,
    fraud_posterior_monte_carlo,
    frechet_bounds,
    make_bayes_surface_figure,
    make_conditional_grid_figure,
    make_confusion_matrix_figure,
    make_event_grid_figure,
    make_expected_cost_figure,
    make_expected_value_figure,
    make_fraud_population_figure,
    make_joint_heatmap,
    make_llm_quality_comparison_figure,
    make_probability_flow_figure,
    make_reliability_comparison_figure,
    make_squared_deviation_figure,
    make_variance_distribution_figure,
    simulate_binary_joint_distribution,
    validate_discrete_distribution,
)


st.set_page_config(
    page_title="Probability Essentials — Visual Lab",
    page_icon="🎲",
    layout="wide",
)

TOPIC_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = TOPIC_DIR / "outputs"
PLOTLY_CONFIG = {
    "displaylogo": False,
    "scrollZoom": False,
    "toImageButtonOptions": {"format": "png", "scale": 2},
}

SECTIONS = (
    "Sample space and events",
    "Conditional probability",
    "Independence and dependence",
    "Bayes' theorem",
    "Expected value",
    "Variance and standard deviation",
    "Monte Carlo convergence",
    "Expected cost and decision thresholds",
    "Production connections",
)


@st.cache_data(max_entries=40, show_spinner=False)
def cached_die_simulation(max_sample_size: int, seed: int) -> pd.DataFrame:
    """Cache deterministic fair-die convergence paths."""
    return die_monte_carlo(max_sample_size, seed)


@st.cache_data(max_entries=40, show_spinner=False)
def cached_fraud_simulation(max_sample_size: int, seed: int) -> pd.DataFrame:
    """Cache deterministic fraud-posterior convergence paths."""
    return fraud_posterior_monte_carlo(max_sample_size, seed)


def show_interpretation_cards(
    observation: str,
    interview_mistake: str,
    production_takeaway: str,
) -> None:
    """Render consistent interpretation, interview, and production cards."""
    columns = st.columns(3)
    with columns[0].container(border=True, height="stretch"):
        st.markdown("**What to observe**")
        st.write(observation)
    with columns[1].container(border=True, height="stretch"):
        st.markdown("**Common interview mistake**")
        st.write(interview_mistake)
    with columns[2].container(border=True, height="stretch"):
        st.markdown("**Production takeaway**")
        st.write(production_takeaway)


def show_generated_asset(filename: str, caption: str) -> None:
    """Display a generated asset or actionable generation instructions."""
    asset_path = OUTPUT_DIR / filename
    if asset_path.exists():
        st.image(asset_path, caption=caption, width="stretch")
    else:
        st.warning(
            f"`{filename}` is missing. Run "
            "`python generate_visual_assets.py` to generate the animation "
            "or static preview.",
            icon=":material/build:",
        )


with st.sidebar:
    st.header("Visual lab")
    selected_section = st.selectbox(
        "Choose a concept",
        SECTIONS,
        index=0,
        key="section_selector",
    )
    st.caption(
        "All scenarios are synthetic, deterministic where seeded, and run "
        "without network access."
    )
    st.markdown("**Generated assets**")
    available_assets = sum(
        (OUTPUT_DIR / filename).exists()
        for filename in (
            "event_operations.png",
            "conditional_probability.gif",
            "independence_comparison.png",
            "bayes_base_rate.gif",
            "bayes_surface.png",
            "expected_value_balance.png",
            "variance_spread.gif",
            "bernoulli_variance.png",
            "monte_carlo_convergence.gif",
            "expected_cost.png",
        )
    )
    st.metric("Available", f"{available_assets}/10", border=True)
    st.caption("Generate missing files with `python generate_visual_assets.py`.")

st.title("Probability Essentials — Visual Lab")
st.caption(
    "An interactive, production-oriented laboratory for reasoning under "
    "uncertainty. Synthetic data only."
)


if selected_section == "Sample space and events":
    st.header("Sample space and events")
    st.write(
        "The sample space contains integers 1–100. Event A contains values "
        "divisible by 2; event B contains values divisible by 5. Marker shape "
        "identifies event membership, while size and opacity identify the "
        "selected operation."
    )
    st.latex(
        r"P(A \cup B)=P(A)+P(B)-P(A\cap B)"
    )

    operation = st.segmented_control(
        "Select an event operation",
        ["A", "B", "A ∪ B", "A ∩ B", "Aᶜ", "Bᶜ"],
        default="A ∪ B",
        required=True,
        key="event_operation",
        width="stretch",
    )
    st.plotly_chart(
        make_event_grid_figure(operation),
        width="stretch",
        key="event_grid",
        config=PLOTLY_CONFIG,
    )

    with st.container(horizontal=True):
        st.metric("P(A)", "0.50", "50 / 100", border=True)
        st.metric("P(B)", "0.20", "20 / 100", border=True)
        st.metric("P(A ∩ B)", "0.10", "10 / 100", border=True)
        st.metric("P(A ∪ B)", "0.60", "50 + 20 − 10", border=True)

    st.success(
        "Addition rule check: 0.50 + 0.20 − 0.10 = 0.60.",
        icon=":material/check_circle:",
    )
    with st.expander(
        "Generated static comparison",
        icon=":material/image:",
    ):
        show_generated_asset(
            "event_operations.png",
            "Six event operations on the same 100-outcome sample space.",
        )
    show_interpretation_cards(
        "The intersection belongs to both events, so adding P(A) and P(B) "
        "counts it twice. The subtraction removes that duplicate count.",
        "Adding P(A)+P(B) without subtracting the intersection unless the "
        "events are known to be mutually exclusive.",
        "Set operations appear in cohort definitions, eligibility rules, "
        "monitoring filters, and metric denominators. Define them explicitly.",
    )


elif selected_section == "Conditional probability":
    st.header("Conditional probability")
    st.write(
        "Conditioning restricts the sample space. Outcomes outside B fade "
        "because B becomes the new denominator; only observations also in A "
        "form the numerator."
    )
    st.latex(r"P(A\mid B)=\frac{P(A\cap B)}{P(B)},\quad P(B)>0")
    st.latex(
        r"\max(0, P(A)+P(B)-1)\leq P(A\cap B)"
        r"\leq\min(P(A),P(B))"
    )

    controls = st.columns(3)
    total_population = controls[0].slider(
        "Total population",
        min_value=20,
        max_value=200,
        value=100,
        step=10,
        key="conditional_total",
    )
    count_a = controls[1].slider(
        "Observations in A",
        min_value=0,
        max_value=total_population,
        value=total_population // 2,
        key="conditional_a",
    )
    count_b = controls[2].slider(
        "Observations in B",
        min_value=0,
        max_value=total_population,
        value=max(1, total_population // 5),
        key="conditional_b",
    )
    lower_count = max(0, count_a + count_b - total_population)
    upper_count = min(count_a, count_b)
    default_intersection = int(np.clip(total_population // 10, lower_count, upper_count))
    count_intersection = st.number_input(
        "Observations in A ∩ B",
        min_value=0,
        max_value=total_population,
        value=default_intersection,
        step=1,
        key=(
            f"conditional_intersection_{total_population}_{count_a}_{count_b}"
        ),
        help=(
            f"For these marginals, the valid count range is "
            f"[{lower_count}, {upper_count}]."
        ),
    )

    if not lower_count <= count_intersection <= upper_count:
        st.error(
            "Invalid intersection. The selected marginals require "
            f"{lower_count} ≤ |A∩B| ≤ {upper_count}. This is the count form "
            "of the Fréchet bounds.",
            icon=":material/error:",
        )
    elif count_b == 0:
        st.warning(
            "P(A | B) is undefined because B contains no observations, so "
            "the denominator is zero.",
            icon=":material/warning:",
        )
    else:
        conditional_probability = count_intersection / count_b
        st.plotly_chart(
            make_conditional_grid_figure(
                total_population,
                count_a,
                count_b,
                int(count_intersection),
            ),
            width="stretch",
            key="conditional_grid",
            config=PLOTLY_CONFIG,
        )
        with st.container(horizontal=True):
            st.metric("Original denominator", f"{total_population}", border=True)
            st.metric("New denominator |B|", f"{count_b}", border=True)
            st.metric("Numerator |A∩B|", f"{count_intersection}", border=True)
            st.metric("P(A | B)", f"{conditional_probability:.3f}", border=True)

    with st.expander(
        "Denominator-change animation",
        icon=":material/animation:",
    ):
        show_generated_asset(
            "conditional_probability.gif",
            "The full sample space narrows to B before A∩B is counted.",
        )
    show_interpretation_cards(
        "The denominator changes from the whole population to B. P(A|B) asks "
        "for A's share inside that restricted population.",
        "Using the original population size after conditioning, or reversing "
        "P(A|B) and P(B|A).",
        "Conditional metrics expose subgroup behavior: latency given cache "
        "miss, correctness given good retrieval, or fraud given an alert.",
    )


elif selected_section == "Independence and dependence":
    st.header("Independence and dependence")
    st.write(
        "For fixed marginal probabilities, a dependence parameter δ moves "
        "probability mass within the valid 2×2 joint distribution."
    )
    st.latex(r"P(A\cap B)=P(A)P(B)+\delta")
    st.latex(
        r"\delta\in"
        r"[\max(0,P(A)+P(B)-1)-P(A)P(B),"
        r"\min(P(A),P(B))-P(A)P(B)]"
    )

    controls = st.columns(4)
    probability_a = controls[0].slider(
        "P(A)",
        min_value=0.05,
        max_value=0.95,
        value=0.45,
        step=0.01,
        key="independence_probability_a",
    )
    probability_b = controls[1].slider(
        "P(B)",
        min_value=0.05,
        max_value=0.95,
        value=0.35,
        step=0.01,
        key="independence_probability_b",
    )
    sample_size = controls[2].select_slider(
        "Simulation sample size",
        options=[1_000, 5_000, 20_000, 100_000],
        value=20_000,
        key="independence_sample_size",
    )
    seed = controls[3].number_input(
        "Random seed",
        min_value=0,
        max_value=1_000_000,
        value=42,
        step=1,
        key="independence_seed",
    )
    lower, upper = frechet_bounds(probability_a, probability_b)
    independent_intersection = probability_a * probability_b
    delta_lower = lower - independent_intersection
    delta_upper = upper - independent_intersection
    delta = st.slider(
        "Dependence parameter δ",
        min_value=float(delta_lower),
        max_value=float(delta_upper),
        value=0.0,
        step=0.001,
        format="%.3f",
        key=f"dependence_delta_{probability_a:.2f}_{probability_b:.2f}",
    )
    intersection = independent_intersection + delta

    sample = simulate_binary_joint_distribution(
        probability_a,
        probability_b,
        intersection,
        sample_size=int(sample_size),
        seed=int(seed),
    )
    empirical_a = float(sample["event_a"].mean())
    empirical_b = float(sample["event_b"].mean())
    empirical_intersection = float(
        (sample["event_a"] & sample["event_b"]).mean()
    )
    theoretical_table = pd.DataFrame(
        {
            "Quantity": ["P(A)", "P(B)", "P(A)P(B)", "P(A∩B)", "Difference"],
            "Theoretical": [
                probability_a,
                probability_b,
                independent_intersection,
                intersection,
                delta,
            ],
            "Empirical": [
                empirical_a,
                empirical_b,
                empirical_a * empirical_b,
                empirical_intersection,
                empirical_intersection - empirical_a * empirical_b,
            ],
        }
    )

    left, right = st.columns([1.25, 1.0])
    with left:
        st.plotly_chart(
            make_joint_heatmap(
                probability_a,
                probability_b,
                intersection,
            ),
            width="stretch",
            key="joint_heatmap",
            config=PLOTLY_CONFIG,
        )
    with right:
        st.dataframe(
            theoretical_table,
            hide_index=True,
            width="stretch",
            column_config={
                "Theoretical": st.column_config.NumberColumn(format="%.4f"),
                "Empirical": st.column_config.NumberColumn(format="%.4f"),
            },
        )
        relationship = (
            "independent"
            if np.isclose(delta, 0.0, atol=5e-4)
            else ("positively dependent" if delta > 0 else "negatively dependent")
        )
        st.metric(
            "P(A∩B) − P(A)P(B)",
            f"{delta:+.4f}",
            relationship,
            border=True,
        )

    with st.expander(
        "Static dependence comparison",
        icon=":material/image:",
    ):
        show_generated_asset(
            "independence_comparison.png",
            "Negative dependence, independence, and positive dependence.",
        )
    show_interpretation_cards(
        "At δ=0, the joint intersection equals the product of marginals. "
        "Changing δ redistributes all four cells while preserving P(A), P(B).",
        "Equating independence with mutual exclusivity, or claiming zero "
        "covariance proves independence in general.",
        "Shared regions, databases, credentials, and upstream APIs create "
        "correlated failures. Multiplying marginals can underestimate risk.",
    )


elif selected_section == "Bayes' theorem":
    st.header("Bayes’ theorem")
    st.write(
        "A synthetic fraud detector turns a prior prevalence and two "
        "class-conditional alert rates into the posterior probability that an "
        "alerted transaction is actually fraudulent."
    )
    st.latex(
        r"P(F\mid A)="
        r"\frac{P(A\mid F)P(F)}"
        r"{P(A\mid F)P(F)+P(A\mid F^c)P(F^c)}"
    )

    controls = st.columns(4)
    prior = controls[0].slider(
        "Prior fraud probability P(F)",
        min_value=0.001,
        max_value=0.20,
        value=0.01,
        step=0.001,
        format="%.3f",
        key="bayes_prior",
    )
    true_positive_rate = controls[1].slider(
        "True-positive rate P(A|F)",
        min_value=0.50,
        max_value=1.00,
        value=0.90,
        step=0.01,
        key="bayes_tpr",
    )
    false_positive_rate = controls[2].slider(
        "False-positive rate P(A|Fᶜ)",
        min_value=0.001,
        max_value=0.20,
        value=0.05,
        step=0.001,
        format="%.3f",
        key="bayes_fpr",
    )
    population_size = controls[3].select_slider(
        "Synthetic population",
        options=[1_000, 5_000, 10_000, 50_000, 100_000],
        value=10_000,
        key="bayes_population",
    )
    posterior = bayes_posterior(
        prior,
        true_positive_rate,
        false_positive_rate,
    )
    counts = fraud_outcome_counts(
        prior,
        true_positive_rate,
        false_positive_rate,
        int(population_size),
    )
    alert_count = counts["True positive"] + counts["False positive"]

    with st.container(horizontal=True):
        st.metric("Prior P(F)", f"{prior:.2%}", border=True)
        st.metric("True positives", f"{counts['True positive']:,}", border=True)
        st.metric("False positives", f"{counts['False positive']:,}", border=True)
        st.metric("Alert denominator", f"{alert_count:,}", "TP + FP", border=True)
        st.metric("Posterior P(F|A)", f"{posterior:.2%}", border=True)

    grid_column, matrix_column = st.columns(2)
    with grid_column.container(border=True, height="stretch"):
        st.subheader("Population frequency grid")
        st.caption(
            "A representative grid is scaled to the selected synthetic "
            "population. Shape, label, and color distinguish outcomes."
        )
        st.plotly_chart(
            make_fraud_population_figure(
                prior,
                true_positive_rate,
                false_positive_rate,
                int(population_size),
            ),
            width="stretch",
            key="fraud_population_grid",
            config=PLOTLY_CONFIG,
        )
    with matrix_column.container(border=True, height="stretch"):
        st.subheader("Confusion matrix")
        st.caption("Cells show absolute counts and population percentages.")
        st.plotly_chart(
            make_confusion_matrix_figure(
                prior,
                true_positive_rate,
                false_positive_rate,
                int(population_size),
            ),
            width="stretch",
            key="fraud_confusion_matrix",
            config=PLOTLY_CONFIG,
        )

    with st.container(border=True):
        st.subheader("Probability flow")
        st.caption(
            "The final alert node combines true positives and false positives; "
            "this is the denominator of P(Fraud | Alert)."
        )
        st.plotly_chart(
            make_probability_flow_figure(
                prior,
                true_positive_rate,
                false_positive_rate,
                int(population_size),
            ),
            width="stretch",
            key="bayes_probability_flow",
            config=PLOTLY_CONFIG,
        )

    with st.container(border=True):
        st.subheader("Interactive 3D Bayes surface")
        st.caption(
            "The third dimension is the posterior. Change the true-positive "
            "rate above to regenerate the surface; the diamond marks the "
            "selected scenario."
        )
        st.plotly_chart(
            make_bayes_surface_figure(
                true_positive_rate,
                prior,
                false_positive_rate,
            ),
            width="stretch",
            key="bayes_surface",
            config=PLOTLY_CONFIG,
        )

    with st.expander(
        "Base-rate animation and static 3D preview",
        icon=":material/animation:",
    ):
        animation_column, surface_column = st.columns(2)
        with animation_column:
            show_generated_asset(
                "bayes_base_rate.gif",
                "The detector stays fixed while the prior changes.",
            )
        with surface_column:
            show_generated_asset(
                "bayes_surface.png",
                "Static 3D Bayes surface at a 90% true-positive rate.",
            )
    show_interpretation_cards(
        "With rare fraud, the large legitimate population can create more "
        "false positives than true positives—even with high sensitivity.",
        "Answering P(Fraud|Alert) with the detector recall P(Alert|Fraud), "
        "thereby ignoring prevalence and false positives.",
        "Posterior quality depends on calibrated class-conditional rates and "
        "deployment prevalence. Monitor both under distribution shift.",
    )


elif selected_section == "Expected value":
    st.header("Expected value")
    st.write(
        "Expected value is the sum of each outcome multiplied by its "
        "probability. The vertical line marks the balance point of the "
        "probability mass."
    )
    st.latex(r"\mathbb{E}[X]=\sum_x x\,P(X=x)")

    presets: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "Fair six-sided die": (
            np.arange(1, 7, dtype=float),
            np.full(6, 1 / 6),
        ),
        "Biased die": (
            np.arange(1, 7, dtype=float),
            np.array([0.05, 0.10, 0.15, 0.20, 0.20, 0.30]),
        ),
        "Bernoulli event": (
            np.array([0.0, 1.0]),
            np.array([0.70, 0.30]),
        ),
        "Synthetic profit/loss": (
            np.array([-100.0, 20.0, 200.0]),
            np.array([0.05, 0.80, 0.15]),
        ),
        "Synthetic AI request cost": (
            np.array([0.002, 0.020, 0.200]),
            np.array([0.70, 0.25, 0.05]),
        ),
    }
    preset_name = st.selectbox(
        "Distribution preset",
        list(presets),
        key="expected_value_preset",
    )
    preset_values, preset_probabilities = presets[preset_name]
    distribution = pd.DataFrame(
        {
            "Value": preset_values,
            "Probability": preset_probabilities,
        }
    )
    editor_column, result_column = st.columns([0.85, 1.5])
    with editor_column:
        st.caption(
            "Edit values or probabilities. The chart updates only when the "
            "probabilities are valid."
        )
        edited_distribution = st.data_editor(
            distribution,
            hide_index=True,
            num_rows="fixed",
            key=f"distribution_editor_{preset_name}",
            column_config={
                "Value": st.column_config.NumberColumn(format="%.4f"),
                "Probability": st.column_config.NumberColumn(
                    min_value=0.0,
                    max_value=1.0,
                    format="%.4f",
                ),
            },
        )

    try:
        values, probabilities = validate_discrete_distribution(
            edited_distribution["Value"].to_numpy(),
            edited_distribution["Probability"].to_numpy(),
        )
    except ValueError as error:
        with result_column:
            st.error(str(error), icon=":material/error:")
            st.caption(
                f"Current probability sum: "
                f"{edited_distribution['Probability'].sum():.6f}"
            )
    else:
        mean = expected_value(values, probabilities)
        variance = discrete_variance(values, probabilities)
        weighted_table = pd.DataFrame(
            {
                "Value x": values,
                "Probability P(X=x)": probabilities,
                "Weighted contribution xP(X=x)": values * probabilities,
            }
        )
        with result_column:
            st.plotly_chart(
                make_expected_value_figure(
                    values,
                    probabilities,
                    f"{preset_name}: probability mass and balance point",
                ),
                width="stretch",
                key="expected_value_chart",
                config=PLOTLY_CONFIG,
            )
        with st.container(horizontal=True):
            st.metric("Expected value E[X]", f"{mean:.4f}", border=True)
            st.metric("Variance", f"{variance:.4f}", border=True)
            st.metric(
                "Probability sum",
                f"{probabilities.sum():.6f}",
                "valid",
                border=True,
            )
        st.dataframe(
            weighted_table,
            hide_index=True,
            width="stretch",
            column_config={
                "Value x": st.column_config.NumberColumn(format="%.4f"),
                "Probability P(X=x)": st.column_config.NumberColumn(format="%.4f"),
                "Weighted contribution xP(X=x)": st.column_config.NumberColumn(
                    format="%.4f"
                ),
            },
        )

    with st.expander(
        "Expected-value balance preview",
        icon=":material/balance:",
    ):
        show_generated_asset(
            "expected_value_balance.png",
            "A fair die balances at 3.5 even though 3.5 is not a possible roll.",
        )
    show_interpretation_cards(
        "Each xP(X=x) term contributes to the final sum. Rare, high-magnitude "
        "outcomes can move the mean substantially.",
        "Calling the expected value the most likely outcome, or assuming it "
        "must be observable in one trial.",
        "Expected token cost, fraud loss, or review load is useful for "
        "planning, but tail percentiles are still needed for capacity risk.",
    )


elif selected_section == "Variance and standard deviation":
    st.header("Variance and standard deviation")
    st.write(
        "Variance measures squared dispersion around the mean. Standard "
        "deviation returns that spread to the original unit."
    )
    formula_columns = st.columns(2)
    formula_columns[0].latex(r"\operatorname{Var}(X)=\mathbb{E}[(X-\mu)^2]")
    formula_columns[1].latex(
        r"\operatorname{Var}(X)=\mathbb{E}[X^2]-\mathbb{E}[X]^2"
    )

    standard_deviation = st.slider(
        "Selected Gaussian standard deviation",
        min_value=0.40,
        max_value=4.00,
        value=2.50,
        step=0.05,
        key="variance_sigma",
    )
    st.plotly_chart(
        make_variance_distribution_figure(standard_deviation),
        width="stretch",
        key="variance_distributions",
        config=PLOTLY_CONFIG,
    )
    distribution_stats = pd.DataFrame(
        {
            "Distribution": ["Narrow", "Medium", "Selected"],
            "Mean": [0.0, 0.0, 0.0],
            "Variance": [
                0.75**2,
                1.75**2,
                standard_deviation**2,
            ],
            "Standard deviation": [0.75, 1.75, standard_deviation],
            "5th percentile": [
                -1.64485 * 0.75,
                -1.64485 * 1.75,
                -1.64485 * standard_deviation,
            ],
            "Median": [0.0, 0.0, 0.0],
            "95th percentile": [
                1.64485 * 0.75,
                1.64485 * 1.75,
                1.64485 * standard_deviation,
            ],
        }
    )
    st.dataframe(
        distribution_stats,
        hide_index=True,
        width="stretch",
        column_config={
            column: st.column_config.NumberColumn(format="%.3f")
            for column in distribution_stats.columns
            if column != "Distribution"
        },
    )

    st.subheader("Squared deviations")
    outer_value = st.slider(
        "Symmetric outer observation",
        min_value=1.0,
        max_value=8.0,
        value=4.0,
        step=0.5,
        key="squared_deviation_outer",
    )
    observations = np.array([-outer_value, -1.0, 0.0, 1.0, outer_value])
    observation_mean = float(observations.mean())
    squared_deviations = (observations - observation_mean) ** 2
    deviation_table = pd.DataFrame(
        {
            "Observation": observations,
            "Deviation from mean": observations - observation_mean,
            "Squared deviation": squared_deviations,
            "Variance contribution": squared_deviations / len(observations),
        }
    )
    deviation_chart, deviation_data = st.columns([1.45, 1.0])
    with deviation_chart:
        st.plotly_chart(
            make_squared_deviation_figure(observations),
            width="stretch",
            key="squared_deviations",
            config=PLOTLY_CONFIG,
        )
    with deviation_data:
        st.dataframe(
            deviation_table,
            hide_index=True,
            width="stretch",
            column_config={
                column: st.column_config.NumberColumn(format="%.3f")
                for column in deviation_table.columns
            },
        )
        st.metric(
            "Population variance",
            f"{squared_deviations.mean():.3f}",
            border=True,
        )

    st.subheader("Bernoulli variance")
    bernoulli_probability = st.slider(
        "Bernoulli success probability p",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.01,
        key="bernoulli_probability",
    )
    probability_grid = np.linspace(0.0, 1.0, 301)
    bernoulli_figure = go.Figure()
    bernoulli_figure.add_trace(
        go.Scatter(
            x=probability_grid,
            y=bernoulli_variance(probability_grid),
            mode="lines",
            name="Var(X)=p(1−p)",
            line={"width": 3, "color": "#2563EB"},
        )
    )
    bernoulli_figure.add_trace(
        go.Scatter(
            x=[bernoulli_probability],
            y=[bernoulli_variance(bernoulli_probability)],
            mode="markers+text",
            name="Selected p",
            text=[f"{bernoulli_variance(bernoulli_probability):.3f}"],
            textposition="top center",
            marker={
                "symbol": "diamond",
                "size": 13,
                "color": "#DC2626",
                "line": {"color": "#111827", "width": 1},
            },
        )
    )
    bernoulli_figure.update_layout(
        title="Bernoulli uncertainty peaks at p=0.5",
        xaxis_title="Success probability p",
        yaxis_title="Variance p(1−p)",
        xaxis={"range": [0.0, 1.0]},
        yaxis={"range": [0.0, 0.28]},
        height=430,
    )
    st.plotly_chart(
        bernoulli_figure,
        width="stretch",
        key="bernoulli_variance_chart",
        config=PLOTLY_CONFIG,
    )
    with st.container(horizontal=True):
        st.metric("p", f"{bernoulli_probability:.2f}", border=True)
        st.metric(
            "Var(X)",
            f"{bernoulli_variance(bernoulli_probability):.4f}",
            border=True,
        )
        st.metric(
            "Standard deviation",
            f"{np.sqrt(bernoulli_variance(bernoulli_probability)):.4f}",
            border=True,
        )

    with st.expander(
        "Variance animation and Bernoulli preview",
        icon=":material/animation:",
    ):
        animation_column, curve_column = st.columns(2)
        with animation_column:
            show_generated_asset(
                "variance_spread.gif",
                "The mean remains fixed while spread increases.",
            )
        with curve_column:
            show_generated_asset(
                "bernoulli_variance.png",
                "Bernoulli variance is maximal at p=0.5.",
            )
    show_interpretation_cards(
        "Distributions can share the same mean while their variance, "
        "standard deviation, and tail percentiles differ substantially.",
        "Treating variance as a measure of center, forgetting its squared "
        "units, or assuming the mean alone characterizes reliability.",
        "For latency and AI quality, pair averages with dispersion, upper "
        "percentiles, subgroup variation, and catastrophic-tail frequency.",
    )


elif selected_section == "Monte Carlo convergence":
    st.header("Monte Carlo convergence")
    st.write(
        "Empirical estimates are random. More observations usually reduce "
        "instability, but the path continues to fluctuate around the "
        "theoretical target."
    )
    st.latex(
        r"\widehat{\theta}_n\longrightarrow\theta"
        r"\quad\text{as sample size }n\text{ grows}"
    )

    controls = st.columns(4)
    experiment_type = controls[0].selectbox(
        "Experiment type",
        ["Fair die mean", "Bayesian fraud posterior"],
        key="monte_carlo_experiment",
    )
    number_of_simulations = controls[1].slider(
        "Independent simulation paths",
        min_value=1,
        max_value=8,
        value=4,
        key="monte_carlo_paths",
    )
    max_sample_size = controls[2].select_slider(
        "Maximum sample size",
        options=[1_000, 5_000, 10_000, 20_000, 50_000, 100_000],
        value=20_000,
        key="monte_carlo_max_n",
    )
    base_seed = controls[3].number_input(
        "Base random seed",
        min_value=0,
        max_value=1_000_000,
        value=42,
        step=1,
        key="monte_carlo_seed",
    )

    if experiment_type == "Fair die mean":
        theoretical_value = 3.5
        simulation_paths = [
            cached_die_simulation(int(max_sample_size), int(base_seed) + index)
            for index in range(number_of_simulations)
        ]
        estimate_name = "Empirical expected die value"
    else:
        theoretical_value = bayes_posterior(0.01, 0.90, 0.05)
        simulation_paths = [
            cached_fraud_simulation(
                int(max_sample_size),
                int(base_seed) + index,
            )
            for index in range(number_of_simulations)
        ]
        estimate_name = "Empirical P(Fraud | Alert)"

    convergence_figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(estimate_name, "Absolute estimation error"),
    )
    terminal_estimates: list[float] = []
    for index, path in enumerate(simulation_paths):
        step = max(1, len(path) // 2_000)
        display_path = path.iloc[::step].copy()
        if display_path.index[-1] != path.index[-1]:
            display_path = pd.concat([display_path, path.tail(1)])
        valid_estimates = path["estimate"].dropna()
        terminal_estimates.append(float(valid_estimates.iloc[-1]))
        convergence_figure.add_trace(
            go.Scatter(
                x=display_path["sample_size"],
                y=display_path["estimate"],
                mode="lines",
                name=f"Path {index + 1}",
                line={"width": 1.6},
                opacity=0.78,
                hovertemplate=(
                    f"Path {index + 1}<br>n=%{{x:,}}"
                    "<br>estimate=%{y:.5f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        convergence_figure.add_trace(
            go.Scatter(
                x=display_path["sample_size"],
                y=display_path["absolute_error"],
                mode="lines",
                name=f"Path {index + 1} error",
                showlegend=False,
                line={"width": 1.2, "dash": "dot"},
                opacity=0.70,
                hovertemplate="n=%{x:,}<br>|error|=%{y:.5f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    convergence_figure.add_hline(
        y=theoretical_value,
        line_dash="dash",
        line_color="#DC2626",
        annotation_text=f"Theory={theoretical_value:.5f}",
        row=1,
        col=1,
    )
    convergence_figure.update_xaxes(title_text="Sample size", row=2, col=1)
    convergence_figure.update_yaxes(title_text="Estimate", row=1, col=1)
    convergence_figure.update_yaxes(
        title_text="Absolute error",
        row=2,
        col=1,
    )
    convergence_figure.update_layout(
        title="Independent Monte Carlo paths converge without becoming deterministic",
        height=650,
        legend={"orientation": "h", "y": -0.13},
        margin={"l": 60, "r": 20, "t": 85, "b": 85},
    )
    st.plotly_chart(
        convergence_figure,
        width="stretch",
        key="monte_carlo_chart",
        config=PLOTLY_CONFIG,
    )
    terminal_array = np.asarray(terminal_estimates)
    with st.container(horizontal=True):
        st.metric("Theoretical value", f"{theoretical_value:.5f}", border=True)
        st.metric(
            "Mean terminal estimate",
            f"{terminal_array.mean():.5f}",
            border=True,
        )
        st.metric(
            "Terminal estimate spread",
            f"{terminal_array.std(ddof=0):.5f}",
            border=True,
        )
        st.metric(
            "Mean terminal absolute error",
            f"{np.mean(np.abs(terminal_array - theoretical_value)):.5f}",
            border=True,
        )

    with st.expander(
        "Monte Carlo convergence animation",
        icon=":material/animation:",
    ):
        show_generated_asset(
            "monte_carlo_convergence.gif",
            "A fair-die empirical mean stabilizes as the sample grows.",
        )
    show_interpretation_cards(
        "Small samples can be far from theory. Larger samples stabilize the "
        "estimate, but different seeds still create different paths.",
        "Treating an empirical frequency as the exact theoretical probability, "
        "or expecting error to decrease monotonically at every sample.",
        "Rare-event simulation may require very large samples. Report sample "
        "size, seed, uncertainty, and analytical checks when available.",
    )


elif selected_section == "Expected cost and decision thresholds":
    st.header("Expected cost and decision thresholds")
    st.write(
        "Probability describes uncertainty; costs determine the action. Review "
        "when its fixed cost is lower than the expected cost of missing fraud."
    )
    st.latex(
        r"\operatorname{Cost}(\mathrm{review})=C_R,\qquad"
        r"\mathbb{E}[\operatorname{Cost}(\mathrm{no\ review})]=pC_M"
    )
    st.latex(r"p^*=\frac{C_R}{C_M}")

    controls = st.columns(3)
    review_cost = controls[0].number_input(
        "Review cost Cᵣ",
        min_value=0.0,
        max_value=1_000.0,
        value=5.0,
        step=1.0,
        key="review_cost",
    )
    missed_event_cost = controls[1].number_input(
        "Missed-fraud cost Cₘ",
        min_value=0.01,
        max_value=100_000.0,
        value=100.0,
        step=10.0,
        key="missed_event_cost",
    )
    selected_posterior = controls[2].slider(
        "Posterior fraud probability p",
        min_value=0.0,
        max_value=1.0,
        value=0.15,
        step=0.01,
        key="decision_posterior",
    )
    threshold = expected_decision_threshold(review_cost, missed_event_cost)
    no_review_cost = selected_posterior * missed_event_cost
    decision = "Review" if review_cost < no_review_cost else "Do not review"

    st.plotly_chart(
        make_expected_cost_figure(
            review_cost,
            missed_event_cost,
            selected_posterior,
        ),
        width="stretch",
        key="expected_cost_chart",
        config=PLOTLY_CONFIG,
    )
    with st.container(horizontal=True):
        st.metric("Decision threshold p*", f"{threshold:.2%}", border=True)
        st.metric("Cost(review)", f"{review_cost:,.2f}", border=True)
        st.metric(
            "Expected cost(no review)",
            f"{no_review_cost:,.2f}",
            border=True,
        )
        st.metric("Lower-cost action", decision, border=True)
    if threshold > 1.0:
        st.warning(
            "The threshold exceeds 100% because review costs more than the "
            "maximum modeled missed-event cost. Under this simplified model, "
            "review is never the lower-cost action.",
            icon=":material/warning:",
        )

    with st.expander(
        "Static expected-cost preview",
        icon=":material/image:",
    ):
        show_generated_asset(
            "expected_cost.png",
            "A fixed review cost intersects the probability-weighted miss cost.",
        )
    show_interpretation_cards(
        "The intersection is the probability at which both actions have equal "
        "expected cost. On either side, choose the lower curve.",
        "Defaulting to a 0.5 classification threshold without connecting it to "
        "calibration, error costs, or operational capacity.",
        "Real policies also include review capacity, delayed labels, legal "
        "constraints, unequal impact, and the cost of review errors.",
    )


else:
    st.header("Production connections")
    st.write(
        "Probability becomes operational when it is tied to ranking, "
        "multi-stage pipelines, shared failure modes, quality dispersion, and "
        "distribution monitoring."
    )
    st.latex(r"P(Y=1\mid X=x)")

    classification_column, rag_column = st.columns(2)
    with classification_column.container(border=True, height="stretch"):
        st.subheader("Classification")
        st.write(
            "Case-level probabilities support ranking, thresholding, and "
            "expected-loss decisions. Calibration asks whether predicted "
            "probabilities match observed frequencies."
        )
        predicted_probability = np.linspace(0.05, 0.95, 10)
        calibration_figure = go.Figure()
        calibration_figure.add_trace(
            go.Scatter(
                x=predicted_probability,
                y=predicted_probability,
                mode="lines",
                name="Perfect calibration",
                line={"dash": "dash", "color": "#111827"},
            )
        )
        calibration_figure.add_trace(
            go.Scatter(
                x=predicted_probability,
                y=np.clip(predicted_probability**1.25, 0.0, 1.0),
                mode="lines+markers",
                name="Synthetic model",
                marker={"symbol": "diamond", "size": 9},
                line={"color": "#2563EB"},
            )
        )
        calibration_figure.update_layout(
            title="Synthetic reliability diagram",
            xaxis_title="Predicted probability",
            yaxis_title="Observed frequency",
            xaxis={"range": [0.0, 1.0]},
            yaxis={"range": [0.0, 1.0]},
            height=390,
            legend={"orientation": "h", "y": -0.22},
        )
        st.plotly_chart(
            calibration_figure,
            width="stretch",
            key="classification_calibration",
            config=PLOTLY_CONFIG,
        )

    with rag_column.container(border=True, height="stretch"):
        st.subheader("RAG")
        st.latex(
            r"P(C)=P(C\mid G)P(G)+P(C\mid G^c)P(G^c)"
        )
        probability_good_retrieval = st.slider(
            "P(good retrieval)",
            min_value=0.0,
            max_value=1.0,
            value=0.75,
            step=0.01,
            key="rag_good_retrieval",
        )
        correct_given_good = st.slider(
            "P(correct | good retrieval)",
            min_value=0.0,
            max_value=1.0,
            value=0.90,
            step=0.01,
            key="rag_correct_good",
        )
        correct_given_poor = st.slider(
            "P(correct | poor retrieval)",
            min_value=0.0,
            max_value=1.0,
            value=0.20,
            step=0.01,
            key="rag_correct_poor",
        )
        good_contribution = correct_given_good * probability_good_retrieval
        poor_contribution = correct_given_poor * (
            1.0 - probability_good_retrieval
        )
        rag_correctness = good_contribution + poor_contribution
        rag_figure = go.Figure()
        rag_figure.add_trace(
            go.Bar(
                y=["P(correct answer)"],
                x=[good_contribution],
                name="Good-retrieval path",
                orientation="h",
                marker={
                    "color": "#2563EB",
                    "pattern": {"shape": ""},
                    "line": {"color": "#111827", "width": 1},
                },
                text=[f"{good_contribution:.1%}"],
                textposition="inside",
            )
        )
        rag_figure.add_trace(
            go.Bar(
                y=["P(correct answer)"],
                x=[poor_contribution],
                name="Poor-retrieval path",
                orientation="h",
                marker={
                    "color": "#D97706",
                    "pattern": {"shape": "/"},
                    "line": {"color": "#111827", "width": 1},
                },
                text=[f"{poor_contribution:.1%}"],
                textposition="inside",
            )
        )
        rag_figure.update_layout(
            title=f"Total probability decomposition = {rag_correctness:.1%}",
            barmode="stack",
            xaxis_title="Probability contribution",
            xaxis={"range": [0.0, 1.0]},
            height=300,
            legend={"orientation": "h", "y": -0.30},
        )
        st.plotly_chart(
            rag_figure,
            width="stretch",
            key="rag_decomposition",
            config=PLOTLY_CONFIG,
        )
        st.metric(
            "P(correct answer)",
            f"{rag_correctness:.2%}",
            border=True,
        )
        st.caption(
            "Vector similarity and reranker scores are not automatically "
            "calibrated probabilities."
        )

    reliability_column, quality_column = st.columns(2)
    with reliability_column.container(border=True, height="stretch"):
        st.subheader("Distributed-system reliability")
        service_success = st.slider(
            "Per-service success probability",
            min_value=0.90,
            max_value=0.999,
            value=0.99,
            step=0.001,
            format="%.3f",
            key="service_success",
        )
        region_failure = st.slider(
            "Shared cloud-region failure probability",
            min_value=0.0,
            max_value=0.10,
            value=0.02,
            step=0.005,
            key="region_failure",
        )
        reliability_figure = make_reliability_comparison_figure(
            service_success,
            region_failure,
        )
        st.plotly_chart(
            reliability_figure,
            width="stretch",
            key="reliability_comparison",
            config=PLOTLY_CONFIG,
        )
        independent_joint_success = service_success**2
        shared_joint_success = (
            (1.0 - region_failure) * service_success**2
        )
        st.metric(
            "Shared-failure reduction",
            f"{independent_joint_success - shared_joint_success:.3%}",
            border=True,
        )
        st.caption(
            "Multiplying component probabilities assumes independence. A "
            "shared region creates a common failure path."
        )

    with quality_column.container(border=True, height="stretch"):
        st.subheader("LLM evaluation")
        st.write(
            "These two synthetic systems have similar average quality but "
            "different variance and tail-risk behavior."
        )
        st.plotly_chart(
            make_llm_quality_comparison_figure(seed=42),
            width="stretch",
            key="llm_quality_comparison",
            config=PLOTLY_CONFIG,
        )
        st.caption(
            "A slightly higher mean can hide more catastrophic low-quality "
            "outcomes. Inspect dispersion and lower tails."
        )

    with st.container(border=True):
        st.subheader("Monitoring and distribution shift")
        st.write(
            "Even if detector behavior stays fixed, a prevalence shift changes "
            "the posterior. Historical probabilities require monitoring and "
            "recalibration."
        )
        shifted_prior = st.slider(
            "Shifted fraud prior",
            min_value=0.001,
            max_value=0.20,
            value=0.04,
            step=0.001,
            format="%.3f",
            key="shifted_prior",
        )
        prior_grid = np.linspace(0.001, 0.20, 300)
        posterior_grid = np.array(
            [bayes_posterior(value, 0.90, 0.05) for value in prior_grid]
        )
        original_posterior = bayes_posterior(0.01, 0.90, 0.05)
        shifted_posterior = bayes_posterior(shifted_prior, 0.90, 0.05)
        shift_figure = go.Figure()
        shift_figure.add_trace(
            go.Scatter(
                x=prior_grid,
                y=posterior_grid,
                mode="lines",
                name="Posterior curve",
                line={"color": "#2563EB", "width": 3},
            )
        )
        shift_figure.add_trace(
            go.Scatter(
                x=[0.01, shifted_prior],
                y=[original_posterior, shifted_posterior],
                mode="markers+text",
                name="Prior states",
                text=[
                    f"Original {original_posterior:.1%}",
                    f"Shifted {shifted_posterior:.1%}",
                ],
                textposition=["bottom right", "top left"],
                marker={
                    "symbol": ["circle", "diamond"],
                    "size": [12, 14],
                    "color": ["#64748B", "#DC2626"],
                    "line": {"color": "#111827", "width": 1},
                },
            )
        )
        shift_figure.update_layout(
            title="Prior probability shift changes the posterior",
            xaxis_title="Fraud prevalence P(F)",
            yaxis_title="P(Fraud | Alert)",
            yaxis={"range": [0.0, 1.0]},
            height=430,
            legend={"orientation": "h", "y": -0.18},
        )
        st.plotly_chart(
            shift_figure,
            width="stretch",
            key="distribution_shift",
            config=PLOTLY_CONFIG,
        )
        with st.container(horizontal=True):
            st.metric("Original prior", "1.00%", border=True)
            st.metric(
                "Original posterior",
                f"{original_posterior:.2%}",
                border=True,
            )
            st.metric("Shifted prior", f"{shifted_prior:.2%}", border=True)
            st.metric(
                "Shifted posterior",
                f"{shifted_posterior:.2%}",
                border=True,
            )

    show_interpretation_cards(
        "The same probability foundations recur across calibration, RAG "
        "decomposition, reliability, evaluation variance, and drift.",
        "Treating similarity, logits, or self-reported model confidence as "
        "calibrated probabilities without empirical evidence.",
        "Monitor priors, conditional performance, calibration, dependencies, "
        "and tails—not only one global average or offline accuracy score.",
    )

"""Local interactive statistical power lab. Launch from this topic directory."""

from pathlib import Path
import pandas as pd
import streamlit as st

from from_scratch import sample_size_per_arm
from power_math import (
    approximation_warning, cluster_information, detectable_effect, required_n,
    sampling_model, theoretical_power, traffic_plan, validate_design,
)
from simulation import simulate_experiments, synthetic_intervals
import visual_charts as charts

st.set_page_config(page_title="Statistical power lab · Day 46", page_icon="📐", layout="wide")

DEFAULTS = dict(baseline=0.10, effect_pp=1.0, alpha=0.05, target=0.80,
                sample_n=15000, seed=46, module="01 · Hypothesis distributions")
MODULES = [
    "01 · Hypothesis distributions", "02 · Power and sample size",
    "03 · 3D power surface", "04 · Effect size sensitivity",
    "05 · Monte Carlo simulation", "06 · Correlated observations",
    "07 · Business feasibility", "08 · Statistical vs. practical significance",
]


def reset_defaults():
    st.session_state.clear()
    st.session_state.update(DEFAULTS)


for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def plot(fig, key):
    st.plotly_chart(fig, key=key, width="stretch", theme=None,
                    config=dict(displaylogo=False, scrollZoom=True))


def explanation(seeing, changes, formula, implication):
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown("**What you are seeing**")
        st.write(seeing)
        st.markdown("**What changes when parameters move**")
        st.write(changes)
    with right:
        with st.container(border=True):
            st.markdown("**Mathematical interpretation**")
            st.latex(formula)
            st.markdown("**Production implication**")
            st.write(implication)


@st.cache_data(max_entries=16, show_spinner=False)
def surface(baseline, alpha, target, max_n, max_effect):
    return charts.surface_figure(baseline, alpha, target, max_n, max_effect)


@st.cache_data(max_entries=32, show_spinner=False)
def sensitivity(baseline, alpha, target):
    return charts.sensitivity_figure(baseline, alpha, target)


with st.sidebar:
    st.caption("APPLIED AI ENGINEERING LAB / DAY 46")
    st.subheader("Design your experiment")
    module = st.selectbox("Learning module", MODULES, key="module")
    st.button("Reset to defaults", on_click=reset_defaults, width="stretch")
    st.markdown("**Hypothetical population inputs**")
    p = st.number_input("Baseline probability", min_value=0.001, max_value=0.999,
                        step=0.01, format="%.3f", key="baseline")
    delta = st.number_input("Absolute effect (percentage points)", min_value=-99.0,
                            max_value=99.0, step=0.1, key="effect_pp") / 100
    alpha = st.number_input("Significance level α", min_value=0.001, max_value=0.20,
                            step=0.01, format="%.3f", key="alpha")
    target = st.slider("Target power", 0.60, 0.99, step=0.01, key="target")
    n = st.slider("Independent users per arm", 100, 100000, step=100, key="sample_n")
    seed = st.number_input("Random seed", min_value=0, max_value=2**32 - 1, step=1, key="seed")
    st.caption("Two-sided equality test · balanced allocation · fixed horizon")

st.caption("STATISTICS / EXPERIMENT DESIGN / INTERACTIVE STUDY")
st.title("Statistical power lab")
st.markdown("### From effect size to experimental decisions")
st.write("Plan a hypothetical RAG assistant experiment, inspect uncertainty, and challenge the design before collecting data.")

try:
    validate_design(p, delta, alpha)
except ValueError as error:
    st.error(str(error))
    st.info("Choose population probabilities strictly between 0 and 1. Boundary probabilities need a different inference model.")
    st.stop()

power = theoretical_power(p, delta, n, alpha)
try:
    needed = required_n(p, delta, alpha, target)
except ValueError as error:
    needed = None
    st.info(str(error))

cards = st.columns(4)
cards[0].metric("Population treatment", f"{p + delta:.1%}", f"{p:.1%} control", delta_color="off", border=True)
cards[1].metric("Absolute effect", f"{delta * 100:+.2f} pp", f"{delta / p:+.1%} relative uplift", delta_color="off", border=True)
cards[2].metric("Calculated power at selected n", f"{power:.1%}", border=True)
cards[3].metric("Calculated required n / arm", f"{needed:,}" if needed else "Not available", border=True)
st.caption("Population assumptions are hypothetical. Blue/teal curves are analytical approximations. Monte Carlo outcomes and interval examples are synthetic.")
warning = approximation_warning(p, delta, n)
if warning:
    st.warning(warning)

st.subheader(module)

if module == MODULES[0]:
    model = sampling_model(p, delta, n, alpha)
    plot(charts.distribution_figure(p, delta, n, alpha), "distributions")
    with st.container(horizontal=True):
        st.metric("Type I error · under H0", f"{model['alpha']:.2%}", border=True)
        st.metric("Type II error · under selected H1", f"{model['beta']:.2%}", border=True)
        st.metric("Detection probability · under H1", f"{model['power']:.2%}", border=True)
    st.caption(f"H0 reference common rate = expected pooled rate {(model['null_rate']):.3%}; critical boundaries = ±{model['critical'] * 100:.3f} pp. This freezes the pooled SE for planning.")
    explanation(
        "Both curves describe the estimated difference on the same scale. Red tails are false rejection under H0; amber H1 mass between the cutoffs is β. Teal H1 tails are power. Curve overlap is not β.",
        "Larger n narrows both distributions and moves the critical boundaries toward zero. A fixed nonzero population effect then becomes easier to detect.",
        r"c=z_{1-\alpha/2}SE_0,\quad \beta=P_{H_1}(-c\leq\hat\Delta\leq c)",
        "Count independent users, not all conversations, before interpreting the sensitivity of an assistant evaluation.",
    )

elif module == MODULES[1]:
    if needed:
        plot(charts.power_curve_figure(p, delta, n, alpha, target), "curve")
        st.write(f"Standard-library dominant-tail estimate: **{sample_size_per_arm(p, p + delta, alpha, target):,}/arm**. Statsmodels two-tail numerical estimate: **{needed:,}/arm**.")
    explanation(
        "The marker reports prospective power for the selected sample. The vertical line is the rounded sample requirement at your target.",
        "Power grows nonlinearly and approaches one. Increasing target power or reducing alpha shifts the required sample upward.",
        r"n\approx\frac{[z_{1-\alpha/2}\sqrt{2\bar p(1-\bar p)}+z_q\sqrt{p_A(1-p_A)+p_B(1-p_B)}]^2}{\Delta^2}",
        "A powered design is conditional on the assumed effect; it does not guarantee a significant result or approve a launch.",
    )
    with st.expander("Read the formula"):
        st.write("n is users per arm; α is Type I error; β is missed-detection probability at a specified H1; q=1−β is target power. pA and pB are assumed population probabilities, p̄=(pA+pB)/2, and z denotes a standard-normal quantile. The closed form ignores the remote rejection tail; the numerical solution keeps both.")

elif module == MODULES[2]:
    max_n = st.slider("Surface maximum users per arm", 5000, 200000, 60000, 5000)
    max_effect_pp = st.slider("Surface maximum positive effect (pp)", 0.1, 8.0, 3.0, 0.1)
    plot(surface(p, alpha, target, max_n, max_effect_pp / 100), "surface")
    explanation(
        "Rotate, zoom and hover to inspect calculated power. The translucent plane and contour mark target power; green surface regions meet the requirement. Only valid positive improvements are shown.",
        "A smaller effect moves the target crossing toward much larger samples. Baseline and alpha change the entire landscape.",
        r"\pi=\pi(n,\Delta\mid p_A,\alpha)",
        "A 3D surface exposes regions where the desired sensitivity is beyond the team's traffic budget.",
    )

elif module == MODULES[3]:
    plot(sensitivity(p, alpha, target), "sensitivity")
    mde = detectable_effect(p, n, alpha, target)
    business_pp = st.number_input("Smallest worthwhile improvement (pp)", 0.0, 20.0, 0.5, 0.1)
    st.write(f"Business threshold: **{business_pp:.2f} pp**. Positive MDE at {n:,}/arm: **{mde * 100:.3f} pp**." if mde is not None else "Target power is unattainable for an interior positive effect at this sample.")
    if mde is not None:
        st.caption(f"This MDE corresponds to {mde / p:.1%} relative uplift; it is a calculated design property, not a benefit valuation.")
    explanation(
        "The solid line is required n for positive absolute improvements. The dashed inverse-square reference is anchored at the largest effect, with variance treated as fixed.",
        "Both curves rise sharply for small effects. Bernoulli variance changes with pB, so the inverse-square reference is not an identity across these rates.",
        r"n\propto\Delta^{-2}\ \text{at fixed variance};\quad \text{uplift}=\Delta/p_A",
        "Set the minimum worthwhile improvement from product value, then compare it with the design's MDE; do not silently replace one with the other.",
    )

elif module == MODULES[4]:
    with st.form("simulation_controls"):
        mode_name = st.selectbox("Population scenario", ["H1 · selected population effect", "H0 · no population effect"], key="simulation_mode")
        repetitions = st.select_slider("Monte Carlo repetitions", [1000, 5000, 20000, 100000], value=5000)
        run = st.form_submit_button("Run simulation", type="primary")
    mode = "null" if mode_name.startswith("H0") else "alternative"
    if run:
        with st.spinner("Generating independent binomial experiments…"):
            st.session_state["simulation_result"] = simulate_experiments(p, delta, n, alpha, repetitions, seed, mode)
    result = st.session_state.get("simulation_result")
    if result is not None:
        expected = dict(baseline=p, effect=0.0 if mode == "null" else delta, n=n, alpha=alpha,
                        repetitions=repetitions, seed=seed, mode=mode)
        if result["config"] != expected:
            st.warning("Inputs changed. The plots below retain the previous run's configuration; press Run simulation to replace it.")
        st.caption(f"Run configuration: {result['config']}")
        with st.container(horizontal=True):
            st.metric("Empirical Type I error" if result["config"]["mode"] == "null" else "Empirical power", f"{result['rejection_rate']:.2%}", border=True)
            st.metric("95% Monte Carlo interval", f"{result['ci_low']:.2%} – {result['ci_high']:.2%}", border=True)
            if result["config"]["mode"] == "alternative":
                st.metric("Empirical Type II error", f"{1 - result['rejection_rate']:.2%}", border=True)
        plot(charts.monte_carlo_figure(result), "mc_histograms")
        plot(charts.rejection_figure(result), "mc_rate")
        if result["degenerate_draws"]:
            st.warning(f"{result['degenerate_draws']} all-zero/all-one pooled tables were conservatively assigned p=1. Normal inference is unreliable here.")
        frame = pd.DataFrame({k: result["outcomes"][k] for k in ("control_rate", "treatment_rate", "effect", "p_value", "reject")})
        st.download_button("Download synthetic run (CSV)", frame.to_csv(index=False), "day46_simulation.csv", "text/csv")
    else:
        st.info("Choose a population scenario and press Run simulation. Slider changes do not launch simulations.")
    explanation(
        "Each draw is a complete synthetic A/B experiment with binomial arm counts, observed rates, an estimated difference, a pooled Z p-value and a rejection decision. H0 estimates false positives; H1 estimates detection.",
        "More repetitions reduce Monte Carlo error, not experimental standard error. Larger n changes each experiment's power. Changing the seed changes the sampled experiments.",
        r"\hat\pi=R/B,\quad MCSE\approx\sqrt{\hat\pi(1-\hat\pi)/B}",
        "Validate the intended analysis, especially for rare outcomes. The 95% Wilson interval quantifies simulation error only; it excludes uncertainty in population assumptions.",
    )
    st.caption("The simulated pooled Z-test uses its observed pooled SE. The analytical model freezes that SE at its population planning value; finite-sample discreteness and random SE can cause differences.")

elif module == MODULES[5]:
    a, b, c = st.columns(3)
    users = a.number_input("Distinct users", 1, 1000000, 1000, 100)
    conversations = b.slider("Conversations per user", 1, 40, 10)
    icc = c.slider("Intraclass correlation", 0.0, 1.0, 0.10, 0.01)
    raw, de, effective = cluster_information(users, conversations, icc)
    with st.container(horizontal=True):
        st.metric("Raw conversations", f"{raw:,}", border=True)
        st.metric("Approximate design effect", f"{de:.2f}", border=True)
        st.metric("Equivalent independent conversations", f"{effective:,.0f}", border=True)
    plot(charts.cluster_figure(users, conversations, icc), "clusters")
    plot(charts.conversation_figure(users, conversations), "conversation_groups")
    st.caption("The dot diagram shows up to 12 users × 12 conversations. Colors indicate groups, not simulated outcome dependence.")
    explanation(
        "Raw conversations stay fixed while equivalent independent information falls with ICC. Conversations from a shared user form a cluster.",
        "At ICC=0 all conversations are independent in the heuristic; at ICC=1, each equal-sized cluster contributes roughly one conversation's information. With one conversation per user, ICC has no effect.",
        r"DE=1+(m-1)\rho,\qquad N_{eff}\approx N_{raw}/DE",
        "This equal-cluster heuristic does not size an arbitrary cluster-randomized experiment. Use a cluster-aware estimator, cluster counts and an appropriate power model.",
    )

elif module == MODULES[6]:
    a, b, c = st.columns(3)
    required = a.number_input("Required independent users per arm", 1, 1000000000, int(needed or 10000), 100)
    daily = b.number_input("Daily potential distinct users", 1, 10000000, 2500, 100)
    eligibility = c.slider("Eligible proportion of potential traffic", 0.01, 1.0, 0.40, 0.01)
    a, b, c = st.columns(3)
    share = a.slider("Eligible traffic admitted to experiment", 0.01, 1.0, 1.0, 0.01)
    delay = b.number_input("Outcome maturation delay (days)", 0, 730, 2, 1)
    window = c.number_input("Available calendar window (days)", 1, 730, 14, 1)
    plan = traffic_plan(required, daily, eligibility, share, delay, window)
    st.caption(f"{daily * eligibility:,.0f} expected eligible users/day before traffic admission. Admitted users are split 50/50; eligibility is applied once.")
    plot(charts.feasibility_figure(plan, required, window), "feasibility")
    with st.container(horizontal=True):
        st.metric("Expected mature n / arm at deadline", f"{plan['available_n']:,}", border=True)
        st.metric("Day required mature sample is reached", f"{plan['reach_day']:,}", border=True)
        st.metric("Window sufficient for supplied n", "Yes" if plan["sufficient"] else "No", border=True)
    if plan["available_n"] > 0:
        available_power = theoretical_power(p, delta, plan["available_n"], alpha)
        mde = detectable_effect(p, plan["available_n"], alpha, target)
        st.write(f"At the original population effect: **{available_power:.2%} power**. Positive MDE at the selected target: **{mde * 100:.3f} pp**." if mde is not None else f"Available power: {available_power:.2%}; target MDE is outside the feasible interior probability range.")
        sparse = approximation_warning(p, delta, plan["available_n"])
        if sparse:
            st.warning(sparse)
    else:
        st.warning("No outcomes have matured by the deadline. Power and MDE are undefined at zero observed sample.")
    if needed and required != needed:
        st.warning(f"Your supplied n differs from the calculated requirement ({needed:,}/arm). Window sufficiency refers to your supplied n, not automatically the target-power plan.")
    explanation(
        "The assigned curve leads the fully observed curve by the maturation delay. The threshold is per arm, and the deadline applies to mature outcomes.",
        "Eligibility and experiment traffic share multiply once. Longer outcome delays reduce usable sample at a fixed deadline, even when assignments look sufficient.",
        r"n_{mature}(D)=\max(D-L,0)\,U e f/2",
        "Stable traffic is a planning assumption, not a guaranteed finish date. Seasonality, missing outcomes, re-enrollment and cluster dependence require additional modeling.",
    )

elif module == MODULES[7]:
    threshold = st.number_input("Minimum worthwhile improvement (pp)", 0.0, 20.0, 0.5, 0.1) / 100
    rows = synthetic_intervals(p, delta, (300, 3000, 30000, 300000), alpha, seed)
    plot(charts.interval_figure(rows, threshold, alpha), "practical_intervals")
    table = pd.DataFrame(rows)
    table["pooled_p_below_alpha"] = table["p_value"] < alpha
    table["interval_vs_business_threshold"] = [
        "Entire CI above threshold" if r["ci_low"] > threshold else
        "Entire CI below threshold" if r["ci_high"] < threshold else
        "CI spans threshold" for r in rows
    ]
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption("Every point is synthetic and seed-specific; no experiments were filtered for their p-values. Colors indicate whether the Wald CI excludes zero. The pooled test and unpooled interval need not invert each other exactly.")
    explanation(
        "Intervals show estimation uncertainty. The zero reference answers a different question from the business threshold. Try +0.2 pp with a +0.5 pp threshold to examine a small effect at large n.",
        "Larger n usually narrows the intervals; each row remains one random experiment. Moving the business threshold changes practical interpretation without changing p-values.",
        r"\hat\Delta\pm z_{1-\alpha/2}\sqrt{\hat p_A(1-\hat p_A)/n+\hat p_B(1-\hat p_B)/n}",
        "A nonsignificant result does not prove equivalence. Interval width and decision utility differ; deployment still needs cost, safety, quality and risk evaluation.",
    )

animation = {
    MODULES[0]: "sampling_distributions.gif",
    MODULES[1]: "power_growth.gif",
    MODULES[5]: "effective_sample_size.gif",
}.get(module)
if animation:
    preview = Path(__file__).resolve().parent / "assets" / animation
    if preview.is_file():
        with st.expander("Watch the animated explanation (default export configuration)"):
            st.image(str(preview))
            st.caption("This reproducible animation uses the documented export defaults, independently of the current dashboard controls.")

st.caption("Educational approximation · hypothetical parameters · synthetic simulations · interpretation review required. All calculations and charts run locally.")

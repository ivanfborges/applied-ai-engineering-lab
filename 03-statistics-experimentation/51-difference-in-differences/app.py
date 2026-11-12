"""Run locally: streamlit run app.py. All displayed data are synthetic."""

from pathlib import Path
import sys

import numpy as np
import streamlit as st

# Also supports execution from the repository root and headless AppTest.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from charts import (coefficient_plot, event_plot, geometry, plot_3d, sampling_plot,
                    staggered_plot, trend_chart)
from estimation import (event_study, oracle_decomposition, placebo, regression_summary,
                        sampling_distribution, trajectories)
from simulation import expected_bias, simulate

st.set_page_config(page_title="DiD · Causal learning lab", layout="wide")


@st.cache_data(max_entries=24, show_spinner=False)
def cached_regression(effect, baseline_gap, noise, slope=0.0):
    return regression_summary(simulate(effect=effect, baseline_gap=baseline_gap, noise=noise, slope=slope))


@st.cache_data(max_entries=12, show_spinner="Generating independent synthetic panels…")
def cached_sampling(draws, effect, baseline_gap, noise, slope):
    return sampling_distribution(draws=draws, effect=effect, baseline_gap=baseline_gap, noise=noise, slope=slope)


def chart(fig, key, alt):
    st.plotly_chart(fig, key=key, width="stretch", alt=alt,
                    config={"displaylogo": False, "scrollZoom": False})


def metrics(panel):
    values = oracle_decomposition(panel)
    columns = st.columns(3)
    for column, label, value in zip(columns, ["True post ATT · synthetic", "Estimated DiD", "Estimation error"],
                                    [values["truth"], values["estimate"], values["error"]]):
        column.metric(label, f"{value:+.3f} pts", border=True)
    st.caption(f"Synthetic-only decomposition: identification bias {values['identification_bias']:+.3f} "
               f"+ random noise contribution {values['noise_error']:+.3f} = error {values['error']:+.3f} points. "
               "Real observational data do not reveal these components.")
    return values


st.caption("APPLIED AI ENGINEERING LAB  /  DAY 51  /  SYNTHETIC CAUSAL EXPERIMENTS")
st.title("Difference-in-Differences")
st.markdown("**Change the assumptions. Watch the counterfactual change.** "
            "An AI assistant reaches 60 of 120 support teams at period 0.")
with st.sidebar:
    st.header("Rollout controls")
    st.caption("Shared across tabs · fixed seed 51")
    effect = st.slider("True treatment effect (points)", -8.0, 12.0, 4.0, 0.5, key="effect")
    baseline = st.slider("Baseline difference (points)", -15.0, 15.0, 5.0, 0.5, key="baseline")
    noise = st.slider("Measurement noise (SD)", 0.0, 8.0, 1.5, 0.5, key="noise")
    st.divider()
    st.markdown("**Design:** 120 teams · 12 periods · 6 before / 6 after. "
                "Higher productivity points are better; points are not percentages.")
    st.caption("Blue: control · orange: treated · dashed gray: estimated counterfactual · "
               "dotted purple: synthetic untreated truth · green: estimated gap · red: bias.")

panel = simulate(effect=effect, baseline_gap=baseline, noise=noise)
labels = ["Counterfactual Intuition", "DiD Estimator", "Parallel Trends", "Regression & Uncertainty",
          "Event Study", "When DiD Fails", "3D Exploration"]
tabs = st.tabs(labels, key="chapter", on_change="rerun")

with tabs[0]:
    if tabs[0].open:
        st.subheader("1 · Build the missing untreated path")
        st.caption("Objective: separate what happened from what would have happened without the assistant.")
        reveal = st.toggle("Reveal true untreated outcomes — simulation only", key="truth")
        selected = st.slider("Post period to inspect", 0, 5, 5, key="cf_period")
        chart(trend_chart(panel, reveal, selected), "counterfactual", "Observed team-group outcomes and estimated untreated counterfactual")
        values = metrics(panel)
        before_after = trajectories(panel).treated
        change = before_after.loc[0:].mean() - before_after.loc[: -1].mean()
        st.markdown(f"**Observed:** treated productivity changed by **{change:+.2f} points**. "
                    f"**DiD:** the additional change beyond control is **{values['estimate']:+.2f}**.")
        st.latex(r"\widehat{Y}_{Tt}(0)=\bar Y_{Ct}+\overline{(Y_T-Y_C)}_{pre}")
        st.info("The dashed path extends the average pre-period group gap along the observed control trend. "
                "Its average post-period gap equals pooled DiD. A gap at one selected period need not equal that average. "
                "True untreated post outcomes are missing in real treated teams.")
        if reveal:
            st.caption("Purple uses the simulator's no-rollout Y(0), including realized noise. "
                       "It is neither fitted nor available in a real observational study.")

with tabs[1]:
    if tabs[1].open:
        st.subheader("2 · Subtract the change that also happened without treatment")
        st.caption("Objective: derive the four-mean estimator. These illustrative cells are independent of the rollout sliders.")
        cols = st.columns(4)
        tpre = cols[0].number_input("Treated before", value=100.0, key="tpre")
        tpost = cols[1].number_input("Treated after", value=125.0, key="tpost")
        cpre = cols[2].number_input("Control before", value=100.0, key="cpre")
        cpost = cols[3].number_input("Control after", value=110.0, key="cpost")
        stage = st.slider("Derivation step", 1, 6, 6, key="geometry_stage")
        chart(geometry(tpre, tpost, cpre, cpost, stage-1), "geometry", "Six-step geometric subtraction of treated and control changes")
        st.latex(rf"DiD=({tpost:g}-{tpre:g})-({cpost:g}-{cpre:g})="
                 rf"{tpost-tpre:g}-{cpost-cpre:g}={tpost-tpre-cpost+cpre:g}")
        st.caption("Use the step slider or ▶ Derive DiD inside the chart. Different starting levels cancel; "
                   "different untreated changes do not. Causal interpretation still requires a credible comparison.")

with tabs[2]:
    if tabs[2].open:
        st.subheader("3 · Change the unobserved untreated trend")
        st.caption("Objective: see identical successful estimators disagree with synthetic causal truth.")
        slope = st.slider("Differential untreated trend (points / period)", -2.0, 2.0, 0.0, 0.1, key="slope")
        altered = simulate(effect=effect, baseline_gap=baseline, noise=noise, slope=slope)
        chart(trend_chart(altered, True), "parallel", "Observed outcomes, true untreated path and assumption-based counterfactual under differential trends")
        values = metrics(altered)
        st.latex(r"E[\widehat{DiD}]-ATT=s(\bar t_{post}-\bar t_{pre})=6s")
        if slope:
            st.warning(f"Parallel trends intentionally violated: the expected untreated change difference is {6*slope:+.2f} points. "
                       "The treatment effect is unchanged. A small p-value would not repair this identification problem.")
        else:
            st.info("Scenario A: parallel untreated trends hold in expectation. Realized trajectories still contain finite-sample noise. "
                    "Move the slope to construct Scenario B.")
        st.caption("At noise = 0 the estimation error equals the constructed bias. At positive noise the two differ; "
                   "the decomposition uses unavailable synthetic truth, not a real-world bias diagnostic.")

with tabs[3]:
    if tabs[3].open:
        st.subheader("4 · Precision is different from causal identification")
        st.caption("Objective: connect manual DiD, statsmodels TWFE and repeated-sampling variation.")
        reg_slope = st.slider("Untreated trend in regression scenario", -2.0, 2.0, 0.0, 0.1, key="reg_slope")
        st.caption("Adjust Measurement noise (SD) in the sidebar to change precision while holding the standardized noise draw fixed.")
        summary = cached_regression(effect, baseline, noise, reg_slope)
        cols = st.columns(4)
        for col, label, value in zip(cols, ["Manual DiD", "TWFE coefficient", "True ATT", "Team-cluster SE"],
                                     [summary["manual"], summary["twfe"], effect, summary["se"]]):
            col.metric(label, f"{value:.3f}", border=True)
        chart(coefficient_plot(summary, effect), "coefficient", "TWFE effect and clustered 95 percent confidence interval compared with synthetic truth and zero")
        st.latex(r"Y_{it}=\alpha_i+\lambda_t+\tau D_i Post_t+\epsilon_{it}")
        st.caption(f"95% CI [{summary['low']:.3f}, {summary['high']:.3f}] · CRV1 team clusters · "
                   f"finite-sample correction · Student t({summary['df']:.0f}). "
                   f"Manual–TWFE difference: {summary['manual']-summary['twfe']:.2e}.")
        if reg_slope:
            st.warning("This regression scenario violates parallel trends. Its confidence interval reflects sampling precision, not confounding bias.")
        else:
            st.info("More measurement noise typically widens intervals across repeated samples; it does not itself create differential untreated trends. "
                    "The current coefficient can still move with its realized noise draw.")
        draws = st.select_slider("Independent datasets", options=[100, 200, 500], value=200, key="draws")
        config = (draws, effect, baseline, noise, reg_slope)
        if st.button("Run sampling experiment", key="run_sampling", type="primary"):
            st.session_state["sampling_run"] = (config, cached_sampling(*config))
        saved = st.session_state.get("sampling_run")
        if saved and saved[0] == config:
            distribution = saved[1]
            chart(sampling_plot(distribution, effect + expected_bias(effect, reg_slope)), "sampling", "Distribution of DiD estimates across independent simulated panels")
            mcse = distribution.estimate.std(ddof=1) / np.sqrt(draws)
            coverage = distribution.covered.mean()
            st.markdown(f"Mean estimate **{distribution.estimate.mean():.3f}** · empirical SD "
                        f"**{distribution.estimate.std(ddof=1):.3f}** · mean MCSE **{mcse:.3f}** · "
                        f"fraction of intervals covering truth **{coverage:.3f}** "
                        f"(MCSE {np.sqrt(coverage*(1-coverage)/draws):.3f}).")
            st.caption("Each histogram entry comes from a separate independently seeded dataset, root seed 5100. "
                       "This is a sampling distribution, not the confidence interval of the current dataset. "
                       "Interpretation candidate: author review required.")
            if coverage in (0, 1):
                st.caption("At an observed coverage fraction of 0 or 1 the plug-in MCSE is zero. "
                           "That boundary estimate does not imply the true coverage probability is known exactly.")
        elif saved:
            st.info("Parameters changed. Run the sampling experiment again to display results for this configuration.")

with tabs[4]:
    if tabs[4].open:
        st.subheader("5 · Look for pre-treatment warning signs")
        st.caption("Objective: distinguish relative pre-period differences from post-period effect estimates.")
        event_slope = st.slider("Differential trend for diagnostics", -2.0, 2.0, 0.0, 0.1, key="event_slope")
        event_anticipation = st.slider("Anticipation in last two pre-periods (points)", 0.0, 8.0, 0.0, 0.5, key="event_anticipation")
        diagnostic = simulate(effect=effect, baseline_gap=baseline, noise=noise,
                              slope=event_slope, anticipation=event_anticipation)
        events = event_study(diagnostic)
        chart(event_plot(events), "event", "Pre and post event-time coefficients relative to the last pre-treatment period with pointwise intervals")
        pre_events = events.loc[(events.period < 0) & ~events.reference]
        excluded = int(((pre_events.low > 0) | (pre_events.high < 0)).sum())
        st.caption(f"{excluded} of {len(pre_events)} non-reference pre-period pointwise intervals exclude zero. "
                   "An isolated exclusion can occur by chance even under a valid DGP; this count is not a joint pre-trends test.")
        st.latex(r"\widehat\beta_k=(\bar Y_{Tk}-\bar Y_{Ck})-(\bar Y_{T,-1}-\bar Y_{C,-1})")
        test = placebo(diagnostic)
        st.metric("Pre-only placebo · fake rollout at -3", f"{test['estimate']:+.3f} pts", border=True)
        st.caption(f"95% CI [{test['low']:.3f}, {test['high']:.3f}]. Only periods -6..-1 were used; "
                   "windows -6..-4 versus -3..-1. Event reference -1 is fixed at zero, with no estimated interval.")
        if event_slope or event_anticipation:
            st.warning("Assumptions intentionally violated. Anticipation contaminates reference period -1 and can shift every event coefficient. "
                       "The placebo may also contain anticipatory outcomes even though every row precedes official adoption.")
        if test["low"] > 0 or test["high"] < 0:
            st.warning("This placebo interval excludes zero: a warning about this design, not proof of a particular failure mechanism.")
        st.info("Near-zero pre-coefficients and a small placebo do not prove future parallel trends. "
                "Intervals are pointwise, not simultaneous; multiple-testing and pretest-selection risks remain.")
        with st.expander("Advanced · why staggered adoption changes comparisons"):
            chart(staggered_plot(), "staggered", "Early, late and never-treated adoption heatmap; conceptual schedule only")
            st.markdown("At period 3, early adopters are already treated. Their outcomes can contain evolving effects, "
                        "so using them as controls for late adopters can contaminate a conventional TWFE comparison. "
                        "This does not make every TWFE estimate biased. Define cohort-time ATT and aggregation; "
                        "consider Callaway–Sant'Anna or Sun–Abraham designs. No modern estimator is approximated here.")

with tabs[5]:
    if tabs[5].open:
        st.subheader("6 · When DiD Fails")
        st.caption("Objective: manipulate one causal failure at a time, holding the true post-treatment effect fixed.")
        scenario = st.selectbox("Failure mode", ["Differential untreated trends", "Spillover effects", "Anticipation effects"], key="failure")
        config = dict(effect=effect, baseline_gap=baseline, noise=noise)
        if scenario == "Differential untreated trends":
            parameter = st.slider("Failure: differential slope", -2.0, 2.0, 0.8, 0.1, key="failure_slope")
            config["slope"] = parameter
            explanation = "Parallel untreated trends fail: the treated no-rollout path has an extra slope."
            mitigation = "Investigate concurrent changes and assignment; choose more credible controls and conduct sensitivity analysis."
        elif scenario == "Spillover effects":
            parameter = st.slider("Control spillover fraction of treatment effect", 0.0, 1.0, 0.5, 0.05, key="spillover")
            config["spillover"] = parameter
            explanation = "No-interference fails: controls receive a fraction of the own-treatment effect after rollout. For a positive effect this attenuates DiD."
            mitigation = "Redesign exposure boundaries, isolate comparisons or explicitly estimate direct and spillover effects."
        else:
            parameter = st.slider("Anticipatory gain (last two pre-periods)", 0.0, 8.0, 3.0, 0.5, key="anticipation")
            config["anticipation"] = parameter
            explanation = "No-anticipation fails: treated outcomes increase at -2 and -1, before official exposure at 0."
            mitigation = "Audit announcement/training dates; redefine the treatment start and use uncontaminated pre-periods with a defensible comparison."
        failed = simulate(**config)
        chart(trend_chart(failed, True), "failure_chart", "Observed outcomes and synthetic no-rollout truth under the selected assumption violation")
        metrics(failed)
        st.latex(r"E[\widehat{DiD}]-ATT=6s-f\tau-a/3")
        st.caption("s: untreated slope · f: control spillover fraction · a: anticipatory gain in two of six pre-periods. "
                   "Only the selected failure mechanism is enabled here.")
        if parameter:
            st.warning(explanation)
        else:
            st.info("At zero this mechanism is disabled; only measurement noise remains.")
        st.markdown(f"**Design response:** {mitigation}")
        st.caption("True post ATT compares the treated rollout with its no-rollout world, not with spillover-contaminated controls. "
                   "Anticipation changes two pre-periods; the post ATT remains the configured effect.")

with tabs[6]:
    if tabs[6].open:
        st.subheader("7 · Explore the same counterfactual in three dimensions")
        st.caption("Objective: locate the effect at the treated group and a selected time; 2D remains the primary teaching view.")
        period = st.slider("3D post period", 0, 5, 5, key="period_3d")
        truth_3d = st.toggle("Reveal synthetic Y(0) in 3D", key="truth_3d")
        chart(plot_3d(panel, period, truth_3d), "three_d", "Rotatable trajectories in time, group and productivity; green segment is the estimated counterfactual gap")
        data = trajectories(panel).loc[period]
        st.markdown(f"At period **{period}**, treated observed = **{data.treated:.2f}**, estimated untreated = "
                    f"**{data.estimated_cf:.2f}**, estimated gap = **{data.estimated_gap:+.2f} points**.")
        st.latex(r"\widehat{gap}_t=\bar Y_{Tt}-\widehat{Y}_{Tt}(0)")
        st.caption("Drag to rotate, use the toolbar to zoom, hover for values and click legend entries to toggle trajectories. "
                   "The group axis is categorical, not a continuous dose. A green estimated gap need not equal the true effect.")

st.divider()
st.caption("Synthetic educational laboratory · local computation only · interpretation review required · "
           "no measured production gains, cost or quality guardrails.")

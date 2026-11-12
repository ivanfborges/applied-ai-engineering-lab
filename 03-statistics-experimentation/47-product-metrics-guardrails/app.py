"""Run locally from this folder: streamlit run app.py."""

from dataclasses import asdict
from inspect import signature
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

from metrics import (binomial_interval, difference_interval, policy_recommendation,
                     rollout_policy, scorecard, zero_incident_upper_bound)
from simulation import (CONTROL, TREATMENT, System, cohort_heatmap, configurations,
                        repeated_experiments, retention_table, simpson_counts,
                        simulate, uncertainty_path)
from visual_charts import (COLORS, conversion_chart, economics_heatmap, finish,
                           latency_charts, rare_event_chart, repeated_chart,
                           retention_chart, safety_chart, tail_animation,
                           tradeoff_chart, uncertainty_chart)

TOPIC = Path(__file__).resolve().parent
MODULES = ["01 · Conversion and A/B testing", "02 · Retention and churn",
           "03 · Latency and tail behavior", "04 · Revenue and unit economics",
           "05 · Safety and rare events", "06 · 3D product trade-offs",
           "07 · Simulated rollout decision", "08 · Aggregation traps", "09 · Export GIFs"]
DEFAULTS = {"n": 3000, "seed": 47, "revenue": 2., "snapshot": 35, "detection": 1.,
            "critical_fraction": 0., "max_latency": 5., "max_cost_conversion": 1.2,
            "max_safety_delta": .003, "max_error": .02, "confidence": .95,
            "practical_pp": .5, "module": MODULES[0]}
for _arm, _cfg in (("control", CONTROL), ("treatment", TREATMENT)):
    DEFAULTS.update({f"{_arm}_{k}": v for k, v in asdict(_cfg).items()})


def reset():
    st.session_state.update(DEFAULTS)


def plot(fig, key):
    # Accessible chart names were added after the repository's minimum Streamlit version.
    kwargs = {"alt": fig.layout.title.text} if "alt" in signature(st.plotly_chart).parameters else {}
    st.plotly_chart(fig, key=key, width="stretch", config={"displaylogo": False}, **kwargs)


def explain(seeing, experiment, takeaway):
    st.markdown(f"**What am I seeing?** {seeing}")
    st.markdown(f"**What should I experiment with?** {experiment}")
    st.markdown(f"**Senior-level takeaway.** {takeaway}")


@st.cache_data(max_entries=3, show_spinner=False)
def cached_simulation(control, treatment, n, seed, revenue, snapshot, detection, critical_fraction):
    return simulate(control, treatment, n, seed, revenue, snapshot, detection, critical_fraction)


st.set_page_config(page_title="Day 47 · Product metrics lab", layout="wide")
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

with st.sidebar:
    st.title("Experiment controls")
    st.caption("Synthetic RAG assistant · one first request per randomized user")
    st.button("Reset to defaults", key="reset", on_click=reset)
    module = st.selectbox("Learning module", MODULES, key="module")
    st.slider("Users per variant", 100, 20000, step=100, key="n")
    st.number_input("Random seed", 0, 1000000, step=1, key="seed")
    st.number_input("Revenue per conversion (fictional units)", 0., 100., step=.5, key="revenue")
    for arm in ("control", "treatment"):
        with st.expander("Control · current production model" if arm == "control" else "Treatment · candidate AI model"):
            st.number_input("Conversion probability | request served", 0., 1., step=.01, key=f"{arm}_conversion")
            st.number_input("D7 exact-day activity probability", .01, .99, step=.01, key=f"{arm}_retention_d7")
            st.number_input("Request success probability", 0., 1., step=.005, key=f"{arm}_success", format="%.3f")
            st.number_input("Median first-attempt latency (s)", .1, 10., step=.1, key=f"{arm}_median")
            st.number_input("Lognormal tail variability (sigma)", .1, 1.5, step=.05, key=f"{arm}_sigma")
            st.number_input("Mean attributable cost / first request", .001, 1., step=.01, key=f"{arm}_cost", format="%.3f")
            st.number_input("Unsafe probability | request served", 0., .1, step=.001, key=f"{arm}_unsafe", format="%.4f")
            st.number_input("Daily subscription churn hazard", .001, .2, step=.005, key=f"{arm}_churn", format="%.3f")
    with st.expander("Observation and detection"):
        st.slider("Calendar snapshot day (entry on days 0–27)", 28, 65, key="snapshot")
        st.slider("Incident detection sensitivity (specificity = 1)", 0., 1., step=.05, key="detection")
        st.slider("Critical share of detected incidents", 0., .2, step=.01, key="critical_fraction")
    with st.expander("Illustrative rollout policy", expanded=True):
        st.number_input("Maximum p95 latency (s)", .1, 60., step=.5, key="max_latency")
        st.number_input("Maximum cost / successful conversion", .01, 20., step=.1, key="max_cost_conversion")
        st.number_input("Maximum safety degradation (absolute rate)", 0., .1, step=.001, key="max_safety_delta", format="%.4f")
        st.number_input("Maximum request error rate", 0., 1., step=.005, key="max_error", format="%.3f")
        st.slider("Family confidence level", .80, .99, step=.01, key="confidence")
        st.number_input("Minimum worthwhile conversion lift (pp)", 0., 20., step=.1, key="practical_pp")

systems = {arm: System(**{k: st.session_state[f"{arm}_{k}"] for k in asdict(CONTROL)})
           for arm in ("control", "treatment")}
c, t = systems["control"], systems["treatment"]
n, seed, revenue = st.session_state.n, st.session_state.seed, st.session_state.revenue
policy = {"latency": st.session_state.max_latency, "cost_conversion": st.session_state.max_cost_conversion,
          "safety_delta": st.session_state.max_safety_delta, "error": st.session_state.max_error,
          "confidence": st.session_state.confidence, "practical": st.session_state.practical_pp/100}

st.title("Product metrics & guardrails")
st.caption("DAY 47 · SYNTHETIC DATA ONLY · Blue: control · Orange: treatment · Interpretation: author review required")
st.info("A higher-converting model still needs acceptable latency, economics, reliability and safety. "
        "Every decision here is an illustrative simulation, never production release approval.")
users, histories = cached_simulation(c, t, n, seed, revenue, st.session_state.snapshot,
                                     st.session_state.detection, st.session_state.critical_fraction)
summary = scorecard(users)
pc, pt = [float(summary.loc[arm,"conversion"]) for arm in ("control", "treatment")]
truth = t.success*t.conversion-c.success*c.conversion
kc, kt = [int(summary.loc[arm,"converted_users"]) for arm in ("control", "treatment")]
ci = difference_interval(kc,n,kt,n)
st.session_state["observed_lift"] = pt-pc
st.session_state["sample_counts"] = users.groupby("arm").size().to_dict()
columns = st.columns(4)
columns[0].metric("Control conversion", f"{100*pc:.2f}%")
columns[1].metric("Treatment conversion", f"{100*pt:.2f}%")
columns[2].metric("Absolute lift", f"{100*(pt-pc):+.2f} pp")
columns[3].metric("Relative lift", f"{100*(pt-pc)/pc:+.1f}%" if pc else "Unavailable: zero baseline")
st.header(module)

if module == MODULES[0]:
    plot(conversion_chart(users), "conversion")
    st.write(f"Conservative 95% conversion-difference interval: [{100*ci[0]:.2f}, {100*ci[1]:.2f}] pp. "
             f"Generating marginal rates: {100*c.success*c.conversion:.2f}% / {100*t.success*t.conversion:.2f}%. "
             f"Minimum worthwhile effect: {policy['practical']*100:.2f} pp.")
    plot(repeated_chart(repeated_experiments(c,t,n,seed+101),truth), "repetitions")
    plot(uncertainty_chart(uncertainty_path(c,t,max(100,n),seed+202),truth), "uncertainty")
    st.caption("Play reveals a separate nested sample. Each displayed interval is valid at its own fixed horizon; "
               "repeatedly looking and stopping when favorable is not a valid sequential test.")
    explain("The funnel retains all assigned users; bars and lift intervals quantify sampling uncertainty.",
            "Set equal generating conversions, then reduce n. Next raise n and use Play on the sample path.",
            "Significance concerns uncertainty around zero. Practical significance concerns a worthwhile effect, "
            "with a prespecified horizon and policy. Neither establishes product value alone.")

elif module == MODULES[1]:
    table = retention_table(histories)
    plot(retention_chart(table), "retention")
    st.dataframe(table.loc[table.day.isin([1,7,14,30]),
                          ["arm","day","eligible","exact_activity","rolling_to_d30","subscription_survival","subscription_churn"]],
                 hide_index=True)
    cols=st.columns(2)
    for col,arm in zip(cols,("control","treatment")):
        heat=cohort_heatmap(histories[arm])
        fig=go.Figure(go.Heatmap(z=100*heat.to_numpy(),x=heat.columns,y=heat.index,colorscale="Cividis",zmin=0,zmax=100,
                                text=np.round(100*heat.to_numpy(),1),texttemplate="%{text}%",hoverongaps=False))
        with col:
            plot(finish(fig,f"{arm}: mature exact-day cohorts; blank means unavailable",height=330),f"heat-{arm}")
    selected = st.selectbox("Cohort timeline", ["control","treatment"], key="timeline_arm")
    h=histories[selected]
    fig=go.Figure(go.Scatter(x=np.arange(31),y=table.loc[table.arm==selected,"oracle_alive"],fill="tozeroy",name="Complete synthetic future"))
    plot(finish(fig,"Subscription members remaining — simulator oracle, not observed future", "Day", "Starting subscribers not yet canceled"),"leaving")
    sub=np.flatnonzero(h["subscriber"])[:100]
    observed=np.minimum(h["cancel_time"][sub],h["age"][sub])
    event=h["cancel_time"][sub]<=h["age"][sub]
    fig=go.Figure()
    for flag,label,symbol in ((True,"Observed cancellation","circle"),(False,"Right-censored","x")):
        mask=event==flag
        fig.add_trace(go.Scatter(x=observed[mask],y=np.arange(len(sub))[mask],mode="markers",name=label,marker_symbol=symbol))
    plot(finish(fig,"Observed subscription follow-up — first 100 starting subscribers", "Observed time (days)", "Subscriber index"),"censoring")
    explain("Exact-day activity uses [d,d+1); finite rolling return means any activity from d through D30 "
            "and needs 31 days of follow-up. KM estimates subscription survival with administrative censoring.",
            "Move the snapshot from day 28 to 65. Then raise subscription churn hazard and compare survival with exact-day activity.",
            "An active free-tier user may have canceled a subscription. Only subscription survival and its "
            "matching cancellation probability are complements. KM assumes noninformative censoring; it does not predict beyond observed support.")

elif module == MODULES[2]:
    threshold=st.slider("Interactive duration threshold (seconds)",.1,60.,float(policy["latency"]),.1,key="latency_threshold")
    log_latency=st.checkbox("Logarithmic latency axis",key="log_latency")
    for i,fig in enumerate(latency_charts(users,systems,threshold)):
        if log_latency:
            fig.update_xaxes(type="log")
        plot(fig,f"latency-{i}")
    rows=[]
    for arm,g in users.groupby("arm"):
        rows.append({"arm":arm, **{f"p{int(q*100)} (s)":np.quantile(g.latency_s,q,method="inverted_cdf") for q in (.5,.9,.95,.99)},
                     "exceed threshold":float((g.latency_s>threshold).mean()),"error rate":float((~g.served_ok).mean())})
    st.dataframe(pd.DataFrame(rows),hide_index=True)
    plot(tail_animation(t.median),"tail-animation")
    st.caption("All first-attempt end-to-end durations are fully observed, including failures. No timeout censoring, "
               "retries or time-to-first-token (TTFT) telemetry is simulated; TTFT is a separate streaming metric.")
    explain("Histograms estimate density; curves show the generating PDFs; the ECDF gives the fraction at or below a duration. "
            "p95 is the threshold for 95% of attempts, leaving about one in twenty above it; p99 leaves about one in a hundred.",
            "Give both arms the same median, increase treatment sigma, and press Play on the fixed-median tail animation.",
            "Mean or median latency can hide tail regressions. Define the latency population and monitor failures and deadline censoring separately.")

elif module == MODULES[3]:
    fields=["revenue_per_user","cost_per_request","cost_per_conversion","contribution_per_user"]
    economics=summary[fields].copy()
    economics["total_variable_cost"]=users.groupby("arm").cost.sum()
    st.dataframe(economics)
    delta=float(summary.loc["treatment","contribution_per_user"]-summary.loc["control","contribution_per_user"])
    st.metric("Observed incremental contribution / assigned user",f"{delta:+.4f}")
    baseline=c.success*c.conversion*revenue-c.cost
    plot(economics_heatmap(baseline,revenue,max(.25,1.5*t.cost)),"economics")
    target_conversion=(baseline+t.cost)/revenue if revenue else np.nan
    delta_p=t.success*t.conversion-c.success*c.conversion
    if revenue:
        st.write(f"At this revenue, treatment needs marginal conversion > {100*target_conversion:.2f}% to outperform control contribution "
                 "under the generating model. Values above 100% are infeasible.")
    else:
        st.write("Zero revenue: conversion cannot offset cost; compare cost alone.")
    if delta_p>0:
        st.write(f"If only revenue per conversion changes equally for both arms, break-even is {(t.cost-c.cost)/delta_p:.4f} "
                 "fictional units per conversion (a negative threshold means treatment already costs less).")
    else:
        st.write("No positive generating conversion advantage: the usual positive-lift break-even revenue formula is not applicable.")
    explain("The heatmap is population arithmetic: conversion × revenue minus mean cost. Magenta marks equal contribution to control.",
            "Raise treatment cost while holding conversion fixed; then vary revenue per conversion and watch the contour.",
            "This is contribution, not full profit. First-attempt cost is an attributed all-in variable budget, drawn Uniform[0.5,1.5] × mean. "
            "Real embeddings, retrieval, reranking, tools, retries and human review must be reconciled explicitly; fixed/shared costs are excluded.")

elif module == MODULES[4]:
    confidence=st.slider("Safety exercise confidence level",.80,.99,.95,.01,key="safety_confidence")
    plot(safety_chart(users,confidence),"safety")
    st.dataframe(users.groupby("arm")[["unsafe","unsafe_detected","critical_detected"]].sum().rename(
        columns={"unsafe":"Latent simulator incidents","unsafe_detected":"Detected incidents","critical_detected":"Detected critical incidents"}))
    p=st.number_input("Rare-event incident probability per eligible trial",0.,.05,.001,.0005,format="%.4f",key="rare_p")
    trials=st.slider("Independent safety evaluation trials",10,50000,1000,10,key="rare_n")
    detected_probability=p*st.session_state.detection
    count=int(np.random.default_rng(seed+303).binomial(trials,detected_probability))
    bounds=binomial_interval(count,trials,confidence)
    st.write(f"Separate fixed-seed sample: {count}/{trials} detected; two-sided interval [{100*bounds[0]:.4f}%, {100*bounds[1]:.4f}%]. "
             f"Probability of zero detected incidents: {(1-detected_probability)**trials:.4f}.")
    st.write(f"If zero are detected: exact one-sided upper = {100*zero_incident_upper_bound(trials,1-confidence):.4f}%. "
             "This conditional exercise is separate from the actual sample count.")
    plot(rare_event_chart(detected_probability,trials,confidence),"rare-events")
    explain("Bars estimate detected incidence per first request. Oracle labels are known only because we constructed the simulation. "
            "Detection has configurable sensitivity and assumes specificity one.",
            "Set a rare probability and small n, then increase n. Reduce detection sensitivity and compare detected versus latent counts.",
            "Independent trials and a trustworthy detector are assumptions. Correlated prompts reduce effective evidence. "
            "Automated classifiers, LLM judges and calibrated human review are distinct evidence sources. Critical security/privacy incidents can force a hard stop.")

elif module == MODULES[5]:
    table=configurations(c,t,revenue,policy,seed+404)
    compliant=st.checkbox("Show only configurations within population guardrails",key="only_feasible")
    shown=table.loc[table.feasible] if compliant else table
    scene,decision=st.columns([3,1])
    with scene:
        plot(tradeoff_chart(shown),"tradeoffs")
    with decision:
        st.subheader("Population constraints")
        for arm in ("control","treatment"):
            model=table.loc[table.name==arm].iloc[0]
            st.markdown(f"**{arm}: {'WITHIN LIMITS' if model.feasible else 'VIOLATES LIMITS'}**")
            st.write(model.reason)
        st.caption("Known generating values. Use section 07 for finite-sample evidence and uncertainty.")
    st.dataframe(table.loc[table.name.isin(["control","treatment"]),
                          ["name","conversion","p95","cost","unsafe","contribution","feasible","reason","pareto"]],hide_index=True)
    st.caption(f"{int(table.feasible.sum())}/{len(table)} configurations within population guardrails; "
               f"{int(table.pareto.sum())} in the feasible Pareto set. Symbols distinguish feasible circles from violating crosses; "
               "production/candidate are overlaid as diamond/square.")
    explain("X is marginal conversion, Y population p95, Z mean cost/request; color is population unsafe/request. "
            "Marker diameter = 8 + 15 × normalized contribution, always positive. These are generating-model values, not noisy experiment estimates.",
            "Tighten p95 or cost/conversion in the sidebar, enable the feasible-only toggle, then rotate and zoom the scene.",
            "Pareto dominance is checked only within the feasible finite set: maximize conversion and minimize p95, cost and safety incidence, "
            "with at least one strict improvement. Population feasibility is not a sample-based rollout conclusion; omitted objectives and untested models can change the frontier.")

elif module == MODULES[6]:
    injected=st.checkbox("Inject a confirmed critical security/privacy incident (teaching scenario)",key="inject_critical")
    critical=bool(users.loc[users.arm=="treatment","critical_detected"].any()) or injected
    checks=rollout_policy(users,policy,t.cost,st.session_state.detection,critical)
    recommendation=policy_recommendation(checks)
    st.session_state["decision_status"]=recommendation
    st.subheader(recommendation)
    st.dataframe(checks,hide_index=True)
    st.caption("FAIL, PASS and INCONCLUSIVE are text labels. Five statistical checks share a Bonferroni family error budget; "
               "the confirmed-critical check is a hard stop. All bounds assume this fixed-horizon synthetic design.")
    st.warning("Illustrative policy only. Changing thresholds after observing outcomes explores the policy; "
               "it is not a preregistered production experiment or release authorization.")
    explain("The primary bound must exceed the practical lift. Latency tests P(duration > maximum) ≤ 5%, equivalent to a continuous-population "
            "p95 constraint. Economics bounds mean cost / conversion probability; reliability and safety use exact binomial building blocks.",
            "Increase n with all else fixed, relax one boundary, then inject a critical incident. Also reduce detection below one.",
            "A nonsignificant harm is not a pass. Conservative bounds may leave a decision inconclusive; perfect synthetic detection and "
            "known bounded costs do not transfer automatically to real traffic. Inspect every failed or unresolved check.")

elif module == MODULES[7]:
    st.subheader("A · Deliberately constructed Simpson reversal")
    share_c=st.slider("Easy-segment share: control",.1,.9,.2,.1,key="share_c")
    share_t=st.slider("Easy-segment share: treatment",.1,.9,.8,.1,key="share_t")
    counts=simpson_counts(share_c,share_t)
    st.dataframe(counts,hide_index=True)
    aggregate=counts.groupby("arm")[["converted","eligible"]].sum()
    aggregate["rate"]=aggregate.converted/aggregate.eligible
    st.dataframe(aggregate)
    fig=go.Figure()
    for arm,color in COLORS.items():
        g=counts.loc[counts.arm==arm]
        fig.add_trace(go.Bar(x=list(g.segment)+["aggregate"],y=list(100*g.rate)+[100*aggregate.loc[arm,"rate"]],name=arm,marker_color=color))
    plot(finish(fig,"Different mixtures can reverse the aggregate comparison", "Segment", "Conversion (%)"),"simpson")
    explain("Transparent counts use worse treatment rates in both segments but different segment mixtures.",
            "Make segment shares equal; compare the aggregate with both segment effects.",
            "This is a constructed selection example, separate from the randomized main experiment. It does not imply that randomization automatically creates Simpson's paradox.")
    st.subheader("B · User weighting versus request weighting")
    volume=st.slider("Requests from the high-activity user",1,100,10,key="request_volume")
    users_demo=pd.DataFrame({"user":["high activity","low activity"],"requests":[volume,1],"successes":[round(.9*volume),0]})
    users_demo["user_success_rate"]=users_demo.successes/users_demo.requests
    st.dataframe(users_demo,hide_index=True)
    user_rate=float(users_demo.user_success_rate.mean())
    request_rate=float(users_demo.successes.sum()/users_demo.requests.sum())
    plot(finish(go.Figure(go.Bar(x=["Equal user weights","Equal request weights"],y=[100*user_rate,100*request_rate],
                                 marker_color=[COLORS["control"],COLORS["treatment"]])),
                "The same outcomes, different implicit weights", y="Success (%)"),"weights")
    explain("One user has many requests at approximately 90% success; the other has one failure. Integer success counts are shown.",
            "Change high-user request volume while keeping the low-activity user's single failure.",
            "Equal-user and request-weighted means answer different questions. Neither grants independent-request standard errors in a user-randomized test.")

else:
    st.write("Exports use the current arms, revenue, seed and sample size. They run only after an explicit button press or CLI command.")
    st.caption("Two GIFs, 24 frames each by default, 900 × 500 pixels; saved under ignored outputs/. No automatic public preview selection.")
    if st.button("Generate both educational GIFs",key="export_gifs"):
        from animations import export_gifs
        with st.spinner("Rendering two bounded animations locally…"):
            paths=export_gifs(TOPIC/"outputs",c,t,n,seed,revenue)
        st.session_state["export_paths"]=[str(p) for p in paths]
    if "export_paths" in st.session_state:
        st.success("Artifacts generated on explicit request. Review labels and interpretations before selecting previews.")
        st.code("\n".join(st.session_state.export_paths))
    explain("Latency keeps median fixed while spread grows. The uncertainty GIF reveals a single nested Bernoulli path around a fixed true effect.",
            "Export once with equal medians but different tail spreads; then use a larger n for the uncertainty endpoint.",
            "Individual estimates fluctuate. Intervals at repeatedly inspected horizons do not justify optional stopping; GIFs are teaching artifacts.")

with st.expander("Generating parameters, observed scorecard and reproducibility"):
    st.dataframe(pd.DataFrame({arm:asdict(cfg) for arm,cfg in systems.items()}))
    st.dataframe(summary.rename(columns={"unsafe_incidents":"latent_simulator_incidents",
                                          "unsafe_per_request":"latent_simulator_incidence"}))
    st.caption(f"Seed {seed}; {n:,} independent users per arm. Assignments are exactly balanced and randomly permuted. "
               "All users generate one first request; no post-treatment exclusions. This differs from the historical textual study generator.")

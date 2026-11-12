"""Numerical contracts for the visual lab; no rendering or performance claims."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from metrics import (binomial_interval, bounded_mean_interval, classify_bound, cost_per_outcome,
                     difference_interval, pareto_mask, policy_recommendation, rollout_policy, scorecard)
from simulation import (CONTROL, TREATMENT, System, cohort_heatmap, configurations, kaplan_meier,
                        retention_table, simulate, simpson_counts, uncertainty_path)
from visual_charts import (conversion_chart, economics_heatmap, latency_charts, rare_event_chart,
                           retention_chart, safety_chart, tail_animation, tradeoff_chart, uncertainty_chart)

POLICY={"latency":5.,"cost_conversion":1.2,"safety_delta":.003,"error":.02,"confidence":.95,"practical":.005}


@pytest.mark.parametrize("k,n",[(0,10),(1,1000),(50,100),(100,100)])
def test_exact_binomial_intervals_match_scipy(k,n):
    reference=binomtest(k,n).proportion_ci(.95,method="exact")
    assert binomial_interval(k,n)==pytest.approx([reference.low,reference.high])


def test_difference_bounds_work_at_boundaries_and_swap_arms():
    low,high=difference_interval(0,100,0,100)
    assert low<0<high
    a=difference_interval(5,100,30,200)
    b=difference_interval(30,200,5,100)
    assert a==pytest.approx([-b[1],-b[0]])


def test_known_cost_support_not_observed_range_and_zero_outcomes():
    assert np.isnan(cost_per_outcome(3.,0))
    assert cost_per_outcome(8.,4)==2
    small=bounded_mean_interval([.1]*100,.05,.15)
    large=bounded_mean_interval([.1]*1000,.05,.15)
    assert small[0]<large[0]<.1<large[1]<small[1]
    with pytest.raises(ValueError):
        bounded_mean_interval([.2],.05,.15)


@pytest.mark.parametrize("low,high,threshold,higher,expected",[
    (.01,.03,.04,False,"PASS"),(.01,.05,.04,False,"INCONCLUSIVE"),
    (.05,.08,.04,False,"FAIL"),(.01,.03,0,True,"PASS"),
    (-.01,.03,0,True,"INCONCLUSIVE"),(-.03,-.01,0,True,"FAIL"),
    (np.nan,np.nan,0,True,"INCONCLUSIVE")])
def test_three_state_bound_decisions(low,high,threshold,higher,expected):
    assert classify_bound(low,high,threshold,higher)==expected


def test_reproducible_assignments_denominators_and_counterfactual_streams():
    a,ha=simulate(n=200,seed=47)
    b,hb=simulate(n=200,seed=47)
    pd.testing.assert_frame_equal(a,b)
    assert a.user_id.is_unique
    assert a.groupby("arm").size().tolist()==[200,200]
    assert scorecard(a).assigned_users.tolist()==[200,200]
    changed,_=simulate(treatment=replace(TREATMENT,median=4),n=200,seed=47)
    pd.testing.assert_frame_equal(a.loc[a.arm=="control"],changed.loc[changed.arm=="control"])
    assert np.array_equal(ha["treatment"]["activity"],hb["treatment"]["activity"])


def test_coherent_activity_and_administrative_censoring():
    users,h=simulate(n=300,seed=47,snapshot=28)
    for arm,history in h.items():
        assert np.all(history["activity"][:,0])
        days=np.arange(31)
        assert not np.any(history["activity"][:,1:] & (history["engagement_exit"][:,None]<=days)[:,1:])
        assert np.all(history["rolling"][:,:-1]>=history["rolling"][:,1:])
        assert np.all(history["rolling"]>=history["activity"])
        assert np.isnan(cohort_heatmap(history)["D30"]).all()
    table=retention_table(h)
    assert table.loc[table.day==30,"exact_activity"].isna().all()
    assert users.returned_d7.notna().equals(users.followup_days>=8)
    for arm,g in table.groupby("arm"):
        valid=g.subscription_survival.dropna()
        assert (np.diff(valid)<=0).all()
        assert (g.subscription_survival+g.subscription_churn).dropna().to_numpy()==pytest.approx(1.)


def test_km_known_censoring_example_no_extrapolation():
    result=kaplan_meier([1,2,3,4],[True,False,True,False],[0,1,2,3,4,5])
    assert result[:5]==pytest.approx([1,.75,.75,.375,.375])
    assert np.isnan(result[5])


def test_d7_calibration_is_mathematical_not_an_independent_daily_probability():
    for p7 in (.01,.38,.99):
        frequency=(1+p7)/2
        hazard=-np.log(p7/frequency)/7
        assert frequency*np.exp(-hazard*7)==pytest.approx(p7)


def test_pareto_dominance_ties_and_feasible_subset():
    assert pareto_mask([[1,2],[2,3],[1,2],[3,1]]).tolist()==[True,False,True,True]
    table=configurations(CONTROL,TREATMENT,2.,POLICY,47)
    feasible=table.loc[table.feasible]
    assert not table.loc[~table.feasible,"pareto"].any()
    objective=feasible[["conversion","p95","cost","unsafe"]].to_numpy(copy=True)
    objective[:,0]*=-1
    assert feasible.pareto.tolist()==pareto_mask(objective).tolist()
    assert table["size"].between(8,23).all()


def test_policy_pass_fail_inconclusive_detection_and_hard_stop():
    good=System(conversion=.5,success=.999,median=.2,sigma=.1,cost=.01,unsafe=0)
    baseline=replace(CONTROL,conversion=.05,unsafe=0)
    users,_=simulate(baseline,good,n=10000)
    permissive={**POLICY,"safety_delta":.01}
    checks=rollout_policy(users,permissive,good.cost)
    assert set(checks.status)=={"PASS"}
    assert policy_recommendation(checks)=="Candidate meets the simulated rollout criteria"
    critical=rollout_policy(users,permissive,good.cost,confirmed_critical=True)
    assert critical.iloc[-1].status=="FAIL"
    assert "violates" in policy_recommendation(critical)
    imperfect=rollout_policy(users,permissive,good.cost,detection=.5)
    assert imperfect.loc[imperfect.check=="Safety incidence degradation","status"].iloc[0]=="INCONCLUSIVE"
    assert "Insufficient evidence" in policy_recommendation(imperfect)
    invalid={**POLICY,"confidence":1.}
    with pytest.raises(ValueError):
        rollout_policy(users,invalid,good.cost)


def test_no_conversions_does_not_crash_or_pass_economics():
    zero=replace(TREATMENT,conversion=0)
    users,_=simulate(treatment=zero,n=100)
    checks=rollout_policy(users,POLICY,zero.cost)
    row=checks.loc[checks.check=="Cost per conversion"].iloc[0]
    assert np.isnan(row.observed)
    assert np.isinf(row.upper)
    assert row.status!="PASS"


def test_nested_path_and_transparent_simpson_counts():
    path=uncertainty_path(CONTROL,TREATMENT,3000,47)
    pd.testing.assert_frame_equal(path,uncertainty_path(CONTROL,TREATMENT,3000,47))
    assert np.all(np.diff(path.n)>0)
    assert ((path.lift>=path.low)&(path.lift<=path.high)).all()
    counts=simpson_counts()
    assert counts.groupby("arm").eligible.sum().tolist()==[1000,1000]
    rates=counts.groupby("arm")[["converted","eligible"]].sum()
    assert rates.loc["treatment","converted"]>rates.loc["control","converted"]
    for _,g in counts.groupby("segment"):
        assert g.loc[g.arm=="treatment","rate"].iloc[0]<g.loc[g.arm=="control","rate"].iloc[0]


def test_all_plotly_figures_serialize_with_censored_and_zero_event_data():
    users,h=simulate(n=100, snapshot=28)
    figures=[conversion_chart(users),retention_chart(retention_table(h)),
             *latency_charts(users,{"control":CONTROL,"treatment":TREATMENT},5),
             economics_heatmap(.2,0),safety_chart(users,.95),rare_event_chart(0,1000,.95),
             tail_animation(),uncertainty_chart(uncertainty_path(CONTROL,TREATMENT,1000,47),.03),
             tradeoff_chart(configurations(CONTROL,TREATMENT,2,POLICY,47))]
    for fig in figures:
        assert len(fig.to_json())>100


@pytest.mark.parametrize("kwargs",[{"conversion":1.1},{"retention_d7":0},{"cost":0},{"sigma":np.nan}])
def test_invalid_system_configuration(kwargs):
    with pytest.raises(ValueError):
        replace(CONTROL,**kwargs)

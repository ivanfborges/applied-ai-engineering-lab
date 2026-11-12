"""Independent numerical references and bounded rendering checks for the visual lab."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image
from scipy.optimize import brentq
from scipy.stats import chi2_contingency, ttest_ind

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import ab_visual_core as core
import visualize_ab_testing as visual
from from_scratch import two_proportion_test, sample_size_per_arm


def test_binary_core_agrees_with_independent_pearson_and_previous_implementation():
    xa=np.array([100,1500,400]);na=np.array([1000,15000,900])
    xb=np.array([120,1650,500]);nb=np.array([1100,15000,1100])
    d=core.binary_inference(xa,na,xb,nb)
    for i in range(3):
        reference=chi2_contingency([[xa[i],na[i]-xa[i]],[xb[i],nb[i]-xb[i]]],correction=False)
        old=two_proportion_test(xa[i],na[i],xb[i],nb[i])
        assert d['z'][i]**2==pytest.approx(reference.statistic)
        assert d['p_value'][i]==pytest.approx(reference.pvalue,rel=1e-10)
        assert d['low'][i]==pytest.approx(old.ci_low)
        assert d['high'][i]==pytest.approx(old.ci_high)
        se=np.sqrt((xa[i]/na[i])*(1-xa[i]/na[i])/na[i]+(xb[i]/nb[i])*(1-xb[i]/nb[i])/nb[i])
        assert d['se'][i]==pytest.approx(se)


@pytest.mark.parametrize('args',[(0,100,0,100),(100,100,100,100),(-1,100,10,100),(110,100,10,100),(10.,100,12,100),(10,0,10,100)])
def test_invalid_or_sparse_binary_inputs_fail(args):
    with pytest.raises(ValueError):core.binary_inference(*args)


@pytest.mark.parametrize('kwargs',[{'seed':-1},{'n_per_arm':999},{'repetitions':0},{'repetitions':20001},{'baseline':0},{'alpha':1},{'treatment':float('nan')}])
def test_configuration_contract(kwargs):
    with pytest.raises(ValueError):core.LabConfig(**kwargs)


def test_streams_are_reproducible_and_independent_of_module_execution_order():
    c=core.LabConfig(repetitions=100)
    a=core.trials(c,module=3)
    core.trials(c,module=8)
    b=core.trials(c,module=3)
    np.testing.assert_array_equal(a['difference'],b['difference'])
    assert not np.array_equal(a['x_a'],core.trials(c,module=3,scenario=1)['x_a'])
    assert not np.array_equal(a['x_a'],core.trials(replace(c,seed=46),module=3)['x_a'])


def test_potential_outcomes_are_coherent_and_monotone_in_this_declared_dgp():
    d=core.population(core.LabConfig(),n=10000)
    np.testing.assert_array_equal(d['observed'],np.where(d['assignment']==1,d['y1'],d['y0']))
    assert (d['y1']>=d['y0']).all()
    assert d['finite_ate']==pytest.approx((d['y1']-d['y0']).mean())
    assert d['expected_ate']==pytest.approx((d['p1']-d['p0']).mean())


def test_aa_false_positive_rate_and_wald_coverage_with_mc_uncertainty():
    c=core.LabConfig(repetitions=12000)
    d=core.trials(c,p_b=c.baseline,module=8)
    rate=core.rate_summary(d['p_value']<c.alpha)
    assert abs(rate['rate']-.05)<5*np.sqrt(.05*.95/c.repetitions)
    coverage=((d['low']<=0)&(d['high']>=0)).mean()
    assert abs(coverage-.95)<5*np.sqrt(.95*.05/c.repetitions)
    assert rate['mc_ci95'][0]<rate['rate']<rate['mc_ci95'][1]


def test_power_matches_simulation_and_independent_normal_tail_inversion():
    c=core.LabConfig(repetitions=12000)
    analytical=float(core.analytical_power(15000,.01))
    actual=(core.trials(c,module=5)['p_value']<.05).mean()
    assert abs(actual-analytical)<5*np.sqrt(analytical*(1-analytical)/c.repetitions)
    assert core.analytical_power(15000,0)==pytest.approx(.05)
    assert core.analytical_power(5000,.01)<analytical
    assert core.analytical_power(15000,.005)<analytical
    n=brentq(lambda n:float(core.analytical_power(n,.01))-.8,100,100000)
    assert abs(sample_size_per_arm(.1,.11)-n)<1
    assert core.planning_sizes(.1,[.005])[0]>core.planning_sizes(.1,[.01])[0]


def test_peeking_compares_paired_experiments_with_same_horizon():
    c=core.LabConfig(repetitions=2000)
    d=core.peeking(c);p=d['p_values']
    assert d['looks'][-1]==c.n_per_arm
    assert d['fixed']['count']==int((p[:,-1]<c.alpha).sum())
    assert d['naive']['count']==int((p<c.alpha).any(1).sum())
    assert d['naive']['count']>=d['fixed']['count']
    expected=np.where((p<c.alpha).any(1),(p<c.alpha).argmax(1),p.shape[1]-1)
    np.testing.assert_array_equal(d['stopped_n'],d['looks'][expected])


def test_srm_uses_intended_allocation_not_assumption_of_balance():
    healthy,bug,configured=core.allocation_cases(core.LabConfig())
    assert healthy['p_value']>.001
    assert bug['p_value']<.001
    assert configured['n_b']==bug['n_b']
    assert configured['p_value']>.001


def test_user_mean_welch_reference_and_equal_event_estimand():
    c=core.LabConfig(repetitions=500)
    d=core.clustered_trials(c,rho=.2)
    a,b=d['counts'][0]/20,d['counts'][1]/20
    trusted=ttest_ind(b,a,equal_var=False)
    assert d['user_first']['p_value']==pytest.approx(trusted.pvalue)
    assert d['user_first']['low']==pytest.approx(trusted.confidence_interval().low)
    assert d['user_first']['high']==pytest.approx(trusted.confidence_interval().high)
    assert d['naive_first']['difference']==pytest.approx(d['user_first']['difference'])
    assert d['design_effect']==pytest.approx(4.8)
    assert d['user_first']['se']>d['naive_first']['se']


def test_beta_binomial_variance_matches_declared_icc_model():
    c=core.LabConfig(repetitions=100)
    d=core.clustered_trials(c,rho=.2,users_per_arm=10000,repetitions=1)
    observed=(d['counts']/20).var()
    expected=c.baseline*(1-c.baseline)/20*(1+19*.2)
    assert observed==pytest.approx(expected,rel=.06)


def test_synthetic_product_policy_does_not_equate_significance_with_launch():
    d=core.product_experiment(core.LabConfig())
    assert d['gates']['cost']==(d['metrics']['cost_b']-d['metrics']['cost_a']<=d['policy']['max_cost_increment'])
    assert d['gates']['safety']==(d['metrics']['safety_upper_b']<=d['policy']['max_safety_upper'])
    assert d['metrics']['safety_upper_b']>=d['metrics']['safety_b']
    assert d['status']!='Ready for production'
    assert not all(d['gates'].values())


def test_power_surface_html_and_png_in_temporary_directory(tmp_path):
    visual.style()
    record=visual.render_module('power_surface',core.LabConfig(repetitions=100),tmp_path,quality='light',gifs=False)
    with Image.open(tmp_path/'06_power_surface.png') as png:assert png.width>=1000
    html=(tmp_path/'06_power_surface.html').read_text(encoding='utf-8')
    assert 'cdn.plot.ly' not in html.split('<script>')[0]
    assert 'labScenarios' in html and '80% power contour' in html
    assert 'surface' in html and record['scenario_count']==1


def test_animation_contains_distinct_frames_and_reported_scenarios(tmp_path):
    visual.style()
    record=visual.render_module('aa_testing',core.LabConfig(repetitions=100),tmp_path,quality='light')
    with Image.open(tmp_path/'08_aa_testing.gif') as image:
        assert image.n_frames>1
        image.seek(0);first=np.array(image.convert('RGB'))
        image.seek(image.n_frames-1);last=np.array(image.convert('RGB'))
        assert not np.array_equal(first,last)
    assert record['gif_frames']>1


def test_cli_rejects_existing_run_without_overwriting(tmp_path):
    marker=tmp_path/'existing.txt';marker.write_text('preserve',encoding='utf-8')
    with pytest.raises(SystemExit):visual.main(['--quick','--output',str(tmp_path)])
    assert marker.read_text()=='preserve'


def test_negative_effect_is_visible_on_the_z_axis():
    visual.style()
    fig,scenarios,report,_=visual.sampling_distributions(core.LabConfig(
        repetitions=100,baseline=.2,treatment=.1))
    try:
        observed=report['first_observed']['z']
        assert observed < -5
        for _,figure in scenarios:
            low,high=figure.layout.xaxis2.range
            assert low <= observed <= high
            assert low < 0 < high
    finally:
        visual.plt.close(fig)


def test_confidence_level_label_follows_alpha():
    visual.style()
    fig,scenarios,_,_=visual.confidence_intervals(core.LabConfig(repetitions=100,alpha=.01))
    try:
        assert '99%' in fig.axes[0].get_xlabel()
        assert all('99%' in p.layout.xaxis.title.text for _,p in scenarios)
    finally:
        visual.plt.close(fig)


def test_cluster_batches_keep_every_repetition():
    c=core.LabConfig(repetitions=257)
    result=core.clustered_trials(c)
    assert result['batch_size']==128
    assert result['naive']['repetitions']==257
    assert result['user']['repetitions']==257
    assert result['user']==core.clustered_trials(c)['user']


def test_observed_only_view_does_not_reveal_oracle_effects():
    visual.style()
    fig,scenarios,_,_=visual.potential_outcomes(core.LabConfig(repetitions=100))
    try:
        oracle,observed=scenarios[0][1],scenarios[1][1]
        assert len(oracle.data[-1].x)==3
        assert list(observed.data[-1].x)==['Randomized estimate']
        for trace in observed.data[:-1]:
            assert 'Individual effect' not in trace.hovertemplate
            assert np.asarray(trace.customdata).shape[1]==1
    finally:
        visual.plt.close(fig)

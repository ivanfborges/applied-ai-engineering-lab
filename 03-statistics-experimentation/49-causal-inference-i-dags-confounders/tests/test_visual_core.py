"""SCM invariants, external graph agreement and bounded rendering checks."""

from html.parser import HTMLParser
import json
from pathlib import Path
import sys

import networkx as nx
import numpy as np
import pytest
from PIL import Image
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from causal_visual_core import (ADJUSTMENTS, MODULES, bias_study, collider_selection,
    graph_payload, model, population_values, random_tape, residuals, slope, summary,
    validate_graph_algorithm)
from example import simulate_scm, study_graph
from from_scratch import DAG
from visualize_causal_inference import generate, gif_frames, main


def test_exact_eight_edges_and_networkx_agreement():
    graph = graph_payload()
    expected = {('C','A'),('C','Y'),('A','Y'),('A','M'),('M','Y'),('A','S'),('U','S'),('U','Y')}
    assert set(map(tuple, graph['edges'])) == expected
    assert nx.is_directed_acyclic_graph(nx.DiGraph(graph['edges']))
    assert validate_graph_algorithm() == 240
    assert graph['cases']['C']['backdoor']
    for name in ('None','C + M','C + S'):
        assert not graph['cases'][name]['backdoor']
    collider = next(p for p in graph['cases']['None']['paths'] if 'S' in p['nodes'])
    assert not collider['active']
    assert next(p for p in graph['cases']['C + S']['paths'] if 'S' in p['nodes'])['active']


def test_collider_descendant_opens_path_against_networkx():
    edges = [('A','S'),('U','S'),('S','D'),('U','Y')]
    graph, reference = DAG('ASUDY', edges), nx.DiGraph(edges)
    for z, expected in [(set(),True),({'D'},False),({'D','U'},True)]:
        assert graph.is_d_separated('A','Y',z) == expected
        assert nx.is_d_separator(reference,{'A'},{'Y'},z) == expected


def test_baseline_population_and_same_draws_as_historical_model():
    values = population_values()
    assert values == {'total':2.,'direct':1.3,'indirect':.7,'naive':2.9,'collider':1.04}
    d = model(random_tape(1500,49))
    historical = simulate_scm(1500,49)
    for name in historical:
        np.testing.assert_allclose(d[name],historical[name],atol=1e-14)


@pytest.mark.parametrize('alpha,beta,confounding,noise',[(.7,1,1.5,.5),(-.4,1.5,-1,1.2),(0,0,0,.1)])
def test_modified_equations_potential_outcomes_and_fwl(alpha,beta,confounding,noise):
    d = model(random_tape(8000,52),alpha=alpha,beta=beta,confounding=confounding,noise=noise)
    values = population_values(alpha,beta,confounding,noise)
    np.testing.assert_allclose(d.Y1-d.Y0,values['total'],atol=1e-12)
    np.testing.assert_allclose(d.Y,np.where(d.A==1,d.Y1,d.Y0),atol=1e-12)
    _,_,fwl = residuals(d)
    expected = LinearRegression().fit(d[['A','C']],d.Y).coef_[0]
    assert fwl == pytest.approx(expected,abs=1e-12)
    for name, controls in ADJUSTMENTS.items():
        expected = LinearRegression().fit(d[['A',*controls]],d.Y).coef_[0]
        assert slope(d,controls) == pytest.approx(expected,abs=1e-11)


def test_large_sample_approaches_analytical_coefficients():
    s = summary(model(random_tape(80_000,490)))
    for name,target in [('None',2.9),('C',2),('C + M',1.3),('C + S',1.04),('Standardized C',2)]:
        assert s['estimates'][name] == pytest.approx(target,abs=.05)


def test_collider_dependence_selection_and_noise_do_not_change_outcome():
    tape=random_tape(35_000,49)
    d=model(tape)
    result=collider_selection(d,center=.5,width=.15)
    assert abs(result['delta_u_full'])<.06
    assert result['delta_u_selected']<-.65
    assert result['full_C']==pytest.approx(2,abs=.08)
    assert result['full_CS']<1.2
    noisy=model(tape,noise=1.5)
    np.testing.assert_array_equal(d.Y,noisy.Y)
    assert not np.array_equal(d.S,noisy.S)
    high=collider_selection(d,center=.5,width=.15,mode='high')
    assert high['n']>result['n']
    assert high['delta_u_selected']!=pytest.approx(result['delta_u_selected'],abs=.1)
    empty=collider_selection(d,center=100,width=.05)
    assert empty['n']==0 and empty['selected_C'] is None and empty['delta_u_selected'] is None


def test_reproducible_bias_study_sampling_sd_and_mcse():
    first=bias_study(49,sizes=(200,1800),repetitions=12)
    assert first==bias_study(49,sizes=(200,1800),repetitions=12)
    for record in first:
        assert record['sd']>0
        assert record['mcse']==pytest.approx(record['sd']/np.sqrt(12))
        assert record['low']<record['high']
    for estimator in ('None','C','C + S'):
        records=[r for r in first if r['estimator']==estimator]
        assert records[1]['sd']<records[0]['sd']
    assert next(r['mean'] for r in first if r['n']==1800 and r['estimator']=='None')>2.7


@pytest.mark.parametrize('kwargs',[{'noise':-1},{'alpha':np.nan},{'n':1001},{'n':10},{'n':100.5}])
def test_invalid_model_parameters(kwargs):
    with pytest.raises(ValueError): model(random_tape(1000,49),**kwargs)


@pytest.mark.parametrize('n,seed',[(10,49),(100,-1),(True,49),(100,1.5)])
def test_invalid_tape_configuration(n,seed):
    with pytest.raises(ValueError): random_tape(n,seed)


class Resources(HTMLParser):
    def __init__(self):
        super().__init__(); self.remote=[]
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ('src','href') and value and value.startswith(('https:','http:','//')):
                self.remote.append(value)


def test_offline_html_smoke_and_preserve_existing_runs(tmp_path):
    report=generate(tmp_path/'lab',seed=49,fast=True,modules=('collider',),dashboard=True,gifs=False)
    assert report['graph_comparisons']==240
    file=tmp_path/'lab/collider.html'
    content=file.read_text(encoding='utf-8')
    parser=Resources(); parser.feed(content)
    assert not parser.remote
    assert 'plotly.js' in content and 'CausalMath' in content
    for id_ in ('noise','width','center','mode','seed','n'):
        assert f'id="{id_}"' in content
    assert json.loads((tmp_path/'lab/run.json').read_text())['population']['total']==2
    with pytest.raises(ValueError,match='nonempty'):
        generate(tmp_path/'lab',fast=True,modules=('collider',),gifs=False)


@pytest.mark.parametrize('kind',['confounding','collider','mediation'])
def test_gif_frames_are_real_and_save_to_temporary_directory(kind,tmp_path):
    frames=gif_frames(kind,model(random_tape(600,49)),6)
    assert len(frames)==6 and len({f.tobytes() for f in frames})>=4
    file=tmp_path/f'{kind}.gif'
    frames[0].save(file,save_all=True,append_images=frames[1:],duration=600,loop=0,disposal=2)
    with Image.open(file) as image:
        assert image.n_frames==6
        assert image.width>=1000
    for frame in frames: frame.close()


@pytest.mark.parametrize('args',[['--seed','-1'],['--repetitions','2'],['--module','unknown']])
def test_cli_rejects_invalid_inputs(args):
    with pytest.raises(SystemExit) as error: main(args)
    assert error.value.code==2

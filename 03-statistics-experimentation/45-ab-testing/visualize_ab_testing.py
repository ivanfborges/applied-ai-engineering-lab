"""Offline A/B testing visual learning lab: 12 mechanisms, no server required."""
from __future__ import annotations
import argparse
import os
from dataclasses import asdict, replace
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import platform
import textwrap
import time

# Keep the standalone renderer from starting a BLAS thread pool per operation.
for _thread_variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_thread_variable] = '1'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs
from plotly.utils import PlotlyJSONEncoder
import scipy
from scipy.stats import norm
from PIL import Image

from ab_visual_core import (LabConfig, allocation_cases, analytical_power, binary_inference,
    clustered_trials, peeking, planning_sizes, population, product_experiment, rate_summary, trials)

BLUE, ORANGE, TEAL, PURPLE = '#0072B2', '#D55E00', '#009E73', '#AA4499'
NAVY, GRAY = '#15283F', '#8593A3'
MODULES = ('randomization','potential_outcomes','sampling_distributions','confidence_intervals',
           'power','power_surface','errors','aa_testing','peeking','srm','unit_of_analysis','ai_product')
QUESTIONS = (
    'What does random assignment balance?', 'Which outcomes can an experiment observe?',
    'How does a sampling distribution become a test?', 'What does interval coverage mean?',
    'How much data does a smaller effect need?', 'Where does a design reach 80% power?',
    'Which errors change with the threshold?', 'How often does no effect look significant?',
    'What happens when we stop on significance?', 'Does allocation match the intended design?',
    'How many independent observations do we have?', 'Is a significant AI workflow ready to launch?')
LESSONS = (
    'Random assignment balances covariates in expectation. A realized imbalance is possible; observed balance cannot prove valid assignment.',
    'Both potential outcomes are simulation-only oracle information. A real experiment reveals only Y(Z). This DGP assumes monotone benefit.',
    'Effect estimates have percentage-point units; Z statistics do not. Only the Z panel uses the pooled-test rejection rule.',
    'Coverage describes repeated intervals containing a fixed effect. It is not a posterior probability for this particular interval.',
    'Lines are normal planning approximations. Dots and their intervals are Monte Carlo rejection frequencies, not exact power.',
    'Analytical normal approximation with independent users, balanced arms and one two-sided final analysis. Rotation changes the view, not the model.',
    'Normal sampling models, not posterior probabilities. At fixed separation, lowering alpha trades fewer false positives for lower power.',
    'A/A p-values are approximately uniform here. The empirical rejection frequency has Monte Carlo uncertainty; no rate is forced to equal alpha.',
    'Both policies see the same null experiments and maximum horizon. Naive peeking reuses a fixed-horizon test; valid sequential designs need other methods.',
    'A 40/60 split is healthy if configured as 40/60. SRM compares observed counts with the intended allocation and triggers investigation.',
    'Assignment is by user. Beta-binomial events share a user risk; equal events per user keep both estimators on the same scale. Welch uses independent user means.',
    'Synthetic primary, operational and diagnostic metrics are distinct. Predefined illustrative gates support human review; none proves production readiness.')
HYPOTHESES = (
    'Across random allocations of a fixed population, standardized baseline mean differences center near zero.',
    'An observed randomized mean difference can differ from the finite-population ATE; only assigned outcomes are observed.',
    'Independent effect estimates concentrate around their generating difference, while pooled Z supports a null rejection rule.',
    'Regular Wald intervals have approximately nominal repeated coverage and shrink with more independent users.',
    'Detection frequency increases with sample size and effect size; normal planning approximately tracks Monte Carlo power.',
    'The normal power model has a visible sample-size/effect boundary at 80% power.',
    'At fixed separation, changing alpha trades Type I error against Type II error in the declared normal model.',
    'Under the ideal null generator, final-test rejection frequency is near the configured alpha, with Monte Carlo variation.',
    'Any-crossing rejection is at least as frequent as final-only rejection on the same null experiment paths.',
    'An unintended allocation mismatch can trigger SRM while an intentionally configured unequal allocation need not.',
    'Ignoring positive within-user correlation understates uncertainty; independent user-mean inference addresses this model.',
    'A synthetic primary-metric improvement can coexist with practical-value or operational guardrail failures.')


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
        'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,
        'axes.labelcolor':NAVY,'text.color':NAVY,'xtick.color':NAVY,'ytick.color':NAVY,
        'grid.color':'#DFE7EF','grid.alpha':.65,'figure.facecolor':'#F8FAFD','axes.facecolor':'white'})


def canvas(number, columns=2, rows=1):
    fig,axes = plt.subplots(rows,columns,figsize=(12.8,7),squeeze=False)
    fig.suptitle(f'{number:02d}  /  {QUESTIONS[number-1]}',x=.06,y=.97,ha='left',fontsize=18,fontweight='bold')
    fig.text(.06,.90,'DAY 45     A/B TESTING FROM SCRATCH     /     SYNTHETIC LEARNING LAB',fontsize=9,color=BLUE)
    fig.text(.06,.035,textwrap.fill(LESSONS[number-1],145),fontsize=9,va='bottom',linespacing=1.5)
    fig.subplots_adjust(left=.08,right=.96,top=.80,bottom=.22,wspace=.34,hspace=.55)
    return fig,axes.ravel()


def plot_layout(fig,title):
    fig.update_layout(template='plotly_white',title=dict(text=title,font=dict(size=19)),
        height=580,font=dict(family='Arial',color=NAVY,size=13),
        colorway=[BLUE,ORANGE,TEAL,PURPLE],margin=dict(l=65,r=45,t=100,b=75),
        paper_bgcolor='#FFFFFF',legend=dict(orientation='h',y=1.12),barmode='overlay')
    return fig


def compact(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    raise TypeError(type(value).__name__)


def write_html(path,number,scenarios,config):
    payload = json.dumps([dict(name=name,data=fig.to_plotly_json()['data'],
                              layout=fig.to_plotly_json().get('layout',{})) for name,fig in scenarios],
                         cls=PlotlyJSONEncoder).replace('</','<\\/')
    options = ''.join(f'<option value="{i}">{escape(name)}</option>' for i,(name,_) in enumerate(scenarios))
    content = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Day 45 visual lab</title><style>body{margin:0;background:#eef3f9;color:#15283f;font:16px Arial,sans-serif}main{max-width:1320px;margin:28px auto;padding:0 22px}a{color:#0072b2}header{border-top:5px solid #0072b2;background:white;padding:24px 30px;border-radius:12px}h1{font-size:28px;margin:12px 0}p{line-height:1.6}.tag{font-size:12px;letter-spacing:2px;color:#0072b2;font-weight:bold}#plot{margin-top:18px;min-height:580px;border-radius:12px;background:white}select,button{font:inherit;padding:10px;border:1px solid #b8c7d8;border-radius:6px;background:white;margin:5px 8px 5px 0}.note{padding:18px 25px;background:white;border-radius:12px;margin-top:18px}img{max-width:100%}</style><main><header>
<a href="index.html">All generated modules</a><div class="tag">DAY 45 / OFFLINE EXPERIMENTATION LAB</div>
'''
    content += f'<h1>{number:02d} / {escape(QUESTIONS[number-1])}</h1><p>{escape(LESSONS[number-1])}</p>'
    content += f'<label for="scenario">Precomputed scenario: </label><select id="scenario">{options}</select><button id="play">Play scenarios</button>'
    content += f'<p class="tag">Seed {config.seed} | HTML does not recompute arbitrary inputs. Regenerate with the Python CLI.</p></header><div id="plot"></div>'
    content += '<div class="note"><b>Evidence boundary</b><p>All quantities are synthetic or analytical. Interpretation requires author review. Hover, zoom and inspect traces. Simulation parameters are recorded in run.json.</p></div>'
    content += '<script>'+get_plotlyjs()+'</script><script>const labScenarios='+payload+''';window.labScenarios=labScenarios;
const plot=document.getElementById('plot'),select=document.getElementById('scenario');
let pending=Promise.resolve(),timer=null;
function render(i){pending=pending.then(async()=>{const s=labScenarios[i];await Plotly.react(plot,s.data,s.layout,{responsive:true,scrollZoom:true,displaylogo:false});plot.dataset.state=String(i);plot.dataset.ready='true';});return pending;}
select.addEventListener('change',()=>render(+select.value));
if(labScenarios.length===1){select.disabled=true;document.getElementById('play').disabled=true;}
document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;document.getElementById('play').textContent='Play scenarios';return;}document.getElementById('play').textContent='Pause';timer=setInterval(()=>{select.value=String((+select.value+1)%labScenarios.length);select.dispatchEvent(new Event('change'));},1200);};
render(0);</script></main></html>'''
    path.write_text(content,encoding='utf-8')


def randomization(c):
    d = population(c)
    z,f = d['assignment'],d['features']
    labels = ['Engagement','Activity','History','Latent (oracle)']
    fig,ax = canvas(1,2,2)
    def draw(k):
        for a in ax: a.clear()
        count = max(2,int(k)); idx=np.arange(len(z)); revealed=idx<count
        positions=np.where(revealed, np.where(z==0,-1.,1.),0.)
        colors=np.where(revealed,np.where(z==0,BLUE,ORANGE),'#CCD5E0')
        ax[0].scatter(positions+.12*np.sin(idx*2.4),idx,s=10,c=colors)
        ax[0].set(xticks=[-1,0,1],xticklabels=['Control A','Unassigned','Treatment B'],xlim=(-1.5,1.5),ylabel='Synthetic user index')
        sizes=[int(((z==g)&revealed).sum()) for g in (0,1)]
        ax[1].bar(['Control A','Treatment B'],sizes,color=[BLUE,ORANGE]);ax[1].set(ylim=(0,len(z)*.65),ylabel='Assigned users',title=f'{min(count,len(z))} of {len(z)} assignments revealed')
        for g,col in [(0,BLUE),(1,ORANGE)]:
            ax[2].hist(f[(z==g)&revealed,1],bins=np.linspace(0,6,22),density=True,alpha=.5,color=col,label=['Control A','Treatment B'][g])
        ax[2].set(xlabel='Baseline activity (synthetic units)',ylabel='Density',xlim=(0,6));ax[2].legend(fontsize=8)
        if min(sizes):
            a=d['standardized'][(z==1)&revealed].mean(0)-d['standardized'][(z==0)&revealed].mean(0)
            ax[3].barh(labels,a,color=[BLUE]*3+[PURPLE])
        ax[3].axvline(0,color=GRAY);ax[3].set(xlim=(-.6,.6),xlabel='Mean B minus A / population SD')
    draw(len(z))
    scenarios=[]
    for scenario in range(6):
        assignment=c.rng(1,scenario+1).integers(0,2,len(z))
        p=make_subplots(rows=1,cols=2,subplot_titles=['Baseline characteristic distributions','Realized baseline mean differences'])
        for g,col in [(0,BLUE),(1,ORANGE)]:
            for j,label in enumerate(labels):
                p.add_trace(go.Violin(y=d['standardized'][assignment==g,j],x=[label]*int((assignment==g).sum()),
                    name=['Control A','Treatment B'][g],legendgroup=str(g),showlegend=j==0,
                    line_color=col,side=['negative','positive'][g],points=False),1,1)
        diff=d['standardized'][assignment==1].mean(0)-d['standardized'][assignment==0].mean(0)
        p.add_trace(go.Bar(x=labels,y=diff,name='B - A / population SD',marker_color=PURPLE),1,2)
        p.update_yaxes(title_text='Standardized baseline value',row=1,col=1)
        p.update_yaxes(title_text='Standardized difference',range=[-.6,.6],row=1,col=2)
        scenarios.append((f'Fixed population / random allocation {scenario+1} / nB={(assignment==1).sum()}',plot_layout(p,'Observed balance varies across randomized allocations')))
    rng=c.rng(1,99);alloc=rng.integers(0,2,(c.repetitions,len(z)),dtype=np.int8)
    sb=alloc.sum(1);means_b=alloc@d['standardized']/sb[:,None]
    means_a=(1-alloc)@d['standardized']/(len(z)-sb[:,None])
    summary=dict(n_users=len(z),n_a=int((z==0).sum()),n_b=int((z==1).sum()),
                 realized_smd=d['mean_differences'],rerandomization_mean_smd=(means_b-means_a).mean(0),
                 balance_repetitions=c.repetitions,features=labels)
    return fig,scenarios,summary,(draw,len(z))


def potential_outcomes(c):
    d=population(c,n=2400,effect=.08,module=2)
    fig,ax=canvas(2)
    shown=np.arange(60)
    for i in shown:
        ax[0].plot([i,i],[d['y0'][i],d['y1'][i]+.06],color=TEAL if d['y1'][i]!=d['y0'][i] else '#CBD6E2',alpha=.7)
    ax[0].scatter(shown,d['y0'][shown],color=BLUE,s=14,label='Y(0), oracle')
    ax[0].scatter(shown,d['y1'][shown]+.06,color=ORANGE,s=14,label='Y(1), oracle (+0.06 display offset)')
    ax[0].set(yticks=[0,1],ylabel='Binary potential outcome',xlabel='Synthetic user index (first 60)',title='One pair per user; vertical links show individual effects')
    ax[0].legend(loc='center',fontsize=7)
    values=[d['expected_ate'],d['finite_ate'],d['estimate']]
    names=['Expected effect (oracle)','Finite-population ATE (oracle)','Randomized estimate']
    ax[1].barh(names,np.array(values)*100,color=[GRAY,TEAL,ORANGE]);ax[1].set(xlabel='Absolute effect (percentage points)',title='Three quantities; only estimate is observed')
    scenarios=[]
    for oracle in [True,False]:
        p=make_subplots(rows=1,cols=2,subplot_titles=['First 60 users','Population effects'])
        for g,col in [(0,BLUE),(1,ORANGE)]:
            mask=np.ones(60,bool) if oracle else d['assignment'][shown]==g
            custom=np.column_stack((d[f'y{g}'][shown[mask]],(d['y1']-d['y0'])[shown[mask]])) if oracle else d[f'y{g}'][shown[mask]][:,None]
            hover='User %{x}<br>Binary outcome=%{customdata[0]}'
            if oracle: hover+='<br>Individual effect=%{customdata[1]} (oracle only)'
            p.add_trace(go.Scatter(x=shown[mask],y=d[f'y{g}'][shown[mask]]+.06*g,customdata=custom,
                hovertemplate=hover+'<extra>%{fullData.name}</extra>',mode='markers',name=f'Y({g})'+(' oracle' if oracle else ' observed'),marker=dict(color=col,size=8)),1,1)
        p.add_trace(go.Bar(x=names if oracle else names[-1:],y=np.array(values if oracle else values[-1:])*100,
            name='Effect (pp)',marker_color=[GRAY,TEAL,ORANGE] if oracle else [ORANGE]),1,2)
        p.update_yaxes(title_text='Binary outcome (slight display offset)',row=1,col=1)
        p.update_yaxes(title_text='Effect (pp)',row=1,col=2)
        scenarios.append(('Simulator: both outcomes' if oracle else 'Experiment: assigned outcome only',plot_layout(p,'Potential outcomes are jointly known only inside the simulator')))
    summary=dict(n_users=2400,probability_increment=.08,expected_ate=d['expected_ate'],finite_ate=d['finite_ate'],estimate=d['estimate'],ite_counts=np.bincount(d['y1']-d['y0'],minlength=2))
    return fig,scenarios,summary,None


def sampling_distributions(c):
    null=trials(c,p_b=c.baseline,module=3,scenario=0)
    alt=trials(c,module=3,scenario=1)
    fig,ax=canvas(3)
    fig.text(.06,.84,f'First unselected alternative experiment: SE(null)={alt["se_null"][0]*100:.3f} pp; SE(effect)={alt["se"][0]*100:.3f} pp; Z={alt["z"][0]:.3f}',fontsize=10)
    se=np.sqrt(2*c.baseline*(1-c.baseline)/c.n_per_arm)
    delta=c.treatment-c.baseline; critical=norm.isf(c.alpha/2)
    effect_bins=np.linspace(min(-4*se,delta-4*se,null['difference'].min(),alt['difference'].min())*100,
                            max(4*se,delta+4*se,null['difference'].max(),alt['difference'].max())*100,40)
    z_bins=np.linspace(min(-5,null['z'].min(),alt['z'].min())-.2,
                       max(5,null['z'].max(),alt['z'].max())+.2,42)
    def draw(k):
        for a in ax:a.clear()
        for data,col,name in [(null,BLUE,'Null simulations'),(alt,ORANGE,'Alternative simulations')]:
            ax[0].hist(data['difference'][:k]*100,bins=effect_bins,density=True,alpha=.5,color=col,label=name)
            ax[1].hist(data['z'][:k],bins=z_bins,density=True,alpha=.5,color=col,label=name)
        ax[0].axvline(delta*100,color=ORANGE,ls='--');ax[0].set(xlim=(effect_bins[0],effect_bins[-1]),ylim=(0,1.5/(se*100)),xlabel='Estimated B - A (percentage points)',ylabel='Density / pp',title=f'{k:,} independent experiments per hypothesis')
        ax[1].axvspan(z_bins[0],-critical,color=PURPLE,alpha=.12);ax[1].axvspan(critical,z_bins[-1],color=PURPLE,alpha=.12)
        ax[1].axvline(alt['z'][0],color=NAVY,label='First alternative experiment')
        ax[1].set(xlim=(z_bins[0],z_bins[-1]),ylim=(0,.6),xlabel='Pooled Z statistic (unitless)',ylabel='Density',title=f'Reject |Z| > {critical:.3f}; observed p={alt["p_value"][0]:.4f}')
        ax[0].legend(fontsize=8);ax[1].legend(fontsize=8)
    draw(c.repetitions)
    scenarios=[]
    for k in sorted(set([min(100,c.repetitions),min(500,c.repetitions),c.repetitions])):
        p=make_subplots(rows=1,cols=2,subplot_titles=['Estimator distribution (pp)','Z distribution and rejection rule'])
        for d,col,name in [(null,BLUE,'Null'),(alt,ORANGE,'Alternative')]:
            p.add_trace(go.Histogram(x=d['difference'][:k]*100,histnorm='probability density',name=name,marker_color=col,opacity=.55,nbinsx=35),1,1)
            p.add_trace(go.Histogram(x=d['z'][:k],histnorm='probability density',name=name,marker_color=col,opacity=.55,showlegend=False,nbinsx=35),1,2)
        for bound in [-critical,critical]:p.add_vline(x=bound,line_dash='dash',line_color=PURPLE,row=1,col=2)
        p.add_vline(x=float(alt['z'][0]),line_color=NAVY,row=1,col=2)
        p.update_xaxes(title_text='Estimated difference (pp)',range=[effect_bins[0],effect_bins[-1]],row=1,col=1)
        p.update_xaxes(title_text='Z (unitless)',range=[z_bins[0],z_bins[-1]],row=1,col=2)
        scenarios.append((f'{k} independent experiments per hypothesis',plot_layout(p,f'Pooled Z: first observed p={alt["p_value"][0]:.5f}; critical +/-{critical:.3f}')))
    summary=dict(null_rejections=rate_summary(null['p_value']<c.alpha),alternative_rejections=rate_summary(alt['p_value']<c.alpha),
                 first_observed={key:float(alt[key][0]) for key in ['difference','z','p_value','se_null','se','low','high']},known_null_se=se)
    return fig,scenarios,summary,(draw,c.repetitions)


def confidence_intervals(c):
    sizes=sorted(set([1000,c.n_per_arm]));data=[trials(c,n=n,module=4,scenario=i) for i,n in enumerate(sizes)]
    fig,ax=canvas(4,len(sizes));delta=c.treatment-c.baseline;shown=min(80,c.repetitions)
    limits=[min(float(d['low'][:shown].min()) for d in data)*100-.2,max(float(d['high'][:shown].max()) for d in data)*100+.2]
    def draw(k):
        for a,d,n in zip(ax,data,sizes):
            a.clear();covered=(d['low']<=delta)&(delta<=d['high'])
            for i in range(k):
                a.plot([d['low'][i]*100,d['high'][i]*100],[i,i],color=TEAL if covered[i] else PURPLE,lw=1.7)
                a.plot(d['difference'][i]*100,i,'.',color=NAVY,ms=3)
            a.axvline(delta*100,color=ORANGE,ls='--');a.set(xlim=limits,ylim=(shown+1,-2),xlabel=f'Effect and Wald {1-c.alpha:.0%} interval (pp)',ylabel='Independent experiment',title=f'n={n:,}/arm | shown coverage={covered[:k].mean():.1%}')
    draw(shown);scenarios=[]
    for n,d in zip(sizes,data):
        p=go.Figure();covered=(d['low'][:shown]<=delta)&(delta<=d['high'][:shown])
        for good,col,name in [(True,TEAL,'Contains true effect'),(False,PURPLE,'Misses true effect')]:
            mask=covered==good;ids=np.arange(shown)[mask]
            p.add_trace(go.Scatter(x=d['difference'][:shown][mask]*100,y=ids,mode='markers',name=name,marker_color=col,
                error_x=dict(type='data',array=(d['high'][:shown][mask]-d['difference'][:shown][mask])*100,color=col)))
        p.add_vline(x=delta*100,line_dash='dash',line_color=ORANGE)
        p.update_xaxes(title_text=f'Estimated effect and {1-c.alpha:.0%} Wald interval (pp)',range=limits);p.update_yaxes(title_text='Independent experiment',autorange='reversed')
        scenarios.append((f'n={n:,} per arm / {shown} displayed intervals',plot_layout(p,f'Full {c.repetitions:,}-experiment coverage: {((d["low"]<=delta)&(delta<=d["high"])).mean():.2%}')))
    summary=[dict(n_per_arm=n,coverage=rate_summary((d['low']<=delta)&(delta<=d['high'])),mean_width=float((d['high']-d['low']).mean())) for n,d in zip(sizes,data)]
    return fig,scenarios,summary,(draw,shown)

def power(c):
    fig,ax=canvas(5)
    ns=np.linspace(1000,40000,80);effects=[.005,.01,.02]
    scenarios=[];observations=[]
    designs=[(c.baseline,c.alpha),(.05,.05),(.20,.05),(c.baseline,.01),(c.baseline,.10)]
    designs=list(dict.fromkeys(designs))
    for base,alpha in designs:
        p=make_subplots(rows=1,cols=2,subplot_titles=['Power: analytical curves and Monte Carlo checks','Planning users per arm vs MDE'])
        for j,(effect,col) in enumerate(zip(effects,[BLUE,ORANGE,TEAL])):
            curve=analytical_power(ns,effect,base,alpha)
            p.add_trace(go.Scatter(x=ns,y=curve,name=f'+{effect*100:g} pp (normal)',line_color=col),1,1)
            if (base,alpha)==designs[0]:ax[0].plot(ns,curve,color=col,label=f'+{effect*100:g} pp')
            for k,n in enumerate([1000,5000,15000,30000]):
                cfg=replace(c,baseline=base,treatment=base+effect,alpha=alpha)
                d=trials(cfg,n=n,module=5,scenario=100*designs.index((base,alpha))+j*10+k)
                r=rate_summary(d['p_value']<alpha);lo,hi=r['mc_ci95']
                p.add_trace(go.Scatter(x=[n],y=[r['rate']],mode='markers',showlegend=False,marker=dict(color=col,size=8),
                    error_y=dict(type='data',array=[hi-r['rate']],arrayminus=[r['rate']-lo])),1,1)
                observations.append(dict(baseline=base,alpha=alpha,effect=effect,n_per_arm=n,analytical=float(analytical_power(n,effect,base,alpha)),monte_carlo=r))
                if (base,alpha)==designs[0]:ax[0].errorbar(n,r['rate'],yerr=[[r['rate']-lo],[hi-r['rate']]],fmt='o',color=col,ms=4)
        mdes=np.linspace(.0025,.02,40)
        for target,col in [(.8,BLUE),(.9,ORANGE)]:
            sizes=planning_sizes(base,mdes,alpha,target)
            p.add_trace(go.Scatter(x=mdes*100,y=sizes,name=f'{target:.0%} target',line_color=col),1,2)
            if (base,alpha)==designs[0]:ax[1].plot(mdes*100,sizes,color=col,label=f'{target:.0%} power')
        for target in [.8,.9]:p.add_hline(y=target,line_dash='dash',line_color=GRAY,row=1,col=1)
        p.update_xaxes(title_text='Independent users per arm',row=1,col=1);p.update_yaxes(title_text='Rejection probability',range=[0,1],row=1,col=1)
        p.update_xaxes(title_text='MDE (absolute percentage points)',row=1,col=2);p.update_yaxes(title_text='Approximate users per arm',type='log',row=1,col=2)
        scenarios.append((f'Baseline {base:.0%} / alpha {alpha:g}',plot_layout(p,f'Independent binary outcomes | {c.repetitions:,} Monte Carlo repetitions per dot')))
    for target in [.8,.9]:ax[0].axhline(target,color=GRAY,ls='--',lw=1)
    ax[0].set(xlabel='Independent users per arm',ylabel='Power / rejection frequency',ylim=(0,1),title=f'Baseline {c.baseline:.0%}, alpha {c.alpha:g}')
    ax[0].legend(fontsize=9);ax[1].set(xlabel='MDE (percentage points)',ylabel='Approximate users per arm',yscale='log',title='Smaller effects demand substantially more users');ax[1].legend()
    return fig,scenarios,observations,None


def power_surface(c):
    ns=np.linspace(1000,40000,55);effects=np.linspace(0,.025,45)
    xx,yy=np.meshgrid(ns,effects);zz=analytical_power(xx,yy,c.baseline,c.alpha)
    fig,ax=canvas(6,1)
    contour=ax[0].contourf(xx,yy*100,zz,levels=np.linspace(0,1,21),cmap='cividis')
    ax[0].clabel(ax[0].contour(xx,yy*100,zz,levels=[.8],colors=[ORANGE],linewidths=2),fmt={.8:'80% power'})
    fig.colorbar(contour,ax=ax[0],label='Analytical power')
    ax[0].set(xlabel='Users per arm',ylabel='True absolute effect (pp)',title=f'2D companion / baseline={c.baseline:.0%}, alpha={c.alpha:g}')
    p=go.Figure(go.Surface(x=ns,y=effects*100,z=zz,colorscale='Cividis',cmin=0,cmax=1,
        colorbar=dict(title='Power'),hovertemplate='n/arm=%{x:.0f}<br>effect=%{y:.2f} pp<br>power=%{z:.3f}<extra></extra>',
        contours=dict(z=dict(show=True,start=.8,end=.8,size=.1,color=ORANGE,width=4,project_z=True))))
    from scipy.optimize import brentq
    line=[(brentq(lambda n: float(analytical_power(n,e,c.baseline,c.alpha))-.8,10,1e8),e) for e in effects[1:]]
    line=[(n,e) for n,e in line if ns[0]<=n<=ns[-1]]
    p.add_trace(go.Scatter3d(x=[n for n,e in line],y=[e*100 for n,e in line],z=[.8]*len(line),mode='lines',line=dict(color=ORANGE,width=7),name='80% power contour'))
    p.update_layout(scene=dict(xaxis_title='Users per arm',yaxis_title='True effect (pp)',zaxis=dict(title='Power',range=[0,1]),aspectratio=dict(x=1.4,y=1,z=.8)))
    plot_layout(p,f'Normal planning approximation | baseline={c.baseline:.0%}, alpha={c.alpha:g}')
    return fig,[('Fixed baseline and alpha / rotate, zoom and hover',p)],dict(baseline=c.baseline,alpha=c.alpha,n_range=[1000,40000],effect_pp_range=[0,2.5],model='Two-tail normal approximation'),None


def errors(c):
    fig,ax=canvas(7,1);scenarios=[];summary=[]
    for effect in [.005,.01,.02]:
        for alpha in [.01,.05,.10]:
            p0,p1=c.baseline,c.baseline+effect;pooled=(p0+p1)/2
            se0=np.sqrt(2*pooled*(1-pooled)/c.n_per_arm)
            se1=np.sqrt((p0*(1-p0)+p1*(1-p1))/c.n_per_arm)
            mu,sigma=effect/se0,se1/se0;critical=norm.isf(alpha/2)
            xx=np.linspace(-5,max(8,mu+4*sigma),1600)
            null,alt=norm.pdf(xx),norm.pdf(xx,mu,sigma)
            type2=norm.cdf(critical,mu,sigma)-norm.cdf(-critical,mu,sigma)
            p=go.Figure()
            p.add_trace(go.Scatter(x=xx,y=null,name='Null Z ~ N(0,1)',line_color=BLUE))
            p.add_trace(go.Scatter(x=xx,y=alt,name='Alternative Z approximation',line_color=ORANGE))
            regions=[(np.abs(xx)>=critical,null,PURPLE,'Type I: both null tails'),
                     (np.abs(xx)<critical,alt,GRAY,'Type II: alternative non-rejection'),
                     (np.abs(xx)>=critical,alt,TEAL,'Power: alternative rejection')]
            for mask,density,col,name in regions:
                # Separate tails with NaNs, avoiding a polygon across the non-rejection region.
                p.add_trace(go.Scatter(x=xx,y=np.where(mask,density,np.nan),fill='tozeroy',mode='lines',line=dict(width=0,color=col),opacity=.45,name=name,connectgaps=False))
            p.add_vrect(x0=-critical,x1=critical,fillcolor=GRAY,opacity=.08,line_width=0,
                        layer='below',annotation_text='Non-rejection region',annotation_position='top')
            for bound in [-critical,critical]:p.add_vline(x=bound,line_dash='dash',line_color=NAVY)
            p.update_xaxes(title_text='Z (unitless)');p.update_yaxes(title_text='Sampling density')
            scenarios.append((f'Effect +{effect*100:g} pp / alpha={alpha:g}',plot_layout(p,f'Normal model: alpha={alpha:.1%}, beta={type2:.1%}, power={1-type2:.1%}')))
            summary.append(dict(effect=effect,alpha=alpha,z_mean=mu,z_sd=sigma,type1=alpha,type2=type2,power=1-type2))
            if effect==.01 and alpha==.05:
                ax[0].plot(xx,null,color=BLUE,label='Null sampling distribution');ax[0].plot(xx,alt,color=ORANGE,label='Alternative sampling distribution')
                for mask,density,col,name in regions:ax[0].fill_between(xx,0,density,where=mask,color=col,alpha=.4,label=name)
                ax[0].axvspan(-critical,critical,color=GRAY,alpha=.08,label='Non-rejection region')
                for bound in [-critical,critical]:ax[0].axvline(bound,color=NAVY,ls='--')
                ax[0].set(xlabel='Z (unitless)',ylabel='Sampling density',title=f'Two-sided alpha={alpha:.0%} | beta={type2:.1%} | power={1-type2:.1%}')
    ax[0].legend(fontsize=8,ncol=1,loc='upper right')
    return fig,scenarios,summary,None


def aa_testing(c):
    d=trials(c,p_b=c.baseline,module=8);pvalues=d['p_value'];reject=pvalues<c.alpha
    cumulative=np.cumsum(reject)/np.arange(1,c.repetitions+1)
    report=rate_summary(reject)
    fig,ax=canvas(8)
    def draw(k):
        for a in ax:a.clear()
        ax[0].hist(pvalues[:k],bins=np.linspace(0,1,21),density=True,color=BLUE,alpha=.8)
        ax[0].axhline(1,color=GRAY,ls='--',label='Uniform density reference')
        ax[0].axvline(c.alpha,color=PURPLE,ls='--');ax[0].set(xlim=(0,1),ylim=(0,2),xlabel='Two-sided pooled-test p-value',ylabel='Density',title=f'{k:,} independent null experiments');ax[0].legend(fontsize=9)
        ax[1].plot(np.arange(1,k+1),cumulative[:k]*100,color=BLUE)
        ax[1].axhline(c.alpha*100,color=PURPLE,ls='--',label=f'Nominal alpha={c.alpha:.0%}')
        ax[1].set(xlim=(0,c.repetitions),ylim=(0,max(15,c.alpha*180)),xlabel='Independent experiments accumulated',ylabel='Cumulative rejection frequency (%)',title=f'Observed: {cumulative[k-1]:.2%} at {k:,} runs');ax[1].legend()
    draw(c.repetitions);scenarios=[]
    for k in sorted(set([100,min(500,c.repetitions),c.repetitions])):
        p=make_subplots(rows=1,cols=2,subplot_titles=['Null p-values','Cumulative false positives'])
        p.add_trace(go.Histogram(x=pvalues[:k],xbins=dict(start=0,end=1,size=.05),histnorm='probability density',name='A/A p-values',marker_color=BLUE),1,1)
        p.add_trace(go.Scatter(x=np.arange(1,k+1),y=cumulative[:k]*100,name='Observed cumulative rate',line_color=BLUE),1,2)
        p.add_hline(y=1,line_dash='dash',line_color=GRAY,row=1,col=1)
        p.add_hline(y=c.alpha*100,line_dash='dash',line_color=PURPLE,row=1,col=2)
        p.update_xaxes(title_text='p-value',range=[0,1],row=1,col=1);p.update_yaxes(title_text='Density',range=[0,2],row=1,col=1)
        p.update_xaxes(title_text='Independent experiments',range=[0,c.repetitions],row=1,col=2);p.update_yaxes(title_text='Rejection frequency (%)',range=[0,max(15,c.alpha*180)],row=1,col=2)
        scenarios.append((f'{k:,} independent A/A experiments',plot_layout(p,f'No true effect | baseline={c.baseline:.0%}, n={c.n_per_arm:,}/arm')))
    return fig,scenarios,report,(draw,c.repetitions)

def peeking_visual(c):
    d=peeking(c);looks,pvalues=d['looks'],d['p_values']
    fig,ax=canvas(9,3)
    maxn=c.n_per_arm;fixed=d['fixed'];naive=d['naive']
    def draw(k):
        for a in ax:a.clear()
        for i in range(5):ax[0].plot(looks[:k],pvalues[i,:k],lw=1.2,alpha=.7,label=f'Experiment {i+1}')
        ax[0].axhline(c.alpha,color=PURPLE,ls='--');ax[0].set(xlim=(0,maxn),ylim=(.0001,1),yscale='log',xlabel='Users accumulated per arm',ylabel='Fixed-horizon p-value at each look',title='First five unselected null experiments')
        now=(pvalues[:,:k]<c.alpha).any(1)
        rates=[(pvalues[:,k-1]<c.alpha).mean(),now.mean()]
        ax[1].bar(['Test at current look','Any crossing so far'],np.array(rates)*100,color=[BLUE,ORANGE]);ax[1].axhline(c.alpha*100,color=PURPLE,ls='--');ax[1].set(ylim=(0,max(30,naive['rate']*140)),ylabel='False-positive frequency (%)',title=f'{c.repetitions:,} paired experiments')
        crossed=pvalues[:,:k]<c.alpha;hit=crossed.any(1)
        index=np.where(hit,crossed.argmax(1),k-1);stops=looks[index]
        ax[2].hist(stops[hit],bins=np.linspace(0,maxn,21),color=ORANGE,label='First significant look')
        ax[2].bar(looks[k-1],int((~hit).sum()),width=maxn/35,color=GRAY,label='No crossing: censored at horizon')
        ax[2].set(xlim=(0,maxn*1.05),ylim=(0,c.repetitions),xlabel='Stopping users per arm',ylabel='Experiments',title=f'Observed through look {k}/{len(looks)}');ax[2].legend(fontsize=7)
    draw(len(looks));scenarios=[]
    for k in sorted(set([5,15,len(looks)])):
        p=make_subplots(rows=1,cols=3,subplot_titles=['Unselected null p-value paths','Same data, different policies','First crossing or censoring'])
        for i in range(5):p.add_trace(go.Scatter(x=looks[:k],y=pvalues[i,:k],name=f'Experiment {i+1}',showlegend=False,line=dict(width=1.5)),1,1)
        p.add_hline(y=c.alpha,line_dash='dash',line_color=PURPLE,row=1,col=1)
        any_so_far=(pvalues[:,:k]<c.alpha).any(1)
        rates=[(pvalues[:,k-1]<c.alpha).mean(),any_so_far.mean()]
        p.add_trace(go.Bar(x=['Test at this look','Any crossing'],y=np.array(rates)*100,marker_color=[BLUE,ORANGE],name='Null rejection frequency'),1,2)
        first=np.where(any_so_far,(pvalues[:,:k]<c.alpha).argmax(1),k-1)
        p.add_trace(go.Histogram(x=looks[first[any_so_far]],name='False-positive stops',marker_color=ORANGE,nbinsx=20),1,3)
        p.add_trace(go.Bar(x=[int(looks[k-1])],y=[int((~any_so_far).sum())],width=maxn/35,name='No rejection / censored',marker_color=GRAY),1,3)
        p.update_xaxes(title_text='Users per arm',range=[0,maxn],row=1,col=1);p.update_yaxes(title_text='p-value',type='log',range=[-4,0],row=1,col=1)
        p.update_yaxes(title_text='False positives (%)',row=1,col=2);p.update_xaxes(title_text='Users per arm at stop/censor',row=1,col=3)
        scenarios.append((f'Look {k}/{len(looks)}: horizon {looks[k-1]:,}/arm',plot_layout(p,f'Paired null simulation | alpha={c.alpha:g} | full maximum {maxn:,}/arm')))
    summary=dict(looks=looks,fixed=fixed,naive=naive,max_horizon=maxn,
                 median_stopping_n=float(np.median(d['stopped_n'])),non_rejecting_censored=int((~d['any_cross']).sum()))
    return fig,scenarios,summary,(draw,len(looks))


def srm(c):
    cases=allocation_cases(c);fig,ax=canvas(10,3);scenarios=[]
    for a,d in zip(ax,cases):
        counts=[d['n']-d['n_b'],d['n_b']];expected=np.array([1-d['intended_q'],d['intended_q']])*d['n']
        a.bar(['Control A','Treatment B'],counts,color=[BLUE,ORANGE]);a.plot([0,1],expected,'_',ms=35,mew=3,color=NAVY,label='Intended expected counts')
        a.set(ylim=(0,d['n']*.75),ylabel='Assigned users',title=f'{d["label"]}\np(SRM)={d["p_value"]:.3g}');a.legend(fontsize=7)
        p=go.Figure(go.Bar(x=['Control A','Treatment B'],y=counts,marker_color=[BLUE,ORANGE],name='Observed users'))
        p.add_trace(go.Scatter(x=['Control A','Treatment B'],y=expected,mode='markers',marker=dict(symbol='line-ew',size=35,color=NAVY,line=dict(width=3)),name='Intended expected count'))
        p.update_yaxes(title_text='Users',range=[0,d['n']*.75]);plot_layout(p,f'{d["label"]} | exact binomial p={d["p_value"]:.4g}')
        scenarios.append((d['label'],p))
    return fig,scenarios,cases,None

def unit_of_analysis(c):
    fig,ax=canvas(11,3);scenarios=[];summary=[]
    for rho in [0,.05,.20]:
        d=clustered_trials(c,rho=rho)
        naive,user=d['naive_first'],d['user_first']
        p=make_subplots(rows=1,cols=2,subplot_titles=['Same estimate, different uncertainty','Repeated null false positives'])
        for i,(r,name,col) in enumerate([(naive,'Naive request-level',ORANGE),(user,'Independent user means (Welch)',BLUE)]):
            p.add_trace(go.Scatter(x=[r['difference']*100],y=[name],mode='markers',name=name,marker=dict(color=col,size=12),
                error_x=dict(type='data',array=[(r['high']-r['difference'])*100],color=col)),1,1)
        for report,name,col in [(d['naive'],'Naive requests',ORANGE),(d['user'],'Welch users',BLUE)]:
            lo,hi=report['mc_ci95'];p.add_trace(go.Bar(x=[name],y=[report['rate']*100],name=name,showlegend=False,marker_color=col,
                error_y=dict(type='data',array=[(hi-report['rate'])*100],arrayminus=[(report['rate']-lo)*100])),1,2)
        p.add_vline(x=0,line_dash='dash',line_color=GRAY,row=1,col=1);p.add_hline(y=c.alpha*100,line_dash='dash',line_color=PURPLE,row=1,col=2)
        p.update_xaxes(title_text=f'Effect and {1-c.alpha:.0%} interval (pp)',row=1,col=1);p.update_yaxes(title_text='False positives (%)',row=1,col=2)
        scenarios.append((f'ICC={rho:g} / 20 events per user',plot_layout(p,f'500 users/arm, equal event counts | design effect = {d["design_effect"]:.2f}')))
        summary.append({k:v for k,v in d.items() if k!='counts'})
        if rho==.20:
            rng=c.rng(11,999);counts=np.r_[d['counts'][0,:20],d['counts'][1,:20]]
            event_matrix=np.array([rng.permutation(np.arange(20)<count) for count in counts])
            ax[0].imshow(event_matrix,aspect='auto',cmap='Blues',vmin=0,vmax=1,interpolation='nearest')
            ax[0].axhline(19.5,color=ORANGE,lw=2);ax[0].set(xlabel='Event index within user',ylabel='Independent user (20 A, 20 B)',title='Events share a latent user risk')
            for i,(r,name,col) in enumerate([(naive,'Naive requests',ORANGE),(user,'Welch user means',BLUE)]):
                ax[1].errorbar(r['difference']*100,i,xerr=[[(r['difference']-r['low'])*100],[(r['high']-r['difference'])*100]],fmt='o',color=col,capsize=4)
            ax[1].set(yticks=[0,1],yticklabels=['Naive requests','Welch users'],xlabel=f'Effect and {1-c.alpha:.0%} interval (pp)',title=f'ICC={rho:g}, equal 20 events/user');ax[1].axvline(0,color=GRAY,ls='--')
            ax[2].bar(['Naive requests','Welch users'],[d['naive']['rate']*100,d['user']['rate']*100],color=[ORANGE,BLUE]);ax[2].axhline(c.alpha*100,color=PURPLE,ls='--');ax[2].set(ylabel='Null rejection frequency (%)',title=f'{c.repetitions:,} independent experiments')
    return fig,scenarios,summary,None


def ai_product(c):
    d=product_experiment(c);m,r,policy=d['metrics'],d['primary'],d['policy']
    fig,ax=canvas(12,2,2)
    ax[0].errorbar(r['difference']*100,0,xerr=[[(r['difference']-r['low'])*100],[(r['high']-r['difference'])*100]],fmt='o',color=BLUE,capsize=5)
    ax[0].axvline(0,color=GRAY);ax[0].axvline(policy['minimum_effect']*100,color=TEAL,ls='--',label='Minimum worthwhile effect')
    ax[0].set(xlabel='Task completion effect (pp)',yticks=[],title=f'Primary p={r["p_value"]:.4g} | {d["status"]}');ax[0].legend(fontsize=8)
    fig.text(.06,.84,f'Absolute effect: {r["difference"]*100:+.3f} pp | Relative lift: {r["relative_lift"]:+.2%}',fontsize=10)
    ax[1].bar(['Control A','Treatment B'],[m['p95_a'],m['p95_b']],color=[BLUE,ORANGE]);ax[1].axhline(m['p95_a']*policy['max_latency_ratio'],color=PURPLE,ls='--');ax[1].set(ylabel='P95 latency (seconds)',title='Point-estimate guardrail: ratio <= 1.25')
    ax[2].bar(['Control A','Treatment B'],[m['cost_a'],m['cost_b']],color=[BLUE,ORANGE]);ax[2].axhline(m['cost_a']+policy['max_cost_increment'],color=PURPLE,ls='--');ax[2].set(ylabel='Cost per eligible assigned user (USD)',title='Illustrative cost increment <= $0.005')
    names=list(d['gates']);values=[int(d['gates'][name]) for name in names]
    ax[3].barh([name.replace('_',' ') for name in names],[1]*len(names),color=[TEAL if v else PURPLE for v in values]);ax[3].set(xticks=[],title='Predefined illustrative policy: human review required')
    for i,v in enumerate(values):ax[3].text(.5,i,'PASS' if v else 'HOLD',ha='center',va='center',color='white',fontweight='bold')
    p=make_subplots(rows=2,cols=2,subplot_titles=['Primary effect and Wald interval (pp)','Operational guardrails (value / limit)','Synthetic diagnostics','Predefined review gates'])
    p.add_trace(go.Scatter(x=[r['difference']*100],y=['Task success'],mode='markers',name='Estimated effect',marker_color=BLUE,
        error_x=dict(type='data',array=[(r['high']-r['difference'])*100])),1,1)
    p.add_vline(x=policy['minimum_effect']*100,line_dash='dash',line_color=TEAL,row=1,col=1)
    ratios=[(m['p95_b']/m['p95_a'])/policy['max_latency_ratio'],(m['cost_b']-m['cost_a'])/policy['max_cost_increment'],m['safety_upper_b']/policy['max_safety_upper']]
    p.add_trace(go.Bar(x=['P95 ratio','Cost increment','Safety upper bound'],y=ratios,marker_color=[PURPLE if v>1 else TEAL for v in ratios],name='Value / policy maximum'),1,2);p.add_hline(y=1,line_dash='dash',row=1,col=2)
    # Numeric diagnostics have different units; normalized ratios are explicitly labeled.
    p.add_trace(go.Bar(x=['Retrieval proxy','Documents','Mean tokens'],y=[m['quality_b']/m['quality_a'],m['docs_b']/m['docs_a'],m['tokens_b']/m['tokens_a']],name='Diagnostic B / A',marker_color=BLUE),2,1)
    p.add_trace(go.Bar(x=[name.replace('_',' ') for name in names],y=[1]*len(names),text=['PASS' if v else 'HOLD' for v in values],textposition='inside',marker_color=[TEAL if v else PURPLE for v in values],name='Policy gates'),2,2)
    p.update_yaxes(title_text='Diagnostic B / A',row=2,col=1);p.update_yaxes(showticklabels=False,row=2,col=2)
    plot_layout(p,f'ALL SYNTHETIC | completion p={r["p_value"]:.4g} | {d["status"]}');p.update_layout(height=760)
    detail=go.Figure(go.Table(header=dict(values=['Metric / gate','Control A','Treatment B / decision'],fill_color='#DBE9F5',font=dict(size=14)),cells=dict(values=[['Completion rate','P95 latency (s)','Cost / assigned user (USD)','Unacceptable-response rate','Safety: one-sided exact 95% upper','Retrieval proxy','Retrieved docs','Mean tokens']+[name.replace('_',' ') for name in names],
        [f'{r["rate_a"]:.3%}',f'{m["p95_a"]:.3f}',f'{m["cost_a"]:.5f}',f'{m["safety_a"]:.3%}','Not a treatment decision bound',f'{m["quality_a"]:.3f}',m['docs_a'],f'{m["tokens_a"]:.0f}']+['Prespecified policy']*len(names),
        [f'{r["rate_b"]:.3%}',f'{m["p95_b"]:.3f}',f'{m["cost_b"]:.5f}',f'{m["safety_b"]:.3%}',f'{m["safety_upper_b"]:.3%}',f'{m["quality_b"]:.3f}',m['docs_b'],f'{m["tokens_b"]:.0f}']+['PASS' if v else 'HOLD' for v in values]],height=29)))
    plot_layout(detail,'Synthetic metric ledger | point-estimate latency/cost gates have no uncertainty guarantee')
    return fig,[('Decision dashboard',p),('Metric values and gate ledger',detail)],d,None

BUILDERS = dict(zip(MODULES,[randomization,potential_outcomes,sampling_distributions,
    confidence_intervals,power,power_surface,errors,aa_testing,peeking_visual,srm,unit_of_analysis,ai_product]))
QUICK = ('randomization','sampling_distributions','confidence_intervals','aa_testing','peeking')


def render_module(name,config,directory,quality='normal',gifs=True):
    number=MODULES.index(name)+1;stem=f'{number:02d}_{name}'
    start=time.perf_counter()
    fig,scenarios,summary,animation=BUILDERS[name](config)
    dpi,frames,fps={'light':(85,8,3),'normal':(110,20,5),'high':(140,32,7)}[quality]
    try:
        fig.savefig(directory/f'{stem}.png',dpi=dpi)
        write_html(directory/f'{stem}.html',number,scenarios,config)
        assets=[f'{stem}.png',f'{stem}.html']
        frame_count=0
        if animation is not None and gifs:
            draw,total=animation
            steps=np.unique(np.linspace(max(2,total//frames),total,frames,dtype=int))
            movie=FuncAnimation(fig,lambda k: draw(int(k)),frames=steps,interval=1000/fps,repeat=True,cache_frame_data=False)
            movie.save(directory/f'{stem}.gif',writer=PillowWriter(fps=fps),dpi=dpi)
            with Image.open(directory/f'{stem}.gif') as image:frame_count=image.n_frames
            if frame_count<2:raise RuntimeError('Animation unexpectedly has fewer than two frames.')
            assets.append(f'{stem}.gif')
    finally:
        plt.close(fig)
    return dict(module=name,number=number,question=QUESTIONS[number-1],hypothesis=HYPOTHESES[number-1],interpretation_candidate=LESSONS[number-1],
                review_status='Author review required',parameters=asdict(config),summary=summary,
                assets=assets,scenario_count=len(scenarios),gif_frames=frame_count,
                elapsed_seconds=time.perf_counter()-start)


def write_index(directory,records,config):
    cards=[]
    for record in records:
        stem=f'{record["number"]:02d}_{record["module"]}'
        gif=f'<a href="{stem}.gif">GIF animation</a> | ' if record['gif_frames'] else ''
        cards.append(f'<article><div class="tag">MODULE {record["number"]:02d}</div><h2><a href="{stem}.html">{escape(record["question"])}</a></h2><a href="{stem}.html"><img src="{stem}.png" alt="{escape(record["question"])}"></a><p>{escape(record["interpretation_candidate"])}</p><p>{gif}<a href="{stem}.png">PNG</a> | <a href="{stem}.html">Interactive HTML</a></p></article>')
    content='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Day 45 - A/B visual lab</title><style>body{margin:0;background:#edf2f8;color:#15283f;font:16px Arial,sans-serif}main{max-width:1250px;margin:40px auto;padding:0 24px}header{padding:35px;background:#15283f;color:white;border-radius:14px}h1{font-size:36px}.tag{letter-spacing:2px;font-size:12px;color:#0072b2;font-weight:bold}header .tag{color:#85d4fa}p{line-height:1.6}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:22px;margin-top:25px}article{padding:23px;background:white;border-radius:12px;box-shadow:0 5px 15px #15283f0a}img{max-width:100%}a{color:#0072b2;text-decoration:none}a:hover{text-decoration:underline}@media(max-width:550px){.grid{grid-template-columns:1fr}h1{font-size:28px}}</style><main><header><div class="tag">DAY 45 / OFFLINE VISUAL LEARNING LAB</div><h1>From random assignment<br>to a defensible experiment decision.</h1><p>Explore mechanisms, not just definitions. Open a module, choose a precomputed scenario, hover and zoom. Each HTML is self-contained. Free-form updates require rerunning the Python CLI.</p>'''
    content+=f'<p>Seed {config.seed} | {config.repetitions:,} repetitions where applicable | All data synthetic or analytical.<br>Interpretations require author review. No production approval is implied.</p></header><div class="grid">'+''.join(cards)+'</div><p>Detailed numerical records, parameters and environment: <a href="run.json">run.json</a>. Operational thresholds are hypothetical.</p></main></html>'
    (directory/'index.html').write_text(content,encoding='utf-8')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--quick',action='store_true',help='Five process modules, 600 repetitions and lightweight GIFs.')
    mode.add_argument('--all',action='store_true',help='All twelve modules, sequentially; 2000 repetitions and light rendering by default.')
    mode.add_argument('--module',choices=MODULES,help='Generate one selected module.')
    parser.add_argument('--seed',type=int,default=45)
    parser.add_argument('--repetitions',type=int,help='100 to 20000 independent experiments where applicable.')
    parser.add_argument('--n-per-arm',type=int,default=15000)
    parser.add_argument('--baseline',type=float,default=.10)
    parser.add_argument('--treatment',type=float,default=.11)
    parser.add_argument('--alpha',type=float,default=.05)
    parser.add_argument('--quality',choices=['light','normal','high'],help='PNG/GIF resolution and frame budget.')
    parser.add_argument('--no-gifs',action='store_true',help='Explicitly skip GIF rendering; keep PNG and HTML.')
    parser.add_argument('--output',type=Path,help='New or empty run directory. Existing runs are never overwritten.')
    args=parser.parse_args(argv)
    quick=args.quick or not (args.all or args.module)
    try:
        config=LabConfig(seed=args.seed,repetitions=args.repetitions or (600 if quick else 2000),
                         n_per_arm=args.n_per_arm,baseline=args.baseline,treatment=args.treatment,alpha=args.alpha)
        if args.repetitions is not None and args.repetitions==0:raise ValueError('repetitions must be positive.')
        directory=args.output or Path(__file__).resolve().parent/'visualizations'/'outputs'/datetime.now(timezone.utc).strftime(f'run-%Y%m%dT%H%M%S%fZ-seed{config.seed}')
        if directory.exists() and any(directory.iterdir()):raise ValueError('Output directory is not empty; choose a new run path.')
        directory.mkdir(parents=True,exist_ok=True)
    except (ValueError,OSError) as error:
        parser.error(str(error))
    selected=QUICK if quick else (MODULES if args.all else (args.module,))
    style();records=[]
    for name in selected:
        print(f'Rendering {name} ...',flush=True)
        records.append(render_module(name,config,directory,args.quality or 'light',not args.no_gifs))
        print(f'  Saved {len(records[-1]["assets"])} assets; {records[-1]["elapsed_seconds"]:.1f}s',flush=True)
    report=dict(run_utc=datetime.now(timezone.utc).isoformat(),config=asdict(config),
                versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                              matplotlib=matplotlib.__version__,plotly=plotly.__version__),
                mode='quick' if quick else ('all' if args.all else args.module),quality=args.quality or 'light',
                resources=dict(numerical_threads=1,cluster_batch_size=128),
                review_status='Author review required',modules=records)
    (directory/'run.json').write_text(json.dumps(report,indent=2,default=compact,allow_nan=False)+'\n',encoding='utf-8')
    write_index(directory,records,config)
    print(f'Open: {(directory/"index.html").resolve()}',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())

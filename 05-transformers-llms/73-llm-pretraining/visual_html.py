"""Offline Plotly documents with real controls; no remote JavaScript or runtime."""

import html
import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.offline import get_plotlyjs


STYLE = """
body{margin:0;color:#183247;background:#f1f5f9;font:16px system-ui,sans-serif}
header{background:#163247;color:white;padding:24px 4vw}h1{font-size:26px;margin:0 0 12px}
header p{max-width:1100px;line-height:1.5;margin:0;color:#dce8f2}
main{display:grid;grid-template-columns:270px minmax(0,1fr);gap:20px;padding:22px 3vw}
aside,article{padding:20px;background:white;border:1px solid #dbe4ed;border-radius:10px}
label{display:block;margin:0 0 16px;font-size:14px}input,select{width:100%;margin-top:7px;padding:7px;box-sizing:border-box;font:inherit}
#chart{height:620px}#metrics{white-space:pre-wrap;line-height:1.7;font:14px ui-monospace,monospace;padding:12px;background:#edf4f9}
.limit{font-size:14px;color:#52687b;line-height:1.6}a{color:#16749a}@media(max-width:800px){main{grid-template-columns:1fr}#chart{height:500px}}
"""


def local_plotly(folder):
    path = Path(folder) / "plotly.min.js"
    if not path.exists():
        path.write_text(get_plotlyjs(), encoding="utf-8")


def write_plotly(figure, path, title, limitation):
    path = Path(path)
    local_plotly(path.parent)
    figure.update_layout(template="plotly_white", font={"family": "Arial", "color": "#183247"},
                         title={"text": title + "<br><sup>" + limitation + "</sup>"},
                         margin={"t": 100, "b": 80, "l": 80, "r": 40})
    figure.write_html(path, include_plotlyjs="directory", include_mathjax=False,
                      config={"responsive": True, "displaylogo": False}, div_id="chart")


def control(identifier, label, value, minimum, maximum, step=1, kind="number"):
    return f'<label for="{identifier}">{html.escape(label)}<input id="{identifier}" type="{kind}" value="{value}" min="{minimum}" max="{maximum}" step="{step}"></label>'


def selector(identifier, label, choices):
    return f'<label for="{identifier}">{html.escape(label)}<select id="{identifier}">' + "".join(
        f'<option value="{html.escape(str(value))}">{html.escape(str(text))}</option>' for value, text in choices) + '</select></label>'


def write_page(path, title, description, controls, payload, script, limitation):
    path = Path(path)
    local_plotly(path.parent)
    data = json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
    shell = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{STYLE}</style><script src="plotly.min.js"></script></head>
<body><header><h1>{html.escape(title)}</h1><p>{html.escape(description)}</p></header><main><aside>{controls}
<p><a href="index.html">Return to gallery</a></p></aside><article><div id="chart"></div><pre id="metrics"></pre>
<p class="limit">{html.escape(limitation)}</p></article></main><script id="lab-data" type="application/json">{data}</script><script>
const P=JSON.parse(document.getElementById('lab-data').textContent);
const num=id=>Number(document.getElementById(id).value), val=id=>document.getElementById(id).value;
function show(traces,layout,state){{window.labReady=false;window.labState=state;document.getElementById('metrics').textContent=JSON.stringify(state,null,2);
if(typeof layout.title==='string')layout.title={{text:layout.title}};
if(layout.scene)for(const name of ['xaxis','yaxis','zaxis'])if(typeof layout.scene[name]?.title==='string')layout.scene[name].title={{text:layout.scene[name].title}};
return Plotly.react('chart',traces,{{paper_bgcolor:'#fff',plot_bgcolor:'#fff',font:{{family:'Arial',color:'#183247'}},margin:{{t:80,l:75,r:45,b:100}},legend:{{orientation:'h',x:0,y:-.2}},...layout}},{{responsive:true,displaylogo:false}}).then(()=>{{window.labReady=true}})}}
function valid(){{for(const input of document.querySelectorAll('input'))if(!input.checkValidity()){{document.getElementById('metrics').textContent='Enter finite values inside the displayed bounds.';return false}}return true}}
{script}
for(const element of document.querySelectorAll('input,select'))element.addEventListener('input',()=>{{if(valid())update()}});update();
</script></body></html>'''
    path.write_text(shell, encoding="utf-8")


def probability_html(report, path):
    controls = control("checkpoint", "Recorded checkpoint", 0, 0, len(report["snapshots"]) - 1, kind="range")
    controls += selector("probe", "Fixed context", [(0, report["probe_labels"][0]), (1, report["probe_labels"][1])])
    write_page(path, "Measured token distributions", "Every slider position is a recorded model prediction, with stable vocabulary identities and probability limits.",
               controls, report, r"""
function update(){let s=P.snapshots[num('checkpoint')],i=num('probe'),p=s.probabilities[i],fish=P.corpus.vocabulary.indexOf('fish');
show([{type:'bar',x:P.corpus.vocabulary,y:p,marker:{color:P.corpus.vocabulary.map((_,j)=>j===fish?'#d87c31':'#287da8')}}],
{title:P.probe_labels[i]+' · step '+s.step,yaxis:{title:{text:'Probability'},range:[0,1]},xaxis:{tickangle:-45}},
{step:s.step,entropy_nats:s.entropy[i],reference_token:'fish',reference_nll:s.reference_token_nll[i],reference_probability:p[fish],argmax:P.corpus.vocabulary[p.indexOf(Math.max(...p))]})}
""", "Synthetic CLM. Fish is one admissible continuation; confidence and correctness are different. Held-out subject/verb combinations were absent from training.")


def attention_html(reports, path):
    data = {name: reports[name] for name in ("clm", "mlm")}
    controls = selector("objective", "Objective", [("clm", "CLM: causal"), ("mlm", "MLM: bidirectional")])
    controls += control("checkpoint", "Recorded checkpoint", 0, 0, len(reports["clm"]["snapshots"]) - 1, kind="range")
    controls += control("head", "Head in recorded layer 0", 0, 0, reports["clm"]["configuration"]["heads"] - 1, kind="range")
    controls += selector("probe", "Probe", [(0, "Observed context"), (1, "Held-out context")])
    write_page(path, "Actual per-head attention", "Inspect the trained model's layer-0 attention at initialization, intermediate updates and the final update.", controls, data, r"""
function update(){let r=P[val('objective')],s=r.snapshots[Math.min(num('checkpoint'),r.snapshots.length-1)],i=num('probe'),h=num('head'),z=s.attention[i][h],tokens=s.probe_inputs[i].map(t=>r.corpus.vocabulary[t]);
show([{type:'heatmap',z,x:tokens,y:tokens,zmin:0,zmax:1,colorscale:'Blues',colorbar:{title:{text:'Weight'}}}],
{title:val('objective').toUpperCase()+' · layer 0 · head '+h+' · step '+s.step,xaxis:{title:{text:'Key position'},tickangle:-45},yaxis:{title:{text:'Query position'},autorange:'reversed'}},
{objective:val('objective'),step:s.step,head:h,row_sums:z.map(row=>row.reduce((a,b)=>a+b,0))})}
""", "Weights come from this model's actual snapshot forward pass. Attention is not causal feature importance or a complete explanation. Color range stays 0–1.")


def embedding_html(report, coords, variance, path):
    controls = control("checkpoint", "Recorded checkpoint", 0, 0, len(coords) - 1, kind="range")
    payload = {"coords": coords.tolist(), "variance": variance.tolist(), "vocabulary": report["corpus"]["vocabulary"],
               "categories": report["corpus"]["categories"], "steps": [s["step"] for s in report["snapshots"]]}
    write_page(path, "Embedding trajectories in shared PCA coordinates", "PCA is fitted once on pooled snapshots; checkpoint axes never rotate independently.", controls, payload, r"""
const colors={subject:'#287da8',verb:'#ae67a5',object:'#d87c31',special:'#788897',function:'#328c74'};
const flat=P.coords.flat(),mins=[0,1,2].map(k=>Math.min(...flat.map(p=>p[k]))),maxs=[0,1,2].map(k=>Math.max(...flat.map(p=>p[k])));
function update(){let i=num('checkpoint'),pts=P.coords[i],traces=[{type:'scatter3d',mode:'markers+text',x:pts.map(p=>p[0]),y:pts.map(p=>p[1]),z:pts.map(p=>p[2]),text:P.vocabulary,textposition:'top center',marker:{size:5,color:P.categories.map(c=>colors[c])},name:'Current snapshot'}];
for(let t=0;t<P.vocabulary.length;t++){let a=P.coords.slice(0,i+1).map(c=>c[t]);traces.push({type:'scatter3d',mode:'lines',x:a.map(p=>p[0]),y:a.map(p=>p[1]),z:a.map(p=>p[2]),line:{width:2,color:colors[P.categories[t]]},name:P.vocabulary[t],showlegend:false})}
show(traces,{title:'Measured CLM embeddings · step '+P.steps[i],scene:{xaxis:{title:'PC1',range:[mins[0]-1,maxs[0]+1]},yaxis:{title:'PC2',range:[mins[1]-1,maxs[1]+1]},zaxis:{title:'PC3',range:[mins[2]-1,maxs[2]+1]},uirevision:'camera'}},
{step:P.steps[i],explained_variance:P.variance,cumulative_3d:P.variance.reduce((a,b)=>a+b,0)})}
""", "Static token embeddings are not contextual hidden states. Predefined category colors do not establish semantic clusters; PCA discards dimensions. Neighbors are measured separately in original coordinates.")


def cross_entropy_html(path):
    write_page(path, "Cross-entropy versus target probability", "Analytical curve L = −ln(p_target), not a Transformer training trace.",
               control("probability", "Probability of reference target", 0.5, 0.0001, 0.9999, 0.0001, "range"), {}, r"""
function update(){let p=num('probability'),x=Array.from({length:500},(_,i)=>.0001+i*(.9998/499)),y=x.map(p=>-Math.log(p));
show([{x,y,mode:'lines',name:'Analytical NLL'},{x:[p],y:[-Math.log(p)],mode:'markers',marker:{size:12,color:'#d87c31'},name:'Selected probability'}],
{xaxis:{title:{text:'Target probability'},range:[0,1]},yaxis:{title:{text:'Negative log-likelihood (nats)'},range:[0,10]}},{probability:p,nll:-Math.log(p),derivative_wrt_probability:-1/p})}
""", "A confident wrong prediction assigns very little probability to the observed target and incurs large loss. This probability-space derivative differs from the logit derivative p − y.")


def perplexity_html(path):
    write_page(path, "Analytical perplexity relationship", "Perplexity = exp(mean autoregressive NLL in nats). This curve contains no measured runs.",
               control("loss", "Mean causal NLL (nats)", 1.5, 0, 6, 0.01, "range"), {}, r"""
function update(){let n=num('loss'),x=Array.from({length:301},(_,i)=>i*.02);show([{x,y:x.map(Math.exp),mode:'lines',name:'exp(NLL)'},{x:[n],y:[Math.exp(n)],mode:'markers',marker:{size:12,color:'#d87c31'},name:'Selected NLL'}],
{xaxis:{title:{text:'Mean causal NLL (nats)'},range:[0,6]},yaxis:{title:{text:'Autoregressive perplexity'},range:[0,410]}},{mean_causal_nll:n,perplexity:Math.exp(n)})}
""", "Aligned tokenizer, context and scoring policy are required for comparison. MLM reconstruction loss is not ordinary autoregressive perplexity. Lower perplexity does not establish truthfulness.")


SCALING_JS = r"""
function coeff(){return {E:num('E'),A:num('A'),B:num('B'),a:num('alpha'),b:num('beta')}}
function loss(n,d,c){return c.E+c.A/Math.pow(n/1e9,c.a)+c.B/Math.pow(d/1e9,c.b)}
function optimum(C,c){let n=Math.pow(c.a*c.A/(c.b*c.B)*Math.pow(C/6e18,c.b),1/(c.a+c.b))*1e9;return {n,d:C/(6*n),loss:loss(n,C/(6*n),c)}}
"""


def scaling_controls(scaling):
    return "".join(control(name, "Illustrative " + name, getattr(scaling, name), 0.01, 20 if name in ("E", "A", "B") else 2, 0.01)
                   for name in ("E", "A", "B", "alpha", "beta"))


def scaling_html(path, scaling, surface=True):
    controls = scaling_controls(scaling) + control("budget", "log10 training FLOPs", 22, 18, 26, 0.1, "range")
    script = SCALING_JS + r"""
function update(){let c=coeff(),C=10**num('budget'),o=optimum(C,c),ns=Array.from({length:45},(_,i)=>10**(6+i*7/44)),ds=Array.from({length:45},(_,i)=>10**(7+i*8/44));
let z=ds.map(d=>ns.map(n=>loss(n,d,c))),traces=[];
if(P.surface){traces.push({type:'surface',x:ns.map(Math.log10),y:ds.map(Math.log10),z,colorscale:'Viridis',opacity:.9,name:'Illustrative loss',colorbar:{title:{text:'Loss'}}});
for(let f of [.1,1,10]){let n=ns.filter(n=>C*f/(6*n)>=1e7&&C*f/(6*n)<=1e15),d=n.map(n=>C*f/(6*n));traces.push({type:'scatter3d',x:n.map(Math.log10),y:d.map(Math.log10),z:n.map((n,i)=>loss(n,d[i],c)),mode:'lines',line:{width:6},name:'C × '+f})}}
else {for(let f of [.1,1,10]){let n=ns,d=n.map(n=>C*f/(6*n));traces.push({x:n.map(n=>n/1e9),y:d.map(d=>d/1e9),mode:'lines',name:'C × '+f})}traces.push({x:[o.n/1e9],y:[o.d/1e9],mode:'markers',marker:{size:12,color:'#d87c31'},name:'Analytical optimum'})}
show(traces,P.surface?{title:'Illustrative scaling surface; capacity-limited left / data-limited front',scene:{xaxis:{title:'log10 parameters'},yaxis:{title:'log10 training tokens'},zaxis:{title:'Assumed loss'},uirevision:'camera'}}:
{title:'Fixed-compute contours with assumed optimum',xaxis:{title:{text:'Parameters (billions)'},type:'log'},yaxis:{title:{text:'Training tokens (billions)'},type:'log'}},
{coefficients:c,coefficient_units:'N and D in billions',training_flops:C,optimum_parameters:o.n,optimum_tokens:o.d,estimated_loss:o.loss,empirical_fit:false})}
"""
    write_page(path, "Illustrative scaling law" if surface else "Fixed-compute trade-offs", "L = E + A/(N/1e9)^alpha + B/(D/1e9)^beta. Change coefficients and budgets; no empirical constants are used.",
               controls, {"surface": surface}, script,
               "Capacity and data limitations are terms in the assumed equation, not measured regions. C≈6ND excludes attention/context, optimizer, recomputation, communication and downtime. No synthetic scaling-law fit.")


def allocation_html(path, scaling):
    controls = scaling_controls(scaling) + control("budget", "log10 training FLOPs", 22, 18, 26, 0.1, "range")
    write_page(path, "Compute-optimal allocation under an assumed law", "Substitute D=C/(6N) and derive the minimum. The highlighted point is calculated, never manually placed.",
               controls, {}, SCALING_JS + r"""
function update(){let c=coeff(),C=10**num('budget'),o=optimum(C,c),n=Array.from({length:250},(_,i)=>o.n*10**(-2+4*i/249)),d=n.map(n=>C/(6*n)),l=n.map((n,i)=>loss(n,d[i],c));
show([{x:n.map(n=>n/1e9),y:l,mode:'lines',name:'Fixed compute loss'},{x:[o.n/1e9],y:[o.loss],mode:'markers',marker:{size:13,color:'#d87c31'},name:'Derived optimum'}],
{title:'Small capacity ← allocation → insufficient token exposure',xaxis:{title:{text:'Parameters (billions)'},type:'log'},yaxis:{title:{text:'Illustrative loss'}}},
{training_flops:C,optimal_parameters:o.n,optimal_tokens:o.d,tokens_per_parameter:o.d/o.n,minimum_loss:o.loss,stationarity:'alpha*A*n^-alpha = beta*B*d^-beta'})}
""", "Normalized illustrative coefficients are adjustable and are not Chinchilla's fitted coefficients. The optimum depends on assumptions and ignores inference cost. Twenty tokens per parameter is not a universal rule.")


def compute_html(path):
    controls = (control("parameters", "Model parameters (billions)", 7, 0.001, 10000, 0.001)
                + control("tokens", "Training tokens (billions)", 140, 0.001, 1e7, 0.001)
                + control("accelerators", "Number of accelerators", 8, 1, 100000, 1)
                + control("peak", "Peak TFLOP/s per accelerator", 300, 0.01, 100000, 0.01)
                + control("utilization", "Assumed model FLOPs utilization", .4, .001, 1, .001)
                + control("cost", "Hypothetical cost per accelerator-hour", 2, 0, 10000, .01))
    write_page(path, "Analytical pretraining compute simulator", "Change all inputs locally. Every cost and time result is an assumption-based estimate, not a hardware benchmark or provider price.",
               controls, {}, r"""
function update(){let N=num('parameters')*1e9,D=num('tokens')*1e9,k=num('accelerators'),peak=num('peak')*1e12,u=num('utilization'),rate=num('cost');
let C=6*N*D,throughput=k*peak*u,seconds=C/throughput,hours=seconds/3600*k;
let counts=[1,2,4,8,16,32,64,128,256,512],days=counts.map(k=>C/(k*peak*u)/86400);
show([{x:counts,y:days,mode:'lines+markers',name:'Idealized constant-utilization duration'},{x:[k],y:[seconds/86400],mode:'markers',marker:{size:13,color:'#d87c31'},name:'Selected allocation'}],
{xaxis:{title:{text:'Accelerators'},type:'log'},yaxis:{title:{text:'Estimated days'},type:'log'}},
{analytical_estimate:true,training_flops:C,effective_flops_per_second:throughput,training_seconds:seconds,training_days:seconds/86400,accelerator_hours:hours,estimated_accelerator_cost:hours*rate,tokens_per_parameter:D/N})}
""", "Dense C≈6ND. Utilization is assumed constant and bundles inefficiency; real scaling may degrade. Attention, communication, recomputation, optimizer work, memory feasibility, checkpointing and downtime require separate assessment. Sparse/MoE models need active-parameter accounting. No inference workload is modeled.")


def corpus_html(reports, path):
    names = ["clm", "duplicated", "single_grammar", "conflicting", "domain_imbalanced"]
    data = {name: {"stats": reports[name]["dataset_statistics"], "vocabulary": reports[name]["corpus"]["vocabulary"],
                   "final": reports[name]["snapshots"][-1], "processed_tokens": reports[name]["processed_training_tokens"]} for name in names}
    write_page(path, "Controlled synthetic corpus conditions", "All conditions use the same tokenizer, model, optimizer, update count and common held-out validation distribution.",
               selector("condition", "Training condition", [(name, name.replace('_', ' ')) for name in names]), data, r"""
function update(){let r=P[val('condition')];show([{x:r.vocabulary,y:r.stats.token_counts,type:'bar',marker:{color:'#287da8'},name:'Corpus token occurrences'}],
{title:val('condition')+' · token frequencies',xaxis:{tickangle:-45},yaxis:{title:{text:'Occurrences in training pool'}}},
{condition:val('condition'),examples:r.stats.examples,unique_examples:r.stats.unique_examples,duplicate_fraction:r.stats.duplicate_fraction,grammars:r.stats.grammars,animal_examples:r.stats.animal_examples,technical_examples:r.stats.technical_examples,processed_training_tokens:r.processed_tokens,final_train_nll:r.final.train.nll,final_common_validation_nll:r.final.validation.nll})}
""", "Similar exposure is not equal information. Duplication, grammar coverage, intentional role conflicts and domain mixture change together with corpus composition; no internet-scale causal conclusion follows from one seed.")

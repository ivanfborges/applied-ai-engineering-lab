"""Standalone, offline Plotly laboratories with explicit model/data provenance."""

from html import escape
import json

import numpy as np
from plotly.offline import get_plotlyjs
from sklearn.decomposition import PCA

from from_scratch import sinusoidal_positions
from transformer_lab import shared_pca


def json_data(value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return np.round(item, 6).tolist() if np.issubdtype(item.dtype, np.floating) else item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    return json.dumps(value, default=convert, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")


def select(name, label, values):
    return f'<label for="{name}">{escape(label)}<select id="{name}">' + "".join(
        f'<option value="{escape(str(value))}">{escape(str(text))}</option>'
        for value, text in values) + '</select></label>'


COMMON = r"""
const cfg={responsive:true,displaylogo:false,typesetMath:false};
const val=id=>document.getElementById(id).value;
const byid=id=>document.getElementById(id);
const opt=(id,items)=>{const old=val(id);byid(id).replaceChildren(...items.map(([v,t])=>new Option(t,v)));if(items.some(([v])=>String(v)===old))byid(id).value=old;};
const layout=(title)=>({title:{text:title,font:{size:16}},margin:{t:60,b:55,l:60,r:20},uirevision:'keep-camera',paper_bgcolor:'#fff',plot_bgcolor:'#fff'});
const heat=(z,title)=>[{type:'heatmap',z,colorscale:'Viridis',hovertemplate:'row %{y}, column %{x}<br>value %{z:.6f}<extra></extra>'}];
function scene(title,range){return {...layout(title),margin:{t:60,b:0,l:0,r:0},scene:{xaxis:{title:{text:'PC1'},range},yaxis:{title:{text:'PC2'},range},zaxis:{title:{text:'PC3'},range},aspectmode:'cube',uirevision:'keep-camera'}};}
let running=false,dirty=false;
async function redraw(){window.labReady=false;if(running){dirty=true;return;}running=true;try{do{dirty=false;await update();}while(dirty);window.labReady=true;}catch(e){byid('summary').textContent=String(e);window.labError=String(e);throw e;}finally{running=false;}}
window.addEventListener('error',e=>{window.labError=String(e.message)});
window.addEventListener('unhandledrejection',e=>{window.labError=String(e.reason)});
"""


def page(path, title, controls, script, payload, plots=2, explanation=""):
    source = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title>
<style>body{margin:0;background:#f1f5f9;color:#172033;font:16px system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:22px}
h1{font-size:28px}p{line-height:1.6}.card{background:white;border:1px solid #cbd5e1;border-radius:10px;padding:18px;margin:14px 0}
.controls{display:flex;gap:15px;align-items:end;flex-wrap:wrap}label{display:grid;gap:6px;font-size:14px}select,button,input{font:inherit;padding:7px;max-width:230px}
button:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #2563eb}.plots{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
.plot{height:440px;min-width:0;background:white;border:1px solid #e2e8f0;border-radius:8px}#summary{white-space:pre-wrap;line-height:1.7;font-variant-numeric:tabular-nums}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #e2e8f0;padding:8px;text-align:left}footer{font-size:13px;color:#475569}
@media(max-width:760px){.plots{grid-template-columns:1fr}main{padding:12px}.plot{height:390px}}</style>
<script>__PLOTLY__</script></head><body><main><h1>__TITLE__</h1><p>__EXPLANATION__</p>
<div class="card controls">__CONTROLS__<button id="reset">Reset</button></div><div class="card" id="summary" role="status" aria-live="polite">Loading local data…</div>
<div class="plots">__PLOTS__</div><div class="card"><table id="numbers"></table></div>
<footer><p>Synthetic educational laboratory · author interpretation review required. All data and Plotly are bundled locally; no server or external model.
Attention weights are not a complete explanation. PCA distances omit unshown dimensions.</p></footer></main>
<script>const DATA=__DATA__;__COMMON____SCRIPT__
const inputs=[...document.querySelectorAll('select,input')];const defaults=inputs.map(e=>[e,e.value]);
inputs.forEach(e=>e.addEventListener('change',redraw));byid('reset').onclick=()=>{defaults.forEach(([e,v])=>e.value=v);redraw();};
redraw();</script></body></html>'''
    replacements = {"__TITLE__": escape(title), "__EXPLANATION__": escape(explanation), "__CONTROLS__": controls,
                    "__PLOTS__": "".join(f'<div id="plot{i}" class="plot"></div>' for i in range(plots)),
                    "__DATA__": json_data(payload), "__COMMON__": COMMON, "__SCRIPT__": script, "__PLOTLY__": get_plotlyjs()}
    for key, value in replacements.items():
        source = source.replace(key, value)
    path.write_text(source, encoding="utf-8")


def positions_page(path):
    controls = select("length", "Calculated positions", [(16, 16), (32, 32), (64, 64)]) + select("width", "Encoding dimensions", [(8, 8), (16, 16), (32, 32)])
    script = r"""
async function update(){const n=+val('length'),d=+val('width');const z=Array.from({length:d},(_,j)=>Array.from({length:n},(_,p)=>j%2===0?Math.sin(p/10000**(2*Math.floor(j/2)/d)):Math.cos(p/10000**(2*Math.floor(j/2)/d))));
await Plotly.react('plot0',[{type:'surface',x:Array.from({length:n},(_,i)=>i),y:Array.from({length:d},(_,i)=>i),z,colorscale:'RdBu',cmin:-1,cmax:1,hovertemplate:'position %{x}<br>feature %{y}<br>PE %{z:.6f}<extra></extra>'}],
{...layout('Original sinusoidal positional surface'),scene:{xaxis:{title:{text:'Position'}},yaxis:{title:{text:'Feature dimension'}},zaxis:{title:{text:'Encoding value'},range:[-1,1]},uirevision:'keep-camera'}},cfg);
const cosine=Array.from({length:n},(_,p)=>Array.from({length:n},(_,q)=>{let dot=0,a=0,b=0;for(let j=0;j<d;j++){dot+=z[j][p]*z[j][q];a+=z[j][p]**2;b+=z[j][q]**2;}return dot/Math.sqrt(a*b);}));
await Plotly.react('plot1',heat(cosine),{...layout('Cosine similarity of positional vectors'),xaxis:{title:{text:'Position'}},yaxis:{title:{text:'Position'}}},cfg);
byid('summary').textContent=`Mathematical visualization · recalculated in JavaScript from the original sin/cos formula. Shape ${n} × ${d}. Rotate/zoom the surface; controls recompute the mathematics, not a trained model.`;
window.labState={length:n,width:d,first:z.map(row=>row[0]),sample:z[0][1],cosine00:cosine[0][0]};
}
"""
    page(path, "02 · Position, dimension and encoding value", controls, script, {},
         explanation="Sinusoidal encoding is calculated exactly here. The trained reversal model separately uses learned positional embeddings.")


def attention_page(path, traces, title="06 · Inspect attention heads", evolution=False):
    controls = select("checkpoint", "Optimizer update", [(i, label) for i, (label, _) in enumerate(traces)])
    controls += select("mechanism", "Attention mechanism", [("encoder", "Encoder self-attention"), ("decoder", "Decoder causal self-attention"), ("cross", "Encoder-decoder cross-attention")])
    controls += select("layer", "Layer (zero based)", [(i, i) for i in range(max(a["layer"] for a in traces[0][1]["attention"]) + 1)])
    controls += select("head", "Head (zero based)", [(i, i) for i in range(len(traces[0][1]["attention"][0]["probability"]))])
    controls += select("query", "Active query", [(0, 0)])
    payload = [{"step": label, "trace": trace} for label, trace in traces]
    script = r"""
async function update(){const item=DATA[+val('checkpoint')],t=item.trace,a=t.attention.find(a=>a.mechanism===val('mechanism')&&a.layer===+val('layer')),h=+val('head'),p=a.probability[h];
opt('query',p.map((_,i)=>[i,i]));const q=+val('query');
await Plotly.react('plot0',[{type:'heatmap',z:p,zmin:0,zmax:1,colorscale:'Viridis',hovertemplate:'query %{y}, key %{x}<br>probability %{z:.6f}<extra></extra>'}],{...layout(`${a.mechanism} · layer ${a.layer} · head ${h}`),shapes:[{type:'rect',x0:-.5,x1:p[0].length-.5,y0:q-.5,y1:q+.5,line:{color:'#f97316',width:3}}],xaxis:{title:{text:'Key position'}},yaxis:{title:{text:'Query position'}}},cfg);
await Plotly.react('plot1',[{type:'bar',y:p[q],marker:{color:'#7c3aed'}}],{...layout('Selected row: actual attention probabilities'),yaxis:{range:[0,1],title:{text:'Probability'}},xaxis:{title:{text:'Key position'}}},cfg);
const stage=t.stages[t.stages.length-1];await Plotly.react('plot2',heat(stage.values),layout('Final encoder memory · actual activation values'),cfg);
await Plotly.react('plot3',[{type:'bar',y:t.probability[Math.min(q,t.probability.length-1)],marker:{color:'#d97706'}}],{...layout('Vocabulary probabilities at selected target slot'),yaxis:{range:[0,1]},xaxis:{title:{text:'Token ID'}}},cfg);
byid('summary').textContent=`${t.classification} · optimizer update ${item.step}. Shape: ${p.length} queries × ${p[0].length} keys; ${a.probability.length} heads. Source [${t.source}]. Shifted target input [${t.decoder_input}]. No linguistic head roles or alignment target imposed.`;
byid('numbers').innerHTML='<tr><th>Key position</th><th>Probability</th></tr>'+p[q].map((v,i)=>`<tr><td>${i}</td><td>${v.toFixed(6)}</td></tr>`).join('');
window.labState={checkpoint:+val('checkpoint'),step:item.step,mechanism:a.mechanism,layer:a.layer,head:h,query:q,weights:p[q],rows:p.length,columns:p[0].length};
}
"""
    page(path, title, controls, script, payload, plots=4,
         explanation="Controls select real captured per-head weights, encoder activations and vocabulary probabilities. Training snapshots use the same held-out sequence; no checkpoint is selected for best quality.")


def residual_page(path, trace):
    rows = trace["residuals"]
    fitted = PCA(n_components=3, svd_solver="full").fit(np.concatenate([row[key] for row in rows for key in ("input", "update", "sum")]))
    vectors = [{"input": (row["input"][0] - fitted.mean_) @ fitted.components_.T,
                "update": row["update"][0] @ fitted.components_.T,
                "sum": (row["sum"][0] - fitted.mean_) @ fitted.components_.T} for row in rows]
    for row in vectors:
        np.testing.assert_allclose(row["input"] + row["update"], row["sum"], atol=2e-6)
    script = r"""
async function update(){const i=+val('layer'),r=DATA.vectors[i],zero=[0,0,0],end=r.input.map((v,j)=>v+r.update[j]);
const line=(start,end,name,color)=>({type:'scatter3d',mode:'lines+markers',x:[start[0],end[0]],y:[start[1],end[1]],z:[start[2],end[2]],name,line:{color,width:8},marker:{size:5},hovertemplate:name+'<br>%{x:.4f}, %{y:.4f}, %{z:.4f}<extra></extra>'});
const range=[-DATA.limit,DATA.limit];await Plotly.react('plot0',[line(zero,r.input,'Centered input x','#2563eb'),line(r.input,end,'Linear projection of update F(x)','#db2777'),line(zero,r.sum,'Centered sum x + F(x)','#7c3aed')],scene('One shared PCA basis: translated vector addition',range),cfg);
await Plotly.react('plot1',[{type:'bar',x:['x','F(x)','x+F(x)'],y:[r.input,r.update,r.sum].map(v=>Math.hypot(...v)),marker:{color:['#2563eb','#db2777','#7c3aed']}}],layout('Projected vector lengths (not full-state norms)'),cfg);
byid('summary').textContent=`${DATA.classification}. Encoder layer ${i+1}, token 0, before normalization. PCA retained ${(DATA.variance*100).toFixed(1)}% variance over all source residual triplets. Inputs and sums are centered; updates are projected linearly without subtracting the PCA mean, preserving vector addition. Identity derivative contributes I + J_F; following normalization changes the full block Jacobian.`;
window.labState={layer:i,input:r.input,update:r.update,sum:r.sum};}
"""
    limit = float(max(abs(np.concatenate([list(row.values()) for row in vectors]).ravel()))) * 1.25
    page(path, "08 · Residual vector addition in a common projection", select("layer", "Encoder layer", [(i, i + 1) for i in range(len(rows))]), script,
         {"vectors": vectors, "variance": sum(fitted.explained_variance_ratio_), "limit": limit, "classification": trace["classification"]},
         explanation="Rotate the actual projected vectors. A branch update is added to the input; this view is before layer normalization, not the complete high-dimensional block.")


def representations_page(path, traces):
    # A common basis spans initialized/trained snapshots as well as every encoder stage.
    stages = [stage for _, trace in traces for stage in trace["stages"]]
    projection = shared_pca(stages)
    count = len(traces[0][1]["stages"])
    coordinates = projection["coordinates"].reshape(len(traces), count, len(traces[0][1]["source"]), 3)
    payload = {"coordinates": coordinates, "names": projection["names"][:count],
               "checkpoints": [step for step, _ in traces], "tokens": traces[0][1]["source"],
               "variance": projection["variance_ratio"], "range": [float(coordinates.min()) - .3, float(coordinates.max()) + .3]}
    controls = select("checkpoint", "Optimizer update", [(i, step) for i, (step, _) in enumerate(traces)])
    controls += select("stage", "Encoder stage", list(enumerate(payload["names"])))
    controls += '<button id="play">Play stages</button><button id="pause">Pause</button>'
    script = r"""
const palette=['#2563eb','#d97706','#15803d','#db2777','#7c3aed','#0891b2','#64748b','#dc2626'];let timer=null;
byid('play').onclick=()=>{if(timer)clearInterval(timer);timer=setInterval(()=>{if(running)return;byid('stage').value=(+val('stage')+1)%DATA.names.length;redraw();},900);};byid('pause').onclick=()=>{clearInterval(timer);timer=null;};
async function update(){const checkpoint=+val('checkpoint'),stage=+val('stage'),all=DATA.coordinates[checkpoint],points=all[stage],traces=[];
for(let token=0;token<points.length;token++){const path=all.slice(0,stage+1).map(s=>s[token]);traces.push({type:'scatter3d',mode:'lines+markers+text',x:path.map(v=>v[0]),y:path.map(v=>v[1]),z:path.map(v=>v[2]),text:path.map((_,i)=>i===path.length-1?`pos ${token}: ${DATA.tokens[token]}`:''),textposition:'top center',name:`position ${token}`,line:{color:palette[token%palette.length],width:5},marker:{size:4},hovertemplate:`token ID ${DATA.tokens[token]} at source position ${token}<br>%{x:.4f}, %{y:.4f}, %{z:.4f}<extra></extra>`});}
await Plotly.react('plot0',traces,scene(DATA.names[stage],DATA.range),cfg);
await Plotly.react('plot1',[{type:'scatter',mode:'markers+text',x:points.map(p=>p[0]),y:points.map(p=>p[1]),text:points.map((_,i)=>`pos ${i}`),textposition:'top center',marker:{size:14,color:points.map((_,i)=>palette[i%palette.length])}}],{...layout('Same PCA coordinates in 2D'),xaxis:{title:{text:'PC1'},range:DATA.range},yaxis:{title:{text:'PC2'},range:DATA.range}},cfg);
byid('summary').textContent=`${DATA.checkpoints[checkpoint]===0?'Untrained numerical demonstration':'Trained model observation'} · optimizer update ${DATA.checkpoints[checkpoint]} · ${DATA.names[stage]}. Shared PCA retains ${(DATA.variance.reduce((a,b)=>a+b,0)*100).toFixed(1)}% variance (PC ratios ${DATA.variance.map(v=>(v*100).toFixed(1)+'%').join(', ')}). One fit across all stages and checkpoints; axes are fixed. Branch outputs and residual states are different computational quantities; geometric proximity is not a semantic guarantee.`;
window.labState={checkpoint,stage,points,variance:DATA.variance};}
"""
    page(path, "10 · Token trajectories across encoder stages and training", controls, script, payload,
         explanation="Actual activation trajectories with consistent token colors and one PCA coordinate system. Stage and checkpoint controls select recorded computation; camera rotation and zoom change only the view.")


def memory_page(path):
    controls = select("batch", "Batch size", [(1, 1), (4, 4), (8, 8)]) + select("heads", "Attention heads", [(1, 1), (4, 4), (8, 8)])
    controls += select("bytes", "Score precision", [(4, "FP32"), (2, "FP16")])
    script = r"""
async function update(){const batch=+val('batch'),heads=+val('heads'),bytes=+val('bytes'),n=Array.from({length:9},(_,i)=>128*2**i),z=n.map(m=>n.map(source=>batch*heads*m*source*bytes/2**20));
await Plotly.react('plot0',[{type:'surface',x:n,y:n,z,colorscale:'Viridis',hovertemplate:'source %{x}<br>target %{y}<br>score memory %{z:.2f} MiB<extra></extra>'}],{...layout('Materialized cross-attention scores only'),scene:{xaxis:{title:{text:'Source length n'},type:'log'},yaxis:{title:{text:'Target length m'},type:'log'},zaxis:{title:{text:'MiB'}},uirevision:'keep-camera'}},cfg);
await Plotly.react('plot1',[{type:'scatter',mode:'lines+markers',x:n,y:n.map(v=>batch*heads*v*v*bytes/2**20),name:'Self-attention n²'},{type:'scatter',mode:'lines+markers',x:n,y:n.map(v=>batch*heads*128*v*bytes/2**20),name:'Cross-attention, target m=128'}],{...layout('Quadratic versus linear slice'),xaxis:{type:'log',title:{text:'Source length'}},yaxis:{type:'log',title:{text:'MiB'}}},cfg);
byid('summary').textContent=`Theoretical complexity estimate · B=${batch}, h=${heads}, ${bytes} bytes/value. Cross scores: B × h × m × n × bytes. At n=m=32768: ${(z[8][8]/1024).toFixed(2)} GiB. This is not total model/activation memory, measured runtime or an allocation. Fused exact attention may avoid materializing all scores while retaining dense pair arithmetic.`;
window.labState={batch,heads,bytes,maxMiB:z[8][8]};}
"""
    page(path, "14 · Source length, target length and materialized score memory", controls, script, {},
         explanation="This surface evaluates a formula for lengths 128–32,768. It never constructs large attention tensors or measures hardware performance.")


def playground_page(path, cases):
    fitted = PCA(n_components=3, svd_solver="full").fit(np.concatenate([stage["values"] for case in cases for stage in case["stages"]]))
    payload = []
    for case in cases:
        compact = {key: value for key, value in case.items() if key not in ("stages",)}
        compact["points"] = fitted.transform(case["stages"][-1]["values"])
        payload.append(compact)
    checkpoints = sorted({case["checkpoint"] for case in cases})
    lengths = sorted({len(case["source"]) for case in cases})
    controls = select("checkpoint", "Optimizer update", [(step, step) for step in checkpoints])
    controls += select("length", "Precomputed source length", [(n, n) for n in lengths])
    controls += select("positions", "Learned positions", [(1, "Enabled"), (0, "Disabled")])
    controls += select("causal", "Decoder visibility", [(1, "Causal mask"), (0, "No causal mask")])
    controls += select("norm", "Norm placement", [("post", "Post-norm"), ("pre", "Pre-norm")])
    controls += select("mechanism", "Mechanism", [("encoder", "Encoder self"), ("decoder", "Decoder self"), ("cross", "Cross-attention")])
    controls += select("layer", "Layer", [(i, i) for i in range(max(a["layer"] for a in cases[0]["attention"]) + 1)])
    controls += select("head", "Head", [(i, i) for i in range(len(cases[0]["attention"][0]["probability"]))])
    controls += select("query", "Query", [(0, 0)])
    script = r"""
async function update(){const t=DATA.find(t=>t.checkpoint===+val('checkpoint')&&t.source.length===+val('length')&&+t.positions===+val('positions')&&+t.causal===+val('causal')&&t.norm===val('norm'));
if(!t)throw new Error('No precomputed case for this selection');const a=t.attention.find(a=>a.mechanism===val('mechanism')&&a.layer===+val('layer')),h=+val('head'),p=a.probability[h],scores=a.scores[h];
opt('query',p.map((_,i)=>[i,i]));const q=+val('query'),masked=scores.map((row,i)=>row.map((value,j)=>t.causal&&a.mechanism==='decoder'&&j>i?null:value));
await Plotly.react('plot0',[{type:'scatter3d',mode:'markers+text',x:t.points.map(p=>p[0]),y:t.points.map(p=>p[1]),z:t.points.map(p=>p[2]),text:t.source.map((v,i)=>`pos ${i}: ${v}`),marker:{size:7,color:t.source.map((_,i)=>i),colorscale:'Turbo'}}],scene('Final encoder memory · shared PCA across cases',null),cfg);
await Plotly.react('plot1',[{type:'heatmap',z:masked,colorscale:'RdBu',hovertemplate:'query %{y}, key %{x}<br>scaled logit %{z:.6f}<extra></extra>'}],layout('Scaled attention logits · blank = blocked (-infinity)'),cfg);
await Plotly.react('plot2',[{type:'heatmap',z:p,zmin:0,zmax:1,colorscale:'Viridis',hovertemplate:'query %{y}, key %{x}<br>probability %{z:.6f}<extra></extra>'}],{...layout('Normalized attention probabilities'),shapes:[{type:'rect',x0:-.5,x1:p[0].length-.5,y0:q-.5,y1:q+.5,line:{color:'#d97706',width:3}}]},cfg);
const edges=[];for(let k=0;k<p[q].length;k++){if(p[q][k]>0)edges.push({type:'scatter',mode:'lines',x:[q,k],y:[0,1],line:{color:`rgba(124,58,237,${.15+.85*p[q][k]})`,width:.5+12*p[q][k]},showlegend:false,hoverinfo:'skip'});}
edges.push({type:'scatter',mode:'markers+text',x:p[q].map((_,i)=>i),y:p[q].map(()=>1),text:p[q].map((_,i)=>`key ${i}`),textposition:'top center',marker:{size:10,color:'#2563eb'},name:'Keys'});
edges.push({type:'scatter',mode:'markers+text',x:[q],y:[0],text:[`query ${q}`],textposition:'bottom center',marker:{size:13,color:'#d97706'},name:'Query'});
await Plotly.react('plot3',edges,{...layout('Connectivity · edge thickness = actual probability'),yaxis:{range:[-.4,1.4],visible:false},xaxis:{title:{text:'Position (source and target axes differ for cross-attention)'}}},cfg);
await Plotly.react('plot4',[{type:'bar',y:a.output[q],marker:{color:'#15803d'}}],{...layout('All-head output after concatenation and W_O'),xaxis:{title:{text:'Feature dimension'}},yaxis:{title:{text:'Activation'}}},cfg);
await Plotly.react('plot5',[{type:'bar',y:t.probability[Math.min(q,t.probability.length-1)],marker:{color:'#d97706'}}],{...layout('Teacher-forced vocabulary probabilities at target slot'),yaxis:{range:[0,1]},xaxis:{title:{text:'Token ID'}}},cfg);
byid('summary').textContent=`${t.classification} · ${t.counterfactual?'COUNTERFACTUAL forward: altered computation, no retraining':'Baseline configuration'}. Source [${t.source}], shifted target [${t.decoder_input}]. ${a.probability.length} heads, ${p.length} queries × ${p[0].length} keys; output ${a.output.length} × ${a.output[0].length}. Attention = softmax(QKᵀ / √d_head + mask)V; residual Add/Norm and FFN compose the layer. Controls select actual precomputed model computations. Head/query/layer select a view; camera controls only the view. Head count is fixed by the CLI configuration. PCA is lossy, not a semantic map. Instrumentation max forward error ${t.instrumentation_max_error.toExponential(2)}.`;
byid('numbers').innerHTML='<tr><th>Key</th><th>Scaled score</th><th>Probability</th></tr>'+p[q].map((v,k)=>`<tr><td>${k}</td><td>${masked[q][k]===null?'blocked (-∞)':masked[q][k].toFixed(6)}</td><td>${v.toFixed(6)}</td></tr>`).join('');
window.labState={checkpoint:t.checkpoint,length:t.source.length,positions:t.positions,causal:t.causal,norm:t.norm,mechanism:a.mechanism,layer:a.layer,head:h,query:q,weights:p[q],scores:masked[q],rows:p.length,columns:p[0].length,counterfactual:t.counterfactual};}
"""
    page(path, "16 · Transformer playground: computation, visibility and geometry", controls, script, payload, plots=6,
         explanation="An offline laboratory of real precomputed forwards. Changes to length, positions, visibility and normalization select measured activations of the same checkpoint; they do not train a new model. The trained baseline uses post-norm and learned positions.")

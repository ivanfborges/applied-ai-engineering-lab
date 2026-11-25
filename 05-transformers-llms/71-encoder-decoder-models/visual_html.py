"""Standalone Plotly pages: embedded library, computed payloads, local controls."""

import html
import json
import math

import numpy as np
from plotly.offline import get_plotlyjs

from visual_experiments import COLORS, SOURCE, TARGET, shared_pca


def builtin(value):
    if isinstance(value, np.ndarray):
        return builtin(value.tolist())
    if isinstance(value, np.generic):
        return builtin(value.item())
    if isinstance(value, dict):
        return {key: builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [builtin(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


COMMON_JS = r"""
const D=JSON.parse(document.getElementById('lab-data').textContent);
const C={encoder:'#27834c',decoder:'#2473bd',cross:'#8154af',active:'#dc791c',masked:'#a4aab2'};
const config={responsive:true,displaylogo:false};
function plot(id,data,layout={}) {return Plotly.react(id,data,{paper_bgcolor:'#f7f9fc',plot_bgcolor:'white',font:{family:'Arial',size:13,color:'#223047'},margin:{t:65,b:90,l:80,r:45},...layout},config);}
function control(id){return document.getElementById(id);}
function finish(promises){window.labReady=false;window.lastRender=Promise.all(promises).then(()=>{window.labReady=true;});}
"""


def page(path, title, description, payload, controls, charts, script):
    content = json.dumps(builtin(payload), ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>body{{margin:0;background:#f7f9fc;color:#223047;font:16px Arial,sans-serif}}main{{max-width:1400px;margin:auto;padding:24px}}
h1{{font-size:27px}}p{{line-height:1.55;max-width:1150px}}.controls{{display:flex;flex-wrap:wrap;gap:20px;padding:18px;background:white;border:1px solid #dce2eb;border-radius:8px}}
label{{display:flex;align-items:center;gap:8px}}select,input,button{{font:inherit;padding:5px}}.chart{{min-height:430px;width:100%;margin-top:14px}}.status{{white-space:pre-wrap;padding:14px;background:#eaf0f7;border-radius:6px}}
</style><script>{get_plotlyjs()}</script></head><body><main><h1>{html.escape(title)}</h1>
<p>{html.escape(description)}</p><div class="controls">{controls}</div><p id="status" class="status"></p>
{''.join(f'<div id="{name}" class="chart" role="img" aria-label="{html.escape(name)} plot"></div>' for name in charts)}
<p>Computed synthetic data, seed 271. Green: encoder; blue: decoder; purple: cross-attention; orange: intervention/current token; gray: inaccessible. Author interpretation review required.</p>
</main><script id="lab-data" type="application/json">{content}</script><script>{COMMON_JS}\n{script}</script></body></html>'''
    path.write_text(document, encoding="utf-8")


ATTENTION_JS = r"""
function update(){
 const family=control('mechanism').value,A=D[family],i=Math.min(+control('query').value,A.query_labels.length-1);
 control('query').max=A.query_labels.length-1;control('query').value=i;
 const p=A.weights[i],scores=A.logits[i],mask=A.allowed[i],q=A.query_labels[i],x=A.key_labels.map((t,j)=>`${j}: ${t}`);
 const info=x.map((t,j)=>[q,i,A.key_labels[j],j,scores[j],p[j],mask[j]?'accessible':'masked']);
 const hover='Query %{customdata[0]} (i=%{customdata[1]})<br>Key %{customdata[2]} (j=%{customdata[3]})<br>Score %{customdata[4]:.4f}<br>P %{customdata[5]:.6f}<br>%{customdata[6]}<extra></extra>';
 const color=family==='cross'?C.cross:family==='encoder'?C.encoder:C.decoder;
 const plots=[];
 plots.push(plot('probabilities',[{type:'bar',x,y:p,marker:{color:mask.map(ok=>ok?color:C.masked)},customdata:info,hovertemplate:hover}],{title:{text:`Query ${i}: ${q} · normalized probabilities`},yaxis:{title:{text:'Probability'},range:[0,1]},xaxis:{title:{text:'Key position / token'}}}));
 plots.push(plot('scores',[{type:'bar',x,y:scores,marker:{color:mask.map(ok=>ok?color:C.masked)},customdata:info,hovertemplate:hover}],{title:{text:'Raw scaled scores (masked keys still show their pre-mask score)'},yaxis:{title:{text:'QKᵀ / √d'}},xaxis:{title:{text:'Key position / token'}}}));
 const edges=[];
 for(let j=0;j<p.length;j++)edges.push({type:'scatter',x:[(p.length-1)/2,j],y:[0,1],mode:'lines',line:{color:mask[j]?color:C.masked,width:mask[j]?1+16*p[j]:1,dash:mask[j]?'solid':'dot'},opacity:mask[j]?0.15+0.85*p[j]:0.5,customdata:[info[j],info[j]],hovertemplate:hover,showlegend:false});
 edges.push({type:'scatter',x:p.map((_,j)=>j),y:p.map(()=>1),text:x,mode:'markers+text',textposition:'top center',marker:{size:14,color:mask.map(ok=>ok?C.encoder:C.masked)},customdata:info,hovertemplate:hover,showlegend:false});
 edges.push({type:'scatter',x:[(p.length-1)/2],y:[0],text:[`Q ${i}: ${q}`],mode:'markers+text',textposition:'bottom center',marker:{size:17,color:C.decoder},showlegend:false});
 plots.push(plot('connections',edges,{title:{text:'Attention-weighted token connections'},xaxis:{visible:false,range:[-0.8,p.length-0.2]},yaxis:{visible:false,range:[-0.4,1.4]},height:440}));
 if(control('vectors')){
   plots.push(plot('vectors',[{type:'heatmap',z:[A.q[i],...A.k,...A.v,A.output[i]],y:[`Q target ${i}`,...A.key_labels.map(t=>'K source '+t),...A.key_labels.map(t=>'V source '+t),`Output target ${i}`],colorscale:'RdBu',zmid:0,colorbar:{title:{text:'Feature value'}}}],{title:{text:'Actual projected Q, all source K/V and weighted output'},xaxis:{title:{text:'Projected feature'}},height:650}));
 }
 control('status').textContent=`${family} | query ${i}: ${q} | accessible keys ${mask.filter(Boolean).length}/${mask.length} | probability sum ${p.reduce((a,b)=>a+b,0).toFixed(12)}\nGray dotted connections are blocked, not low-probability valid links. Cross-attention uses random projections: no learned translation alignment.`;
 finish(plots);
}
control('mechanism').onchange=()=>{control('query').value=0;update();};control('query').oninput=update;update();
"""


def attention_page(path, data, cross_only=False):
    options = '<option value="cross">Encoder-decoder cross-attention</option>' if cross_only else '<option value="encoder">Bidirectional</option><option value="decoder">Causal</option><option value="cross">Encoder-decoder cross-attention</option>'
    controls = f'<label>Mechanism <select id="mechanism">{options}</select></label><label>Query position <input id="query" type="range" min="0" max="5" value="0" step="1"></label>'
    page(path, "Cross-attention: target Q reads source K/V" if cross_only else "Interactive attention inspector",
         "Inspect accessible and masked keys, raw scores, probabilities and weighted connections. Query positions refer to decoder inputs, including the start token. All values come from one shared NumPy implementation.",
         data, controls, ["probabilities", "scores", "connections"]+(["vectors"] if cross_only else []), ATTENTION_JS)


def geometry_payload(model):
    families = ["encoder", "decoder", "cross"]
    arrays = [model["embeddings"][f] for f in families]+[model["baseline"][f] for f in families]+[model["perturbed"][f] for f in families]
    pca = shared_pca(arrays)
    return {"explained_ratio": pca["explained_ratio"], "basis": pca["basis"], "mean": pca["mean"],
            "families": {f: {"labels": SOURCE if f != "cross" else ["<start>", *TARGET[:-1]],
                             "input": pca["coordinates"][i], "baseline": pca["coordinates"][i+3],
                             "perturbed": pca["coordinates"][i+6], "full_l2": model["deltas"][f]}
                         for i, f in enumerate(families)}}


GEOMETRY_JS = r"""
function update(){
 const f=control('family').value,A=D.families[f],traces=[],view=control('states').value;
 for(const [stage,color,symbol] of [['input',C.masked,'diamond'],['baseline',C[f],'circle'],['perturbed',C.active,'square']]){
   if((view==='context'&&stage==='perturbed')||(view==='perturbation'&&stage==='input'))continue;
   const xyz=A[stage];traces.push({type:'scatter3d',x:xyz.map(p=>p[0]),y:xyz.map(p=>p[1]),z:xyz.map(p=>p[2]),text:A.labels,mode:'markers+text',textposition:'top center',name:stage,marker:{color,size:6,symbol},customdata:A.full_l2,hovertemplate:'%{text}<br>PCs %{x:.3f}, %{y:.3f}, %{z:.3f}<br>Full-space perturbation L2 %{customdata:.6f}<extra>'+stage+'</extra>'});
 }
 for(let i=0;i<A.labels.length&&view!=='context';i++){
   const a=A.baseline[i],b=A.perturbed[i],delta=b.map((x,j)=>x-a[j]);
   traces.push({type:'scatter3d',x:[a[0],b[0]],y:[a[1],b[1]],z:[a[2],b[2]],mode:'lines',line:{color:C.active,width:5},showlegend:false,hoverinfo:'skip'});
   if(Math.hypot(...delta)>1e-10)traces.push({type:'cone',x:[b[0]],y:[b[1]],z:[b[2]],u:[delta[0]],v:[delta[1]],w:[delta[2]],anchor:'tip',sizemode:'absolute',sizeref:0.18,colorscale:[[0,C.active],[1,C.active]],showscale:false,showlegend:false,hoverinfo:'skip'});
 }
 control('status').textContent=`Shared PCA fitted once to all input, baseline and source-perturbed states. First three PCs explain ${(100*D.explained_ratio.reduce((a,b)=>a+b,0)).toFixed(2)}% of fitted variance. Hover shows original 16D L2 displacement; projected distances may shrink. No semantic cluster claim.`;
 finish([plot('geometry',traces,{title:{text:`${f}: input → context → source intervention`},scene:{xaxis:{title:{text:'PC1'}},yaxis:{title:{text:'PC2'}},zaxis:{title:{text:'PC3'}},aspectmode:'data'},height:760,uirevision:'shared-pca'})]);
}control('family').onchange=update;control('states').onchange=update;update();
"""


def geometry_page(path, model, perturbation=False):
    page(path, "Hidden representation geometry · one shared PCA basis",
         "Rotate and zoom the three-dimensional projection. Gray diamonds are input embeddings; family-colored circles are contextual states; orange squares are source-perturbed states. Arrows show baseline-to-intervention displacement, not a learned semantic trajectory.",
         geometry_payload(model), '<label>Architecture <select id="family"><option value="encoder">Encoder-only</option><option value="decoder">Decoder-only</option><option value="cross">Encoder-decoder</option></select></label><label>States <select id="states">'+
         f'<option value="context" {"selected" if not perturbation else ""}>Input and contextual states</option><option value="perturbation" {"selected" if perturbation else ""}>Baseline and perturbed states</option><option value="all">All states</option></select></label>',
         ["geometry"], GEOMETRY_JS)


GENERATION_JS = r"""
function update(){
 const mode=control('mode').value,temp=+control('temperature').value,run=D.runs.find(r=>r.mode===mode&&r.temperature===temp);
 const index=Math.min(+control('step').value,run.steps.length-1);control('step').max=run.steps.length-1;control('step').value=index;
 const s=run.steps[index],colors=D.vocab.map((_,j)=>j===s.chosen?C.active:C.decoder);
 control('status').textContent=`${mode} | temperature ${temp} | step ${index} | context: ${s.context.join(' ')}\nAppend: ${s.token} | complete rollout: ${run.steps.map(x=>x.token).join(' ')}\nFixed random bigram generator, seed 271. EOS stops decoding; the sampled sequence depends on temperature. Greedy argmax is unchanged by positive temperature.`;
 finish([plot('generation',[{type:'bar',x:D.vocab,y:s.logits,marker:{color:colors},name:'Raw logits',xaxis:'x',yaxis:'y'},
 {type:'bar',x:D.vocab,y:s.probabilities,marker:{color:colors},name:'Probabilities',xaxis:'x2',yaxis:'y2'}],
 {title:{text:'Generation temperature acts on vocabulary logits, not QK scaling'},grid:{rows:1,columns:2,pattern:'independent'},xaxis:{title:{text:'Vocabulary'}},xaxis2:{title:{text:'Vocabulary'}},yaxis:{title:{text:'Logit'}},yaxis2:{title:{text:'Probability'},range:[0,1]},showlegend:false})]);
}for(const id of ['mode','temperature','step'])control(id).oninput=update;update();
"""


def generation_page(path, data):
    controls = '<label>Decoding <select id="mode"><option value="greedy">Greedy</option><option value="sampling">Seeded sampling</option></select></label><label>Temperature <select id="temperature"><option>0.25</option><option>0.5</option><option selected>1</option><option>2</option></select></label><label>Generation step <input id="step" type="range" min="0" max="5" step="1" value="0"></label>'
    page(path, "Autoregressive generation: greedy and temperature sampling",
         "Each step looks up fixed random logits from the last token, applies softmax(logits / temperature), selects a token and appends it. The model is an untrained bigram probability generator, not GPT. Sampling uses a reset fixed RNG seed for each temperature.", data, controls, ["generation"], GENERATION_JS)


CACHE_JS = r"""
function update(){
 const layers=+control('layers').value,heads=+control('heads').value,dim=+control('dimension').value,bytes=+control('precision').value,batch=+control('batch').value;
 if(['layers','heads','dimension','batch'].some(id=>!control(id).checkValidity())){control('status').textContent='Enter positive integer dimensions within the displayed input limits.';return;}
 const lengths=Array.from({length:64},(_,i)=>(i+1)*256),memory=lengths.map(n=>2*batch*layers*heads*dim*n*bytes/2**20);
 control('status').textContent=`KV bytes = 2 × batch × layers × KV heads × head dimension × cached tokens × bytes/element. At ${lengths.at(-1)} tokens: ${memory.at(-1).toFixed(2)} MiB.\nExcludes weights, activations, allocator overhead and encoder source memory. This is theory, not a memory allocation or speed benchmark. Grouped-query attention uses KV-head count, not query-head count.`;
 finish([plot('memory',[{type:'scatter',mode:'lines',x:lengths,y:memory,line:{color:C.cross},name:'Theoretical KV cache'}],{title:{text:'Configurable theoretical decoder self-attention cache'},xaxis:{title:{text:'Cached sequence length'}},yaxis:{title:{text:'MiB'}}})]);
}for(const id of ['layers','heads','dimension','precision','batch'])control(id).oninput=update;update();
"""


def cache_page(path, cache):
    controls = ''.join(f'<label>{label} <input id="{key}" type="number" min="1" max="{maximum}" value="{value}" step="1"></label>'
                       for key, label, value, maximum in [("layers","Layers",12,128),("heads","KV heads",4,64),("dimension","Head dimension",64,512),("batch","Batch",1,128)])
    controls += '<label>Precision <select id="precision"><option value="2">float16 / bfloat16 (2 bytes)</option><option value="4">float32 (4 bytes)</option><option value="8">float64 (8 bytes)</option></select></label>'
    page(path, "KV cache memory: explicit theoretical dimensions",
         f"The numerical one-head cache experiment matched recomputed attention with maximum absolute error {cache['max_abs_error']:.3e}. This independent calculator exposes the multi-layer memory formula; it does not change the recorded numerical experiment.",
         cache, controls, ["memory"], CACHE_JS)


COMPLEXITY_JS = r"""
function counts(n,m){return {encoder:n*n,decoder:(n+m)*(n+m+1)/2,cross:n*n+m*(m+1)/2+n*m};}
function update(){
 const n=+control('source').value,m=+control('target').value,view=control('view').value,grid=D.grid;
 const names={encoder:'Encoder-only n²',decoder:'Decoder-only (n+m)(n+m+1)/2',cross:'Encoder-decoder n²+m(m+1)/2+nm'};
 const traces=[];
 for(const f of ['encoder','decoder','cross'])if(view==='all'||view===f){
  const z=grid.map(mv=>grid.map(nv=>counts(nv,mv)[f]));traces.push({type:'surface',x:grid,y:grid,z,opacity:view==='all'?0.7:1,colorscale:[[0,C[f]],[1,C[f]]],showscale:false,name:names[f],hovertemplate:`${names[f]}<br>n=%{x}, m=%{y}<br>Connections=%{z:.0f}<extra></extra>`});
  traces.push({type:'scatter3d',x:[n],y:[m],z:[counts(n,m)[f]],mode:'markers',marker:{size:7,color:C.active},name:'Selected n,m: '+names[f]});
 }
 const c=counts(n,m);control('status').textContent=`Selected n=${n}, m=${m}: encoder ${c.encoder.toLocaleString()}, decoder ${c.decoder.toLocaleString()}, encoder-decoder ${c.cross.toLocaleString()} permitted connections.\nOne instance of each relevant attention operation; no depth, FFN, width, caching or kernel effects. Encoder-only does not generate m target tokens. These are not equivalent end-to-end workloads or measured FLOPs.`;
 finish([plot('complexity',traces,{title:{text:'Source length × target length × permitted connections'},scene:{xaxis:{title:{text:'Source n'}},yaxis:{title:{text:'Target m'}},zaxis:{title:{text:'Connections'}},aspectmode:'cube'},height:760,uirevision:'complexity'})]);
}for(const id of ['source','target','view'])control(id).oninput=update;update();
"""


def complexity_page(path, quick=False):
    controls = '<label>Source n <input id="source" type="range" min="1" max="4096" value="1024" step="1"></label><label>Target m <input id="target" type="range" min="0" max="4096" value="256" step="1"></label><label>Surface <select id="view"><option value="all">All architectures</option><option value="encoder">Encoder-only</option><option value="decoder">Decoder-only</option><option value="cross">Encoder-decoder</option></select></label>'
    page(path, "Attention complexity as idealized permitted connections",
         "Rotate the surfaces, inspect individual families, and move the source/target sliders. Causal masking reduces permitted edges but does not necessarily halve a dense kernel's actual computation. Curves are exact formulas evaluated at a finite grid, not empirical benchmarks.",
         {"grid": np.linspace(1, 4096, 22 if quick else 45).tolist()}, controls, ["complexity"], COMPLEXITY_JS)


RAG_JS = r"""
function update(){
 const k=+control('candidates').value,order=[...D.cosine.keys()].sort((a,b)=>D.cosine[b]-D.cosine[a]||a-b).slice(0,k),reranked=[...order].sort((a,b)=>D.joint_scores[b]-D.joint_scores[a]||a-b);
 control('status').textContent=`Query: ${D.query}\nRetrieved: ${order.map(i=>'D'+i).join(' → ')}\nToy reranked: ${reranked.map(i=>'D'+i).join(' → ')}\nDocument vectors are independently computed and reusable. Only ${k} candidate scores are displayed; all six toy fixture scores were precomputed for this offline control. A real pipeline would score its selected candidates. No LLM or answer validation runs.`;
 finish([plot('ranking',[{type:'bar',x:order.map(i=>'D'+i),y:order.map(i=>D.cosine[i]),name:'Independent cosine',marker:{color:C.encoder},customdata:order.map(i=>D.documents[i]),hovertemplate:'%{x}: %{customdata}<br>Cosine %{y:.4f}<extra></extra>'},{type:'bar',x:order.map(i=>'D'+i),y:order.map(i=>D.joint_scores[i]),name:'Toy pair rule',marker:{color:C.cross},customdata:order.map(i=>D.documents[i]),hovertemplate:'%{x}: %{customdata}<br>Pair score %{y:.4f}<extra></extra>'}],{title:{text:'Retrieval and joint pair scoring have different score definitions'},barmode:'group',yaxis:{title:{text:'Score'}}}),
 plot('vectors',[{type:'heatmap',z:[D.query_vector,...D.document_vectors],x:D.vocabulary,y:['Query',...D.documents.map((_,i)=>'D'+i)],colorscale:'Greens',zmin:0,zmax:1}],{title:{text:'Independently normalized lexical count vectors'},xaxis:{title:{text:'Vocabulary coordinate'}}})]);
}control('candidates').oninput=update;update();
"""


def rag_page(path, data):
    page(path, "Synthetic retrieval and reranking: separate component roles",
         "This fixture simulates indexable bi-encoder representations with normalized lexical count vectors. The joint query-document rule is 0.4 × cosine + 0.4 × [bank AND loan] + 0.2 × [approved OR approval]. It is not a trained cross-encoder. Encoder-decoder cross-attention is a different operation. Generation and response checks remain conceptual.",
         data, '<label>Candidate count <input id="candidates" type="range" min="1" max="6" value="4" step="1"></label>', ["ranking", "vectors"], RAG_JS)

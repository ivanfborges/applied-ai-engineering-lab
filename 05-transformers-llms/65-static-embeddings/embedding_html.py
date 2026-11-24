"""Standalone offline HTML explorers; all model rankings use original vectors."""

from html import escape
import json
from pathlib import Path

import numpy as np
from plotly.offline import get_plotlyjs

from embedding_lab import COLORS, category, fixed_projection
from from_scratch import analogy, nearest


def json_data(value):
    return json.dumps(value, allow_nan=False, default=lambda a: a.tolist() if isinstance(a, np.ndarray) else float(a)).replace("<", "\\u003c")


def page(path, title, controls, script, caption):
    html = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title>
<style>body{margin:0;background:#f1f5f9;color:#172033;font:16px system-ui,sans-serif}
main{max-width:1320px;margin:auto;padding:24px}h1{font-size:28px;margin:4px 0 12px}
.card{background:white;border:1px solid #dbe3ef;border-radius:12px;padding:18px;margin:12px 0}
.controls{display:flex;gap:24px;align-items:center;flex-wrap:wrap}label{display:grid;gap:6px}
select,input,button{font:inherit;padding:6px}#plot{height:680px}#summary{white-space:pre-wrap;line-height:1.5}
.caption{max-width:1100px;line-height:1.6;color:#475569}input[type=range]{width:180px}
@media(max-width:600px){main{padding:10px}#plot{height:520px}}</style>
<script>__PLOTLY__</script></head><body><main><h1>__TITLE__</h1>
<p>Day 65 · Synthetic educational laboratory · Author interpretation review required</p>
<div class="card controls">__CONTROLS__</div><div class="card"><div id="plot"></div></div>
<div class="card" id="summary" role="status" aria-live="polite"></div><p class="caption">__CAPTION__</p>
</main><script>window.labReady=false;const config={responsive:true,displaylogo:false,typesetMath:false};
__SCRIPT__</script></body></html>"""
    html = html.replace("__TITLE__", escape(title)).replace("__PLOTLY__", get_plotlyjs())
    html = html.replace("__CONTROLS__", controls).replace("__CAPTION__", escape(caption)).replace("__SCRIPT__", script)
    Path(path).write_text(html, encoding="utf-8")


def embedding_space(path, model, frequencies, title=None, note="", focus="doctor"):
    words = model.fit.words
    xyz, evr, _ = fixed_projection([model.fit.vectors], 3)
    xyz = xyz[0]
    neighbors = {w: nearest(words, model.fit.vectors, model.fit.vectors[i], 5, (w,)) for i, w in enumerate(words)}
    payload = dict(words=words, xyz=xyz, norms=np.linalg.norm(model.fit.vectors, axis=1),
                   frequencies=[frequencies[w] for w in words], categories=[category(w) for w in words],
                   colors=[COLORS.get(category(w), "#94a3b8") for w in words], neighbors=neighbors, evr=evr,
                   focus=focus if focus in words else words[0])
    options = "".join(f'<option value="{i}">{escape(w)}</option>' for i, w in enumerate(words))
    controls = f'<label>Highlight word and original-space neighbors<select id="word">{options}</select></label><label><input id="labels" type="checkbox">Show word labels</label>'
    script = "const data=" + json_data(payload) + ";\n" + r"""
const select=document.getElementById('word'), labels=document.getElementById('labels');
select.value=String(data.words.indexOf(data.focus));
async function update(){window.labReady=false;const i=+select.value, selected=data.words[i];
const neighbors=data.neighbors[selected], near=new Set(neighbors.map(x=>x[0]));
const traces=[];for(const category of [...new Set(data.categories)]){
const ids=data.words.map((_,j)=>j).filter(j=>data.categories[j]===category);
traces.push({type:'scatter3d',mode:labels.checked?'markers+text':'markers',name:category,
x:ids.map(j=>data.xyz[j][0]),y:ids.map(j=>data.xyz[j][1]),z:ids.map(j=>data.xyz[j][2]),
text:ids.map(j=>data.words[j]),textposition:'top center',
marker:{color:ids.map(j=>data.colors[j]),size:ids.map(j=>j===i?13:near.has(data.words[j])?9:4),opacity:.85},
customdata:ids.map(j=>[data.norms[j],data.frequencies[j]]),
hovertemplate:'%{text}<br>norm=%{customdata[0]:.3f}<br>frequency=%{customdata[1]}<extra>%{fullData.name}</extra>'});}
await Plotly.react('plot',traces,{title:'Final trained word vectors · PCA projection',uirevision:'keep-camera',
scene:{xaxis:{title:{text:'PC1'}},yaxis:{title:{text:'PC2'}},zaxis:{title:{text:'PC3'}}},margin:{l:0,r:0,t:45,b:0}},config);
document.getElementById('summary').textContent='Selected: '+selected+' | category: '+data.categories[i]+
'\nOriginal-dimensional cosine top 5: '+neighbors.map(x=>x[0]+' '+x[1].toFixed(4)).join(', ')+
'\nPCA explained variance: '+data.evr.map(x=>(100*x).toFixed(1)+'%').join(', ');
window.labState={selected,neighbors,explainedVariance:data.evr,norm:data.norms[i]};window.labReady=true;}
select.addEventListener('change',update);labels.addEventListener('change',update);update();
"""
    page(path, title or f"{model.mode.upper()} · final embedding space", controls, script,
         "Rotate and zoom the real 3D scene. Colors are predefined synthetic categories. Highlighting uses cosine in original dimensions; PCA distances and apparent clusters are approximate. " + note)


def cosine_explorer(path):
    controls = '<label>Angle (degrees)<input id="angle" type="range" min="0" max="180" step="5" value="45"></label><label>Magnitude A<input id="a" type="range" min="0" max="4" step="0.25" value="2"></label><label>Magnitude B<input id="b" type="range" min="0" max="4" step="0.25" value="1"></label><label><input id="normalize" type="checkbox">Normalize nonzero vectors</label>'
    script = r"""
async function update(){window.labReady=false;const angle=+document.getElementById('angle').value;
const a=+document.getElementById('a').value,b=+document.getElementById('b').value;
const normalize=document.getElementById('normalize').checked,theta=angle*Math.PI/180;
const na=normalize&&a>0?1:a, nb=normalize&&b>0?1:b;
const av=[na,0],bv=[nb*Math.cos(theta),nb*Math.sin(theta)];
const dot=av[0]*bv[0],cosine=na>0&&nb>0?dot/(na*nb):null;
const distance=Math.hypot(av[0]-bv[0],av[1]-bv[1]);
const unitA=a>0?[1,0]:null,unitB=b>0?[Math.cos(theta),Math.sin(theta)]:null;
await Plotly.react('plot',[{x:[0,av[0]],y:[0,av[1]],mode:'lines+markers',name:'A',line:{width:6,color:'#2563eb'}},
{x:[0,bv[0]],y:[0,bv[1]],mode:'lines+markers',name:'B',line:{width:6,color:'#be185d'}}],
{title:'Actual vector geometry',xaxis:{range:[-4.5,4.5],title:{text:'x'}},yaxis:{range:[-4.5,4.5],scaleanchor:'x',title:{text:'y'}},margin:{t:50}},config);
document.getElementById('summary').textContent=JSON.stringify({angle,dot,cosine,distance,normA:na,normB:nb,unitA,unitB,normalized:normalize},null,2)+
'\nCosine is undefined for a zero vector. Positive scaling preserves cosine; it changes the dot product and usually distance.';
window.labState={angle,dot,cosine,distance,normA:na,normB:nb,normalized:normalize};window.labReady=true;}
for(const id of ['angle','a','b','normalize'])document.getElementById(id).addEventListener('input',update);update();
"""
    page(path, "Cosine, magnitude and normalization", controls, script,
         "Analytical vectors, not trained word embeddings. For unit vectors, squared distance = 2 - 2 cosine. No semantic probability is implied.")


def analogy_explorer(path, model):
    fit = model.fit
    matrices, evr, pca = fixed_projection([fit.vectors], 3)
    experiments = []
    for a, b, c, expected in (("man", "king", "woman", "queen"), ("france", "paris", "germany", "berlin"), ("technology", "engineer", "healthcare", None)):
        va, vb, vc = (fit.vectors[fit.words.index(w)] for w in (a, b, c))
        query = vb - va + vc
        target = pca.transform(query[None, :])[0]
        target = np.pad(target, (0, 3-len(target)))
        experiments.append({"label": f"{b} - {a} + {c}", "sources": [a,b,c], "query": target,
                            "expected": expected, "ranking": analogy(fit, a, b, c, 5)})
    payload = dict(words=fit.words, xyz=matrices[0], experiments=experiments, evr=evr)
    controls = '<label>Prespecified raw-vector analogy<select id="analogy">' + "".join(f'<option value="{i}">{escape(row["label"])}</option>' for i,row in enumerate(experiments)) + '</select></label>'
    script = "const data=" + json_data(payload) + ";\n" + r"""
async function update(){window.labReady=false;const test=data.experiments[+document.getElementById('analogy').value];
const selected=new Set([...test.sources,...test.ranking.map(x=>x[0])]);if(test.expected)selected.add(test.expected);
const ids=data.words.map((_,i)=>i).filter(i=>selected.has(data.words[i]));
const traces=[{type:'scatter3d',mode:'markers+text',name:'Trained vectors',x:ids.map(i=>data.xyz[i][0]),
y:ids.map(i=>data.xyz[i][1]),z:ids.map(i=>data.xyz[i][2]),text:ids.map(i=>data.words[i]),marker:{size:7,color:'#2563eb'}},
{type:'scatter3d',mode:'markers+text',name:'Arithmetic query',x:[test.query[0]],y:[test.query[1]],z:[test.query[2]],text:['b - a + c'],marker:{size:10,color:'#dc2626',symbol:'diamond'}}];
const pts=[test.sources[0],test.sources[1],test.sources[2]].map(w=>data.xyz[data.words.indexOf(w)]);
traces.push({type:'scatter3d',mode:'lines',name:'a → b relationship',x:[pts[0][0],pts[1][0]],y:[pts[0][1],pts[1][1]],z:[pts[0][2],pts[1][2]],line:{width:7,color:'#d97706'}});
traces.push({type:'scatter3d',mode:'lines',name:'same offset at c',x:[pts[2][0],test.query[0]],y:[pts[2][1],test.query[1]],z:[pts[2][2],test.query[2]],line:{width:7,color:'#dc2626'}});
await Plotly.react('plot',traces,{title:test.label,uirevision:'analogy',scene:{xaxis:{title:{text:'PC1'}},yaxis:{title:{text:'PC2'}},zaxis:{title:{text:'PC3'}}}},config);
document.getElementById('summary').textContent='Original-space top 5 cosine: '+test.ranking.map(x=>x[0]+' '+x[1].toFixed(4)).join(', ')+
'\nExpected fixture answer: '+(test.expected||'exploratory; no predefined correct answer')+
'\nSource words excluded. Raw vectors before cosine. PCA explained variance: '+data.evr.map(x=>(100*x).toFixed(1)+'%').join(', ');
window.labState=test;window.labReady=true;}
document.getElementById('analogy').addEventListener('change',update);update();
"""
    page(path, "Learned analogy geometry · successes and failures", controls, script,
         "Learned Skip-gram vectors, with no adjustment to force analogy success. Candidate ranking uses the original space. A visible projected gap does not equal full-dimensional error.")


def retrieval_space(path, retrieval):
    vectors = retrieval["vectors"]["skipgram"]
    xyz, evr, _ = fixed_projection([np.vstack([vectors["documents"], vectors["queries"]])], 3)
    payload = dict(xyz=xyz[0], nd=len(retrieval["documents"]), documents=retrieval["documents"],
                   queries=[q[0] for q in retrieval["queries"]], details=retrieval["details"]["skipgram"], evr=evr)
    controls = '<label>Query<select id="query">'+"".join(f'<option value="{i}">{escape(q[0])}</option>' for i,q in enumerate(retrieval["queries"]))+'</select></label>'
    script = "const data=" + json_data(payload) + ";\n" + r"""
async function update(){window.labReady=false;const i=+document.getElementById('query').value;
const row=data.details[i],n=data.nd,q=data.xyz[n+i];
await Plotly.react('plot',[{type:'scatter3d',mode:'markers+text',name:'Documents',
x:data.xyz.slice(0,n).map(x=>x[0]),y:data.xyz.slice(0,n).map(x=>x[1]),z:data.xyz.slice(0,n).map(x=>x[2]),
text:data.documents.map((_,j)=>'D'+j),customdata:data.documents,hovertemplate:'%{text}: %{customdata}<extra></extra>',
marker:{size:data.documents.map((_,j)=>row.ranking.slice(0,3).includes(j)?11:5),color:'#2563eb'}},
{type:'scatter3d',mode:'markers+text',name:'Selected query',x:[q[0]],y:[q[1]],z:[q[2]],text:[data.queries[i]],marker:{size:10,color:'#dc2626'}}],
{title:'Mean Skip-gram vectors · projected document/query space',uirevision:'retrieval',scene:{xaxis:{title:{text:'PC1'}},yaxis:{title:{text:'PC2'}},zaxis:{title:{text:'PC3'}}}},config);
document.getElementById('summary').textContent=JSON.stringify(row,null,2)+'\nAll rankings use original-dimensional vectors. An all-OOV query abstains. PCA variance: '+data.evr.map(x=>(100*x).toFixed(1)+'%').join(', ');
window.labState=row;window.labReady=true;}
document.getElementById('query').addEventListener('change',update);update();
"""
    page(path, "Synthetic retrieval: documents and queries", controls, script,
         "Judgments are manually assigned to this tiny fixture. Document IDs, original-dimensional scores and coverage remain inspectable. The zero vector for an OOV query is a storage placeholder; the retriever abstains.")

"""Self-contained Plotly pages with local JavaScript controls; no server or CDN."""

from html import escape
import json
from pathlib import Path

import numpy as np
from plotly.offline import get_plotlyjs

from attention_lab import COLORS, context_fixture, memory_estimates, value_fixture


def json_data(value):
    return json.dumps(value, allow_nan=False, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else float(x)).replace("<", "\\u003c")


def slider(name, label, minimum, maximum, step, value):
    return (f'<label for="{name}">{escape(label)}<input id="{name}" type="range" '
            f'min="{minimum}" max="{maximum}" step="{step}" value="{value}"></label>')


COMMON = r"""
const C={query:'#2563eb',key:'#d97706',value:'#15803d',output:'#7c3aed',mask:'#94a3b8'};
const cfg={responsive:true,displaylogo:false,typesetMath:false};
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
const norm=a=>Math.sqrt(dot(a,a));
const transpose=a=>a[0].map((_,j)=>a.map(row=>row[j]));
const mm=(a,b)=>a.map(row=>transpose(b).map(col=>dot(row,col)));
function soft(a){const m=Math.max(...a),e=a.map(x=>Math.exp(x-m)),s=e.reduce((x,y)=>x+y,0);return e.map(x=>x/s);}
const entropy=p=>-p.reduce((s,x)=>s+(x>0?x*Math.log(x):0),0);
function attend(q,k,v){const raw=mm(q,transpose(k)),scaled=raw.map(row=>row.map(x=>x/Math.sqrt(q[0].length)));
 const weights=scaled.map(soft);return {raw,scaled,weights,output:mm(weights,v)};}
const val=id=>+document.getElementById(id).value;
const f=x=>Number(x).toFixed(4);
function table(headers,rows){document.getElementById('numbers').innerHTML='<thead><tr>'+headers.map(x=>'<th>'+x+'</th>').join('')+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+row.map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('')+'</tbody>';}
function vector3(v,name,color){return {type:'scatter3d',mode:'lines+markers',x:[0,v[0]],y:[0,v[1]],z:[0,v[2]],name,
 line:{color,width:9},marker:{size:[2,7],color},hovertemplate:name+'<br>(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>'};}
function arrowhead3(v,color){const n=norm(v)||1;return {type:'cone',x:[v[0]],y:[v[1]],z:[v[2]],u:[v[0]/n],v:[v[1]/n],w:[v[2]/n],
 anchor:'tip',sizemode:'absolute',sizeref:.24,colorscale:[[0,color],[1,color]],showscale:false,hoverinfo:'skip',showlegend:false};}
let userCamera=null;
function scene(title,limit=3){return {title:{text:title,font:{size:17}},uirevision:'keep-camera',
 scene:{xaxis:{title:{text:'x'},range:[-limit,limit]},yaxis:{title:{text:'y'},range:[-limit,limit]},zaxis:{title:{text:'z'},range:[-limit,limit]},aspectmode:'cube',camera:userCamera||{eye:{x:1.35,y:-1.45,z:.9}}},margin:{l:0,r:0,b:0,t:60},legend:{orientation:'h'}};}
function bar(p,title){return Plotly.react('aux',[{type:'bar',x:p.map((_,i)=>'Key '+(i+1)),y:p,marker:{color:C.output}}],
 {title:{text:title},yaxis:{title:{text:'Normalized weight'},range:[0,1]},margin:{t:60,b:45,l:55,r:15}},cfg);}
let rendering=false,dirty=false;
async function redraw(){window.labReady=false;if(rendering){dirty=true;return;}rendering=true;
 try{do{dirty=false;await update();
 const plot=document.getElementById('plot');if(plot._fullLayout.scene&&!plot._attentionCameraHook){
  plot.on('plotly_relayout',event=>{if(event['scene.camera'])userCamera=JSON.parse(JSON.stringify(event['scene.camera']));});
  plot._attentionCameraHook=true;
 }
 }while(dirty);window.labReady=true;}
 catch(error){document.getElementById('summary').textContent=String(error);throw error;}
 finally{rendering=false;}}
"""


def page(path, title, controls, script, caption, payload=None):
    template = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title>
<style>body{margin:0;background:#f1f5f9;color:#172033;font:16px system-ui,sans-serif}main{max-width:1380px;margin:auto;padding:22px}
h1{font-size:27px}.card{background:white;border:1px solid #dbe3ef;border-radius:10px;padding:18px;margin:14px 0}
.controls{display:flex;gap:18px;align-items:center;flex-wrap:wrap}label{display:grid;gap:7px}input[type=range]{width:175px}
button,select{font:inherit;padding:7px}button:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #2563eb}
.plots{display:grid;grid-template-columns:1.2fr 1fr;gap:12px}.plot{min-width:0;height:520px}#summary{white-space:pre-wrap;line-height:1.6}
.tablewrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:9px;border-bottom:1px solid #dbe3ef;text-align:right}th:first-child,td:first-child{text-align:left}
.caption{line-height:1.6;color:#475569}@media(max-width:760px){main{padding:10px}.plots{grid-template-columns:1fr}.plot{height:430px}}
</style><script>__PLOTLY__</script></head><body><main><h1>__TITLE__</h1>
<p>Day 66 · Synthetic attention laboratory · Author interpretation review required</p>
<div class="card controls" id="controls">__CONTROLS__<button id="reset">Reset</button></div>
<div class="card plots"><div id="plot" class="plot"></div><div id="aux" class="plot"></div></div>
<div class="card" id="summary" role="status" aria-live="polite"></div>
<div class="card tablewrap"><table id="numbers"></table></div><p class="caption">__CAPTION__</p>
</main><script>window.labReady=false;const data=__DATA__;
__COMMON__
__SCRIPT__
const defaults=[...document.querySelectorAll('input,select')].map(el=>[el.id,el.type==='checkbox'?el.checked:el.value]);
document.querySelectorAll('input,select').forEach(el=>el.addEventListener('input',redraw));
document.getElementById('reset').addEventListener('click',()=>{if(window.stopAnimation)window.stopAnimation();defaults.forEach(([id,value])=>{const el=document.getElementById(id);if(el.type==='checkbox')el.checked=value;else el.value=value;});redraw();});
redraw();</script></body></html>"""
    for old, new in (("__TITLE__", escape(title)), ("__CONTROLS__", controls), ("__CAPTION__", escape(caption)),
                     ("__DATA__", json_data(payload)), ("__COMMON__", COMMON), ("__SCRIPT__", script)):
        template = template.replace(old, new)
    # Insert the licensed runtime last so replacements cannot touch its contents.
    Path(path).write_text(template.replace("__PLOTLY__", get_plotlyjs()), encoding="utf-8")


def geometry_page(path):
    controls = (slider("yaw", "Query yaw (degrees)", 0, 360, 5, 30) + slider("pitch", "Query elevation (degrees)", -80, 80, 5, 20)
                + slider("qnorm", "Query magnitude", .2, 3, .1, 1) + slider("kgain", "Key 1 magnitude", .2, 3, .1, 1)
                + '<label><input id="unequal" type="checkbox">Unequal key 2/3 magnitudes</label><button id="rotate">Rotate query</button>')
    script = r"""
let timer=null;window.stopAnimation=()=>{clearInterval(timer);timer=null;document.getElementById('rotate').textContent='Rotate query';};
document.getElementById('rotate').onclick=()=>{if(timer){window.stopAnimation();return;}
 document.getElementById('rotate').textContent='Pause rotation';timer=setInterval(()=>{document.getElementById('yaw').value=(val('yaw')+5)%360;redraw();},180);};
async function update(){const a=val('yaw')*Math.PI/180,b=val('pitch')*Math.PI/180;
 const q=[Math.cos(b)*Math.cos(a),Math.cos(b)*Math.sin(a),Math.sin(b)].map(x=>x*val('qnorm'));
 const gains=[val('kgain'),document.getElementById('unequal').checked ? 1.6 : 1,document.getElementById('unequal').checked ? .55 : 1,1];
 const keys=[[1,0,0],[0,1,0],[0,0,1],[-1/Math.sqrt(3),-1/Math.sqrt(3),-1/Math.sqrt(3)]].map((k,i)=>k.map(x=>x*gains[i]));
 const raw=keys.map(k=>dot(q,k)),scaled=raw.map(x=>x/Math.sqrt(3)),weights=soft(scaled),cosine=keys.map((k,i)=>raw[i]/norm(q)/norm(k));
 const angles=cosine.map(x=>Math.acos(Math.max(-1,Math.min(1,x)))*180/Math.PI);
 await Promise.all([Plotly.react('plot',[vector3(q,'Query',C.query),...keys.map((k,i)=>vector3(k,'Key '+(i+1),C.key)),arrowhead3(q,C.query),...keys.map(k=>arrowhead3(k,C.key))],scene('Rotate the scene and change vector coordinates',3.2),cfg),bar(weights,'Softmax(QKᵀ / √3)')]);
 table(['Key','Norm','Angle °','Cosine','Dot','Scaled','Weight'],keys.map((k,i)=>[i+1,f(norm(k)),f(angles[i]),f(cosine[i]),f(raw[i]),f(scaled[i]),f(weights[i])]));
 document.getElementById('summary').textContent='Query = ['+q.map(f).join(', ')+'] | norm = '+f(norm(q))+'\nCosine ignores positive norm changes; dot products and attention do not. Blue: Q; orange: K; purple: weights.';
 window.labState={query:q,keys,raw,scaled,weights,cosine,angles};}
"""
    page(path, "02 · Query–key geometry in 3D", controls, script,
         "Initial keys have unit norm. Key 1 and unequal-key controls change magnitudes without changing key directions. The 3D experiment uses d_k=3; its separate planar GIF uses d_k=2. All vectors are synthetic.")


def softmax_page(path):
    controls = "".join(slider(f"logit{i}", f"Raw logit {i+1}", -20, 20, .5, value) for i, value in enumerate((2, 1, 0, -1)))
    controls += slider("temperature", "Temperature (extra educational parameter)", .1, 5, .1, 1)
    controls += '<label>Preset<select id="preset"><option value="custom">Custom / initial</option><option value="uniform">Uniform</option><option value="moderate">Moderate</option><option value="saturated">Nearly one-hot</option><option value="large">Large common offset</option></select></label>'
    script = r"""
document.getElementById('preset').addEventListener('input',()=>{const presets={uniform:[0,0,0,0],moderate:[2,1,0,-1],saturated:[20,-20,-20,-20],large:[20,19,18,17]};
 const a=presets[document.getElementById('preset').value];if(a)a.forEach((x,i)=>document.getElementById('logit'+i).value=x);redraw();});
async function update(){const raw=[0,1,2,3].map(i=>val('logit'+i)),temperature=val('temperature');
 const scaled=raw.map(x=>x/2/temperature),m=Math.max(...scaled),stabilized=scaled.map(x=>x-m),exp=stabilized.map(Math.exp),weights=soft(scaled);
 let j2=0;weights.forEach((p,i)=>weights.forEach((r,j)=>{const derivative=p*((i===j?1:0)-r);j2+=derivative*derivative;}));
 await Promise.all([Plotly.react('plot',[{type:'bar',name:'Raw QK scores',x:['K1','K2','K3','K4'],y:raw,marker:{color:C.key}},
 {type:'bar',name:'Scaled / temperature',x:['K1','K2','K3','K4'],y:scaled,marker:{color:C.query}}],{title:{text:'d_k = 4 · scaling then temperature'},barmode:'group',yaxis:{title:{text:'Logit'}},margin:{t:65,b:45,l:55,r:15}},cfg),bar(weights,'Normalized weights · competition across keys')]);
 table(['Key','Raw','Scaled / T','Subtract max','exp(shifted)','Weight'],raw.map((x,i)=>[i+1,f(x),f(scaled[i]),f(stabilized[i]),exp[i].toExponential(3),f(weights[i])]));
 const h=entropy(weights),maximum=Math.max(...weights),jacobianNorm=Math.sqrt(j2);
 document.getElementById('summary').textContent='T = '+f(temperature)+' | max weight = '+f(maximum)+' | entropy = '+f(h)+' nats\n||d softmax / d scaled-logits||_F = '+f(jacobianNorm)+'\nAll exponentials ≤ 1 after maximum subtraction. Tiny coefficients may underflow. The derivative with respect to raw scores also has the factor 1/(√4 T).';
 window.labState={raw,scaled,stabilized,exponentials:exp,weights,entropy:h,maximum,jacobian_norm:jacobianNorm,temperature};}
"""
    page(path, "05 · Stable softmax playground", controls, script,
         "Softmax is row-wise across four keys. Temperature is additional educational control, not a required extra factor in standard SDPA. The large-offset preset demonstrates shift invariance; maximum subtraction handles much larger finite offsets in the tested Python core.")


def values_page(path):
    controls = "".join(slider(f"mass{i}", f"Value {i+1} nonnegative mass", 0, 1, .05, value) for i, value in enumerate((.4, .3, .2, .1)))
    controls += '<label>Weight source<select id="source"><option value="manual">Normalize masses</option><option value="qk">Fixed Q/K softmax</option></select></label>'
    controls += slider("vshift", "Translate every value along x", -2, 2, .1, 0)
    script = r"""
async function update(){const q=[1,.5,-.5],keys=[[1,0,0],[0,1,0],[0,0,1],[-1,-1,-1]],raw=keys.map(k=>dot(q,k));
 const fixedWeights=soft(raw.map(x=>x/Math.sqrt(3))),mass=[0,1,2,3].map(i=>val('mass'+i)),s=mass.reduce((a,b)=>a+b,0);
 const fixed=document.getElementById('source').value==='qk',weights=fixed?fixedWeights:mass.map(x=>s?x/s:.25);
 const values=data.values.map(v=>[v[0]+val('vshift'),v[1],v[2]]),weighted=values.map((v,i)=>v.map(x=>x*weights[i]));
 const output=[0,1,2].map(j=>weighted.reduce((s,v)=>s+v[j],0));
 const vectors=[...values.map((v,i)=>vector3(v,'Value '+(i+1),C.value)),...weighted.map((v,i)=>vector3(v,'α'+(i+1)+'V'+(i+1),C.mask)),vector3(output,'Output',C.output)];
 vectors.push({type:'mesh3d',x:values.map(v=>v[0]),y:values.map(v=>v[1]),z:values.map(v=>v[2]),i:[0,0,0,1],j:[1,1,2,2],k:[2,3,3,3],opacity:.12,color:C.value,name:'Convex hull',hoverinfo:'skip'});
 await Promise.all([Plotly.react('plot',vectors,scene('Values, weighted arrows and convex hull',4),cfg),bar(weights,'Normalized mixture coefficients')]);
 table(['Value','Weight','x','y','z','Weighted x'],values.map((v,i)=>[i+1,f(weights[i]),...v.map(f),f(weighted[i][0])]));
 document.getElementById('summary').textContent='O = ['+output.map(f).join(', ')+'] | Σ weights = '+f(weights.reduce((a,b)=>a+b,0))+'\n'+(fixed?'Q/K fixed: scores = ['+raw.map(f).join(', ')+']. Value translations leave scores and weights unchanged.':'Mass controls are normalized. All-zero masses explicitly fall back to uniform weights.')+'\nGreen: V; gray: αV; purple: O. Convex hull applies before dropout and output projection.';
 window.labState={raw,fixedWeights,weights,values,weighted,output,fixed,uniformFallback:!fixed&&s===0};}
"""
    page(path, "06 · Weighted values and output geometry", controls, script,
         "The tetrahedron is the convex hull of four synthetic 3D values. Each output is a nonnegative normalized mixture. In fixed Q/K mode only V changes: use translation to inspect a different output with identical scores and weights.", dict(values=value_fixture()))


def context_page(path):
    contexts = []
    for financial in (False, True):
        fixture = context_fixture(financial)
        contexts.append(dict(words=fixture["words"], x=fixture["x"], bank=fixture["bank"],
                             query=fixture["attention"].query, key=fixture["attention"].key, value=fixture["attention"].value))
    controls = '<label>Engineered context<select id="context"><option value="0">River bank</option><option value="1">Financial bank</option></select></label>'
    controls += '<label><input id="outputs" type="checkbox" checked>Show contextual outputs</label>'
    script = r"""
async function update(){const d=data.contexts[val('context')],a=attend(d.query,d.key,d.value),show=document.getElementById('outputs').checked;
 const neutral=d.x.map((v,i)=>norm(v)===0?i:-1).filter(i=>i>=0),bankOutput=a.output[d.bank];
 const labels=d.words.map((word,i)=>i===d.bank||word==='river'||word==='money'?word:i===neutral[0]?'neutral ×'+neutral.length:'');
 const input={type:'scatter',mode:'markers+text',name:'Before attention',x:d.x.map(v=>v[0]),y:d.x.map(v=>v[1]),text:labels,customdata:d.words,
 hovertemplate:'%{customdata}<br>(%{x:.3f}, %{y:.3f})<extra>Input</extra>',textposition:'bottom right',marker:{color:C.value,size:9}};
 const out={type:'scatter',mode:'markers+text',name:'After attention (all rows identical)',x:[bankOutput[0]],y:[bankOutput[1]],text:['bank output'],textposition:'top right',marker:{color:C.output,size:10}};
 const positions=d.words.map((word,i)=>i+': '+word);
 await Promise.all([Plotly.react('plot',show?[input,out]:[input],{title:{text:'Direct synthetic axes · water / finance'},xaxis:{range:[-.2,2.5],title:{text:'Engineered water feature'}},yaxis:{range:[-.2,2.3],title:{text:'Engineered finance feature'}},legend:{orientation:'h',y:-.25},margin:{t:65,b:110,l:60,r:25}},cfg),
 Plotly.react('aux',[{type:'heatmap',z:a.weights,x:positions,y:positions,zmin:0,zmax:1,colorscale:[[0,'#f5f3ff'],[1,C.output]],colorbar:{title:{text:'Weight'}}}],{title:{text:'Calculated attention matrix'},yaxis:{autorange:'reversed'},xaxis:{tickangle:-45},margin:{t:65,b:120,l:120,r:50}},cfg)]);
 table(['Token position','Q','K','V','Bank-query weight'],d.words.map((word,i)=>[positions[i],d.query[i].map(f).join(', '),d.key[i].map(f).join(', '),d.value[i].map(f).join(', '),f(a.weights[d.bank][i])]));
 document.getElementById('summary').textContent=d.words.join(' ')+'\nBank before = ['+d.x[d.bank].map(f).join(', ')+']; after = ['+a.output[d.bank].map(f).join(', ')+']\nEvery Q is deliberately [1,1]; K=2X and V=X. This demonstrates designed aggregation, not learned language understanding.';
 window.labState={context:val('context'),bank:d.bank,weights:a.weights,output:a.output,input:d.x};}
"""
    page(path, "07 · Engineered contextual aggregation", controls, script,
         "Features explicitly encode water/finance information. No trained model or semantic benchmark is involved. All queries are identical in this deliberately simplified fixture, so output rows within a context are identical.", dict(contexts=contexts))


def memory_page(path):
    controls = slider("length", "Sequence length (theory only)", 128, 16384, 128, 2048)
    script = r"""
async function update(){const n=val('length'),pairs=n*n,fp32=pairs*4/2**20,fp16=pairs*2/2**20;
 const lengths=data.lengths;
 await Promise.all([Plotly.react('plot',[{type:'scatter',mode:'lines+markers',x:lengths,y:data.interactions,name:'n² interactions',line:{color:C.output}},
 {type:'scatter',mode:'markers',x:[n],y:[pairs],name:'Selected',marker:{size:13,color:C.query}}],{title:{text:'No matrices allocated by this explorer'},xaxis:{title:{text:'Tokens n'},type:'log',dtick:1},yaxis:{title:{text:'Pairs per head/example'},type:'log',dtick:1},legend:{orientation:'h',y:1.13},margin:{t:90,b:70,l:75,r:15}},cfg),
 Plotly.react('aux',[{type:'bar',x:['FP32','FP16'],y:[fp32,fp16],marker:{color:[C.query,C.key]}}],{title:{text:'One materialized matrix · MiB'},yaxis:{title:{text:'MiB'},range:[0,1100]},margin:{t:65,b:60,l:70,r:15}},cfg)]);
 table(['Tokens','Shape','Pairs','FP32 MiB','FP16 MiB'],[[n,n+' × '+n,pairs,f(fp32),f(fp16)]]);
 document.getElementById('summary').textContent='One score OR weight matrix only. Excludes other arrays, gradients, projections and model parameters.\nDense arithmetic remains quadratic for fixed width. FlashAttention reduces materialization and memory traffic; no optimized kernel is benchmarked here.';
 window.labState={length:n,interactions:pairs,fp32_mib:fp32,fp16_mib:fp16};}
"""
    page(path, "11 · Attention interactions and memory", controls, script,
         "FP16 is a byte-size estimate, not a NumPy float16 execution or latency measurement. The full CPU timing experiment is separate and records its actual environment and repetitions.", memory_estimates())

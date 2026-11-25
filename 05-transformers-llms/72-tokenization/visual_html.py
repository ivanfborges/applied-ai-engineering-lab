"""Standalone Plotly laboratories; all scripts embedded, with no remote runtime."""

import html
import json
from pathlib import Path

from plotly.offline import get_plotlyjs

from visual_core import (PRESETS, PARALLEL, RAG_DOCUMENT, compare_text, chunk_document,
                         chunk_metrics)


DESCRIPTIONS = {
    "inference": ("03 · BPE inference", "Step through actual stored merge priorities; unsupported café has an explicit failure state."),
    "association": ("04 · BPE and WordPiece-style scores", "Synthetic marginal and joint counts. Association scoring is an educational reconstruction, not the original WordPiece trainer."),
    "unigram": ("05 · Unigram lattice", "Change manual probability weights; normalized probabilities sum to one. Viterbi and exhaustive paths are recomputed in your browser."),
    "vocabulary": ("09 · Vocabulary trade-off", "Measured training and held-out lengths. Unsupported words are excluded from compression denominators and separately reported."),
    "embedding": ("10 · Embedding parameters", "Theoretical weight-only storage. Input/output tying shares weights; projection computation remains vocabulary-dependent."),
    "attention": ("11 · Attention scaling", "Theoretical per-head interaction and n²d proxies. No GPU memory or latency measurement; FlashAttention can avoid materializing scores."),
    "multilingual": ("12 · Multilingual probes", "Approximately equivalent handwritten passages. Results describe these samples, not language-wide fairness or quality."),
    "ocr": ("13 · Synthetic OCR experiment", "Actual counts across controlled mechanisms and seeded trials. Inflation may be negative; retrieval and answers are not evaluated."),
    "rag": ("14 · Token-aware chunking", "Measured precomputed parameter grid; token chunks preserve whole Unicode code points. Evidence containment is not retrieval accuracy."),
    "context": ("15 · Context window simulator", "Hypothetical decoder capacity; actual usage and effective reasoning depend on the model. Chunk pruning removes lowest-priority retrieval counts first."),
    "cost": ("16 · Cost simulator", "All rates are hypothetical, per million tokens. Context reduction does not establish preserved answer quality or real savings."),
    "monthly": ("16 · Monthly cost surface", "Hypothetical monthly requests × input tokens with configurable output and prices. Excludes cache, tools, reasoning categories and infrastructure."),
}


def payload_for(kind, data, local):
    payload = {"kind": kind, "data": data}
    if kind == "rag":
        payload["document"] = RAG_DOCUMENT
        payload["grid"] = []
        for encoding in ("cl100k_base", "o200k_base"):
            for size in (24, 40, 64):
                for overlap in (0, 4, 8):
                    for strategy in ("characters", "tokens", "sentences"):
                        chunks = chunk_document(RAG_DOCUMENT, encoding, strategy, size, overlap)
                        payload["grid"].append({"encoding": encoding, "size": size, "overlap": overlap,
                                                "strategy": strategy, "chunks": chunks,
                                                "metrics": chunk_metrics(chunks, size,
                                                    "o200k_base" if encoding == "cl100k_base" else "cl100k_base")})
    if kind == "multilingual":
        payload["fragmentation"] = {enc: {lang: compare_text(enc, text, local) for lang, text in PARALLEL.items()}
                                      for enc in ("cl100k_base", "o200k_base")}
    return payload


STYLE = """
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#193246;font:16px system-ui,sans-serif}
header{padding:26px 5vw 18px;background:#142e40;color:#fff}header small{color:#b8d6e7;letter-spacing:.1em}h1{margin:8px 0;font-size:28px}
header p{max-width:1100px;line-height:1.5;color:#dae5ee}.layout{display:grid;grid-template-columns:285px minmax(0,1fr);gap:20px;padding:24px 4vw}
aside,.panel{background:white;border:1px solid #dce5ee;border-radius:12px;padding:20px}label{display:block;font-size:14px;margin:0 0 18px}label span{float:right;color:#1a7193;font-weight:600}
input,select,button{font:inherit}input[type=range],input[type=number],select{width:100%;margin-top:8px}input[type=number],select{padding:7px;border:1px solid #c1d0dc;border-radius:5px}
button{padding:9px 15px;background:#216e93;color:#fff;border:0;border-radius:5px;margin:0 6px 12px 0;cursor:pointer}button:hover{background:#124d6b}
.chart{min-height:500px;width:100%}.state{white-space:pre-wrap;font:14px ui-monospace,monospace;background:#f5f8fb;padding:16px;max-height:480px;overflow:auto;line-height:1.6}
.tokens{display:flex;flex-wrap:wrap;gap:7px;margin:16px 0}.token{color:white;padding:7px 10px;border-radius:6px;font:14px ui-monospace,monospace;max-width:100%;overflow-wrap:anywhere}
.limits{color:#526779;font-size:13px;line-height:1.55}.kpi{font-size:17px;padding:10px 0;color:#1a7193}a{color:#2878a1}@media(max-width:850px){.layout{grid-template-columns:1fr}.chart{min-height:420px}}
"""


SCRIPT = r"""
const P=JSON.parse(document.getElementById('lab-data').textContent),D=P.data,K=P.kind;
const controls=document.getElementById('controls'),state=document.getElementById('state'),metrics=document.getElementById('metrics');
const palette=['#2878a1','#cc6a33','#37856b','#9765a3','#b18c21','#b85472'];
const value=id=>document.getElementById(id).value;
const num=id=>Number(value(id));
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
let activeTimer=null, defaults={};
function slider(id,label,min,max,initial,step=1){controls.insertAdjacentHTML('beforeend',`<label for="${id}">${esc(label)} <span id="${id}-value"></span><input type="range" id="${id}" min="${min}" max="${max}" value="${initial}" step="${step}"></label>`);defaults[id]=String(initial)}
function number(id,label,initial,min=0,max=1e9,step=1){controls.insertAdjacentHTML('beforeend',`<label for="${id}">${esc(label)}<input type="number" id="${id}" min="${min}" max="${max}" step="${step}" value="${initial}"></label>`);defaults[id]=String(initial)}
function select(id,label,options,initial=options[0]){controls.insertAdjacentHTML('beforeend',`<label for="${id}">${esc(label)}<select id="${id}">${options.map(o=>`<option value="${esc(o)}" ${o===initial?'selected':''}>${esc(o)}</option>`).join('')}</select></label>`);defaults[id]=String(initial)}
async function tokens(pieces){const indices=await Promise.all(pieces.map(async p=>{const bytes=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(String(p)));return new DataView(bytes).getUint32(0)%palette.length}));document.getElementById('tokens').innerHTML=pieces.map((p,i)=>`<span class="token" style="background:${palette[indices[i]]}">${esc(p).replaceAll(' ','·').replaceAll('\n','↵')}</span>`).join('')}
function plot(traces,layout={}){return Plotly.react('chart',traces,{paper_bgcolor:'#fff',plot_bgcolor:'#fff',font:{family:'system-ui',color:'#193246'},margin:{t:65,l:80,r:35,b:80},autosize:true,...layout},{responsive:true,displaylogo:false}).then(()=>{window.labReady=true})}
function lines(x,y,name,color){return {x,y,name,mode:'lines+markers',line:{color}}}
function detail(obj){window.labState=obj;state.textContent=JSON.stringify(obj,null,2)}
function surface(x,y,z,title,xlabel,ylabel,zlabel){return plot([{type:'surface',x,y,z,colorscale:'Viridis',colorbar:{title:{text:zlabel}}}],{title,scene:{xaxis:{title:{text:xlabel}},yaxis:{title:{text:ylabel}},zaxis:{title:{text:zlabel}},camera:{eye:{x:1.65,y:1.65,z:1.1}}}})}
function setup(){
 if(K==='inference'){select('word','Encoding target',Object.keys(D.inference));slider('step','Algorithm step',0,8,0)}
 if(K==='association'){slider('jointA','Joint frequency A',10,100,100);slider('leftB','Left-symbol frequency B',40,10000,40,40)}
 if(K==='unigram'){for(const [piece,p] of Object.entries(D.unigram.probabilities))slider('p-'+piece,piece+' probability weight',1,100,Math.round(p*100));slider('step','DP end position / backtrack',1,13,13)}
 if(K==='vocabulary')select('partition','Partition',['both','train','held_out']);
 if(K==='embedding'){slider('vocab','Vocabulary size',16000,250000,50000,1000);slider('width','Embedding dimension',512,8192,4096,128);select('precision','Ideal weight storage',['FP32','FP16/BF16','INT8']);select('tying','Input/output weights',['tied','untied'])}
 if(K==='attention'){slider('length','Sequence length n',128,4096,1024,128);slider('head','Head dimension',16,256,64,16);select('workload','Interactions',['dense','causal','one cached query'])}
 if(K==='multilingual'){select('encoding','Pretrained encoding',['both','cl100k_base','o200k_base']);select('language','Inspect fragmentation',Object.keys(P.fragmentation.cl100k_base))}
 if(K==='ocr'){select('encoding','Encoding',['cl100k_base','o200k_base']);select('passage','Passage',['technical','legal']);select('mechanism','Corruption mechanism',['substitution','insert_space','break_word','remove_space','deletion']);slider('rate','Corruption probability',0,30,15,5);slider('trial','Seeded trial',0,Math.max(...D.ocr.map(r=>r.trial)),0)}
 if(K==='rag'){select('encoding','Embedding tokenizer',['cl100k_base','o200k_base']);select('strategy','Chunking rule',['characters','tokens','sentences'],'tokens');select('size','Limit (code points or tokens)',['24','40','64'],'40');select('overlap','Overlap in strategy units',['0','4','8'],'4')}
 if(K==='context'){select('limit','Hypothetical context capacity',['8192','32768','128000','1000000'],'32768');for(const [id,label,n] of [['system','System',1500],['history','History',4000],['query','Query',300],['retrieval','Retrieved context',14000],['tools','Tools / outputs',500],['overhead','Structural overhead',200],['output','Reserved output',4000]])number(id,label,n,0,1000000);select('prune','Remove lower-priority retrieval chunks',['off','on'])}
 if(K==='cost'||K==='monthly'){number('input','Input tokens/request',6000,0,1000000);number('output','Output tokens/request',1000,0,1000000);number('pin','Hypothetical input USD/million',2,0,1000,.1);number('pout','Hypothetical output USD/million',8,0,1000,.1);number('daily','Daily requests',333,0,10000000);number('monthly','Monthly requests',10000,0,100000000);slider('reduction','Input-context reduction (%)',0,80,0,5)}
 controls.insertAdjacentHTML('beforeend','<button id="play">Play / pause</button><button id="reset">Reset</button><p class="limits">All controls use local calculations or measured parameter grids. No API request or remote script is required.</p>');
 for(const e of controls.querySelectorAll('input,select'))e.addEventListener('input',render);
 document.getElementById('reset').onclick=()=>{clearInterval(activeTimer);activeTimer=null;for(const [id,v] of Object.entries(defaults))document.getElementById(id).value=v;render()};
 document.getElementById('play').onclick=()=>{if(activeTimer){clearInterval(activeTimer);activeTimer=null;return}const input=controls.querySelector('input[type=range]');if(!input)return;activeTimer=setInterval(()=>{let n=Number(input.value)+Number(input.step);input.value=n>Number(input.max)?input.min:n;render()},1000)};
}
async function render(){window.labReady=false;metrics.textContent='';document.getElementById('tokens').innerHTML='';
 for(const e of controls.querySelectorAll('input[type=range]')){document.getElementById(e.id+'-value').textContent=e.value}
 const numeric=[...controls.querySelectorAll('input[type=number]')];if(numeric.some(e=>!e.checkValidity()||!Number.isFinite(Number(e.value)))){metrics.textContent='Enter finite values within the displayed bounds.';window.labReady=true;return}
 if(K==='inference'){
  const frames=D.inference[value('word')],step=Math.min(num('step'),frames.length-1),s=frames[step];await tokens(s.pieces);detail(s);
  await plot([{type:'bar',x:s.candidates.map(c=>c.pair.join('+')),y:s.candidates.map(c=>c.rank),marker:{color:palette[0]}}],{title:'Available stored pair ranks (smaller has priority)',yaxis:{title:{text:'Learned rank'}}});
 }else if(K==='association'){
  const frequency=[num('jointA'),30],association=[frequency[0]/(1000*500),30/(num('leftB')*50)];detail({frequency,association,bpe_winner:frequency[0]>=30?'A':'B',association_winner:association[0]>=association[1]?'A':'B'});
  await plot([{type:'bar',x:['A','B'],y:frequency,name:'Joint count',yaxis:'y',marker:{color:palette[0]}},{type:'bar',x:['A','B'],y:association,name:'Association',yaxis:'y2',marker:{color:palette[1]}}],{title:'Different units: count versus association',barmode:'group',yaxis:{title:{text:'Joint count'}},yaxis2:{title:{text:'Association score'},overlaying:'y',side:'right'},margin:{t:65,l:80,r:100,b:80}});
 }else if(K==='unigram'){
  const probs={},keys=Object.keys(D.unigram.probabilities),total=keys.reduce((s,k)=>s+num('p-'+k),0);keys.forEach(k=>probs[k]=num('p-'+k)/total);
  const text=D.unigram.text,n=text.length,best=Array(n+1).fill(-Infinity),prev=Array(n+1).fill(null),edges=[];best[0]=0;
  for(let end=1;end<=n;end++)for(const piece of keys.sort()){let start=end-piece.length;if(start>=0&&text.slice(start,end)===piece){let score=best[start]+Math.log(probs[piece]);edges.push({start,end,piece,p:probs[piece],score:Number.isFinite(score)?score:null});if(score>best[end]){best[end]=score;prev[end]=[start,piece]}}}
  const path=[];let at=n;while(at>0){path.unshift(prev[at][1]);at=prev[at][0]}
  const pos=Math.min(num('step'),n),traces=edges.map((e,i)=>({x:[e.start,(e.start+e.end)/2,e.end],y:[0,.3+(e.end-e.start)*.08,0],mode:'lines+text',text:['',e.piece+' p='+e.p.toFixed(3),''],textposition:'top center',line:{color:e.end<=pos?palette[0]:'#d0dbe4',shape:'spline'},showlegend:false,hovertext:'log p='+Math.log(e.p).toFixed(4)}));
  traces.push({x:Array.from({length:n+1},(_,i)=>i),y:Array(n+1).fill(0),mode:'markers+text',text:Array.from({length:n+1},(_,i)=>String(i)),textposition:'bottom center',marker:{color:palette[1],size:10},showlegend:false});
  const paths=D.unigram.paths.map(p=>({pieces:p.pieces,probability:p.pieces.reduce((s,k)=>s*probs[k],1),log_probability:p.pieces.reduce((s,k)=>s+Math.log(probs[k]),0)}));
  await tokens(path);detail({probabilities:probs,sum:Object.values(probs).reduce((a,b)=>a+b,0),step:pos,best:best.map(v=>Number.isFinite(v)?v:null),best_predecessor:prev[pos],selected:path,probability:Math.exp(best[n]),log_probability:best[n],paths});
  await plot(traces,{title:'Candidate paths; step '+num('step')+(num('step')>n?' · backtrack best path':''),xaxis:{title:{text:'Character boundary position'}},yaxis:{visible:false,range:[-.3,1.9]}});
 }else if(K==='vocabulary'){
  const groups=value('partition')==='both'?['train','held_out']:[value('partition')];const traces=groups.map((g,i)=>{const r=D.vocabulary.filter(r=>r.partition===g);return lines(r.map(r=>r.vocabulary),r.map(r=>r.mean_tokens_per_supported_word),g,palette[i])});detail(D.vocabulary.filter(r=>groups.includes(r.partition)));
  await plot(traces,{title:'Actual vocabulary versus supported tokens/word',xaxis:{title:{text:'Vocabulary entries'}},yaxis:{title:{text:'Mean tokens / supported word'}}});
 }else if(K==='embedding'){
  const multiplier=value('tying')==='tied'?1:2,bytes={'FP32':4,'FP16/BF16':2,'INT8':1}[value('precision')],x=Array.from({length:20},(_,i)=>16000+i*(250000-16000)/19),y=Array.from({length:17},(_,i)=>512+i*(8192-512)/16),parameters=num('vocab')*num('width')*multiplier;
  detail({vocabulary:num('vocab'),width:num('width'),parameters,ideal_weight_bytes:parameters*bytes,decimal_GB:parameters*bytes/1e9,GiB:parameters*bytes/2**30,excluded:'optimizer, gradients, activations, quantization metadata, other parameters'});metrics.textContent=`Selected: ${parameters.toLocaleString()} parameters · ${(parameters*bytes/1e9).toFixed(3)} GB ideal weights`;
  await surface(x,y,y.map(d=>x.map(v=>v*d*multiplier)),'Embedding + output weights','Vocabulary entries','Embedding dimension','Parameters');
 }else if(K==='attention'){
  const n=num('length'),h=num('head'),mode=value('workload'),f=x=>mode==='dense'?x*x:mode==='causal'?x*(x+1)/2:x,x=Array.from({length:25},(_,i)=>128+i*(4096-128)/24),y=Array.from({length:17},(_,i)=>16+i*15);
  detail({length:n,head_dimension:h,workload:mode,interactions:f(n),proxy:f(n)*h,interpretation:'core interaction × width proxy, not end-to-end FLOPs or latency'});
  await surface(x,y,y.map(d=>x.map(n=>f(n)*d)),mode+' interaction × head dimension proxy','Sequence length n','Head dimension','Proxy units');
 }else if(K==='multilingual'){
  const names=value('encoding')==='both'?['cl100k_base','o200k_base']:[value('encoding')];const traces=names.map((name,i)=>{let r=D.multilingual.filter(r=>r.name===name);return {type:'bar',name,x:r.map(r=>r.language),y:r.map(r=>r.tokens),marker:{color:palette[i]}}});const frag=P.fragmentation[names[0]][value('language')];await tokens(frag.pieces);detail({metrics:D.multilingual.filter(r=>names.includes(r.name)),fragmentation:frag});
  await plot(traces,{title:'Selected approximately equivalent translations',barmode:'group',yaxis:{title:{text:'Actual tokens'}}});
 }else if(K==='ocr'){
  const rows=D.ocr.filter(r=>r.encoding===value('encoding')&&r.passage===value('passage')&&r.mechanism===value('mechanism')),rates=[...new Set(rows.map(r=>r.rate))];
  let rate=rates.reduce((a,b)=>Math.abs(b-num('rate')/100)<Math.abs(a-num('rate')/100)?b:a);let chosen=rows.find(r=>r.rate===rate&&r.trial===num('trial'));detail({chosen,rate_note:'Nearest measured rate from 0, .05, .15, .30; no interpolation of token counts'});metrics.textContent=chosen.text;
  await plot(rates.map((r,i)=>({type:'box',name:String(r),y:rows.filter(x=>x.rate===r).map(x=>x.inflation_pct),marker:{color:palette[i]}})),{title:'Measured inflation distributions by corruption rate',yaxis:{title:{text:'100 × (noisy - clean) / clean (%)'}},xaxis:{title:{text:'Measured corruption rate'}}});
 }else if(K==='rag'){
  const row=P.grid.find(r=>r.encoding===value('encoding')&&r.strategy===value('strategy')&&r.size===num('size')&&r.overlap===num('overlap'));detail(row);metrics.textContent=`${row.metrics.chunks} chunks · ${row.metrics.token_limit_violations} embedding limit violations · evidence contained: ${row.metrics.evidence_contained}`;
  await tokens(row.chunks.map(c=>c.text));
  await plot(row.chunks.map((c,i)=>({type:'bar',orientation:'h',x:[c.end-c.start],base:[c.start],y:['chunk '+(i+1)],name:c.tokens+' tokens',marker:{color:palette[i%6]},hovertext:esc(c.text),showlegend:false})),{title:'Original document spans; overlap duplicates evidence positions',xaxis:{title:{text:'Document code-point offset'}},yaxis:{autorange:'reversed'}});
 }else if(K==='context'){
  let names=['system','history','query','retrieval','tools','overhead','output'],counts=names.map(num),limit=num('limit'),removed=0;if(value('prune')==='on'){const chunk=1000;while(counts.reduce((a,b)=>a+b,0)>limit&&counts[3]>0){counts[3]=Math.max(0,counts[3]-chunk);removed++}}
  const allocated=counts.reduce((a,b)=>a+b,0),remaining=limit-allocated;detail({counts:Object.fromEntries(names.map((n,i)=>[n,counts[i]])),limit,allocated,remaining,utilization_pct:100*allocated/limit,overflow:Math.max(0,-remaining),must_remove:Math.max(0,-remaining),removed_lower_priority_chunks:removed,fits:remaining>=0});metrics.textContent=`${allocated.toLocaleString()} / ${limit.toLocaleString()} allocated · ${remaining.toLocaleString()} remaining`;
  await plot(names.map((n,i)=>({type:'bar',orientation:'h',name:n,x:[counts[i]],y:['context'],marker:{color:palette[i%6]}})),{title:'Hypothetical shared decoder context',barmode:'stack',xaxis:{title:{text:'Tokens'},range:[0,Math.max(limit,allocated)*1.05]},shapes:[{type:'line',x0:limit,x1:limit,y0:0,y1:1,yref:'paper',line:{color:'#a12637',dash:'dash'}}]});
 }else if(K==='cost'||K==='monthly'){
  const input=num('input')*(1-num('reduction')/100),out=num('output'),pin=num('pin'),pout=num('pout'),per=(input*pin+out*pout)/1e6,before=(num('input')*pin+out*pout)/1e6,monthly=per*num('monthly');detail({input_tokens:input,output_tokens:out,input_cost:input*pin/1e6,output_cost:out*pout/1e6,per_request:per,daily:per*num('daily'),monthly,before_monthly:before*num('monthly'),projected_difference:(before-per)*num('monthly'),status:'hypothetical; quality and production savings unmeasured'});metrics.textContent=`USD ${per.toFixed(5)}/request · USD ${monthly.toFixed(2)}/month`;
  if(K==='monthly'){const x=Array.from({length:25},(_,i)=>i*32000/24),y=Array.from({length:21},(_,i)=>i*100000/20);await surface(x,y,y.map(v=>x.map(t=>(t*pin+out*pout)/1e6*v)),'Hypothetical monthly cost','Input tokens/request','Monthly requests','USD/month')}
  else {const x=Array.from({length:30},(_,i)=>i*32000/29);await plot([lines(x,x.map(t=>t*pin/1e6*num('monthly')),'Input cost',palette[0]),lines(x,x.map(t=>out*pout/1e6*num('monthly')),'Output cost',palette[1]),lines(x,x.map(t=>(t*pin+out*pout)/1e6*num('monthly')),'Total',palette[2])],{title:'Monthly cost sensitivity (fixed output length)',xaxis:{title:{text:'Input tokens/request'}},yaxis:{title:{text:'Hypothetical monthly USD'}}})}
 }
}
setup();render();window.addEventListener('resize',()=>Plotly.Plots.resize('chart'));
"""


def write_page(path: Path, kind, data, local):
    heading, explanation = DESCRIPTIONS[kind]
    payload = json.dumps(payload_for(kind, data, local), ensure_ascii=True, allow_nan=False).replace("<", "\\u003c")
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(heading)}</title><style>{STYLE}</style><script>{get_plotlyjs()}</script></head><body>
<header><small>DAY 72 · TOKENIZATION LEARNING LAB · OFFLINE</small><h1>{html.escape(heading)}</h1><p>{html.escape(explanation)}</p></header>
<main class="layout"><aside id="controls" aria-label="Experiment controls"></aside><section class="panel"><div id="metrics" class="kpi" aria-live="polite"></div><div id="chart" class="chart" aria-label="Interactive experiment chart"></div><div id="tokens" class="tokens" aria-label="Token or chunk boundaries"></div><h2>Computed state</h2><pre id="state" class="state"></pre><p class="limits">Synthetic experiments and theoretical simulations are distinguished in the labels. Author review required. Use the Streamlit dashboard for arbitrary text and live tokenization; standalone pages use embedded measurements or explicit local formulas.</p></section></main>
<script type="application/json" id="lab-data">{payload}</script><script>{SCRIPT}</script></body></html>"""
    path.write_text(document, encoding="utf-8")


def write_gallery(path, artifacts):
    cards = []
    for artifact in artifacts:
        name = artifact["name"]
        if name.endswith((".png", ".gif")):
            cards.append(f'<article><a href="{html.escape(name)}"><img loading="lazy" src="{html.escape(name)}" alt="{html.escape(name)}"></a><p>{html.escape(name)}</p></article>')
        elif name.endswith(".html"):
            cards.append(f'<article><a href="{html.escape(name)}">Open interactive lab: {html.escape(name)}</a></article>')
    path.write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><title>Day 72 local gallery</title><style>{STYLE}article{{background:white;margin:20px;padding:15px;border-radius:10px}}img{{max-width:100%;height:auto}}</style><header><small>DAY 72</small><h1>Tokenization visual learning lab</h1><p>Local generated gallery. For arbitrary text and all live controls, launch app.py with Streamlit. Synthetic measurements; interpretation review pending.</p></header><main>{"".join(cards)}</main></html>', encoding="utf-8")

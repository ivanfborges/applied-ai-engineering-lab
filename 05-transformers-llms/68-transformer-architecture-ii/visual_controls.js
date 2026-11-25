/* Offline controls. Python supplies actual tensors; small geometric/count formulas
   are recomputed here so exported documents remain interactive without a server. */
"use strict";
const $ = id => document.getElementById(id);
const palette = DATA.headColors;
const tokenColors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#997000", "#6955A5"];
const values = {};
const sum = a => a.reduce((x, y) => x + y, 0);
const dot = (a, b) => sum(a.map((x, i) => x * b[i]));
const norm = a => Math.sqrt(dot(a, a));
const transpose = a => a[0].map((_, j) => a.map(row => row[j]));
const softmax = a => {const m = Math.max(...a), e = a.map(x => Math.exp(x - m)), z = sum(e); return e.map(x => x / z);};
const entropy = a => -sum(a.map(x => x > 0 ? x * Math.log(x) : 0));
const fmt = (x, n=4) => x === null ? "−∞ / unavailable" : Number(x).toLocaleString("en-US", {maximumFractionDigits:n});
const gib = bytes => bytes / 2**30;
const cache = (b,l,t,k,d,bits) => 2*b*l*t*k*d*bits/8;
function counts(h,k,d) {
  if (!Number.isInteger(h) || !Number.isInteger(k) || !Number.isInteger(d) || h<1 || k<1 || d<1 || h%k) throw Error("Invalid head grouping");
  const w=h*d; return {width:w, parameters:2*w*w+2*w*k*d, assignment:Array.from({length:h},(_,i)=>Math.floor(i/(h/k)))};
}
window.labMath = {softmax, entropy, cache, counts};
let renderFunction, pending = Promise.resolve(), scheduledRenders = 0;
function queueRender() {
  scheduledRenders++;
  window.labBusy = true;
  pending = pending.then(async () => {await renderFunction(); window.labReady=true; $("status").textContent="Ready · all values computed locally · no network required";})
    .catch(error => {window.labError=String(error); $("status").textContent="ERROR: "+error; console.error(error);})
    .finally(()=>{scheduledRenders--;window.labBusy=scheduledRenders>0;});
}
function select(id, label, options, initial) {
  const field=document.createElement("label"); field.textContent=label;
  const input=document.createElement("select"); input.id=id;
  for(const option of options) {const [value,text]=Array.isArray(option)?option:[option,option]; input.add(new Option(text,value));}
  values[id]=String(initial ?? (Array.isArray(options[0])?options[0][0]:options[0])); input.value=values[id];
  input.addEventListener("change",()=>{values[id]=input.value; queueRender();}); field.append(input); $("controls").append(field); return input;
}
function slider(id,label,min,max,step,initial) {
  const field=document.createElement("label"), span=document.createElement("span"), input=document.createElement("input");
  input.type="range"; input.id=id; input.min=min; input.max=max; input.step=step; input.value=initial; values[id]=initial;
  span.textContent=label+": "+initial; field.append(span,input); $("controls").append(field);
  input.addEventListener("input",()=>{values[id]=Number(input.value);span.textContent=label+": "+input.value;queueRender();});
}
function panels(number) {for(let i=0;i<number;i++){const div=document.createElement("div");div.id="plot"+i;div.className="plot";$("plots").append(div);}}
function table(headers,rows) {return `<table><thead><tr>${headers.map(x=>`<th>${x}</th>`).join("")}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(x=>`<td>${x}</td>`).join("")}</tr>`).join("")}</tbody></table>`;}
function metrics(text) {$("metrics").innerHTML=text;}
function plot(i,data,title,extra={}) {
  return Plotly.react("plot"+i,data,{title:{text:title,font:{size:15}},font:{family:"Segoe UI,Arial",color:"#172c42"},paper_bgcolor:"white",plot_bgcolor:"white",margin:{l:65,r:35,t:62,b:58},uirevision:DATA.page,legend:{orientation:"h",y:-.18},...extra},{responsive:true,displaylogo:false,scrollZoom:true});
}
function heat(i,z,title,options={}) {
  const {probability=false,labels,query,signed=false,xTitle="Key / feature",yTitle="Query / token",...rest}=options;
  const rows=z.length, cols=z[0].length;
  const x=labels && labels.length===cols?labels:Array.from({length:cols},(_,j)=>String(j));
  const y=labels && labels.length===rows?labels:Array.from({length:rows},(_,j)=>String(j));
  const trace={type:"heatmap",z,x,y,colorscale:signed?"RdBu": "Viridis",reversescale:signed,
    hovertemplate:"row %{y}, column %{x}<br>%{z:.5f}<extra></extra>",colorbar:{thickness:12}};
  if(probability){trace.zmin=0;trace.zmax=1;}
  if(signed){const max=Math.max(...z.flat().filter(x=>x!==null).map(Math.abs),1e-10);trace.zmin=-max;trace.zmax=max;}
  const shapes=query===undefined?[]:[{type:"rect",xref:"paper",yref:"y",x0:0,x1:1,y0:query-.48,y1:query+.48,line:{color:"#D55E00",width:3},fillcolor:"rgba(0,0,0,0)"}];
  return plot(i,[trace],title,{xaxis:{title:{text:xTitle}},yaxis:{title:{text:yTitle},autorange:"reversed"},shapes,...rest});
}
const bars = (x,y,name,color="#0072B2") => ({type:"bar",x,y,name,marker:{color}});
function scatter3(i,points,title,axes="coordinate") {
  return plot(i,[{type:"scatter3d",mode:"markers+text",x:points.map(p=>p[0]),y:points.map(p=>p[1]),z:points.map(p=>p[2]),text:DATA.tokens,textposition:"top center",marker:{size:6,color:tokenColors},hovertemplate:"%{text}<br>(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>"}],title,
    {margin:{l:0,r:0,t:55,b:0},scene:{xaxis:{title:{text:axes+" 1"}},yaxis:{title:{text:axes+" 2"}},zaxis:{title:{text:axes+" 3"}},aspectmode:"data",uirevision:"camera"}});
}
function headControls(query=true,mode=true) {
  if(query)select("query","Query token",DATA.tokens.map((t,i)=>[i,`${i}: ${t}`]),2);
  if(mode)select("mode","Attention mask",[["bidirectional","Bidirectional"],["causal","Causal"]]);
}

function subspaces() {
  select("space","Coordinate view",[["direct","Direct 3D (geometric analogy)"],["pca","16D → 4D heads, PCA views"]]);
  select("head","Head",[0,1,2,3]);select("kind","Projection",["Q","K","V"]);
  select("a","Token A",DATA.tokens.map((t,i)=>[i,`${i}: ${t}`]),0);select("b","Token B",DATA.tokens.map((t,i)=>[i,`${i}: ${t}`]),1);panels(5);
  renderFunction=async()=>{
    const c=DATA.cases[values.space],h=+values.head,e=c.entries.find(e=>e.head===h&&e.kind===values.kind),a=+values.a,b=+values.b;
    metrics(`<strong>Head ${h} ${values.kind}</strong> · matrix ${e.weight.length}×${e.weight[0].length}; every input feature may contribute.<br>Exact distance A↔B: ${fmt(c.distances[a][b])} → ${fmt(e.distances[a][b])}; exact angle: ${fmt(c.angles[a][b],2)}° → ${fmt(e.angles[a][b],2)}°.<br>Centered variance retained: input ${fmt(c.retained*100,1)}%, projected ${fmt(e.retained*100,1)}%. ${values.space==="pca"?"PCA is fitted separately; compare exact numbers, not distances between plots.":"Direct maps are full 3D coordinates; full-rank 3→3 column spaces all overlap."}`);
    await Promise.all([scatter3(0,c.original,"Same input token representations",values.space==="pca"?"Input PC":"Input"),scatter3(1,e.view,`Head ${h}: ${values.kind} projection`,values.space==="pca"?"Head PC":"Projected"),heat(2,e.weight,`Actual W_${values.kind} for head ${h}`,{signed:true,xTitle:"Projected coordinate",yTitle:"Input feature"}),heat(3,c.minimum_principal_angles,"Minimum principal angle between Q column spans (°)",{xTitle:"Q head",yTitle:"Q head"}),
      plot(4,[bars(DATA.tokens,c.distances[a],"Input-space distance"),bars(DATA.tokens,e.distances[a],"Projected-space distance",palette[h])],`Exact distances from token ${a}`,{yaxis:{title:{text:"Euclidean distance (own coordinate space)"}}})]);
  };
}

function geometry() {
  select("head","Constructed head",[0,1,2,3]);headControls(true,false);
  slider("direction","Query angle offset (degrees)",-180,180,5,0);slider("qnorm","Query magnitude",.1,5,.1,1);slider("knorm","Key magnitude multiplier",.1,5,.1,1);select("scale","Scale sqrt(2)",[[1,"Enabled"],[0,"Disabled"]]);panels(4);
  renderFunction=async()=>{
    const h=+values.head,t=+values.query,rotation=h*Math.PI/8,angle=(+values.direction)*Math.PI/180+t*Math.PI/12;
    const rotate=p=>[p[0]*Math.cos(rotation)-p[1]*Math.sin(rotation),p[0]*Math.sin(rotation)+p[1]*Math.cos(rotation)];
    const q=[Math.cos(angle)*values.qnorm,Math.sin(angle)*values.qnorm];
    const keys=[[1,0],[3,0],[0,1],[-1,.5],[0,-2],[1,1]].map(rotate).map(p=>p.map(x=>x*values.knorm));
    const vals=[[1,0],[0,2],[-1,1],[1,-1],[-2,0],[.5,1.5]].map(rotate);
    const dots=keys.map(k=>dot(q,k)),cos=keys.map((k,j)=>dots[j]/(norm(q)*norm(k))),logits=dots.map(x=>x/(+values.scale?Math.sqrt(2):1)),probs=softmax(logits);
    const context=[0,1].map(j=>sum(vals.map((v,i)=>v[j]*probs[i])));
    const arrows=(vectors,names,colors)=>vectors.map((p,i)=>({x:[0,p[0]],y:[0,p[1]],mode:"lines+markers+text",text:["",names[i]],textposition:"top center",name:names[i],line:{width:i===vectors.length-1?4:2,color:colors[i]}}));
    metrics(`<strong>Context = Σ probability × value = (${fmt(context[0])}, ${fmt(context[1])})</strong><br>First two keys have identical cosine (${fmt(cos[0])}, ${fmt(cos[1])}) but different norms (${fmt(norm(keys[0]))}, ${fmt(norm(keys[1]))}) and dot products (${fmt(dots[0])}, ${fmt(dots[1])}). Query is a constructed intervention, not a model embedding.`+
      table(["Key","Norm","Angle to q (°)","Cosine","Dot","Logit","Probability"],keys.map((k,i)=>[DATA.tokens[i],fmt(norm(k)),fmt(Math.acos(Math.max(-1,Math.min(1,cos[i])))*180/Math.PI,2),fmt(cos[i]),fmt(dots[i]),fmt(logits[i]),fmt(probs[i])])));
    const vectorLayout=(vectors,colors)=>({xaxis:{title:{text:"Coordinate 1"},zeroline:true},yaxis:{title:{text:"Coordinate 2"},scaleanchor:"x",zeroline:true},annotations:vectors.map((p,i)=>({x:p[0],y:p[1],ax:0,ay:0,xref:"x",yref:"y",axref:"x",ayref:"y",text:"",showarrow:true,arrowhead:3,arrowsize:1,arrowwidth:i===vectors.length-1?3:1.5,arrowcolor:colors[i]}))});
    await Promise.all([plot(0,arrows([...keys,q],[...DATA.tokens,"query"],[...tokenColors,"#111111"]),"Query and keys: directed vectors",vectorLayout([...keys,q],[...tokenColors,"#111111"])),plot(1,[bars(DATA.tokens,probs,"Attention probability",tokenColors)],"Softmax over candidate keys",{yaxis:{range:[0,1],title:{text:"Probability"}}}),plot(2,arrows([...vals,context],[...DATA.tokens,"context"],[...tokenColors,palette[h]]),"Value vectors and weighted context",vectorLayout([...vals,context],[...tokenColors,palette[h]])),plot(3,[bars(DATA.tokens,dots,"Dot product"),bars(DATA.tokens,cos,"Cosine similarity","#D55E00")],"Magnitude-sensitive dot product vs cosine")]);
  };
}

function heads() {
  headControls();select("stage","Tensor stage",[["raw","Raw QKᵀ"],["scaled","Scaled logits"],["masked","Masked logits (blank = −∞)"],["attention","Softmax probabilities"]],"attention");panels(6);
  renderFunction=async()=>{
    const t=DATA.traces[values.mode],a=t[values.stage][0],q=+values.query,average=a[0].map((row,i)=>row.map((_,j)=>a.some(m=>m[i][j]===null)?null:sum(a.map(m=>m[i][j]))/a.length));
    const spread=Math.max(...t.attention[0].map(m=>m[q][q]))-Math.min(...t.attention[0].map(m=>m[q][q]));
    metrics(`<strong>Query ${q}: ${DATA.tokens[q]}</strong> · per-head shape (1,4,6,6); output (1,6,16). Self-weight spread across heads: ${fmt(spread)}. An average map hides this spread.<br>`+
      table(["Head",...DATA.tokens],t.attention[0].map((m,h)=>[h,...m[q].map(x=>fmt(x))])));
    await Promise.all([...a.map((z,h)=>heat(h,z,`Head ${h} · ${values.stage}`,{probability:values.stage==="attention",signed:values.stage!=="attention",query:q})),heat(4,average,"Mean over heads (inspection only)",{probability:values.stage==="attention",signed:values.stage!=="attention",query:q}),heat(5,t.output[0],"Final contextualized representations (T×D)",{signed:true,query:q})]);
  };
}

function graph() {
  select("head","Head",[["all","All heads"],0,1,2,3]);select("query","Query token",[["all","All queries"],...DATA.tokens.map((t,i)=>[i,`${i}: ${t}`])],2);headControls(false,true);slider("threshold","Minimum attention",0,1,.02,.12);panels(2);
  renderFunction=async()=>{
    const t=DATA.traces[values.mode],positions=DATA.tokens.map((_,i)=>[Math.cos(2*Math.PI*i/6),Math.sin(2*Math.PI*i/6)]),traces=[],annotations=[];
    let edges=0;
    for(let h=0;h<4;h++)if(values.head==="all"||+values.head===h){
      for(let q=0;q<6;q++)if(values.query==="all"||+values.query===q){for(let k=0;k<6;k++){
        const w=t.attention[0][h][q][k];if(w<values.threshold||w===0)continue;edges++;
        const a=positions[q],b=positions[k],bend=(h-1.5)*.08,pts=[];
        for(let s=0;s<=12;s++){const u=s/12;if(q===k){const angle=2*Math.PI*u;pts.push([a[0]+.14*Math.sin(angle),a[1]+.15*(1-Math.cos(angle))]);}else pts.push([a[0]*(1-u)+b[0]*u+bend*Math.sin(Math.PI*u)*(b[1]-a[1]),a[1]*(1-u)+b[1]*u-bend*Math.sin(Math.PI*u)*(b[0]-a[0])]);}
        traces.push({x:pts.map(p=>p[0]),y:pts.map(p=>p[1]),mode:"lines",name:`H${h} ${q}→${k}`,line:{width:1+8*w,color:palette[h]},opacity:.18+.82*w,showlegend:false,hovertemplate:`Head ${h}: query ${q} → key ${k}<br>weight ${w.toFixed(5)}<extra></extra>`});
        const end=pts[q===k?10:10],start=pts[q===k?8:8];annotations.push({x:end[0],y:end[1],ax:start[0],ay:start[1],xref:"x",yref:"y",axref:"x",ayref:"y",showarrow:true,arrowhead:3,arrowsize:1.3,arrowwidth:1+3*w,arrowcolor:palette[h],text:""});
      }}
      traces.push({x:[null],y:[null],mode:"lines",name:`Head ${h}`,line:{color:palette[h],width:4}});
    }
    traces.push({x:positions.map(p=>p[0]),y:positions.map(p=>p[1]),mode:"markers+text",text:DATA.tokens.map((t,i)=>`${i}: ${t}`),textposition:"bottom center",marker:{size:20,color:tokenColors,line:{color:"white",width:2}},name:"tokens",showlegend:false});
    metrics(`<strong>${edges} directed edges</strong> above threshold ${fmt(values.threshold,2)}. Direction is query → key; key/value information is aggregated back into the query output. Color identifies the head, including when all heads are visible.`);
    const selected=values.head==="all"?0:+values.head;
    await Promise.all([plot(0,traces,"Directed attention routing",{annotations,xaxis:{range:[-1.5,1.5],visible:false},yaxis:{range:[-1.5,1.6],visible:false,scaleanchor:"x"}}),heat(1,t.attention[0][selected],`Exact weights · head ${selected}`,{probability:true,query:values.query==="all"?undefined:+values.query})]);
  };
}

function scaling() {
  select("dim","Head dimension",DATA.samples.map(s=>s.dimension),32);slider("qm","Query magnitude factor",.25,4,.25,1);slider("km","Key magnitude factor",.25,4,.25,1);select("scaled","Attention scaling",[[1,"Enabled"],[0,"Disabled"]]);panels(4);
  renderFunction=async()=>{
    const s=DATA.samples.find(s=>s.dimension===+values.dim),factor=values.qm*values.km,raw=s.raw.map(x=>x*factor),scaled=s.scaled.map(x=>x*factor),logits=s.logits.map(x=>x*factor/(+values.scaled?Math.sqrt(s.dimension):1)),probs=softmax(logits);
    metrics(`<strong>${s.raw.length} paired random dot products</strong> · empirical raw variance ${fmt(s.raw_variance*factor**2)} (expected ${fmt(s.dimension*factor**2)}); scaled ${fmt(s.scaled_variance*factor**2)} (expected ${fmt(factor**2)}).<br>Six-key row: max probability ${fmt(Math.max(...probs))}; entropy ${fmt(entropy(probs))} nats. ${+values.scaled?"Scaling enabled":"Unscaled control"}. Magnitude factors multiply the original IID samples; seed and components stay paired.`);
    await Promise.all([plot(0,[{type:"histogram",x:raw,name:"Raw dot products",opacity:.6,histnorm:"probability density",marker:{color:palette[0]},nbinsx:50},{type:"histogram",x:scaled,name:"Scaled by sqrt(d)",opacity:.6,histnorm:"probability density",marker:{color:palette[1]},nbinsx:50}],"Raw vs scaled dot-product distributions",{barmode:"overlay",xaxis:{title:{text:"Dot-product value"}},yaxis:{title:{text:"Density"}}}),plot(1,[bars(DATA.tokens,probs,"Attention",tokenColors)],"Same six-key softmax row",{yaxis:{range:[0,1],title:{text:"Probability"}}}),plot(2,[{x:DATA.samples.map(x=>x.dimension),y:DATA.samples.map(x=>x.raw_variance*factor**2),name:"Empirical raw",mode:"lines+markers"},{x:DATA.samples.map(x=>x.dimension),y:DATA.samples.map(x=>x.dimension*factor**2),name:"Expected raw d·q²k²",mode:"lines",line:{dash:"dash"}},{x:DATA.samples.map(x=>x.dimension),y:DATA.samples.map(x=>x.scaled_variance*factor**2),name:"Empirical scaled",mode:"lines+markers"},{x:DATA.samples.map(x=>x.dimension),y:DATA.samples.map(()=>factor**2),name:"Expected scaled q²k²",mode:"lines",line:{dash:"dot"}}],"Variance vs dimension (paired samples)",{xaxis:{title:{text:"Head dimension"}},yaxis:{title:{text:"Variance"}}}),plot(3,[bars(DATA.tokens,logits,"Softmax logits")],"Selected scaled/unscaled logits",{yaxis:{title:{text:"Logit"}}})]);
  };
}

function headCount() {
  select("width","Model dimension",[16,32,64,128,256,512],64);select("h","Heads",[1,2,4,8,16],4);slider("length","Sequence length",16,4096,16,512);select("bits","Map precision bits",[16,32],16);panels(3);
  renderFunction=async()=>{
    const d=+values.width,h=+values.h,t=+values.length,bits=+values.bits,hs=[1,2,4,8,16].filter(x=>d%x===0);
    metrics(`<strong>D=${d}, H=${h}, d_head=${d/h}</strong> · each head output (T,${d/h}); concatenate (T,${d}).<br>Bias-free parameters: ${fmt(4*d*d,0)}. H=${h} probability matrices contain ${fmt(h*t*t,0)} entries, ${fmt(h*t*t*bits/8/2**20,2)} MiB for B=1. Counts do not measure quality.`);
    await Promise.all([plot(0,Array.from({length:h},(_,i)=>bars(["Concatenated feature width"],[d/h],`Head ${i}`,palette[i%palette.length])),"Same model width, narrower heads",{barmode:"stack",yaxis:{title:{text:"Feature count"}}}),plot(1,[bars(hs.map(String),hs.map(()=>4*d*d),"Projection parameters")],"4D² parameters independent of H",{xaxis:{title:{text:"Heads"}},yaxis:{title:{text:"Parameters (no bias)"}}}),plot(2,[bars(hs.map(String),hs.map(n=>n*t*t*bits/8/2**20),"One explicit map tensor")],"More independent maps at fixed width",{xaxis:{title:{text:"Heads"}},yaxis:{title:{text:"MiB (B=1)"}}})]);
  };
}

function diversity() {
  select("query","Query token",DATA.tokens.map((t,i)=>[i,`${i}: ${t}`]),2);select("ablated","Zero head output",[0,1,2,3],0);panels(4);
  renderFunction=async()=>{
    const d=DATA.diversity,q=+values.query,h=+values.ablated;
    const original=d.trace.output[0][q],ablated=original.map((v,j)=>v-d.contributions[h][q][j]);
    metrics(`<strong>Heads 0/1 map max difference: ${fmt(d.identical_map_error,12)}</strong>; head-value output L2 difference ${fmt(d.distinct_value_output_difference)}.<br>Zeroing head ${h} changes the full layer output by L2 ${fmt(d.ablation_l2[h])}. This is a numerical intervention on an untrained layer, not downstream task importance.`);
    await Promise.all([heat(0,d.map_similarity,"Cosine similarity: flattened attention maps",{signed:true,xTitle:"Head",yTitle:"Head"}),heat(1,d.output_similarity,"Cosine similarity: flattened head outputs",{signed:true,xTitle:"Head",yTitle:"Head"}),plot(2,[bars([0,1,2,3],d.entropy,"Mean entropy",palette.slice(0,4))],"Per-head attention entropy (nats)"),plot(3,[{x:original.map((_,i)=>i),y:original,name:"Original output",mode:"lines+markers"},{x:ablated.map((_,i)=>i),y:ablated,name:`Head ${h} zeroed`,mode:"lines+markers"}],`Output features for query ${q}`,{xaxis:{title:{text:"Output feature"}},yaxis:{title:{text:"Activation"}}})]);
  };
}

function sharingControls() {
  const h=select("qh","Query heads",[4,8,16,32],8),k=select("kvh","KV heads (divisor)",[1,2,4,8],2);
  h.addEventListener("change",()=>{const selected=+values.kvh; k.replaceChildren();for(const n of [1,2,4,8,16,32].filter(n=>+h.value%n===0))k.add(new Option(n,n));values.kvh=String(Math.min(selected,+h.value));k.value=values.kvh;queueRender();});
  return {h,k};
}
function sharing() {
  sharingControls();select("dh","Head dimension",[32,64,128],64);panels(3);
  renderFunction=async()=>{
    const h=+values.qh,k=+values.kvh,d=+values.dh,c=counts(h,k,d),connections=[];
    for(let i=0;i<h;i++)connections.push({x:[0,1],y:[i,c.assignment[i]*(h-1)/Math.max(1,k-1)],mode:"lines+markers",line:{color:palette[c.assignment[i]%8],width:2},marker:{size:8},name:`Q${i} → K/V${c.assignment[i]}`,showlegend:false});
    metrics(`<strong>${k===h?"MHA":k===1?"MQA":"GQA"}: ${h} query heads, ${k} KV heads</strong> · D=${c.width}, d_head=${d}.<br>Q+O: 2D²; K+V: 2D·H_KV·d_head. Total ${fmt(c.parameters,0)} parameters (no bias). No training or checkpoint conversion was performed.`);
    await Promise.all([plot(0,connections,"Query → shared key/value heads",{xaxis:{range:[-.3,1.3],tickvals:[0,1],ticktext:["Query heads","Key AND value heads"]},yaxis:{title:{text:"Head placement (schematic)"}}}),plot(1,[bars(["MHA","Selected sharing","MQA"],[counts(h,h,d).parameters,c.parameters,counts(h,1,d).parameters],"Projection parameters")],"Same query width, different K/V widths"),plot(2,[bars(["MHA","Selected sharing","MQA"],[h,k,1].map(n=>gib(cache(1,32,8192,n,d,16))),"Cache GiB")],"Theoretical KV payload · B=1,L=32,T=8192,16 bits",{yaxis:{title:{text:"GiB"}}})]);
  };
}

function kvCache() {
  slider("layers","Layers",1,96,1,32);sharingControls();select("dh","Head dimension",[32,64,128,256],128);slider("batch","Batch size",1,32,1,1);slider("length","Cached tokens",512,131072,512,8192);select("precision","Storage",[[32,"FP32 (4 bytes)"],[16,"FP16 (2 bytes)"],["bf16","BF16 (2 bytes)"],[8,"INT8 (1 byte, scales excluded)"],[4,"INT4 (ideal packed, 0.5 byte)"]],16);select("budget","Hypothetical capacity plane",[8,16,24,48,80],24);panels(3);
  renderFunction=async()=>{
    const b=+values.batch,l=+values.layers,t=+values.length,h=+values.qh,k=+values.kvh,d=+values.dh,bits=values.precision==="bf16"?16:+values.precision,ts=[512,1024,2048,4096,8192,16384,32768,65536,131072],bs=[1,2,4,8,16,32];
    const result=cache(b,l,t,k,d,bits),z=bs.map(bb=>ts.map(tt=>gib(cache(bb,l,tt,k,d,bits))));
    metrics(`<strong>KV bytes = 2 × ${b} batch × ${l} layers × ${fmt(t,0)} tokens × ${k} KV heads × ${d} head width × ${bits/8} bytes</strong><br>= ${fmt(result,0)} bytes = ${fmt(gib(result),3)} GiB. Query heads (${h}) determine grouping, not a separate cache multiplier. Capacity plane excludes all other memory consumers.`);
    await Promise.all([plot(0,[...[[h,"MHA"],[k,"Selected GQA/MQA"],[1,"MQA"]].map(([n,name])=>({x:ts,y:ts.map(tt=>gib(cache(b,l,tt,n,d,bits))),mode:"lines+markers",name}))],"Cache payload vs context length",{xaxis:{title:{text:"Cached tokens"}},yaxis:{title:{text:"GiB"}}}),plot(1,[bars(["MHA","Selected","MQA"],[h,k,1].map(n=>gib(cache(b,l,t,n,d,bits))),"Theoretical payload")],"Architecture comparison at selected context",{yaxis:{title:{text:"GiB"}}}),plot(2,[{type:"surface",x:ts,y:bs,z,colorscale:"Viridis",name:"KV payload",colorbar:{title:{text:"GiB"},thickness:12}},{type:"surface",x:ts,y:bs,z:bs.map(()=>ts.map(()=>+values.budget)),opacity:.25,showscale:false,colorscale:[[0,"#D55E00"],[1,"#D55E00"]],name:"Hypothetical budget"}],"Cache payload surface + hypothetical budget",{scene:{xaxis:{title:{text:"Tokens"}},yaxis:{title:{text:"Batch"}},zaxis:{title:{text:"GiB"}},uirevision:"cache-camera"},margin:{l:0,r:20,t:55,b:0}})]);
  };
}

function complexity() {
  select("width","Model width D",[32,64,128,256,512,1024],512);select("h","Heads H",[1,2,4,8,16,32],8);slider("length","Reference tokens",128,16384,128,2048);panels(4);
  renderFunction=async()=>{
    const d=+values.width,h=+values.h,t=+values.length,ts=[128,256,512,1024,2048,4096,8192,16384],hs=[1,2,4,8,16,32],maps=hs.map(hh=>ts.map(tt=>hh*tt*tt*2/2**20));
    metrics(`<strong>Dense prefill B=1</strong>: projections 4TD² = ${fmt(4*t*d*d,0)} MACs; QKᵀ + AV = 2T²D = ${fmt(2*t*t*d,0)} MACs.<br>One cached decoding step: attention ≈2TD MACs, projections ≈4D² MACs. One explicit FP16 map tensor: ${fmt(h*t*t*2/2**20,2)} MiB. KV payload grows linearly with tokens. Memory traffic/bandwidth and actual latency require device profiling.`);
    await Promise.all([plot(0,[{x:ts,y:ts.map(tt=>tt/128),mode:"lines+markers",name:"Linear T / 128"},{x:ts,y:ts.map(tt=>(tt/128)**2),mode:"lines+markers",name:"Quadratic (T/128)²"}],"Normalized growth, common baseline",{xaxis:{title:{text:"Tokens"},type:"log"},yaxis:{title:{text:"Relative count"},type:"log"}}),plot(1,[{x:ts,y:ts.map(tt=>4*tt*d*d),mode:"lines+markers",name:"Prefill projections"},{x:ts,y:ts.map(tt=>2*tt*tt*d),mode:"lines+markers",name:"Prefill attention"},{x:ts,y:ts.map(tt=>2*tt*d),mode:"lines+markers",name:"One-token cached attention"}],"Arithmetic counts (MACs), not timing",{xaxis:{title:{text:"Tokens"},type:"log"},yaxis:{title:{text:"MACs"},type:"log"}}),plot(2,[{x:ts,y:ts.map(tt=>h*tt*tt*2/2**20),mode:"lines+markers",name:"Explicit FP16 maps"},{x:ts,y:ts.map(tt=>cache(1,1,tt,h,d/h,16)/2**20),mode:"lines+markers",name:"One-layer FP16 KV cache"}],"Quadratic maps vs linear cache",{xaxis:{title:{text:"Tokens"},type:"log"},yaxis:{title:{text:"MiB"},type:"log"}}),plot(3,[{type:"surface",x:ts,y:hs,z:maps,colorscale:"Viridis",colorbar:{title:{text:"MiB"},thickness:12}}],"One explicit map tensor vs tokens and heads",{scene:{xaxis:{title:{text:"Tokens"}},yaxis:{title:{text:"Heads"}},zaxis:{title:{text:"FP16 MiB"}},uirevision:"complexity-camera"},margin:{l:0,r:20,t:55,b:0}})]);
  };
}

function validation() {
  panels(2);renderFunction=async()=>{
    const r=DATA.report;metrics(`<strong>Computed seed ${DATA.seed} · CPU float64 · bias/dropout disabled · tolerance 1e-12</strong>`+table(["Control","Observed result"],[...r.parity.map(p=>[p.mode+" output / map max abs error",p.output_max_abs_error.toExponential(3)+" / "+p.map_max_abs_error.toExponential(3)]),["Causal prefix change",fmt(r.causal_prefix_max_abs_change,12)],["Bidirectional prefix change",fmt(r.bidirectional_prefix_max_abs_change)],["Identical-map error",fmt(r.identical_map_error,12)],["Distinct-value head output L2",fmt(r.distinct_value_output_l2)],["Online-softmax max abs error",r.online_softmax_max_abs_error.toExponential(3)]]));
    await Promise.all([plot(0,[bars(r.scaling.map(s=>s.dimension),r.scaling.map(s=>s.scaled_variance),"Scaled empirical variance")],"IID scaling check (expected ≈1)",{xaxis:{title:{text:"Head dimension"}},yaxis:{title:{text:"Variance"}}}),plot(1,[bars(["32 KV heads","8 KV heads","1 KV head"],[r.cache_gib["32"],r.cache_gib["8"],r.cache_gib["1"]],"Analytical GiB")],"Payload formula check · fixed hypothetical model",{yaxis:{title:{text:"GiB"}}})]);
  };
}
const initializers={subspaces,geometry,heads,graph,scaling,"head-count":headCount,diversity,sharing,"kv-cache":kvCache,complexity,validation};
initializers[DATA.page]();queueRender();

/* Live operations mirror the tested NumPy definitions; no network or model calls. */
"use strict";
const D = JSON.parse(document.getElementById("lab-data").textContent);
const C = D.config, group = D.group;
const color = {q:"#2473b9", k:"#168467", p:"#d47b22", r:"#8755ad", gray:"#8290a3"};
const values = {}, chartNodes = [], controls = document.getElementById("controls");
let animationControl = null, timer = null, rendering = false, pending = false;
const seq = n => Array.from({length:n}, (_,i)=>i);
const dot = (a,b) => a.reduce((s,x,i)=>s+x*b[i],0);
const norm = a => Math.sqrt(dot(a,a));
const add = (a,b) => a.map((x,i)=>x+b[i]);
const sub = (a,b) => a.map((x,i)=>x-b[i]);
const transpose = a => a[0].map((_,i)=>a.map(row=>row[i]));
const mm = (a,b) => {const bt=transpose(b);return a.map(row=>bt.map(col=>dot(row,col)));};
const maxError = (a,b) => Math.max(...a.flat().map((x,i)=>Math.abs(x-b.flat()[i])));
const cosine = (a,b) => dot(a,b)/(norm(a)*norm(b));
const fmt = x => typeof x==="number" ? (Math.abs(x)<1e-4 && x!==0?x.toExponential(3):x.toFixed(5)) : x;
function softmax(a){const max=Math.max(...a), v=a.map(x=>Math.exp(x-max)), total=v.reduce((s,x)=>s+x,0);if(!Number.isFinite(total)||!total)throw Error("No finite visible softmax row");return v.map(x=>x/total);}
function pe(n,d,b,start=0){return seq(n).map(p=>seq(d).map(j=>{const angle=(p+start)*b**(-2*Math.floor(j/2)/d);return j%2?Math.cos(angle):Math.sin(angle);}));}
function rotate2(v,theta){return [v[0]*Math.cos(theta)-v[1]*Math.sin(theta),v[0]*Math.sin(theta)+v[1]*Math.cos(theta)];}
function rope(x,positions,b){return x.map((row,p)=>{const out=[];for(let j=0;j<row.length;j+=2)out.push(...rotate2(row.slice(j,j+2),positions[p]*b**(-j/row.length)));return out;});}
function attention(x,w,method,table,b,slope=.1){
 const n=x.length,d=x[0].length,e=pe(n,d,b);
 let z=x;if(method==="sinusoidal")z=x.map((row,i)=>add(row,e[i]));
 if(method==="absolute")z=x.map((row,i)=>add(row,table[i]));
 let [q,k,v]=w.map(weight=>mm(z,weight));if(method==="rope"){q=rope(q,seq(n),b);k=rope(k,seq(n),b);}
 const logits=q.map((row,p)=>k.map((key,r)=>r>p?-Infinity:dot(row,key)/Math.sqrt(d)+(method==="relative"?-.08*Math.abs(p-r):method==="alibi"?-slope*(p-r):0)));
 const probs=logits.map(softmax);return {logits,probs,output:mm(probs,v),entropy:probs.map(row=>-row.reduce((s,p)=>s+(p?p*Math.log(p):0),0)),distance:probs.map((row,p)=>row.reduce((s,a,r)=>s+a*Math.abs(p-r),0))};
}
function control(id,label,min,max,value,step=1){
 values[id]=value;const wrap=document.createElement("label");wrap.htmlFor=id;const title=document.createElement("span");title.textContent=label;
 const input=document.createElement("input");input.type="range";input.id=id;input.min=min;input.max=max;input.step=step;input.value=value;
 const out=document.createElement("output");out.htmlFor=id;out.textContent=value;wrap.append(title,input,out);controls.append(wrap);
 input.addEventListener("input",()=>{values[id]=Number(input.value);out.textContent=input.value;schedule();});return input;
}
function select(id,label,options,value){values[id]=value;const wrap=document.createElement("label");wrap.textContent=label;const input=document.createElement("select");input.id=id;options.forEach(x=>{const opt=document.createElement("option");opt.value=x;opt.textContent=x;input.append(opt);});input.value=value;wrap.append(input);controls.append(wrap);input.addEventListener("change",()=>{values[id]=input.value;schedule();});return input;}
function setup(){
 if(["sinusoidal","construction","similarity"].includes(group)){
  control("length","Sequence length",8,512,C.seq_len);select("dim","Embedding width",[...new Set([8,16,32,64,128,C.dim])].sort((a,b)=>a-b),String(C.dim));
  control("logbase","log10 frequency base",.3,8,Math.log10(C.base),.01);animationControl=control("position","Selected position",0,511,Math.min(10,C.seq_len-1));
  if(group==="sinusoidal")control("pair","Dimension pair index",0,63,0);
  if(group==="construction")control("columns","Constructed columns",1,512,Math.min(24,C.seq_len));
 }
 if(group==="embedding-space"){
  select("dim","Embedding width",Object.keys(D.geometry),String(C.dim));select("view","Coordinates",["pca","exact"],D.initial_view);
  control("length","Trajectory prefix",8,Math.max(128,C.seq_len),C.seq_len);animationControl=control("position","Highlighted position",0,Math.max(128,C.seq_len)-1,Math.min(10,C.seq_len-1));
  [0,1,2].forEach((v,i)=>control("axis"+i,"Exact axis "+["x","y","z"][i],0,127,v));
 }
 if(group==="addition"){
  select("token","Content occurrence",[0,1,2,3,4,5],"0");animationControl=control("position","Assigned position",0,127,3);select("view","Vector coordinates",["exact","linear PCA basis"],"exact");
 }
 if(group==="permutation")animationControl=control("stage","Original / permuted",0,1,0);
 if(group==="absolute-relative")animationControl=control("shift","Global shift of pattern 5–10",0,20,20);
 if(group==="rope"){
  animationControl=control("p","Query position",0,64,2);control("k","Key position",0,64,5);control("frequency","Angular frequency (rad/token)",.02,2,.5,.02);
  control("qa","Initial query angle (rad)",-3.2,3.2,.3,.05);control("ka","Initial key angle (rad)",-3.2,3.2,1,.05);
  control("qm","Query magnitude",.1,2,1,.1);control("km","Key magnitude",.1,2,.8,.1);control("shift","Common position shift",0,64,0);
 }
 if(group==="rope-multifrequency")animationControl=control("position","Key position; query fixed at 2",0,127,5);
 if(group==="alibi"){
  control("length","Sequence length",8,128,32);animationControl=control("query","Query position",0,127,24);control("slope","Head slope",.01,1,.1,.01);
  control("strength","Relevant content score",0,20,0,.2);control("relevant","Relevant key position",0,127,2);
 }
 if(group==="comparison"){
  control("length","Sequence length",8,64,Math.min(C.seq_len,32));animationControl=control("query","Query position",0,63,Math.min(C.seq_len,24)-1);select("method","Inspect mechanism",["none","sinusoidal","absolute","relative","rope","alibi"],"rope");
 }
 if(group==="extrapolation"){
  select("length","Represented context length",[128,256,512,1024,4096],"4096");animationControl=control("position","Selected position",0,4095,256);control("interpolate","Coordinate scale (p / factor)",1,32,1,.25);
 }
 if(group==="rag"){
  control("chunks","Retrieved + distractor chunks",3,20,8);animationControl=control("evidence","Evidence slot",0,19,3);control("words","Whitespace units per chunk",8,80,24);
 }
 const play=document.createElement("button");play.id="play";play.textContent="Play / pause";play.addEventListener("click",()=>{if(timer){clearInterval(timer);timer=null;return;}timer=setInterval(()=>{if(rendering)return;const id=animationControl.id;let next=Number(animationControl.value)+Number(animationControl.step);if(next>Number(animationControl.max))next=Number(animationControl.min);animationControl.value=next;animationControl.dispatchEvent(new Event("input"));},500);});controls.append(play);
}
function axes(title){const wrapped=title.length>53?title.replace(/^(.{20,50})\s/,"$1<br>"):title;return {title:{text:wrapped,font:{size:14},x:.03,xanchor:"left"},height:440,margin:{l:60,r:35,t:65,b:100},legend:{orientation:"h",y:-.25,x:0,font:{size:10}},paper_bgcolor:"white",plot_bgcolor:"#f8fafc",font:{family:"Segoe UI,Arial",color:"#18334e"},uirevision:group};}
async function plot(index,data,title,extra={}){
 if(!chartNodes[index]){const node=document.createElement("div");node.className="chart";node.id="plot"+index;document.getElementById("charts").append(node);chartNodes[index]=node;}
 await Plotly.react(chartNodes[index],data,Object.assign(axes(title),extra),{responsive:true,displaylogo:false,scrollZoom:true});
}
const line=(x,y,name,c)=>({x,y,type:"scatter",mode:"lines",name,line:{color:c,width:2}});
const heat=(z,lo,hi)=>({z:z.map(row=>row.map(v=>Number.isFinite(v)?v:null)),type:"heatmap",zmin:lo,zmax:hi,colorscale:lo<0?"RdBu":"Blues",reversescale:lo<0,hoverongaps:false});
const scene={xaxis:{title:{text:"Coordinate 1"}},yaxis:{title:{text:"Coordinate 2"}},zaxis:{title:{text:"Coordinate 3"}},aspectmode:"data"};
function trajectory(coords,meta,name="positions"){
 return {type:"scatter3d",x:coords.map(v=>v[0]),y:coords.map(v=>v[1]),z:coords.map(v=>v[2]),mode:"lines+markers",name,
  marker:{size:4,color:seq(coords.length),colorscale:"Viridis",colorbar:{title:{text:"Position"}}},line:{color:color.p,width:3},customdata:meta,
  hovertemplate:"position=%{customdata[0]}<br>(%{x:.4f}, %{y:.4f}, %{z:.4f})<br>original norm=%{customdata[1]:.5f}<br>previous cosine=%{customdata[2]:.5f}<br>previous L2=%{customdata[3]:.5f}<extra></extra>"};
}
function vectors2(items){return items.map(([v,name,c,origin=[0,0]])=>({x:[origin[0],origin[0]+v[0]],y:[origin[1],origin[1]+v[1]],type:"scatter",mode:"lines+markers",name,line:{color:c,width:4},marker:{size:[0,9],symbol:["circle","arrow"],angleref:"previous"}}));}
function vectors3(items){return items.flatMap(([v,name,c,origin=[0,0,0]])=>[{type:"scatter3d",x:[origin[0],origin[0]+v[0]],y:[origin[1],origin[1]+v[1]],z:[origin[2],origin[2]+v[2]],mode:"lines+markers",name,line:{color:c,width:7},marker:{size:3,color:c}},{type:"cone",x:[origin[0]+v[0]],y:[origin[1]+v[1]],z:[origin[2]+v[2]],u:[v[0]],v:[v[1]],w:[v[2]],anchor:"tip",sizemode:"absolute",sizeref:.25,colorscale:[[0,c],[1,c]],showscale:false,showlegend:false}]);}
function metrics(rows,state){document.getElementById("metrics").textContent=rows.join("\n");window.labState=state;}
function limit(id,max){const node=document.getElementById(id);if(!node)return;node.max=max;values[id]=Math.min(values[id],max);node.value=values[id];node.parentElement.querySelector("output").textContent=values[id];}
async function render(){
 const v=values;chartNodes.forEach(node=>node.style.display="block");
 if(["sinusoidal","construction","similarity"].includes(group)){limit("position",v.length-1);limit("pair",Number(v.dim)/2-1);limit("columns",v.length);}
 if(group==="embedding-space"){limit("position",v.length-1);[0,1,2].forEach(i=>limit("axis"+i,Number(v.dim)-1));}
 if(group==="alibi"){limit("query",v.length-1);limit("relevant",v.length-1);}
 if(group==="comparison")limit("query",v.length-1);
 if(group==="extrapolation")limit("position",Number(v.length)-1);
 if(group==="rag")limit("evidence",v.chunks-1);
 if(["sinusoidal","construction","similarity"].includes(group)){
  const n=v.length,d=Number(v.dim),base=10**v.logbase,e=pe(n,d,base),p=Math.min(v.position,n-1),dims=seq(d);
  if(group==="sinusoidal"){
   const pair=Math.min(v.pair,d/2-1),indices=[0,1,2*pair,2*pair+1,d-2,d-1].filter((j,i,a)=>a.indexOf(j)===i);
   await plot(0,indices.map((j,i)=>line(seq(n),e.map(row=>row[j]),`${j%2?"cos":"sin"} dim ${j}`, [color.q,color.k,color.p,color.r,"#ba4b6e","#6f7e92"][i])).concat([{x:[p,p],y:[-1.1,1.1],mode:"lines",name:"selected position",line:{color:"#333",dash:"dash"}}]),"Sinusoidal frequencies",{xaxis:{title:{text:"Position (tokens)"}},yaxis:{title:{text:"Encoding value"},range:[-1.15,1.15]}});
   await plot(1,[{x:dims,y:e[p],type:"bar",marker:{color:dims.map(j=>j%2?color.k:color.q)}}],`Positional vector at ${p}`,{yaxis:{range:[-1,1]},xaxis:{title:{text:"Embedding dimension"}}});
   const omega=base**(-2*pair/d);metrics([`width=${d}; base=${fmt(base)}; selected position=${p}; pair=${pair}`,`omega=${fmt(omega)} rad/token; wavelength=${fmt(2*Math.PI/omega)} tokens/cycle`,`sin=${fmt(e[p][2*pair])}; cos=${fmt(e[p][2*pair+1])}; norm=${fmt(norm(e[p]))}`],{position:p,dim:d,base,vector:e[p],frequency:omega});
  }else if(group==="construction"){
   const columns=Math.min(n,v.columns);await plot(0,[heat(transpose(e).map(row=>row.map((x,i)=>i<columns?x:null)),-1,1)],`Constructed ${columns}/${n} positions`,{xaxis:{title:{text:"Token position"}},yaxis:{title:{text:"Dimension; even=sin, odd=cos"}}});
   for(const [i,width] of [8,16,32,64,128].entries())await plot(i+1,[heat(transpose(pe(n,width,base)),-1,1)],`Width ${width}; common scale [-1,1]`);
   metrics([`Selected p=${p}, width=${d}; columns visible=${columns}`,`Selected vector: ${e[p].slice(0,8).map(fmt).join(", ")}…`,`Blank is unconstructed; all dimension comparisons use the same base.`],{columns,position:p,vector:e[p]});
  }else{
   const denom=d/2,cos=e.map(a=>e.map(b=>dot(a,b)/denom));
   await plot(0,[heat(cos,-1,1)],"Pure positional cosine similarity",{xaxis:{title:{text:"Key position"}},yaxis:{title:{text:"Reference position"}}});
   const stride=Math.max(1,Math.ceil(n/64)),sample=seq(n).filter(i=>i%stride===0);
   await plot(1,[{type:"surface",x:sample,y:sample,z:sample.map(i=>sample.map(j=>cos[i][j])),cmin:-1,cmax:1,colorscale:"RdBu"}],"Similarity surface (bounded sampling)",{scene:{xaxis:{title:{text:"Position r"}},yaxis:{title:{text:"Position p"}},zaxis:{title:{text:"Cosine"},range:[-1,1]}}});
   await plot(2,[line(seq(n).map(i=>i-p),cos[p],`reference ${p}`,color.r)],"Cosine against signed displacement",{xaxis:{title:{text:"r - p (tokens)"}},yaxis:{range:[-1,1]}});
   const expected=seq(d/2).reduce((s,i)=>s+Math.cos(5*base**(-2*i/d)),0);metrics([`reference=${p}; width=${d}; base=${fmt(base)}`,`dot(PE0,PE5)=${fmt(dot(e[0],e[5]))}; sum cosine=${fmt(expected)}`,"Original-dimensional cosine; no semantic or trained-model claim."],{formulaError:Math.abs(dot(e[0],e[5])-expected),reference:p,cosine:cos[p]});
  }
 }
 if(group==="embedding-space"){
  const g=D.geometry[v.dim],n=v.length,p=Math.min(v.position,n-1),dim=Number(v.dim),indices=[0,1,2].map(i=>Math.min(v["axis"+i],dim-1));
  const original=pe(n,dim,C.base),coords=v.view==="pca"?g.coordinates.slice(0,n):original.map(row=>indices.map(i=>row[i]));
  const meta=seq(n).map(i=>[i,g.norm[i],g.previous_cosine[i],g.previous_distance[i]]);
  await plot(0,[trajectory(coords,meta),{type:"scatter3d",x:[coords[p][0]],y:[coords[p][1]],z:[coords[p][2]],mode:"markers",name:`selected ${p}`,marker:{size:9,color:color.r}}],v.view==="pca"?"Fixed PCA basis; prefix trajectory":`Exact original dimensions ${indices.join(",")}`,{scene});
  metrics([`view=${v.view}; width=${dim}; prefix=${n}; selected=${p}`,`PCA fit positions=0..${g.fit_positions-1}; explained ratios=${g.explained_variance_ratio.map(fmt).join(", ")}`,`original norm=${fmt(g.norm[p])}; previous cosine=${fmt(g.previous_cosine[p])}; previous L2=${fmt(g.previous_distance[p])}`,"Changing width uses its own incompatible basis; do not compare PCA coordinates across widths."],{position:p,coordinates:coords[p],explained:g.explained_variance_ratio,view:v.view});
 }
 if(group==="addition"){
  const a=D.addition,t=Number(v.token),p=v.position,x=a.content[t],e=pe(p+1,C.dim,C.base)[p],z=add(x,e),projection=v.view==="exact"?row=>row.slice(0,3):row=>mm([row],a.basis)[0];
  const xp=projection(x),ep=projection(e),zp=projection(z),reference=add(a.content[0],pe(1,C.dim,C.base)[0]);
  await plot(0,vectors3([[xp,"content",color.q],[ep,"position from origin",color.p],[ep,"position at content tip",color.p,xp],[zp,"sum",color.r]]),v.view==="exact"?"Exact dimensions 0,1,2":"Shared uncentered linear PCA basis",{scene});
  const other=projection(add(a.content[1],e));await plot(1,[{type:"scatter3d",x:[zp[0],other[0]],y:[zp[1],other[1]],z:[zp[2],other[2]],mode:"markers+text",text:[`selected token at ${p}`,`different token at ${p}`],marker:{size:7,color:[color.q,color.k]}}],"Different content at the same position",{scene});
  const error=Math.max(...sub(zp,add(xp,ep)).map(Math.abs));metrics([`token occurrence=${t}; assigned position=${p}; width remains ${C.dim}`,`norm content=${fmt(norm(x))}; position=${fmt(norm(e))}; combined=${fmt(norm(z))}`,`cosine to occurrence 0: before=${fmt(cosine(x,a.content[0]))}; after (reference at p=0)=${fmt(cosine(z,reference))}`,`projection addition error=${error.toExponential(3)}; coordinates=${v.view}`,"Occurrences 0 and 3 have identical content. The high-dimensional projection never subtracts a mean from vectors."],{position:p,token:t,additionError:error,combined:z,projectedCombined:zp});
 }
 if(group==="permutation"){
  const p=D.permutation,stage=v.stage,key=stage?"permuted":"original",labels=stage?p.permuted_tokens:p.tokens;
  for(const [i,method] of ["none","sinusoidal"].entries()){
   const item=p[method],state=item[key];await plot(i*3,[heat(state.input,-3,3)],`${method}: content/input · ${labels.join(" ")}`);
   await plot(i*3+1,[heat(state.logits,-5,5)],`${method}: scaled QK logits`);
   await plot(i*3+2,[heat(state.probabilities,0,1)],`${method}: attention probabilities`);
   await plot(6+i,[heat(stage?item.aligned_output:state.output,-3,3)],`${method}: output ${stage?"after undoing permutation":"original"}`);
  }
  metrics([`stage=${stage?"The cat chased the dog.":"The dog chased the cat."}`,`no-position max aligned error=${p.none.error.toExponential(4)}`,`fixed-position max aligned difference=${fmt(p.sinusoidal.error)}`,`joint token+coordinate permutation error=${p.joint_error.toExponential(4)}`,"Identical projections throughout. Matrices show actual endpoints, not invented interpolated tokens."],{stage,noPositionError:p.none.error,fixedPositionDifference:p.sinusoidal.error,jointError:p.joint_error});
 }
 if(group==="absolute-relative"){
  const positions=seq(6).map(i=>5+i),moved=positions.map(i=>i+v.shift),a=pe(6,C.dim,C.base,5),b=pe(6,C.dim,C.base,5+v.shift),table=D.table;
  const relative=positions.map(p=>positions.map(r=>-.08*Math.abs(p-r))),shifted=moved.map(p=>moved.map(r=>-.08*Math.abs(p-r)));
  await plot(0,[heat(transpose(a),-1,1)],"Sinusoidal pattern at positions 5–10");await plot(1,[heat(transpose(b),-1,1)],`Same content at positions ${moved[0]}–${moved[5]}`);
  await plot(2,[heat(positions.map(i=>table[i]),-2,2)],"Absolute table: original random rows");await plot(3,[heat(moved.map(i=>table[i]),-2,2)],"Absolute table: shifted random rows");await plot(4,[heat(shifted,-2,0)],"Relative bias indexed by query/key slots");
  metrics([`global shift=${v.shift}; sinusoidal coordinates max difference=${fmt(maxError(a,b))}`,`relative bias global-shift error=${maxError(relative,shifted).toExponential(3)}`,"Pure sinusoidal Gram relationships also survive a common shift; additive content attention need not.","The supplied absolute table is random initialization, not trained embeddings."],{shift:v.shift,relativeError:maxError(relative,shifted),sinusoidalDifference:maxError(a,b)});
 }
 if(group==="rope"){
  const p=v.p+v.shift,k=v.k+v.shift,w=v.frequency,q=[v.qm*Math.cos(v.qa),v.qm*Math.sin(v.qa)],key=[v.km*Math.cos(v.ka),v.km*Math.sin(v.ka)],qr=rotate2(q,p*w),kr=rotate2(key,k*w),relative=rotate2(key,(k-p)*w);
  const circle=seq(129).map(i=>2*Math.PI*i/128),traces=vectors2([[q,"original Q",color.gray],[key,"original K", "#a5b8ac"],[qr,"rotated Q",color.q],[kr,"rotated K",color.k]]);
  for(const radius of [v.qm,v.km])traces.push(line(circle.map(a=>radius*Math.cos(a)),circle.map(a=>radius*Math.sin(a)),`norm ${radius}`,"#c3cad5"));
  const arc=seq(65).map(i=>p*w*i/64);traces.push(line(arc.map(a=>.35*Math.cos(a)),arc.map(a=>.35*Math.sin(a)),"Q rotation arc",color.p));
  await plot(0,traces,"Two exact rotation planes in the same 2D coordinates",{xaxis:{range:[-2.4,2.4],title:{text:"Pair coordinate 0"}},yaxis:{range:[-2.4,2.4],scaleanchor:"x",title:{text:"Pair coordinate 1"}}});
  const offsets=seq(81).map(i=>i-40),scores=offsets.map(delta=>dot(q,rotate2(key,delta*w)));await plot(1,[line(offsets,scores,"fixed content dot",color.r),{x:[k-p],y:[dot(qr,kr)],type:"scatter",mode:"markers",name:"current",marker:{size:10,color:color.p}}],"Change relative displacement; scores can oscillate",{xaxis:{title:{text:"Key - query position"}},yaxis:{title:{text:"Unscaled Q·K"}}});
  metrics([`p=${p}; k=${k}; omega=${fmt(w)} rad/token; common shift=${v.shift}`,`absolute rotation angles: Q=${fmt(p*w)}, K=${fmt(k*w)}; relative=${fmt((k-p)*w)} rad`,`norms Q ${fmt(norm(q))} → ${fmt(norm(qr))}; K ${fmt(norm(key))} → ${fmt(norm(kr))}`,`dot before=${fmt(dot(q,key))}; rotated=${fmt(dot(qr,kr))}; relative-form=${fmt(dot(q,relative))}`],{p,k,frequency:w,rotatedQuery:qr,rotatedKey:kr,dot:dot(qr,kr),relativeDot:dot(q,relative),normError:Math.max(Math.abs(norm(q)-norm(qr)),Math.abs(norm(key)-norm(kr)))});
 }
 if(group==="rope-multifrequency"){
  const r=D.rope,qr=rope([r.q],[2],C.base)[0],kr=rope([r.k],[v.position],C.base)[0],contrib=seq(C.dim/2).map(i=>qr[2*i]*kr[2*i]+qr[2*i+1]*kr[2*i+1]);
  for(let pair=0;pair<Math.min(4,C.dim/2);pair++){
   const bound=Math.max(norm(r.q.slice(2*pair,2*pair+2)),norm(r.k.slice(2*pair,2*pair+2)))+.3;
   await plot(pair,vectors2([[qr.slice(2*pair,2*pair+2),"rotated Q",color.q],[kr.slice(2*pair,2*pair+2),"rotated K",color.k]]),`Pair ${pair}; ω=${fmt(C.base**(-2*pair/C.dim))}; K angle=${fmt(v.position*C.base**(-2*pair/C.dim))}`,{xaxis:{range:[-bound,bound]},yaxis:{range:[-bound,bound],scaleanchor:"x"}});
  }
  await plot(4,[{x:seq(contrib.length),y:contrib,type:"bar",marker:{color:color.r}}],"Every pair's contribution to the unscaled dot");
  const coords=r.coordinates.slice(0,Math.min(v.position+1,r.coordinates.length)),meta=seq(coords.length).map(i=>[i,norm(r.q),null,null]);await plot(5,[trajectory(coords,meta,"rotated query trajectory")],"High-dimensional rotary path in a fixed PCA projection",{scene});
  metrics([`key position=${v.position}; query position=2; all ${C.dim/2} pairs contribute`,`sum pair contributions=${fmt(contrib.reduce((a,b)=>a+b,0))}; full dot=${fmt(dot(qr,kr))}; scaled score=${fmt(dot(qr,kr)/Math.sqrt(C.dim))}`,`trajectory PCA explained=${r.explained_variance_ratio.map(fmt).join(", ")}`,"The four displayed planes are exact. All pair contributions are included in the bar plot."],{position:v.position,contributions:contrib,dot:dot(qr,kr),sum:contrib.reduce((a,b)=>a+b,0)});
 }
 if(group==="alibi"){
  const n=v.length,p=Math.min(v.query,n-1),relevant=Math.min(v.relevant,n-1),keys=seq(n),content=keys.map(i=>i===relevant?v.strength:0),bias=keys.map(i=>i<=p?-v.slope*(p-i):-Infinity),logits=add(content,bias),probs=softmax(logits),full=keys.map(q=>keys.map(k=>k<=q?-v.slope*(q-k):null));
  await plot(0,[line(keys,content,"content logits",color.q),line(keys,bias.map(x=>Number.isFinite(x)?x:null),"distance bias",color.p),line(keys,logits.map(x=>Number.isFinite(x)?x:null),"combined logits",color.r)],`Query ${p}; future keys > ${p} masked`,{xaxis:{title:{text:"Key position"}},yaxis:{title:{text:"Logit units"}}});
  await plot(1,[{x:keys,y:probs,type:"bar",marker:{color:keys.map(i=>i>p?color.gray:i===relevant?color.k:color.q)}}],"Normalized attention probabilities",{yaxis:{range:[0,1]}});await plot(2,[heat(full,-n,0)],"Causal bias matrix; blank = future mask");
  const slopes=seq(25).map(i=>.02+i*.98/24),dist=seq(32),surface=slopes.map(s=>softmax(dist.map(d=>-s*d)));
  await plot(3,[{type:"surface",x:dist,y:slopes,z:surface,colorscale:"Viridis"}],"Flat-content probability: 32 visible past distances",{scene:{xaxis:{title:{text:"Past distance (tokens)"}},yaxis:{title:{text:"Slope"}},zaxis:{title:{text:"Probability"}}}});
  metrics([`length=${n}; query=${p}; slope=${fmt(v.slope)}; relevant key=${relevant}; content strength=${fmt(v.strength)}`,`sum probabilities=${fmt(probs.reduce((a,b)=>a+b,0))}; expected past distance=${fmt(probs.reduce((s,a,i)=>s+a*Math.abs(p-i),0))}`,"Surface holds visible support fixed at 32 keys and isolates distance bias. It is distinct from the editable content plot."],{query:p,slope:v.slope,probabilities:probs,logits:logits.map(x=>Number.isFinite(x)?x:null),relevant});
 }
 if(group==="comparison"){
  const n=v.length,p=Math.min(v.query,n-1),methods=["none","sinusoidal","absolute","relative","rope","alibi"],results={};
  for(const [i,method] of methods.entries()){results[method]=attention(D.x.slice(0,n),D.weights,method,D.table,C.base);await plot(i,[heat(results[method].probs,0,1)],`${method}: same causal mask / projections`);}
  const chosen=results[v.method];await plot(6,[line(seq(n),chosen.logits[p].map(x=>Number.isFinite(x)?x:null),"logits",color.q)],`Selected ${v.method}; query ${p} logits`);await plot(7,methods.map((method,i)=>line(seq(n),results[method].probs[p],method,[color.q,color.p,color.k,color.r,"#b93f61",color.gray][i])),"Selected query probability distributions");
  metrics([`query=${p}; length=${n}; width=${C.dim}; additive methods modify input before Q/K/V projection`,...methods.map(method=>`${method}: entropy=${fmt(results[method].entropy[p])} nats; expected |distance|=${fmt(results[method].distance[p])}`),"All metrics describe synthetic distributions, not method quality."],{query:p,method:v.method,probabilities:chosen.probs[p],entropy:chosen.entropy[p],distance:chosen.distance[p]});
 }
 if(group==="extrapolation"){
  const n=Number(v.length),p=Math.min(v.position,n-1),sample=seq(96).map(i=>Math.round(i*(n-1)/95)),enc=sample.map(pos=>pe(1,C.dim,C.base,pos)[0]);
  await plot(0,[line(sample,enc.map(row=>row[0]),"sin dim 0",color.q),line(sample,enc.map(row=>row[C.dim-2]),"slow sine",color.p)],"Sampled positions; hypothetical training boundary at 128",{shapes:[{type:"line",x0:128,x1:128,y0:-1,y1:1,line:{color:color.r,dash:"dash"}}],xaxis:{title:{text:"Position (tokens)"}},yaxis:{range:[-1.1,1.1]}});
  const ref=pe(1,C.dim,C.base,p)[0];await plot(1,[line(sample,enc.map(row=>cosine(ref,row)),`cosine to position ${p}`,color.r),line(sample,sample.map(k=>-.1*Math.abs(p-k)),"ALiBi -0.1 distance (logit units)",color.p)],"Cosine and distance penalty use separate axes",{yaxis:{title:{text:"Cosine"},range:[-1,1]},yaxis2:{title:{text:"Distance penalty"},overlaying:"y",side:"right"}});
  await Plotly.restyle(chartNodes[1],{yaxis:"y2"},[1]);
  const rows=D.precision.rows;await plot(2,[line(rows.map(r=>r.position+1),rows.map(r=>Math.max(r.float32_error,1e-18)),"float32 construction",color.p),line(rows.map(r=>r.position+1),rows.map(r=>Math.max(r.float64_error,1e-18)),"float64 construction",color.q)],"Fixed relative offset 5; measured NumPy phase precision",{xaxis:{type:"log",title:{text:"Absolute origin + 1"}},yaxis:{type:"log",title:{text:"Absolute dot error vs relative float64 reference"}}});
  const coords=sample.map(i=>i/v.interpolate);await plot(3,[line(sample,coords,"effective coordinate p / factor",color.p)],"Coordinate interpolation only; no trained-model result",{shapes:[{type:"line",x0:0,x1:n-1,y0:127,y1:127,line:{color:color.r,dash:"dash"}}],xaxis:{title:{text:"Original position"}},yaxis:{title:{text:"Effective coordinate"}}});
  metrics([`represented length=${n}; selected=${p}; hypothetical table supports only 0..127`,`absolute table lookup at ${p}: ${p<128?"defined (random row)":"UNAVAILABLE — no learned extrapolation"}`,`coordinate interpolation factor=${v.interpolate}; effective selected coordinate=${fmt(p/v.interpolate)}`,"Curves use 96 sampled positions. Precision rows are actual Python float32/64 measurements; browser geometry uses JavaScript double precision.","No trained model, supported runtime capacity or effective-context accuracy was evaluated."],{length:n,position:p,tableAvailable:p<128,interpolatedPosition:p/v.interpolate,precision:rows});
 }
 if(group==="rag"){
  const count=v.chunks,slot=Math.min(v.evidence,count-1),units=v.words,system=12,query=8,chunks=seq(count).map(i=>({name:i===slot?"RELEVANT EVIDENCE":`distractor ${i}`,start:system+i*units,end:system+(i+1)*units}));
  const traces=[{x:[system],y:["prompt"],base:[0],type:"bar",orientation:"h",name:"system instructions",marker:{color:color.q}}];chunks.forEach((chunk,i)=>traces.push({x:[units],y:["prompt"],base:[chunk.start],type:"bar",orientation:"h",name:chunk.name,marker:{color:i===slot?color.p:color.gray},hovertemplate:`${chunk.name}: [${chunk.start}, ${chunk.end}) display units<extra></extra>`}));traces.push({x:[query],y:["prompt"],base:[system+count*units],type:"bar",orientation:"h",name:"user query",marker:{color:color.k}});
  await plot(0,traces,"Move the fact while keeping context size fixed",{barmode:"overlay",xaxis:{title:{text:"Whitespace display units — not model tokens"}},showlegend:false});
  const pad=(words,length,word)=>words.concat(Array(Math.max(0,length-words.length)).fill(word)).slice(0,length);
  const systemWords=pad(["SYSTEM","Answer","only","from","retrieved","evidence","and","cite","its","chunk","identifier","carefully"],system,"instruction");
  const blockWords=chunks.map((chunk,i)=>i===slot?pad(["Chunk",String(i),"Fact","access","code","is","ORCHID-69"],units,"evidence"):pad(["Chunk",String(i),"contains","unrelated","weather","information"],units,"distractor"));
  const queryWords=pad(["QUERY","What","is","the","access","code","Cite","evidence"],query,"query");
  const prompt=[systemWords,...blockWords,queryWords].map(words=>words.join(" ")).join("\n\n");
  let details=document.getElementById("prompt-details");if(!details){details=document.createElement("details");details.id="prompt-details";const summary=document.createElement("summary");summary.textContent="Inspect the actual synthetic prompt";const pre=document.createElement("pre");pre.id="prompt-text";pre.style.cssText="white-space:pre-wrap;background:white;padding:16px";details.append(summary,pre);document.querySelector("main").append(details);}document.getElementById("prompt-text").textContent=prompt;
  metrics([`context length=${system+count*units+query} display units; evidence slot=${slot+1}/${count}; evidence span=[${chunks[slot].start}, ${chunks[slot].end})`,"Synthetic fact: the access code is ORCHID-69. Distractors contain no answer.","Proposed measures: exact match, grounded answer accuracy, citation correctness, retrieval recall; control context budget and examples.","No LLM evaluated; there are no observed accuracy values or simulated quality curves."],{evidenceSlot:slot,contextUnits:prompt.trim().split(/\s+/).length,evidenceStart:chunks[slot].start,evidenceEnd:chunks[slot].end,modelEvaluated:false,prompt});
 }
}
async function schedule(){if(rendering){pending=true;return;}rendering=true;window.labReady=false;try{await render();document.getElementById("error").textContent="";window.labReady=true;}catch(error){document.getElementById("error").textContent=String(error);window.labError=String(error);throw error;}finally{rendering=false;if(pending){pending=false;schedule();}}}
setup();
// Reserve every chart before Plotly measures widths; adding later charts changes the grid.
const chartCount={sinusoidal:2,construction:6,"embedding-space":1,addition:2,similarity:3,permutation:8,"absolute-relative":5,rope:2,"rope-multifrequency":6,alibi:4,comparison:8,extrapolation:4,rag:1}[group];
for(let index=0;index<chartCount;index++){const node=document.createElement("div");node.className="chart";node.id="plot"+index;node.style.minWidth="0";document.getElementById("charts").append(node);chartNodes[index]=node;}
schedule();

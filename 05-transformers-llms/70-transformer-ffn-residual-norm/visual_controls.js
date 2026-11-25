"use strict";
// All inputs are synthetic; controls compute from embedded arrays, never a service.
const D = JSON.parse(document.getElementById("lab-data").textContent);
const C = ["#0072B2", "#D55E00", "#008767", "#7B4AA1", "#B83E50"];
const mean = a => a.reduce((s,x)=>s+x,0)/a.length;
const rms = a => Math.sqrt(mean(a.map(x=>x*x)));
const dot = (a,b) => a.reduce((s,x,i)=>s+x*b[i],0);
const mv = (x,w,b) => w[0].map((_,j)=>x.reduce((s,v,i)=>s+v*w[i][j],b?b[j]:0));
const add = (a,b) => a.map((x,i)=>x+b[i]);
const mul = (a,k) => a.map(x=>x*k);
const flatten = a => a.flat(Infinity);
function act(x,name){
  if(name==="identity")return x;
  if(name==="relu")return Math.max(x,0);
  if(name==="gelu")return .5*x*(1+Math.tanh(Math.sqrt(2/Math.PI)*(x+.044715*x*x*x)));
  if(name==="silu")return x*(x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x)));
  throw Error("Unknown activation");
}
function derivative(x,name){
  if(name==="identity")return 1;
  if(name==="relu")return x>0?1:0;
  if(name==="gelu"){const t=Math.tanh(Math.sqrt(2/Math.PI)*(x+.044715*x*x*x));return .5*(1+t)+.5*x*(1-t*t)*Math.sqrt(2/Math.PI)*(1+3*.044715*x*x);}
  const sigmoid=x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x));return sigmoid+x*sigmoid*(1-sigmoid);
}
function ln(x,eps=1e-5,gamma=1,beta=0){const m=mean(x),v=mean(x.map(z=>(z-m)**2));return x.map(z=>gamma*(z-m)/Math.sqrt(v+eps)+beta);}
function rn(x,eps=1e-5){const s=Math.sqrt(mean(x.map(z=>z*z))+eps);return x.map(z=>z/s);}
function scalar(a,L,alpha,residual){
  const base=Math.abs(residual?1+alpha*a:a);
  if(L===0)return {value:1,log10:0,status:"finite"};
  if(base===0)return {value:0,log10:null,status:"exact zero"};
  const log10=L*Math.log10(base);
  if(log10>Math.log10(Number.MAX_VALUE))return {value:null,log10,status:"float64 overflow"};
  if(log10<Math.log10(Number.MIN_VALUE))return {value:0,log10,status:"float64 underflow"};
  return {value:base**L,log10,status:"finite"};
}
const controls=document.getElementById("controls");
function slider(id,label,min,max,step,value){
  const wrapper=document.createElement("label");wrapper.innerHTML=`<span>${label}: <output id="${id}-value">${value}</output></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${value}">`;controls.append(wrapper);
  wrapper.querySelector("input").addEventListener("input",()=>{document.getElementById(id+"-value").textContent=get(id);queueRender();});
}
function select(id,label,options,value){const wrapper=document.createElement("label");wrapper.innerHTML=`<span>${label}</span><select id="${id}">${options.map(x=>`<option value="${x}" ${x===value?"selected":""}>${x}</option>`).join("")}</select>`;controls.append(wrapper);wrapper.querySelector("select").addEventListener("change",queueRender);}
function get(id){return Number(document.getElementById(id).value);}
function chosen(id){return document.getElementById(id).value;}
const mode=D.mode;
if(mode==="geometry"){
  slider("stage","Stage (input / expansion / activation / output)",0,3,1,0);
  select("activation","Activation",["identity","relu","gelu"],"gelu");
  select("camera","Camera preset",["oblique","front","top"],"oblique");
}
if(mode==="activations"){slider("scale","Input standard deviation",.1,3,.1,1);slider("offset","Input mean",-3,3,.1,0);select("view","Curves",["functions","derivatives"],"functions");}
if(mode.startsWith("residual")){slider("alpha","Residual branch scale",0,2,.05,1);if(mode==="residual-stream")slider("layers","Layers",1,8,1,8);}
if(mode==="scalar-gradients"){slider("depth","Depth",1,2000,1,24);slider("local","Local derivative a",-2,2,.05,.1);slider("alpha","Residual branch scale",0,2,.05,1);}
if(mode==="layernorm"){
  slider("scale","Input scale",0,3,.01,1);slider("offset","Input offset",-3,3,.1,0);
  slider("gamma","Shared gamma",-2,2,.1,1);slider("beta","Shared beta",-2,2,.1,0);
  slider("epsilon","log10 epsilon",-12,-1,1,-5);select("variance","Input fixture",["nonconstant","near-constant","constant"],"nonconstant");
}
if(mode.startsWith("normalization")){
  select("intervention","Intervention",Object.keys(D.normalization.variants),"base");
  select("method","Normalization",["LayerNorm","RMSNorm","BatchNorm train","BatchNorm eval"],"LayerNorm");
}
if(mode==="pre-post-norm"){
  const depths=[...new Set(D.transformer.map(r=>r.depth))].map(String);
  slider("depth-index","Measured depth index",0,depths.length-1,1,depths.length-1);
  select("metric","Measured metric",["gradient_l2","gradient_rms","relative_gradient","activation_rms","attention_update_ratio","ffn_update_ratio","consecutive_cosine","parameter_gradient_l2"],"gradient_l2");
  select("seed","Seed",["mean",...[...new Set(D.transformer.map(r=>r.seed))].map(String)],"mean");
  select("view","Representation",["profiles","surface"],"profiles");
}
if(mode==="swiglu"){slider("gate","Gate preactivation",-6,6,.1,1);slider("feature","Independent feature",-3,3,.1,1);}
if(mode==="parameters"||mode==="complexity"){
  slider("width","Model dimension",8,1024,8,D.config.d_model);slider("expansion","Expansion ratio",1,8,.25,D.config.expansion);
  slider("blocks","Blocks",1,96,1,D.config.depth);slider("tokens","Sequence/context length",8,4096,8,128);
  select("ffn","FFN type",["GELU","SwiGLU"],"GELU");select("workload","Workload",["full sequence","cached new token"],"full sequence");
}
const reset=document.createElement("button");reset.id="reset";reset.textContent="Reset controls";controls.append(reset);
const defaults=Object.fromEntries([...controls.querySelectorAll("input,select")].map(e=>[e.id,e.value]));
reset.addEventListener("click",()=>{for(const [id,value] of Object.entries(defaults)){document.getElementById(id).value=value;const label=document.getElementById(id+"-value");if(label)label.textContent=value;}queueRender();});
const animator=controls.querySelector('input[type="range"]');
if(animator){const play=document.createElement("button");play.id="play";play.textContent="Play / pause";controls.append(play);let timer;
  play.addEventListener("click",()=>{if(timer){clearInterval(timer);timer=null;return;}timer=setInterval(()=>{if(!window.labReady)return;let value=Number(animator.value)+Number(animator.step);if(value>Number(animator.max))value=Number(animator.min);animator.value=value;animator.dispatchEvent(new Event("input",{bubbles:true}));},650);});}
function line(x,y,name,color){return {x,y,name,type:"scatter",mode:"lines",line:{color,width:2}};}
function points(xyz,name,color,ids){return {type:"scatter3d",mode:"markers",x:xyz.map(p=>p[0]),y:xyz.map(p=>p[1]),z:xyz.map(p=>p[2]),name,ids:ids?.map(String),text:ids?.map(i=>"sample "+i),marker:{size:4,color},hovertemplate:"%{text}<br>(%{x:.4f}, %{y:.4f}, %{z:.4f})<extra>%{fullData.name}</extra>"};}
function path(xyz,name,color){return {...points(xyz,name,color),mode:"lines+markers",line:{color,width:5}};}
function heat(z,name){return {type:"heatmap",z,name,colorscale:"RdBu",zmid:0,hovertemplate:"row=%{y}, feature=%{x}<br>value=%{z:.6g}<extra></extra>"};}
function layout(title,three=false){return {title:{text:title,font:{size:17}},margin:{t:60,r:35,b:60,l:70},paper_bgcolor:"white",plot_bgcolor:"white",uirevision:"fixed",legend:{orientation:"h",y:-.13},font:{family:"system-ui",color:"#17283c"},
  xaxis:{title:{text:"Coordinate / layer"}},yaxis:{title:{text:"Value"}},...(three?{scene:{aspectmode:"cube",xaxis:{title:{text:"Feature 0"}},yaxis:{title:{text:"Feature 1"}},zaxis:{title:{text:"Feature 2"}},camera:{eye:{x:1.6,y:1.6,z:1.2}}}}:{})};}
const options={responsive:true,displaylogo:false,scrollZoom:true};
let pending=false;
function queueRender(){window.labReady=false;if(pending)return;pending=true;requestAnimationFrame(()=>{pending=false;render().catch(error=>{window.labError=String(error);console.error(error);});});}
async function render(){
  window.labReady=false;let traces=[],extra=[],state={},L=layout(""),E=layout("");
  if(mode==="geometry"){
    const name=chosen("activation"),stage=get("stage"),xyz=D.geometry.variants[name][stage];
    for(let k=0;k<3;k++){const indices=D.geometry.labels.map((c,i)=>c===k?i:-1).filter(i=>i>=0);traces.push(points(indices.map(i=>xyz[i]),"Synthetic cluster "+k,C[k],indices));}
    L=layout(["Input 3D","Expansion: selected coordinates 0–2 / 12","Activation: selected coordinates 0–2 / 12","Output 3D"][stage]+" · "+name,true);
    const bound=Math.max(...Object.values(D.geometry.variants).flat(3).map(Math.abs))*1.1;
    for(const axis of ["xaxis","yaxis","zaxis"])L.scene[axis].range=[-bound,bound];
    const camera=chosen("camera");L.scene.camera.eye=camera==="front"?{x:0,y:0,z:2.3}:camera==="top"?{x:0,y:2.3,z:0}:{x:1.6,y:1.6,z:1.2};L.scene.uirevision=camera;
    state={activation:name,stage,point0:xyz[0],sampleIDs:D.geometry.ids,affineCompositionError:D.geometry.affine_error,coordinatePolicy:D.geometry.coordinate_policy};
  }
  if(mode==="activations"){
    const x=Array.from({length:301},(_,i)=>-6+12*i/300),sample=D.activation_inputs.map(v=>get("offset")+get("scale")*v);
    const moments={};["identity","relu","gelu","silu"].forEach((name,i)=>{traces.push(line(x,x.map(v=>chosen("view")==="functions"?act(v,name):derivative(v,name)),name,C[i]));const output=sample.map(v=>act(v,name));extra.push({type:"histogram",x:output,name,opacity:.45,histnorm:"probability density",nbinsx:45,marker:{color:C[i]}});moments[name]={mean:mean(output),rms:rms(output),negativeFraction:output.filter(v=>v<0).length/output.length};});
    L=layout("Activation "+chosen("view"));L.xaxis.title.text="Input x";L.yaxis.title.text=chosen("view")==="functions"?"f(x)":"df/dx";
    E=layout("Same shifted/scaled synthetic input distribution");E.barmode="overlay";E.xaxis.title.text="Output value";E.yaxis.title.text="Density";
    state={scale:get("scale"),mean:get("offset"),moments,reluDerivativeAtZero:derivative(0,"relu"),geluApproximation:"tanh"};
  }
  if(mode.startsWith("residual")){
    const f=D.residual,alpha=get("alpha"),count=mode==="residual-stream"?get("layers"):1,x=f.x,update=f.first_update,y=add(x,mul(update,alpha));
    if(mode==="residual"){
      traces=[path([[0,0,0],x],"x",C[0]),path([x,y],"alpha F(x) translated to x tip",C[1]),path([[0,0,0],y],"x + alpha F(x)",C[2]),path([[0,0,0],update],"replacement F(x)",C[3])];
      traces.push({type:"cone",x:[y[0]],y:[y[1]],z:[y[2]],u:[alpha*update[0]],v:[alpha*update[1]],w:[alpha*update[2]],sizemode:"absolute",sizeref:.1,showscale:false,anchor:"tip",name:"branch direction",colorscale:[[0,C[1]],[1,C[1]]]});
      L=layout("Uncentered vector addition · original feature coordinates",true);state={alpha,x,branch:update,output:y,replacement:update,additionError:Math.max(...y.map((v,i)=>Math.abs(v-x[i]-alpha*update[i])))};
    }else{
      const run=residual=>{let states=[x];for(let i=0;i<count;i++){const branch=mv(states.at(-1),f.weights[i],f.biases[i]).map(Math.tanh);states.push(residual?add(states.at(-1),mul(branch,alpha)):branch);}return states;};
      const residual=run(true),replacement=run(false);traces=[path(residual,"residual",C[2]),path(replacement,"replacement",C[3])];L=layout("Same transformations, diverging state trajectories",true);state={alpha,layers:count,residual,replacement};
    }
    const coordinates=traces.flatMap(trace=>[...(trace.x||[]),...(trace.y||[]),...(trace.z||[])]);
    const bound=Math.max(1.5,...coordinates.map(Math.abs))*1.15;
    for(const axis of ["xaxis","yaxis","zaxis"])L.scene[axis].range=[-bound,bound];
    state.axisBound=bound;
  }
  if(mode==="scalar-gradients"){
    const a=get("local"),depth=get("depth"),alpha=get("alpha"),layers=Array.from({length:depth+1},(_,i)=>i),zeroCounts=[];
    [false,true].forEach((residual,i)=>{const values=layers.map(l=>scalar(a,l,alpha,residual));traces.push(line(layers,values.map(v=>v.value>0?v.value:null),residual?"|1+alpha a|^L":"|a|^L",C[i]));zeroCounts.push(values.filter(v=>v.value===0).length);});
    L=layout("Analytical scalar gradient magnitude · logarithmic axis");L.xaxis.title.text="Depth L";L.yaxis={type:"log",title:{text:"Absolute gradient (zeros omitted)"}};
    state={a,depth,alpha,plain:scalar(a,depth,alpha,false),residual:scalar(a,depth,alpha,true),zeroPointsOmitted:zeroCounts,interpretation:"Analytical model, not PyTorch measurement"};
  }
  if(mode==="layernorm"){
    const kind=chosen("variance"),scale=get("scale"),offset=get("offset"),eps=10**get("epsilon"),gamma=get("gamma"),beta=get("beta");
    const base=D.norm_vectors.map(x=>kind==="constant"?[1,1,1]:kind==="near-constant"?x.map(v=>1+v*1e-7):x);
    const input=base.map(x=>x.map(v=>v*scale+offset)),centered=input.map(x=>x.map(v=>v-mean(x))),normalized=input.map(x=>ln(x,eps)),output=input.map(x=>ln(x,eps,gamma,beta));
    traces=[points(input,"input",C[0]),points(centered,"centered",C[1]),points(normalized,"standardized",C[2]),points(output,"affine output",C[3])];
    const basis1=[1/Math.sqrt(2),-1/Math.sqrt(2),0],basis2=[1/Math.sqrt(6),1/Math.sqrt(6),-2/Math.sqrt(6)],circle=Array.from({length:101},(_,i)=>add(mul(basis1,Math.sqrt(3)*Math.cos(2*Math.PI*i/100)),mul(basis2,Math.sqrt(3)*Math.sin(2*Math.PI*i/100))));
    traces.push({...path(circle,"sqrt(3) reference circle",C[4]),mode:"lines"});
    traces.push({type:"surface",x:[[-2,2],[-2,2]],y:[[-2,-2],[2,2]],z:[[4,0],[0,-4]],opacity:.15,showscale:false,colorscale:[[0,C[2]],[1,C[2]]],name:"zero-mean plane",showlegend:true});
    L=layout("LayerNorm geometry · zero-mean plane and radius sqrt(3)",true);const bound=Math.max(2,...flatten(input).map(Math.abs),...flatten(output).map(Math.abs))*1.1;for(const axis of ["xaxis","yaxis","zaxis"])L.scene[axis].range=[-bound,bound];
    extra=[heat([input[0],centered[0],normalized[0],output[0]],"First sample actual stages")];E=layout("First vector: input / centered / standardized / affine output");E.yaxis.tickvals=[0,1,2,3];E.yaxis.ticktext=["input","centered","normalized","affine"];
    state={scale,offset,eps,gamma,beta,kind,firstInput:input[0],firstOutput:output[0],centeredPlaneMaxError:Math.max(...centered.map(x=>Math.abs(x.reduce((a,b)=>a+b,0)))),normalizedNormRange:[Math.min(...normalized.map(rms))*Math.sqrt(3),Math.max(...normalized.map(rms))*Math.sqrt(3)],gaussianAssumption:false};
  }
  if(mode.startsWith("normalization")){
    const intervention=chosen("intervention"),method=chosen("method"),record=D.normalization.variants[intervention],input=record.input,output=record[method];
    if(mode==="normalization"){
      traces=[heat(output.flat(),method+" output")];extra=[heat(input.flat(),"input")];L=layout(method+" · rows=(batch,token); columns=features");E=layout("Input tensor under intervention: "+intervention);
    }else{traces=[points(input.flat(),"input",C[0]),points(record.LayerNorm.flat(),"LayerNorm",C[2]),points(record.RMSNorm.flat(),"RMSNorm",C[3])];L=layout("Original three-dimensional features · "+intervention,true);}
    const base=D.normalization.variants.base[method][0],current=output[0],coupling=Math.max(...flatten(current).map((v,i)=>Math.abs(v-flatten(base)[i])));
    state={method,intervention,batch:input.length,tokens:input[0].length,features:input[0][0].length,firstVector:output[0][0],firstSampleChangeFromBase:coupling,runningMean:D.normalization.running_mean,runningVariance:D.normalization.running_var,layout:D.normalization.layout};
  }
  if(mode==="pre-post-norm"){
    const depths=[...new Set(D.transformer.map(r=>r.depth))],depth=depths[get("depth-index")],metric=chosen("metric"),seed=chosen("seed"),conditions=["pre","post","pre+final-LN"];
    const rows=D.transformer.filter(r=>seed==="mean"||r.seed===Number(seed));
    const values=(condition,d)=>Array.from({length:d+1},(_,i)=>{const entries=rows.filter(r=>r.depth===d&&r.condition===condition&&r.layer===i&&r[metric]!=null).map(r=>r[metric]);return entries.length?mean(entries):null;});
    if(chosen("view")==="surface"){
      for(const [j,condition] of conditions.entries())traces.push({type:"surface",x:Array.from({length:Math.max(...depths)+1},(_,i)=>i),y:depths,z:depths.map(d=>{const v=values(condition,d);return Array.from({length:Math.max(...depths)+1},(_,i)=>i<v.length?v[i]:null);}),name:condition,showscale:j===0,opacity:.65,colorscale:j===0?"Blues":j===1?"Oranges":"Purples",showlegend:true});
      L=layout("Measured depth × layer × "+metric,true);L.scene.xaxis.title.text="Layer position";L.scene.yaxis.title.text="Network depth";L.scene.zaxis.title.text=metric;
    }else{for(const [i,condition] of conditions.entries())traces.push(line(Array.from({length:depth+1},(_,j)=>j),values(condition,depth),condition,C[i]));L=layout("Measured profiles · depth="+depth);L.yaxis.title.text=metric;if(metric.includes("gradient"))L.yaxis.type="log";}
    state={depth,seed,metric,view:chosen("view"),values:Object.fromEntries(conditions.map(c=>[c,values(c,depth)])),source:"Actual PyTorch CSV measurements; no interpolation between depths",optimizerSteps:0};
  }
  if(mode==="swiglu"){
    const gate=get("gate"),feature=get("feature"),activated=act(gate,"silu"),product=activated*feature;
    traces=[{type:"bar",x:["gate preactivation","SiLU(gate)","independent feature","product"],y:[gate,activated,feature,product],marker:{color:[C[3],C[3],C[0],C[2]]},text:[gate,activated,feature,product].map(v=>v.toFixed(3)),textposition:"auto"}];
    L=layout("Manual signed feature gating · no probability interpretation");extra=[heat(D.swiglu.gated_features,"Actual projected token features")];E=layout("Computed 8 → "+D.swiglu.gated_width+" → 8 SwiGLU product");
    state={gate,activated,feature,product,denseWidth:D.swiglu.dense_width,gatedWidth:D.swiglu.gated_width,denseParameters:D.swiglu.dense_count,gatedParameters:D.swiglu.gated_count,bias:true};
  }
  if(mode==="parameters"||mode==="complexity"){
    const d=get("width"),h=Math.round(d*get("expansion")),layers=get("blocks"),T=get("tokens"),gated=chosen("ffn")==="SwiGLU",cached=chosen("workload")==="cached new token",factor=gated?3:2;
    const attention=4*d*d+4*d,ffn=factor*d*h+(gated?2*h+d:h+d),norms=4*d,total=(attention+ffn+norms)*layers;
    const macs=n=>({attention:layers*(4*(cached?1:n)*d*d+2*(cached?n:n*n)*d),ffn:layers*factor*(cached?1:n)*d*h});
    if(mode==="parameters"){
      traces=[{type:"bar",x:["attention","FFN","two affine LayerNorms"],y:[attention,ffn,norms].map(v=>v*layers),marker:{color:[C[1],C[0],C[3]]}}];L=layout("Parameters including biases · all blocks");L.yaxis.title.text="Parameters";
      const widths=Array.from({length:32},(_,i)=>(i+1)*32);extra=[line(widths,widths.map(w=>(4*w*w+4*w+factor*w*Math.round(w*get("expansion"))+(gated?2*Math.round(w*get("expansion"))+w:Math.round(w*get("expansion"))+w)+4*w)*layers),"Total parameters versus d",C[0])];E=layout("Model width changes the parameter budget");E.xaxis.title.text="Model width d";E.yaxis.title.text="Parameters";
    }else{
      const lengths=Array.from({length:64},(_,i)=>(i+1)*64);traces=[line(lengths,lengths.map(t=>macs(t).attention),"attention MACs",C[1]),line(lengths,lengths.map(t=>macs(t).ffn),"FFN MACs",C[0])];L=layout("Theoretical "+chosen("workload")+" matrix arithmetic");L.xaxis.title.text="Sequence / cache length T";L.yaxis.title.text="MACs";
      const expansions=Array.from({length:29},(_,i)=>1+i*.25);extra=[line(expansions,expansions.map(e=>layers*factor*(cached?1:T)*d*Math.round(d*e)),"FFN MACs versus expansion",C[0])];E=layout("Expansion cost at selected sequence length");E.xaxis.title.text="Expansion ratio h/d";E.yaxis.title.text="FFN MACs";
    }
    state={d,h,layers,T,gated,cached,parameterCounts:{attention:attention*layers,ffn:ffn*layers,normalization:norms*layers,total},weightBytes:{FP32:total*4,FP16:total*2,BF16:total*2,INT8_ideal:total},macs:macs(T),flops:2*(macs(T).attention+macs(T).ffn),convention:"1 MAC = 2 FLOPs; excludes nonlinearities, norms, softmax and memory movement",memoryScope:"weights only; no KV cache, optimizer or activations"};
  }
  await Plotly.react("plot",traces,L,options);
  const second=document.getElementById("secondary");if(extra.length){second.style.display="block";await Plotly.react(second,extra,E,options);}else second.style.display="none";
  window.labState=state;document.getElementById("metrics").textContent=JSON.stringify(state,null,2);window.labReady=true;
}
window.labMath={activation:act,derivative,layerNorm:ln,rmsNorm:rn,scalar};
queueRender();

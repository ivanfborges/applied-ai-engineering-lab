/* Finite precomputed choices. This document never runs or approximates a network in JS. */
"use strict";
const LAB = JSON.parse(document.getElementById("lab-data").textContent);
const palette = {small:"#c08615",xavier:"#2878b5",he:"#16846b",large:"#c74440",he_layer:"#8458aa"};
const $ = id => document.getElementById(id);
const log = a => a.map(v => v !== null && Number.isFinite(v) && v > 0 ? Math.log10(v) : null);
const fmt = v => v === null ? "non-finite" : Number(v).toExponential(3);
const options = {responsive:true,displaylogo:false,scrollZoom:false};
const layout = (title, x, y) => ({title:{text:title,font:{size:16}},xaxis:{title:{text:x}},yaxis:{title:{text:y}},
  margin:{l:72,r:24,b:64,t:60},paper_bgcolor:"white",plot_bgcolor:"white",font:{family:"system-ui, sans-serif",color:"#17324d"},
  legend:{orientation:"h",y:-.23,font:{size:11}},uirevision:title});
function plot(id, traces, config) { return Plotly.react(id,traces,config,{...options,scrollZoom:id==="surface"}); }
function line(x, raw, name, color, dash="solid") {
  return {x,y:log(raw),customdata:raw,type:"scatter",mode:"lines+markers",name,
    line:{color,dash,width:2},marker:{size:4},connectgaps:false,
    hovertemplate:"%{x}<br>measured=%{customdata:.4e}<br>log10=%{y:.3f}<extra>%{fullData.name}</extra>"};
}
function fill(id, values) {
  $(id).replaceChildren(...values.map(v=>{const o=document.createElement("option");o.value=v;o.textContent=v;return o;}));
}
for (const [id,key] of [["depth","depth"],["width","width"],["activation","activation"],["normalization-choice","normalization"],["seed","seed"]]) {
  const values=[...new Set(LAB.grid.map(r=>r.config[key]))];
  fill(id,values); $(id).value=String(LAB.config[key]);
}
for (const strategy of Object.keys(palette).filter(k=>k!=="he_layer")) {
  const option=document.createElement("option");option.value=strategy;option.textContent=strategy;$("strategy").appendChild(option);
}
fill("train-seed", LAB.seeds);

async function signals() {
  const c={depth:+$("depth").value,width:+$("width").value,activation:$("activation").value,
    normalization:$("normalization-choice").value,seed:+$("seed").value};
  const record=LAB.grid.find(r=>Object.keys(c).every(k=>r.config[k]===c[k]));
  const strategies=$("strategy").value==="all" ? Object.keys(record.results) : [$("strategy").value];
  const forward=[], backward=[], parameters=[], hist=[];
  let zeros=0,nonfinite=0,subnormal=0;
  const objective=$("objective").value;
  for (const name of strategies) {
    const r=record.results[name], x=r.layers.map(l=>l.layer), color=palette[name];
    forward.push(line(x,r.layers.map(l=>l.q),name,color));
    backward.push(line(x,r.layers.map(l=>l[objective]),name,color));
    parameters.push(line(x.slice(1),r.weight_loss_rms,name,color));
    for(const v of r.weight_loss_rms) {
      if(v===null)nonfinite++;else if(v===0)zeros++;
      else if(Math.abs(v)<2.2250738585072014e-308)subnormal++;
    }
    for (const row of r.layers) for(const key of ["q",objective]) {
      if(row[key]===null)nonfinite++;else if(row[key]===0)zeros++;
      if(row.status[key]==="subnormal")subnormal++;
    }
    const histogramNames=$("strategy").value==="all" ? ["xavier","he"] : strategies;
    if (!histogramNames.includes(name))continue;
    Object.entries(r.samples).forEach(([depth,values],index)=>{
      const row=r.layers[+depth];
      hist.push({type:"histogram",x:values.filter(v=>v!==null),name:`${name} / layer ${depth}`,
        xaxis:index===0?"x":`x${index+1}`,yaxis:index===0?"y":`y${index+1}`,opacity:.5,nbinsx:35,
        marker:{color},meta:`q=${fmt(row.q)}, var=${fmt(row.variance)}, mean=${fmt(row.mean)}`,
        hovertemplate:"Value %{x}<br>Subsample count %{y}<br>%{meta}<extra>%{fullData.name}</extra>"});
    });
  }
  $("configuration").textContent=`Measured batch ${record.config.batch}; CPU float64; depth ${c.depth}, width ${c.width}, ${c.activation}, ${c.normalization}, seed ${c.seed}. BN is in training mode.`;
  $("status").textContent=`Selected log metrics: ${zeros} zero entries, ${nonfinite} non-finite entries, ${subnormal} subnormal entries. Zeros/non-finite values are gaps, never replaced by a positive floor.`;
  const histLayout=layout("A3 | Activation distributions: early / middle / final", "Activation", "Subsample count");
  histLayout.grid={rows:1,columns:3,pattern:"independent"};histLayout.barmode="overlay";
  histLayout.xaxis.domain=[0,.28];histLayout.xaxis2={domain:[.36,.64],title:{text:"Activation"}};
  histLayout.xaxis3={domain:[.72,1],title:{text:"Activation"}};
  await Promise.all([
    plot("forward",forward,layout("A2 | Forward magnitude","Layer depth","log10(second moment q)")),
    plot("backward",backward,layout("C2 | Activation gradients","Layer depth",`log10(${objective==="probe_rms"?"fixed-adjoint RMS":"loss gradient RMS"})`)),
    plot("parameters",parameters,layout("C2 | Linear-weight gradients: squared-output loss","Weight layer (1-based)","log10(parameter-gradient RMS)")),
    plot("histograms",hist,histLayout)
  ]);
  window.labDiagnostics={configuration:c,zeros,nonfinite,subnormal};
}

async function theory() {
  const traces=[];
  for(const name of ["xavier","he"]) {
    LAB.repeats[name].forEach((r,i)=>{
      const t=line(r.layers.map(l=>l.layer),r.layers.map(l=>l.q/r.layers[0].q),`${name}, seed ${r.config.seed}`,palette[name]);
      t.opacity=.5;traces.push(t);
    });
    const x=LAB.repeats[name][0].layers.map(l=>l.layer);
    const t=line(x,x.map(d=>name==="xavier"?Math.pow(.5,d):1),`${name} theoretical`,palette[name],"dash");
    t.hovertemplate="Depth %{x}<br>Analytical log10(q/q0)=%{y:.3f}<extra>%{fullData.name}</extra>";traces.push(t);
  }
  const r=LAB.relu;
  await Promise.all([
    plot("paired",traces,layout(`B1 | Baseline ReLU/no norm: ${LAB.seeds.length} paired seeds`,"Layer depth","log10(q / q_input)")),
    plot("relu",[
      {type:"histogram",x:r.before,name:"Z (symmetric sample)",opacity:.5,marker:{color:"#53738c"},nbinsx:60},
      {type:"histogram",x:r.after,name:"ReLU(Z)",opacity:.5,marker:{color:palette.he},nbinsx:60}
    ],{...layout(`B3 | q after/before=${(r.q_after/r.q_before).toFixed(3)}; variance ratio=${(r.var_after/r.var_before).toFixed(3)}`,"Value","Count in displayed subsample"),barmode:"overlay"})
  ]);
}

async function surface() {
  const s=LAB.surface,key=$("surface-metric").value,raw=s[key];
  const traces=[{type:"surface",x:s.depths,y:s.scales,z:raw.map(log),customdata:raw,colorscale:"RdBu",reversescale:true,
    colorbar:{title:{text:"log10(metric)"},len:.7},connectgaps:false,
    hovertemplate:"Depth %{x}<br>Weight std / He %{y:.3f}<br>Measured %{customdata:.5e}<br>log10 %{z:.3f}<extra></extra>"}];
  for(const [name,scale] of [["xavier",1/Math.sqrt(2)],["he",1]]) {
    const i=s.scales.findIndex(v=>Math.abs(v-scale)<1e-12);
    traces.push({type:"scatter3d",x:s.depths,y:s.depths.map(()=>scale),z:log(raw[i]),mode:"lines+markers",name:`${name} reference`,
      line:{color:palette[name],width:7},marker:{size:3},customdata:raw[i],
      hovertemplate:"Depth %{x}<br>Scale %{y:.3f}<br>Measured %{customdata:.4e}<extra>%{fullData.name}</extra>"});
  }
  const bad=raw.flat().filter(v=>v===null||v<=0).length;
  $("surface-note").textContent=`Measured ReLU/no-normalization grid: width ${s.config.width}, batch 16, seed ${s.config.seed}; ${s.depths.length} depths × ${s.scales.length} scales. ${bad} zero/non-finite cells are gaps. Each depth has its own measured backward pass, with a shared output adjoint of RMS 1.`;
  await plot("surface",traces,{...layout("D | Depth × initialization scale","",""),
    scene:{xaxis:{title:{text:"Network depth"}},yaxis:{title:{text:"Weight std / He std"}},
      zaxis:{title:{text:key==="forward"?"log10(activation second moment)":"log10(input probe RMS)"}},aspectmode:"cube"},
    margin:{t:65,l:10,r:20,b:40},uirevision:"preserve-camera"});
}

async function normalization() {
  const n=LAB.normalization,key=$("norm-view").value,values=n[key];
  const before=n.before.flat(),chosen=values.flat();
  const limit=Math.max(...before.map(Math.abs));
  await Promise.all([
    plot("heatmap",[{type:"heatmap",z:values,zmin:-limit,zmax:limit,colorscale:"RdBu",reversescale:true,
      hovertemplate:"Feature %{x}<br>Example %{y}<br>Value %{z:.4f}<extra></extra>"}],layout(`E2 | ${key}: shared color scale`,"Feature (columns)","Example (rows)")),
    plot("norm-hist",[{type:"histogram",x:before,name:"Before",opacity:.4,marker:{color:"#53738c"}},
      {type:"histogram",x:chosen,name:key,opacity:.6,marker:{color:palette.he_layer}}],
      {...layout("E1 | Before and after distributions","Activation","Count"),barmode:"overlay"}),
    plot("batchsize",[0,1,2,3].map(f=>({type:"scatter",mode:"lines+markers",x:n.sensitivity.map(r=>r.batch),
      y:n.sensitivity.map(r=>r.feature_mean_sd[f]),name:`Feature ${f}`})),
      {...layout(`E3 | Mean-estimate variability over ${n.repetitions} batches`,"Batch size","SD of estimated mean"),xaxis:{type:"log",title:{text:"Batch size"}}}),
    plot("anchor",[{x:[0,1,2,3],y:n.anchor_bn_a,name:"BN companions A",line:{color:palette.he}},
      {x:[0,1,2,3],y:n.anchor_bn_b,name:"BN shifted companions B",line:{color:palette.large}},
      {x:[0,1,2,3],y:n.anchor_ln_a,name:"LN A = B",line:{color:palette.he_layer}}],
      layout("E3 | Same anchor, different batch companions","Feature","Normalized anchor value"))
  ]);
}

async function bnMode() {
  const n=LAB.normalization,key=$("bn-mode").value;
  $("bn-note").textContent=`Running buffers populated by ${n.training_batches_for_eval} actual forward batches (100 reference batches + the held batch's training-mode pass). Evaluation uses these stored buffers. Means: ${n.running_mean.map(v=>v.toFixed(2)).join(", ")}; variances: ${n.running_var.map(v=>v.toFixed(2)).join(", ")}. No weights were optimized.`;
  const limit=Math.max(...n.train_test.flat().map(Math.abs),...n.eval_test.flat().map(Math.abs));
  await plot("bn-eval",[{type:"heatmap",z:n[key],zmin:-limit,zmax:limit,colorscale:"RdBu",reversescale:true}],
    layout(`E4 | Same held batch: ${key==="train_test"?"training batch statistics":"populated running statistics"}`,"Feature","Example"));
}

async function training() {
  const seed=+$("train-seed").value,lr=+$("lr").value;
  const runs=LAB.training.filter(r=>r.seed===seed&&r.lr===lr),losses=[],updates=[];
  if($("clipping").value==="yes"&&lr===.15)runs.push(...LAB.clipping.filter(r=>r.seed===seed));
  for(const r of runs) {
    const name=r.strategy+(r.clip?" + clip=1":""),color=r.clip?palette.he_layer:palette[r.strategy],h=r.history,x=h.map(v=>v.epoch);
    losses.push(line(x,h.map(v=>v.train_loss),name+" train",color));
    losses.push(line(x,h.map(v=>v.validation_loss),name+" validation",color,"dash"));
    updates.push(line(x,h.map(v=>v.gradient_norm),name+" gradient (pre-clip)",color));
    updates.push(line(x,h.map(v=>v.update_norm),name+" update",color,"dash"));
    if(r.clip)updates.push(line(x,h.map(v=>v.clipped_gradient_norm),name+" gradient after clip",color,"dot"));
  }
  $("training-status").textContent=runs.map(r=>`${r.strategy}${r.clip?" +clip":""}: ${r.nonfinite?`STOP at epoch ${r.nonfinite.epoch}, first non-finite ${r.nonfinite.stage}`:`finite through epoch ${r.history.at(-1).epoch}`}`).join(" · ");
  await Promise.all([plot("losses",losses,layout(`F | Actual regression, lr=${lr}, seed=${seed}`,"Completed full-batch updates","log10(MSE)")),
    plot("updates",updates,layout("F | Different quantities: gradient and actual update norms","Update from epoch","log10(global L2 norm)"))]);
}

async function conditioning() {
  const traces=[];
  if(LAB.jacobian) {
    for(const r of LAB.jacobian.records)traces.push(line(r.singular_values.map((_,i)=>i+1),r.singular_values,
      `${r.strategy}, depth ${r.depth}, numerical rank ${r.numerical_rank}`,palette[r.strategy],r.depth===1?"dot":r.depth===5?"dash":"solid"));
    $("jacobian-note").textContent="Full mode: measured 12×12 Jacobians from one synthetic input, exact autograd followed by SVD. Zero singular values are gaps. Numerical rank uses width × machine epsilon × largest singular value. A reasonable q does not establish directional conditioning.";
  } else {
    traces.push({x:[1,2,3,4,5,6,7,8,9,10,11,12],y:Array(12).fill(1),name:"Identity (analytical control)",mode:"lines+markers"},
      {x:[1,2,3,4,5,6,7,8,9,10,11,12],y:[Math.sqrt(12),...Array(11).fill(0)],name:"Rank-one (analytical control)",mode:"lines+markers"});
    $("jacobian-note").textContent="Analytical linear-map control: both have mean(singular value²)=1, but one loses eleven directions. This is constructed, not a measured neural network. Run --mode full for actual input-output Jacobian SVDs.";
  }
  await plot("jacobian",traces,layout("G | Conditioning and directional information","Singular-value rank (descending)",LAB.jacobian?"log10(singular value)":"Singular value (linear scale)"));
}

function guard(fn) {return async()=>{window.labReady=false;try{await fn();window.labReady=true;}catch(error){window.labError=String(error);console.error(error);}};}
for(const id of ["depth","width","activation","normalization-choice","seed","strategy","objective"])$(id).addEventListener("change",guard(signals));
$("surface-metric").addEventListener("change",guard(surface));
$("norm-view").addEventListener("change",guard(normalization));
$("bn-mode").addEventListener("change",guard(bnMode));
for(const id of ["lr","train-seed","clipping"])$(id).addEventListener("change",guard(training));
$("protocol").textContent=`Seed set ${LAB.seeds.join(", ")}; baseline depth ${LAB.config.depth}, width ${LAB.config.width}, batch ${LAB.config.batch}. Synthetic data only. `+
  `Small std=0.25×He, oversized std=1.6×He; Xavier gain=1, He fan-in/ReLU gain. Tanh is a controlled mismatch with the same scales, not an activation-tuned recommendation. `+
  `Normalization epsilon=1e-5, before activation, gamma=1/beta=0. Python ${LAB.versions.python}; NumPy ${LAB.versions.numpy}; PyTorch ${LAB.versions.torch}.`;
guard(async()=>{await signals();await theory();await surface();await normalization();await bnMode();await training();await conditioning();})();

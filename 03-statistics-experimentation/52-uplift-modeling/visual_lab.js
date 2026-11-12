/* Offline view logic. Python owns data generation, fitting and statistical records. */
"use strict";
const DATA = JSON.parse(document.getElementById("lab-data").textContent);
window.LAB_DATA = DATA;
const $ = id => document.getElementById(id);
const C = {positive:"#167d9a", negative:"#b23a48", control:"#60788b", light:"#f5f2e8", gold:"#b88926"};
const EFFECT = [[0,C.negative],[0.5,C.light],[1,C.positive]];
const labels = ["Potential outcomes","Response vs effect","CATE surfaces","S- and T-Learners","Sample size & noise","Uplift curves","Costs & targeting","Confounding & overlap","Production decisions"];
const sum = a => a.reduce((x,y)=>x+y,0);
const mean = a => sum(a)/a.length;
const fmt = (x,d=3) => Number(x).toFixed(d);
const signed = (x,d=3) => (x>0?"+":"")+fmt(x,d);
const config = {responsive:true, displaylogo:false, scrollZoom:true, showLink:false,
  modeBarButtonsToRemove:["sendDataToCloud","lasso2d","select2d"], toImageButtonOptions:{format:"png",scale:1.5}};
let active = 1, revision = 0;
const timers = {};
function baseLayout(title, extra={}) {
  return {title:{text:title,font:{size:15,color:"#193449"}}, paper_bgcolor:"white",plot_bgcolor:"white",
    margin:{t:55,l:62,r:24,b:58},font:{family:"system-ui,Segoe UI,sans-serif",size:12,color:"#304c60"},
    xaxis:{zerolinecolor:"#c5d4df",gridcolor:"#edf1f4"},yaxis:{zerolinecolor:"#c5d4df",gridcolor:"#edf1f4"},
    legend:{orientation:"h",y:-.24},uirevision:"retain-view",...extra};
}
async function plot(id,traces,title,layout={}) {
  await Plotly.react($(id),traces,baseLayout(title,layout),config);
}
function stat(value,label) { return `<div class="stat"><strong>${value}</strong><span>${label}</span></div>`; }
function heat(id,z,title,options={}) {
  return plot(id,[{type:"heatmap",x:DATA.axis,y:DATA.axis,z,colorscale:EFFECT,zmin:-1,zmax:1,
    colorbar:{title:{text:"Probability<br>points"},thickness:12},
    hovertemplate:"Engagement %{x:.2f}<br>Price sensitivity %{y:.2f}<br>Value %{z:.4f}<extra></extra>",...options}],title,
    {xaxis:{title:{text:"Engagement (standard units)"}},yaxis:{title:{text:"Price sensitivity (standard units)"}}});
}
function surfaces(strength) {
  const axis=DATA.axis, p0=[],p1=[],tau=[];
  for(const b of axis) {
    const r0=[],r1=[],rt=[];
    for(const a of axis) {
      const l=-.3+.85*a-.65*b+.25*Math.sin(a*b);
      const h=.9*Math.tanh(b)-.7*Math.tanh(a)+.5*Math.sin(a*b);
      const c=1/(1+Math.exp(-l)),t=1/(1+Math.exp(-(l+strength*h)));
      r0.push(c);r1.push(t);rt.push(t-c);
    }
    p0.push(r0);p1.push(r1);tau.push(rt);
  }
  return {p0,p1,tau};
}
function selectPolicy(value,cost,capacity) {
  const scores=DATA.test.score;
  const order=scores.map((_,i)=>i).sort((a,b)=>scores[b]-scores[a]||a-b);
  const k=Math.floor(scores.length*capacity/100);
  return order.slice(0,k).filter(i=>value*scores[i]>cost);
}
function policyMetrics(indices,value,cost) {
  const d=DATA.test;
  const predicted=sum(indices.map(i=>d.score[i]));
  const truth=sum(indices.map(i=>d.tau[i]));
  const ipw=sum(indices.map(i=>d.z[i]));
  return {selected:indices.length,predicted_conversions:predicted,true_conversions:truth,
    ipw_conversions:ipw,contact_cost:cost*indices.length,predicted_net:value*predicted-cost*indices.length,
    true_net:value*truth-cost*indices.length,ipw_net:value*ipw-cost*indices.length};
}
function gainAt(curve,q) {
  if(q<=0)return 0;
  if(q>=1)return curve.gain[curve.gain.length-1];
  let low=0,high=curve.q.length-1;
  while(high-low>1){const mid=Math.floor((low+high)/2);if(curve.q[mid]<=q)low=mid;else high=mid;}
  const weight=(q-curve.q[low])/(curve.q[high]-curve.q[low]);
  return curve.gain[low]*(1-weight)+curve.gain[high]*weight;
}
function toggleTimer(name,callback,delay=600) {
  if(timers[name]) {clearInterval(timers[name]);delete timers[name];return;}
  timers[name]=setInterval(()=>{callback();update();},delay);
}

async function potential() {
  const d=DATA.individual, stage=+$('stage').value, reveal=$('reveal').checked;
  const steps=["Define pre-treatment X","Two hypothetical actions","Randomize assignment, e=0.5","Reveal only the assigned outcome","Keep the counterfactual missing"];
  $('stage-value').textContent=steps[stage];
  $('person').textContent=`One synthetic person · Engagement ${fmt(d.x[0],2)} · Price sensitivity ${fmt(d.x[1],2)}`;
  for(const arm of [0,1]) {
    const observed=stage>=3&&arm===d.t;
    $('branch'+arm).classList.toggle('observed',observed);
    const outcome=observed?`Observed Y(${arm}) = ${d['y'+arm]}`:
      reveal?`SIMULATION ORACLE ONLY: Y(${arm}) = ${d['y'+arm]}`:"Y("+arm+"): HIDDEN";
    $('branch'+arm).innerHTML=`<h3>${arm?'Treatment':'Control'} · T=${arm}</h3><p>Known synthetic μ${arm} = ${fmt(d['p'+arm])}</p><strong>${outcome}</strong>`;
  }
  $('assignment-text').textContent=(stage>=2?`Fixed seeded assignment: T=${d.t}. `:"")+steps[stage]+". Known probabilities are not observed binary outcomes.";
  window.labState={module:1,stage,reveal,assigned:d.t,observed:stage>=3?d['y'+d.t]:null};
}
async function segments() {
  const strength=+$('segment-strength').value, type=$('segment-ranking').value;
  $('segment-strength-value').textContent=fmt(strength,1);
  const d=DATA.segments, p1=d.p0.map((p,i)=>p+strength*(d.p1[i]-p));
  const tau=p1.map((p,i)=>p-d.p0[i]);
  const estimates=DATA.sample_sizes['5000'][0].map(x=>x.estimate);
  const ranking={p0:d.p0,p1,tau,estimated:estimates}[type];
  const order=[0,1,2,3].sort((a,b)=>ranking[b]-ranking[a]||a-b);
  const names=order.map(i=>d.names[i]);
  await Promise.all([
    plot('segment-bars',[
      {type:"bar",name:"Control probability",x:names,y:order.map(i=>d.p0[i]),marker:{color:C.control}},
      {type:"bar",name:"Treatment probability",x:names,y:order.map(i=>p1[i]),marker:{color:C.positive}}
    ],"Response probabilities",{barmode:"group",yaxis:{title:{text:"Probability"},range:[0,1]},margin:{t:50,l:50,r:15,b:100}}),
    plot('segment-effects',[
      {type:"bar",name:"Known hypothetical CATE",x:names,y:order.map(i=>tau[i]),marker:{color:order.map(i=>tau[i]>=0?C.positive:C.negative)}},
      {type:"scatter",mode:"markers",name:"Fixed-trial estimated CATE (strength 1)",x:names,y:order.map(i=>estimates[i]),marker:{color:C.gold,size:10,symbol:"diamond"}}
    ],"Incremental effect, including harm",{yaxis:{title:{text:"Probability-point difference"},range:[-.3,.7]},margin:{t:50,l:60,r:15,b:100}})
  ]);
  $('segment-table').innerHTML=`<h3>Current ranking · highest first</h3><table><tr><th>Segment</th><th>μ₀</th><th>μ₁</th><th>Known CATE</th><th>Estimated CATE, fixed trial</th></tr>${order.map(i=>`<tr><td>${d.names[i]}</td><td>${fmt(d.p0[i])}</td><td>${fmt(p1[i])}</td><td>${signed(tau[i])}</td><td>${signed(estimates[i])}</td></tr>`).join('')}</table>`;
  window.labState={module:2,strength,p0:d.p0,p1,tau,order};
}
let cameraLock=false, cameraInstalled=false;
async function geometry() {
  const strength=+$('strength').value, s=surfaces(strength);
  $('strength-value').textContent=fmt(strength,1);
  const surface=(z,effect)=>({type:"surface",x:DATA.axis,y:DATA.axis,z,
    colorscale:effect?EFFECT:"Blues",cmin:effect?-1:0,cmax:1,
    colorbar:{thickness:10,len:.6,title:{text:effect?"CATE":"P(Y=1)"}},
    hovertemplate:"X₁ %{x:.2f}<br>X₂ %{y:.2f}<br>Value %{z:.4f}<extra></extra>"});
  const layout=effect=>({margin:{t:50,l:0,r:0,b:0},scene:{
    xaxis:{title:{text:"Engagement"},range:[-2.5,2.5]},yaxis:{title:{text:"Price sensitivity"},range:[-2.5,2.5]},
    zaxis:{title:{text:effect?"CATE (probability points)":"Outcome probability"},range:effect?[-1,1]:[0,1]},
    camera:{eye:{x:1.5,y:1.5,z:1.1}},aspectmode:"cube"}});
  await Promise.all([
    plot('surface0',[surface(s.p0,false)],"Control μ₀(X)",layout(false)),
    plot('surface1',[surface(s.p1,false)],"Treatment μ₁(X)",layout(false)),
    plot('surface-tau',[surface(s.tau,true)],"CATE = μ₁ − μ₀",layout(true)),
    heat('true-heatmap',s.tau,"The same CATE from above")
  ]);
  if(!cameraInstalled){
    const ids=['surface0','surface1','surface-tau'];
    for(const id of ids)$(id).on('plotly_relayout',event=>{
      if(cameraLock||!event['scene.camera'])return;
      cameraLock=true;
      Promise.all(ids.filter(other=>other!==id).map(other=>Plotly.relayout($(other),{'scene.camera':event['scene.camera']})))
        .finally(()=>{cameraLock=false;});
    });
    cameraInstalled=true;
  }
  window.labState={module:3,strength,surfaces:s};
}
async function models() {
  const error=$('model-view').value==='errors',g=DATA.grid;
  const minus=(a,b)=>a.map((row,i)=>row.map((v,j)=>v-b[i][j]));
  $('model-metrics').innerHTML=stat(fmt(DATA.metrics.s_rmse,4),'S-Learner test CATE RMSE')+stat(fmt(DATA.metrics.t_rmse,4),'T-Learner test CATE RMSE')+stat(DATA.metrics.n_train,'Shared randomized training rows')+stat(DATA.metrics.n_test,'Independent test rows');
  await Promise.all([
    heat('model-truth',g.tau,"Known synthetic CATE"),
    heat('model-s',error?minus(g.s,g.tau):g.s,error?"S estimation error: estimated − true":"S-Learner estimated CATE"),
    heat('model-t',error?minus(g.t,g.tau):g.t,error?"T estimation error: estimated − true":"T-Learner estimated CATE")
  ]);
  window.labState={module:4,view:error?'errors':'effects',s_rmse:DATA.metrics.s_rmse,t_rmse:DATA.metrics.t_rmse};
}
async function samples() {
  const n=$('sample-n').value, segment=+$('sample-segment').value;
  const rows=DATA.sample_sizes[n].map(row=>row[segment]), truth=DATA.segments.p1[segment]-DATA.segments.p0[segment];
  const coverage=mean(rows.map(r=>+(r.low<=truth&&truth<=r.high)));
  const widths=rows.map(r=>r.high-r.low);
  $('sample-metrics').innerHTML=stat(rows.length,'Independent replications')+stat(fmt(mean(widths),3),'Mean interval width, probability points')+
    stat(fmt(100*coverage,1)+'%','Empirical pointwise coverage')+stat(`${Math.min(...rows.map(r=>Math.min(r.n0,r.n1)))}–${Math.max(...rows.map(r=>Math.max(r.n0,r.n1)))}`,'Observed N per subgroup arm');
  const y=rows.map((_,i)=>i+1);
  await Promise.all([
    plot('sample-intervals',[{type:"scatter",mode:"markers",x:rows.map(r=>r.estimate),y,
      marker:{color:rows.map(r=>r.low<=truth&&truth<=r.high?C.positive:C.negative),size:7},
      error_x:{type:"data",symmetric:false,array:rows.map(r=>r.high-r.estimate),arrayminus:rows.map(r=>r.estimate-r.low),color:C.control},
      customdata:rows.map(r=>[r.n0,r.n1]),hovertemplate:"Effect %{x:.3f}<br>Replication %{y}<br>N₀ %{customdata[0]}, N₁ %{customdata[1]}<extra></extra>"}],
      `${DATA.segments.names[segment]} · total N=${n}`,
      {xaxis:{title:{text:"Subgroup effect (probability points)"},range:[-.65,.9]},yaxis:{title:{text:"Independent replication"}},
       shapes:[{type:"line",xref:"x",yref:"paper",x0:truth,x1:truth,y0:0,y1:1,line:{dash:"dash",color:C.gold}}]}),
    plot('sample-spread',Object.entries(DATA.sample_sizes).map(([size,reps])=>({type:"box",name:`N=${size}`,y:reps.map(row=>row[segment].estimate),boxpoints:"all",jitter:.35,pointpos:0,marker:{size:4}})),
      "Repeated estimates across sample sizes",{yaxis:{title:{text:"Subgroup effect"},range:[-.65,.9]},showlegend:false,
      shapes:[{type:"line",xref:"paper",yref:"y",x0:0,x1:1,y0:truth,y1:truth,line:{dash:"dash",color:C.gold}}]})
  ]);
  window.labState={module:5,n:+n,segment,coverage,mean_width:mean(widths)};
}
async function gains() {
  const requested=+$('target').value, n=DATA.test.z.length;
  const count=Math.floor(n*requested/100), q=count/n;
  const uplift=gainAt(DATA.curves.uplift,q),response=gainAt(DATA.curves.response,q),ate=DATA.metrics.ipw_ate,random=q*ate;
  $('target-value').textContent=requested+'%';
  $('gain-metrics').innerHTML=stat(`${count} / ${n}`,'Eligible people selected')+stat(signed(uplift*n,2),'Estimated incremental outcomes, selected set')+
    stat(signed(uplift*1000,2),'Incremental outcomes per 1,000 eligible')+stat(signed((uplift-random)*1000,2),'Per 1,000 versus random')+stat(signed((uplift-response)*1000,2),'Per 1,000 versus response ranking');
  await plot('gain-chart',[
    {type:"scatter",mode:"lines",name:"Estimated uplift ranking",x:DATA.curves.uplift.q,y:DATA.curves.uplift.gain,line:{color:C.positive}},
    {type:"scatter",mode:"lines",name:"Control-response ranking",x:DATA.curves.response.q,y:DATA.curves.response.gain,line:{color:C.control}},
    {type:"scatter",mode:"lines",name:"Expected random targeting",x:[0,1],y:[0,ate],line:{color:C.gold,dash:"dash"}},
    {type:"scatter",mode:"markers",name:`Selected ${requested}%`,x:[q,q,q],y:[uplift,response,random],marker:{color:[C.positive,C.control,C.gold],size:10}}
  ],"Cumulative incremental gain on the randomized holdout",{
    xaxis:{title:{text:"Fraction of eligible population targeted"},range:[0,1],tickformat:".0%"},
    yaxis:{title:{text:"Incremental outcomes / eligible person"}},
    shapes:[{type:"line",xref:"x",yref:"paper",x0:q,x1:q,y0:0,y1:1,line:{color:"#b8c8d1",dash:"dot"}}]});
  $('gain-areas').innerHTML=`<h3>Areas, with explicit normalization</h3><table><tr><th>Ranking</th><th>AUUC = ∫Ĝ(q)dq</th><th>Qini area = AUUC − Ĝ(1)/2</th></tr>${['uplift','response'].map(name=>`<tr><td>${name}</td><td>${fmt(DATA.curves[name].auuc,6)}</td><td>${fmt(DATA.curves[name].qini_area,6)}</td></tr>`).join('')}</table><p class="caption">Complete tie blocks define vertices; inside a tie, interpolation averages over random tie order. This differs from the deterministic cohort tie rule in Module 7. No uncertainty band is claimed for this single fitted ranking.</p>`;
  window.labState={module:6,count,q,uplift,response,random,ate};
}
async function policies() {
  const value=+$('value').value,cost=+$('cost').value,capacity=+$('capacity').value;
  $('value-value').textContent=fmt(value,0);$('cost-value').textContent=fmt(cost,0);$('capacity-value').textContent=capacity+'%';
  const selected=selectPolicy(value,cost,capacity), m=policyMetrics(selected,value,cost), n=DATA.test.score.length;
  const all=policyMetrics(DATA.test.score.map((_,i)=>i),value,cost), none=policyMetrics([],value,cost);
  const q=Math.floor(n*capacity/100)/n;
  const random={selected:q*n,predicted_conversions:q*sum(DATA.test.score),true_conversions:q*sum(DATA.test.tau),ipw_conversions:q*sum(DATA.test.z),contact_cost:q*n*cost,
    predicted_net:q*all.predicted_net,true_net:q*all.true_net,ipw_net:q*all.ipw_net};
  const cutoff=selected.length?Math.min(...selected.map(i=>DATA.test.score[i])):Infinity;
  const region=DATA.grid.t.map(row=>row.map(score=>+(score>=cutoff&&value*score>cost)));
  $('policy-metrics').innerHTML=stat(`${m.selected} / ${n}`,'Customers selected')+stat(fmt(m.predicted_conversions,2),'Predicted incremental conversions')+
    stat(fmt(m.ipw_conversions,2),'IPW-estimated incremental conversions')+stat(fmt(m.true_conversions,2),'True expected conversions (synthetic only)')+
    stat(fmt(m.contact_cost,2),'Total intervention cost')+stat(signed(m.ipw_net,2),'IPW total incremental net value');
  await Promise.all([
    heat('policy-region',region,"Estimated positive-net treatment regions",{zmin:0,zmax:1,colorscale:[[0,"#edf1f4"],[1,C.positive]],showscale:false}),
    plot('policy-comparison',[
      {type:"bar",name:"Predicted",x:["Policy","All","None","Random"],y:[m,all,none,random].map(x=>x.predicted_net),marker:{color:C.control}},
      {type:"bar",name:"IPW estimated",x:["Policy","All","None","Random"],y:[m,all,none,random].map(x=>x.ipw_net),marker:{color:C.positive}},
      {type:"bar",name:"True synthetic",x:["Policy","All","None","Random"],y:[m,all,none,random].map(x=>x.true_net),marker:{color:C.gold}}
    ],"Policy net value: estimation versus synthetic truth",{barmode:"group",yaxis:{title:{text:"Total incremental currency units"}}})
  ]);
  $('policy-table').innerHTML=`<h3>Incremental value relative to treating nobody</h3><table><tr><th>Policy</th><th>Contacts</th><th>Predicted net / eligible</th><th>IPW net / eligible</th><th>True net / eligible</th></tr>${[m,all,none,random].map((r,i)=>`<tr><td>${['Cost-aware policy','Treat all (capacity ignored)','Treat none','Expected random at capacity'][i]}</td><td>${fmt(r.selected,0)}</td><td>${signed(r.predicted_net/n)}</td><td>${signed(r.ipw_net/n)}</td><td>${signed(r.true_net/n)}</td></tr>`).join('')}</table><p class="caption">Random is an expectation over uniformly selected subsets. Bars show totals; table shows values per eligible person. Interactive exploration is not final policy validation.</p>`;
  window.labState={module:7,value,cost,capacity,...m};
}
async function overlap() {
  const regime=$('regime').value,d=DATA.overlap[regime],r=d.records;
  const armValues=(values,arm)=>values.filter((_,i)=>d.t[i]===arm);
  const traces=(values,title)=>[0,1].map(arm=>({type:"histogram",histnorm:"probability density",opacity:.6,name:arm?'Treatment':'Control',x:armValues(values,arm),nbinsx:25,marker:{color:arm?C.positive:C.control}}));
  const first=r[0];
  $('overlap-metrics').innerHTML=stat(fmt(first.naive,4),'First dataset: naive difference')+stat(fmt(first.ipw,4),'First dataset: known-propensity IPW')+
    stat(fmt(first.truth,4),'First dataset: known sample mean CATE')+stat(`${fmt(first.ess[0],1)} / ${fmt(first.ess[1],1)}`,'Control / treatment ESS')+stat(fmt(first.max_weight,1),'Largest realized assignment weight');
  const scale=regime==='observational'?.8:3.5;
  const assignment=DATA.axis.map(b=>DATA.axis.map(a=>regime==='randomized'?.5:Math.min(.999,Math.max(.001,1/(1+Math.exp(-scale*(a-1.2*b)))))));
  await Promise.all([
    plot('feature-hist',traces(d.x.map(x=>x[0])),"Engagement distribution by assigned arm",{barmode:"overlay",xaxis:{title:{text:"Engagement"}},yaxis:{title:{text:"Density"}}}),
    plot('propensity-hist',traces(d.e),"Known treatment-propensity distributions",{barmode:"overlay",xaxis:{title:{text:"e(X) = P(T=1|X)"},range:[0,1]},yaxis:{title:{text:"Density"}}}),
    heat('assignment-map',assignment,"Assignment probability over feature space",{zmin:0,zmax:1,colorscale:"Blues",colorbar:{title:{text:"P(T=1)"},thickness:12}}),
    plot('overlap-errors',[
      {type:"box",name:"Naive difference",y:r.map(x=>x.naive-x.truth),boxpoints:"all",jitter:.3,marker:{color:C.negative,size:5}},
      {type:"box",name:"Known-propensity IPW",y:r.map(x=>x.ipw-x.truth),boxpoints:"all",jitter:.3,marker:{color:C.positive,size:5}}
    ],`Estimation error across ${r.length} independent datasets`,{yaxis:{title:{text:"Estimate − known sample mean CATE"}},showlegend:false}),
    plot('weight-chart',[{type:"scatter",mode:"markers",x:d.e,y:d.weights,
      marker:{color:d.t.map(t=>t?C.positive:C.control),size:4,opacity:.5},
      hovertemplate:"e(X) %{x:.4f}<br>Assignment weight %{y:.2f}<extra></extra>"}],
      "Rare assignments carry large inverse-probability weights",{xaxis:{title:{text:"Known treatment propensity"}},yaxis:{title:{text:"Realized weight (log scale)"},type:"log"}})
  ]);
  window.labState={module:8,regime,naive:first.naive,ipw:first.ipw,ess:first.ess,max_weight:first.max_weight};
}
const pipeline=[
  ["Pre-treatment features","Freeze engagement, price sensitivity, eligibility and case attributes at the decision timestamp. A difficulty score predicts failure, not response to AI assistance."],
  ["Experiment logs","Record randomization unit, assigned action, actual assignment probability, policy version, received exposure and mature outcome. Assignment and receipt can differ; define the ITT estimand."],
  ["Effect estimation","Fit response models only on development data under a defensible causal design. LLM text or a confidence score is not a causal label. Check arm support and model error."],
  ["Independent evaluation","Evaluate a frozen ranking or policy on independent randomized outcomes. Use correct probabilities, costs, uncertainty and outcome windows. Reserve fresh data after policy selection."],
  ["Cost-aware policy","Combine estimated incremental success with the value of a resolved case, assistance cost, eligibility, capacity and guardrails. A hard case can have little treatment benefit."],
  ["Intervention","Execute the assigned workflow. Preserve consent, escalation, capacity and treatment versions. Log failures and exposure instead of silently replacing assignment with receipt."],
  ["Outcome measurement","Wait for comparable, mature outcome windows. Measure success, latency, safety and actual intervention costs. This lab measures no production guardrail."],
  ["Monitoring & retraining","Maintain randomized comparison traffic, diagnose drift and overlap, version policies and re-evaluate before changing the decision rule. Account for interference and feedback."]
];
let pipelineStep=0;
async function production() {
  $('pipeline').innerHTML=pipeline.map((step,i)=>`<button data-step="${i}" class="${pipelineStep===i?'active':''}">${i+1} · ${step[0]} ${i<pipeline.length-1?'→':''}</button>`).join('');
  for(const button of $('pipeline').querySelectorAll('button'))button.onclick=()=>{pipelineStep=+button.dataset.step;update();};
  $('pipeline-detail').innerHTML=`<h3>${pipeline[pipelineStep][0]}</h3><p>${pipeline[pipelineStep][1]}</p>`;
  const traffic=+$('traffic').value, logging=$('logging').checked;
  $('traffic-value').textContent=traffic+'%';
  $('monitoring').textContent=traffic===0?"No randomized monitoring traffic: a deterministic policy can eliminate comparisons in excluded regions. Logged probabilities alone cannot repair absent support.":
    !logging?"Randomized traffic is present, but assignment probabilities are not logged. Valid offline weighting cannot be reconstructed from a model score alone.":
    `Illustrative monitoring mixture: ${traffic}% of eligible traffic is randomized 50/50; the rest follows the policy. Log e_total(X) = (${traffic}/100)×0.5 + (1−${traffic}/100)×π(X). This provides design support for both actions for eligible traffic, assuming execution and outcome collection follow the design.`;
  window.labState={module:9,step:pipelineStep,traffic,logging};
}

const renderers=[potential,segments,geometry,models,samples,gains,policies,overlap,production];
async function update() {
  const current=++revision;
  document.body.dataset.ready='false';
  try {
    await renderers[active-1]();
    if(current===revision)document.body.dataset.ready='true';
  } catch(error) {
    $('error').style.display='block';$('error').textContent=error.stack||String(error);
    console.error(error);document.body.dataset.ready='error';
  }
}
async function navigate(index) {
  for(const key of Object.keys(timers)){clearInterval(timers[key]);delete timers[key];}
  active=index;
  document.querySelectorAll('section.module').forEach((section,i)=>section.classList.toggle('active',i===index-1));
  document.querySelectorAll('nav button').forEach((button,i)=>{button.classList.toggle('active',i===index-1);button.setAttribute('aria-current',i===index-1?'step':'false');});
  history.replaceState(null,'',`#module${index}`);
  await update();
}
$('navigation').innerHTML=labels.map((name,i)=>`<button data-module="${i+1}"><span class="num">${String(i+1).padStart(2,'0')}</span>${name}</button>`).join('');
for(const button of $('navigation').querySelectorAll('button'))button.onclick=()=>navigate(+button.dataset.module);
$('sample-segment').innerHTML=DATA.segments.names.map((name,i)=>`<option value="${i}" ${i===1?'selected':''}>${name}</option>`).join('');
for(const element of document.querySelectorAll('input,select'))element.addEventListener(element.type==='range'?'input':'change',update);
$('play-potential').onclick=()=>toggleTimer('potential',()=>{$('stage').value=(+$('stage').value+1)%5;},1000);
$('play-segments').onclick=()=>toggleTimer('segments',()=>{$('segment-strength').value=((Math.round(+$('segment-strength').value*10)+1)%21)/10;},400);
$('play-samples').onclick=()=>toggleTimer('samples',()=>{const el=$('sample-n');el.selectedIndex=(el.selectedIndex+1)%4;},1400);
$('run-info').textContent=`Seed ${DATA.config.seed} · ${DATA.config.n_train} train / ${DATA.config.n_test} test · ${DATA.config.repetitions} repetitions per sampling design`;
window.UpliftLab={surfaces,selectPolicy,policyMetrics,gainAt,navigate,update};
const initial=Number(location.hash.replace('#module',''));
navigate(initial>=1&&initial<=9?initial:1);

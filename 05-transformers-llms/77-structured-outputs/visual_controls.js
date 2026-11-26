/* Offline controls mirror the tested Python mathematics over bounded inputs. */
const C = {allowed:'#0072B2', rejected:'#C44E52', pending:'#D89021', accepted:'#00866A', evidence:'#7859A6'};
const cfg = {responsive:true, displaylogo:false, showLink:false};
const el = id => document.getElementById(id);
const value = id => Number(el(id).value);
const layout = (title,x,y) => ({title:{text:title,font:{size:18}},margin:{l:70,r:30,t:65,b:70},
  font:{family:'system-ui',color:'#243746'},paper_bgcolor:'white',plot_bgcolor:'white',
  xaxis:{title:{text:x}},yaxis:{title:{text:y}},legend:{orientation:'h',y:-.22},uirevision:'keep'});
const chart = async (id,data,settings) => Plotly.react(id,data,settings,cfg);
const softmax = scores => {const m=Math.max(...scores), w=scores.map(x=>Math.exp(x-m)), s=w.reduce((a,b)=>a+b,0);return w.map(x=>x/s);};
function retry(p,n,c=.02,t=.8){
  const success=p===1?1:-Math.expm1(n*Math.log1p(-p));
  const attempts=p===0?n:success/p;
  return {success,attempts,cost:c*attempts,latency:t*attempts,exhaustion:(1-p)**n,
    marginal:p*(1-p)**(n-1),perAccepted:success?c*attempts/success:null};
}
function minimum(p,target){
  if(target===0)return 1;
  if(p===0||(target===1&&p<1))return null;
  if(p===1)return 1;
  let n=Math.max(1,Math.ceil(Math.log1p(-target)/Math.log1p(-p)));
  while(n>1&&retry(p,n-1).success>=target)n--;
  return n;
}
function rng(seed){return ()=>{seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return ((t^t>>>14)>>>0)/4294967296;};}
window.labState={};
async function update(drawSamples=false){
  el('detail').textContent='';
  const demo=LAB.demo;
  el('plot2').style.display='none';
  if(demo==='masking'){
    const tokens=['100','200','"1500"','null','true'];
    const scores=tokens.map((_,i)=>value('logit'+i)), temp=value('temperature');
    const allowed=tokens.map((_,i)=>el('allow'+i).checked);
    const original=softmax(scores.map(x=>x/temp)), permitted=scores.filter((_,i)=>allowed[i]);
    let q=null;
    if(permitted.length){const probs=softmax(permitted.map(x=>x/temp));let k=0;q=scores.map((_,i)=>allowed[i]?probs[k++]:0);}
    const colors=tokens.map((_,i)=>allowed[i]?C.allowed:C.rejected);
    const traces=[{type:'bar',name:'Original p',x:tokens,y:original,marker:{color:colors,opacity:.45}},
      {type:'bar',name:'Constrained q',x:tokens,y:q||[],marker:{color:colors},text:q?q.map(x=>x.toFixed(4)):[],textposition:'outside'}];
    await chart('plot',traces,{...layout('Original and constrained probabilities','Toy token','Probability'),barmode:'group',xaxis:{type:'category',title:{text:'Toy token'}},yaxis:{range:[0,1.1],title:{text:'Probability'}}});
    let samples=null;
    if(q&&drawSamples){const random=rng(value('seed'));samples=Array(5).fill(0);for(let j=0;j<1000;j++){let u=random(),cum=0;for(let k=0;k<5;k++){cum+=q[k];if(u<cum||k===4){samples[k]++;break;}}}}
    const mass=original.reduce((sum,p,i)=>sum+(allowed[i]?p:0),0);
    el('status').textContent=q?`Admissible original mass: ${mass.toFixed(6)} | Sum(q): ${q.reduce((a,b)=>a+b,0).toFixed(6)}\nForbidden probabilities: exactly zero.`:'No admissible token: generation blocked. No constrained distribution exists.';
    el('detail').textContent=tokens.map((t,i)=>`${t.padEnd(8)} logit=${scores[i].toFixed(2)} masked=${allowed[i]?(scores[i]/temp).toFixed(2):'-Infinity'} p=${original[i].toFixed(6)} q=${q?q[i].toFixed(6):'undefined'}`).join('\n')+(samples?'\nSeeded 1000-sample counts: '+JSON.stringify(samples):'');
    window.labState={original,q,allowed,mass,samples,blocked:!q};
  } else if(demo==='space'){
    const gate=value('gate'), rows=LAB.validation.rows;
    const groups=[['Invalid JSON',r=>r.passed[0]===false,C.rejected],
      ['JSON / schema or domain failure',r=>r.passed[0]===true&&(r.passed[1]===false||r.passed[2]===false),C.pending],
      ['Schema compliant / inconsistent',r=>r.passed[3]===false,C.evidence],
      ['Consistent / ungrounded',r=>r.passed[4]===false,'#9C755F'],
      ['Valid / grounded',r=>r.passed[5]===true,C.accepted]];
    const eligible=r=>gate===0||(gate===1&&r.passed[0]===true)||(gate===2&&r.passed[2]===true)||(gate===3&&r.passed[3]===true)||(gate===4&&r.passed[4]===true);
    const traces=groups.map(([name,test,color],i)=>{const subset=rows.filter(r=>test(r)&&eligible(r));return {type:'scatter3d',mode:'markers',name,x:subset.map(r=>i*1.4+r.position_noise[0]),y:subset.map(r=>Math.sin(i)*1.5+r.position_noise[1]),z:subset.map(r=>r.position_noise[2]+i*.4),text:subset.map(r=>r.id+'<br>'+r.reason),hovertemplate:'%{text}<extra></extra>',marker:{size:5,color,symbol:i%2?'diamond':'circle'}};});
    await chart('plot',traces,{...layout('Eligible records after active constraints','',''),scene:{xaxis:{title:{text:'Illustrative coordinate x'}},yaxis:{title:{text:'Illustrative coordinate y'}},zaxis:{title:{text:'Illustrative coordinate z'}}},height:620});
    const count=rows.filter(eligible).length;
    el('status').textContent=`Active constraint level: ${gate}. Eligible: ${count} / ${rows.length}. Coordinates are illustrative only.`;
    window.labState={gate,eligible:count};
  } else if(demo==='semantics'){
    const rows=LAB.validation.rows;
    const categories=[...new Set(rows.map(r=>r.category))];
    const palette=[C.rejected,C.pending,'#CC79A7','#999999','#9C755F','#D55E00',C.evidence,C.allowed,C.accepted];
    const traces=categories.map((name,i)=>{const s=rows.filter(r=>r.category===name);return {type:'scatter',mode:'markers',name,x:s.map(r=>r.schema_score),y:s.map(r=>r.semantic_score),text:s.map(r=>r.category+'<br>'+r.reason+'<br>Input: '+r.raw+'<br>Reference: '+JSON.stringify(r.expected)),hovertemplate:'Compliance=%{x:.3f}<br>Semantic=%{y:.3f}<br>%{text}<extra></extra>',marker:{size:12,color:palette[i],symbol:i%2?'diamond':'circle',opacity:.7}};});
    await chart('plot',traces,{...layout('High structural scores can coexist with incorrect information','Schema check fraction (17 checks)','Semantic check fraction (9 checks)'),xaxis:{range:[-.03,1.05],title:{text:'Schema check fraction'}},yaxis:{range:[-.03,1.05],title:{text:'Semantic check fraction'}}});
    el('status').textContent=`${rows.length} synthetic records. Identical scores overlap; no jitter or probability interpretation.`;
    el('detail').textContent=categories.map(name=>{const r=rows.find(r=>r.category===name);return `${name}: schema=${r.schema_score.toFixed(3)}, semantic=${r.semantic_score.toFixed(3)}\n  ${r.raw}`;}).join('\n');
    window.labState={records:rows.length};
  } else if(demo==='reliability'||demo==='cost'){
    const p=value('p'), rawN=value('n'), target=value('target');
    const c=demo==='cost'?value('cost'):.02,t=demo==='cost'?value('duration'):.8;
    if(!Number.isFinite(p)||p<0||p>1||!Number.isInteger(rawN)||rawN<1||rawN>30||target<0||target>1||!Number.isFinite(target)||!Number.isFinite(c)||!Number.isFinite(t)||c<0||t<0){el('status').textContent='Invalid input: p/target in [0,1], integer n in [1,30], nonnegative cost/duration.';return;}
    const n=rawN, budgets=Array.from({length:n},(_,i)=>i+1), selected=retry(p,n,c,t), needed=minimum(p,target);
    const series=[.2,.525,.8,p].filter((v,i,a)=>a.indexOf(v)===i);
    if(demo==='reliability'){
      const traces=series.map(prob=>({type:'scatter',mode:'lines+markers',name:`p=${prob.toFixed(3)}`,x:budgets,y:budgets.map(k=>retry(prob,k).success)}));
      traces.push({type:'scatter',mode:'lines',name:'Target',x:[1,n],y:[target,target],line:{dash:'dash',color:C.rejected}});
      await chart('plot',traces,{...layout('Diminishing reliability gains from independent attempts','Maximum total attempts (integer)','P(at least one success)'),xaxis:{dtick:1,title:{text:'Maximum total attempts'}},yaxis:{range:[0,1.03],title:{text:'Reliability'}}});
    } else {
      const metrics=budgets.map(k=>retry(p,k,c,t)), budget=value('budget');
      const feasible=budgets.filter((k,i)=>metrics[i].success>=target&&metrics[i].cost<=budget);
      const traces=[{type:'scatter',mode:'lines+markers',name:'Reliability',x:budgets,y:metrics.map(r=>r.success),line:{color:C.allowed}},
        {type:'scatter',mode:'lines+markers',name:'Expected cost ($)',x:budgets,y:metrics.map(r=>r.cost),yaxis:'y2',line:{color:C.pending}},
        {type:'scatter',mode:'markers',name:'Feasible budgets',x:feasible,y:feasible.map(k=>metrics[k-1].success),marker:{color:C.accepted,size:13,symbol:'diamond'}},
        {type:'scatter',mode:'lines+markers',name:'Marginal success gain',x:budgets,y:metrics.map(r=>r.marginal),line:{color:C.evidence,dash:'dot'}}];
      await chart('plot',traces,{...layout('Reliability, cost and marginal gains','Maximum total attempts (integer)','Reliability / marginal gain'),yaxis2:{title:{text:'Expected cost per request ($)'},overlaying:'y',side:'right',rangemode:'tozero'},margin:{l:70,r:80,t:65,b:90}});
      el('plot2').style.display='block';
      await chart('plot2',[{type:'scatter',mode:'lines+markers',name:'Reliability / expected cost',x:metrics.map(r=>r.cost),y:metrics.map(r=>r.success),text:budgets.map(k=>'n='+k),hovertemplate:'%{text}<br>E[cost]=$%{x:.4f}<br>Reliability=%{y:.5f}<extra></extra>'}],layout('Reliability versus expected generation cost','Expected cost per request ($)','Reliability'));
      el('detail').textContent='Feasible integer budgets: '+(feasible.join(', ')||'none')+'\nBudget: $'+budget.toFixed(3)+'\n'+metrics.map((r,i)=>`n=${i+1} success=${r.success.toFixed(6)} E[N]=${r.attempts.toFixed(4)} E[cost]=$${r.cost.toFixed(4)} E[latency]=${r.latency.toFixed(4)}s marginal=${r.marginal.toFixed(6)}`).join('\n');
      window.labState.feasible=feasible;
    }
    el('status').textContent=`Final reliability: ${selected.success.toFixed(6)} | E[N]: ${selected.attempts.toFixed(4)} | Exhaustion: ${selected.exhaustion.toFixed(6)}\nMinimum total attempts for target: ${needed===null?'not finitely achievable':needed}.`+(demo==='cost'?`\nE[cost/request]=$${selected.cost.toFixed(4)} | E[latency]=${selected.latency.toFixed(4)}s | Cost/accepted=${selected.perAccepted===null?'undefined':'$'+selected.perAccepted.toFixed(4)}`:'');
    window.labState={...window.labState,p,n,target,needed,...selected};
  } else if(demo==='surface'||demo==='cost-surface'){
    const p=Array.from({length:41},(_,i)=>i/40), n=Array.from({length:10},(_,i)=>i+1);
    const success=n.map(k=>p.map(prob=>retry(prob,k).success));
    const z=demo==='surface'?success:n.map(k=>p.map(prob=>retry(prob,k).cost));
    const selected=value('threshold'), traces=[{type:'surface',x:p,y:n,z,surfacecolor:success,colorscale:'Viridis',cmin:0,cmax:1,
      colorbar:{title:{text:'Reliability'}},customdata:success,contours:{z:{show:true,usecolormap:true,project:{z:true}}},
      hovertemplate:'p=%{x:.3f}<br>Integer budget=%{y}<br>z=%{z:.6f}<br>Reliability=%{customdata:.6f}<extra></extra>'}];
    for(const threshold of [.9,.95,.99]){
      const valid=n.filter(k=>minimum(1,threshold)<=k), xs=valid.map(k=>-Math.expm1(Math.log1p(-threshold)/k));
      traces.push({type:'scatter3d',mode:'lines+markers',name:`${(threshold*100).toFixed(0)}% reliability contour`,x:xs,y:valid,z:valid.map((k,i)=>demo==='surface'?threshold:retry(xs[i],k).cost),line:{width:threshold===selected?9:3},marker:{size:3}});
    }
    await chart('plot',traces,{...layout(demo==='surface'?'Reliability at computed integer budgets':'Expected generation cost; color gives reliability','',''),height:650,
      scene:{xaxis:{title:{text:'Per-attempt success p'}},yaxis:{title:{text:'Maximum total attempts n'},dtick:1},zaxis:{title:{text:demo==='surface'?'P(success)':'Expected generation cost ($)'}}}});
    el('status').textContent=demo==='surface'?`41 p values x 10 integer budgets. Highlighted contour: ${(100*selected).toFixed(0)}%. Rotate or zoom to inspect.`:'41 p values x 10 integer budgets. Assumed cost=$0.02/call; color encodes reliability.';
    window.labState={p,n,z,success,threshold:selected};
  } else if(demo==='correlation'){
    const metric=el('metric').value, sim=LAB.simulation;
    const scenarios=[...new Set(sim.results.map(r=>r.scenario))], traces=[];
    scenarios.forEach((name,i)=>{const rows=sim.results.filter(r=>r.scenario===name), colors=[C.allowed,C.rejected,C.accepted];
      const predicted=rows.map(r=>metric==='success'?r.analytical_success:metric==='attempts'?r.analytical_attempts:1-r.analytical_success);
      const measured=rows.map(r=>metric==='success'?r.simulated_success:metric==='attempts'?r.simulated_attempts:r.exhaustion);
      traces.push({type:'scatter',mode:'lines',name:name+' prediction',x:rows.map(r=>r.n),y:predicted,line:{color:colors[i]}});
      traces.push({type:'scatter',mode:'markers',name:name+' Monte Carlo',x:rows.map(r=>r.n),y:measured,marker:{color:colors[i],symbol:'diamond'},error_y:{type:'data',array:rows.map(r=>1.96*(metric==='attempts'?r.attempts_se:r.success_se)),visible:true}});
    });
    await chart('plot',traces,layout('Analytical predictions versus actual synthetic simulation','Maximum total attempts (integer)',metric));
    el('status').textContent=`Simulated requests per scenario: ${sim.samples.toLocaleString()} | seed=${sim.seed} | permanent failure mixture=25%.`;
    el('detail').textContent='Configured feedback conditional p_i: '+sim.feedback_conditional_p.map(p=>p.toFixed(3)).join(', ');
    window.labState={metric,samples:sim.samples};
  }
}
document.querySelectorAll('input,select').forEach(node=>node.addEventListener('change',()=>{
  if(LAB.demo==='masking'){
    if(node.id==='schema'&&node.value!=='custom'){
      for(let i=0;i<5;i++)el('allow'+i).checked=node.value==='all'||i<2||(node.value==='nullable'&&i===3);
    }else if(node.id.startsWith('allow'))el('schema').value='custom';
  }
  update().catch(e=>{el('status').textContent='Calculation failed: '+e.message;throw e;});
}));
if(el('sample'))el('sample').addEventListener('click',()=>update(true));
update().then(()=>document.body.dataset.ready='true');

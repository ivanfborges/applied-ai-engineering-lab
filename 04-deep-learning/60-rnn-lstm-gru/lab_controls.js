/* Embedded in each self-contained export; no fetch, server or CDN. */
(() => {
  'use strict';
  const data = window.LAB_DATA;
  const $ = id => document.getElementById(id);
  const number = id => Number($(id).value);
  const sig = x => 1 / (1 + Math.exp(-x));
  const blue = '#2463a6', orange = '#ca6028', green = '#21846d';
  const round = x => Number(x).toFixed(4);
  let drawing=[], renderVersion=0;
  const config = {responsive:true, displaylogo:false, toImageButtonOptions:{format:'png'}};
  const base = {paper_bgcolor:'#ffffff', plot_bgcolor:'#f5f8fc', font:{family:'Arial',color:'#20354e'},
    margin:{l:70,r:30,t:65,b:65}, height:400, legend:{orientation:'h',y:-0.22}};
  const line = (x,y,name,color) => ({x,y,name,type:'scatter',mode:'lines+markers',line:{color},
    hovertemplate:'position=%{x}<br>value=%{y:.6g}<extra>'+name+'</extra>'});
  function plot(id,traces,layout) {
    $(id).hidden = false;
    let title=layout.title?.text||'';
    if(title.length>62){const split=title.lastIndexOf(' ',62);title=title.slice(0,split)+'<br>'+title.slice(split+1);}
    const result=Plotly.react(id,traces,{...base,...layout,title:{text:title,font:{size:14},x:.03,xanchor:'left'}},config);
    drawing.push(result);
    return result;
  }
  function flow(nodes,edges) {
    const width=1000, height=250;
    let content='<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#667d96"/></marker></defs>';
    for (const [a,b,label] of edges) {
      const p=nodes[a],q=nodes[b];
      content+=`<path d="M${p[0]+65},${p[1]} L${q[0]-65},${q[1]}" stroke="#667d96" stroke-width="2" fill="none" marker-end="url(#arrow)"/>`;
      if(label) content+=`<text x="${(p[0]+q[0])/2}" y="${(p[1]+q[1])/2-10}" text-anchor="middle" font-size="13">${label}</text>`;
    }
    for(const [x,y,label,color] of nodes) content+=`<rect x="${x-64}" y="${y-22}" width="128" height="44" rx="9" fill="${color||'#eaf1fa'}" stroke="#95aac0"/><text x="${x}" y="${y+5}" text-anchor="middle" font-size="14">${label}</text>`;
    $('diagram').innerHTML=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Computed state flow">${content}</svg>`;
  }
  function seriesRNN() {
    const values=$('sequence').value.split(',').map(v=>Number(v.trim()));
    if(values.length<1 || values.length>20 || !values.every(Number.isFinite)) throw Error('Enter 1 to 20 comma-separated finite numbers.');
    const strength=number('strength'), states=[[0,0]];
    for(const x of values) {
      const h=states.at(-1);
      states.push([Math.tanh(.8*x+strength*(.8*h[0]+.25*h[1])),Math.tanh(-.5*x+strength*(-.1*h[0]+.7*h[1]))]);
    }
    $('t').max=values.length;
    const t=Math.min(number('t'),values.length); $('t').value=t;
    const time=Array.from({length:t+1},(_,i)=>i);
    plot('plot',[line(time,states.slice(0,t+1).map(h=>h[0]),'hidden dimension 1',blue),line(time,states.slice(0,t+1).map(h=>h[1]),'hidden dimension 2',orange)],
      {title:{text:'Actual tanh updates · handcrafted weights'},xaxis:{title:{text:'timestep (0 = initial state)'},range:[0,values.length]},yaxis:{title:{text:'hidden state'},range:[-1.05,1.05]}});
    plot('extra',[{x:values.map((_,i)=>i+1),y:values,type:'bar',marker:{color:values.map((_,i)=>i===t-1?orange:blue)},name:'input x_t'}],
      {title:{text:'Input sequence · edit values above'},xaxis:{title:{text:'timestep'}},yaxis:{title:{text:'input scalar'}}});
    const h=states[t],prev=states[Math.max(0,t-1)];
    flow([[110,80,`x_t = ${t?round(values[t-1]):'not read'}`],[360,150,`h_prev[1]=${round(prev[0])}`],
      [640,80,'tanh(Wx x + Wh h)'],[875,150,`h_t[1]=${round(h[0])}`]],[[0,2,'same Wx'],[1,2,'same Wh'],[2,3,'update']]);
    $('readout').textContent=`t=${t}; h_t=[${h.map(round).join(', ')}]. Wx=[0.8,-0.5]; Wh=${strength.toFixed(2)} × [[0.8,0.25],[-0.1,0.7]]; b=0. These are fixed, untrained weights.`;
  }
  function gates() {
    const isLSTM=data.kind==='lstm', manual=$('mode').value==='manual';
    const t=number('t'), row=data.rows[t-1];
    let h, c, f, i, g, o, z, r, retained, written, candidate;
    if(isLSTM) {
      c=manual?number('c'):row.c_prev; f=manual?number('f'):row.f;
      i=manual?number('i'):row.i; g=manual?number('g'):row.g; o=manual?number('o'):row.o;
      retained=f*c; written=i*g; h=o*Math.tanh(retained+written);
      plot('plot',[{x:['f · retain','i · write','o · expose'],y:[f,i,o],type:'bar',marker:{color:[blue,orange,green]}}],
        {title:{text:manual?'Manual gate interventions':'Gates computed from fixed NumPy weights'},yaxis:{title:{text:'gate value'},range:[0,1]}});
      plot('extra',[{x:['f × c_prev','i × g','c_t','h_t'],y:[retained,written,retained+written,h],type:'bar',marker:{color:[blue,orange,green,'#7555aa']}}],
        {title:{text:'Retain + write, then expose'},yaxis:{title:{text:'signed state value'}}});
      flow([[95,55,`c_prev ${round(c)}`],[310,55,`× f ${round(f)}`],[535,55,`+ i·g ${round(written)}`],
        [785,55,`c_t ${round(retained+written)}`],[95,180,`x_t ${round(row.x)}`],[310,180,`h_prev ${round(row.h_prev)}`],
        [535,180,`g ${round(g)}`],[785,180,`h_t ${round(h)}`]],[[0,1,'retain'],[1,2,'add'],[2,3,'memory'],[3,7,'o × tanh(c)'],[4,6,'input'],[5,6,'history']]);
      $('readout').textContent=`${manual?'Manual intervention: gates are independently chosen, not learned.':'Computed mode: untrained seed-160 NumPy cell; x and previous states come from its actual recurrence.'} c_t=${round(f)}×${round(c)}+${round(i)}×${round(g)}=${round(retained+written)}; h_t=${round(o)}×tanh(c_t)=${round(h)}. Output gating changes h_t, not this step’s c_t.`;
    } else {
      const previous=manual?number('h'):row.h_prev, x=manual?number('x'):row.x;
      z=manual?number('z'):row.z; r=manual?number('r'):row.r;
      const before=$('convention').value==='before';
      candidate=Math.tanh(.8*x+.1+(before?.9*(r*previous)+.3:r*(.9*previous+.3)));
      retained=z*previous; written=(1-z)*candidate; h=retained+written;
      plot('plot',[{x:['z · retain','r · reset'],y:[z,r],type:'bar',marker:{color:[blue,orange]}}],
        {title:{text:manual?'Manual gate interventions':'Gates from fixed scalar weights'},yaxis:{title:{text:'gate value'},range:[0,1]}});
      plot('extra',[{x:['candidate n','z × h_prev','(1-z) × n','h_t'],y:[candidate,retained,written,h],type:'bar',marker:{color:[green,blue,orange,'#7555aa']}}],
        {title:{text:'One state: retention + candidate update'},yaxis:{title:{text:'signed state value'}}});
      flow([[110,70,`h_prev ${round(previous)}`],[350,70,`z ${round(z)}`],[610,70,`z·h ${round(retained)}`],
        [865,125,`h_t ${round(h)}`],[110,190,`x_t ${round(x)}`],[350,190,`r ${round(r)}`],[610,190,`n ${round(candidate)}`]],
        [[0,1,'retain'],[1,2,'multiply'],[2,3,'add'],[4,5,'reset candidate'],[5,6,'tanh'],[6,3,'× (1-z)']]);
      $('readout').textContent=`${manual?'Manual intervention, not learned behavior.':'Fixed, untrained scalar gates: z=sigmoid(0.6x+0.4h+0.5), r=sigmoid(-0.4x+0.3h).'} ${before?'Reset-before variant: n=tanh(0.8x+0.1+0.9(rh)+0.3).':'PyTorch convention: n=tanh(0.8x+0.1+r(0.9h+0.3)).'} h_t=z h_prev+(1-z)n=${round(h)}. z near 1 retains previous state; no separate c_t.`;
    }
    document.querySelectorAll('[data-manual]').forEach(el=>{el.disabled=!manual;el.closest('label').hidden=!manual;});
    $('t').disabled=manual;
    $('play').disabled=manual;
  }
  function selectedRows() {
    const seed=$('seed').value, length=number('length');
    return data.report.results.filter(row=>(seed==='all'||row.seed===Number(seed))&&row.length===length);
  }
  function gradients() {
    const a=number('a'),k=number('t'),time=Array.from({length:k+1},(_,i)=>i);
    plot('plot',[line(time,time.map(n=>Math.max(1e-18,Math.abs(a)**n)),`|a|^k; a=${a}`,blue)],
      {title:{text:'Analytical scalar product · NOT measured training gradients'},xaxis:{title:{text:'recurrent transitions k'},range:[0,50]},yaxis:{type:'log',title:{text:'|a|^k (floor 1e-18 for display)'},range:[-18,10]}});
    plot('extra',selectedRows().map(row=>line(row.input_gradient_norm.map((_,i)=>row.length-1-i),row.input_gradient_norm.map(v=>Math.max(v,1e-18)),row.cell.toUpperCase(),{rnn:blue,lstm:orange,gru:green}[row.cell])),
      {title:{text:'Actual autograd · validation-selected models, independent probe'},xaxis:{title:{text:'distance from final input (0 = last step)'}},yaxis:{type:'log',title:{text:'mean ||d BCE_i / d x_it||₂ (floor 1e-18)'}}});
    $('readout').textContent='The scalar model has constant derivative a. Real gradients depend on weights, directions, nonlinearities, gates, target and loss saturation. Empirical norms sum neither signed examples nor parameter gradients. Input sensitivity is not causal importance; clipping and these curves do not identify the cause of a failed run.';
  }
  function comparison() {
    const rows=data.report.results, seed=$('seed').value, filtered=rows.filter(r=>seed==='all'||r.seed===Number(seed));
    const lengths=data.report.configuration.lengths;
    const average=arr=>arr.reduce((a,b)=>a+b,0)/arr.length;
    const traces=[];
    for(const cell of ['rnn','lstm','gru']) {
      const groups=lengths.map(t=>filtered.filter(r=>r.cell===cell&&r.length===t).map(r=>r.test.accuracy*100));
      traces.push({...line(lengths,groups.map(average),cell.toUpperCase(),{rnn:blue,lstm:orange,gru:green}[cell]),
        error_y:{type:'data',array:groups.map(v=>v.length>1?Math.sqrt(v.reduce((s,x)=>s+(x-average(v))**2,0)/(v.length-1)):0),visible:groups[0].length>1}});
    }
    const bounds=traces.flatMap(trace=>trace.y.flatMap((mean,index)=>[mean-trace.error_y.array[index],mean+trace.error_y.array[index]]));
    plot('plot',traces,{title:{text:'Held-out accuracy · mean ± SD when multiple seeds'},xaxis:{title:{text:'sequence length'}},yaxis:{title:{text:'test accuracy (%)'},range:[Math.min(0,...bounds)-2,Math.max(103,...bounds)+2]}});
    const selected=selectedRows(), firstSeed=selected[0].seed, curves=selected.filter(r=>r.seed===firstSeed);
    plot('extra',curves.map(r=>line(r.history.map(v=>v.epoch),r.history.map(v=>v.validation_bce),r.cell.toUpperCase(),{rnn:blue,lstm:orange,gru:green}[r.cell])),
      {title:{text:`Validation curves · T=${number('length')} · seed=${firstSeed}`},xaxis:{title:{text:'epoch'}},yaxis:{title:{text:'validation BCE'},type:'log'}});
    plot('third',curves.map(r=>({x:[r.cell],y:[r.parameters],name:`${r.cell} H=${r.hidden}`,type:'bar'})),
      {title:{text:'Actual trainable parameters · includes scalar head'},yaxis:{title:{text:'parameter count'}}});
    plot('fourth',curves.map(r=>({x:[r.cell],y:[r.training_seconds],name:r.cell,type:'bar'})),
      {title:{text:'Fit wall time · one selected seed/length, not inference latency'},yaxis:{title:{text:'seconds including validation'}}});
    plot('fifth',curves.map(r=>line(r.input_gradient_norm.map((_,i)=>r.length-1-i),r.input_gradient_norm.map(v=>Math.max(v,1e-18)),r.cell.toUpperCase(),{rnn:blue,lstm:orange,gru:green}[r.cell])),
      {title:{text:`Actual loss sensitivities · T=${number('length')} · seed=${firstSeed}`},xaxis:{title:{text:'distance from final input'}},yaxis:{type:'log',title:{text:'mean ||d BCE_i / d x_it||₂'}}});
    $('readout').textContent=`Protocol ${data.report.configuration.protocol}; ${data.report.configuration.epochs} epochs; ${data.report.configuration.matched?'approximately matched parameter budgets':'equal hidden sizes, unequal parameter budgets'}. Each checkpoint is selected using validation only. Every predefined model is reported. Aggregate mode averages paired seeds; curves and cost use the first displayed seed. No architecture superiority is presumed. See the separate empirical gradient explorer.`;
  }
  function hidden3d() {
    const row=selectedRows().find(r=>r.cell===$('architecture').value);
    $('t').max=row.length; const t=Math.min(number('t'),row.length); $('t').value=t;
    const traces=[];
    row.projection.forEach((trajectory,index)=>{
      const points=trajectory.slice(0,t+1), label=row.trajectory_labels[index], color=label?orange:blue;
      traces.push({type:'scatter3d',mode:'lines+markers',name:`class ${label}, probe ${row.trajectory_probe_indices[index]}`,
        x:points.map(v=>v[0]),y:points.map(v=>v[1]),z:points.map(v=>v[2]),line:{color,width:5},
        marker:{size:points.map((_,j)=>j===0||j===t?6:3),color:points.map((_,j)=>j===0?'#18283d':color)},
        text:points.map((_,j)=>`t=${j}; class=${label}; final logit=${round(row.trajectory_logits[index])}`),
        hovertemplate:'%{text}<br>PC1=%{x:.4f}<br>PC2=%{y:.4f}<br>PC3=%{z:.4f}<extra></extra>'});
    });
    const flat=row.projection.flat(), ranges=[0,1,2].map(i=>{const vals=flat.map(v=>v[i]),lo=Math.min(...vals),hi=Math.max(...vals),pad=Math.max(.05,(hi-lo)*.08);return [lo-pad,hi+pad];});
    plot('plot',traces,{height:650,title:{text:`${row.cell.toUpperCase()} · trained H=${row.hidden} · PC variance ${row.explained_variance.map(v=>(100*v).toFixed(1)+'%').join(' / ')}`},
      scene:{xaxis:{title:{text:'PC1'},range:ranges[0]},yaxis:{title:{text:'PC2'},range:ranges[1]},zaxis:{title:{text:'PC3'},range:ranges[2]},aspectmode:'cube'},uirevision:row.key});
    $('readout').textContent=`t=${t}/${row.length}; black start markers are h0=0; larger colored endpoints identify the current step. Play reveals chronological direction. PCA fits all four trajectories/timesteps for this checkpoint only, including future plotted states. It is a post-hoc view, not model input or literal 3D state. Axes across architectures are not aligned. Separation does not establish generalization.`;
  }
  function attention() {
    const q=number('t'),weights=data.weights,embedding=data.embeddings;
    plot('plot',[{type:'heatmap',z:weights,x:weights.map((_,i)=>i+1),y:weights.map((_,i)=>i+1),zmin:0,zmax:1,colorscale:'Blues',colorbar:{title:{text:'weight'}},hovertemplate:'query=%{y}, key=%{x}<br>weight=%{z:.4f}<extra></extra>'}],
      {title:{text:'Illustrative QKᵀ / √3 → causal mask → softmax'},xaxis:{title:{text:'key position'}},yaxis:{title:{text:'query position'},autorange:'reversed'},shapes:[{type:'rect',x0:.5,x1:6.5,y0:q+.5,y1:q+1.5,line:{color:orange,width:3}}]});
    const value=[0,1,2].map(d=>weights[q].reduce((sum,w,i)=>sum+w*embedding[i][d],0));
    plot('extra',[{type:'bar',x:['value dimension 1','value dimension 2','value dimension 3'],y:value,marker:{color:green}}],
      {title:{text:`Query ${q+1} · actual weighted sum Σ weight_j V_j`},yaxis:{title:{text:'weighted value coordinate'}}});
    const nodes=Array.from({length:6},(_,i)=>[85+165*i,65,`h${i+1}`,i===q?'#f8dccb':'#eaf1fa']);
    const edges=Array.from({length:5},(_,i)=>[i,i+1,'']);
    for(let i=0;i<=q;i++){ nodes.push([85+165*i,190,`key${i+1}: ${round(weights[q][i])}`]); if(i!==q) edges.push([6+i,6+q,'direct']); }
    flow(nodes,edges);
    $('readout').textContent=`Illustrative, untrained Q=K=V=[sin(t),cos(t),t/6]. Query ${q+1} attends only to keys 1..${q+1}. Future entries have weight exactly zero; allowed rows sum to one. Recurrence transports information through consecutive states; attention takes a weighted combination of accessible values. Standard full attention has O(T²d) score/value work. Autoregressive token generation remains sequential.`;
  }
  let timer=null;
  function render() {
    const version=++renderVersion;
    drawing=[]; window.LAB_READY=false;
    try {
      document.querySelectorAll('input[type=range]').forEach(el=>{const out=$(el.id+'Value');if(out)out.textContent=el.value;});
      ({rnn:seriesRNN,lstm:gates,gru:gates,gradients,comparison,hidden3d,attention}[data.kind])();
      $('error').textContent='';
      Promise.all(drawing).then(()=>new Promise(resolve=>requestAnimationFrame(resolve)))
        .then(()=>Promise.all([...document.querySelectorAll('.js-plotly-plot')].filter(el=>!el.hidden).map(el=>Plotly.Plots.resize(el))))
        .then(()=>{if(version===renderVersion)window.LAB_READY=true;})
        .catch(error=>{$('error').textContent=error.message;});
    } catch(error) { $('error').textContent=error.message; window.LAB_READY=false; }
  }
  document.querySelectorAll('input,select,textarea').forEach(el=>el.addEventListener('input',render));
  if($('play')) $('play').onclick=()=>{
    if(timer){clearInterval(timer);timer=null;$('play').textContent='Play';return;}
    $('play').textContent='Pause';
    timer=setInterval(()=>{const el=$('t');el.value=number('t')>=Number(el.max)?el.min:number('t')+1;render();},600);
  };
  render();
})();

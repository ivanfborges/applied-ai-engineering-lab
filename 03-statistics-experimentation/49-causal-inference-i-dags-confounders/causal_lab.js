/* Offline interactions over NumPy-generated exogenous tapes. No network or random JS draws. */
(() => {
  'use strict';
  const mean = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : null;
  const difference = (v, a, rows) => {
    const groups = [[], []];
    for (const i of rows) groups[a[i]].push(v[i]);
    return groups.every(g => g.length) ? mean(groups[1]) - mean(groups[0]) : null;
  };
  function construct(t, p, n) {
    const d = {};
    for (const k of ['A', 'C', 'U']) d[k] = t[k].slice(0, n);
    for (const k of ['M', 'S', 'Y', 'Y0', 'Y1']) d[k] = [];
    for (let i = 0; i < n; i++) {
      d.M.push(p.alpha * d.A[i] + t.eM[i]);
      d.S.push(d.A[i] + d.U[i] + p.noise * t.eS[i]);
      d.Y0.push(p.beta * t.eM[i] + p.confounding * d.C[i] + 1.2 * d.U[i] + t.eY[i]);
      d.Y1.push(d.Y0[i] + 1.3 + p.alpha * p.beta);
      d.Y.push(1.3 * d.A[i] + p.beta * d.M[i] + p.confounding * d.C[i] + 1.2 * d.U[i] + t.eY[i]);
    }
    return d;
  }
  function regress(d, controls, rows = d.A.map((_, i) => i)) {
    const keys = ['A', ...controls], k = keys.length + 1;
    if (rows.length <= k) return null;
    const columns = [rows.map(() => 1), ...keys.map(key => rows.map(i => d[key][i]))];
    const q = [], r = Array.from({length: k}, () => Array(k).fill(0));
    // Modified Gram-Schmidt QR; rank failures yield an explicit unavailable estimate.
    for (let j = 0; j < k; j++) {
      const v = columns[j].slice();
      for (let h = 0; h < j; h++) {
        r[h][j] = v.reduce((s, x, i) => s + x * q[h][i], 0);
        for (let i = 0; i < v.length; i++) v[i] -= r[h][j] * q[h][i];
      }
      r[j][j] = Math.sqrt(v.reduce((s, x) => s + x*x, 0));
      if (r[j][j] < 1e-9) return null;
      q.push(v.map(x => x / r[j][j]));
    }
    const b = q.map(col => col.reduce((s, x, i) => s + x * d.Y[rows[i]], 0));
    for (let j = k-1; j >= 0; j--) {
      for (let h = j+1; h < k; h++) b[j] -= r[j][h] * b[h];
      b[j] /= r[j][j];
    }
    return b[1];
  }
  function statistics(d) {
    const rows = d.A.map((_, i) => i), strata = [];
    for (const c of [0, 1]) {
      const ids = rows.filter(i => d.C[i] === c);
      strata.push({C: c, n: ids.length, weight: ids.length / rows.length,
        difference: difference(d.Y, d.A, ids)});
    }
    const estimates = {'None': regress(d, []), C: regress(d, ['C']),
      'C + M': regress(d, ['C', 'M']), 'C + S': regress(d, ['C', 'S'])};
    estimates['Standardized C'] = strata.every(s => s.difference !== null)
      ? strata.reduce((s, v) => s + v.weight * v.difference, 0) : null;
    const groupMeans = [0, 1].map(c => {
      const ids = rows.filter(i => d.C[i] === c);
      return {a: mean(ids.map(i => d.A[i])), y: mean(ids.map(i => d.Y[i]))};
    });
    const ra = rows.map(i => d.A[i] - groupMeans[d.C[i]].a);
    const ry = rows.map(i => d.Y[i] - groupMeans[d.C[i]].y);
    const denominator = ra.reduce((s, v) => s + v*v, 0);
    const fwl = denominator > 1e-9 ? ra.reduce((s, v, i) => s + v*ry[i], 0) / denominator : null;
    return {estimates, strata, fwl, ra, ry,
      means: [0, 1].map(a => mean(rows.filter(i => d.A[i] === a).map(i => d.Y[i]))),
      p_c1_by_a: [0, 1].map(a => mean(rows.filter(i => d.A[i] === a).map(i => d.C[i])))};
  }
  function selection(d, center, width, mode) {
    const all = d.A.map((_, i) => i);
    const rows = all.filter(i => mode === 'high' ? d.S[i] >= center : Math.abs(d.S[i]-center) <= width);
    return {indices: rows, n: rows.length, delta_u_full: difference(d.U, d.A, all),
      delta_u_selected: difference(d.U, d.A, rows), full_C: regress(d, ['C']),
      full_CS: regress(d, ['C', 'S']), selected_C: regress(d, ['C'], rows),
      selected_CS: regress(d, ['C', 'S'], rows)};
  }
  globalThis.CausalMath = {construct, regress, statistics, selection};
  if (typeof document === 'undefined') return;

  const L = globalThis.LAB, mod = L.module, $ = id => document.getElementById(id);
  const fmt = x => x === null || !Number.isFinite(x) ? 'Unavailable (overlap/rank)' : x.toFixed(4);
  const palette = ['#2563a6', '#b66a0b', '#7954a1', '#be4350', '#087e64'];
  let plots = [], timer = null, phase = 0;
  function chart(id, traces, layout={}) {
    const base = {paper_bgcolor: '#ffffff', plot_bgcolor: '#f6f8fb', font: {family: 'Arial, sans-serif', size: 14, color: '#192d42'},
      margin: {l: 70, r: 30, t: 55, b: 80}, hovermode: 'closest',
      legend: {orientation: 'h', y: -0.18}, uirevision: mod, ...layout};
    plots.push(Plotly.react(id, traces, base, {responsive: true, displaylogo: false, scrollZoom: true,
      toImageButtonOptions: {format: 'png', filename: 'day49_' + mod}}));
  }
  const line = (value, color, text) => ({type: 'line', xref: 'paper', x0: 0, x1: 1, y0: value, y1: value,
    line: {color, dash: 'dash', width: 2}, label: {text, textposition: 'end'}});
  const ids = d => d.A.map((_, i) => i);
  function sampled(rows, limit=700) {
    const step = Math.max(1, Math.ceil(rows.length / limit));
    return rows.filter((_, j) => j % step === 0);
  }
  const jitter = i => .09 * Math.sin(i * 2.399963229728653);
  function scatter(d, rows, x, y, name, color, extra={}) {
    const r = sampled(rows);
    return {type: 'scatter', mode: 'markers', name,
      x: r.map(i => x === 'A' ? d.A[i] + jitter(i) : d[x][i]), y: r.map(i => d[y][i]),
      customdata: r.map(i => [i, d.A[i], d.C[i], d.S[i]]),
      hovertemplate: 'case %{customdata[0]}<br>A=%{customdata[1]}, C=%{customdata[2]}<br>value=%{y:.3f}<br>S=%{customdata[3]:.3f}<extra>%{fullData.name}</extra>',
      marker: {color, size: 5, opacity: .48}, ...extra};
  }
  function effects(labels, values, total=2, direct=1.3, title='Estimated treatment coefficients') {
    chart('detail', [{type: 'scatter', mode: 'markers', x: labels, y: values,
      marker: {color: palette.slice(0, labels.length), size: 13, symbol: 'diamond'}, name: 'Finite-sample estimate'}],
    {title: {text: title}, yaxis: {title: {text: 'Score difference / treatment coefficient'}},
      shapes: [line(total, '#087e64', 'Known total = '+total.toFixed(2)), line(direct, '#7954a1', 'Direct path = '+direct.toFixed(2))]});
  }
  function graphFigure(g, path, active, controls, mediation=false) {
    const traces = [], annotations = [], positions = g.positions;
    for (const [u, v] of g.edges) {
      const chosen = path && path.some((node, i) => i &&
        ((path[i-1] === u && node === v) || (path[i-1] === v && node === u)));
      const causal = path && path.every((node, i) => !i || g.edges.some(e => e[0]===path[i-1] && e[1]===node));
      const color = chosen ? (active ? (causal ? '#087e64' : '#b66a0b') : '#be4350') : '#acb6c1';
      const [x0, y0] = positions[u], [x1, y1] = positions[v];
      annotations.push({x: .84*x1+.16*x0, y: .84*y1+.16*y0, ax: .84*x0+.16*x1, ay: .84*y0+.16*y1,
        xref: 'x', yref: 'y', axref: 'x', ayref: 'y', showarrow: true, arrowhead: 3,
        arrowsize: 1.2, arrowwidth: chosen ? 4 : 2, arrowcolor: color});
    }
    for (const node of g.nodes) traces.push({type: 'scatter', mode: 'markers+text', name: node+' · '+g.roles[node],
      x: [positions[node][0]], y: [positions[node][1]], text: [node], textposition: 'middle center',
      textfont: {color: 'white', size: 20}, marker: {size: 48, color: g.colors[node],
        symbol: node === 'U' ? 'diamond' : 'circle', line: {color: controls.includes(node) ? '#122334' : 'white', width: controls.includes(node) ? 5 : 2}},
      hovertemplate: node+' · '+g.roles[node]+(controls.includes(node) ? '<br>Conditioned' : '')+'<extra></extra>'});
    chart('chart', traces, {title: {text: mediation ? 'Trace the direct and mediated pathways' : 'Assumed causal graph · inspect the selected path'},
      xaxis: {visible: false, range: [-2.5, 2.5]}, yaxis: {visible: false, range: [-1.9, 2.1]}, annotations,
      margin: {l: 20, r: 20, t: 55, b: 40}});
  }
  function renderDag() {
    const c = L.graph.cases[$('adjust').value], index = +$('path').value;
    const path = c.paths[index];
    graphFigure(L.graph, path.nodes, path.active, c.controls);
    $('detail').style.display = 'none';
    $('table').innerHTML = '<table><thead><tr><th>Path</th><th>Type</th><th>State</th><th>Reason</th></tr></thead><tbody>' + c.paths.map(p =>
      '<tr><td>'+p.label+'</td><td>'+(p.causal ? 'Directed causal' : 'Noncausal')+'</td><td>'+(p.active ? 'Active' : 'Blocked')+'</td><td>'+p.reason+'</td></tr>').join('')+'</tbody></table>';
    const message = $('adjust').value === 'None' ? 'Causal paths remain, but confounding is open.' :
      $('adjust').value === 'C' ? 'C blocks confounding; both total-effect pathways remain active.' :
      $('adjust').value === 'C + M' ? 'M blocks the mediated pathway. This no longer preserves the total effect.' :
      'Conditioning on S opens the collider path through U; adding a control introduces bias.';
    $('metrics').textContent = 'Original backdoor criterion: '+(c.backdoor ? 'PASS' : 'FAIL')+'. '+message;
    return {adjustment: $('adjust').value, path: index, ...c};
  }
  function confounding(d, s) {
    const revealed = $('reveal').checked;
    const traces = [];
    for (const c of (revealed ? [0, 1] : [null])) {
      const rows = ids(d).filter(i => c === null || d.C[i] === c);
      traces.push(scatter(d, rows, 'A', 'Y', c === null ? 'Observed cases' : 'C='+c, c === null ? palette[0] : palette[c]));
    }
    chart('chart', traces, {title: {text: revealed ? 'Reveal complexity composition' : 'Observed treatment–outcome association'},
      xaxis: {title: {text: 'A (binary; jitter only for display)'}, tickvals: [0, 1], range: [-.3, 1.3]},
      yaxis: {title: {text: 'Operational performance Y (score units)'}}});
    const boxes = [];
    for (const c of [0, 1]) for (const a of [0, 1]) {
      const rows = ids(d).filter(i => d.C[i] === c && d.A[i] === a);
      boxes.push({type: 'box', name: 'C='+c+', A='+a, y: rows.map(i => d.Y[i]), boxpoints: false,
        marker: {color: palette[c]}});
    }
    chart('detail', boxes, {title: {text: 'Outcome distributions within complexity strata'}, yaxis: {title: {text: 'Y (score units)'}}});
    $('contrast').style.display='block';
    chart('contrast',[{type:'scatter',mode:'markers',name:'Finite-sample difference',
      x:['Marginal','Within C=0','Within C=1','Standardized','OLS(C)'],
      y:[s.estimates.None,...s.strata.map(v=>v.difference),s.estimates['Standardized C'],s.estimates.C],
      marker:{size:13,color:palette,symbol:'diamond'}}],
    {title:{text:'Compare associations and adjusted estimates'},yaxis:{title:{text:'Score difference'}},shapes:[line(2,'#087e64','Known total = 2.0')]});
    $('table').innerHTML = metricTable(['Comparison', 'Estimated difference', 'Target weight'], [
      ['Marginal association', fmt(s.estimates.None), 'Different C composition by arm'],
      ...s.strata.map(v => ['Within C='+v.C, fmt(v.difference), v.weight.toFixed(4)]),
      ['Target-standardized', fmt(s.estimates['Standardized C']), 'Whole-sample P(C)'],
      ['C-adjusted OLS', fmt(s.estimates.C), 'Constant-effect model'], ['Known synthetic total', '2.0000', 'Population truth']]);
    $('metrics').textContent = 'P(C=1 | A=0)='+fmt(s.p_c1_by_a[0])+'; P(C=1 | A=1)='+fmt(s.p_c1_by_a[1])+'. Population association = 2 + 1.5(0.8−0.2) = 2.9.';
  }
  function metricTable(headers, rows) {
    return '<table><thead><tr>'+headers.map(h => '<th>'+h+'</th>').join('')+'</tr></thead><tbody>'+rows.map(row => '<tr>'+row.map(v => '<td>'+v+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
  }
  function collider(d, p) {
    const selected = selection(d, +$('center').value, +$('width').value, $('mode').value);
    const traces = [scatter(d, ids(d), 'A', 'U', 'Full population', '#a8b2bf', {marker: {color: '#a8b2bf', size: 4, opacity: .22}})];
    for (const a of [0, 1]) traces.push(scatter(d, selected.indices.filter(i => d.A[i] === a), 'A', 'U', 'Selected A='+a, palette[a]));
    chart('chart', traces, {title: {text: 'Selection on S creates A–U dependence'},
      xaxis: {title: {text: 'A (binary; display jitter only)'}, tickvals: [0, 1], range: [-.3, 1.3]},
      yaxis: {title: {text: 'U · latent capability (simulator oracle)'}, range: [-4.5, 4.5]}});
    effects(['Full: C', 'Full: C + S', 'Selected: C', 'Selected: C + S'],
      [selected.full_C, selected.full_CS, selected.selected_C, selected.selected_CS]);
    $('metrics').textContent = 'Selected '+selected.n+'/'+d.A.length+' cases. ΔU = E[U|A=1]−E[U|A=0]: full '+fmt(selected.delta_u_full)+', selected '+fmt(selected.delta_u_selected)+'. '+
      ($('mode').value === 'interval' ? 'Interval: |S−center| ≤ half-width.' : 'High-S rule: S ≥ threshold; half-width is ignored.');
    $('table').innerHTML = '<p>Changing noise in S does not change Y. Selection and continuous S adjustment are different operations. Full C-adjusted OLS targets total ATE; selected estimates generally do not. U is displayed only because this is a simulator.</p>';
    return selected;
  }
  function mediation(d, s, p) {
    const total = 1.3+p.alpha*p.beta;
    const g = {...L.graph, nodes: ['A', 'M', 'Y'], edges: [['A','M'], ['M','Y'], ['A','Y']],
      positions: {A: [-1.7,0], M: [0,1.3], Y: [1.7,0]}};
    graphFigure(g, phase % 2 ? ['A','M','Y'] : ['A','Y'], true, [], true);
    chart('detail', [{type:'bar', name:'Known structural component', x:['Direct','Indirect','Total'],
      y:[1.3,p.alpha*p.beta,total], marker:{color:['#7954a1','#b66a0b','#087e64']}},
      {type:'scatter',mode:'markers',name:'Estimated treatment coefficient',x:['Direct','Total'],
        y:[s.estimates['C + M'],s.estimates.C],marker:{size:14,color:'#192d42',symbol:'diamond'}}],
    {title:{text:'Linear additive decomposition'},yaxis:{title:{text:'Score units'}}});
    $('metrics').textContent = 'Direct = 1.3; indirect = '+p.alpha.toFixed(2)+' × '+p.beta.toFixed(2)+' = '+(p.alpha*p.beta).toFixed(3)+'; total = '+total.toFixed(3)+'. OLS(C)='+fmt(s.estimates.C)+'; OLS(C,M)='+fmt(s.estimates['C + M'])+'.';
    $('table').innerHTML = '<p>Dark diamonds are finite-sample estimates. Controlling M removes the mediated path in this SCM. This direct-effect interpretation requires additive linear structure, no treatment–mediator interaction and no mediator–outcome confounding.</p>';
  }
  function intervention(d, s) {
    const a = +$('treatment').value, natural = ids(d).filter(i => d.A[i] === a), potential = 'Y'+a;
    chart('chart', [{type:'histogram',x:natural.map(i=>d.Y[i]),histnorm:'probability density',name:'Observed A='+a,
      marker:{color:'#b66a0b'},xbins:{start:-10,end:16,size:.5}}],
    {title:{text:'Observe A='+a+' · naturally assigned subset'}, xaxis:{title:{text:'Y (score units)'},range:[-10,16]}, yaxis:{title:{text:'Density'},range:[0,.25]}});
    chart('detail', [{type:'histogram',x:d[potential],histnorm:'probability density',name:'do(A='+a+') · all cases',
      marker:{color:'#087e64'},xbins:{start:-10,end:16,size:.5}}],
    {title:{text:'Intervene · replace C → A, regenerate M and Y'},xaxis:{title:{text:'Y(a) · simulated counterfactual'},range:[-10,16]},yaxis:{title:{text:'Density'},range:[0,.25]}});
    const doMeans=[mean(d.Y0),mean(d.Y1)];
    $('metrics').textContent = 'Observed mean for A='+a+': '+fmt(s.means[a])+'; paired-intervention mean: '+fmt(doMeans[a])+'. Both panels use fixed axes.';
    $('table').innerHTML = metricTable(['Quantity','Observed sample','Analytical population'],[
      ['E[Y | A=0]',fmt(s.means[0]),'0.3'],['E[Y | A=1]',fmt(s.means[1]),'3.2'],
      ['E[Y(0)]',fmt(doMeans[0]),'0.75'],['E[Y(1)]',fmt(doMeans[1]),'2.75'],
      ['Difference',fmt(s.means[1]-s.means[0])+' observational; '+fmt(doMeans[1]-doMeans[0])+' intervention','2.9 association; 2.0 causal']]);
    return {do_means:doMeans, observational_means:s.means};
  }
  function geometry(d) {
    const traces=[], expected=[];
    for(const c of [0,1]) for(const a of [0,1]) {
      const rows=sampled(ids(d).filter(i=>d.A[i]===a && d.C[i]===c),250);
      const visible=$('a'+a).checked && $('c'+c).checked;
      traces.push({type:'scatter3d',mode:'markers',name:'A='+a+', C='+c,visible,
        x:rows.map(i=>a+jitter(i)),y:rows.map(i=>c+jitter(i+19)),z:rows.map(i=>d.Y[i]),
        customdata:rows.map(i=>[d.A[i],d.C[i]]),marker:{size:3,color:palette[2*c+a],opacity:.55},
        hovertemplate:'Actual A=%{customdata[0]}, C=%{customdata[1]}<br>Y=%{z:.3f}<extra>%{fullData.name}</extra>'});
      if(visible) expected.push([a,c,2*a+1.5*c]);
    }
    traces.push({type:'scatter3d',mode:'markers+text',name:'Known E[Y | A,C]',visible:$('expected').checked,
      x:expected.map(v=>v[0]),y:expected.map(v=>v[1]),z:expected.map(v=>v[2]),
      text:expected.map(v=>'μ='+v[2].toFixed(1)),textposition:'top center',marker:{color:'#102d40',size:8,symbol:'diamond'}});
    if($('comparisons').checked) for(const c of [0,1]) if($('c'+c).checked && $('a0').checked && $('a1').checked) traces.push({type:'scatter3d',mode:'lines',name:'Within C='+c+': Δ=2',
      x:[0,1],y:[c,c],z:[1.5*c,2+1.5*c],line:{color:'#087e64',width:7,dash:'dash'},hoverinfo:'name'});
    chart('chart',traces,{title:{text:'Four binary cells · no continuous response surface'},height:600,
      scene:{xaxis:{title:{text:'A · treatment'},tickvals:[0,1],ticktext:['0 disabled','1 enabled']},
        yaxis:{title:{text:'C · complexity'},tickvals:[0,1],ticktext:['0 low','1 high']},
        zaxis:{title:{text:'Y · score'}},camera:{eye:{x:1.5,y:1.6,z:1.0}}},
      legend:{orientation:'h',y:-.1},margin:{l:0,r:0,t:60,b:90}});
    $('detail').style.display='none';
    $('metrics').textContent='Observed points are jittered by at most 0.09. Expected markers use exact binary coordinates: μ(A,C)=2A+1.5C. Dashed segments connect categories to show contrasts, not a continuous response.';
    return {expected};
  }
  function adjustment(d,s,p) {
    const names=['None','C','C + M','C + S'], total=1.3+p.alpha*p.beta;
    chart('chart',[{type:'bar',x:names.map(v=>'Controls: '+v),y:names.map(v=>s.estimates[v]),
      marker:{color:palette},name:'Finite-sample OLS coefficient'}],
    {title:{text:'Same sample, different adjustment choices'},yaxis:{title:{text:'Treatment coefficient'}},
      shapes:[line(total,'#087e64','Known total = '+total.toFixed(2)),line(1.3,'#7954a1','Direct path = 1.30')]});
    const rows=sampled(ids(d));
    chart('detail',[{type:'scatter',mode:'markers',x:rows.map(i=>s.ra[i]),y:rows.map(i=>s.ry[i]),
      marker:{color:rows.map(i=>palette[d.C[i]]),size:5,opacity:.45},name:'Residualized cases'},
      {type:'scatter',mode:'lines',x:[-.85,.85],y:[-.85*s.fwl,.85*s.fwl],name:'FWL slope = '+fmt(s.fwl),line:{color:'#087e64',width:3}}],
    {title:{text:'FWL · remove intercept and linear C from A and Y'},xaxis:{title:{text:'r_A = A − fitted E_linear[A | C]'}},yaxis:{title:{text:'r_Y = Y − fitted E_linear[Y | C]'}}});
    $('metrics').textContent='FWL='+fmt(s.fwl)+'; joint OLS(C)='+fmt(s.estimates.C)+'; absolute difference='+Math.abs(s.fwl-s.estimates.C).toExponential(2)+'. Population omitted-variable bias = 0.6 × '+p.confounding.toFixed(2)+'.';
    $('table').innerHTML=metricTable(['Controls','What the coefficient represents here'],[
      ['None','Confounded association: total + 0.6 × C coefficient'],['C','Total effect under this constant-effect SCM'],
      ['C + M','Direct path under additional mediation assumptions'],['C + S','Total minus 1.2/(1 + collider noise²): collider bias']]);
  }
  function sampleSize(seed) {
    const records=L.bias[String(seed)], traces=[];
    const band=$('range').value==='range';
    for(const [j,name] of ['None','C','C + S'].entries()) {
      const r=records.filter(x=>x.estimator===name);
      traces.push({type:'scatter',mode:'lines+markers',name:'Controls: '+name,x:r.map(v=>v.n),y:r.map(v=>v.mean),
        line:{color:palette[j]},error_y:{type:'data',symmetric:false,array:r.map(v=>band?v.high-v.mean:v.sd),
          arrayminus:r.map(v=>band?v.mean-v.low:v.sd),visible:true},
        customdata:r.map(v=>[v.sd,v.mcse]),hovertemplate:'n=%{x}<br>mean=%{y:.4f}<br>SD=%{customdata[0]:.4f}<br>MCSE(mean)=%{customdata[1]:.4f}<extra>%{fullData.name}</extra>'});
    }
    chart('chart',traces,{title:{text:'More data reduces spread around each model’s target'},
      xaxis:{type:'log',title:{text:'Cases per independent synthetic sample'}},yaxis:{title:{text:'Mean estimated treatment coefficient'}},
      shapes:[line(2,'#087e64','Known total effect = 2.0')]});
    $('detail').style.display='none';
    $('table').innerHTML=metricTable(['n','Controls','Mean','SD','MCSE(mean)'],records.map(r=>[r.n,r.estimator,fmt(r.mean),fmt(r.sd),fmt(r.mcse)]));
    $('metrics').textContent=L.repetitions+' repetitions per size. Bars show '+(band?'empirical 10th–90th percentile sampling ranges':'±1 sample SD')+', not confidence intervals for a causal effect. Curves use fresh samples, not the interactive sample-size prefix. Population limits: 2.9, 2.0, 1.04.';
    return {records};
  }
  function parameters() {
    const get=(id,fallback)=>$(id)?+$(id).value:fallback;
    return {alpha:get('alpha',.7),beta:get('beta',1),confounding:get('confounding',1.5),noise:get('noise',.5)};
  }
  function render() {
    plots=[];
    document.body.dataset.ready='false';
    $('chart').style.height=mod==='geometry'?'600px':'480px';
    $('contrast').style.display='none';
    $('detail').style.display='block'; $('table').innerHTML='';
    for(const el of document.querySelectorAll('input[type=range]')) if($(el.id+'-value')) $(el.id+'-value').textContent=el.value;
    const seed=+$('seed').value, p=parameters(), n=+$('n').value;
    const d=construct(L.tapes[String(seed)],p,n), s=statistics(d);
    let extra={};
    if(mod==='dag') extra=renderDag();
    if(mod==='confounding') confounding(d,s);
    if(mod==='collider') extra=collider(d,p);
    if(mod==='mediation') mediation(d,s,p);
    if(mod==='intervention') extra=intervention(d,s);
    if(mod==='geometry') extra=geometry(d);
    if(mod==='adjustment') adjustment(d,s,p);
    if(mod==='sample-size') extra=sampleSize(seed);
    globalThis.labState={module:mod,seed,n,parameters:p,summary:s,extra};
    globalThis.labReady=Promise.all(plots).then(()=>{document.body.dataset.ready='true';});
    globalThis.labReady.catch(error=>{$('metrics').textContent='Rendering error: '+error.message;throw error;});
  }
  document.addEventListener('change',e=>{
    if(e.target.matches('input,select')) render();
  });
  if($('play')) $('play').addEventListener('click',()=>{
    if(timer){clearInterval(timer);timer=null;$('play').textContent='Animate paths';}
    else {timer=setInterval(()=>{phase++;render();},1300);$('play').textContent='Pause animation';}
  });
  render();
})();

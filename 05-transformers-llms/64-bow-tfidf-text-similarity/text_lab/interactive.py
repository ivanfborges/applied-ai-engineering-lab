"""Self-contained Plotly pages: live mathematical controls, no remote assets."""

import html
import json

import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from . import core


SCRIPT = r"""
const el=id=>document.getElementById(id), value=id=>Number(el(id).value);
const norm=a=>Math.hypot(...a), dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
const unit=a=>norm(a)?a.map(x=>x/norm(a)):a.map(()=>0);
const tfFactor=(t,r,k,b)=>t===0?0:t*(k+1)/(t+k*(1-b+b*r));
const idfs=(n,d)=>[Math.log(n/d),1+Math.log((n+1)/(d+1)),Math.log1p((n-d+.5)/(d+.5))];
const line=(x,y,name,color)=>({x,y,name,type:'scatter',mode:'lines',line:{color}});
const colors=['#2673b8','#238457','#d07826','#8962ab','#65758b'];
function arrow3(v,name,color,points){
 const p=points||[0,0,0];
 return [{type:'scatter3d',mode:'lines+markers',name,x:[p[0],v[0]],y:[p[1],v[1]],z:[p[2],v[2]],line:{color,width:6},marker:{size:2}},
 {type:'cone',x:[v[0]],y:[v[1]],z:[v[2]],u:[v[0]-p[0]],v:[v[1]-p[1]],w:[v[2]-p[2]],anchor:'tip',sizemode:'absolute',sizeref:.14,showscale:false,colorscale:[[0,color],[1,color]],showlegend:false}];
}
function sphere(positive=false){
 const x=[],y=[],z=[];
 for(let i=0;i<21;i++){const a=(positive?Math.PI/2:Math.PI)*i/20;const xr=[],yr=[],zr=[];
  for(let j=0;j<31;j++){const b=(positive?Math.PI/2:2*Math.PI)*j/30;xr.push(Math.sin(a)*Math.cos(b));yr.push(Math.sin(a)*Math.sin(b));zr.push(Math.cos(a));}
  x.push(xr);y.push(yr);z.push(zr);
 }return {type:'surface',x,y,z,opacity:.14,showscale:false,colorscale:[[0,'#8da8ba'],[1,'#8da8ba']],hoverinfo:'skip',name:'unit sphere'};
}
async function update(){
 let traces=[], layout={title:TITLE,margin:{t:70,l:65,r:35,b:70},paper_bgcolor:'white',uirevision:'keep-camera',legend:{orientation:'h',y:-.2}},summary={};
 if(KIND==='tf'){
  const length=value('length'), average=value('average'), k=value('k1'), b=value('b');
  el('tf').max=Math.min(100,length);if(value('tf')>length)el('tf').value=length;
  const selected=value('tf'), x=Array.from({length:Math.min(100,length)+1},(_,i)=>i);
  const arrays=[x.map(t=>+(t>0)),x,x.map(t=>t/length),x.map(t=>t?1+Math.log(t):0),x.map(t=>tfFactor(t,length/average,k,b))];
  const tfColors=['#65758b','#2673b8','#8962ab','#238457','#d07826'];
  ['Binary','Raw','Relative TF','Sublinear TF','BM25 TF factor'].forEach((name,i)=>traces.push(line(x,arrays[i],name,tfColors[i])));
  layout.xaxis={title:{text:'Term occurrences; fixed document length='+length}};layout.yaxis={title:{text:'TF factor (different conventions)'}};
  summary={tf:selected,document_length:length,average_length:average,bm25_factor:tfFactor(selected,length/average,k,b),relative_tf:selected/length,sublinear_tf:selected?1+Math.log(selected):0,binary:+(selected>0),assumption:'Other tokens are replaced; length stays fixed. Curves restricted to TF <= length.'};
  traces.push({x:[selected],y:[summary.bm25_factor],mode:'markers',name:'Selected BM25 point',marker:{size:12,color:'#d07826'}});
 }else if(KIND==='idf'){
  const n=value('N');el('DF').max=n;if(value('DF')>n)el('DF').value=n;const d=value('DF');
  const x=Array.from({length:n},(_,i)=>i+1);['Classical','Smoothed TF-IDF','Positive BM25'].forEach((name,i)=>traces.push(line(x,x.map(df=>idfs(n,df)[i]),name,colors[i])));
  const current=idfs(n,d);traces.push({x:[d,d,d],y:current,mode:'markers',name:'Selected DF',marker:{size:10,color:colors.slice(0,3)}});
  layout.xaxis={title:{text:'Document frequency (1 <= DF <= N)'}};layout.yaxis={title:{text:'IDF (natural log)'}};
  summary={N:n,DF:d,probability:d/n,self_information:-Math.log(d/n),classical:current[0],smoothed:current[1],positive_bm25:current[2]};
 }else if(KIND==='surface'||KIND==='bm25'){
  const xs=Array.from({length:41},(_,i)=>i), ys=Array.from({length:41},(_,i)=>KIND==='surface'?1+(value('N')-1)*i/40:.4+3.6*i/40);
  // DF is discrete: integer-valued sampled rows; duplicates at small N are harmless.
  if(KIND==='surface')ys.forEach((y,i)=>ys[i]=Math.round(y));
  const z=ys.map(y=>xs.map(t=>KIND==='surface'?(el('mode').value==='log'?(t?1+Math.log(t):0):t)*idfs(value('N'),y)[1]:tfFactor(t,y,value('k1'),value('b'))));
  traces=[{type:'surface',x:xs,y:ys,z,colorscale:KIND==='surface'?'Greens':'Oranges',colorbar:{title:{text:KIND==='surface'?'TF-IDF':'TF factor'}}}];
  layout.scene={xaxis:{title:{text:'Term frequency'}},yaxis:{title:{text:KIND==='surface'?'Document frequency':'Document length / average length'}},zaxis:{title:{text:KIND==='surface'?'Unnormalized TF-IDF':'BM25 TF factor'}},aspectmode:'cube'};
  summary=KIND==='surface'?{N:value('N'),tf_mode:el('mode').value,example_tf20_df1:z[0][20],domain:'0 <= TF <= 40; 1 <= DF <= N; smoothed IDF'}:{k1:value('k1'),b:value('b'),tf20_ratio1:tfFactor(20,1,value('k1'),value('b')),assumption:'Average length=100; ratio>=0.4 implies length>=40; TF<=40. TF factor only, before IDF.'};
 }else if(KIND==='vectors'){
  const stage=el('stage').value, matrix=PAYLOAD[stage];traces=[sphere(true)];
  matrix.forEach((v,i)=>traces.push(...arrow3(v,'D'+i,colors[i])));
  const limit=stage==='unit'?1.15:Math.max(...PAYLOAD.weighted.flat())*1.2;
  layout.scene={xaxis:{title:{text:'machine'},range:[0,limit]},yaxis:{title:{text:'learning'},range:[0,limit]},zaxis:{title:{text:'data'},range:[0,limit]},aspectmode:'cube'};
  summary={stage,coordinates:matrix,norms:matrix.map(norm),idf:PAYLOAD.idf,explanation:'Exactly three vocabulary axes; no projection. Unit vectors lie on the positive-octant sphere.'};
 }else if(KIND==='cos2'||KIND==='cos3'){
  const a=value('a'),b=value('bmag'),angle=value('angle')*Math.PI/180,elev=KIND==='cos3'?value('elevation')*Math.PI/180:0;
  const A=KIND==='cos3'?[a,0,0]:[a,0],B=KIND==='cos3'?[b*Math.cos(elev)*Math.cos(angle),b*Math.cos(elev)*Math.sin(angle),b*Math.sin(elev)]:[b*Math.cos(angle),b*Math.sin(angle)];
  const c=a&&b?Math.max(-1,Math.min(1,dot(A,B)/(a*b))):null;
  summary={A,B,dot:dot(A,B),normA:norm(A),normB:norm(B),cosine:c,angle_degrees:c===null?null:Math.acos(c)*180/Math.PI,euclidean:norm(A.map((v,i)=>v-B[i])),zero_policy:c===null?'Cosine and angle undefined: zero vector':'Nonzero vectors'};
  if(KIND==='cos2'){
   const theta=Array.from({length:101},(_,i)=>2*Math.PI*i/100);traces=[line(theta.map(Math.cos),theta.map(Math.sin),'Unit circle','#cbd5e1')];
   [A,B].forEach((v,i)=>traces.push({x:[0,v[0]],y:[0,v[1]],mode:'lines+markers',name:i?'B':'A',line:{color:colors[i],width:5}}));
   layout.xaxis={title:{text:'x'},range:[-4.5,4.5],zeroline:true};layout.yaxis={title:{text:'y'},range:[-4.5,4.5],scaleanchor:'x'};
  }else{
   const projection=a?[B[0],0,0]:[0,0,0];traces=[sphere(),...arrow3(A,'A',colors[0]),...arrow3(B,'B',colors[1]),...arrow3(unit(A),'unit A','#65758b'),...arrow3(unit(B),'unit B','#8962ab')];
   if(a)traces.push(...arrow3(projection,'B projected on A',colors[2]));
   summary.projection=a?projection:null;
   layout.scene={xaxis:{title:{text:'x'},range:[-4.5,4.5]},yaxis:{title:{text:'y'},range:[-4.5,4.5]},zaxis:{title:{text:'z'},range:[-4.5,4.5]},aspectmode:'cube'};
  }
 }
 document.querySelectorAll('input').forEach(input=>el(input.id+'-value').textContent=input.value);
 await Plotly.react('plot',traces,layout,{responsive:true,displaylogo:false});
 el('metrics').textContent=JSON.stringify(summary,null,2);window.labState=summary;window.labReady=true;
}
document.querySelectorAll('input,select').forEach(input=>input.addEventListener('input',()=>update().catch(e=>{el('metrics').textContent=String(e);throw e;})));
update();
"""


def explorer(export, filename, kind, title, controls, payload=None):
    if not export.wants(filename):
        return
    fields = []
    for key, label, minimum, maximum, step, value in controls:
        if isinstance(minimum, list):
            options = "".join(f'<option value="{html.escape(option)}" {"selected" if option == value else ""}>{html.escape(option)}</option>' for option in minimum)
            fields.append(f'<label>{html.escape(label)} <select id="{key}">{options}</select></label>')
        else:
            fields.append(f'<label for="{key}">{html.escape(label)}: <output id="{key}-value">{value}</output>'
                          f'<input id="{key}" type="range" min="{minimum}" max="{maximum}" step="{step}" value="{value}"></label>')
    data = json.dumps(payload or {}, allow_nan=False).replace("</", "<\\/")
    contents = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
                f'<title>{html.escape(title)}</title><style>body{{font:16px system-ui;margin:24px;color:#203047}}'
                '#controls{display:flex;flex-wrap:wrap;gap:20px}label{display:grid;min-width:170px}#plot{height:660px}'
                'pre{white-space:pre-wrap;background:#f0f5f8;padding:16px}input,select{padding:6px}</style></head><body>'
                f'<h1>{html.escape(title)}</h1><p>Synthetic educational calculations. Controls recompute the displayed mathematics. '
                'Drag 3D plots to rotate; use the toolbar to zoom. No external network or model.</p>'
                f'<div id="controls">{"".join(fields)}</div><div id="plot" role="img" aria-label="{html.escape(title)}"></div>'
                '<h2>Calculated state</h2><pre id="metrics" aria-live="polite"></pre>'
                '<p>Interpretation: author review required. Cosine is not a relevance probability. Signed vectors are not nonnegative TF-IDF.</p>'
                f'<script>{get_plotlyjs()}</script><script>const KIND={json.dumps(kind)},TITLE={json.dumps(title)},PAYLOAD={data};\n{SCRIPT}</script></body></html>')
    path = export.path(filename); path.write_text(contents, encoding="utf-8"); export.paths.append(path)


def generate(export, section, seed=64):
    if section == "tf":
        explorer(export, "08_term_frequency_interactive.html", "tf", "Term repetition under explicit length assumptions", [
            ("tf", "Occurrences", 0, 100, 1, 3), ("length", "Document length", 1, 1000, 1, 100),
            ("average", "Average document length", 1, 1000, 1, 100), ("k1", "BM25 k1", .1, 3, .1, 1.2), ("b", "BM25 b", 0, 1, .05, .75)])
    elif section == "idf":
        explorer(export, "11_idf_interactive.html", "idf", "Term prevalence and information", [
            ("N", "Corpus documents", 2, 1000, 1, 100), ("DF", "Documents containing term", 1, 100, 1, 5)])
    elif section == "tfidf":
        explorer(export, "12_tfidf_weight_surface_3d.html", "surface", "TF × smoothed IDF: a genuine weight surface", [
            ("N", "Corpus size", 10, 1000, 10, 100), ("mode", "TF convention", ["raw", "log"], None, None, "raw")])
    elif section == "vectors":
        data = core.three_vectors(); payload = {key: value.tolist() for key, value in data.items() if isinstance(value, np.ndarray)}
        for filename, default in (("13_bow_vector_space_3d.html", "counts"), ("15_normalized_vector_space_3d.html", "unit")):
            explorer(export, filename, "vectors", "Three vocabulary axes: counts, IDF and normalization", [
                ("stage", "Representation", ["counts", "weighted", "unit"], None, None, default)], payload)
    elif section == "cosine":
        for filename, kind in (("16_cosine_similarity_2d.html", "cos2"), ("17_cosine_similarity_3d.html", "cos3")):
            controls = [("angle", "Angle / azimuth (degrees)", 0, 180 if kind == "cos2" else 360, 1, 45),
                        ("a", "Magnitude A", 0, 4, .1, 1), ("bmag", "Magnitude B", 0, 4, .1, 2)]
            if kind == "cos3": controls.append(("elevation", "Elevation (degrees)", -90, 90, 1, 30))
            explorer(export, filename, kind, "Cosine geometry: direction, scale and zero vectors", controls)
    elif section == "bm25":
        explorer(export, "33_bm25_saturation_3d.html", "bm25", "BM25: saturation and corpus-relative length", [
            ("k1", "TF saturation k1", .1, 3, .1, 1.2), ("b", "Length normalization b", 0, 1, .05, .75)])
    elif section == "projections":
        documents, labels, query, result = core.projection_fixture(seed)
        for dim, filename in ((2, "23_document_projection_2d.html"), (3, "24_document_projection_3d.html")):
            if not export.wants(filename): continue
            figure = go.Figure(); buttons = []
            groups = sorted(set(labels))
            for method, data in result.items():
                for category in groups:
                    indices = np.flatnonzero(np.array(labels) == category)
                    points = data["points"][indices]
                    custom = [[int(i), documents[i]] for i in indices]
                    common = dict(name=category, mode="markers", customdata=custom,
                                  hovertemplate="D%{customdata[0]}<br>%{customdata[1]}<extra>%{fullData.name}</extra>", visible=method == "tfidf")
                    trace = go.Scatter(x=points[:, 0], y=points[:, 1], **common) if dim == 2 else go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], **common)
                    figure.add_trace(trace)
                point = data["query"]
                common = dict(name="Query: " + query, mode="markers", marker=dict(symbol="diamond", size=12, color="black"), visible=method == "tfidf")
                figure.add_trace(go.Scatter(x=[point[0]], y=[point[1]], **common) if dim == 2 else go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]], **common))
            for method, data in result.items():
                visibility = [method == name for name in result for _ in range(len(groups) + 1)]
                buttons.append(dict(label=method, method="update", args=[{"visible": visibility}, {"title.text": f"{method} SVD {dim}D; explained variance={sum(data['variance'][:dim]):.3f}; original ranking may differ"}]))
            figure.update_layout(title=f"TF-IDF SVD {dim}D; explained variance={sum(result['tfidf']['variance'][:dim]):.3f}; projection is approximate",
                                 updatemenus=[dict(buttons=buttons, active=1)], xaxis_title="SVD 1", yaxis_title="SVD 2",
                                 scene=dict(xaxis_title="SVD 1", yaxis_title="SVD 2", zaxis_title="SVD 3"), height=720,
                                 annotations=[dict(text="Synthetic corpus; each query uses the document-fitted SVD. Low-dimensional distance is not original cosine.", x=.5, y=-.12, xref="paper", yref="paper", showarrow=False)])
            figure.write_html(export.path(filename), include_plotlyjs=True, config={"displaylogo": False}, auto_open=False)
            export.paths.append(export.path(filename))

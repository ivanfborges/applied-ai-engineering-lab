"""Standalone, offline HTML explorations and reusable Plotly figures."""

from __future__ import annotations

import html
import json
from pathlib import Path

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from . import core as c


COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
STYLE = """
body{font-family:system-ui,sans-serif;background:#f5f7fb;color:#142536;margin:0}
main{max-width:1250px;margin:30px auto;padding:24px;background:white;border-radius:14px}
h1{font-size:1.8rem}h2{font-size:1.2rem}.note{background:#eaf2f8;padding:14px;border-left:5px solid #0072b2}
label{display:inline-flex;flex-direction:column;gap:5px;margin:10px 18px 12px 0}
input,select,button{font:inherit;padding:8px;border:1px solid #8995a0;border-radius:5px}
textarea{width:96%;font:inherit;padding:10px}pre{white-space:pre-wrap;overflow-wrap:anywhere}
.tokens{display:flex;flex-wrap:wrap;gap:7px;margin:16px 0}.token{padding:9px;border:2px solid #0072b2;border-radius:7px;white-space:pre}
.active{background:#0072b2;color:white}.muted{color:#546573}.metrics{font-size:1.05rem;line-height:1.6}
table{border-collapse:collapse;width:100%}td,th{text-align:left;border-bottom:1px solid #dce3eb;padding:8px}
#plot{width:100%;min-height:570px}button{cursor:pointer;background:#eaf2f8}
"""


def js_data(value) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False).replace("</", "<\\/")


def write_page(path: Path, title: str, question: str, controls: str, script: str,
               figure: go.Figure | None = None, body: str = "") -> None:
    chart = pio.to_html(figure, full_html=False, include_plotlyjs=True, div_id="plot",
                        config={"displaylogo": False, "responsive": True}) if figure is not None else ""
    path.write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                    f'<title>{html.escape(title)}</title><link rel="icon" href="data:,"><style>{STYLE}</style><main><h1>{html.escape(title)}</h1>'
                    f'<p class="note">{html.escape(question)}</p>{controls}{body}{chart}'
                    f'<script>{script}</script><footer class="muted">Synthetic educational lab · Offline · Author interpretation review required</footer></main></html>', encoding="utf-8")


def tokenization_html(path: Path) -> None:
    data = {}
    for strategy in c.STRATEGIES:
        try:
            rows = [c.segment(s, strategy) for s in c.SENTENCES]
            vocab = len({p["token"] for row in rows for p in row})
            data[strategy] = {"rows": rows, "vocabulary": vocab}
        except c.ResourceUnavailable as exc:
            data[strategy] = {"error": str(exc)}
    controls = '<label>Sentence<select id="sentence">' + "".join(f'<option value="{i}">{html.escape(s)}</option>' for i, s in enumerate(c.SENTENCES)) + '</select></label>'
    controls += '<label>Strategy<select id="strategy">' + "".join(f'<option>{s}</option>' for s in c.STRATEGIES) + '</select></label><button id="play">Animate boundaries</button>'
    body = '<pre id="source"></pre><div id="summary" class="metrics"></div><div id="boxes" class="tokens"></div><pre id="offsets"></pre><details><summary>Actual educational BPE model</summary><pre id="model"></pre></details>'
    script = f'const data={js_data(data)}, sentences={js_data(c.SENTENCES)}, model={js_data({"algorithm":"character BPE; 40 frequency merges maximum; lexicographic ties; merges may span spaces", "corpus_sha256":c.trained_bpe().corpus_sha256,"vocabulary":c.trained_bpe().vocabulary,"merges":c.trained_bpe().merges})};' + """
const el=id=>document.getElementById(id);let timer=null;
function render(){clearInterval(timer);let strategy=el('strategy').value,index=+el('sentence').value,d=data[strategy];
el('source').textContent=sentences[index];el('boxes').replaceChildren();el('offsets').textContent='';
if(d.error){el('summary').textContent=d.error;return;}
let row=d.rows[index];el('summary').textContent=`${row.length} tokens · ${d.vocabulary} observed distinct tokens across six sentences · offsets: ${strategy==='byte'?'UTF-8 bytes':'Python Unicode code points'} [start,end)`;
for(let p of row){let box=document.createElement('span');box.className='token';box.textContent=p.token===' '?'␠':p.token;box.title=`[${p.start}, ${p.end})`;el('boxes').append(box);}
el('offsets').textContent=row.map(p=>`[${p.start},${p.end}) ${JSON.stringify(p.token)}`).join(' | ');
el('model').textContent=JSON.stringify(model,null,2);}
el('sentence').onchange=el('strategy').onchange=render;
el('play').onclick=()=>{clearInterval(timer);let i=0,boxes=[...el('boxes').children];boxes.forEach(b=>b.classList.remove('active'));timer=setInterval(()=>{if(i>=boxes.length){clearInterval(timer);return;}boxes[i++].classList.add('active');},180);};render();
"""
    write_page(path, "1 · How do token boundaries change?", "Boundary segmentation on original text; regex does not normalize here. Linguistic: NLTK Treebank (English), Toktok for the supplied Portuguese examples. Character = code point, not grapheme. Bytes use all 256 possible values; BPE has a learned alphabet and <unk> for unseen characters. Observed vocabulary differs from model vocabulary; no pretrained outputs are invented.", controls, script, body=body)


def ngram_html(path: Path) -> None:
    controls = '<label>Input sentence<input id="text" size="65" value="machine learning models learn useful patterns"></label><label>Unit<select id="unit"><option>word</option><option>character</option></select></label><label>Order n<input id="n" type="number" min="1" max="30" value="2"></label><button id="step">Next window</button><button id="reset">Reset</button>'
    script = r"""
const el=id=>document.getElementById(id);let index=0,grams=[];
function compute(){let n=+el('n').value,text=el('text').value.normalize('NFC').toLowerCase().replaceAll('’',"'");
if(!Number.isInteger(n)||n<1||n>30){el('summary').textContent='Choose an integer n from 1 to 30';return;}
let tokens=el('unit').value==='character'?Array.from(text):Array.from(text.matchAll(/[\p{L}\p{N}]+(?:'[\p{L}\p{N}]+)*/gu),m=>m[0]);
// The character unit includes spaces. Word mode is a browser NFC/lowercase regex baseline, not full Python casefold.
grams=[];for(let i=0;i<=tokens.length-n;i++)grams.push(tokens.slice(i,i+n));index=Math.min(index,grams.length);
el('boxes').replaceChildren();tokens.forEach((t,i)=>{let b=document.createElement('span');b.className='token'+(index>0&&i>=index-1&&i<index-1+n?' active':'');b.textContent=t===' '?'␠':t;el('boxes').append(b);});
el('summary').textContent=`m=${tokens.length} tokens; n=${n}; max(0,m−n+1)=${Math.max(0,tokens.length-n+1)} occurrences; extracted=${index}`;
el('collection').textContent=grams.slice(0,index).map((g,i)=>`${i+1}. ${JSON.stringify(g)}`).join('\n');}
el('step').onclick=()=>{index=Math.min(index+1,grams.length);compute();};el('reset').onclick=()=>{index=0;compute();};
['text','n','unit'].forEach(id=>el(id).oninput=()=>{index=0;compute();});compute();
""".replace(r"\\p", r"\p")
    write_page(path, "4 · Which windows become features?", "Contiguous occurrences without padding. m is sequence length; n is window size. Try 'a a a', an empty sentence, or n > m. Browser word mode uses NFC/lowercase (not full Python casefold); the Python dashboard uses the shared study analyzer. Character mode includes spaces and uses Unicode code points.", controls, script, body='<div id="summary" class="metrics"></div><div id="boxes" class="tokens"></div><h2>Generated collection</h2><pre id="collection"></pre>')


def feature_html(path: Path) -> None:
    data = {}
    for n in (1, 2, 3):
        for df in (1, 2, 4):
            for limit in (0, 30, 60):
                try:
                    stats, matrix, features = c.feature_stats(c.synthetic_corpus(), n, df, limit or None)
                    data[f"{n}:{df}:{limit}"] = {"stats": stats, "matrix": matrix.toarray().tolist(), "features": features}
                except ValueError as exc:
                    data[f"{n}:{df}:{limit}"] = {"error": str(exc)}
    fig = go.Figure(go.Heatmap(z=[[0]], colorscale="Blues", colorbar={"title": "Count"}))
    fig.update_layout(xaxis_title="Feature index (hover for term)", yaxis_title="Synthetic document ID", height=600)
    controls = '<label>Maximum order<select id="n"><option>1</option><option>2</option><option>3</option></select></label><label>min_df (documents)<select id="df"><option>1</option><option>2</option><option>4</option></select></label><label>max_features<select id="limit"><option value="0">No cap</option><option>30</option><option>60</option></select></label>'
    script = f'const data={js_data(data)};' + """
const el=id=>document.getElementById(id);
function render(){let d=data[`${el('n').value}:${el('df').value}:${el('limit').value}`];if(d.error){el('summary').textContent=d.error;return;}
el('summary').textContent=JSON.stringify(d.stats,null,2);let custom=d.matrix.map(row=>row.map((_,i)=>d.features[i]));
Plotly.react('plot',[{type:'heatmap',z:d.matrix,customdata:custom,hovertemplate:'D%{y}: %{customdata} = %{z}<extra></extra>',colorscale:'Blues',colorbar:{title:'Count'}}],{height:600,xaxis:{title:'Feature index'},yaxis:{title:'Document ID'},margin:{t:30}});}
['n','df','limit'].forEach(id=>el(id).onchange=render);render();
"""
    write_page(path, "5 · What does pruning change?", "27 exact precomputed scikit-learn configurations on the same 45 synthetic documents. Counts, nnz, density and storage are recomputed for each configuration; sparse bytes = data + indices + indptr, excluding Python object overhead. This is representation accounting, not measured model quality.", controls, script, fig, body='<pre id="summary"></pre>')


def morphology_figure(rows: list[dict], method: str = "porter") -> go.Figure:
    rows = [r for r in rows if method in r]
    targets = sorted({r[method] for r in rows})
    labels = [f'{r["word"]} [{r["pos"]}]' for r in rows] + targets
    links = {"source": list(range(len(rows))), "target": [len(rows) + targets.index(r[method]) for r in rows],
             "value": [1] * len(rows), "color": ["rgba(0,114,178,.20)"] * len(rows)}
    original = len({r["word"] for r in rows})
    reduction = 100 * (1 - len(targets) / original) if original else 0
    fig = go.Figure(go.Sankey(node={"label": labels, "pad": 10, "thickness": 15,
                                    "color": [COLORS[0]] * len(rows) + [COLORS[2]] * len(targets)}, link=links))
    fig.update_layout(title=f"{method}: {original} surface forms → {len(targets)} outputs ({reduction:.1f}% reduction)",
                      height=950, font={"size": 13}, margin={"l": 25, "r": 25, "t": 70, "b": 30})
    return fig


def morphology_html(path: Path, rows: list[dict]) -> None:
    methods = [m for m in ("porter", "snowball", "lemma", "portuguese") if any(m in r for r in rows)]
    figures = {method: json.loads(pio.to_json(morphology_figure(rows, method))) for method in methods}
    controls = '<label>Actual algorithm<select id="method">' + "".join(f'<option>{m}</option>' for m in methods) + '</select></label>'
    script = f'const figures={js_data(figures)};document.getElementById("method").onchange=function(){{const f=figures[this.value];Plotly.react("plot",f.data,f.layout);}};'
    ambiguity = [r for r in rows if r["word"] in ("saw", "leaves")]
    write_page(path, "3 · Which forms collapse, and which remain apart?", "Computed NLTK Porter / English Snowball / Portuguese Snowball; WordNet uses supplied POS, not predicted tags. In 'I saw a bird', the expected contextual lemma is see; in 'I bought a saw', it is saw. WordNet can return saw even with verb POS because saw is also a verb. Candidates below are actual resource outputs; no contextual selection is invented. Leaves illustrates differing noun/verb lemmas. Inspect universe/university for conflation and mouse/mice, run/ran for missed irregular relations. Whether a collision is harmful depends on the task.", controls, script, morphology_figure(rows, methods[0]), body='<details open><summary>Actual candidates: POS alone may be insufficient</summary><pre>' + html.escape(json.dumps(ambiguity, indent=2)) + '</pre></details>')


def tfidf_surface(documents: int = 100, df: int = 10, tf: int = 3) -> go.Figure:
    xs, ys = np.arange(1, documents + 1), np.arange(0, 11)
    x, y = np.meshgrid(xs, ys)
    z = c.tfidf_weight(documents, x, y)
    fig = go.Figure(go.Surface(x=xs.tolist(), y=ys.tolist(), z=z.tolist(), colorscale="Cividis", opacity=.88,
                              colorbar={"title": "TF × IDF"}))
    fig.add_trace(go.Scatter3d(x=[df], y=[tf], z=[float(c.tfidf_weight(documents, df, tf))], mode="markers",
                               marker={"size": 7, "color": COLORS[1]}, name="Selected term"))
    fig.add_trace(go.Scatter3d(x=xs.tolist(), y=[tf] * len(xs), z=c.tfidf_weight(documents, xs, tf).tolist(),
                               mode="lines", line={"width": 6, "color": COLORS[1]}, name="Fixed TF section"))
    fig.update_layout(title=f"Unnormalized TF-IDF · D={documents}", height=700,
                       margin={"t": 70, "b": 90, "r": 120}, legend={"orientation": "h", "x": 0, "y": -.10},
                       uirevision="tfidf", scene={"xaxis_title": "Document frequency df (1…D)", "yaxis_title": "Term frequency TF", "zaxis_title": "TF × IDF"})
    return fig


def tfidf_html(path: Path) -> None:
    controls = '<label>Corpus D<input id="D" type="number" min="1" max="500" value="100"></label><label>Document frequency df<input id="df" type="number" min="1" value="10"></label><label>Term frequency TF<input id="tf" type="number" min="0" max="10" value="3"></label><button id="update">Recalculate surface</button><pre id="summary"></pre>'
    script = """
const el=id=>document.getElementById(id);
async function update(){let D=+el('D').value,df=+el('df').value,tf=+el('tf').value;
if(!Number.isInteger(D)||D<1||D>500||!Number.isInteger(df)||df<1||df>D||!Number.isInteger(tf)||tf<0||tf>10){el('summary').textContent='Require integer 1 ≤ D ≤ 500, 1 ≤ df ≤ D, 0 ≤ TF ≤ 10';return;}
window.labReady=false;el('update').disabled=true;
let xs=Array.from({length:D},(_,i)=>i+1),ys=Array.from({length:11},(_,i)=>i),idf=x=>Math.log((1+D)/(1+x))+1,z=ys.map(t=>xs.map(x=>t*idf(x)));
await Plotly.restyle('plot',{x:[xs],y:[ys],z:[z]},[0]);await Plotly.restyle('plot',{x:[[df]],y:[[tf]],z:[[tf*idf(df)]]},[1]);
await Plotly.restyle('plot',{x:[xs],y:[xs.map(()=>tf)],z:[xs.map(x=>tf*idf(x))]},[2]);await Plotly.relayout('plot',{'title.text':`Unnormalized TF-IDF · D=${D}`});
el('summary').textContent=`IDF = ln((1+${D})/(1+${df}))+1 = ${idf(df).toFixed(5)}; unnormalized weight = ${(tf*idf(df)).toFixed(5)}`;window.labReady=true;el('update').disabled=false;}
el('update').onclick=update;update();
"""
    write_page(path, "7 · How do corpus rarity and within-document frequency interact?", "Smoothed IDF uses natural log. This surface is before L2 normalization; TF alone does not set a normalized document weight. df is an integer count of indexed documents containing the term. Invalid counts are rejected. Drag the 3D camera or inspect the orange fixed-TF cross-section.", controls, script, tfidf_surface())


def space_figure(result: dict, query: str) -> go.Figure:
    fig = go.Figure()
    ranking = result["ranking"]
    rank_map = {doc: i + 1 for i, doc in enumerate(ranking)}
    for j, topic in enumerate(dict.fromkeys(label for label, _ in c.TOPIC_DOCS)):
        indices = [i for i, (label, _) in enumerate(c.TOPIC_DOCS) if label == topic]
        xyz = result["documents"][indices]
        fig.add_trace(go.Scatter3d(x=xyz[:, 0].tolist(), y=xyz[:, 1].tolist(), z=xyz[:, 2].tolist(), mode="markers+text",
            text=[f"D{i}" for i in indices], name=topic,
            marker={"size": [10 if i in ranking[:3] else 5 for i in indices], "color": COLORS[j]},
            customdata=[[c.TOPIC_DOCS[i][1], float(result["scores"][i]), rank_map.get(i, "no overlap")] for i in indices],
            hovertemplate="%{text}: %{customdata[0]}<br>Original cosine=%{customdata[1]:.4f}<br>Rank=%{customdata[2]}<extra></extra>"))
    q = result["query"]
    fig.add_trace(go.Scatter3d(x=[q[0]], y=[q[1]], z=[q[2]], mode="markers", name="Query",
                               marker={"size": 10, "symbol": "diamond", "color": "#111111"}, text=[query]))
    fig.update_layout(title=f"Three-component SVD · explained variance {result['variance']:.1%} · {result['features']} original features",
                       height=700, scene={"xaxis_title": "SVD 1", "yaxis_title": "SVD 2", "zaxis_title": "SVD 3"}, margin={"t": 65, "b": 20})
    return fig


def space_html(path: Path, seed: int) -> None:
    figures, rankings = {}, {}
    for rep in ("word", "word12", "char"):
        for i, query in enumerate(c.SPACE_QUERIES):
            result = c.vector_space(rep, query, seed)
            figures[f"{rep}:{i}"] = json.loads(pio.to_json(space_figure(result, query)))
            rankings[f"{rep}:{i}"] = [{"id": d, "cosine": round(float(result["scores"][d]), 6), "text": c.TOPIC_DOCS[d][1]} for d in result["ranking"]]
    controls = '<label>Representation<select id="rep"><option>word</option><option>word12</option><option>char</option></select></label><label>Query<select id="query">' + "".join(f'<option value="{i}">{html.escape(q)}</option>' for i, q in enumerate(c.SPACE_QUERIES)) + '</select></label>'
    script = f'const figures={js_data(figures)},rankings={js_data(rankings)};' + """
function render(){let key=document.getElementById('rep').value+':'+document.getElementById('query').value,f=figures[key];Plotly.react('plot',f.data,f.layout);document.getElementById('ranking').textContent=JSON.stringify(rankings[key],null,2);}
document.getElementById('rep').onchange=document.getElementById('query').onchange=render;render();
"""
    write_page(path, "8 · What does a lexical vector space preserve?", "Large points = top three positive-score documents, not human relevance judgments. Ranking and hover scores use cosine in the original sparse feature space. Three SVD coordinates are only a projection and can distort distances. Queries are transformed by the document-fitted SVD; they do not refit the index. Lexical synonyms and negation can fail. No dense neural embeddings are used.", controls, script, space_figure(c.vector_space("word", c.SPACE_QUERIES[0], seed), c.SPACE_QUERIES[0]), body='<details open><summary>Authoritative original-space ranking</summary><pre id="ranking"></pre></details>')


def geometry_figure(angle: float = 45, magnitude: float = 1, elevation: float = 0) -> go.Figure:
    theta, phi = np.deg2rad([angle, elevation])
    a = [1., 0., 0.]
    b = (magnitude * np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])).tolist()
    fig = go.Figure()
    for vector, label, color in ((a, "a = [1,0,0]", COLORS[0]), (b, "b (signed generic coordinates)", COLORS[1])):
        fig.add_trace(go.Scatter3d(x=[0, vector[0]], y=[0, vector[1]], z=[0, vector[2]], mode="lines+markers", name=label, line={"width": 9, "color": color}))
    sim = c.cosine(np.array(a), np.array(b))
    label = "undefined (zero vector)" if sim is None else f"{sim:.5f}"
    fig.update_layout(title=f"cos(a,b) = {label}", height=650,
                       scene={"xaxis": {"range": [-3, 3]}, "yaxis": {"range": [-3, 3]}, "zaxis": {"range": [-3, 3]}, "aspectmode": "cube"})
    return fig


def geometry_html(path: Path) -> None:
    controls = '<label>Azimuth (degrees)<input id="angle" type="range" min="0" max="180" value="45"></label><label>Magnitude<input id="mag" type="range" min="0" max="3" value="1" step="0.1"></label><label>Elevation (degrees)<input id="elevation" type="range" min="-90" max="90" value="0"></label><pre id="summary"></pre>'
    script = """
const el=id=>document.getElementById(id);function update(){let a=+el('angle').value*Math.PI/180,p=+el('elevation').value*Math.PI/180,m=+el('mag').value,b=[m*Math.cos(a)*Math.cos(p),m*Math.sin(a)*Math.cos(p),m*Math.sin(p)],sim=m===0?null:Math.cos(a)*Math.cos(p);Plotly.restyle('plot',{x:[[0,b[0]]],y:[[0,b[1]]],z:[[0,b[2]]]},[1]);let text=sim===null?'undefined (zero vector)':sim.toFixed(5);Plotly.relayout('plot',{'title.text':`cos(a,b) = ${text}`});el('summary').textContent=`azimuth=${el('angle').value}°; elevation=${el('elevation').value}°; norm(a)=1; norm(b)=${m}; dot=${b[0].toFixed(5)}; cosine=${text}; actual angle=${sim===null?'undefined':(Math.acos(Math.max(-1,Math.min(1,sim)))*180/Math.PI).toFixed(2)+'°'}`;}
['angle','mag','elevation'].forEach(id=>el(id).oninput=update);update();
"""
    write_page(path, "9 · Why does direction matter more than magnitude?", "cos(a,b) = dot(a,b) / (norm(a) norm(b)). Opposite vectors have cosine −1 in a generic signed space. Ordinary nonnegative TF-IDF vectors cannot have negative cosine. A zero vector has undefined geometric cosine; retrieval libraries commonly use a zero score as a convention.", controls, script, geometry_figure())


def transition_html(path: Path, alpha: float = 0.1) -> None:
    model = c.NGramLM([c.tokenize(s) for s in c.LM_SENTENCES], 2)
    graph = nx.DiGraph()
    for context, counts in model.counts.items():
        probabilities = model.distribution(context, alpha)
        for word in counts: graph.add_edge(context[0], word, probability=float(probabilities[model.vocabulary.index(word)]))
    positions = nx.spring_layout(graph, seed=63)
    fig = go.Figure()
    annotations = []
    for source, target, data in graph.edges(data=True):
        a, b = positions[source], positions[target]
        fig.add_trace(go.Scatter(x=[a[0], b[0]], y=[a[1], b[1]], mode="lines", showlegend=False,
                                 line={"width": 1 + 9 * data["probability"], "color": COLORS[0]}, hoverinfo="skip"))
        annotations.append({"x": .35 * a[0] + .65 * b[0], "y": .35 * a[1] + .65 * b[1], "ax": a[0], "ay": a[1],
                            "xref": "x", "yref": "y", "axref": "x", "ayref": "y", "showarrow": True, "arrowhead": 3,
                            "text": f'{data["probability"]:.2f}', "font": {"size": 11}, "bgcolor": "white"})
    fig.add_trace(go.Scatter(x=[positions[n][0] for n in graph], y=[positions[n][1] for n in graph],
                             mode="markers+text", text=list(graph), marker={"size": 24, "color": COLORS[2]}, textposition="top center", name="Tokens"))
    fig.update_layout(title=f"Observed bigram transitions · add-alpha={alpha}", height=700, annotations=annotations,
                       xaxis={"visible": False}, yaxis={"visible": False}, margin={"t": 70, "b": 50})
    distributions = {" ".join(context): model.distribution(context, alpha).tolist() for context in model.counts}
    controls = '<label>Observed context<select id="context">' + "".join(f'<option>{html.escape(key)}</option>' for key in distributions) + '</select></label>'
    script = f'const distributions={js_data(distributions)},vocabulary={js_data(model.vocabulary)};' + r"""
function render(){let p=distributions[document.getElementById('context').value];document.getElementById('distribution').textContent=vocabulary.map((w,i)=>`${w}: ${p[i].toFixed(5)}`).join('\n')+`\nSum = ${p.reduce((a,b)=>a+b,0).toFixed(8)}`;}
document.getElementById('context').onchange=render;render();
"""
    write_page(path, "10 · Which next-token transitions were observed?", "Directed arrows and edge widths show conditional probabilities. Only observed edges are drawn; with smoothing, omitted unseen edges still have positive probability. BOS is context only and EOS is a predicted token. The complete selected distribution below includes all prediction-vocabulary tokens; this is a finite-context Counter model, not a Transformer.", controls, script, fig, body='<pre id="distribution"></pre>')


def ocr_html(path: Path, report: dict) -> None:
    controls = '<label>Corruption rate<select id="rate">' + "".join(f'<option>{r}</option>' for r in report["rates"]) + '</select></label><label>Representation<select id="rep">' + "".join(f'<option>{rep}</option>' for rep in c.REPRESENTATIONS) + '</select></label>'
    script = f'const examples={js_data(report["examples"])};' + """
function render(){const rows=examples.filter(r=>r.rate===+document.getElementById('rate').value&&r.representation===document.getElementById('rep').value);document.getElementById('examples').textContent=JSON.stringify(rows,null,2);}
document.getElementById('rate').onchange=document.getElementById('rep').onchange=render;render();
"""
    write_page(path, "12 · Which lexical matches survive synthetic OCR edits?", f"Actual trial-0 queries and rankings from the paired experiment ({report['trials']} trials per rate, six queries, seed {report['seed']}). Selection probability is not edit distance: insertions and multi-character confusions change length. Original document index remains clean. These examples illustrate retrieval failures and successes, not real OCR accuracy.", controls, script, body='<pre id="examples"></pre>')

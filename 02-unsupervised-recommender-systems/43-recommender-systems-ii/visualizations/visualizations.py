"""Generate ten offline, CPU-only educational scenes for Recommender Systems II."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import Image, ImageChops
import plotly
import plotly.graph_objects as go
import sklearn
from sklearn.decomposition import TruncatedSVD

from visual_core import (
    build_lab, candidate_analysis, cold_states, exposure_simulation,
    history_matrix, model_metrics, objective_losses, rank_items,
    ranking_contributions, shared_projection, trace_bpr,
)

BLUE, ORANGE, GREEN, GRAY = "#0072B2", "#D55E00", "#009E73", "#697586"
MODULES = ("matrix", "embeddings", "dot-product", "bpr", "objectives", "hybrid",
           "candidates", "metrics", "cold-start", "feedback")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fafcfe", "axes.facecolor": "white",
                     "axes.titleweight": "bold", "savefig.facecolor": "#fafcfe"})


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def encode(value):
    return json.dumps(value, default=serializable, allow_nan=False).replace("</", "<\\/")


def prepare_output(output):
    output = Path(output).resolve()
    for directory in ("gifs", "images", "html", "records"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    return output


def decorate(fig, title, explanation):
    fig.suptitle(title, x=0.04, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.04, 0.025, explanation, ha="left", va="bottom", fontsize=9, color="#344054")
    fig.subplots_adjust(top=0.79, bottom=0.23, left=0.075, right=0.95, wspace=0.35, hspace=0.55)


def png(fig, output, name):
    path = output / "images" / name
    fig.savefig(path, dpi=115)
    return path


def gif(fig, update, frames, output, name, fps=4):
    path = output / "gifs" / name
    animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, repeat=True)
    animation.save(path, writer=PillowWriter(fps=fps), dpi=85)
    return path


def html_scene(output, filename, title, explanation, controls, data, script):
    """Native controls drive Plotly.react with computed states, without CDN or fetch."""
    figure = go.Figure()
    figure.update_layout(template="plotly_white", height=620)
    chart = figure.to_html(full_html=False, include_plotlyjs=True, include_mathjax=False,
                           div_id="lab-plot", config={"responsive": True, "displaylogo": False})
    document = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title>
<style>body{font:16px system-ui,sans-serif;margin:0;background:#f3f6fa;color:#172b45}
main{max-width:1400px;margin:auto;padding:24px}h1{font-size:26px;margin:0 0 12px}
p{max-width:1150px;line-height:1.55}label{display:inline-flex;gap:10px;align-items:center;
margin:8px 24px 8px 0}select,input{font:inherit}select{padding:5px}input{width:190px}
#controls{background:white;border:1px solid #ccd5df;border-radius:8px;padding:12px}
#status{padding:12px;background:#e9f2fa;min-height:48px;line-height:1.5}
#lab-plot{min-height:620px}details{padding:12px;background:white}code{font-size:14px}</style>
</head><body><main><h1>__TITLE__</h1><p>__EXPLANATION__</p><div id="controls">__CONTROLS__</div>
<div id="status" aria-live="polite"></div>__CHART__
<details><summary>How to use this scene</summary><p>Use the controls above to change computed states.
Hover to inspect values. Drag to rotate a 3D scene; use the wheel to zoom.
All libraries and numerical states are embedded in this file; no server is needed.
Changing model dimensionality or training duration requires regeneration with the CLI.</p></details>
</main><script>const D=__DATA__;const graph=document.getElementById('lab-plot');
const el=id=>document.getElementById(id);const base={height:620,plot_bgcolor:'white',
margin:{l:85,r:60,t:55,b:70},font:{family:'Arial',size:14},paper_bgcolor:'white',
colorway:['#0072B2','#D55E00','#009E73','#CC79A7'],uirevision:'stable'};
window.labState={};window.labReady=false;
__SCRIPT__
document.querySelectorAll('#controls select,#controls input').forEach(c=>c.addEventListener('input',render));
render().then(()=>{window.labReady=true});</script></body></html>"""
    for key, value in {"TITLE": html.escape(title), "EXPLANATION": html.escape(explanation),
                       "CONTROLS": controls, "CHART": chart, "DATA": encode(data), "SCRIPT": script}.items():
        document = document.replace(f"__{key}__", value)
    path = output / "html" / filename
    path.write_text(document, encoding="utf-8")
    return path


def user_control(n, initial=0):
    return '<label>User <select id="user">' + ''.join(
        f'<option value="{u}" {"selected" if u == initial else ""}>User {u}</option>' for u in range(n)) + '</select></label>'


def k_control():
    return '<label>Top-K <select id="topk"><option>5</option><option selected>10</option><option>20</option></select></label>'


def matrix_scene(lab, output, frames):
    states = []
    for count in range(1, 16):
        matrix = history_matrix(lab.sequences, len(lab.features), count)
        svd = TruncatedSVD(n_components=lab.factors, n_iter=7, random_state=lab.seed)
        p = svd.fit_transform(matrix)
        approximation = p @ svd.components_
        states.append((matrix, approximation, np.where(matrix == 1, 1 - approximation, np.nan)))
    fig, axes = plt.subplots(1, 3, figsize=(13, 6))
    vmax = max(np.abs(a).max() for _, a, _ in states)
    residual_max = max(np.nanmax(np.abs(r)) for _, _, r in states)
    images = [axes[0].imshow(states[0][0], cmap=ListedColormap(["#e4e9ef", BLUE]), vmin=0, vmax=1, aspect="auto"),
              axes[1].imshow(states[0][1], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto"),
              axes[2].imshow(states[0][2], cmap="RdBu_r", vmin=-residual_max, vmax=residual_max, aspect="auto")]
    for ax, image, title in zip(axes, images, ["Observed labels", "Uncentered SVD scores", "Residual on observed pairs"]):
        ax.set(xlabel="Item ID", ylabel="User ID", title=title)
        bar = fig.colorbar(image, ax=ax, shrink=0.65, pad=0.03)
        if ax is axes[0]:
            bar.set_ticks([0, 1], labels=["Unknown", "Positive"])
        else:
            bar.set_label("Score" if ax is axes[1] else "1 − score")
    status = fig.text(0.04, 0.835, "", fontsize=11)
    decorate(fig, "01  From sparse interactions to low-rank scores",
             f"R (80×100) ≈ P (80×{lab.factors}) Qᵀ ({lab.factors}×100). Each frame refits SVD using one more past round.\n"
             "Unknown is not a confirmed negative. Residuals are masked on unknown pairs; scores are not probabilities. Zero-filling still affects SVD.")
    def update(index):
        matrix, approximation, residual = states[index]
        for image, values in zip(images, [matrix, approximation, residual]):
            image.set_data(values)
        status.set_text(f"Past rounds: {index + 1}/15    Observed: {int(matrix.sum()):,}    Unobserved: {1 - matrix.mean():.1%}    Test round 15 excluded")
    path = gif(fig, update, range(len(states)), output, "01_matrix_factorization.gif", 3)
    update(14)
    still = png(fig, output, "01_matrix_factorization.png")
    plt.close(fig)
    return [path, still]


def embeddings_scene(lab, output, frames):
    p, q = lab.trace["p"], lab.trace["q"]
    pp, qq, projection = shared_projection(p, q)
    # A single final shared basis is reused for every epoch, avoiding rotating PCA frames.
    epoch_states = []
    for state in lab.trace["states"]:
        if lab.factors > 3:
            ps = (state["p"] - projection["mean"]) @ projection["components"].T
            qs = (state["q"] - projection["mean"]) @ projection["components"].T
        else:
            ps, qs, _ = shared_projection(state["p"], state["q"])
        scores = state["p"] @ state["q"].T
        top = [rank_items(row, mask, 20) for row, mask in zip(scores, lab.seen)]
        epoch_states.append({"epoch": state["epoch"], "p": ps, "q": qs,
                             "scores": scores, "norms": np.linalg.norm(state["q"], axis=1), "top": top})
    controls = user_control(len(p)) + k_control() + (
        f'<label>BPR epoch <input id="epoch" type="range" min="0" max="{lab.epochs}" value="{lab.epochs}" step="1"></label>')
    script = """async function render(){const u=+el('user').value,k=+el('topk').value,e=+el('epoch').value;
const s=D.states[e],ids=s.top[u].slice(0,k),axis=j=>s.q.map(v=>v[j]),pa=j=>s.p.map(v=>v[j]);
const itemInfo=s.q.map((v,i)=>[i,D.genres[i],s.scores[u][i],s.norms[i],D.seen[u][i]?'seen':'eligible']);
const traces=[{type:'scatter3d',mode:'markers',name:'Items',x:axis(0),y:axis(1),z:axis(2),
marker:{size:4,color:'#D55E00',symbol:'square',opacity:.55},customdata:itemInfo,
hovertemplate:'Item %{customdata[0]} · genre %{customdata[1]}<br>full dot = %{customdata[2]:.4f}<br>norm = %{customdata[3]:.4f}<br>%{customdata[4]}<extra></extra>'},
{type:'scatter3d',mode:'markers',name:'Users',x:pa(0),y:pa(1),z:pa(2),marker:{size:3,color:'#0072B2',symbol:'circle'},text:s.p.map((_,i)=>'User '+i),hovertemplate:'%{text}<extra></extra>'},
{type:'scatter3d',mode:'markers',name:'Selected user '+u,x:[s.p[u][0]],y:[s.p[u][1]],z:[s.p[u][2]],marker:{size:9,color:'#000',symbol:'diamond'}},
{type:'scatter3d',mode:'markers+text',name:'Eligible Top-'+k,x:ids.map(i=>s.q[i][0]),y:ids.map(i=>s.q[i][1]),z:ids.map(i=>s.q[i][2]),text:ids.map(i=>'i'+i),marker:{size:7,color:'#009E73',symbol:'square'},customdata:ids.map(i=>s.scores[u][i]),hovertemplate:'%{text}<br>full dot = %{customdata:.4f}<extra></extra>'}];
const links=[[],[],[]];ids.forEach(i=>{for(let j=0;j<3;j++)links[j].push(s.p[u][j],s.q[i][j],null)});
traces.push({type:'scatter3d',mode:'lines',name:'Compatibility links (not distance)',x:links[0],y:links[1],z:links[2],line:{color:'#009E73',width:2},hoverinfo:'skip'});
window.labState={user:u,k,epoch:e,ids,scores:ids.map(i=>s.scores[u][i])};
el('status').textContent='Epoch '+e+' · '+D.projection+' · Scores use full '+D.factors+'D factors. Norms affect dot products. Links show ranked compatibility, not nearest Euclidean neighbors.';
await Plotly.react(graph,traces,{...base,title:{text:'Shared coordinates · selected user '+u+' · epoch '+e},scene:{xaxis:{title:{text:'Latent coordinate 1'}},yaxis:{title:{text:'Latent coordinate 2'}},zaxis:{title:{text:'Latent coordinate 3'}},aspectmode:'cube'}});}
"""
    path = html_scene(output, "02_embeddings_3d.html", "02  Embeddings evolve into compatibility scores",
                      "Change user, training epoch and Top-K. Hover on items for exact scores. A shared PCA basis is used only above three factors; centering/projection can distort original scores and distances.",
                      controls, {"states": epoch_states, "genres": lab.features.argmax(axis=1), "seen": lab.seen,
                                 "projection": projection["mode"], "factors": lab.factors}, script)
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    top = rank_items(lab.scores["BPR"][0], lab.seen[0], 10)
    ax.scatter(*qq.T, c=ORANGE, alpha=0.45, s=17, label="Items")
    ax.scatter(*pp.T, c=BLUE, alpha=0.5, s=12, label="Users")
    ax.scatter(*pp[0], c="black", marker="D", s=65, label="Selected user 0")
    ax.scatter(*qq[top].T, c=GREEN, marker="s", s=38, label="Eligible top 10")
    for i in top:
        ax.plot(*np.stack([pp[0], qq[i]]).T, c=GREEN, alpha=0.3)
    ax.set(xlabel="Coordinate 1", ylabel="Coordinate 2", zlabel="Coordinate 3")
    ax.legend(loc="upper left", fontsize=9)
    decorate(fig, "02  Shared embedding space — computed companion preview",
             "Dots are fitted BPR factors; lines connect user 0 to eligible high-dot-product items.\n"
             + projection["mode"] + ". Full scores are shown in the interactive HTML; these are not probability estimates.")
    preview = png(fig, output, "02_embeddings_preview.png")
    plt.close(fig)
    return [path, preview]


def dot_product_scene(lab, output, frames):
    # Separate exact 2D examples expose norm versus angle without PCA distortion.
    items = np.array([[2.5, 0.2], [0.5, 0.5], [-0.5, 1.4], [1.0, -1.5], [-1, -0.5]])
    users = np.array([[1, 0.6], [0.2, 1.2], [-1, 0.5], [1.5, -0.8]])
    states = []
    for p in users:
        dot = items @ p
        cosine = dot / (np.linalg.norm(items, axis=1) * np.linalg.norm(p))
        angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        states.append({"user": p, "dot": dot, "cosine": cosine, "angle": angle,
                       "distance": np.linalg.norm(items - p, axis=1),
                       "dot_order": np.argsort(-dot, kind="stable"),
                       "cos_order": np.argsort(-cosine, kind="stable")})
    controls = '<label>User direction <select id="user">' + ''.join(
        f'<option value="{i}">Example {i}: {p.tolist()}</option>' for i, p in enumerate(users)) + '</select></label>'
    controls += '<label>Inspect item <select id="item">' + ''.join(f'<option>{i}</option>' for i in range(5)) + '</select></label>'
    script = """async function render(){const u=+el('user').value,item=+el('item').value,s=D.states[u],tr=[];
D.items.forEach((v,i)=>tr.push({type:'scatter',mode:'lines+markers+text',x:[0,v[0]],y:[0,v[1]],text:['','i'+i],textposition:'top center',name:'Item '+i,
customdata:[[i,s.dot[i],s.cosine[i],s.angle[i],s.distance[i]],[i,s.dot[i],s.cosine[i],s.angle[i],s.distance[i]]],
hovertemplate:'i%{customdata[0]}<br>dot %{customdata[1]:.3f}<br>cos %{customdata[2]:.3f}<br>angle %{customdata[3]:.1f}°<br>Euclidean distance %{customdata[4]:.3f}<extra></extra>'}));
tr.push({type:'scatter',mode:'lines+markers',name:'User vector',x:[0,s.user[0]],y:[0,s.user[1]],line:{color:'black',width:6}});
const start=Math.atan2(s.user[1],s.user[0]),end=Math.atan2(D.items[item][1],D.items[item][0]);
let delta=end-start;while(delta>Math.PI)delta-=2*Math.PI;while(delta< -Math.PI)delta+=2*Math.PI;
const angles=Array.from({length:30},(_,i)=>start+delta*i/29);
tr.push({type:'scatter',mode:'lines',name:'Selected angle',x:angles.map(a=>.55*Math.cos(a)),y:angles.map(a=>.55*Math.sin(a)),line:{color:'black',dash:'dot',width:3},hoverinfo:'skip'});
tr.push({type:'bar',x:s.dot_order.map(i=>'i'+i),y:s.dot_order.map(i=>s.dot[i]),name:'Dot-product ranking',xaxis:'x2',yaxis:'y2',marker:{color:'#0072B2'}});
tr.push({type:'bar',x:s.cos_order.map(i=>'i'+i),y:s.cos_order.map(i=>s.cosine[i]),name:'Cosine ranking',xaxis:'x3',yaxis:'y3',marker:{color:'#D55E00'}});
el('status').textContent='Exact 2D illustration (separate from the 80-user experiment). Item '+item+': ||p||='+Math.hypot(...s.user).toFixed(3)+', ||q||='+Math.hypot(...D.items[item]).toFixed(3)+', angle='+s.angle[item].toFixed(1)+'°. p·q = ||p|| ||q|| cos(angle) = '+s.dot[item].toFixed(3)+'. Dot order: '+s.dot_order.join(', ')+'; cosine order: '+s.cos_order.join(', ')+'. Dotted arc marks the selected angle; proximity need not match dot-product rank.';
window.labState={user:u,item,dot:s.dot,cosine:s.cosine,dotOrder:s.dot_order,cosineOrder:s.cos_order};
await Plotly.react(graph,tr,{...base,title:{text:'Same item vectors; changing user direction changes ranking'},showlegend:false,
xaxis:{domain:[0,.46],title:{text:'Coordinate 1'},range:[-2,3]},yaxis:{domain:[0,1],title:{text:'Coordinate 2'},range:[-2,2],scaleanchor:'x'},
xaxis2:{domain:[.6,1],anchor:'y2',title:{text:'Items ordered by dot product'}},yaxis2:{domain:[.58,1],anchor:'x2',title:{text:'Dot product'}},
xaxis3:{domain:[.6,1],anchor:'y3',title:{text:'Items ordered by cosine'}},yaxis3:{domain:[0,.4],anchor:'x3',title:{text:'Cosine'},range:[-1.1,1.1]}});}
"""
    return [html_scene(output, "03_dot_product.html", "03  Direction, magnitude and ranking are different",
                       "Select an illustrative user vector. The five item vectors stay fixed; exact angles, norms, distances, dot products and cosine rankings change. This geometry example uses no relevance labels or fitted model.",
                       controls, {"items": items, "states": states}, script)]


def bpr_scene(lab, output, frames):
    trace = trace_bpr(lab.matrix, factors=2, epochs=lab.epochs, seed=lab.seed, frames=frames)
    fig, axes = plt.subplots(1, 3, figsize=(13, 6))
    status = fig.text(0.04, 0.84, "", fontsize=10)
    bound = max(np.abs(s["after"]).max() for s in trace["steps"]) * 1.25
    h = trace["history"]
    def update(index):
        state = trace["steps"][index]
        for ax in axes:
            ax.clear()
        for j, (name, color) in enumerate(zip(["User", "Positive", "Sampled unknown"], [BLUE, GREEN, ORANGE])):
            before, after = state["before"][j], state["after"][j]
            axes[0].plot([0, before[0]], [0, before[1]], "--", color=color, alpha=0.45)
            axes[0].arrow(0, 0, *after, width=0.007, head_width=0.06, color=color,
                          length_includes_head=True, label=name)
        axes[0].set(xlim=(-bound, bound), ylim=(-bound, bound), xlabel="Factor 1", ylabel="Factor 2", title="Actual 2D SGD step")
        axes[0].legend(fontsize=8, loc="lower left")
        completed_epochs = state["step"] // int(lab.matrix.sum())
        past = [row for row in h if row["epoch"] <= completed_epochs]
        axes[1].plot([r["epoch"] for r in h], [r["pair_loss"] for r in h], c=GRAY, alpha=0.3)
        axes[1].plot([r["epoch"] for r in past], [r["pair_loss"] for r in past], c=BLUE, marker=".")
        axes[1].set(xlabel="Completed epoch", ylabel="Mean softplus(−margin)", title="All training positive/unknown pairs")
        for key, name, color in [("positive_score", "Fixed positive", GREEN), ("unobserved_score", "Fixed unknown", ORANGE)]:
            axes[2].plot([r["epoch"] for r in h], [r[key] for r in h], c=color, alpha=0.2)
            axes[2].plot([r["epoch"] for r in past], [r[key] for r in past], c=color, label=name)
        axes[2].set(xlabel="Completed epoch", ylabel="Full 2D dot-product score", title=f"Fixed probe (u,i,j) = {trace['tracked_pair']}")
        axes[2].legend(fontsize=8)
        a, b = state["after_scores"]
        status.set_text(f"Step {state['step']:,} · epoch {state['epoch']} · (u,i,j)=({state['user']},{state['positive']},{state['unobserved']})\n"
                        f"After update: s+={a:.4f}, s?={b:.4f}, Δ={a-b:.4f}, −log σ(Δ)={state['pair_loss']:.4f}")
    decorate(fig, "04  BPR updates three vectors using their old values",
             "Dashed = before; arrows = after. Recorded steps span a real, separate 2-factor fit on the same past-only matrix.\n"
             "g=σ(−Δ); p moves toward q+−q?; item updates use old p. L2 shrinkage is included in SGD, excluded from plotted data loss.\n"
             "Pair loss and scores can fluctuate; each shown triplet is different. Unknown samples are not confirmed dislikes.")
    path = gif(fig, update, range(len(trace["steps"])), output, "04_bpr_training.gif", 4)
    update(len(trace["steps"]) - 1)
    still = png(fig, output, "04_bpr_loss.png")
    plt.close(fig)
    return [path, still]


def objectives_scene(lab, output, frames):
    labels = np.array([1, 0, 1, 0, 0])
    logits = np.array([0.3, 1.2, 0.8, -0.5, 0.1])
    loss = objective_losses(logits, labels)
    fig, axes = plt.subplots(1, 3, figsize=(13, 6))
    ids = np.arange(5)
    colors = [GREEN if label else GRAY for label in labels]
    axes[0].bar(ids, logits, color=colors)
    axes[0].set(xlabel="Candidate item ID", ylabel="Logit (not probability)", title=f"Pointwise BCE = {loss['pointwise']:.4f}")
    axes[0].set_xticks(ids)
    pairs = [(i, j) for i in ids[labels == 1] for j in ids[labels == 0]]
    pair_losses = [np.logaddexp(0, -(logits[i] - logits[j])) for i, j in pairs]
    axes[1].bar(np.arange(len(pairs)), pair_losses, color=BLUE)
    axes[1].set_xticks(np.arange(len(pairs)), [f"{i}>{j}" for i, j in pairs])
    axes[1].set(xlabel="Judged positive > negative", ylabel="softplus(−score difference)", title=f"Pairwise mean = {loss['pairwise']:.4f}")
    target = np.exp(labels - np.logaddexp.reduce(labels))
    predicted = np.exp(logits - np.logaddexp.reduce(logits))
    axes[2].bar(ids - .18, target, .36, color=GREEN, label="softmax(judgments)")
    axes[2].bar(ids + .18, predicted, .36, color=ORANGE, label="softmax(logits)")
    axes[2].set(xlabel="Candidate item ID", ylabel="Top-one list probability", title=f"ListNet top-one CE = {loss['listwise']:.4f}")
    axes[2].set_xticks(ids)
    axes[2].legend(fontsize=8)
    decorate(fig, "05  Same five candidates, three different learning signals",
             "Toy complete judgments: green=positive, gray=judged negative (unlike unknown entries in the main experiment).\n"
             "BCE evaluates individual labels; pairwise softplus compares pairs; top-one ListNet CE compares distributions over the list.\n"
             "These are computed differentiable losses, not trained models in this scene. NDCG is an evaluation metric, not this listwise loss.")
    path = png(fig, output, "05_ranking_objectives.png")
    plt.close(fig)
    return [path]


def hybrid_scene(lab, output, frames):
    states = []
    for user in range(len(lab.matrix)):
        rows = []
        for alpha in np.linspace(0, 1, 11):
            scores = lab.hybrid(float(alpha))[user]
            ids = rank_items(scores, lab.seen[user], 20)
            rows.append({"ids": ids, "scores": scores[ids], "cf": alpha * lab.cf_z[user, ids],
                         "content": (1 - alpha) * lab.content_z[user, ids]})
        states.append(rows)
    controls = user_control(len(lab.matrix)) + k_control() + '<label>CF weight α <input id="alpha" type="range" min="0" max="10" step="1" value="5"></label>'
    script = """async function render(){const u=+el('user').value,k=+el('topk').value,a=+el('alpha').value,s=D.states[u][a],ids=s.ids.slice(0,k),names=ids.map(i=>'Item '+i);
const tr=[{type:'bar',orientation:'h',name:'α × z(CF)',x:s.cf.slice(0,k),y:names,marker:{color:'#0072B2'}},
{type:'bar',orientation:'h',name:'(1−α) × z(content)',x:s.content.slice(0,k),y:names,marker:{color:'#D55E00'}},
{type:'scatter',mode:'markers',name:'Hybrid sum',x:s.scores.slice(0,k),y:names,marker:{color:'#111',size:9,symbol:'diamond'}}];
el('status').textContent='User '+u+' · α='+a/10+' · Top-'+k+'. Per-user z-scores use the complete eligible catalog before retrieval. Constant components map to zero. α is an educational control, not an optimized production weight. Markers show the sum; negative components extend left.';
window.labState={user:u,k,alpha:a/10,ids,scores:s.scores.slice(0,k),cf:s.cf.slice(0,k),content:s.content.slice(0,k)};
await Plotly.react(graph,tr,{...base,title:{text:'Weighted component contributions and changing item order',x:.5,xanchor:'center',y:.98,yanchor:'top'},margin:{l:110,r:60,t:115,b:80},barmode:'relative',xaxis:{title:{text:'Normalized contribution / fused score'}},yaxis:{title:{text:'Eligible ranked items'},categoryorder:'array',categoryarray:names,autorange:'reversed'},legend:{orientation:'h',y:1.02,yanchor:'bottom',x:0,xanchor:'left'}});}
"""
    path = html_scene(output, "06_hybrid_ranking.html", "06  Hybrid ranking is a policy over two signals",
                      "Move α or choose another user/K. Bars show signed component contributions; black diamonds show their sum. Items are sorted by the actual sum and consumed items are excluded. Scaling is not probability calibration.",
                      controls, {"states": states}, script)
    fig, ax = plt.subplots(figsize=(11, 7))
    state = states[0][5]
    y = np.arange(10)
    ax.barh(y - .18, state["cf"][:10], height=.34, color=BLUE, label="0.5 × z(CF)")
    ax.barh(y + .18, state["content"][:10], height=.34, color=ORANGE, label="0.5 × z(content)")
    ax.scatter(state["scores"][:10], y, c="black", marker="D", label="Sum", zorder=5)
    ax.set_yticks(y, [f"Item {i}" for i in state["ids"][:10]])
    ax.invert_yaxis()
    ax.set(xlabel="Contribution / fused score", ylabel="Eligible item rank", title="User 0 · α=0.5 · Top-10")
    ax.legend(loc="lower right", fontsize=9)
    decorate(fig, "06  Inspect each source before trusting their combination",
             "Computed from the same BPR/content scores as the interactive slider; weights here are illustrative.\n"
             "Per-user eligible-catalog z-scores are fixed across α. A negative component can lower an otherwise strong candidate.")
    preview = png(fig, output, "06_hybrid_preview.png")
    plt.close(fig)
    return [path, preview]


def candidates_scene(lab, output, frames):
    analysis = {str(k): candidate_analysis(lab, k=k) for k in (5, 10, 20)}
    controls = user_control(len(lab.matrix)) + k_control() + '<label>Per-source budget <select id="budget">' + ''.join(
        f'<option {"selected" if b == 10 else ""}>{b}</option>' for b in (5, 10, 20, 50)) + '</select></label>'
    script = """async function render(){const u=+el('user').value,k=el('topk').value,b=+el('budget').value,
s=D.analysis[k].find(r=>r.budget===b),pool=s.pools[u],full=s.full_rankings[u],ranked=s.rankings[u],t=D.targets[u];
const ys=['Full eligible catalog','BPR retrieval','Content retrieval','Candidate union','Ranked Top-'+k];
const cf=D.cf[u].slice(0,b),content=D.content[u].slice(0,b),counts=[85,cf.length,content.length,pool.length,ranked.length];
const tr=[{type:'bar',orientation:'h',x:counts,y:ys,text:counts,textposition:'auto',name:'Stage sizes',marker:{color:['#697586','#0072B2','#D55E00','#CC79A7','#009E73']},xaxis:'x',yaxis:'y'},
{type:'bar',x:['Candidate recall','Restricted Recall@'+k,'Full Recall@'+k,'Restricted NDCG@'+k,'Full NDCG@'+k],y:[s.candidate_recall,s.metrics.recall,s.full_metrics.recall,s.metrics.ndcg,s.full_metrics.ndcg],name:'All 80 users',marker:{color:'#0072B2'},xaxis:'x2',yaxis:'y2'}];
el('status').textContent='User '+u+' held-out target '+t+': '+(pool.includes(t)?'retrieved':'EXCLUDED before ranking')+'. Full order: '+full.join(', ')+'. Restricted order: '+ranked.join(', ')+'. Union IDs: '+pool.join(', ')+'. All-user excluded target owners: '+s.excluded_users.join(', ')+'. Metrics retain all 80 users.';
window.labState={user:u,k:+k,budget:b,target:t,pool,ranked,full,candidateRecall:s.candidate_recall,recall:s.metrics.recall,ndcg:s.metrics.ndcg};
await Plotly.react(graph,tr,{...base,title:{text:'Full catalog → source retrieval → union → ranking → Top-K'},showlegend:false,margin:{l:175,r:70,t:65,b:110},
xaxis:{domain:[0,.43],title:{text:'Actual number of items'},range:[0,100]},yaxis:{domain:[0,1],autorange:'reversed'},
xaxis2:{domain:[.58,1],anchor:'y2',tickangle:-25,title:{text:'Aggregate evaluation'}},yaxis2:{domain:[0,1],anchor:'x2',range:[0,1.05],title:{text:'Metric value'}}});}
"""
    cf = [rank_items(row, seen, 50) for row, seen in zip(lab.scores["BPR"], lab.seen)]
    content = [rank_items(row, seen, 50) for row, seen in zip(lab.scores["Content"], lab.seen)]
    path = html_scene(output, "07_candidate_pipeline.html", "07  Retrieval sets the ranker's opportunity",
                      "Choose budget, user and K. Source lists use past-only scores, then their union is ranked with fixed α=0.5. Held-out targets are used only to mark evaluation hits/misses after retrieval. More candidates cannot lower candidate recall; final ranking quality can move either way.",
                      controls, {"analysis": analysis, "targets": lab.targets, "cf": cf, "content": content}, script)
    fig, ax = plt.subplots(figsize=(11, 6))
    rows = analysis["10"]
    budgets = [r["budget"] for r in rows]
    ax.plot(budgets, [r["candidate_recall"] for r in rows], "o-", c=BLUE, label="Candidate recall")
    ax.plot(budgets, [r["metrics"]["recall"] for r in rows], "s-", c=GREEN, label="Restricted Recall@10")
    ax.plot(budgets, [r["metrics"]["ndcg"] for r in rows], "^-", c=ORANGE, label="Restricted NDCG@10")
    ax.axhline(rows[0]["full_metrics"]["recall"], c=GREEN, ls="--", label="Full-catalog Recall@10")
    ax.set(xlabel="Candidates per source (BPR + content union)", ylabel="Macro metric over 80 users", ylim=(0, 1), xticks=budgets)
    ax.legend(fontsize=9)
    decorate(fig, "07  A ranker cannot recover a discarded target",
             "Same target set and denominator at every budget; α=0.5 and score normalization stay fixed.\n"
             "Union size is at most twice the source budget; this exhaustive small-catalog computation is not a serving-speed benchmark.")
    preview = png(fig, output, "07_candidate_recall.png")
    plt.close(fig)
    return [path, preview]


def metrics_scene(lab, output, frames):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    status = fig.text(0.04, 0.84, "", fontsize=11)
    positions = np.arange(1, 11)
    def update(index):
        for ax in axes:
            ax.clear()
        rank = 8 - index
        rel = np.zeros(10)
        rel[rank - 1] = 1
        values = ranking_contributions(rel)
        axes[0].bar(positions, rel, color=[GREEN if r else GRAY for r in rel])
        for pos, discount in zip(positions, values["discounts"]):
            axes[0].text(pos, .08, f"{discount:.2f}", ha="center", color="white", fontsize=8, rotation=90,
                         bbox={"facecolor": GRAY, "edgecolor": "none", "pad": 2})
        axes[0].set(xlabel="Ranking position (numbers show 1/log₂(rank+1))", ylabel="Judged relevance", ylim=(0, 1.2), xticks=positions, title=f"One known relevant item at rank {rank}")
        axes[1].step(positions, values["dcg"], where="mid", c=BLUE, label="Cumulative DCG@K")
        axes[1].step(positions, values["ndcg"], where="mid", c=ORANGE, ls="--", label="NDCG@K (IDCG=1)")
        axes[1].step(positions, np.cumsum(rel), where="mid", c=GREEN, alpha=.7, label="Recall@K = hit rate")
        axes[1].set(xlabel="K", ylabel="Metric value", ylim=(-.03, 1.1), xticks=positions)
        axes[1].legend(fontsize=8)
        status.set_text(f"Relevant item moves from rank 8 toward rank 2 · DCG@10=NDCG@10={values['dcg'][-1]:.4f} · Recall@10=1")
    decorate(fig, "08  The same hit matters more near the top",
             "Toy judged list: green=relevant, gray=judged irrelevant. These labels are illustrative, not claims about unknown catalog items.\n"
             "Binary gain=2ʳᵉˡ−1; DCG sums discounted gains. This one-positive list has ideal DCG=1; recall alone ignores hit position.")
    path = gif(fig, update, range(7), output, "08_ranking_metrics.gif", 2)
    plt.close(fig)
    metrics = model_metrics(lab)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for name, rows in metrics.items():
        ks = [int(k) for k in rows]
        axes[0].plot(ks, [r["recall"] for r in rows.values()], "o-", label=name)
        axes[1].plot(ks, [r["ndcg"] for r in rows.values()], "o-", label=name)
    for ax, metric in zip(axes, ["Recall@K (= hit rate)", "NDCG@K"]):
        ax.set(xlabel="K", ylabel=f"Mean {metric}", ylim=(0, 1), xticks=[1, 3, 5, 10, 20])
        ax.legend(fontsize=8)
    decorate(fig, "08  Compare actual models on the same held-out targets",
             f"Synthetic experiment: 80 users, 100 items, {lab.factors} factors, {lab.epochs} epochs, seed {lab.seed}. One target/user; 85 eligible items/user.\n"
             "All consumed items excluded. Hybrid α=0.5 is illustrative. These aggregate test metrics are separate from the toy animation above.")
    preview = png(fig, output, "08_model_comparison.png")
    plt.close(fig)
    return [path, preview]


def cold_scene(lab, output, frames):
    states = cold_states(lab)
    fig, axes = plt.subplots(1, 3, figsize=(13, 6))
    status = fig.text(0.04, 0.84, "", fontsize=10)
    def update(index):
        state = states[index]
        for ax in axes:
            ax.clear()
        for key, name, color in [("cf", "Collaborative", BLUE), ("content", "Content", ORANGE)]:
            values = state[key][~state["seen"]]
            axes[0].plot(np.sort(values), color=color, label=name)
        axes[0].set(xlabel="Sorted eligible score index (different scales)", ylabel="Raw component score", title="No history → uninformative scores")
        axes[0].legend(fontsize=8)
        top = state["top"]
        axes[1].barh(np.arange(len(top)), state["hybrid"][top], color=[ORANGE if i == state["new_item"] else BLUE for i in top])
        axes[1].set_yticks(np.arange(len(top)), [f"Item {i}" for i in top])
        axes[1].invert_yaxis()
        axes[1].set(xlabel="Hybrid α=0.5 score", ylabel="Top-10 eligible items", title="BPR ∪ content retrieval (20/source)")
        axes[2].plot([s["count"] for s in states], [np.std(s["cf"][~s["seen"]]) for s in states], "o-", color=BLUE, label="CF score std")
        axes[2].plot([s["count"] for s in states], [np.std(s["content"][~s["seen"]]) for s in states], "s-", color=ORANGE, label="Content score std")
        axes[2].axvline(state["count"], color="black", ls="--")
        axes[2].set(xlabel="New-user interactions observed", ylabel="Score variation (not quality)", title="Sparsity changes discrimination")
        axes[2].legend(fontsize=8)
        status.set_text(f"New user history: {state['count']}/15 · Unknown pairs: {1 - state['count']/101:.1%} · Metadata-only item {state['new_item']}\n"
                        f"CF fallback={state['cf'][-1]:.3f} · content={state['content'][-1]:.3f} · in union={state['new_item_in_pool']} · ties use item ID")
    decorate(fig, "09  Cold start needs information and an admission path",
             "New user excluded from item-model fitting; fold-in updates only their vector as past interactions arrive. New item has a zero factor.\n"
             "Content source is included in the union. With no user history, both signals are zero and ordering is arbitrary ID tie-breaks.\n"
             "This separate onboarding probe measures score/coverage changes, not cold-start accuracy. Hybrid does not guarantee new-item inclusion.")
    path = gif(fig, update, range(len(states)), output, "09_cold_start.gif", 1)
    update(len(states) - 1)
    preview = png(fig, output, "09_cold_start.png")
    plt.close(fig)
    return [path, preview]


def feedback_scene(lab, output, frames):
    states = exposure_simulation(lab.seed, rounds=min(frames, 24))
    fig, axes = plt.subplots(1, 3, figsize=(13, 6))
    status = fig.text(0.04, 0.84, "", fontsize=11)
    items = np.arange(len(states[0]["true_ctr"]))
    def update(index):
        state = states[index]
        for ax in axes:
            ax.clear()
        for policy, name, color in [(0, "Uniform exposure", BLUE), (1, "Click-popularity policy", ORANGE)]:
            offset = -.18 if policy == 0 else .18
            axes[0].bar(items + offset, state["exposure"][policy], .36, color=color, label=name)
            axes[1].bar(items + offset, state["clicks"][policy], .36, color=color, label=name)
            axes[2].scatter(state["true_ctr"], state["observed_ctr"][policy], color=color, label=name)
        axes[0].set(xlabel="Item ID", ylabel="Cumulative impressions", title="Exposure is chosen by policy", xticks=items)
        axes[1].set(xlabel="Item ID", ylabel="Observed clicks", title="Clicks combine exposure + preference", xticks=items)
        axes[2].plot([0, .5], [0, .5], "--", c=GRAY)
        axes[2].set(xlabel="Hidden stationary click probability", ylabel="Observed clicks / impressions", xlim=(0, .5), ylim=(0, .5), title="Rates and uncertainty differ")
        axes[0].legend(fontsize=8)
        shares = state["exposure"][1] / state["exposure"][1].sum()
        status.set_text(f"Simulation round {index}/{len(states)-1} · Equal initial 10 impressions + 1 pseudoclick/item/policy\n"
                        f"Popularity policy: 10% uniform exploration + 90% ∝ cumulative clicks² · exposure concentration Σshare²={np.sum(shares**2):.3f}")
    decorate(fig, "10  What gets exposed changes what can become training data",
             "Two simulated policies share hidden stationary click probabilities; each draws 120 impressions/round, then Bernoulli clicks.\n"
             "The popularity policy updates from cumulative clicks. Initial pseudocounts are declared, not measured behavior.\n"
             "Conceptual feedback-loop simulation, not evidence about every recommender. Click-only evaluation omits unseen preferences and exposure opportunities.")
    path = gif(fig, update, range(len(states)), output, "10_feedback_loop.gif", 3)
    plt.close(fig)
    return [path]


GENERATORS = dict(zip(MODULES, [matrix_scene, embeddings_scene, dot_product_scene, bpr_scene,
                               objectives_scene, hybrid_scene, candidates_scene, metrics_scene,
                               cold_scene, feedback_scene]))


def inspect_artifact(path):
    result = {"path": str(path), "bytes": path.stat().st_size,
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if not result["bytes"]:
        raise ValueError(f"empty artifact: {path}")
    if path.suffix in (".png", ".gif"):
        with Image.open(path) as image:
            result["size"] = list(image.size)
            result["frames"] = getattr(image, "n_frames", 1)
            image.seek(0)
            first = image.convert("RGB")
            changed = False
            for index in range(1, result["frames"]):
                image.seek(index)
                if ImageChops.difference(first, image.convert("RGB")).getbbox():
                    changed = True
            result["frame_changes"] = changed
            if path.suffix == ".gif" and (result["frames"] < 2 or not changed):
                raise ValueError(f"animation has no meaningful frame changes: {path}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--all", action="store_true")
    choice.add_argument("--module", choices=MODULES)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--factors", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="relative paths resolve against this script's directory")
    args = parser.parse_args()
    if not 8 <= args.frames <= 60 or not 1 <= args.epochs <= 100:
        parser.error("require 8 <= frames <= 60 and 1 <= epochs <= 100 for bounded CPU rendering")
    output = args.output_dir
    if not output.is_absolute():
        output = Path(__file__).resolve().parent / output
    output = prepare_output(output)
    start = time.perf_counter()
    lab = build_lab(args.seed, args.factors, args.epochs, args.frames)
    artifacts, modules = [], list(MODULES) if args.all else [args.module]
    for name in modules:
        tick = time.perf_counter()
        artifacts += GENERATORS[name](lab, output, args.frames)
        print(f"{name}: generated ({time.perf_counter()-tick:.1f}s)", flush=True)
    inspection = [inspect_artifact(path) for path in artifacts]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    result = {"executed_at_utc": datetime.now(timezone.utc).isoformat(),
              "hypothesis": "Make real training, score fusion and retrieval truncation inspectable; quantify candidate exclusion under fixed illustrative fusion.",
              "configuration": {"seed": args.seed, "factors": args.factors, "epochs": args.epochs,
                                "frames": args.frames, "users": 80, "items": 100, "genres": 8,
                                "fit_rounds": [0, 14], "test_round": 15, "alpha": 0.5,
                                "lr": 0.05, "reg": 0.01, "modules": modules,
                                "normalization": "per-user complete eligible catalog z-score; constant maps to zero"},
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__,
                           "plotly": plotly.__version__, "pillow": Image.__version__},
              "test_metrics": model_metrics(lab),
              "candidate_analysis": candidate_analysis(lab), "artifacts": inspection,
              "elapsed_seconds": time.perf_counter() - start,
              "interpretation_candidate": "Compare candidate recall to final recall and inspect source contributions; neither smooth loss curves nor hybrid fusion establish general recommendation gains.",
              "interpretation_status": "author review required",
              "limitations": "Single genre-driven synthetic seed, one final positive/user, no main-dataset impressions or confirmed negatives. Fixed alpha=0.5 and factors differ from historical study. Toy objective/dot-product/metric scenes and separate cold/exposure simulations are not aggregated test evidence."}
    record = output / "records" / f"visual_run_{stamp}.json"
    record.write_text(encode(result) + "\n", encoding="utf-8")
    print(json.dumps({"modules": modules, "artifacts": len(artifacts),
                      "seconds": round(result["elapsed_seconds"], 2),
                      "test_at_10": {name: rows["10"] for name, rows in result["test_metrics"].items()},
                      "record": str(record)}, indent=2))


if __name__ == "__main__":
    main()

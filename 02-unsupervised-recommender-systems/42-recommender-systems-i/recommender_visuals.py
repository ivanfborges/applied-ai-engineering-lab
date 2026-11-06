"""Ten bounded, offline learning scenes, rendered from numerical calculations."""

from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image
import plotly.graph_objects as go

import recommender_visual_core as core
from factorization import fit_factorization
from from_scratch import UserNeighborhood, overlap_pearson


BLUE, ORANGE, GREEN, GRAY, RED = "#2563a8", "#d97706", "#147d64", "#dce3eb", "#b64040"
COLORS = [BLUE, ORANGE, GREEN]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.titleweight": "bold"})

SCENES = {
    1: ("User–item matrix", "Which cells contain evidence?",
        "A sparse explicit matrix distinguishes low ratings from missing observations.",
        "The generator has fully known synthetic preferences; only the training mask is shown."),
    2: ("Explicit vs. implicit", "Does a click mean a five-star preference?",
        "Preference targets and confidence weights encode different assumptions.",
        "Counts are fictional event logs; confidence is not a calibrated probability."),
    3: ("User similarity", "Which agreement is actually supported by co-ratings?",
        "Cosine normalization, overlap-local centering and shrinkage produce different similarities.",
        "The similarity graph uses a fixed threshold and circular layout, not learned geometry."),
    4: ("Recommendation arithmetic", "Where does one score come from?",
        "Positive neighbors contribute normalized rating deviations to a user baseline.",
        "The animation reveals a frozen calculation, not incremental model training."),
    5: ("Learned factors in 3D", "What does a dot product predict?",
        "A masked observed-rating objective can learn three-dimensional compatibility factors.",
        "Axes are not named preferences; dot-product order is not Euclidean proximity."),
    6: ("Matrix reconstruction", "Which errors can we actually verify?",
        "Training errors exist only on observed entries; predictions fill every pair.",
        "Unobserved predictions are estimates; dimension comparisons measure training fit only."),
    7: ("Cold start and sparsity", "When does a new user acquire collaborative support?",
        "Adding onboarding observations can enable supported neighborhoods; thinning changes support.",
        "Onboarding copies a synthetic training profile; sparsity repeats are not real-user cohorts."),
    8: ("Evaluation", "Is rating accuracy the same as a good ranking?",
        "Ordering changes NDCG even when Precision@K and Recall@K stay fixed.",
        "Ranking uses known synthetic catalog preferences, unlike ordinary incomplete click logs."),
    9: ("Exposure and feedback", "Who had an opportunity to click?",
        "Policy-dependent exposure can concentrate clicks even with equal item click propensities.",
        "One seeded toy trajectory is not a causal estimate for a deployed recommender."),
    10: ("Recommendation pipeline", "Where do models, fallbacks and feedback fit?",
         "Retrieval, ranking, evaluation and monitoring have distinct responsibilities.",
         "This is an architectural teaching diagram; no service is deployed."),
}


def finish(fig, title, note):
    fig.suptitle(title, x=.05, y=.98, ha="left", fontsize=17, fontweight="bold", color="#17304a")
    fig.text(.05, .025, note, fontsize=9, color="#475569", va="bottom")
    fig.tight_layout(rect=(.02, .09, .98, .91))
    return fig


def png(fig, out, name):
    fig.savefig(out / name, dpi=140, facecolor="white")
    plt.close(fig)
    return name


def gif(draw, states, out, name, duration=750):
    frames = []
    for state in states:
        fig = draw(state)
        fig.canvas.draw()
        frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        frames.append(frame)
        plt.close(fig)
    frames[0].save(out / name, save_all=True, append_images=frames[1:], duration=duration, loop=0)
    for frame in frames:
        frame.close()
    return name


def matrix(ax, values, title, labels=True, vmin=1, vmax=5, cmap="viridis", annotate=True):
    palette = plt.get_cmap(cmap).with_extremes(bad=GRAY)
    image = ax.imshow(np.ma.masked_invalid(values), cmap=palette, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Items"); ax.set_ylabel("Users")
    if labels:
        ax.set_xticks(range(values.shape[1]), core.ITEMS[:values.shape[1]], rotation=45, ha="right")
        ax.set_yticks(range(values.shape[0]), core.USERS[:values.shape[0]])
    if annotate:
        for (u, i), value in np.ndenumerate(values):
            text = "?" if not np.isfinite(value) else f"{value:.0f}"
            rgb = np.array(palette((value - vmin) / (vmax - vmin))[:3]) if np.isfinite(value) else np.ones(3)
            foreground = "#17304a" if rgb @ np.array([.2126, .7152, .0722]) > .5 else "white"
            ax.text(i, u, text, ha="center", va="center", fontsize=8,
                    color="#475569" if not np.isfinite(value) else foreground)
    return image


def scene1(ctx, out):
    data = ctx["data"]
    values = data["frame"].to_numpy()
    observed = values[np.isfinite(values)]
    def draw(count):
        shown = np.full_like(values, np.nan)
        positions = np.argwhere(np.isfinite(values))
        for u, i in positions[:count]:
            shown[u, i] = values[u, i]
        fig, axes = plt.subplots(1, 2, figsize=(12, 6.8), gridspec_kw={"width_ratios": [2.5, 1]})
        im = matrix(axes[0], shown, r"$R\in\mathbb{R}^{12\times10}$ • gray ? = unknown")
        fig.colorbar(im, ax=axes[0], label="Explicit rating (1–5)", shrink=.8)
        axes[1].hist(shown[np.isfinite(shown)], bins=np.arange(.5, 6.5), color=BLUE, rwidth=.8)
        axes[1].set(xticks=range(1, 6), xlabel="Observed star rating", ylabel="Number of ratings",
                    title=f"{count}/120 observed\n{1-count/120:.1%} sparsity")
        return finish(fig, "01  A missing cell is a question, not a one-star rating",
                      "Synthetic 12-user / 10-item training snapshot. Empty cells reveal no expressed preference.")
    files = [png(draw(len(observed)), out, "01_user_item_matrix.png")]
    if ctx.get("animate_matrix"):
        files.append(gif(draw, np.linspace(0, len(observed), ctx["frames"], dtype=int), out,
                         "01_interactions_appear.gif", 350))
    return dict(files=files, observations=dict(training_ratings=len(observed), sparsity=float(np.isnan(values).mean())))


def scene2(ctx, out):
    data = ctx["data"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    matrix(axes[0, 0], data["train"][:4, :6], "Explicit: observed ratings (gray = unknown)")
    n = data["clicks"][:4, :6]
    im = matrix(axes[0, 1], n.astype(float), "Implicit: clicks, including observed zero counts", vmin=0, vmax=max(1, n.max()), cmap="Blues")
    fig.colorbar(im, ax=axes[0, 1], label="Click count", shrink=.7)
    counts = np.arange(11)
    for alpha in (0., 1., 3., 8.):
        _, confidence = core.implicit_targets(counts, alpha)
        axes[1, 0].plot(counts, confidence, label=f"α={alpha:g}", marker=".")
    axes[1, 0].set(xlabel="Qualifying click count n", ylabel="Confidence weight c = 1 + αn",
                   title="Binary target p = 1[n > 0]; confidence is separate")
    axes[1, 0].legend()
    event_counts = [data[key].sum() for key in ("impressions", "views", "clicks", "purchases")]
    axes[1, 1].bar(["Shown", "Views", "Clicks", "Purchases"], event_counts, color=[GRAY, BLUE, ORANGE, GREEN])
    axes[1, 1].set(ylabel="Synthetic event count", title="Same environment, different event semantics")
    finish(fig, "02  Behavior is evidence with uncertainty",
           "p=0 is a modeling convention, not proof of dislike. No impression means no chance to click.\n"
           "Views, clicks and purchases are separate nested synthetic events; ratings are not converted into clicks.")
    return dict(files=[png(fig, out, "02_explicit_implicit.png")],
                observations=dict(event_counts=[int(x) for x in event_counts], alphas=[0, 1, 3, 8]))


def scene3(ctx, out):
    data = ctx["data"]; ratings = data["train"]
    sims = core.similarity_matrices(ratings)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    labels = [("raw_cosine", "Raw cosine • missing→0 as vector convention"),
              ("centered_cosine", "Full-row centered cosine • missing→0 deviation"),
              ("pearson", "Pearson • co-rated means; min overlap=3"),
              ("shrunk", "Shrunk Pearson • β=5; self excluded in prediction")]
    for ax, (key, title) in zip(axes.flat, labels):
        im = ax.imshow(sims[key], vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set(title=title, xlabel="User ID", ylabel="User ID", xticks=[0, 4, 8, 11], yticks=[0, 4, 8, 11])
        fig.colorbar(im, ax=ax, shrink=.65)
    ax = axes[1, 1]
    angles = np.arange(12) * 2 * np.pi / 12
    xy = np.column_stack((np.cos(angles), np.sin(angles)))
    edges = 0
    for u in range(12):
        for v in range(u):
            if sims["shrunk"][u, v] > .3:
                ax.plot(*xy[[u, v]].T, color=GREEN, alpha=.6, lw=2 * sims["shrunk"][u, v]); edges += 1
    ax.scatter(*xy.T, c=np.repeat(COLORS, 4), s=180, zorder=3)
    for i, (x, y) in enumerate(xy):
        ax.text(x * 1.2, y * 1.2, core.USERS[i], ha="center", va="center", fontsize=8)
    ax.set(title="Edges: shrunk similarity > 0.30", xlabel="Circular layout x (not latent)", ylabel="Layout y")
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax = axes[1, 2]
    upper = np.triu_indices(12, 1)
    ax.scatter(sims["overlap"][upper], sims["pearson"][upper], c=BLUE, label="Unshrunk", alpha=.6)
    ax.scatter(sims["overlap"][upper], sims["shrunk"][upper], c=ORANGE, label="β=5", alpha=.6)
    ax.set(xlabel="Co-rated item count", ylabel="Similarity", title="Support changes shrinkage; not certainty")
    ax.legend()
    finish(fig, "03  Similarity depends on centering and shared evidence",
           "Raw zero-fill is a vector approximation, not a zero-star observation. Gray unknown ratings remain masked in Pearson.")
    files = [png(fig, out, "03_similarity_comparison.png")]
    a, b = ratings[0], ratings[1]
    mask = np.isfinite(a) & np.isfinite(b)
    x, y = a[mask], b[mask]
    corr = overlap_pearson(a, b, 3, 0)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.8))
    indices = np.arange(mask.sum())
    axes[0].bar(indices - .18, x, .35, label="U00", color=BLUE)
    axes[0].bar(indices + .18, y, .35, label="U01", color=ORANGE)
    axes[0].set(xticks=indices, xticklabels=np.array(core.ITEMS)[mask], ylabel="Explicit rating", title="Only co-rated items")
    axes[0].tick_params(axis="x", rotation=45); axes[0].legend()
    axes[1].scatter(x - x.mean(), y - y.mean(), s=80, color=GREEN)
    for i, label in enumerate(np.array(core.ITEMS)[mask]):
        axes[1].annotate(label, (x[i] - x.mean(), y[i] - y.mean()), fontsize=8, xytext=(4, 4), textcoords="offset points")
    numerator = np.dot(x - x.mean(), y - y.mean())
    denominator = np.linalg.norm(x - x.mean()) * np.linalg.norm(y - y.mean())
    axes[1].set(xlabel=f"U00 deviation from overlap mean {x.mean():.2f}",
                ylabel=f"U01 deviation from overlap mean {y.mean():.2f}",
                title=f"Pearson = {numerator:.2f}/{denominator:.2f} = {corr:.3f}")
    axes[1].axhline(0, color=GRAY); axes[1].axvline(0, color=GRAY)
    betas = np.linspace(0, 20, 81)
    for n in (2, int(mask.sum()), 10):
        axes[2].plot(betas, corr * n / (n + betas), label=f"n={n}, fixed r={corr:.2f}")
    axes[2].set(xlabel="Shrinkage coefficient β", ylabel="n/(n+β) × r", title="Controlled support curves (not extra users)")
    axes[2].legend(fontsize=8)
    finish(fig, "03b  A pairwise calculation you can inspect",
           "Two nonconstant shared ratings give |Pearson|=1; this does not establish reliability.\n"
           "Curves hold r fixed to isolate shrinkage. Equal-support scaling cancels in normalized prediction weights.")
    files.append(png(fig, out, "03_pair_and_shrinkage.png"))
    return dict(files=files, observations=dict(pair=[0, 1], overlap=int(mask.sum()), pearson=corr, graph_edges=edges))


def scene4(ctx, out):
    model = ctx["model"]
    ranking = model.recommend(0, k=10)
    supported = [r for r in ranking if r["source"] == "neighborhood"]
    if not supported:
        raise ValueError("the teaching snapshot must contain a supported candidate for U00")
    item = supported[0]["item"]
    trace = core.prediction_trace(model, 0, item)
    stages = ["Select U00", "Find positive-similarity users", "Keep users who rated this item",
              "Retrieve supported weights", "Calculate normalized deviations", "Add target baseline",
              "Clip to rating scale", "Rank unseen supported items"]
    def draw(stage):
        fig, axes = plt.subplots(1, 3, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1.1, 1.4, 1.2]})
        weights = model.similarities_[0]
        colors = [GREEN if any(r["user"] == i for r in trace["rows"]) else GRAY for i in range(12)]
        axes[0].barh(core.USERS, weights if stage >= 1 else np.zeros(12), color=colors)
        axes[0].set(xlabel="Shrunk Pearson weight", title=f"Candidate: {core.ITEMS[item]}")
        axes[0].axvline(0, color="#475569")
        axes[1].axis("off")
        lines = [f"Step {stage + 1}/8: {stages[stage]}", "", f"Target training mean = {trace['baseline']:.3f}"]
        if stage >= 3:
            lines += [f"Σw = {trace['weight_sum']:.4f}", ""]
        if stage >= 4:
            for row in trace["rows"]:
                lines.append(f"U{row['user']:02d}: {row['weight']:.3f} × ({row['rating']:.1f} − {row['mean']:.3f}) / Σw")
                lines.append(f"      normalized contribution = {row['contribution']:+.3f}")
        if stage >= 5:
            raw = trace["baseline"] + sum(r["contribution"] for r in trace["rows"])
            lines += ["", f"Unclipped prediction = {raw:.4f}"]
        if stage >= 6:
            lines += [f"Final score = {trace['result']['score']:.4f}", "No neighbors → item/user/global mean"]
        axes[1].text(0, .98, "\n".join(lines), va="top", fontsize=10, linespacing=1.6)
        if stage < 7:
            contributions = [r["contribution"] if stage >= 4 else 0 for r in trace["rows"]]
            axes[2].bar([f"U{r['user']:02d}" for r in trace["rows"]], contributions, color=GREEN)
            axes[2].set(ylabel="Contribution to target baseline (rating units)", title="Each supported neighbor has a role")
            axes[2].axhline(0, color=GRAY)
        else:
            axes[2].barh([core.ITEMS[r["item"]] for r in ranking][::-1],
                         [r["score"] for r in ranking][::-1], color=BLUE)
            axes[2].set(xlim=(0, 5.3), xlabel="Predicted rating", title="Final unseen-item ranking")
        return finish(fig, "04  One recommendation, eight inspectable steps",
                      "Green users contribute positive weights, rated the candidate and pass the top-3 rule.\n"
                      "Means use training rows; similarity centers only common items. The model stays frozen throughout this reveal.")
    files = [png(draw(7), out, "04_neighborhood_calculation.png"),
             gif(draw, range(8), out, "04_neighborhood_steps.gif", 1100 if not ctx["fast"] else 550)]
    empty_item = np.column_stack((ctx["data"]["train"], np.full(12, np.nan)))
    fallback = UserNeighborhood().fit(empty_item).predict(0, 10)
    return dict(files=files, observations=dict(trace=trace, no_neighbor_example=fallback))


def factor_html(ctx, out):
    factor = ctx["factor"]; data = ctx["data"]
    p, q = factor["p"], factor["q"]
    fig = go.Figure([
        go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers+text", text=core.USERS,
                     textposition="top center", name="Users P", marker=dict(color=BLUE, size=6),
                     hovertemplate="User %{text}<br>learned vector (%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>"),
        go.Scatter3d(x=q[:, 0], y=q[:, 1], z=q[:, 2], mode="markers+text", text=core.ITEMS,
                     textposition="bottom center", name="Items Q", marker=dict(color=ORANGE, size=7),
                     customdata=np.column_stack((q @ p[0], factor["predictions"][0])),
                     hovertemplate="Item %{text}<br>P·Q = %{customdata[0]:.4f}<br>Predicted rating = %{customdata[1]:.4f}<br>with selected user's biases<extra></extra>"),
        go.Scatter3d(x=[0, p[0, 0]], y=[0, p[0, 1]], z=[0, p[0, 2]], mode="lines+markers",
                     name="Selected user vector", line=dict(color=GREEN, width=7)),
        go.Scatter3d(x=[], y=[], z=[], mode="lines", name="Candidate item vectors",
                     line=dict(color=ORANGE, width=3), hoverinfo="skip"),
    ])
    bound = float(max(np.max(np.abs(s[axis])) for s in factor["states"] for axis in ("p", "q"))) * 1.2
    fig.update_layout(title="05  Learned 3D factors • rotate, zoom, inspect",
                      height=600, margin=dict(l=0, r=0, b=0, t=55),
                      scene=dict(xaxis=dict(title="Latent 1", range=[-bound, bound]),
                                 yaxis=dict(title="Latent 2", range=[-bound, bound]),
                                 zaxis=dict(title="Latent 3", range=[-bound, bound]), aspectmode="cube"),
                      legend=dict(orientation="h", y=-.04), template="plotly_white")
    payload = dict(mean=factor["mean"], states=factor["states"], users=core.USERS, items=core.ITEMS,
                   observed=data["train_mask"], train=data["train"])
    # NaN ratings are unnecessary in the interactive payload; only masks and learned state are needed.
    payload.pop("train")
    payload = json.loads(json.dumps(payload, default=lambda x: x.tolist(), allow_nan=False))
    chart = fig.to_html(full_html=False, include_plotlyjs=True, div_id="factor-plot",
                        config=dict(responsive=True, displaylogo=False), auto_play=False,
                        include_mathjax=False)
    options = "".join(f'<option value="{i}">{name}</option>' for i, name in enumerate(core.USERS))
    document = """<!doctype html><html lang="en"><meta charset="utf-8"><link rel="icon" href="data:,"><title>Day 42 • 3D factors</title>
<style>body{font:16px system-ui;margin:24px auto;max-width:1250px;color:#17304a;background:#f8fafc}
main{background:white;padding:24px;border-radius:16px}label{display:inline-block;margin:12px 24px 12px 0}
table{border-collapse:collapse;width:100%}td,th{padding:8px;text-align:right;border-bottom:1px solid #dce3eb}
td:first-child,th:first-child{text-align:left}.note{padding:14px;background:#eaf1f7;border-radius:8px}
a{color:#2563a8}</style><main><a href="index.html">← Gallery</a><h1>Dot products, not distances</h1>
<p class="note">Prediction = global mean + user bias + item bias + P·Q. Axes have no fixed semantic meaning.
Orange rays are candidate item vectors from the origin. Their lengths and norms affect dot products;
closeness to the green user endpoint is not the scoring rule. Scores below are unbounded model estimates.</p>
<label>User <select id="user-select">OPTIONS</select></label>
<label>Saved training epoch <input id="epoch-slider" type="range" min="0" max="MAX" value="MAX"><b id="epoch-label"></b></label>
CHART<h2>Unseen candidate arithmetic</h2><p id="model-info"></p>
<table><thead><tr><th>Item</th><th>μ</th><th>b user</th><th>b item</th><th>P·Q</th><th>Predicted rating</th><th>Euclidean distance</th></tr></thead><tbody id="score-table"></tbody></table>
<p>Unknown preferences are not verified ratings. This model fits only the fixed training mask.
Training slider states are real optimizer checkpoints. Learned factors may rotate together without changing dot products.</p></main>
<script>const lab=PAYLOAD; const gd=document.getElementById('factor-plot');let revision=0;
async function update(){const u=+document.getElementById('user-select').value;
const thisRevision=++revision;gd.dataset.ready='false';
const s=lab.states[+document.getElementById('epoch-slider').value];
document.getElementById('epoch-label').textContent=' '+s.epoch;
const coords=a=>[0,1,2].map(d=>a.map(v=>v[d]));const pc=coords(s.p),qc=coords(s.q);
await Plotly.restyle(gd,{x:[pc[0]],y:[pc[1]],z:[pc[2]]},[0]);
if(thisRevision!==revision)return;
const sizes=lab.observed[u].map(seen=>seen?5:10);
const scoreData=s.q.map((q,i)=>{const dot=s.p[u].reduce((a,v,d)=>a+v*q[d],0);return[dot,lab.mean+s.bu[u]+s.bi[i]+dot];});
await Plotly.restyle(gd,{x:[qc[0]],y:[qc[1]],z:[qc[2]],'marker.size':[sizes],customdata:[scoreData]},[1]);
if(thisRevision!==revision)return;
await Plotly.restyle(gd,{x:[[0,s.p[u][0]]],y:[[0,s.p[u][1]]],z:[[0,s.p[u][2]]]},[2]);
if(thisRevision!==revision)return;
let rays=[[],[],[]],rows=[];
lab.items.forEach((name,i)=>{if(lab.observed[u][i])return;
const dot=s.p[u].reduce((a,v,d)=>a+v*s.q[i][d],0),score=lab.mean+s.bu[u]+s.bi[i]+dot;
const distance=Math.sqrt(s.p[u].reduce((a,v,d)=>a+(v-s.q[i][d])**2,0));
rows.push({name,dot,score,distance,bi:s.bi[i]});for(let d=0;d<3;d++)rays[d].push(0,s.q[i][d],null);});
await Plotly.restyle(gd,{x:[rays[0]],y:[rays[1]],z:[rays[2]]},[3]);
if(thisRevision!==revision)return;
rows.sort((a,b)=>b.score-a.score);
document.getElementById('score-table').innerHTML=rows.map(r=>'<tr><td>'+r.name+'</td>'+[lab.mean,s.bu[u],r.bi,r.dot,r.score,r.distance].map(v=>'<td>'+v.toFixed(4)+'</td>').join('')+'</tr>').join('');
document.getElementById('model-info').textContent='Selected '+lab.users[u]+' • epoch '+s.epoch+' • '+rows.length+' unknown candidates';
gd.dataset.selectedUser=u;gd.dataset.epoch=s.epoch;gd.dataset.ready='true';}
document.getElementById('user-select').addEventListener('change',update);
document.getElementById('epoch-slider').addEventListener('input',update);update();</script></html>"""
    document = document.replace("OPTIONS", options).replace("MAX", str(len(factor["states"]) - 1))
    document = document.replace("CHART", chart).replace("PAYLOAD", json.dumps(payload, allow_nan=False))
    path = out / "05_factorization_3d.html"
    path.write_text(document, encoding="utf-8")
    return path.name


def scene5(ctx, out):
    factor = ctx["factor"]
    history = factor["history"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    axes[0].plot([h["epoch"] for h in history], [h["objective"] for h in history], label="Training MSE + L2", color=BLUE)
    axes[0].plot([h["epoch"] for h in history], [h["mse"] for h in history], label="Training MSE only", color=ORANGE)
    axes[0].set(xlabel="Full-batch gradient-descent epoch", ylabel="Masked training objective / MSE", title="Only observed training pairs contribute residuals")
    axes[0].legend()
    p, q = factor["p"][0], factor["q"]
    dot = q @ p
    distance = np.linalg.norm(q - p, axis=1)
    axes[1].scatter(distance, dot, color=GREEN, s=70)
    for i, name in enumerate(core.ITEMS):
        axes[1].annotate(name, (distance[i], dot[i]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[1].set(xlabel="Euclidean distance ||P[U00] − Q[item]||", ylabel="Dot product P[U00] · Q[item]",
                title="Compatibility is not minimum distance")
    finish(fig, "05  Learn from a mask; inspect the actual 3D factors",
           "Unclipped score = μ + b(user) + b(item) + P·Q. Loss curves are training fit, not held-out quality.\n"
           "Orthogonal rotations preserve dot products; learned axes do not automatically describe human preferences.")
    files = [png(fig, out, "05_factorization_training.png"), factor_html(ctx, out)]
    def draw(state):
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.8))
        im = matrix(axes[0], state["predictions"], f"Predictions at saved epoch {state['epoch']}", annotate=False)
        fig.colorbar(im, ax=axes[0], label="Predicted rating (display scale 1–5)")
        axes[1].plot([h["epoch"] for h in history], [h["mse"] for h in history], color=GRAY)
        prefix = history[:state["epoch"] + 1]
        axes[1].plot([h["epoch"] for h in prefix], [h["mse"] for h in prefix], color=BLUE)
        axes[1].scatter([state["epoch"]], [history[state["epoch"]]["mse"]], color=ORANGE)
        axes[1].set(xlabel="Epoch", ylabel="Training masked MSE", title="Recorded optimizer states; no test residuals")
        return finish(fig, "05b  Predictions change during actual gradient descent",
                      "Unobserved predictions have no verified label in training. Identical fit configuration in fast/full modes.")
    files.append(gif(draw, factor["states"], out, "05_factor_training.gif", 250))
    return dict(files=files, observations=dict(configuration=factor["configuration"],
                initial_mse=history[0]["mse"], final_mse=history[-1]["mse"],
                final_objective=history[-1]["objective"]))


def scene6(ctx, out):
    data, factor = ctx["data"], ctx["factor"]
    error = np.where(data["train_mask"], factor["predictions"] - data["train"], np.nan)
    fig, axes = plt.subplots(1, 4, figsize=(16, 6.3))
    for ax, values, title, cmap, low, high in (
        (axes[0], data["train"], "Observed TRAIN ratings", "viridis", 1, 5),
        (axes[1], factor["predictions"], "Model estimates • all pairs", "viridis", 1, 5),
        (axes[2], error, "Predicted − observed • TRAIN only", "RdBu_r", -2, 2),
        (axes[3], data["train_mask"].astype(float), "TRAIN mask • 1 observed / 0 unknown", "Blues", 0, 1),
    ):
        im = matrix(ax, values, title, vmin=low, vmax=high, cmap=cmap, annotate=False)
        fig.colorbar(im, ax=ax, shrink=.65)
    finish(fig, "06  Reconstruction fills a matrix; it does not reveal truth",
           "Gray residuals have no training label. Predictions are unbounded; display colors saturate at 1 and 5.\n"
           "Held-out errors are measured separately in scene 8. No unknown prediction is presented as an observation.")
    files = [png(fig, out, "06_matrix_reconstruction.png")]
    dimensions = [1, 2, 3, 5]
    fits = [fit_factorization(data["train"], dimensions=d, snapshots=1) for d in dimensions]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(dimensions, [f["history"][-1]["mse"] for f in fits], "o-", color=BLUE)
    ax.set(xticks=dimensions, xlabel="Latent dimensions", ylabel="Training MSE after 240 epochs",
           title="Capacity comparison • same mask, L2 and optimizer budget")
    finish(fig, "06b  Capacity is not held-out quality",
           "Each dimension is fitted independently with seed 42. Optimization and regularization affect fit; no monotonic guarantee.")
    files.append(png(fig, out, "06_capacity_comparison.png"))
    return dict(files=files, observations=dict(dimensions=dimensions,
                training_mse=[f["history"][-1]["mse"] for f in fits]))


def scene7(ctx, out):
    data = ctx["data"]
    states = core.cold_start_states(data["train"])
    def draw(state):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        rows = state["predictions"]
        colors = [GREEN if r["source"] == "neighborhood" else ORANGE for r in rows]
        bars = axes[0].barh([core.ITEMS[r["item"]] for r in rows][::-1], [r["score"] for r in rows][::-1], color=colors[::-1])
        for bar, row in zip(bars, rows[::-1]):
            axes[0].text(.04, bar.get_y() + bar.get_height()/2, row["source"], va="center", color="white", fontsize=9)
        axes[0].set(xlabel="Training rating count" if state["step"] == 0 else "Predicted rating (1–5)",
                    title="Popularity fallback" if state["step"] == 0 else "Neighborhood + labeled fallback scores")
        axes[1].plot([s["step"] for s in states], [s["positive_neighbors"] for s in states], color=GRAY, marker="o")
        prefix = states[:state["step"] + 1]
        axes[1].plot([s["step"] for s in prefix], [s["positive_neighbors"] for s in prefix], color=GREEN, marker="o")
        axes[1].set(xlabel="Onboarding ratings observed", ylabel="Positive supported user neighbors",
                    title=f"{state['step']} ratings • {state['personalized']}/4 list items personalized",
                    xticks=range(len(states)))
        return finish(fig, "07  Cold start changes as evidence arrives",
                      "Orange = fallback; green = collaborative score. Step 0 uses count-popularity units; later steps use rating units.\n"
                      "New user copies U00's known training profile, one rating at a time; means/similarities are refitted each step.")
    files = [gif(draw, states, out, "07_cold_start.gif", 1000 if not ctx["fast"] else 500),
             png(draw(states[-1]), out, "07_cold_start.png")]
    repeats = 4 if ctx["fast"] else 12
    sweep = core.sparsity_sweep(data, repeats=repeats)
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.4))
    x = [r["sparsity"] for r in sweep]
    for ax, key, title in zip(axes, ("neighbors", "coverage", "stability_sd"),
                              ("Mean contributors per held-out pair", "Fraction with collaborative support", "Prediction SD across repeated thinning")):
        ax.plot(x, [r[key] for r in sweep], "o-", color=BLUE)
        ax.set(xlabel="Measured training sparsity", ylabel=key.replace("_", " "), title=title)
    finish(fig, "07b  Sparse evidence affects support and repeatability",
           f"{repeats} seeded training-mask thinnings at each keep fraction. Fixed future-label-free model inputs; fixed held-out queries.\n"
           "Coverage means a neighborhood-backed prediction, not merely a finite fallback. Stability is measured SD, not accuracy.")
    files.append(png(fig, out, "07_sparsity.png"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    extended = np.column_stack((data["train"], np.full(12, np.nan)))
    model = UserNeighborhood().fit(extended)
    fallback = model.predict(0, 10)
    axes[0].axis("off")
    axes[0].text(.05, .9, "New catalog item: no ratings\n\nShared ratings: 0\nCollaborative neighbors: 0\n"
                 f"Diagnostic fallback: {fallback['score']:.3f}\nSource: {fallback['source']}\n\nExcluded from pure-CF candidate list",
                 va="top", fontsize=13, linespacing=1.6)
    metadata = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0.]])
    profile = np.array([1., 0, 0])
    content = metadata @ profile
    axes[1].bar(["Sci-fi A", "Cooking B", "Nature C", "NEW sci-fi"], content, color=[GRAY, GRAY, GRAY, GREEN])
    axes[1].set(ylim=(0, 1.2), ylabel="Category-profile dot product", title="Separate content rule: known sci-fi preference")
    axes[1].tick_params(axis="x", rotation=15)
    finish(fig, "07c  New-item discovery needs additional evidence",
           "Right: hand-declared one-hot metadata and user category profile, not learned collaborative factors.\n"
           "A content rule can score a new item; a user-mean fallback contains no evidence about that item.")
    files.append(png(fig, out, "07_new_item.png"))
    return dict(files=files, observations=dict(cold_states=states, thinning_repeats=repeats,
                sparsity=sweep, new_item_fallback=fallback))


def scene8(ctx, out):
    result = core.evaluation(ctx["data"], ctx["factor"])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    errors = result["errors"]
    axes[0, 0].bar(["MF unbounded", "CF clipped"], [e["mae"] for e in errors.values()], color=[BLUE, GREEN])
    axes[0, 0].set(ylabel="Held-out rating MAE", title=f"Rating task • {result['test_pairs']} held-out explicit pairs")
    candidates = ["A", "B", "C", "D", "E"]
    relevant = {"A", "C"}
    orders = [["A", "C", "B", "D", "E"], ["B", "C", "A", "D", "E"]]
    hand_metrics = []
    for ax, order, name in zip((axes[0, 1], axes[1, 0]), orders, ("Ranking A", "Ranking B")):
        metrics = core.ranking_metrics(order, relevant, 3, candidates)
        hand_metrics.append(metrics)
        ax.bar(np.arange(1, 6), [1 if i in relevant else .12 for i in order],
               color=[GREEN if i in relevant else GRAY for i in order])
        for rank, label in enumerate(order, 1):
            ax.text(rank, .05, label, ha="center")
            if rank <= 3 and label in relevant:
                ax.text(rank, .7, f"DCG +{1/np.log2(rank+1):.3f}", ha="center", fontsize=8)
        ax.axvline(3.5, color=ORANGE, ls="--")
        ax.set(xticks=range(1, 6), xlabel="Rank (dashed line = K=3)", ylabel="Binary relevance", ylim=(0, 1.4),
               title=f"{name}: P@3={metrics['precision']:.3f}, R@3={metrics['recall']:.3f}, NDCG={metrics['ndcg']:.3f}")
    axes[1, 1].axis("off")
    actual_names = [core.ITEMS[i] for i in result["order"]]
    relevant_names = [core.ITEMS[i] for i in result["relevant"]]
    m = result["metrics"]
    lines = ["Learned MF ranking • U00", f"Candidate universe: {len(result['candidates'])} training-unseen items",
             "Order: " + ", ".join(actual_names), "Relevant (synthetic truth ≥3): " + ", ".join(relevant_names),
             f"P@3={m['precision']:.3f}   R@3={m['recall']:.3f}   NDCG@3={m['ndcg']:.3f}", "",
             "A/B lists are hand-specified metric demonstrations.", "Both have 2 relevant items in top 3; ordering differs.",
             "Learned ranking uses full synthetic candidate labels.", "Those labels never enter model training."]
    axes[1, 1].text(0, .98, "\n".join(lines), va="top", fontsize=10, linespacing=1.7, wrap=True)
    finish(fig, "08  Numerical error, retrieval and ordering answer different questions",
           "Hand example: candidates A–E; relevant={A,C}; binary gain; DCG discount=1/log₂(rank+1); ideal DCG=1+1/log₂3.\n"
           "No sampled negatives. Synthetic offline metrics do not establish online effectiveness; rating and ranking label populations differ.")
    return dict(files=[png(fig, out, "08_evaluation.png")], observations=dict(**result, hand_metrics=hand_metrics))


def scene9(ctx, out):
    records = core.exposure_simulation()
    def draw(time):
        fig, axes = plt.subplots(1, 3, figsize=(14, 5.8))
        state = records["popularity"][time]
        axes[0].bar(np.arange(6) - .18, state["impressions"], .35, color=BLUE, label="Shown")
        axes[0].bar(np.arange(6) + .18, state["clicks"], .35, color=ORANGE, label="Clicked")
        axes[0].set(xlabel="Item ID", ylabel="Cumulative events", title=f"Popularity policy • round {time}")
        axes[0].legend()
        for color, (name, history) in zip((RED, GREEN), records.items()):
            prefix = history[:time + 1]
            axes[1].plot([h["round"] for h in prefix], [h["hhi"] for h in prefix], label=name, color=color)
        axes[1].set(xlim=(0, 30), ylim=(1/6, .55), xlabel="Round", ylabel="HHI of prior + cumulative clicks", title="Policy can affect concentration")
        axes[1].legend(fontsize=8)
        axes[2].axis("off")
        axes[2].text(.02, .96, "An observation has stages\n\nShown → not clicked\nShown → clicked\nNever shown → no click opportunity\n\n"
                     "80 impressions / round\nClick probability = 0.25 for ALL items\nInitial count prior = [8,2,1,1,1,1]\n"
                     "Popularity: sample ∝ prior + clicks\nExplore: 75% popularity + 25% uniform\n\n"
                     "Impression logs reveal opportunity,\nnot unbiased preferences by themselves.", va="top", fontsize=10, linespacing=1.5)
        return finish(fig, "09  More clicks can follow more exposure",
                      "Seed 42; fixed equal click propensities, sampling with replacement, no position or fatigue effects.\n"
                      "Only policy differs. This single simulated path is not an estimate of a real platform's bias or exploration benefit.")
    states = np.unique(np.linspace(0, 30, ctx["frames"], dtype=int))
    files = [png(draw(30), out, "09_exposure_bias.png"), gif(draw, states, out, "09_feedback_loop.gif", 250)]
    return dict(files=files, observations=dict(final={name: history[-1] for name, history in records.items()},
                rounds=30, impressions_per_round=80, click_probability=.25, seed=42))


def scene10(ctx, out):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis("off")
    boxes = [(.2, 4.7, "Interaction events", "Ratings • clicks • impressions"),
             (3.7, 4.7, "User–item matrix", "Masks • feedback semantics"),
             (7.2, 4.7, "Model training", "Neighborhood CF / masked factors"),
             (10.7, 4.7, "Candidate retrieval", "Seen / eligible item filtering"),
             (10.7, 1.8, "Ranking", "Scores • policy • constraints"),
             (7.2, 1.8, "Recommendations", "List delivered to a user"),
             (3.7, 1.8, "New interactions", "Exposure → behavior → new data")]
    for x, y, title, description in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 3.05, 1.05, boxstyle="round,pad=.12", fc="#eaf1f7", ec=BLUE))
        ax.text(x + 1.52, y + .72, title, ha="center", va="center", fontsize=12, fontweight="bold")
        ax.text(x + 1.52, y + .28, description, ha="center", va="center", fontsize=8)
    links = [((3.4, 5.2), (3.55, 5.2)), ((6.9, 5.2), (7.05, 5.2)), ((10.4, 5.2), (10.55, 5.2)),
             ((12.2, 4.55), (12.2, 3.0)), ((10.55, 2.3), (10.4, 2.3)), ((7.05, 2.3), (6.9, 2.3))]
    for start, end in links:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15, color=BLUE, lw=2))
    ax.add_patch(FancyArrowPatch((3.55, 2.3), (1.7, 4.55), connectionstyle="arc3,rad=-.3", arrowstyle="-|>", mutation_scale=18, color=ORANGE, lw=2))
    ax.text(.15, 1.1, "Feedback loop:\npolicy shapes future exposure", color=ORANGE, fontsize=11)
    ax.text(7, 3.7, "Offline evaluation: fixed held-out labels, separate from training\n"
            "Cold-start branch: count popularity / content evidence before ranking\n"
            "Online monitoring: latency, support, concentration, freshness, product outcomes",
            ha="center", va="center", fontsize=11, color=GREEN, linespacing=1.7)
    finish(fig, "10  A recommendation system keeps several boundaries explicit",
           "Conceptual architecture: no deployment or cloud dependency. Current permissions and inventory belong in eligibility checks.\n"
           "Offline quality, online monitoring and experimental product impact require different evidence.")
    return dict(files=[png(fig, out, "10_recommendation_pipeline.png")], observations=dict(deployment="conceptual only"))


RENDERERS = {i: globals()[f"scene{i}"] for i in SCENES}


def write_gallery(out, records):
    sections = []
    for number in sorted(records, key=int):
        record = records[number]
        name, question, hypothesis, limitation = SCENES[int(number)]
        assets = []
        for filename in record["files"]:
            label = escape(filename)
            if Path(filename).suffix in (".png", ".gif"):
                assets.append(f'<a href="{label}"><img loading="lazy" src="{label}" alt="{escape(question)}"></a>')
            else:
                assets.append(f'<p><a href="{label}">Open interactive 3D explorer →</a></p>')
        sections.append(f'<section><h2>{number}. {escape(name)}</h2><h3>{escape(question)}</h3>'
                        f'<p>{escape(hypothesis)}</p>{"".join(assets)}<p><b>Limit:</b> {escape(limitation)}</p></section>')
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><link rel="icon" href="data:,"><title>Day 42 Visual Lab</title><style>' \
               'body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#17304a;background:#f8fafc}' \
               'section{background:white;border:1px solid #dce3eb;border-radius:12px;padding:22px;margin:24px 0}' \
               'img{width:100%;height:auto;margin-top:12px}a{color:#2563a8}p{line-height:1.6}</style>' \
               '<h1>Day 42 • Recommender Systems I</h1><p>Follow scenes 1–10, or start with 1, 3, 4, 5, 7 and 8. ' \
               'Everything is synthetic and local. Animations reveal actual calculations or declared simulations. ' \
               'Interpretations require author review. Scene 5 contains the offline user/epoch controls.</p>' \
               + ''.join(sections) + '</html>'
    (out / "index.html").write_text(document, encoding="utf-8")

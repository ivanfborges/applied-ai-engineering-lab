"""CPU-only visual laboratory. Run --help for independent demonstrations."""

import os

# Bound numeric child pools before importing numerical packages; users may override.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import logging
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import PIL
import plotly
import sklearn

from embedding_lab import (
    ANALOGIES, CATEGORIES, COLORS, DOCUMENTS, QUERIES, RARE_WORDS, association_matrices,
    category, diagnostics, fixed_projection, forward_trace, make_corpus, mean_vectors,
    negative_trace, noise_distribution, pmi_experiment, retrieval_experiment, sigmoid,
    snapshot_vectors, sweep_experiments, tokenize, train_model,
)
from embedding_html import analogy_explorer, cosine_explorer, embedding_space, retrieval_space
from from_scratch import analogy, cosine, nearest, training_data, unit_rows


LOG = logging.getLogger("day65")
DEMO_NAMES = ("all", "representations", "window", "cbow", "skipgram", "negative_sampling", "pmi", "glove", "cosine", "analogies", "polysemy", "experiments", "retrieval")
LABEL_WORDS = "doctor physician nurse developer programmer engineer car bus king queen man woman france paris germany berlin bank money river hot cold apple banana".split()
MATRIX_WORDS = "doctor physician patient hospital developer programmer software computer bank money river water".split()


def matrix_plot(ax, matrix, rows, title, columns=None, limits=None, values=False, cmap="viridis"):
    data = np.ma.masked_invalid(matrix)
    color = plt.get_cmap(cmap).with_extremes(bad="#dbe3ee")
    image = ax.imshow(data, aspect="auto", cmap=color, vmin=limits[0] if limits else None, vmax=limits[1] if limits else None)
    ax.set_title(title, loc="left", fontsize=11, pad=10)
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    ax.set_xticks(range(matrix.shape[1]), columns if columns is not None else [str(i) for i in range(matrix.shape[1])], rotation=60, ha="right", fontsize=8)
    if values:
        for (i, j), value in np.ndenumerate(matrix):
            if np.isfinite(value):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=7, color="white" if value < np.nanmean(matrix) else "#0f172a")
    plt.colorbar(image, ax=ax, shrink=.8, pad=.02)
    return image


def decorate(fig, title, caption):
    fig.suptitle(title, x=.03, ha="left", fontsize=17, fontweight="bold", color="#0f172a")
    previous = getattr(fig, "_lab_caption", None)
    if previous is not None and previous in fig.texts:
        previous.set_text(caption)
    else:
        fig._lab_caption = fig.text(.03, .015, caption, ha="left", fontsize=9, color="#475569", wrap=True)
    fig.tight_layout(rect=(.01, .065, .99, .92))


def export_png(path, fig):
    fig.savefig(path, dpi=120, facecolor="white")
    plt.close(fig)


def export_gif(path, fig, update, frames, fps=2):
    animation = FuncAnimation(fig, update, frames=frames, repeat=True, interval=1000/fps)
    try:
        animation.save(path, writer=PillowWriter(fps=fps), dpi=85)
    finally:
        plt.close(fig)


def display_ids(words, limit=16):
    selected = [words.index(w) for w in MATRIX_WORDS if w in words]
    return selected[:limit]


def scatter(ax, coordinates, words, title, labels=True):
    for group in [*CATEGORIES, "other"]:
        ids = [i for i, word in enumerate(words) if category(word) == group]
        if ids:
            ax.scatter(coordinates[ids, 0], coordinates[ids, 1], c=COLORS.get(group, "#cbd5e1"), s=30 if group != "other" else 9, label=group, alpha=.8)
    if labels:
        for i, word in enumerate(words):
            if word in LABEL_WORDS:
                ax.annotate(word, coordinates[i], xytext=(3, 3), textcoords="offset points", fontsize=8)
    ax.set_title(title, loc="left", fontsize=12)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=.15)


def limits_for(coordinates):
    points = np.vstack(coordinates)
    low, high = points.min(axis=0), points.max(axis=0)
    margin = np.maximum(high - low, .2) * .15
    return low - margin, high + margin


def evolution(path, model):
    coords, evr, _ = fixed_projection(snapshot_vectors(model))
    low, high = limits_for(coords)
    fig, ax = plt.subplots(figsize=(12, 7))
    def update(frame):
        ax.clear()
        state = model.snapshots[frame]
        scatter(ax, coords[frame], model.fit.words, f"{model.mode.upper()} · epoch {state.epoch} · own loss {state.loss:.4f}")
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]))
        ax.legend(ncol=5, loc="upper center", fontsize=7)
        decorate(fig, "Recorded embedding evolution", f"One PCA across all snapshots; variance {evr[0]:.1%} + {evr[1]:.1%}. Fixed axes. Colors are synthetic categories; rankings use original dimensions.")
    export_gif(path, fig, update, len(coords))


def window_animation(path, radius):
    sentence = "the doctor treats a patient in the hospital".split()
    fig, ax = plt.subplots(figsize=(12, 5))
    def update(center):
        ax.clear()
        ax.set(xlim=(-.5, 8), ylim=(0, 5))
        ax.axis("off")
        ids = [j for j in range(max(0, center-radius), min(len(sentence), center+radius+1)) if j != center]
        for i, word in enumerate(sentence):
            color = "#fed7aa" if i == center else "#bfdbfe" if i in ids else "#f1f5f9"
            ax.text(i, 3.8, word, ha="center", bbox=dict(boxstyle="round,pad=.5", facecolor=color, edgecolor="white"), fontsize=12)
        ax.text(0, 2.6, f"Center: {sentence[center]} (position {center})   Context positions: {ids}", fontsize=12)
        ax.text(0, 1.9, f"CBOW: {', '.join(sentence[j] for j in ids)}  →  {sentence[center]}", fontsize=12)
        pairs = [f"({sentence[center]}, {sentence[j]})" for j in ids]
        ax.text(0, 1.0, "Skip-gram events:\n" + "  ".join(pairs[:3]) + "\n" + "  ".join(pairs[3:]), fontsize=11)
        decorate(fig, "Fixed sliding context window", f"Radius {radius}; boundaries clipped within the sentence. Repeated occurrences are separate. Dynamic random-radius Word2Vec windows are not used here.")
    export_gif(path, fig, update, len(sentence), fps=1)


def architecture_animation(path, focus, seed):
    traces = {mode: forward_trace(mode, seed) for mode in ("cbow", "skipgram")}
    stages = [("Select occurrences", "features"), ("Embedding lookup", "inputs"), ("Mean / center hidden state", "hidden"),
              ("Output logits", "logits"), ("Full softmax", "probabilities"), ("Target negative log-likelihood", "loss"),
              ("Analytical gradients", "dv"), ("Simultaneous update", "after_inputs")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    def update(frame):
        for ax, mode in zip(axes, traces):
            ax.clear(); ax.axis("off")
            trace = traces[mode]
            label, key = stages[frame]
            ax.set_title(mode.upper() + (" · focus" if mode == focus else " · comparison"), loc="left", fontsize=14)
            ax.text(0, .96, "context → mean → center" if mode == "cbow" else "center → each context", transform=ax.transAxes, fontsize=12)
            for i,node in enumerate(("select","lookup","hidden","logits","softmax","NLL","gradient","update")):
                left=.005+i*.124
                active=i==frame
                ax.add_patch(FancyBboxPatch((left,.82),.112,.065,boxstyle="round,pad=.004",transform=ax.transAxes,
                                           facecolor="#bfdbfe" if active else "#f1f5f9",edgecolor="#2563eb" if active else "#cbd5e1"))
                ax.text(left+.056,.852,node,transform=ax.transAxes,ha="center",va="center",fontsize=7)
                if i<7:
                    ax.annotate("",(left+.123,.852),(left+.112,.852),xycoords=ax.transAxes,arrowprops=dict(arrowstyle="->",lw=.7))
            ax.text(0, .76, f"Step {frame+1}: {label}", transform=ax.transAxes, fontsize=12, fontweight="bold")
            value = np.array2string(np.asarray(trace[key]), precision=3, suppress_small=True)
            ax.text(0, .65, value, transform=ax.transAxes, family="monospace", fontsize=11, va="top")
            extra = f"Vocabulary: {', '.join(trace['words'])}\nTarget IDs: {trace['targets'].tolist()}\nMean NLL: {trace['loss']:.6f} → {trace['after_loss']:.6f}"
            if frame == 6:
                extra += "\nOutput-table gradient:\n" + np.array2string(trace["du"], precision=3)
            ax.text(0, .05, extra, transform=ax.transAxes, fontsize=10, va="bottom", family="monospace")
        decorate(fig, "CBOW and Skip-gram · actual numerical forward/backward pass", "Four-word vocabulary; dimension 2; lr 0.4. Full-softmax mean loss. Values and gradients are computed, not illustrative placeholders.")
    export_gif(path, fig, update, len(stages), fps=1)


def sgns_animation(path, trace):
    all_points = [p for row in trace for p in row["after"]]
    low, high = limits_for(all_points)
    fig, (ax, info) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1.1, 1]})
    def update(frame):
        ax.clear(); info.clear(); info.axis("off")
        row = trace[frame]
        oldv, oldu = row["before"]
        v, u = row["after"]
        for before, after, label, color in ((oldv[0], v[0], "center: doctor", "#2563eb"), (oldu[1], u[1], "positive: patient", "#16a34a"), (oldu[2], u[2], "noise: computer", "#dc2626"), (oldu[0], u[0], "context: doctor (noise on odd steps)", "#d97706")):
            ax.arrow(0, 0, after[0], after[1], color=color, width=.006, length_includes_head=True)
            short_label=label.split(" (")[0]
            offset=(4,-17) if label.startswith("center") else (4,12) if label.startswith("positive") else (4,-7) if label.startswith("context") else (4,5)
            ax.annotate(short_label, after, xytext=offset, textcoords="offset points", fontsize=9)
            ax.annotate("", after, before, arrowprops=dict(arrowstyle="->", color=color, linestyle="--", lw=2))
            ax.scatter(*before, c=color, s=20, marker="x")
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), xlabel="Actual dimension 1", ylabel="Actual dimension 2")
        ax.grid(alpha=.2)
        positive = float(v[0] @ u[1])
        negatives = u[row["negatives"]] @ v[0]
        text = (f"Actual SGD step {frame+1}\nNoise IDs: {row['negatives']}\nLoss before update: {row['loss_before']:.4f}\n\n"
                f"Positive dot: {row['dot_before']:.3f} → {positive:.3f}\nPositive sigmoid after: {sigmoid(positive):.3f}\n"
                f"Positive loss after: {np.logaddexp(0,-positive):.3f}\nNoise scores after: {np.round(negatives,3)}\n"
                f"Noise sigmoid after: {np.round(sigmoid(negatives),3)}\nNoise losses after: {np.round(np.logaddexp(0,negatives),3)}\n\n"
                f"Cosine: {row['cos_before']:.3f} → {row['cos_after']:.3f}\nCenter norm: {np.linalg.norm(oldv[0]):.3f} → {np.linalg.norm(v[0]):.3f}\n"
                f"Center gradient: {np.round(row['gradients'][0][0],3)}\nΔv = -lr × gradient, lr={row['lr']}")
        info.text(0, 1, text, va="top", fontsize=11, family="monospace")
        decorate(fig, "Negative sampling · real two-table updates", "Solid arrows: post-update vectors; crosses: previous state; dashed arrows: parameter displacement. The direct objective uses dot products, not cosine or distance.")
    export_gif(path, fig, update, len(trace), fps=2)


def matrix_reconstruction(path, model, counts, ids):
    observed = counts[np.ix_(ids,ids)] > 0
    target = np.full(observed.shape, np.nan)
    target[observed] = np.log(counts[np.ix_(ids,ids)][observed])
    words = [model.fit.words[i] for i in ids]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    def update(frame):
        fig.clear()
        ax1, ax2 = fig.subplots(1, 2)
        state = model.snapshots[frame]
        v, u, b, c = state.parameters
        reconstruction = (v @ u.T + b[:,None] + c[None,:])[np.ix_(ids,ids)]
        reconstruction[~observed] = np.nan
        bound = max(1, np.nanmax(np.abs(target)))
        matrix_plot(ax1, target, words, "Fixed target log counts", words, (-bound,bound))
        matrix_plot(ax2, reconstruction, words, f"Actual reconstruction · epoch {state.epoch}", words, (-bound,bound))
        decorate(fig, "GloVe: corpus → weighted counts → log targets → gradients → vectors", f"Observed cells only; gray = omitted, not a zero target. Actual weighted mean loss {state.loss:.6f}. Fixed color scale; subset of vocabulary.")
    export_gif(path, fig, update, len(model.snapshots))


def analogy_animation(path, model=None):
    words = ("man", "king", "woman", "queen")
    if model is None:
        raw = np.array([[1.,0], [1,1], [2,0], [2,1]])
        coords = raw
        query = raw[1]-raw[0]+raw[2]
        ranking = [("queen", 1.)]
        label = "Constructed idealized vectors · conceptual geometry"
    else:
        fit = model.fit
        projected, _, pca = fixed_projection([fit.vectors])
        raw = np.array([fit.vectors[fit.words.index(w)] for w in words])
        coords = np.array([projected[0][fit.words.index(w)] for w in words])
        query = pca.transform((raw[1]-raw[0]+raw[2])[None,:])[0]
        ranking = analogy(fit, "man", "king", "woman", 3)
        label = "Actual trained Skip-gram · raw-vector offset, fixed PCA"
    low, high = limits_for([np.vstack([coords, query, [0,0]])])
    fig, ax = plt.subplots(figsize=(11, 7))
    def update(frame):
        ax.clear()
        ax.scatter(coords[:,0], coords[:,1], s=70, c=["#2563eb", "#2563eb", "#be185d", "#16a34a"])
        for word, xy in zip(words, coords):
            ax.annotate(word, xy, xytext=(5,5), textcoords="offset points", fontsize=12)
        if frame >= 1:
            ax.annotate("",coords[1],coords[0],arrowprops=dict(arrowstyle="->",lw=3,color="#d97706"))
        if frame >= 2:
            ax.annotate("",query,coords[2],arrowprops=dict(arrowstyle="->",lw=3,color="#dc2626"))
        if frame >= 3:
            ax.scatter(*query, marker="X", c="#dc2626", s=150, label="b - a + c")
            ax.plot([query[0],coords[3,0]],[query[1],coords[3,1]], "--", color="#64748b", label="projected gap to queen")
            ax.legend()
        titles = ["1 · Source vectors and expected fixture word", "2 · Relationship: king - man", "3 · Apply the offset at woman", "4 · Inspect predicted location and actual candidates"]
        ax.set(title=titles[frame], xlim=(low[0],high[0]), ylim=(low[1],high[1]), xlabel="Actual x" if model is None else "PC1", ylabel="Actual y" if model is None else "PC2")
        ax.grid(alpha=.2)
        boundary=". Constructed geometry, not a learned result." if model is None else ". A learned failure is retained."
        decorate(fig, label, "Top candidates in original space (sources excluded): " + "; ".join(f"{w} {s:.4f}" for w,s in ranking) + boundary)
    export_gif(path, fig, update, 4, fps=1)


def polysemy_animation(path, models, ratios):
    coords, evr, _ = fixed_projection([m.fit.vectors for m in models])
    words = models[0].fit.words
    selected = "bank money finance loan river water shore fisherman".split()
    ids = [words.index(w) for w in selected]
    low, high = limits_for([xy[ids] for xy in coords])
    fig, ax = plt.subplots(figsize=(11,7))
    def update(frame):
        ax.clear()
        xy = coords[frame]
        for i in ids:
            ax.scatter(*xy[i], c="#dc2626" if words[i] == "bank" else COLORS.get(category(words[i]),"#94a3b8"), s=120 if words[i]=="bank" else 50)
            ax.annotate(words[i],xy[i],xytext=(4,4),textcoords="offset points",fontsize=11)
        fit = models[frame].fit
        vec = fit.vectors[words.index("bank")]
        neighbors = nearest(words,fit.vectors,vec,3,("bank",))
        ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),xlabel="PC1",ylabel="PC2",title=f"Additional bank sentences: {ratios[frame]:.0%} financial · one static bank row")
        ax.grid(alpha=.2)
        decorate(fig,"Changing sense frequency · independently trained fixed-seed models",f"One common PCA across all fits; variance {sum(evr):.1%}. Base support includes both senses. Neighbors: "+", ".join(w for w,_ in neighbors)+". No midpoint assumption.")
    export_gif(path,fig,update,len(models),fps=1)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo",choices=DEMO_NAMES,default="all")
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument("--quick",action="store_true")
    modes.add_argument("--extended",action="store_true")
    parser.add_argument("--seed",type=int,default=65)
    parser.add_argument("--corpus-size",type=int)
    parser.add_argument("--repetitions",type=int,default=1)
    parser.add_argument("--epochs",type=int)
    parser.add_argument("--window",type=int,default=3)
    parser.add_argument("--dimension",type=int,default=10)
    parser.add_argument("--output-dir",type=Path,default=Path(__file__).parent/"outputs")
    parser.add_argument("--no-show",action="store_true",help="Explicit headless mode (exports are always nonblocking)")
    args=parser.parse_args(argv)
    if args.dimension < 2 or args.window < 1 or args.repetitions < 1 or args.seed < 0:
        parser.error("dimension >= 2, window >= 1, repetitions >= 1 and seed >= 0 are required")
    if args.corpus_size is not None and args.corpus_size < 1 or args.epochs is not None and args.epochs < 1:
        parser.error("corpus-size and epochs must be positive")
    logging.basicConfig(level=logging.INFO,format="%(message)s")
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root=args.output_dir/stamp
    for sub in ("figures","gifs","html","results"):
        (root/sub).mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False,"axes.titlesize":12,"figure.facecolor":"white"})
    epochs=args.epochs if args.epochs is not None else 80 if args.quick else 400 if args.extended else 240
    size=args.corpus_size if args.corpus_size is not None else 50 if args.quick else 500 if args.extended else 180
    corpus=make_corpus(size,args.repetitions,args.seed)
    words,_,_,pairs,counts=training_data(corpus,args.window)
    frequencies=Counter(w for sentence in corpus for w in sentence)
    report={"created_at_utc":datetime.now(timezone.utc).isoformat(),"configuration":vars(args)|{"epochs":epochs,"corpus_size":size},
            "environment":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"matplotlib":matplotlib.__version__,"pillow":PIL.__version__,"plotly":plotly.__version__,"sklearn":sklearn.__version__},
            "data":{"source":"code-defined synthetic grammatical templates","sentences":len(corpus),"tokens":sum(map(len,corpus)),"vocabulary":len(words),"positive_pairs":len(pairs),"frequencies":dict(frequencies),
                    "sha256":hashlib.sha256(json.dumps(corpus,separators=(",",":")).encode()).hexdigest()},
            "hypothesis":"Recorded context objectives can fit; geometry, analogy and judged retrieval quality require independent diagnostics.",
            "interpretation":"author review required","limitations":"Synthetic categories and judgments; small CPU configurations; different objectives and representation policies; no natural-language benchmark or production validation.","assets":[]}
    models={}
    def wanted(*names):
        return args.demo=="all" or args.demo in names
    def model(mode):
        if mode not in models:
            LOG.info("Training %s: dim=%d window=%d epochs=%d",mode,args.dimension,args.window,epochs)
            models[mode]=train_model(corpus,mode,dim=args.dimension,window=args.window,epochs=epochs,seed=args.seed)
        return models[mode]
    def asset(name):
        sub="figures" if name.endswith(".png") else "gifs" if name.endswith(".gif") else "html"
        path=root/sub/name
        report["assets"].append(path.relative_to(root).as_posix())
        LOG.info("Export %s",name)
        return path
    ids=display_ids(words,12)
    names=[words[i] for i in ids]
    if wanted("representations"):
        trained=model("cbow").fit.vectors
        rng=np.random.default_rng(args.seed)
        random=rng.normal(0,.1,trained.shape)
        selected=ids[:8]; labels=[words[i] for i in selected]
        matrices=[np.eye(len(words))[np.ix_(selected,selected)],counts[np.ix_(selected,selected)],random[selected],trained[selected]]
        titles=["One-hot (selected identity columns)","Sparse word-context counts","Random dense initialization","Trained CBOW dense rows"]
        fig,axes=plt.subplots(2,2,figsize=(14,9))
        for ax,data,title in zip(axes.flat,matrices,titles):
            matrix_plot(ax,data,labels,f"{title} · {data.shape} · zeros {np.mean(data==0):.0%}",labels if data.shape[1]==len(selected) else None,values=True)
        similarity=cosine(trained[words.index("doctor")],trained[words.index("physician")])
        a,b=words.index("doctor"),words.index("physician")
        report["representation_controls"]={"onehot_cosine":0.,"count_cosine":cosine(counts[a],counts[b]),"random_cosine":cosine(random[a],random[b]),"trained_cosine":similarity,"count_sparsity":float(np.mean(counts==0))}
        decorate(fig,"From discrete identities to learned representations",f"Full one-hot/count shapes: {len(words)}×{len(words)}; dense: {trained.shape}. Full count sparsity: {np.mean(counts==0):.1%}.\nDoctor/physician cosine: one-hot=0; counts={cosine(counts[a],counts[b]):.3f}; random={cosine(random[a],random[b]):.3f}; trained={similarity:.3f}.")
        export_png(asset("01_one_hot_vs_dense_embeddings.png"),fig)
    if wanted("window"):
        window_animation(asset("02_sliding_context_window.gif"),args.window)
    if wanted("cbow"):
        architecture_animation(asset("03_cbow_architecture.gif"),"cbow",args.seed)
        evolution(asset("08_cbow_training_evolution.gif"),model("cbow"))
        embedding_space(asset("10_cbow_embeddings_3d.html"),model("cbow"),frequencies)
    if wanted("skipgram"):
        architecture_animation(asset("04_skipgram_architecture.gif"),"skipgram",args.seed)
        evolution(asset("09_skipgram_training_evolution.gif"),model("skipgram"))
        embedding_space(asset("11_skipgram_embeddings_3d.html"),model("skipgram"),frequencies)
    if wanted("negative_sampling"):
        evolution(asset("44_sgns_training_evolution.gif"),model("sgns"))
        embedding_space(asset("43_sgns_embeddings_3d.html"),model("sgns"),frequencies,note="Actual SGNS, with fixed-noise diagnostic loss and multiple minibatch updates per epoch; budgets differ from full-batch softmax.")
        trace=negative_trace(args.seed)
        sgns_animation(asset("05_negative_sampling_dynamics.gif"),trace)
        report["negative_sampling_trace"]=trace
        z=np.linspace(-8,8,300)
        fig,axes=plt.subplots(1,3,figsize=(14,5))
        axes[0].plot(z,sigmoid(z));axes[0].set(title="Sigmoid",ylabel="Binary score")
        axes[1].plot(z,np.logaddexp(0,-z),label="positive: softplus(-z)");axes[1].plot(z,np.logaddexp(0,z),label="noise: softplus(z)");axes[1].set(title="Stable loss terms");axes[1].legend()
        axes[2].plot(z,sigmoid(z)-1,label="positive gradient");axes[2].plot(z,sigmoid(z),label="noise gradient");axes[2].set(title="Derivative with respect to z");axes[2].legend()
        for ax in axes: ax.set_xlabel("Dot product z");ax.grid(alpha=.2)
        decorate(fig,"SGNS is binary discrimination, not normalized softmax","Scores do not measure semantic equivalence. Positive pairs optimize larger dot products; cosine and magnitude can change differently.")
        export_png(asset("06_negative_sampling_loss.png"),fig)
        freq=np.array([frequencies[w] for w in words]);top=np.argsort(-freq)[:18]
        fig,ax=plt.subplots(figsize=(13,5));x=np.arange(len(top))
        ax.bar(x-.2,noise_distribution(freq,1)[top],.4,label="unigram exponent 1");ax.bar(x+.2,noise_distribution(freq,.75)[top],.4,label="noise exponent .75")
        ax.set_xticks(x,[words[i] for i in top],rotation=45,ha="right");ax.legend();ax.set_ylabel("Probability over complete vocabulary")
        decorate(fig,"What changes when unigram noise is smoothed?","Shown: 18 most frequent entries; both complete distributions sum to one. Low-frequency entries receive relatively more mass after smoothing.")
        export_png(asset("07_negative_sampling_distribution.png"),fig)
    if wanted("pmi"):
        matrices=association_matrices(counts)
        for filename,keys,titles in [("12_cooccurrence_matrix.png",("counts","log_counts"),("Distance-weighted raw counts","Log observed counts")),("13_conditional_probabilities.png",("conditional",),("P(context | center)",)),("14_pmi_ppmi_heatmaps.png",("pmi","ppmi"),("PMI · unobserved entries masked","PPMI · unobserved entries become zero"))]:
            fig,axes=plt.subplots(1,len(keys),figsize=(max(10,7*len(keys)),6),squeeze=False)
            for ax,key,title in zip(axes.flat,keys,titles): matrix_plot(ax,matrices[key][np.ix_(ids,ids)],names,title,names)
            decorate(fig,"Distributional association from real synthetic-corpus counts","Statistics use full vocabulary marginals; displayed subset only. Gray entries are not finite PMI values. PPMI uses the conventional zero truncation.")
            export_png(asset(filename),fig)
        result=pmi_experiment(args.seed);report["pmi_experiment"]=result
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for ax,row in zip(axes,result):
            ax.scatter(row["target"].ravel(),row["learned"].ravel(),s=65)
            bounds=[min(row["target"].min(),row["learned"].min()),max(row["target"].max(),row["learned"].max())]
            ax.plot(bounds,bounds,"--",color="#94a3b8");ax.set(title=f"q exponent {row['exponent']} · RMSE {row['rmse']:.3f}",xlabel="log p(c|w) / (K q(c))",ylabel="Learned center/context dot product")
        decorate(fig,"Shifted PMI and generalized SGNS association scores","Controlled 3×3 positive counts; K=5; rank 2; 1,200 gradient steps. Exact expected noise objective, no sampling. Unigram exponent 1 gives PMI-log(K); .75 uses generalized target.")
        export_png(asset("15_shifted_pmi_vs_learned_dot_products.png"),fig)
    if wanted("glove"):
        m=model("glove")
        fig,ax=plt.subplots(figsize=(10,5));x=np.linspace(.001,100,400)
        for xmax,alpha in ((10,.75),(100,.75),(10,.5)):
            ax.plot(x,np.minimum(x/xmax,1)**alpha,label=f"x_max={xmax}, alpha={alpha}")
        ax.legend();ax.set(xlabel="Observed co-occurrence x",ylabel="Weight f(x)")
        decorate(fig,"GloVe weights: downweight rare observed pairs, cap frequent ones","Zero cells are omitted from the objective. This demonstration uses x_max=10 and alpha=.75.")
        export_png(asset("16_glove_weighting_function.png"),fig)
        matrix_reconstruction(asset("17_glove_matrix_reconstruction.gif"),m,counts,ids)
        state=m.snapshots[-1];v,u,b,c=state.parameters
        target=association_matrices(counts)["log_counts"];reconstruction=v@u.T+b[:,None]+c[None,:];reconstruction[counts==0]=np.nan
        fig,axes=plt.subplots(1,3,figsize=(16,6))
        for ax,data,title in zip(axes,(target,reconstruction,reconstruction-target),("Target log counts","Reconstructed observed cells","Residual (unweighted)")):
            matrix_plot(ax,data[np.ix_(ids,ids)],names,title,names,cmap="coolwarm")
        decorate(fig,"Inspect GloVe's observed-cell reconstruction","Gray cells did not enter the objective. Residual visualization is unweighted; the optimized loss additionally applies f(x). Subset shown.")
        export_png(asset("18_glove_reconstruction_residuals.png"),fig)
        fig,ax=plt.subplots(figsize=(10,5));ax.plot(m.fit.losses);ax.set(xlabel="Full-batch gradient updates",ylabel="Mean weighted squared log-count residual")
        decorate(fig,"Actual GloVe fitting curve","Batch gradient descent, not original AdaGrad. This scale must not be compared with Word2Vec NLL or SGNS loss.")
        export_png(asset("19_glove_training_loss.png"),fig)
        evolution(asset("20_glove_embedding_evolution.gif"),m)
        embedding_space(asset("21_glove_embeddings_3d.html"),m,frequencies,note="Final representation is center + context vectors.")
    if wanted("cosine"):
        cosine_explorer(asset("22_cosine_similarity_interactive.html"))
        for number,mode in ((23,"cbow"),(24,"skipgram"),(25,"glove")):
            fit=model(mode).fit;unit=unit_rows(fit.vectors[ids]);fig,ax=plt.subplots(figsize=(9,8))
            matrix_plot(ax,unit@unit.T,names,f"{mode.upper()} · original-space cosine",names,(-1,1),cmap="coolwarm")
            decorate(fig,"Same angular scale, different learned representations","Cosine scale fixed at [-1, 1]. Diagonal self-similarity is 1. Similarity is not a probability or a guarantee of synonymy.")
            export_png(asset(f"{number:02d}_{mode}_similarity_heatmap.png"),fig)
        fig,axes=plt.subplots(1,3,figsize=(16,6))
        for ax,mode in zip(axes,("cbow","skipgram","glove")):
            unit=unit_rows(model(mode).fit.vectors[ids])
            matrix_plot(ax,unit@unit.T,names,mode.upper(),names,(-1,1),cmap="coolwarm")
        decorate(fig,"Compare actual original-dimensional cosine matrices","Same selected words, fixed [-1,1] scale. Predictive vectors use input tables; GloVe uses center+context. A visually coherent pattern is not a semantic benchmark.")
        export_png(asset("42_similarity_comparison.png"),fig)
    if wanted("analogies"):
        analogy_animation(asset("26_idealized_vector_analogy.gif"))
        analogy_animation(asset("27_learned_vector_analogy.gif"),model("skipgram"))
        analogy_explorer(asset("28_vector_analogies_3d.html"),model("skipgram"))
    if wanted("polysemy"):
        embedding_space(asset("29_polysemy_embedding_space.html"),model("skipgram"),frequencies,title="Bank: one static row, multiple contexts",note="One bank row represents every occurrence; the space does not disambiguate a query's sense.",focus="bank")
        ratios=[0,.25,.5,.75,1.]
        bank_models=[train_model(make_corpus(size,args.repetitions,args.seed,r),"skipgram",dim=args.dimension,window=args.window,epochs=epochs,seed=args.seed) for r in ratios]
        polysemy_animation(asset("30_polysemy_frequency_experiment.gif"),bank_models,ratios)
        report["polysemy"]=[]
        for r,m in zip(ratios,bank_models):
            local=make_corpus(size,args.repetitions,args.seed,r)
            financial=sum("bank" in sentence and any(w in sentence for w in ("loan","deposited","finance")) for sentence in local)
            total=sum("bank" in sentence for sentence in local)
            report["polysemy"].append({"additional_finance_ratio":r,"actual_financial_bank_sentences":financial,"total_bank_sentences":total,"actual_financial_sentence_fraction":financial/total,
                                      "bank_neighbors":nearest(m.fit.words,m.fit.vectors,m.fit.vectors[m.fit.words.index("bank")],5,("bank",)),"configuration":m.configuration})
        np.savez_compressed(root/"results"/"polysemy_fits.npz",words=np.array(words),ratios=np.array(ratios),vectors=np.stack([m.fit.vectors for m in bank_models]))
        fig,ax=plt.subplots(figsize=(11,5));x=np.arange(3)
        report["antonym_cosines"]={}
        for offset,(a,b) in zip((-.18,.18),(("hot","cold"),("large","small"))):
            scores=[]
            for mode in ("cbow","skipgram","glove"):
                fit=model(mode).fit;scores.append(cosine(fit.vectors[words.index(a)],fit.vectors[words.index(b)]))
            report["antonym_cosines"][f"{a}/{b}"]=dict(zip(("cbow","skipgram","glove"),scores))
            ax.bar(x+offset,scores,.36,label=f"{a} / {b}")
        ax.set_xticks(x,["CBOW","Skip-gram","GloVe"]);ax.set_ylim(-1,1);ax.legend();ax.set_ylabel("Original-dimensional cosine")
        decorate(fig,"Antonyms can share context distributions","Actual cosine measurements on synthetic shared templates. High cosine does not mean logical equivalence; no antonym rejection rule is learned.")
        export_png(asset("31_antonym_similarity.png"),fig)
        fit=model("skipgram").fit
        order,_=mean_vectors(["the company complied with regulations","regulations with complied company the"],fit)
        neg,_=mean_vectors(["the company complied with regulations","the company did not comply with regulations"],fit)
        report["limitations_controls"]={"order_permutation_cosine":cosine(*order),"negation_pair_cosine":cosine(*neg),"oov":"kubernetes not in vocabulary","bank_rows":sum(w=="bank" for w in words)}
        fig,(ax,info)=plt.subplots(1,2,figsize=(14,6));ax.axis("off");info.axis("off")
        ax.text(.02,.95,"Static lookup (executed)",fontsize=15,weight="bold",va="top")
        ax.text(.02,.8,"financial bank ─┐\n               ├─ same bank row\nriver bank ────┘",fontsize=14,family="monospace",va="top")
        ax.text(.02,.4,f"Order permutation cosine: {cosine(*order):.6f}\nNegation pair cosine: {cosine(*neg):.6f}\nOOV: explicit coverage loss / abstention",fontsize=12,va="top")
        info.text(.02,.95,"Contextual encoder (conceptual)",fontsize=15,weight="bold",va="top")
        info.text(.02,.8,"tokens + surrounding sequence\n            ↓\ncontext-dependent hidden states\n            ↓\noptional pooling / retrieval objective",fontsize=14,va="top")
        info.text(.02,.35,"No contextual model was downloaded or run.\nNo fabricated contextual vectors are displayed.\nA token state is not automatically a sentence encoder.",fontsize=12,va="top")
        decorate(fig,"Static limitations and a conceptual contextual alternative","Order permutation preserves the arithmetic mean. The negation comparison also changes inflection (comply/complied); its cosine is a diagnostic, not contradiction detection.")
        export_png(asset("32_static_vs_contextual_embeddings.png"),fig)
    if wanted("experiments"):
        rows=sweep_experiments(corpus,args.seed,epochs=max(8,epochs//2),extended=args.extended)
        report["sweeps"]=rows
        pd.DataFrame(rows).to_csv(root/"results"/"hyperparameters.csv",index=False)
        def sweep_figure(group,filename,label):
            data=pd.DataFrame([r for r in rows if r["group"]==group])
            fig,axes=plt.subplots(1,2,figsize=(12,5))
            grouped=data.groupby("value",sort=False)
            labels=list(grouped.groups)
            for ax,metric in zip(axes,("purity_at3","analogy_accuracy")):
                means=grouped[metric].mean().reindex(labels);std=grouped[metric].std().fillna(0).reindex(labels)
                ax.bar(range(len(labels)),means,yerr=std,capsize=4,color="#2563eb")
                ax.set_xticks(range(len(labels)),[str(x) for x in labels]);ax.set_ylim(0,1.08);ax.set(title=metric,xlabel=label)
            decorate(fig,"Measured synthetic diagnostics · "+group,"Entire vocabulary competes for neighbors. Categories and two analogy answers are predefined synthetic fixtures. Architecture bars repeat seeds; error bars are sample SD, not confidence intervals.")
            export_png(asset(filename),fig)
        sweep_figure("architecture","33_architecture_comparison.png","Architecture")
        sweep_figure("window","34_window_size_comparison.png","Fixed window radius")
        sweep_figure("dimension","35_embedding_dimension_comparison.png","Dimension")
        sweep_figure("negative","36_negative_sampling_comparison.png","SGNS noise samples K")
        architecture=pd.DataFrame([r for r in rows if r["group"]=="architecture"])
        fig,axes=plt.subplots(1,3,figsize=(16,5))
        architecture.groupby("mode")[["purity_at3","rare_purity_at3"]].mean().plot.bar(ax=axes[0],ylim=(0,1),rot=0)
        extra=pd.DataFrame([r for r in rows if r["group"] in ("corpus_size","epochs")])
        for ax,group in zip(axes[1:],("corpus_size","epochs")):
            subset=extra[extra.group==group];ax.plot(subset.value,subset.rare_purity_at3,"o-")
            ax.set(xlabel="Actual sentence count" if group=="corpus_size" else "Epochs",ylabel="Rare-word purity@3",ylim=(-.03,1.03))
        decorate(fig,"Rare vocabulary and additional training controls","surgeon and architect appear once per generated collection before repetitions. Corpus-size and epoch controls use separate axes. No general rare-word advantage is assumed.")
        export_png(asset("37_rare_word_experiment.png"),fig)
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for mode in ("cbow","skipgram","glove","sgns"):
            chosen=[r for r in rows if r["mode"]==mode]
            axes[0].scatter([r["seconds"] for r in chosen],[r["purity_at3"] for r in chosen],label=mode)
        axes[0].legend();axes[0].set(xlabel="Measured local training seconds",ylabel="Synthetic purity@3")
        dimrows=[r for r in rows if r["group"]=="dimension"]
        axes[1].plot([r["dim"] for r in dimrows],[r["two_training_tables_bytes"]/1024 for r in dimrows],"o-")
        axes[1].set(xlabel="Embedding dimension",ylabel="Two raw float64 tables (KiB)")
        decorate(fig,"Quality, local computation and raw parameter storage","Timings include training preparation; one local environment, no matched tuning budget. Raw table bytes exclude vocabulary, activations, dense features, biases and optimizer state.")
        export_png(asset("38_quality_vs_computational_cost.png"),fig)
    if wanted("retrieval"):
        retrieval=retrieval_experiment({mode:model(mode) for mode in ("skipgram","glove")})
        report["retrieval"]=retrieval
        fig,ax=plt.subplots(figsize=(12,5));pd.DataFrame(retrieval["summary"]).T.plot.bar(ax=ax,ylim=(0,1.08),rot=15)
        decorate(fig,"TF-IDF versus averaged static embeddings · measured judged retrieval","Eight synthetic documents, six queries, manually assigned grades. Recall@3, full-ranking reciprocal rank and nDCG@3; fully OOV queries abstain. No real-world superiority claim.")
        export_png(asset("39_semantic_retrieval_comparison.png"),fig)
        retrieval_space(asset("40_document_query_embedding_space.html"),retrieval)
        fig,axes=plt.subplots(1,3,figsize=(16,6))
        for ax,mode in zip(axes,("tfidf","skipgram","glove")):
            matrix_plot(ax,retrieval["scores"][mode], [f"Q{i}: {q[2]}" for i,q in enumerate(QUERIES)],mode,[f"D{i}" for i in range(len(DOCUMENTS))],(-1,1),cmap="coolwarm")
        decorate(fig,"Inspect every score, including negation, polysemy and OOV","Manual relevant IDs: Q0→D0; Q1→D2 (D3 contradicts); Q2→D5; Q3→D1; Q4→D6 grade2,D7 grade1; Q5→D1 but OOV. A score is not logical entailment.")
        export_png(asset("41_retrieval_failure_cases.png"),fig)
    report["models"]={mode:{"configuration":m.configuration,"diagnostics":diagnostics(m,frequencies),
                             "loss_definition":"fixed-noise mean SGNS diagnostic" if mode=="sgns" else "mean weighted observed log-count residual" if mode=="glove" else "full-softmax mean NLL",
                             "pca_animation_explained_variance":fixed_projection(snapshot_vectors(m))[1],
                             "pca_final_3d_explained_variance":fixed_projection([m.fit.vectors],3)[1],
                             "initial_loss":m.fit.losses[0],"final_loss":m.fit.losses[-1],"losses":m.fit.losses,
                             "analogy_rankings":[{"relation":[a,b,c],"expected":answer,"candidates":analogy(m.fit,a,b,c,5)} for a,b,c,answer in ANALOGIES]} for mode,m in models.items()}
    for mode,m in models.items():
        parameters={f"parameter_{i}":np.stack([s.parameters[i] for s in m.snapshots]) for i in range(len(m.snapshots[0].parameters))}
        np.savez_compressed(root/"results"/f"{mode}_snapshots.npz",words=np.array(m.fit.words),vectors=np.stack(snapshot_vectors(m)),epochs=np.array([s.epoch for s in m.snapshots]),losses=m.fit.losses,**parameters)
    def encode(value):
        if isinstance(value,np.ndarray): return value.tolist()
        if isinstance(value,np.generic): return value.item()
        if isinstance(value,Path): return value.as_posix()
        raise TypeError(type(value).__name__)
    for relative in report["assets"]:
        path=root/relative
        if not path.is_file(): raise RuntimeError(f"missing artifact: {path}")
    (root/"results"/"run.json").write_text(json.dumps(report,indent=2,allow_nan=False,default=encode)+"\n",encoding="utf-8")
    links="\n".join(f'<li><a href="{escape(relative)}">{escape(Path(relative).name.replace("_"," "))}</a> <small>{Path(relative).suffix[1:].upper()}</small></li>' for relative in sorted(report["assets"],key=lambda name:Path(name).name))
    (root/"index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Day 65 visual laboratory</title><style>body{font:17px system-ui;background:#f1f5f9;color:#172033;max-width:1100px;margin:30px auto;padding:20px}li{padding:10px;background:white;margin:6px;border-radius:8px}a{color:#1d4ed8}small{color:#64748b}</style><h1>Day 65 · Static embeddings visual lab</h1><p>Start with 02 (windows), 03–04 (full softmax), 05 (SGNS updates), 08–09 and 44 (recorded evolution), 17 (GloVe reconstruction), then explore analogies and judged retrieval. All data are synthetic. Interpretations require author review.</p><p>GIFs show recorded states; HTML explorers work offline. JSON/CSV/NPZ results are in the results folder. These artifacts do not establish benchmark or production performance.</p><ol>'+links+'</ol></html>',encoding="utf-8")
    LOG.info("Completed %d artifacts. Run record: %s",len(report["assets"]),root/"results"/"run.json")
    return root


if __name__=="__main__":
    main()

"""Generate the Day 49 offline causal learning lab and deterministic local GIFs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
import platform
import time

# Bound BLAS work and use file rendering; no GUI/server or GPU is required.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image
import plotly
from plotly.offline import get_plotlyjs
import sklearn

from causal_visual_core import (ADJUSTMENTS, COLORS, MODULES, POSITIONS, ROLES, TITLES,
    bias_study, collider_selection, graph_payload, model, population_values,
    random_tape, summary, validate_graph_algorithm)


HERE = Path(__file__).resolve().parent
LESSONS = {
    "dag": (
        "Which controls block confounding without blocking the total effect?",
        "Choose a path: green means active causal, amber active noncausal and red blocked. The table explains each state. A dark node outline means conditioned.",
        "Change the adjustment set and inspect every path, including the collider path.",
        "Active path: all noncolliders outside Z; every collider has itself or a descendant in Z. Original backdoor: no treatment descendants, all incoming paths blocked.",
        "A plausible RAG DAG may include difficulty → routing and difficulty → answer quality. Validate those assumptions before selecting controls."),
    "confounding": (
        "Why is the observed difference near 2.9 when the known total effect is 2.0?",
        "Treated cases contain more high-complexity cases in this SCM. Compare outcomes within C before averaging with a common target distribution.",
        "Reveal complexity, change the seed and inspect within-stratum distributions and weighted effect markers.",
        "ATE = Σc [E(Y|A=1,C=c) − E(Y|A=0,C=c)] P(C=c); observational difference = 2 + 1.5 × 0.6.",
        "GenAI adoption may favor simple documents in another product. This example intentionally assigns AI more often to complex cases; the bias direction must follow the assumed mechanism."),
    "collider": (
        "How can observing feedback create a relationship that was absent before selection?",
        "A → S ← U → Y is the mechanism of interest. The full SCM also contains C and M. Compare full and selected A–U distributions; U is an oracle variable visible only in simulation.",
        "Change S noise, interval center, half-width, selection rule, sample size and seed. A high-S threshold is a separate rule from a narrow interval.",
        "S = A + U + σS εS. At nearly fixed S, A=1 tends to require lower U. Full-sample OLS(C,S) approaches 2 − 1.2/(1+σS²).",
        "If flags and user difficulties both affect recorded feedback, feedback-only evaluation may introduce selection bias. The selection mechanism requires validation."),
    "mediation": (
        "Which part of the total effect disappears when we control adoption?",
        "Follow the highlighted direct or mediated path; compare structural bars with estimated OLS diamonds. The smaller diagram is a motif from the full DAG.",
        "Change A → M and M → Y coefficients, then animate the two pathways.",
        "Total = direct + indirect = 1.3 + αAM × βMY, only under the specified additive linear SCM.",
        "Retrieval quality may mediate a reranker’s end-to-end effect. Controlling it changes the question; a direct-effect interpretation needs additional assumptions."),
    "intervention": (
        "What changes when we set A instead of selecting cases that received A?",
        "The observed panel selects a naturally assigned subset. The intervention panel uses every case, replaces the assignment equation and regenerates M and Y. Axes are synchronized.",
        "Switch A between disabled and enabled; compare both observed and interventional group means.",
        "Observe: E[Y|A=a]. Intervene: E[Y(a)]. Here population means are (0.3, 3.2) versus (0.75, 2.75).",
        "High-risk customers may receive retention actions more often. The treated population can differ systematically from the population under an intervention."),
    "geometry": (
        "How does group composition distort a marginal comparison?",
        "Four discrete cells use actual binary A and C. Small jitter separates observations visually; hover reports the original categories. Diamonds are known conditional means.",
        "Rotate and zoom; toggle A groups, C strata, expected markers and within-stratum contrasts.",
        "E[Y|A,C] = 2A + 1.5C. Within either C, the expected contrast is 2. No continuous response surface is fitted.",
        "A 3D view can expose composition imbalance, but showing a surface between binary categories would add an unsupported continuous interpretation."),
    "adjustment": (
        "Why can more regression controls make a causal estimate worse?",
        "Compare four specifications on the same sample. Below, remove the intercept and linear C from both A and Y: the residual slope equals joint OLS(C).",
        "Vary C → Y strength and A → M; inspect the estimates and FWL agreement.",
        "Omitted-variable bias = βC Cov(A,C)/Var(A) = 0.6βC. FWL = (rAᵀrY)/(rAᵀrA).",
        "Predictive feature selection cannot decide an adjustment set. FWL is an algebraic identity and remains true with causally inappropriate controls."),
    "sample-size": (
        "Does a larger sample repair structural bias?",
        "Error bars describe the sampling distribution of point estimates. They shrink around different limits: 2.9, 2.0 and 1.04. Ten-to-ninety percent ranges are not causal confidence intervals.",
        "Choose SD or empirical sampling ranges and compare reproducible seed banks. Generate more repetitions with the Python CLI when desired.",
        "Sampling variance can fall with n while bias persists. MCSE(mean) = SD / √repetitions; consistency depends on identification and specification.",
        "More observational product logs do not repair confounding or collider adjustment. Revisit the design and assumptions before increasing estimator complexity."),
}


def select_control(id_, label, choices):
    return f'<label>{escape(label)}<select id="{id_}">' + ''.join(
        f'<option value="{escape(str(value))}">{escape(str(text))}</option>' for value, text in choices) + '</select></label>'


def slider(id_, label, low, high, step, value):
    return (f'<label for="{id_}">{escape(label)}: <output id="{id_}-value">{value}</output>'
            f'<input id="{id_}" type="range" min="{low}" max="{high}" step="{step}" value="{value}"></label>')


def checkbox(id_, label, checked=True):
    return f'<label class="check"><input id="{id_}" type="checkbox" {"checked" if checked else ""}>{escape(label)}</label>'


def module_controls(module, graph):
    if module == "dag":
        paths = graph["cases"]["None"]["paths"]
        return select_control("adjust", "Adjustment set", [(s, s) for s in ADJUSTMENTS]) + select_control(
            "path", "Inspect path", [(i, p["label"]) for i, p in enumerate(paths)])
    if module == "confounding":
        return checkbox("reveal", "Reveal C", False)
    if module == "collider":
        return (slider("noise", "S noise SD", .05, 2, .05, .5)
                + slider("center", "S interval center / high-S threshold", -2, 3, .1, .5)
                + slider("width", "Interval half-width (interval rule only)", .05, 2, .05, .35)
                + select_control("mode", "Selection rule", [("interval", "Narrow interval"), ("high", "High-S threshold")]))
    if module == "mediation":
        return (slider("alpha", "A → M coefficient", -1, 1.5, .1, .7)
                + slider("beta", "M → Y coefficient", -1, 2, .1, 1)
                + '<button id="play" type="button">Animate paths</button>')
    if module == "intervention":
        return select_control("treatment", "Treatment value", [(0, "A = 0 · disabled"), (1, "A = 1 · enabled")])
    if module == "geometry":
        return ''.join(checkbox(id_, label) for id_, label in [("a0", "Show A=0"), ("a1", "Show A=1"),
            ("c0", "Show C=0"), ("c1", "Show C=1"), ("expected", "Expected means"), ("comparisons", "Within-stratum contrasts")])
    if module == "adjustment":
        return slider("confounding", "C → Y coefficient", -2, 3, .1, 1.5) + slider("alpha", "A → M coefficient", -1, 1.5, .1, .7)
    return select_control("range", "Sampling variability", [("range", "Empirical 10th–90th percentiles"), ("sd", "±1 sample SD")])


STYLE = """
*{box-sizing:border-box}body{margin:0;background:#eef2f6;color:#192d42;font:16px/1.6 Arial,sans-serif}
main{max-width:1280px;margin:auto;padding:22px}h1{font-size:30px;line-height:1.25;margin:10px 0}h2{font-size:20px;margin:8px 0}
.tag{font-size:13px;font-weight:bold;letter-spacing:.08em;color:#2563a6}.card{background:white;border:1px solid #d3dce6;border-radius:10px;padding:20px;margin:16px 0}
.intro{display:grid;grid-template-columns:1.4fr 1fr;gap:24px}.controls{display:flex;gap:18px;flex-wrap:wrap;align-items:center}
label{display:flex;flex-direction:column;font-size:14px;font-weight:bold;min-width:150px}label.check{flex-direction:row;gap:8px;align-items:center;min-width:auto}
select,button{background:white;color:#192d42;border:1px solid #8394a6;padding:9px;border-radius:5px;font-size:14px}button{cursor:pointer}
input[type=range]{width:190px}input:focus,select:focus,button:focus,a:focus{outline:3px solid #7954a1;outline-offset:3px}
.chart{height:480px;width:100%}#detail{height:370px}#metrics{background:#e9f4ef;border-left:5px solid #087e64;padding:15px;font-weight:bold}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #d3dce6;padding:10px;text-align:left}th{background:#f2f5f9}
nav{display:flex;flex-wrap:wrap;gap:10px}a{color:#174f88}.formula{font-family:Consolas,monospace;background:#f5f7fa;padding:12px}
.small{font-size:13px;color:#4a5b6c}iframe{width:100%;height:1700px;border:1px solid #c5cfdb;border-radius:10px;background:white}
@media(max-width:800px){.intro{grid-template-columns:1fr}main{padding:10px}.card{padding:12px}.chart{height:480px}table{display:block;overflow-x:auto}}
"""


def write_module(destination: Path, module: str, payload: dict, bundle: str) -> None:
    question, notice, interaction, formula, takeaway = LESSONS[module]
    i = MODULES.index(module)
    data = {**payload, "module": module}
    controls = select_control("seed", "Synthetic seed", [(k, k) for k in payload["tapes"]])
    if module in ("dag", "sample-size"):
        controls += f'<input type="hidden" id="n" value="{payload["max_samples"]}">'
    else:
        controls += slider("n", "Cases (prefix of fixed tape)", 500, payload["max_samples"], 100, payload["max_samples"])
    controls += module_controls(module, payload["graph"])
    encoded = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Day 49 · {escape(TITLES[i])}</title><style>{STYLE}</style></head><body><main>
<div class="tag">DAY 49 · SYNTHETIC CAUSAL LAB · MODULE {i+1}/8</div><h1>{escape(TITLES[i])}</h1>
<a href="index.html">All eight modules</a>
<div class="card intro"><section><h2>Question</h2><p>{escape(question)}</p><h2>Visual explanation</h2><p>{escape(notice)}</p></section>
<section><h2>Interaction</h2><p>{escape(interaction)}</p><h2>Senior-level takeaway</h2><p>{escape(takeaway)}</p></section></div>
<div class="card controls">{controls}</div><p class="small">Controls apply on release/change. Points may be thinned for display; estimates use all selected cases. Four offline seed tapes; use --seed to generate another bank.</p>
<div id="metrics" role="status" aria-live="polite">Computing…</div>
<div class="card"><div id="chart" class="chart"></div></div><div class="card"><div id="detail" class="chart"></div><div id="contrast" class="chart" style="display:none"></div><div id="table"></div></div>
<div class="card"><h2>Mathematical connection</h2><p class="formula">{escape(formula)}</p>
<details><summary>Shared SCM and evidence boundary</summary><p>C ~ Bernoulli(0.5); P(A=1|C)=0.2+0.6C; U ~ N(0,1); M=0.7A+εM; S=A+U+0.5εS; Y=1.3A+M+1.5C+1.2U+εY. All standardized disturbances are independent. Parameter controls explicitly alter these coefficients locally.</p>
<p>Baseline known total effect 2.0, direct 1.3, naive population difference 2.9. These are constructed truths. U and potential outcomes are simulator-only oracles. No product effect or author approval is established.</p></details></div>
<p class="small">Offline HTML · no server, CDN or external data · interpretation review required · keyboard-accessible native controls</p>
</main><script>{bundle}</script><script>globalThis.LAB={encoded};</script><script>{(HERE/'causal_lab.js').read_text(encoding='utf-8')}</script></body></html>'''
    destination.write_text(html, encoding="utf-8")


def write_dashboard(destination: Path, modules) -> None:
    links = ''.join(f'<a target="lesson" href="{m}.html">{MODULES.index(m)+1}. {escape(TITLES[MODULES.index(m)])}</a>' for m in modules)
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Day 49 · Causal Inference Laboratory</title><style>{STYLE}</style></head>
<body><main><div class="tag">APPLIED AI ENGINEERING LAB · DAY 49</div><h1>From association to intervention</h1>
<p>Eight experiments connect causal assumptions, selection and regression. Start with the DAG, then follow the learning sequence. All cases are synthetic. Open any module in a new tab for a full-width standalone experience.</p>
<div class="card"><nav aria-label="Learning sequence">{links}</nav></div>
<iframe name="lesson" title="Interactive causal inference lesson" src="{modules[0]}.html"></iframe>
<p>Baseline: total effect 2.0; direct path 1.3; observational population difference 2.9. These values describe the constructed SCM, not a real AI deployment. See VISUAL_LAB.md for configurations and limitations.</p></main></body></html>'''
    destination.write_text(html, encoding="utf-8")


def draw_dag(ax, *, edges=None, path=(), title="Assumed SCM · eight edges", position=None, highlight_edges=None):
    graph = nx.DiGraph(graph_payload()["edges"] if edges is None else edges)
    pos = POSITIONS if position is None else position
    selected = set(zip(path, path[1:])) if highlight_edges is None else set(highlight_edges)
    colors = ["#087e64" if e in selected else "#8394a6" for e in graph.edges]
    nx.draw_networkx(graph, pos=pos, ax=ax, node_color=[COLORS[n] for n in graph.nodes],
        node_size=1600, font_color="white", font_weight="bold", font_size=15,
        edge_color=colors, width=[3.5 if e in selected else 1.8 for e in graph.edges],
        arrowsize=20, connectionstyle="arc3,rad=0.03", min_source_margin=22, min_target_margin=22)
    ax.set_title(title, loc="left", fontsize=15, pad=18)
    ax.set_xlim(-2.4, 2.4); ax.set_ylim(-1.9, 2.05); ax.axis("off")


def gif_frames(kind, d, count):
    """Render directly to in-memory indexed frames, never export every PNG frame."""
    s = summary(d)
    y_limits = (float(d.Y.min())-1, float(d.Y.max())+1)
    points = np.arange(0, len(d), max(1, len(d)//600))
    jitter = .08 * np.sin(np.arange(len(d))*2.399963229728653)
    frames = []
    steps = ["Observe the association", "Reveal case complexity C", "Separate complexity strata",
             "Compare A within each stratum", "Use the target population weights", "Compare adjustment with known truth"]
    for frame in range(count):
        fig, (ax, right) = plt.subplots(1, 2, figsize=(11, 5.8), gridspec_kw={"width_ratios": [1.35, 1]})
        fig.patch.set_facecolor("#f4f7fa")
        for panel in (ax, right):
            panel.set_facecolor("white")
        if kind == "confounding":
            stage = min(5, frame*6//count)
            progress = min(1, max(0, (frame/(count-1)*6-2)))
            shift = 2*progress if stage >= 2 else 0
            x = d.A + shift*d.C + jitter
            ax.scatter(x.iloc[points], d.Y.iloc[points], s=10, alpha=.4,
                c=[COLORS['C'] if d.C.iloc[i] else COLORS['A'] for i in points] if stage else COLORS['A'])
            ax.set(xlim=(-.4, 3.4), ylim=y_limits, ylabel="Performance Y (score units)")
            ax.set_xticks([0, 1, 2, 3], ["A=0\nC=0", "A=1\nC=0", "A=0\nC=1", "A=1\nC=1"] if stage>=2 else ["A=0", "A=1", "", ""])
            if stage >= 1:
                for c in (0,1):
                    ax.scatter([],[],color=COLORS['C'] if c else COLORS['A'],label=f'C={c}')
                ax.legend(loc='upper left',fontsize=9)
            if stage >= 3:
                for c in (0, 1):
                    means = [d.loc[(d.A==a)&(d.C==c), 'Y'].mean() for a in (0, 1)]
                    ax.plot([2*c, 2*c+1], means, "D-", lw=3, color=COLORS['C'] if c else COLORS['A'])
            ax.set_title(steps[stage], fontsize=13, loc="left")
            if stage < 5:
                right.axis("off")
                text = [f"Observed difference\n{s['estimates']['None']:.3f}\n\nAssociation is not intervention.",
                        "Colors reveal C.\nTreatment assignment depends on C.\n\nC also affects Y.",
                        "Move the SAME cases into strata.\nDo not compare different\ncomplexity mixes.",
                        '\n'.join(f"C={v['C']}: within-stratum\ndifference = {v['difference']:.3f}\n" for v in s['strata']),
                        '\n'.join(f"P(C={v['C']})={v['weight']:.3f}" for v in s['strata'])+f"\n\nWeighted difference\n{s['estimates']['Standardized C']:.3f}"][stage]
                right.text(.04, .8, text, va="top", fontsize=13, linespacing=1.6)
            else:
                right.scatter([0, 1], [s['estimates']['None'], s['estimates']['Standardized C']], s=90, color=[COLORS['C'], COLORS['Y']])
                right.axhline(2, color=COLORS['Y'], ls="--", label="Known total = 2.0")
                right.set(xlim=(-.4,1.4),ylim=(.5, 3.5), xticks=[0, 1], xticklabels=["Observed", "Standardized"], ylabel="Score difference")
                right.legend(loc="lower left", fontsize=10)
        elif kind == "collider":
            width = 3 * (.05 / 3)**(frame/(count-1))
            selected = collider_selection(d, center=.5, width=width)
            rows = np.array(selected['indices'], dtype=int)
            ax.scatter((d.A+jitter).iloc[points], d.U.iloc[points], color="#aab4bf", alpha=.25, s=10)
            for a in (0, 1):
                pick = rows[d.A.iloc[rows].to_numpy()==a]
                ax.scatter(d.A.iloc[pick]+jitter[pick], d.U.iloc[pick], s=14, alpha=.65, color=COLORS['A'] if a==0 else COLORS['S'])
                if len(pick):
                    ax.plot([a-.16, a+.16], [d.U.iloc[pick].mean()]*2, color="#142d40", lw=3)
            ax.set(xlim=(-.35, 1.35), ylim=(-4.5, 4.5), xticks=[0, 1], xlabel="A · binary (display jitter)", ylabel="U · simulator-only latent capability")
            ax.set_title("Narrow the selected S interval", loc="left", fontsize=13)
            right.hist(d.S, bins=np.linspace(-4,5,45), color="#c4ccd5", density=True)
            right.axvspan(.5-width, .5+width, color=COLORS['S'], alpha=.4)
            right.set(xlim=(-4,5), ylim=(0,.48), xlabel="S = A + U + 0.5 εS", ylabel="Density")
            delta = selected['delta_u_selected']
            right.set_title(f"|S − 0.5| ≤ {width:.2f}\nSelected n={selected['n']}; ΔU={delta:.2f}" if delta is not None else "Insufficient selected overlap", fontsize=12)
            fig.text(.5,.065,"A → S ← U → Y · fixing S makes A and U trade off. Selection is not continuous S adjustment.",ha="center",fontsize=10)
        else:
            t = frame/(count-1)*3
            stage = min(2, int(t))
            path = ['A','Y'] if stage==0 else ['A','M','Y']
            pos={'A':(-1.7,0),'M':(0,1.3),'Y':(1.7,0)}
            draw_dag(ax,edges=[('A','M'),('M','Y'),('A','Y')],path=path,
                title=["Direct path: 1.3", "Mediated path: 0.7 × 1.0", "Total effect: 1.3 + 0.7 = 2.0"][stage],position=pos,
                highlight_edges=[('A','M'),('M','Y'),('A','Y')] if stage==2 else None)
            segment=(t-stage)* (len(path)-1)
            edge=min(len(path)-2,int(segment)); frac=min(1,segment-edge)
            start,end=np.array(pos[path[edge]]),np.array(pos[path[edge+1]])
            point=start+(end-start)*frac
            ax.scatter(*point,s=150,color="#e3b444",edgecolor="#192d42",zorder=5)
            right.bar(['Direct','Indirect','Total'],[1.3,.7 if stage>=1 else 0,2 if stage>=2 else 0],color=[COLORS['M'],COLORS['C'],COLORS['Y']])
            right.set(ylim=(0,2.4),ylabel="Known structural effect (score units)")
            right.set_title("Baseline additive linear SCM",fontsize=12)
            fig.text(.5,.065,"Schematic path traversal, not elapsed time. OLS(C) targets 2.0; OLS(C,M) targets 1.3 here.",ha="center",fontsize=10)
        fig.suptitle("Day 49 · "+kind.capitalize()+" · synthetic data",fontsize=16,fontweight="bold",y=.98)
        fig.subplots_adjust(left=.08,right=.97,top=.84,bottom=.20,wspace=.3)
        fig.canvas.draw()
        image=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3]).convert('P',palette=Image.Palette.ADAPTIVE,colors=128)
        frames.append(image)
        plt.close(fig)
    return frames


def export_assets(folder: Path, d: pd.DataFrame, fast: bool, modules) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    assets=[]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,"axes.spines.right":False})
    for kind in ("confounding","collider","mediation"):
        if kind not in modules:
            continue
        frames=gif_frames(kind,d,18 if fast else 30)
        file=folder/f'{kind}_explained.gif' if kind!='collider' else folder/'collider_bias_explained.gif'
        # Required stable export names.
        if kind=='confounding': file=folder/'confounding_explained.gif'
        frames[0].save(file,save_all=True,append_images=frames[1:],duration=600 if fast else 450,loop=0,optimize=False,disposal=2)
        for frame in frames: frame.close()
        assets.append(file)
    if "dag" in modules:
        fig,ax=plt.subplots(figsize=(11,6))
        draw_dag(ax,path=['A','M','Y'])
        fig.text(.5,.04,' · '.join(f'{n}: {ROLES[n]}' for n in ['A','Y','C','M','S','U']),ha='center',fontsize=10)
        file=folder/'causal_dag.png'; fig.savefig(file,dpi=120,bbox_inches='tight'); plt.close(fig); assets.append(file)
    if "adjustment" in modules:
        s=summary(d); names=list(ADJUSTMENTS)
        fig,ax=plt.subplots(figsize=(10,5.6))
        ax.scatter(names,[s['estimates'][n] for n in names],s=100,color=[COLORS[n] for n in ['C','Y','M','S']])
        ax.axhline(2,color=COLORS['Y'],ls='--',label='Known total effect = 2.0')
        ax.axhline(1.3,color=COLORS['M'],ls=':',label='Direct path coefficient = 1.3')
        ax.set(xlabel='Regression controls',ylabel='Estimated A coefficient (score units)',title='Same synthetic cases, different adjustment choices')
        ax.legend(); ax.grid(axis='y',alpha=.2)
        file=folder/'treatment_effect_comparison.png'; fig.savefig(file,dpi=120,bbox_inches='tight'); plt.close(fig); assets.append(file)
    return assets


def generate(output: Path, *, seed=49, fast=False, repetitions=None, modules=MODULES,
             dashboard=True, gifs=True) -> dict:
    started=time.perf_counter()
    if seed < 0 or not modules or any(m not in MODULES for m in modules):
        raise ValueError('invalid seed or module list')
    repetitions = repetitions if repetitions is not None else (12 if fast else 24)
    if not 8 <= repetitions <= 200:
        raise ValueError('repetitions must be between 8 and 200')
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('output directory is nonempty; select a new directory to preserve past runs')
    output.mkdir(parents=True,exist_ok=True)
    max_samples=4000 if fast else 8000
    tape=random_tape(max_samples,seed); d=model(tape)
    numerical=summary(d)
    comparison_count=validate_graph_algorithm()
    np.testing.assert_allclose(d.Y1-d.Y0,2,atol=1e-12)
    payload={"tapes":{},"bias":{},"graph":graph_payload(),"max_samples":max_samples,"repetitions":repetitions}
    outputs=[]
    if dashboard:
        for current in range(seed,seed+4):
            payload['tapes'][str(current)]={k:v.tolist() for k,v in random_tape(max_samples,current).items()}
            if 'sample-size' in modules:
                payload['bias'][str(current)]=bias_study(current*10000, sizes=(200,600,1800) if fast else (200,600,1800,5400),repetitions=repetitions)
        bundle=get_plotlyjs()
        for module in modules:
            file=output/f'{module}.html'; write_module(file,module,payload,bundle); outputs.append(file)
        file=output/'index.html'; write_dashboard(file,list(modules)); outputs.append(file)
    if gifs:
        outputs.extend(export_assets(output/'assets',d,fast,modules))
    report={"created_utc":datetime.now(timezone.utc).isoformat(),"seed":seed,"fast":fast,
        "max_samples":max_samples,"repetitions":repetitions,"modules":list(modules),
        "versions":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,
            "matplotlib":matplotlib.__version__,"plotly":plotly.__version__,"networkx":nx.__version__,"sklearn":sklearn.__version__},
        "population":population_values(),"baseline":numerical,"graph_comparisons":comparison_count,
        "bias":payload['bias'],"outputs":[str(p.relative_to(output)) for p in outputs],
        "seconds":time.perf_counter()-started,"interpretations":"author review required"}
    (output/'run.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all',action='store_true',help='HTML modules, dashboard, GIFs and PNGs')
    parser.add_argument('--dashboard',action='store_true',help='interactive offline HTML')
    parser.add_argument('--gifs',action='store_true',help='local GIFs and PNG previews')
    parser.add_argument('--module',choices=MODULES,help='restrict generation to one module')
    parser.add_argument('--seed',type=int,default=49)
    parser.add_argument('--fast',action='store_true',help='4000-case tape, 12 repetitions, 18 GIF frames')
    parser.add_argument('--repetitions',type=int,help='8..200 Monte Carlo repetitions per sample size')
    parser.add_argument('--output-dir',type=Path,help='new output directory; existing runs are preserved')
    args=parser.parse_args(argv)
    if args.seed<0 or args.repetitions is not None and not 8<=args.repetitions<=200:
        parser.error('seed must be nonnegative; repetitions must be in 8..200')
    output=args.output_dir
    if output is None:
        base=HERE/'outputs'/f'lab-seed{args.seed}-{"fast" if args.fast else "standard"}'
        output=base; index=2
        while output.exists() and any(output.iterdir()):
            output=base.with_name(base.name+f'-{index:03d}'); index+=1
    default=not (args.all or args.dashboard or args.gifs)
    try:
        report=generate(output,seed=args.seed,fast=args.fast,repetitions=args.repetitions,
            modules=(args.module,) if args.module else MODULES,
            dashboard=args.all or args.dashboard or default,gifs=args.all or args.gifs or default)
    except ValueError as error:
        parser.error(str(error))
    for file in [*report['outputs'],'run.json']:
        print((output/file).resolve())
    print(f"Generated in {report['seconds']:.2f}s; interpretations require author review.")
    return 0


if __name__=='__main__':
    raise SystemExit(main())

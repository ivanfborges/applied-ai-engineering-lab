"""Generate offline recurrent-memory animations, experiments and interactive HTML."""

import argparse
import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image
from plotly.offline import get_plotlyjs
import torch

from experiments import ROOT, configuration, load_or_run
from visual_core import causal_attention, computed_gru, computed_lstm, lstm_control, rnn_trace, scalar_gradients


BLUE, ORANGE, GREEN, INK = "#2463a6", "#ca6028", "#21846d", "#20354e"
SEQUENCE = [1., 0., 0.2, -0.4, 0., 0., 0., 0.]
EXPECTED = (
    "01_rnn_unrolled.gif", "01_rnn_hidden_state.html",
    "02_gradient_vanishing_exploding.gif", "02_gradient_analysis.html",
    "03_lstm_memory_flow.gif", "03_lstm_gates_interactive.html",
    "04_gru_memory_flow.gif", "04_gru_gates_interactive.html",
    "05_architecture_comparison.html", "05_training_curves.png", "05_experiment_results.csv",
    "06_hidden_state_3d.html", "07_recurrence_vs_attention.gif", "07_attention_matrix.html",
)


def slider(name, label, lo, hi, step, value, manual=False):
    return (f'<label>{label} <output id="{name}Value">{value}</output>'
            f'<input id="{name}" type="range" min="{lo}" max="{hi}" step="{step}" value="{value}"'
            + (' data-manual="true"' if manual else '') + '></label>')


def select(name, label, options):
    return f'<label>{label}<select id="{name}">' + ''.join(
        f'<option value="{html.escape(str(value))}">{html.escape(str(text))}</option>' for value, text in options) + '</select></label>'


def experiment_controls(report, *, aggregate=False):
    seeds = [(s, str(s)) for s in report["configuration"]["seeds"]]
    if aggregate:
        seeds.insert(0, ("all", "Mean across seeds"))
    return select("seed", "Experiment seed", seeds) + select("length", "Sequence length", [(t, t) for t in report["configuration"]["lengths"]])


def interactive(path, title, explanation, mechanism, controls, kind, payload):
    """One offline file, including Plotly runtime/license and all numerical data."""
    encoded = json.dumps(dict(kind=kind, **payload), allow_nan=False).replace("<", "\\u003c")
    script = (ROOT / "lab_controls.js").read_text(encoding="utf-8")
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>[hidden]{{display:none!important}}body{{margin:0;background:#edf2f8;color:{INK};font:16px/1.6 Arial,sans-serif}}main{{max-width:1250px;margin:24px auto;padding:30px;background:white;border-radius:18px}}h1{{font-size:30px;line-height:1.2}}.eyebrow{{font-weight:bold;letter-spacing:.13em;color:{GREEN};font-size:12px}}.controls{{display:flex;flex-wrap:wrap;gap:18px;background:#f0f5fb;padding:18px;border-radius:12px}}label{{display:flex;flex-direction:column;min-width:140px;font-size:14px;font-weight:bold}}input,select,textarea,button{{font:inherit;padding:7px;border:1px solid #a3b5c9;border-radius:6px;accent-color:{BLUE}}}input[type=range]{{width:160px}}output{{font-weight:normal}}button{{cursor:pointer;background:#e7f0fc}}#readout,.mechanism{{padding:16px;background:#f1f6fc;border-left:4px solid {BLUE}}}#error{{color:#a12121;font-weight:bold}}.plots{{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr))}}.plot{{min-width:0}}#diagram svg{{width:100%;height:auto}}#plot.full{{grid-column:1/-1}}footer{{font-size:13px;color:#50637b}}@media(max-width:650px){{main{{padding:14px;margin:8px}}.plots{{display:block}}h1{{font-size:24px}}}}</style>
<script>{get_plotlyjs()}</script></head><body><main><div class="eyebrow">DAY 60 · RECURRENT MEMORY LAB · OFFLINE</div>
<h1>{html.escape(title)}</h1><p>{html.escape(explanation)}</p><p class="mechanism">{html.escape(mechanism)}</p>
<div class="controls">{controls}</div><p id="error" role="alert"></p><div id="diagram"></div><p id="readout" aria-live="polite"></p>
<div class="plots"><div class="plot {'full' if kind=='hidden3d' else ''}" id="plot"></div><div class="plot" id="extra" hidden></div><div class="plot" id="third" hidden></div><div class="plot" id="fourth" hidden></div><div class="plot" id="fifth" hidden></div></div>
<footer>Educational synthetic study. Computed states, analytical illustrations and trained-model measurements are labeled separately. Interpretation: author review required. No network connection or server is used.</footer>
<script>window.LAB_DATA={encoded};</script><script>{script}</script></main></body></html>'''
    Path(path).write_text(page, encoding="utf-8")


def canvas(title, description, mechanism, rows=2):
    fig, axes = plt.subplots(rows, 1, figsize=(10.5, 6.8), squeeze=False)
    fig.patch.set_facecolor("#f7faff")
    fig.suptitle(title, fontsize=17, weight="bold", color=INK, x=.06, ha="left", y=.97)
    fig.text(.06, .9, description, fontsize=10, color=INK)
    fig.text(.06, .035, mechanism, fontsize=10, color=INK)
    fig.subplots_adjust(top=.82, bottom=.17, hspace=.52, left=.09, right=.96)
    return fig, axes[:, 0]


def node(ax, x, y, text, active=False, width=.95):
    ax.add_patch(FancyBboxPatch((x-width/2, y-.22), width, .44, boxstyle="round,pad=0.04",
                               facecolor="#fae0cd" if active else "#e4eefb", edgecolor=ORANGE if active else BLUE))
    ax.text(x, y, text, ha="center", va="center", fontsize=10, color=INK)


def arrow(ax, start, end, color=BLUE, width=1.5):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, color=color, lw=width))


def save_gif(path, draw, frames, duration=350):
    images = []
    try:
        for frame in frames:
            fig = draw(frame)
            try:
                fig.canvas.draw()
                images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()).convert("P", palette=Image.Palette.ADAPTIVE))
            finally:
                plt.close(fig)
        images[0].save(path, save_all=True, append_images=images[1:], loop=0, duration=duration, optimize=True)
    finally:
        for image in images:
            image.close()


def rnn_figure(step):
    states = rnn_trace(SEQUENCE)
    fig, (network, trajectory) = canvas("01 · The same recurrent cell, applied across time",
        f"Reading input {step}/{len(SEQUENCE)} · handcrafted tanh RNN, hidden dimension 2",
        "h_t = tanh(Wx x_t + Wh h_(t-1)); shared weights at every step, h0=0.\nObserve repeated transformation, not exact storage. These weights are untrained.")
    network.set(xlim=(0, 9), ylim=(-.8, 1.2)); network.axis("off")
    for t in range(1, 9):
        node(network, t, .1, f"h{t}\n{states[t, 0]:+.2f}" if t <= step else f"h{t}\n?", t == step, width=.72)
        network.text(t, .9, f"x={SEQUENCE[t-1]:+.1f}", ha="center", fontsize=10, color=ORANGE if t == step else INK)
        arrow(network, (t, .75), (t, .36), ORANGE if t == step else BLUE)
        if t > 1:
            arrow(network, (t-.6, .1), (t-.43, .1), ORANGE if t == step else BLUE, 2)
    for dimension, color in enumerate((BLUE, ORANGE)):
        trajectory.plot(np.arange(step+1), states[:step+1, dimension], "o-", color=color, label=f"h dimension {dimension+1}")
    trajectory.set(xlim=(0, 8), ylim=(-1, 1), xlabel="timestep (0 = initial state)", ylabel="hidden value")
    trajectory.legend(loc="upper right"); trajectory.grid(alpha=.15)
    return fig


def gradient_figure(distance):
    fig, (curves, path) = canvas("02 · Backward sensitivity across repeated transitions",
        f"Analytical scalar model · constant derivative a · distance k={distance}",
        "Magnitude = |a|^k, not neural-network training measurements.\nReal BPTT multiplies changing Jacobians; clipping limits explosions, not vanishing signals.")
    for a, color in ((.8, BLUE), (1., GREEN), (1.2, ORANGE)):
        curves.semilogy(np.arange(distance+1), scalar_gradients(a, distance), "o-", ms=3, color=color, label=f"a={a}")
    curves.set(xlim=(0, 50), ylim=(1e-6, 2e4), xlabel="recurrent transitions k", ylabel="gradient magnitude")
    curves.grid(alpha=.2); curves.legend(loc="upper left")
    path.axis("off"); path.set(xlim=(0, 10), ylim=(0, 1))
    count=min(distance,7)
    if not count:
        path.text(5,.5,"k=0: no recurrent transition; the empty product is 1",ha="center",color=INK,fontsize=13)
    elif distance>7:
        path.text(5,.05,f"Schematic segment: 7 displayed factors out of k={distance}",ha="center",fontsize=10,color=INK)
    for index in range(count):
        node(path, 1+index*1.3, .5, f"J{index+1}=a", index == count-1)
        if index:
            arrow(path, (.55+index*1.3, .5), (.22+index*1.3, .5), ORANGE)
    return fig


def lstm_gate_schedule():
    """Hand-selected interventions isolate retention, overwrite and exposure.

    This schedule is deliberately not represented as gates learned from x.
    The separate HTML computed mode uses the existing NumPy cell instead.
    """
    cell, hidden = .8, 0.
    rows = []
    schedule = [(1,0,0,1),(.95,0,0,0),(.95,.6,-.8,1),(0,0,0,1),
                (1,.9,.9,.8),(1,0,0,.1),(.9,0,0,1),(.8,0,0,1)]
    for x,(f,i,g,o) in zip(SEQUENCE,schedule):
        result=lstm_control(cell,f,i,g,o)
        rows.append(dict(x=x,c_prev=cell,h_prev=hidden,f=f,i=i,g=g,o=o,**result))
        cell,hidden=result["cell"],result["hidden"]
    return rows


def gate_figure(kind, step):
    rows = lstm_gate_schedule() if kind == "lstm" else computed_gru()
    row = rows[step-1]
    is_lstm = kind == "lstm"
    title = "03 · LSTM: retain, write, expose" if is_lstm else "04 · GRU: one state, two control gates"
    formula = ("c_t = f*c_prev + i*g; h_t = o*tanh(c_t). Hand-selected gate schedule, not learned behavior."
               if is_lstm else "PyTorch convention: n=tanh(0.8x+0.1+r*(0.9h_prev+0.3)); h=z*h_prev+(1-z)*n.")
    description = (f"Manual intervention {step}/8 · initial c=0.8 · inputs shown for context; gates independently chosen"
                   if is_lstm else f"Computed scalar forward step {step}/8 · x={row['x']:+.2f} · fixed, untrained weights")
    fig, axes = canvas(title, description,
                       formula+"\nObserve signed contributions. Gate values alone are not explanations of a trained decision.", rows=3)
    network, gates, state = axes
    network.axis("off"); network.set(xlim=(0, 10), ylim=(-.4, 1.3))
    if is_lstm:
        labels = [(1, .8, f"c_prev\n{row['c_prev']:+.3f}"), (3.5, .8, f"f*c_prev\n{row['retained']:+.3f}"),
                  (6, .8, f"+ i*g\n{row['written']:+.3f}"), (8.6, .8, f"c_t\n{row['cell']:+.3f}"),
                  (1, -.1, f"x={row['x']:+.2f}"), (3.5, -.1, f"h_prev\n{row['h_prev']:+.3f}"),
                  (6, -.1, f"g={row['g']:+.3f}"), (8.6, -.1, f"h_t\n{row['hidden']:+.3f}")]
        gate_names, gate_values = ["forget f", "input i", "output o"], [row["f"], row["i"], row["o"]]
        for start, end in (((1.6,.8),(2.9,.8)),((4.1,.8),(5.4,.8)),((6.6,.8),(8,.8)),((8.6,.55),(8.6,.12)),((1.6,-.1),(2.9,-.1)),((4.1,-.1),(5.4,-.1))):
            arrow(network, start, end)
        network.text(9.25,.35,"o × tanh",fontsize=9,ha="center")
    else:
        labels = [(1,.8,f"h_prev\n{row['h_prev']:+.3f}"),(3.5,.8,f"z={row['z']:.3f}"),
                  (6,.8,f"z*h_prev\n{row['retained']:+.3f}"),(8.6,.35,f"h_t\n{row['hidden']:+.3f}"),
                  (1,-.1,f"x={row['x']:+.2f}"),(3.5,-.1,f"r={row['r']:.3f}"),(6,-.1,f"n\n{row['candidate']:+.3f}")]
        gate_names, gate_values = ["update z · retain", "reset r · candidate"], [row["z"], row["r"]]
        for start,end in (((1.6,.8),(2.9,.8)),((4.1,.8),(5.4,.8)),((6.6,.8),(8,.4)),((1.6,-.1),(2.9,-.1)),((4.1,-.1),(5.4,-.1)),((6.6,-.1),(8,.3))):
            arrow(network,start,end)
    for x,y,label in labels:
        node(network,x,y,label,active="_t" in label,width=1.2)
    bars = gates.bar(gate_names, gate_values, color=[BLUE,ORANGE,GREEN][:len(gate_names)])
    gates.bar_label(bars, fmt="%.3f", padding=2); gates.set(ylim=(0,1.15),ylabel="gate value")
    state.plot(range(1,step+1),[r["hidden"] for r in rows[:step]],"o-",color=ORANGE,label="hidden h")
    if is_lstm:
        state.plot(range(1,step+1),[r["cell"] for r in rows[:step]],"o-",color=BLUE,label="cell c")
    state.set(xlim=(.8,8.2),ylim=(-1,1),xlabel="timestep",ylabel="state value"); state.legend(loc="upper right")
    return fig


def attention_figure(query):
    weights, _, _ = causal_attention()
    fig, (network, matrix) = canvas("07 · Recurrence passes state; attention accesses values",
        f"Conceptual, untrained attention · query position {query+1} · future keys masked",
        "Attention = softmax(mask(QK^T / sqrt(d))) V; here Q=K=V are illustrative vectors.\nFull attention has quadratic score/value work; autoregressive token generation remains sequential.")
    network.axis("off"); network.set(xlim=(0,7),ylim=(-.7,1.6))
    for t in range(6):
        node(network,t+1,1,f"h{t+1}",t==query,width=.7)
        node(network,t+1,0,f"v{t+1}",t==query,width=.7)
        if t<5: arrow(network,(t+1.4,1),(t+1.6,1),ORANGE if t<query else BLUE)
        if t<=query and t!=query:
            network.add_patch(FancyArrowPatch((query+1,-.24),(t+1,-.24),connectionstyle="arc3,rad=-.25",arrowstyle="-|>",
                                             mutation_scale=10,color=GREEN,lw=1+4*weights[query,t]))
    matrix.imshow(weights,vmin=0,vmax=1,cmap="Blues",aspect="auto")
    for j in range(6):
        matrix.text(j,query,f"{weights[query,j]:.2f}",ha="center",va="center",color=ORANGE,weight="bold",fontsize=10)
    matrix.set(xticks=range(6),xticklabels=range(1,7),yticks=range(6),yticklabels=range(1,7),xlabel="key position",ylabel="query position")
    return fig


def training_png(path, report):
    lengths = report["configuration"]["lengths"]
    fig, axes = plt.subplots(1,len(lengths),figsize=(12,4.4),squeeze=False)
    for ax,length in zip(axes[0],lengths):
        for row in report["results"]:
            if row["length"] == length:
                ax.semilogy([h["epoch"] for h in row["history"]],[h["validation_bce"] for h in row["history"]],
                            color={"rnn":BLUE,"lstm":ORANGE,"gru":GREEN}[row["cell"]],alpha=.7,label=f"{row['cell']} s{row['seed']}")
                ax.scatter(row["selected_epoch"],row["selected_validation"]["bce"],color={"rnn":BLUE,"lstm":ORANGE,"gru":GREEN}[row["cell"]],s=18)
        ax.set(title=f"Sequence length {length}",xlabel="epoch",ylabel="validation BCE"); ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=7)
    fig.suptitle("05 · Measured validation trajectories · dots = restored checkpoints",fontsize=15,weight="bold",color=INK)
    fig.text(.05,.02,"Synthetic delayed-bit recall. Same epoch budget; unequal parameter counts in equal-width mode. Author review required.",fontsize=10)
    fig.tight_layout(rect=(0,.07,1,.92)); fig.savefig(path,dpi=120); plt.close(fig)


def generate(output, *, only=None, full=False, matched=False, cache_dir=None, progress=print):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be new or empty; preserve existing exports")
    output.mkdir(parents=True,exist_ok=True)
    groups = {only} if only else {"rnn","gradients","lstm","gru","comparison","hidden3d","attention"}
    unknown = groups - {"rnn","gradients","lstm","gru","comparison","hidden3d","attention"}
    if unknown:
        raise ValueError(f"unknown visualization group: {unknown}")
    report = None
    if groups & {"gradients","comparison","hidden3d"}:
        report, folder = load_or_run(cache_dir or ROOT / "outputs" / "cache", configuration(full,matched),progress)
        (output / "experiment_report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    playback = '<button id="play" type="button">Play</button>'
    if "rnn" in groups:
        progress("Generating 01: unrolled RNN and editable hidden-state explorer")
        save_gif(output/"01_rnn_unrolled.gif",rnn_figure,range(1,9),500)
        controls = '<label>Input sequence<input id="sequence" value="1,0,0.2,-0.4,0,0,0,0"></label>'
        controls += slider("strength","Recurrent weight strength",0,2,.05,1)+slider("t","Timestep",0,8,1,8)+playback
        interactive(output/"01_rnn_hidden_state.html","Repeated transformations create recurrent memory",
                    "Change the sequence and recurrent strength. Watch the state trajectory and the current numerical update.",
                    "h_t=tanh(Wx x_t+Wh h_prev+b); all positions reuse the same weights. Fixed weights, not a trained model.",controls,"rnn",{})
    if "gradients" in groups:
        progress("Generating 02: analytical products and empirical PyTorch loss gradients")
        save_gif(output/"02_gradient_vanishing_exploding.gif",gradient_figure,range(0,51,2),180)
        interactive(output/"02_gradient_analysis.html","Two views of backward sensitivity",
                    "Move a below, at and above one. Compare the analytical decay with actual trained-model input gradients below.",
                    "Scalar |a|^k assumes constant derivative. Empirical curves are autograd derivatives of individual BCE losses across independent synthetic probe examples.",
                    slider("a","Scalar derivative a",0,1.5,.02,.8)+slider("t","Transitions k",0,50,1,50)+playback+experiment_controls(report),"gradients",dict(report=report))
    for kind,index in (("lstm","03"),("gru","04")):
        if kind not in groups:
            continue
        progress(f"Generating {index}: {kind.upper()} gate flow and controls")
        save_gif(output/f"{index}_{kind}_memory_flow.gif",lambda step:gate_figure(kind,step),range(1,9),600)
        controls=select("mode","Gate mode",[("manual","Manual intervention"),("computed","Fixed weights + actual inputs")])
        controls+=slider("t","Computed sequence step",1,8,1,1)+playback
        if kind=="lstm":
            for args in (("f","Forget gate",0,1,.05,.8),("i","Input gate",0,1,.05,.5),("o","Output gate",0,1,.05,.7),
                         ("c","Previous cell state",-2,2,.05,1),("g","Candidate g",-1,1,.05,-.5)):
                controls+=slider(*args,manual=True)
            rows=computed_lstm()
            mechanism="c_t=f*c_prev+i*g; h_t=o*tanh(c_t). Manual gates isolate the direct effect. Computed gates use an untrained NumPy LSTM, seed 160, H=1."
        else:
            for args in (("z","Update z (retain)",0,1,.05,.8),("r","Reset r",0,1,.05,.5),("h","Previous hidden",-1,1,.05,.7),("x","Input",-2,2,.05,.2)):
                controls+=slider(*args,manual=True)
            controls+=select("convention","Candidate convention",[("pytorch","PyTorch: reset AFTER affine"),("before","Variant: reset BEFORE affine")])
            rows=computed_gru()
            mechanism="h_t=z*h_prev+(1-z)*n. PyTorch: n=tanh(W_in*x+b_in+r*(W_hn*h_prev+b_hn)). Reset-before is a distinct function; fixed scalar weights are shown in the readout."
        interactive(output/f"{index}_{kind}_gates_interactive.html",f"{kind.upper()} · control what memory retains and exposes",
                    "Compare manual gate interventions with gates computed from fixed weights. Observe signed retention and write contributions; these are not learned explanations.",
                    mechanism,controls,kind,dict(rows=rows))
    if "comparison" in groups:
        progress("Generating 05: architecture comparison, training curves and CSV")
        interactive(output/"05_architecture_comparison.html","What did each recurrent model actually learn?",
                    "Inspect validation histories, held-out recall, actual model sizes and fit wall time. No architecture is assumed to win.",
                    "All models use independent train/validation/test streams and the same minibatch orders. Validation chooses epochs; test does not tune architecture or hyperparameters.",
                    experiment_controls(report,aggregate=True),"comparison",dict(report=report))
        training_png(output/"05_training_curves.png",report)
        fields=["seed","length","cell","hidden","parameters","selected_epoch","validation_bce","test_bce","test_accuracy","training_seconds","majority_accuracy","first_sign_accuracy","last_sign_accuracy"]
        with (output/"05_experiment_results.csv").open("w",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
            for row in report["results"]:
                writer.writerow({**{key:row[key] for key in fields if key in row},"validation_bce":row["selected_validation"]["bce"],"test_bce":row["test"]["bce"],"test_accuracy":row["test"]["accuracy"]})
    if "hidden3d" in groups:
        progress("Generating 06: trained hidden states projected with NumPy SVD")
        interactive(output/"06_hidden_state_3d.html","Learned hidden states · a 3D projection, not literal memory",
                    "Rotate, zoom and hover. Select architecture, sequence length and seed; use the timestep slider or Play to reveal temporal direction.",
                    "One PCA basis is fitted to all plotted states of this checkpoint, including h0. Explained variance describes this sample; bases are different across architectures. Visual class separation does not prove generalization.",
                    experiment_controls(report)+select("architecture","Architecture",[(k,k.upper()) for k in ("rnn","lstm","gru")])+slider("t","Timestep",0,50,1,50)+playback,
                    "hidden3d",dict(report=report))
    if "attention" in groups:
        progress("Generating 07: recurrence vs causal attention")
        save_gif(output/"07_recurrence_vs_attention.gif",attention_figure,range(6),700)
        weights,_,embeddings=causal_attention()
        interactive(output/"07_attention_matrix.html","Different paths to historical context",
                    "Select the query position. Observe a local recurrent chain, direct access to earlier values, the causal mask and the resulting weighted value vector.",
                    "Illustrative Q=K=V vectors, not a trained attention model. Softmax normalizes each allowed row; future scores are masked before normalization.",
                    slider("t","Query index (zero based)",0,5,1,5)+playback,"attention",dict(weights=weights.tolist(),embeddings=embeddings.tolist()))
    files=sorted(path.name for path in output.iterdir() if path.is_file())
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),groups=sorted(groups),
                  mode="full" if full else "quick",matched=matched,cache_key=report["cache_key"] if report else None,
                  files=files,interpretation_status="author review required")
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    progress(f"Export complete: {output}")
    return output,report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument("--quick",action="store_true",help="Default: one seed, H=12, 12 epochs, all sections")
    modes.add_argument("--full",action="store_true",help="Three seeds, H=16, 20 epochs, all sections")
    parser.add_argument("--only",choices=["rnn","gradients","lstm","gru","comparison","hidden3d","attention"])
    parser.add_argument("--matched-budget",action="store_true",help="Approximate the same parameter count as the mode's LSTM")
    parser.add_argument("--output-dir",type=Path,help="New or empty export directory")
    parser.add_argument("--cache-dir",type=Path,default=ROOT/"outputs"/"cache")
    args=parser.parse_args()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output=args.output_dir or ROOT/"outputs"/f"visual_{stamp}"
    try:
        generate(output,only=args.only,full=args.full,matched=args.matched_budget,cache_dir=args.cache_dir,progress=lambda message:print(message,flush=True))
    except (ValueError,FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__=="__main__":
    main()

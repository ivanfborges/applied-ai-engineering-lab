"""Export four small deterministic teaching animations with Matplotlib/Pillow."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import metrics as lab


KINDS = {"positions": "01_positions_precision_ap.gif", "ap": "02_ap_cumulative.gif",
         "rr": "03_first_hit_rr.gif", "ndcg": "04_ndcg_gain.gif"}


def states(kind):
    if kind == "ap":
        order, labels = lab.fixture([1, 0, 1, 1, 0, 1])
        return [{"order": order, "labels": labels, "step": step, "k": 5} for step in range(6)]
    if kind == "rr":
        return [{"order": lab.first_hit_order(pos)[0], "labels": lab.first_hit_order(pos)[1],
                 "position": pos, "k": 5} for pos in range(5, 0, -1)]
    if kind == "positions":
        order, labels = lab.fixture([1, 0, 1, 0, 0, 1])
        selected, start = "D06", 6
    elif kind == "ndcg":
        order, labels = lab.fixture()
        selected, start = "D01", 5
    else:
        raise ValueError(f"Unknown animation: {kind}")
    rest = [item for item in order if item != selected]
    result = []
    for position in range(start, 0, -1):
        ranking = rest.copy()
        ranking.insert(position - 1, selected)
        result.append({"order": ranking, "labels": labels, "position": position, "k": 5})
    return result


def draw_cards(ax, order, labels, k, cursor=None):
    ranks = np.arange(1, len(order) + 1)
    ax.barh(ranks, [1] * len(order), color=[lab.COLORS[int(labels[item])] for item in order], height=0.73)
    for rank, item in zip(ranks, order):
        grade = labels[item]
        ax.text(0.04, rank, f"{item}  |  grade {grade}", va="center", fontsize=11,
                color="white" if grade >= 2 else "#152238", weight="bold")
        if rank > k:
            ax.text(0.86, rank, "> K", va="center", color="#152238", fontsize=10,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.9))
    if k < len(order):
        ax.axhline(k + 0.5, color="#2463EB", linestyle="--", linewidth=1.5)
    if cursor:
        ax.barh(cursor, [1.01], fill=False, edgecolor="#152238", height=0.85, linewidth=2)
    ax.set(yticks=ranks, ylabel="Rank (1 is best)", xlim=(0, 1.03), ylim=(len(order)+0.6, 0.4))
    ax.set_xticks([])
    ax.set_title("Full judged ranking", fontsize=12, loc="left", pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)


def render_animation(kind, path, fps=2, dpi=100):
    """Render a forward scan, end hold and explicit rewind for a gentle GIF loop."""
    if not 1 <= fps <= 8 or not 50 <= dpi <= 150:
        raise ValueError("Use fps in [1,8] and dpi in [50,150].")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = states(kind)
    sequence = list(range(len(frames))) + [len(frames)-1]*2 + list(range(len(frames)-2, 0, -1)) + [0]*2
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.8), gridspec_kw={"width_ratios": [0.9, 1.25]})
    fig.patch.set_facecolor("#F7F9FC")
    titles = {"positions": "Same judged documents: move a relevant result forward",
              "ap": "Average precision: scan, count hits, add only at relevant ranks",
              "rr": "Reciprocal rank: only the first relevant result matters",
              "ndcg": "NDCG: move a high-gain document toward the first position"}

    def update(index):
        state = frames[index]
        order, labels, k = state["order"], state["labels"], state["k"]
        values = lab.evaluate_query(order, labels, k)
        rows = lab.trace(order, labels, k)
        for ax in axes:
            ax.clear()
            ax.set_facecolor("#F7F9FC")
        draw_cards(axes[0], order, labels, k, state.get("step"))
        ax = axes[1]
        if kind == "ap":
            step = state["step"]
            contributions = [row["ap_contribution"] if row["rank"] <= step else 0 for row in rows[:k]]
            ax.bar(range(1, k+1), contributions, color=[lab.COLORS[int(labels[item])] for item in order[:k]])
            ax.set(xlabel="Rank i", ylabel="Added precision P@i x relevance_i", ylim=(0, 1.17), xticks=range(1, k+1))
            for rank, value in enumerate(contributions, 1):
                if value:
                    ax.text(rank, value+0.025, f"{value:.3f}", ha="center", fontsize=10)
            numerator = sum(contributions)
            detail = "Before scanning" if not step else f"rank {step}: hits={rows[step-1]['hits_so_far']}, P@i={rows[step-1]['precision_at_rank']:.3f}"
            ax.set_title(f"Step {step}/5 | {detail}\nAP numerator={numerator:.4f}; AP={numerator:.4f}/4={numerator/4:.4f}", fontsize=11, loc="left", pad=12)
            footer = "R_q = 4 includes rank 6. Forward scan -> hold -> rewind to replay."
        elif kind == "positions":
            positions = [frame["position"] for frame in frames]
            for metric, color in (("precision", "#2463EB"), ("ap", "#23866F")):
                metric_values = [lab.evaluate_query(frame["order"], labels, k)[metric] for frame in frames]
                ax.plot(positions, metric_values, "o-", color=color, label=f"{metric.upper()}@5")
                ax.scatter([state["position"]], [values[metric]], s=150, color=color, edgecolors="white", zorder=4)
            ax.set(xlabel="D06 position (moving toward rank 1)", ylabel="Metric", ylim=(0, 1.08), xticks=range(1, 7))
            ax.invert_xaxis()
            ax.legend(loc="lower right", frameon=False)
            ax.set_title(f"P@5 = {values['precision']:.4f} | AP@5 = {values['ap']:.4f}\nFull relevant count R_q = 3", fontsize=12, loc="left", pad=12)
            footer = "Crossing K changes hits; moving within K can change AP while P stays fixed."
        elif kind == "rr":
            positions = np.arange(1, 6)
            ax.plot(positions, 1/positions, "o-", color="#2463EB")
            ax.scatter([state["position"]], [values["rr"]], s=180, color="#23866F", edgecolors="white", zorder=4)
            ax.set(xlabel="First relevant position h", ylabel="Reciprocal rank", ylim=(0, 1.12), xticks=positions)
            ax.set_title(f"First hit = {state['position']}\nRR@5 = 1/{state['position']} = {values['rr']:.4f}", fontsize=13, loc="left", pad=12)
            footer = "RR describes one query. MRR is the mean RR across multiple queries."
        else:
            details = lab.dcg_details(order, labels, k)
            ax.bar(range(1, 7), [row["dcg_contribution"] for row in rows], color=[lab.COLORS[int(labels[item])] for item in order])
            ax.set(xlabel="Rank i", ylabel="Selected gain / log2(i+1)", ylim=(0, 8), xticks=range(1, 7))
            ax.set_title(f"DCG@5={details['dcg']:.4f} | IDCG@5={details['idcg']:.4f}\nNDCG@5 = {values['ndcg']:.4f}", fontsize=12, loc="left", pad=12)
            footer = "Exponential gain: 2^grade - 1. Full-universe IDCG stays fixed as order changes."
        ax.grid(axis="y", alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(titles[kind], x=0.06, ha="left", fontsize=14, weight="bold", color="#152238")
        # Figure-level text is recreated to avoid accumulation across frames.
        for artist in list(fig.texts)[1:]:
            artist.remove()
        fig.text(0.06, 0.055, footer, fontsize=10, color="#152238")
        fig.text(0.06, 0.02, "Synthetic data | Day 44 | Forward -> hold -> rewind | Author review required", fontsize=9, color="#526174")
        return []

    fig.subplots_adjust(top=0.75, bottom=0.20, left=0.07, right=0.97, wspace=0.35)
    try:
        movie = animation.FuncAnimation(fig, update, frames=sequence, interval=1000/fps, repeat=True)
        movie.save(str(path), writer=animation.PillowWriter(fps=fps), dpi=dpi)
    finally:
        plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "assets")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--only", choices=list(KINDS), help="export one animation instead of all four")
    args = parser.parse_args()
    selected = [args.only] if args.only else list(KINDS)
    inventory = []
    for kind in selected:
        path = render_animation(kind, args.output_dir / KINDS[kind], args.fps, args.dpi)
        with Image.open(path) as gif:
            durations = []
            for frame in range(gif.n_frames):
                gif.seek(frame)
                gif.load()
                durations.append(gif.info.get("duration", 0))
            inventory.append({"kind": kind, "file": path.name, "bytes": path.stat().st_size,
                              "size": list(gif.size), "decoded_frames": gif.n_frames,
                              "duration_ms": sum(durations), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        print(f"Exported {path.name}: {inventory[-1]['decoded_frames']} decoded frames, {inventory[-1]['bytes']} bytes")
    order, labels = lab.fixture()
    details = lab.dcg_details(order, labels, 5)
    record = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "Moving relevant IDs changes rank-sensitive metrics; rank-wise contributions explain AP and NDCG.",
        "configuration": {"source": "deterministic synthetic fixtures in export_gifs.states", "gain": "exponential",
                          "k": 5, "fps": args.fps, "dpi": args.dpi, "randomness": "none",
                          "ap_denominator": "full-universe relevant count", "loop": "forward, hold, explicit rewind"},
        "versions": {name: version(name) for name in ["numpy", "matplotlib", "Pillow", "plotly", "streamlit"]},
        "python": platform.python_version(), "artifacts": inventory,
        "reference_query": {**details["metrics"], "dcg": details["dcg"], "idcg": details["idcg"]},
        "interpretation_candidate": "Inspect AP additions only at relevant ranks and fixed-IDCG changes as the high-grade document moves earlier.",
        "interpretation_status": "author review required",
        "limitations": "Constructed rankings; no empirical utility model, training, real retrieval or answer evaluation. Rewind is an animation replay, not a forward evaluation step.",
    }
    output = Path(__file__).resolve().parent / "outputs"
    output.mkdir(exist_ok=True)
    path = output / f"visual_run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Saved ignored execution record: {path}")


if __name__ == "__main__":
    main()

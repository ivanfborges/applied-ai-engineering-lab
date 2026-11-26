"""Bounded Matplotlib/Pillow exports; every panel uses computed lab records."""

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image

from visual_core import COLORS, PIPELINE, STAGES, TOKENS, masking


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelcolor": COLORS["ink"], "text.color": COLORS["ink"],
                     "figure.facecolor": "#F5F8FC", "axes.facecolor": "#FFFFFF",
                     "savefig.facecolor": "#F5F8FC"})


def figure(title, subtitle, preview=False, shape=(12.8, 7.2)):
    fig = plt.figure(figsize=shape, dpi=80 if preview else 110)
    fig.suptitle(title, x=.055, y=.97, ha="left", fontsize=19, fontweight="bold")
    fig.text(.055, .91, subtitle, fontsize=10)
    fig.text(.055, .025, "DAY 77  |  Synthetic educational lab  |  Author interpretation review required",
             fontsize=8, color="#536879")
    return fig


def save_png(fig, path):
    fig.savefig(path)
    plt.close(fig)


def save_gif(path, count, draw, preview=False, duration=1100):
    frames = []
    for index in range(count):
        fig = draw(index, preview)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()))
        plt.close(fig)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration,
                   loop=0, optimize=False)
    for frame in frames:
        frame.close()


def decoding_frame(row, preview=False):
    fig = figure("Constrained decoding: one admissible token at a time",
                 f"Step {row['step']:02d}  |  state: {row['phase']}  |  field index: {row['field']}", preview)
    prefix = row["prefix"] or "<empty prefix>"
    fig.text(.055, .83, "Prefix: " + prefix, fontfamily="monospace", fontsize=10)
    fig.text(.055, .78, f"Allowed: {', '.join(row['allowed'])}     Selected: {row['selected']}",
             fontsize=11, fontweight="bold")
    y = np.arange(len(TOKENS))
    for bounds, values, title in (([.14, .14, .32, .58], row["original"], "Original probabilities"),
                                  ([.61, .14, .32, .58], row["constrained"], "Masked + normalized")):
        ax = fig.add_axes(bounds)
        colors = [COLORS["allowed"] if keep else COLORS["rejected"] for keep in row["mask"]]
        bars = ax.barh(y, values, color=colors)
        for i, bar in enumerate(bars):
            if TOKENS[i] == row["selected"]:
                bar.set_edgecolor("black"); bar.set_linewidth(2)
            if not row["mask"][i]:
                bar.set_hatch("//")
        ax.set_yticks(y, TOKENS, fontsize=8)
        ax.invert_yaxis(); ax.set_xlim(0, 1.03)
        ax.set_title(title, fontsize=12); ax.set_xlabel("Probability")
    fig.text(.56, .065, f"sum(q) = {sum(row['constrained']):.6f}  |  forbidden q = 0", fontsize=10)
    return fig


def redistribution_frame(index, preview=False):
    tokens = ["100", "200", 'quote', "null", "true"]
    masks = [[True] * 5, [True, True, False, True, True],
             [True, True, False, False, True], [True, True, False, False, False]]
    row = masking([1.2, .8, 2., -.2, .1], masks[index])
    fig = figure("Token masking redistributes probability mass",
                 "Numeric, non-nullable field: remove incompatible continuations progressively.", preview)
    colors = ["#0072B2", "#E69F00", "#7859A6", "#00866A", "#C44E52"]
    values = [row["logits"], row["original"], row["masked_logits"], row["constrained"]]
    for i, (title, value) in enumerate(zip(("1. Original logits", "2. Original probabilities",
                                          "3. Masked logits", "4. Constrained probabilities"), values)):
        ax = fig.add_axes([.075 + (i % 2) * .49, .53 - (i // 2) * .35, .38, .25])
        shown = [v if v is not None else 0 for v in value]
        bars = ax.bar(tokens, shown, color=colors)
        for j, keep in enumerate(row["mask"]):
            if not keep:
                bars[j].set_hatch("//")
                if i == 2:
                    ax.text(j, .03, "-inf", ha="center", fontweight="bold")
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_ylim(-.4 if i in (0, 2) else 0, 2.4 if i in (0, 2) else 1)
    fig.text(.07, .08, f"Admissible original mass = {row['mass']:.4f}   |   sum(q) = {sum(row['constrained']):.6f}",
             fontsize=12, fontweight="bold")
    return fig


def grammar_frame(row, preview=False):
    fig = figure("A restricted JSON grammar is a state machine",
                 "Three required fields, fixed order, whole-token vocabulary; no nesting or recursion.", preview)
    ax = fig.add_axes([.04, .36, .92, .45]); ax.set_xlim(-.7, 6.7); ax.set_ylim(-1, 1); ax.axis("off")
    phases = ["start", "property", "colon", "value", "separator", "completed", "terminated"]
    labels = ["Start\nobject", "Property\nname", "Colon", "Property\nvalue", "Comma /\nclose", "Completed\nobject", "EOS"]
    for i, (phase, label) in enumerate(zip(phases, labels)):
        active = phase == row["phase"]
        ax.add_patch(FancyBboxPatch((i - .4, -.25), .8, .5, boxstyle="round,pad=.04",
                                   facecolor=COLORS["allowed"] if active else "#E5EBF1", edgecolor="#536879"))
        ax.text(i, 0, label, ha="center", va="center", color="white" if active else COLORS["ink"], fontsize=9)
        if i < 6:
            ax.add_patch(FancyArrowPatch((i + .44, 0), (i + .56, 0), arrowstyle="->", mutation_scale=12))
    ax.add_patch(FancyArrowPatch((4, -.3), (1, -.3), connectionstyle="arc3,rad=-.45",
                               arrowstyle="->", color=COLORS["allowed"], mutation_scale=12))
    ax.text(2.5, -.91, "comma -> next required field", ha="center", fontsize=10)
    fig.text(.07, .3, "Prefix: " + (row["prefix"] or "<empty>"), fontfamily="monospace", fontsize=11)
    selection = "Rejected trial" if "trial" in row else "Selected"
    fig.text(.07, .23, f"Valid outgoing: {', '.join(row['allowed'])}     {selection}: {row['selected']}", fontsize=12)
    forbidden = next(token for token in ('<EOS>', '}', 'true') if token not in row["allowed"])
    message = row["trial"] + ": state unchanged" if "trial" in row else f"Forbidden example: {forbidden} -> rejected before state changes"
    fig.text(.07, .15, message, color=COLORS["rejected"], fontsize=11)
    fig.text(.07, .08, "Early EOS and early } are disallowed until all three required fields are complete.", fontsize=10)
    return fig


def funnel_figure(data, current=5, preview=False):
    fig = figure("Validation funnel: each gate answers a different question",
                 f"{data['population']} hand-authored synthetic records; counts come from executed Python validators.", preview)
    ax = fig.add_axes([.26, .24, .34, .53])
    stats = data["stages"]
    survivors = [r["survivors"] for r in stats]
    ax.barh(np.arange(6), [r["entering"] for r in stats], color="#E1E7EF", label="Entering")
    ax.barh(np.arange(6), [v if i <= current else 0 for i, v in enumerate(survivors)],
            color=[COLORS["accepted"] if i == 5 else COLORS["allowed"] for i in range(6)], label="Surviving")
    ax.set_yticks(range(6), STAGES, fontsize=10); ax.invert_yaxis()
    ax.set_xlim(0, data["population"] * 1.1); ax.set_xlabel("Records (fixed original population)")
    for i, row in enumerate(stats):
        if i <= current:
            ax.text(row["entering"] + 2, i, str(row["survivors"]), va="center")
    ax.legend(loc="lower right", fontsize=9)
    row = stats[current]
    text = [f"Gate {current + 1}: {row['stage']}", "", f"Entering: {row['entering']}",
            f"Rejected: {row['rejected']}", f"Surviving: {row['survivors']}",
            f"Conditional pass: {row['conditional_pass']:.1%}", f"End-to-end: {row['end_to_end']:.1%}", "", "Rejection reasons:"]
    text.extend(f"{reason}: {count}" for reason, count in row["reasons"].items())
    if not row["reasons"]:
        text.append("None at this gate")
    fig.text(.66, .76, "\n".join(text), va="top", fontsize=11, linespacing=1.5)
    if current == 5:
        fig.text(.07, .1, "Rejected by gate: " + " | ".join(f"{i+1}: {r['rejected']}" for i, r in enumerate(stats)), fontsize=10)
    return fig


def strictness_figure(rows, preview=False):
    fig = figure("Strict typing and extra-field policy are separate controls",
                 "int amount_cents >= 0; bool approved; currency enum; required nullable string note. C = coerced; D = dropped.",
                 preview, (16, 10))
    cases = list(dict.fromkeys(r["case"] for r in rows))
    columns = [(False, "ignore"), (False, "forbid"), (True, "ignore"), (True, "forbid")]
    cells = []
    for name in cases:
        selected = [next(r for r in rows if r["case"] == name and (r["strict"], r["extra"]) == column) for column in columns]
        case_input = selected[0]["input"]
        summary = json_summary(case_input)
        cells.append([name + "\n" + summary] + [
            ("PASS\n" + f"amount={r['output']['amount_cents']}; bool={r['output']['approved']}" +
             ("\nC:" + ",".join(r["coercion"]) if r["coercion"] else "") +
             ("\nD:" + ",".join(r["dropped"]) if r["dropped"] else "")) if r["accepted"] else "REJECT\n" + "\n".join(textwrap.wrap(r["error"], 25))
            for r in selected])
    ax = fig.add_axes([.035, .09, .93, .75]); ax.axis("off")
    table = ax.table(cellText=cells, colLabels=["Original synthetic input", "Permissive / ignore", "Permissive / forbid",
                                               "Strict / ignore", "Strict / forbid"],
                     colWidths=[.34, .165, .165, .165, .165], bbox=[0, 0, 1, 1], cellLoc="left")
    table.auto_set_font_size(False); table.set_fontsize(8)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D9E2")
        if r == 0:
            cell.set_facecolor(COLORS["ink"]); cell.set_text_props(color="white", weight="bold")
        elif c > 0:
            label = cell.get_text().get_text()
            cell.set_facecolor("#FCE8E8" if label.startswith("REJECT") else "#E0F2EC")
        else:
            cell.set_facecolor("#EDF2F7")
    return fig


def json_summary(value):
    import json
    return "\n".join(textwrap.wrap(json.dumps(value, separators=(",", ":")), width=65))


def workflow_frame(row, preview=False):
    fig = figure("Bounded regeneration: stop, recover or exhaust",
                 "Scripted responses; costs ($0.02/call) and latency (0.8 s/call) are assumed, not measured.", preview)
    ax = fig.add_axes([.05, .36, .9, .4]); ax.axis("off")
    labels = ["Generate", "Parse", "Schema", "Business", "Evidence", "Accept"]
    active = {"parse": 1, "schema": 2, "business": 3, "reference": 4, "accepted": 5}.get(row["stage"], -1)
    for i, label in enumerate(labels):
        x = i / 6
        color = COLORS["rejected"] if i == active and row["stage"] != "accepted" else COLORS["accepted"] if i <= active else "#DDE5EE"
        ax.add_patch(FancyBboxPatch((x, .3), .135, .24, boxstyle="round,pad=.008", facecolor=color, edgecolor="none"))
        ax.text(x + .0675, .42, label, ha="center", va="center", color="white" if i <= active else COLORS["ink"])
        if i < 5:
            ax.annotate("", (x + .163, .42), (x + .139, .42), arrowprops=dict(arrowstyle="->"))
    fig.text(.07, .78, f"Scenario: {row['scenario']}  |  attempt: {row['number']} / {row['budget']} total", fontsize=13, weight="bold")
    fig.text(.07, .33, f"Classification: {row['stage']} / {row['code']}", fontsize=13)
    fig.text(.07, .26, f"Feedback: {row['feedback']}", fontsize=12)
    fig.text(.07, .19, f"Cumulative synthetic cost: ${row['cost']:.2f}    Simulated latency: {row['latency']:.1f} s", fontsize=12)
    fig.text(.07, .12, f"Outcome: {row['final']}", fontsize=15, weight="bold")
    fig.text(.07, .065, "Regeneration != transport retry. Backoff/jitter are not simulated here.", fontsize=10)
    return fig


def simulation_figure(simulation, preview=False):
    fig = figure("Persistent failures place a ceiling on retry recovery",
                 f"Monte Carlo: N={simulation['samples']:,}, seed={simulation['seed']}. Same first-call success p=0.525.", preview)
    axes = [fig.add_axes([.075, .2, .4, .58]), fig.add_axes([.57, .2, .37, .58])]
    for name, color in zip(("independent", "persistent mixture", "feedback aware"),
                           (COLORS["allowed"], COLORS["rejected"], COLORS["accepted"])):
        rows = [r for r in simulation["results"] if r["scenario"] == name]
        n = [r["n"] for r in rows]
        axes[0].plot(n, [r["analytical_success"] for r in rows], color=color, label=name)
        axes[0].errorbar(n, [r["simulated_success"] for r in rows],
                         yerr=[1.96 * r["success_se"] for r in rows], fmt="o", ms=4, color=color, capsize=2)
        axes[1].plot(n, [r["analytical_attempts"] for r in rows], color=color)
        axes[1].scatter(n, [r["simulated_attempts"] for r in rows], s=17, color=color)
    axes[0].set(xlabel="Maximum total attempts (integer)", ylabel="Request success probability", ylim=(.45, 1.02))
    axes[1].set(xlabel="Maximum total attempts (integer)", ylabel="Expected attempts per request", ylim=(1, 5))
    axes[0].legend(fontsize=9); axes[0].grid(alpha=.2); axes[1].grid(alpha=.2)
    fig.text(.075, .09, "Lines: analytical predictions. Points: simulation. Bars: +/- 1.96 binomial SE (normal approximation).", fontsize=10)
    fig.text(.075, .055, "Mixture: 25% permanently unsuccessful requests; remaining requests have p=0.7 per attempt.", fontsize=10)
    return fig


def reliability_surface_figure(grid, preview=False):
    fig = figure("Independent attempts: reliability depends on p and the budget",
                 "P(success within n) = 1 - (1-p)^n. Integer n samples; surface interpolation is illustrative.", preview)
    ax = fig.add_axes([.07, .13, .75, .67], projection="3d")
    p = np.array(sorted(set(row["p"] for row in grid)))
    n = np.arange(1, 11)
    x, y = np.meshgrid(p, n)
    z = 1 - (1 - x) ** y
    surface = ax.plot_surface(x, y, z, cmap="viridis", vmin=0, vmax=1, linewidth=.2)
    ax.contour(x, y, z, levels=[.9, .95, .99], zdir="z", offset=0, cmap="viridis")
    ax.set(xlabel="Per-attempt success p", ylabel="Maximum attempts n", zlabel="Final success probability", zlim=(0, 1))
    ax.view_init(24, -125)
    fig.colorbar(surface, ax=ax, shrink=.65, label="Reliability")
    return fig


def guardrail_frame(path, index, preview=False):
    fig = figure("The model proposes; the application controls execution",
                 "Illustrative architecture with actual synthetic validation/permission decisions. No live tools or provider.", preview)
    ax = fig.add_axes([.035, .23, .93, .58]); ax.set_xlim(-.6, 5.6); ax.set_ylim(-.7, 1.8); ax.axis("off")
    positions = [(i, 1) for i in range(6)] + [(i, 0) for i in range(4, -1, -1)]
    for i, (label, (x, y)) in enumerate(zip(PIPELINE, positions)):
        color = COLORS["allowed"] if i == index else "#DDE6F0"
        if i == index == path["stop"]:
            color = COLORS["accepted"] if path["executed"] else COLORS["rejected"]
        ax.add_patch(FancyBboxPatch((x - .44, y - .24), .88, .48, boxstyle="round,pad=.015", facecolor=color, edgecolor="#9CAEBE"))
        ax.text(x, y, "\n".join(textwrap.wrap(label, 15)), ha="center", va="center", fontsize=8,
                color="white" if i == index else COLORS["ink"])
        if i:
            px, py = positions[i - 1]
            direction = 1 if x > px else -1
            start = (px + direction * .46, py) if py == y else (px, py - .27)
            end = (x - direction * .46, y) if py == y else (x, y + .27)
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=10,
                                        shrinkA=0, shrinkB=0, color="#667D91"))
    x, y = positions[index]
    ax.plot(x, y + .38, marker="v", color=COLORS["ink"], ms=10)
    fig.text(.065, .84, f"Scenario: {path['scenario']}  |  attempt {path['attempt']} / 3 total", fontsize=13, weight="bold")
    fig.text(.065, .18, "Tool proposal: post_invoice(SYN-0)  |  authorized=" + str(path["authorized"]), fontsize=11)
    outcome = path["branch"] if index == path["stop"] else "Proceed to the next independent gate"
    fig.text(.065, .11, outcome, fontsize=13, weight="bold",
             color=(COLORS["accepted"] if path["executed"] else COLORS["rejected"]) if index == path["stop"] else COLORS["allowed"])
    fig.text(.065, .065, "External tools executed: 0. Scenario final eligibility after all gates: " + str(path['executed']), fontsize=10)
    return fig


def render_static(demo: str, root: Path, lab: dict, preview=False) -> list[str]:
    files = []
    if demo == "decoding":
        files = ["constrained_decoding.gif"]
        save_gif(root / files[0], len(lab["decoding"]), lambda i, p: decoding_frame(lab["decoding"][i], p), preview)
    elif demo == "masking":
        files = ["probability_redistribution.gif"]
        save_gif(root / files[0], 4, redistribution_frame, preview, 1600)
    elif demo == "grammar":
        files = ["grammar_state_transitions.gif"]
        rows = lab["decoding"] + lab["grammar_rejections"]
        save_gif(root / files[0], len(rows), lambda i, p: grammar_frame(rows[i], p), preview)
    elif demo == "validation":
        files = ["validation_funnel.gif", "validation_funnel.png"]
        save_gif(root / files[0], 6, lambda i, p: funnel_figure(lab["validation"], i, p), preview, 1800)
        save_png(funnel_figure(lab["validation"], 5, preview), root / files[1])
    elif demo == "strictness":
        files = ["strict_vs_permissive_validation.png"]
        save_png(strictness_figure(lab["strictness"], preview), root / files[0])
    elif demo == "retries":
        files = ["structured_output_retry_workflow.gif"]
        save_gif(root / files[0], len(lab["workflows"]), lambda i, p: workflow_frame(lab["workflows"][i], p), preview, 1600)
    elif demo == "surface":
        files = ["retry_reliability_surface.png"]
        save_png(reliability_surface_figure(lab["retry_grid"], preview), root / files[0])
    elif demo == "correlation":
        files = ["independent_vs_correlated_failures.png"]
        save_png(simulation_figure(lab["simulation"], preview), root / files[0])
    elif demo == "guardrails":
        files = ["production_guardrails_pipeline.gif"]
        states = [(path, i) for path in lab["guardrails"] for i in range(path["stop"] + 1)]
        save_gif(root / files[0], len(states), lambda i, p: guardrail_frame(*states[i], preview=p), preview, 850)
    return files

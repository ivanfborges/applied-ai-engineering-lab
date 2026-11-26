"""Question-specific Matplotlib/Pillow animations and offline Plotly explorers."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from visual_core import parameter_counts, serializable
from visual_html import write_gallery, write_interactive


BLUE, GREEN, RED, ORANGE, PURPLE, GRAY = "#236995", "#20815e", "#bd4354", "#bc6c16", "#7955a2", "#637487"
COLORS = [BLUE, GREEN, RED, PURPLE]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.facecolor": "white"})


def indices(length, count):
    return np.unique(np.linspace(0, length - 1, min(length, count)).astype(int))


def png(fig, root, name):
    fig.savefig(Path(root) / name, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def animate(root, name, records, draw, c, size=(11, 6)):
    fig = plt.figure(figsize=size, layout="constrained")
    selected = indices(len(records), c.frames)
    def update(index):
        fig.clear()
        draw(fig, int(index))
        fig.suptitle(name.removesuffix(".gif").replace("_", " ") + " · synthetic", fontsize=13)
    animation = FuncAnimation(fig, update, frames=selected, interval=220, repeat=True)
    animation.save(Path(root) / name, writer=PillowWriter(fps=4), dpi=85)
    plt.close(fig)


def probability_bars(ax, probabilities, labels=None, title="Response probabilities", colors=None):
    labels = labels or [f"Response {i}" for i in range(len(probabilities))]
    ax.bar(labels, probabilities, color=colors or COLORS[:len(probabilities)], edgecolor="white", hatch="//")
    ax.set(ylim=(0, 1), ylabel="Probability", title=title)
    ax.tick_params(axis="x", labelrotation=15)


def curve(ax, records, key, color=ORANGE, label=None, end=None):
    values = [row[key] for row in records]
    selected = records if end is None else records[:end + 1]
    ax.plot([row["step"] for row in selected], [row[key] for row in selected], color=color, label=label or key)
    low, high = min(values), max(values)
    pad = max((high - low) * .1, .01)
    ax.set(xlim=(0, max(1, records[-1]["step"])), ylim=(low - pad, high + pad), xlabel="Optimization step", ylabel=key)
    ax.grid(alpha=.2)


def interactive(root, name, title, report, views, description):
    safe_views = {k: [(fig, serializable(state)) for fig, state in frames] for k, frames in views.items()}
    write_interactive(Path(root) / name, title, description, safe_views, report["limitation"])


def landscape(root, r, c):
    axis = np.linspace(-4, 3, 45)
    x, y = np.meshgrid(axis, axis)
    surfaces = []
    for center, hessian in zip(r["centers"], r["hessians"]):
        delta = np.stack((x - center[0], y - center[1]), -1)
        surfaces.append(.5 * np.einsum("...i,ij,...j->...", delta, hessian, delta))
    records = r["records"]
    def draw(fig, i):
        ax, loss = fig.subplots(1, 2)
        stage = 0 if records[i]["stage"] == "pretraining" else 1
        ax.contour(x, y, surfaces[stage], levels=15, cmap="Blues")
        for name, color in (("pretraining", BLUE), ("fine-tuning", GREEN)):
            pts = np.array([v["position"] for v in records[:i + 1] if v["stage"] == name])
            if len(pts):
                ax.plot(pts[:, 0], pts[:, 1], ".-", color=color, label=name)
        for k, center in enumerate(r["centers"]):
            ax.scatter(*center, marker="*", s=150, color=(BLUE, GREEN)[k])
        ax.set(xlim=(-4, 3), ylim=(-4, 3), xlabel="Parameter theta 1", ylabel="Parameter theta 2",
               title=f'{records[i]["stage"]} · step {records[i]["step"]}')
        ax.legend()
        loss.plot([v["loss"] for v in records[:i + 1]], color=ORANGE)
        loss.axvline(c.steps, linestyle="--", color=GRAY)
        loss.set(xlim=(0, len(records)), ylim=(0, max(v["loss"] for v in records) * 1.1),
                 xlabel="Sequential recorded state", ylabel="Current-stage quadratic loss",
                 title="Objective changes at the stage boundary")
    animate(root, "01_pretraining_vs_fine_tuning.gif", records, draw, c)
    fig = plt.figure(figsize=(11, 5), layout="constrained"); draw(fig, len(records) - 1)
    png(fig, root, "01_parameter_landscape_contour.png")
    views = {}
    for k, name in enumerate(("pretraining", "fine-tuning")):
        pts = np.array([v["position"] for v in records if v["stage"] == name])
        z = [v["loss"] for v in records if v["stage"] == name]
        figure = go.Figure(go.Surface(x=x.tolist(), y=y.tolist(), z=surfaces[k].tolist(), opacity=.8, colorscale="Blues", name="Analytical loss"))
        figure.add_trace(go.Scatter3d(x=pts[:, 0].tolist(), y=pts[:, 1].tolist(), z=z, mode="lines+markers", name="Actual gradient descent", line={"color": GREEN, "width": 6}))
        figure.add_trace(go.Scatter3d(x=[float(pts[0, 0]), float(pts[-1, 0])], y=[float(pts[0, 1]), float(pts[-1, 1])], z=[z[0], z[-1]], mode="markers+text", text=["Start", "End"], marker={"size": 7, "color": [RED, GREEN]}, name="Stage endpoints"))
        figure.update_layout(scene={"xaxis_title": "theta 1", "yaxis_title": "theta 2", "zaxis_title": "Quadratic loss"})
        views[name] = [(figure, {"stage": name, "start": pts[0].tolist(), "end": pts[-1].tolist()})]
    interactive(root, "01_parameter_landscape_3d.html", "Where fine-tuning starts", r, views,
                "Actual gradient descent on two analytical objectives. The adaptation start equals the pretraining endpoint.")


def sft(root, r, c):
    records = r["records"]
    def draw(fig, i):
        ax, loss = fig.subplots(1, 2)
        probability_bars(ax, records[i]["probabilities"], r["labels"], f'SFT step {i} · target=chosen', [GRAY, GRAY, GREEN, GRAY])
        ax.text(.02, .96, f'p(target)={records[i]["probabilities"][r["target"]]:.3f}', transform=ax.transAxes, va="top")
        curve(loss, records, "loss", end=i)
        loss.set_title(f'Cross-entropy · gradient norm={records[i]["gradient"]:.3f}')
    animate(root, "02_sft_probability_evolution.gif", records, draw, c)
    p = np.linspace(.001, 1, 400)
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.plot(p, -np.log(p), color=ORANGE)
    ax.set(xlabel="Target probability p", ylabel="-ln(p), nats", title="Analytical negative log-likelihood; not a training curve")
    png(fig, root, "02_cross_entropy_landscape.png")
    token = r["token"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), layout="constrained")
    axes[0].axis("off")
    table = axes[0].table(cellText=[token["inputs"], token["targets"], [int(v) for v in token["mask"]]],
                          rowLabels=["Input (true prefix)", "Next-token target", "Response loss mask"], loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.6)
    axes[0].set_title("Teacher forcing: labels are shifted once; prompt and padding excluded")
    axes[1].bar(range(6), token["cross_entropy"], color=GRAY); axes[1].set(ylabel="Raw token CE, nats")
    axes[2].bar(range(6), np.asarray(token["cross_entropy"]) * token["mask"], color=GREEN)
    axes[2].set(xlabel="Aligned prediction position", ylabel="Selected contribution", title=f'Selected sequence sum={token["selected_sum"]:.4f}; arithmetic fixture, no generation')
    png(fig, root, "02_token_loss_masking.png")
    frames = []
    for i in indices(len(records), c.frames):
        row = records[i]
        fig = go.Figure(go.Bar(x=r["labels"], y=list(row["probabilities"]), marker_color=[GRAY, GRAY, GREEN, GRAY], name="Model response probability"))
        fig.update_layout(yaxis={"title": "Probability", "range": [0, 1]}, title=f"Recorded SFT step {i}")
        frames.append((fig, {"step": int(i), "loss": row["loss"], "gradient_norm": row["gradient"], "logits": list(row["logits"])}))
    interactive(root, "02_sft_probabilities.html", "SFT changes a categorical distribution", r, {"SFT checkpoints": frames}, "Each frame uses stored logits, softmax probabilities and gradients from a real PyTorch update.")


def instruction(root, r, c):
    names = list(r["models"])
    pooled = np.concatenate([np.asarray([row["hidden"] for row in r["models"][name]]) for name in names])
    flat = pooled.reshape(-1, pooled.shape[-1]); mean = flat.mean(0)
    _, singular, vh = np.linalg.svd(flat - mean, full_matrices=False)
    coords = {name: (np.asarray([row["hidden"] for row in r["models"][name]]) - mean) @ vh[:2].T for name in names}
    explained = float((singular[:2] ** 2).sum() / (singular ** 2).sum())
    limits = np.concatenate(list(coords.values())).reshape(-1, 2)
    records = r["models"]["multitask"]
    def draw(fig, i):
        axes = fig.subplots(1, 3)
        for ax, name in zip(axes, names):
            points = coords[name][i]
            for task, label in enumerate(r["task_labels"]):
                selected = np.asarray(r["probe_tasks"]) == task
                ax.scatter(points[selected, 0], points[selected, 1], color=COLORS[task], marker=["o", "s", "D", "+"][task], s=30, label=label)
            ax.legend(fontsize=6, loc="upper right")
            ax.set(xlim=(limits[:, 0].min() - .2, limits[:, 0].max() + .2), ylim=(limits[:, 1].min() - .2, limits[:, 1].max() + .2), xlabel="Pooled PC1", ylabel="Pooled PC2", title=f'{name} · update {i}')
        axes[1].text(.02, .02, f'Fixed pooled PCA; 2D variance={explained:.1%}', transform=axes[1].transAxes)
    animate(root, "03_instruction_tuning_tasks.gif", records, draw, c)
    final = np.array([r["models"][name][-1]["accuracy"] for name in names])
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    image = ax.imshow(final, vmin=0, vmax=1, cmap="Blues")
    ax.set(xticks=range(4), xticklabels=r["task_labels"], yticks=range(3), yticklabels=names, title="Held-out synthetic variant accuracy; names label numeric rules")
    for i in range(3):
        for j in range(4):
            ax.text(j, i, f"{final[i,j]:.3f}", ha="center", va="center", color="white" if final[i,j] > .65 else "black")
    fig.colorbar(image, ax=ax, label="Accuracy")
    png(fig, root, "03_multitask_performance_heatmap.png")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for j, ax in enumerate(axes.flat):
        for name, color in zip(names, COLORS):
            ax.plot([row["step"] for row in r["models"][name]], [row["accuracy"][j] for row in r["models"][name]], label=name, color=color)
        ax.set(ylim=(0, 1), xlabel="Adaptation update", ylabel="Held-out accuracy", title=r["task_labels"][j]); ax.legend()
    png(fig, root, "03_task_training_curves.png")
    views = {}
    for name in names:
        frames = []
        for i in indices(len(records), c.frames):
            points = coords[name][i]
            fig = go.Figure()
            for task, label in enumerate(r["task_labels"]):
                selected = np.asarray(r["probe_tasks"]) == task
                fig.add_trace(go.Scatter(x=points[selected, 0].tolist(), y=points[selected, 1].tolist(), mode="markers", name=label, marker={"color": COLORS[task], "symbol": ["circle", "square", "diamond", "cross"][task]}))
            fig.update_layout(xaxis={"title": "Pooled PC1", "range": [float(limits[:,0].min() - .2), float(limits[:,0].max() + .2)]}, yaxis={"title": "Pooled PC2", "range": [float(limits[:,1].min() - .2), float(limits[:,1].max() + .2)]}, title=f"{name}: fixed pooled PCA")
            frames.append((fig, {"step": int(i), "task_accuracy": r["models"][name][i]["accuracy"], "pca_variance_2d": explained}))
        views[name] = frames
    interactive(root, "03_instruction_generalization.html", "Single-task versus multitask adaptation", r, views, "One PCA basis is fitted to all models and snapshots; axes do not rotate between frames. Task colors are supplied labels, not discovered clusters.")


def reward(root, r, c):
    axis = np.linspace(0, 1, 24); x, y = np.meshgrid(axis, axis)
    allweights = [v["weights"] for rows in r["noise"].values() for v in rows]
    bound = max(np.abs(w).sum() for w in allweights) + .1
    def surface(w):
        return w[0] * x + w[2] * y + .5 * (w[1] + w[3] + w[4])
    rows = r["noise"]["0.0"]
    def draw(fig, i):
        ax, loss = fig.subplots(1, 2)
        image = ax.imshow(surface(rows[i]["weights"]), origin="lower", extent=(0, 1, 0, 1), vmin=-bound, vmax=bound, cmap="RdBu")
        fig.colorbar(image, ax=ax, label="Predicted reward")
        ax.set(xlabel="Synthetic factuality feature", ylabel="Synthetic relevance feature", title=f'Reward plane · step {i}')
        scores = rows[i]["probe_scores"]
        ax.text(.02, .98, f'Probe reward 1/2: {scores[0]:.2f} / {scores[1]:.2f}\nDifference: {scores[0]-scores[1]:.2f}; P(1 preferred)={rows[i]["probe_preference_probability"]:.3f}',
                transform=ax.transAxes, va="top", fontsize=8, bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
        for noise, color in zip(r["noise"], COLORS):
            curve(loss, r["noise"][noise], "loss", color, f"Label flips={noise}", i)
        loss.set_ylim(0, 1); loss.legend(); loss.set_title("BT training loss; reward scores are proxies")
    animate(root, "04_reward_model_training.gif", rows, draw, c)
    views = {}
    for noise, records in r["noise"].items():
        views[f"flip probability={noise}"] = []
        for i in indices(len(records), c.frames):
            row = records[i]
            fig = go.Figure(go.Surface(x=x.tolist(), y=y.tolist(), z=surface(row["weights"]).tolist(), colorscale="RdBu", cmin=-bound, cmax=bound, colorbar={"title": "Reward"}))
            fig.update_layout(scene={"xaxis_title": "Factuality feature", "yaxis_title": "Relevance feature", "zaxis": {"title": "Predicted reward", "range": [-bound, bound]}}, title="Other feature values fixed at 0.5")
            views[f"flip probability={noise}"].append((fig, {"step": int(i), "loss": row["loss"], "probe_response1_response2_scores": list(row["probe_scores"]), "probe_reward_difference": float(row["probe_scores"][0]-row["probe_scores"][1]), "probe_preference_probability": row["probe_preference_probability"]}))
    interactive(root, "04_reward_surface_3d.html", "Learning a reward proxy", r, views, "Linear Bradley-Terry fitting to stochastic synthetic utility comparisons. The 3D plane is a slice over two features.")
    delta = np.linspace(-6, 6, 300)
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.plot(delta, 1 / (1 + np.exp(-delta)), color=PURPLE); ax.axvline(0, color=GRAY, ls="--")
    ax.set(xlabel="Reward(chosen) - reward(rejected)", ylabel="BT preference probability", title="Analytical preference probability; not truth probability")
    png(fig, root, "04_bradley_terry_probability.png")
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.bar(list(r["noise"]), [rows[-1]["expected_test_loss"] for rows in r["noise"].values()], color=ORANGE)
    ax.set(xlabel="Extra annotation-flip probability", ylabel="Expected held-out BT cross-entropy", title="Measured models evaluated against the same synthetic population")
    png(fig, root, "04_preference_noise_comparison.png")


def policy_animation(root, name, runs, c, extra=False):
    records = next(iter(runs.values()))["records"]
    keys = ("proxy", "intended", "length", "kl", "objective") if extra else ("reward", "kl", "objective")
    def draw(fig, i):
        axes = fig.subplots(2, 3 if extra else 2).flatten()
        for (label, report), color in zip(runs.items(), COLORS):
            current = report["records"][i]
            axes[0].plot(range(3), current["probabilities"], "o-", color=color, label=label)
            for ax, key in zip(axes[1:], keys):
                curve(ax, report["records"], key, color, label, i)
        axes[0].plot(range(3), next(iter(runs.values()))["reference"], "s--", color=GRAY, label="Frozen reference")
        axes[0].set(ylim=(0, 1), xticks=range(3), xlabel="Response category", ylabel="Probability", title=f"Actual state {i}"); axes[0].legend(fontsize=7)
        for ax, key in zip(axes[1:], keys):
            values = [row[key] for report in runs.values() for row in report["records"]]
            pad = max(.01, (max(values) - min(values)) * .1)
            ax.set_ylim(min(values) - pad, max(values) + pad)
            ax.set_title("Current reward signal minus beta KL" if key == "objective" else key)
            ax.legend(fontsize=7)
    animate(root, name, records, draw, c, (14, 8) if extra else (11, 8))


def rlhf(root, r, c):
    policy_animation(root, "05_rlhf_policy_evolution.gif", r["runs"], c)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for (beta, report), color in zip(r["runs"].items(), COLORS):
        for ax, key in zip(axes, ("reward", "kl", "objective")):
            curve(ax, report["records"], key, color, f"beta={beta}")
            ax.legend(fontsize=8)
    for ax, key in zip(axes, ("reward", "kl", "objective")):
        values = [row[key] for report in r["runs"].values() for row in report["records"]]
        ax.set_ylim(min(values) - .05, max(values) + .05)
    png(fig, root, "05_kl_regularization_comparison.png")
    axis = np.linspace(0, 1, 35); x, y = np.meshgrid(axis, axis); third = 1 - x - y
    views = {}
    for beta, report in r["runs"].items():
        q, rewards = np.array(report["reference"]), np.array(report["reward"])
        p = np.stack((x, y, np.maximum(0, third)), -1)
        kl = np.sum(np.where(p > 0, p * (np.log(np.maximum(p, 1e-300)) - np.log(q)), 0), -1)
        z = (p * rewards).sum(-1) - float(beta) * kl
        z[third < -1e-12] = np.nan
        rows = report["records"]; pts = np.array([row["probabilities"] for row in rows]); objectives = [row["objective"] for row in rows]
        optimum = np.array(report["optimum"]); active = optimum > 0
        optimal_z = float(optimum @ rewards - float(beta) * np.sum(optimum[active] * np.log(optimum[active] / q[active])))
        figure = go.Figure(go.Surface(x=x.tolist(), y=y.tolist(), z=z.tolist(), colorscale="Viridis", opacity=.8, name="Exact objective", colorbar={"title": "J"}))
        figure.add_trace(go.Scatter3d(x=pts[:,0].tolist(), y=pts[:,1].tolist(), z=objectives, name="Mirror ascent", mode="lines", line={"color": RED, "width": 6}))
        figure.add_trace(go.Scatter3d(x=[float(q[0]), float(optimum[0])], y=[float(q[1]), float(optimum[1])], z=[rows[0]["objective"], optimal_z], mode="markers+text", text=["Reference", "Analytic optimum"], name="Reference / optimum", marker={"color": [BLUE, GREEN], "size": 7}))
        figure.update_layout(scene={"xaxis_title": "p1: response 1 probability", "yaxis_title": "p2; p3 = 1-p1-p2", "zaxis_title": "E[reward] - beta KL"}, title=f"Valid simplex only · beta={beta}")
        views[f"beta={beta}"] = [(figure, {"beta": float(beta), "final_probabilities": pts[-1].tolist(), "final_expected_reward": rows[-1]["reward"], "final_kl": rows[-1]["kl"], "final_objective": rows[-1]["objective"], "optimum": list(report["optimum"]), "max_abs_optimum_error": report["max_abs_optimum_error"]})]
    interactive(root, "05_rlhf_objective_surface_3d.html", "Reward and KL on the probability simplex", r, views, "The triangular domain enforces nonnegative probabilities summing to one. Surface height is an exact finite-action objective, not PPO loss.")


def dpo(root, r, c):
    rows = r["runs"]["0.5"]
    def draw(fig, i):
        axes = fig.subplots(2, 2)
        probability_bars(axes[0,0], rows[i]["probabilities"], ["Chosen", "Rejected", "Other"], f'DPO update {i}', [GREEN, RED, GRAY])
        axes[0,0].plot(range(3), r["reference"], "s--", color=GRAY, label="Frozen reference"); axes[0,0].legend()
        for ax, key in zip((axes[0,1], axes[1,0], axes[1,1]), ("loss", "margin", "gradient")):
            curve(ax, rows, key, end=i)
        axes[1,0].set_title(f'BT preference probability={rows[i]["bt_preference_probability"]:.3f}')
    animate(root, "06_dpo_training_dynamics.gif", rows, draw, c)
    margin = np.linspace(-8, 8, 300)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for beta, color in zip((.1, .5, 2), COLORS):
        for ax, values in zip(axes, (np.logaddexp(0, -beta * margin), beta / (1 + np.exp(beta * margin)), 1 / (1 + np.exp(-beta * margin)))):
            ax.plot(margin, values, color=color, label=f"beta={beta}"); ax.set_xlabel("Reference-adjusted margin"); ax.legend()
    for ax, label in zip(axes, ("DPO loss", "Absolute margin derivative", "Implied BT preference probability")):
        ax.set_ylabel(label)
    png(fig, root, "06_dpo_loss_and_gradients.png")
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    x = np.arange(3)
    for offset, (name, p), color in zip((-.18, .18), r["counterexample"].items(), (BLUE, GREEN)):
        ax.bar(x + offset, p, .35, label=f'{name}; chosen/rejected odds={p[0]/p[1]:.2f}', color=color)
    ax.set(xticks=x, xticklabels=["Chosen", "Rejected", "Other"], ylim=(0, 1), ylabel="Absolute response probability", title="Constructed counterexample: ratio rises while both pair probabilities fall"); ax.legend()
    png(fig, root, "06_relative_vs_absolute_probability.png")
    views = {}; axis = np.linspace(-5, 3, 35); x, y = np.meshgrid(axis, axis)
    ref_margin = float(np.log(r["reference"][0] / r["reference"][1]))
    for beta, records in r["runs"].items():
        surface = np.logaddexp(0, -float(beta) * (x - y - ref_margin))
        frames = []
        for i in indices(len(records), c.frames):
            pts = np.array([row["logits"] for row in records[:i+1]])
            fig = go.Figure(go.Surface(x=x.tolist(), y=y.tolist(), z=surface.tolist(), opacity=.8, colorscale="Oranges", name="Analytical loss", colorbar={"title": "DPO loss"}))
            fig.add_trace(go.Scatter3d(x=pts[:,0].tolist(), y=pts[:,1].tolist(), z=[row["loss"] for row in records[:i+1]], mode="lines+markers", name="Actual NumPy gradient descent", line={"color": PURPLE, "width": 6}, marker={"size": 2}))
            fig.update_layout(scene={"xaxis_title": "Chosen logit", "yaxis_title": "Rejected logit", "zaxis_title": "DPO loss"}, title=f"beta={beta}; third logit fixed")
            frames.append((fig, {"step": int(i), "beta": float(beta), "response_probabilities": list(records[i]["probabilities"]), "bt_preference_probability": records[i]["bt_preference_probability"], "relative_margin": records[i]["margin"]}))
        views[f"beta={beta}"] = frames
    interactive(root, "06_dpo_loss_surface_3d.html", "DPO: response probability versus preference probability", r, views, "Beta selects independently executed trajectories. A checkpoint never fabricates interpolation between optimization states.")


def comparison(root, r, c):
    first, second = r["rlhf"]["records"], r["dpo"]
    def draw(fig, i):
        axes = fig.subplots(2, 3).flatten()
        for rows, label, color in ((first, "Exact reward/KL", BLUE), (second, "Population DPO", PURPLE)):
            axes[0].plot(range(3), rows[i]["probabilities"], "o-", label=label, color=color)
            curve(axes[1], rows, "reward", color, label, i); curve(axes[2], rows, "kl", color, label, i)
            margin = [float(np.log(row["probabilities"][2]/row["probabilities"][0]) - np.log(r["reference"][2]/r["reference"][0])) for row in rows]
            axes[3].plot([row["step"] for row in rows[:i+1]], margin[:i+1], color=color, label=label)
        axes[0].set(ylim=(0, 1), xlabel="Response category", ylabel="Probability", title=f'Common initial reference · update {i}')
        for ax, key in ((axes[1], "reward"), (axes[2], "kl")):
            values = [row[key] for records in (first, second) for row in records]
            ax.set_ylim(min(values) - .05, max(values) + .05)
        axes[3].set(xlim=(0,first[-1]["step"]),ylim=(-.1,3),xlabel="Update",ylabel="Adjusted response-2/0 log-odds",title="Common-reference preference margin")
        curve(axes[4], first, "objective", BLUE, "Reward/KL objective (maximize)", i)
        curve(axes[5], second, "loss", PURPLE, "Population DPO CE (minimize)", i)
        axes[4].set_title("Reward optimization objective"); axes[5].set_title("DPO objective; separate definition / axis")
        for ax in axes: ax.legend(fontsize=8)
    animate(root, "07_rlhf_vs_dpo.gif", second, draw, c, (14, 8))
    fig = plt.figure(figsize=(14, 8), layout="constrained"); draw(fig, len(second)-1)
    png(fig, root, "07_alignment_metrics_comparison.png")
    frames = []
    for i in indices(len(second), c.frames):
        fig = go.Figure()
        for rows, name, color in ((first, "Exact reward/KL", BLUE), (second, "Population DPO", PURPLE)):
            fig.add_trace(go.Bar(x=["Response 0", "Response 1", "Response 2"], y=list(rows[i]["probabilities"]), name=name, marker_color=color))
        fig.add_trace(go.Scatter(x=["Response 0", "Response 1", "Response 2"], y=list(r["rlhf"]["optimum"]), name="Analytic optimum", mode="markers", marker={"color": GREEN, "size": 12, "symbol": "diamond"}))
        fig.update_layout(yaxis={"title": "Probability", "range": [0, 1]}, title="Same rewards, BT assumptions, reference and beta")
        frames.append((fig, {"step": int(i), "reward_kl_objective": first[i]["objective"], "dpo_pair_cross_entropy": second[i]["loss"], "expected_reward": [first[i]["reward"], second[i]["reward"]], "kl": [first[i]["kl"], second[i]["kl"]], "reference_adjusted_response2_vs0_margin": [float(np.log(first[i]["probabilities"][2] / first[i]["probabilities"][0]) - np.log(r["reference"][2] / r["reference"][0])), second[i]["margin"]]}))
    interactive(root, "07_alignment_comparison.html", "A controlled theoretical connection", r, {"Common-reference trajectories": frames}, "Expected DPO loss uses population BT probabilities on all pairs. Loss magnitudes are different objectives and are not compared as quality scores.")


def lora(root, r, c):
    selected = r["targets"]["rank-2 target"]; rows = selected["ranks"]["2"]["records"]
    base, target = np.array(r["base"]), np.array(selected["target_update"])
    labels = ["Frozen W0", "A (trainable)", "B (trainable)", "B @ A", "Effective W0 + B @ A", "Target W0 + delta", "Residual: target - effective"]
    def matrices(row):
        update = np.array(row["update"])
        return [base, row["A"], row["B"], update, base + update, base + target, target - update]
    bounds = [max(np.abs(np.asarray(matrices(row)[j])).max() for row in rows) + 1e-9 for j in range(7)]
    def draw(fig, i):
        axes = fig.subplots(2, 4)
        for j, (ax, matrix, label) in enumerate(zip(axes.flat, matrices(rows[i]), labels)):
            image = ax.imshow(matrix, cmap="RdBu_r", vmin=-bounds[j], vmax=bounds[j], aspect="auto")
            ax.set(title=label, xlabel="Column", ylabel="Row"); fig.colorbar(image, ax=ax, shrink=.7)
        axes.flat[-1].axis("off")
        axes.flat[-1].text(0, .9, f'Rank 2 · step {i}\nFrobenius error {rows[i]["error"]:.4f}\nFrozen W0 unchanged\nalpha/rank = 1', va="top")
    animate(root, "08_lora_matrix_evolution.gif", rows, draw, c, (14, 7))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    fig2, ax2 = plt.subplots(figsize=(8, 4), layout="constrained")
    for ax, (name, report) in zip(axes, r["targets"].items()):
        ranks = [int(k) for k in report["ranks"]]
        ax.plot(ranks, [v["records"][-1]["error"] for v in report["ranks"].values()], "o-", color=GREEN, label="Learned finite-budget LoRA")
        ax.plot(ranks, [v["svd_error"] for v in report["ranks"].values()], "s--", color=BLUE, label="Truncated SVD lower bound")
        ax.set(xlabel="LoRA rank", ylabel="Frobenius update error", title=name); ax.legend(fontsize=8)
        ax2.semilogy(range(1, len(report["singular_values"])+1), np.maximum(report["singular_values"], 1e-16), "o-", label=name)
    ax2.set(xlabel="Singular-value index", ylabel="Singular value (log scale)", title="Actual target update spectra"); ax2.legend()
    png(fig, root, "08_lora_rank_comparison.png"); png(fig2, root, "08_singular_value_spectrum.png")
    views = {}
    for name, report in r["targets"].items():
        surface_bound = max(np.abs(report["target_update"]).max(), max(np.abs(row["update"]).max() for run in report["ranks"].values() for row in run["records"])) + .01
        for rank, run in report["ranks"].items():
            frames = []
            for i in indices(len(run["records"]), c.frames):
                row = run["records"][i]
                fig = make_subplots(rows=1, cols=2, specs=[[{"type": "surface"}, {"type": "surface"}]], subplot_titles=["Target update", f"Learned rank {rank}"])
                for col, z in enumerate((report["target_update"], row["update"]), 1):
                    fig.add_trace(go.Surface(z=np.asarray(z).tolist(), colorscale="RdBu_r", cmin=-surface_bound, cmax=surface_bound, showscale=col == 2), row=1, col=col)
                for scene in ("scene", "scene2"):
                    fig.update_layout(**{scene: {"xaxis_title": "Column", "yaxis_title": "Row", "zaxis": {"title": "Update weight", "range": [-surface_bound, surface_bound]}}})
                frames.append((fig, {"step": int(i), "rank": int(rank), "frobenius_error": row["error"], "svd_lower_bound": run["svd_error"], "frozen_unchanged": run["frozen_unchanged"]}))
            views[f"{name}; rank={rank}"] = frames
    interactive(root, "08_lora_reconstruction_3d.html", "Learned matrix updates and SVD bounds", r, views, "Ranks 1/2/4/8/16 use the same frozen matrix and target per scenario. Learning trajectories are measured; SVD is the analytical best Frobenius approximation.")


def geometry(root, r, c):
    rows = r["runs"]["1"]
    def draw(fig, i):
        axes = fig.subplots(1, 3)
        bound = np.abs(r["target"]).max()
        for ax, matrix, title in zip(axes, (r["target"], rows[i]["matrix"], r["runs"]["2"][i]["matrix"]), ("Unconstrained target", "Rank-1 projected update", "Full-rank update")):
            image = ax.imshow(matrix, vmin=-bound, vmax=bound, cmap="RdBu_r"); ax.set(title=title, xlabel="Column", ylabel="Row")
            fig.colorbar(image, ax=ax)
        axes[1].text(.02, -.22, f'Update {i}; rank-1 error={rows[i]["error"]:.3f}', transform=axes[1].transAxes)
    animate(root, "09_full_vs_lora_optimization.gif", rows, draw, c)
    a, b = np.meshgrid(np.linspace(.15, 1.5, 35), np.linspace(-1, 1, 35)); fixed_c = .4; d = fixed_c * b / a
    views = {}
    for rank, records in r["runs"].items():
        pts = np.array([row["matrix"] for row in records])
        figure = go.Figure(go.Surface(x=a.tolist(), y=b.tolist(), z=d.tolist(), name="Rank-1 slice", opacity=.6, colorscale="Blues", colorbar={"title": "W11"}))
        figure.add_trace(go.Scatter3d(x=pts[:,0,0].tolist(), y=pts[:,0,1].tolist(), z=pts[:,1,1].tolist(), mode="lines+markers", name=f"Projected rank-{rank} trajectory", marker={"size": 2}, line={"color": GREEN, "width": 6}))
        target = np.array(r["target"])
        figure.add_trace(go.Scatter3d(x=[float(target[0,0])], y=[float(target[0,1])], z=[float(target[1,1])], mode="markers", marker={"size": 8, "symbol": "diamond", "color": RED}, name="Full optimum projection"))
        figure.update_layout(scene={"xaxis_title": "W00", "yaxis_title": "W01", "zaxis_title": "W11"}, title="Rank-1 slice: W10=0.4, W11=0.4 W01 / W00")
        views[f"trajectory rank={rank}"] = [(figure, {"final_error": records[-1]["error"], "slice_fixed_W10": .4, "trajectory_W10_varies": True})]
    interactive(root, "09_lora_constraint_geometry_3d.html", "Bounded rank is not a fixed linear subspace", r, views, "The surface is a fixed-W10 slice. Trajectories are projections of four-coordinate matrices with varying W10, so need not lie on that slice. SVD-projected gradient is a geometric analogue, not factorized LoRA training.")


def efficiency(root, r, c):
    dims = np.array(r["dimensions"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    for key, color in zip(("full", "lora", "adapter"), COLORS):
        counts = np.array([v[key] for v in r["counts"]])
        axes[0].loglog(dims, counts, "o-", label=key, color=color)
        axes[1].plot(dims, counts * 2 / 1024, "o-", label=key, color=color)
    axes[0].set(xlabel="Square-layer width", ylabel="Trainable entries (log)", title="Four adapted layers; r=8, bottleneck=8")
    axes[1].set(xlabel="Square-layer width", ylabel="Theoretical FP16 checkpoint KiB", title="Trainable entries only; no optimizer/base/activations")
    for width,color in zip((64,256,1024),COLORS):
        ranks=np.arange(1,33)
        axes[2].plot(ranks,100*2*ranks/width,label=f"width={width}",color=color)
    axes[2].set(xlabel="LoRA rank",ylabel="Trainable fraction of full weights (%)",title="Fraction versus rank; no memory measurement")
    for ax in axes: ax.legend()
    png(fig, root, "10_parameter_efficiency.png")
    fig, axes = plt.subplots(1, 5, figsize=(14, 4), layout="constrained")
    fixture = r["adapter_fixture"]
    for ax, key, title in zip(axes, ("input", "down", "relu", "up", "output"), ("Input h (8)", "Down (2), with bias", "ReLU (2)", "Up (8), with bias", "h + adapter(h)")):
        ax.barh(range(len(fixture[key])), fixture[key], color=GREEN if key != "input" else BLUE)
        ax.set(title=title, xlabel="Activation", ylabel="Coordinate")
    fig.suptitle(f'Actual bottleneck forward pass: 8 -> 2 -> ReLU -> 8 + residual; {fixture["count"]} trainable entries')
    png(fig, root, "10_adapter_bottleneck.png")
    views = {}
    for layers in (1, 4, 12):
        ranks = np.arange(1, 33); widths = np.array([32, 64, 128, 256, 512, 1024])
        x, y = np.meshgrid(widths, ranks); z = 100 * 2 * y / x
        figure = go.Figure(go.Surface(x=x.tolist(), y=y.tolist(), z=z.tolist(), colorscale="Viridis", colorbar={"title": "% of full weights"}))
        figure.update_layout(scene={"xaxis_title": "Layer width d", "yaxis_title": "LoRA rank r", "zaxis_title": "Trainable fraction 100*2r/d (%)"}, title=f"{layers} layers scale both counts equally; fraction unchanged")
        views[f"adapted layers={layers}"] = [(figure, {"layers": layers, "example_d128_r8_b8": parameter_counts(128, 8, 8, layers), "full_count_excludes_bias": True, "adapter_count_includes_bias": True})]
    interactive(root, "10_parameter_efficiency_3d.html", "Trainable counts and checkpoint assumptions", r, views, "Theoretical LoRA weight-entry fraction; can exceed 100% at large ranks. Layer count changes absolute counts, not this ratio.")


def qlora(root, r, c):
    original, q4, q8 = np.array(r["original"]), np.array(r["q4"]["reconstructed"]), np.array(r["q8"]["reconstructed"])
    probe = original.ravel()[:64]; quantized = q4.ravel()[:64]
    frames = list(range(c.frames))
    def draw(fig, i):
        ax = fig.subplots()
        alpha = i / max(1, c.frames - 1)
        displayed = (1 - alpha) * probe + alpha * quantized
        ax.scatter(range(len(probe)), probe, marker="x", color=BLUE, label="Original FP32")
        ax.scatter(range(len(probe)), displayed, color=GREEN, label="Display interpolation toward 4-bit levels")
        ax.hlines(np.unique(q4), 0, 64, color=GRAY, alpha=.2)
        ax.set(xlabel="Weight index", ylabel="Weight value", ylim=(original.min()-.1, original.max()+.1), title="Mapping animation; interpolation is illustrative, not training")
        ax.legend()
    animate(root, "11_quantization_animation.gif", frames, draw, c)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    axes[0].hist(original.ravel(), bins=30, alpha=.6, label="FP32", color=BLUE); axes[0].hist(q4.ravel(), bins=15, alpha=.6, label="Uniform 4-bit", color=GREEN)
    axes[0].set(xlabel="Weight value", ylabel="Count", title="Original and reconstructed distributions"); axes[0].legend()
    axes[1].hist((original-q4).ravel(), bins=25, color=ORANGE); axes[1].set(xlabel="Original - reconstructed weight", ylabel="Count", title="Uniform 4-bit error")
    values = [0., float(np.sqrt(np.mean((original - np.array(r["fp16"]))**2))), r["q8"]["error"], r["q4"]["error"]]
    axes[2].bar(["FP32", "FP16", "8-bit", "4-bit"], values, color=COLORS); axes[2].set(ylabel="Measured weight RMSE", title="Errors relative to original FP32 values")
    png(fig, root, "11_quantization_error.png")
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.bar(["FP32", "FP16", "Uniform 8-bit", "Uniform 4-bit"], [original.size*4, original.size*2, r["q8"]["theoretical_bytes"], r["q4"]["theoretical_bytes"]], color=COLORS)
    ax.set(ylabel="Theoretical packed weight bytes", title="Uniform formats include one 4-byte scale; simulated arrays are larger")
    png(fig, root, "11_quantization_storage_comparison.png")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, matrix, title in zip(axes, (original, q4, r["records"][-1]["effective"]), ("Original FP32", "Frozen 4-bit reconstruction", "Frozen reconstruction + rank-2 update")):
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-.6, vmax=.6); ax.set(title=title, xlabel="Column", ylabel="Row"); fig.colorbar(im, ax=ax)
    png(fig, root, "11_quantized_base_with_lora.png")
    views = {}
    for name, matrix in (("FP32", original), ("FP16", r["fp16"]), ("uniform 8-bit", q8), ("uniform 4-bit", q4), ("4-bit reconstruction + LoRA", r["records"][-1]["effective"])):
        matrix = np.array(matrix)
        fig = go.Figure(go.Heatmap(z=matrix.tolist(), colorscale="RdBu_r", zmin=-.6, zmax=.6, colorbar={"title": "Weight"}))
        fig.update_layout(xaxis_title="Column", yaxis_title="Row", title=name)
        views[name] = [(fig, {"rmse_vs_original_fp32": float(np.sqrt(np.mean((original-matrix)**2))), "frozen_reconstructed_base_unchanged": r["base_unchanged"]})]
    interactive(root, "11_quantization_explorer.html", "Quantization storage versus numerical error", r, views, "Uniform symmetric quantization is an educational simulation. LoRA trains against a dequantized floating-point copy; no real low-bit kernel or packed-memory measurement is performed.")


def forgetting(root, r, c):
    rows = r["runs"]["full"]; axis = np.array(r["grid_axis"])
    def draw(fig, i):
        axes = fig.subplots(1, 3)
        for ax, (name, records) in zip(axes, r["runs"].items()):
            ax.imshow(records[i]["boundary"], origin="lower", extent=(axis[0], axis[-1], axis[0], axis[-1]), vmin=0, vmax=1, cmap="RdBu_r")
            ax.axvline(0, color=BLUE, ls="--", label="Task A boundary"); ax.axhline(0, color=GREEN, ls=":", label="Task B boundary")
            ax.set(xlabel="x0", ylabel="x1", title=f'{name} · update {i}\nA={records[i]["task_a"]:.3f}, B={records[i]["task_b"]:.3f}')
            ax.legend(fontsize=7)
    animate(root, "12_catastrophic_forgetting.gif", rows, draw, c, (13, 5))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for (name, records), color in zip(r["runs"].items(), COLORS):
        for ax, key in zip(axes, ("task_a", "task_b")):
            ax.plot([v["step"] for v in records], [v[key] for v in records], color=color, label=name)
            ax.set(ylim=(0, 1), xlabel="Task B adaptation update", ylabel="Held-out accuracy", title=key); ax.legend()
    png(fig, root, "12_capability_retention_comparison.png")
    views = {}
    for name, records in r["runs"].items():
        frames = []
        for i in indices(len(records), c.frames):
            row = records[i]
            fig = go.Figure(go.Contour(x=axis.tolist(), y=axis.tolist(), z=np.asarray(row["boundary"]).tolist(), zmin=0, zmax=1, colorscale="RdBu_r", colorbar={"title": "p(class 1)"}))
            fig.add_trace(go.Scatter(x=[0, 0], y=[-2.5, 2.5], mode="lines", name="A: x0=0", line={"color": BLUE, "dash": "dash"}))
            fig.add_trace(go.Scatter(x=[-2.5, 2.5], y=[0, 0], mode="lines", name="B: x1=0", line={"color": GREEN, "dash": "dot"}))
            fig.update_layout(xaxis_title="x0", yaxis_title="x1", title=f'{name}: recorded boundary')
            frames.append((fig, {"step": int(i), "heldout_task_a": row["task_a"], "heldout_task_b": row["task_b"]}))
        views[name] = frames
    interactive(root, "12_forgetting_boundaries.html", "Specialization and capability regression", r, views, "The same inputs have conflicting A/B labels with no task ID. Evaluation samples never enter gradient updates. Replay uses A training samples only.")


def hacking(root, r, c):
    policy_animation(root, "13_reward_hacking.gif", r["runs"], c, extra=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for (name, report), color in zip(r["runs"].items(), COLORS):
        for ax, key in zip(axes.flat, ("proxy", "intended", "length", "kl")):
            curve(ax, report["records"], key, color, name); ax.legend(fontsize=8)
    for ax, key in zip(axes.flat, ("proxy", "intended", "length", "kl")):
        values = [row[key] for report in r["runs"].values() for row in report["records"]]
        pad = max(.01, (max(values) - min(values)) * .1)
        ax.set_ylim(min(values) - pad, max(values) + pad)
    png(fig, root, "13_reward_hacking_metrics.png")
    views = {}
    for name, report in r["runs"].items():
        rows = report["records"]
        frames = []
        for i in indices(len(rows), c.frames):
            partial = rows[:i+1]
            fig = go.Figure(go.Scatter3d(x=[row["length"] for row in partial], y=[row["proxy"] for row in partial], z=[row["intended"] for row in partial], mode="lines+markers", line={"color": PURPLE, "width": 6}, marker={"color": [row["step"] for row in partial], "colorscale": "Viridis", "size": 3}, name="Actual optimization trajectory"))
            fig.update_layout(scene={"xaxis": {"title": "Expected response length", "range": [20, 120]}, "yaxis": {"title": "Expected proxy reward", "range": [0, 1]}, "zaxis": {"title": "Expected intended utility", "range": [-.2, 1]}}, title=name)
            frames.append((fig, {"step": int(i), "proxy": rows[i]["proxy"], "intended": rows[i]["intended"], "kl": rows[i]["kl"], "length": rows[i]["length"]}))
        views[name] = frames
    interactive(root, "13_proxy_vs_true_utility_3d.html", "Optimizing the wrong signal", r, views, "Axes use separate supplied proxy and intended utility functions. KL can limit distribution change while leaving reward misspecification intact.")


def pipeline(root, r, c):
    rows = r["records"]
    def draw(fig, i):
        axes = fig.subplots(2, 2)
        row = rows[i]
        probability_bars(axes[0,0], row["probabilities"][0], ["T0 concise", "T0 verbose", "T1 concise", "T1 verbose"], f'{row["stage"]} · step {row["step"]}; fixed evaluation probe', [GREEN,GRAY,GREEN,GRAY])
        selected = [v for v in rows[:i+1] if v["stage"] == row["stage"]]
        axes[0,1].plot([v["step"] for v in selected], [v["loss"] for v in selected], color=ORANGE, label=f'{row["stage"]}: own objective')
        axes[0,1].set(xlim=(0, c.steps), ylim=(0, max(v["loss"] for v in rows)*1.05), xlabel="Within-stage update", ylabel="Current stage loss", title="Only the current objective; no cross-stage loss ranking"); axes[0,1].legend(fontsize=8)
        for key, color in (("task_accuracy", BLUE), ("chosen_response_accuracy", GREEN), ("mean_chosen_probability", PURPLE)):
            axes[1,0].plot(range(i+1), [v["metrics"][key] for v in rows[:i+1]], label=key, color=color)
        axes[1,0].set(xlim=(0,len(rows)-1), ylim=(0,1), xlabel="Sequential recorded state", ylabel="Held-out metric"); axes[1,0].legend(fontsize=7)
        axes[1,1].bar(range(min(12,len(row["weight_updates"]))), row["weight_updates"][:12], color=GREEN)
        bound = max(abs(v) for row_ in rows for v in row_["weight_updates"][:12]) + .1
        axes[1,1].set(ylim=(-bound,bound), xlabel="First 12 currently trainable entries", ylabel="Delta from this stage's initial values", title=f'{row["trainable_parameters"]} trainable; frozen integrity={row["base_frozen_unchanged"]}')
    animate(root, "14_complete_post_training_pipeline.gif", rows, draw, c, (13, 8))
    fig = plt.figure(figsize=(13, 8), layout="constrained"); draw(fig,len(rows)-1)
    png(fig, root, "14_final_evaluation_summary.png")
    frames = []
    for i in indices(len(rows), c.frames):
        row = rows[i]
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Fixed evaluation probe", "Held-out stage trajectory"])
        fig.add_trace(go.Bar(x=["T0 concise","T0 verbose","T1 concise","T1 verbose"],y=list(row["probabilities"][0]),name="Response probability",marker_color=[GREEN,GRAY,GREEN,GRAY]),row=1,col=1)
        for key,color in (("task_accuracy",BLUE),("chosen_response_accuracy",GREEN),("mean_chosen_probability",PURPLE)):
            fig.add_trace(go.Scatter(x=list(range(i+1)),y=[v["metrics"][key] for v in rows[:i+1]],name=key,line={"color":color}),row=1,col=2)
        fig.update_yaxes(range=[0,1],title_text="Probability / fraction"); fig.update_xaxes(title_text="Sequential recorded state",row=1,col=2,range=[0,len(rows)-1])
        frames.append((fig,{"stage":row["stage"],"within_stage_step":row["step"],"current_objective_loss":row["loss"],"trainable_count":row["trainable_parameters"],"frozen_base_integrity":row["base_frozen_unchanged"],"metrics":row["metrics"],"final_sft_reference_integrity":r["final_reference_unchanged"]}))
    interactive(root,"14_post_training_dashboard.html","The complete measured post-training pipeline",r,{"Recorded base / SFT / DPO":frames},"All KL/margin metrics use the same final SFT reference. Stage losses retain different definitions. The fixed evaluation population never selects a checkpoint.")


RENDERERS = {"landscape":landscape,"sft":sft,"instruction":instruction,"reward":reward,"rlhf":rlhf,
             "dpo":dpo,"comparison":comparison,"lora":lora,"geometry":geometry,"efficiency":efficiency,
             "qlora":qlora,"forgetting":forgetting,"hacking":hacking,"pipeline":pipeline}


def render_demo(name, root, report, config):
    RENDERERS[name](Path(root),report,config)

"""Question-led figures and bounded animations for the Visual Validation Lab."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import visual_experiments as exp

TRAIN, VALID, TEST = "#2878B5", "#E59128", "#258B70"
INK, MUTED, RED, PAPER = "#183348", "#74818C", "#C84E59", "#F7F8FA"
COLORS = [TRAIN, VALID, TEST]
NAMES = ["Train", "Validation", "Test"]
CMAP = ListedColormap(["#E4E9EE", TRAIN, VALID, TEST])
plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": "white", "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#C8D1D9", "savefig.facecolor": PAPER,
})


def canvas(title, subtitle, rows=1, cols=1, size=(12, 7)):
    fig, axes = plt.subplots(rows, cols, figsize=size, squeeze=False)
    fig.suptitle(title, x=0.065, y=0.97, ha="left", fontsize=20, weight="bold")
    fig.text(0.065, 0.90, subtitle, color=MUTED, fontsize=11)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.81, bottom=0.16, hspace=0.65, wspace=0.35)
    return fig, axes


def footer(fig, text):
    fig.text(0.065, 0.035, text, fontsize=9, color=MUTED, va="bottom")


def partition_legend(ax, names=NAMES):
    ax.legend(handles=[Patch(color=c, label=n) for c, n in zip(COLORS, names)],
              loc="upper right", fontsize=9)


def timeline(ax, parts, n, title, row=0):
    for label, color, indices in zip(NAMES, COLORS, parts):
        ax.scatter(indices, np.full(len(indices), row), c=color, s=35, marker="s", label=label)
    ax.set(xlim=(-2, n + 1), ylim=(-0.7, 0.7), yticks=[], xlabel="Observation time / row index", title=title)


def record(config, result, interpretation, limitation):
    return {"configuration": config, "result": result,
            "interpretation_candidate_for_author_review": interpretation,
            "limitation": limitation}


class Renderer:
    def __init__(self, output):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.artifacts = []

    def png(self, fig, name):
        path = self.output / (name + ".png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        self.artifacts.append(path.name)

    def gif(self, fig, update, frames, name, fps=3):
        animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
        path = self.output / (name + ".gif")
        animation.save(path, writer=PillowWriter(fps=fps), dpi=85)
        plt.close(fig)
        self.artifacts.append(path.name)

    def html(self, fig, name):
        path = self.output / (name + ".html")
        fig.write_html(path, include_plotlyjs=True, full_html=True, auto_open=False)
        self.artifacts.append(path.name)


def visualize_train_val_test(render, seed):
    n = 120
    parts = exp.split_three(n, seed)
    rng = np.random.default_rng(seed)
    start = rng.uniform([3.8, 1.0], [6.2, 3.7], size=(n, 2))
    end = np.empty_like(start)
    assigned = np.empty(n, dtype=int)
    for k, ids in enumerate(parts):
        assigned[ids] = k
        end[ids] = np.c_[0.4 + k * 3.5 + (np.arange(len(ids)) % 9) * 0.30,
                         0.7 + (np.arange(len(ids)) // 9) * 0.34]
    point_colors = np.array(COLORS)[assigned]
    fig, axs = canvas("Where does each observation go?", "A fixed 60 / 20 / 20 holdout: roles change, row identities do not.")
    ax = axs[0, 0]
    dots = ax.scatter(start[:, 0], start[:, 1], c=point_colors, s=38, edgecolors="white")
    for k, (name, role, ids) in enumerate(zip(NAMES, ["Fit parameters", "Choose decisions", "Frozen final audit"], parts)):
        ax.text(1.6 + k * 3.5, 4.1, f"{name}: {len(ids)}\n{role}", ha="center", color=COLORS[k], weight="bold")
    ax.set(xlim=(0, 10.7), ylim=(0, 4.8), xticks=[], yticks=[], xlabel="Schematic partition position", ylabel="Observation layout")
    footer(fig, "Synthetic layout, not feature coordinates. Colors identify the final split from the first frame.")
    def update(frame):
        alpha = min(1, max(0, (frame - 2) / 12))
        dots.set_offsets((1 - alpha) * start + alpha * end)
        return dots,
    render.gif(fig, update, 20, "train_val_test_split", fps=6)
    fig, axs = canvas("One dataset, three different jobs", "The test partition stays outside model and threshold selection.", cols=2)
    for k, ids in enumerate(parts):
        axs[0, 0].scatter(end[ids, 0], end[ids, 1], color=COLORS[k], s=24, label=NAMES[k])
    axs[0, 0].set(xlabel="Schematic partition position", ylabel="Observation layout", title="Each dot is one row")
    axs[0, 0].legend()
    axs[0, 1].barh(NAMES[::-1], [len(p) for p in parts][::-1], color=COLORS[::-1])
    axs[0, 1].set(xlabel="Number of observations", ylabel="Partition", title="Partition sizes")
    for i, count in enumerate([len(p) for p in parts][::-1]):
        axs[0, 1].text(count + 1, i, str(count), va="center")
    footer(fig, "Split percentages are design inputs, not a guarantee of deployment representativeness.")
    render.png(fig, "train_val_test_split")
    return record({"rows": n, "fractions": [0.6, 0.2, 0.2]},
                  {"partition_sizes": [len(p) for p in parts]},
                  "Every row moves into exactly one role; validation is part of development.",
                  "Schematic movement demonstrates membership, not predictive performance.")


def animate_kfold(render, seed):
    data = exp.kfold_experiment(seed)
    folds, scores = data["folds"], data["scores"]
    order = np.concatenate([v for _, v in folds])
    membership = np.empty((5, 600), dtype=int)
    for i, (t, v) in enumerate(folds):
        row = np.ones(600, dtype=int)
        row[v] = 2
        membership[i] = row[order]
    fig, axs = canvas("Watch the validation fold move", "Ordinary shuffled KFold; a fresh scaled logistic model is fitted in every row.", cols=2)
    left, right = axs[0]
    left.imshow(membership, aspect="auto", cmap=CMAP, vmin=0, vmax=3, interpolation="nearest")
    left.set(xlabel="Development observations (ordered by held-out fold)", ylabel="CV fit", yticks=range(5), yticklabels=range(1, 6))
    right.set(xlim=(0.5, 5.5), ylim=(0.4, 1.0), xlabel="Completed fold", ylabel="ROC-AUC")
    right.grid(axis="y", alpha=0.2)
    bars = right.bar(np.arange(1, 6), np.zeros(5), color=VALID)
    highlight = Rectangle((-0.5, -0.45), 600, 0.9, facecolor="none", edgecolor=INK, linewidth=2)
    left.add_patch(highlight)
    score_labels = [right.text(j+1, scores[j]+0.025, "", ha="center", fontsize=10) for j in range(5)]
    fig.legend(handles=[Patch(color=TRAIN, label="Train"), Patch(color=VALID, label="Validation")], loc="center", bbox_to_anchor=(0.28, 0.075), ncol=2)
    footer(fig, "Every row: validation once, training four times. Fold SD is descriptive, not a confidence interval.")
    def update(frame):
        i = min(frame, 4)
        highlight.set_y(i - 0.45)
        for j, bar in enumerate(bars):
            bar.set_height(scores[j] if j <= i else 0)
        for j, label in enumerate(score_labels):
            label.set_text(f"{scores[j]:.3f}" if j <= i else "")
        right.set_title(f"Mean {scores.mean():.3f} | SD {scores.std(ddof=1):.3f}" if i == 4 else f"Completed fits: {i+1}/5")
    render.gif(fig, update, 7, "kfold_animation", fps=1)
    return record({"rows": 600, "features": 10, "folds": 5, "model": "StandardScaler + LogisticRegression C=1"},
                  {"fold_auc": scores, "mean_auc": scores.mean(), "sample_sd": scores.std(ddof=1)},
                  f"Mean fold AUC is {scores.mean():.4f}; each observation has one held-out prediction.",
                  "IID-like synthetic rows; no hyperparameter search or final test in this demonstration.")


def compare_kfold_stratified(render, seed):
    data = exp.stratification_experiment(seed)
    fig, axs = canvas("Does every fold represent the rare class?", "Same 200 synthetic labels, same split seed; only the splitting rule changes.", cols=2)
    for ax, name, rates in zip(axs[0], ["KFold", "StratifiedKFold"], data["rates"]):
        ax.barh(np.arange(1, 6), 1-rates, color=TRAIN, label="Class 0")
        ax.barh(np.arange(1, 6), rates, left=1-rates, color=VALID, label="Class 1")
        ax.axvline(1-data["global_rate"], color=INK, linestyle="--", label="Global class boundary")
        for i, rate in enumerate(rates, 1):
            ax.text(1.02, i, f"{rate:.1%}", va="center", fontsize=9)
        ax.set(xlim=(0, 1.15), xlabel="Validation class proportion", ylabel="Fold", title=name, yticks=np.arange(1, 6))
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.17), fontsize=8)
    footer(fig, "Stratification balances labels; it cannot remove customer dependence or future information.")
    render.png(fig, "stratified_kfold")
    return record({"rows": 200, "positive_labels": 20, "folds": 5},
                  {"global_positive_rate": data["global_rate"], "fold_positive_rates": data["rates"]},
                  "Compare each fold's positive proportion with the fixed global rate.",
                  "Exact balance here follows divisible class counts; other group constraints may prevent it.")


def visualize_group_leakage(render, seed):
    d = exp.group_experiment(seed)
    fig, axs = canvas("Are we recognizing customers or generalizing?", "100 customers x 20 transactions; deployment target: previously unseen customers.", rows=2, cols=2, size=(13, 9))
    customer_colors = plt.get_cmap("tab10")
    for ax, name in zip(axs[0], d["folds"]):
        train, valid = d["folds"][name][0]
        for g in range(8):
            for ids, marker in ((train, "o"), (valid, "x")):
                chosen = ids[d["groups"][ids] == g]
                ax.scatter(d["X"][chosen, 0], d["X"][chosen, 1], color=customer_colors(g), marker=marker, s=27, alpha=0.8)
            center = d["X"][d["groups"] == g, :2].mean(axis=0)
            ax.annotate(f"C{g}", center, xytext=(4, 6), textcoords="offset points", fontsize=9)
        ax.set(xlabel="Signature feature 1", ylabel="Signature feature 2", title=f"{name}: fold 1, customers 0..7")
        ax.legend(handles=[Line2D([], [], marker="o", color=MUTED, linestyle="", label="Train"),
                           Line2D([], [], marker="x", color=MUTED, linestyle="", label="Validation")], fontsize=8)
    heat = axs[1, 0].imshow(d["fractions"]["GroupKFold"], aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    axs[1, 0].set(xlabel="Validation fold", ylabel="Customer ID", title="Fraction of each customer's rows held out", xticks=range(5), xticklabels=range(1, 6))
    fig.colorbar(heat, ax=axs[1, 0], label="Held-out fraction", pad=0.02)
    for name, color in zip(d["scores"], [RED, TRAIN]):
        score = d["scores"][name]
        axs[1, 1].plot(range(1, 6), score, "o-", color=color, label=f"{name}: mean {score.mean():.3f}")
    axs[1, 1].set(xlabel="Fold", ylabel="Accuracy", ylim=(0, 1.08), title="Same 1-nearest-neighbor model")
    axs[1, 1].legend(fontsize=8)
    footer(fig, "Color = customer; marker = split. Two of eight model features shown. Labels are random per customer, stable within customer.")
    render.png(fig, "group_leakage")
    fig, axs = canvas("Which customers cross the validation boundary?", "Each cell is the fraction of a customer's transactions assigned to that validation fold.", cols=2)
    for ax, name in zip(axs[0], d["fractions"]):
        im = ax.imshow(d["fractions"][name], aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set(title=name, xlabel="Validation fold", ylabel="Customer ID", xticks=range(5), xticklabels=range(1, 6))
        fig.colorbar(im, ax=ax, label="Held-out fraction")
    footer(fig, "A fractional cell means this customer's other rows remain in training. GroupKFold cells are either zero or one.")
    render.png(fig, "group_membership")
    means = {k: float(v.mean()) for k, v in d["scores"].items()}
    return record({"customers": 100, "rows_per_customer": 20, "features": 8, "signature_noise_sd": 0.06, "model": "1-NN", "labels": "Bernoulli(0.5) per customer"},
                  {"accuracy": d["scores"], "mean_accuracy": means, "overlapping_groups": d["overlap"]},
                  f"Random accuracy {means['Random KFold']:.3f}; group accuracy {means['GroupKFold']:.3f}. Interpret against the new-customer target.",
                  "Designed stable signatures; real customer structure varies. Known-customer future prediction needs a different boundary.")


def visualize_temporal_split(render, seed):
    n = 120
    random_parts = exp.split_three(n, seed)
    temporal = [np.arange(72), np.arange(72, 96), np.arange(96, 120)]
    exp.assert_partition(n, *temporal)
    fig, axs = canvas("Does training contain the validation future?", "The horizontal coordinate is time in both panels; colors encode the information role.", rows=2)
    for ax, parts, name in zip(axs[:, 0], [random_parts, temporal], ["Random split: periods interleave", "Temporal split: past -> later -> latest"]):
        timeline(ax, parts, n, name)
        partition_legend(ax)
    footer(fig, "Chronology is necessary for this future-deployment target. Features and labels must also be available at the fitting cutoff.")
    render.png(fig, "temporal_split")
    return record({"rows": n, "temporal_cutoffs": [72, 96]},
                  {"random_latest_train": int(random_parts[0].max()), "random_earliest_validation": int(random_parts[1].min()),
                   "temporal_latest_train": 71, "temporal_earliest_validation": 72},
                  "Random partitioning mixes time periods; chronological boundaries preserve their order.",
                  "Membership diagram only; no performance score is implied.")


def animate_windows(render, seed):
    d = exp.drift_experiment(seed)
    for name in ["Expanding", "Rolling"]:
        folds = d["folds"][name]
        fig, axs = canvas(f"{name} window: what history survives?", "Fit on blue history, evaluate the next orange period; green final test is reserved.", rows=2)
        top, bottom = axs[:, 0]
        image = top.imshow(np.zeros((1, 600)), aspect="auto", cmap=CMAP, vmin=0, vmax=3, extent=(0, 600, 0, 1))
        top.set(xlabel="Time step", yticks=[], title="Gray: not used in this fit")
        partition_legend(top)
        bottom.plot(d["t"], d["slope"], color=MUTED, linestyle="--", label="True generating slope")
        fitted_line, = bottom.plot([], [], "o-", color=TRAIN, label="Fitted slope at validation start")
        bottom.set(xlabel="Time step", ylabel="Slope of y versus x", xlim=(0, 600), ylim=(-1.2, 1.2))
        bottom.legend(loc="lower left", fontsize=9)
        caption = top.text(0.01, 1.15, "", transform=top.transAxes, fontsize=10)
        footer(fig, "One sample per step; labels immediately available. Same linear model. Final test is never used to choose a window.")
        def update(frame, folds=folds, name=name, image=image, line=fitted_line, caption=caption):
            i = min(frame, len(folds)-1)
            t, v = folds[i]
            state = np.zeros(600)
            state[t], state[v], state[480:] = 1, 2, 3
            image.set_data(state[None, :])
            line.set_data([folds[j][1][0] for j in range(i+1)], d["coefficients"][name][:i+1])
            caption.set_text(f"Fit {i+1}: train {t[0]}..{t[-1]} ({len(t)} rows) | validation {v[0]}..{v[-1]} | MSE {d['scores'][name][i]:.3f}")
        render.gif(fig, update, 6, name.lower() + "_window", fps=1)
    return record({"rows": 600, "development": 480, "test": 120, "validation_window": 60, "rolling_history": 120,
                   "model": "LinearRegression(x)", "labels": "immediate", "true_slope": "1 - 2*t/599", "noise_sd": 0.25},
                  {"validation_mse": d["scores"], "final_test_mse": d["test_mse"], "fitted_slopes": d["coefficients"]},
                  f"Final future MSE: expanding {d['test_mse']['Expanding']:.3f}, rolling {d['test_mse']['Rolling']:.3f}. Shorter history changes both recency and sample size.",
                  "One linear drift trajectory; the window comparison is predeclared, not selected using test results.")


def simulate_concept_drift(render, seed):
    d = exp.drift_experiment(seed)
    fig, axs = canvas("The relationship changes while time advances", "Synthetic conditional mean: y = slope(t) * x; slope moves from +1 toward -1.", cols=2)
    left, right = axs[0]
    dots = left.scatter([], [], color=TRAIN, alpha=0.65, label="Current 60-row block")
    true_line, = left.plot([], [], color=RED, label="True slope at block midpoint")
    fitted_line, = left.plot([], [], color=VALID, linestyle="--", label="Fit using earlier rows")
    left.set(xlim=(-3.5, 3.5), ylim=(-3.5, 3.5), xlabel="Feature x", ylabel="Target y")
    left.legend(fontsize=8, loc="upper left")
    right.plot(d["t"], d["slope"], color=RED, label="True conditional slope")
    cursor = right.axvline(0, color=INK)
    right.set(xlabel="Time step", ylabel="Generating slope", ylim=(-1.2, 1.2))
    right.legend(fontsize=9)
    footer(fig, "Only earlier rows enter each fit; training freezes at row 479 for the final test period. Frames show different regimes.")
    def update(frame):
        end = min(600, 120 + frame*60)
        ids = np.arange(end-60, end)
        dots.set_offsets(np.c_[d["x"][ids], d["y"][ids]])
        xx = np.array([-3, 3])
        true_line.set_data(xx, d["slope"][ids].mean()*xx)
        from sklearn.linear_model import LinearRegression
        # Freeze the fitted data boundary throughout the final test period.
        cutoff = min(end-60, 480)
        fit = LinearRegression().fit(d["x"][:cutoff, None], d["y"][:cutoff])
        fitted_line.set_data(xx, fit.predict(xx[:, None]))
        cursor.set_xdata([ids.mean(), ids.mean()])
        left.set_title(f"Time {ids[0]}..{ids[-1]}; fit rows 0..{cutoff-1}")
    render.gif(fig, update, 9, "concept_drift", fps=2)
    fig, axs = canvas("What does each validation design measure?", "Same synthetic trajectory and linear model; development comparisons and reserved future test.", cols=2)
    for name, color in zip(d["scores"], [MUTED, TRAIN, VALID]):
        axs[0, 0].plot(range(1, 5), d["scores"][name], "o-", color=color, label=f"{name}: mean {np.mean(d['scores'][name]):.3f}")
    axs[0, 0].set(xlabel="Fold index (time order only for temporal methods)", ylabel="Validation MSE", title="Four development folds")
    axs[0, 0].legend(fontsize=9)
    names, values = list(d["test_mse"]), list(d["test_mse"].values())
    axs[0, 1].bar(names, values, color=[TRAIN, VALID])
    for i, value in enumerate(values):
        axs[0, 1].text(i, value, f"{value:.3f}", ha="center", va="bottom")
    axs[0, 1].set(xlabel="Predeclared history policy", ylabel="Final future MSE", title="Untouched times 480..599")
    footer(fig, "Lower MSE is better. Random folds mix regimes; temporal folds also differ in sample size and evaluation population.")
    render.png(fig, "concept_drift_comparison")
    return record({"rows": 600, "development": 480, "test": 120, "slope": "1 - 2*t/599", "noise_sd": 0.25, "model": "LinearRegression(x)"},
                  {"cv_mse": d["scores"], "test_mse": d["test_mse"]},
                  "Read the computed curves without assuming random CV must win; they estimate different prediction conditions.",
                  "A single synthetic trajectory does not isolate drift from training-size and period effects.")


def flow_box(ax, x, y, text, color):
    ax.text(x, y, text, ha="center", va="center", fontsize=10,
            bbox={"boxstyle": "round,pad=0.65", "facecolor": color, "edgecolor": "none"},
            color="white", transform=ax.transAxes)


def demonstrate_preprocessing_leakage(render, seed):
    d = exp.preprocessing_experiment(seed)
    fig, axs = canvas("Who was allowed to influence preprocessing?", "Two distinct controls: scaling uses covariates; supervised selection also uses labels.", rows=2, cols=2, size=(13, 9))
    for ax, unsafe in zip(axs[0], [True, False]):
        ax.set_axis_off()
        ax.set_title("INVALID: global fitting" if unsafe else "VALID: a fresh pipeline per fold")
        flow_box(ax, 0.5, 0.77, "All rows, including future validation" if unsafe else "Training fold only", RED if unsafe else TRAIN)
        flow_box(ax, 0.5, 0.30, "StandardScaler.fit -> CV" if unsafe else "Scaler.fit -> model.fit", RED if unsafe else TRAIN)
        ax.annotate("", xy=(0.5, 0.46), xytext=(0.5, 0.62), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "color": INK})
        if not unsafe:
            ax.text(0.5, -0.10, "Validation: transform + predict only", ha="center", transform=ax.transAxes, color=VALID)
    train = d["folds"][0][0]
    raw = d["X"][:, 0]
    axs[1, 0].hist(raw, bins=25, alpha=0.4, color=RED, label="All periods")
    axs[1, 0].hist(raw[train], bins=25, alpha=0.55, color=TRAIN, label="First training fold")
    for name, color in [("global", RED), ("local", TRAIN)]:
        scaler = d[name]
        axs[1, 0].axvline(scaler.mean_[0], color=color, linestyle="--",
                         label=f"{name}: mean {scaler.mean_[0]:.2f}, SD {scaler.scale_[0]:.2f}")
    axs[1, 0].set(xlabel="Raw feature 1", ylabel="Rows", title="Future shift changes global statistics")
    axs[1, 0].legend(fontsize=8)
    for name, color in [("global", RED), ("local", TRAIN)]:
        transformed = d[name].transform(d["X"][train])[:, 0]
        axs[1, 1].hist(transformed, bins=18, histtype="step", linewidth=2, color=color, label=f"Training rows / {name} scaler")
    axs[1, 1].axvline(0, color=MUTED, linestyle=":")
    axs[1, 1].set(xlabel="Transformed feature 1", ylabel="Training rows", title="Same rows; different fitted transformation")
    axs[1, 1].legend(fontsize=8)
    footer(fig, "First training fold shown. The model input uses both features; labels are generated from features plus noise.")
    render.png(fig, "preprocessing_leakage")
    fig, axs = canvas("Does every leak create a large score change?", "Paired folds within each experiment; the two experiments use different synthetic datasets.", cols=2)
    for name, values, color in [("Global scaling", d["global_auc"], RED), ("Fold-local scaling", d["local_auc"], TRAIN)]:
        axs[0, 0].plot(range(1, 4), values, "o-", color=color, label=f"{name}: {values.mean():.3f}")
    for (name, values), color in zip(d["selection"].items(), [RED, TRAIN]):
        axs[0, 1].plot(range(1, 5), values, "o-", color=color, label=f"{name}: {values.mean():.3f}")
    for ax, title in zip(axs[0], ["Scaling: temporal covariate shift", "Feature selection: random labels"]):
        ax.set(xlabel="Validation fold", ylabel="ROC-AUC", ylim=(0, 1.05), title=title)
        ax.axhline(0.5, linestyle=":", color=MUTED)
        ax.legend(fontsize=8)
    footer(fig, "A small scaling effect does not validate global fitting. Feature selection must also live inside the pipeline.")
    render.png(fig, "preprocessing_scores")
    fig, axs = canvas("Held-out covariates can move the global mean", "Sensitivity demonstration: shift a copied future block while training rows stay fixed.")
    ax = axs[0, 0]
    future = d["X"][-100:, 0].copy()
    fixed = d["X"][:-100, 0]
    train_points = ax.scatter(fixed, np.zeros(len(fixed)), s=12, color=TRAIN, alpha=0.5, label="Fixed earlier rows")
    moving = ax.scatter(future, np.ones(len(future)), s=18, color=VALID, alpha=0.6, label="Held-out future block")
    global_line = ax.axvline(raw.mean(), color=RED, label="Global fitted mean")
    ax.axvline(fixed.mean(), color=TRAIN, linestyle="--", label="Earlier-row fitted mean")
    ax.set(xlim=(-4, 11), ylim=(-0.5, 1.6), xlabel="Feature 1", yticks=[0, 1], yticklabels=["Earlier", "Held out"])
    ax.legend(loc="upper left", fontsize=8)
    annotation = ax.text(0.98, 0.95, "", ha="right", va="top", transform=ax.transAxes)
    footer(fig, "This perturbation is a statistics sensitivity check, not a new model-performance experiment.")
    def update(frame):
        delta = frame / 2
        shifted = future + delta
        moving.set_offsets(np.c_[shifted, np.ones(len(shifted))])
        mean = np.r_[fixed, shifted].mean()
        global_line.set_xdata([mean, mean])
        annotation.set_text(f"Held-out shift: +{delta:.1f}\nGlobal mean: {mean:.3f}")
    render.gif(fig, update, 9, "preprocessing_influence", fps=2)
    return record({"scaling_rows": 500, "features": 2, "last_100_feature_shift": 4, "temporal_folds": 3,
                   "model": "scaled LogisticRegression C=0.1", "selection_rows": 240, "noise_features": 500, "selected_features": 15, "selection_folds": 4},
                  {"global_mean": d["global"].mean_, "global_sd": d["global"].scale_,
                   "first_training_mean": d["local"].mean_, "first_training_sd": d["local"].scale_,
                   "global_scaling_auc": d["global_auc"], "local_scaling_auc": d["local_auc"], "selection_auc": d["selection"]},
                  f"Scaling mean AUC: global {d['global_auc'].mean():.4f}, fold-local {d['local_auc'].mean():.4f}. The information boundary is distinct from effect size.",
                  "One covariate-shift scenario and one noise-only selection scenario; no general leakage magnitude is claimed.")


def demonstrate_target_leakage(render, seed):
    d = exp.target_experiment(seed)
    fig, axs = canvas("A post-outcome feature changes the problem", "Both models are scored offline; one uses information that cannot exist at prediction time.", rows=2, cols=2, size=(13, 9))
    for (name, (fpr, tpr)), color in zip(d["curves"].items(), [TRAIN, RED]):
        axs[0, 0].plot(fpr, tpr, color=color, label=f"{name}\nAUC {d['aucs'][name]:.3f}")
    axs[0, 0].plot([0, 1], [0, 1], ":", color=MUTED)
    axs[0, 0].set(xlabel="False positive rate", ylabel="True positive rate", title="Same held-out rows")
    axs[0, 0].legend(fontsize=8, loc="lower right")
    for target, color in [(0, TRAIN), (1, VALID)]:
        axs[0, 1].hist(d["leaked"][d["y"] == target], bins=25, alpha=0.55, color=color, label=f"Target {target}")
    axs[0, 1].set(xlabel="Post-outcome measurement = y + noise", ylabel="Rows", title="Why the leaked feature is predictive")
    axs[0, 1].legend(fontsize=9)
    for ax, (name, matrix) in zip(axs[1], d["matrices"].items()):
        ax.imshow(matrix, cmap="Blues")
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(col, row, str(value), ha="center", va="center",
                    color="white" if value > matrix.max()/2 else INK, fontsize=14)
        ax.set(title=name, xlabel="Predicted class (threshold 0.5)", ylabel="True class",
               xticks=[0, 1], yticks=[0, 1])
    footer(fig, "Predeclared threshold 0.5, no tuning. This is an invalid offline control, not a deployable model comparison.")
    render.png(fig, "target_leakage")
    fig, axs = canvas("What exists at the prediction timestamp?", "A strong offline ROC curve does not make future features available.")
    ax = axs[0, 0]
    ax.axvline(0, color=INK, linewidth=2, label="Prediction timestamp")
    ax.axvspan(-3, 0, color=TRAIN, alpha=0.08)
    ax.axvspan(0, 4, color=RED, alpha=0.08)
    events = [(-2, 2, "Legitimate features", TRAIN), (1.2, 1, "Target occurs", VALID), (2.8, 0, "Post-outcome feature", RED)]
    for x, y, label, color in events:
        ax.scatter(x, y, color=color, s=130)
        ax.annotate(label, (x, y), xytext=(8, 8), textcoords="offset points", color=color)
    ax.set(xlim=(-3, 4.4), ylim=(-0.5, 2.8), xlabel="Relative event time (schematic)", yticks=[])
    ax.legend(loc="upper right")
    footer(fig, "Allowed information ends at prediction time, including ingestion delay. A pipeline cannot repair this feature.")
    render.png(fig, "target_information_timeline")
    return record({"rows": 900, "features": 10, "holdout_fraction": 0.3, "leaked_feature": "y + Normal(0, 0.3)", "model": "scaled logistic regression", "threshold": 0.5},
                  {"auc": d["aucs"], "confusion_matrices": d["matrices"]},
                  f"Available-feature AUC {d['aucs']['Available features']:.3f}; invalid post-outcome AUC {d['aucs']['With post-outcome feature']:.3f}. The latter solves a different information problem.",
                  "The leaked measurement is deliberately target-derived; the result is not a production forecast.")


def visualize_temporal_feature_leakage(render, seed):
    d = exp.purchase_experiment(seed)
    fig, axs = canvas("Can March know the full-year purchase average?", "One synthetic customer's monthly purchases; prediction occurs at the end of March.")
    ax = axs[0, 0]
    months = np.arange(1, 13)
    ax.bar(months, d["monthly"], color=[TRAIN]*3 + [RED]*9, label="Monthly purchase amount")
    ax.axvline(3.5, color=INK, linewidth=2, label="Prediction cutoff")
    ax.hlines(d["asof_march"], 0.5, 3.5, color=TEST, linewidth=3, label=f"Available average: {d['asof_march']:.2f}")
    ax.axhline(d["full_year"], color=RED, linestyle="--", label=f"Leaking average: {d['full_year']:.2f}")
    ax.set(xticks=months, xticklabels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
           xlabel="Month", ylabel="Synthetic purchase units", ylim=(0, d["monthly"].max()*1.5))
    ax.legend(loc="upper left", fontsize=9)
    footer(fig, "The full-year average uses future purchases. Values are in synthetic units; no classifier is fitted in this view.")
    render.png(fig, "temporal_feature_leakage")
    return record({"customers": 1, "months": 12, "generator": "LogNormal(4, 0.35) + 3*month_index", "cutoff": "end of March"},
                  d, "The two computed averages use different availability windows, regardless of their numerical gap.",
                  "A single customer's feature computation, not a predictive performance experiment.")


def simulate_validation_overfitting(render, seed):
    d = exp.overfitting_experiment(seed)
    fig, axs = canvas("Can model selection fit validation noise?", "200 hypothetical null classifiers; every score is computed from generated predictions.", cols=2)
    count = np.arange(1, len(d["winners"])+1)
    axs[0, 0].plot(count, d["best_validation"], color=VALID, label="Best validation so far")
    axs[0, 0].plot(count, d["winner_test"], color=TEST, label="Same winner: independent test")
    axs[0, 0].axhline(0.5, color=MUTED, linestyle=":", label="Population AUC under null")
    axs[0, 0].set(xlabel="Configurations examined", ylabel="ROC-AUC", title="Winner chosen using validation only", ylim=(0.35, 0.85))
    axs[0, 0].legend(fontsize=8)
    axs[0, 1].scatter(d["validation"], d["test"], color=MUTED, alpha=0.45, s=18, label="Candidate")
    winner = d["winners"][-1]
    axs[0, 1].scatter(d["validation"][winner], d["test"][winner], color=VALID, s=100, marker="*", label=f"Final winner #{winner+1}")
    axs[0, 1].set(xlabel="Validation AUC (80 rows)", ylabel="Independent test AUC (2,000 rows)", title="Selection favors a fortunate validation draw")
    axs[0, 1].legend(fontsize=9)
    footer(fig, "Test curves are a post-hoc audit of predeclared prefixes, never a selection input. These are simulated score functions, not trained models.")
    render.png(fig, "validation_overfitting")
    return record({"candidates": 200, "validation_rows": 80, "test_rows": 2000, "predictions": "independent Normal scores", "labels": "balanced", "selection": "argmax validation AUC; first tie wins"},
                  {"validation_auc": d["validation"], "test_auc": d["test"], "winner_indices": d["winners"],
                   "final_winner": int(winner+1), "final_validation_auc": float(d["best_validation"][-1]), "final_test_auc": float(d["winner_test"][-1])},
                  f"Selected candidate {winner+1}: validation AUC {d['best_validation'][-1]:.3f}, independent test AUC {d['winner_test'][-1]:.3f}. More candidates provide more chances to select noise.",
                  "Independent hypothetical candidates simplify real correlated searches. One trajectory does not estimate selection-bias uncertainty.")


def visualize_nested_cv(render, seed):
    d = exp.nested_experiment(seed)
    matrix, labels = [], []
    # Order by outer holdout to expose containment rather than scrambled row IDs.
    order = np.concatenate([v for _, v in d["outer"]])
    for i, ((train, valid), inner) in enumerate(zip(d["outer"], d["inner"])):
        outer_row = np.ones(d["n"], dtype=int)
        outer_row[valid] = 3
        matrix.append(outer_row[order])
        labels.append(f"Outer {i+1}: refit + audit")
        for j, (t, v) in enumerate(inner):
            row = np.zeros(d["n"], dtype=int)
            row[t], row[v], row[valid] = 1, 2, 3
            matrix.append(row[order])
            labels.append(f"    Inner {j+1}: select C")
    fig, axs = canvas("Selection stays inside each outer training set", "The green outer holdout never enters the inner search. Selected pipelines are refitted before outer scoring.", size=(13, 8))
    ax = axs[0, 0]
    ax.imshow(matrix, cmap=CMAP, vmin=0, vmax=3, aspect="auto", interpolation="nearest")
    ax.set(yticks=range(len(labels)), yticklabels=labels, xlabel="Rows ordered by outer holdout membership", ylabel="Evaluation stage")
    ax.legend(handles=[Patch(color=TRAIN, label="Fit"), Patch(color=VALID, label="Inner validation"), Patch(color=TEST, label="Outer holdout")],
              loc="upper left", bbox_to_anchor=(0, -0.14), ncol=3)
    fig.subplots_adjust(left=0.22, bottom=0.22)
    footer(fig, "Three outer folds x three inner folds x four C values. Outer holdouts assess the search procedure, not one fixed hyperparameter.")
    render.png(fig, "nested_cv")
    fig, axs = canvas("What did each inner search select?", "Actual GridSearchCV runs; no outer score participates in choosing C.")
    ax = axs[0, 0]
    positions = np.arange(1, 4)
    inner_scores = [r["inner_best_auc"] for r in d["rows"]]
    outer_scores = [r["outer_auc"] for r in d["rows"]]
    ax.bar(positions-0.18, inner_scores, width=0.36, color=VALID, label="Winning inner CV AUC")
    ax.bar(positions+0.18, outer_scores, width=0.36, color=TEST, label="Outer holdout AUC")
    for i, row in enumerate(d["rows"], 1):
        ax.text(i, max(row["inner_best_auc"], row["outer_auc"])+0.03, f"C = {row['C']:g}", ha="center")
    ax.set(xlabel="Outer fold", ylabel="ROC-AUC", ylim=(0, 1.12), xticks=positions)
    ax.legend(loc="lower right")
    footer(fig, "Differences may have either sign. Three outer scores are not independent evidence of a universal selection-bias magnitude.")
    render.png(fig, "nested_cv_scores")
    return record({"rows": 360, "features": 10, "outer_folds": 3, "inner_folds": 3, "C_grid": [0.01, 0.1, 1, 10], "model": "scaled logistic regression"},
                  {"outer_results": d["rows"], "mean_outer_auc": float(np.mean(outer_scores))},
                  f"Outer mean AUC {np.mean(outer_scores):.3f}; each displayed C was selected exclusively inside that outer training partition.",
                  "IID-like synthetic classification; outer fold dispersion is not an exact confidence interval.")


def plot_holdout_variability(render, seed):
    d = exp.holdout_experiment(seed)
    fig, axs = canvas("How sensitive is the estimate to partition choice?", "Fixed dataset and model; 16 split seeds. Both methods train each fit on 80% of the rows.", cols=2)
    all_values = np.r_[d["holdout"], d["cv"]]
    bins = np.linspace(all_values.min()-0.01, all_values.max()+0.01, 10)
    for name, key, color in [("Single holdout", "holdout", VALID), ("Mean of 5 folds", "cv", TRAIN)]:
        values = d[key]
        axs[0, 0].hist(values, bins=bins, alpha=0.5, color=color, label=f"{name}: SD {values.std(ddof=1):.3f}")
        axs[0, 1].plot(range(1, 17), values, "o-", color=color, label=name)
    axs[0, 0].set(xlabel="Validation ROC-AUC estimate", ylabel="Split seeds", title="Observed distribution")
    axs[0, 1].set(xlabel="Split-seed repetition", ylabel="ROC-AUC estimate", title="Paired partition sensitivity")
    for ax in axs[0]:
        ax.legend(fontsize=8)
    footer(fig, "Repeated splits reuse the same rows. This shows partition variability, not bias or population-level confidence intervals.")
    render.png(fig, "holdout_variability")
    return record({"rows": 400, "features": 10, "repetitions": 16, "holdout_fraction": 0.2, "cv_folds": 5, "model": "scaled logistic regression"},
                  {"scores": d, "holdout_sd": float(d["holdout"].std(ddof=1)), "cv_mean_sd": float(d["cv"].std(ddof=1))},
                  f"Observed SD: holdout {d['holdout'].std(ddof=1):.4f}, CV means {d['cv'].std(ddof=1):.4f}. Compare this draw without assuming CV always reduces variance.",
                  "No population ground truth or independent repeated datasets; bias is not measured.")


def plot_temporal_split_3d(render, seed):
    rng = np.random.default_rng(seed)
    n = 240
    X = rng.normal(size=(n, 2))
    t = np.arange(n)
    random_parts = exp.split_three(n, seed)
    temporal_parts = [np.arange(144), np.arange(144, 192), np.arange(192, 240)]
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                        subplot_titles=["Random: periods mixed", "Temporal: ordered slices"])
    for col, parts in enumerate([random_parts, temporal_parts], 1):
        for name, color, ids in zip(NAMES, COLORS, parts):
            fig.add_trace(go.Scatter3d(
                x=X[ids, 0], y=X[ids, 1], z=t[ids], mode="markers",
                marker={"size": 3, "color": color, "opacity": 0.75},
                name=name, legendgroup=name, showlegend=col == 1,
                customdata=ids, hovertemplate="Row %{customdata}<br>x1=%{x:.2f}<br>x2=%{y:.2f}<br>time=%{z}<extra>%{fullData.name}</extra>",
            ), row=1, col=col)
    for cutoff in [143.5, 191.5]:
        fig.add_trace(go.Surface(
            x=[-3.5, 3.5], y=[-3.5, 3.5], z=np.full((2, 2), cutoff),
            opacity=0.15, showscale=False, colorscale=[[0, INK], [1, INK]],
            hoverinfo="skip", name="Temporal cutoff",
        ), row=1, col=2)
    scene = {"xaxis_title": "Feature 1", "yaxis_title": "Feature 2", "zaxis_title": "Time step",
             "xaxis": {"range": [-3.5, 3.5]}, "yaxis": {"range": [-3.5, 3.5]}, "zaxis": {"range": [0, 240]},
             "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.0}}}
    fig.update_layout(
        title={"text": "The temporal boundary is geometric<br><sup>Same rows in both panels. Rotate, zoom and hover; no model is trained.</sup>", "x": 0.04},
        scene=scene, scene2=scene, height=760, template="plotly_white",
        paper_bgcolor=PAPER, margin={"l": 15, "r": 15, "t": 100, "b": 50},
    )
    fig.add_annotation(text="Synthetic features; z is ordered time. Planes mark the predeclared temporal cutoffs.", x=0.5, y=-0.05, xref="paper", yref="paper", showarrow=False)
    render.html(fig, "temporal_split_3d")
    return record({"rows": n, "features": "two independent Normal variables", "time": "row index", "temporal_cutoffs": [144, 192]},
                  {"partition_sizes": [144, 48, 48]},
                  "Temporal splitting creates disjoint time ranges; random splitting interleaves them along the z-axis.",
                  "Geometric membership demonstration; 3D position does not establish predictive difficulty.")


def visualize_production_boundary(render, seed):
    fig, axs = canvas("What can the model know at inference time?", "Availability, not retrospective database presence, defines the permitted feature set.")
    ax = axs[0, 0]
    ax.axvspan(-5, 0, color=TRAIN, alpha=0.07)
    ax.axvspan(0, 5, color=RED, alpha=0.07)
    ax.axvline(0, color=INK, linewidth=2)
    events = [
        (-4.3, 4, "Customer history", TRAIN), (-2.8, 3, "Past transactions", TRAIN),
        (-1.4, 2, "Ingested features", TRAIN), (1.2, 4, "Target event", RED),
        (2.4, 3, "Collections event", RED), (3.5, 2, "Future purchases", RED),
    ]
    for x, y, label, color in events:
        ax.scatter(x, y, color=color, marker="o" if x < 0 else "x", s=90)
        ax.text(x, y-0.45, label, ha="center", fontsize=10, color=color)
    ax.text(-2.5, 5.1, "AVAILABLE", ha="center", color=TRAIN, weight="bold")
    ax.text(2.5, 5.1, "NOT AVAILABLE", ha="center", color=RED, weight="bold")
    ax.text(0, 0.6, "PREDICTION", ha="center", weight="bold", bbox={"facecolor": PAPER, "edgecolor": "none"})
    ax.set(xlim=(-5.5, 5.5), ylim=(0, 5.8), xlabel="Relative event / availability time (schematic)", yticks=[])
    ax.legend(handles=[Line2D([], [], marker="o", color=TRAIN, linestyle="", label="Permitted"),
                       Line2D([], [], marker="x", color=RED, linestyle="", label="Unavailable")], loc="lower right")
    footer(fig, "Also preserve the deployment population: new entities, later periods, or both. Audit label delay and point-in-time joins.")
    render.png(fig, "production_information_boundary")
    return record({"type": "conceptual availability diagram"}, {"available_events": 3, "unavailable_events": 3},
                  "A legitimate validation design reproduces the information available for the intended production prediction.",
                  "Event placement is schematic. A real application must establish its own availability timestamps.")


def visualize_rag_leakage(render, seed):
    fig, axs = canvas("Are chunks independent evaluation units?", "Example deployment target: generalize to unseen document families and held-out questions.", rows=2, cols=2, size=(13, 9))
    chunk_split = np.tile(np.arange(1, 4), (3, 1))
    document_split = np.repeat(np.arange(1, 4)[:, None], 3, axis=1)
    for ax, matrix, title in zip(axs[0], [chunk_split, document_split],
                                  ["Mixed chunks: a document crosses splits", "Grouped documents: one split per family"]):
        ax.imshow(matrix, cmap=CMAP, vmin=0, vmax=3)
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(col, row, f"{chr(65+row)}{col+1}\n{NAMES[value-1]}", ha="center", va="center", color="white", fontsize=10)
        ax.set(title=title, xlabel="Derived chunk", ylabel="Source document", xticks=range(3), xticklabels=[1, 2, 3], yticks=range(3), yticklabels=["A", "B", "C"])
    for ax in axs[1]:
        ax.set_axis_off()
    flow_box(axs[1, 0], 0.5, 0.75, "Prompt development questions", TRAIN)
    flow_box(axs[1, 0], 0.5, 0.15, "Validation questions -> prompt choices", VALID)
    axs[1, 0].annotate("iteration", xy=(0.80, 0.63), xytext=(0.80, 0.30), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "color": INK})
    flow_box(axs[1, 1], 0.5, 0.75, "Freeze prompt + retrieval decisions", INK)
    flow_box(axs[1, 1], 0.5, 0.15, "Final held-out evaluation questions", TEST)
    axs[1, 1].annotate("", xy=(0.5, 0.32), xytext=(0.5, 0.57), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "color": INK})
    footer(fig, "A source document may legitimately be retrievable at test time. Keep reference answers out of tuning; match the retrieval contract.")
    render.png(fig, "rag_document_leakage")
    return record({"documents": 3, "chunks_per_document": 3, "type": "conceptual split diagram"},
                  {"mixed_document_splits": chunk_split, "grouped_document_splits": document_split},
                  "Repeated prompt edits turn inspected evaluation questions into development data. Grouping targets unseen-document generalization.",
                  "No RAG model was executed. Legitimate retrieval of test evidence is not automatically leakage.")

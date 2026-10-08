"""Generate Day 34 scientific figures, GIF, offline 3D views, and measured reports.

Run: python visualize_missing_data.py
No network, GPU, dashboard, notebook, or web backend is used.
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import sklearn
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits

import missing_data_visual_core as core

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
BLUE, ORANGE, TEAL, PURPLE = "#24678d", "#d16a28", "#16856e", "#8456a6"
GRAY = "#a5adb4"
METHOD_COLORS = {"Mean": ORANGE, "Median": PURPLE, "KNN": TEAL}
STRATEGY_COLORS = dict(zip(core.STRATEGIES, (BLUE, ORANGE, TEAL, PURPLE)))
GENERATED = []


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10, "figure.titlesize": 16,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": .16, "savefig.facecolor": "white",
    })


def register(path):
    GENERATED.append(path)
    print(f"Generated: {path}", flush=True)


def title(fig, heading, subtitle, footer=None):
    fig.suptitle(heading, y=.99)
    fig.text(.5, .92, subtitle, ha="center", fontsize=10, color="#404b53")
    if footer:
        fig.text(.5, .015, footer, ha="center", fontsize=9, color="#404b53")


def save(fig, filename):
    fig.tight_layout(rect=(0, .065, 1, .88))
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=140)
    plt.close(fig)
    register(path)
    return path


def mechanism_scatter(ax, X, missing, heading):
    ax.scatter(X[~missing, 0], X[~missing, 1], s=14, color=BLUE, alpha=.50, label="Observed")
    ax.scatter(X[missing, 0], X[missing, 1], s=24, marker="x", color=ORANGE,
               alpha=.80, label="Hidden income: true position")
    ax.set(xlabel="Age-like (observed)", ylabel="Income-like (synthetic units)", title=heading)


def plot_missingness_mechanisms(X):
    sample = X[:450]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True, sharey=True)
    equations = {
        "MCAR": r"$P(R\mid X_{obs},X_{mis})=P(R)$",
        "MAR": r"$P(R\mid X_{obs},X_{mis})=P(R\mid X_{obs})$",
        "MNAR": "Missingness still depends on the hidden value",
    }
    meanings = {"MCAR": "Unrelated to data", "MAR": "Depends on observed age", "MNAR": "Depends on true income"}
    for ax, mechanism in zip(axes, ("MCAR", "MAR", "MNAR")):
        _, mask, _ = core.inject_mechanism(sample, mechanism)
        mechanism_scatter(ax, sample, mask, f"{mechanism}: {meanings[mechanism]}\n{mask.mean():.1%} hidden")
        ax.text(.03, .97, equations[mechanism], transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(facecolor="white", alpha=.9, edgecolor="none"))
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -.16), ncols=2, fontsize=9)
    title(fig, "Three ways to lose income-like values",
          "Same complete point cloud and random uniforms; different missingness probabilities",
          "R=1 means observed. Hidden true positions are visible only because this dataset is synthetic.")
    save(fig, "01_mcar_mar_mnar.png")


def plot_missingness_probability():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    age = np.linspace(18, 80, 300)
    income = np.linspace(5, 150, 300)
    for ax, mechanism in zip(axes, ("MCAR", "MAR", "MNAR")):
        X = np.column_stack((age, income))
        predictor = income if mechanism == "MNAR" else age
        ax.plot(predictor, core.missingness_probability(X, mechanism), lw=3, color=ORANGE)
        ax.set(ylim=(0, 1), ylabel="P(income-like value is missing)", title=mechanism,
               xlabel="Income-like: eventually hidden" if mechanism == "MNAR" else "Age-like: remains observed")
        formula = {"MCAR": "p = 0.30",
                   "MAR": "p = 0.05 + 0.75 sigmoid((age - 45) / 5)",
                   "MNAR": "p = 0.05 + 0.75 sigmoid((log(income) - 3.65) / 0.20)"}[mechanism]
        ax.text(.5, .05, formula, ha="center", transform=ax.transAxes, fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.9))
    title(fig, "Missingness is a probability model",
          "R=1 is observed; these curves show P(R=0), the probability of absence",
          "Mechanisms are known by construction. Observed real data generally cannot conclusively distinguish MAR from MNAR.")
    save(fig, "02_missingness_probability.png")


def create_missingness_animation(X):
    X = X[:350]
    fractions = np.r_[np.linspace(0, 1, 25), np.ones(6)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharex=True, sharey=True)
    def update(frame):
        fraction = fractions[frame]
        previous = fractions[max(0, frame - 1)]
        for ax, mechanism in zip(axes, ("MCAR", "MAR", "MNAR")):
            ax.clear()
            _, mask, _ = core.inject_mechanism(X, mechanism, fraction=fraction)
            _, old, _ = core.inject_mechanism(X, mechanism, fraction=previous)
            newly = mask & ~old
            ax.scatter(X[old, 0], X[old, 1], s=18, facecolors="none", edgecolors=GRAY, alpha=.5, label="Already hidden: truth")
            ax.scatter(X[~mask, 0], X[~mask, 1], s=13, color=BLUE, alpha=.6, label="Observed")
            ax.scatter(X[newly, 0], X[newly, 1], s=40, marker="x", color=ORANGE, label="Newly hidden: truth")
            ax.set(xlim=(16, 82), ylim=(0, X[:, 1].max() * 1.05),
                   xlabel="Age-like (observed)", ylabel="Income-like",
                   title=f"{mechanism} | {mask.mean():.1%} income hidden")
        axes[0].legend(loc="upper left", fontsize=8)
        fig.suptitle(f"Missingness unfolding | mechanism exposure {fraction:.0%}", y=.99)
    update(0)
    fig.text(.5, .035, "Colored crosses become gray reference outlines; true hidden positions are synthetic-only information.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, .91))
    animation = FuncAnimation(fig, update, frames=len(fractions), interval=250, repeat=True)
    path = OUTPUT_DIR / "03_missingness_process.gif"
    animation.save(path, writer=PillowWriter(fps=4), dpi=100)
    plt.close(fig)
    register(path)


def plot_imputation_distribution(X):
    masked, mask, _ = core.inject_mnar(X)
    filled = core.imputed_versions(masked)
    versions = {"Complete truth": X[:, 1], "Observed only": X[~mask, 1]}
    versions.update({f"{name} imputed": values[:, 1] for name, values in filled.items()})
    fig, axes = plt.subplots(1, 5, figsize=(16, 5), sharex=True, sharey=True)
    bins = np.linspace(X[:, 1].min(), X[:, 1].max(), 38)
    stats = {}
    for ax, (name, values) in zip(axes, versions.items()):
        color = METHOD_COLORS.get(name.split()[0], BLUE)
        ax.hist(values, bins=bins, density=True, color=color, alpha=.75)
        stats[name] = core.distribution_stats(values)
        s = stats[name]
        ax.set(title=f"{name}\nn={s['n']:,}", xlabel="Income-like (synthetic units)")
        ax.text(.97, .95, f"mean {s['mean']:.1f}\nvariance {s['variance']:.1f}\nstd {s['std']:.1f}\nmedian {s['median']:.1f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
        if name in ("Mean imputed", "Median imputed"):
            value = np.mean(X[~mask, 1]) if name == "Mean imputed" else np.median(X[~mask, 1])
            ax.axvline(value, color="#222222", ls="--", lw=1)
    axes[0].set_ylabel("Density (shared bins and scale)")
    title(fig, "Keeping rows does not restore their distribution",
          "MNAR hides high income-like values; constant filling creates a pile-up",
          "Descriptive same-sample filling, not predictive evaluation. Variance uses ddof=0; complete truth is synthetic-only.")
    save(fig, "04_imputation_distribution_distortion.png")
    return stats


def geometry_data():
    rng = np.random.default_rng(43)
    x = rng.normal(size=320)
    X = np.column_stack((x, 1.2 * x + rng.normal(scale=.45, size=len(x))))
    mask = rng.random(len(X)) < .38
    masked = X.copy()
    masked[mask, 1] = np.nan
    return X, mask, core.imputed_versions(masked)


def plot_imputation_geometry():
    X, mask, filled = geometry_data()
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharex=True, sharey=True)
    errors = {}
    for ax, (name, values) in zip(axes, filled.items()):
        ax.scatter(X[~mask, 0], X[~mask, 1], s=13, color=BLUE, alpha=.35, label="Observed")
        ax.scatter(X[mask, 0], X[mask, 1], s=28, facecolors="none", edgecolors=GRAY, label="Hidden true position")
        ax.scatter(values[mask, 0], values[mask, 1], s=20, marker="x", color=METHOD_COLORS[name], label="Imputed position")
        for index in np.flatnonzero(mask)[::9]:
            ax.plot([X[index, 0], values[index, 0]], [X[index, 1], values[index, 1]], color=METHOD_COLORS[name], alpha=.4, lw=.8)
        error = float(np.sqrt(np.mean((values[mask, 1] - X[mask, 1]) ** 2)))
        errors[name] = error
        ax.set(xlabel="Observed X1", ylabel="Partly hidden X2", title=f"{name} imputation | hidden RMSE {error:.3f}")
    axes[0].legend(loc="upper left", fontsize=8)
    title(fig, "Imputation moves points in feature space",
          "Mean/median collapse points to a line; KNN varies with local observed X1",
          "Identical MCAR mask in every panel. Segments show a small subset of errors; this is a descriptive same-sample example.")
    save(fig, "05_imputation_geometry.png")
    return errors


def plot_correlation_distortion():
    X, mask, filled = geometry_data()
    versions = {"Complete truth": X, **filled}
    fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharex=True, sharey=True)
    records = {}
    for ax, (name, values) in zip(axes, versions.items()):
        color = METHOD_COLORS.get(name, BLUE)
        ax.scatter(values[~mask, 0], values[~mask, 1], color=BLUE, s=12, alpha=.4)
        ax.scatter(values[mask, 0], values[mask, 1], color=color, s=22, marker="x")
        r = float(np.corrcoef(values.T)[0, 1])
        cov = float(np.cov(values.T, ddof=0)[0, 1])
        var = float(np.var(values[:, 1]))
        records[name] = dict(pearson=r, covariance=cov, variance_x2=var)
        ax.set(title=f"{name}\nPearson r = {r:.3f}", xlabel="X1", ylabel="X2")
        ax.text(.03, .97, f"Cov(X1,X2)={cov:.3f}\nVar(X2)={var:.3f}\nn={len(values)}", transform=ax.transAxes,
                va="top", fontsize=9, bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
    title(fig, "Equal sample counts, different covariance structure",
          "Crosses mark the same hidden rows, restored by different estimators",
          "KNN can also inflate correlation by reducing conditional residual variation; closeness to r=1 is not proof of recovery.")
    save(fig, "06_correlation_distortion.png")
    return records


def plot_missing_indicator_effect(result):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    X, y, mask = result["complete"], result["y"], result["missing"]
    sample = np.arange(min(400, len(y)))
    for label, color in ((0, BLUE), (1, ORANGE)):
        ids = sample[y[sample] == label]
        axes[0].scatter(X[ids, 0], X[ids, 1], s=16, color=color, alpha=.55, label=f"Target {label}")
    hidden = sample[mask[sample]]
    axes[0].scatter(X[hidden, 0], X[hidden, 1], s=35, facecolors="none", edgecolors="#222222", lw=.7, label="Measurement hidden")
    axes[0].set(xlabel="Observed context", ylabel="Measurement: complete synthetic truth", title="Which class loses measurements?")
    axes[0].legend(fontsize=8)
    proportions = []
    for i, (name, group) in enumerate((("Observed", ~mask), ("Missing", mask))):
        proportion = float(y[group].mean())
        proportions.append(dict(group=name, n=int(group.sum()), target_prevalence=proportion))
        axes[1].bar(i, proportion, color=(BLUE, ORANGE)[i], width=.5)
        axes[1].text(i, proportion + .035, f"{proportion:.1%}\nn={group.sum()}", ha="center", fontsize=9)
    axes[1].set(xticks=[0, 1], xticklabels=["Observed", "Missing"], ylim=(0, 1),
                ylabel="Held-out positive fraction", title="Absence reveals the latent regime")
    for i, (name, (fpr, tpr)) in enumerate(result["curves"].items()):
        axes[2].plot(fpr, tpr, lw=2.5, color=(BLUE, TEAL)[i], label=f"{name}: AUC {result['auc'][name]:.3f}")
    axes[2].plot([0, 1], [0, 1], ls="--", color=GRAY)
    axes[2].set(xlabel="False positive rate", ylabel="True positive rate", title="Same logistic classifier and split")
    axes[2].legend(loc="lower right", fontsize=8)
    title(fig, "A missingness indicator can carry predictive signal",
          "An unobserved regime causes both target propensity and measurement absence; the mask never reads labels",
          "Usefulness depends on this generator. A changed collection policy can make this indicator a fragile shortcut.")
    save(fig, "07_missing_indicator.png")
    return dict(auc=result["auc"], groups=proportions, test_n=int(len(y)))


def flow_box(ax, xy, label, color, width=.23):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), width, .14, boxstyle="round,pad=.012",
                               facecolor=color, alpha=.13, edgecolor=color))
    ax.text(x + width / 2, y + .07, label, ha="center", va="center", fontsize=9)


def arrow(ax, start, end, color=BLUE):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, color=color, lw=1.8))


def plot_leakage_diagram():
    rng = np.random.default_rng(44)
    train, test = rng.normal(0, 1, 200), rng.normal(4, 1, 200)
    train[rng.random(200) < .30] = np.nan
    test[rng.random(200) < .30] = np.nan
    proper = float(SimpleImputer(strategy="median").fit(train[:, None]).statistics_[0])
    # Intentionally invalid information flow, isolated to this diagram.
    leaked = float(SimpleImputer(strategy="median").fit(np.r_[train, test][:, None]).statistics_[0])
    fig = plt.figure(figsize=(13, 6.3))
    hist = fig.add_subplot(2, 1, 1)
    bins = np.linspace(-3, 8, 35)
    hist.hist(train[np.isfinite(train)], bins=bins, alpha=.5, color=BLUE, label="Training observed values")
    hist.hist(test[np.isfinite(test)], bins=bins, alpha=.5, color=ORANGE, label="Held-out observed values (shifted)")
    hist.axvline(proper, color=BLUE, lw=2, label=f"Training median: {proper:.2f}")
    hist.axvline(leaked, color=ORANGE, lw=2, ls="--", label=f"Pooled median: {leaked:.2f}")
    hist.set(xlabel="Synthetic feature", ylabel="Observed count")
    hist.legend(fontsize=9, ncols=2)
    diagram = fig.add_subplot(2, 1, 2)
    diagram.set(xlim=(0, 1), ylim=(0, 1))
    diagram.axis("off")
    flow_box(diagram, (.02, .68), "Split first\ntrain / held-out", BLUE)
    flow_box(diagram, (.37, .68), f"Fit using train only\nmedian = {proper:.2f}", BLUE)
    flow_box(diagram, (.72, .68), "Transform both\nfrozen statistic", BLUE)
    arrow(diagram, (.25, .75), (.36, .75))
    arrow(diagram, (.60, .75), (.71, .75))
    flow_box(diagram, (.02, .21), "Pool train + held-out\nfuture information", ORANGE)
    flow_box(diagram, (.37, .21), f"Fit before split\nmedian = {leaked:.2f}", ORANGE)
    flow_box(diagram, (.72, .21), "Split afterwards\ncontaminated statistic", ORANGE)
    arrow(diagram, (.25, .28), (.36, .28), ORANGE)
    arrow(diagram, (.60, .28), (.71, .28), ORANGE)
    title(fig, "Imputation leakage is an information-flow violation",
          "The held-out distribution changes the fitted statistic when fitting happens before the split",
          "The shifted distributions expose the dependency. No predictive improvement or benchmark gain is claimed.")
    save(fig, "08_imputation_leakage.png")
    return dict(training_median=proper, pooled_median=leaked, n_train=200, n_test=200)

def plot_curves(rows, filename, shifted=False):
    summary = core.summarize(rows)
    fig, ax = plt.subplots(figsize=(10, 5.7))
    for strategy in dict.fromkeys(row["strategy"] for row in summary):
        points = sorted((row for row in summary if row["strategy"] == strategy), key=lambda row: row["rate"])
        rates = 100 * np.array([row["rate"] for row in points])
        means = np.array([row["mean"] for row in points])
        std = np.array([row["std"] for row in points])
        ax.plot(rates, means, marker="o", lw=2.2, color=STRATEGY_COLORS[strategy], label=strategy)
        ax.fill_between(rates, means - std, means + std, color=STRATEGY_COLORS[strategy], alpha=.12)
    ax.set(xlabel="Requested MCAR cell missingness (%)", ylabel="Held-out ROC-AUC", ylim=(.45, 1))
    ax.legend(loc="lower left", fontsize=10)
    if shifted:
        ax.axvline(10, color="#222222", ls="--", lw=1.5)
        ax.text(11, .965, "Training condition: 10%", fontsize=10)
        heading = "Frozen models under missingness shift"
        subtitle = r"$P_{train}(R) \ne P_{prod}(R)$ | Same complete test values and labels; only availability changes"
        footer = "Five fixed seeds; shading is +/- one sample SD, not a confidence interval. Fitted models never see stressed test statistics."
    else:
        heading = "Prediction quality as information becomes unavailable"
        subtitle = "Refit at each rate | Same histogram boosting settings; four preprocessing / NaN-routing strategies"
        footer = "Five fixed seeds; shading is +/- one sample SD, not a confidence interval. Matched train/test MCAR, training-only preprocessing."
    title(fig, heading, subtitle, footer)
    save(fig, filename)
    return summary


def plot_feature_dropout(rows):
    names = list(dict.fromkeys(row["scenario"] for row in rows))
    means, stds, aucs = [], [], []
    for name in names:
        group = [row for row in rows if row["scenario"] == name]
        deltas = np.array([row["delta_from_baseline"] for row in group])
        means.append(float(deltas.mean()))
        stds.append(float(deltas.std(ddof=1)) if len(deltas) > 1 else 0)
        aucs.append(float(np.mean([row["auc"] for row in group])))
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(names, means, xerr=stds, color=[BLUE] + [ORANGE] * (len(names) - 1), alpha=.85, capsize=4)
    ax.axvline(0, color="#222222", lw=1)
    ax.invert_yaxis()
    ax.set(xlabel="ROC-AUC change from all-features-available baseline")
    low = min(np.array(means) - np.array(stds))
    ax.set_xlim(min(-.05, low - .025), .065)
    for i, value in enumerate(aucs):
        ax.text(.008, i, f"AUC {value:.3f}", va="center", fontsize=10)
    title(fig, "Feature availability is an engineering contract",
          "Train with complete features; hide entire test columns and reuse the training-median fallback",
          "Feature-store failure | upstream API timeout | failed JOIN | sensor outage | document extraction failure")
    save(fig, "11_feature_dropout_robustness.png")
    return [dict(scenario=name, mean_auc=value, mean_delta=delta, std_delta=std)
            for name, value, delta, std in zip(names, aucs, means, stds)]


def create_interactive_3d(X):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Skipping optional 3D views. Install with: python -m pip install plotly", flush=True)
        return []
    X = X[:500, :3]
    fig = go.Figure()
    mechanisms = ("Complete", "MCAR", "MAR", "MNAR")
    for mechanism in mechanisms:
        mask = np.zeros(len(X), dtype=bool) if mechanism == "Complete" else core.inject_mechanism(X, mechanism)[1]
        for state, selected, color in (("Observed", ~mask, BLUE), ("Hidden income: synthetic truth", mask, ORANGE)):
            fig.add_trace(go.Scatter3d(
                x=X[selected, 0], y=X[selected, 1], z=X[selected, 2],
                mode="markers", name=state, visible=mechanism == "Complete",
                marker=dict(size=3.5, color=color, opacity=.65,
                            symbol="circle" if state == "Observed" else "x"),
            ))
    buttons = []
    for i, mechanism in enumerate(mechanisms):
        visible = [j // 2 == i for j in range(8)]
        buttons.append(dict(label=mechanism, method="update",
                            args=[{"visible": visible}, {"title": {"text": f"3D missingness | {mechanism}", "x": .5, "xanchor": "center", "y": .98}}]))
    fig.update_layout(
        title=dict(text="3D missingness | Complete", x=.5, xanchor="center", y=.98),
        height=800, template="plotly_white",
        scene=dict(xaxis_title="Age-like", yaxis_title="Income-like",
                   zaxis_title="Risk-like", aspectmode="cube"),
        updatemenus=[dict(buttons=buttons, x=0, y=1.12)],
        margin=dict(l=20, r=20, t=115, b=130),
        legend=dict(orientation="h", x=0, y=-.05),
        annotations=[dict(text="Rotate to inspect selection. Orange points show hidden synthetic truth, unavailable in real observed-only data.",
                          x=.5, y=-.12, xref="paper", yref="paper", showarrow=False)],
    )
    path = OUTPUT_DIR / "12_missingness_3d.html"
    fig.write_html(path, include_plotlyjs=True, full_html=True, auto_open=False)
    register(path)
    create_imputation_3d(go, X)
    return ["12_missingness_3d.html", "13_imputation_3d.html"]


def create_imputation_3d(go, X):
    masked, mask, _ = core.inject_mnar(X)
    versions = core.imputed_versions(masked)
    fig = go.Figure()
    def points(values, selected, label, color, symbol="circle"):
        return go.Scatter3d(x=values[selected, 0], y=values[selected, 1], z=values[selected, 2],
                           mode="markers", name=label,
                           marker=dict(size=3.5, color=color, symbol=symbol, opacity=.65))
    fig.add_trace(points(X, ~mask, "Observed", BLUE))
    fig.add_trace(points(X, mask, "Hidden true positions", GRAY, "circle-open"))
    for method in ("Mean", "KNN"):
        filled = versions[method]
        fig.add_trace(points(filled, mask, f"{method} imputed", METHOD_COLORS[method], "x"))
        coordinates = [[], [], []]
        for i in np.flatnonzero(mask)[:16]:
            for j in range(3):
                coordinates[j].extend([float(X[i, j]), float(filled[i, j]), None])
        fig.add_trace(go.Scatter3d(
            x=coordinates[0], y=coordinates[1], z=coordinates[2], mode="lines",
            name=f"{method} error segments (16 rows)", showlegend=False,
            line=dict(width=2, color=METHOD_COLORS[method]),
        ))
    buttons = [
        dict(label="Mean", method="update", args=[{"visible": [True, True, True, True, False, False]}]),
        dict(label="KNN", method="update", args=[{"visible": [True, True, False, False, True, True]}]),
        dict(label="Both", method="update", args=[{"visible": [True] * 6}]),
    ]
    fig.update_layout(
        title=dict(text="3D imputation | displacement from hidden truth", x=.5, xanchor="center", y=.98),
        height=800, template="plotly_white",
        scene=dict(xaxis_title="Age-like", yaxis_title="Income-like",
                   zaxis_title="Risk-like", aspectmode="cube"),
        updatemenus=[dict(buttons=buttons, x=0, y=1.12, active=2)],
        margin=dict(l=20, r=20, t=115, b=130), legend=dict(orientation="h", x=0, y=-.05),
        annotations=[dict(text="Same MNAR mask. Lines show 16 errors in raw coordinate units; visual length is not a standardized error metric.",
                          x=.5, y=-.12, xref="paper", yref="paper", showarrow=False)],
    )
    path = OUTPUT_DIR / "13_imputation_3d.html"
    fig.write_html(path, include_plotlyjs=True, full_html=True, auto_open=False)
    register(path)


def experiment_record(hypothesis, configuration, result, candidate, limitations):
    return dict(hypothesis=hypothesis, configuration=configuration, result=result,
                interpretation_candidate=candidate, limitations=limitations,
                review_status="Interpretation pending author review.")


def print_metrics(summary, heading):
    print("\n" + heading)
    print(f"{'Strategy':23} {'Missing':>8} {'Mean AUC':>10} {'SD':>9}")
    for row in summary:
        print(f"{row['strategy']:23} {row['rate']:8.0%} {row['mean']:10.4f} {row['std']:9.4f}")


def main():
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    X_complete, _ = core.generate_dataset()
    # Separate complete-preserving datasets are useful for inspection/debugging.
    X_mcar = core.inject_mechanism(X_complete, "MCAR")[0]
    X_mar = core.inject_mar(X_complete)[0]
    X_mnar = core.inject_mnar(X_complete)[0]
    assert all(values.shape == X_complete.shape for values in (X_mcar, X_mar, X_mnar))
    with threadpool_limits(limits=1):
        plot_missingness_mechanisms(X_complete)
        plot_missingness_probability()
        create_missingness_animation(X_complete)
        distribution = plot_imputation_distribution(X_complete)
        geometry = plot_imputation_geometry()
        correlation = plot_correlation_distortion()
        indicator = plot_missing_indicator_effect(core.run_indicator_experiment())
        leakage = plot_leakage_diagram()
        robustness = core.run_missingness_robustness_experiment()
        robustness_summary = plot_curves(robustness, "09_performance_vs_missingness.png")
        shift = core.run_missingness_shift_experiment()
        shift_summary = plot_curves(shift, "10_missingness_shift.png", shifted=True)
        dropout = core.run_feature_dropout()
        dropout_summary = plot_feature_dropout(dropout)
        interactive = create_interactive_3d(X_complete)
    configuration = {
        "n": 2000, "train_test": "stratified 70/30, split before all fitting",
        "seeds": core.SEEDS, "mask_seeds": "train seed+1000; test seed+2000; nested masks across rates",
        "data": "Entirely custom synthetic; generator coefficients are fixed in generate_dataset",
        "target": "Bernoulli with sigmoid(1.5*risk - .65*behavior + .35*age_z + .7*(log_income-3.4))",
        "learner": {"name": "HistGradientBoostingClassifier", "max_iter": 65, "max_leaf_nodes": 9,
                    "min_samples_leaf": 20, "learning_rate": .1, "l2_regularization": 1,
                    "early_stopping": False, "random_state": "repeat seed"},
        "knn": {"n_neighbors": 5, "scaling": "NaN-preserving observed training StandardScaler"},
        "other_parameters": "Library defaults; versions below", "numerical_threads": 1,
        "error_band": "sample standard deviation across five datasets/splits/masks, ddof=1; not a confidence interval",
    }
    limited = ["Synthetic generator and MCAR masks; no measured production behavior or universal method ranking.",
               "Five prespecified seeds are illustrative, not a population uncertainty estimate.",
               "ROC-AUC measures ranking; calibration, costs, latency and fallback safety are not evaluated.",
               "No tuning or test-based model selection; repeated conditions are diagnostic."]
    report = {
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__},
        "configuration": configuration,
        "mechanisms": {"MCAR": "p=.30", "MAR": "p=.05+.75*sigmoid((observed age-45)/5)",
                       "MNAR": "p=.05+.75*sigmoid((log(true income)-3.65)/.20); mask computed before hiding"},
        "distribution": experiment_record(
            "Selective hiding and deterministic filling may change feature distributions.",
            {"n": 2400, "seed": 42, "mask": "MNAR income", "filling": "descriptive same-sample; no predictive evaluation"},
            distribution, "Compare observed and filled values to synthetic complete truth; no distribution recovery is established.",
            ["Truth is known only in synthetic data.", "The MAR/MNAR mechanism is not inferred."]),
        "geometry_and_correlation": experiment_record(
            "Constant filling collapses points; local filling changes the joint structure differently.",
            {"n": 320, "seed": 43, "mask": "38% requested MCAR on X2", "knn_neighbors": 7},
            {"hidden_rmse": geometry, "correlation": correlation},
            "Use measured displacement and covariance to distinguish sample retention from faithful recovery.",
            ["Same-sample descriptive donors; not a held-out imputation benchmark."]),
        "indicator": experiment_record(
            "An indicator may help when a latent regime influences both missingness and target propensity.",
            {"n": 2400, "seed": 42, "split": "70/30 stratified", "mask_probabilities": [.05, .85],
             "target_logit": "1.3*context+2.3*latent_regime-1.1",
             "classifier": "identical LogisticRegression C=1 max_iter=500 after train-only median/scaling"},
            indicator, "Compare the computed AUC and conditional target prevalence; review fragility if the latent collection policy changes.",
            ["One engineered latent-regime scenario; not a universal indicator benefit."]),
        "leakage": experiment_record(
            "A shifted held-out distribution can change a pooled imputation statistic.",
            {"seed": 44, "train": "N(0,1), 200", "test": "N(4,1), 200", "mask": "30% MCAR per partition"},
            leakage, "A different median shows held-out information influencing preprocessing, without claiming an accuracy gain.",
            ["Intentionally leaky fitting exists only in this statistic-only diagram."]),
        "matched_missingness": experiment_record(
            "Prediction quality may change when training and evaluation lose more information.",
            {**configuration, "rates": core.RATES, "fit": "refit all workflows at every rate"},
            {"rows": robustness, "summary": robustness_summary},
            "Inspect actual curve directions and spread; compare strategies under the same learner.",
            limited),
        "missingness_shift": experiment_record(
            "Frozen models may degrade when test availability differs from training.",
            {**configuration, "train_rate": .10, "test_rates": core.SHIFT_RATES, "fit": "once per strategy and seed"},
            {"rows": shift, "summary": shift_summary},
            "Changes at fixed complete values isolate availability as an evaluation stress; clinical or business safety is not established.",
            limited),
        "feature_dropout": experiment_record(
            "Removing full feature columns may reduce prediction quality despite successful fallback.",
            {**configuration, "train_rate": 0, "strategy": "training median + histogram boosting", "fallback": "stored training median"},
            {"rows": dropout, "summary": dropout_summary},
            "Compare AUC changes to distinguish software input acceptance from reliable model behavior.",
            limited),
        "artifacts": [path.name for path in GENERATED],
        "optional_3d": interactive,
        "omitted": {"14_robustness_degradation.gif": "Duplicates the mechanism animation and quantitative robustness curves."},
    }
    path = OUTPUT_DIR / "visual_experiments.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    register(path)
    print_metrics(robustness_summary, "Refitted workflows: matched training and test missingness")
    print_metrics(shift_summary, "Frozen workflows: training missingness fixed at 10%")
    print("\nIndicator test ROC-AUC:", indicator["auc"])
    print("Dropout test means:", dropout_summary)
    print("\nGenerated visualizations and measured report:")
    for path in GENERATED:
        print(f"- {path}")
    print(f"Finished in {time.perf_counter() - start:.1f}s. Interpretations pending author review.")


if __name__ == "__main__":
    main()

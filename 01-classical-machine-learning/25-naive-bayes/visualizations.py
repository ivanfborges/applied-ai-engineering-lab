"""Generate bounded, offline visual explanations of Multinomial Naive Bayes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression

from example import TRAIN_LABELS, TRAIN_TEXTS
from visual_math import (
    accumulated_scores,
    bayes_update,
    class_indices,
    correlated_copy_posteriors,
    fit_count_model,
    fit_text_model,
    score_difference,
    smoothed_probabilities,
    token_contrasts,
)


BILLING = "#237a80"
ACCESS = "#c15c4a"
DARK = "#253746"
PALE_BILLING = "#d7eeeb"
PALE_ACCESS = "#f5ddd7"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs"


def _destination(output_dir, filename):
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save(fig, output_dir, filename):
    path = _destination(output_dir, filename)
    fig.savefig(path, dpi=135, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_bayes_update(output_dir=DEFAULT_OUTPUT):
    """How does one observed word update the class prior?"""
    vectorizer, model = fit_text_model()
    billing, access = class_indices(model)
    order = [billing, access]
    token = "invoice"
    column = vectorizer.vocabulary_[token]
    prior = np.exp(model.class_log_prior_[order])
    likelihood = np.exp(model.feature_log_prob_[order, column])
    unnormalized, posterior = bayes_update(prior, likelihood)

    stages = [
        ("1. Prior P(class)", prior),
        (f"2. Likelihood P({token} | class)", likelihood),
        ("3. Prior × likelihood", unnormalized),
        (f"4. Posterior P(class | {token})", posterior),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6))
    for ax, (title, values) in zip(axes.flat, stages):
        ax.barh(["billing", "access"], values, color=[BILLING, ACCESS])
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.27)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Token-draw probability" if title.startswith("2.") else "Joint probability" if title.startswith("3.") else "Probability")
        for i, value in enumerate(values):
            ax.text(value + max(values) * 0.02, i, f"{value:.3f}", va="center")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Bayes update for one token draw: invoice", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.01, "Values come from the fitted synthetic-text model. Each panel has its own horizontal scale.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    return _save(fig, output_dir, "01_bayes_update.png")


def plot_conditional_independence(output_dir=DEFAULT_OUTPUT):
    """How can words remain dependent even after conditioning on a class?"""
    rng = np.random.default_rng(25)
    words = ["invoice", "payment", "refund"]
    latent_subtype = rng.random(1200) < 0.5
    rates = np.where(latent_subtype[:, None], 0.8, 0.15)
    presence = rng.random((1200, 3)) < rates
    marginal = presence.mean(axis=0)
    observed = presence.astype(float).T @ presence.astype(float) / len(presence)
    product = np.outer(marginal, marginal)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    for ax, matrix, title in zip(
        axes,
        (observed, product),
        ("Observed P(both | billing)", "Naive product of marginals"),
    ):
        data = np.ma.array(matrix, mask=np.eye(3, dtype=bool))
        image = ax.imshow(data, vmin=0, vmax=max(observed.max(), product.max()), cmap="YlGnBu")
        ax.set_xticks(range(3), words)
        ax.set_yticks(range(3), words)
        ax.set_title(title, fontsize=11, fontweight="bold")
        for i in range(3):
            for j in range(3):
                if i != j:
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color=DARK)
        ax.set_xlabel("Second token present")
    axes[0].set_ylabel("First token present")
    fig.subplots_adjust(left=0.10, right=0.85, bottom=0.19, top=0.80, wspace=0.27)
    color_axis = fig.add_axes([0.88, 0.25, 0.02, 0.45])
    fig.colorbar(image, cax=color_axis, label="Joint occurrence probability")
    fig.suptitle("Conditional independence is an assumption, not a text fact", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.02, "Synthetic billing class with a hidden issue subtype; no causal direction is implied. Diagonal omitted.", ha="center", fontsize=9)
    return _save(fig, output_dir, "02_conditional_independence.png")


def plot_token_evidence(output_dir=DEFAULT_OUTPUT):
    """Which fitted words favor billing versus access?"""
    vectorizer, model = fit_text_model()
    contrasts = token_contrasts(vectorizer, model)
    chosen = ["invoice", "payment", "refund", "amount", "login", "password", "account", "blocked"]
    values = [contrasts[token] for token in chosen]
    colors = [BILLING if value > 0 else ACCESS for value in values]

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.barh(chosen, values, color=colors)
    ax.axvline(0, color=DARK, linewidth=1)
    ax.set_xlabel("log P(token | billing) − log P(token | access)")
    ax.set_title("Which tokens move the class score?", loc="left", fontweight="bold", fontsize=14)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.025, "\u2190 access evidence    |    billing evidence \u2192   Each token adds to log prior odds.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return _save(fig, output_dir, "03_token_evidence.png")


def animate_evidence_accumulation(output_dir=DEFAULT_OUTPUT):
    """How do cumulative log scores change as a document is read?"""
    vectorizer, model = fit_text_model()
    tokens = ["invoice", "payment", "amount", "incorrect"]
    scores = accumulated_scores(vectorizer, model, tokens)
    billing, access = class_indices(model)
    order = [billing, access]
    lowest = scores.min() - 1
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    footer = fig.text(0.5, 0.04, "", ha="center", fontsize=9)
    fig.subplots_adjust(bottom=0.21, top=0.86)

    def draw(frame):
        ax.clear()
        values = scores[frame, order]
        ax.barh(["billing", "access"], values, color=[BILLING, ACCESS])
        ax.invert_yaxis()
        ax.set_xlim(lowest, 0)
        ax.set_xlabel("Joint log score (higher is preferred)")
        ax.set_title("Prior only" if frame == 0 else "After: " + " ".join(tokens[:frame]), loc="left", fontweight="bold")
        for i, value in enumerate(values):
            ax.text(value - 0.08, i, f"{value:.2f}", ha="right", va="center", color=DARK)
        winner = "tie (library selects access)" if np.isclose(values[0], values[1]) else "billing" if values[0] > values[1] else "access"
        footer.set_text(f"Score = log prior + \u03a3 token log likelihoods   |   predicted: {winner}")
        ax.spines[["top", "right"]].set_visible(False)

    movie = animation.FuncAnimation(fig, draw, frames=len(tokens) + 1, interval=1100, repeat=True)
    path = _destination(output_dir, "04_log_evidence_accumulation.gif")
    movie.save(path, writer=animation.PillowWriter(fps=1))
    plt.close(fig)
    return path


def plot_log_underflow(output_dir=DEFAULT_OUTPUT):
    """At which computed token count does a raw float64 product become zero?"""
    terms = np.arange(1, 351)
    single_probability = 0.05
    product = np.cumprod(np.full(len(terms), single_probability, dtype=float))
    log_sum = np.cumsum(np.full(len(terms), np.log(single_probability)))
    zero_at = terms[np.flatnonzero(product == 0)[0]] if np.any(product == 0) else None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    positive = product > 0
    axes[0].plot(terms[positive], product[positive], color=ACCESS, linewidth=2)
    axes[0].axhline(np.finfo(float).tiny, color=DARK, linestyle="--", linewidth=1, label="Smallest normal float64")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Raw product, log axis")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].plot(terms, log_sum, color=BILLING, linewidth=2)
    axes[1].set_ylabel("Sum of log probabilities")
    for ax in axes:
        ax.set_xlabel("Number of factors, each equal to 0.05")
        ax.spines[["top", "right"]].set_visible(False)
    caption = f"On this runtime, the raw float64 product first equals zero at {zero_at} factors." if zero_at else "No zero was reached in the displayed range."
    fig.suptitle("Log space preserves the score after raw products underflow", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.01, caption, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    return _save(fig, output_dir, "04b_log_underflow.png")


def animate_laplace_smoothing(output_dir=DEFAULT_OUTPUT):
    """How does alpha repair a zero token probability?"""
    words = ["invoice", "payment", "refund", "login", "password"]
    counts = np.array([8, 5, 3, 0, 0])
    alphas = [0, 0.01, 0.1, 0.5, 1, 2, 5]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    fig.text(0.5, 0.09, "P(token j | class) = (count j + alpha) / (total + alpha \u00d7 V)", ha="center", fontsize=9)
    footer = fig.text(0.5, 0.04, "", ha="center", fontsize=9)
    fig.subplots_adjust(bottom=0.23, top=0.88)

    def draw(frame):
        ax.clear()
        alpha = alphas[frame]
        probabilities = smoothed_probabilities(counts, alpha)
        ax.bar(words, probabilities, color=[BILLING] * 3 + [ACCESS] * 2)
        ax.set_ylim(0, 0.6)
        ax.set_ylabel("P(token | billing)")
        ax.set_title(f"Additive smoothing: alpha = {alpha:g}", loc="left", fontweight="bold")
        footer.set_text(f"P(login | billing) = {probabilities[3]:.4f}; invoice \u00d7 login = {probabilities[0] * probabilities[3]:.4f}")
        ax.spines[["top", "right"]].set_visible(False)

    movie = animation.FuncAnimation(fig, draw, frames=len(alphas), interval=1000, repeat=True)
    path = _destination(output_dir, "05_laplace_smoothing.gif")
    movie.save(path, writer=animation.PillowWriter(fps=1))
    plt.close(fig)
    return path


def _count_grid():
    x1, x2 = np.meshgrid(np.linspace(0, 8, 121), np.linspace(0, 8, 121))
    points = np.column_stack((x1.ravel(), x2.ravel()))
    return x1, x2, points


def plot_decision_boundary(output_dir=DEFAULT_OUTPUT):
    """Why is the Multinomial NB decision rule linear in count space?"""
    X, y, model = fit_count_model()
    x1, x2, points = _count_grid()
    difference = score_difference(model, points).reshape(x1.shape)
    fig, ax = plt.subplots(figsize=(7.2, 6))
    ax.contourf(x1, x2, difference, levels=[-100, 0, 100], colors=[PALE_ACCESS, PALE_BILLING], alpha=0.8)
    ax.contour(x1, x2, difference, levels=[0], colors=[DARK], linewidths=2)
    for label, color in (("billing", BILLING), ("access", ACCESS)):
        rows = X[y == label]
        ax.scatter(rows[:, 0], rows[:, 1], s=55, color=color, edgecolor="white", alpha=0.8, label=label)
    billing, access = class_indices(model)
    weights = model.feature_log_prob_[billing] - model.feature_log_prob_[access]
    intercept = model.class_log_prior_[billing] - model.class_log_prior_[access]
    ax.set(xlim=(0, 8), ylim=(0, 8), xlabel="Billing-related token count", ylabel="Access-related token count")
    ax.set_title("Multinomial NB has a straight log-score boundary", loc="left", fontweight="bold", fontsize=13)
    ax.text(0.02, 0.98, f"Δscore = {intercept:+.2f} {weights[0]:+.2f}x₁ {weights[1]:+.2f}x₂", transform=ax.transAxes, va="top", fontsize=10, bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"))
    ax.legend(loc="upper right")
    fig.text(0.5, 0.02, "Integer points are observed counts; the shaded plane extends the score rule continuously.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, output_dir, "06_decision_boundary.png")


def plot_posterior_surface_3d(output_dir=DEFAULT_OUTPUT):
    """Can rotation clarify the score-difference plane and its zero boundary?"""
    _, _, model = fit_count_model()
    x1, x2, points = _count_grid()
    z = score_difference(model, points).reshape(x1.shape)
    fig = go.Figure()
    fig.add_trace(go.Surface(x=x1, y=x2, z=z, colorscale=[[0, ACCESS], [0.5, "#f7f7f7"], [1, BILLING]], cmin=-max(abs(z.min()), abs(z.max())), cmax=max(abs(z.min()), abs(z.max())), colorbar=dict(title="Δscore"), name="Class score difference"))
    fig.add_trace(go.Surface(x=x1, y=x2, z=np.zeros_like(z), opacity=0.32, colorscale=[[0, "#6b747c"], [1, "#6b747c"]], showscale=False, name="Decision level: zero"))
    billing, access = class_indices(model)
    weights = model.feature_log_prob_[billing] - model.feature_log_prob_[access]
    intercept = model.class_log_prior_[billing] - model.class_log_prior_[access]
    boundary_y = np.linspace(0, 8, 100)
    boundary_x = -(intercept + weights[1] * boundary_y) / weights[0]
    visible = (boundary_x >= 0) & (boundary_x <= 8)
    fig.add_trace(go.Scatter3d(
        x=boundary_x[visible], y=boundary_y[visible], z=np.zeros(visible.sum()),
        mode="lines", line=dict(color=DARK, width=8), name="Decision boundary: score difference = 0",
    ))
    fig.update_layout(
        title="Rotate the Multinomial NB score plane; the zero plane is the decision boundary",
        scene=dict(
            xaxis_title="Billing token count",
            yaxis_title="Access token count",
            zaxis_title="score(billing) − score(access)",
            aspectratio=dict(x=1, y=1, z=0.7),
        ),
        annotations=[dict(text="Above zero: billing  |  Below zero: access. Continuous surface extends an integer-count rule.", x=0.5, y=0, xref="paper", yref="paper", showarrow=False)],
        margin=dict(l=0, r=0, t=70, b=55),
    )
    path = _destination(output_dir, "09_posterior_surface.html")
    fig.write_html(path, include_plotlyjs=True, full_html=True)
    return path


def animate_prior_shift(output_dir=DEFAULT_OUTPUT):
    """How does changing only the prior move the boundary?"""
    _, _, model = fit_count_model()
    x1, x2, points = _count_grid()
    priors = np.linspace(0.1, 0.9, 9)
    fig, ax = plt.subplots(figsize=(7, 5.6))

    def draw(frame):
        ax.clear()
        prior = priors[frame]
        difference = score_difference(model, points, prior_billing=prior).reshape(x1.shape)
        ax.contourf(x1, x2, difference, levels=[-100, 0, 100], colors=[PALE_ACCESS, PALE_BILLING])
        if difference.min() < 0 < difference.max():
            ax.contour(x1, x2, difference, levels=[0], colors=[DARK], linewidths=2)
        ax.set(xlim=(0, 8), ylim=(0, 8), xlabel="Billing-related token count", ylabel="Access-related token count")
        ax.set_title(f"Prior shift: P(billing) = {prior:.1f}", loc="left", fontweight="bold")
        ax.text(0.02, 0.97, "Fixed token likelihoods; only log prior odds change.", transform=ax.transAxes, va="top", fontsize=9, bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"))
        ax.text(0.02, 0.03, "Teal: billing prediction   |   coral: access prediction", transform=ax.transAxes, fontsize=9, bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"))

    movie = animation.FuncAnimation(fig, draw, frames=len(priors), interval=900, repeat=True)
    path = _destination(output_dir, "07_prior_shift.gif")
    movie.save(path, writer=animation.PillowWriter(fps=1))
    plt.close(fig)
    return path


def plot_correlated_evidence(output_dir=DEFAULT_OUTPUT):
    """What if several observed features are copies of one latent event?"""
    copies, exact, naive = correlated_copy_posteriors(copies=4)
    fig, ax = plt.subplots(figsize=(8, 4.7))
    ax.plot(copies, naive, marker="o", color=ACCESS, linewidth=2, label="Naive multiplication of copies")
    ax.plot(copies, exact, marker="s", color=BILLING, linewidth=2, label="Correct posterior for one latent event")
    ax.set(xlim=(0.8, 4.2), ylim=(0.65, 1.01), xticks=copies, xlabel="Number of perfectly copied positive signals", ylabel="P(billing | observed signals)")
    ax.set_title("Copied evidence can make a posterior too extreme", loc="left", fontweight="bold", fontsize=13)
    ax.text(0.02, 0.05, "Toy model: prior 0.5; P(signal | billing)=0.75; P(signal | access)=0.25.\nFour features encode the same latent event, not four independent events.", transform=ax.transAxes, fontsize=9, bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"))
    ax.legend(loc="center right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, output_dir, "10_correlated_evidence.png")


def compare_nb_logistic(output_dir=DEFAULT_OUTPUT):
    """How do generative token contrasts differ from discriminative coefficients?"""
    vectorizer, nb = fit_text_model()
    X = vectorizer.transform(TRAIN_TEXTS)
    logistic = LogisticRegression(max_iter=1000, random_state=25).fit(X, TRAIN_LABELS)
    words = ["invoice", "payment", "refund", "amount", "login", "password", "account", "reset"]
    indices = [vectorizer.vocabulary_[word] for word in words]
    contrast = token_contrasts(vectorizer, nb)
    nb_values = np.array([contrast[word] for word in words])
    lr_values = logistic.coef_[0, indices]
    if logistic.classes_[1] != "billing":
        lr_values = -lr_values

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, values, title, xlabel in (
        (axes[0], nb_values, "Multinomial NB", "Log token-likelihood contrast"),
        (axes[1], lr_values, "Logistic Regression", "Coefficient for billing class"),
    ):
        ax.barh(words, values, color=[BILLING if value > 0 else ACCESS for value in values])
        ax.axvline(0, color=DARK, linewidth=1)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle("Same token counts, different fitted quantities", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.01, "Positive favors billing. NB contrasts and logistic coefficients have different definitions and scales.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    return _save(fig, output_dir, "08_nb_vs_logistic.png")


VIEWS = {
    "bayes-update": plot_bayes_update,
    "conditional-independence": plot_conditional_independence,
    "token-evidence": plot_token_evidence,
    "evidence-accumulation": animate_evidence_accumulation,
    "log-underflow": plot_log_underflow,
    "smoothing": animate_laplace_smoothing,
    "decision-boundary": plot_decision_boundary,
    "posterior-3d": plot_posterior_surface_3d,
    "prior-shift": animate_prior_shift,
    "correlated-evidence": plot_correlated_evidence,
    "nb-vs-logistic": compare_nb_logistic,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true", help="Generate every view (default).")
    selection.add_argument("--view", choices=VIEWS, help="Generate one named view.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Artifact directory.")
    args = parser.parse_args()
    selected = [args.view] if args.view else VIEWS
    for name in selected:
        print(f"{name}: {VIEWS[name](args.output_dir)}", flush=True)


if __name__ == "__main__":
    main()

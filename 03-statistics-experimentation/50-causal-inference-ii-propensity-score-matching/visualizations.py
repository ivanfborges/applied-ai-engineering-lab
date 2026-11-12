"""Consistent Matplotlib chapters, Pillow animations and an independent viewer."""

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.widgets import Button, CheckButtons, Slider
import numpy as np

from from_scratch import matching_weights, standardized_mean_difference
from visual_core import Design, analyze_teams, assignment, diagnostics, simulate_teams

CONTROL, TREATED, TRUTH, ACCENT = "#247BA0", "#D66032", "#25785A", "#7554A3"
COLORS = (CONTROL, TREATED)
METHOD_COLORS = {"Naive": "#657384", "Matching ATT": ACCENT, "IPW ATE": TRUTH}
NAMES = ("Control: no AI", "Treated: AI")


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
        "axes.labelsize": 10, "legend.fontsize": 8, "figure.facecolor": "#F8FAFC",
        "axes.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "grid.alpha": .18, "savefig.facecolor": "#F8FAFC",
    })


def frame(fig, title, caption):
    fig.clear()
    fig.suptitle(title, fontsize=16, fontweight="bold", x=.055, ha="left", y=.97)
    fig.subplots_adjust(left=.075, right=.96, bottom=.20, top=.85, wspace=.35, hspace=.55)
    fig.text(.055, .065, textwrap.fill(caption, 155), va="center", fontsize=9,
             color="#334155", linespacing=1.6)
    fig.text(.055, .014, "SYNTHETIC SUPPORT TEAMS  |  negative effect = minutes saved  |  author interpretation review required",
             fontsize=8, color="#64748B")


def scatter_groups(ax, data, weights=None, alpha=.6):
    t, x = data["t"], data["x"]
    for arm in (0, 1):
        selected = t == arm
        size = 15 if weights is None else 15*weights[selected]
        ax.scatter(*x[selected].T, c=COLORS[arm], s=size, alpha=alpha,
                   marker="o" if arm == 0 else "^", label=NAMES[arm], linewidths=.3)
    ax.set(xlim=(.5, 9.5), ylim=(8, 52), xlabel="Baseline complexity (index)",
           ylabel="Baseline workload (open tickets)")
    ax.grid(True)


def outcome_bars(ax, result):
    data, summaries = result["data"], result["summaries"]
    means = [data["y"][data["t"] == arm].mean() for arm in (0, 1)]
    ax.bar([0, 1], means, color=COLORS, width=.55)
    for i, mean in enumerate(means):
        ax.text(i, mean+1.5, f"{mean:.1f} min", ha="center", fontsize=9)
    ax.set(xticks=[0, 1], xticklabels=["Control", "AI"], ylim=(0, 95),
           ylabel="Observed mean resolution time (min)", title="Observed outcomes mix selection and effect")
    ax.text(.02, .95, f"Naive treated − control = {summaries['Naive']['estimate']:+.2f} min",
            transform=ax.transAxes, va="top", fontsize=9)


def draw_confounding(fig, result):
    a = result["data"]["design"].separation
    frame(fig, f"01  Selection changes the comparison  |  assignment strength = {a:.2f}",
          "The assignment coefficients change while baseline units, assignment uniforms and potential outcomes stay fixed. "
          "Harder cases adopt AI more often. The observed difference includes baseline differences; truth is an intervention contrast.")
    ax = fig.add_subplot(221)
    scatter_groups(ax, result["data"])
    ax.set_title("Same baseline units; treatment labels change")
    ax.legend(loc="upper left")
    ax = fig.add_subplot(222)
    for arm in (0, 1):
        means = result["data"]["z"][result["data"]["t"] == arm].mean(axis=0)
        ax.bar(np.arange(2)+(arm-.5)*.32, means, width=.32, color=COLORS[arm], label=NAMES[arm])
    ax.axhline(0, color="gray", lw=.7)
    ax.set(xticks=[0, 1], xticklabels=["Complexity", "Workload"], ylim=(-1.6, 1.6),
           ylabel="Baseline mean (centered units)", title="Groups differ before treatment")
    ax.legend()
    outcome_bars(fig.add_subplot(223), result)
    ax = fig.add_subplot(224)
    values = [result["sample_ate"], result["summaries"]["Naive"]["estimate"]]
    ax.barh([0, 1], values, color=[TRUTH, "#657384"], height=.5)
    ax.axvline(0, color="black", lw=.8)
    ax.set(yticks=[0, 1], yticklabels=["Known true ATE", "Naive association"],
           xlim=(-16, 25), xlabel="Treated − control (minutes)", title="Association can reverse the effect's sign")
    for y, v in enumerate(values):
        ax.text(v+.5, y, f"{v:+.2f}", va="center", fontsize=9)


def surface_axes(ax, result, angle=-55):
    grid = np.linspace(-2, 2, 32)
    g0, g1 = np.meshgrid(grid, grid)
    z = np.column_stack([g0.ravel(), g1.ravel()])
    estimated = result["model"].predict_proba(z)[:, 1].reshape(g0.shape)
    true = assignment(z, result["data"]["design"]).reshape(g0.shape)
    x0, x1 = 5+2*g0, 30+10*g1
    ax.plot_surface(x0, x1, estimated, cmap="viridis", alpha=.56, vmin=0, vmax=1,
                    linewidth=0, rstride=2, cstride=2)
    ax.plot_wireframe(x0, x1, true, color="#475569", alpha=.4, rstride=6, cstride=6, linewidth=.6)
    data = result["data"]
    for arm in (0, 1):
        selected = np.flatnonzero(data["t"] == arm)[::max(1, len(data["t"])//200)]
        ax.scatter(data["x"][selected, 0], data["x"][selected, 1], result["scores"][selected],
                   color=COLORS[arm], s=10, marker="o" if arm == 0 else "^", label=NAMES[arm])
    ax.set(xlabel="Complexity", ylabel="Workload", zlabel="P(AI | baseline)", zlim=(0, 1),
           title="Estimated surface + true wireframe")
    ax.view_init(elev=26, azim=angle)
    ax.legend(loc="upper left", fontsize=8)
    return x0, x1, estimated, true


def draw_surface(fig, result, angle=-55):
    frame(fig, "02  A probability of assignment, not a treatment effect",
          "Height is estimated P(AI | baseline); point color is actual adoption. The gray wireframe is the known assignment function. "
          "Purple stars have different covariates but exactly the same fitted score. Rotation reveals probability geometry, not outcome geometry.")
    x0, x1, estimated, true = surface_axes(fig.add_subplot(121, projection="3d"), result, angle)
    ax = fig.add_subplot(122)
    filled = ax.contourf(x0, x1, estimated, levels=np.linspace(0, 1, 11), cmap="viridis", vmin=0, vmax=1)
    lines = ax.contour(x0, x1, true, levels=[.2, .5, .8], colors="white", linestyles="--", linewidths=1)
    ax.clabel(lines, fmt="true %.1f", fontsize=8)
    beta = result["model"].coef_[0]
    if abs(beta[1]) > .15:
        extent = min(1.0, 1.5*abs(beta[1]/beta[0])) if abs(beta[0]) > .01 else 1.0
        z0 = np.array([-extent, extent])
        z1 = -beta[0]/beta[1]*z0
        ax.plot(5+2*z0, 30+10*z1, "*-", color="#DF9FFA", ms=13, lw=2, label="Different X; same fitted e(X)")
        probability = result["model"].predict_proba([[z0[0], z1[0]]])[0, 1]
        ax.text(.04, .95, f"Star pair: fitted e(X) = {probability:.3f}", transform=ax.transAxes,
                va="top", color="white", bbox=dict(facecolor="#334155", alpha=.8, edgecolor="none"))
        ax.legend(loc="lower right")
    ax.set(xlabel="Baseline complexity", ylabel="Baseline workload", title="2D contour projection; dashed = true")
    fig.colorbar(filled, ax=ax, shrink=.8, label="Estimated propensity")


def overlap_hist(ax, result, annotate=True):
    for arm in (0, 1):
        values = result["scores"][result["data"]["t"] == arm]
        ax.hist(values, bins=np.linspace(0, 1, 17), density=True, histtype="stepfilled",
                alpha=.35, color=COLORS[arm], label=NAMES[arm])
    lower, upper = result["overlap"]["range"]
    if lower < upper:
        ax.axvspan(lower, upper, color=TRUTH, alpha=.08, label="Empirical common range")
    ax.axvspan(0, lower, color="#B91C1C", alpha=.08)
    ax.axvspan(upper, 1, color="#B91C1C", alpha=.08)
    ax.set(xlim=(0, 1), xlabel="Estimated propensity score", ylabel="Density")
    if annotate:
        ax.text(.5, .97, f"Histogram intersection = {result['overlap']['intersection']:.2f}\n"
                f"Outside common range = {result['overlap']['outside_range']:.1%}",
                transform=ax.transAxes, ha="center", va="top", fontsize=9,
                bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))


def draw_overlap(fig, result, structural_data):
    a = result["data"]["design"].separation
    label = "Strong" if a < .7 else "Moderate" if a < 2 else "Poor"
    frame(fig, f"03  {label} practical overlap  |  assignment strength = {a:.2f}",
          "The left panel changes assignment strength with the same exogenous tape. Common ranges can contain sparse regions; histogram intersection is bin-dependent. "
          "The separate right panel forces AI for complexity > 7.5: those units have no observed no-AI counterpart by design.")
    ax = fig.add_subplot(121)
    overlap_hist(ax, result)
    ax.set_title("Practical overlap; theoretical probabilities stay in (0, 1)")
    ax.legend(loc="center", fontsize=8)
    ax = fig.add_subplot(122)
    for arm in (0, 1):
        take = structural_data["t"] == arm
        ax.scatter(structural_data["x"][take, 0], structural_data["true_scores"][take],
                   s=12, color=COLORS[arm], alpha=.55, label=NAMES[arm])
    ax.axvspan(7.5, 9.5, color="#B91C1C", alpha=.12)
    ax.text(.04, .08, "TRUE e(X) = 1 in shaded region\nFull-population ATE not identified\nNo adjusted estimate computed here",
            transform=ax.transAxes, fontsize=10, color="#991B1B",
            bbox=dict(facecolor="white", edgecolor="none", alpha=.9))
    ax.set(xlabel="Baseline complexity", ylabel="True propensity", ylim=(-.04, 1.07),
           title="Separate structural nonpositivity demonstration")


def draw_matching(fig, result, processed=None, caliper=.2):
    data, full, matches = result["data"], result["full_matches"], result["matches"]
    total = len(full.treated)
    processed = total if processed is None else min(processed, total)
    frame(fig, f"04  Matching on fitted logit scores  |  processed {processed}/{total}",
          f"Every treated row in this small illustration is processed once; controls can be reused. Caliper = {caliper:g} absolute logit units, inclusive. "
          "Lines connect score neighbors, which can be far apart in covariate space. Rejected treated units change the effective target population.")
    ax = fig.add_subplot(221)
    scatter_groups(ax, data)
    ax.set_title(f"Before matching: {len(data['t'])} synthetic units")
    ax.legend(loc="upper left")
    ax = fig.add_subplot(222)
    scatter_groups(ax, data, alpha=.35)
    for k in range(processed):
        i, j, distance = full.treated[k], full.control[k], full.distance[k]
        good = distance <= caliper
        ax.plot(data["x"][[i, j], 0], data["x"][[i, j], 1],
                color=ACCENT if good else "#B91C1C", alpha=.5, lw=1,
                ls="-" if good else ":")
        if not good:
            ax.scatter(*data["x"][i], marker="x", color="#B91C1C", s=65)
    if processed < total:
        i, j = full.treated[processed], full.control[processed]
        ax.scatter(*data["x"][[i, j]].T, s=130, facecolors="none", edgecolors="black", linewidths=1.5)
        ax.text(.01, .99, f"Active T row {i}: e={result['scores'][i]:.3f}\n"
                f"Nearest C row {j}: e={result['scores'][j]:.3f}\n"
                f"Logit distance={full.distance[processed]:.3f}",
                transform=ax.transAxes, va="top", fontsize=8, bbox=dict(facecolor="white", alpha=.9))
    ax.set_title("All controls are candidates; ring = current pair")
    ax = fig.add_subplot(223)
    if "Caliper ATT" in result["weights"]:
        scatter_groups(ax, data, weights=result["weights"]["Caliper ATT"], alpha=.6)
    else:
        ax.text(.5, .5, "No matches retained", transform=ax.transAxes, ha="center")
    ax.set_title(f"Final sample: {len(matches.treated)} retained, {len(matches.unmatched)} rejected")
    ax = fig.add_subplot(224)
    ax.hist(full.distance, bins=np.linspace(0, max(.4, full.distance.max()+.05), 14), color=ACCENT, alpha=.75)
    ax.axvline(caliper, color="#B91C1C", ls="--", label=f"Caliper = {caliper:g}")
    ax.set(xlabel="Absolute logit-score distance", ylabel="Treated units", title="Actual score distances; no geometric shortcut")
    if matches.treated.size:
        counts = np.bincount(matches.control, minlength=len(data["t"]))
        ax.text(.98, .9, f"Distinct controls: {np.unique(matches.control).size}\nMax reuse: {counts.max()}\n"
                f"Retained true effect: {result['retained_truth']:+.2f} min", ha="right", va="top", transform=ax.transAxes, fontsize=8)
    ax.legend()


def covariate_hist(ax, result, feature, weights, title, original=False):
    data = result["data"]
    edges = np.linspace(1, 9, 17) if feature == 0 else np.linspace(10, 50, 17)
    for arm in (0, 1):
        take = data["t"] == arm
        if original:
            ax.hist(data["x"][take, feature], bins=edges, density=True, histtype="step",
                    color=COLORS[arm], ls="--", alpha=.5)
        ax.hist(data["x"][take, feature], weights=weights[take], bins=edges,
                density=True, histtype="step", lw=1.8, color=COLORS[arm], label=NAMES[arm])
    ax.set(xlabel="Baseline complexity" if feature == 0 else "Baseline workload",
           ylabel="Within-arm density", title=title)


def draw_ipw(fig, result, fraction=1.0, estimand="ATE"):
    w = result["weights"][f"IPW {estimand}"]
    interpolated = (1-fraction)+fraction*w
    summary = diagnostics(result["data"], interpolated)
    formula = "T/e + (1-T)/(1-e)" if estimand == "ATE" else "T + (1-T)e/(1-e)"
    frame(fig, f"05  {estimand} weighting changes contributions  |  transition = {fraction:.0%}",
          f"Final weights: w = {formula}. Marker AREA is proportional to the displayed weight, with no resampling or coordinate movement. "
          "Intermediate frames interpolate contributions for teaching; only the final frame uses the stated IPW estimator. ATT targets the treated population.")
    ax = fig.add_subplot(221)
    scatter_groups(ax, result["data"], alpha=.5)
    ax.set_title("Recorded covariates and treatment stay fixed")
    ax.legend(loc="upper left")
    ax = fig.add_subplot(222)
    scatter_groups(ax, result["data"], interpolated, alpha=.4)
    for arm in (0, 1):
        selected = np.flatnonzero(result["data"]["t"] == arm)
        index = selected[np.argmax(w[selected])]
        point = result["data"]["x"][index]
        ax.annotate(f"T={arm}; e={result['scores'][index]:.3f}\nfinal w={w[index]:.2f}", point,
                    xytext=(5, -22 if arm else 10), textcoords="offset points", fontsize=7,
                    bbox=dict(facecolor="white", alpha=.9, edgecolor=COLORS[arm]))
    ax.set_title(f"Weighted contrast now: {summary['estimate']:+.2f} min")
    covariate_hist(fig.add_subplot(223), result, 0, np.ones_like(w), "Before weighting")
    ax = fig.add_subplot(224)
    covariate_hist(ax, result, 0, interpolated, "Contributions after displayed weighting")
    ax.legend()


def draw_balance(fig, result):
    frame(fig, "06  Validate the comparison with measured balance",
          "All SMDs use the original unweighted pooled sample SD, held fixed across methods. The 0.1 line is a diagnostic heuristic, not an identification test. "
          "Right panels compare full marginal distributions: dashed = original, solid = adjusted. Hidden common causes cannot be checked here.")
    ax = fig.add_subplot(121)
    for offset, name in enumerate(("Naive", "Matching ATT", "IPW ATE")):
        values = np.abs(result["summaries"][name]["smd"])
        ax.plot(values, np.arange(2)+(offset-1)*.09, "o-", label=name, color=METHOD_COLORS[name], ms=7)
    ax.axvline(.1, color="#B91C1C", ls="--", label="0.1 heuristic")
    ax.set(yticks=[0, 1], yticklabels=["Complexity", "Workload"], ylim=(-.5, 1.5),
           xlabel="Absolute standardized mean difference", title="Love plot; fixed reference SD")
    ax.legend(loc="upper right")
    covariate_hist(fig.add_subplot(222), result, 0, result["weights"]["Matching ATT"], "Complexity: matching vs original", original=True)
    ax = fig.add_subplot(224)
    covariate_hist(ax, result, 1, result["weights"]["IPW ATE"], "Workload: IPW ATE vs original", original=True)
    ax.legend()


def draw_extremes(fig, sweep, index):
    result = sweep[index]
    summary = result["summaries"]["IPW ATE"]
    counts = np.bincount(result["data"]["t"])
    warning = "WEAK EFFECTIVE INFORMATION" if np.min(summary["ess"]/counts) < .25 else "Inspect balance and support"
    frame(fig, f"07  Extreme weights and effective information  |  separation = {result['data']['design'].separation:.2f}",
          f"{warning}. ESS = (sum w)^2 / sum(w^2), calculated separately by arm. A few heavily weighted observations can dominate. "
          "The path is one fixed random tape, not a universal bias curve; error need not increase monotonically with separation.")
    ax = fig.add_subplot(221)
    overlap_hist(ax, result, annotate=False)
    ax.set_title("Assignment becomes more predictable")
    ax.legend()
    ax = fig.add_subplot(222)
    largest = max(r["summaries"]["IPW ATE"]["max_weight"] for r in sweep)
    edges = np.geomspace(1, max(3, largest*1.01), 20)
    for arm in (0, 1):
        take = result["data"]["t"] == arm
        ax.hist(result["weights"]["IPW ATE"][take], bins=edges, histtype="step", color=COLORS[arm], lw=2)
    ax.set(xscale="log", xlabel="Actual IPW ATE weight (log axis)", ylabel="Observations",
           title=f"Largest weight = {summary['max_weight']:.1f}; no truncation")
    ax = fig.add_subplot(223)
    ax.bar([0, 1], counts, color=COLORS, alpha=.2, label="Raw arm count")
    ax.bar([0, 1], summary["ess"], color=COLORS, width=.45, label="Arm ESS")
    ax.set(xticks=[0, 1], xticklabels=["Control", "Treated"], ylim=(0, len(result["data"]["t"])*.7),
           ylabel="Count / effective sample size", title=f"ESS C/T = {summary['ess'][0]:.1f} / {summary['ess'][1]:.1f}")
    ax.legend()
    ax = fig.add_subplot(224)
    x = [r["data"]["design"].separation for r in sweep]
    y = [r["summaries"]["IPW ATE"]["estimate"] for r in sweep]
    ax.plot(x[:index+1], y[:index+1], "o-", color=TRUTH, label="IPW ATE")
    ax.axhline(result["sample_ate"], color="black", ls="--", label="Known true ATE")
    ax.set(xlim=(0, 4), ylim=(min(min(y), -8)-3, max(max(y), -8)+3),
           xlabel="Treatment-assignment separation", ylabel="Effect estimate (min)",
           title="One sample path; no monotonic bias guarantee")
    ax.legend()


def draw_estimators(fig, rows):
    frame(fig, "08  Same algorithm family, different causal questions",
          "Each scenario uses independent generated datasets. Matching targets population ATT; IPW targets population ATE. Naive error is benchmarked to ATE as an association diagnostic. "
          "Reference truths integrate the known generator; error distributions include sampling variation. No observational confidence intervals are shown.")
    scenarios = ("Constant effect", "Heterogeneous effect")
    for i, scenario in enumerate(scenarios):
        selected = [row for row in rows if row["scenario"] == scenario]
        truth_ate = next(r["target"] for r in selected if r["method"] == "IPW ATE")
        truth_att = next(r["target"] for r in selected if r["method"] == "Matching ATT")
        for j, field in enumerate(("estimate", "error")):
            ax = fig.add_subplot(2, 2, i*2+j+1)
            values = [row[field] for row in selected]
            edges = np.linspace(min(values)-.3, max(values)+.3, 17)
            for method, color in METHOD_COLORS.items():
                series = [row[field] for row in selected if row["method"] == method]
                ax.hist(series, bins=edges, histtype="step", lw=2, color=color, label=method)
            if field == "estimate":
                ax.axvline(truth_ate, color=TRUTH, ls="--", label=f"True ATE {truth_ate:.2f}")
                ax.axvline(truth_att, color=ACCENT, ls=":" , lw=2, label=f"True ATT {truth_att:.2f}")
            else:
                ax.axvline(0, color="black", ls="--", label="Zero target-specific error")
            ax.set(title=f"{scenario}; {len(selected)//3} independent datasets",
                   xlabel="Estimate (minutes)" if field == "estimate" else "Estimate − appropriate population target (min)",
                   ylabel="Simulations")
            ax.legend(fontsize=7, ncol=2)


def draw_hidden(fig, result):
    frame(fig, "09  Observed balance does not establish exchangeability",
          "U influences adoption and potential outcomes but is withheld from the propensity fit. U is visible here only because this is a simulation. "
          "Real observational data cannot validate no unmeasured confounding by inspecting measured covariate balance. This illustration is separate from the main generator.")
    fig.subplots_adjust(left=.19, wspace=.5)
    ax = fig.add_subplot(121)
    data = result["data"]
    x_with_oracle = np.column_stack([data["x"], data["hidden_u"]])
    for method in ("Naive", "Matching ATT", "IPW ATE"):
        smd = np.abs(standardized_mean_difference(x_with_oracle, data["t"], result["weights"][method]))
        ax.plot(smd, [0, 1, 2], "o-", color=METHOD_COLORS[method], label=method)
    ax.axvline(.1, color="#B91C1C", ls="--")
    ax.set(yticks=[0, 1, 2], yticklabels=["Observed complexity", "Observed workload", "Hidden U (oracle only)"],
           xlabel="Absolute SMD (fixed original SD)", title="Observed rows and oracle-only U")
    ax.legend()
    ax = fig.add_subplot(122)
    methods = list(METHOD_COLORS)
    estimates = [result["summaries"][method]["estimate"] for method in methods]
    ax.barh(methods, estimates, color=list(METHOD_COLORS.values()))
    ax.axvline(result["sample_ate"], color="black", ls="--", label="Known true ATE = ATT")
    ax.axvline(0, color="gray", lw=.7)
    ax.set(xlabel="Effect estimate (minutes)", title="Residual confounding persists")
    ax.legend()


def save_chapter(output, name, draw, frames, dpi=110, fps=5):
    """Render one figure at a time; Pillow buffers only this animation's frames."""
    output = Path(output)
    (output/"figures").mkdir(parents=True, exist_ok=True)
    (output/"animations").mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12.8, 8))
    files = []
    try:
        draw(fig, frames[-1])
        png = output/"figures"/f"{name}.png"
        fig.savefig(png, dpi=max(140, dpi))
        files.append(png)
        if len(frames) > 1:
            animation = FuncAnimation(fig, lambda index: draw(fig, frames[index]),
                                      frames=len(frames), interval=1000/fps,
                                      repeat=True, blit=False, cache_frame_data=False)
            gif = output/"animations"/f"{name}.gif"
            animation.save(gif, writer=PillowWriter(fps=fps), dpi=dpi)
            files.append(gif)
    finally:
        plt.close(fig)
    return files


def make_dashboard(n=600, seed=50):
    """Native sliders and mouse-rotatable 3D surface; no export/backend side effects."""
    style()
    fig = plt.figure(figsize=(14, 9))
    fig.subplots_adjust(top=.88, bottom=.32, left=.07, right=.95, wspace=.35, hspace=.45)
    fig.suptitle("Causal comparison workbench  |  synthetic support teams", fontsize=16, fontweight="bold")
    axes = [fig.add_subplot(221, projection="3d"), fig.add_subplot(222),
            fig.add_subplot(223), fig.add_subplot(224)]
    controls = {}
    settings = [("Separation", 0, 4, 1, .01), ("Caliper", 0, 1.5, .2, .01),
                ("Effect (min)", -14, 0, -8, .1), ("Seed", 0, 999, seed, 1)]
    # Keep seeds outside the slider range usable through the CLI by widening the control.
    for i, (label, low, high, initial, step) in enumerate(settings):
        ax = fig.add_axes([.17, .245-i*.047, .48, .022])
        controls[label] = Slider(ax, label, low, max(high, seed) if label == "Seed" else high,
                                 valinit=initial, valstep=step)
    toggle = CheckButtons(fig.add_axes([.73, .14, .22, .09]), ["Heterogeneous effect"], [False])
    reset = Button(fig.add_axes([.77, .085, .13, .04]), "Reset controls")
    status = fig.text(.07, .015, "", fontsize=9, color="#334155")
    state = {"result": None, "updates": 0}

    def update(_=None):
        design = Design(float(controls["Separation"].val), float(controls["Effect (min)"].val),
                        3.0 if toggle.get_status()[0] else 0.0)
        data = simulate_teams(n, int(controls["Seed"].val), design)
        result = analyze_teams(data, float(controls["Caliper"].val))
        view = (axes[0].elev, axes[0].azim)
        for ax in axes:
            ax.clear()
        surface_axes(axes[0], result, view[1])
        axes[0].view_init(elev=view[0], azim=view[1])
        overlap_hist(axes[1], result)
        axes[1].set_title("Overlap: inspect sparse regions, not only range")
        for method in ("Naive", "Matching ATT", "IPW ATE", "Caliper ATT"):
            if method in result["summaries"]:
                axes[2].plot(np.abs(result["summaries"][method]["smd"]), [0, 1], "o-", label=method,
                             color=METHOD_COLORS.get(method, "#AB7A32"))
        axes[2].axvline(.1, color="#B91C1C", ls="--")
        axes[2].set(yticks=[0, 1], yticklabels=["Complexity", "Workload"], xlabel="Absolute SMD (fixed original SD)")
        axes[2].legend(fontsize=7, ncol=2)
        methods = list(result["summaries"])
        axes[3].barh(methods, [result["summaries"][m]["estimate"] for m in methods],
                     color=[METHOD_COLORS.get(m, "#AB7A32" if m == "Caliper ATT" else "#507A85") for m in methods])
        axes[3].axvline(result["sample_ate"], color=TRUTH, ls="--", label="Sample true ATE")
        axes[3].axvline(result["sample_att"], color=ACCENT, ls=":", label="Sample true ATT")
        if result["retained_truth"] is not None:
            axes[3].axvline(result["retained_truth"], color="#AB7A32", ls="-.", label="Retained treated truth")
        axes[3].set(xlabel="Effect (minutes); negative saves time")
        axes[3].legend(fontsize=7)
        s = result["summaries"]["IPW ATE"]
        no_matches = " NO CALIPER MATCHES." if not result["matches"].treated.size else ""
        status.set_text(f"IPW max weight {s['max_weight']:.1f}; ESS control/treated {s['ess'][0]:.1f}/{s['ess'][1]:.1f}; "
                        f"caliper retains {result['matches'].treated.size}/{data['t'].sum()}."+no_matches+
                        "\nMouse-drag the 3D axes. Caliper ATT targets retained treated units; no confidence intervals or causal guarantees.")
        state.update(result=result, updates=state["updates"]+1)
        fig.canvas.draw_idle()

    def reset_controls(_):
        for slider in controls.values():
            slider.reset()
        if toggle.get_status()[0]:
            toggle.set_active(0)

    for slider in controls.values():
        slider.on_changed(update)
    toggle.on_clicked(update)
    reset.on_clicked(reset_controls)
    update()
    return fig, controls, toggle, reset, state

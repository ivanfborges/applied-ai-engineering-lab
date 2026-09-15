"""Run: python -m streamlit run linear_regression_visual_lab.py"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import regression_math as rm
import regression_charts as rc

CATALOG = [
    "01 · Fitting a line", "02 · Residual distances", "03 · Why squared error?",
    "04 · OLS loss landscape", "05 · Gradient descent", "06 · Multiple regression plane",
    "07 · Geometric projection", "08 · Residual diagnostics", "09 · Nonlinear misspecification",
    "10 · Heteroskedasticity", "11 · Multicollinearity", "12 · Outlier sensitivity",
    "13 · Leverage vs residual", "14 · Extrapolation", "15 · OLS vs Ridge",
    "16 · Holding other variables constant", "17 · Intercept interpretation",
    "18 · R² and the mean baseline", "19 · Training vs generalization",
    "20 · Normal equations",
]

LESSONS = [
    ("Noisy observations, fitted line, and the generating mean.",
     "OLS selects the intercept and slope with the smallest training RSS.",
     "Noise changes estimates; one fitted sample cannot establish unbiasedness."),
    ("Vertical signed distances from observations to their predictions.",
     "OLS squares vertical errors in y, not perpendicular distances to the line.",
     "Residuals are observed minus predicted; their training sum is zero with an intercept."),
    ("Signed, absolute, and squared errors at the same residual values.",
     "Doubling an error doubles absolute loss but quadruples squared loss.",
     "Squared loss places much more weight on large errors."),
    ("RSS across pairs of intercept and slope, with the analytic OLS minimum.",
     "A full-rank design gives a convex quadratic bowl. The generating coefficients need not minimize this sample's RSS.",
     "Normal equations characterize the global minimum; the grid only visualizes it."),
    ("The same parameter updates shown as a moving line and a path on the loss contour.",
     "The MSE gradient is 2 Dᵀ(D beta − y)/n. Learning rates above the stability boundary can diverge.",
     "This is an educational optimizer, not a replacement for QR/SVD least squares."),
    ("Two predictors define a fitted plane through a cloud of target observations.",
     "Predictions are an affine combination of x1 and x2; residuals are in the target direction.",
     "Linear coefficients do not require normally distributed predictors."),
    ("A target vector, its projection onto a two-dimensional column space, and the remaining error.",
     "This example has three observations: each axis is an observation, not a feature.",
     "Dᵀe is approximately zero even when a chosen feature set is scientifically inadequate."),
    ("Three training diagnostic views of the same fit.",
     "Patterns in residual location, spread, or time can indicate inadequacy; a histogram alone cannot validate assumptions.",
     "Errors need not be Gaussian to compute OLS; residuals are not independent copies of errors."),
    ("The same observations fitted using x alone and then x plus x².",
     "The missing quadratic term can leave curvature in residuals despite training orthogonality.",
     "Linear in parameters does not mean linear in raw inputs."),
    ("Residuals from constant and increasing conditional noise variance.",
     "Both datasets use the same standardized noise draws and matched average conditional variance.",
     "Under exogeneity, heteroskedasticity need not bias OLS; classical standard errors can be wrong."),
    ("Independent versus correlated predictors, coefficient paths, predictions, and loss contours.",
     "A narrow range of independent feature variation creates weakly identified coefficient directions.",
     "Predictions on familiar feature combinations can remain stable while coefficients vary."),
    ("Fits before and after adding a point whose target is offset from the generating mean.",
     "Squared loss can pull the fitted line toward a single unusual observation.",
     "Compare both fits on the same uncontaminated holdout; training error alone is misleading."),
    ("A point's predictor extremity, fitted residual, and effect on the slope.",
     "Leverage is about feature geometry; influence combines geometry with response discrepancy.",
     "A high-leverage point can pull the line toward itself and end with a small residual."),
    ("The training domain and three possible extensions of the same observed relationship.",
     "All extensions agree on [0, 10], so training data cannot distinguish them.",
     "Extrapolation is an assumption about an unobserved domain, not learned evidence."),
    ("OLS, a scaled Ridge fit, coefficient paths, and repeated-sample prediction error.",
     "Ridge penalizes standardized slopes while leaving the intercept unpenalized.",
     "Shrinkage introduces bias and can reduce variance; a held-out benefit is not guaranteed."),
    ("One latency-model input moves while the other stays fixed.",
     "An additive coefficient gives the fitted mean change per unit, conditional on the other inputs.",
     "Conditional association is not a causal intervention; ensure the comparison has data support."),
    ("The intercept at x=0 alongside the actual observed predictor range.",
     "A line has an intercept even if x=0 is far from the training domain.",
     "Centering changes the reference point of the intercept, not fitted predictions."),
    ("Distances to the sample mean versus distances to OLS predictions.",
     "R² compares RSS with SST on the evaluated sample.",
     "Training R² is not a generalization or causality guarantee; constant targets make the ratio undefined."),
    ("Training and validation errors as polynomial degree changes.",
     "Feature transformations increase model capacity even though coefficients enter linearly.",
     "Choosing a degree from this validation set requires a separate final test for an unbiased performance estimate."),
    ("The design, Gram matrix, right-hand side, and three numerical solutions.",
     "Forming DᵀD squares the condition number; pseudoinversion does not undo lost precision.",
     "Use a stable least-squares solver; near-equal predictions do not imply stable coefficients."),
]


def chart(fig, key):
    st.plotly_chart(fig, key="chart_"+key, width="stretch", theme=None,
                    config={"displaylogo": False, "scrollZoom": False})


def metric_row(values):
    with st.container(horizontal=True):
        for label, value in values.items():
            shown = f"{value:.4g}" if np.isfinite(value) else "Undefined"
            st.metric(label, shown, border=True)


def equation(beta):
    st.latex(r"\hat y = " + f"{beta[0]:.3f}" +
             "".join(f" {v:+.3f} x_{{{i}}}" for i, v in enumerate(beta[1:], 1)))


@st.cache_data(max_entries=24, show_spinner="Calculating repeated training samples...")
def repeated(n, rho, noise, alpha, repeats, seed):
    return rm.stability(n, rho, noise, alpha, repeats, seed)


@st.cache_data(max_entries=24)
def polynomial(n, noise, seed):
    return rm.polynomial_experiment(n, noise, seed)


st.set_page_config(page_title="Visual linear regression lab", page_icon=":material/scatter_plot:",
                   layout="wide")
st.title("Visual linear regression lab")
st.caption("DAY 18 · Synthetic systems · See the fit, question the assumptions")
view = st.sidebar.selectbox("Explore a concept", CATALOG, key="view")
section = CATALOG.index(view)+1
st.sidebar.caption("Controls below apply only to this view.")
st.subheader(view.split(" · ", 1)[1])
seed, n, noise = 42, 80, 1.
if section not in (3, 7, 16, 20):
    seed = int(st.sidebar.number_input("Random seed", 0, 9999, 42, key="seed"))
    n = st.sidebar.slider("Sample size", 20, 150 if section == 19 else 300, 80, 10, key="n")
    if section != 10:
        noise = st.sidebar.slider("Noise standard deviation", 0., 5., 1., .1, key="noise")
if section in (1, 2, 4, 5, 8, 17, 18):
    intercept = st.sidebar.slider("True intercept", -5., 5., 2., .25, key="intercept")
    slope = st.sidebar.slider("True slope", -3., 3., 1.5, .1, key="slope")
    domain = (100., 110.) if section == 17 and st.sidebar.toggle(
        "Observe only x > 100", value=True, key="far_zero") else (-3., 3.)
    x, y, mean = rm.line_data(n, noise, intercept, slope, seed, domain)
    fit = rm.fit_ols(x, y)
    beta = fit["beta"]

what, why, takeaway = LESSONS[section-1]
st.markdown(f"**What you are seeing:** {what}")

if section in (1, 2):
    show = st.sidebar.toggle("Show residual lines", value=section == 2, key="residual_lines")
    chart(rc.line_chart(x, y, beta, true_beta=(intercept, slope), residual_lines=show), "line")
    equation(beta)
    s = rm.scores(y, fit["pred"])
    metric_row({f"Training {k}": s[k] for k in ("RSS", "MSE", "RMSE", "R2")})
    metric_row({"Estimated intercept": beta[0], "Estimated slope": beta[1]})
    if section == 2:
        st.latex(r"e_i=y_i-\hat y_i,\quad RSS=\sum_i e_i^2")
        metric_row({"Sum of training residuals": fit["residual"].sum()})

elif section == 3:
    e = np.array([-10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10])
    fig = go.Figure()
    for label, values, color in (("Signed residual", e, rc.BLUE),
                                 ("Absolute residual", abs(e), rc.TEAL),
                                 ("Squared residual", e*e, rc.RED)):
        fig.add_trace(go.Bar(x=e.astype(str), y=values, name=label, marker_color=color))
    chart(rc.style(fig, "Large errors receive disproportionate squared loss",
                   "Residual", "Value / loss"), "penalty")
    st.dataframe(pd.DataFrame({"Residual": e, "Absolute": abs(e), "Squared": e*e}),
                 hide_index=True)

elif section == 4:
    surface, contour = rc.loss_figures(x, y, (intercept, slope))
    chart(surface, "surface")
    chart(contour, "contour")
    metric_row({"OLS minimum RSS": rm.scores(y, fit["pred"])["RSS"],
                "RSS at true coefficients": rm.scores(y, mean)["RSS"]})
    st.caption("The red point is the exact least-squares solution, not a grid search estimate.")

elif section == 5:
    ratio = st.sidebar.slider("Learning rate / stability boundary", .05, 1.3, .6, .05,
                              key="rate")
    steps = st.sidebar.slider("Optimization steps", 10, 150, 80, 10, key="steps")
    result = rm.gradient_descent(x, y, rate_ratio=ratio, steps=steps)
    chart(rc.descent_animation(x, y, result), "descent")
    metric_row({"Learning rate": result["rate"], "Stability boundary": result["critical"],
                "Final MSE": result["losses"][-1], "OLS MSE": rm.scores(y, fit["pred"])["MSE"]})
    st.caption(result["status"]+". Frame controls animate both panels. Divergent paths can leave the viewport.")
    loss = go.Figure(rc.trace(np.arange(len(result["losses"])), result["losses"], "MSE"))
    chart(rc.style(loss, "Loss per optimization step", "Step", "MSE"), "steps_loss")
    if ratio >= 1:
        st.warning("This rate is at or above the quadratic stability boundary; convergence is not guaranteed.")

elif section == 6:
    b1 = st.sidebar.slider("True coefficient x1", -4., 4., 3., .25, key="b1")
    b2 = st.sidebar.slider("True coefficient x2", -4., 4., -2., .25, key="b2")
    X, _, _ = rm.correlated_data(n, 0, noise, seed)
    y = 2+X@np.array([b1, b2])+rm.generator(seed+1).normal(0, noise, n)
    fit = rm.fit_ols(X, y)
    chart(rc.plane(X, y, fit["beta"]), "plane")
    equation(fit["beta"])
    metric_row({"Training RMSE": rm.scores(y, fit["pred"])["RMSE"]})

elif section == 7:
    target = [st.sidebar.slider(f"Target coordinate y{i+1}", -5., 5., v, .25,
                                key=f"coord{i}") for i, v in enumerate([2., -1., 4.])]
    fit = rm.projection(target)
    chart(rc.projection_chart(fit), "projection")
    st.latex(r"y=\hat y+e,\qquad D^Te\approx 0")
    st.dataframe(pd.DataFrame({"y": fit["y"], "y_hat": fit["pred"], "e": fit["residual"]}))
    metric_row({"max |Dᵀe|": np.max(abs(fit["design"].T@fit["residual"])),
                "Reconstruction error": np.linalg.norm(fit["y"]-fit["pred"]-fit["residual"])})
    st.caption("D = [ones, (-1, 0, 1)]. The translucent plane is its column space. "
               "The right-angle marker is omitted when the residual is zero.")

elif section == 8:
    chart(rc.diagnostics(y, fit["pred"]), "diagnostics")
    metric_row({"Training residual mean": fit["residual"].mean(),
                "Training residual SD": fit["residual"].std()})
    st.info("A centered histogram is not an assumption test. Training residuals are constrained by fitting; "
            "normal predictors are not required and residuals need not look perfectly Gaussian.")

elif section == 9:
    x, y, linear, quadratic = rm.curvature(n, noise, seed)
    chart(rc.curvature_chart(x, y, linear, quadratic), "curvature")
    metric_row({"x-only training RMSE": rm.scores(y, linear["pred"])["RMSE"],
                "Quadratic training RMSE": rm.scores(y, quadratic["pred"])["RMSE"]})
    st.latex(r"\hat y=\beta_0+\beta_1 x+\beta_2 x^2")
    st.caption("These are training diagnostics. Adding a feature cannot increase optimized training RSS; "
               "that alone does not establish generalization.")

elif section == 10:
    strength = st.sidebar.slider("Noise growth with x", 0., 2., .6, .1, key="growth")
    x, ya, yb, sigma = rm.variance_data(n, strength, seed)
    chart(rc.variance_chart(x, ya, yb), "variance")
    st.latex(r"\sigma(x)=0.3+g x,\quad \mathrm{Var}(\varepsilon\mid x)=\sigma(x)^2")
    metric_row({"Minimum noise SD": sigma.min(), "Maximum noise SD": sigma.max()})
    st.info("Zero conditional error mean is built into both generators. No classical or robust "
            "standard errors are estimated here.")

elif section in (11, 15):
    rho = st.sidebar.slider("Population feature correlation", 0., .999, .99, .001, key="rho")
    alpha = st.sidebar.slider("Ridge alpha", 0., 100., 10., .5, key="alpha") if section == 15 else 10.
    repeats = st.sidebar.slider("Independent training samples", 20, 150, 60, 10, key="repeats")
    X, y, _ = rm.correlated_data(n, rho, noise, seed)
    Xt, yt, _ = rm.correlated_data(250, rho, noise, seed+1)
    ols = rm.fit_ols(X, y)
    ridge, rb = rm.fitted_model(X, y, alpha)
    result = repeated(n, rho, noise, alpha, repeats, seed)
    equation(ols["beta"])
    if section == 11:
        independent = repeated(n, 0., noise, alpha, repeats, seed)
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Independent predictors", "Correlated predictors"))
        for col, r in enumerate((independent, result), 1):
            for j, color in ((1, rc.BLUE), (2, rc.RED)):
                fig.add_trace(rc.trace(np.arange(repeats), r["beta"][:, 0, j],
                              f"beta{j}", color, showlegend=col == 1), row=1, col=col)
        fig.update_yaxes(matches="y")
        chart(rc.style(fig, "Coefficient estimates across independent samples",
                       "Training sample", "Coefficient"), "stability")
        rows = []
        contour = make_subplots(rows=1, cols=2, subplot_titles=("Independent loss", "Correlated loss"))
        axis = np.linspace(-8, 8, 70)
        a, b = np.meshgrid(axis, axis)
        for col, correlation in enumerate((0., rho), 1):
            xx, yy, _ = rm.correlated_data(n, correlation, noise, seed)
            model = rm.fit_ols(xx, yy)
            # Profile out the intercept by centering; axes are slope deviations from OLS.
            centered = xx-xx.mean(axis=0)
            increments = np.stack((a, b), axis=-1)
            gram = centered.T@centered
            excess = np.einsum("...i,ij,...j->...", increments, gram, increments)
            contour.add_trace(go.Contour(x=axis, y=axis, z=excess, showscale=False,
                contours=dict(start=0, end=100, size=10), colorscale="Blues"), row=1, col=col)
            xt, ytest, _ = rm.correlated_data(250, correlation, noise, seed+1)
            r = independent if col == 1 else result
            rows.append({"Design": "Independent" if col == 1 else "Correlated",
                         "Sample correlation": np.corrcoef(xx.T)[0, 1],
                         "beta1": model["beta"][1], "beta2": model["beta"][2],
                         "Holdout RMSE": rm.scores(ytest, rm.design_matrix(xt)@model["beta"])["RMSE"],
                         "Var(beta1)": r["beta"][:, 0, 1].var(ddof=1),
                         "Var(beta2)": r["beta"][:, 0, 2].var(ddof=1),
                         "Mean prediction variance": r["variance"][0]})
        chart(rc.style(contour, "Equal excess RSS: weak identification stretches the valley",
                       "beta1 − OLS beta1", "beta2 − OLS beta2"), "collinear_contour")
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        predfig = go.Figure()
        for name, r, color in (("Independent", independent, rc.BLUE), ("Correlated", result, rc.RED)):
            predfig.add_trace(rc.trace(r["mean"], r["predictions"][:, 0].mean(axis=0),
                name, color, "markers", error_y=dict(type="data",
                array=r["predictions"][:, 0].std(axis=0), visible=True)))
        chart(rc.style(predfig, "Mean prediction ± one SD across training samples",
                       "Generating mean at fixed test inputs", "OLS prediction"), "pred_stability")
        st.caption("Each design has its own matched test distribution. Error bars summarize repeated fits, "
                   "not prediction intervals. Loss contours use the same excess-RSS levels.")
    else:
        st.dataframe(pd.DataFrame({"Parameter": ["Intercept", "x1", "x2"],
                                   "OLS": ols["beta"], "Ridge (original units)": rb}), hide_index=True)
        metric_row({"OLS holdout RMSE": rm.scores(yt, rm.design_matrix(Xt)@ols["beta"])["RMSE"],
                    "Ridge holdout RMSE": rm.scores(yt, ridge.predict(Xt))["RMSE"],
                    "OLS slope norm": np.linalg.norm(ols["beta"][1:]),
                    "Ridge slope norm": np.linalg.norm(rb[1:])})
        alphas = np.logspace(-4, 3, 45)
        paths = np.array([rm.fitted_model(X, y, a)[1] for a in alphas])
        fig = go.Figure()
        for j, color in ((1, rc.BLUE), (2, rc.RED)):
            fig.add_trace(rc.trace(alphas, paths[:, j], f"beta{j}", color))
            fig.add_hline(y=ols["beta"][j], line_dash="dot", line_color=color)
        fig.update_xaxes(type="log")
        chart(rc.style(fig, "Ridge coefficient paths; dotted lines show OLS",
                       "Alpha (log scale)", "Coefficient in original units"), "ridge_path")
        predictions = go.Figure()
        predictions.add_trace(rc.trace(yt, rm.design_matrix(Xt)@ols["beta"], "OLS", rc.BLUE, "markers"))
        predictions.add_trace(rc.trace(yt, ridge.predict(Xt), "Ridge", rc.RED, "markers"))
        chart(rc.style(predictions, "Predictions on the same held-out rows", "Actual", "Predicted"), "ridge_pred")
        bars = go.Figure()
        for label, values, color in (("Squared bias", result["bias2"], rc.ORANGE),
                                     ("Variance", result["variance"], rc.BLUE)):
            bars.add_trace(go.Bar(x=["OLS", "Ridge"], y=values, name=label, marker_color=color))
        bars.update_layout(barmode="stack")
        chart(rc.style(bars, "Finite-ensemble error against the known conditional mean",
                       "Estimator", "Mean squared prediction error"), "bias_variance")
        st.dataframe(pd.DataFrame({"Model": ["OLS", "Ridge"],
            "Squared bias": result["bias2"], "Prediction variance": result["variance"],
            "Error vs true mean": result["mean_error"],
            "Mean noisy holdout RMSE": result["rmse"].mean(axis=0)}), hide_index=True)
        st.caption("Bias and variance use a finite ensemble, not population estimates. Their sum equals "
                   "ensemble MSE against the noiseless mean; observation noise is excluded from that sum.")

elif section in (12, 13):
    px = st.sidebar.slider("Added point x", 0., 30., 8., .5, key="point_x")
    offset = st.sidebar.slider("Added point offset from true line", -40., 40., 15., 1., key="offset")
    r = rm.outlier_case(n, noise, px, offset, seed)
    grid = np.linspace(min(r["x"].min(), px), max(r["x"].max(), px), 200)
    left, right = st.columns(2)
    with left:
        chart(rc.line_chart(r["x"], r["y"], r["before"]["beta"], "Before added point",
                            grid=grid), "before")
    with right:
        f = rc.line_chart(r["xa"], r["ya"], r["after"]["beta"], "After added point", grid=grid)
        f.add_trace(rc.trace([px], [r["ya"][-1]], "Added point", rc.RED, "markers",
                            marker_size=14))
        chart(f, "after")
    st.dataframe(pd.DataFrame({"Fit": ["Before", "After"],
        "Intercept": [r[k]["beta"][0] for k in ("before", "after")],
        "Slope": [r[k]["beta"][1] for k in ("before", "after")],
        "Clean holdout RMSE": r["rmse"]}), hide_index=True)
    metric_row({"Added point leverage": r["leverage"],
                "Added point fitted residual": r["point_residual"],
                "Slope change": r["after"]["beta"][1]-r["before"]["beta"][1]})
    if section == 13:
        st.caption("Try x near 0 with a large offset, then x near 30 with the same offset. "
                   "A high-leverage point on the true line need not distort the fit.")

elif section == 14:
    mode = st.sidebar.selectbox("Unobserved continuation", ["Linear", "Saturation", "Accelerating"],
                                key="continuation")
    x, y, _ = rm.line_data(n, noise, seed=seed, domain=(0, 10))
    fit = rm.fit_ols(x, y)
    grid = np.linspace(0, 30, 300)
    mean = 2+1.5*grid
    if mode == "Saturation":
        mean = np.where(grid <= 10, mean, 17+3*(1-np.exp(-(grid-10)/2)))
    if mode == "Accelerating":
        mean = mean + .25*np.maximum(grid-10, 0)**2
    fig = rc.line_chart(x, y, fit["beta"], "Identical training data, different possible futures", grid=grid)
    fig.add_trace(rc.trace(grid, mean, "Chosen generating mean", rc.ORANGE, line_dash="dash"))
    fig.add_vrect(x0=0, x1=10, fillcolor=rc.TEAL, opacity=.07, line_width=0,
                  annotation_text="Training domain", annotation_position="top left")
    fig.add_vrect(x0=10, x1=30, fillcolor=rc.RED, opacity=.06, line_width=0,
                  annotation_text="Extrapolation", annotation_position="top right")
    chart(fig, "extrapolation")
    metric_row({"RMSE vs chosen mean for x > 10": rm.scores(mean[grid > 10],
        (rm.design_matrix(grid)@fit["beta"])[grid > 10])["RMSE"]})

elif section == 16:
    tokens = st.sidebar.slider("Input tokens", 0, 4000, 1000, 100, key="tokens")
    concurrency = st.sidebar.slider("Concurrent requests", 1, 30, 5, key="concurrency")
    moving = st.sidebar.selectbox("Move this feature", ["Input tokens", "Concurrent requests"], key="moving")
    grid = np.linspace(0, 4000, 100) if moving == "Input tokens" else np.arange(1, 31)
    response = 120+.08*grid+12*concurrency if moving == "Input tokens" else 120+.08*tokens+12*grid
    current = tokens if moving == "Input tokens" else concurrency
    fig = go.Figure(rc.trace(grid, response, "Conditional mean", rc.TEAL))
    fig.add_trace(rc.trace([current], [120+.08*tokens+12*concurrency], "Selected input",
                          rc.RED, "markers"))
    chart(rc.style(fig, "Synthetic additive latency model", moving, "Mean latency (ms)"), "conditional")
    st.latex(r"E[latency\mid tokens,concurrency]=120+0.08\,tokens+12\,concurrency")
    metric_row({"Selected mean latency (ms)": 120+.08*tokens+12*concurrency})
    st.caption("Fixed hypothetical equation, not a fit to real infrastructure. Token coefficient units: "
               "ms per token at fixed concurrency. This is not a causal estimate.")

elif section == 17:
    grid = np.linspace(min(0, x.min()), x.max(), 250)
    fig = rc.line_chart(x, y, beta, "The intercept can be far outside observed support", grid=grid)
    fig.add_trace(rc.trace([0], [beta[0]], "Intercept at x = 0", rc.RED, "markers"))
    fig.add_vrect(x0=x.min(), x1=x.max(), fillcolor=rc.TEAL, opacity=.1, line_width=0,
                  annotation_text="Observed domain")
    chart(fig, "intercept")
    metric_row({"Intercept": beta[0], "Prediction at training mean x": beta[0]+beta[1]*x.mean()})

elif section == 18:
    s = rm.scores(y, fit["pred"])
    fig = make_subplots(rows=1, cols=2, subplot_titles=("SST: distance to sample mean", "RSS: distance to fitted line"))
    for col, pred, name, color in ((1, np.full(n, y.mean()), "Mean", rc.ORANGE),
                                   (2, fit["pred"], "OLS", rc.TEAL)):
        fig.add_trace(rc.segments(x, y, pred, "Distances", color), row=1, col=col)
        fig.add_trace(rc.trace(x, y, "Observed", rc.BLUE, "markers", showlegend=col == 1), row=1, col=col)
        fig.add_trace(rc.trace(x, pred, name, color), row=1, col=col)
    chart(rc.style(fig, "Same observations, two reference predictions", "x", "y"), "r2")
    metric_row({k: s[k] for k in ("SST", "RSS", "R2")})
    st.latex(r"R^2=1-\frac{RSS}{SST}")
    st.caption("Here the mean uses the training sample because this is training R². For held-out R², "
               "the scoring reference uses the evaluation mean; a deployable baseline uses the training mean.")

elif section == 19:
    # Keep high-degree comparisons bounded and avoid interpolating fewer than 16 rows.
    sample = min(n, 150)
    degree = st.sidebar.select_slider("Displayed polynomial degree", list(range(1, 16)), 5, key="degree")
    r = polynomial(sample, noise, seed)
    fig = go.Figure()
    for i, label in enumerate(("Training", "Validation")):
        fig.add_trace(rc.trace(r["degrees"], r["errors"][:, i], label, rc.COLORS[i], "lines+markers"))
    chart(rc.style(fig, "Model capacity changes with polynomial degree", "Degree", "RMSE"), "degree_error")
    curve = go.Figure(rc.trace(r["train"], r["ytrain"], "Training", rc.BLUE, "markers"))
    curve.add_trace(rc.trace(r["valid"], r["yvalid"], "Validation", rc.ORANGE, "markers"))
    curve.add_trace(rc.trace(r["grid"], r["curves"][degree-1], f"Degree {degree}", rc.RED))
    curve.add_trace(rc.trace(r["grid"], r["mean"], "Generating mean", rc.TEAL, line_dash="dash"))
    chart(rc.style(curve, "Polynomial feature regression", "x", "y"), "polynomial")
    metric_row({"Training RMSE": r["errors"][degree-1, 0],
                "Validation RMSE": r["errors"][degree-1, 1],
                "Design condition number": r["condition"][degree-1]})
    st.caption(f"{sample} training observations, 180 validation observations. "
               "Raw input range [-1, 1] limits scale growth, but high-degree designs can still be ill-conditioned.")

elif section == 20:
    exponent = st.sidebar.slider("Log10 separation between feature columns", -10., 0., 0., .5,
                                 key="separation")
    r = rm.normal_equations(10**exponent)
    a, b, c = st.columns(3)
    with a:
        st.markdown("**Design D (includes intercept)**")
        st.dataframe(pd.DataFrame(r["X"], columns=["Intercept", "x1", "x2"]))
    with b:
        st.markdown("**DᵀD**")
        st.dataframe(pd.DataFrame(r["gram"]))
    with c:
        st.markdown("**Dᵀy**")
        st.dataframe(pd.DataFrame({"Dᵀy": r["rhs"]}))
    st.dataframe(pd.DataFrame({"Parameter": ["Intercept", "x1", "x2"],
        "pinv(DᵀD) Dᵀy": r["normal"], "lstsq(D, y)": r["stable"],
        "scikit-learn": r["library"]}), hide_index=True)
    metric_row({"cond(D)": r["condition"],
        "Normal vs lstsq coefficient distance": np.linalg.norm(r["normal"]-r["stable"]),
        "Normal vs lstsq prediction distance": np.linalg.norm(r["X"]@(r["normal"]-r["stable"])),
        "sklearn vs lstsq prediction distance": np.linalg.norm(r["X"]@(r["library"]-r["stable"]))})
    st.latex(r"\hat\beta_{demo}=\mathrm{pinv}(D^TD)D^Ty")
    st.warning("The Gram-matrix pseudoinverse is an educational comparison. Main fits use stable solvers.")

st.markdown(f"**Why this happens:** {why}")
st.info(f"Interview takeaway: {takeaway}")
st.caption("All displayed values are calculated for the current controls. Synthetic demonstrations "
           "are not benchmarks or real-world causal evidence.")
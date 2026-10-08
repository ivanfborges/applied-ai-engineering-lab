"""Start locally with: streamlit run app.py"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

import visual_core as core
import visualizations as viz

st.set_page_config(page_title="Day 35 | Explainability lab", page_icon=":material/insights:", layout="wide")


@st.cache_resource(max_entries=6, show_spinner="Fitting a reproducible CPU forest...")
def trained(config):
    return core.build_lab(config)


@st.cache_data(max_entries=12, show_spinner="Measuring frozen-model reliance...")
def importances(config, metric, repeats):
    return core.importance_results(trained(config), metric, repeats)


@st.cache_data(max_entries=16, show_spinner=False)
def curves(config, feature):
    return core.response_curves(trained(config), feature, rows=min(100, len(trained(config)["X_test"])), grid_size=25)


@st.cache_data(max_entries=8, show_spinner="Averaging the 3D replacement grid...")
def surface(config, row):
    return core.surfaces(trained(config), row=row)


@st.cache_data(max_entries=12, show_spinner="Computing probability SHAP on 80 held-out rows...")
def shap_values(config, size, seed):
    return core.forest_shap(trained(config), size, seed, explained_count=min(80, len(trained(config)["X_test"])))


@st.cache_data(max_entries=8, show_spinner=False)
def shuffled_path(config, feature, repeat, joint):
    columns = (0, 1) if joint else (feature,)
    return core.permutation_path(trained(config), columns, repeat)


@st.cache_data(max_entries=1, show_spinner="Running the random-label split-search control...")
def cardinality():
    return core.cardinality_experiment()


@st.cache_data(max_entries=6, show_spinner="Fitting the deliberately invalid leakage model...")
def leakage(config):
    return core.leakage_experiment(trained(config))


def chart(fig, key):
    st.plotly_chart(fig, key=key, width="stretch", theme=None, config={
        "displaylogo": False, "scrollZoom": False,
        "toImageButtonOptions": {"format": "png", "scale": 3, "filename": key},
    })


def interpretation(observing, misinterpretation, engineering):
    st.markdown("#### What you are observing")
    st.write(observing)
    st.markdown("#### Common misinterpretation")
    st.write(misinterpretation)
    st.markdown("#### Production engineering takeaway")
    st.write(engineering)


def download_png(label, builder, name):
    st.download_button(label, data=lambda: viz.png_bytes(builder()), file_name=name,
                       mime="image/png", on_click="ignore", key=f"download_{name}")


def experiment_record(lab, hypothesis, result, candidate, limitations):
    return {
        "hypothesis": hypothesis,
        "configuration": {"synthetic": True, "seed": core.SEED, "config_order":
            ["n", "rho", "interaction", "trees", "depth", "leaf"],
            "config": list(lab["config"]), "split": "stratified 75/25",
            "evaluation": "held-out rows"},
        "result": result,
        "interpretation_candidate": candidate,
        "review_status": "Pending author review.",
        "limitation": limitations,
    }


if "lab_config" not in st.session_state:
    st.session_state.lab_config = core.DEFAULT_CONFIG
if "shap_config" not in st.session_state:
    st.session_state.shap_config = (50, 35)

st.title("Explainability for classical machine learning")
st.caption("Day 35 · CPU-only synthetic learning lab · seed 35 · quantities before conclusions")

with st.sidebar:
    st.header("Dataset and forest")
    st.caption("These controls change training. Apply them together.")
    with st.form("training_controls"):
        n = st.slider("Dataset sample size", 300, 2000, 900, 100, key="sample_size")
        rho = st.slider("Signal/proxy correlation parameter", 0.0, 0.995, 0.95, 0.005, format="%.3f", key="rho")
        interaction = st.slider("Interaction strength", 0.0, 3.0, 1.6, 0.1, key="interaction")
        trees = st.slider("Forest trees", 20, 120, 60, 20, key="trees")
        depth = st.slider("Maximum tree depth", 2, 12, 6, 1, key="depth")
        leaf = st.slider("Minimum samples per leaf", 1, 20, 5, 1, key="leaf")
        apply = st.form_submit_button("Apply dataset and forest", type="primary")
    if apply:
        st.session_state.lab_config = (n, rho, interaction, trees, depth, leaf)
    config = tuple(st.session_state.lab_config)
    st.caption(f"Active: n={config[0]}, rho={config[1]:.3f}, interaction={config[2]:.1f}; "
               f"{config[3]} trees, depth {config[4]}, leaf {config[5]}.")
    st.divider()
    st.header("Inspect the fitted model")
    st.caption("These controls update explanations or displays; they do not retrain.")
    feature_name = st.selectbox("Selected explanation feature", core.FEATURES, key="feature")
    feature = core.FEATURES.index(feature_name)
    max_observation = min(80, (config[0]+3)//4)-1
    if st.session_state.get("observation", 0) > max_observation:
        st.session_state.observation = max_observation
    selected = st.slider("Selected held-out observation", 0, max_observation, 0, key="observation")
    background_options = [size for size in [10, 25, 50, 100, 200, 300]
                          if size <= config[0] - (config[0]+3)//4]
    if st.session_state.shap_config[0] not in background_options:
        st.session_state.shap_config = (background_options[-1], st.session_state.shap_config[1])
    if st.session_state.get("background_size", 50) not in background_options:
        st.session_state.background_size = background_options[-1]
    with st.form("shap_controls"):
        background_size = st.select_slider("SHAP training background size", background_options,
                                            value=st.session_state.get("background_size", 50), key="background_size")
        background_seed = st.number_input("SHAP background seed", 0, 10000, 35, key="background_seed")
        apply_shap = st.form_submit_button("Apply SHAP background")
    if apply_shap:
        st.session_state.shap_config = (background_size, int(background_seed))
    st.caption("Up to 80 held-out rows are selectable for local explanations. "
               "PDP/ICE use up to 100; surfaces average 60.")
    st.caption("Plots: drag to inspect, hover for values, camera icon to export PNG.")

labels = [
    "Overview & Dataset", "Tree Importance / MDI", "Permutation Importance",
    "PDP & 3D Surfaces", "ICE", "SHAP", "Explainability Pitfalls",
    "Summary & Interview Review",
]
tabs = st.tabs(labels, on_change="rerun", key="lab_tab")
lab = trained(config)
probability = core.positive_probability(lab["model"], lab["X_test"])
with st.container(horizontal=True):
    st.metric("Held-out ROC-AUC", f"{core.metric_score(lab['y_test'], probability, 'ROC-AUC'):.4f}", border=True)
    st.metric("Held-out positive rate", f"{lab['y_test'].mean():.1%}", border=True)
    st.metric("Observed signal/proxy correlation",
              f"{np.corrcoef(lab['X_test'][:, :2].T)[0, 1]:.3f}", border=True)
    st.metric("Train / test rows", f"{len(lab['train_ids'])} / {len(lab['test_ids'])}", border=True)

if tabs[0].open:
    with tabs[0]:
        st.subheader("Known data generation, separate from the fitted forest")
        st.write("signal and context are independent standard normals. The proxy carries redundant "
                 "signal; noise has no direct target term. All data are synthetic.")
        st.latex(r"x_{\mathrm{proxy}}=\rho x_{\mathrm{signal}}+\sqrt{1-\rho^2}\epsilon,\quad \epsilon\sim N(0,1)")
        st.latex(r"p(y=1\mid x)=\sigma(1.4x_{\mathrm{signal}}+0.9x_{\mathrm{context}}+\gamma x_{\mathrm{signal}}x_{\mathrm{context}})")
        st.caption("rho is the population correlation parameter; gamma is interaction strength; "
                   "sigma(z)=1/(1+exp(-z)). A seeded uniform draw samples binary labels.")
        chart(viz.dataset_scatter(lab), "dataset")
        st.caption("Axes are actual input values. Blue circles are observed class 0; orange crosses "
                   "are class 1. Colors encode sampled labels, not model probabilities.")
        data = pd.DataFrame(lab["X_test"][:12], columns=core.FEATURES)
        data["observed_y"] = lab["y_test"][:12]
        data["predicted_P(y=1)"] = probability[:12]
        st.dataframe(data, hide_index=True)
        interpretation(
            "Applying training controls regenerates the seeded dataset and fitted forest. Changing "
            "feature, row, grid position or centering reuses the trained forest.",
            "A high ROC-AUC does not prove calibrated probabilities or a valid explanation. "
            "The known generator and the learned forest are different functions.",
            "Fix evaluation boundaries and feature availability before investigating explanations."
        )

if tabs[1].open:
    with tabs[1]:
        st.subheader("Move a candidate split and watch impurity change")
        X, y = core.tree_split_rows(lab)
        split_feature = st.selectbox("Candidate split feature", ["signal", "context"], key="split_feature")
        split_index = ["signal", "context"].index(split_feature)
        lo, hi = np.quantile(X[:, split_index], [0.02, 0.98])
        threshold = st.slider("Candidate split threshold", float(lo), float(hi), float((lo+hi)/2), key="threshold")
        stats = core.split_statistics(X, y, split_index, threshold)
        chart(viz.split_plot(X, y, split_index, threshold, animate=True), "candidate_split")
        st.caption("Axes: signal and context. Color/symbol: observed class. Dashed line: candidate "
                   "threshold. Slider statistics below refer to the selected threshold. Play independently "
                   "sweeps the boundary and gain on the same rows without refitting the forest.")
        with st.container(horizontal=True):
            for name, value in [("Parent Gini", stats["parent"]), ("Weighted child Gini", stats["weighted_children"]),
                                ("Weighted impurity reduction", stats["gain"])]:
                st.metric(name, f"{value:.4f}", border=True)
        chart(viz.split_distributions(stats), "split_counts")
        chart(viz.tree_diagram(stats, split_feature, threshold), "stump")
        st.latex(r"G(t)=1-\sum_k p_{k,t}^2,\quad \Delta G=G(t)-\frac{n_L}{n_t}G(L)-\frac{n_R}{n_t}G(R)")
        st.caption("t is a parent node; L and R are its children; n counts rows; p(k,t) is class k's "
                   "proportion. MDI sums (n_t/n_root)*Delta G over nodes using a feature, then "
                   "normalizes and averages tree importances.")
        chart(viz.importance_plot(importances(config, "ROC-AUC", 8), "ROC-AUC"), "forest_mdi")
        if st.toggle("Show high-cardinality random-label control", key="cardinality_toggle"):
            control = cardinality()
            chart(viz.cardinality_plot(control), "cardinality")
            st.caption(f"Both nuisance features are independent of random labels. Training split candidates: "
                       f"continuous {control['candidates'][0]}, binary {control['candidates'][1]}. "
                       f"Held-out AUC: {control['auc']:.4f}. This is a single-seed control, not a benchmark.")
            record = experiment_record(lab, "More candidate splits can fit random-label fluctuations.",
                {"control_seed": 35, "control_rows": 600, "control_auc": control["auc"],
                 "mdi": control["mdi"].tolist(), "permutation": control["pi"].tolist()},
                "Compare split-search credit with held-out reliance; pending author review.",
                "The control uses its own 50-tree unrestricted-depth forest and a 60/40 split; "
                "it changes feature cardinality and distribution jointly.")
            record["configuration"] = control["configuration"]
            st.download_button("Download control experiment record", json.dumps(record, indent=2),
                               "cardinality_record.json", "application/json", on_click="ignore")
        interpretation(
            "The threshold changes class proportions and weighted impurity reduction. The manual "
            "stump is illustrative; final MDI comes from the separately fitted full forest.",
            "MDI is not held-out performance loss. Many available split positions can generate "
            "training credit for noise; a noise score in one run does not prove a universal bias size.",
            "Compare internal split credit with held-out, metric-specific reliance and audit candidate feature availability."
        )

if tabs[2].open:
    with tabs[2]:
        st.subheader("Transport feature values while targets stay attached to their rows")
        metric = st.selectbox("Evaluation metric", list(core.METRICS), key="permutation_metric")
        repeats = st.slider("Permutation repetitions", 2, 20, 8, key="permutation_repeats")
        repeat = st.slider("Animated shuffle seed offset", 0, 9, 0, key="shuffle_repeat")
        joint = st.toggle("Animate a shared signal/proxy shuffle", key="joint_shuffle")
        path = shuffled_path(config, feature, repeat, joint)
        shown_feature = 0 if joint else feature
        chart(viz.permutation_animation(lab, path, metric, shown_feature), "permutation_transport")
        baseline = core.metric_score(lab["y_test"], path["probabilities"][0], metric)
        shuffled = core.metric_score(lab["y_test"], path["probabilities"][-1], metric)
        with st.container(horizontal=True):
            st.metric(f"Baseline {metric}", f"{baseline:.4f}", border=True)
            st.metric(f"Endpoint shuffled {metric}", f"{shuffled:.4f}", border=True)
            st.metric("Baseline minus shuffled score", f"{baseline-shuffled:.4f}", border=True)
        st.caption("X: selected feature value; Y: model P(y=1). Color and symbol retain each row's "
                   "observed target. Both panels have fixed axes. Intermediate positions interpolate "
                   "for visual motion and are not permutation-importance samples. The endpoint is one actual shuffle.")
        results = importances(config, metric, repeats)
        chart(viz.importance_plot(results, metric), "importance_comparison")
        chart(viz.group_plot(results, metric), "grouped_permutation")
        st.caption("The animation uses seed 35 + offset. Repeated estimates use independent seeded "
                   "streams; the displayed animation is not one of the library's exact draws. "
                   "Error bars are standard deviations of shuffled score differences, not confidence intervals.")
        st.latex(r"\widehat{PI}_G=s(f,D)-\frac{1}{R}\sum_{r=1}^R s(f,D_G^{\pi_r})")
        st.caption("f is the frozen model, D the held-out data, G a feature or group, R the number "
                   "of repeats, and pi a row permutation. s is higher-is-better; negative log loss "
                   "means this difference is the increase in positive log loss. Negative estimates are retained.")
        download_png("Download comparison PNG (200 dpi)",
                     lambda: viz.matplotlib_importance(results, metric), "importance_comparison.png")
        interpretation(
            "The model stays fixed while selected input values lose their alignment with labels. "
            "The pair shares one shuffle, preserving its internal relationship. Different metrics "
            "can change the importance ranking.",
            "A low individual value does not prove shared information is useless. Redundancy does "
            "not guarantee both individual importances are small. Permutation is neither causality nor retraining ablation.",
            "Record the metric, population, frozen model and perturbation scheme; investigate dependence before deleting features."
        )

if tabs[3].open:
    with tabs[3]:
        mode = st.segmented_control("Response view", ["Replacement calculation", "3D comparison"],
                                    default="Replacement calculation", key="pdp_mode")
        if mode == "Replacement calculation":
            st.subheader("Replace, predict every reference row, then average")
            response = curves(config, feature)
            index = st.slider("Feature grid position", 0, len(response["grid"])-1, 12, key="pdp_grid")
            chart(viz.replacement_scatter(lab, response, feature, index), "replacement_inputs")
            left, right = st.columns(2)
            with left:
                chart(viz.prediction_histogram(response, index), "replacement_histogram")
            with right:
                chart(viz.pdp_plot(response, feature, index), "pdp_curve")
            st.metric("Mean replacement probability", f"{response['pdp'][index]:.4f}")
            st.caption("Input panels use actual feature units, with probability encoded by Viridis "
                       "(0 to 1). Histogram X is predicted probability; Y is count. Orange marks "
                       "the mean, which is one point on the PDP; no labels are averaged.")
            if feature in (0, 1):
                replaced = core.replacement_rows(response["reference"], feature, response["grid"][index])
                chart(viz.support_plot(lab, replaced), "pdp_support")
                st.warning(f"{core.proxy_outside_band(replaced, config[1]).mean():.1%} of these replacements "
                           "lie outside the generator's 95% proxy|signal band. This is not a formal empirical support test.")
        else:
            st.subheader("Compare averaging with fixed nuisance inputs")
            comparison = surface(config, selected)
            kind = st.selectbox("3D surface", ["pdp", "fixed"], format_func=lambda x:
                "Partial dependence: average 60 reference rows" if x == "pdp" else "Model prediction: selected row's proxy/noise fixed",
                key="surface_kind")
            chart(viz.surface_plot(comparison, kind, lab=lab, selected=selected), "response_surface")
            slice_index = st.slider("Context slice position", 0, len(comparison["context"])-1, 8, key="surface_slice")
            chart(viz.surface_slices(comparison, slice_index), "surface_slices")
            st.caption(f"Axes: signal, context, predicted P(y=1). Surface color repeats probability. "
                       f"Fixed slice holds proxy={comparison['row'][1]:.3f}, noise={comparison['row'][3]:.3f}. "
                       "Actual sample dots use their own proxy/noise, so they need not lie on that fixed slice. "
                       "Orange diamond marks the selected actual prediction. These are prediction/PDP surfaces, not SHAP.")
            if st.toggle("Compare weak and strong interaction models", key="interaction_compare"):
                weak_config, strong_config = list(config), list(config)
                weak_config[2], strong_config[2] = 0.0, 2.5
                weak = surface(tuple(weak_config), selected)
                strong = surface(tuple(strong_config), selected)
                left, right = st.columns(2)
                with left:
                    chart(viz.surface_plot(weak, "pdp"), "weak_interaction")
                    st.caption("Generator interaction gamma=0.0; separate fixed forest.")
                with right:
                    chart(viz.surface_plot(strong, "pdp"), "strong_interaction")
                    st.caption("Generator interaction gamma=2.5; separate fixed forest.")
                st.caption("The same latent feature draws and uniform label draws are reused; labels, "
                           "stratified splits and models can change. This does not isolate a single fitted coefficient.")
        st.latex(r"\widehat{PD}_S(z)=\frac{1}{n}\sum_{i=1}^n f(z,x_{i,-S})")
        st.caption("S is the replaced feature set, z its grid value, i a reference row, n the "
                   "reference count, and f is the positive-class probability of the frozen forest.")
        interpretation(
            "The PDP averages replacement predictions over fixed reference rows. A fixed-input "
            "surface evaluates one nuisance-feature setting. A 2D slice reveals their difference.",
            "PDP is generally not E[f(X)|X_S=z] and is not a causal effect. Percentile grids "
            "restrict univariate extremes but cannot guarantee plausible joint inputs.",
            "Inspect observed joint support and reference-population choice alongside every replacement response."
        )

if tabs[4].open:
    with tabs[4]:
        st.subheader("Follow one row, then compare it with the average")
        response = curves(config, feature)
        count = st.slider("Number of thin ICE curves", 1, 80, 25, key="ice_count")
        centered = st.toggle("Center at the first grid value", key="centered")
        chart(viz.ice_plot(response, feature, selected, count, centered), "ice")
        st.dataframe(pd.DataFrame([response["reference"][selected]], columns=core.FEATURES), hide_index=True)
        st.caption(f"Selected row {selected}: only {feature_name} varies. All other displayed feature "
                   "values remain fixed along its curve. Thin-line colors encode context from low "
                   "purple to high yellow; black dashed line is selected; orange is the mean of ALL "
                   f"{len(response['ice'])} reference curves, even when fewer thin curves are shown.")
        st.latex(r"ICE_i(z)=f(z,x_{i,-j}),\quad \widehat{PD}_j(z)=\frac{1}{n}\sum_i ICE_i(z)")
        if centered:
            st.latex(r"cICE_i(z)=ICE_i(z)-ICE_i(z_0)")
        st.caption("i indexes a reference row, j the selected feature, z its grid value, z0 the "
                   "first grid anchor, and f outputs P(y=1). Centering changes vertical offsets, not the model.")
        download_png("Download ICE/PDP PNG (200 dpi)",
                     lambda: viz.matplotlib_ice(response, feature), "ice_pdp.png")
        interpretation(
            "Different curves can reveal response heterogeneity that averaging obscures. "
            "Changing the selected row, line count or centering only changes the display.",
            "Nonparallel probability curves alone do not prove interactions on a log-odds scale: "
            "a nonlinear sigmoid can produce that geometry for additive logits. Replacement may also be unrealistic.",
            "State the output scale and compare subgroup responses in supported feature regions."
        )

if tabs[5].open:
    with tabs[5]:
        view = st.segmented_control("SHAP view", ["First principles", "Waterfall", "Beeswarm",
            "Dependence", "Background sensitivity"], default="First principles", key="shap_view")
        if view == "First principles":
            st.subheader("Two feature orders allocate the product interaction")
            x1 = st.slider("Product x1", -4.0, 4.0, 2.0, 0.5, key="product_x1")
            x2 = st.slider("Product x2", -4.0, 4.0, 3.0, 0.5, key="product_x2")
            game = core.shapley_product(x1, x2)
            st.latex(r"f(x_1,x_2)=x_1x_2,\quad x^{\mathrm{missing}}=(0,0)")
            chart(viz.shapley_animation(game), "coalition")
            st.dataframe(pd.DataFrame({
                "Coalition": ["empty", "x1", "x2", "x1, x2"],
                "Coalition value": [0, 0, 0, game["prediction"]],
            }), hide_index=True)
            st.dataframe(pd.DataFrame({
                "Order": ["x1 then x2", "x2 then x1", "Average (Shapley)"],
                "x1 contribution": [* [p["contributions"][0] for p in game["paths"]], game["values"][0]],
                "x2 contribution": [* [p["contributions"][1] for p in game["paths"]], game["values"][1]],
            }), hide_index=True)
            st.success(f"Verified: baseline 0 + {game['values'][0]:.3f} + {game['values'][1]:.3f} "
                       f"= prediction {game['prediction']:.3f}. Both orders are averaged.")
            st.caption("Colored nodes indicate participating features; faded nodes use zero. "
                       "The diagram's coordinates are layout positions, not input or probability axes. "
                       "Coalition values have product-output units; they are not probabilities.")
        else:
            size, seed = st.session_state.shap_config
            result = shap_values(config, size, seed)
            st.caption(f"Interventional/marginal replacement; P(y=1); {size} training background rows, "
                       f"seed {seed}; {len(result['rows'])} explained held-out rows. Max reconstruction error {result['error']:.2e}.")
            if view == "Waterfall":
                chart(viz.waterfall_plot(result, selected), "waterfall")
                st.caption("X: baseline, allocated feature contributions, final probability. "
                           "Y: probability units; orange increases, blue decreases, gray anchors. "
                           "The display order is bookkeeping, not the Shapley averaging algorithm.")
                download_png("Download waterfall PNG (200 dpi)",
                             lambda: viz.matplotlib_waterfall(result, selected), "shap_waterfall.png")
            elif view == "Beeswarm":
                chart(viz.beeswarm_plot(result), "beeswarm")
                st.caption("Each dot is one attribution. X: probability contribution; Y: feature, "
                           "ranked by mean absolute SHAP on the explained rows. Red is high and blue is low "
                           "actual feature value, normalized separately per feature. Vertical packing "
                           "has no numerical meaning. This is a deterministic beeswarm-style rendering.")
            elif view == "Dependence":
                color_name = st.selectbox("Color dependence points by", core.FEATURES, index=2, key="shap_color")
                chart(viz.dependence_plot(result, feature, core.FEATURES.index(color_name)), "shap_dependence")
                st.caption("X: actual feature value; Y: its SHAP contribution to P(y=1). "
                           f"Viridis encodes actual {color_name}. Color structure suggests heterogeneity, not proof of causality.")
            else:
                if st.session_state.get("other_background_size", 100) not in background_options:
                    st.session_state.other_background_size = background_options[-1]
                other_size = st.select_slider("Comparison background size", background_options,
                                               value=st.session_state.get("other_background_size", 100), key="other_background_size")
                other_seed = st.number_input("Comparison background seed", 0, 10000, 99, key="other_background_seed")
                other = shap_values(config, other_size, int(other_seed))
                st.dataframe(pd.DataFrame({
                    "Quantity": ["Baseline", *core.FEATURES, "Prediction"],
                    f"A: {size} rows, seed {seed}": [result["baseline"][selected], *result["values"][selected], result["predictions"][selected]],
                    f"B: {other_size} rows, seed {other_seed}": [other["baseline"][selected], *other["values"][selected], other["predictions"][selected]],
                }), hide_index=True)
                comparison = pd.DataFrame({
                    "feature": core.FEATURES,
                    "A": result["values"][selected], "B": other["values"][selected],
                })
                import plotly.graph_objects as go
                fig = go.Figure()
                for name, color in (("A", viz.COLORS[0]), ("B", viz.COLORS[1])):
                    fig.add_trace(go.Bar(x=comparison["feature"], y=comparison[name], name=name, marker_color=color))
                fig.update_yaxes(title="Contribution to P(y=1)", zeroline=True)
                chart(viz.style(fig, "Same prediction, two background games"), "background_sensitivity")
                st.caption("Both decompositions reconstruct the same selected prediction. Baseline "
                           "and allocation can differ. Size comparisons with the same seed use nested "
                           "training samples; different seeds change the reference composition.")
            st.latex(r"f(x)=\phi_0+\sum_{j=1}^p\phi_j")
        st.latex(r"\phi_j=\sum_{S\subseteq F\setminus\{j\}}\frac{|S|!(p-|S|-1)!}{p!}\,[v(S\cup\{j\})-v(S)]")
        st.caption("F is the feature set, p its size, j the attributed feature, S a coalition, "
                   "v its value under the specified reference game, phi0 the empty-coalition baseline, "
                   "and phi(j) the order-averaged marginal contribution.")
        interpretation(
            "The product demo gives (3,3) at (2,3) with zero reference. Forest SHAP allocates "
            "the selected probability relative to a sampled training background and verifies the sum.",
            "Mean |SHAP| is not performance importance. Ordinary interventional TreeSHAP is not "
            "a causal estimate or the observational conditional-expectation game. Backgrounds need not agree.",
            "Version the model, class, scale, effective background and dependence convention; "
            "check reconstruction and test reference sensitivity."
        )

if tabs[6].open:
    with tabs[6]:
        st.subheader("When Explainability Misleads")
        pitfall = st.selectbox("Pitfall to investigate", ["Correlated features", "Unrealistic feature combinations",
                                                        "Data leakage (deliberately invalid)"], key="pitfall")
        if pitfall == "Correlated features":
            chart(viz.support_plot(lab), "correlated_support")
            chart(viz.group_plot(importances(config, "ROC-AUC", 8), "ROC-AUC"), "pitfall_group")
            st.caption("Change rho in the training form and apply. The support band and fitted "
                       "model change. Individual ranking can shift when multiple inputs carry similar information.")
            st.latex(r"x_{\mathrm{proxy}}=\rho x_{\mathrm{signal}}+\sqrt{1-\rho^2}\epsilon")
        elif pitfall == "Unrealistic feature combinations":
            response = curves(config, 0)
            index = st.slider("Signal replacement grid position", 0, 24, 12, key="pitfall_grid")
            replaced = core.replacement_rows(response["reference"], 0, response["grid"][index])
            chart(viz.support_plot(lab, replaced), "outside_support")
            st.metric("Replacement rows outside known 95% conditional band",
                      f"{core.proxy_outside_band(replaced, config[1]).mean():.1%}")
            st.latex(r"x_{\mathrm{proxy}}\mid x_{\mathrm{signal}}\sim N(\rho x_{\mathrm{signal}},\,1-\rho^2)")
            st.caption("Blue: observed joint geometry. Shaded band: known generator's 95% "
                       "proxy|signal interval. Green/red crosses: replacements inside/outside. "
                       "This diagnostic uses synthetic knowledge unavailable for most real datasets.")
        else:
            st.error("DELIBERATELY INVALID: post_outcome equals the outcome and is unavailable "
                     "at prediction time. It is included in both training and held-out rows to demonstrate leakage.")
            if st.toggle("Run the invalid leakage comparison", key="run_leakage"):
                invalid = leakage(config)
                with st.container(horizontal=True):
                    st.metric("Leakage-free AUC", f"{invalid['clean_auc']:.4f}", border=True)
                    st.metric("INVALID apparent AUC", f"{invalid['invalid_auc']:.4f}", border=True)
                chart(viz.leakage_plot(invalid), "leakage")
                st.latex(r"x_{\mathrm{post\ outcome}}=y \quad \text{(not available at prediction time)}")
                record = experiment_record(lab, "A post-outcome field can inflate apparent performance and importance.",
                    {"clean_auc": invalid["clean_auc"], "invalid_auc": invalid["invalid_auc"],
                     "invalid_mdi": invalid["invalid_mdi"].tolist()},
                    "High importance identifies reliance on leaked labels; it cannot validate prediction-time availability.",
                    "Deliberate label copying is an extreme synthetic construction, not an estimate of real leakage prevalence.")
                record["configuration"].update(invalid["configuration"])
                st.download_button("Download leakage experiment record", json.dumps(record, indent=2),
                                   "invalid_leakage_record.json", "application/json", on_click="ignore")
        interpretation(
            "Correlated predictors redistribute reliance, feature replacement can leave plausible "
            "joint geometry, and post-outcome information can produce convincing but invalid metrics.",
            "Neither a plausible chart nor an exact attribution sum establishes causal influence, "
            "fairness, safety, feature availability, or a justified decision.",
            "Review feature timing and dependency constraints; audit apparent shortcuts before interpreting performance."
        )

if tabs[7].open:
    with tabs[7]:
        st.subheader("Explain the measurement before explaining the model")
        table = pd.DataFrame([
            ("MDI", "Which splits reduced training impurity?", "Normalized split credit", "Split-search bias"),
            ("Permutation", "What score changes after disruption?", "Chosen score units", "Dependence and metric"),
            ("PDP", "What is the average replacement response?", "Probability", "Joint support"),
            ("ICE", "How does one fixed row respond?", "Probability or centered change", "Output scale and support"),
            ("SHAP", "How is deviation allocated?", "Probability contribution", "Reference coalition game"),
        ], columns=["Method", "Question", "Units here", "Main boundary"])
        st.dataframe(table, hide_index=True)
        st.latex(r"\widehat{PD}_j(z)=\frac1n\sum_i ICE_i(z),\qquad f(x)=\phi_0+\sum_j\phi_j")
        st.caption("These are two different identities: averaging replacement responses, and "
                   "reconstructing an output from attributed deviations.")
        with st.expander("Interview prompts and reference answers"):
            st.markdown(
                "**Why can MDI and permutation rankings disagree?** Split-search credit and "
                "held-out score degradation measure different quantities.\n\n"
                "**Why compare a correlated pair jointly?** It tests group reliance while "
                "retaining within-pair dependence; it is not generally the sum of individual importance.\n\n"
                "**Why does a fixed surface differ from PDP?** One fixes nuisance inputs; the "
                "other averages over reference nuisance values.\n\n"
                "**Does SHAP prove intervention benefit?** No; its allocation is defined by a "
                "reference game, model and output scale.\n\n"
                "**What does reconstruction validate?** Additive accounting on the specified "
                "class and scale, not explanation reliability across populations or backgrounds."
            )
        st.info("First experiments: increase rho; compare individual and joint permutation. "
                "Then switch gamma from 0 to 2.5, inspect ICE and the paired surfaces, and "
                "compare two SHAP backgrounds for the same observation.")
        record = experiment_record(lab,
            "The applied synthetic configuration provides a reproducible baseline for model inspection.",
            {"test_auc": core.metric_score(lab["y_test"], probability, "ROC-AUC"),
             "test_positive_rate": float(lab["y_test"].mean()),
             "observed_signal_proxy_correlation": float(np.corrcoef(lab["X_test"][:, :2].T)[0, 1]),
             "selected_row": selected, "selected_inputs": lab["X_test"][selected].tolist(),
             "selected_probability": float(probability[selected])},
            "Use these measured diagnostics to contextualize the selected explanation; pending author review.",
            "Single synthetic split and seed; no stability, causality or deployment claim.")
        record["configuration"]["model"] = lab["model"].get_params()
        st.download_button("Download current configuration and baseline measurements",
                           json.dumps(record, indent=2), "current_lab_record.json",
                           "application/json", on_click="ignore")
        st.caption("These are suggested experiments until you apply and record them. Interactive "
                   "outputs are computed measurements; interpretation remains for author review.")
        st.code("python export_gifs.py\nstreamlit run app.py", language="bash")
        interpretation(
            "A method, quantity, units and boundary can be paired before interpreting a chart.",
            "Agreement between methods is not mandatory, and disagreement is not automatically a bug.",
            "Keep hypotheses, configurations, actual results, interpretation candidates and limitations "
            "in any experiment record you retain."
        )

# Day 27 Random Forest visual lab

The command-line generator [visualizations.py](visualizations.py) creates 16 views from seeded synthetic data. It needs NumPy, scikit-learn, matplotlib, and Pillow; Plotly is optional for the self-contained 3D HTML view. The shared dependencies are declared in [pyproject.toml](../../pyproject.toml). No notebook, server, network access, or GPU is needed.

From the repository root:

    python 01-classical-machine-learning/27-random-forest/visualizations.py --all
    python 01-classical-machine-learning/27-random-forest/visualizations.py --static
    python 01-classical-machine-learning/27-random-forest/visualizations.py --animations
    python 01-classical-machine-learning/27-random-forest/visualizations.py --interactive
    python 01-classical-machine-learning/27-random-forest/visualizations.py --only 07_correlation_variance
    python -m pytest -q 01-classical-machine-learning/27-random-forest/tests

The default without flags is the complete lab. Every generated view prints its question, a run-specific observation or theoretical reading, and its limitation. Output goes to outputs/visualizations/ inside this topic. **All generated assets are ignored and regenerable local artifacts.** None is linked as a public preview or selected for Git in this phase.

## Visual route

| Output | Question it answers | Evidence type |
|---|---|---|
| 01_tree_instability.png | How do bootstrap changes alter one tree's boundary? | Fitted synthetic models |
| 02_bootstrap_sampling.png | How can N draws produce duplicates and OOB rows? | One seeded bootstrap sample |
| 03_bootstrap_632.png | Why does the unique fraction approach 1 - 1/e? | Simulation plus exact expectation |
| 04_oob_mechanism.png | Which fitted trees may vote for one OOB row? | Membership and fitted votes |
| 05_feature_randomness.png | Where are candidate features sampled? | Explicitly conceptual schematic |
| 06_bagging_vs_random_forest.png | What changes beyond bootstrap bagging? | Fitted models and measured agreement |
| 07_correlation_variance.png | How does tree correlation limit averaging? | Theoretical equal-correlation model |
| 08_forest_growth.gif | How does one forest's aggregate boundary change? | Seven fitted warm-start stages |
| 09_n_estimators.png | How do scores and CPU fit time vary with tree count? | Measured single-split experiment |
| 10_max_features.png | How do tree strength and similarity vary with candidate count? | Measured single-split experiment |
| 11_regression_ensemble.png | What do individual step predictions and their mean look like? | Fitted synthetic regression |
| 12_probability_surface.html | What shape can class probabilities take in two features? | Optional fitted Plotly surface |
| 13_mdi_importance.png | Where did trees allocate training impurity reduction? | Fitted MDI values |
| 14_mdi_vs_permutation.png | How does held-out score degradation differ from MDI? | Fitted MDI and held-out shuffles |
| 15_correlated_features.png | How can a near-duplicate share permutation importance? | Two fitted feature sets |
| 16_tree_vs_forest.png | How does aggregation alter probability regions? | Fitted tree and forest |

The 05 diagram samples fresh candidate lists at three illustrated nodes. Its winners are illustrative choices, not scikit-learn split traces. The 07 curves set individual-tree variance to one and assume equal pairwise correlation. Neither view is an empirical forest benchmark. The other plots compute their displayed values from their fitted models or simulations.

## Executed checks for author review

**Bagging and forest diversity.** Hypothesis: restricting candidate features may reduce similarity among trees. Configuration: 360 synthetic moons rows, a seeded stratified 70/30 split, 80 bagged full-feature trees, and 80 forest trees using one candidate feature per split. Result: mean pairwise agreement on held-out labels was 0.887 for bagging and 0.853 for the forest. Interpretation candidate for author review: this run is consistent with additional split randomness reducing agreement. Limitation: pairwise agreement is affected by this dataset and class mix; no universal ordering follows.

**Tree count.** Hypothesis: adding trees may stabilize scores while increasing fit cost. Configuration: the same moons split; tree counts 1, 2, 5, 10, 20, 50, 100, and 200; forest max_features=1 and min_samples_leaf=2. Result: held-out accuracy was 0.907 at one tree and 0.926 at 200; eligible full-coverage OOB points ranged from 0.917 to 0.925. The chart contains fit times measured on the executing machine. Interpretation candidate for author review: scores were fairly close over the larger counts in this run. Limitation: one split and machine-specific timings do not identify a generally optimal count.

**Candidate-feature count.** Hypothesis: more available features may strengthen individual trees while increasing similarity. Configuration: 500 Gaussian six-feature synthetic rows, a rule using two features, 9% independent label flips, a seeded 70/30 split, and 100 trees for each candidate count 1, sqrt (=2), 3, 4, and all 6. Result: from one to six candidates, mean individual-tree held-out accuracy was 0.681 to 0.757 and pairwise agreement was 0.660 to 0.776. Interpretation candidate for author review: these endpoints show the expected strength/similarity direction in this run. Limitation: forest accuracy and OOB points are dataset-specific and need not change monotonically.

**Importance.** Hypothesis: training impurity credit can differ from held-out score dependence, especially for many-valued noise. Configuration: 600 synthetic rows, two rule features, continuous, binary, and 30-value unrelated noise, 10% label flips, a seeded 70/30 split, 150 trees, and ten held-out permutations. Result: continuous noise had MDI 0.139 and mean permutation accuracy decrease -0.006; held-out accuracy was 0.878. Interpretation candidate for author review: in this run the model allocated training split credit to a noise feature without a positive held-out permutation effect. Limitation: one split and one metric; the result does not quantify general MDI bias or causal relevance.

**Correlated substitute.** Hypothesis: a near-duplicate may reduce the individual permutation importance of the original feature. Configuration: 700 synthetic rows with x2 = x1 plus Gaussian noise of SD 0.04, a third unrelated feature, 10% label flips, the same train/test indices, and 150-tree forests with and without x2. Result: training correlation was 0.999; x1 permutation importance was 0.380 alone and 0.093 beside x2. Interpretation candidate for author review: x2 can substitute for some predictive information from x1 in this example. Limitation: the models use different feature sets, so the bars are not a causal decomposition.

The Monte Carlo view separately measured a mean unique fraction of 0.632 at N=500 across 300 draws, close to the finite-N expectation of 0.632. Its theoretical 1 - 1/e line is labeled as a limit. Other view-specific observations and limitations print during execution.

## Candidate public previews

If the author later selects a small public preview set, the strongest candidates are 02 (bootstrap multiplicity and OOB), 05 (per-node feature randomness), 07 (the correlation/variance equation), and 14 (MDI versus held-out permutation). That later step would require deliberately unignoring the chosen assets, verifying their sizes, reviewing the interpretations, and only then linking them from public documentation. The HTML and the remaining generated outputs can stay local.

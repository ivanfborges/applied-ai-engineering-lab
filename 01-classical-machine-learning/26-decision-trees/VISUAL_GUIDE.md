# Decision Tree visual lab

This lab follows the sequence from class mixture to impurity, greedy split selection, recursive regions, growth, regularization, and ensembles. The generator implements all 20 requested core views. An optional 3D view shows the height of leaf probabilities using genuinely flat rectangles.

## Run locally

Install the repository's shared dependencies from [pyproject.toml](../../pyproject.toml):

    python -m pip install -e ".[dev]"

From the repository root, run all core views:

    python 01-classical-machine-learning/26-decision-trees/visualize_decision_trees.py

An interactive Matplotlib backend shows each figure after saving and closes it when the window is dismissed. For a batch run without windows:

    python 01-classical-machine-learning/26-decision-trees/visualize_decision_trees.py --no-show

Select a few views, list the available names, or include the optional offline HTML:

    python 01-classical-machine-learning/26-decision-trees/visualize_decision_trees.py --view candidate-splits information-gain recursive-splits --no-show
    python 01-classical-machine-learning/26-decision-trees/visualize_decision_trees.py --list
    python 01-classical-machine-learning/26-decision-trees/visualize_decision_trees.py --include-3d --no-show

The default output directory is the topic's `outputs/`, resolved from the script location rather than the current working directory. `--output-dir PATH` overrides it. The console lists files only after successful generation. A full core run is designed to produce 18 PNGs, three GIFs, and `experiment_report.json`; the HTML requires the optional Plotly view.

## Questions and planned filenames

These are generator targets, not a claim that the files have already been generated in the topic directory.

| View name | Visual question | Output filename |
|---|---|---|
| gini | Where is binary class mixture highest? | 01_gini_impurity.png |
| entropy | How does uncertainty behave at pure and balanced nodes? | 02_entropy.png |
| gini-vs-entropy | What shapes and raw scales do the two criteria have? | 03_gini_vs_entropy.png |
| candidate-splits | Which threshold gives the greatest weighted Gini reduction? | 04_candidate_splits.png |
| information-gain | What does greedy maximization look like for each criterion? | 05_information_gain_by_threshold.png |
| depth-boundaries | How do depth caps change axis-aligned regions? | 06_tree_depth_decision_boundaries.png |
| tree-growth | How do geometry, leaf count, and observed scores change across depth caps? | 07_tree_growth.gif |
| tree-structure | Which if/else rules correspond to the geometry? | 08_tree_structure.png |
| recursive-splits | Why does a child split stop at its parent region? | 09_recursive_partitioning.gif |
| depth-generalization | Does additional depth help this validation sample? | 10_depth_vs_generalization.png |
| depth-leaves | How many terminal regions are fitted under each depth cap? | 11_depth_vs_leaves.png |
| min-samples-leaf | How does a minimum leaf count restrict specificity and affect observed scores? | 12_min_samples_leaf.png and 12b_leaf_size_decision_boundaries.png |
| pruning | How do leaves and scores change along a training-only pruning path? | 13_cost_complexity_pruning.png |
| pruning-animation | Which regions disappear as the penalty increases? | 14_pruning.gif |
| probability-regions | Why is a leaf probability constant throughout its region? | 15_probability_regions.png |
| leaf-probabilities | Do small leaves create more extreme probability regions here? | 16_leaf_size_probability_effect.png |
| interaction | How do conditional splits express a feature interaction? | 17_feature_interaction.png |
| instability | How much do fixed-hyperparameter trees differ after bootstrap resampling? | 18_tree_instability.png |
| tree-vs-logistic | How do global linear and conditional axis-aligned boundaries differ? | 19_tree_vs_logistic_regression.png |
| tree-vs-forest | How does averaging tree probabilities change the surface? | 20_tree_vs_random_forest.png |
| probability-3d (optional) | What does the height of each flat probability region mean? | interactive_decision_tree_lab.html |

## What the figures establish

The impurity curves are mathematical illustrations with unnormalized scales. The candidate-split views calculate exact training impurity reductions on eight code-defined rows. Gini reduction and entropy information gain are labeled separately.

The primary data is synthetic `make_moons`: 450 rows, noise 0.28, seed 26, with a fixed stratified 315/135 training/validation split. Fits use training rows only. The pruning path uses training data only. Depth, leaf-size, pruning, and model comparisons display actual computed accuracy and complexity. Validation is repeatedly inspected for education; no independent test score or universal best configuration is claimed.

The growth GIF shows separately fitted depth caps. The recursive GIF reveals the actual splits of one depth-three fitted tree in breadth-first order. Each segment is restricted to the corresponding parent rectangle.

Leaf-probability maps display empirical class fractions, not a calibration assessment. The reported fraction of mesh cells at probability zero or one describes plotted area, not population prevalence. The 3D surface uses independent flat leaf polygons, avoiding interpolation that could suggest sloping probabilities inside leaves.

The clean interaction grid uses the rule `X1 > 0 and X2 > -0.2`. It is a mechanism illustration without held-out evaluation. The instability grid uses six full-size bootstrap resamples at depth cap five. Bootstrapping may replace many rows; this demonstrates sampling sensitivity rather than the effect of a tiny perturbation. Mesh prediction disagreement is not a formal variance estimate.

The logistic model uses two features and a scaler fitted on training data. The forest comparison averages 64 trees and reports computed scores. That picture alone does not establish stability or calibration.

## Evidence and output policy

Each run writes `experiment_report.json` with the seed, dataset configuration, library versions, successful asset filenames, and measured experiments. Each experiment records a hypothesis, configuration, result, interpretation candidate marked for author review, and limitation. Existing topic notes describe a different synthetic experiment; these noisy-moons measurements should be reviewed on their own.

All generated files are **regenerable artifacts** and remain ignored. No public preview is selected or unignored by this implementation, and neither README is updated. Test renderings are temporary validation artifacts.

Start by inspecting candidate-splits, information-gain, depth-boundaries, recursive-splits, depth-generalization, pruning, and probability-regions. After manual review, the split table, depth comparison, observed depth/accuracy chart, and probability map are strong candidates for a small GitHub preview set. The growth GIF is a candidate for a future LinkedIn post because it connects partition complexity to measured scores; the recursive-split GIF is useful for explaining the algorithm itself.

## Validate the implementation

The numerical helper is [decision_tree_visual_math.py](decision_tree_visual_math.py). The executable renderer is [visualize_decision_trees.py](visualize_decision_trees.py).

    python -m pytest -q 01-classical-machine-learning/26-decision-trees/tests
    python scripts/validate_repo.py syntax
    python scripts/validate_repo.py links

The visual tests check impurity endpoints, weighted gains against library stumps, deterministic data, pruning-path independence from validation labels, region geometry, PNG/GIF rendering, offline HTML, report contents, and figure closure. They generate their assets in temporary test directories.

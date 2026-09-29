# Classification Metrics II Visual Lab

The [generator](visualizations.py) creates an offline visual story from synthetic scores to thresholded decisions, ROC/PR operating points, calibration, and operational cost. All charts label their synthetic or mathematical setting. The [tests](tests/test_visualizations.py) cover the reusable calculations and output creation.

## Run

From the repository root, using the shared dependencies in [pyproject.toml](../../pyproject.toml):

```bash
python 01-classical-machine-learning/23-classification-metrics-ii/visualizations.py
python -m pytest -q 01-classical-machine-learning/23-classification-metrics-ii/tests
```

The generator writes to `outputs/` beside the script. Use `--output-dir PATH` to choose another directory, or `--cost-fp`, `--cost-fn`, and `--capacity` to change the illustrative policy assumptions. It needs NumPy, Matplotlib, scikit-learn, Pillow, and Plotly from the repository's shared dependencies. If Plotly is absent, static figures and GIFs still generate and the two HTML views are skipped. No external data, API, notebook, or server is needed.

The default synthetic split has 3,000 training, 1,000 calibration, 1,000 validation, and 1,000 test cases. The model and scaler fit only on training data. Calibration mappings fit only on the calibration split. Curves and threshold policies use validation data. Probability comparisons use the held-out test split. The animated dot plot uses a visible 65-case validation subset; its counts therefore differ from the full-validation curves and policy figures. On fixed scores, lowering a threshold cannot lower recall, but precision can fluctuate rather than decline monotonically.

## Visual questions and output policy

All files are created under `outputs/`. The five **public previews** below are intentionally unignored and referenced from the [topic README](README.md). Other files are **regenerable artifacts** and remain ignored. No intermediate assets are published.

| Output | Question answered | Policy |
|---|---|---|
| `threshold_sweep.gif` | Which individual cases become TP, FP, FN, or TN as one threshold moves? | Public preview |
| `roc_threshold_animation.gif` | Where does the current threshold appear on the full ROC curve? | Regenerable |
| `pr_threshold_animation.gif` | Where does it appear on the PR curve? | Regenerable |
| `roc_vs_pr.png` | How do the same scores and selected threshold read on both curves? | Public preview |
| `prevalence_pr_curves.png` | How does PR behavior change under controlled prevalence weights? | Public preview |
| `precision_vs_prevalence.png` | What does Bayes' precision formula imply at fixed TPR and FPR? | Regenerable |
| `metrics_vs_threshold.png` | How do precision, recall, F1, and specificity vary over a grid? | Public preview |
| `alert_volume_vs_threshold.png` | How many cases cross each threshold? | Regenerable |
| `ranking_vs_calibration.png` | How can ranking stay fixed while probability magnitudes move? | Public preview |
| `calibration_distortion.gif` | How does reliability move as the monotonic exponent changes? | Regenerable |
| `reliability_diagram.png` | How do observed rates compare with predicted probabilities? | Regenerable |
| `probability_distribution.png` | Where do original and transformed probability masses lie? | Regenerable |
| `calibration_methods.png` | How do separately fitted sigmoid and isotonic mappings score on held-out data? | Regenerable |
| `cost_vs_threshold.png` | Which threshold minimizes the illustrated empirical error cost? | Regenerable |
| `threshold_cost_surface.html` | How does the minimum-cost threshold move with the FN/FP cost ratio? | Regenerable, local interactive |
| `decision_cost_threshold.png` | Why does the simple theoretical cutoff depend on relative costs? | Regenerable |
| `capacity_constraint.png` | Which policies fit an illustrative 100-case review capacity? | Regenerable |
| `interactive_threshold_explorer.html` | How do confusion counts and ROC/PR points update together? | Regenerable, local interactive |

The two HTML files are self-contained Plotly pages of about 4.7 MiB each. Open them locally in a browser with JavaScript support. They are useful for study, but too large to embed as routine GitHub previews. The 3D surface reports **cost per 1,000 validation cases**: `(C_FP·FP + C_FN·FN) / N × 1000`, where `N=1000` for this validation split. The surface fixes `C_FP=1` by default and varies `C_FN/C_FP`; the red trace shows the grid minimum at each ratio. The figures use illustrative costs, not measured business loss.

## Executed synthetic experiments

### A. Threshold movement and policy choice

- **Hypothesis:** threshold movement changes counts, operating points, and alert volume without retraining; selected thresholds depend on the stated objective.
- **Configuration:** `make_classification`, 6,000 rows, 12 features (5 informative, 2 redundant), requested 95/5 weights, `class_sep=1.3`, `flip_y=0.01`, seed 23. Stratified split seeds 23–25. Scaled logistic regression fitted on training data. Full validation has 55 positives of 1,000. The plotted policy grid has 101 thresholds from 0 to 1; minimum precision is 0.80, illustrative costs are `C_FP=1, C_FN=9`, and capacity is 100 alerts.
- **Result:** validation ROC-AUC 0.9209, AP 0.8114, Brier 0.0189, log loss 0.0857. Grid-max F1 occurs at threshold 0.30 (precision 0.8077, recall 0.7636, 52 alerts). The maximum-recall threshold satisfying precision 0.80 is also 0.30 on this grid. Minimum illustrative empirical cost occurs at 0.17 (precision 0.6716, recall 0.8182, 67 alerts). The highest-recall capacity-feasible choice occurs at 0.11 (84 alerts).
- **Interpretation candidate — author review required:** different decision objectives can select different cutoffs on the same fixed scores. Coincidence of F1 and precision-constrained choices here is a calculated result, not a design requirement.
- **Limitations:** one synthetic generator, model, split, and grid. Validation selection is noisy. Costs and capacity are illustrative; none of these thresholds is a deployment recommendation.

### B. Controlled prevalence sensitivity

- **Hypothesis:** reweighting identical class-conditional score sets changes precision and AP while leaving pairwise ROC-AUC fixed.
- **Configuration:** 2,500 positive scores from Beta(4,3) and 2,500 negative scores from Beta(2,7), seed 231. Class-specific sample weights represent 50%, 10%, 1%, and 0.1% positive populations without resampling the score sets. A common threshold is the 95th percentile of the negative scores.
- **Result:** ROC-AUC is 0.9371 in all four weighted populations. AP is 0.9392, 0.7299, 0.3872, and 0.2147, respectively. Precision at the common threshold is 0.9332, 0.6080, 0.1236, and 0.0138.
- **Interpretation candidate — author review required:** changing prevalence alone can strongly change positive retrieval quality at an unchanged score threshold and conditional ranking.
- **Limitations:** these are weighted synthetic evaluation populations with exactly fixed conditional scores, not independently sampled deployments. Real conditional score distributions and model calibration can also shift.

### C. Monotonic distortion and held-out calibration

- **Hypothesis:** `p**4` preserves ranking when finite precision does not introduce ties, but changes probability quality.
- **Configuration:** apply the deterministic transform to the same 1,000 held-out test scores from experiment A; evaluate both against identical test labels. Ten quantile reliability bins are used for plotting.
- **Result:** original and transformed ROC-AUC are both 0.8998; AP is 0.7557 for both. Brier score changes from 0.0218 to 0.0308; log loss changes from 0.1016 to 0.3044. The script checks exact score ordering for this run.
- **Interpretation candidate — author review required:** the observed unchanged discrimination and worse probability error illustrate distinct evaluation questions.
- **Limitations:** the baseline classifier is not certified perfectly calibrated. Bins have noisy event rates, and the transform is deliberately artificial.

### D. Fitted calibration mappings

- **Hypothesis:** calibration mappings can change probability scores and may improve one held-out probability metric without uniformly improving every metric.
- **Configuration:** fit sigmoid logistic mapping on model logit scores and isotonic regression on model probabilities using only the 1,000-case calibration split. Evaluate original, sigmoid, and isotonic scores on the same 1,000-case test split.
- **Result:** original / sigmoid / isotonic Brier scores are 0.02176 / 0.02151 / 0.02064. Log losses are 0.10162 / 0.10277 / 0.18711. ROC-AUC values are 0.89983 / 0.89983 / 0.89887; AP values are 0.75573 / 0.75573 / 0.71571.
- **Interpretation candidate — author review required:** isotonic improves Brier in this split but worsens log loss and introduces enough ties to reduce ranking summaries. Calibration is not a guarantee of universal metric improvement.
- **Limitations:** small rare-positive calibration and test splits, one model, one split, no uncertainty intervals. A production calibration choice would need a broader evaluation design.

## Preview selection

The five referenced previews total about 1.0 MiB: the threshold GIF, ROC/PR comparison, controlled prevalence curves, threshold landscape, and ranking/calibration comparison. The ROC and PR threshold GIFs repeat much of the selected threshold story and are kept for local study. The separate probability histogram and alert-volume plot are also useful locally but less informative alone than the selected panels. Both HTML files remain ignored because each is roughly 4.7 MiB.

The observed interpretation candidates above await author review. Synthetic patterns do not establish production performance, fairness, calibration under shift, or a business threshold.

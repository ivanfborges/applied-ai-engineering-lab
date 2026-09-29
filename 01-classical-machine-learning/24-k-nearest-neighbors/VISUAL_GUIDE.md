# Day 24 visual lab

Run from the repository root or from this directory:

```bash
python 01-classical-machine-learning/24-k-nearest-neighbors/visualizations.py
python 01-classical-machine-learning/24-k-nearest-neighbors/interactive_3d.py
```

Both scripts use only synthetic or code-defined data. The nine PNG/GIF outputs are generated under `assets/`; the Plotly script writes a standalone `assets/knn_3d.html` that opens locally. The three [README previews](README.md#visual-lab) are public Git candidates. Other generated files, including the 4.9 MB HTML, stay ignored and regenerate on demand.

| Output | Question it answers |
|---|---|
| `knn_neighbors.png` | Which five stored rows are closest to a separate query? |
| `distance_metrics.png` | Why do Euclidean circles and Manhattan diamonds define different neighborhoods? |
| `scaling_neighbors.png` | Does standardization change the actual neighbor indices? |
| `k_decision_boundary.gif` | How do local decision regions change as `k` moves from 1 to 50? |
| `weighted_knn.png` | How can close rows dominate a distance-weighted vote? |
| `curse_dimensionality.png` | How large must an idealized axis-aligned neighborhood become to cover a fixed volume fraction? |
| `distance_concentration.png` | How does relative distance spread change in one seeded Gaussian simulation? |
| `irrelevant_features.png` | What happens to held-out KNN accuracy when independent noise features are added? |
| `vector_retrieval.png` | What is shared by neighbor classification and vector retrieval, and what happens afterward? |
| `knn_3d.html` | What does an exact neighborhood look like with three actual informative features? |

## Executed observations

**Scaling neighborhood.** Hypothesis: the income coordinate's larger numerical units will change raw Euclidean neighbor selection. Configuration: 70 synthetic reference rows with age in 22-68 years and income in USD 30,000-150,000; separate query `[44, 81000]`; `k=5`; `StandardScaler` fitted only on reference rows. Result: raw indices `[48, 53, 1, 11, 22]`; standardized indices `[1, 40, 26, 56, 37]`. Interpretation candidate **for author review**: scale changes which examples contribute to a prediction. Limitation: these coordinates are deliberately constructed and the plot measures neighborhood identity, not predictive quality.

**Voting and changing `k`.** Hypothesis: weighting nearby examples more strongly can change a vote, while increasing `k` changes the locality of the decision surface. Configuration: a five-row handcrafted synthetic vote with `k=5`; separately, 180 synthetic moon observations with noise 0.24 and `k` in `[1, 3, 5, 9, 15, 25, 50]`. Result: uniform voting predicted class 0; inverse-distance voting predicted class 1. The GIF contains seven rendered decision surfaces. Interpretation candidates **for author review**: close minority rows can overcome a farther majority; larger `k` appears to smooth local regions in this training-set view. Limitations: the vote is intentionally constructed, and the animation reports no validation accuracy or optimal `k`.

**Distance concentration.** Hypothesis: distances to independent Gaussian references become less distinguishable relative to their mean as feature count grows. Configuration: one Gaussian query and 1,000 independent Gaussian references for each dimension in `[2, 5, 10, 20, 50, 100, 250, 500, 1000]`, seed 42; spread = `(maximum - minimum) / mean`. Result: relative spread was 2.659 at dimension 2, 0.597 at 50, 0.199 at 500, and 0.118 at 1,000. The script prints all dimensions and min/mean/max/SD values. Interpretation candidate **for author review**: this seeded Gaussian configuration shows decreasing relative spread. Limitation: one query per dimension, finite sample extremes, and an isotropic Gaussian are not a general performance guarantee.

**Irrelevant dimensions.** Hypothesis: independent noise coordinates can degrade distance-based classification. Configuration: 500 synthetic moon rows with noise 0.18; one stratified 375/125 train/test split, seed 42; 0-250 appended Gaussian noise features from seed 43; fixed `k=9`; training-fitted `StandardScaler` in a pipeline. Result: held-out accuracy was 0.992 with 0 noise features and 0.656 with 250; it rose from 0.728 at 50 to 0.736 at 100. Interpretation candidate **for author review**: more irrelevant coordinates coincided with lower accuracy overall in this constructed split, without a monotonic guarantee. Limitation: one split, fixed model settings, no uncertainty interval, and deliberately independent noise. Test outcomes here describe the plotted configurations; selecting a production feature set requires development-only selection and fresh test evaluation.

## Boundaries

The volume curve uses `side length = fraction^(1/d)` for an axis-aligned subcube inside a unit cube. It is a geometric approximation, not a Euclidean radius or an empirical KNN score. The 3D scene uses exactly three generated informative coordinates; it does not project the 50-1,000-dimensional simulations. Mock document vectors have no learned semantics, and retrieval stops before any downstream model. The visual lab makes no latency or production retrieval claim.

Every output can be regenerated without network access. Only `scaling_neighbors.png`, `k_decision_boundary.gif`, and `distance_concentration.png` are intentionally public previews. Review the interpretation candidates before treating them as the author's conclusions.

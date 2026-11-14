# Unsupervised Learning and Recommender Systems

[Repository overview](../README.md) · [Full roadmap](../ROADMAP.md)

Module 2 spans global Days 36–44. **9 of 9 studies are implemented**: Day 36
connects the K-Means objective to representation choices, cluster-count
diagnostics and tested Lloyd updates. Day 37 adds density connectivity,
core/border/noise semantics and DBSCAN parameter sensitivity. Day 38 adds
agglomerative linkage criteria, dendrogram structure and nested cuts with tied
height boundaries. Day 39 adds PCA projection, variance and reconstruction,
a fold-local Wine classification comparison and a synthetic low-variance-signal
counterexample. Day 40 adds t-SNE/UMAP neighborhood fidelity, parameter and seed
comparisons on public digits, and a tested fixed-bandwidth KL objective/gradient.
Its ten-module offline visual lab adds real 3D fits, adaptive affinities, saved
optimization steps, fuzzy graphs and measured interpretation counterexamples.
Day 41 adds normal-only anomaly detection with MAD, Isolation Forest and
One-Class SVM, independent calibration, frozen-score policy comparisons and a
tested educational isolation forest. Day 42 adds explicit-rating collaborative
filtering, overlap-aware similarity, cold-start fallback policies and a synthetic
temporal rating-error comparison by history segment. Its ten-scene offline visual
lab adds recommendation arithmetic, masked 3D factorization, onboarding,
ranking evaluation and exposure-loop simulations. Day 43 adds SVD/BPR/content
and popularity comparisons, tested pairwise gradients, validation-selected
fusion, candidate-recall boundaries and metadata-only item admission on
globally time-split synthetic interactions. Day 44 adds tested Precision@K,
Recall@K, AP/MAP, RR/MRR and graded NDCG over explicit item IDs, multiple
relevant items per query, scikit-learn agreement and synthetic order/cutoff
and candidate-universe diagnostics with declared evaluation policies.
Its [eight-section Streamlit lab](44-ranking-metrics/ranking_metrics_visual_lab/README.md)
adds editable rankings, AP/RR/NDCG animations, a discrete 3D landscape, four GIF
exporters and a synthetic RAG reranking/context-budget simulation. Two teaching
GIFs are selected previews; visual interpretations remain pending author review.

Day 43's [interactive visual lab](43-recommender-systems-ii/visualizations/README.md)
makes factorization, embedding training, BPR steps, hybrid contributions and
candidate budgets inspectable in ten scenes, with separate cold-start and
exposure simulations. Its three-factor visual run is recorded separately from
the topic's historical twelve-factor study; interpretation review remains pending.

| Day | Implemented study | Evidence |
|---:|---|---|
| 36 | [K-Means clustering](36-k-means-clustering/) | Tested NumPy Lloyd loop, scaling/K diagnostics, frozen inference and seven-scene offline visual lab with exact-state GIF and 3D slider |
| 37 | [DBSCAN and density-based clustering](37-dbscan-density-based-clustering/) | Tested exact expansion and ambiguous borders, density sweeps, k-distance and K-Means comparison; 11-scene visual lab with offline parameter controls and 3D geometry |
| 38 | [Hierarchical clustering](38-hierarchical-clustering/) | Tested NumPy average linkage, full-tree conversion, Ward height units and tie-aware cuts; four-linkage synthetic comparison, nested K sweep and ten-scene visual lab with exact merge/dendrogram GIFs and offline 2D/3D explorers |
| 39 | [PCA in practice](39-pca-in-practice/) | Tested covariance PCA, full-SVD subspace agreement, reconstruction identities and fold-local fitting; Wine/synthetic comparisons plus eight-demo visual lab with basis/projection GIFs and offline k-selectable 3D reconstruction |
| 40 | [t-SNE and UMAP](40-t-sne-umap/) | Public digits fidelity and tested KL gradient; ten-module offline visual lab with Swiss Roll, real 3D, adaptive affinities, saved optimization, fuzzy graphs, controlled sweeps and interpretation traps |
| 41 | [Anomaly detection](41-anomaly-detection/) | Synthetic MAD/Isolation Forest/One-Class SVM comparison, held-out normal calibration, threshold-policy metrics and tested leaf correction/robust statistics |
| 42 | [Recommender systems I](42-recommender-systems-i/) | Tested overlap Pearson, cold-start fallbacks and temporal holdout; ten-scene offline lab with neighbor arithmetic, masked 3D factors, onboarding, ranking and exposure loops |
| 43 | [Recommender systems II](43-recommender-systems-ii/) | Tested BPR gradients/SGD, synthetic global-round SVD/BPR/content/popularity comparison, validation-selected fusion, candidate budgets and metadata-only item admission |
| 44 | [Ranking metrics](44-ranking-metrics/) | Tested ID-based metrics and macro aggregation; eight-section Streamlit lab, AP/RR/NDCG animations, exact 3D permutations, synthetic RAG/context budgets, four GIF exporters and two selected previews |

Implementation and automated validation are separate from author review of
interpretations. The dated experiment records in Days 36–44 contain measured
observations and explicitly pending interpretation candidates. A scoped
AI-assistant technical review of four items in Days 39/43 is identified in the
[curation record](../docs/CURATION.md); other candidates remain pending. These establish
neither real-world segmentation or incident-detection quality nor recommendation or retrieval gains.
See the [publication methodology](../docs/methodology.md).

Follow the topic's commands to run its examples and focused tests. Repository
checks and their limits are described in the [validation contract](../docs/validation.md).
The [visual guide](36-k-means-clustering/VISUAL_LAB.md) documents three selected
public previews; the bundled HTML explorer and other exports regenerate locally
and stay ignored. Visual interpretation candidates remain pending author review.

The [DBSCAN visual guide](37-dbscan-density-based-clustering/VISUAL_LAB.md)
documents exact neighborhood/expansion GIFs, density and scaling controls,
standalone offline parameter/3D explorers and three inspected preview candidates.
Its visual interpretation candidates also remain pending author review.

Continue with [Module 3](../03-statistics-experimentation/), which contains all
eight implemented studies from A/B testing through uplift modeling (Days 45–52).
No placeholder topic folders are published.

# Unsupervised Learning and Recommender Systems

[Repository overview](../README.md) · [Full roadmap](../ROADMAP.md)

Module 2 spans global Days 36–44. **2 of 9 studies are implemented**: Day 36
connects the K-Means objective to representation choices, cluster-count
diagnostics and tested Lloyd updates. Day 37 adds density connectivity,
core/border/noise semantics and DBSCAN parameter sensitivity. The other seven
topics remain planned.

| Day | Implemented study | Evidence |
|---:|---|---|
| 36 | [K-Means clustering](36-k-means-clustering/) | Tested NumPy Lloyd loop, scaling/K diagnostics, frozen inference and seven-scene offline visual lab with exact-state GIF and 3D slider |
| 37 | [DBSCAN and density-based clustering](37-dbscan-density-based-clustering/) | Tested exact expansion and ambiguous borders, density sweeps, k-distance and K-Means comparison; 11-scene visual lab with offline parameter controls and 3D geometry |

Implementation and automated validation are separate from author review of
interpretations. The dated experiment records in Days 36–37 contain measured
observations and explicitly pending interpretation candidates. These establish
neither real-world segmentation quality nor recommendation or retrieval gains.
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

Future work follows the roadmap: hierarchical clustering,
PCA, t-SNE/UMAP, anomaly detection, recommender systems and ranking metrics.
No placeholder topic folders are published.

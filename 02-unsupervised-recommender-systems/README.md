# Unsupervised Learning and Recommender Systems

[Repository overview](../README.md) · [Full roadmap](../ROADMAP.md)

Module 2 spans global Days 36–44. **1 of 9 studies is implemented**: Day 36
connects the K-Means objective to representation choices, cluster-count
diagnostics and tested Lloyd updates. The other eight topics remain planned.

| Day | Implemented study | Evidence |
|---:|---|---|
| 36 | [K-Means clustering](36-k-means-clustering/) | Tested NumPy Lloyd loop, scaling/K diagnostics, frozen inference and seven-scene offline visual lab with exact-state GIF and 3D slider |

Implementation and automated validation are separate from author review of
interpretations. The dated experiment record in Day 36 contains measured
observations and explicitly pending interpretation candidates. It establishes
neither real-world segmentation quality nor recommendation or retrieval gains.
See the [publication methodology](../docs/methodology.md).

Follow the topic's commands to run its examples and focused tests. Repository
checks and their limits are described in the [validation contract](../docs/validation.md).
The [visual guide](36-k-means-clustering/VISUAL_LAB.md) documents three selected
public previews; the bundled HTML explorer and other exports regenerate locally
and stay ignored. Visual interpretation candidates remain pending author review.

Future work follows the roadmap: density-based and hierarchical clustering,
PCA, t-SNE/UMAP, anomaly detection, recommender systems and ranking metrics.
No placeholder topic folders are published.

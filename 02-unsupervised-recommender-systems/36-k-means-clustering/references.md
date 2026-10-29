# References

Official scikit-learn pages checked on October 29, 2026. The stable pages
identified version 1.9.1; the recorded run also used 1.9.1. Stable URLs can
change with later releases; measured results retain their environment in
[the experiment record](README.md#executed-study-record--october-29-2026).

- [Clustering user guide: K-Means](https://scikit-learn.org/stable/modules/clustering.html#k-means)
  — objective, Lloyd iteration, initialization and computational considerations.
- [KMeans API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
  — `n_init`, K-Means++, tolerance, fitted centers and prediction behavior.
- [StandardScaler API](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
  — fitted statistics, population variance and outlier sensitivity.
- [Silhouette score API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
  — definition, Euclidean distance default and valid occupied-cluster counts.
- [Silhouette analysis example](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)
  — use individual coefficients, sizes and separation when assessing K.
- [Demonstration of K-Means assumptions](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html)
  — shape, variance and size failure cases. Further reading, not an experiment
  executed in this topic.
- [make_blobs API](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html)
  — source of the synthetic generator. The example specifies all centers,
  dispersions, sample count, random seed and the additional scale distortion
  directly in code; no external dataset is downloaded.

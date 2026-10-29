# References

- Ester, Kriegel, Sander and Xu (1996),
  [A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise](https://cdn.aaai.org/KDD/1996/KDD96-037.pdf).
  Original neighborhood, reachability and connectivity definitions. Its
  historical evaluation is not a benchmark run by this repository.
- [scikit-learn DBSCAN API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html):
  self-inclusive `min_samples`, metric options, sample weights, core indices,
  noise labels and memory caveats.
- [scikit-learn clustering guide: DBSCAN](https://scikit-learn.org/stable/modules/clustering.html#dbscan):
  density connectivity, input-order effects, implementation costs and related
  OPTICS/HDBSCAN approaches.
- [NearestNeighbors API](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html):
  `kneighbors(X)` versus a query without `X`; neighborhood boundary semantics.
- [Adjusted Rand index API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html):
  a partition comparison independent of arbitrary numeric label IDs.
- [Haversine distances API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.haversine_distances.html):
  geographic coordinate order and radians for spherical angular distance.
- [Matplotlib PillowWriter](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.PillowWriter.html):
  GIF export without FFmpeg for the visual companion.
- [Plotly interactive HTML export](https://plotly.com/python/interactive-html-export/)
  and [slider controls](https://plotly.com/python/sliders/): local interactive
  figures with JavaScript bundled for offline use.

The paper and DBSCAN/clustering/neighbor documentation were consulted during
curation on 2026-10-29. Stable documentation URLs can change; measured results
in [notes.md](notes.md) retain the versions actually used.

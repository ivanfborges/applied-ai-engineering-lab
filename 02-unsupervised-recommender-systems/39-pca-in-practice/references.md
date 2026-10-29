# References

Primary sources checked on 2026-10-29. Stable documentation can change; the
installed versions used for measurements are recorded in [notes.md](notes.md).

- [scikit-learn PCA API](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html):
  centering, component layout, full SVD, strict float-threshold rule, sample
  variance and covariance-solver numerical limitations.
- [Importance of feature scaling](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html):
  Wine as an illustration of how units change principal directions.
- [scikit-learn load_wine](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html):
  locally bundled dataset dimensions and loading API.
- [UCI Wine](https://archive.ics.uci.edu/dataset/109/wine): Aeberhard, S. and
  Forina, M. (1992), *Wine* [Dataset], UCI Machine Learning Repository,
  DOI [10.24432/C5PC7J](https://doi.org/10.24432/C5PC7J), CC BY 4.0.
  Chemical measurements of three cultivars; no dataset files are redistributed here.

The scripts and prose are a topic-specific implementation and synthesis. Wine
results concern this dataset and evaluation design; synthetic results concern
the explicitly documented generators.

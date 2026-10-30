# References and data provenance

## Dataset

- [scikit-learn `load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html):
  the bundled 1,797-row, 64-feature subset used in this study. No remote fetch,
  dataset copy or MNIST dependency is required. Values are counts of active
  pixels in blocks of preprocessed handwritten images, rather than arbitrary
  continuous sensor features.
- Alpaydin, E. and Kaynak, C. (1998), [Optical Recognition of Handwritten Digits,
  UCI, DOI 10.24432/C50P49](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits).
  UCI lists the full 5,620-row collection under CC BY 4.0; the bundled subset
  must not be described as the full collection. The original collection uses
  distinct writers for its training and test partitions; this exploratory
  sample is not a writer-held-out classifier evaluation.

## Objectives and implementation

- van der Maaten, L. and Hinton, G. (2008), [Visualizing Data using t-SNE](https://jmlr.org/papers/v9/vandermaaten08a.html).
  Original similarity objective, heavy-tailed embedding and gradient.
- McInnes, L., Healy, J. and Melville, J. (2018/2020), [UMAP: Uniform Manifold
  Approximation and Projection for Dimension Reduction](https://arxiv.org/abs/1802.03426).
  Manifold assumptions, fuzzy graph construction and embedding formulation.
- [How UMAP works](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html)
  and [UMAP parameters](https://umap-learn.readthedocs.io/en/latest/parameters.html).
  Read the graph construction and practical optimizer separately from a dense
  conceptual objective; negative sampling is an implementation choice.
- [UMAP reproducibility](https://umap-learn.readthedocs.io/en/latest/reproducibility.html)
  and [transforming new data](https://umap-learn.readthedocs.io/en/latest/transform.html).
  Seeds, multithreading trade-offs and approximate inference on new observations.
- [scikit-learn `TSNE`](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html).
  Perplexity constraints and iteration configuration. `max_iter` replaced
  `n_iter` in 1.5; the example selects the supported name to retain compatibility
  with the repository's scikit-learn 1.4 lower bound.
- [scikit-learn `trustworthiness`](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.trustworthiness.html).
  Rank-weighted penalty for introduced neighbors and the k < n/2 constraint.

Official pages were consulted during implementation on 2026-10-30. Their live
versions are not an environment lock; executed package versions are recorded
separately in [the experiment notes](notes.md#executed-experiments--2026-10-30).

## Offline visual laboratory

- [UMAP API guide](https://umap-learn.readthedocs.io/en/latest/api.html):
  exact supplied-neighbor inputs, graph controls and local connectivity. The
  visual graph audit also checks the installed implementation; private hooks
  and weak-edge scheduling are version-sensitive.
- [Plotly dropdown controls](https://plotly.com/python/dropdowns/) and
  [interactive HTML export](https://plotly.com/python/interactive-html-export/):
  selecting genuinely computed traces with `update` and loading a local JS
  bundle for offline execution.

The [visual guide](VISUAL_GUIDE.md) distinguishes analytical kernel illustrations,
actual optimizer states, independent parameter fits and final UMAP embeddings.

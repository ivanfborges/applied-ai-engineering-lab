# Primary references

- [Goodfellow, Bengio and Courville — Deep Learning, chapter 14](https://www.deeplearningbook.org/contents/autoencoders.html): undercomplete, regularized and denoising autoencoders, representation capacity and limitations.
- [Baldi and Hornik (1989), Neural networks and principal component analysis](https://doi.org/10.1016/0893-6080(89)90014-2): linear networks, squared reconstruction error and principal subspaces.
- [PyTorch MSELoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html): mean reduction across elements; explains the $1/(nd)$ normalization used here.
- [scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html): centered linear decomposition and inverse transformation; scaling is a separate preprocessing step.
- [scikit-learn IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html): scoring direction and fitted isolation baseline.
- [NumPy quantile](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html): explicit `higher` quantile convention used for normal-only calibration.

All study observations are generated in code. No external dataset is downloaded
or redistributed. Library versions used for executed results are recorded in
[the dated study](notes.md#executed-study--2026-11-19); dependency ranges remain
centralized in [pyproject.toml](../../pyproject.toml).

# References

- [scikit-learn: permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html).
  Score-decrease definition, tree importance bias and correlated-feature limitations.
- [scikit-learn: partial dependence and ICE](https://scikit-learn.org/stable/modules/partial_dependence.html).
  Marginal replacement definitions and brute versus recursion methods.
- [SHAP TreeExplainer API](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html).
  Explicit feature perturbation, probability output, background and multi-output shapes.
- [Lundberg and Lee (2017), A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874).
  Additive explanations and Shapley allocation.
- [Goldstein et al., Peeking Inside the Black Box](https://arxiv.org/abs/1309.6392).
  Individual conditional expectation and heterogeneous response inspection.

These primary sources were consulted for technical review. Documentation URLs
follow current releases; the executed versions are recorded in
[notes.md](notes.md#executed-forest-experiment). Data in this topic are generated
locally from the stated equations; no public dataset is used.
## Visual implementation references

- [Plotly animations](https://plotly.com/python/animations/): graph-object frames, controls and fixed axis ranges.
- [Matplotlib PillowWriter](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.PillowWriter.html): local GIF writing without ffmpeg.
- [SHAP Independent masker](https://shap.readthedocs.io/en/latest/generated/shap.maskers.Independent.html): explicit effective background size.
- [Streamlit caching](https://docs.streamlit.io/develop/concepts/architecture/caching): model and numerical-result caching; exact APIs were checked against the installed 1.60 documentation.

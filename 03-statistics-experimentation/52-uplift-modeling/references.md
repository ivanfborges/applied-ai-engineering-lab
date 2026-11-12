# References

- Künzel, S. R., Sekhon, J. S., Bickel, P. J., and Yu, B. (2019).
  [Metalearners for estimating heterogeneous treatment effects using machine learning](https://doi.org/10.1073/pnas.1804597116).
  Primary source for S-, T- and X-Learners and their structural trade-offs.
- Hernán, M. A., and Robins, J. M. (2020).
  [Causal Inference: What If](https://miguelhernan.org/whatifbook).
  Potential outcomes, consistency, exchangeability, positivity and identification.
- Athey, S., and Wager, S. (2021).
  [Policy Learning with Observational Data](https://doi.org/10.3982/ECTA15732).
  Policy objectives, observational identification and doubly robust evaluation.
  This educational example does not reproduce that paper's policy learner.
- scikit-learn:
  [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
  Class-probability columns follow `classes_`; random states and explicit forest
  parameters control this study's fits. Installed versions belong in run records.

The gain and centered-area conventions are defined explicitly in
[notes.md](notes.md), rather than asserted to match a library's Qini coefficient.
No external dataset or illustrative image is used.

## Visual laboratory

- Newcombe, R. G. (1998).
  [Interval estimation for the difference between independent proportions: comparison of eleven methods](https://pubmed.ncbi.nlm.nih.gov/9595617/).
  Source for combining independent Wilson intervals into an unpooled
  difference-of-proportions interval. Module 5 uses the approximate pointwise
  score method without continuity correction.
- Plotly: [interactive HTML export](https://plotly.com/python/interactive-html-export/)
  and [JavaScript configuration](https://plotly.com/javascript/configuration-options/).
  The generator embeds the installed Plotly bundle locally and uses browser
  controls without a CDN.
- Matplotlib: [PillowWriter](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.PillowWriter.html).
  GIF export uses Pillow rather than an external video encoder.

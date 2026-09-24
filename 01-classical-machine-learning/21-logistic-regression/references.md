# References

These primary documentation pages were checked during implementation.
The recorded execution used scikit-learn 1.9.0 and NumPy 2.4.6; online stable
documentation may describe a newer release.

- [scikit-learn: logistic regression objective](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)  
  Binary and multinomial models, penalty normalization, and solver distinctions.
  Used to check the C=1/(n*lambda) mapping for the unweighted L2 comparison.
- [scikit-learn: LogisticRegression API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)  
  Parameters, class ordering, probability outputs, and solver compatibility.
  The example uses lbfgs with default L2 behavior and avoids the deprecated
  penalty parameter.
- [NumPy: logaddexp](https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html)  
  Stable computation of the log of a sum of exponentials; used to evaluate
  binary cross-entropy directly from signed logits.
- [scikit-learn: decision threshold tuning](https://scikit-learn.org/stable/modules/classification_threshold.html)  
  Distinguishes probability estimation from decisions and explains validation
  requirements for threshold selection.

- [scikit-learn: calibration_curve](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html)  
  Reliability-bin definitions, empty-bin handling, and uniform binning.
- [Plotly: write_html](https://plotly.com/python-api-reference/generated/plotly.io.write_html.html)  
  Offline HTML export with a shared local JavaScript bundle.

The core examples use make_synthetic_data in [from_scratch.py](from_scratch.py).
The visual lab additionally uses scikit-learn's synthetic make_classification
and make_moons generators plus a small Bernoulli simulation, all configured in
[visual_experiments.py](visual_experiments.py). No external dataset is used.
# Explainability interview questions

1. **What does high forest impurity importance establish?**
   It reports split impurity reduction in fitted trees. It can reflect useful
   signal, selection bias, or training overfit. Validate reliance on held-out
   data before treating it as predictive evidence.

2. **Why can individual permutation disagree with grouped permutation?**
   A remaining correlated input may preserve predictive information after one
   feature is shuffled. A shared shuffle of a group tests reliance on the
   group while retaining internal relationships. The example's joint signal
   pair decrease is larger than either individual decrease; this is not
   guaranteed for every fitted model.

3. **Is permutation importance the effect of deleting a feature?**
   It measures a frozen model under a specific disruption. Deletion followed
   by retraining allows alternative predictors and answers a different question.

4. **How do you interpret a negative permutation estimate?**
   The shuffled score exceeded the baseline. Preserve the sign and investigate
   sampling variation or harmful reliance. Do not automatically replace it
   with zero or claim the feature causes harm.

5. **When is PDP the average of ICE?**
   When the output, reference observations, grid and weighting match. The code
   checks this relationship and library agreement directly. A displayed ICE
   subsample need not average to a PDP computed on more rows.

6. **Why can a percentile grid still produce unrealistic observations?**
   Univariate plausibility does not guarantee joint plausibility. Replacing
   signal while keeping its proxy fixed breaks their relationship.

7. **Do nonparallel probability ICE curves prove interactions?**
   No. A nonlinear link can produce that geometry even for additive logits.
   Specify the prediction scale and inspect relevant support.

8. **What must accompany a local SHAP value?**
   Model version, class, output units, background, masking/dependence convention
   and reconstruction check. In probability units +0.20 is 20 percentage points
   of allocated deviation from the baseline; it is not an intervention estimate.

9. **Why do SHAP magnitude and permutation rankings differ?**
   The former allocates prediction deviations; the latter measures score
   degradation after input disruption. They depend on different reference
   choices and cannot be interpreted as the same quantity.

10. **What would you check before using explanations operationally?**
    Evaluation validity and feature timing, plausible perturbations, class and
    scale consistency, stability across backgrounds/cohorts/retraining,
    computational cost, and domain usefulness. Additive fidelity does not
    prove that a decision policy is appropriate.

A concise interview answer: choose an explanation method for a defined question,
validate the prediction model first, state the inspection population and output,
and examine dependency violations. Use global reliance and response diagnostics
alongside local attribution; keep causal and policy claims separate from those
measurements.
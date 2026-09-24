# Logistic regression interview questions

1. **What is being regressed?**  
   The conditional log-odds are affine in the supplied features. A sigmoid
   produces a positive-class probability, and a separate rule produces a label.

2. **Why sigmoid and cross-entropy?**  
   Sigmoid inverts the logit link. For independent Bernoulli observations,
   negative average log-likelihood is binary cross-entropy. A different link
   is possible; sigmoid is a modeling choice rather than the only function
   mapping real scores to probabilities.

3. **What does a coefficient of log(2) imply?**  
   A one-unit increase doubles conditional odds when other supplied predictors
   are fixed. Starting at p=0.2, the new probability is 1/3. State feature
   units, transformations, and interactions before interpreting the number.

4. **Derive the gradient and explain the test.**  
   The score derivative of BCE is p-y. Therefore the mean weight gradient is
   X.T@(p-y)/n, plus lambda*w for L2; the intercept gradient is mean(p-y).
   Central finite differences in the tests independently check both.

5. **Does a convex loss guarantee a finite, unique fit?**  
   No. Redundant predictors undermine uniqueness, and separation can prevent
   a finite unpenalized maximum-likelihood estimate. Regularization changes
   the problem. A small gradient does not by itself diagnose separation.

6. **What does changing the threshold do geometrically?**  
   The boundary becomes w.T@x+b=logit(t). For a nonzero fixed weight vector,
   this shifts a parallel hyperplane. The probabilities remain unchanged.
   Nonlinear feature transformations change the geometry in raw input space.

7. **Why is the scikit-learn comparison easy to get wrong?**  
   Penalty normalizations, intercept treatment, preprocessing, sample weights,
   and convergence tolerances must agree. Here mean BCE plus lambda/2 times
   squared weight norm matches unweighted L2 with C=1/(n*lambda).

8. **Why compute loss from logits?**  
   A sigmoid may round to zero or one, making direct logarithms unstable.
   Signed-margin logaddexp evaluates binary loss without clipping scores.
   The extreme-logit tests also ensure confident errors retain large losses.

9. **Is 100% precision at t=0.8 enough to adopt that threshold?**  
   No. In the recorded split it covers only 22 predictions and misses 109
   positives. A policy needs declared costs and capacity, validation selection,
   and a final held-out assessment. The example selected no operational policy.

10. **Are probability outputs automatically calibrated?**  
    No. Misspecification, penalization, weighting, sampling, and shift can
    affect calibration. A lower log loss on one sample is useful evidence
    about aggregate probability quality, not proof of subgroup calibration.

11. **Where does this fit in an AI system?**  
    It can classify frozen embeddings or score pre-decision routing signals.
    Validate whether labels capture the intended outcome and whether features
    exist at decision time. Monitor probability quality and the decision
    policy separately; do not infer usefulness from this synthetic experiment.

For the derivations and measured limitations, see [notes.md](notes.md).
# Interview questions: regularized regression

1. **Why can a biased estimator predict better?**
   Prediction error includes squared bias, variance across training samples,
   and irreducible noise. Shrinkage can lower variance enough to offset its
   bias, especially in weakly identified directions. It can also underfit.
   A single train/test comparison does not estimate that decomposition.

2. **Explain exact zeros without relying only on the diamond picture.**
   At zero, the L1 subgradient spans [-1, 1]. For Lasso, a feature whose
   residual correlation satisfies \(|X_j^\top r/n|\le\alpha\) can have zero
   weight at the optimum. The proximal step implements this interval using
   soft thresholding. A smooth L2 penalty does not create that interval.

3. **Your NumPy Ridge and scikit-learn Ridge disagree at the same alpha. Why?**
   Check scaling, centering, intercept treatment, sample weights, and loss
   normalization first. For this study's SSE/(2n) + alpha*L2/2 objective,
   use `Ridge(alpha=n * alpha)` on the same rows. The parity tests encode
   this conversion.

4. **Why is a pipeline containing LassoCV not enough to guarantee fold-local scaling?**
   The scaler runs before LassoCV starts its internal folds. Wrap a pipeline
   ending in ordinary Lasso with GridSearchCV, so each outer training fold
   fits both preprocessing and estimation. Keep final test rows outside
   that search entirely.

5. **Lasso removed one of two correlated predictors. Is the removed feature useless?**
   No. Conditional redundancy, penalty strength, sampling, and representation
   all affect that choice. Refit across valid resamples and inspect selection
   frequency. Predictive inclusion is not causal identification.

6. **What does ElasticNet add beyond a second hyperparameter?**
   Positive L2 regularization makes the coefficient objective strictly convex
   and encourages sharing weight across similar predictors, while L1 retains
   thresholding. It does not guarantee perfect support recovery or stable
   selection in every sample.

7. **What evidence does the executed example actually provide?**
   It shows different coefficient allocations and sparsity with similar
   test RMSE on one known synthetic generator. Lasso was chosen by development
   CV. It does not establish a universal winner, coefficient stability, or
   statistically meaningful superiority. See [the record](notes.md#executed-experiments).

8. **How would you use a regularized baseline on OCR or document-processing features?**
   Define a continuous target, such as processing time, and validate whether
   squared error matches the operational cost. Keep preprocessing and the
   estimator together; split by customer or time where dependence requires
   it. Monitor schema, missingness, drift, and prediction error. Sparsity saves
   serving cost only if unneeded feature computation is also removed.
   Neither sparsity nor a linear form makes explanations causal.

9. **How do you know the teaching solver finished?**
   Inspect `converged` and the KKT residual. Small parameter updates can
   reflect a small step size rather than optimality. Analytic fixtures and
   independent scikit-learn comparisons provide stronger checks than merely
   confirming that the loss decreased.

10. **Does a Laplace prior mean the posterior selects features with probability one?**
    No. Lasso is a posterior-mode result under a Gaussian likelihood and a
    Laplace coefficient prior. A continuous posterior does not assign an
    atom of probability to zero. Posterior uncertainty and model selection
    probabilities require a separate treatment.

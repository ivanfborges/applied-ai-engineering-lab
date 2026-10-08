# SVM interview questions

1. **Why is the full margin width $2/\|w\|$?**
   Under canonical scaling, the supporting hyperplanes have scores -1 and +1.
   Their score difference is two; dividing by the norm of the normal vector
   converts that difference to Euclidean distance. Multiplying the score by
   an arbitrary constant cannot improve the actual geometric margin.

2. **Can a correctly classified observation have positive hinge loss?**
   Yes. When $0<y f(x)<1$, it is correctly classified but inside the margin.
   Hinge loss is $1-y f(x)$. A misclassification has margin below zero and
   loss above one. A score exactly at zero requires a tie convention.

3. **What is a support vector in the soft-margin case?**
   A training observation with a nonzero dual multiplier. It can lie on the
   margin, inside it, or on the wrong side of the boundary. Free multipliers
   $0<\alpha<C$ imply margin one; bounded multipliers $\alpha=C$ imply margin
   at most one. These statements assume ideal KKT conditions and common `C`.

4. **How do `C` and RBF `gamma` interact?**
   `C` prices margin violations relative to the norm penalty. `gamma` determines
   the distance scale of similarity. A local kernel and a high violation cost
   can fit noisy training structure. Joint validation selects their combination;
   no setting guarantees better held-out performance.

5. **Why can two linear SVM implementations disagree at equal `C`?**
   They may use mean versus summed hinge, hinge versus squared hinge, different
   intercept penalties, solver tolerances, or stopping rules. For this scratch
   objective, matched linear SVC uses `C_library = C_mean / n_train`.
   The [normalization derivation](notes.md#normalization-must-match-the-implementation)
   and controlled comparison make that distinction concrete.

6. **What error occurs when a scratch solver averages only violating rows?**
   Its gradient uses $1/|A|$ instead of $1/n$, changing the effective loss weight
   as the active set changes. Rows with zero hinge still belong in the empirical
   mean denominator. Inspect the regression and finite-difference checks in
   [test_svm.py](tests/test_svm.py).

7. **What does the kernel trick require?**
   A kernel computes an inner product in an implicit feature space. Symmetric
   positive semidefinite Gram matrices preserve the standard convex formulation.
   An arbitrary similarity is not automatically valid. RBF score contours in
   input space do not represent equal Euclidean distances to the boundary.

8. **Can you interpret a decision score as a probability?**
   No. Neither the score nor its sigmoid is automatically calibrated. Fit and
   evaluate calibration using appropriate validation boundaries. Logistic
   regression supplies a probabilistic model, but its calibration also requires
   checking against observed outcomes.

9. **How do you avoid scaling and hyperparameter leakage?**
   Reserve test rows before tuning; put the scaler inside the pipeline evaluated
   on training folds. Refit the selected pipeline on the complete training split,
   then evaluate the reserved test. The example tests inspect actual scaler fit
   inputs, including each fold and the final refit.

10. **How would you assess an SVM document router?**
    Version feature extraction, preprocessing, classifier, and decision policy
    together. Use representative labels and a split reflecting document sources
    and time. Compare class-specific errors, abstention or calibration, and
    measured latency against baselines. Kernel prediction cost grows with
    support-vector count; a linear classifier over fixed embeddings is a
    candidate when scale or latency makes kernels impractical. No router or
    deployment is implemented by this study.

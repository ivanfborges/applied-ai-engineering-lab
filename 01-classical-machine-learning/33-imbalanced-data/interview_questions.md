# Interview questions: imbalanced classification

1. **What do you inspect before balancing a 99:1 dataset?**  
   Positive counts, label reliability, minority subgroups, validation boundaries,
   deployment prevalence, costs, and downstream capacity. The proportion alone
   does not establish that the learner needs resampling.

2. **How do weighting and thresholding differ?**  
   Weighting changes the fitted loss and can change ranking and output
   probabilities. Thresholding changes actions on fixed scores. In the
   [example](example.py), each model's two policies therefore share AP and
   ROC-AUC, while confusion counts and alert volume differ.

3. **Why is balanced class weighting not a business-cost model?**  
   It equalizes aggregate class contribution using observed frequencies.
   The value of a missed event or a false alert is a separate assumption.
   Weighting can shift outputs away from natural-prevalence posteriors.

4. **When are random oversampling and weighting equivalent?**  
   Exact replication is equivalent to integer row weights in a summed data
   loss. Random multiplicities, normalization, regularization, and the learner
   can break practical equivalence. Holding nominal regularization fixed
   across fit sizes is not a controlled equivalence test.

5. **What does SMOTE assume, and why does scaling matter?**  
   A segment between nearby minority observations must be a plausible minority
   region. Scaling changes neighbors; noise, disconnected clusters, and
   categorical codes can invalidate interpolation. The
   [scratch implementation](from_scratch.py) handles numerical geometry only.

6. **Where does resampling belong in cross-validation?**  
   Inside each training fold, after any training-fitted representation used for
   distances. Evaluate on unchanged held-out rows. Pre-split duplication or
   interpolation can put related evidence on both sides of the boundary.

7. **Derive a cost-based probability threshold.**  
   Positive action costs $(1-p)C_{FP}$ and negative action costs $pC_{FN}$.
   With calibrated probabilities, constant costs, and zero correct-action
   costs, choose positive when $p\ge C_{FP}/(C_{FP}+C_{FN})$.
   Review constraints or case-dependent costs require a different policy.

8. **Why does the example select a threshold different from $1/11$?**  
   It minimizes observed validation cost, rather than applying the Bayes rule
   to established calibrated probabilities. Sampling variability and
   probability error can change the chosen threshold. The test set cannot be
   used to repair it after seeing results.

9. **Can a high ROC-AUC coexist with low alert precision?**  
   Yes. With many negatives, a small FPR can produce numerous false alerts.
   Report AP with prevalence and the exact metric definition, then inspect
   operating-point precision, recall, and workload. AP and trapezoidal PR area
   are different summaries.

10. **Does a lower Brier loss prove better calibration?**  
    No. It combines calibration and discrimination, with outcome uncertainty.
    Inspect reliability estimates with sufficient positive counts and evaluate
    calibration on appropriate held-out data.

11. **Analysts can review only 500 alerts per batch. What changes?**  
    Define the batch, eligibility, tie handling, and delay constraints; evaluate
    precision and recall at that capacity. A fixed threshold alone cannot
    guarantee 500 cases as volume and prevalence change. This experiment
    implements constant error cost without a capacity constraint.

12. **What can the recorded experiment establish?**  
    Reproducible behavior for the stated synthetic configuration and software
    environment. It does not establish a generally best resampler, probability
    calibration, financial return, or production performance. Interpretations
    in [the experiment record](notes.md#executed-experiment) await author review.

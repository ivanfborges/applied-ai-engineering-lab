# Interview questions: modern gradient boosting

1. **What is the difference between a residual and a Newton leaf update?**
   A squared-error residual is a negative gradient. A Newton update uses both
   gradient sums and local curvature; with L2 penalty its leaf value is
   $-G/(H+\lambda)$. Binary classification adds that value to a logit, not to a
   probability. Derive it from the quadratic objective in [notes.md](notes.md).

2. **What does a positive split gain establish?**
   Improvement in the local quadratic objective after paying the additional
   leaf penalty. It does not establish generalization or exact loss improvement.
   The tests compare gain with independently evaluated parent/child objectives.

3. **Is `min_child_weight=20` a twenty-row constraint?**
   No. In XGBoost it constrains Hessian mass. Logistic curvature depends on the
   current probabilities; confident rows contribute little. LightGBM also has
   curvature controls, alongside row-support parameters.

4. **Does XGBoost always grow level-wise while LightGBM grows leaf-wise?**
   XGBoost can use depthwise or loss-guided growth. LightGBM's best-first growth
   makes leaf count central. Ask which settings are in use before comparing.

5. **Why does a common depth limit fail to equalize models?**
   Symmetric trees repeat one condition across each level. Other policies may
   use different conditions on different branches. Leaves, bins, regularization,
   sampling, and early stopping further change effective capacity.

6. **What are GOSS and EFB, and did this example test them?**
   GOSS retains large-gradient rows and samples/reweights others. EFB bundles
   sufficiently exclusive sparse features. The example disables row sampling
   and does not isolate EFB, so it establishes no speed benefit from either.

7. **How do ordered target statistics differ from ordered boosting?**
   The former restrict label information when representing categories. The
   latter restricts information used in training predictions/gradients to
   reduce prediction shift. This example explicitly enables ordered boosting;
   a default CPU CatBoost fit need not do so.

8. **Can CatBoost repair temporal leakage?**
   No. A feature calculated after the decision, or an unrealistic random split,
   remains invalid. Category safeguards address a specific training mechanism.

9. **Why does the mixed example use different representations?**
   To make practical workflows executable: XGBoost receives train-fitted one-hot
   columns, LightGBM receives categories, and CatBoost receives strings. The
   comparison confounds representation and algorithm; it cannot isolate either.
   XGBoost native categorical support is a suggested follow-up.

10. **What would you do if AUC improved but operational performance declined?**
    Inspect probability reliability, thresholds, costs, prevalence, subgroup
    behavior, latency, and deployment population. Ranking improvement alone
    does not validate a decision rule.

11. **Where do early stopping, model selection, and final evaluation belong?**
    Early stopping and selection use validation information; the fixed winner
    is then evaluated on test rows. For repeated tuning, use cross-validation
    or another appropriate development procedure and preserve final evaluation.

12. **How would you choose a framework for a large categorical dataset?**
    Assess cardinality, rare/unseen values, sparsity, memory, training budget,
    serving compatibility, and validation constraints. Benchmark representative
    workloads with documented configurations and repeated measurements. This
    study's synthetic results are not evidence for a production choice.

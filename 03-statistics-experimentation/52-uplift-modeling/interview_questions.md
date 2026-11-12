# Uplift modeling: interview questions

1. **Why is churn risk insufficient for retention targeting?** Risk describes
   what happens under the observed policy. Targeting requires the change caused
   by an offer. A high-risk customer may be unaffected; a moderate-risk customer
   may respond. An offer also has a cost.

2. **What does a T-Learner estimate?** Two conditional outcome means,
   $\hat\mu_1(x)$ and $\hat\mu_0(x)$, whose difference estimates CATE under
   identification assumptions. Separate fits lose sample size per arm, and
   their errors both affect the difference.

3. **Can we validate individual treatment effects with RMSE?** Usually only one
   potential outcome is observed. This study reports RMSE against synthetic
   CATE, a conditional probability difference, not individual counterfactual
   ground truth. Real evaluation needs randomized policy comparisons or causal
   scores under defensible assumptions.

4. **What does the IPW gain endpoint measure?** Incremental positive outcomes
   per eligible person if everybody is treated, relative to treating nobody:
   the IPW ATE. Every ranking has the same endpoint; ranking quality concerns
   where gain appears before it. Tie handling and normalization must be stated.

5. **Why could a high AUUC be misleading?** Its scale depends on ATE and the
   curve definition. Sampling variation, test-set selection and arbitrary
   treatment of score ties can inflate it. Report the random reference and
   policy values, and reserve fresh data after tuning.

6. **When is positive uplift insufficient to justify treatment?** When
   incremental outcome value is smaller than contact cost, or capacity,
   eligibility and risk constraints prevent the action. A conversion-maximizing
   ranking need not maximize profit when values/costs differ.

7. **What changes with nonrandom treatment?** Specify the estimand, pre-treatment
   adjustment variables, exchangeability argument and overlap. Outcome models
   or cross-fitted DR methods can help estimation; none repairs unmeasured
   confounding automatically. Coarse bins may leave confounding within strata.

8. **Why run a homogeneous-effect control?** True ranking advantage disappears
   when all effects are equal, yet noisy fitted scores may still seem to find
   segments. Replications expose variation. A few favorable holdout estimates
   do not establish useful heterogeneity.

9. **What is needed after offline evaluation?** Log eligibility, features as
   available at decision time, assignment and its probability, actual exposure,
   policy version, costs and mature outcomes. Preserve randomized comparison
   data, assess spillovers, and test the frozen policy prospectively.

10. **How do uplift, DiD and contextual bandits differ?** Uplift is about
    heterogeneity and targeting; DiD supplies an identification design under
    assumptions such as parallel untreated trends. Bandits adapt actions online
    with exploration. A targeting learner alone supplies neither a DiD design
    nor an online exploration policy.

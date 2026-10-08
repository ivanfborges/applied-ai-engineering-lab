# Gradient boosting interview questions

1. **What is being differentiated?**  
   The loss with respect to the model's raw prediction at each training row.
   A learner approximates the negative gradient as a function of features.
   For half squared error this is $y-F$; for binary logistic loss with a
   raw logit it is $y-\sigma(F)$.

2. **Which errors does the next tree learn?**  
   Errors of the entire current ensemble. In the scratch code,
   `residual = y - prediction` runs inside the boosting loop after all
   previous corrections have been added. Fitting the original target at
   each round would not implement this mechanism.

3. **Why are leaf means correct for squared error?**  
   For a fixed region, differentiating
   $\sum_{i\in R}(r_i-\gamma)^2$ gives
   $\gamma=\bar r_R$. That is the optimal unshrunk leaf correction.
   Other losses can require different leaf values.

4. **Can more trees make training error worse?**  
   With full-data squared-error residual-mean leaves and learning rate in
   $(0,1]$, the exact-arithmetic update cannot increase training SSE.
   That guarantee does not extend to held-out error, arbitrary losses,
   or the same full-data objective under subsampling.

5. **Is a smaller learning rate equivalent to scaling a trained ensemble?**  
   No. It changes the predictions used to form the next gradient targets,
   which can change later splits and leaf values. Tune the rate jointly
   with stage count and weak-learner capacity.

6. **What can a sum of stumps fail to learn?**  
   With multiple features, each stump uses only one feature. Their sum is
   additive in individual features and cannot express arbitrary interactions
   such as XOR. Deeper trees can model interactions. The public scratch model
   uses one feature to isolate correction mechanics.

7. **How does this differ from Random Forest?**  
   Forest members fit target predictions with injected randomness and then
   average predictions. Boosted learners depend on earlier predictions and
   contribute signed additions. Forests support parallel tree training;
   boosting rounds have a sequential dependency. Validation must decide
   which approach suits a particular problem.

8. **How would you select the number of trees without corrupting evaluation?**  
   Track validation loss, choose a prefix or apply a predeclared stopping
   rule, and evaluate the chosen procedure on untouched test rows. The
   example scans a fully fitted path; it demonstrates stage selection but
   does not avoid the computation of later trees. Validation minima have
   selection bias.

9. **Why do classification boosters often contain regression trees?**  
   Their targets are numeric loss-gradient signals, even though the outcome
   is categorical. For logistic loss they modify a raw logit, then transform
   it to a probability. A sum of class votes is a different operation.

10. **What would block use of a booster for an AI routing policy?**  
    Unrepresentative review labels, future information in routing features,
    entity leakage, and a mismatch between training loss and the policy's
    operational costs. A good offline score also needs calibration checks
    when decisions depend on probabilities. This topic's synthetic regression
    results do not establish that a routing policy works.

See [the notes](notes.md) for derivations and measured evidence, and inspect
[the implementation](from_scratch.py) to connect the equations to the loop.
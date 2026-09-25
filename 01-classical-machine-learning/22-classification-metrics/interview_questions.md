# Interview questions: classification metrics

1. **Why can 95% accuracy describe a useless classifier?**  
   If positives are 5% of the sample, predicting negative for every row earns 95% accuracy and 0 recall. State prevalence, the positive class, and the confusion counts before judging usefulness.

2. **How do precision and recall differ?**  
   Precision is TP/(TP+FP), conditioned on predicted positives. Recall is TP/(TP+FN), conditioned on actual positives. Precision concerns alert reliability; recall concerns coverage.

3. **What happens when a classifier makes no positive predictions?**  
   Precision has a zero denominator and is undefined. This topic reports zero by an explicit software convention; the report should still show that the predicted-positive count is zero.

4. **What does F1 omit?**  
   The count formula `2TP/(2TP+FP+FN)` omits TN. Equal F1 values can still represent different FP/FN balances and operational costs.

5. **Does raising the decision threshold always raise precision?**  
   No. On a fixed score set, raising it can only reduce predicted positives and cannot improve recall, but precision may rise or fall depending on which examples are removed.

6. **How would you evaluate an alert model with limited analyst capacity?**  
   Define the positive event and review capacity, inspect expected alert counts and FP/FN costs, choose a threshold on representative validation data under the capacity constraint, then evaluate the frozen policy on held-out test data. Monitor prevalence and error counts after deployment.

7. **Why can precision fall after deployment even if recall stays similar?**  
   A lower positive prevalence can reduce the share of true alerts if TPR and FPR are otherwise similar. Conditional error rates may also change, so inspect both prevalence and the confusion matrix.

8. **How do these ideas transfer to an LLM or RAG evaluator?**  
   Define a concrete binary event, such as a relevant retrieved document or an unsupported answer, and specify the reference labels. Then FP and FN describe different product failures. Label quality and representative sampling remain part of the evaluation.

# Interview questions: classification metrics II

**What does ROC-AUC measure, including ties?**  
It is the probability that a randomly chosen positive outranks a randomly chosen negative, counting equal scores as half a win. It measures ordering across thresholds, not whether a score of 0.8 means 80% risk.

**Why report AP with prevalence?**  
AP summarizes precision over recall increments, and the no-skill level is approximately the positive prevalence. Precision also changes with prevalence at fixed TPR and FPR, so an AP number without the evaluation population can mislead. Trapezoidal PR area and AP can differ.

**Does imbalance invalidate ROC-AUC?**  
No. ROC-AUC remains a legitimate ranking measure. However, a low FPR can still produce many false alerts when negatives are numerous; inspect counts and PR behavior as well.

**How would you set a threshold for a review queue?**  
Define the target, such as a minimum precision or maximum daily alerts. Select the cutoff on validation data, report the resulting counts and uncertainty, then apply it once to an untouched test set. For a hard fixed capacity, compare a top-\(K\) policy.

**When does the cost-based threshold equal \(C_{FP}/(C_{FP}+C_{FN})\)?**  
When scores are calibrated probabilities and the only outcomes are constant false-positive and false-negative costs. Capacity limits, intervention effects, or heterogeneous costs break that simple rule.

**Can AUC stay fixed while calibration changes?**  
Yes. A strictly increasing transformation preserves ordering and hence ROC-AUC and AP, but changes the numerical score. In the [executed synthetic example](notes.md#1-ranking-preserving-probability-distortion), raising probabilities to the fourth power left both ranking summaries fixed while Brier score changed.

**How should calibration be checked?**  
Inspect reliability bins with counts, Brier score or log loss, and relevant segments. Fit any calibration mapping on data separate from the final evaluation data. Reassess it when prevalence or feature distributions change.

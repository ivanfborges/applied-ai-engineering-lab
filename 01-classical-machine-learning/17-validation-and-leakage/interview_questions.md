# Validation design: interview questions

**Why have validation and test data?**  
Validation influences the chosen system, even without fitting its coefficients.
A final holdout assesses the frozen choices. CV can replace a fixed validation
partition; it does not make the winning CV score independent of selection.

**A patient contributes 30 images. What should the split unit be?**  
For new-patient deployment, hold out patients, including related scans and
derived augmentations. Count independent patients and class support, not just
images. For future monitoring of known patients, time and available history
define a different evaluation target.

**Can unsupervised preprocessing leak?**  
Yes: scaling, imputation and PCA learn from covariates. Fit them inside each
training fold. A pipeline does not fix a precomputed future-derived feature.

**Your labels describe the next 30 days. Is sorting by time enough?**  
No. At a simulated training cutoff, only use labels whose outcome windows and
reporting delays have completed. Audit feature availability separately.
Choose exclusions from actual timestamps and overlapping intervals; a fixed
row gap is sufficient only under explicit sampling and delay assumptions.

**Why might pooled out-of-fold AUC differ from mean fold AUC?**  
AUC is based on pairs, not additive per-row losses. Pooling introduces
cross-fold comparisons between scores from different fitted models.
Specify the aggregation and intended population before comparing scores.

**Does mean plus or minus fold SD give a confidence interval?**  
No. Training sets overlap, so fold estimates are dependent. SD is descriptive
dispersion. An uncertainty method must respect independent sampling units and
the model-selection procedure.

**When does nested CV help?**  
When estimating an entire tuning procedure with limited data. Inner folds
choose configurations; outer holdouts evaluate those choices. Both levels
must respect the same deployment boundary. It adds considerable fitting cost.

**Can a test document be in a RAG index?**  
Yes, when deployment legitimately retrieves it. Keep evaluation queries and
reference answers independent of tuning. Hold out document families when the
question is generalization to new documents; distinguish that from withholding
the evidence needed to answer a question.

**A random-fold model scores perfectly but group CV is mediocre. What next?**  
Inspect entity overlap, duplicate structure and features that identify entities.
Clarify whether deployment targets known or new entities. In this study the
labels are random per customer, so recognizing the signature is sufficient for
the random folds; the result is not evidence for unseen-customer prediction.

**Give a short design explanation.**  
Define the prediction timestamp, available information and deployment population.
Reserve a relevant final holdout. Use development folds respecting groups or
time, fit transformations within those folds, freeze selection decisions, then
evaluate the complete system. Report the metric, aggregation and limitations.

# Anomaly detection: interview questions

**How can an unsupervised workflow leak information?** Fitting a scaler or a
normal reference using future periods leaks information even without labels.
Use chronological fit/calibration/test boundaries and feature availability times.

**Why can a detector with higher AP have lower recall at deployment?** AP
evaluates the full ranking. Recall at one operating point depends on calibration
and policy. In the dated synthetic record, Isolation Forest has higher AP than
MAD and lower recall at the 2% calibration setting. That is not a contradiction.

**Does nu specify future anomaly prevalence?** No. Its constraints concern
training violations and support vectors; future behavior also depends on data,
kernel, scaling and threshold. `contamination` in IsolationForest determines
its built-in score cutoff. Neither is an incident-probability estimate.

**What does the terminal-leaf correction do?** If a depth limit or duplicates
leave multiple fitting rows in a leaf, depth alone underestimates the remaining
isolation effort. Add the expected search path $c(n)$ for the leaf population.

**When is MAD sufficient?** A stable, interpretable metric with an appropriate
contextual baseline may not need a learned multivariate model. Featurewise MAD
can miss unusual combinations, and zero MAD needs an explicit policy rather
than an arbitrary epsilon.

**How would you evaluate with few labels?** Replay known incidents, inject
targeted faults to test specific failure modes, and review stratified samples
from both flagged and unflagged rows. Account for review sampling and delayed
labels. Alert count is measurable without labels; true precision and recall are not.

**Why not alert on the top 1% of every incoming batch?** That fixes workload
relative to each batch, potentially forcing alerts during healthy periods and
hiding large incidents. A frozen reference cutoff can reveal population shifts,
but still requires investigation capacity and context.

**What would you inspect after an alert surge?** Schema, units, missingness,
feature timing, model/threshold versions, score distributions, recent deployments
and workload mix. Examine incident traces before relaxing thresholds or retraining.

**How does this apply to agents?** Normalize token/tool-call behavior by task
type and inspect retry/transition patterns. Use hard execution budgets for known
limits; anomaly detection can prioritize investigation of unusual behavior. It
does not establish answer correctness or authorization violations.

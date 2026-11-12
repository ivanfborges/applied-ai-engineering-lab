# Product metrics: senior interview questions

**What would make a new RAG configuration successful?**
Define a user outcome such as independently verified task resolution, its
eligible population and observation window. Choose reliability, latency,
economics and safety guardrails, with diagnostics for retrieval and token use.
Acceptance is not correctness. Establish telemetry quality before interpreting
a treatment difference.

**Conversion moves from 12% to 15%. What changed?**
Three percentage points of absolute lift and 25% relative lift. Ask about
randomization, uncertainty, practical importance, maturity, segments and
guardrails. Relative lift without the baseline can exaggerate the impression
of value. Observational before/after differences need causal justification.

**Can D7 retention be 40% while seven-day churn is 5%?**
Yes. Exact-day activity and subscription cancellation need different events,
populations and windows. A weekly workflow may be valuable without exact D7
activity, and canceled subscribers may continue using a free tier. State both
contracts before comparing them.

**Why does this implementation require eight days for exact D7 retention?**
Its interval is `[7, 8)` elapsed days after entry. Seven days of follow-up only
reach the start of that interval. Other products use calendar days or another
convention; document the boundary instead of inheriting a metric name.

**The treatment has lower average latency. Can its experience be worse?**
Yes. The tail, errors and timeouts can deteriorate while the mean improves.
Inspect distribution and workload mix, TTFT versus completion time, and failure
semantics. Averaging host p95s does not produce an overall p95.

**A new model doubles cost/request. Is that unacceptable?**
It depends on incremental contribution and the constraints agreed before the
test. Include costs of failed attempts, retries, downstream review and shared
resources as appropriate. Cheap requests may fail often, so also assess cost
per verified successful task. Neither revenue nor contribution alone settles
critical safety constraints.

**No harmful outputs were detected in 1,000 trials. Is the model safe?**
Under independent trials and perfect detection, the one-sided 95% upper
incident probability is $1-0.05^{1/1000}\approx0.299\%$, not zero. Representative
exposure, evaluator recall, severity and unseen attack coverage remain separate
questions. Repeated prompts or users can violate the independence assumption.

**A safety degradation is nonsignificant. Has the guardrail passed?**
No. Absence of significance may reflect low power. Predefine a tolerable
degradation and use a suitable non-inferiority rule, with rare-event and
clustering considerations. A critical incident may be a hard stop even when
its aggregate rate is low.

**Why reject duplicates instead of silently removing them?**
Duplicate events can describe retries, conflicting assignments or legitimate
repeated requests. Deduplication needs stable IDs and a declared event model.
The prepared ledger requires one user row; guessing which row to keep changes
the measurement contract.

**Why does one request per user matter statistically?**
It aligns the request and randomization units in this teaching example.
Repeated requests create correlation and activity weighting. Define the
estimand first, retain nonrequesting assigned users when appropriate, and
use user-level or cluster-aware analysis instead of independent-request errors.

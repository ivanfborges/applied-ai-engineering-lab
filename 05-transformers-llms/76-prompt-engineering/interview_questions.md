# Questions to test engineering judgment

**What does a prompt change mathematically?**
The conditioning information in $P_\theta(y\mid p,x)$. Ordinary prompting leaves
the model parameters fixed; fine-tuning updates weights or adapters.

**How would you show that V3 improves on V2?**
Freeze the candidates, evaluate matched inputs under the same model/configuration,
repeat requests and inspect paired errors on an independent holdout. Account for
failed calls and context cost. This repository's authored response fixtures cannot
answer that comparison, because no model generated them under either prompt.

**Why separate JSON, schema and semantic scores?**
`{"invoice_id":"B22","total_usd":999999}` can parse and satisfy the schema
while contradicting a source total of USD 100. Each layer answers a different
question; a format check cannot establish source support.

**Should a missing total count as a failure?**
If the source omits it, a null may be the correct extraction. Exact match rewards
that policy, while complete-output coverage still records a need for additional
processing. Penalizing every null would encourage unsupported guesses.

**Does a retry loop solve incorrect extraction?**
Only if its acceptance checks can detect the relevant error. This controller
stops on schema validity and can return a wrong total. Additional retries also
consume budget and can repeat errors; evaluate first-attempt and final results.

**Why not evaluate only responses that parsed?**
That would hide operational/format failures and change the denominator. Report
all-case accuracy and structural rates first; a conditional score can be added
with its denominator clearly stated.

**When should deterministic logic replace prompt wording?**
For exact computations, type checks, supported-unit policies and authorization.
Language interpretation can remain model-assisted, with independently verifiable
intermediate evidence when the task warrants decomposition. More stages add
latency and failure paths, so compare against a simpler baseline.

**Do separate instruction and document fields prevent injection?**
They communicate a trust boundary but do not enforce one inside the model.
Authorization and tool permissions must be checked outside generation. A single
injection-themed fixture verifies scoring, not model robustness.

**How can a prompt overfit without changing model weights?**
Its designer repeatedly selects instructions that fit a finite development set.
The selection process can exploit incidental wording and example patterns. Keep
development and holdout data separate, version prompts and test perturbations.

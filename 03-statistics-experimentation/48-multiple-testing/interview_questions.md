# Questions about the full testing procedure

**Why can twenty individually valid 5% tests be misleading?**
Under independent exact true nulls, the chance of at least one rejection is
$1-0.95^{20}\approx0.6415$. Each test's calibration does not control that
family-level event. Define the claim family before choosing the correction.

**Is FDR 5% a promise that at most 5% of this report's findings are false?**
No. FDR is $E[V/\max(R,1)]$ across repeated applications. An individual report
can have a much larger FDP. With no discoveries FDP is zero; omitting such
repetitions estimates a different quantity. In the global null FDR equals FWER,
which explains why a null-only simulation cannot demonstrate their distinction.

**How do Holm and BH differ algorithmically?**
Holm steps down, testing ordered p-values against $\alpha/(m-i+1)$ until the
first failure. BH steps up: find the largest passing rank under $iq/m$, then
reject everything through it. For p-values 0.03 and 0.04 at 5%, Holm rejects
neither and BH rejects both. BH adjusted p-values need a reverse cumulative
minimum; the input order must be restored afterward.

**Which correction would you choose for a RAG quality report?**
First specify what constitutes a decision: a confirmatory gate, exploratory
screening or a conjunction of operational requirements. Use valid paired or
cluster-aware endpoint tests as appropriate. Bonferroni/Holm protect a declared
family against any false rejection; BH protects expected FDP under suitable
dependence assumptions. Correlation among quality metrics does not by itself
justify BH. Report the complete family, effect estimates and uncertainty;
retain latency, cost and safety constraints as explicit decision criteria.

**Why does checking an ordinary p-value daily change inference?**
Stopping at the first crossing uses the minimum over looks, not the p-value
at a fixed horizon. Nested looks are correlated but still offer additional
rejection opportunities. A prespecified finite-look Bonferroni budget is valid
with valid marginal p-values; calibrated group-sequential or anytime-valid
inference can support other monitoring policies. Correcting across endpoints
at each look alone does not cover the whole repeated decision procedure.

**Can correction rescue choosing the best of 100 prompts on one benchmark?**
Only under a valid, fully specified inferential design. A selected maximum
score is subject to winner's curse; adaptively proposing more prompts using
the same benchmark also compromises the underlying p-values. Freeze candidates
and hypotheses, separate development from final holdout evaluation, and account
for multiplicity where making inferential claims. Correction cannot repair
leakage, confounding, biased sampling or an invalid analysis unit.

**What would you inspect in this simulation's results?**
Check global-null raw FWER against its analytical value using Monte Carlo
uncertainty, confirm the FDR/FWER identity there, and examine mixed-family FDR
and power separately. Compare peeking policies on the same trajectories with
paired uncertainty. A finite seeded run illustrates the procedure under its
generator; it does not prove the theorem or establish performance on product data.

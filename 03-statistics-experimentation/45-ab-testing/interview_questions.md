# Interview questions: experiment design before the p-value

**Why can a randomized difference be causal?**

Assignment is independent of potential outcomes under valid randomization.
With consistent treatments, comparable measurement and no relevant interference,
the assigned-group difference estimates the average assignment effect for the
enrolled population. It does not identify each user's individual effect or
guarantee generalization to another population.

**How would you compare two RAG workflows?**

Define the intervention and eligible user population, randomize users with
stable assignments, and prespecify a user outcome and its window. Use offline
retrieval and answer checks before launch; monitor latency, cost and safety
alongside the primary outcome. Keep unrelated components fixed if attribution
to a particular change matters. Analyze all assigned users and audit telemetry.

**Why use different standard errors for the test and interval?**

The pooled test estimates one rate under the equality null. The unpooled Wald
interval estimates variance without that constraint. It is not the exact
inversion of the pooled test, and both rely on normal approximations. In sparse
data, choose more suitable inference rather than accepting the formula blindly.

**A result has $p=0.08$. Is the feature ineffective?**

The prespecified 5% test does not reject equality. This is not evidence of
equivalence. The effect estimate and interval may allow meaningful benefit,
harm or both. Assess design, precision and the worthwhile-effect threshold;
another experiment needs its own plan rather than an extension until significance.

**The primary metric improved but SRM failed. What next?**

Investigate the assignment, eligibility, exposure and analysis pipelines before
using the effect. Check the expected allocation and whether the test matches
the randomization design. Do not rebalance by removing users after the fact.
A favorable primary p-value cannot repair a compromised comparison.

**Why not filter to users who tried the new assistant?**

Trying it is potentially affected by treatment. Conditioning on that behavior
can break comparability. ITT estimates offering the change; an exposure effect
requires a different estimand and identification assumptions, potentially an
instrumental-variable approach.

**What if randomization is at organization level?**

Users within an organization can share outcomes and influence each other.
Power and variance must reflect the number of independent clusters and their
correlation. Counting every employee as an independent assignment can grossly
overstate precision. A sticky user hash does not solve this dependence.

**Can statistical significance justify higher inference costs?**

Only after relating the effect and its uncertainty to economic value and
prespecified operational constraints. A small positive difference can be
statistically detectable yet economically unattractive. This study's synthetic
completion counts do not measure costs or demonstrate guardrail compliance.

**Does A/A validate the platform?**

Repeated A/A runs can reveal calibration and instrumentation problems. A single
A/A rejection can occur by chance. These ideal binomial repetitions test the
inference procedure under a known generator; they do not exercise a real
assignment or logging platform. Report Monte Carlo uncertainty as well.

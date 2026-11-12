# Experimentation, Causality and Product Thinking

[Repository overview](../README.md) · [Full roadmap](../ROADMAP.md)

Module 3 spans global Days 45–52. **5 of 8 studies are implemented.** The public
directory follows the canonical `03-statistics-experimentation` name in the
repository structure. Future topics remain planned in the roadmap.

| Day | Implemented study | Evidence |
|---:|---|---|
| 45 | [A/B testing from scratch](45-ab-testing/) | Synthetic user-level ITT experiment and tested inference; twelve visual modules, five GIFs and offline browser verification across 42 scenarios; interpretation review pending |
| 46 | [Power analysis and sample size](46-power-analysis/) | Tested sizing and statsmodels agreement; eight-module Streamlit lab, 3D power surface, seeded Monte Carlo, three GIFs and maturity-aware traffic planning; interpretation review pending |
| 47 | [Product metrics and guardrails](47-product-metrics-guardrails/) | Nine-section Streamlit lab, coherent activity and censored survival, latency/economics/safety, feasible 3D Pareto set, conservative rollout policy, aggregation traps and two on-demand GIFs; interpretation review pending |
| 48 | [Multiple testing and false positives](48-multiple-testing/) | Tested Bonferroni/Holm/BH/BY, six-experiment offline visual lab, four GIFs, interactive 3D FWER controls, synthetic FDR/power and shared-path peeking with Monte Carlo uncertainty; interpretation review pending |
| 49 | [Causal inference I: DAGs and confounders](49-causal-inference-i-dags-confounders/) | Tested d-separation and original backdoor criterion, FWL/OLS agreement and standardization; repeated synthetic adjustment comparison with known total/direct effects; interpretation review pending |

Day 45 connects a hypothetical AI-workflow change to a binary user outcome,
making the randomization unit, observation window, denominator and stopping
rule explicit. Its [dated experiment record](45-ab-testing/notes.md) separates
generating assumptions, measured results and interpretation candidates.
**Author review of interpretations is pending.** No production benefit or
guardrail compliance is established.

The [visual guide](45-ab-testing/VISUAL_LAB.md) separates analytical curves,
Monte Carlo measurements and hypothetical product decisions. Generated HTML,
PNGs and GIFs stay ignored pending preview selection; no public asset is
automatically selected. Root README updates are deferred to author review.

Day 46 connects prospective sensitivity to feasible sample counts, allocation,
calendar time and illustrative serving expense. Its
[dated analytical record](46-power-analysis/notes.md#executed-analytical-study--2026-11-11)
uses hypothetical assumptions without generating experimental outcomes.
**Author review of its interpretation candidates is pending.**
The separate [visual laboratory](46-power-analysis/VISUAL_LAB.md) adds
synthetic outcomes and Monte Carlo uncertainty; its run configuration and
interpretations are recorded separately from that analytical study.

Day 47 turns the metric contract into an executable
[product scorecard](47-product-metrics-guardrails/). Its
[dated synthetic study](47-product-metrics-guardrails/notes.md#executed-synthetic-study--2026-11-12)
separates generating probabilities, measured outcomes, conversion uncertainty
and illustrative guardrail point checks. **Author review is pending.** The
checks establish no statistical non-inferiority, production safety or rollout
authorization.

Its separate [visual lab](47-product-metrics-guardrails/VISUAL_LAB.md) adds
interactive parameters, coherent activity trajectories, censored subscription
survival, latency-tail animations, break-even economics, detector sensitivity,
population trade-off geometry and an illustrative uncertainty-aware policy.
Generated GIFs stay ignored; author interpretation review remains pending.

Day 48 separates family-level false positives, expected false discovery
proportions and repeated significance decisions. Its
[dated synthetic record](48-multiple-testing/notes.md#executed-synthetic-study--2026-11-12)
reports global-null and mixed-family simulations with Monte Carlo uncertainty,
plus a shared-trajectory comparison of fixed horizon, naive peeking and
Bonferroni across ten planned looks. **Author review is pending.** No
arbitrary-dependence or anytime-valid guarantee is inferred from the run.

Its separate [visual laboratory](48-multiple-testing/VISUAL_LAB.md) adds
null-test accumulation, correction-rule animation, effect-size and PSD
correlation sweeps, a disclosed selected peeking path and an offline 3D
analytical surface. The original study record remains separate; generated
reports stay ignored and interpretation review remains pending.

Day 49 connects assumed causal paths to executable adjustment checks and a
[dated synthetic study](49-causal-inference-i-dags-confounders/notes.md#executed-synthetic-study--2026-11-12).
Four regression specifications separate confounding, mediation and collider
bias, with FWL agreement and empirical standardization. **Author review of
interpretations is pending.** The constructed model establishes no causal
claim about a real AI assistant.

Days 50–52 are planned: propensity scores and matching,
difference-in-differences and uplift modeling. No placeholder folders are added.

Use each topic's execution instructions and focused tests. The shared
[validation contract](../docs/validation.md) describes repository checks and
their limits; [methodology](../docs/methodology.md) separates implementation,
automated verification and author interpretation review.

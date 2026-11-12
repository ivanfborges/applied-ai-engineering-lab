# Experimentation, Causality and Product Thinking

[Repository overview](../README.md) · [Full roadmap](../ROADMAP.md)

Module 3 spans global Days 45–52. **1 of 8 studies is implemented.** The public
directory follows the canonical `03-statistics-experimentation` name in the
repository structure. Future topics remain planned in the roadmap.

| Day | Implemented study | Evidence |
|---:|---|---|
| 45 | [A/B testing from scratch](45-ab-testing/) | Synthetic user-level ITT experiment and tested inference; twelve visual modules, five GIFs and offline browser verification across 42 scenarios; interpretation review pending |

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

Days 46–52 are planned: power and sample size, product metrics and guardrails,
multiple testing, DAGs and confounders, propensity scores and matching,
difference-in-differences and uplift modeling. No placeholder folders are added.

Use each topic's execution instructions and focused tests. The shared
[validation contract](../docs/validation.md) describes repository checks and
their limits; [methodology](../docs/methodology.md) separates implementation,
automated verification and author interpretation review.

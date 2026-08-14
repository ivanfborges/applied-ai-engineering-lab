# Hypothesis Testing

## Overview

Hypothesis testing evaluates how compatible observed data are with a defined
null model. It does not prove a hypothesis, assign a probability to the null,
or decide whether an effect is useful. A responsible analysis connects the
test to an estimand, a sampling design, an effect estimate, uncertainty, and a
decision threshold.

This study uses a paired AI-system evaluation. A baseline and candidate are
scored on the same synthetic query set, so the relevant observations are the
query-level differences

\[
d_i = \text{candidate}_i - \text{baseline}_i.
\]

The example compares a paired Student-\(t\) test with an educational sign-flip
test. It also reports the mean difference, a 95% confidence interval, and
Cohen's \(d_z\) rather than reducing the result to `p < 0.05`.

## Concepts and relevance

- Null and alternative hypotheses define the claim and direction before data
  are inspected.
- A p-value measures tail probability under the null model; it is not
  \(P(H_0\mid\text{data})\).
- Type I error is a false rejection, Type II error is a missed effect, and
  power is \(1-\beta\) for a particular effect and design.
- Paired designs preserve query-level matching and can remove nuisance
  variability that an independent-sample analysis would retain.
- Statistical significance, effect magnitude, uncertainty, and practical
  relevance answer different questions.

These distinctions matter when comparing model versions, RAG pipelines,
prompts, OCR stages, latency changes, or human-evaluation protocols. Even a
well-calibrated test cannot repair leakage, biased sampling, pseudo-replication,
an invalid metric, or a mismatch between the evaluation population and the
deployment population.

## Files

- [`notes.md`](notes.md): theory, assumptions, power, test selection, failure
  modes, and engineering interpretation.
- [`example.py`](example.py): deterministic paired analysis using NumPy and
  SciPy on explicitly synthetic scores.
- [`from_scratch.py`](from_scratch.py): educational paired statistics and
  exact/Monte Carlo sign-flip test.
- [`hypothesis_testing_visual_lab.py`](hypothesis_testing_visual_lab.py):
  responsive Streamlit laboratory with eight selectively rendered sections.
- [`statistical_utils.py`](statistical_utils.py): validated power,
  simulation, pairing, multiplicity, and interval calculations used by the lab.
- [`gif_exports.py`](gif_exports.py): bounded generators for three educational
  animations.
- [`tests/`](tests/): formula, validation, reproducibility, and inference
  consistency checks.
- [`interview_questions.md`](interview_questions.md): senior-level conceptual
  and applied questions.
- [`references.md`](references.md): documentation and further reading.

## Run

From the repository root, install the shared dependencies if needed:

```bash
python -m pip install -e ".[dev]"
```

Run both educational paths:

```bash
python 00-foundations/12-hypothesis-testing/from_scratch.py
python 00-foundations/12-hypothesis-testing/example.py
```

Start the interactive laboratory:

```bash
streamlit run 00-foundations/12-hypothesis-testing/hypothesis_testing_visual_lab.py
```

Generate all three GIFs:

```bash
python 00-foundations/12-hypothesis-testing/gif_exports.py
```

The repository environment is canonical. For a standalone minimal install,
the visual lab uses:

```bash
python -m pip install numpy pandas scipy plotly streamlit matplotlib pillow
```

Run the focused tests:

```bash
python -m pytest -q 00-foundations/12-hypothesis-testing/tests
```

The two educational examples print to the console and create no files. The GIF
command writes only to the ignored `outputs/` directory. No external dataset,
credential, network call, or notebook is required.

## Visual laboratory

The sidebar controls sample size, effect, standard deviation, alpha, test
direction, Monte Carlo experiments, sign-flip permutations, minimum practically
important difference (MPID), and random seed. Only the selected section is
computed, and deterministic simulations use bounded Streamlit caches.

| Section | Question made visible |
|---|---|
| Test intuition | Where do the rejection and p-value regions sit under H₀? |
| Errors and power | How do alpha, beta, power, sample size, variance, and effect size interact? |
| Repeated experiments | Why do true nulls still produce false positives, and how do p-values change under H₁? |
| Paired AI experiment | What do query-level gains, regressions, effect uncertainty, and pairing contribute? |
| Randomization test | How does sign flipping construct an empirical null distribution? |
| Multiple testing | How do uncorrected, Bonferroni, and Benjamini-Hochberg decisions differ? |
| Confidence intervals | When does a matching two-sided interval agree with a hypothesis test, and what does repeated coverage mean? |
| Practical significance | How can statistical detectability and decision magnitude disagree? |

### Calculation assumptions

- The introductory p-value view uses a known-SD normal reference model for a
  mean and supports two-sided or greater alternatives.
- The alpha/beta diagram uses a one-sided normal superiority test so every
  shaded area exactly matches the displayed error probability.
- Power curves and the 3D surface use the noncentral t distribution for a
  one-sample test and standardized effect `d = delta / sigma`.
- Repeated tests use IID normal samples, an estimated sample SD, and a
  Student-t reference distribution.
- The paired RAG example treats queries as independent experimental units and
  analyzes candidate-minus-baseline differences from synthetic bounded scores.
- The sign-flip test requires sign exchangeability under the null; it is not
  assumption-free.
- The multiple-testing simulation uses independent uniform p-values with every
  null true. Only in that configured simulation is every rejection false.
- The interactive CI equivalence uses matching two-sided normal procedures;
  repeated intervals use Student-t intervals on IID normal samples.

### GIF exports

The generator creates these regenerable artifacts:

```text
outputs/sample_size_uncertainty.gif
outputs/statistical_power.gif
outputs/false_positive_simulation.gif
```

They remain ignored until the author deliberately selects and unignores a
small preview set. The strongest candidates after visual inspection are
`statistical_power.gif` for a GitHub README and
`false_positive_simulation.gif` for a short explanatory post. The uncertainty
GIF is clear but overlaps more with the Day 11 standard-error material.

### Executed visual evidence

The complete GIF generator and every interactive section and subview were
executed with the checked-in implementation. These observations concern
synthetic configurations only.

- **Hypotheses:** standard error should contract as `1 / sqrt(n)`; for a fixed
  positive alternative, increasing n should reduce beta and increase power;
  and repeated calibrated tests under a true null should produce a finite-run
  false-positive rate near, but not forced to equal, alpha.
- **Configuration:** uncertainty frames at `n = 10, 20, 50, 100, 250, 500`
  with population SD 1; one-sided power frames with effect 0.35, SD 1, and
  alpha 0.05 at the same sample sizes; and 250 two-sided Student-t experiments
  under H₀ with `n = 30`, SD 1, alpha 0.05, and seed 42.
- **Observed results:** the displayed theoretical SE decreased from 0.3162 at
  `n = 10` to 0.0447 at `n = 500`. The final configured power value was 1.0000
  at `n = 500`. The true-null simulation produced 15 false positives among 250
  experiments, an empirical rate of 0.0600.
- **Interpretation candidate for author review:** the rendered evidence is
  consistent with the square-root uncertainty relationship, the separation of
  H₀ and H₁ as information increases, and finite Monte Carlo fluctuation around
  the nominal Type I rate.
- **Limitations:** the animations use constructed normal models and only a few
  selected sample sizes. A final power value rounded to 1.0000 is
  configuration-specific, not a guarantee. The false-positive run contains
  only 250 experiments. None of the visuals validates a real metric, sampling
  frame, evaluator, production population, or deployment trade-off.

## Experiment record

The practical example was executed with the checked-in source. The result is
evidence about a constructed distribution, not a benchmark or production
claim.

- **Hypothesis:** under the two-sided null, the population mean of
  `candidate - baseline` is zero; the configured alternative is that it is not
  zero.
- **Configuration:** 80 synthetic paired scores; seed 12; baseline scores drawn
  from a normal distribution with mean 0.72 and SD 0.08; independent paired
  increments drawn with mean 0.018 and SD 0.04; both system scores clipped to
  `[0, 1]`; alpha 0.05; a paired Student-t test; 50,000 seeded sign-flip draws;
  and an explicitly illustrative practical-improvement threshold of 0.02.
- **Observed result:** baseline and candidate means were 0.7237 and 0.7407.
  The mean paired difference was 0.0170, with 95% t interval
  `[0.0086, 0.0255]`, `t = 4.0147`, paired t-test `p = 0.000135`, sign-flip
  `p = 0.000200`, and Cohen's `dz = 0.4489`.
- **Interpretation candidate for author review:** both configured procedures
  find the positive synthetic difference difficult to reconcile with the
  zero-effect null. The interval excludes zero but spans values below and
  above the illustrative 0.02 threshold, so this run does not establish that
  the effect clears that practical threshold.
- **Limitations:** all scores, distribution parameters, and the practical
  threshold are constructed. Clipping changes the generated difference
  distribution, the sign-flip result relies on its null-invariance assumption,
  and one deterministic sample does not demonstrate calibration, power, or
  performance on real AI evaluations. The example does not model evaluator
  error, clustered queries, leakage, multiplicity, or distribution shift.

## Key takeaways

- State the estimand, experimental unit, hypotheses, direction, alpha, and
  stopping rule before inspecting the result.
- Read a p-value as a probability about data under a model, not a probability
  that a hypothesis is true.
- Failure to reject is not evidence of equality; the confidence interval shows
  which effect sizes remain compatible with the procedure.
- Power is defined relative to a target effect and depends on sample size,
  noise, alpha, and design.
- Paired analysis is appropriate only when the pairing is real and preserved.
- Deployment decisions also require effect magnitude, uncertainty, latency,
  cost, reliability, safety, and evaluation validity.

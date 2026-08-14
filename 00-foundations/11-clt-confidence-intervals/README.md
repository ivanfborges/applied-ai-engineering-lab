# Central Limit Theorem and Confidence Intervals

## Overview

A point estimate is incomplete without a description of how it could vary
across samples. The Central Limit Theorem (CLT) explains why the sampling
distribution of many averages becomes approximately normal under suitable
conditions, even when the observations themselves are not normal. Standard
errors quantify that repeated-sampling variation, and confidence intervals
turn it into an uncertainty range.

For independent, identically distributed observations with finite mean
\(\mu\) and finite variance \(\sigma^2\),

\[
\frac{\sqrt{n}(\bar X-\mu)}{\sigma}
\xrightarrow{d} N(0,1),
\qquad
\operatorname{SE}(\bar X)=\frac{\sigma}{\sqrt n}.
\]

The CLT concerns the distribution of an estimator across repeated samples. It
does not make the original data normal, and it does not make biased or
dependent data reliable.

## Technical questions

This study uses synthetic data to investigate three connected questions:

1. Do means of samples from a right-skewed exponential population become less
   skewed as sample size grows?
2. Does the empirical spread of those means follow \(\sigma/\sqrt n\)?
3. Does a nominal 95% Student-\(t\) procedure obtain approximately 95%
   repeated-sampling coverage in the configured simulation?

It also implements a normal-approximation interval for a mean and a Wilson
score interval for a proportion from first principles. These are educational
implementations, not replacements for a statistical library or a review of
the sampling design.

## Why it matters in applied AI

Offline model scores, LLM evaluation rates, conversion metrics, human-review
scores, and mean latency are estimates from a particular sample. Reporting an
estimate with its interval helps distinguish effect magnitude from sampling
noise. The interval is meaningful only when the estimand, target population,
sampling unit, and dependence structure match the decision.

A narrow interval can still be misleading when evaluation data are selected,
labels are noisy, users contribute correlated observations, leakage is
present, or distribution shift is ignored. Statistical precision does not
repair a biased measurement process.

## Files

- [`notes.md`](notes.md): intuition, derivations, assumptions, interval choices,
  uncertainty communication, applications, and failure modes.
- [`example.py`](example.py): deterministic CLT and Student-\(t\) coverage
  experiments using synthetic exponential samples.
- [`from_scratch.py`](from_scratch.py): educational mean, variance, standard
  error, normal mean interval, and Wilson proportion interval formulas.
- [`visual_lab.py`](visual_lab.py): CLI for the complete static, animated, and
  standalone interactive laboratory.
- [`visualizations/`](visualizations/): reusable numerical, rendering,
  animation, applied-experiment, and Plotly modules.
- [`tests/`](tests/): formula, validation, reproducibility, simulation, and
  visual-manifest checks.
- [`interview_questions.md`](interview_questions.md): senior-level conceptual
  and applied questions.
- [`references.md`](references.md): primary documentation and further reading.

## Run

From the repository root, install the shared dependencies if needed:

```bash
python -m pip install -e ".[dev]"
```

Run both educational paths:

```bash
python 00-foundations/11-clt-confidence-intervals/from_scratch.py
python 00-foundations/11-clt-confidence-intervals/example.py
```

Run the focused tests:

```bash
python -m pytest -q 00-foundations/11-clt-confidence-intervals/tests
```

The two educational examples print to the console and create no files.

Generate the complete visual laboratory:

```bash
python 00-foundations/11-clt-confidence-intervals/visual_lab.py
```

Generate a single conceptual section or use the reduced rendering mode:

```bash
python 00-foundations/11-clt-confidence-intervals/visual_lab.py --section clt
python 00-foundations/11-clt-confidence-intervals/visual_lab.py --section ci
python 00-foundations/11-clt-confidence-intervals/visual_lab.py --section se
python 00-foundations/11-clt-confidence-intervals/visual_lab.py --section dependence
python 00-foundations/11-clt-confidence-intervals/visual_lab.py --section comparison
python 00-foundations/11-clt-confidence-intervals/visual_lab.py --quick
```

The shared environment provides NumPy, SciPy, Matplotlib, Plotly, and Pillow.
No notebook, server, credential, network call, or external dataset is needed.

## Visual laboratory

Every generated asset answers one statistical question:

| Asset | Question |
|---|---|
| `01_population_vs_sampling.png` | How can skewed observations produce a more symmetric sampling distribution of the mean? |
| `02_clt_convergence.gif` | How do sampling-distribution shape and standard error change as sample size grows? |
| `03_standard_error_vs_n.png` | Why does four times the independent data give approximately half the standard error? |
| `04_standard_error_surface.html` | How do sample size and intrinsic population variability jointly determine standard error? |
| `05_sd_vs_se.png` | Why are observation-level SD and estimator-level SE different quantities? |
| `06_ci_construction.png` | How do sample SD, SE, a Student-t critical value, and margin of error assemble an interval? |
| `07_ci_coverage.gif` | What does 95% repeated-sampling coverage mean? |
| `08_confidence_level_width.png` | Why does higher confidence require a wider interval for the same sample? |
| `09_z_vs_t_distribution.png` | Why do small Student-t degrees of freedom produce wider intervals? |
| `10_skewness_and_sample_size.png` | Why is `n >= 30` not a universal normal-approximation rule? |
| `11_independence_violation.png` | Why can 5,000 correlated rows contain far less information than 5,000 independent rows? |
| `12_model_comparison_ci.png` | Why should two models be compared through uncertainty in their difference? |
| `13_practical_vs_statistical_significance.png` | Why can a detectable effect still fall below a decision-relevant magnitude? |

The outputs are written to `assets/` and remain ignored, regenerable artifacts
until the author explicitly curates a small preview set. The standalone Plotly
HTML embeds its JavaScript for offline rotation, zoom, and hover and is about
4.7 MiB, so it should remain local unless a specific publication strategy
justifies that size.

### Executed visual evidence

The complete full-resolution laboratory was executed with the current source.
These results describe deterministic synthetic configurations, not real-system
performance or benchmark evidence.

- **Hypotheses:** IID sample-mean spread will follow \(\sigma/\sqrt n\);
  sample-mean skewness will decrease at a source-dependent rate; a normal-data
  Student-\(t\) interval procedure will show coverage close to its nominal
  level; positive within-user correlation will make the IID SE optimistic;
  and a large sample can distinguish a small effect from zero without making
  that effect practically important.
- **Configuration:** fixed seeds per experiment; exponential population with
  mean and SD 2.0; 10,000 samples at \(n=30\) for the population comparison;
  4,000 samples in each of nine CLT animation frames; 5,000 standardized means
  per source/size panel; 60 normal-data Student-\(t\) intervals at \(n=25\);
  2,000 dependence trials with 100 users and 50 rows per user; 120 independent
  scores per model; and 80,000 observations per group for the illustrative
  practical-significance comparison.
- **Observed results:** at \(n=30\), theoretical SE was 0.3651 and empirical
  sampling SD was 0.3618. At the CLT animation's final \(n=200\) frame, those
  values were 0.1414 and 0.1442. Of 60 generated intervals, 57 covered the true
  mean, yielding 95.00% empirical coverage. Log-normal standardized-mean
  skewness decreased from 2.0228 at \(n=5\) to 0.8255 at \(n=30\) and 0.4171
  at \(n=100\). With configured ICC 0.80, naive IID SE was 0.0316 while
  clustered empirical SE was 0.2010 and the cluster-aware formula gave 0.2005.
  Model B's point estimate exceeded Model A's by 0.5200, while the 95% Welch
  interval for the difference was [-1.3988, 2.4388]. In the large-sample
  example, the estimated effect was 0.1423 with 95% CI [0.0932, 0.1914], below
  the explicitly illustrative practical threshold of 0.50.
- **Interpretation candidate for author review:** the rendered evidence is
  consistent with the mechanisms the lab was designed to expose: CLT
  convergence and the square-root SE law under IID finite-variance sampling;
  slower convergence for a skewed log-normal source; long-run rather than
  per-interval confidence; severe IID underestimation under clustering; and
  distinct questions for statistical detectability and decision value.
- **Limitations:** every distribution, effect, correlation, and threshold is
  constructed. The coverage animation has only 60 intervals and its exact
  95.00% realization is simulation-specific. The cluster-aware expression
  assumes a balanced Gaussian random-intercept design with known variance
  components. Welch intervals are approximate for the model examples. The
  practical threshold is explicitly illustrative and not an industry
  standard. None of the visuals accounts for selection bias, leakage,
  measurement error, evaluator disagreement, or distribution shift.

After author review, the strongest public-preview candidates are
`02_clt_convergence.gif`, `07_ci_coverage.gif`, and
`11_independence_violation.png`. They cover the central theorem, the correct
confidence interpretation, and the most production-relevant assumption
failure. They are recommendations only and remain ignored in the current
worktree.

## Executed experiment record

The practical example was executed with the current source configuration. Its
results are evidence about a constructed population, not benchmark claims.

- **Hypothesis:** as sample size increases, means drawn from the exponential
  population will have empirical standard error close to
  \(\sigma/\sqrt n\) and decreasing skewness; the nominal 95% Student-\(t\)
  intervals will have coverage reasonably close to 95% at \(n=50\).
- **Configuration:** exponential synthetic population with mean and standard
  deviation 2.0; seed 42; 10,000 simulated samples at each of \(n=5,30,100\);
  and a separate seed-2024 experiment with 10,000 samples of size 50. Sampling
  is direct from the population distribution. Sampling-distribution standard
  deviations use `ddof=0`; within-sample standard deviations use `ddof=1`.
- **Observed result:** for \(n=5,30,100\), the empirical standard errors were
  0.8928, 0.3604, and 0.1994, compared with theoretical values 0.8944, 0.3651,
  and 0.2000. Sampling-distribution skewness decreased from 0.9114 to 0.3330
  and 0.1608. The separate Student-\(t\) experiment produced 93.61% empirical
  coverage and mean interval width 1.1162 at \(n=50\).
- **Interpretation candidate for author review:** the configured results are
  consistent with the predicted square-root reduction in standard error and
  increasing symmetry of the sample mean. Coverage was close to, but below,
  the nominal 95%, which is consistent with treating the Student-\(t\) interval
  as an approximation rather than an exact interval for skewed exponential
  observations.
- **Limitation:** exponential data have finite variance and a known generating
  process. One deterministic simulation does not establish adequate CLT
  behavior for other distributions, estimators, sample sizes, dependence
  structures, or real evaluation data. The Student-\(t\) interval is exact for
  normal observations, but only approximate for these exponential samples.

## Key takeaways

- The population distribution and an estimator's sampling distribution are
  different objects.
- Standard deviation measures variability among observations; standard error
  measures variability of an estimator across samples.
- Mean uncertainty shrinks at the \(1/\sqrt n\) rate under the IID variance
  calculation, so halving standard error requires about four times the data.
- A frequentist 95% interval describes the long-run coverage of a procedure,
  not generally a 95% posterior probability for one realized interval.
- There is no universal \(n=30\) guarantee; skewness, tails, dependence, and
  the estimator determine whether an approximation is adequate.
- Communicate the estimate, interval, confidence level, sample size, sampling
  unit, method, assumptions, and practical decision threshold together.

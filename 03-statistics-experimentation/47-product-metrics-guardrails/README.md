# Day 47 — Product Metrics and Guardrails

**Can higher conversion coexist with an unacceptable product trade-off?**
A model-quality gain needs a measurement contract before it can support a
product decision. This study connects conversion, retention, subscription
churn, latency, revenue, serving cost and safety incidence in one scorecard.
The local Streamlit lab lets you change both systems and inspect seven linked
visual sections, two aggregation demonstrations and on-demand GIFs.
Primary outcomes describe intended value; guardrails constrain acceptable
degradation; diagnostics explain movement; data-quality checks assess whether
the measurements can be trusted.

The executable example uses **synthetic data only**: independent assigned users,
one first request per user, fixed seed, hypothetical revenue and cost units.
It prints an arm-level scorecard, a conversion-lift interval and illustrative
treatment point checks. It performs no release action.

## Start the interactive lab on Windows

From the repository root, select a **stable Python 3.11+** interpreter in VS Code.
Create a separate environment if an existing environment uses a prerelease Python:

```powershell
python -m venv .local/environments/day47
.\.local\environments\day47\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Set-Location 03-statistics-experimentation/47-product-metrics-guardrails
streamlit run app.py
```

If activation is restricted by your shell, use the environment executable
directly from the repository root:

```powershell
.\.local\environments\day47\Scripts\python.exe -m streamlit run 03-statistics-experimentation/47-product-metrics-guardrails/app.py --browser.gatherUsageStats=false
```

Dependencies are centralized in [pyproject.toml](../../pyproject.toml).
No additional framework, topic requirements file, credentials, external API or
real customer dataset is needed. The topic configuration disables Streamlit
usage telemetry; when launching from the repository root, append
`--browser.gatherUsageStats=false` for the same setting. Stop with `Ctrl+C`.

### Explore the nine sections

| Section | Visual question |
|---|---|
| Conversion and A/B testing | What remains in the denominator, and how much can estimated lift vary? |
| Retention and churn | Why do exact-day activity, finite rolling return and subscription survival differ? |
| Latency and tail behavior | How can the median stay fixed while p95/p99 worsen? |
| Revenue and unit economics | When does higher conversion pay for a more expensive system? |
| Safety and rare events | What can a small or zero incident count actually establish? |
| 3D product trade-offs | Which known-population configurations satisfy constraints and are nondominated? |
| Simulated rollout decision | Which checks pass, fail or remain inconclusive with uncertainty? |
| Aggregation traps | How do segment mixtures and unequal request volumes change weighting? |
| Export GIFs | How do tail spread and sample size change the pictures over time? |

Use the sidebar to change sample size, seed, arm parameters, observation date,
detector sensitivity and policy limits. Every section includes what to inspect,
what to change and the operational implication. Control is blue; treatment is
orange. Symbols and text supplement color. Reset restores the global defaults.

The visual generator is a **separate model** from the historical textual example:
it adds coherent activity trajectories, censored subscription lifetimes and
bounded serving costs. Its scorecard must not be substituted into the original
dated study. Read [the visual model and policy](VISUAL_LAB.md) for definitions,
conservative confidence bounds and recorded validation scope.

### Export the two GIFs explicitly

In the topic directory, use section 09's button or run:

```powershell
python animations.py --users-per-arm 3000 --seed 47 --frames 24
```

The CLI uses documented default arms; the dashboard button uses current arm
controls. Both create a unique `outputs/gif-run-<UTC timestamp>/` containing
`latency_percentiles.gif`, `sample_size_uncertainty.gif` and a configuration
manifest. Default size is 900 × 500 pixels with 24 frames. Nothing renders on
import or routine dashboard refresh. All exports remain ignored and require
review before any deliberate public-preview selection.

Three useful starting experiments:

1. Set equal medians, keep control sigma at 0.5 and raise treatment sigma to
   1.0; compare the ECDF, p95/p99 and fixed-median animation.
2. Raise treatment conversion and serving cost together; vary revenue and
   inspect incremental contribution, the break-even contour and rollout checks.
3. Reduce sample size and safety probability, then lower detector sensitivity;
   compare zero-count bounds with the policy's inconclusive safety check.

## What the example measures

| Measure | Population and interpretation |
|---|---|
| 24-hour conversion | Converted users / all assigned users, including failed requests |
| Exact D7 retention | Active in days `[7, 8)` / users with at least eight days of follow-up |
| Seven-day subscription churn | Cancellations in `[0, 7)` / starting subscribers with seven days of follow-up |
| Error rate and p50/p95 latency | All first requests; duration includes failed attempts |
| Revenue and contribution per user | Revenue and revenue minus attributed variable cost / assigned users |
| Cost per conversion | All first-request costs / converted users, including the cost of failures |
| Unsafe-output incidence | Synthetic incident labels / all first requests, assuming complete review |

Each changing cohort denominator is printed. Empty retention/churn populations
and zero-conversion cost ratios are unavailable (`NaN` in tables, `null` in JSON),
never silently zero. The validator rejects duplicate users, incomplete required
telemetry and missing outcomes in mature cohorts. It cannot detect users absent
from the supplied assignment ledger.

## Run from the repository root

Use stable Python 3.11+ and the shared dependencies in
[pyproject.toml](../../pyproject.toml). No topic-specific extra is needed.

```bash
python -m pip install -e ".[dev]"
python 03-statistics-experimentation/47-product-metrics-guardrails/example.py
python -m pytest -q 03-statistics-experimentation/47-product-metrics-guardrails/tests/test_metrics.py
```

Optionally retain a dated, non-overwriting JSON record under ignored `outputs/`:

```bash
python 03-statistics-experimentation/47-product-metrics-guardrails/example.py --users-per-arm 3000 --seed 47 --output-dir 03-statistics-experimentation/47-product-metrics-guardrails/outputs
```

Small samples can fail the conversion interval's explicit minimum-count check;
the interval is a large-sample teaching approximation. The JSON includes the
configuration, package versions, hypothesis, observations, interpretation
candidate and limitations. Numeric results repeat with the same configuration
and environment; timestamps and output filenames change.

## Read and inspect

- [notes.md](notes.md): metric contracts, formulas, inference boundaries,
  pitfalls, sources and dated experiment evidence.
- [metrics.py](metrics.py): educational aggregation, validation, inverse-ECDF
  percentile, conversion interval and zero-incident upper bound.
- [example.py](example.py): synthetic generator, textual scorecard and optional
  run record; no plots or external calls.
- [tests/test_metrics.py](tests/test_metrics.py): known-answer cases, maturity
  boundaries, missingness, unequal sample sizes and NumPy/SciPy agreement.
- [app.py](app.py): Streamlit controls and educational explanations.
- [simulation.py](simulation.py): stable assignments, activity/cancellation
  trajectories, repeated experiments and synthetic model configurations.
- [visual_charts.py](visual_charts.py): interactive Plotly figures and animations.
- [animations.py](animations.py): explicit GIF export and dated render manifest.
- [tests/test_visual_math.py](tests/test_visual_math.py),
  [tests/test_visual_app.py](tests/test_visual_app.py) and
  [tests/test_visual_exports.py](tests/test_visual_exports.py): mathematical,
  interface and temporary-render checks.
- [interview_questions.md](interview_questions.md): decision-oriented questions
  about metric selection, economics, safety and analysis units.

## Evidence and limits

The [2026-11-12 run](notes.md#executed-synthetic-study--2026-11-12) records
measured outputs separately from generating assumptions.
**Interpretation status: author review required.**
The constructed trade-offs demonstrate arithmetic and decision boundaries;
they establish no benefit, acceptable risk level or causal result on real users.

The conversion difference has an approximate two-sided 95% unpooled Wald
interval. Latency, cost, error and observed safety checks are descriptive:
falling within an illustrative point boundary establishes neither
non-inferiority nor rollout eligibility. The separate exact zero-incident
exercise assumes independent trials and perfect detection; the arm data contain
their own observed incident counts.

The historical textual example has no real safety evaluator, retention mechanism, repeated-request clustering,
timeout censoring, revenue attribution pipeline or full operating-cost model
is implemented. A canceled subscriber may still use a free product; churn and
activity labels are generated independently. The visual lab models activity
and subscription trajectories separately, uses a configurable synthetic detector,
and adds uncertainty-aware policy checks; neither implementation validates a
real safety evaluator or authorizes deployment.

For all Day 47 tests, from the repository root:

```powershell
python -m pytest -q 03-statistics-experimentation/47-product-metrics-guardrails/tests
```

The lab's conversion intervals combine exact binomial arm bounds conservatively.
The policy additionally allocates a family error budget across five checks and
uses known bounded-cost assumptions. These differ deliberately from the
historical example's Wald interval and point checks. Repeatedly inspecting a
sample path or tuning thresholds after seeing results is exploratory, not valid
sequential inference. No CI, cross-browser or production validation is implied.

The main lessons are to retain the intended denominator, wait for outcome
maturity, distinguish activity from subscription loss, and assess contribution
and uncertainty alongside the primary metric. Tests verify these implementation
contracts without approving their business meaning.

Continue from [Day 45: A/B inference](../45-ab-testing/) and
[Day 46: prospective power](../46-power-analysis/), or return to the
[module index](../README.md).

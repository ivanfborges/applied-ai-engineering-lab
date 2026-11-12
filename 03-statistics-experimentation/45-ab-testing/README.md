# Day 45 — A/B Testing from Scratch

[Module index](../README.md) · [Roadmap](../../ROADMAP.md)

Does a new AI workflow improve user task completion, beyond improving offline
ranking scores? This study turns that question into a randomized design and a
fixed-horizon analysis. It covers experimental units, metric denominators,
intention-to-treat (ITT), pooled two-proportion testing, uncertainty and sample
ratio mismatch (SRM). All observations here are **synthetic**.

The educational implementation and local verification are documented below.
**Interpretation candidates require author review.** These simulations do not
establish product benefit or readiness to launch.

The separate [offline visual learning lab](VISUAL_LAB.md) includes twelve
locally verified modules: standalone HTML, PNGs and five process animations.
Its Python 3.13 execution record, offline browser checks and resource limits
are documented separately from the earlier interrupted run. Generated assets
remain ignored pending author selection of public previews.
Its synthetic product guardrails are a separate experiment from the original
binary-outcome study; they are not production measurements.

## Design before inference

| Decision | Contract used in the example |
|---|---|
| Treatment | Hypothetical replacement of an assistant workflow |
| Population and unit | 30,000 eligible synthetic users; one assignment and one binary outcome each |
| Assignment | Independent Bernoulli allocation with treatment probability 0.5; seed 45 |
| Primary metric | Target task completed within a hypothetical 24-hour window, divided by **all assigned users** in each arm |
| Generating assumptions | Control probability 0.10; treatment probability 0.11; independent outcomes |
| Hypotheses | $H_0:p_B=p_A$ versus $H_1:p_B\ne p_A$; two-sided $\alpha=0.05$ |
| Horizon | Enroll the fixed sample and conceptually wait for every outcome window to mature; one final analysis |
| Planning | 80% target power for a 1 percentage point absolute effect, balanced arms, normal approximation |
| Practical value | Illustrative minimum worthwhile improvement of 0.5 percentage points; distinct from the planning MDE |
| Validity diagnostic | Exact binomial allocation test; investigate if SRM $p<0.001$ (illustrative prespecified threshold) |
| Guardrails | Latency, serving cost and safety must be specified for a real test; **not simulated or evaluated here** |

The time window is a metric specification, not a modeled event clock. Every
generated user has a complete outcome. A production service also needs stable
assignment across visits, exposure and outcome logging, eligibility checks and
protection against interference. Reproducible random draws alone provide none
of those guarantees.

## Execute from the repository root

Use stable Python 3.11+ and the canonical dependencies in
[pyproject.toml](../../pyproject.toml):

```bash
python -m pip install -e ".[dev]"
python 03-statistics-experimentation/45-ab-testing/from_scratch.py
python 03-statistics-experimentation/45-ab-testing/example.py
python -m pytest -q 03-statistics-experimentation/45-ab-testing/tests/test_ab_testing.py
```

For the visual extension, start with a lightweight run, then generate the full
lab. Run one generation at a time; see the visual guide for controls and limits.

```powershell
python 03-statistics-experimentation/45-ab-testing/visualize_ab_testing.py --quick --no-gifs
python 03-statistics-experimentation/45-ab-testing/visualize_ab_testing.py --all --quality light
```

The first script uses only the standard library and illustrates **assumed**
counts, 1,500/15,000 versus 1,650/15,000. The main example generates user data
with NumPy, compares its pooled Z-test with SciPy's uncorrected Pearson
chi-square test, checks allocation using `binomtest`, then runs 2,000 independent
fixed-horizon repetitions per configuration: A/A at 15,000 users per arm and
A/B at 1,000, 5,000 and 15,000 per arm. Repetitions use binomial counts with
exactly balanced group sizes; they are separate experiments, never repeated
looks at the same experiment.

Optionally retain a new local report (choose a new filename for every run):

```bash
python 03-statistics-experimentation/45-ab-testing/example.py --seed 45 --repetitions 2000 --output 03-statistics-experimentation/45-ab-testing/outputs/my-run.json
```

`outputs/` is ignored by the repository. The script refuses to overwrite an
existing report. It generates no plots or public assets. The curated dated
results are in [notes.md](notes.md); new executions do not replace that record.

## Read the evidence in order

1. Investigate broken assignment, incomplete windows or inconsistent logging
   before interpreting an effect. A large SRM p-value does not prove validity.
2. Report $\hat p_B-\hat p_A$ in percentage points, relative lift and the interval.
   The test uses **pooled null variance**; the separate Wald interval uses
   **unpooled estimated variance**. They are not exact inverses of each other.
3. A p-value is neither the probability that the null is true nor a launch rule.
   Compare uncertainty with the minimum worthwhile effect and guardrails.
4. A/A rejection frequency has Monte Carlo uncertainty. Even calibration near
   5% under this ideal generator does not validate a real experimentation stack.

The core refuses sparse tables with fewer than five observed successes or
failures in either arm. This is a conservative teaching heuristic, not proof
of valid normal inference. Wald intervals can under-cover or extend beyond
the feasible difference range; they are deliberately not clipped. Use suitable
score/exact methods for sparse data. Clustered events, ratio metrics, missing
outcomes, sequential testing and multiple comparisons need additional methods.

## Files and study route

| File | Purpose |
|---|---|
| [from_scratch.py](from_scratch.py) | Validated count inputs, pooled Z-test, unpooled Wald interval and balanced planning approximation |
| [example.py](example.py) | User-level randomization, ITT counts, SRM, SciPy agreement and A/A/power repetitions |
| [notes.md](notes.md) | Estimand, formulas, failure modes, dated observations and pending interpretations |
| [tests/test_ab_testing.py](tests/test_ab_testing.py) | Numerical references, symmetry, invalid data, assignment, SRM and simulation contracts |
| [interview_questions.md](interview_questions.md) | Design and interpretation questions for AI systems |
| [references.md](references.md) | Primary sources and official numerical documentation |
| [visualize_ab_testing.py](visualize_ab_testing.py) | CLI for twelve offline visual modules, PNGs, GIFs, HTML and per-run records |
| [ab_visual_core.py](ab_visual_core.py) | Reproducible mechanisms, normal power, paired peeking and user-level Welch inference |
| [VISUAL_LAB.md](VISUAL_LAB.md) | Navigation, mathematical interpretation, parameters, limitations and preview recommendations |
| [tests/test_visual_lab.py](tests/test_visual_lab.py) | Independent numerical checks and temporary-directory rendering tests |

For a 45–60 minute session, read the design, derive the two standard errors,
run both scripts and the tests, then explain what would still block a real
launch. The central lesson is to connect a well-defined randomized intervention
to a measured user outcome with uncertainty and operational constraints.

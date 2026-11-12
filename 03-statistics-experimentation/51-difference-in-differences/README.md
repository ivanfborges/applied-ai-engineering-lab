# Day 51 — Difference-in-Differences

How much did a treated group change beyond the change in an untreated group?
Difference-in-Differences (DiD) uses that contrast to estimate an average
treatment effect on the treated (ATT), **when its identification assumptions
are credible**. It can support evaluations of a nonrandom AI-workflow rollout,
but a post-rollout improvement alone does not establish causality.

This study connects the four-mean estimator to unit/time fixed effects and
team-clustered uncertainty. It then challenges parallel trends using a known
synthetic differential trend and a placebo restricted to genuine pre-periods.
All outcomes are **synthetic productivity points**, with higher values meaning
higher productivity. They are neither minutes nor percentages, and no real
dataset, deployed assistant or business benefit is measured.

## The contrast and its boundary

$$
\widehat{\tau}_{DiD} =
(\bar Y_{T,post}-\bar Y_{T,pre})-(\bar Y_{C,post}-\bar Y_{C,pre}).
$$

For hypothetical means of 80 → 105 in treatment and 100 → 110 in control,
the contrast is 15 points despite different starting levels. The causal step
is assuming that, without treatment, the treated group's expected change
would match the control group's change. Fixed effects cannot establish this.

The executable example deliberately supports a complete balanced panel, a
single common adoption date, stable team membership and never-treated controls.
It rejects missing cells, duplicates and changing group membership. This narrow
contract makes the manual/TWFE equivalence interpretable; it does not implement
staggered-adoption estimation or repair selective missingness.

## Run from the repository root

Use Python 3.11+ and the canonical extras from [pyproject.toml](../../pyproject.toml):

```powershell
python -m pip install -e ".[dev,experimentation]"
python 03-statistics-experimentation/51-difference-in-differences/example.py
python -m pytest -q 03-statistics-experimentation/51-difference-in-differences/tests/test_did.py
```

## Interactive visual laboratory

The separate [visual lab guide](VISUAL_LAB.md) explains seven interactive
chapters, the 120-team/12-period generator, synthetic truth, failure modes and
the visual-phase experiment record. The original analytical example below
retains its 100-team/6-period configuration and dated results.

From this topic directory, with your Python environment active:

```powershell
pip install -r requirements.txt
streamlit run app.py
```

The [dependency entry point](requirements.txt) selects the repository's canonical
`experimentation` extra; it does not duplicate version bounds. Run GIF export
separately:

```powershell
python export_gifs.py
```

This creates `assets/did_counterfactual_animation.gif`,
`assets/parallel_trends_violation.gif` and `assets/visual_run.json` locally.
All exports remain ignored pending author preview selection. No generated
asset is linked as a versioned preview. Existing exports are protected; use a
new `--out` directory or deliberately pass `--overwrite` to regenerate.

| Visual source | Responsibility |
|---|---|
| [app.py](app.py) | Seven Streamlit tabs, shared controls, bounded caches and on-demand sampling |
| [simulation.py](simulation.py) | 120-team synthetic panel, true no-rollout outcomes and explicit violations |
| [estimation.py](estimation.py) | Manual/TWFE comparison, clustered intervals, event study, pre-only placebo and sampling |
| [charts.py](charts.py) | Interactive Plotly 2D, animated estimator geometry and optional 3D |
| [export_gifs.py](export_gifs.py) | Independent, headless Matplotlib/Pillow GIF generation |
| [tests/test_visual_lab.py](tests/test_visual_lab.py) | Numerical identities, covariance comparisons and temporary-directory GIF export |
| [tests/test_app.py](tests/test_app.py) | Headless tab, parameter, failure-mode and sampling controls |

For the visual tests, install `.[dev,experimentation]` from the repository root,
then run these files separately to preserve the repository's test isolation:

```powershell
python -m pytest -q 03-statistics-experimentation/51-difference-in-differences/tests/test_visual_lab.py
python -m pytest -q 03-statistics-experimentation/51-difference-in-differences/tests/test_app.py
```

### Original analytical script

The script prints four rows: baseline rollout, baseline pre-placebo,
nonparallel rollout and nonparallel pre-placebo. Each includes manual and TWFE
estimates, treated/control changes, a team-clustered standard error and 95%
interval. It writes no output files. The same seed is used across scenarios;
the known slope change is isolated from simulation noise.

## Reading the evidence

The [dated study record](notes.md#executed-synthetic-study--2026-11-12) records
the executed configuration, versions, numerical observations and limitations.
**Interpretation candidates require author review.** Numerical agreement checks
an implementation identity, not the credibility of an observational design.
A small placebo contrast cannot establish future untreated parallel trends.

| File | Purpose |
|---|---|
| [example.py](example.py) | Four-mean arithmetic, panel validation, synthetic generation, TWFE and diagnostics |
| [notes.md](notes.md) | Identification derivation, inference choices, design risks, run record and interview questions |
| [tests/test_did.py](tests/test_did.py) | Hand calculations, invariances, regression agreement, trend contamination and invalid inputs |
| [references.md](references.md) | Primary methodological and statsmodels sources |

The key lessons are to distinguish ATT from general rollout effects, level
differences from trend differences, and uncertainty in an estimate from bias
in its causal interpretation. Real evaluations also need treatment/exposure
audits, a stable outcome population, concurrent-change checks and quality/cost
guardrails. The code is educational and does not supply those design guarantees.

[Module index](../README.md) · [Roadmap](../../ROADMAP.md)

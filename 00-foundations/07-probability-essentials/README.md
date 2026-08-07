# Probability Essentials

## Overview

Probability is the language used to reason about uncertain events and noisy
observations. This topic connects event algebra, conditional probability,
independence, expected value, variance, and Bayes' theorem to decisions in Data
Science and AI Engineering.

The practical example uses a deterministic synthetic fraud scenario to show a
counterintuitive but important result: a detector with high sensitivity can
still produce mostly false-positive alerts when fraud is rare. The code
compares the theoretical Bayesian posterior with a Monte Carlo estimate and
saves a chart of the base-rate effect.

## Concepts Covered

- Sample spaces, events, complements, unions, and intersections
- Probability axioms and addition rules
- Conditional probability and the multiplication rule
- Independence, conditional independence, and mutual exclusivity
- Law of total probability and Bayes' theorem
- Random variables, expected value, variance, and standard deviation
- Covariance and variance of sums
- Population versus sample variance
- Base rates, calibration, and expected-loss decisions

## Why It Matters

Probability supports probabilistic classification, anomaly and fraud
detection, A/B testing, model calibration, reliability analysis, threshold
selection, and cost-sensitive decisions. It also helps decompose multi-stage AI
systems: for example, answer quality can be studied conditionally on successful
retrieval rather than hidden inside one aggregate score.

Correct formulas are not sufficient on their own. Their conclusions depend on
representative data, stable probabilities, defensible independence assumptions,
and a decision rule that reflects operational costs.

## Files

- `notes.md`: intuition, theory, formulas, assumptions, trade-offs,
  applications, limitations, and common mistakes.
- `example.py`: synthetic fraud-alert simulation, theoretical and empirical
  results, and a base-rate visualization.
- `from_scratch.py`: educational implementations of expected value, variance,
  conditional probability, and a binary Bayes update.
- `visual_lab.py`: nine-section interactive Streamlit probability laboratory.
- `visualizations.py`: reusable, tested probability and Plotly helpers.
- `generate_visual_assets.py`: offline generator for the PNG and GIF learning
  assets.
- `VISUAL_GUIDE.md`: concept, controls, interpretation, production connection,
  and interview takeaway for every visual.
- `tests/test_probability_functions.py`: unit tests for the centralized
  probability calculations and deterministic simulation.
- `interview_questions.md`: senior-level conceptual, mathematical, and
  production questions with answers.
- `references.md`: books, course material, and official library documentation.

## How to Run

From the repository root, install the shared dependencies if needed:

```bash
python -m pip install -e ".[dev]"
```

Run the main experiment:

```bash
python 00-foundations/07-probability-essentials/example.py
```

The script uses a fixed random seed and writes:

```text
00-foundations/07-probability-essentials/outputs/base_rate_effect.png
```

Generated outputs are ignored by the repository-level `.gitignore`. To display
the chart as well as save it:

```bash
python 00-foundations/07-probability-essentials/example.py --show
```

Run the first-principles utilities:

```bash
python 00-foundations/07-probability-essentials/from_scratch.py
```

Use `python .../example.py --help` to inspect options for the number of
transactions, random seed, and output directory.

## Key Takeaways

- \(P(A \mid B)\) and \(P(B \mid A)\) answer different questions; Bayes'
  theorem relates them but does not make them equal.
- Posterior probabilities depend on base rates. Sensitivity and false-positive
  rate cannot be interpreted without prevalence.
- Independence is a strong modeling assumption, especially when services or
  signals share causes.
- Expected value is a long-run probability-weighted average, not necessarily a
  typical or even observable outcome.
- Variance describes instability that the mean hides; production decisions
  often also require percentiles and tail-risk analysis.
- A classification threshold should reflect calibrated probabilities, error
  costs, capacity, and policy—not default automatically to 0.5.

## Visual Probability Lab

The visual lab turns the Day 7 formulas into an interactive technical
laboratory. It includes:

- a 100-outcome event grid for unions, intersections, and complements;
- a denominator-change animation for conditional probability;
- theoretical and simulated 2×2 joint distributions for dependence;
- a four-view Bayes experiment with a population grid, confusion matrix,
  probability flow, and interactive 3D surface;
- expected value as weighted contributions and a balance point;
- same-mean distributions with different spreads, squared deviations, and the
  Bernoulli variance curve;
- reproducible Monte Carlo convergence paths;
- expected-cost curves and the implied decision threshold;
- production connections to calibration, RAG, shared infrastructure failure,
  LLM quality variance, and prior shift.

All data and metrics are synthetic. The application and asset generator make no
network requests and require no external dataset, API, ImageMagick, or FFmpeg.

### Create a virtual environment

Run these commands from the repository root:

```bash
python -m venv .venv
```

### Activate on Windows PowerShell

```bash
.venv\Scripts\Activate.ps1
```

### Activate on Linux or macOS

```bash
source .venv/bin/activate
```

### Install dependencies

The repository-level `pyproject.toml` is the canonical dependency source and
contains NumPy, pandas, Matplotlib, Plotly, Streamlit, and Pillow:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Move into the topic folder:

```bash
cd 00-foundations/07-probability-essentials
```

### Generate static assets

```bash
python generate_visual_assets.py
```

The generator creates:

```text
outputs/
├── event_operations.png
├── conditional_probability.gif
├── independence_comparison.png
├── bayes_base_rate.gif
├── bayes_surface.png
├── expected_value_balance.png
├── variance_spread.gif
├── bernoulli_variance.png
├── monte_carlo_convergence.gif
└── expected_cost.png
```

Every GIF is written with Matplotlib's `PillowWriter`. The script uses the
non-interactive `Agg` backend, closes figures after saving, and prints each
generated path.

### Start the interactive application

```bash
streamlit run visual_lab.py
```

Streamlit normally opens the application in the default browser automatically.
Select a concept in the sidebar, change its controls, and compare the updated
metrics with the formula shown above the chart.

### What to observe

- In the conditional-probability animation, the denominator changes before the
  intersection becomes the numerator.
- In the independence tool, only \(\delta=0\) makes
  \(P(A\cap B)=P(A)P(B)\).
- In the Bayes lab, changing prevalence can move alert precision dramatically
  even when detector sensitivity and false-positive rate stay fixed.
- In the variance animation, the mean remains fixed while dispersion grows.
- Monte Carlo paths stabilize with sample size but continue to fluctuate.
- The expected-cost threshold follows \(p^*=C_R/C_M\), not a universal 0.5.

See `VISUAL_GUIDE.md` for a visualization-by-visualization study guide.

### Run tests

From the topic folder:

```bash
python -m unittest discover -s tests
```

### Troubleshooting

- **An animation is missing:** run `python generate_visual_assets.py`. The app
  shows an actionable message instead of crashing when an asset is absent.
- **Streamlit is not found:** activate the repository virtual environment and
  run `python -m pip install -e ".[dev]"` from the repository root.
- **A browser does not open:** use the local URL printed by Streamlit, normally
  `http://localhost:8501`.
- **Port 8501 is busy:** run
  `streamlit run visual_lab.py --server.port 8502`.
- **Matplotlib reports a GUI backend problem:** use the asset generator rather
  than running plotting functions interactively; it selects the `Agg` backend.
- **Asset generation seems slow:** four GIFs are rendered frame by frame. On
  the verified environment, the full deterministic suite completes in roughly
  one to two minutes; runtime varies by machine.

# Day 52 — Uplift Modeling

[Module index](../README.md) · [Roadmap](../../ROADMAP.md)

Which customers benefit **because of** an intervention? A response model
estimates who is likely to convert; an uplift model estimates how treatment
changes that probability. A high baseline response can coexist with low or
negative uplift. This distinction matters for retention offers, proactive
support and limited AI-assisted interventions.

The target is the conditional average treatment effect (CATE):

$$
\tau(x)=E[Y(1)-Y(0)\mid X=x].
$$

Two Random Forest classifiers form a T-Learner:
$\hat\tau(x)=\hat\mu_1(x)-\hat\mu_0(x)$. The example compares its ranking with
predicted **control response** and a one-feature stratified estimator. Known
randomization probabilities support holdout inverse-propensity-weighted (IPW)
gain and incremental net-value evaluation. Neither a classifier nor IPW
automatically identifies effects in confounded observational data.

## Executable study

All data are **synthetic**. There is no downloaded dataset, live campaign or
measured AI product benefit. Each repetition generates 12,000 independent
customers, three pre-treatment features, 50/50 treatment assignment and one
binary outcome per customer. A fixed 60/40 split separates training and
evaluation. Models, bins and policy rules never use holdout outcomes or oracle
CATE. Random Forest settings and a 30% capacity are fixed in advance.

The heterogeneous scenario includes positive and negative effects. The
homogeneous control has exactly +5 percentage points of CATE; noisy fitted
rankings can still appear to find subgroups. Both scenarios use ten independent
replications by default. Sample standard deviations and Monte Carlo standard
errors describe variation across complete refits, not individual CATE intervals.

From the repository root, in an activated stable Python 3.11+ environment:

```powershell
python -m pip install -e ".[dev]"
python 03-statistics-experimentation/52-uplift-modeling/from_scratch.py
python 03-statistics-experimentation/52-uplift-modeling/example.py
python 03-statistics-experimentation/52-uplift-modeling/example.py --scenario homogeneous
python -m pytest -q 03-statistics-experimentation/52-uplift-modeling/tests/test_uplift.py
```

Dependency bounds come from [pyproject.toml](../../pyproject.toml); this topic
needs no specialized causal framework or optional runtime extra. The CLI
prints configuration, versions and numerical summaries. To retain every
replication, pass `--output` with a new filename under this topic's ignored
`outputs/` directory. Existing files are refused to preserve prior records.
The analytical command generates no plots. Use the separate visual generator
below for the interactive laboratory.

## Interactive visual laboratory

From this topic directory, run:

```powershell
python visualize_uplift.py
```

This creates and opens `visualizations/index.html`: nine offline modules with
interactive 2D charts, synchronized rotatable 3D surfaces, four GIFs and
controls for response/effect ranking, sample size, targeting, costs, capacity
and overlap. The HTML embeds Plotly, data and images and works without a
server or connection. Use `--no-open` for headless generation and `--no-gifs`
to skip exports while retaining the interactive modules.

See [VISUAL_LAB.md](VISUAL_LAB.md) for the walkthrough, mathematical contract,
configuration, dated measured results and validation evidence. The visual
phase uses a separate two-feature generator and leaves the historical
analytical record unchanged. Exports remain ignored; no public preview is
selected. **Author interpretation review remains pending.**

## Read the evidence correctly

Gain is incremental positive outcomes **per eligible person**, including
untargeted people in the denominator. All ranking endpoints estimate the same
holdout ATE. AUUC uses trapezoidal integration over targeted fractions;
`qini_area = AUUC - IPW_ATE/2` subtracts the random-targeting line. Tie blocks
are interpolated as random order within each tie; these conventions may differ
from a library's Qini normalization.

The predeclared policies include top-30% uplift, top-30% control response,
positive predicted net value with a 30% cap, treat-all, treat-none and the
expected random-30% policy. Illustrative outcome value is 100 currency units
and each contact costs 5. All contacts incur that cost, including unsuccessful
ones. Scores rank people; observed randomized outcomes evaluate decisions.
The script never chooses a winner, treatment fraction or threshold on test
outcomes. Oracle CATE RMSE and oracle policy values are synthetic diagnostics.

See the [dated run record](notes.md#executed-synthetic-study--2026-11-12)
for actual configuration, results and limitations. **Interpretation status:
author review required.** Local verification does not approve interpretations.

## Files and next questions

| File | Purpose |
|---|---|
| [example.py](example.py) | Synthetic RCT, T-Learner, fixed policies, repeated holdout comparison and optional JSON |
| [from_scratch.py](from_scratch.py) | Training-only quantile strata, IPW scores, tie-aware gains and equal-cost policies |
| [tests/test_uplift.py](tests/test_uplift.py) | Exact IPW expectations, gain normalization, costs/capacity, bin boundaries and holdout isolation |
| [notes.md](notes.md) | Identification, formulas, learner trade-offs, executed record and unrun extensions |
| [interview_questions.md](interview_questions.md) | Questions about counterfactuals, ranking, uncertainty and operating a policy |
| [references.md](references.md) | Primary causal sources and implementation documentation |
| [visualize_uplift.py](visualize_uplift.py) | Generate and optionally open the self-contained offline dashboard |
| [uplift_visual_core.py](uplift_visual_core.py) | Separate synthetic DGP, S/T models, uncertainty and overlap mechanisms |
| [uplift_animations.py](uplift_animations.py) | Four bounded Pillow GIFs with static fallbacks |
| [visual_lab.html](visual_lab.html), [visual_lab.js](visual_lab.js) | Page template and interactive controls embedded during generation |
| [VISUAL_LAB.md](VISUAL_LAB.md), [tests/test_visual_lab.py](tests/test_visual_lab.py) | Visual guide, measured record and focused numerical/rendering tests |

The stratified estimator teaches a conditional difference in means; it does
not replace a causal inference library. Its quantile boundaries come from
training data, repeated cuts collapse, and missing treatment arms in a bin
raise an error. It only conditions on one feature and can miss heterogeneity.

CATE is a conditional expectation, not an observed individual effect. The
identification assumptions remain consistency, exchangeability, positivity,
pre-treatment features and no interference. These experiments establish no
calibration, fairness, real-world profitability or production readiness.

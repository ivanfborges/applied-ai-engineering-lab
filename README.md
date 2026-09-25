# Applied AI Engineering Lab

**English** | [Português](README.pt-BR.md)

A public, AI-assisted study curriculum and technical portfolio connecting
theory, executable code, visual experiments, interview preparation, and
production-oriented reasoning across Data Science and Applied AI Engineering.

[![Repository quality](https://github.com/ivanfborges/applied-ai-engineering-lab/actions/workflows/quality.yml/badge.svg)](https://github.com/ivanfborges/applied-ai-engineering-lab/actions/workflows/quality.yml)

## Current Status

**22 implemented studies: 15 foundations and 7 classical ML topics.** The 140-topic roadmap is a learning plan, not 140 delivered projects. Numerical implementations, reported observations and author-reviewed conclusions have different evidence requirements; see the [curation record](docs/CURATION.md).

The implemented material connects mathematical foundations to model evaluation
and inference through synthetic experiments and tested numerical code. Topics
16–22 cover the ML lifecycle, validation boundaries, linear regression theory
and implementation, regularization, logistic regression, and classification metrics.

Day 19 adds direct OLS and batch gradient descent implementations, scikit-learn
parity checks, and a measured feature-scaling experiment. Day 20 adds fold-local
regularization tuning, a tested proximal gradient solver, and measured shrinkage
and sparsity comparisons. Day 21 adds stable logistic loss, tested binary
optimization, conditional odds interpretation, and a measured comparison of
fixed classification thresholds. Its 14-view visual lab adds animated decisions
and optimization, offline 3D geometry, and measured representation and
calibration experiments. Day 22 adds tested confusion-matrix metrics, a measured synthetic
baseline comparison, and a nine-view visual lab spanning thresholds,
prevalence, and review capacity.

- Latest topic: [Classification Metrics I](01-classical-machine-learning/22-classification-metrics/)
- Current module: [Classical Machine Learning](01-classical-machine-learning/) — 7 of 20 topics completed
- Completed module: [Foundations](00-foundations/)
- Full plan: [140-day study roadmap](ROADMAP.md)

Planned modules are not presented as completed work. A module directory is
published when it contains an implemented study; interpretation-review status
is documented separately.

## Start with a technical question

| Question | Selected study | Evidence and boundary |
|---|---|---|
| How do data, validation and inference fit together? | [End-to-end ML pipeline](01-classical-machine-learning/16-end-to-end-ml-pipeline/) | Synthetic churn, fold-local preprocessing, threshold policy, artifact round-trip and tests. No live retention deployment. |
| Could the score be inflated by leakage? | [Validation and leakage](01-classical-machine-learning/17-validation-and-leakage/) | Random-label control, group boundaries and temporal label timing. Single-seed demonstrations, not general effect sizes. |
| What does a fitted linear model actually establish? | [Linear regression theory](01-classical-machine-learning/18-linear-regression-theory/) | OLS geometry, residual diagnostics and tests. Training identities do not establish generalization or causality. |
| Can I inspect the optimizer itself? | [Gradient descent from scratch](00-foundations/06-gradient-descent-from-scratch/) | NumPy implementation and convergence diagnostics on synthetic data. Educational implementation; no dedicated test suite yet. |
| Why is association insufficient for intervention? | [Correlation vs causation](00-foundations/13-correlation-causation/) | Known synthetic generators for confounding and selection. No causal identification on business data. |

For a short visit, read the selected study's question and limitations, then
inspect its code and tests. For the full curriculum, use the inventory below.
Interpretations in topics 16–18 explicitly marked for author review remain
pending; this curation does not approve them on the author's behalf.

For applied studies using external datasets, see the separate
[TopVistos classification study](https://github.com/ivanfborges/ML_olympiad_for_students-topvistos_EUA)
and [Rio Airbnb geospatial ML study](https://github.com/ivanfborges/eng_dados-analytics_engineering).

## Implemented studies

| Day | Topic | Main evidence |
|---:|---|---|
| 1 | [AI/ML/GenAI landscape](00-foundations/01-ai-ml-genai-landscape/) | Executable routing and retrieval example |
| 2 | [Vectors and matrices](00-foundations/02-linear-algebra-vectors-matrices/) | Visual explorer, GIFs, and unit tests |
| 3 | [Vector spaces, bases, and projections](00-foundations/03-vector-spaces-bases-projections/) | Projection implementations and visualization generators |
| 4 | [Eigenvalues, eigenvectors, PCA, and SVD](00-foundations/04-eigenvalues-eigenvectors-pca-svd/) | Visual laboratory, notebook, and numerical tests |
| 5 | [Calculus for ML](00-foundations/05-calculus-for-ml/) | Gradient checks and guided visual exploration |
| 6 | [Gradient descent from scratch](00-foundations/06-gradient-descent-from-scratch/) | NumPy optimizer and convergence diagnostics |
| 7 | [Probability essentials](00-foundations/07-probability-essentials/) | Streamlit laboratory, curated assets, and tests |
| 8 | [Probability distributions](00-foundations/08-probability-distributions/) | Dashboard, simulations, estimators, and visual assets |
| 9 | [Exploratory data analysis with statistical rigor](00-foundations/09-exploratory-data-analysis/) | Controlled synthetic experiment and tested descriptive-statistics core |
| 10 | [Sampling, bias, and variance](00-foundations/10-sampling-bias-variance/) | Repeated-sampling experiment and tested estimator diagnostics |
| 11 | [Central Limit Theorem and confidence intervals](00-foundations/11-clt-confidence-intervals/) | CLT and interval-coverage simulations with tested numerical helpers |
| 12 | [Hypothesis testing](00-foundations/12-hypothesis-testing/) | Interactive statistical lab, paired inference, GIF generators, and tested simulations |
| 13 | [Correlation vs causation](00-foundations/13-correlation-causation/) | Visual causality lab with tested confounding, Simpson reversal, collider, and intervention simulations |
| 14 | [Maximum likelihood estimation and MAP](00-foundations/14-maximum-likelihood-map/) | Tested Beta-Bernoulli estimators and from-scratch logistic MLE/MAP optimization |
| 15 | [Entropy, cross-entropy, and KL divergence](00-foundations/15-entropy-cross-entropy-kl-divergence/) | Visual lab, tested information measures, stable logit loss, and masked next-token NLL |
| 16 | [End-to-end ML pipeline](01-classical-machine-learning/16-end-to-end-ml-pipeline/) | Interactive lifecycle lab, leakage-safe training, validation policy, drift simulations, persisted inference bundle, and tests |
| 17 | [Train/validation/test split, cross-validation and leakage](01-classical-machine-learning/17-validation-and-leakage/) | Tested K-fold indices, development-only tuning, synthetic leakage controls, group boundaries, and temporal gaps |
| 18 | [Linear regression theory](01-classical-machine-learning/18-linear-regression-theory/) | 20-view visual lab, OLS geometry, animated optimization, residual diagnostics, Ridge bias-variance experiments, and numerical/app tests |
| 19 | [Linear regression from scratch](01-classical-machine-learning/19-linear-regression-from-scratch/) | Tested OLS and batch gradient descent, scikit-learn comparison, and a synthetic feature-scaling experiment |
| 20 | [Regularization: Ridge, Lasso and ElasticNet](01-classical-machine-learning/20-regularization/) | Fold-local tuning, tested proximal optimization, and synthetic shrinkage and sparsity comparisons |
| 21 | [Logistic Regression](01-classical-machine-learning/21-logistic-regression/) | Tested numerical core and 14-view visual lab with threshold/learning animations, offline 3D surfaces, and measured synthetic experiments |
| 22 | [Classification Metrics I](01-classical-machine-learning/22-classification-metrics/) | Tested metrics, synthetic baseline and threshold experiments, and a nine-view visual lab |

## Roadmap

| Sequence | Module | Days | Status |
|---:|---|---:|---|
| 0 | Foundations | 1–15 | **Complete — Day 15 completed** |
| 1 | Classical Machine Learning | 16–35 | **In progress — 7 of 20 topics completed** |
| 2 | Unsupervised Learning and Recommender Systems | 36–44 | Planned |
| 3 | Experimentation, Causality, and Product Thinking | 45–52 | Planned |
| 4 | Deep Learning | 53–62 | Planned |
| 5 | NLP, Transformers, and LLMs | 63–77 | Planned |
| 6 | Embeddings, Semantic Search, and RAG | 78–88 | Planned |
| 7 | Agents and Agentic Workflows | 89–100 | Planned |
| 8 | MLOps and LLMOps | 101–114 | Planned |
| 9 | AI System Design | 115–122 | Planned |
| 10 | Interview Preparation | 123–132 | Planned |
| 11 | Portfolio Projects | 133–140 | Planned |

See [ROADMAP.md](ROADMAP.md) for the complete day-by-day plan.

## Quick Start

Create and activate a virtual environment from the repository root:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Install the shared runtime, test, and notebook dependencies from the canonical
`pyproject.toml`:

```bash
python -m pip install -e ".[dev,notebooks]"
```

`requirements.txt` remains as a compatibility shortcut for the same command.

Run a lightweight example:

```bash
python 00-foundations/01-ai-ml-genai-landscape/example.py
```

Run a first-principles implementation:

```bash
python 00-foundations/06-gradient-descent-from-scratch/from_scratch.py
```

Run all repository quality checks:

```bash
python scripts/validate_repo.py all
```

The syntax, internal-link, test, and Streamlit smoke-test checks can also be
run separately with `syntax`, `links`, `tests`, or `apps` in place of `all`.

Start a local visual laboratory:

```bash
streamlit run 00-foundations/07-probability-essentials/visual_lab.py
```

Dependencies and version bounds are centralized in `pyproject.toml`. Topic
READMEs document execution commands but do not maintain independent dependency
lists.

## Study Format

Published topics are curated from a broader study session. Depending on the
subject, a topic may contain:

```text
README.md
notes.md
example.py
from_scratch.py
notebook.ipynb
interview_questions.md
references.md
tests/
visualizations/
```

Not every topic needs every file. The structure should follow the technical
question rather than force a template.

The intended evidence is:

- theory connected to executable behavior;
- first-principles implementations where they clarify internals;
- comparisons, failure modes, assumptions, and trade-offs;
- deterministic synthetic experiments without invented benchmarks;
- tests for reusable numerical logic;
- selected visual assets when they materially improve understanding;
- interview questions linked back to the underlying implementation;
- observations and conclusions from experiments that were actually run.

## Repository Layout

```text
applied-ai-engineering-lab/
├── 00-foundations/
├── 01-classical-machine-learning/          # published when started
├── 02-unsupervised-recommender-systems/    # published when started
├── 03-statistics-experimentation/          # published when started
├── 04-deep-learning/                       # published when started
├── 05-transformers-llms/                   # published when started
├── 06-rag-semantic-search/                 # published when started
├── 07-agents/                              # published when started
├── 08-mlops-llmops/                        # published when started
├── 09-ai-system-design/                    # published when started
├── 10-interview-preparation/               # published when started
└── portfolio-projects/                     # Days 133–140
```

The public Git tree may contain only the modules that already have content.
Future directories remain part of the roadmap without being presented as
finished work.

## Scope and Reproducibility

- Examples run locally and do not require paid services or credentials.
- Current studies use synthetic or code-defined data.
- Generated outputs are ignored by default; only selected documentation
  previews are versioned.
- Visual and educational scripts are not production-grade library
  replacements.
- Reported values are demonstrations, not benchmark claims.
- Deployment will be considered only when it adds technical evidence without
  requiring unnecessary ongoing cost.

## AI-Assisted Workflow

AI tools were used as copilots for research, drafting, code suggestions, and
editorial refinement. Automated checks validate specific software behavior; they do not establish
that every interpretation has been personally reviewed by the author. Items
explicitly marked for author review remain pending. The author remains
responsible for technical decisions and published conclusions.

See [Study and Publication Methodology](docs/methodology.md) for the public
workflow and curation principles.

## License

This repository is available under the [MIT License](LICENSE). The examples and
study material may be reused with attribution and without warranty.

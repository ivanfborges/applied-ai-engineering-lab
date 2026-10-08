# Curation and evidence record

[Português](CURATION.pt-BR.md) · [Repository entry](../README.md)

Curation dated September 18, 2026, based on source revision `b369445`.
Scope: select useful entry points, repair navigation and execution paths, and
make review status explicit. This does not rerun or replace historical
experiment records, certify every scientific statement, or change authorship.

Navigation update on October 8, 2026: the [ML module index](../01-classical-machine-learning/)
now covers implemented topics 16–35 and the root entry includes questions from
32, 33 and 35. The dated selection and validation below remain historical, not
the scope or results of the current checks. See the [validation contract](validation.md).

A separate [stable reference check on October 8, 2026](validation.md#stable-reference-check--october-8-2026)
covers the current review working tree. It does not replace the September
record or approve pending interpretations.

## Delegated technical review — October 8, 2026

The author explicitly delegated technical curation to the AI assistant. The
assistant reviewed nine interpretation items against the local study source,
metric definitions and existing public records, and adopted bounded wording
for the scopes below. **This was an AI-assisted review performed under delegation,
not a personal analysis performed by the author.** The author retains editorial
responsibility; that does not change who performed this review.

| Study / items | Reviewed scope | Decision and retained boundary |
|---|---|---|
| 32 · 32-A/B | [Representation comparison and arithmetic demonstration](../01-classical-machine-learning/32-feature-engineering/notes.md#executed-experiments) | Restrict the ranking comparison to the favorable generator and recorded split; distinguish new history from representation and regularization changes. Retain training-parameter reuse as an arithmetic check, not predictive benefit. |
| 33 · 33-A/B/C | [Recorded training strategies and threshold policies](../01-classical-machine-learning/33-imbalanced-data/notes.md#executed-experiment) | Retain the fixed-score threshold trade-off. Restrict balancing conclusions to measured AP/ROC-AUC/recall; keep fit-size/weight confounding and no test-selected winner. Higher Brier does not identify miscalibration or its mechanism. |
| 35 · 35-A/B/C/D | [Forest record](../01-classical-machine-learning/35-explainability-classical-ml/notes.md#executed-forest-experiment) and [linear control](../01-classical-machine-learning/35-explainability-classical-ml/notes.md#executed-linear-control) | Restrict group reliance to the perturbation protocol; distinguish importance quantities and the gap diagnostic from a support test. Retain reconstruction/agreement and the known-function control as numerical checks, not causal, stability or deployment validation. |

No experiments were rerun for this editorial review. Numerical tables,
configurations and environment records remain historical. The separate visual
experiments in 32/33/35 and interpretation candidates in other topics are outside
this review and retain their pending status. Reports produced by future script
runs start pending: reviewing a dated public interpretation does not pre-approve
new measurements, configurations or explanations.

## Selection

The five entry points cover complementary questions: lifecycle (16), evaluation
boundaries (17), model assumptions (18), numerical implementation (6), and
causal reasoning (13). Each has executable source and explanations. Topics 13 and 16–18 have
public tests; topic 6 has no dedicated automated test file in this revision
and is selected for code inspection, not as equally tested software.
Other studies remain in the complete inventory; none was deleted or downgraded
solely because it is educational. The curriculum remains a learning resource.

## Author review still pending

| Study | Where the candidate interpretations are recorded | Review decision still needed |
|---|---|---|
| 16 · Pipeline | [README experiment records](../01-classical-machine-learning/16-end-to-end-ml-pipeline/README.md), [notes](../01-classical-machine-learning/16-end-to-end-ml-pipeline/notes.md) | Confirm whether to adopt, narrow or rewrite the proposed explanations for the churn, leakage, search and drift simulations. Keep the synthetic-data and single-configuration limits. |
| 17 · Validation | [Notes](../01-classical-machine-learning/17-validation-and-leakage/notes.md), [visual guide](../01-classical-machine-learning/17-validation-and-leakage/VISUAL_GUIDE.md) | Review the explanations for feature-selection optimism, repeated-entity recognition and validation design; select any new public previews separately. |
| 18 · Regression | [Notes](../01-classical-machine-learning/18-linear-regression-theory/notes.md), [visual guide](../01-classical-machine-learning/18-linear-regression-theory/VISUAL_GUIDE.md) | Review coefficient recovery, residual orthogonality under misspecification, Ridge trade-offs and proposed public visual selections. |

These are review items already labeled in the source material. No personal
approval is inferred from test success or permission to edit the portfolio.
Before promoting a candidate to an author conclusion, record the decision,
its scope and any wording changes. Unlisted topics are not thereby certified
as scientifically reviewed in full.

## Repairs and validation scope

- The pipeline folder was renamed to `16-end-to-end-ml-pipeline`; its README
  commands, smoke-test registry, Git preview rules and profile links still
  used the previous name. They now target the implemented directory.
- Six pipeline image links pointed to absent files. The README now documents
  local generation instead of embedding unavailable previews. No figures or
  experimental values were regenerated for this editorial repair.
- English remains the technical entry; Portuguese navigation and methodology
  are provided in separate documents. Detailed topic files retain their language.
- Use `python scripts/validate_repo.py all` for syntax, local-link checks,
  isolated test files and headless smoke tests of registered Streamlit apps.
  Smoke checks do not certify every widget state, browser layout or notebook.
- Dependencies remain the declared ranges in `pyproject.toml`, not a frozen
  numerical environment. Historical runs retain their originally reported
  configurations; results from them are not represented as new measurements.

There is no real-data benchmark, live service, production impact or new
human interpretation approval in this curation delivery.

Verified on September 18, 2026: syntax for 133 Python files, 218 local links
in 85 Markdown files, 276 tests across 25 isolated files, and all six registered
Streamlit smoke checks passed on Windows with Python 3.11.14 and Streamlit 1.64.0.
These checks do not approve the interpretation candidates above.

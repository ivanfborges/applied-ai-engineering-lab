# Validation contract

Run commands from the repository root with **stable Python 3.11 or newer**.
The canonical dependency bounds are in [pyproject.toml](../pyproject.toml).
Ranges are not a frozen numerical environment: retain versions with measured
results, and do not silently replace historical experiment records.

## Choose the scope

```bash
python -m pip install -e ".[dev,imbalance,explainability,boosting]"
python -m pip check
python scripts/validate_repo.py syntax
python scripts/validate_repo.py links
python scripts/validate_repo.py tests
python scripts/validate_repo.py apps
```

`python scripts/validate_repo.py all` combines the last four checks and is the
local check sequence used by [CI](../.github/workflows/quality.yml). CI also
checks installed optional-library imports. Install the complete extras above
before interpreting a full-suite result; otherwise optional tests can skip.
Notebooks additionally use `.[notebooks]` and are not executed by this runner.

For a small change, run its specific test file, then syntax and links:

```bash
python -m pytest -q scripts/tests/test_validate_repo.py
```

## What the runner establishes

| Check | Contract | Boundary |
|---|---|---|
| `syntax` | AST-parses public Python source without compiling bytecode | Does not execute imports or algorithms |
| `links` | Checks local paths against tracked and non-ignored Git candidates | Does not check remote URLs, heading existence or rendered layout; checks HTML attributes inside Markdown, not standalone HTML |
| `tests` | Discovers public `test_*.py` and runs each file in a separate Python process | Passing assertions cover their stated cases, not every topic claim; inspect skips/warnings |
| `apps` | Runs the explicitly registered Streamlit entries through headless AppTest | Initial app smoke, not every interaction or browser appearance |

The [runner](../scripts/validate_repo.py) obtains publication candidates using
`git ls-files --cached --others --exclude-standard`. A local file does not
become a valid documentation target just because it exists under ignored outputs.

The plain `python -m pytest` default targets Foundations as configured in
`pyproject.toml`; it is **not the repository-wide command**. Repeated names such
as `example.py` and `from_scratch.py` can collide if all topic tests are imported
in one process. Use the isolated runner for the complete curriculum. Its own
[contract tests](../scripts/tests/test_validate_repo.py) protect that behavior.

Topic tests also cover selected app branches, including the Day 18 and Day 35
labs; this is separate from the registered `apps` smoke list. Neither mechanism
launches a server. Full visual exports, notebook execution, screenshot review,
Linux clean-install CI and scientific interpretation require separate evidence.

## Publication gate

Record the exact commands, environment, failures/skips and scope actually run.
Do not describe focused checks as the full suite. Do not infer author approval
from test success or represent synthetic measurements as production results.
See [methodology](methodology.md) and the [ML module](../01-classical-machine-learning/).

## Stable reference check — October 8, 2026

The AI assistant ran the four commands above on the review working tree on
Windows x64, using an isolated Python 3.11.14 environment with all four extras.
This includes the new tests from the review, not just the earlier committed
revision. Dependency compatibility and optional-library import checks passed.

- Syntax: 231 public Python files parsed.
- Local links: 659 targets checked across 162 Markdown files.
- Tests: 999 passed, plus 133 subtests, across all 61 discovered isolated files;
  no skips or failures. Three SHAP/Matplotlib pending-deprecation warnings.
- Apps: all six registered headless AppTest smoke checks passed. Bare-mode
  ScriptRunContext warnings were emitted; no app exceptions were found.

Key versions: NumPy 2.4.6, pandas 3.0.6, SciPy 1.17.1, scikit-learn 1.9.1,
Matplotlib 3.11.2, pytest 9.1.1, Streamlit 1.65.0, SHAP 0.50.0,
imbalanced-learn 0.14.2, XGBoost 3.2.0, LightGBM 4.7.0 and CatBoost 1.2.10.
This is a dated compatibility reference, not a dependency lock or replacement
for the environments recorded with historical experiments.

Tracked-file hashes remained unchanged throughout the run; the author's
original environment configuration and installed-package metadata were also
unchanged. Complete logs and the 66-package environment manifest are retained
as ignored local review records, not public assets. No historical tables or
tracked previews were regenerated. This run does not establish Linux CI,
notebook execution, browser/GitHub rendering, production readiness or personal
review of scientific interpretations.

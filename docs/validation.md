# Validation contract

Run commands from the repository root with **stable Python 3.11 or newer**.
The canonical dependency bounds are in [pyproject.toml](../pyproject.toml).
Ranges are not a frozen numerical environment: retain versions with measured
results, and do not silently replace historical experiment records.

## Choose the scope

```bash
python -m pip install -e ".[dev,imbalance,explainability,boosting,manifold,experimentation]"
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
The `manifold` extra supplies UMAP for Day 40; its integration smoke test is
required and does not skip when that dependency is absent.
The `experimentation` extra supplies statsmodels for the Day 46 visual lab,
Day 48 correction checks and Day 51 clustered inference; their relevant tests
require the dependency.
Notebooks additionally use `.[notebooks]` and are not executed by this runner.

For a small change, run its specific test file, then syntax and links:

```bash
python -m pytest -q scripts/tests/test_validate_repo.py
```

## Focused iteration and resource limits

Do not rerun the whole curriculum after every local change. Select a public
topic directory or test file; the runner still isolates each selected file:

```bash
python scripts/validate_repo.py tests --scope scripts/tests
python scripts/validate_repo.py tests --scope 03-statistics-experimentation/47-product-metrics-guardrails
python scripts/validate_repo.py tests --scope 02-unsupervised-recommender-systems/37-dbscan-density-based-clustering --exclude-rendering
```

`--exclude-rendering` requires an explicit test scope and deselects only tests
marked `rendering`. Seven real-export functions in Days 37/38/40/48 are initially
classified; other legacy rendering may be unmarked and remains eligible. This
is a **partial check**, not a guarantee of zero rendering or full coverage.
Numerical/integration checks in those visual test files remain eligible.
Read pytest's actual passed, skipped and deselected counts. An empty selection
fails instead of being presented as a successful validation.

Keep real-export tests when changing visual behavior. `tests` without a scope
and `all` retain every test, including rendering; `all` rejects selection flags.
Scope also works for `apps`, but only selects entries from its explicit registry.
For documentation-only edits, run `syntax` and `links`; for shared logic or
dependency changes, use wider regression coverage and the complete CI gate.

Test/app child processes default to one numeric thread per OpenBLAS, MKL,
OpenMP, NumExpr and Numba pool, with the Agg backend and Streamlit telemetry off.
Use `--threads 2` explicitly to request two. These settings are applied before
child imports without changing the parent shell; they are not a total CPU/RAM
cap. Do not add process-level parallelism on top without measuring resource use.
Avoid running installation, training and full visual exports concurrently.

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

## Browser review — November 13, 2026

The AI assistant checked existing local exports for Days 49/52 and the running
Streamlit apps for Days 44/46/47/51 using cached Playwright CLI 0.1.24 and
Chrome 157.0.8103.51 on Windows. HTTP servers bound only to loopback; external
page HTTP(S) requests were blocked, and Streamlit telemetry was disabled.
No application changes or historical-asset regeneration were needed for this
browser review. Implementation fixes elsewhere in the review have separate tests.

| Surface | Verified scope |
|---|---|
| Day 49 causal explorer | 55 scenarios, including 35 stored Python expectation cases, path/adjustment controls, keyboard operation, animation and real 3D camera interaction |
| Day 52 uplift explorer | 33 checks across nine modules: surface identities, ranking endpoints, fixed policies, overlap regimes, synchronized camera and selected mobile layout |
| Day 44 ranking app | Eight sections; rank swap updates RR@5; desktop and selected mobile rendering |
| Day 46 power app | Eight sections and reset; desktop and selected mobile rendering |
| Day 47 product-metrics app | Nine sections, confirmed-incident hard stop and reset; desktop and selected mobile rendering; no GIF-export button triggered |
| Day 51 DiD app | Seven tabs, keyboard-operated oracle toggle, spillover/anticipation modes and 3D rendering; selected mobile layout |

The completed runs had no JavaScript/console errors or external page requests.
Ten Plotly Canvas2D readback-performance warnings were observed. Headless Chrome
used software WebGL (SwiftShader): this is not a native-GPU performance test,
cross-browser review, GitHub rendering check or full accessibility audit.
Desktop viewport was 1440×1000; selected mobile views used 430×900.
Day 49 mobile was not checked. Scientific interpretations remain a separate review.

Initial harness attempts read an asynchronous metric too early, carried a mobile
viewport into another app and misidentified a native switch's interaction model.
The harness was corrected using rendered DOM and keyboard behavior; failed
attempts and completed-run logs are preserved separately as ignored local records.
Browser checks do not replace the canonical automated suite or Linux CI.

## Full reference check — November 14, 2026

The AI assistant ran `python scripts/validate_repo.py all` on the review working
tree on Windows x64 in a separate stable Python 3.11.14 environment. All six
extras in the installation command above were installed together. Dependency
compatibility and imports of imbalanced-learn, SHAP, XGBoost, LightGBM,
CatBoost, UMAP, statsmodels and NetworkX passed before the suite.

- Syntax: 355 public Python files parsed.
- Local links: 1,195 targets checked across 238 Markdown files.
- Tests: 2,214 passed, plus 133 subtests, in 103 isolated files; no failures or skips.
- Apps: all six registered headless AppTest smoke checks passed.

The tests emitted three SHAP/Matplotlib pending-deprecation warnings. The app
smokes emitted six bare-mode ScriptRunContext warnings and no app exceptions.
The canonical command took 960 seconds locally; this is validation duration,
not an application-performance benchmark.

Key versions: NumPy 2.4.6, pandas 3.1.1, SciPy 1.17.1, scikit-learn 1.9.1,
Matplotlib 3.11.2, pytest 9.2.0, Streamlit 1.67.0, Plotly 6.9.0,
imbalanced-learn 0.14.2, SHAP 0.50.0, XGBoost 3.2.0, LightGBM 4.7.0,
CatBoost 1.2.10, UMAP 0.5.12, statsmodels 0.15.0 and NetworkX 3.6.1.
The complete 72-distribution manifest and logs remain ignored local review records.

Tracked-file hashes were unchanged during validation. No historical tables,
notebooks or tracked previews were regenerated, and existing environments were
not modified. This dated compatibility reference is not a lockfile or a rerun
of historical experiments. Linux CI for the published revision is separate
evidence; this local run does not establish notebook execution, cross-browser
compatibility, production readiness or personal author approval.

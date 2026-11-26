# Structured Outputs: interactive visual learning lab

This companion to the [analytical study](README.md) covers thirteen technical
questions with six GIF animations, four PNGs and eight standalone Plotly HTMLs.
All inputs, prices, durations, probabilities, documents and permissions are
synthetic or explicitly assumed. There is no provider integration or live tool
execution. Interpretation candidates require author review.

## Execute locally

From the repository root, use Python 3.11+ and the existing canonical dependency
configuration. No new framework or model download is required:

```powershell
python -m pip install -e ".[dev,nlp]"
python 05-transformers-llms/77-structured-outputs/visualizations.py --list
python 05-transformers-llms/77-structured-outputs/visualizations.py --all
python 05-transformers-llms/77-structured-outputs/visualizations.py --demo decoding
python 05-transformers-llms/77-structured-outputs/visualizations.py --demo validation
python 05-transformers-llms/77-structured-outputs/visualizations.py --demo retries
python 05-transformers-llms/77-structured-outputs/visualizations.py --demo guardrails
python 05-transformers-llms/77-structured-outputs/visualizations.py --demo masking --preview --show
```

Without a demo selection the CLI renders everything. `--demo` may be repeated.
`--show` opens the finished local gallery; exports work headlessly without it.
Default output is a fresh topic-relative `outputs/visual-<UTC timestamp>/`.
`--output-dir outputs/my-run` selects a **new** topic-relative directory; absolute
paths are also supported. Existing directories are rejected to preserve run
history and public previews. An initial `--output-dir assets` works only when
that directory does not already exist; prefer ignored dated output directories.

`--seed 777` is the default. Full runs simulate 100,000 requests per retry model;
`--preview` defaults to 10,000 requests and 80 dpi instead of 110 dpi. The grammar
and validation fixtures are unchanged. `--samples N` overrides the Monte Carlo
count in either mode. Each selected renderer can run independently; the run
record retains the small shared calculation snapshot for all demonstrations.

Open the generated `index.html` or individual HTML files directly in a browser.
Each interactive artifact bundles Plotly and its local controls, requiring no
server, CDN, credentials or network connection. GitHub does not run an HTML file
inside a Markdown README; generate the full local artifacts to explore them.

## What to explore

| CLI demo | Generated artifacts | Learning question / interaction |
|---|---|---|
| `space` | `structured_output_space.html` | Rotate an explicitly illustrative population, filter categories through its legend and activate progressively stricter gates. Coordinates are not embeddings or probabilities. |
| `decoding` | `constrained_decoding.gif` | Inspect prefix, state, candidate probabilities, admissibility and actual seeded token selection at each step. |
| `masking` | `probability_redistribution.gif`, `interactive_token_masking.html` | Compare logits, probabilities, negative-infinity masks and renormalization. Edit temperature/logits/schema/mask; draw reproducible samples. An empty mask blocks generation explicitly. |
| `grammar` | `grammar_state_transitions.gif` | Follow valid transitions plus executed rejected trials for early EOS, missing fields and incompatible values. |
| `validation` | `validation_funnel.gif`, `validation_funnel.png` | Read actual entering/rejected/surviving counts, reasons, conditional pass rates and end-to-end acceptance. |
| `strictness` | `strict_vs_permissive_validation.png` | Compare coercion, errors and dropped extras for strict/permissive typing crossed with ignore/forbid policies. |
| `semantics` | `structural_vs_semantic_correctness.html` | Hover over explicit check scores, raw responses and known reference records; identify high structural scores with low semantic scores. |
| `retries` | `structured_output_retry_workflow.gif` | Follow first-pass success, format recovery, schema/business failures, exhaustion and permanent preflight rejection; inspect assumed accumulated cost/time. |
| `reliability` | `retry_reliability_curves.html` | Change p, total attempts and target reliability. Inspect diminishing returns and the minimum achievable integer budget. |
| `surface` | `retry_reliability_surface.html`, `retry_reliability_surface.png` | Rotate reliability over p and integer budgets; highlight 90%, 95% and 99% contours. |
| `cost` | `retry_cost_reliability_tradeoff.html`, `expected_retry_cost_surface.html` | Change success/cost/duration/target/budget; inspect feasible configurations, marginal reliability and expected cost. The 3D surface colors cost by reliability. |
| `correlation` | `independent_vs_correlated_failures.png`, `retry_failure_simulation.html` | Compare actual simulation with analytical success, exhaustion and expected attempts; switch the displayed metric. |
| `guardrails` | `production_guardrails_pipeline.gif` | Track synthetic records through validation, evidence and permission decisions, including regeneration, human review and a denied valid tool call. |

The [calculation core](visual_core.py), [static renderer](visual_render.py),
[HTML exporter](visual_html.py), [offline controls](visual_controls.js) and
[CLI](visualizations.py) have separate responsibilities. The original
`example.py` and `from_scratch.py` remain the analytical examples.

## Mathematical scope

For a permitted token set $A$ and original probabilities $p_v$, the masked
distribution is

$$
q_v=\begin{cases}
p_v/\sum_{u\in A}p_u,&v\in A,\\
0,&v\notin A.
\end{cases}
$$

The NumPy implementation normalizes permitted finite logits after shifting by
their permitted maximum. Forbidden logits are represented as `None` in JSON
records, displayed as negative infinity; they are never a fabricated finite
logit. Empty masks are errors. The HTML implements the same equations over
bounded finite inputs; browser checks compare controls with expected values.
Temperature is positive; a zero-temperature greedy policy is outside this lab.

The grammar has three required fields in a fixed order: `document_type` is
`invoice` or `receipt`, `amount` is one of the whole numeric toy tokens 1000,
1500 or 2000, and `currency` is BRL or USD. The finite vocabulary is intentionally
restricted. This is not a general JSON parser, character tokenizer or commercial
constrained decoder. Nested/recursive data needs richer grammar state. Grammar
admissibility is independent of source truth. Local renormalization also differs
from conditioning a whole original sequence on eventual validity.

For independent attempts with constant pass probability $p$, stop at first
success or at $n$ total calls:

$$
R_n=1-(1-p)^n,\qquad
\mathbb{E}[N]=\sum_{k=0}^{n-1}(1-p)^k.
$$

At $p=0$, success is zero and all $n$ calls are consumed. At $p=1$, one call
suffices. A positive target is impossible at $p=0$, and target 1 has no finite
solution for $0<p<1$. Numerical reliability uses `expm1`/`log1p` for stability.
For target $0<r<1$ and $0<p<1$,

$$
n_{\min}=\left\lceil\frac{\log(1-r)}{\log(1-p)}\right\rceil.
$$

All displayed budgets are integers. The 3D surfaces connect integer rows for
visual continuity; fractional retry counts have no operational interpretation.

Assumed constant cost $c$ and duration $t$ give expected request cost
$c\mathbb{E}[N]$ and expected request latency $t\mathbb{E}[N]$. Population cost per
accepted response is $c\mathbb{E}[N]/R_n$, undefined when $R_n=0$.
For $p>0$ this ratio simplifies to **$c/p$, independent of the retry budget**.
Retries improve request coverage but do not improve this ratio in the simplified
model. Fixed setup cost, variable attempts, correlated errors, validation overhead
and queueing would change the result. These are theoretical costs and durations,
not provider billing or measured performance.

For conditional pass probabilities $p_i$ given all preceding failures,

$$
R_n=1-\prod_{i=1}^{n}(1-p_i),\qquad
\mathbb{E}[N]=\sum_{k=0}^{n-1}\prod_{i=1}^{k}(1-p_i).
$$

The persistent mixture samples a permanent-failure label once per request, with
probability $f=0.25$. Other requests use $p_r=0.7$ independently each attempt.
Its first-call success equals the independent baseline, $(1-f)p_r=0.525$, but
its population success approaches $1-f=0.75$:

$$
R_n=(1-f)[1-(1-p_r)^n],\qquad
\mathbb{E}[N]=fn+(1-f)\sum_{k=0}^{n-1}(1-p_r)^k.
$$

The feedback-aware model uses configured conditional probabilities
`min(0.525 + 0.12 * (i-1), 0.95)`. This tests a changing-hazard model. It does
not measure the causal benefit of providing feedback to an LLM.

## Validation data and scores

The visual funnel uses 12 variations of each of nine authored response types,
for 108 synthetic records. Monetary fields are integer cents. The structural
stage uses strict field types and enums; domain bounds/identifier constraints
are checked separately by the full analytical `Invoice` model. Then arithmetic
and exact-reference equality are checked. Unreached stages are `None`, not
failures. Acceptance is the conjunction of all configured gates. Because the
fixtures have known answers, the evidence check is an oracle, not a general
inference-time grounding validator.

The semantic scatter uses **17 selected schema checks**: object shape, exact
key set, eight field-type checks, three literals/enums, three monetary bounds
and identifier format. This fraction is diagnostic, not an exhaustive count of
every Pydantic validator. Its **nine semantic checks** are eight typed exact-field
matches plus the arithmetic identity. Identical coordinates overlap without
jitter; hover/category details expose the original records. Neither axis is a
calibrated probability, confidence score or a claim about learned embeddings.

The strictness matrix compares nine cases under four policies: typing strictness
crossed with extra-field handling. A nullable note remains required. Dropped
extra fields are recorded separately from type coercions. Python-object validation
is used after parsing; strict JSON-input conversions can differ for other types.

## Execution record — 2026-11-26

**Hypotheses:** each deterministic gate rejects a distinct failure class;
local token masking preserves normalized support; a latent permanently failing
population limits retry recovery even at the same initial pass probability.

Full configuration: Windows CPU, Python 3.11.14, NumPy 2.4.6, pandas 3.1.1,
Pydantic 2.14.4, Matplotlib 3.11.2, Plotly 6.9.0 and Pillow 12.3.0. Seed 777;
100,000 simulated requests per model; integer budgets 1–10; 41 p values in
theoretical surfaces. The first complete run was recorded at
`2026-11-26T13:28:56.458618+00:00`. The original analytical run remains separate
in [notes.md](notes.md#execution-record--2026-11-26).

| Gate | Entering | Rejected | Surviving | Conditional pass | End-to-end |
|---|---:|---:|---:|---:|---:|
| Parse | 108 | 12 | 96 | 88.89% | 88.89% |
| Schema/types | 96 | 48 | 48 | 50.00% | 44.44% |
| Domain | 48 | 12 | 36 | 75.00% | 33.33% |
| Cross-field | 36 | 12 | 24 | 66.67% | 22.22% |
| Source reference | 24 | 12 | 12 | 50.00% | 11.11% |
| Acceptance | 12 | 0 | 12 | 100.00% | 11.11% |

| Retry model, n=10 | Analytical success | Simulated success | Simulated exhaustion | Analytical E[N] | Simulated mean N |
|---|---:|---:|---:|---:|---:|
| Independent | 99.94153% | 99.937% | 0.063% | 1.903648 | 1.90061 |
| Persistent mixture | 74.99956% | 75.107% | 24.893% | 3.571422 | 3.56070 |
| Feedback-aware | 99.999999993% | 100.000% | 0.000% | 1.688049 | 1.68428 |

Simulation points fluctuate around predictions: the observed mixture rate can
exceed its population ceiling in a finite sample. Zero observed failures also
does not establish failure impossibility. The tests compare errors with six
standard errors plus one discrete outcome for all 30 scenario/budget rows.
Displayed bars use the simpler normal approximation of +/-1.96 binomial SE.

The sampled 14-step toy decode ended in
`{"document_type":"receipt","amount":1500,"currency":"BRL"}` plus EOS.
Scripted retry examples accepted first-pass and repaired cases within their
three-call budget; two permanent preflight scenarios consumed zero model calls.
The valid synthetic `post_invoice` proposal with no permission did not reach
the simulated execution eligibility state. No external tool was invoked.

**Interpretation candidates — author review required:** structural restrictions
and source fidelity constrain different failure modes. Retry gains depend on
request-level failure structure; merely increasing the cap cannot recover the
permanent portion of this mixture. A valid proposed action still needs an
independent permission decision.

**Limitations:** authored case frequencies determine the funnel, not an estimated
LLM response distribution. One simulation seed, a configured mixture and manually
specified feedback probabilities do not establish real extraction accuracy,
causal feedback improvements, injection resistance or production reliability.
The architecture diagram is illustrative; only synthetic validation/permission
decisions are implemented. Regeneration is distinct from transport retries:
no exponential-backoff/jitter implementation or live recovery was evaluated.

## Verification and artifact policy

```powershell
python scripts/validate_repo.py tests --scope 05-transformers-llms/77-structured-outputs
python scripts/validate_repo.py syntax
python scripts/validate_repo.py links
git diff --check
```

[Core tests](tests/test_visual_core.py) cover grammar states, empty masks,
probability identities, coercion, stage counts, terminal permissions, budgets,
minimum attempts and analytical/Monte Carlo agreement.
[Export tests](tests/test_visual_rendering.py) write real temporary images and
standalone HTML, check CLI errors and preserve existing output directories.
Browser interaction and image inspection are separate from these automated
checks; neither is evidence of scientific author approval or production readiness.

`experiment.json` retains hypothesis, configuration, versions, results,
interpretation candidate and limitations. `manifest.json` records SHA256 hashes
of calculation/rendering sources, experiment records and artifacts. Files are
dated and not overwritten. Galleries, complete HTMLs, alternate PNG/GIFs and
experiment records remain ignored. Selected previews are curated separately
after inspection, with their exact sizes and verification scope recorded below.

## Selected previews and final verification

Final export: `2026-11-26T15:09:34.597658+00:00`, using the same seed and full
simulation settings. Earlier exports remain intact. The final source/artifact
hashes were verified against its manifest. All eight final interactive HTMLs
were byte-identical to the browser-audited exports; this was checked by SHA256.

| Public preview | Bytes | Resolution / progression | Selection rationale |
|---|---:|---|---|
| [constrained_decoding.gif](assets/constrained_decoding.gif) | 594,814 | 1408 x 792; 14 frames, 15.4 s | Shows every actual sampled state, including numeric masks and EOS. |
| [validation_funnel.png](assets/validation_funnel.png) | 78,184 | 1408 x 792 | Compact, computed counts with conditional and end-to-end denominators. |
| [retry_reliability_surface.png](assets/retry_reliability_surface.png) | 179,947 | 1408 x 792 | Lightweight GitHub preview of the local interactive 3D relationship. |
| [production_guardrails_pipeline.gif](assets/production_guardrails_pipeline.gif) | 3,505,487 | 1408 x 792; 76 frames, 64.6 s | Traces distinct validation, evidence, authorization, regeneration and exhaustion paths. |

Total selected images: 4,358,432 bytes (4.16 MiB). The guardrails GIF deliberately
exceeds the repository's 1 MiB per-GIF review target: all 76 states preserve the
successful, semantic-failure, missing-evidence, permission-denied, repaired and
three-attempt exhaustion paths. It is linked, not embedded in the README.
Removing these states would hide the different stopping decisions. The other
inline previews total 672,998 bytes. The eight standalone HTMLs bundle roughly
4.9 MB each and remain local; this selection uses a lightweight static surface
preview instead of adding a large HTML to the public set.

Actual automated verification on the final implementation:

- 141 topic cases passed across three isolated files: 54 analytical, 69 visual
  core and 18 export/CLI checks; no skips, warnings or failures.
- The full CLI generated 18 artifacts: six GIFs, four PNGs and eight HTMLs,
  plus the gallery and separate JSON experiment/manifest records.
- All selected files were decoded and representative animation states were
  visually inspected, including the numeric-value step and terminal guardrail
  branches. Copies match the generated originals by SHA256.
- 37 browser assertions passed across all eight HTMLs in Chrome headless at
  1500 x 1100, using offline `file:` URLs and software WebGL. Checks exercised
  schema/mask/logit/temperature controls, seeded sampling, empty masks, gate
  counts, camera rotation/reset, boundary probabilities, integer budgets,
  minimum attempts, feasible cost budgets and simulation metrics.
- No JavaScript/console errors and no external HTTP(S) requests were observed.
  Seven performance warnings occurred: three Canvas2D readback warnings and
  four WebGL ReadPixels stall warnings. These do not measure native GPU speed.
- Repository syntax: 575 Python files parsed. Internal links: 2,111 targets in
  348 Markdown files passed. `node --check` accepted the offline controls.
  `uv --no-cache --offline pip check --python <stable-interpreter>` verified 93
  installed packages; `git diff --check` passed. These are local checks, not a
  complete repository test-suite or Linux CI run.

The first browser inspection exposed an automatically inferred numeric token
axis; the final controls explicitly use a categorical axis and passed the
five-category regression assertion. Earlier failed inspection outputs remain
separate from the completed browser checks.

Browser verification establishes the inspected desktop flow only. Mobile,
cross-browser behavior, a complete accessibility audit, native GPU performance,
GitHub rendering and scientific author review were not established.

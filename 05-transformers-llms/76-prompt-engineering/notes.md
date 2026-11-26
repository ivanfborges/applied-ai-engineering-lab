# Prompt specifications, evaluation and failure boundaries

## Conditioning rather than parameter updates

For fixed model parameters $\theta$, instructions/examples $p$, document $x$
and generated tokens $y$, autoregressive generation factors as

$$
P_\theta(y\mid p,x)=\prod_{t=1}^{T}P_\theta(y_t\mid p,x,y_{<t}).
$$

Changing the prompt changes conditioning, without training the weights. Ordinary
few-shot prompting is in-context adaptation; fine-tuning changes parameters or
adapters. Retrieval changes the evidence supplied in context. The three techniques
can be combined, but each should address an observed need. Day 75's decoding
controls determine how scores become output tokens; prompt design determines
part of the information on which those scores depend.

Prompt selection can be written as an empirical decision problem:

$$
p^*=\arg\min_{p\in\mathcal P}
\mathbb E_{(x,y^*)\sim\mathcal D,\;\hat y\sim P_\theta(\cdot\mid p,x)}
[L(\hat y,y^*)].
$$

The distribution and objective matter. Repeatedly editing against the same tiny
development set overfits that set. A prompt's apparent improvement can depend on
wording, example order, selected model and decoding settings. Lower output entropy
or self-reported confidence does not imply better factual accuracy. No output
probabilities or entropy are measured in this implementation.

## Six components tied to one task

| Component | Decision in this study | Failure it makes explicit |
|---|---|---|
| Instructions | Extract an identifier and final payable total | Vague requests for “important information” |
| Context | Supplied invoice text only | Unsupported knowledge or inferred totals |
| Examples | Three distinct complete, missing and conflicting cases | Unspecified edge-case behavior |
| Constraints | Whole USD, explicit evidence, no calculations or conversion | Silent inference and unit errors |
| Format | Exactly two typed keys | Prose, coercion and unexpected fields |
| Failure handling | Field-specific nulls and bounded structural retries | Invented values or endless retries |

V1 intentionally leaves ambiguity for later empirical comparison. V2 expresses
the policy. V3 illustrates it, at additional context cost. No evidence here says
that longer prompts or demonstrations improve a particular model. The baseline
must ultimately be scored against the same references as all other variants.

Provider message roles differ. Keep trusted application policy separate from
documents and tool results when adapting these logical request channels. A string
inside a document cannot grant permission to use a tool or access another tenant.
JSON-encoding that string preserves its bytes and helps inspection; it does not
enforce a trust boundary inside an LLM. External authorization and restricted tools
remain application responsibilities, consistent with the
[OWASP guidance](https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html).
Our single authored injection response is a scoring example, not an attack test
against a model or a security evaluation.

## Four distinct questions

1. Can strict JSON parsing read the response? Duplicate keys and nonstandard
   NaN/Infinity constants are rejected as part of this study's parsing policy.
2. Does the parsed object satisfy the extraction contract? Python treats booleans
   as integers in some contexts, so the total check uses `type(total) is int`.
3. Does the output match independently authored reference values? This is an
   offline semantic score, not an online oracle.
4. May downstream software act on it? Real acceptance would require source
   verification, domain checks and an explicit operational policy.

Prompted JSON, provider-constrained decoding and application validation address
different mechanisms. Constrained decoding can improve structural reliability
without proving that a value occurs in the source. This study implements only the
application validation boundary; it does not invoke a provider's decoding feature.

## Metric definitions

All rates use all $N$ cases, including malformed responses and operational errors.
We require exactly one response per case; missing, extra or duplicate records
raise an error rather than silently changing the denominator. In a real run,
record a failed request explicitly instead of dropping it.

Let $S_i$ indicate schema validity, $J_i$ strict parseability, $\hat y_i$ the
validated object, $y_i^*$ its reference, and $F=2$ fields. Invalid responses earn
zero field credit. Equality includes correct nulls, without type coercion.

$$
A_{\mathrm{exact}}=\frac{1}{N}\sum_i S_i\,\mathbb 1[\hat y_i=y_i^*],
\qquad
A_{\mathrm{field}}=\frac{1}{NF}\sum_i S_i\sum_{f=1}^{F}
\mathbb 1[\hat y_{if}=y_{if}^*].
$$

JSON and schema rates are $\sum_i J_i/N$ and $\sum_i S_i/N$. Operational errors
are counted separately and receive zero JSON/schema/accuracy credit. This is an
end-to-end denominator policy; it differs from accuracy conditional on successful
API calls. Field accuracy can be inflated by many easy missing fields, so inspect
categories and non-null fields before using it to select a model.

Define $C_i=1$ only when the response satisfies the schema and both fields are
non-null. The reported complete-output coverage and selective risk are

$$
\mathrm{coverage}=\frac{\sum_i C_i}{N},\qquad
R_{\mathrm{complete}}=\frac{\sum_i C_i\mathbb 1[\hat y_i\ne y_i^*]}{\sum_i C_i}.
$$

Risk is undefined (JSON null in the report) when no output is complete. A correct
partial extraction counts toward exact match but not complete-output coverage.
This is not calibrated confidence thresholding. Completeness does not establish
correctness; a fully fabricated object can be complete. An all-null strategy can
reduce coverage to zero while failing complete-document extraction.

## Bounded retries cannot verify meaning

`request_with_validation` calls a supplied generator at most `max_attempts` times.
It retries a timeout or invalid structure and stops at the first schema-valid
object, including partial/all-null objects. Unexpected exceptions propagate so
programming and configuration defects are visible. The demonstration feeds an
invalid string followed by a valid object with a wrong total. The controller
returns that wrong object; only the offline reference scorer can identify the
mismatch here.

The controller does not change instructions, repair responses, enforce deadlines
or implement provider backoff. Repeating the same call can repeat the same error.
A production design would classify retryable failures and bound both calls and
elapsed time. Semantic verification or human review remains a separate step.

## Execution record — 2026-11-26

The AI assistant ran the default `example.py` on Windows with Python 3.11.14,
at `2026-11-26T12:37:06.660353+00:00`. This is a deterministic evaluator control,
not a prompt-comparison experiment. Runtime dependencies are standard library
only; there is no SDK/model version, sampling configuration or random seed.

**Hypothesis:** a schema-only stopping rule can return an incorrect extraction,
and separate structural/semantic metrics will expose the difference.

**Configuration:** nine authored synthetic development documents, two output
fields, one fixed response per document, no API requests and no generation.
Six cases have schema-valid responses, including three deliberately wrong
objects. One simulated operational failure remains in all-case denominators.
The separate retry control uses a two-attempt budget: invalid prose followed
by an object with the wrong total. There is no holdout or statistical sampling.

| Diagnostic | Observed result on authored fixtures |
|---|---:|
| Strict JSON validity | 7/9 = 77.78% |
| Schema validity | 6/9 = 66.67% |
| Exact match | 3/9 = 33.33% |
| Field accuracy, invalid outputs earn zero | 9/18 = 50.00% |
| Simulated operational errors | 1 |
| Complete-output coverage | 4/9 = 44.44% |
| Incorrect among complete outputs | 3/4 = 75.00% |
| Retry control | Stopped after 2 calls at total 999999 for B22; reference total 100 |

The ignored local record is named
`fixture-evaluation-20261126T123706660353Z.json` under `outputs/`; its per-case
rows and input SHA-256 hashes support local inspection. Input hashes were
`0db05c3d569cfee90f95c7c4a8f0b0c17fe3606411a488edf420e7a853c644a7`
(documents) and
`ba7d344a787a60b722173fcf3d20e737835f48adb94821b8929ee03dc6c9d5a5`
(responses). Each new invocation creates a distinct dated record and leaves
earlier records intact.

**Interpretation candidate — author review required:** on these designed controls,
structural validity and completeness cannot substitute for factual verification.
The retry controller has the demonstrated stopping behavior and does not detect
the semantic error.

**Limitation:** the failure rates were determined by authored responses. They are
not empirical LLM accuracy, an estimate of natural failure frequency, prompt
injection success or evidence that any of V1/V2/V3 is superior. No model, provider
latency, cost or production outcome was measured. This is an executable
counterexample to schema-only acceptance, not a model benchmark.

## Proposed real-model evaluation (not executed)

Freeze prompt text, model identifier/version, API/SDK version, supported decoding
settings and an evaluation manifest. Select prompts on development cases; reserve
independent holdout documents until selection is frozen. Keep examples out of both
evaluation sets. Randomize or counterbalance request order and repeat each
configuration to examine variation; temperature zero alone is not evidence of
repeatability.

Use the same cases and scorer for V1/V2/V3. Preserve raw responses, refusal/timeout
labels, per-attempt latency, token usage and retry counts in private logs. Report
first-attempt and post-retry metrics separately, including all failed calls. Record
actual billing/configuration before estimating cost. Quantify paired uncertainty
only with a justified sample; nine designed cases cannot support a population
benchmark. Verify references independently, including how ambiguous documents
should be labeled.

Then consider single-component ablations, example-order perturbations and
provider-constrained generation as separate comparisons. Structured generation
changes another variable and should not be attributed to prompt wording alone.
Expand fractional/currency/OCR/contradiction cases before generalization claims.
No such model experiment, latency measurement or business validation ran here.

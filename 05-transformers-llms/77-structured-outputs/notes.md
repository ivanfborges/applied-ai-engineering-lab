# Structured representation and justified acceptance

## Four predicates, four different questions

For a response $y$ and source $x$, define application acceptance as

$$
A(y,x)=J(y)\land S(y)\land B(y)\land G(y,x).
$$

$J$ is the serialization policy, $S$ the typed contract, $B$ the business rules
and $G$ the configured source check. Authorization for downstream effects is an
additional application decision. This conjunction describes acceptance by the
configured checks; imperfect checks can still accept false information.

In the implementation, the stages are observable and sequential. An output
rejected by the parser is not counted as a schema failure; later checks were
not reached. Arithmetic failure is distinguished from field/type failure.

JSON Schema describes properties and constraints on JSON instances. A property
listed under `properties` becomes mandatory only through `required`.
`additionalProperties: false` rejects undeclared fields. A required nullable
field must appear, with either an allowed value or `null`.
See the [official object reference](https://json-schema.org/understanding-json-schema/reference/object).

The `Invoice` model generates this structural contract through
`model_json_schema()`. Pydantic strict mode rejects numeric strings, floats and
booleans for the integer monetary fields when validating the parsed Python
object. `extra="forbid"` makes drift visible. Integer cents avoid binary decimal
rounding for this example; currency, units, bounds and business assumptions
must still be specified. These three currencies use two decimal minor units;
this is not a universal monetary representation.

JSON Schema considers a number with zero fractional part an integer; a JSON
`100000.0` can satisfy its `integer` constraint, while this Python contract
rejects the parsed float. A provider's accepted schema and local strict types
therefore need compatibility tests. The exported schema also does not encode
our arithmetic or exact-reference equality checks. No exported schema was
submitted to a provider in this study.

`purchase_order: str | None` has no default and is required. Null means the order
reference is unavailable; it does not mean zero, an empty string or a missing
JSON property. The known money fields are deliberately non-nullable. A wider
extraction product should explicitly represent unknown amounts and review
states instead of asking the model to invent them.

## Parsing is an application boundary

Python's default JSON parser permits non-standard constants and keeps the last
value for duplicate keys. Our policy rejects both, including duplicates inside
nested objects. It also rejects float overflow such as `1e999`, using a finite
float hook. This avoids differences between consumers silently changing the
meaning of a response. See [Python's documented parser behavior](https://docs.python.org/3.11/library/json.html#standard-compliance-and-interoperability).

The 16,384-character cap is an educational local guardrail. It is not a byte,
token, CPU or wall-clock limit. Real adapters should limit incoming payloads
before allocation, impose nesting/transport limits and enforce a request deadline.
No automatic removal of fences, dropping of extra fields or coercion of amounts
is performed: normalization would be a separate, explicitly evaluated policy.

## Prompting, validation and constrained generation

| Mechanism | Responsibility | Remaining concern |
|---|---|---|
| Prompt format instruction | Requests a representation | Generation can violate it |
| JSON-oriented generation mode | Aims at JSON syntax under its documented conditions | A specific application schema may remain unenforced |
| Post-generation validation | Detects unsupported representations or values | Detection does not generate a replacement |
| Constrained decoding | Restricts continuations using a supported grammar/contract | Completion, semantic fidelity and application rules remain separate |

These mechanisms can coexist. Provider implementations support different schema
subsets and completion behavior. This offline implementation makes no provider
integration claim. Structured tool arguments also require validation; the tool
name and arguments alone grant no authority.

## First principles: local token masking

For logits $z_v$ and nonempty permitted token set $A_t$, the implemented step is

$$
q_t(v)=\begin{cases}
\dfrac{\exp(z_v-m)}{\sum_{u\in A_t}\exp(z_u-m)}, & v\in A_t,\\
0, & v\notin A_t,
\end{cases}
\qquad m=\max_{u\in A_t} z_u.
$$

The maximum is taken over permitted logits before exponentiation. A forbidden
extreme maximum must not underflow every permitted weight. Finite extreme input
differences can overflow to negative infinity in float64; their exponentials
correctly become zero, while at least one maximum retains weight one.
The function requires a one-dimensional finite array and an actual boolean mask.
An empty permitted set raises an explicit error.

This is one synthetic step with a supplied mask. A real grammar decoder must
derive $A_t$ from prefix state, tokenizer-specific continuations and remaining
termination conditions. A syntactically viable prefix can still exceed a token
budget before completion. Local masking is generally **not** sampling the
original sequence distribution conditioned on eventual validity: that conditional
distribution also weights each continuation by its future probability of valid
completion. The toy mask contains no evidence that `100` is a truer value than
`200`.

## Bounded retries and terminal decisions

The controller accepts a callback receiving an attempt number and fixed feedback
code. The demonstration callback returns prewritten responses and ignores the
feedback. It tests orchestration, not whether a model understands or improves
from that feedback. It does not repair values or disclose the reference to the
callback. An incorrect but internally consistent reference match failure is
terminal rather than an opportunity to tune responses against an answer key.

Parsing, type/field and arithmetic failures are retryable here. Refusal,
incomplete envelopes, reference mismatch and the response-length guardrail are
terminal. Stopping on incomplete output is this example's conservative policy,
not a universal provider requirement. A real adapter may support safe bounded
regeneration, but should avoid concatenating incomplete JSON blindly.

No broad exception handler retries programming, transport, configuration or
permission errors. Implement transport retries separately, classifying transient
errors and respecting shared call/time/cost budgets. Repeating a tool action
requires an idempotency strategy; a valid extraction should not by itself trigger
a payment or database mutation. No tool execution or authorization subsystem is
implemented here.

With independent attempts and constant pass probability $p$, a maximum of $n$
**total attempts** gives

$$
P(\text{success within }n)=1-(1-p)^n,
\qquad
\mathbb{E}[N]=\sum_{k=0}^{n-1}(1-p)^k.
$$

For $p>0$, this expectation is $[1-(1-p)^n]/p$. At $p=0$, it is $n$; at $p=1$,
it is one. With constant attempt cost $c$, expected generation cost is
$c\mathbb{E}[N]$. These are theoretical statements, not measured probabilities
or billing estimates in this study. Correlated errors and validation feedback
make the constant-independent model questionable for real LLM requests.

If $q_k$ denotes success on attempt $k$ **conditional on all previous failures**,
and the policy continues after every failure up to $n$, the more general
success probability is $1-\prod_{k=1}^n(1-q_k)$. Terminal failures such as refusals
require additional states; the simple formula does not describe this entire
controller. Passing a validator also differs from being factually correct.

## Evaluation and common traps

Use a frozen held-out collection of documents and references for an actual
extraction evaluation. Report first-call and final contract compliance,
field/reference accuracy, missing-value behavior, attempts, terminal outcomes
and latency/cost across **all requests**. Declare the denominator for every
metric; accuracy among accepted outputs hides rejected and exhausted requests.
Also measure incorrect accepted records and coverage so that rejecting everything
cannot appear to solve extraction.

An exact known reference is available only because these fixtures are synthetic.
In RAG, retrieved document-ID membership can check provenance but not whether a
passage entails a claim. In document extraction, OCR spans and deterministic
cross-checks can help but remain fallible. Schema compliance, source membership,
model confidence and authorization must not be substituted for one another.

Keep schema versions alongside model/prompt/validator versions. Adding a new
required field breaks older payloads unless an explicit migration is introduced.
Strictness should preserve legitimate uncertainty, not just maximize rejections.

## Execution record — 2026-11-26

**Hypothesis:** separate gates will distinguish malformed syntax, wrong types,
inconsistent arithmetic and fabricated values; the configured terminal policy
will prevent additional calls even if a later fixture would be correct. A local
mask will zero forbidden probabilities and normalize the permitted pair.

**Configuration:** Windows CPU; Python 3.11.14, Pydantic 2.14.4, NumPy 2.4.6;
pytest 9.2.0 for verification. Five hand-authored synthetic response sequences,
four total calls permitted per sequence. The single reference is `SYN-77`,
schema `1`, type `invoice`, currency `BRL`, subtotal 100,000 cents, tax 5,000,
total 105,000 and null purchase order. No stochastic draws, seed, model, document
corpus or credentials are involved.

Executed from the repository root using the stable local interpreter:

```text
python 05-transformers-llms/77-structured-outputs/example.py --save
python 05-transformers-llms/77-structured-outputs/from_scratch.py
```

The invoice run was timestamped `2026-11-26T13:10:42.738008+00:00`. Its uniquely
named JSON record remains ignored; the observed results below are the intentional
public record, without linking to an ignored artifact.

| Synthetic sequence | Observed stages | Calls | Final state |
|---|---|---:|---|
| Repair | Parse failure, schema failure, arithmetic failure, accepted | 4 | Accepted |
| Fabricated subtotal/total | Contract and arithmetic pass; reference mismatch | 1 | Rejected; correct second fixture unused |
| Repeated empty objects | Schema failure on every call | 4 | Exhausted |
| Refusal envelope | Provider refusal | 1 | Refusal; correct second fixture unused |
| Incomplete envelope with valid-looking JSON | Incomplete status checked before parsing | 1 | Incomplete |

The masking input was `[1.2, 0.8, 2.0, -0.2, 0.1]` for synthetic tokens `100`,
`200`, quote, `null`, `true`; only the first two were permitted. Their original
combined mass was `0.3732284348`. The constrained distribution was
`[0.5986876601, 0.4013123399, 0, 0, 0]`.

**Interpretation candidate — author review required:** the observed stages are
consistent with a distinction between representation, internal consistency and
reference fidelity. Budgeted retries can expose recovery paths without allowing
terminal cases to continue. Masking redistributes representation probability,
with no factual correctness signal.

**Limitations:** fixture ordering determines every outcome; the repair sequence
would occur regardless of the feedback text. One synthetic reference makes
exact comparison unusually easy. These observations do not estimate LLM success,
causal retry benefits, general grounding accuracy, authorization, throughput,
latency, cost or adversarial robustness. Author review remains pending.

## Suggested follow-ups — not executed

The later [visual companion](VISUAL_LAB.md) executes separate synthetic coercion
fixtures, a restricted whole-token grammar and Monte Carlo retry models.
Those runs do not replace this historical analytical record or constitute the
held-out/provider experiments suggested below.

- Compare strict and permissive Pydantic models on the same held-out payloads;
  count changed values separately from accepted records.
- Add an explicit unknown-amount envelope and check missing-data coverage without
  allowing a partially populated financial record to reach a consumer.
- Evaluate a real provider's supported schema subset and completed/refused/
  incomplete responses under a separately approved integration scope.
- Vary retry budgets on a fixed held-out model evaluation; report request-level
  recovery, false acceptance, cost and latency rather than fixture frequencies.
- Add a small JSON-prefix grammar and tokenizer integration as a separate study;
  the current softmax function is not such a decoder.

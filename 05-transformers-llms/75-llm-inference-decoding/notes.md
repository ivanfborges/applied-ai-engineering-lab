# Decoding mechanics and their limits

## Model, policy and sequence

A causal model defines conditional probabilities, while a decoder defines a
possibly different sampling policy $q$. With prompt $x$ and output $y$:

$$
P(y_{1:n}\mid x)=\prod_{t=1}^{n}P(y_t\mid x,y_{<t}).
$$

Changing an early selected token changes later contexts. A single fixed-logit
experiment isolates one decision; it cannot describe full language generation.
The transition fixtures here expose this dependency without training a model.

Greedy decoding maximizes each conditional probability, not the joint sequence
probability. For a complete two-step tree, let $P(A)=0.6$, $P(B)=0.4$,
$P(X\mid A)=0.55$, $P(Y\mid A)=0.45$, $P(X\mid B)=0.95$ and
$P(Y\mid B)=0.05$. Greedy chooses $AX$ with probability $0.33$, but
$BX$ has probability $0.38$. This is a constructed calculation, not an LLM run.
Beam search approximates sequence search with additional compute; it is not
implemented here and does not guarantee factual correctness.

## Temperature and entropy

For finite logits $z_i$ and $T>0$:

$$
p_i(T)=\frac{\exp((z_i-z_{\max})/T)}
{\sum_j\exp((z_j-z_{\max})/T)},\qquad
\frac{p_i(T)}{p_j(T)}=\exp((z_i-z_j)/T).
$$

Subtracting the maximum prevents positive exponential overflow. Extremely
negative centered scores may underflow to zero; this implementation accepts
that finite-precision limit. It rejects NaN and infinite input logits rather
than supporting masked `-inf` inputs. Sorting uses original logits, preserving
the intended ordering when small probabilities underflow or round to ties.
The maximal weight stays one, so normalization always has positive mass.

In exact arithmetic, positive temperature preserves rank. As $T\to\infty$,
the distribution tends to uniform; as $T\to0^+$, mass concentrates uniformly
on all tied maxima. The separate `temperature=0` branch chooses only the first
maximum by vocabulary index. It is a convention, distinct from that tied limit.
Some libraries choose greedy through a sampling flag instead; the conventions
here should not be copied as an API compatibility claim.

Entropy in bits is $H(p)=-\sum_{i:p_i>0}p_i\log_2 p_i$.
For fixed finite logits, before filtering:

$$
\frac{dH}{dT}=\frac{\operatorname{Var}_{p(T)}(z)}{T^3\ln 2}\geq0.
$$

This identity follows from differentiating the log partition function.
It concerns uncertainty over tokens. It says nothing about whether a high
probability continuation is true, grounded or useful. Filtering introduces
discrete support changes, so interpret the unfiltered identity separately.

## Candidate filtering

Let $S_k$ contain the first $k$ logits in descending order, with lower token ID
winning ties. Top-k sets other probabilities to zero and divides retained
entries by $\sum_{j\in S_k}p_j$.

For descending probabilities $r_{(i)}$ **after top-k renormalization**, top-p
keeps the smallest prefix satisfying

$$
m=\min\left\{s:\sum_{i=1}^{s}r_{(i)}\geq\alpha\right\},
\qquad 0<\alpha\leq1.
$$

The threshold-crossing token is included and probabilities are renormalized
again. At least one candidate remains. With $\alpha=1$ no nucleus filtering
occurs. For positive $T$, top-k selects exactly $k$ indices, but finite-precision
underflow can leave fewer than $k$ entries with positive mass. The experiment's
candidate counts refer to **positive numerical support**.

For probabilities $(0.5,0.3,0.2)$, top-k=2 gives $(0.625,0.375,0)$.
Top-p=0.6 then keeps one token. Applying top-p=0.6 to the original distribution
keeps two. Order and renormalization are part of the contract, not incidental
details. Nucleus sampling is motivated in [Holtzman et al.](https://arxiv.org/abs/1904.09751);
library controls are documented separately in the
[Transformers generation reference](https://huggingface.co/docs/transformers/main_classes/text_generation).

## Generation and stopping contract

`generate` receives a callable over immutable context IDs and one finite logit
per vocabulary fragment. It validates parameters before generation, including
at a zero budget. Prompt IDs are copied and affect the model; prompt text is
not returned or searched for stops. Every iteration queries fresh logits.

1. Sample one ID and count it toward `max_new_tokens`.
2. If it is `eos_id`, finish with `eos` and omit the EOS fragment from text.
3. Append a non-EOS fragment and check the whole generated string for stops.
4. At the first step with complete matches, cut before the earliest character
   position and finish with `stop_sequence`. Stop-list order does not matter.
5. Otherwise continue, or finish with `length` at budget exhaustion.

EOS and completed stops take precedence over budget exhaustion on the same
step. Zero budget returns empty text and `length`, with no model call. The
returned `sampled_ids` include EOS and the entire fragment that triggered a
stop, even if visible text ends earlier. This is a diagnostic trace, not an
API billing estimate. A partial stop prefix at budget exhaustion is returned.

Matching means the first **observed completed** match; the loop does not wait
for an unfinished longer stop that might start earlier. Stop matching supports
cross-fragment strings and discards any suffix after the match in that step.
A streaming service would need to buffer possible stop prefixes to avoid
leaking text already delivered. Tokenizer decoding can change prior text and
is not equivalent to concatenating these synthetic fragments.

## Reproducibility and application decisions

Greedy decoding removes random selection given fixed logits and tie rules.
Seeded sampling controls a pseudorandom stream given the same calls and
arguments. Neither controls model updates, floating-point kernels, batching,
hardware or provider routing. NumPy documents strict conditions for
[random stream compatibility](https://numpy.org/doc/stable/reference/random/compatibility.html);
PyTorch likewise documents [reproducibility limits](https://docs.pytorch.org/docs/stable/notes/randomness.html).
Different seeds are allowed to produce identical outputs. Repeating a seed
after changing the number or shape of RNG calls is a different protocol.

| Question | What to evaluate beyond the decoder |
|---|---|
| Is extraction valid? | Schema, required fields, numerical/business rules, truncation and retries |
| Is RAG grounded? | Retrieval coverage, evidence support, abstention and representative held-out questions |
| Is generation diverse? | Multiple prompts/seeds, duplicates, useful variation and task success |
| Is inference reproducible? | Model/tokenizer revision, prompt, parameters, seed, runtime, hardware and exact outputs |
| Does a budget improve serving? | Output lengths, TTFT, per-token latency, tail latency, failures and cost per successful request |

Prefill processes prompt positions and can initialize a KV cache; autoregressive
decode then extends context one token at a time. This loop recomputes a toy
callable and implements neither attention nor caching. Lower temperature does
not reduce the model's forward-pass complexity. A smaller output cap can
truncate JSON or increase retries; a stop string is not schema enforcement.
Tune configurations on development prompts and reserve held-out evaluation.
No universal task-specific temperature ranges are asserted by these fixtures.

## Execution record — 2026-11-26

Executed locally by Codex at **2026-11-26 10:42:35 UTC** (07:42:35 in
America/Sao_Paulo), using Python **3.11.14**, NumPy **2.4.6**, Windows 10,
float64 calculations and `default_rng`/PCG64. No model training or external
inference was performed. Command from the repository root:

```sh
python 05-transformers-llms/75-llm-inference-decoding/example.py --seed 42 --samples 10000
```

The complete record is locally retained as
`outputs/run-20261126T104235733199Z.json` (ignored, not a public link).
Later executions receive new timestamps and do not replace this record.
**All interpretation candidates below remain author review required.**

### 1. Temperature and empirical frequencies

**Hypothesis:** increasing positive temperature increases unfiltered entropy
for fixed nonuniform logits. **Configuration:** logits are the natural logs
of $(0.85,0.07,0.04,0.03,0.01)$; no filters; 10,000 independent draws per
temperature. Each temperature resets PCG64 to seed 42, so comparisons share
an initialization rather than independent streams across conditions.

| Temperature | Maximum theoretical probability | Entropy (bits) | Maximum absolute frequency error |
|---:|---:|---:|---:|
| 0.2 | 0.999996 | 0.000081 | 0.000004 |
| 0.7 | 0.951368 | 0.356784 | 0.000590 |
| 1.0 | 0.850000 | 0.871810 | 0.002500 |
| 1.5 | 0.676153 | 1.514436 | 0.005847 |
| 2.0 | 0.555483 | 1.845767 | 0.003917 |

**Interpretation candidate:** theoretical entropy increased across the sweep;
the largest observed frequency discrepancy was about 0.00585 in absolute
probability. **Limitation:** one seed and sample size do not estimate a
convergence rate or general error bound. At T=0.2 all observed samples chose
the first token, despite nonzero theoretical probabilities for the others.
There is no factual-confidence measurement.

### 2. Filtering and interactions

**Hypothesis:** nucleus support adapts to concentration and preceding filters.
**Configuration:** the peaked vector above and a five-token uniform vector;
temperatures 0.7/1/2, top-k disabled or 3, top-p 0.5/0.9/1: 36 calculated
combinations, without random draws.

**Result:** at T=1 and top-p=0.9 without top-k, the peaked vector retained two
candidates and 0.92 original mass; the uniform vector retained all five and
1.00 original mass. With top-k=3 before the same nucleus filter, the uniform
vector retained three candidates and only 0.60 original mass.

**Interpretation candidate:** top-p=0.9 can retain less than 0.9 original mass
when applied after another filter and renormalization. **Limitation:** five
synthetic tokens do not establish decoding quality. Original mass,
temperature-adjusted mass and post-top-k mass are different quantities.
No runtime measurement was made.

### 3. Controlled reproducibility

**Hypothesis:** the same seed and calls reproduce the local sequence.
**Configuration:** A/B/EOS fixture; before 12 generated IDs, logits favor the
preceding symbol (initially A) by 1 over the other symbol, with EOS logit -10.
At 12 IDs, EOS receives logit 10 and A/B receive -10. T=1, no filters, output
budget 20, seeds 42/42/43; a separate greedy comparison uses T=0 and seeds 42/43.

**Result:** both seed-42 sampling runs returned `BBBBABBBAAAB`; seed 43
returned `AAABBABAAABB`. All three stopped on EOS after 13 sampled IDs.
The two greedy results were equal across seeds.

**Interpretation candidate:** seed replay and greedy seed independence hold
in this fixture. **Limitation:** one different-seed pair does not measure
diversity; distinct seeds can coincide. This is not evidence of cross-version,
hardware or hosted-model reproducibility.

### 4. Termination and visible text

**Hypothesis:** EOS, stop strings and token limits produce distinct finish
reasons and visible output. **Configuration:** handwritten transitions
`Hello` → ` world` → `!` → EOS, T=0, seed 42, default budget 20 except the
budget controls; initial prompt is empty.

| Case | Visible text | Sampled IDs count | Reason |
|---|---|---:|---|
| No text stop | `Hello world!` | 4 | `eos` |
| Stop `!` | `Hello world` | 3 | `stop_sequence` |
| Cross-fragment stop `lo wo` | `Hel` | 2 | `stop_sequence` |
| Budget 2 | `Hello world` | 2 | `length` |
| Budget 0 | Empty | 0 | `length` |

**Interpretation candidate:** visible length and work performed are different
accounting quantities. **Limitation:** fragment concatenation is not tokenizer
decoding, API billing, streaming behavior or proof of task completeness.

The standalone `from_scratch.py` demo was also executed and returned the first,
second and fourth cases above. Tests of the numerical/application contracts
are software verification, separate from these interpretation candidates.

Analytical-phase validation on the same date: 59 topic tests passed, without skips or warnings;
repository parsing passed for 548 Python files and internal links passed for
2,027 links in 338 files. Commands used the stable interpreter specified above:
`python scripts/validate_repo.py tests --scope 05-transformers-llms/75-llm-inference-decoding`,
`python scripts/validate_repo.py syntax`, `python scripts/validate_repo.py links`
and `git diff --check`. The environment has no `pip` module, so dependency
compatibility was checked with
`uv --no-cache --offline pip check --python .local/environments/n09-full-2026-11-13/Scripts/python.exe`:
89 installed packages were compatible. These are local software checks;
GitHub math rendering, hosted inference, full-repository tests and CI were not
validated by this topic run.

The subsequent seed-275 visual laboratory has a separate
[execution and review record](VISUAL_LAB.md#execution-and-review-record),
including additional tests, numerical controls and theoretical cache counts.
It does not replace the analytical observations above.

## Suggested extensions — not executed

- Repeat the frequency experiment over independent seeds and sample sizes;
  report uncertainty rather than interpreting one error as a convergence rate.
- Compare a pinned real-model/tokenizer revision across greedy and sampled
  configurations using held-out task quality, grounding, schema validity and
  latency. Keep this separate from the synthetic study.
- Implement tokenizer-aware stop buffering and test Unicode and streaming
  boundaries. This would require a new contract and validation scope.

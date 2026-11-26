# Objectives, parameterization and evidence

## Behavior adaptation starts with a diagnosis

Pretraining optimizes broad prediction objectives. Continued pretraining uses
similar objectives on a different corpus; SFT uses demonstrations of a desired
conditional output. Instruction tuning is a data/task formulation of SFT, often
spanning many natural-language instructions. Testing unseen instruction families
is necessary before claiming instruction generalization. The categorical toy
below does not perform that test.

| Observed failure | First candidate to investigate | Evidence needed before escalating to fine-tuning |
|---|---|---|
| Missing or stale facts | Retrieval, source freshness, context assembly | Relevant sources actually reach the model |
| Invalid output schema | Schema constraints and application validation | Errors persist under an explicit, validated contract |
| Stable style or task behavior fails | Prompt/examples, then SFT candidate | Representative demonstrations and a held-out baseline |
| Plausible answers differ in judged quality | Preference data and evaluation rubric | Pair quality, coverage, disagreement and regressions |
| Wrong tool arguments or permissions | Tool schema, orchestration and access controls | Training cannot replace runtime authorization |

This is an engineering decision framework, not an experimentally evaluated
ranking of prompting, RAG and fine-tuning. A fine-tuned model can still retrieve
documents and use tools. Weight updates can change stored knowledge, but do
not provide a dependable fresh, attributable fact store.

## From token likelihood to supervised updates

For prompt $x$ and response $y$, autoregressive scoring gives

$$
\log \pi_\theta(y\mid x)=\sum_{t=1}^{T}\log \pi_\theta(y_t\mid x,y_{<t}).
$$

With binary response mask $m_{it}$, a token-normalized SFT loss is

$$
L_{\mathrm{SFT}}=-\frac{\sum_{i,t}m_{it}\log\pi_\theta(y_{it}\mid x_i,y_{i,<t})}
{\sum_{i,t}m_{it}}.
$$

Shift inputs and labels once, include the intended termination token, and mask
prompt/padding positions consistently. Teacher forcing uses true preceding
tokens; it does not measure free generation. Empty response masks are invalid.
Example-normalized sequence sums weight lengths differently from token means.

`response_logps` expects already aligned causal logits and next-token targets;
it returns a response sum per example. Tests compare its masked sum to PyTorch
cross-entropy, including zero gradients at excluded positions. It neither
constructs a chat template nor enforces causal attention. In a real LLM,
formatting and truncation must be validated before scoring. DPO conventionally
uses response sequence sums; replacing them with per-token means changes the
objective and its length behavior.

## RLHF and the reference policy

A classical RLHF pipeline first fits demonstrations, then a reward model from
comparisons, then uses policy optimization such as PPO on sampled responses.
A Bradley–Terry reward model uses

$$
P(y_w\succ y_l\mid x)=\sigma(r_\phi(x,y_w)-r_\phi(x,y_l)).
$$

One regularized policy objective is

$$
\max_\theta\;\mathbb E_{x\sim D,\,y\sim\pi_\theta(\cdot\mid x)}
[r_\phi(x,y)]-\beta\,\mathbb E_{x\sim D}
D_{\mathrm{KL}}(\pi_\theta(\cdot\mid x)\Vert\pi_{\mathrm{ref}}(\cdot\mid x)).
$$

The reference anchors policy drift. Imperfect rewards can incentivize shortcuts;
labeling bias, weak coverage and reward hacking remain evaluation problems.
This study implements neither a learned reward model nor PPO. See
[InstructGPT](https://arxiv.org/abs/2203.02155) for an executed research pipeline;
its reported model results are not results of this repository.

## DPO: ratios, not an explicit KL penalty in the code

Define a same-prompt preference margin

$$
m_i=\log\pi_\theta(y_w\mid x_i)-\log\pi_\theta(y_l\mid x_i)
-\log\pi_{\mathrm{ref}}(y_w\mid x_i)+\log\pi_{\mathrm{ref}}(y_l\mid x_i).
$$

The standard offline loss and derivative are

$$
L_{\mathrm{DPO}}=\frac1N\sum_i\operatorname{softplus}(-\beta m_i),
\qquad \frac{\partial L}{\partial m_i}=-\frac\beta N\sigma(-\beta m_i).
$$

The NumPy implementation differentiates the two policy log-probability columns;
the rejected derivative has the opposite sign. It uses `logaddexp` for numerical
stability. When policy/reference margins match, loss equals $\log 2$.
For same-prompt categorical pairs the softmax normalizer cancels from their
log-ratio; the tests check the resulting logit gradient independently.

[DPO](https://arxiv.org/abs/2305.18290) derives this loss by reparameterizing the
optimal policy of a KL-regularized reward problem under a preference model.
This does not promise an exact solution with finite biased data and restricted
parameters. Beta scales the margin and gradient; its theoretical regularization
role is not a universal monotonic guarantee about observed KL under fixed
learning rates and finite updates. The code measures KL separately rather than
adding an explicit KL term to the DPO loss.

The frozen reference here is an independent copy of the complete **SFT policy**,
including its learned LoRA matrices. It is not the earlier base layer. Preference
pairs share an input and differ only in response category. Their synthetic
labels have no annotator uncertainty. Real data need provenance, consistent
rubrics, disagreement analysis and independent capability/safety evaluations.

Improving a ratio need not increase absolute chosen probability: changing
chosen/rejected probabilities from $(0.4,0.3)$ to $(0.3,0.1)$ improves their ratio
while both decrease. The remaining probability goes to other outcomes.
Pair accuracy also ignores whether another category wins the overall argmax.

## Adapters, LoRA and QLoRA constrain different resources

A bottleneck adapter can transform a hidden state as

$$
h'=h+W_{\mathrm{up}}\,f(W_{\mathrm{down}}h).
$$

It adds trainable modules to an otherwise frozen network. A nonlinear adapter
generally cannot be merged into an existing linear weight. Placement and
normalization details depend on the architecture.

For a linear weight $W_0\in\mathbb R^{d_{out}\times d_{in}}$, LoRA uses

$$
W=W_0+\frac\alpha r BA,\quad
A\in\mathbb R^{r\times d_{in}},\quad B\in\mathbb R^{d_{out}\times r}.
$$

There are $r(d_{in}+d_{out})$ trainable entries instead of $d_{in}d_{out}$ weight
entries. The update rank is at most $r$, and choosing a large rank can erase
parameter savings. In this implementation base bias is also frozen. Random A
and zero B preserve initial outputs: A's first gradient is zero while B can
receive a signal. Making both zero would stall this bilinear path.

`merged()` returns a separate floating-point layer using $W_0+(\alpha/r)BA$.
Finite precision permits small discrepancies. No dropout, quantization, tied
weights, attention targeting or adapter routing is implemented. The original
[LoRA paper](https://arxiv.org/abs/2106.09685) motivates low-rank adaptation;
its quality and efficiency results do not transfer automatically to this toy.

QLoRA additionally stores a frozen quantized base and trains low-rank adapters.
NF4, double quantization and paged optimizers are features of the original
[QLoRA method](https://arxiv.org/abs/2305.14314). Activations, compute buffers,
optimizer state and the reference model still matter. Fewer trainable
parameters do not imply proportionally less total memory, latency or compute.
This study performs no quantization or resource benchmark.

## Evaluation and operational questions

Split real examples by source/entity and time where appropriate; remove near
duplicates before partitioning. Keep task quality, format validity, factual
grounding, refusals, preference judgments and retained capabilities distinct.
Use blinded comparisons with uncertainty and disagreement reporting when
human preference is the target. A judge score is a proxy, not ground truth.

Validate dataset rights, sensitive-data handling and model licenses for an
actual training project. Retain base/tokenizer/chat-template versions, adapter
configuration and evaluation artifacts together. Check base-adapter compatibility,
serving behavior and rollback before deployment. PEFT can reduce update scope
but cannot guarantee freedom from forgetting or cross-task regressions. These
are design considerations; there is no deployed system in this study.

## Execution record — 2026-11-25

**Hypothesis:** SFT and then DPO through LoRA can increase concise-category
preference while retaining task identity on independent synthetic inputs.
**Interpretation status:** author review required.

Executed locally on Windows, Python 3.11.14, PyTorch 2.15.0+cpu and NumPy 2.4.6.
The report timestamp is `2026-11-26T00:17:23.321499+00:00`, corresponding to
2026-11-25 in America/Sao_Paulo. This is a new analytical run, separate from
the proposed experiments in the study material and every earlier day's records.

Configuration: initialization seed 42; independent split seeds 43/44/45/46 and
sizes 512/128/128/256 for base/SFT/preferences/evaluation. Two balanced tasks
have one-hot IDs and six Gaussian nuisance features with standard deviation 0.2.
Base style probability is 0.5; SFT concise probability is 0.8 (sampling
probability, not an enforced sample fraction). Preference/evaluation chosen
category is always the concise version of the correct task.

All stages use full-batch Adam with fresh optimizer state: 200 base updates at
0.03, 80 SFT updates at 0.03 and 100 DPO updates at 0.01. Rank 1, alpha 2,
beta 0.5; CPU float32, one thread, no dropout or weight decay. The final budget
is fixed in advance; no validation search or evaluation-based selection occurs.

Command (from repository root; use the same stable interpreter for both):

```powershell
python 05-transformers-llms/74-fine-tuning-alignment/example.py --output 05-transformers-llms/74-fine-tuning-alignment/outputs/run_20261125_seed42_analytical.json
python 05-transformers-llms/74-fine-tuning-alignment/from_scratch.py
```

The JSON remains ignored and is not a public asset link. Existing filenames
are refused; use a new dated filename for reruns. Numerical repeatability is
tested in the installed environment, not promised across library versions.

| Held-out metric, 256 inputs | Base | SFT + LoRA | DPO + LoRA |
|---|---:|---:|---:|
| Task accuracy (argmax category's task) | 1.000000 | 1.000000 | 1.000000 |
| Exact chosen category accuracy | 0.535156 | 0.968750 | 1.000000 |
| Within-pair accuracy, strict positive log-odds | 0.535156 | 0.968750 | 1.000000 |
| Mean chosen category probability | 0.494234 | 0.776065 | 0.975222 |
| Mean chosen/rejected log-odds | 0.004370 | 1.496050 | 7.692167 |
| Mean reference-adjusted margin | -1.491681 | 0.000000 | 6.196117 |
| Mean KL(policy to SFT reference), nats | 0.306964 | 0.000000 | 0.246170 |
| Mean DPO loss vs SFT reference | 1.151509 | 0.693147 | 0.046714 |

The SFT model is its own reference in the middle column. The base column's
reference is that same later SFT snapshot. KL and adjusted margins therefore
describe distance to SFT, not distance to each stage's predecessor.

Base weight+bias count: 36; trainable LoRA count: 12. The policy stores 48 total
parameters, and the reference is an additional copy. The base and complete
reference state stayed exactly unchanged through their frozen stages.
Merged/unmerged maximum absolute logit discrepancy was $1.907349\times10^{-6}$.

**Interpretation candidate — author review required:** in this configuration,
the preference shift coexisted with unchanged task argmax accuracy. The rising
log-odds and nonzero KL show a distribution change that accuracy alone hides.
This demonstrates the implemented mechanism on its constructed distribution.

**Limitations:** one seed and simple one-hot task identity; continuous nuisance
features distinguish splits but do not introduce new task families. No text
generation, natural-language instructions, humans, factuality test, general
capability retention test or production evidence. SFT/DPO use different signals
and budgets, so this is neither a fair method ranking nor a full-FT comparison.
No memory/timing benchmark or hyperparameter optimum is claimed.

The separate fixed NumPy illustration uses beta 0.5, policy log-probabilities
`[[-2, -4], [-3, -2]]` and reference log-probabilities `[[-3, -4], [-2, -3]]`.
It returned mean loss 0.8936693358491647 and gradient rows
`[-0.0943851672, 0.0943851672]` and `[-0.1827646447, 0.1827646447]`.
This checks the analytical formula on chosen constants, not a trained model's
quality. Interpretation candidate: derivative signs encourage a larger relative
margin; **author review required**. Finite differences and autograd provide
separate numerical checks rather than empirical preference evidence.

### Local validation

Executed with the stable interpreter and the repository validation runner:

```powershell
python scripts/validate_repo.py tests --scope 05-transformers-llms/74-fine-tuning-alignment
python scripts/validate_repo.py syntax
python scripts/validate_repo.py links
uv --no-cache --offline pip check --python .local/environments/n09-full-2026-11-13/Scripts/python.exe
git diff --check
```

The focused suite passed 30 tests in one isolated file, without skips or
warnings. Its repeatability integration uses 20/10/10 updates and checks two
identical runs, restored RNG/thread state, frozen parameters and finite metrics;
those short test runs are not the 200/80/100-update result table above.
The global checks parsed 539 public Python files and checked 1,997 internal
links across 333 Markdown files. Offline dependency checking found all 89
installed packages compatible. The tracked diff had no whitespace errors.
These are local checks, not a full curriculum test run, CI result, GitHub math
rendering check, scientific review or deployment approval.

## Suggested experiments — not executed

- Sweep rank while retaining identical splits, objectives and evaluation;
  inspect both parameter counts and capability regressions.
- Sweep beta with an explicit learning-rate/budget policy; measure KL rather
  than assuming a monotonic result.
- Compare full fine-tuning and LoRA with a prespecified tuning budget, repeated
  seeds and measured resource usage.
- Introduce conflicting or noisy preferences and track task loss as well as
  pair loss. Do not equate a preferred style with factual correctness.

## Separate visual experiments — 2026-11-25

The [visual laboratory](VISUAL_LAB.md) adds separately configured synthetic
experiments, analytical surfaces and uniform quantization simulations. Its
dated record preserves the distinction from the analytical results above.
Matrix rank sweeps, conflicting-task full/LoRA/replay adaptation and reward/KL
calculations there do not execute the original response-selector sweep proposals
under identical settings. Historical numerical records remain unchanged;
new visual interpretations and preview curation require author review.

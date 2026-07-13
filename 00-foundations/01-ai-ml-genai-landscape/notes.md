# Technical Notes: AI/ML/GenAI Landscape

## 1. A Useful Mental Model

The landscape is easier to reason about when separated into three categories:

```text
Model families and learning paradigms
AI -> Machine Learning -> Deep Learning -> Foundation Models -> LLMs

System patterns
retrieval | RAG | tool use | workflows | agents | multi-agent systems

Operational disciplines
data engineering | MLOps | LLMOps | observability | governance
```

Data Science cuts across these categories: it frames decisions, analyzes evidence, designs experiments, builds models when useful, and communicates uncertainty. AI Engineering turns model capabilities into software that remains useful under real traffic, changing data, failures, and organizational constraints.

The nested model-family diagram is a helpful simplification, not a complete taxonomy. Not every foundation model is an LLM, not every AI method learns from data, and a production product usually combines learned and deterministic components.

## 2. From Business Question to System

A sound design sequence is:

```text
user or business objective
-> decision/action to support
-> measurable success and guardrails
-> available data and feedback
-> baseline
-> model/system choice
-> offline and online evaluation
-> deployment, monitoring, and iteration
```

Starting with a framework reverses this reasoning. "Use an agent" is an implementation proposal, not a requirement. The real requirement might be dynamic tool selection, low-latency classification, auditable document lookup, or a fixed approval process.

## 3. Disciplines and Model Families

### 3.1 Data Science

Data Science extracts decision-relevant knowledge from data through statistics, experimentation, causal reasoning, forecasting, optimization, visualization, and sometimes ML. It does not require a learned model: a well-designed experiment or SQL analysis may be the correct deliverable.

Core assumptions depend on the method. Statistical inference may require representative sampling and a valid data-generating model. Causal claims additionally require defensible identification assumptions. Predictive work requires that evaluation data represent the deployment setting.

### 3.2 Machine Learning

Machine Learning estimates behavior from examples rather than encoding every rule explicitly.

- **Supervised learning:** learn from input-target pairs for classification, regression, ranking, or forecasting.
- **Unsupervised learning:** discover structure without explicit target labels, such as clusters or low-dimensional representations.
- **Self-supervised learning:** derive supervision from the data, such as predicting a masked or next token.
- **Reinforcement learning:** learn a policy from state, action, reward, and transition signals.

Most supervised learning can be expressed as regularized empirical risk minimization:

$$
\theta^* = \arg\min_\theta \left[
\frac{1}{n}\sum_{i=1}^{n}\mathcal{L}(f_\theta(x_i), y_i)
+ \lambda\Omega(\theta)
\right].
$$

The loss measures fit to observed examples. The regularizer expresses a preference for a restricted solution, often trading variance for bias. This formula does not guarantee business value: the loss, sample, split strategy, and decision threshold must represent the real task.

### 3.3 Deep Learning

Deep learning uses multilayer neural networks to learn representations and predictions jointly. A layer applies

$$
h^{(l)} = \phi\left(W^{(l)}h^{(l-1)} + b^{(l)}\right),
$$

where $W$ and $b$ are learned parameters and $\phi$ is a nonlinear activation. Composing layers can represent complex functions and hierarchical features.

Deep learning is particularly useful for unstructured data and high-dimensional nonlinear relationships, especially when pretrained models are available. Its costs include compute, data requirements, training instability, weaker interpretability, and more complex serving.

### 3.4 Foundation Models and LLMs

A foundation model is pretrained on broad data and adapted to multiple downstream tasks. An LLM is a language-oriented foundation model, commonly based on the Transformer.

An autoregressive LLM factorizes sequence probability as

$$
P(x_1,\ldots,x_T)=\prod_{t=1}^{T}P(x_t\mid x_{<t}),
$$

and commonly minimizes negative log-likelihood:

$$
\mathcal{L}_{\text{LLM}}=-\sum_{t=1}^{T}\log P_\theta(x_t\mid x_{<t}).
$$

This objective rewards likely continuations; it is not a truth guarantee. Instruction tuning, preference optimization, prompting, tools, and retrieval change behavior or context but do not remove the probabilistic nature of generation.

Scaled dot-product attention is

$$
\operatorname{Attention}(Q,K,V)=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

Queries score compatibility with keys; normalized scores weight values. The scale factor prevents dot products from growing excessively with key dimension.

## 4. System Patterns

### 4.1 Retrieval and RAG

Retrieval ranks external items for a query. Cosine similarity is a common dense or sparse vector comparison:

$$
\operatorname{cosine}(q,d)=\frac{q\cdot d}{\lVert q\rVert_2\lVert d\rVert_2}.
$$

For unit-normalized vectors it equals the dot product. A high similarity score is only a proxy for usefulness; semantic proximity, factual relevance, authorization, freshness, and answer coverage are different properties.

RAG supplies retrieved evidence to a generator at inference time:

```text
offline: documents -> parse -> chunk -> enrich -> index
online: query -> retrieve -> rerank -> construct context -> generate -> validate
```

A conceptual decomposition is

$$
P(y\mid x,D)\approx\sum_{z\in R_k(x,D)}
P_\eta(z\mid x,D)P_\theta(y\mid x,z).
$$

This exposes two separate questions: did retrieval find sufficient evidence, and did generation use it correctly? End-to-end answer scores alone cannot locate the failing component.

Use RAG for private, changing, large, or citation-sensitive knowledge. Avoid it when the input already contains all relevant context, the task is pure transformation, or the corpus and access controls cannot support reliable retrieval.

### 4.2 Tools, Workflows, and Agents

A tool is an externally callable capability. A workflow has a mostly predetermined control graph. An agent gives a model bounded influence over the next action based on current state.

An abstract agent policy is

$$
a_t \sim \pi(a\mid s_t), \qquad
s_{t+1}=T(s_t,a_t,o_{t+1}).
$$

In practical LLM agents, the policy is divided among the model, prompts, schemas, code, permissions, and state machine. The system is not agentic merely because it calls a tool. If every request always performs the same retrieval call, it remains a workflow.

Prefer workflows for stable, auditable, regulated, or irreversible processes. Agents are justified when paths genuinely vary and intermediate observations determine later actions. A common production compromise is a deterministic outer workflow with narrow agentic decisions and explicit approval gates.

### 4.3 Multi-Agent Systems

Multiple agents may specialize as router, researcher, executor, or critic. This can improve separation of concerns, but it adds messages, latency, cost, coordination failures, and a much larger behavioral state space. Use multiple agents only when role separation or parallelism supplies measurable value that a single bounded workflow cannot.

## 5. Operations

### 5.1 MLOps

MLOps manages datasets, features, training code, experiments, model artifacts, validation, deployment, monitoring, rollback, and retraining. Important failure modes include leakage, training-serving skew, data drift, concept drift, dependency failures, and feedback loops.

Deploying a model endpoint is only one step. Reproducibility requires versioning code, data, configuration, and environment; reliable change requires automated tests and evaluation gates.

### 5.2 LLMOps

LLMOps inherits those concerns and adds artifacts and behavior specific to generative applications:

- prompt, model-provider, tool-schema, and knowledge-base versions;
- retrieved contexts and complete execution traces;
- output quality, grounding, citations, and structured-output validity;
- prompt injection, unsafe tool use, and sensitive-data exposure;
- token usage, model routing, caching, and cost per completed task;
- non-determinism and provider behavior changes.

Distribution monitoring alone is insufficient. A generative system may require a versioned evaluation set, deterministic checks, calibrated model-based graders, human review, adversarial cases, and online product metrics.

## 6. Four Planes of AI System Design

| Plane | Responsibilities | Example artifacts |
| --- | --- | --- |
| Data | ingest, validate, transform, store, govern | datasets, documents, features, embeddings |
| Intelligence | predict, retrieve, rank, generate, apply rules | classifiers, LLMs, retrievers, policies |
| Orchestration | manage state and execution | workflows, agents, queues, retries, approvals |
| Operations | run and improve safely | deployment, traces, evals, alerts, IAM, budgets |

The boundaries help assign ownership and diagnose failures, but they are not isolated. An embedding change affects the index, retrieval metrics, downstream answers, cost, and rollout plan.

## 7. Selecting an Approach

| Problem shape | Good starting point | Main reason |
| --- | --- | --- |
| Structured inputs, fixed label | rules or classical ML | low cost, constrained output, measurable target |
| Image/audio/text representation | deep learning or pretrained model | learned features for unstructured data |
| Open-ended language transformation | LLM | flexible language generation |
| Questions over private changing documents | RAG | external evidence and provenance |
| Fixed multi-step process | workflow | predictable and auditable control |
| Variable multi-tool goal | bounded agent | observations determine later actions |
| Exact financial calculation | code, SQL, or rules | deterministic correctness |

Fine-tuning and RAG solve different primary problems. RAG injects knowledge at inference time and can expose sources. Fine-tuning modifies behavior through training and is useful for style, format, or task specialization. They can be combined.

## 8. Assumptions and Trade-offs

Every approach depends on assumptions:

- Supervised ML assumes labels and the evaluation sample represent deployment sufficiently well.
- Deep learning assumes enough signal, data, compute, or transferable pretrained representations.
- RAG assumes relevant evidence exists, can be parsed, indexed, authorized, and retrieved.
- Agents assume the available state and tools are enough to choose and validate useful actions.
- Monitoring assumes production signals correlate with user and business outcomes.

A simple system utility abstraction is

$$
U=Q-\alpha C-\beta L-\gamma R,
$$

where quality $Q$ competes with cost $C$, latency $L$, and risk $R$. The coefficients express product priorities. A larger model can raise model quality while reducing total utility through latency, cost, or operational risk.

## 9. Evaluation by Layer

- **Classifier:** class-level precision/recall, calibration, abstention behavior, slice performance.
- **Retriever:** Recall@k, Precision@k, MRR, nDCG, evidence coverage, authorization correctness.
- **Generator:** correctness, faithfulness to evidence, completeness, citations, format, safety.
- **Agent/workflow:** task completion, invalid actions, loop rate, recovery, approval compliance.
- **System:** user outcome, latency distribution, availability, cost per task, escalation, business impact.

Offline evaluation is necessary but incomplete. Production introduces ambiguous requests, drift, integration failures, malicious input, concurrency, feedback loops, and user behavior. Online rollout needs guardrails and a way to compare outcomes safely.

## 10. Applications and Limitations

Applications span tabular prediction, forecasting, recommendations, document extraction, knowledge assistants, support automation, code generation, and tool-using operations. Hybrid systems are common: a rule can reject invalid input, a classifier can route it, retrieval can supply evidence, an LLM can produce language, and a human can approve a high-risk action.

No architecture removes uncertainty. Classical models fail under distribution shift; deep models can be opaque; LLMs can generate unsupported claims; RAG can retrieve the wrong evidence; agents can misuse tools or fail to terminate. Reliability comes from constrained responsibilities, explicit contracts, component tests, observability, and controlled fallback—not from treating one model as the whole system.

## 11. Common Mistakes

1. Calling RAG a model rather than a system pattern.
2. Calling every tool-using workflow an agent.
3. Assuming an LLM is appropriate because the input is text.
4. Using fine-tuning primarily to keep frequently changing facts current.
5. Evaluating only aggregate accuracy or only the final RAG answer.
6. Ignoring negative cases, uncertainty, and abstention.
7. Applying access control after sensitive retrieval.
8. Treating prompts as security or validation boundaries.
9. Retrying non-idempotent tool calls without safeguards.
10. Logging sensitive prompts, documents, or outputs indiscriminately.
11. Monitoring API health without monitoring behavioral quality.
12. Optimizing model metrics while ignoring the user decision and system constraints.

## 12. Suggested Experiments

1. Run `example.py`, then test classifier thresholds of `0.40`, `0.70`, and `0.90`. Compare automation with review volume.
2. Remove category filtering during retrieval. Observe whether it recovers from or amplifies classifier errors.
3. Try unigram-only, word-bigram, and character n-gram TF-IDF.
4. Add out-of-domain queries and record whether classification, retrieval, or policy fails first.
5. Add a second retrieved document and define how evidence conflicts should be handled.

These examples use synthetic data and are intended to reveal behavior, not establish performance claims.

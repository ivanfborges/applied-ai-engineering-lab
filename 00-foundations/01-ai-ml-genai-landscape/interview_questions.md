# Senior Interview Questions and Answers

## 1. How do Data Science, Machine Learning, and AI Engineering differ?

Data Science frames questions and uses data to support decisions through analysis, statistics, experimentation, causal reasoning, and predictive modeling. Machine Learning focuses on algorithms that learn behavior from data. AI Engineering integrates learned capabilities with software, data, retrieval, APIs, observability, security, and operations. Their boundaries overlap, but their primary outputs are often a decision or insight, a learned model, and a reliable production system respectively.

## 2. When would you prefer classical ML to an LLM?

I would begin with classical ML when inputs are structured or short, outputs belong to a fixed label space, representative labels exist, and latency, cost, calibration, or interpretability matter. Churn, fraud scoring, and ticket routing fit this shape. I would consider an LLM when the task requires open-ended generation, flexible instruction following, or semantic transformation. Text input alone is not enough to justify an LLM.

## 3. What is empirical risk minimization, and what does it omit?

It chooses parameters that minimize average training loss, often plus a regularizer:

$$
\theta^*=\arg\min_\theta\left[\frac{1}{n}\sum_i
\mathcal{L}(f_\theta(x_i),y_i)+\lambda\Omega(\theta)\right].
$$

It formalizes fitting and model preference, but does not by itself guarantee generalization or business value. Results depend on representative data, the loss, validation design, thresholds, and deployment conditions. It also omits latency, cost, fairness, and operational risk unless those enter the objective explicitly.

## 4. Why is an LLM not a database?

An autoregressive LLM models token probabilities from context. Its parameters encode statistical patterns rather than rows with explicit provenance and consistency guarantees. It may reproduce useful knowledge, but it can also generate plausible unsupported statements. For private, dynamic, or source-sensitive information, I would use an authorized external data or retrieval layer and validate the answer against it.

## 5. What problem does RAG solve, and what does it not solve?

RAG supplies external evidence to a generator at inference time. It is valuable when knowledge is private, changing, too large for one prompt, or needs citations. It does not guarantee factuality: parsing, chunking, indexing, retrieval, ranking, context construction, and generation can all fail. I would evaluate evidence retrieval separately from answer faithfulness and correctness.

## 6. How would you evaluate a RAG system?

For retrieval I would measure Recall@k, Precision@k, MRR or nDCG, evidence coverage, and access-control correctness. For generation I would measure correctness, faithfulness, completeness, citation accuracy, abstention, structure, and safety. At product level I would track task completion, latency, cost, escalation, and user or business outcomes. The evaluation set should include positive, negative, historical, adversarial, and human-reviewed cases, with every component version recorded.

## 7. RAG or fine-tuning?

I use RAG primarily to provide current external knowledge and provenance. I use fine-tuning primarily to modify behavior, style, output consistency, or task specialization. RAG adds online retrieval complexity; fine-tuning adds training, data-curation, and model-versioning complexity. They are complementary: a fine-tuned model can consume retrieved evidence.

## 8. What makes a system agentic?

A model must dynamically participate in selecting the next action or transition based on state and observations. A fixed pipeline that always invokes the same search tool is a workflow, even if an LLM writes the final response. Agentic control creates flexibility but expands the behavioral state space, so it needs bounded tools, explicit state, termination conditions, permissions, and complete traces.

## 9. How do you choose between a workflow and an agent?

I ask whether execution paths truly vary. A workflow is preferable for stable, regulated, auditable, or irreversible processes. An agent is useful when different requests require different tools or when intermediate results change the plan. A robust hybrid uses deterministic outer control and narrow agentic decisions, with human approval for consequential actions.

## 10. How does LLMOps differ from MLOps?

LLMOps retains versioning, validation, deployment, monitoring, and governance. It additionally manages prompts, contexts, retrieval indexes, tool schemas, provider versions, execution traces, generative evaluation, prompt injection, structured outputs, token usage, and cost. Conventional monitoring often observes one prediction; an LLM application may require inspecting a multi-step trace and the evidence available at each step.

## 11. Why is offline model quality insufficient?

Offline data approximates production. Real use introduces ambiguity, distribution shift, integration failures, latency and cost limits, adversarial inputs, feedback loops, and human behavior. A model with better offline accuracy can create a worse product if it is uncalibrated, slow, expensive, unsafe, or hard to maintain. I combine offline gates, controlled rollout, online outcome metrics, guardrails, and fallback.

## 12. How would you design a private-document assistant?

I would define user tasks, quality, latency, data residency, and audit requirements first. The data plane would parse, chunk, classify, and version documents. Retrieval would enforce authorization before ranking and preserve source identifiers. A reranker could improve evidence selection; the generator would receive bounded context and return citations in a validated schema. The orchestration layer would support abstention and human escalation. I would trace versions and latency without indiscriminately logging sensitive content, and evaluate retrieval and generation separately.

## 13. What are the main agent failure modes and controls?

Failure modes include wrong tools or arguments, loops, stale state, prompt injection, excessive permission, duplicate side effects, failure to detect completion, and trusting incorrect tool output. Controls include typed schemas, least privilege, allowlists, step and budget limits, explicit state transitions, idempotency keys, output validation, sandboxing, human approval, and replayable traces.

## 14. How do you optimize the cost of an LLM system?

I start at the task level: determine which requests need an LLM at all. Then I examine model routing, prompt and context length, retrieval precision, output limits, caching, batching, precomputation, retries, and duplicate tool calls. I compare cost per successful task—not only cost per token—while holding the required quality and risk thresholds constant.

## 15. Give an interview-ready explanation of the landscape.

Machine Learning, Deep Learning, and LLMs describe learned capabilities at increasing specialization and scale. RAG, workflows, and agents are system patterns built around such capabilities: RAG adds external evidence, while an agent lets a model influence the next action. MLOps manages the learned-model lifecycle, and LLMOps extends it with prompts, retrieval, traces, generative evaluation, safety, and token economics. I would start any design from the decision, data, quality target, and operational constraints, then choose the simplest architecture that meets them.

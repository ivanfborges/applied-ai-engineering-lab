# AI, Machine Learning, and Generative AI Landscape

## Overview

This module maps the modern AI landscape across three distinct levels: model families, system patterns, and operational disciplines. Machine learning, deep learning, foundation models, and large language models describe ways to build learned capabilities. Retrieval-Augmented Generation (RAG), workflows, and agents combine capabilities into applications. MLOps, LLMOps, governance, and observability keep those applications reliable in production.

The practical example connects a classical text classifier, a lexical knowledge retriever, an abstention policy, and structured telemetry. It deliberately uses no external LLM: the goal is to make system boundaries and failure modes visible.

## Concepts Covered

- Data Science, Machine Learning, Deep Learning, foundation models, and LLMs
- Supervised, unsupervised, self-supervised, and reinforcement learning
- RAG, tools, deterministic workflows, and agents
- MLOps and the additional concerns introduced by LLMOps
- Data, intelligence, orchestration, and operations planes
- Quality, latency, cost, safety, and maintainability trade-offs
- Component-level and end-to-end evaluation

## Why It Matters

Senior practitioners must distinguish a model capability from a production system. A strong design starts with the user decision, data, constraints, and evaluation plan, then selects the simplest architecture that can meet them. This avoids using an LLM where rules or classical ML are sufficient, using RAG where all context already fits in the request, or introducing an agent where a fixed workflow is safer.

## Files

- `notes.md`: detailed landscape, theory, formulas, selection criteria, and production trade-offs.
- `example.py`: synthetic support-ticket classification, retrieval, routing, abstention, and telemetry.
- `from_scratch.py`: educational bag-of-words vectorization and cosine-similarity retrieval.
- `interview_questions.md`: senior-level conceptual, mathematical, and system-design questions with answers.
- `references.md`: primary papers, official documentation, and further reading.

## How to Run

From the repository root, install the shared dependencies if needed:

```bash
python -m pip install -r requirements.txt
```

Run the end-to-end example:

```bash
python 00-foundations/01-ai-ml-genai-landscape/example.py
```

Run the first-principles retriever:

```bash
python 00-foundations/01-ai-ml-genai-landscape/from_scratch.py
```

Both programs use small synthetic datasets defined in code. Their outputs are demonstrations, not benchmark results.

## Key Takeaways

- Data Science is a problem-solving discipline; ML, deep learning, and LLMs are overlapping model families or learning paradigms.
- RAG and agents are system patterns, not model families.
- Tool use alone does not make a system agentic; dynamic model-directed action selection does.
- LLMOps extends MLOps with prompts, retrieval, traces, generative evaluation, cost, and safety concerns.
- Evaluate each uncertain component as well as the complete user outcome.
- The best starting architecture is the simplest one that satisfies quality, latency, cost, security, and risk constraints.

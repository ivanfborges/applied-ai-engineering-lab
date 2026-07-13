# Applied AI Engineering Lab

A public learning and portfolio repository focused on **Data Science**, **Machine Learning**, **Deep Learning**, **LLMs**, **RAG**, **AI Agents**, **MLOps/LLMOps**, and **AI System Design**.

This repository is designed as a structured technical lab for strengthening senior-level theoretical foundations while building practical, interview-ready and portfolio-ready examples.

## Purpose

The goal is to document a continuous study path that connects:

- mathematical and statistical foundations;
- classical machine learning theory and implementation;
- deep learning and neural network internals;
- NLP, Transformers and Large Language Models;
- embeddings, semantic search and Retrieval-Augmented Generation;
- conversational agents and tool-using systems;
- MLOps, LLMOps, monitoring, evaluation and production readiness;
- AI system design for real-world business problems.

The repository is not intended to be a collection of disconnected notebooks. It is a structured technical portfolio showing the ability to understand, explain, implement and productionize AI systems.

## Target Roles

This lab is aligned with interviews and practical expectations for roles such as:

- Senior Data Scientist
- Machine Learning Engineer
- Applied AI Engineer
- AI Engineer
- GenAI Engineer
- Forward Deployed Engineer
- LLM Engineer
- AI Solutions Architect

## Repository Structure

```text
applied-ai-engineering-lab/
├── README.md
├── START_HERE.md
├── ROADMAP.md
├── STUDY_CONTEXT.md
├── REPO_STRUCTURE.md
├── DAILY_STUDY_PROMPT.md
├── DAILY_STUDY_PROMPT_SHORT.md
├── CODEX_PROMPT.md
├── LINKEDIN_STRATEGY.md
├── GITHUB_WORKFLOW.md
│
├── 00-foundations/
├── 01-statistics-experimentation/
├── 02-classical-machine-learning/
├── 03-unsupervised-recommender-systems/
├── 04-deep-learning/
├── 05-transformers-llms/
├── 06-rag-semantic-search/
├── 07-agents/
├── 08-mlops-llmops/
├── 09-ai-system-design/
└── portfolio-projects/
```

Each study topic should ideally contain:

```text
README.md
notes.md
notebook.ipynb
example.py
from_scratch.py
interview_questions.md
references.md
```

Not every topic requires all files. For example, some theoretical topics may not need `from_scratch.py`, while implementation-heavy topics should include it.

## Daily Study Format

Each topic is designed to be studied in **30–60 minutes**, following this structure:

1. Executive overview
2. Intuition
3. Theoretical foundations
4. Mathematical/statistical/logical foundations
5. Practical applicability
6. Common mistakes and pitfalls
7. Comparisons with related methods
8. Python example
9. From-scratch implementation when useful
10. Experiments to run
11. Senior interview questions
12. Interview-ready explanation
13. GitHub file structure
14. Short README in English
15. LinkedIn post idea
16. 30–60 minute checklist

## Philosophy

The objective is not only to know how to use tools, but to understand:

- why a method works;
- when it fails;
- what trade-offs it introduces;
- how to evaluate it;
- how to explain it in an interview;
- how to apply it in production.

Senior-level AI work requires more than implementing models. It requires connecting theory, architecture, reliability, business impact and communication.

## Suggested Workflow

1. Study the daily topic using `DAILY_STUDY_PROMPT.md` in ChatGPT.
2. Use the generated study content as input for Codex in VSCode.
3. Ask Codex to create or update the topic folder following `CODEX_PROMPT.md`.
4. Review the generated files manually.
5. Run code locally.
6. Commit with a clear message.
7. Push to GitHub.
8. Once or twice per week, publish a short LinkedIn post based on the strongest insight.

## Tech Stack

Preferred stack for examples:

- Python
- NumPy
- pandas
- scikit-learn
- matplotlib
- PyTorch
- FastAPI
- SQL
- BigQuery
- Google Cloud Platform
- Vertex AI
- Vector Search / FAISS / pgvector when relevant
- LangChain / LangGraph / LlamaIndex when relevant

## License

This repository is intended for public learning and portfolio demonstration. Choose a license before publishing reusable code.

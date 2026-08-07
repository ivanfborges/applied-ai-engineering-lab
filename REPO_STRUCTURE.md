# Repository Structure

This document defines the expected structure for the Applied AI Engineering Lab repository.

## Root Structure

```text
applied-ai-engineering-lab/
│
├── README.md
├── LICENSE
├── ROADMAP.md
├── REPO_STRUCTURE.md
├── docs/
│   └── methodology.md
│
├── 00-foundations/
├── 01-classical-machine-learning/
├── 02-unsupervised-recommender-systems/
├── 03-statistics-experimentation/
├── 04-deep-learning/
├── 05-transformers-llms/
├── 06-rag-semantic-search/
├── 07-agents/
├── 08-mlops-llmops/
├── 09-ai-system-design/
├── 10-interview-preparation/
└── portfolio-projects/
```

Only modules with reviewed public content need to exist in Git. The complete
future sequence belongs in `ROADMAP.md`; empty placeholder directories should
not be committed.

## Module Purpose

### `00-foundations/`
Mathematics, probability, statistics, optimization and basic foundations required for ML and AI.

### `01-classical-machine-learning/`
Supervised ML algorithms, model evaluation, feature engineering and explainability.

### `02-unsupervised-recommender-systems/`
Clustering, dimensionality reduction, anomaly detection and recommendation systems.

### `03-statistics-experimentation/`
Statistical inference, A/B testing, causal inference and product metrics.

### `04-deep-learning/`
Neural networks, backpropagation, optimizers, CNNs, RNNs, autoencoders and PyTorch practice.

### `05-transformers-llms/`
NLP, attention, Transformer architecture, tokenization, pretraining, fine-tuning, inference and prompt engineering.

### `06-rag-semantic-search/`
Embeddings, vector databases, semantic search, RAG, retrieval evaluation and production RAG patterns.

### `07-agents/`
Tool calling, ReAct, planning, memory, multi-agent systems, human-in-the-loop, safety and observability.

### `08-mlops-llmops/`
Model serving, monitoring, evals, CI/CD, cost, latency, security and cloud architecture.

### `09-ai-system-design/`
System design exercises for AI products and senior-level interviews.

### `10-interview-preparation/`
Cross-topic technical interview review, mock interviews and technical storytelling.

### `portfolio-projects/`
Larger projects extracted from the study track and polished as standalone portfolio assets.

---

## Topic Folder Naming Convention

Use numeric prefixes and lowercase kebab-case names.

Examples:

```text
01-classical-machine-learning/
├── 01-linear-regression/
├── 02-logistic-regression/
├── 03-decision-trees/
├── 04-random-forest/
└── 05-gradient-boosting/
```

For each folder, use clear names that communicate the topic without being too long.

Good:

```text
04-random-forest
05-gradient-boosting
03-rag-chunking
02-tool-calling
```

Avoid:

```text
day27
study-today
random-stuff
ml-topic
```

---

## Standard Topic Folder Structure

Each topic folder should ideally contain:

```text
README.md
notes.md
notebook.ipynb
example.py
from_scratch.py
interview_questions.md
references.md
```

Not every file is mandatory for every topic.

## Public and Local Study Material

The public topic folder should contain reviewed material only. Raw study-session
outputs and publication drafts remain useful locally, but they are not finished
portfolio artifacts.

Store local-only material under the ignored `.local/` directory:

```text
.local/
├── workflow/
│   ├── STUDY_CONTEXT.md
│   ├── DAILY_STUDY_PROMPT.md
│   ├── DAILY_STUDY_PROMPT_SHORT.md
│   ├── CODEX_PROMPT.md
│   ├── VISUAL_WORKFLOW.md
│   └── GITHUB_WORKFLOW.md
├── raw-study/
├── linkedin-drafts/
├── visual-prompts/
├── archive/
└── portfolio-review/
```

The author manually saves `day*_whole.md` working material under
`.local/raw-study/` and manually extracts optional LinkedIn drafts under
`.local/linkedin-drafts/`. Codex may read those sources but should not create,
move, rewrite, or publish them. Move reusable technical insights into public
topic documentation only after review.

The public [methodology](docs/methodology.md) explains the publication process
without exposing personal context, operational prompts, or private drafts.

### `README.md`
Short, professional English summary.

Should include:

- topic overview;
- concepts covered;
- why the topic matters;
- how to run the code;
- key takeaways.

### `notes.md`
Deep technical notes.

Should include:

- intuition;
- theory;
- formulas;
- assumptions;
- trade-offs;
- applications;
- limitations;
- common mistakes.

### `notebook.ipynb`
Interactive exploration.

Use notebooks for:

- visualization;
- experiments;
- comparison of methods;
- step-by-step explanation.

### `example.py`
Practical example using standard libraries.

Should be:

- simple;
- executable;
- well commented;
- not over-engineered.

### `from_scratch.py`
Simplified internal implementation.

Use this when the topic benefits from implementation from first principles.

Examples where it makes sense:

- gradient descent;
- linear regression;
- logistic regression;
- decision tree split criteria;
- k-means;
- attention mechanism;
- cosine similarity;
- simple vector search.

Examples where it may not be necessary:

- cloud architecture;
- compliance;
- LinkedIn strategy;
- high-level system design.

### `interview_questions.md`
Senior-level Q&A.

Should include:

- conceptual questions;
- mathematical questions;
- practical questions;
- trade-off questions;
- system design questions when relevant.

### `references.md`
References and further reading.

Prefer:

- official documentation;
- original papers;
- reputable technical blogs;
- books;
- course notes from credible institutions.

---

## README Template for Topic Folders

```markdown
# Topic Name

## Overview

Short explanation of the topic and why it matters.

## Concepts Covered

- Concept 1
- Concept 2
- Concept 3

## Why It Matters

Explain how this topic appears in real Data Science or AI Engineering work.

## Files

- `notes.md`: detailed theory and practical discussion.
- `example.py`: practical implementation with standard libraries.
- `from_scratch.py`: simplified implementation from first principles, when applicable.
- `interview_questions.md`: senior-level interview questions and answers.
- `references.md`: useful references.

## How to Run

```bash
python example.py
```

## Key Takeaways

- Takeaway 1
- Takeaway 2
- Takeaway 3
```

---

## Coding Guidelines

Prefer clean, readable code over clever code.

General rules:

- use Python 3.11+ when possible;
- prefer small functions;
- add comments where they clarify the concept;
- keep examples lightweight;
- use synthetic or simple public datasets;
- avoid unnecessary dependencies;
- never hardcode secrets or credentials;
- keep cloud examples conceptual unless credentials are not required.

Recommended default imports for classical ML examples:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
```

Avoid using heavy frameworks unless the topic requires them.

---

## Language Guidelines

### GitHub

Prefer English for:

- root README;
- topic README files;
- code comments;
- portfolio project documentation.

Portuguese is acceptable for:

- personal study notes;
- interview preparation notes;
- LinkedIn drafts.

### LinkedIn

Use Portuguese.

Tone:

- professional;
- natural;
- concise;
- technical but accessible;
- no hype;
- no exaggerated self-promotion.

---

## When to Extract a Standalone Repository

Keep daily studies in this monorepo.

Create a separate repository only when a topic becomes a polished project with standalone value.

Examples:

```text
rag-evaluation-pipeline
conversational-agent-with-tools
ml-monitoring-drift-lab
transformers-from-scratch
classical-ml-interview-playbook
```

Criteria for extraction:

- complete README;
- clear problem statement;
- runnable code;
- architecture diagram if relevant;
- practical value;
- strong storytelling for recruiters or hiring managers.

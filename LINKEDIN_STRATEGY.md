# LinkedIn Strategy

This document defines how to use LinkedIn as a professional visibility channel while the technical depth remains in GitHub.

## Main Principle

GitHub is the technical proof.

LinkedIn is the distribution layer.

The goal is not to post every daily study note. The goal is to share selected insights that demonstrate senior thinking in Data Science and AI Engineering.

## Recommended Frequency

Post around:

```text
2 times per week
```

Suggested schedule:

```text
Tuesday: short technical insight
Thursday or Friday: weekly recap, mini-case or GitHub update
```

Avoid posting every day unless there is a very strong reason. Daily posts can make the content feel repetitive and reduce audience interest if every post is just a study log.

## What to Post

Prioritize posts that show:

- technical judgment;
- clear explanation;
- trade-off awareness;
- interview-relevant insight;
- practical connection to real projects;
- production thinking;
- humility and consistency.

## What Not to Post

Avoid:

- daily generic updates;
- long mathematical derivations;
- code dumps;
- exaggerated self-promotion;
- AI hype without substance;
- posts that only say “Day X of studying Y”.

## Recommended Post Types

### 1. Interview Insight

Example theme:

```text
A common mistake in interviews is saying ROC-AUC is always the best metric for imbalanced classification. In many real cases, PR-AUC tells a more useful story.
```

### 2. Conceptual Trade-Off

Example theme:

```text
RAG is not just embeddings plus a vector database. In production, the real challenge is retrieval quality, evaluation, latency, cost and failure handling.
```

### 3. Mini-Case

Example theme:

```text
This week I implemented Random Forest from scratch to revisit bootstrap sampling, bagging and feature randomness. The exercise helped clarify why tree decorrelation is central to the method.
```

### 4. Weekly Recap

Example theme:

```text
This week I reviewed classification metrics, threshold tuning and calibration. The main insight: a model is not good just because it has high accuracy. It is good when its decisions match the cost of the business problem.
```

### 5. System Design Insight

Example theme:

```text
When designing an AI assistant, the model is only one part of the system. The architecture also needs retrieval, tools, fallback, observability, evaluation and permission boundaries.
```

## Post Length

Keep posts concise.

Recommended range:

```text
800–1,500 characters
```

Longer posts are acceptable for strong technical storytelling, but most posts should be easy to read quickly.

## Structure Template

```text
Hook: one clear technical observation.

Context: where this appears in real work or interviews.

Insight: what people often miss.

Practical takeaway: how to think about it.

Soft CTA: mention that deeper notes/code are in GitHub or in the first comment.
```

## Example Post Template

```text
Uma coisa que aparece muito em entrevista de Machine Learning é a diferença entre saber usar um algoritmo e saber explicar por que ele funciona.

Random Forest, por exemplo, parece simples: várias árvores e uma votação final.

Mas o ponto mais importante está no efeito combinado de bootstrap sampling, bagging e aleatoriedade nas features. Esses mecanismos reduzem variância e evitam que todas as árvores cometam exatamente os mesmos erros.

É esse tipo de detalhe que diferencia uma explicação superficial de uma explicação mais sênior.

Estou revisando esses fundamentos e documentando exemplos práticos em Python no GitHub.
```

## GitHub Link Strategy

Do not force a GitHub link in every post.

Options:

1. Put the repo link in the first comment.
2. Mention that the implementation is in GitHub and add the link only when relevant.
3. Use GitHub links more directly for weekly recaps or project posts.

## Weekly Content Flow

### Monday
Study and commit.

### Tuesday
Post a short insight from Monday/Tuesday.

### Wednesday
Study and commit.

### Thursday
Study and commit.

### Friday
Post a weekly recap or mini-case.

### Weekend
Review GitHub organization and prepare the next week.

## Positioning

The recurring message should be:

```text
I am a Senior Data Scientist / Applied AI Engineer strengthening deep theory, production thinking and practical implementation around modern AI systems.
```

Do not position the work as “starting from zero”.

Position it as:

```text
consolidating senior foundations and documenting applied AI engineering practice.
```

## Content Buckets

Rotate between these buckets:

1. Classical ML and statistics
2. LLMs and Transformers
3. RAG and embeddings
4. Agents and tool use
5. MLOps/LLMOps and production
6. AI system design
7. Career/interview reflections

## Monthly Strategy

At the end of each month, publish one stronger post:

```text
What I studied this month in Applied AI Engineering and what I learned about building reliable AI systems.
```

This can summarize:

- topics studied;
- strongest insights;
- GitHub improvements;
- practical projects started;
- next focus areas.

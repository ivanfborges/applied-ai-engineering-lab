## 1. Executive overview

The AI/ML/GenAI landscape is best understood as a set of **nested model families combined with system-level architectural patterns and operational disciplines**.

A useful initial map is:

```text
Artificial Intelligence
└── Machine Learning
    └── Deep Learning
        └── Foundation Models
            └── Large Language Models

System patterns built around models:
├── Retrieval-Augmented Generation
├── Tool use
├── Workflows
├── Agents
└── Multi-agent systems

Operational disciplines:
├── Data Engineering
├── MLOps
├── LLMOps
├── AI observability
└── AI governance

Cross-cutting discipline:
└── AI system design
```

This hierarchy needs an important qualification:

* **Machine Learning, Deep Learning and LLMs** describe model families or learning paradigms.
* **RAG and agents** describe system architectures built around models.
* **MLOps and LLMOps** describe how those systems are developed, evaluated, deployed and operated.
* **Data Science** is a broader problem-solving discipline that intersects statistics, experimentation, data engineering, machine learning and business decision-making.
* **AI Engineering** focuses on turning models into reliable, scalable and useful software systems.

Senior-level interviews rarely test only whether you know what each term means. They test whether you can determine:

1. which level of complexity a problem actually requires;
2. whether the problem requires prediction, retrieval, generation, decision-making or automation;
3. how the components interact;
4. how success will be measured;
5. what can fail in production;
6. how cost, latency, risk and maintainability affect the design.

The central senior-level principle is:

> The model is only one component of an AI system. Production value comes from the full system: data, evaluation, orchestration, interfaces, infrastructure, monitoring and operational feedback.

---

## 2. Core intuition

Think of an AI system as a **factory for transforming information into decisions or actions**.

* **Data Engineering** supplies and organizes the raw material.
* **Data Science** identifies the business problem, studies the data and defines how success should be measured.
* **Machine Learning** builds machines that learn patterns from examples.
* **Deep Learning** builds machines that also learn their own internal representations.
* **LLMs** are general-purpose language engines trained on very large datasets.
* **RAG** gives the language engine access to an external knowledge warehouse.
* **Tools** allow the engine to query systems or perform operations.
* **Agents** act as controllers that decide which tool or step to use next.
* **MLOps and LLMOps** keep the factory reproducible, observable and reliable.
* **AI system design** defines how the entire factory should operate under real constraints.

This analogy also clarifies an important misconception:

Giving an LLM access to a database does not automatically create an agent. It may simply create a deterministic workflow in which the application always performs retrieval before generation.

An agent exists when the system has some autonomy to choose actions based on the current state.

---

## 3. Theoretical foundations

### 3.1 Data Science

Data Science is the discipline of extracting actionable knowledge from data.

It may involve:

* exploratory data analysis;
* statistical inference;
* experimentation;
* causal reasoning;
* predictive modeling;
* forecasting;
* optimization;
* visualization;
* decision support;
* machine learning;
* communication with stakeholders.

Data Science does not necessarily require machine learning.

For example, a pricing analysis based on SQL, descriptive statistics and controlled experiments may create more business value than a complex predictive model.

A senior Data Scientist must be able to move between:

```text
Business question
→ measurable objective
→ data requirements
→ analytical or modeling method
→ validation
→ decision or product
```

---

### 3.2 Machine Learning

Machine Learning creates systems whose behavior is learned from data instead of being entirely specified by explicit rules.

The primary paradigms are:

#### Supervised learning

The training data contains input-output pairs:

[
(x_i, y_i)
]

The objective is to learn a function:

[
f_\theta(x) \approx y
]

Typical problems:

* classification;
* regression;
* ranking;
* forecasting;
* anomaly detection with labeled examples.

#### Unsupervised learning

The data contains inputs but no explicit target labels.

Typical problems:

* clustering;
* dimensionality reduction;
* density estimation;
* representation discovery.

#### Self-supervised learning

The learning signal is generated from the data itself.

Examples include:

* predicting masked tokens;
* predicting the next token;
* contrasting related and unrelated examples;
* reconstructing corrupted inputs.

Modern foundation models rely heavily on self-supervised pretraining.

#### Reinforcement learning

An agent interacts with an environment and learns a policy that maximizes cumulative reward.

Typical elements:

* state;
* action;
* reward;
* transition;
* policy.

Reinforcement learning is conceptually related to AI agents, but most LLM agents are not continuously learning through reinforcement learning in production. They usually execute an inference-time control policy defined by prompts, code and model outputs.

---

### 3.3 Deep Learning

Deep Learning is a subset of Machine Learning based on neural networks with multiple representation-learning layers.

The key conceptual difference is that classical ML often depends on manually designed features, while deep learning can learn hierarchical representations directly from data.

For an image model:

```text
Pixels
→ edges
→ textures
→ shapes
→ objects
```

For a language model:

```text
Tokens
→ local patterns
→ syntactic relationships
→ semantic representations
→ task-relevant behavior
```

Deep learning is particularly useful when:

* the data is unstructured;
* the relationship between input and output is highly nonlinear;
* feature engineering is difficult;
* large amounts of data or pretrained models are available.

It introduces higher requirements for:

* compute;
* data;
* monitoring;
* reproducibility;
* inference optimization;
* interpretability.

---

### 3.4 Foundation models and LLMs

A foundation model is trained on broad data and can be adapted to many downstream tasks.

Large Language Models are foundation models specialized in sequences, especially text and code.

Most modern LLMs are autoregressive Transformer models. Their base objective is to predict the next token given previous tokens.

The lifecycle usually includes:

```text
Pretraining
→ instruction tuning
→ preference or alignment tuning
→ deployment
→ prompt/context adaptation
→ optional domain adaptation
```

Important concepts include:

* tokenization;
* embeddings;
* positional information;
* self-attention;
* feed-forward layers;
* residual connections;
* normalization;
* context windows;
* decoding;
* instruction tuning;
* tool calling;
* structured outputs.

An LLM does not behave like a conventional knowledge database. Its parameters encode statistical patterns learned during training. It generates outputs probabilistically and may produce unsupported information.

---

### 3.5 Retrieval-Augmented Generation

RAG combines information retrieval with generative modeling.

A basic RAG flow is:

```text
Documents
→ parsing
→ chunking
→ embedding
→ vector or hybrid index

User query
→ query transformation
→ retrieval
→ optional reranking
→ context construction
→ generation
→ validation and citations
```

The principal objective is to provide the model with relevant external evidence at inference time.

RAG is useful when knowledge is:

* private;
* frequently updated;
* too large to place entirely in the prompt;
* required for traceable answers;
* subject to access control.

RAG does not guarantee factuality. The system can fail because:

1. the relevant document was not indexed;
2. the document was parsed incorrectly;
3. the chunking strategy separated important context;
4. the embedding did not represent the query well;
5. retrieval returned irrelevant documents;
6. the context window was poorly constructed;
7. the model ignored or misinterpreted the evidence;
8. the requested answer does not exist in the corpus.

RAG therefore needs separate evaluation for retrieval and generation.

---

### 3.6 Workflows, tools and agents

A **tool** is an externally callable capability, such as:

* querying BigQuery;
* searching a vector index;
* calling an API;
* executing a calculation;
* creating a ticket;
* sending an email;
* updating a CRM.

A **workflow** has a mostly predefined execution graph.

Example:

```text
Receive document
→ OCR
→ extract metadata
→ retrieve legislation
→ generate summary
→ validate output
→ store result
```

An **agent** uses a model as part of the control mechanism that chooses what to do next.

A simplified agent loop is:

```text
Observe state
→ decide next action
→ call tool
→ observe result
→ update state
→ continue or finish
```

A practical definition is:

> An agent is a system in which a model dynamically participates in selecting actions, tools or transitions based on the current state.

Not every application needs an agent. Deterministic workflows are usually preferable when the task is predictable and the business process is well-defined.

Agents become more useful when:

* the task is open-ended;
* the required sequence varies;
* multiple tools may be relevant;
* intermediate observations influence later decisions;
* the system must recover from certain failures;
* the user goal cannot be mapped to one fixed pipeline.

---

### 3.7 MLOps

MLOps applies software engineering, data engineering and operational practices to the ML lifecycle.

It covers:

* dataset versioning;
* feature pipelines;
* reproducible training;
* experiment tracking;
* validation;
* model registry;
* deployment;
* CI/CD/CT;
* monitoring;
* rollback;
* retraining;
* governance.

A conventional ML system may experience:

* data drift;
* concept drift;
* training-serving skew;
* feature leakage;
* model degradation;
* dependency failures.

MLOps is not simply deploying a model endpoint. It is the discipline of managing the model and its dependencies throughout their lifecycle.

---

### 3.8 LLMOps

LLMOps is best viewed as an extension or specialization of MLOps for foundation-model applications.

It adds operational concerns such as:

* prompt versioning;
* model-provider versioning;
* evaluation datasets;
* retrieval evaluation;
* response-quality evaluation;
* prompt and tool traces;
* token usage;
* context-window management;
* hallucination analysis;
* grounding;
* safety policies;
* prompt injection;
* structured-output validation;
* model routing;
* semantic caching;
* human feedback.

Traditional model monitoring focuses strongly on distributions and prediction performance. LLM monitoring also needs to analyze full execution traces, retrieved context, tool calls and qualitative output dimensions.

---

### 3.9 AI system design

AI system design determines how models, data sources, APIs, infrastructure and user interactions work together.

A senior design process should begin with:

1. **User and business objective**
2. **Decision or action being supported**
3. **Quality requirements**
4. **Latency requirements**
5. **Cost limits**
6. **Availability requirements**
7. **Security and privacy**
8. **Explainability and auditability**
9. **Failure tolerance**
10. **Evaluation strategy**

Only after those elements are defined should the team select models and frameworks.

A production AI system usually contains four planes:

#### Data plane

* source systems;
* ingestion;
* storage;
* transformations;
* document processing;
* feature or embedding generation.

#### Intelligence plane

* ML models;
* LLMs;
* retrievers;
* rerankers;
* classifiers;
* business rules.

#### Orchestration plane

* workflows;
* agents;
* state;
* queues;
* retries;
* tool execution;
* human escalation.

#### Operations plane

* deployment;
* observability;
* evaluation;
* security;
* governance;
* cost management;
* feedback loops.

---

## 4. Mathematical, statistical or logical foundations

Day 1 is a landscape overview, so the goal is to understand how the main mathematical objects connect.

### 4.1 Empirical risk minimization

Most supervised ML training can be represented as:

[
\theta^* =
\arg\min_\theta
\left[
\frac{1}{n}
\sum_{i=1}^{n}
\mathcal{L}(f_\theta(x_i), y_i)
+
\lambda\Omega(\theta)
\right]
]

Where:

* (n) is the number of training examples;
* (x_i) is the input of example (i);
* (y_i) is its target;
* (f_\theta) is the model parameterized by (\theta);
* (\mathcal{L}) is the loss function;
* (\Omega(\theta)) is a regularization term;
* (\lambda) controls the strength of regularization;
* (\theta^*) is the parameter set that minimizes the objective.

The first term measures how well the model fits the observed data. The second discourages overly complex solutions.

This framework includes many methods:

* linear regression;
* logistic regression;
* neural networks;
* gradient boosting;
* support vector machines.

---

### 4.2 Neural representation learning

A neural-network layer can be represented as:

[
h = \phi(Wx + b)
]

Where:

* (x) is the input vector;
* (W) is a learnable weight matrix;
* (b) is a learnable bias vector;
* (\phi) is a nonlinear activation function;
* (h) is the resulting hidden representation.

Multiple layers compose transformations:

[
h^{(l)} =
\phi
\left(
W^{(l)}h^{(l-1)} + b^{(l)}
\right)
]

The superscript (l) identifies the layer.

Deep Learning learns both the representations (h) and the final prediction function.

---

### 4.3 Autoregressive language modeling

An autoregressive LLM models the probability of a sequence:

[
P(x_1, x_2, \ldots, x_T)
========================

\prod_{t=1}^{T}
P(x_t \mid x_1, \ldots, x_{t-1})
]

Where:

* (T) is the sequence length;
* (x_t) is the token at position (t);
* (x_1, \ldots, x_{t-1}) are the previous tokens.

Training usually minimizes negative log-likelihood:

[
\mathcal{L}_{LLM}
=================

-\sum_{t=1}^{T}
\log P_\theta(x_t \mid x_{<t})
]

Where:

* (\theta) represents model parameters;
* (x_{<t}) represents all tokens before position (t).

This objective explains an important limitation: the model is optimized to predict likely continuations, not directly to guarantee truth.

---

### 4.4 Self-attention

Scaled dot-product attention is:

[
\operatorname{Attention}(Q,K,V)
===============================

\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
]

Where:

* (Q) is the query matrix;
* (K) is the key matrix;
* (V) is the value matrix;
* (d_k) is the dimensionality of each key;
* (QK^\top) measures compatibility between queries and keys;
* division by (\sqrt{d_k}) stabilizes the scale;
* softmax converts scores into normalized weights.

Conceptually, each token uses its query representation to determine how much attention it should assign to other tokens.

---

### 4.5 Embedding similarity

A common retrieval metric is cosine similarity:

[
\operatorname{cosine}(q,d)
==========================

\frac{q \cdot d}
{|q|_2 |d|_2}
]

Where:

* (q) is the query embedding;
* (d) is a document embedding;
* (q \cdot d) is their dot product;
* (|q|_2) and (|d|_2) are their Euclidean norms.

For normalized embeddings:

[
|q|_2 = |d|_2 = 1
]

Therefore:

[
\operatorname{cosine}(q,d) = q \cdot d
]

The highest embedding similarity is not necessarily the most useful evidence. Semantic similarity, factual relevance and answer completeness are related but distinct properties.

---

### 4.6 RAG as a latent-document model

A conceptual RAG formulation is:

[
P(y \mid x, D)
\approx
\sum_{z \in R_k(x,D)}
P_\eta(z \mid x,D)
P_\theta(y \mid x,z)
]

Where:

* (x) is the user query;
* (D) is the full document collection;
* (R_k(x,D)) is the set of the top (k) retrieved documents or chunks;
* (z) is one retrieved item;
* (P_\eta(z \mid x,D)) represents retriever relevance;
* (P_\theta(y \mid x,z)) represents the generator probability of output (y) given the query and retrieved evidence.

This decomposition clarifies why RAG should be evaluated in stages:

```text
Did retrieval find the correct evidence?
Did generation use the evidence correctly?
```

---

### 4.7 Agent decision policy

An agent can be represented abstractly as a policy:

[
a_t \sim \pi(a \mid s_t)
]

Where:

* (s_t) is the system state at step (t);
* (a_t) is the chosen action;
* (\pi) is the decision policy;
* (a_t) may be sampled probabilistically or selected deterministically.

The next state is:

[
s_{t+1} = T(s_t, a_t, o_{t+1})
]

Where:

* (T) is the state-transition function;
* (o_{t+1}) is the observation returned after executing the action.

In LLM agents, (\pi) is partly implemented by the model and partly constrained by code, prompts, schemas, permissions and workflow rules.

---

### 4.8 Multi-objective system optimization

A production AI system is rarely optimized only for quality.

A simplified utility function is:

[
U =
Q
-

## \alpha C

## \beta L

\gamma R
]

Where:

* (U) is overall system utility;
* (Q) is output quality;
* (C) is financial cost;
* (L) is latency;
* (R) is operational or business risk;
* (\alpha,\beta,\gamma) express the importance of each constraint.

A larger model may improve (Q), while also increasing (C) and (L). A senior design decision considers the complete utility function, not only benchmark quality.

---

## 5. Practical applicability

### Choosing the appropriate level

| Problem                                             | Likely starting point                         | Why                                               |
| --------------------------------------------------- | --------------------------------------------- | ------------------------------------------------- |
| Predict customer churn                              | Classical supervised ML                       | Structured data and measurable target             |
| Forecast monthly demand                             | Statistical or ML forecasting                 | Temporal dependencies and numeric output          |
| Classify support tickets                            | Classical ML or small language model          | Limited output space and clear labels             |
| Summarize long legal documents                      | LLM with document-processing pipeline         | Unstructured language transformation              |
| Answer questions over private policies              | RAG                                           | External, private and updateable knowledge        |
| Extract fixed fields from invoices                  | OCR plus extraction model or schema-based LLM | Structured output from documents                  |
| Decide which API to call from an open-ended request | Agent or constrained tool router              | Dynamic action selection                          |
| Execute a fixed five-step document pipeline         | Deterministic workflow                        | Predictable sequence; autonomy is unnecessary     |
| Produce financial calculations                      | Deterministic code or SQL                     | Exactness is more important than generation       |
| Recommend actions based on business constraints     | Rules, optimization or hybrid system          | May require guarantees that an LLM cannot provide |

### When classical ML is preferable

Use classical ML when:

* the data is structured;
* the target is clearly defined;
* labeled examples are available;
* low latency matters;
* interpretability is important;
* the output space is constrained;
* cost must remain very low.

Examples:

* fraud scoring;
* credit-risk prediction;
* churn;
* lead scoring;
* tabular classification;
* anomaly detection.

### When an LLM is preferable

Use an LLM when the problem involves:

* open-ended language;
* summarization;
* extraction from heterogeneous text;
* semantic transformation;
* conversational interaction;
* code generation;
* reasoning over instructions;
* flexible tool selection.

### When RAG is appropriate

RAG makes sense when:

* answers depend on an external corpus;
* information changes frequently;
* users need evidence or citations;
* documents are private;
* retrievable access controls are required.

RAG is not automatically necessary when:

* the complete context is already small;
* the task is pure transformation;
* the answer is entirely contained in the user input;
* retrieval quality cannot be made reliable;
* the source material is poorly digitized.

### When agents are appropriate

Agents make sense when the next action depends on intermediate observations.

They should not be the default for:

* fixed ETL processes;
* deterministic document pipelines;
* one-tool calls;
* workflows with strict regulatory sequences;
* operations that cannot tolerate nondeterministic decisions.

---

## 6. Common pitfalls and mistakes

### Conceptual mistakes

1. **Treating all AI systems as Machine Learning**

   Many useful AI products contain rules, search, optimization, retrieval and standard software components.

2. **Describing RAG as a model**

   RAG is an architecture or inference pattern, not a single model family.

3. **Describing every tool-calling application as an agent**

   A predefined call to a search tool is a workflow. Agentic behavior requires dynamic action selection.

4. **Presenting LLMOps as completely separate from MLOps**

   LLMOps extends many MLOps principles while adding prompt, retrieval, tracing and generative-evaluation concerns.

5. **Assuming a larger model always produces a better system**

   A larger model may increase cost and latency without improving the business metric.

### Data and evaluation mistakes

6. **Data leakage**

   Information unavailable at prediction time is included during training or evaluation.

7. **Training-serving skew**

   Features or preprocessing differ between training and production.

8. **Using only aggregate accuracy**

   Important errors may be hidden in specific classes, user groups or critical scenarios.

9. **Evaluating only the final RAG answer**

   Retrieval recall and ranking must be measured separately from generation quality.

10. **Using an LLM as the only evaluator**

    Model-based judges may be useful, but they need calibration, human validation and task-specific criteria.

11. **Ignoring negative and abstention cases**

    A system must know when no reliable answer exists.

### System design mistakes

12. **Starting with the framework**

    “We should use LangGraph” is not a system requirement. First define the task and constraints.

13. **Using prompts as the entire architecture**

    Critical behavior should also be enforced through schemas, code, permissions and validation.

14. **No idempotency or retry strategy**

    Tool calls and asynchronous workflows may execute multiple times.

15. **No state model**

    Agent systems need an explicit representation of what has happened, what remains pending and what actions are allowed.

16. **No human escalation path**

    High-risk or low-confidence cases need controlled fallback.

17. **Authorization after retrieval**

    Access control should be enforced before sensitive documents are retrieved and exposed to the model.

18. **Ignoring prompt injection**

    Retrieved documents and user inputs may contain malicious instructions.

19. **Logging sensitive content indiscriminately**

    Full prompts, documents and outputs may contain personal or confidential information.

20. **Monitoring infrastructure but not behavior**

    A healthy API can still produce low-quality, unsafe or unsupported answers.

---

## 7. Important comparisons

### Machine Learning versus Deep Learning versus LLMs

| Aspect              | Classical ML        | Deep Learning                            | LLMs                                 |
| ------------------- | ------------------- | ---------------------------------------- | ------------------------------------ |
| Typical data        | Structured/tabular  | Images, audio, text, large-scale signals | Text, code and multimodal sequences  |
| Feature engineering | Often explicit      | Mostly learned                           | Learned during pretraining           |
| Data requirements   | Low to medium       | Medium to very high                      | Usually very high during pretraining |
| Inference cost      | Usually low         | Medium to high                           | Medium to very high                  |
| Output structure    | Usually constrained | Task-specific                            | Flexible and generative              |
| Interpretability    | Often higher        | Usually lower                            | Complex and behavior-oriented        |
| Adaptation          | Retraining          | Fine-tuning or transfer learning         | Prompting, RAG, tools, fine-tuning   |

### RAG versus fine-tuning

| RAG                                       | Fine-tuning                                        |
| ----------------------------------------- | -------------------------------------------------- |
| Adds external knowledge at inference time | Changes model behavior through training            |
| Easier to update knowledge                | Knowledge updates require new training             |
| Can expose supporting sources             | Does not inherently provide citations              |
| Depends strongly on retrieval quality     | Depends strongly on training-data quality          |
| Useful for private and dynamic knowledge  | Useful for style, behavior and task adaptation     |
| Increases runtime architecture complexity | Increases training and model-management complexity |

They are not mutually exclusive. A fine-tuned model can still use RAG.

A useful rule is:

* Use **RAG** primarily to provide knowledge.
* Use **fine-tuning** primarily to modify behavior, style, format or task specialization.

### Workflow versus agent

| Workflow                         | Agent                                      |
| -------------------------------- | ------------------------------------------ |
| Execution path mostly predefined | Execution path may be selected dynamically |
| Easier to test                   | Larger behavioral state space              |
| More deterministic               | More flexible                              |
| Easier to audit                  | Requires stronger traceability             |
| Suitable for stable processes    | Suitable for variable or open-ended tasks  |
| Lower operational risk           | Higher autonomy and failure risk           |

### MLOps versus LLMOps

| MLOps focus            | Additional LLMOps focus                     |
| ---------------------- | ------------------------------------------- |
| Features and labels    | Prompts and contexts                        |
| Model artifacts        | Model-provider and prompt versions          |
| Prediction metrics     | Response-quality dimensions                 |
| Data and concept drift | Retrieval, behavior and prompt drift        |
| Model-serving logs     | Full execution traces                       |
| Model registry         | Prompt, tool and evaluation registries      |
| Retraining             | Prompt, retrieval, routing or model updates |

### Model-centric versus system-centric thinking

A model-centric question is:

> Which model has the highest accuracy?

A system-centric question is:

> Which architecture satisfies the required quality, latency, cost, safety and maintainability under real traffic and data conditions?

Senior AI Engineering requires system-centric thinking.

---

## 8. Practical Python example

The following example demonstrates several layers of the landscape without requiring an external LLM:

* classical ML for ticket classification;
* retrieval over a knowledge base;
* a controller that chooses whether to answer or escalate;
* structured observability information.

Install the dependencies:

```bash
pip install numpy scikit-learn
```

```python
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline


TRAINING_DATA = [
    ("I was charged twice for the same purchase", "billing"),
    ("My invoice contains an incorrect amount", "billing"),
    ("I need a refund for a duplicated payment", "billing"),
    ("The subscription price changed unexpectedly", "billing"),
    ("My credit card payment failed", "billing"),
    ("Where can I download my invoice?", "billing"),
    ("The application crashes when I upload a PDF", "technical"),
    ("The API returns a 500 error", "technical"),
    ("Document processing is stuck", "technical"),
    ("The dashboard is not loading", "technical"),
    ("The integration stopped sending events", "technical"),
    ("The system is very slow today", "technical"),
    ("I cannot sign in to my account", "account"),
    ("I forgot my password", "account"),
    ("How can I change my email address?", "account"),
    ("My account has been locked", "account"),
    ("I need to enable two-factor authentication", "account"),
    ("How do I delete my account?", "account"),
]

KNOWLEDGE_BASE = [
    {
        "id": "billing-duplicate",
        "category": "billing",
        "text": (
            "For duplicate charges, confirm whether both payments were settled. "
            "Pending authorizations usually disappear automatically. "
            "If both charges were settled, open a billing dispute."
        ),
    },
    {
        "id": "billing-invoice",
        "category": "billing",
        "text": (
            "Invoices can be downloaded from Settings, Billing, Invoices. "
            "Invoice corrections require a billing support request."
        ),
    },
    {
        "id": "technical-upload",
        "category": "technical",
        "text": (
            "For document upload failures, validate the file format, file size "
            "and API response. Retry only transient 5xx errors with backoff."
        ),
    },
    {
        "id": "technical-api",
        "category": "technical",
        "text": (
            "For API 500 errors, record the request identifier, endpoint and "
            "timestamp. Do not retry non-idempotent operations automatically."
        ),
    },
    {
        "id": "account-access",
        "category": "account",
        "text": (
            "For account access problems, use the password-reset flow. "
            "Locked accounts may require identity verification by support."
        ),
    },
    {
        "id": "account-security",
        "category": "account",
        "text": (
            "Two-factor authentication can be enabled in Settings, Security. "
            "Store recovery codes in a secure location."
        ),
    },
]


@dataclass
class Decision:
    query: str
    predicted_category: str
    classifier_confidence: float
    retrieved_document_id: str
    retrieval_similarity: float
    action: str
    response: str
    latency_ms: float


def build_classifier() -> Pipeline:
    texts = [text for text, _ in TRAINING_DATA]
    labels = [label for _, label in TRAINING_DATA]

    classifier = Pipeline(
        steps=[
            (
                "vectorizer",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    lowercase=True,
                    stop_words="english",
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1_000,
                    random_state=42,
                ),
            ),
        ]
    )
    classifier.fit(texts, labels)
    return classifier


class KnowledgeRetriever:
    def __init__(self, documents: list[dict[str, str]]) -> None:
        self.documents = documents
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            lowercase=True,
            stop_words="english",
        )
        self.document_matrix = self.vectorizer.fit_transform(
            document["text"] for document in documents
        )

    def retrieve(
        self,
        query: str,
        category: str | None = None,
    ) -> tuple[dict[str, str], float]:
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(
            query_vector,
            self.document_matrix,
        )[0]

        candidate_indices = [
            index
            for index, document in enumerate(self.documents)
            if category is None or document["category"] == category
        ]

        if not candidate_indices:
            raise ValueError("No knowledge-base candidates were found.")

        best_index = max(
            candidate_indices,
            key=lambda index: similarities[index],
        )

        return self.documents[best_index], float(similarities[best_index])


def process_ticket(
    query: str,
    classifier: Pipeline,
    retriever: KnowledgeRetriever,
    confidence_threshold: float = 0.55,
    retrieval_threshold: float = 0.05,
) -> Decision:
    started_at = time.perf_counter()

    probabilities = classifier.predict_proba([query])[0]
    classes = classifier.named_steps["model"].classes_
    best_class_index = int(np.argmax(probabilities))

    predicted_category = str(classes[best_class_index])
    confidence = float(probabilities[best_class_index])

    document, similarity = retriever.retrieve(
        query=query,
        category=predicted_category,
    )

    if confidence < confidence_threshold:
        action = "human_review"
        response = (
            "The request could not be classified with sufficient confidence."
        )
    elif similarity < retrieval_threshold:
        action = "human_review"
        response = (
            "The category was identified, but no sufficiently relevant "
            "knowledge-base evidence was found."
        )
    else:
        action = "retrieval_augmented_response"
        response = (
            f"Category: {predicted_category}. "
            f"Relevant guidance: {document['text']}"
        )

    latency_ms = (time.perf_counter() - started_at) * 1_000

    return Decision(
        query=query,
        predicted_category=predicted_category,
        classifier_confidence=confidence,
        retrieved_document_id=document["id"],
        retrieval_similarity=similarity,
        action=action,
        response=response,
        latency_ms=latency_ms,
    )


def main() -> None:
    classifier = build_classifier()
    retriever = KnowledgeRetriever(KNOWLEDGE_BASE)

    example_queries = [
        "I paid the same invoice twice",
        "The API fails while uploading a document",
        "I cannot access my profile and reset does not work",
        "Can you negotiate a new commercial contract for me?",
    ]

    for query in example_queries:
        decision = process_ticket(
            query=query,
            classifier=classifier,
            retriever=retriever,
        )
        print(json.dumps(asdict(decision), indent=2))
        print("-" * 80)


if __name__ == "__main__":
    main()
```

### What each part represents

```text
TfidfVectorizer
→ text representation

LogisticRegression
→ classical supervised ML model

KnowledgeRetriever
→ retrieval component

process_ticket
→ orchestration and decision policy

confidence thresholds
→ guardrails and abstention

Decision dataclass
→ structured trace and observability data
```

In a production GenAI version, the final template could be replaced by an LLM call that receives:

* the original query;
* the predicted category;
* the retrieved document;
* explicit grounding instructions;
* a structured output schema.

The classifier could also be replaced by an LLM router, but that would introduce additional cost, latency and nondeterminism.

---

## 9. From-scratch implementation when useful

A simplified bag-of-words retriever demonstrates what happens before advanced embeddings or vector databases are introduced.

```python
from __future__ import annotations

import re
from collections import Counter

import numpy as np


DOCUMENTS = [
    "duplicate payments require a billing dispute",
    "password reset is available in account security",
    "api failures should include the request identifier",
]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def build_vocabulary(texts: list[str]) -> list[str]:
    vocabulary = sorted(
        {
            token
            for text in texts
            for token in tokenize(text)
        }
    )
    return vocabulary


def vectorize(text: str, vocabulary: list[str]) -> np.ndarray:
    counts = Counter(tokenize(text))
    return np.array(
        [counts[token] for token in vocabulary],
        dtype=float,
    )


def cosine_similarity(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    denominator = np.linalg.norm(first) * np.linalg.norm(second)

    if denominator == 0:
        return 0.0

    return float(np.dot(first, second) / denominator)


def retrieve(query: str) -> tuple[str, float]:
    vocabulary = build_vocabulary(DOCUMENTS + [query])
    query_vector = vectorize(query, vocabulary)

    scores = [
        cosine_similarity(
            query_vector,
            vectorize(document, vocabulary),
        )
        for document in DOCUMENTS
    ]

    best_index = int(np.argmax(scores))
    return DOCUMENTS[best_index], scores[best_index]


if __name__ == "__main__":
    result, score = retrieve("I have two payments for one invoice")
    print(f"Document: {result}")
    print(f"Similarity: {score:.3f}")
```

This implementation has major limitations:

* it cannot understand synonyms;
* it ignores word order;
* it does not weight rare terms;
* it cannot represent semantic similarity;
* it rebuilds the vocabulary inefficiently.

However, it exposes the core retrieval mechanism:

```text
Text
→ numerical vector
→ similarity function
→ ranking
```

TF-IDF improves lexical weighting. Dense embeddings improve semantic representation. Vector databases improve indexing, filtering and large-scale search.

---

## 10. Suggested experiments

### Experiment 1 — Change the abstention threshold

Test:

```python
confidence_threshold = 0.40
confidence_threshold = 0.70
confidence_threshold = 0.90
```

Observe the trade-off between:

* automation rate;
* false routing;
* human-review volume.

This is a production decision, not merely a modeling decision.

### Experiment 2 — Remove category filtering from retrieval

Change:

```python
retriever.retrieve(query, category=predicted_category)
```

to:

```python
retriever.retrieve(query, category=None)
```

Analyze whether classifier errors cause retrieval errors and whether unrestricted retrieval sometimes recovers from incorrect classification.

### Experiment 3 — Compare lexical representations

Test:

* unigram TF-IDF;
* unigram plus bigram TF-IDF;
* raw term counts;
* character n-grams.

Pay attention to spelling variations and short queries.

### Experiment 4 — Add a second-stage reranker

Retrieve the top three documents and rerank them using:

* category match;
* keyword overlap;
* a manually defined business score.

This illustrates the separation between candidate retrieval and final ranking.

### Experiment 5 — Add production metrics

Collect:

* classifier confidence;
* retrieval similarity;
* action selected;
* latency;
* escalation rate;
* category frequency;
* failures by query type.

Then define which metrics are operational and which represent business quality.

---

## 11. Senior interview questions

### 1. How do Data Science, Machine Learning and AI Engineering differ?

**Answer:**

Data Science focuses on using data to produce insights, predictions and decisions. It includes statistics, experimentation, analytics and machine learning.

Machine Learning focuses specifically on algorithms that learn behavior from data.

AI Engineering focuses on integrating models into reliable software systems. It includes model serving, retrieval, orchestration, APIs, observability, security and production operations.

The disciplines overlap, but their primary deliverables differ: analysis and decisions, learned models, and production systems.

---

### 2. When would you choose classical ML instead of an LLM?

**Answer:**

I would prefer classical ML when the input is structured, the output space is constrained, labeled data is available and cost, latency or interpretability matter.

Examples include churn prediction, fraud scoring and ticket classification.

An LLM becomes more appropriate when the task involves open-ended language, semantic transformation, heterogeneous documents or dynamic instruction following. I would not use an LLM merely because it is more modern.

---

### 3. What problem does RAG solve?

**Answer:**

RAG provides an LLM with external evidence at inference time. It is useful for private, dynamic or domain-specific knowledge that should not depend only on model parameters.

It does not inherently solve hallucination. The system still needs reliable ingestion, retrieval, reranking, context construction, generation constraints and evaluation.

I would separately measure retrieval quality and answer quality.

---

### 4. How would you evaluate a RAG system?

**Answer:**

I would separate the pipeline into layers.

For retrieval:

* recall at (k);
* precision at (k);
* mean reciprocal rank;
* normalized discounted cumulative gain;
* evidence coverage;
* access-control correctness.

For generation:

* factual consistency with retrieved context;
* answer correctness;
* completeness;
* citation correctness;
* abstention behavior;
* format compliance;
* safety.

For the complete product:

* task-completion rate;
* user satisfaction;
* latency;
* cost;
* escalation rate;
* business impact.

I would use a versioned evaluation dataset combining synthetic cases, historical cases, adversarial cases and human-reviewed examples.

---

### 5. What makes a system agentic?

**Answer:**

A system is agentic when a model dynamically participates in selecting the next action or transition based on the current state.

Tool use alone is not sufficient. A deterministic pipeline that always invokes the same tool remains a workflow.

I would introduce agentic behavior only when variability in task execution creates enough value to justify greater nondeterminism, evaluation complexity and operational risk.

---

### 6. How do you decide between an agent and a workflow?

**Answer:**

I examine how variable the execution path actually is.

A workflow is preferable when:

* the process is known;
* compliance requires a fixed sequence;
* failure modes need to be bounded;
* auditability is critical.

An agent is useful when:

* goals are open-ended;
* tool selection depends on observations;
* different requests require different plans;
* the system must adapt during execution.

A common production design is hybrid: deterministic outer control with bounded agentic decisions inside specific steps.

---

### 7. How is LLMOps different from MLOps?

**Answer:**

LLMOps retains core MLOps requirements such as versioning, deployment, validation, monitoring and governance.

It adds artifacts and failure modes specific to generative systems:

* prompt versions;
* retrieved contexts;
* tool traces;
* model-provider changes;
* token usage;
* output-quality evaluation;
* grounding;
* prompt injection;
* structured-output validation.

Conventional ML often produces one prediction. An LLM system may produce a multi-step execution trace that must be evaluated and debugged.

---

### 8. How would you design a production RAG system on GCP?

**Answer:**

I would first define quality, latency, security and volume requirements.

A possible architecture would include:

* Cloud Storage for source documents;
* Document AI or specialized parsers for extraction;
* Cloud Run or Dataflow for parsing and chunking;
* BigQuery for metadata, lineage and evaluation data;
* Vertex AI embeddings;
* Vertex AI Vector Search or another index for semantic retrieval;
* optional lexical retrieval and reranking;
* Gemini through Vertex AI for generation;
* Cloud Run for the application API and orchestration;
* Pub/Sub for asynchronous ingestion;
* IAM and document-level access controls;
* Cloud Logging, Trace and custom evaluation telemetry;
* versioned datasets and prompts for regression tests.

I would enforce user authorization before retrieval, preserve document identifiers through the pipeline and separately monitor retrieval and generation.

---

### 9. How would you reduce the cost of an LLM system?

**Answer:**

I would avoid beginning with model downsizing alone.

I would examine:

* whether every request needs an LLM;
* whether classification can be done by rules or a smaller model;
* prompt and context length;
* retrieval precision;
* output-token limits;
* caching;
* batch processing;
* model routing;
* unnecessary retries;
* duplicate tool calls;
* asynchronous execution;
* precomputation.

Cost optimization should preserve the task-level quality target. The best architecture may route simple requests to deterministic logic and complex requests to a stronger model.

---

### 10. What are the main failure modes of an AI agent?

**Answer:**

Important failure modes include:

* selecting the wrong tool;
* supplying invalid arguments;
* repeating actions;
* losing state;
* following malicious instructions;
* operating with excessive permissions;
* producing a correct plan but failing during execution;
* not recognizing that a task is complete;
* executing irreversible operations without confirmation;
* propagating incorrect tool outputs.

Controls include typed tool schemas, permission boundaries, maximum-step limits, idempotency keys, validation, state machines, human approval and complete traces.

---

### 11. Why is offline model quality insufficient?

**Answer:**

Offline metrics measure only a controlled approximation of production behavior.

Production introduces:

* changing data;
* user ambiguity;
* infrastructure failures;
* latency limits;
* cost constraints;
* adversarial inputs;
* integration errors;
* feedback loops;
* human behavior.

A model with higher offline accuracy may create a worse product if it is slower, more expensive, poorly calibrated or difficult to maintain.

---

### 12. What is the most important distinction in the modern AI landscape?

**Answer:**

The most important distinction is between a **model capability** and an **operational system**.

LLMs, classifiers and embedding models provide capabilities. RAG, agents, workflows, evaluation, security and observability turn those capabilities into a system.

Senior engineers are expected to reason about the interaction between both levels.

---

## 12. Interview-ready explanation

> The modern AI landscape can be understood as a combination of model families, system patterns and operational disciplines. Machine Learning learns predictive behavior from data, while Deep Learning learns hierarchical representations using neural networks. LLMs are large autoregressive foundation models specialized in language and related modalities.
>
> RAG and agents are not separate model families. RAG is a system pattern that retrieves external evidence before generation, while agents use a model as part of a control loop that can dynamically select actions or tools.
>
> MLOps manages the lifecycle of learned models, including reproducibility, deployment and monitoring. LLMOps extends this with prompt versioning, retrieval evaluation, execution tracing, cost monitoring and generative safety.
>
> In a real project, I would begin with the business decision, data, quality requirements and operational constraints. I would then choose the simplest architecture capable of meeting them. A structured prediction problem may need classical ML, a private knowledge assistant may need RAG, and an open-ended multi-tool task may justify a bounded agent. The objective is not to use the most advanced model, but to build the most reliable and valuable system.

---

## 13. GitHub file structure

```text
day-01-ai-ml-genai-landscape/
├── README.md
├── notes.md
├── notebook.ipynb
├── example.py
├── from_scratch.py
├── interview_questions.md
├── references.md
└── diagrams/
    └── ai_landscape.md
```

### Suggested responsibilities

```text
README.md
→ concise public overview and execution instructions

notes.md
→ deeper theoretical notes and production trade-offs

notebook.ipynb
→ interactive exploration of classification, retrieval and routing

example.py
→ executable end-to-end example

from_scratch.py
→ simplified vectorization and cosine retrieval

interview_questions.md
→ questions, answers and personal response drafts

references.md
→ books, papers, documentation and further reading

diagrams/ai_landscape.md
→ Mermaid or text diagrams describing the architecture
```

A suitable repository-level organization would be:

```text
applied-ai-engineering-lab/
├── README.md
├── foundations/
│   └── day-01-ai-ml-genai-landscape/
├── machine-learning/
├── deep-learning/
├── llms/
├── rag/
├── agents/
├── mlops-llmops/
└── system-design/
```

---

## 14. Suggested README.md content

# Day 1 — AI, Machine Learning and Generative AI Landscape

## Objective

This study module provides a system-level overview of the modern AI landscape, covering Data Science, Machine Learning, Deep Learning, Large Language Models, Retrieval-Augmented Generation, AI agents, MLOps, LLMOps and AI system design.

The objective is to understand not only how these concepts relate, but also how to select an appropriate approach for a real production problem.

## Concepts covered

* Data Science and analytical decision-making
* Supervised, unsupervised and self-supervised learning
* Deep representation learning
* Foundation models and Large Language Models
* Retrieval-Augmented Generation
* Tools, workflows and AI agents
* MLOps and LLMOps
* Production AI system design
* Quality, latency, cost and risk trade-offs

## Practical example

The example implements a simplified support-ticket system composed of:

1. a TF-IDF text representation;
2. a logistic-regression classifier;
3. a knowledge-base retriever;
4. a routing policy;
5. confidence-based human escalation;
6. structured execution telemetry.

This architecture demonstrates how predictive models, retrieval and orchestration can be combined without introducing unnecessary framework complexity.

## Project structure

```text
.
├── README.md
├── notes.md
├── notebook.ipynb
├── example.py
├── from_scratch.py
├── interview_questions.md
└── references.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scikit-learn
```

On Windows:

```bash
.venv\Scripts\activate
pip install numpy scikit-learn
```

## Running the example

```bash
python example.py
```

Run the simplified retrieval implementation with:

```bash
python from_scratch.py
```

## Key takeaways

* Machine Learning, Deep Learning and LLMs represent different model families and learning paradigms.
* RAG and agents are system architectures built around models.
* Tool use does not automatically make a system agentic.
* LLMOps extends MLOps with prompts, retrieval, traces, generative evaluation and safety.
* Production AI quality depends on the complete system, not only on model performance.
* The simplest architecture that satisfies the business and operational requirements is usually the best starting point.

A stronger future version can include a Mermaid architecture diagram and an automated test suite for routing and retrieval behavior.

---

## 15. LinkedIn post idea

Nem todo sistema com LLM precisa de RAG. Nem todo sistema com ferramentas precisa ser um agente. E nem todo problema de dados precisa de IA generativa.

Ao revisar o panorama de Data Science, Machine Learning e GenAI, uma distinção se torna especialmente importante: modelos e sistemas não são a mesma coisa.

Um classificador, um modelo de embeddings ou um LLM oferecem capacidades específicas. O valor em produção aparece quando essas capacidades são combinadas corretamente com dados, recuperação de informação, regras, APIs, observabilidade, segurança e avaliação.

Na prática, a decisão mais madura nem sempre é escolher o modelo mais avançado. É escolher a arquitetura mais simples que consiga atender aos requisitos de qualidade, custo, latência e risco.

Documentei no GitHub um mapa desse ecossistema e um pequeno exemplo conectando classificação, recuperação de conhecimento, roteamento e fallback para revisão humana.

Esse conteúdo funciona melhor como parte de uma publicação semanal reunindo dois ou três aprendizados, em vez de um registro isolado de cada dia.

---

## 16. 30–60 minute checklist

### Core session — approximately 30 minutes

* [ ] Read sections 1–3 and reproduce the landscape without consulting the text.
* [ ] Explain aloud why RAG is not a model and why tool use is not automatically agentic.
* [ ] Review the empirical-risk, LLM-probability, attention and cosine-similarity formulas.
* [ ] Run `example.py` with the four sample queries.
* [ ] Change one confidence threshold and observe the routing behavior.
* [ ] Write a five-sentence answer to: “How do ML, LLMs, RAG and agents relate?”

### Extended session — up to 60 minutes

* [ ] Complete the core session.
* [ ] Run `from_scratch.py`.
* [ ] Add two documents to the knowledge base.
* [ ] Add one adversarial or out-of-domain query.
* [ ] Record which component failed: classification, retrieval or routing.
* [ ] Answer five interview questions without reading the prepared answers.
* [ ] Create the GitHub folder and commit the initial files.
* [ ] Add one Mermaid diagram showing data, intelligence, orchestration and operations planes.

### Definition of done

By the end of Day 1, you should be able to explain:

```text
1. What belongs to the model layer.
2. What belongs to the system-architecture layer.
3. What belongs to the operational layer.
4. When classical ML is better than an LLM.
5. When RAG is better than putting more information in a prompt.
6. When a workflow is safer than an agent.
7. Why production evaluation must include quality, latency, cost and risk.
```

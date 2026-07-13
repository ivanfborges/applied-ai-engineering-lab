# Study Roadmap — Data Science and Applied AI Engineering

This roadmap is designed for a daily 30–60 minute study routine. It can be followed sequentially, but topics can be reordered depending on interview priorities.

The roadmap is intentionally dense. Some topics may take more than one day if deeper implementation or portfolio work is needed.

---

## Module 0 — Foundations and Study Setup

### Day 1 — AI/ML/GenAI Landscape and Roadmap Overview
Understand the full landscape: Data Science, Machine Learning, Deep Learning, LLMs, RAG, agents, MLOps, LLMOps and AI system design.

### Day 2 — Linear Algebra I: Vectors, Matrices and Operations
Vectors, matrices, dot product, norms, distances, matrix multiplication and geometric interpretation.

### Day 3 — Linear Algebra II: Vector Spaces, Bases and Projections
Vector spaces, basis, dimension, linear independence, projections and why embeddings live in vector spaces.

### Day 4 — Linear Algebra III: Eigenvalues, Eigenvectors, PCA and SVD
Eigen decomposition, singular value decomposition and how they connect to dimensionality reduction.

### Day 5 — Calculus for ML: Derivatives, Gradients and Chain Rule
Derivatives, partial derivatives, gradients and why optimization depends on them.

### Day 6 — Gradient Descent from Scratch
Implement gradient descent for a simple regression problem and visualize convergence.

### Day 7 — Probability Essentials
Events, conditional probability, independence, expected value, variance and Bayes theorem.

### Day 8 — Probability Distributions
Bernoulli, Binomial, Normal, Poisson, Exponential and Log-normal distributions.

### Day 9 — Exploratory Data Analysis with Statistical Rigor
Mean, median, variance, standard deviation, skewness, kurtosis, outliers and correlation.

### Day 10 — Sampling, Bias and Variance
Sampling strategies, sample bias, estimator variance and why data quality matters.

### Day 11 — Central Limit Theorem and Confidence Intervals
CLT, standard error, confidence intervals and uncertainty communication.

### Day 12 — Hypothesis Testing
Null hypothesis, alternative hypothesis, p-value, Type I/II errors and statistical power.

### Day 13 — Correlation vs Causation
Correlation types, confounders, spurious correlation and causal traps.

### Day 14 — Maximum Likelihood Estimation and MAP
Likelihood, log-likelihood, MLE, MAP and connection to model training.

### Day 15 — Entropy, Cross-Entropy and KL Divergence
Information theory foundations for classification, neural networks and LLMs.

---

## Module 1 — Classical Machine Learning

### Day 16 — End-to-End ML Pipeline
Problem framing, data collection, features, split, baseline, training, validation and deployment.

### Day 17 — Train/Validation/Test Split, Cross-Validation and Leakage
Correct validation design, k-fold cross-validation, temporal split and leakage prevention.

### Day 18 — Linear Regression Theory
OLS, assumptions, residuals, coefficients, interpretation and limitations.

### Day 19 — Linear Regression from Scratch
Implement OLS and gradient descent versions; compare with scikit-learn.

### Day 20 — Regularization: Ridge, Lasso and ElasticNet
L1/L2 penalties, shrinkage, feature selection, bias-variance trade-off.

### Day 21 — Logistic Regression
Sigmoid, logit, odds, decision boundary, cross-entropy and classification.

### Day 22 — Classification Metrics I
Accuracy, precision, recall, F1-score and confusion matrix.

### Day 23 — Classification Metrics II
ROC-AUC, PR-AUC, threshold tuning and calibration.

### Day 24 — K-Nearest Neighbors
Distance-based learning, scaling, curse of dimensionality and practical limitations.

### Day 25 — Naive Bayes
Bayes theorem, conditional independence, text classification and probabilistic modeling.

### Day 26 — Decision Trees
Gini, entropy, information gain, splits, pruning and overfitting.

### Day 27 — Random Forest
Bagging, bootstrap sampling, feature randomness, OOB score and feature importance.

### Day 28 — ExtraTrees and Tree Ensemble Variants
How ExtraTrees differs from Random Forest and when additional randomness helps.

### Day 29 — Gradient Boosting Intuition
Weak learners, residuals, additive modeling and sequential correction.

### Day 30 — XGBoost, LightGBM and CatBoost
Modern gradient boosting algorithms for tabular data and their practical differences.

### Day 31 — Support Vector Machines
Maximum margin, support vectors, hinge loss and kernels.

### Day 32 — Feature Engineering for Tabular Data
Encoding, scaling, binning, interaction features and aggregation features.

### Day 33 — Imbalanced Data
Class weights, resampling, SMOTE, thresholding and metric selection.

### Day 34 — Missing Data
MCAR, MAR, MNAR, imputation strategies and model robustness.

### Day 35 — Explainability for Classical ML
Feature importance, permutation importance, PDP, ICE and SHAP.

---

## Module 2 — Unsupervised Learning and Recommender Systems

### Day 36 — K-Means Clustering
Centroids, inertia, elbow method, scaling and limitations.

### Day 37 — DBSCAN and Density-Based Clustering
Density, noise, epsilon, min_samples and non-spherical clusters.

### Day 38 — Hierarchical Clustering
Agglomerative clustering, linkage methods and dendrograms.

### Day 39 — PCA in Practice
Variance explained, projection, dimensionality reduction and interpretation.

### Day 40 — t-SNE and UMAP
High-dimensional visualization and common interpretation mistakes.

### Day 41 — Anomaly Detection
Isolation Forest, One-Class SVM, robust statistics and practical monitoring use cases.

### Day 42 — Recommender Systems I
Collaborative filtering, user-item matrix, implicit/explicit feedback and cold start.

### Day 43 — Recommender Systems II
Matrix factorization, embeddings, ranking models and hybrid recommenders.

### Day 44 — Ranking Metrics
Precision@K, Recall@K, MAP, MRR and NDCG.

---

## Module 3 — Experimentation, Causality and Product Thinking

### Day 45 — A/B Testing from Scratch
Experiment design, randomization, metrics, hypothesis and interpretation.

### Day 46 — Power Analysis and Sample Size
Statistical power, effect size, sample size estimation and business constraints.

### Day 47 — Product Metrics and Guardrails
Conversion, retention, churn, latency, revenue, cost and safety metrics.

### Day 48 — Multiple Testing and False Positives
Bonferroni correction, FDR and peeking problem.

### Day 49 — Causal Inference I: DAGs and Confounders
Directed acyclic graphs, confounding, colliders and backdoor criterion.

### Day 50 — Causal Inference II: Propensity Score and Matching
Treatment assignment, propensity scores, matching and inverse probability weighting.

### Day 51 — Difference-in-Differences
Before/after comparisons with treatment and control groups.

### Day 52 — Uplift Modeling
Incremental impact, treatment effect modeling and targeting interventions.

---

## Module 4 — Deep Learning

### Day 53 — Perceptron and MLPs
Neurons, weights, bias, layers, activations and universal approximation intuition.

### Day 54 — Backpropagation
Chain rule, computational graphs, gradients and parameter updates.

### Day 55 — Activation Functions
Sigmoid, tanh, ReLU, Leaky ReLU, GELU and softmax.

### Day 56 — Optimizers
SGD, Momentum, RMSProp, Adam and AdamW.

### Day 57 — Regularization in Deep Learning
Dropout, weight decay, batch normalization and early stopping.

### Day 58 — Weight Initialization and Training Stability
Xavier, He initialization, vanishing/exploding gradients and normalization.

### Day 59 — CNNs
Convolution, filters, receptive field, pooling and image applications.

### Day 60 — RNNs, LSTMs and GRUs
Sequential modeling before Transformers and their limitations.

### Day 61 — Autoencoders
Encoding, latent spaces, reconstruction and anomaly detection.

### Day 62 — PyTorch Training Loop
Dataset, DataLoader, model, loss, backward pass, optimizer and evaluation.

---

## Module 5 — NLP, Transformers and LLMs

### Day 63 — Classical NLP
Tokenization, normalization, stemming, lemmatization and n-grams.

### Day 64 — Bag of Words, TF-IDF and Text Similarity
Sparse representations and classical information retrieval foundations.

### Day 65 — Word2Vec, GloVe and Static Embeddings
CBOW, Skip-gram, distributional semantics and vector analogies.

### Day 66 — Attention Mechanism
Query, Key, Value, attention scores, softmax and weighted sums.

### Day 67 — Transformer Architecture I
Encoder-decoder architecture, attention blocks and sequence modeling.

### Day 68 — Transformer Architecture II: Multi-Head Attention
Parallel attention heads and representation subspaces.

### Day 69 — Positional Encoding
Why Transformers need position information and how it can be represented.

### Day 70 — Feed-Forward Networks, Residual Connections and Layer Normalization
Training stability and information flow inside Transformer blocks.

### Day 71 — Encoder-only, Decoder-only and Encoder-Decoder Models
BERT, GPT, T5-style architectures and use cases.

### Day 72 — Tokenization for LLMs
BPE, WordPiece, SentencePiece, vocabulary, token limits and cost implications.

### Day 73 — LLM Pretraining
Next-token prediction, masked language modeling, scale, data and compute.

### Day 74 — Fine-Tuning, Instruction Tuning and Alignment
SFT, RLHF, DPO, adapters, LoRA and when fine-tuning makes sense.

### Day 75 — LLM Inference and Decoding
Temperature, top-k, top-p, max tokens, stop sequences and deterministic output.

### Day 76 — Prompt Engineering with Rigor
Instructions, context, examples, constraints, format and failure handling.

### Day 77 — Structured Outputs
JSON schema, output parsing, validation, retries and guardrails.

---

## Module 6 — Embeddings, Semantic Search and RAG

### Day 78 — Modern Embeddings
Dense vectors, similarity, cosine distance, normalization and semantic representation.

### Day 79 — Vector Databases
FAISS, pgvector, Pinecone, Weaviate, Vertex AI Vector Search and retrieval architecture.

### Day 80 — Approximate Nearest Neighbors
HNSW, IVF, ScaNN, recall-latency trade-offs and indexing strategies.

### Day 81 — RAG Architecture I
Ingestion, chunking, embedding, indexing, retrieval and generation.

### Day 82 — RAG Chunking Strategies
Chunk size, overlap, semantic chunking, metadata, parent-child retrieval.

### Day 83 — Retrieval Strategies
Dense retrieval, sparse retrieval, hybrid search and metadata filtering.

### Day 84 — Reranking
Cross-encoders, rerankers, MMR and contextual compression.

### Day 85 — Grounding and Citations
Reducing hallucination, source attribution and traceability.

### Day 86 — RAG Evaluation
Faithfulness, answer relevance, context precision, context recall and golden datasets.

### Day 87 — RAG Chain vs RAG Agent
When retrieval should be a fixed pipeline and when it should be a tool.

### Day 88 — RAG in Production
Caching, latency, cost, fallback, monitoring and continuous evaluation.

---

## Module 7 — Agents and Agentic Workflows

### Day 89 — What Is an AI Agent?
Chatbot vs chain vs workflow vs agent vs multi-agent system.

### Day 90 — Tool Calling
Function schemas, arguments, API calls, validation, permissions and failure handling.

### Day 91 — ReAct Pattern
Reasoning, acting, observations and tool-augmented loops.

### Day 92 — Planning Agents
Plan-and-execute, task decomposition and long-running tasks.

### Day 93 — Reflection and Self-Correction
Critique, retry, self-evaluation and the risk of unproductive loops.

### Day 94 — Agent Memory
Short-term, long-term, episodic, semantic and user-profile memory.

### Day 95 — Multi-Agent Systems
Supervisor-worker patterns, routers, critics and collaboration structures.

### Day 96 — Human-in-the-Loop
Approval flows, escalation, auditability and risk mitigation.

### Day 97 — Agent Safety and Guardrails
Prompt injection, tool misuse, data leakage, PII and permission boundaries.

### Day 98 — Agent Observability
Tracing, tool calls, token usage, latency, errors and cost per task.

### Day 99 — Enterprise Conversational Agent Architecture
RAG, tools, APIs, database, fallback, analytics and human support integration.

### Day 100 — Practical Agent Project
Build a small tool-using agent with validation, logging and fallback.

---

## Module 8 — MLOps and LLMOps

### Day 101 — MLOps Overview
Experiment tracking, model registry, pipelines, deployment and monitoring.

### Day 102 — Feature Stores
Offline/online features, training-serving consistency and feature reuse.

### Day 103 — Model Registry and Versioning
Models, datasets, code, metrics, parameters and reproducibility.

### Day 104 — Model Serving Patterns
Batch, online, streaming, async, serverless and containerized serving.

### Day 105 — APIs for ML and AI Systems
FastAPI, REST, gRPC, validation, contracts and error handling.

### Day 106 — Monitoring Classical ML Models
Data drift, concept drift, prediction drift, skew and performance decay.

### Day 107 — Monitoring LLM Applications
Hallucination, groundedness, toxicity, latency, token usage and cost.

### Day 108 — LLM Evaluation
Golden datasets, regression tests, LLM-as-judge and human evaluation.

### Day 109 — CI/CD for ML and LLM Apps
Unit tests, integration tests, smoke tests, eval gates and deployment safety.

### Day 110 — Cost Optimization in GenAI
Token budget, caching, model routing, batching and fallback models.

### Day 111 — Latency Optimization in LLM Systems
Streaming, parallel calls, context reduction, retrieval latency and caching.

### Day 112 — Full-Stack Observability
Logs, traces, metrics, dashboards and alerts.

### Day 113 — AI Security and Compliance
PII, LGPD/GDPR, IAM, secrets, data residency and audit trails.

### Day 114 — Cloud Architecture for AI Systems
Storage, compute, pipelines, model serving, vector search and orchestration.

---

## Module 9 — AI System Design for Interviews

### Day 115 — How to Answer AI System Design Questions
Requirements, constraints, data, model, evaluation, production and risks.

### Day 116 — Design a Legal RAG Assistant
OCR, document ingestion, chunking, retrieval, citations, evaluation and compliance.

### Day 117 — Design a Ticket Classification System
Baselines, ML model, LLM assist, human review and monitoring.

### Day 118 — Design a Recommendation System
Candidate generation, ranking, feedback loop, metrics and cold start.

### Day 119 — Design a Financial Automation Agent
Tool permissions, human approval, logs, fallback and risk control.

### Day 120 — Design a Fraud/Anomaly Detection System
Features, imbalance, real-time scoring, explainability and drift.

### Day 121 — Design a Multimodal Document Intelligence Pipeline
PDFs, OCR, tables, images, embeddings, summarization and visual RAG.

### Day 122 — Design Continuous Evaluation for an AI Agent
Golden sets, simulated users, regression testing, tracing and dashboards.

---

## Module 10 — Interview Preparation

### Day 123 — Statistics Interview Questions
p-values, confidence intervals, hypothesis testing, bias, variance and sampling.

### Day 124 — Classical ML Interview Questions
Overfitting, regularization, leakage, metrics, imbalance and model selection.

### Day 125 — Model-Specific Interview Questions
Random Forest vs XGBoost, SVM vs Logistic Regression, PCA vs t-SNE.

### Day 126 — Deep Learning Interview Questions
Backpropagation, optimizers, dropout, batch norm and vanishing gradients.

### Day 127 — Transformer and LLM Interview Questions
Self-attention, positional encoding, context window, fine-tuning and decoding.

### Day 128 — RAG Interview Questions
Chunking, embeddings, vector DB, reranking, evaluation and hallucination control.

### Day 129 — Agent Interview Questions
Tool use, memory, planning, guardrails, observability and human-in-the-loop.

### Day 130 — MLOps and LLMOps Interview Questions
Drift, registry, monitoring, evals, CI/CD, rollback and cost control.

### Day 131 — Technical Storytelling
Explain real projects using problem, solution, trade-offs, impact and lessons learned.

### Day 132 — Full Mock Interview
Simulate theory, coding, system design and behavioral questions.

---

## Module 11 — Portfolio Projects

### Day 133 — Organize the Portfolio Repository
Review structure, improve root README and create project templates.

### Day 134 — Portfolio Project 1: Classical ML Tabular Case
End-to-end ML case with EDA, baseline, model comparison, explainability and README.

### Day 135 — Portfolio Project 2: A/B Testing and Causal Analysis
Simulation, hypothesis testing, confidence intervals and executive conclusion.

### Day 136 — Portfolio Project 3: RAG Evaluation Pipeline
Ingestion, chunking, vector search, generation, citations and evaluation metrics.

### Day 137 — Portfolio Project 4: Conversational Agent with Tools
Tool calling, memory, validation, fallback and tracing.

### Day 138 — Portfolio Project 5: LLMOps Mini Stack
Prompt versioning, eval dataset, cost tracking, latency tracking and dashboard.

### Day 139 — Project Documentation and LinkedIn Packaging
Improve READMEs, architecture diagrams and short public explanations.

### Day 140 — Final Review and Next Steps
Identify gaps, prioritize job-specific topics and plan deeper projects.

---

## Suggested Priority If Interviews Are Urgent

If interviews are happening soon, prioritize:

1. Days 11–15: statistics and information theory;
2. Days 17–35: classical ML and metrics;
3. Days 66–77: Transformers and LLMs;
4. Days 81–88: RAG;
5. Days 89–100: agents;
6. Days 106–114: monitoring, LLMOps and production;
7. Days 115–122: system design.

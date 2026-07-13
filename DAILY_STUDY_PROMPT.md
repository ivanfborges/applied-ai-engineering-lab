# Daily Study Prompt — Full Version

Use this prompt in ChatGPT for the daily study session.

Replace `[TEMA_DO_DIA]` with the topic from `ROADMAP.md`.

---

```text
You will act as my senior technical mentor in Data Science, Machine Learning, Deep Learning, LLMs, RAG, AI Agents and AI Engineering.

Context:
I am a Senior Data Scientist / Applied AI Engineer. I work with Python, SQL, GCP, Vertex AI, BigQuery, Cloud Run, Cloud Functions, Document AI, Gemini, RAG, agents and applied AI systems. I am studying daily for 30–60 minutes to strengthen my theoretical foundations and improve my performance in technical interviews for Senior Data Scientist, AI Engineer, Applied AI Engineer, Forward Deployed Engineer and GenAI Engineer roles.

Goal:
I want to study the topic of the day deeply enough to understand the theory, mathematics/statistics, practical applications, limitations, trade-offs, common interview questions and a short practical implementation that can be versioned in a public GitHub repository.

Topic of the day:
[TEMA_DO_DIA]

Expected level:
Do not explain this as if I were a complete beginner. Explain it to someone who already works in the field but wants to consolidate senior-level understanding and defend the topic in technical interviews. Be didactic, but rigorous.

Follow exactly this structure:

1. Executive overview
Explain what the topic is, why it matters and where it appears in real Data Science or AI Engineering work.

2. Core intuition
Explain the central idea with a simple analogy or interpretation, while keeping technical rigor.

3. Theoretical foundations
Explain the essential concepts, including definitions, assumptions, components, internal mechanics and important variations.

4. Mathematical, statistical or logical foundations
Present relevant formulas, explain each term and show how they connect to the method. When useful, derive or explain where the formula comes from.

5. Practical applicability
Explain which types of problems this topic is useful for, when it makes sense, when it does not make sense and what trade-offs it introduces.

6. Common pitfalls and mistakes
List common mistakes in interviews and real projects, including data leakage, wrong metrics, overfitting, incorrect interpretation, misuse and technical limitations.

7. Important comparisons
Compare this topic with related methods or approaches. Explain differences, advantages, disadvantages and criteria for choosing one over another.

8. Practical Python example
Create a short executable Python example. Prefer common market libraries such as numpy, pandas, scikit-learn, matplotlib, PyTorch, LangChain, LlamaIndex, FastAPI or equivalent tools depending on the topic. Do not create unnecessarily complex code. The focus is to demonstrate the concept.

9. From-scratch implementation when useful
If the topic allows it, include a simplified from-scratch implementation using Python/numpy to show how the internal mechanism works. This should be educational, not production-grade.

10. Suggested experiments
Suggest 3 to 5 small variations in the code so I can test and understand the behavior of the method more deeply.

11. Senior interview questions
Create questions and answers in senior technical interview style. Include conceptual, mathematical, practical and system design questions when relevant.

12. Interview-ready explanation
Create a short, mature and professional answer to the question: “Explain [TEMA_DO_DIA] and when you would use it in a real project.”

13. GitHub file structure
Suggest which files to create inside the topic folder, following this pattern:
- README.md
- notes.md
- notebook.ipynb
- from_scratch.py, if useful
- example.py
- interview_questions.md
- references.md

14. Suggested README.md content
Generate a short, professional README in English, including objective, concepts covered, how to run the example and key takeaways.

15. LinkedIn post idea
Generate a short LinkedIn post suggestion in Portuguese, with a professional, natural and non-exaggerated tone. The post should explain the main insight of the topic in an accessible way, without heavy mathematics. The goal is to build authority and curiosity, pointing out that the deeper technical content is documented in GitHub.

16. 30–60 minute checklist
At the end, include a practical checklist for studying this topic in 30–60 minutes.

Restrictions:
- Do not generate superficial content.
- Do not create an excessively long and unstructured text.
- Do not invent benchmark results.
- Prefer synthetic datasets or simple public datasets.
- Code must be executable with few dependencies.
- Explain every mathematical symbol used.
- When mentioning modern tools, also explain the underlying concept.
- Always connect the topic to interviews and real-world applications.
```

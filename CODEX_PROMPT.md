# Codex Prompt for VSCode

Use this prompt inside Codex in VSCode after studying the daily topic in ChatGPT.

The purpose of Codex is to transform the study output into organized repository files.

---

```text
Read the following files before making any changes:

- STUDY_CONTEXT.md
- ROADMAP.md
- REPO_STRUCTURE.md
- CODEX_PROMPT.md

Topic of the day:
[TEMA_DO_DIA]

Objective:
Create or update the corresponding topic folder in this repository, following the project structure and quality standards.

Use the study content below as the base material:

[PASTE_CHATGPT_STUDY_OUTPUT_HERE]

Create the following files when they make sense for the topic:

- README.md
- notes.md
- example.py
- from_scratch.py, if the topic benefits from a simplified implementation from first principles
- interview_questions.md
- references.md

Requirements:

1. Follow the folder naming convention defined in REPO_STRUCTURE.md.
2. The topic README.md must be written in English.
3. The README.md should include overview, concepts covered, why it matters, files, how to run and key takeaways.
4. The notes.md file should be deeper and cover intuition, theory, formulas, assumptions, trade-offs, applications, limitations and common mistakes.
5. The code should be simple, executable and educational.
6. Prefer common dependencies such as numpy, pandas, scikit-learn and matplotlib when possible.
7. Use PyTorch only when the topic requires neural networks or deep learning.
8. Use LangChain, LangGraph, LlamaIndex, FAISS, pgvector or vector search examples only when they are relevant to the topic.
9. Do not create unnecessarily complex architecture for a simple topic.
10. Do not generate generic filler text.
11. Do not invent benchmark results.
12. If using synthetic data, clearly state that it is synthetic.
13. If using a public dataset, include where it comes from in references.md.
14. If creating from_scratch.py, make it educational rather than production-grade.
15. Add comments explaining important logic, but avoid excessive comments.
16. Keep files clean, readable and professional.
17. Ensure imports are valid.
18. Run or reason through the code to avoid syntax errors.
19. Do not commit automatically unless I explicitly ask you to.
20. At the end, suggest a clear commit message.

Before editing files, briefly show the planned files and folder path.

After editing, summarize:

- files created or changed;
- how to run the example;
- what to review manually;
- suggested commit message.
```

---

## Short Codex Prompt

Use this when the daily study output is already very clear.

```text
Read STUDY_CONTEXT.md, ROADMAP.md, REPO_STRUCTURE.md and CODEX_PROMPT.md.

Topic:
[TEMA_DO_DIA]

Create the topic folder and files following the repository standard.

Use this study output as source material:
[PASTE_CHATGPT_STUDY_OUTPUT_HERE]

Generate README.md, notes.md, example.py, from_scratch.py if useful, interview_questions.md and references.md.

Keep the content technical, professional, executable and aligned with senior Data Scientist / Applied AI Engineer interviews.

Do not create generic filler text. Suggest a commit message at the end.
```

# GitHub Workflow

This document defines the recommended workflow for maintaining the repository professionally.

## Initial Setup

```bash
mkdir applied-ai-engineering-lab
cd applied-ai-engineering-lab
git init
```

Copy the starter Markdown files into the folder.

Then:

```bash
git add .
git commit -m "Initialize applied AI engineering study lab"
```

Create the repository on GitHub and connect it:

```bash
git remote add origin https://github.com/ivanfborges/applied-ai-engineering-lab.git
git branch -M main
git push -u origin main
```

## Daily Branch Strategy

For this repository, a simple `main` branch workflow is enough at first.

Use branches only when creating larger portfolio projects.

Examples:

```bash
git checkout -b project/rag-evaluation-pipeline
```

For daily study notes, direct commits to `main` are acceptable.

## Commit Message Guidelines

Use clear, specific commit messages.

Good examples:

```bash
git commit -m "Add gradient descent from-scratch implementation"
git commit -m "Add logistic regression interview notes"
git commit -m "Add random forest theory and sklearn example"
git commit -m "Add RAG chunking strategy notes"
git commit -m "Add tool calling agent example"
```

Avoid:

```bash
git commit -m "update"
git commit -m "changes"
git commit -m "study"
git commit -m "wip"
```

## Suggested Daily Workflow

1. Study the topic in ChatGPT.
2. Generate files with Codex in VSCode.
3. Review all generated content manually.
4. Run examples locally.
5. Fix issues.
6. Commit and push.

Commands:

```bash
git status
git add .
git commit -m "Add [topic] study notes and examples"
git push
```

## Suggested Weekly Workflow

At the end of each week:

1. Review topic folders created during the week.
2. Improve wording and structure.
3. Update module README if needed.
4. Make sure examples run.
5. Select one or two insights for LinkedIn.
6. Create a weekly recap commit if needed.

Example:

```bash
git commit -m "Refine week 1 foundations documentation"
```

## GitHub Issues as Study Backlog

Create one issue per topic or per weekly block.

Example issue title:

```text
[Study] Random Forest theory, implementation and interview questions
```

Example checklist:

```markdown
- [ ] Study theory
- [ ] Write notes.md
- [ ] Implement practical example
- [ ] Add from_scratch.py if useful
- [ ] Add interview questions
- [ ] Add references
- [ ] Run code
- [ ] Commit and push
```

## Project Board

A simple GitHub Project board can have these columns:

```text
Backlog
Studying
Implementing
Reviewing
Done
```

This is optional, but it gives the repository a more professional project-management structure.

## README Quality Checklist

Every important folder should have a README that answers:

- What is this topic/project?
- Why does it matter?
- What concepts are covered?
- How do I run it?
- What are the key takeaways?
- What files should I read first?

## Code Quality Checklist

Before committing code:

- does it run?
- are imports valid?
- is the example simple enough?
- is the output interpretable?
- are comments useful?
- are there unnecessary dependencies?
- are paths relative?
- is any secret or credential exposed?

## Portfolio Extraction Rule

Daily study remains in this repository.

Extract to a standalone repository only when a project has:

- clear problem statement;
- runnable code;
- good README;
- architecture explanation;
- practical value;
- enough polish to be shown independently.

Potential standalone repositories:

```text
rag-evaluation-pipeline
conversational-agent-with-tools
llmops-evaluation-dashboard
classical-ml-tabular-case
ml-monitoring-drift-lab
```

## Pinned Repositories Strategy

Once the first projects are polished, pin repositories that best represent the desired market positioning:

1. applied-ai-engineering-lab
2. rag-evaluation-pipeline
3. conversational-agent-with-tools
4. llmops-evaluation-dashboard
5. classical-ml-tabular-case
6. album-copa-2026-local, if improved as a GenAI/ML portfolio project

## Public Positioning

The repository should communicate:

```text
I understand AI theory, can implement the core ideas, and can think about real production systems.
```

Avoid making the repository look like:

```text
random notebooks from study sessions
```

The difference is documentation, structure and consistency.

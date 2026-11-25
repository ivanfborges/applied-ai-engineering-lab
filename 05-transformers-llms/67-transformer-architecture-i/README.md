# Day 67 — Transformer Architecture I

How do attention blocks become a conditional sequence model? The original
encoder-decoder Transformer connects source self-attention, causal target
self-attention and cross-attention with position-wise feed-forward networks,
residual connections and layer normalization.

This study builds on [Day 66](../66-attention-mechanism/). It focuses on the
composition and the training/inference contract; multi-head mechanics and
positional encodings receive dedicated treatment in Days 68 and 69.
Understanding that contract helps diagnose target leakage and distinguish
teacher-forced predictions from free-running generation in sequence systems.

## Two executable views

- [example.py](example.py) trains a small PyTorch encoder-decoder on **synthetic
  sequence reversal**. It supplies embeddings, positions and a vocabulary head
  around `nn.Transformer`, and evaluates both teacher forcing and greedy generation.
- [from_scratch.py](from_scratch.py) assembles one encoder and decoder block
  with NumPy: single-head attention, sinusoidal positions, ReLU FFNs and post-norm
  residuals. Parameters persist across calls. It is untrained and has no autograd
  or token prediction head; it is an educational forward pass.
- [notes.md](notes.md) explains the equations, masking, tensor flow, assumptions,
  trade-offs and dated execution record.
- [tests/test_transformer_architecture.py](tests/test_transformer_architecture.py)
  checks shifting, split separation, causal invariance, source sensitivity,
  finite gradients, generation and small-run repeatability.
- [interview_questions.md](interview_questions.md) and [references.md](references.md)
  connect the implementation to architectural reasoning and primary sources.

## Run from the repository root

Use a stable Python environment and the canonical repository extras:

```powershell
python -m pip install -e ".[dev,deep-learning]"
python 05-transformers-llms/67-transformer-architecture-i/from_scratch.py
python 05-transformers-llms/67-transformer-architecture-i/example.py --steps 400 --seed 42
python scripts/validate_repo.py tests --scope 05-transformers-llms/67-transformer-architecture-i
```

No downloads, credentials, GPU or external datasets are needed. The example uses
one PyTorch CPU thread and deterministic algorithms. Each invocation creates a
uniquely named JSON record in the topic's ignored `outputs/` directory; previous
records are preserved. `--output-dir` can select another local destination.
Dependency ranges live in [pyproject.toml](../../pyproject.toml), while the
execution record retains actual versions. Seeds do not promise identical
floating-point results across platforms or library versions.

## What was observed

The [2026-11-25 execution record](notes.md#execution-record--2026-11-25) reports
400 updates on 512 unique training sequences and final evaluation on 128 disjoint
sequences. Teacher-forced token accuracy was **98.31%**; autoregressive token
accuracy was **97.14%** and exact match was **89.84%** (115/128). Metrics include
the EOS slot. These are single-configuration synthetic observations, not a
language benchmark. **Interpretation: author review required.**

## Architectural takeaways and limits

- Attention routes information across positions; FFNs transform features within
  each position. Residuals and normalization are part of the architecture.
- Cross-attention queries come from the decoder; keys and values come from the
  final encoder memory. Source and target lengths can differ.
- Shifted targets and a causal mask jointly prevent future-target access during
  teacher forcing. With `nn.Transformer`, boolean `True` means **blocked**.
- Teacher-forced positions can be computed together; greedy autoregressive
  generation still consumes its own previous predictions sequentially.
- This baseline uses fixed-length inputs, learned positions, shared source/target
  embeddings and a fixed generation horizon including one EOS slot. It omits
  padding, EOS stopping, KV caching and variable-length evaluation.

There is no claim of translation ability, long-context generalization, efficient
serving or production readiness. The analytical run did not include a comparative
training ablation; the separate visual protocol below adds a paired masking
experiment. Other proposed studies remain labeled in the notes. Author interpretation review
is separate from implementation and automated checks.

## Interactive visual laboratory

[visualizations.py](visualizations.py) generates 16 groups of GIFs, explanatory
PNGs and seven self-contained Plotly HTMLs. They connect the complete architecture
to actual per-head attention, encoder activation trajectories, shared PCA,
teacher forcing, autoregressive predictions and a paired masked/unmasked training
experiment. Conceptual diagrams, mathematical calculations, untrained computation,
trained observations and theoretical memory estimates are explicitly labeled.

```powershell
python 05-transformers-llms/67-transformer-architecture-i/visualizations.py --all
python 05-transformers-llms/67-transformer-architecture-i/visualizations.py --quick
python 05-transformers-llms/67-transformer-architecture-i/visualizations.py --interactive --skip-training
```

The [visual guide](VISUAL_LAB.md) maps every artifact, control, CLI option,
execution protocol and limitation. It records the new seed-67 experiment
separately from the analytical seed-42 result. Generated files and checkpoints
remain ignored pending author review; run the command and open the printed
local gallery. No preview approval or root-documentation update is implied.

The numerical instrumentation in [transformer_lab.py](transformer_lab.py) reuses
the study model and checks captured operations against its original forward.
[transformer_figures.py](transformer_figures.py) handles Matplotlib/Pillow exports;
[transformer_html.py](transformer_html.py) bundles offline interactive views.
The [numerical visual tests](tests/test_visual_core.py) and
[real export tests](tests/test_visual_exports.py) complement the analytical suite.

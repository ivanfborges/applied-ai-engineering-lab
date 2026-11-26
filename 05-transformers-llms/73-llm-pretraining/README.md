# Day 73 — LLM Pretraining

[Module index](../README.md) · [Study roadmap](../../ROADMAP.md)

How does text become its own supervision, and what does a lower prediction loss
actually establish? This study trains the same miniature Transformer with two
objectives: next-token prediction (CLM) and masked-token reconstruction (MLM).
It connects target construction and attention visibility to data quality,
parameter count, token exposure and approximate training compute.

The examples use **synthetic data**, random initial weights and one CPU thread.
They require no external dataset, model download, account or GPU. This is an
educational implementation, not foundation-model training or a GPT/BERT replica.

## Two objectives, two evaluation questions

| Component | Causal language modeling | Masked language modeling |
|---|---|---|
| Inputs | All tokens except the final token | Full sentence with selected interior tokens replaced by `<mask>` |
| Targets | Same sequence shifted one position forward, including EOS | Original IDs at selected positions; `-100` elsewhere |
| Context | Current and earlier input positions | Both directions in the corrupted sentence |
| Evaluation | Mean token NLL, accuracy and autoregressive perplexity | Mean selected-token NLL and accuracy over all fixed mask combinations |

Training predictions at different CLM positions can run in parallel because
ground-truth prefixes are available. Autoregressive generation has a different,
sequential dependency. MLM reconstruction loss is not autoregressive perplexity,
and the two objectives' loss values do not rank their downstream capabilities.

## Inspect and run

Run from the repository root with stable Python 3.11+. Dependencies remain in
[pyproject.toml](../../pyproject.toml); no topic-specific requirements file is needed.

```powershell
python -m pip install -e ".[deep-learning,dev]"
python 05-transformers-llms/73-llm-pretraining/example.py --objective clm
python 05-transformers-llms/73-llm-pretraining/example.py --objective mlm
python 05-transformers-llms/73-llm-pretraining/from_scratch.py
python scripts/validate_repo.py tests --scope 05-transformers-llms/73-llm-pretraining
```

Both Transformer modes default to seed 42, 200 full-batch AdamW updates,
width 32, two heads, one block and FFN width 64. `--steps`, `--seed`, `--width`,
`--heads`, `--learning-rate` and `--masked-tokens` expose bounded study choices.
MLM defaults to one of four interior tokens masked per sentence, sampled afresh
at each update. This deliberately differs from BERT's original corruption policy.
All sentences have length six, so there is no padding or packed-document support.

Scripts print JSON. `example.py --output <new-path>` also writes a record and
refuses to overwrite an existing file. Choose a new dated path inside the
topic's ignored `outputs/` directory. Runtime, timestamp and versions are recorded;
numerical reproducibility is checked in the installed environment, not promised
across library versions or hardware.

| File | Purpose |
|---|---|
| [example.py](example.py) | Synthetic corpus, held-out combination split, causal/masked batches, one-block training and evaluation |
| [from_scratch.py](from_scratch.py) | Stable NumPy bigram NLL and explicit repeated-row gradient accumulation |
| [tests/test_pretraining.py](tests/test_pretraining.py) | Target shifts, causal invariance, bidirectional positive control, MLM loss selection, gradient references and reproducible CPU training |
| [notes.md](notes.md) | Likelihood, entropy limits, data/compute trade-offs and dated execution evidence |
| [interview_questions.md](interview_questions.md) | Objective, contamination, scaling and adaptation decisions |
| [references.md](references.md) | Original papers and versioned PyTorch documentation |

## Interactive visual laboratory

The separate [visual guide](VISUAL_LAB.md) follows the learning loop from
target construction to measured optimization, probability evolution, embeddings,
actual per-head attention, generalization and leakage. Analytical scaling
surfaces and a local compute simulator are explicitly separated from training.

```powershell
python 05-transformers-llms/73-llm-pretraining/visualize.py --all --quick
python 05-transformers-llms/73-llm-pretraining/visualize.py --all
python 05-transformers-llms/73-llm-pretraining/visualize.py --section scaling --section compute
```

Open the printed run directory's `index.html`. The lab generates nine GIFs,
ten PNGs and twelve local interactive HTMLs, including meaningful 3D loss,
embedding and scaling views. Its controls change recorded checkpoints or
recalculate clearly labeled analytical assumptions. HTMLs load an adjacent
local Plotly script; GIFs need no FFmpeg. Exports remain ignored pending author
curation. A local gallery is not a versioned GitHub preview.

| Visual source | Purpose |
|---|---|
| [visualize.py](visualize.py) | Main entry point, section selection, quick/full modes, saved-run reuse and provenance |
| [visual_core.py](visual_core.py) | Expanded synthetic grammar, configurable exposed-attention Transformer, measured scenarios and tested analytical budgets |
| [visual_render.py](visual_render.py), [visual_html.py](visual_html.py) | Static/animated exports and offline Plotly controls driven by the records |
| [tests/test_visual_core.py](tests/test_visual_core.py), [tests/test_visual_rendering.py](tests/test_visual_rendering.py) | Numerical/training contracts and actual temporary exports |

Visual training reserves complete subject/verb combinations before batching.
Its corpus, architecture and budget differ from the original analytical example;
the records below remain historical evidence for that earlier configuration.
See the guide for exact commands, the visual artifact map, new measured results,
the masking policy and the limits of the synthetic comparisons.

## Evidence and limits

The [2026-11-25 execution record](notes.md#execution-record--2026-11-25) reports
one run per Transformer objective and one NumPy run. CLM train/validation NLL
was **0.498278 / 1.573562**; MLM selected-token NLL was **0.622623 / 1.371625**.
These are different prediction problems and supervision budgets.

Twelve training sentences cover the off-diagonal combinations of four animals
and four foods; four validation sentences reserve diagonal pairs. Tokens are
shared while complete sentences are disjoint. Held-out pairs are intentionally
absent from the training conditional support, so this split probes a specific
distribution shift. It is not a broad test of compositional generalization.
There is no independent test set or task-quality benchmark.

Key takeaways: self-supervision still needs correct labels and masks; ambiguity
can impose nonzero loss even on training data; repeating twelve sentences does
not create new information; and a `6ND` compute proxy does not measure hardware
performance. Lower token loss alone establishes neither truthfulness nor useful
instruction-following behavior. These distinctions help diagnose whether an
application needs domain adaptation, behavioral training or external evidence.

**Implementation and local automated checks are separate from interpretation:
author review required.** The visual phase adds measured records without
replacing the analytical record. Generated JSON, weights and visual exports
remain ignored; public preview selection and interpretation require author review.

# Day 74 — Fine-Tuning, Instruction Tuning and Alignment

[Module index](../README.md) · [Roadmap](../../ROADMAP.md)

When does a persistent behavior problem justify changing model parameters?
This study separates **training objectives** (SFT and preference optimization)
from **adaptation mechanisms** (adapters and LoRA). Instruction tuning uses
supervised demonstrations organized as instructions; alignment is a broader
evaluation goal that no single loss guarantees.

The executable example trains a small response selector, then applies SFT and
DPO through the same LoRA parameters. All data are **synthetic**. No LLM,
tokenizer, public dataset, human annotation, model download, API or GPU is used.
The analytical example does not implement PPO, natural-language instruction
generalization or actual QLoRA. The separate visual laboratory below adds
bounded synthetic experiments and simulations with their own evidence limits.

## What changes at each stage?

| Stage | Signal | Updated parameters |
|---|---|---|
| Toy base training | Task labels with randomly mixed concise/verbose styles | Linear weight and bias |
| SFT + LoRA | Demonstrations sampled with 80% concise style probability | Low-rank matrices A and B |
| DPO + LoRA | Concise preferred over verbose for the same correct task | The existing A and B; base and SFT reference frozen |

The four categories are extraction/concise, extraction/verbose,
summarization/concise and summarization/verbose. Inputs contain a one-hot task
ID plus six nuisance features. These names label categories; the model neither
extracts fields nor summarizes text. Concision is an arbitrary synthetic
preference, not a universal quality criterion.

## Inspect and execute

From the repository root, using stable Python 3.11+ and the canonical
[dependency extras](../../pyproject.toml):

```powershell
python -m pip install -e ".[deep-learning,dev]"
python 05-transformers-llms/74-fine-tuning-alignment/example.py
python 05-transformers-llms/74-fine-tuning-alignment/from_scratch.py
python scripts/validate_repo.py tests --scope 05-transformers-llms/74-fine-tuning-alignment
```

`example.py` prints JSON. `--seed`, `--rank` and `--beta` change study choices;
`--output <new-path>` saves a record, refusing to overwrite existing files.
Use a new dated filename inside the topic's ignored `outputs/` directory.
Defaults use seed 42, one CPU thread, rank 1, alpha 2 and beta 0.5, with
200/80/100 full-batch Adam updates for base/SFT/DPO. No checkpoint or
hyperparameter is selected using evaluation results.

| File | Technical question |
|---|---|
| [example.py](example.py) | How do frozen weights, a copied reference and low-rank updates compose? Includes a separately tested response-token scoring helper. |
| [from_scratch.py](from_scratch.py) | What is the DPO derivative with respect to chosen/rejected log-probabilities? |
| [tests/test_alignment.py](tests/test_alignment.py) | Do finite differences, autograd, masks, freezing, merge and repeatability agree? |
| [notes.md](notes.md) | How do likelihood, preference ratios, RLHF and adaptation constraints differ? Includes dated execution evidence. |
| [interview_questions.md](interview_questions.md) | How would these distinctions guide an adaptation decision? |
| [references.md](references.md) | Original research and PyTorch API sources |

## Interactive visual laboratory

The separate [visual guide](VISUAL_LAB.md) maps fourteen demonstrations to
actual CPU training, analytical surfaces and explicit storage assumptions.
It covers SFT/token masks, symbolic multitask adaptation, reward-model fitting,
exact finite-policy reward/KL optimization, DPO, low-rank reconstruction,
adapter counts, uniform quantization, forgetting, reward hacking and an
integrated base/SFT/DPO pipeline.

```powershell
python 05-transformers-llms/74-fine-tuning-alignment/visualize_fine_tuning.py --all --quick
python 05-transformers-llms/74-fine-tuning-alignment/visualize_fine_tuning.py --demo dpo --quick
python 05-transformers-llms/74-fine-tuning-alignment/visualize_fine_tuning.py --demo lora --quick
python 05-transformers-llms/74-fine-tuning-alignment/visualize_fine_tuning.py --list
```

Open the printed output directory's `index.html`. A complete run exports 13
GIFs, 21 PNGs and 14 interactive HTMLs. HTMLs use one adjacent local Plotly
runtime and work offline as a directory bundle. Checkpoint playback, beta/rank
configuration selection and 3D camera controls explore calculated surfaces and
recorded measurements. The [dated visual record](VISUAL_LAB.md#executed-visual-record--2026-11-25-americasao_paulo)
is separate from the analytical record below.

| Visual source | Purpose |
|---|---|
| [visualize_fine_tuning.py](visualize_fine_tuning.py) | Demo selection, quick/full settings, saved-run reuse and hashed provenance |
| [visual_core.py](visual_core.py) | Fourteen reproducible numerical experiments and simulations |
| [visual_render.py](visual_render.py), [visual_html.py](visual_html.py) | Question-specific Matplotlib/Pillow exports and offline interactive pages |
| [tests/test_visual_core.py](tests/test_visual_core.py), [tests/test_visual_rendering.py](tests/test_visual_rendering.py) | Numerical identities, held-out isolation, real temporary exports and CLI contracts |

Synthetic rule labels do not establish language instruction following; exact
categorical optimization is not PPO; reconstructed uniform low-bit weights are
not actual QLoRA. In the quick forgetting experiment LoRA did not preserve the
earlier conflicting task. All exports remain ignored pending author curation,
and interpretations remain **author review required**.

## Observed scope and takeaways

The [2026-11-25 execution record](notes.md#execution-record--2026-11-25)
reports mean concise-response probability of 0.4942 / 0.7761 / 0.9752 for
base / SFT / DPO, with task accuracy 1.0 at each stage. The 12 trainable LoRA
parameters adapted a frozen 36-parameter base; this count is not a measured
memory or speed improvement. Interpretations: **author review required**.

- SFT imitates demonstrations; DPO fits reference-adjusted relative preference.
- LoRA can serve either objective. It constrains the weight update, not the
  preference definition or evaluation rubric.
- A lower pair loss does not establish absolute chosen-response improvement,
  retained capabilities, factuality or safety. Evaluate those separately.
- Prompting, retrieval, deterministic validation and fine-tuning can be
  complementary. Diagnose the failing system component before choosing training.

The selector is educational numerical machinery, not a production PEFT library
or evidence of natural-language instruction following. General capability
retention, real human preferences, PPO, actual QLoRA and language-quality
comparisons remain untested. The separate visual lab adds synthetic retention
and full/LoRA/replay comparisons plus uniform quantization simulation; these do
not change the analytical run's evidence. Generated outputs stay ignored and
no public preview assets have been selected.

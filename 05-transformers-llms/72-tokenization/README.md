# Day 72 — Tokenization for LLMs

[Module index](../README.md) · [Roadmap](../../ROADMAP.md)

How does a tokenizer change what fits in a request, without changing the
meaning of the original text? This study connects learned subword units to
vocabulary parameters, Unicode boundaries, context limits and hypothetical
cost. Token IDs index learned embeddings; their numerical distance has no
semantic meaning.

Two complementary examples distinguish learning merge rules from using a
pretrained tokenizer. All text probes and weighted training words are synthetic,
defined in code. No external text dataset, model weights or LLM API is used.

| File | Technical purpose |
|---|---|
| [from_scratch.py](from_scratch.py) | Character BPE: weighted pair counts, lexical ties, non-overlapping merges and rank-based inference on unseen words |
| [example.py](example.py) | Real `tiktoken` encoding IDs and byte pieces; tested context, cost and vocabulary calculations |
| [notes.md](notes.md) | BPE vs WordPiece, SentencePiece/Unigram, formulas, operational limits and dated execution evidence |
| [tests/test_tokenization.py](tests/test_tokenization.py) | Merge priorities, refitting, Unicode reconstruction, special-token policy and budget boundaries |
| [interview_questions.md](interview_questions.md) | Model compatibility, multilingual comparisons and RAG decisions |
| [references.md](references.md) | Primary papers, supplier documentation and tokenizer resource provenance |
| [VISUAL_LAB.md](VISUAL_LAB.md) | Separate sixteen-group visual phase, dashboard, experiment configurations and evidence boundaries |
| [visualize.py](visualize.py), [app.py](app.py) | PNG/GIF/offline HTML generator and live Streamlit tokenizer laboratory |
| [visual_core.py](visual_core.py), [visual_figures.py](visual_figures.py), [visual_html.py](visual_html.py) | Shared computed states, raster rendering and offline interactive pages |
| [visual tests](tests/test_visual_core.py), [app tests](tests/test_visual_app.py), [export tests](tests/test_visual_rendering.py) | Algorithm/library contracts, live widgets and temporary real exports |

## Run from the repository root

Python 3.11+. The educational BPE uses only the standard library. The real
encoding comparison needs the canonical `nlp` extra; tests also need `dev`:

```powershell
python -m pip install -e ".[dev,nlp]"
python 05-transformers-llms/72-tokenization/from_scratch.py
$env:TIKTOKEN_CACHE_DIR = Join-Path (Get-Location) ".local/tiktoken"
python 05-transformers-llms/72-tokenization/example.py
python scripts/validate_repo.py tests --scope 05-transformers-llms/72-tokenization
```

The first comparison needs network access to download two public encoding
tables. Later runs reuse their hash-verified cache. There are no credentials or
paid requests. On other shells, set `TIKTOKEN_CACHE_DIR` using that shell's syntax,
or allow the library's default temporary cache. See the
[validation contract](../../docs/validation.md).

`example.py` prints JSON with input strings, IDs, token bytes in hexadecimal,
counts, versions and the run timestamp. For a retained local record, add a
unique output path; the script refuses to overwrite an existing run:

```powershell
python 05-transformers-llms/72-tokenization/example.py --output 05-transformers-llms/72-tokenization/outputs/run-20261125-analytical.json
```

Outputs and caches stay ignored. The public evidence is the small dated table
in [notes.md](notes.md#execution-record--2026-11-25); no preview assets are linked.

## What the implementations establish

BPE training chooses frequent adjacent pairs; encoding uses the learned rank
order. The educational tokenizer splits on whitespace, operates on code points,
rejects unseen characters and returns pieces rather than IDs. Joining pieces
reconstructs one word, not the original multiword document. It is not a byte-BPE
or production tokenizer replacement.

The real example explicitly compares `cl100k_base` with `o200k_base`, without
mapping them to a current hosted model. It treats special-token-looking strings
as ordinary text via `encode_ordinary`. A chat template can inject actual special
tokens and other overhead. These counts therefore do not certify chat request
length or provider billing. One token's bytes can split a UTF-8 character; decode
the full sequence, rather than decoding each piece independently.

The analytical scripts cover WordPiece and SentencePiece conceptually. The
separate visual phase trains actual small HF WordPiece and SentencePiece
BPE/Unigram models on a synthetic corpus; those counts are not a controlled
algorithm benchmark. SentencePiece supports multiple algorithms.

## Separate visual learning lab

```powershell
python 05-transformers-llms/72-tokenization/visualize.py --setup
python 05-transformers-llms/72-tokenization/visualize.py --quick
python 05-transformers-llms/72-tokenization/visualize.py
python -m streamlit run 05-transformers-llms/72-tokenization/app.py --server.address=127.0.0.1
```

The [visual guide](VISUAL_LAB.md) maps all sixteen groups, individual-section
commands, the ten-section dashboard and focused tests. Inspect actual BPE
states, greedy WordPiece, Unigram paths, normalization, Unicode bytes, vocabulary
and OCR experiments, then explore chunking and theoretical budget/cost controls.
Every export run includes a local `index.html` gallery. Generated assets and local
models remain ignored; preview selection and root README highlights await human
review. The original analytical execution record is preserved separately.

## Takeaways and evidence boundary

- Vocabulary size and sequence length trade off; fewer tokens alone do not
  establish better representations, answers or latency.
- Count embedding inputs and assembled generation prompts with their respective
  model tokenizers. Reserve output capacity and enforce separate input/output
  caps where applicable.
- The cost example uses invented prices, and the weight estimate excludes
  gradients, optimizer states, bias and the rest of the network.
- Results describe these synthetic strings. Language fairness, OCR robustness,
  retrieval quality and production cost remain unmeasured.

Implementation and automated validation are separate from interpretation:
**author review required** for the recorded interpretation candidates.

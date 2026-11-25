# References and resource provenance

Primary sources consulted for this study on 2026-11-25:

- Sennrich, Haddow and Birch (2016),
  [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/).
  BPE applied to subword translation; the small implementation here deliberately
  omits the paper's end-of-word marker and production-scale implementation.
- Devlin et al. (2019), [BERT](https://aclanthology.org/N19-1423/), and the
  [original BERT tokenizer](https://github.com/google-research/bert/blob/master/tokenization.py).
  `WordpieceTokenizer` specifies longest-match-first inference, `##` continuation
  and whole-word unknown handling. No BERT model is downloaded or evaluated here.
- Hugging Face,
  [WordPiece course](https://huggingface.co/learn/llm-course/en/chapter6/6).
  Explicitly labels the reconstructed training procedure and association score
  as a best guess; do not attribute that exact trainer to the original BERT team.
- Kudo (2018), [Subword Regularization](https://aclanthology.org/P18-1007/).
  Unigram segmentation and sampling for regularization.
- Kudo and Richardson (2018),
  [SentencePiece](https://aclanthology.org/D18-2012/), plus the
  [official implementation](https://github.com/google/sentencepiece).
  Framework/algorithm distinction, raw-sentence training and whitespace handling.
  Apache-2.0 software; referenced conceptually, not installed for this study.
- [tiktoken](https://github.com/openai/tiktoken),
  [encoding API](https://github.com/openai/tiktoken/blob/main/tiktoken/core.py),
  [encoding definitions](https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py)
  and [cache loader](https://github.com/openai/tiktoken/blob/main/tiktoken/load.py).
  MIT software. The example uses named encodings, ordinary-text encoding,
  byte-level inspection and full-sequence reconstruction, without hosted requests.

## Text and downloaded data

All text in `example.py` and weighted words in `from_scratch.py` are handwritten
synthetic data. There is no external evaluation corpus. Pretrained tokenizer
resources are distinct from the probes: their underlying training distribution
is not controlled by this study. Encoding comparison is not a controlled
vocabulary-size intervention.

`tiktoken` downloads public tables from these supplier endpoints and checks the
expected SHA-256 defined by its installed encoding definitions:

| Encoding | Resource | SHA-256 |
|---|---|---|
| `cl100k_base` | [encoding table](https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken) | `223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7` |
| `o200k_base` | [encoding table](https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken) | `446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d` |

Cache files are local and ignored. No tokenizer files or pretrained weights are
copied into the public topic. The installed library version and probe hash are
recorded with each run. A future rerun should retain its own version and date.
No provider pricing page is needed for the explicitly invented cost scenario.

## Visual-phase implementation sources

- Hugging Face [Tokenizers v0.22.2 WordPiece trainer](https://github.com/huggingface/tokenizers/blob/v0.22.2/tokenizers/src/models/wordpiece/trainer.rs)
  and [underlying BPE trainer](https://github.com/huggingface/tokenizers/blob/v0.22.2/tokenizers/src/models/bpe/trainer.rs).
  The actual library trainer delegates vocabulary construction through its BPE
  trainer. It should not be equated to the manual association-score simulator or
  claimed to reproduce the original unpublished WordPiece training recipe.
  The lab initializes continuation symbols in lexical order to make numeric
  pair tie breaking reproducible, then removes their temporary special status.
  Apache-2.0 software; only a small synthetic corpus is trained locally.
- SentencePiece [Python usage and in-memory model writing](https://github.com/google/sentencepiece/blob/master/python/README.md).
  Local BPE and Unigram models use the same code-defined corpus and explicit
  normalization/threading settings; no pretrained SentencePiece model is downloaded.
- [Streamlit documentation](https://docs.streamlit.io/) and the installed 1.67.0
  package's bundled API/skill references for caching, charts, media and AppTest.
  These guide UI behavior, not scientific approval of the visual interpretations.

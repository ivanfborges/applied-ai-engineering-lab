# References and evidence scope

Consulted on 2026-11-26. All invoice documents, demonstrations and response
fixtures in this topic are authored synthetic examples; no external dataset or
real customer document is used.

- [Google — Prompt design strategies](https://aistudio.google.com/docs/prompting-strategies).
  Primary guidance on instructions, context and demonstrations. It informs prompt
  construction; it is not evidence that our V2/V3 outperform V1. The analytical
  example and default visual simulation make no provider calls. The separate
  [visual guide](VISUAL_LAB.md) documents an optional real-provider adapter;
  its existence does not establish an executed provider comparison.
- [Sclar, Choi, Tsvetkov and Suhr — Quantifying Language Models' Sensitivity to
  Spurious Features in Prompt Design, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6c0e99d736da621403018ca7b32b1a4d-Abstract-Conference.html).
  Original research on formatting sensitivity in the evaluated few-shot settings.
  Motivates perturbation checks; its results are not measurements of our prompts.
- [OWASP — LLM Prompt Injection Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html).
  Guidance for instruction/data separation and controls outside model generation.
  The local fixture does not validate any model's injection resistance.
- [Python — JSON encoder and decoder](https://docs.python.org/3/library/json.html).
  Reference for `object_pairs_hook` and `parse_constant`. This validator deliberately
  rejects duplicate keys and nonstandard numeric constants rather than accepting
  the decoder's permissive defaults.

Connections within the curriculum: [Day 74 — Fine-Tuning and Alignment](../74-fine-tuning-alignment/)
and [Day 75 — LLM Inference and Decoding](../75-llm-inference-decoding/).

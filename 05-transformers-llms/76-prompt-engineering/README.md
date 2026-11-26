# Day 76 — Prompt Engineering with Rigor

How do instructions, context, examples, constraints, output format and failure
handling become a testable interface to a language model?

This study builds three inspectable invoice-extraction prompts and an executable
application-side evaluator. It separates parseability, a typed output contract,
reference accuracy and complete-output coverage. This distinction matters when
document extraction or RAG results feed downstream software: a valid JSON object
can still contain an invented value.

**Implemented:** offline synthetic fixtures, strict validation, reference scoring
and a bounded retry demonstration. **No LLM was evaluated.** The bundled responses
were deliberately written to exercise failures; they are not outputs of any prompt
version. Interpretation candidates remain **author review required**.

## Inspect and execute

Python 3.11+; the example uses only the standard library. From the repository root:

```powershell
python 05-transformers-llms/76-prompt-engineering/example.py --show-prompt v1
python 05-transformers-llms/76-prompt-engineering/example.py --show-prompt v2
python 05-transformers-llms/76-prompt-engineering/example.py --show-prompt v3 --case conflict
python 05-transformers-llms/76-prompt-engineering/example.py
```

V1 specifies the task and JSON format. V2 adds evidence, final-total selection,
types, missing/conflicting/unsupported-field policies and a document-data boundary.
V3 adds three separate synthetic demonstrations. Demonstrations never contain the
evaluation documents. The request's logical `instructions` and `input` fields are
provider-neutral; they must be mapped to the chosen API's supported roles before
any future generation experiment. Serialization is not an injection defense.

The default command evaluates nine hand-authored response fixtures and shows a
retry that stops at a structurally valid, factually wrong response. It writes a
new UTC-dated JSON record under ignored `outputs/`, including Python version,
input hashes, per-case scores and the explicit evidence type. It makes zero API
requests, needs no credentials and measures no provider latency or cost.

For tests, use the repository's canonical `dev` extra:

```powershell
python -m pip install -e ".[dev]"
python scripts/validate_repo.py tests --scope 05-transformers-llms/76-prompt-engineering
python scripts/validate_repo.py syntax
python scripts/validate_repo.py links
```

## Contract and boundaries

Every valid output has exactly `invoice_id` (nonempty string or null) and
`total_usd` (nonnegative integer or null). Booleans, numeric strings, fractional
values and extra keys fail validation. Whole USD amounts are an explicit
educational simplification. Unsupported currencies and fractional-dollar amounts
map to null rather than implicit conversion. Missing evidence or unresolved
conflicts also map to null for the affected field.

JSON syntax, schema validity and exact match are reported separately. Semantic
scoring uses authored reference labels, which are available during evaluation;
the online retry controller has no access to those labels. Partial extraction
can be correct while still requiring review. Complete output means both fields
are non-null; it is a coverage diagnostic, not permission to accept the result.

## Files and learning path

| File | Purpose |
|---|---|
| [prompts.py](prompts.py) | Three prompt versions and serialized document input |
| [evaluation_dataset.json](evaluation_dataset.json) | Nine synthetic development documents and explicit reference labels |
| [response_fixtures.json](response_fixtures.json) | Authored successes, content/format failures and a simulated timeout |
| [evaluation.py](evaluation.py) | Strict validator, fixed-denominator metrics and bounded retry controller |
| [example.py](example.py) | Prompt inspection and dated offline evaluator demonstration |
| [tests/test_prompt_evaluation.py](tests/test_prompt_evaluation.py) | Types, duplicate keys, metrics, missing records, retry bounds and CLI paths |
| [notes.md](notes.md) | Conditional generation, evaluation design, execution record and limitations |
| [interview_questions.md](interview_questions.md) | Questions about generalization, contracts and failure policy |
| [references.md](references.md) | Primary documentation and prompt-sensitivity research |

Prompt components specify intended behavior; evaluation must establish observed
behavior. This example verifies the evaluator rather than ranking V1/V2/V3.
Delimiters cannot enforce authorization, and retries cannot establish truth.
Real-model comparisons, holdout evaluation, prompt ablations and provider-level
structured generation remain proposed work. Day 77 covers structured outputs in
greater depth. No production extraction, OCR, adversarial robustness or business
impact is established by these fixtures.

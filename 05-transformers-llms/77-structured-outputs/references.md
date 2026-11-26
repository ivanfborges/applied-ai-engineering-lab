# References

Primary documentation consulted on 2026-11-26. Local examples were executed with
Pydantic 2.14.4; live documentation and future releases may differ.

- [JSON Schema: objects](https://json-schema.org/understanding-json-schema/reference/object)
  — properties, required keys and additional properties.
- [JSON Schema: numeric types](https://json-schema.org/understanding-json-schema/reference/numeric)
  — integer semantics and numeric bounds; JSON Schema's integer category is not
  identical to Python strict-int validation of parsed JSON floats.
- [Pydantic: strict mode](https://docs.pydantic.dev/latest/concepts/strict_mode/)
  — strict Python-object validation and JSON-input caveats.
- [Pydantic: JSON Schema](https://docs.pydantic.dev/latest/concepts/json_schema/)
  — exporting the structural contract through `model_json_schema()`.
- [Pydantic: validators](https://docs.pydantic.dev/latest/concepts/validators/)
  — Python application validators; the example keeps arithmetic as a separate
  gate so its failure stage is visible.
- [Python 3.11: JSON interoperability](https://docs.python.org/3.11/library/json.html#standard-compliance-and-interoperability)
  — non-finite numbers, duplicate names and parser hooks.
- [Plotly: interactive HTML export](https://plotly.com/python/interactive-html-export/)
  — standalone local HTML with an embedded runtime.
- [Plotly.js: function reference](https://plotly.com/javascript/plotlyjs-function-reference/)
  — updating traces when the offline controls change.
- [Pydantic: dynamic models](https://docs.pydantic.dev/latest/concepts/models/#dynamic-model-creation)
  — constructing comparison models while varying typing and extra-field policy.

The constrained softmax is an educational derivation implemented locally. No
provider SDK, hosted model, OCR system or production decoder was evaluated.

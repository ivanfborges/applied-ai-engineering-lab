# Prompt Engineering — Separate Visual Lab

[Analytical study](README.md) · [Module index](../README.md)

The analytical evaluator uses nine authored invoice-response fixtures. This
separate lab explores explicit synthetic mechanisms and includes replay plus an
optional real-provider adapter. These are different evidence sources, not three
ways to report the same experiment. Interpretations require author review.

## Choose an evidence mode

| Mode | What it does | What it does not establish |
|---|---|---|
| Simulation — default | Computes consequences of declared synthetic probabilities, response rules, confidence, selection and retry policies | Real prompt quality, provider latency/pricing, security or production reliability |
| Replay — `--results-file` | Scores and presents an existing recorded evaluation without new provider requests | New model executions; replaying test fixtures does not turn them into real-provider evidence |
| Real — explicit `--mode real` | Uses the optional adapter to request responses on a small synthetic development set, recording attempts and available usage | Holdout performance, population-level improvement, production readiness, output probabilities or a dollar-cost estimate |

No provider result is claimed here. The adapter is implemented separately from
the published analytical evidence; its API integration has automated tests using
fakes, not live-provider verification. A runnable adapter is not an executed
benchmark.

## Run offline from the repository root

Use Python 3.11+ and the repository's canonical dependencies:

```powershell
python -m pip install -e ".[dev]"
python 05-transformers-llms/76-prompt-engineering/visualize.py --help
python 05-transformers-llms/76-prompt-engineering/visualize.py --experiment reliability --quick --seed 42
```

`--experiment` selects one demonstration; the default is `all`. Use `--quick`
for a smaller run rather than exporting every scenario during routine edits.
The demonstrations cover conditioning, prompt anatomy, in-context examples,
ablation, selection/optimization, generalization, validation, reliability,
cost–quality assumptions, failure handling and instruction/data boundaries.
Their controls and conclusions are conditional on the synthetic mechanisms;
none certifies a model's resistance to malicious inputs.

`--output-dir` selects the parent of a fresh UTC-dated run directory. Simulation
writes a gallery, figures, a local review document and
`results/simulation-manifest.json`, including configuration, environment,
source/artifact hashes, evidence type and interpretation status. The default
parent is this topic's ignored `outputs/`; previous runs are not overwritten.
Keep exports ignored when choosing a custom destination too. No public preview
has been selected for this lab.

Rendering options include `--width`, `--height`, `--dpi`, `--duration` and `--fps`.
`--input-price-per-million` and `--output-price-per-million` are hypothetical
simulation assumptions, not verified provider prices.

### Replay an existing record

Replace the path below with a record you already have:

```powershell
python 05-transformers-llms/76-prompt-engineering/visualize.py --results-file path/to/real-evaluation.json
```

Replay requires no provider SDK or credentials and creates a separate dated
output directory. Retain the record's evidence type and provenance. Supplying
`--results-file` together with `--mode real` is rejected; the CLI does not
silently replace a failed real evaluation with simulation.

## Optional real-provider adapter

This is not part of the offline quick start. It requires a separately installed
`google-genai` SDK and `GEMINI_API_KEY`/`GEMINI_MODEL` environment variables.
There is currently no canonical Gemini extra. Never put credentials in source,
commands saved to the repository, screenshots or committed records. Selecting
`--mode real` sends synthetic documents and prompt instructions to the configured
provider and can incur charges; review the implementation and provider terms
before choosing it. No installation or live call is required for this study.

The adapter validates a worst-case request budget before collection:
cases × prompt variants × repetitions × maximum attempts must fit
`--max-requests`. Relevant controls are `--case-limit`, `--repetitions`,
`--max-attempts`, `--max-requests`, `--max-output-tokens`, `--temperature` and
`--timeout`. SDK internal retries are disabled in the implementation; the
explicit controller owns attempts. A request count is not a spending cap.

Progress is checkpointed to `results/real-evaluation.json`. Records distinguish
operational errors, raw responses, reference scores, attempts, latency and
available token usage. Costs and output probabilities are not inferred.
Failed or partial runs remain partial; compare the observed case sets before
comparing prompt variants. The cases are development fixtures, not a held-out
sample. Live-provider compatibility has not been verified by this documentation
update.

## Files and focused validation

| File | Role |
|---|---|
| [visualize.py](visualize.py) | Mode selection, render limits and dated output directories |
| [visual_core.py](visual_core.py) | Synthetic calculations and evaluation scenarios |
| [visual_render.py](visual_render.py) | Figures, animations and gallery orchestration |
| [visual_html.py](visual_html.py) | Local HTML presentation |
| [visual_real.py](visual_real.py) | Optional provider collection and offline replay |
| [tests/test_visual_core.py](tests/test_visual_core.py) | Numerical, scoring, budget and fake-adapter contracts |
| [tests/test_visual_rendering.py](tests/test_visual_rendering.py) | Temporary real exports and presentation checks |

For a core change, test the core file first. Include export tests when changing
rendering or its interface; the full-topic command includes both and the
analytical tests:

```powershell
python scripts/validate_repo.py tests --scope 05-transformers-llms/76-prompt-engineering/tests/test_visual_core.py
python scripts/validate_repo.py tests --scope 05-transformers-llms/76-prompt-engineering
```

These commands do not select real-provider mode. Passing them validates the
covered contracts, not model behavior or the truth of an extracted value.

# Day 70 — Feed-Forward Networks, Residual Connections and Layer Normalization

[Module index](../README.md) · [Roadmap](../../ROADMAP.md)

**Question:** how do position-wise feature transformations, additive paths and
normalization placement change information and gradients inside a Transformer?

Attention mixes token information. A dense FFN transforms each contextualized
token independently, with shared parameters across positions. Residual additions
provide an identity contribution to the Jacobian; LayerNorm standardizes hidden
features within each token. Together these mechanisms affect optimization and
the parameter budget. They matter when debugging training and adapting open
models, even when an application uses a managed model whose blocks cannot be
changed.

This study contains educational NumPy operations and real PyTorch attention
blocks. All inputs and targets are **synthetic random tensors**. The diagnostics
use untrained weights, one backward pass per condition and **zero optimizer
steps**. Implementation and local checks are complete; interpretation candidates
remain **author review required**.

## Mechanisms to inspect

For one residual sublayer:

$$
\operatorname{PreLN}(x)=x+F(\operatorname{LN}(x)),\qquad
\operatorname{PostLN}(x)=\operatorname{LN}(x+F(x)).
$$

Each implemented block applies this arrangement twice: attention, then FFN.
LayerNorm uses the last feature axis and population variance. GELU uses the
tanh approximation in both implementations. The NumPy backward operation
isolates the effect of LayerNorm on an incoming gradient.

| File | Purpose |
|---|---|
| [from_scratch.py](from_scratch.py) | NumPy FFN, LayerNorm forward/VJP and exact-shape residual composition |
| [example.py](example.py) | Matched Pre/Post-LN stacks, residual ablation, per-depth signals and PyTorch agreement |
| [notes.md](notes.md) | Equations, information loss, costs, pitfalls and dated execution evidence |
| [tests](tests/test_transformer_components.py) | Finite differences, autograd, token independence, official encoder-layer oracle and CLI checks |
| [interview_questions.md](interview_questions.md) | Architectural reasoning and diagnostic trade-offs |
| [references.md](references.md) | Original papers and versioned PyTorch API sources |
| [visualizations.py](visualizations.py) / [visual guide](VISUAL_LAB.md) | Separate eleven-group offline laboratory, animations, interactive geometry and measured diagnostics |
| [Visual numerical tests](tests/test_visual_core.py) / [export tests](tests/test_visual_exports.py) | Norm axes, geometric identities, actual gradient paths and temporary-directory exports |

## Run from the repository root

Use an existing compatible environment or install the canonical extras in an
isolated environment; versions are declared in [pyproject.toml](../../pyproject.toml).

```sh
python -m pip install -e ".[dev,deep-learning]"
python 05-transformers-llms/70-transformer-ffn-residual-norm/from_scratch.py
python 05-transformers-llms/70-transformer-ffn-residual-norm/example.py
python 05-transformers-llms/70-transformer-ffn-residual-norm/example.py --final-norm
python scripts/validate_repo.py tests --scope 05-transformers-llms/70-transformer-ffn-residual-norm
```

The PyTorch CLI defaults to CPU, float64, one thread, seeds 70/71/72, 24 blocks,
width 64, four heads and FFN width 256. Batch size is two and sequence length is
eight. Four conditions share corresponding initial parameters: Pre/Post-LN,
each with and without both residual additions. The objective is mean MSE to
the same independent random target. There is no causal mask, padding,
positional encoding, dropout or training.

JSON is printed to stdout. To retain a new record locally:

```sh
python 05-transformers-llms/70-transformer-ffn-residual-norm/example.py --output 05-transformers-llms/70-transformer-ffn-residual-norm/outputs/my_initialization_run.json
```

`outputs/` is already ignored by repository policy. Existing output files are
rejected to preserve past runs. No generated assets are selected for versioning.
`--depth` and `--seeds` allow explicitly configured follow-up diagnostics.

## Separate interactive visual laboratory

```sh
python 05-transformers-llms/70-transformer-ffn-residual-norm/visualizations.py --all --quick
python 05-transformers-llms/70-transformer-ffn-residual-norm/visualizations.py --topic normalization --quick --show
```

The [visual guide](VISUAL_LAB.md) maps eleven groups to their mathematical
questions, artifacts and limitations. Generated runs have an offline index,
standalone interactive Plotly HTML, PNGs, GIFs, CSV measurements and a dated
manifest. Controls expose FFN geometry, distributions, residual scaling,
LayerNorm/RMSNorm/BatchNorm axes, signed SwiGLU gates and theoretical budgets.
Real PyTorch depth diagnostics and an actual forward/backward block trace
remain separate from the analytical seed-70 records below. All generated
outputs stay ignored; public preview selection and interpretations await
human review. No training or latency benchmark is included.

## Evidence and boundaries

The [2026-11-25 execution record](notes.md#execution-record--2026-11-25) reports
three seeds for the raw readout and a separate shared-final-LayerNorm control.
NumPy/PyTorch maximum absolute differences were below $5\times10^{-15}$ for
LayerNorm forward, its input VJP and the copied-weight FFN.

Raw-readout Pre-LN residual output RMS was 1.903370–1.930206; Post-LN was
0.999995 at six decimal places. A common final normalization changed the
gradients while leaving internal block states unchanged. These measurements
describe initialization under a particular loss. Gradient magnitude alone
does not rank architectures; even residual removal did not consistently reduce
the input gradient across seeds and normalization arrangements.

Key takeaways:

- FFN expansion adds nonlinear feature capacity, without directly mixing tokens.
- An identity path can amplify, transmit or cancel gradients in combination
  with its learned branch; it does not guarantee stable training.
- LayerNorm removes per-token mean and approximately removes positive scale
  before its affine transform. It does not impose a Gaussian distribution.
- Norm placement and final readout must be stated when comparing gradients.
- Parameter counts and asymptotic operation counts do not establish latency,
  convergence, semantic quality or production readiness.

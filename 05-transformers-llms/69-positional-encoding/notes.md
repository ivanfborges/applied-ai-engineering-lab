# Positional information: coordinates, rotations and biases

## What symmetry actually says

Let $X\in\mathbb{R}^{n\times d}$ and let $P$ permute token rows. With shared
projections, $Q=XW_Q$, $K=XW_K$ and $V=XW_V$. For unmasked attention,

$$
F(X)=\operatorname{softmax}_{\mathrm{row}}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V,
\qquad F(PX)=PF(X).
$$

This follows from $Q'K'^\top=PQK^\top P^\top$ and the corresponding permutation
of row-softmax. Equivariance rearranges contextual vectors; invariance would
leave the output unchanged. Symmetric mean pooling after such a block is
invariant. Residual paths, tokenwise feed-forward layers and per-token layer
normalization preserve equivariance when no position-dependent operation is
introduced. Independently sampled dropout masks complicate a samplewise comparison.

A fixed lower-triangular causal mask breaks general permutation equivariance
because visibility is tied to sequence slots. This can provide implicit order
signals; claiming all position-free causal models are incapable of order
reasoning would be too strong. If a mask is permuted along both axes together
with the tokens, the underlying masked operator transforms consistently again.
Causality controls access; it does not explicitly label each relative distance.

For additive positions $E$, the experiment compares $F(PX+E)$ with $PF(X+E)$.
It does **not** permute $E$ in that intervention. In the joint-coordinate control,
$F(P(X+E))=PF(X+E)$ still holds. This distinguishes rearranging content in fixed
slots from simply relabeling all rows.

## Absolute encodings

For even width $d$ and base $b>1$, define $\omega_i=b^{-2i/d}$:

$$
PE(p)_{2i}=\sin(p\omega_i),\qquad PE(p)_{2i+1}=\cos(p\omega_i),
\qquad i=0,\ldots,d/2-1.
$$

The input becomes $z_p=e(t_p)+PE(p)$, preserving width. The original Transformer
scaled token embeddings by $\sqrt d$ before this addition; this demonstration
uses unit-scale random embeddings and documents that choice. It is not a
reproduction of the original training setup. See
[Vaswani et al., section 3.5](https://arxiv.org/html/1706.03762v7#S3.SS5).

For the column vector $u_i(p)=[\sin(p\omega_i),\cos(p\omega_i)]^\top$,

$$
u_i(p+\Delta)=
\begin{bmatrix}
\cos(\Delta\omega_i)&\sin(\Delta\omega_i)\\
-\sin(\Delta\omega_i)&\cos(\Delta\omega_i)
\end{bmatrix}u_i(p).
$$

Each pair contributes one to the squared norm. For **pure, unprojected**
sinusoidal vectors,

$$
PE(p)^\top PE(r)=\sum_{i=0}^{d/2-1}\cos((p-r)\omega_i).
$$

This identity explains equal-displacement dot products, not semantic similarity
or universally relative attention. The attention numerator after addition is
$(e_p+PE(p))^\top W_QW_K^\top(e_r+PE(r))$, which also includes content-content
and cross terms; arbitrary projections need not preserve the pure-position identity.
Addition does not guarantee that content and position can be uniquely recovered.

A learned absolute table $A\in\mathbb{R}^{L\times d}$ instead supplies $A[p]$.
It introduces $Ld$ trainable parameters and requires a policy for unsupported
indices. For $L=4096,d=768$, the count is $3{,}145{,}728$. This is an analytical
parameter count, not a measured memory footprint. `absolute_position_lookup`
only reads a supplied table and rejects out-of-range indices. Interpolation,
resizing and training such a table are outside the implementation.

## Relative representations and rotary geometry

A scalar relative-bias model can write

$$
s_{pr}=\frac{q_p^\top k_r}{\sqrt{d_k}}+b_{r-p}.
$$

Clipping or bucketing distances changes which offsets share parameters. Relative
representations can also enter key/value interactions, as in
[Shaw et al.](https://arxiv.org/abs/1803.02155); a scalar-bias formula is not a
complete implementation of that method. It remains theory here.

The adjacent-pair RoPE convention uses a different coordinate ordering from the
sin/cos column vector above. For each query/key pair $[x_0,x_1]^\top$,

$$
R(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix},
\qquad \theta=p\omega_i.
$$

Apply these $2\times2$ blocks along the feature dimension. With
$\widetilde q_p=R_pq_p$ and $\widetilde k_r=R_rk_r$,

$$
\widetilde q_p^\top\widetilde k_r=q_p^\top R_p^\top R_rk_r
=q_p^\top R_{r-p}k_r.
$$

This is the relative interaction derived in
[RoFormer, section 3](https://arxiv.org/html/2104.09864v5#S3).
The transformation preserves norms; it does not force every dot product to
decrease with distance. In two dimensions, identical $[1,0]$ content gives
$\cos(r-p)$, which oscillates. A common shift preserves the interaction when
the underlying Q/K content and frequency configuration stay fixed. It does
not imply all hidden states of a real model are unchanged after relocating a
prompt with different surrounding content or masks.

`apply_rope` rotates the whole even head dimension. It leaves values untouched
in the attention demonstration. Checkpoint implementations can pair dimensions
differently or rotate only a subset. Adjacent-pair and split-half layouts require
consistent weight/layout conversion; replacing one without it is not safe.

## Causal ALiBi and numerical conventions

For head $h$ with positive slope $m_h$,

$$
s^{(h)}_{pr}=\frac{q_p^\top k_r}{\sqrt{d_k}}-m_h(p-r),\quad r\le p;
\qquad s^{(h)}_{pr}=-\infty,\quad r>p.
$$

The distance bias is added **after** scaling the content score. Increasing
$m_h$ strengthens recency preference with content held fixed, as described by
[Press et al., section 3](https://arxiv.org/html/2108.12409v2#S3). Strong content
compatibility can still overcome the bias. Supplied teaching slopes are not
the paper's head-count-dependent schedule. With equal content logits, the
weights form a truncated geometric distribution in past distance.

`causal_alibi_bias` returns `(heads, query_tokens, key_tokens)`, including
rectangular shapes and absolute offsets. It includes the diagonal, masks future
keys with negative infinity and rejects a query with no visible key. It has no
padding or packed-document boundary mask: integrate those separately. PyTorch
`MultiheadAttention` Boolean masks use `True` for **blocked** links; see the
[versioned API](https://docs.pytorch.org/docs/2.15/generated/torch.nn.MultiheadAttention.html).

All numerical helpers use float64 and explicit, nonnegative integer positions.
Even width is an intentional teaching restriction, not a universal Transformer
requirement. IDs above $2^{53}-1$ are rejected to avoid integer aliasing during
float64 conversion. Exact integer conversion does not guarantee precise
trigonometric phases: multiplying large positions by frequencies introduces
rounding. Test tolerances cover the stated fixtures, not arbitrary context lengths.

## Inference and application trade-offs

| Mechanism | Where position enters | Additional learned positional parameters | Boundary |
|---|---|---|---|
| Sinusoidal | Add to token embeddings | None | Computable beyond training length; generalization unestablished |
| Learned absolute | Add table row | $Ld$ | Unseen indices require adaptation |
| Relative scalar bias | Add to logits | Depends on bias design | Bucketing/clipping may discard distance detail |
| RoPE | Rotate Q/K after projection | None in this standard form | Layout, frequencies and cache positions must agree |
| ALiBi | Add head-specific logit penalty | None with fixed slopes | Recency preference can be undesirable for distant evidence |

Generating sinusoidal vectors or rotating Q/K costs $O(nd)$; dense attention
still costs $O(n^2d)$ for total head width $d$. This educational ALiBi function
materializes $O(Hn_qn_k)$ bias entries; optimized kernels may avoid that allocation.
No timing or memory benchmark was run.

Cached keys in the demonstrated RoPE convention are already rotated. New queries
and keys need positions consistent with that cache. The control compares eight
tokens computed together with the last three queries attending to five cached
keys plus three new keys under a causal mask. Its intentionally wrong branch
resets only the new token IDs. This is different from a consistent common shift
of **all** keys and queries. Actual multilayer caches, sliding windows, packing,
padding and prefix reuse need their own integration tests.

Distinguish training context length, runtime-supported length and task-specific
effective use of context. RAG evaluation should control retrieval quality,
evidence position, distractors and token budget. Changing a pretrained position
mechanism requires compatibility analysis and task evaluation. For time series,
index distance need not equal elapsed time; document layouts and image patches
may need spatial coordinates rather than one text index.

## Execution record — 2026-11-25

Executed by Codex on Windows CPU, Python 3.11.14, NumPy 2.4.6, PyTorch 2.15.0+cpu.
Commands from the repository root:

```powershell
python 05-transformers-llms/69-positional-encoding/from_scratch.py
python 05-transformers-llms/69-positional-encoding/example.py --output 05-transformers-llms/69-positional-encoding/outputs/2026-11-25-analytical.json
```

The actual interpreter was `.local/environments/n09-full-2026-11-13/Scripts/python.exe`.
The JSON record is deliberately ignored and is not a public link target.
These are new analytical runs, not reruns of another day's evidence.

**Hypotheses and configuration.** Unmasked, dropout-free attention should obey
permutation equivariance; holding coordinates or a causal mask fixed should
generally break it. The PyTorch fixture uses seed 69, six tokens, width 16,
four heads, float64, one CPU thread, evaluation/inference mode, no training and
permutation `[2,0,5,1,4,3]`. Embeddings have unit scale; base is 10,000.
The cache fixture uses NumPy seed 169, eight Q/K/V rows, head width 16, prefix
length five and unrotated values. All weights and vectors are synthetic.

| Control | Observed maximum absolute error or difference |
|---|---:|
| No position: $F(PX)$ versus $PF(X)$ | $1.6653\times10^{-16}$ |
| Fixed sinusoidal slots: $F(PX+E)$ versus $PF(X+E)$ | $0.215705$ |
| Joint token/coordinate permutation | $2.2204\times10^{-16}$ |
| Fixed causal mask, no positional vectors | $0.552863$ |
| NumPy versus independently constructed PyTorch sinusoids | $1.1102\times10^{-16}$ |
| Correct RoPE cache offsets versus full causal attention | $1.1102\times10^{-16}$ |
| New token IDs reset while cached keys remain unchanged | $0.426952$ |

**Interpretation candidate — author review required.** These outcomes are
consistent with the stated symmetry and show a detectable offset error in this
fixture. A larger difference is not a better model: there are no labels, fitted
weights, semantic predictions or downstream quality metrics. The fixed-mask
control demonstrates an order signal even without explicit positional vectors.

**Numerical hypotheses and configuration.** The NumPy checks use seed 69, width
16, base 10,000 and float64. The pure sinusoidal Gram matrix for positions 0–11
should agree after shifting both indices by one. Fixed Q/K vectors with distance
five should preserve rotary dot products under a common shift; norms should be
preserved. Equal content logits should expose ALiBi's recency penalty directly.

The shifted Gram error was $2.6645\times10^{-15}$; the inspected RoPE norm error
was zero at printed float64 precision. The reference dot product at `(10,15)`
was $-2.685354516621011$:

| Query/key positions | Absolute dot-product error versus `(10,15)` |
|---|---:|
| `(100,105)` | $1.7319\times10^{-14}$ |
| `(1000,1005)` | $8.6597\times10^{-14}$ |
| `(10000,10005)` | $9.3037\times10^{-13}$ |
| `(1000000,1000005)` | $1.6861\times10^{-10}$ |

With 16 visible keys at positions 0–15, query position 15 and zero content logits:

| ALiBi slope | Self-attention probability | Expected past distance, in tokens |
|---:|---:|---:|
| 0.02 | 0.072307 | 7.075726 |
| 0.1 | 0.119236 | 5.460806 |
| 0.5 | 0.393601 | 1.536125 |
| 1.0 | 0.632121 | 0.581975 |

**Interpretation candidate — author review required.** Numerical agreement
supports the algebra on these fixtures; the growing rotary residual also exposes
finite-precision effects at large indices. Stronger slopes concentrate this
flat-content distribution on recent keys. These observations neither compare
trained positional methods nor establish reliable million-token reasoning.

**Local verification.** With pytest 9.2.0, the focused command
`python scripts/validate_repo.py tests --scope 05-transformers-llms/69-positional-encoding`
passed 49 cases in one isolated file, without skips or warnings. Coverage includes
an independent scalar sinusoidal formula, block-matrix RoPE oracle, geometric
ALiBi probabilities, invalid inputs, the cache intervention and preservation of
an existing output record. Repository checks parsed 493 Python files and checked
1,824 local links in 308 Markdown files. The explicit-interpreter offline
`uv --no-cache --offline pip check --python .local/environments/n09-full-2026-11-13/Scripts/python.exe`
reported 81 compatible installed packages. `git diff --check` found no whitespace
errors in tracked changes. This is focused local evidence; no full-suite, CI,
browser/GitHub-rendering or scientific-author approval is implied.

## Suggested follow-ups — not executed

The separately implemented [visual lab](VISUAL_LAB.md) now explores frequency
bases/widths, pure-position geometry and NumPy float32/64 phase precision.
Its seed-269 measurements are separate from the seed-69 analytical record above.
The trained-task and RAG quality studies below remain unexecuted.

- Compare bases and dimensions under a controlled trained sequence task, holding
  data, optimization and parameter budget explicit. Analytical curves alone
  cannot select a universally better base.
- Evaluate grounded answers and citations while moving evidence across a fixed
  RAG context; vary distractors and token budgets over multiple examples.
- Evaluate bfloat16 or mixed-precision phase construction separately. The visual
  lab's executed float32/64 check does not cover these modes or a trained model's
  inference precision policy.
- Test learned-table adaptation, partial rotary layouts and a real multilayer
  cache separately. None is validated by this one-layer control.

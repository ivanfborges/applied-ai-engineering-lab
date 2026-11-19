# Recurrent state, gated memory and finite learning paths

## Shared state and BPTT

For column-vector input $x_t\in\mathbb{R}^D$ and state
$h_t\in\mathbb{R}^H$, an Elman tanh RNN computes

$$
h_t=\tanh(W_xx_t+W_hh_{t-1}+b).
$$

Here $W_x$ has shape $H\times D$, $W_h$ has shape $H\times H$, and $b$ has
shape $H$. Parameters are reused across positions; sequence length does not
change their count. A sequence-to-one classifier uses logit
$a=w^\top h_T+b_y$ and the stable binary cross-entropy with logits. Applying
sigmoid first and then using `BCEWithLogitsLoss` would apply the wrong objective.
Sequence-to-sequence tasks can instead attach losses at multiple positions.

Unrolling the recurrence gives an ordinary differentiable computation graph.
BPTT sums gradients from all uses of each shared parameter. With a final-state
loss and $J_k=\operatorname{diag}(1-h_k^2)W_h$,

$$
\nabla_{h_t}\mathcal L
=J_{t+1}^{\top}J_{t+2}^{\top}\cdots J_T^{\top}\nabla_{h_T}\mathcal L.
$$

A scalar path with derivative 0.8 repeated 49 times contributes $0.8^{49}$;
a derivative of 1.2 contributes $1.2^{49}$. These are idealized products,
not measurements from trained models. Matrix products depend on directions,
changing activation derivatives and noncommuting factors; a recurrent matrix's
spectral radius alone does not characterize finite-horizon gradient behavior.

Global norm clipping rescales a large gradient vector to a maximum norm before
the update. It helps constrain explosions but cannot restore missing gradients.
The code records preclip norms and clipped-batch counts. It does **not** measure
loss gradients by timestep, and poor recall alone cannot diagnose vanishing
gradients as the cause.

## LSTM: retained memory and exposed state

The NumPy cell concatenates $u_t=[x_t;h_{t-1}]$ and uses gate order
input, forget, candidate, output. Each gate matrix has shape $H\times(D+H)$:

$$
\begin{aligned}
i_t&=\sigma(W_i u_t+b_i),& f_t&=\sigma(W_f u_t+b_f),\\
g_t&=\tanh(W_g u_t+b_g),& o_t&=\sigma(W_o u_t+b_o),\\
c_t&=f_t\odot c_{t-1}+i_t\odot g_t,& h_t&=o_t\odot\tanh(c_t).
\end{aligned}
$$

The cell state carries an additive memory path; the hidden state exposes a
gated nonlinear view of that memory. Gates have values in $(0,1)$ in exact
arithmetic; finite precision can saturate them to endpoints. They are learned
functions of input and previous hidden state, not hard logical switches.

Holding gates and candidate fixed, the direct cell-state Jacobian is
$\operatorname{diag}(f_t)$. Its diagonal path through many steps contains a
product of forget gates. Gates near one can preserve this path. This is not
the full state Jacobian: indirect dependencies through hidden states and gates
also contribute. LSTMs mitigate gradient problems; neither unlimited memory
nor lossless history storage follows from the equations.

`SimpleLSTMCell` is an **educational forward implementation**, with combined
biases and no BPTT, batching, optimized kernels, projections or peepholes.
Its default forget bias of 1 is illustrative. The PyTorch training comparison
uses framework-default initialization without a custom forget bias.
The tests map the NumPy weights to `torch.nn.LSTMCell` and split its combined
bias into two bias vectors, then compare multiple steps with nonzero states.
Agreement checks arithmetic, not model quality or causal gate explanations.

## GRU: specify which candidate equation

The example follows PyTorch's GRU convention:

$$
\begin{aligned}
r_t&=\sigma(W_{ir}x_t+b_{ir}+W_{hr}h_{t-1}+b_{hr}),\\
z_t&=\sigma(W_{iz}x_t+b_{iz}+W_{hz}h_{t-1}+b_{hz}),\\
n_t&=\tanh(W_{in}x_t+b_{in}+r_t\odot(W_{hn}h_{t-1}+b_{hn})),\\
h_t&=z_t\odot h_{t-1}+(1-z_t)\odot n_t.
\end{aligned}
$$

Here $z_t$ near one retains old state. Other presentations reverse the meaning
of the update-gate symbol; check the interpolation equation before interpreting
a gate. PyTorch applies the reset gate **after** the recurrent affine map in
the candidate. Applying it to $h_{t-1}$ **before** the matrix multiplication
defines a different parameterization and generally a different function.
See the [official GRU notes](https://docs.pytorch.org/docs/2.15/generated/torch.nn.GRU.html).

## Cost and access to history

For the exact single-layer, unidirectional PyTorch configuration here, with two
bias vectors per gate and a scalar linear head:

$$
P=kH(D+H+2)+(H+1),\qquad k\in\{1,4,3\}
$$

for RNN, LSTM and GRU respectively. At $D=2,H=16$ this gives 337, 1,297 and
977 trainable parameters. Equal hidden dimensions share state width, not
parameter or compute budgets. All three require sequential recurrent work of
order $O(T(HD+H^2))$ at fixed layer count, with different constant factors.
Inference can retain $O(H)$ state per layer and entity (both $h,c$ for LSTM).
Full BPTT retains intermediate activations growing with $T$; bounded inference
state does not imply bounded training memory.

| Question | Recurrence | Standard Transformer attention |
|---|---|---|
| How is history accessed? | Through a learned compressed state | Through attention to accessible position representations |
| What limits parallelism? | State at $t$ depends on $t-1$ | Training positions can be processed in parallel within a layer |
| How does attention work scale? | No attention matrix in this example | Full attention score/value work is $O(T^2d)$, excluding projections and feed-forward blocks |
| What happens during streaming? | Update bounded state | Autoregressive decoding remains sequential, commonly with a growing KV cache |

Masking and context windows determine which attention connections are allowed.
Attention does not guarantee use of the relevant context, and specialized
attention/recurrent variants change this comparison. No Transformer is trained
here. For applications, compare recurrent models with seasonal persistence,
linear/lagged-feature models and temporal convolutions under the actual budget.

## Sequence boundaries, padding and information availability

With `batch_first=True`, inputs and outputs use `(N,T,D)` and `(N,T,H)`;
states still use `(layers*directions,N,H)`. LSTM returns `(h_n,c_n)`; the
classifier reads `h_n[-1]`. This example has one direction and fixed lengths.
Using `output[:,-1]` with padded sequences can read padding-influenced state.
Use valid lengths with packing or appropriate indexing; handle both directions
explicitly in bidirectional architectures.

At inference, carrying state across ordered chunks of the same sequence
preserves the computation within numerical tolerance. The tests check this
for RNN, LSTM and GRU in evaluation mode. Resetting state at a chunk boundary
discards context. Detaching an existing state preserves its values but breaks
the gradient graph: this is the key trade-off in truncated BPTT. No stateful
training or truncated-BPTT experiment is implemented here.

For real streams, isolate state per entity, reset at real boundaries and
define ordering, expiry and replay rules. A backward recurrence is valid on a
fully observed historical window; leakage occurs when it sees information
unavailable at the decision timestamp. Forecasting requires chronological or
rolling-origin evaluation, with horizon-aware gaps for overlapping windows and
training-only preprocessing. Random IID sequences in this diagnostic have no
shared timeline; independent generated splits are appropriate only for this
stated task.

## Executed study — 2026-11-19

**Hypothesis:** gated recurrence may retain an early bit across distractors more
effectively than a tanh RNN, but the fixed training budget may fail to reveal
this advantage. This is a diagnostic hypothesis, not a promised ranking.

**Execution/configuration:** the default `example.py` command in the README was
run once on 2026-11-19. Its record was created at 08:48:20 UTC (05:48:20 in
America/Sao_Paulo). The ignored JSON basename is
`recall_20261119T084820515223Z.json`; this table intentionally preserves a small
public summary without linking a private or ignored artifact.

- Python 3.11.14, NumPy 2.4.6, PyTorch 2.15.0+cpu; Windows 10, build 19045,
  x86-64, Intel64 Family 6 Model 158 Stepping 9. CPU float32, one PyTorch thread,
  deterministic algorithms enabled. This is not a latency benchmark.
- Train/validation/test sizes: 1,024/256/512, generated with seeds 60/61/62.
  Model initialization seed 63 and shuffle seed 64 are restarted for every
  cell/length. Separate lengths pair labels, but not identical noise prefixes.
- Lengths 10 and 50; one unidirectional recurrent layer with hidden size 16 and
  a scalar linear head; default PyTorch initialization, no custom forget bias.
- Adam at 0.005, other optimizer defaults unchanged; 20 epochs, minibatches of
  64 (320 updates per model), norm clipping at 1. No search or post-result tuning.
- Selection: earliest minimum validation BCE among epoch ends, separately for
  each of the six predefined configurations. Test data are evaluated after
  restoring each checkpoint and are not supplied to the fitting function.

**Measured result** (BCE rounded to six decimals; accuracies are percentages):

| Cell | Length | Parameters | Selected epoch | Validation BCE | Test BCE | Test accuracy |
|---|---:|---:|---:|---:|---:|---:|
| RNN | 10 | 337 | 20 | 0.001730 | 0.001330 | 100.0000% |
| LSTM | 10 | 1,297 | 20 | 0.001064 | 0.001060 | 100.0000% |
| GRU | 10 | 977 | 20 | 0.000724 | 0.000726 | 100.0000% |
| RNN | 50 | 337 | 20 | 0.037086 | 0.023709 | 99.8047% |
| LSTM | 50 | 1,297 | 2 | 0.693586 | 0.694143 | 48.4375% |
| GRU | 50 | 977 | 9 | 0.694999 | 0.694450 | 50.0000% |

The training positive rate was 48.9258%, validation 52.7344% and test 50.9766%
for both lengths. The training-majority classifier therefore predicts zero,
with test accuracy 49.0234%; training-prior test BCE was 0.693798. The exact
first-step sign control attained 100% at both lengths. The last-step sign
control, which uses only a distractor and has population accuracy 50%, attained
50.1953% at length 10 and 48.6328% at length 50. Its finite-sample result is
not a trained last-step classifier or a proof of statistical independence;
independence is specified by the generator.

At length 50, the selected LSTM/GRU training BCEs were 0.692496/0.691565,
consistent with failure to fit useful recall under this configuration. Their
final-epoch validation BCEs were 0.701487/0.702625, greater than the selected
values. Reporting final weights would therefore change the stated protocol.
Across all 320 minibatches per model, clipping activated 41 times for the
length-50 RNN and zero times for the length-50 LSTM/GRU. These aggregate norms
do not localize gradient propagation over time.

**Interpretation candidates — author review required:** all three models
learned short recall here. At length 50, this RNN configuration transferred
successfully to new draws, whereas the gated configurations did not improve
meaningfully on the simple controls. The run does not support the hypothesized
gated advantage under this budget. It also does not contradict the existence
of better gated gradient paths: learning those paths depends on initialization
and optimization. No gate trajectories or temporal derivatives were measured,
so attributing failures specifically to gate saturation or vanishing gradients
would exceed the evidence.

**Limitations:** one training seed, unequal parameter budgets, one rate and
epoch budget, fixed lengths and an artificial bit-recall task. Different
lengths train different models; this is not inference-time length extrapolation.
Held-out metrics have sampling variability and are conditional on this fitted
model and generator. There are no repeated-seed intervals, real temporal data,
Transformer comparison, serving measurements or author-approved conclusions.
The exact sign rule highlights how task design can make a nonrecurrent solution
trivial. No general architecture superiority follows from these values.

The separate `from_scratch.py` command also executed three untrained synthetic
forward steps with seed 60, input size 2, hidden size 3 and forget bias 1.
It exposes hidden/cell states and forget values; it does not collect trained
model evidence. Numerical and integration tests check framework agreement,
chunk continuity, validation checkpoint selection and small deterministic
training. These checks are software evidence, not replications of the table.

## Extensions not executed in the analytical protocol

The separate [visual protocol](VISUAL_LAB.md) subsequently executed paired
seeds, an approximately matched-parameter comparison and temporal input-loss
sensitivities. The earlier analytical record above remains unchanged. The
remaining extensions below retain their proposed status for that protocol.

- Repeat paired training across seeds and summarize variability; do not treat
  the two lengths as independent replications of a general architecture claim.
- Match approximate parameter budgets and tune on validation with equal search
  budgets. Equal parameter counts still do not equalize computation.
- Measure temporal input sensitivities, distinguishing loss gradients from
  output-logit sensitivities and from causal feature importance.
- Test variable-length packing, truncated BPTT and streaming entity boundaries.
- Evaluate real temporal data against lagged-feature and seasonal baselines;
  measure latency and memory separately under a declared runtime protocol.

## Interview checks

**Why can an LSTM still forget?** Forget gates may contract the memory path,
input updates can overwrite content, output saturation can suppress gradients,
and training may never discover a useful retention policy. The direct path
derivative is not the entire recurrent Jacobian.

**Does clipping fix delayed recall?** It constrains large updates, not vanishing
signals. Check task solvability, loss/gradient behavior and optimization before
attributing a failed run to a particular mechanism.

**What does a state carried across minibatches mean?** It is valid only if rows
continue the same entities in order. Shuffling independent examples while
retaining state mixes histories; reset or explicitly align each entity.

**When is bidirectionality safe?** When both directions use observations already
available at prediction time, such as a completed document or historical window.
It is unsafe if later events enter a prediction claimed to occur earlier.

**Why did attention change sequence modeling?** It provides direct access
between accessible positions and more training parallelism, replacing the
requirement that context traverse every recurrent transition. Its context,
attention work and decoding cache still have costs.

## Primary references

- [PyTorch 2.15 RNN](https://docs.pytorch.org/docs/2.15/generated/torch.nn.RNN.html): tanh recurrence, tensor/state shapes and parameter layout.
- [PyTorch 2.15 LSTM](https://docs.pytorch.org/docs/2.15/generated/torch.nn.LSTM.html): gate equations, two states, gate order and biases.
- [PyTorch 2.15 GRU](https://docs.pytorch.org/docs/2.15/generated/torch.nn.GRU.html): update/reset conventions and candidate-equation difference.
- [PyTorch reproducibility](https://docs.pytorch.org/docs/2.15/notes/randomness.html): seeds and deterministic algorithms do not guarantee identical values across releases or platforms.
- [Learning long-term dependencies with gradient descent is difficult](https://doi.org/10.1109/72.279181), Bengio, Simard and Frasconi (1994).
- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735), Hochreiter and Schmidhuber (1997). The implemented forget-gate variant is described by the framework equations above; it is not an exact reproduction of the original architecture.
- [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078), Cho et al. (2014).
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Vaswani et al. (2017).

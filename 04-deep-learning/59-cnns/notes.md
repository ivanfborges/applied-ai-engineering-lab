# CNNs: local operators, spatial context and evidence

## What is shared, and what is learned?

For an input $X$ in NCHW layout, standard convolution has a weight tensor
$W\in\mathbb{R}^{C_{out}\times C_{in}\times K_h\times K_w}$. One output channel
has a spatial kernel for each input channel; their responses are summed.
An RGB input does not produce three independent outputs for each filter.

Using zero-extended input $\widetilde X$, scalar stride $s$, symmetric padding
$p$ and dilation $d$, the operation for one batch element is

$$
Y_{o,i,j}=b_o+\sum_{c=0}^{C_{in}-1}\sum_{u=0}^{K_h-1}\sum_{v=0}^{K_w-1}
W_{o,c,u,v}\widetilde X_{c,is+ud-p,js+vd-p}.
$$

Here zero extension uses coordinates of the original image: out-of-range
indices return zero. The implementation instead materializes a padded array
and starts windows at $(is,js)$. Neither convention reverses the kernel.
Strict mathematical convolution reverses it; a learned unconstrained kernel
can represent either orientation by learning different coefficients.

The weights are reused at every spatial location. Backpropagation therefore
sums contributions from all the locations using each coefficient. A handwritten
edge detector demonstrates the operator, while the classifier's filters are
learned from its loss. No particular channel is guaranteed to represent an
edge, a part or a human-readable concept.

For a standard biased layer,

$$
P=K_hK_wC_{in}C_{out}+C_{out}.
$$

RGB to 32 channels with $3\times3$ kernels has 896 parameters. The implemented
grayscale-to-eight-channel layer has 80. Without a bias, omit the last term.
Spatial arithmetic grows approximately with
$H_{out}W_{out}K_hK_wC_{in}C_{out}$ per image; sharing parameters does not remove
those operations. No timing or hardware benchmark is measured here.

## Output dimensions and the meaning of context

For one axis, with input size $I$, kernel size $K$, padding $p$, dilation $d$
and stride $s$ in floor mode,

$$
O=\left\lfloor\frac{I+2p-d(K-1)-1}{s}+1\right\rfloor.
$$

Stride, kernel and dilation must be positive integers; padding is nonnegative.
The effective kernel must fit the padded input. With stride 1, an odd kernel
and suitable padding can preserve size. “Same” does not mean unchanged
boundary values, and symmetric padding does not preserve size for every
stride/kernel combination. Dilated kernels sample spaced positions; a larger
bounding context does not imply dense coverage of all its pixels.

Track receptive-field span $r_l$ and jump $j_l$ along one sequential path:

$$
r_0=j_0=1,\qquad
r_l=r_{l-1}+d_l(K_l-1)j_{l-1},\qquad
j_l=j_{l-1}s_l.
$$

Apply this independently to height and width. ReLU is pointwise and leaves
the spatial support unchanged.

| Operation | Kernel | Stride | Span | Jump in input pixels |
|---|---:|---:|---:|---:|
| Input | — | — | 1 | 1 |
| Conv | 3 | 1 | 3 | 1 |
| Pool | 2 | 2 | 4 | 2 |
| Hypothetical second conv | 3 | 1 | 8 | 2 |

The classifier stops its local feature extraction at pooling. Its global mean
combines all $8\times8$ pooled positions, so the logits can depend on the entire
$16\times16$ image. The hypothetical second convolution is a separate geometry
example, not a layer in the trained network.

This recurrence gives a theoretical bounding span, not the alignment of the
region, the number of distinct sampled pixels, or the strength of influence.
Padding reduces real-image coverage near boundaries. Dilation/stride can leave
holes; max pooling selects a data-dependent route for local gradients. The
effective receptive field depends on weights, data and nonlinearities.
Branched or residual architectures need support/alignment reasoning per path;
the helper deliberately handles only a sequential path.

## Pooling and translations

Over a window $\mathcal R$, max pooling computes
$\max_{(u,v)\in\mathcal R}X_{u,v}$; average pooling computes
$|\mathcal R|^{-1}\sum_{(u,v)\in\mathcal R}X_{u,v}$. Neither has trainable weights.
Max pooling emphasizes the largest response and locally routes gradients to
the selected maximum; ties need a selection convention. Average pooling
distributes the gradient across its window, and can dilute a sparse response.

The NumPy pooling helper has no padding, no dilation and no ceil mode. In floor
mode a partial trailing window is discarded. Negative values are valid: max
pooling must not introduce artificial zeros. PyTorch's padded max pooling uses
negative infinity; that padding mode is outside the helper's scope.

For a compatible translation operator $T_\Delta$, convolution on an infinite
grid satisfies $f(T_\Delta X)=T_\Delta f(X)$. This is equivariance: the feature
map shifts along with the input. Invariance would mean unchanged output.
Finite borders and cropping complicate this identity; stride $s$ also restricts
exact alignment to shifts compatible with the sampling grid. A one-pixel move
can cross a pooling-window boundary and change the pooled map, as a numerical
counterexample in the tests demonstrates.

Global averaging removes explicit spatial coordinates from the head, but
boundaries, preceding downsampling and altered feature responses still matter.
No shift-invariance claim is made from classification accuracy. Downsampling
can alias high-frequency patterns; pooling is not a general anti-aliasing
guarantee.

## Design choices and application limits

- Stacking local layers expands context and introduces nonlinear compositions.
  The implemented model has only one convolution and does not demonstrate a
  learned hierarchy of semantic parts.
- A $1\times1$ convolution mixes channels without expanding spatial context.
  Depthwise filtering plus pointwise mixing changes channel coupling and cost.
  Neither variation is implemented here.
- Strided convolutions offer learned downsampling. Flattened heads retain
  location information but often add many parameters; global pooling gives a
  compact head while discarding explicit layout.
- Classification can aggregate spatial evidence. Segmentation and small-object
  detection often need higher-resolution features or multiscale reconstruction.
  This classifier provides no localization output.
- For scanned documents, split by document/source where appropriate and check
  duplicate pages, scanner artifacts, stamps and logos. The intended target
  determines whether layout or text should drive prediction. OCR, visual and
  multimodal components require evaluation as a complete pipeline.
- For industrial or medical images, acquisition shift, rare classes and label
  quality can dominate architecture choices. This study offers no evidence of
  suitability for those settings.
- Compare CNNs and attention-based backbones using task data, pretrained
  alternatives and actual hardware constraints. This study does not compare
  model families or establish an accuracy/latency ranking.

The NumPy code deliberately omits batches, channel mixing, bias, autograd and
acceleration. It is educational, not a replacement for a framework kernel.
The PyTorch example fixes CPU float32 and $16\times16$ grayscale inputs; it is
not a general image loader or serving pipeline.

## Executed study — 2026-11-18

**Hypothesis:** learned local filters followed by spatial aggregation can fit
bar orientation and transfer to independent draws from the same synthetic
generator. This tests a basic learning mechanism rather than architectural
superiority.

**Configuration:** 256 training and 128 test images, balanced classes;
$16\times16$ grayscale, independent Gaussian noise with standard deviation 0.1.
Class 0 adds a $9\times3$ vertical bar and class 1 a $3\times9$ horizontal bar,
both with intensity increment 1. Row/column centers are independently sampled
uniformly from integers 5 through 10 inclusive, so bars stay inside the image.
Signal area, intensity and center support are matched across classes.
Pixels are not clipped or normalized. All data are synthetic; no dataset was
downloaded. No real-image provenance or licensing claim is needed.

Train/test generator seeds are 59/60, model initialization seed 61, shuffle
seed 62. The 98-parameter model uses PyTorch default initialization, Adam at
0.01 with its other defaults, mean cross-entropy, batches of 32 and 30 epochs
(240 updates), one CPU thread and float32. No augmentation, weight decay,
validation tuning, early stopping or test-based selection is used. Every
reported epoch loss is evaluated on the full training set after its updates;
test metrics are evaluated only after fitting.

**Environment and execution:** Python 3.11.14, NumPy 2.4.6, PyTorch 2.15.0+cpu,
Windows, using the repository's stable local environment. Command from root:

```powershell
python 04-deep-learning/59-cnns/example.py
```

The actual command used the explicit interpreter path specified by the local
workflow. The report timestamp is `2026-11-19T00:54:21.293074+00:00`, corresponding
to 2026-11-18 21:54 in America/Sao_Paulo. The generated JSON remains ignored;
this Markdown record preserves the public protocol and selected measurements.
Future executions create new timestamped records rather than replacing this
dated observation.

A verification rerun at `2026-11-19T00:59:04.612176+00:00` (2026-11-18 21:59
in America/Sao_Paulo), after making CPU/float32 construction explicit, retained
the same configuration and exactly the same recorded result values, including
all 30 epoch-end losses. It has a separate ignored JSON record; it is a repeat
of this realization, not an additional seed or independent generalization study.

| Measurement | Observed value |
|---|---:|
| Initial full-training cross-entropy | 0.714486778 |
| Final full-training cross-entropy | 0.046860673 |
| Final training accuracy | 1.0 (256/256) |
| Test cross-entropy | 0.045123886 |
| Test accuracy | 1.0 (128/128) |
| Train-majority baseline test accuracy | 0.5 |

The train-majority baseline breaks the balanced-label tie in favor of class 0.
Test confusion matrix, true classes in rows and predictions in columns:

| True / predicted | Vertical (0) | Horizontal (1) |
|---|---:|---:|
| Vertical (0) | 64 | 0 |
| Horizontal (1) | 0 | 64 |

The separate NumPy demonstration produced valid edge responses with rows
`[0, 3, 3]`, max-pooling output `[[3.0]]` and average-pooling output `[[1.5]]`.
The last row/column of that $3\times3$ response is discarded by the $2\times2$,
stride-2 window. The separate Conv3–Pool2–Conv3 geometry calculation returned
span 8 and jump 2.

**Interpretation candidate — author review required:** the decreasing loss and
above-baseline holdout accuracy support successful fitting and same-generator
transfer for this realization. Matching bar area removes a simple total-signal
cue; it does not identify what each learned filter detects. No filter images,
attributions or model-level translation diagnostics were executed.

**Limitations:** one seed and one architecture; easy balanced data; finite
same-generator holdout; no uncertainty interval, real-image baseline, rotation,
translation, corruption, calibration or deployment study. Noise differs between
samples, but robustness to a changed noise distribution is untested. Repeated
synthetic geometries can appear in both splits; distinct noise realizations
and seeds do not simulate generalization to new acquisition sources. CPU
repeatability in the tested environment is not a bitwise portability guarantee.
Passing numerical tests does not approve this interpretation for the author.

## Proposed extensions — not executed

1. Compare kernel sizes with fixed data/splits and explicit parameter/update
   budgets, choosing configurations on validation before a fresh final test.
2. Compare max/average/strided downsampling and global/flattened heads under
   an explicit comparison protocol.
3. Translate bars without wraparound, separating interior shifts, pooling-grid
   alignment and cropping. Measure logit changes separately from accuracy.
4. Use multiple seeds and new noise/rotation distributions; retain class-level
   errors and avoid treating the current test set as a tuning set.

These suggested comparisons have no measured results in this study. The small
training test uses its own reduced configuration as software verification; it
is not pooled into the analytical experiment above.

## Interview checks

**Why does weight sharing help?** It reuses local detectors across positions
and reduces free parameters; it also assumes that such reuse fits the task.
It does not by itself establish generalization or low latency.

**How do receptive field and resolution differ?** Receptive field describes
potential input support. Resolution counts output positions; jump describes
their spacing in input coordinates. Downsampling changes both the grid and
the context of subsequent layers.

**Does pooling make a CNN translation-invariant?** It can reduce sensitivity
to some local changes, but sampling boundaries, padding and the head matter.
Test transformations relevant to the intended application.

**Why no softmax before cross-entropy?** PyTorch's loss accepts raw logits and
internally computes the log-probability operation. A separate softmax is useful
for reporting probabilities, whose calibration still needs evaluation.

## Primary references consulted

- [PyTorch 2.15 Conv2d](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Conv2d.html):
  cross-correlation, channel coupling, parameter layout and output dimensions.
- [PyTorch 2.15 MaxPool2d](https://docs.pytorch.org/docs/2.15/generated/torch.nn.MaxPool2d.html):
  stride, floor/ceil behavior and negative-infinity padding.
- [PyTorch 2.15 CrossEntropyLoss](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html):
  raw-logit and target conventions.
- [Araujo, Norris and Sim (2019), Computing Receptive Fields of Convolutional Neural Networks](https://distill.pub/2019/computing-receptive-fields/):
  single-path receptive-field derivation, boundaries and multipath limitations.

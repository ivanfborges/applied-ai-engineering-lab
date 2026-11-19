# Reconstruction, latent geometry and anomaly assumptions

## Objective and representation

For input $x\in\mathbb{R}^d$, the encoder produces $z=f_\theta(x)\in\mathbb{R}^k$
and the decoder returns $\hat{x}=g_\phi(z)$. For continuous features this study uses

$$
\mathcal{L}(\theta,\phi)=\frac{1}{nd}\sum_{i=1}^n\sum_{j=1}^d(x_{ij}-\hat{x}_{ij})^2.
$$

A linear output permits unrestricted standardized real values. A sigmoid
output would impose an inappropriate $[0,1]$ constraint here. Binary features
can instead motivate a Bernoulli reconstruction objective; mixed-type data
needs feature-specific losses and weights. Squared error corresponds to a
conditional isotropic Gaussian reconstruction noise assumption, up to scale
and constants. It is not by itself a normalized density over arbitrary inputs.

An undercomplete model has $k<d$. A narrow continuous bottleneck constrains
the mapping but does not guarantee semantics, independence, disentanglement
or good anomaly scores. Overcomplete models can learn copying unless constrained.
Denoising learns corrupted-input to clean-target reconstruction; sparse models
penalize activations. These are different objectives, not implemented variants
in this study. A VAE additionally defines a probabilistic latent model and
prior-related regularization; arbitrary latent sampling in a vanilla AE does
not provide the same generative model.

Even in the linear case latent coordinates are not unique: for invertible $A$,
$(WA)(A^{-1}V)=WV$. Compare reconstruction operators/subspaces, not raw columns
of encoder weights. Neural reconstruction also need not be the nearest-point
projection onto a manifold.

## Linear gradients and the PCA boundary

For centered $X\in\mathbb{R}^{n\times d}$, use untied weights
$W\in\mathbb{R}^{d\times k}$ and $V\in\mathbb{R}^{k\times d}$, without biases:

$$
H=XW,\qquad E=HV-X,\qquad \mathcal{L}=\frac{\|E\|_F^2}{nd}.
$$

$$
\nabla_V\mathcal{L}=\frac{2}{nd}H^\top E,\qquad
\nabla_W\mathcal{L}=\frac{2}{nd}X^\top(EV^\top).
$$

Both gradients must be calculated before either parameter is updated.
The tests compare every parameter with central finite differences and PyTorch
autograd in float64, then check one simultaneous update directly.

The globally optimal rank-$k$ linear training reconstruction attains the
truncated-SVD/PCA error on the same centered data. Gradient descent need not
reach that optimum. This does not equate nonlinear AEs with PCA or guarantee
an equivalent test error. The from-scratch script standardizes its one matrix
solely for this training identity demonstration; it claims no holdout result.

## Data roles and anomaly mechanisms

`example.py` samples one orthonormal basis $B\in\mathbb{R}^{12\times12}$ and uses

$$
x=zB_{:,1:2}^{\top}+\epsilon,\qquad
z\sim\mathcal{N}(0,I_2),\quad\epsilon\sim\mathcal{N}(0,0.15^2I_{12}).
$$

Here $z$ is a two-coordinate row vector. All splits share this geometry but
draw independent latent factors and noise. The normal distribution is
approximately linear and low rank, deliberately favorable to PCA.

| Role | Observations | Permitted use |
|---|---:|---|
| Normal training | 1,800 | Fit scaler, AE, PCA and Isolation Forest |
| Normal calibration | 800 | Set one fixed quantile threshold per frozen model |
| Test normal | 360 | Measure false positives |
| Test off-subspace | 20 | Add $4B_{:,3}$; evaluate geometric violations |
| Test on-subspace | 20 | Add $6B_{:,1}$; evaluate a rare position along a normal direction |

Anomaly labels describe these known interventions. Gaussian normal support is
unbounded; these labels are experimental definitions, not proof of impossible
normal events. The on-subspace cases probe a specific weakness of residual
scores: large latent magnitude need not produce large projection residual.
Density/isolation-based scores ask a different question. The learned PCA
subspace is finite-sample, not the exact generating basis.

StandardScaler uses only normal training rows. The AE has a 12→16→2 encoder
and 2→16→12 decoder, with ReLU at both 16-unit hidden layers and linear latent
and output layers. Adam uses learning rate 0.001, batch size 128, 120 epochs,
no weight decay, and the final epoch. The final batch has 8 rows. There is no
early stopping, validation-based checkpoint selection or hyperparameter search.
Calibration is reserved exclusively for thresholds. PCA uses rank 2 and full
SVD; Isolation Forest uses 100 trees, default subsampling/other sklearn settings,
seed 42 and one worker. Their thresholds have different score units.

The strict alert policy is

$$
s(x)=\frac{1}{d}\sum_j(x_j-\hat{x}_j)^2,\qquad
\tau=Q_{0.99}^{\mathrm{higher}}(s(X_{\mathrm{cal}})),\qquad
\hat{y}=\mathbf{1}\{s(x)>\tau\}.
$$

Ties at the threshold are not alerts. With 800 observations this convention
selects sorted index 792 (zero based); only the 7 higher scores can exceed it
in the absence of ties. No formal finite-sample/conformal coverage claim is
made. Future false positives depend on representativeness, independence and
stability. The operational cost or queue capacity is not optimized here.

ROC-AUC and average precision summarize ranking. Precision, recall, false-positive
rate and counts describe the chosen threshold. Test prevalence is 10%, and the
random-ranking AP reference is approximately 0.10, not a deployment forecast.
Each anomaly-kind recall uses just 20 cases, so one case changes it by 0.05.

## Executed study — 2026-11-19

**Hypotheses.** (1) Explicit linear gradient descent can approach the PCA
training bound. (2) Reconstruction scores can detect subspace violations
while missing rare in-subspace observations.

**Execution and configuration.** Codex ran both default scripts on CPU with
seed 42 on 2026-11-19. The nonlinear record was created at 09:32:04 UTC
(06:32:04 America/Sao_Paulo). The environment used Python 3.11.14,
NumPy 2.4.6, scikit-learn 1.9.1 and PyTorch 2.15.0+cpu on Windows.
PyTorch used one thread and deterministic algorithms; OMP, MKL and OpenBLAS
were limited to one thread by the launch command. This is a resource setting,
not a hard machine-wide cap. No packages were installed for this run.

Commands from the repository root, using the stable local interpreter:

```powershell
$labPython = '.local/environments/n09-full-2026-11-13/Scripts/python.exe'
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& $labPython 04-deep-learning/61-autoencoders/from_scratch.py
& $labPython 04-deep-learning/61-autoencoders/example.py --output 04-deep-learning/61-autoencoders/outputs/run-2026-11-19-seed42.json
```

The local JSON remains ignored and is not a public linked asset. Use a new
filename for a new run; this dated record is not replaced by later results.

**Linear training result.** A separate synthetic 500×8 approximately rank-2
matrix used Gaussian mixing and noise SD 0.1, followed by per-feature
centering/scaling. Untied weights were initialized independently with SD 0.1;
full-batch gradient descent used learning rate 0.4 and 2,000 steps. Initial
MSE was 1.0019372134; final AE MSE was 0.0096738016, against the truncated-SVD
optimum 0.0096683618. The gap was approximately $5.44\times10^{-6}$ and the
reconstruction rank was 2. This measures optimization on the training matrix,
not generalization.

**Nonlinear study result.** Under the fixed protocol above, AE training MSE
fell from 1.0635362864 to 0.1487049758 in standardized feature units.

| Model | ROC-AUC | AP | Precision | Recall | TP / FP / FN / TN | Off-subspace recall | On-subspace recall |
|---|---:|---:|---:|---:|---|---:|---:|
| Nonlinear AE | 0.820833 | 0.647967 | 0.833333 | 0.500 | 20 / 4 / 20 / 356 | 1.00 | 0.00 |
| PCA | 0.781181 | 0.622672 | 0.833333 | 0.500 | 20 / 4 / 20 / 356 | 1.00 | 0.00 |
| Isolation Forest | 0.997639 | 0.983430 | 0.928571 | 0.975 | 39 / 3 / 1 / 357 | 1.00 | 0.95 |

Thresholds were respectively 0.3829325867, 0.3808201272 and 0.6209075244;
their units differ. Each produced 7/800 calibration alerts. Test false-positive
rates were 4/360 (1.111%) for AE/PCA and 3/360 (0.833%) for Isolation Forest.
Alert totals were 24, 24 and 42. The linear script and nonlinear study use
different matrices and dimensions; their MSE values must not be compared as
a linear-versus-nonlinear model ranking.

**Interpretation candidates — author review required.** The linear fit approached
the expected training optimum without beating it. In the detection study,
both reconstruction methods missed the injected in-subspace rarity despite
detecting every off-subspace shift. Isolation Forest's score detected most
cases of both mechanisms under this generator. These observations support
examining residual geometry separately from rarity and subgroup recall
separately from aggregate AUC; they do not establish general detector superiority.

**Limitations.** One seed, only 20 examples per anomaly type, linear synthetic
normal structure, large predeclared shifts and fixed untuned configurations
make this a controlled mechanism demonstration. There are no uncertainty
intervals, real operational labels, drift study, latency measurements or
production evidence. The test mixture and score thresholds do not establish
future alert precision or a guaranteed false-positive rate.

**Validation of this delivery.** The scoped repository test runner passed
30 cases in one isolated file (10.97 seconds), with no skips or warnings.
Coverage includes finite differences and float64 autograd, the PCA bound,
latent reparameterization, score/quantile boundaries, deterministic training,
invalid inputs and test-data perturbation through the actual study pipeline.
Repository-wide syntax and internal-path checks passed; they do not render
GitHub mathematics or establish scientific interpretation review.

The local PowerShell launcher was blocked by the shell's execution policy,
so its Python runner was invoked directly. `python -m pip check` was unavailable
because this environment has no pip module. The read-only alternative
`uv --no-cache pip check --python .local/environments/n09-full-2026-11-13/Scripts/python.exe`
checked 78 packages and reported compatibility. No environment or execution
policy change was made. The full curriculum suite and CI were not run.

## Practical limits and proposed extensions

Training MSE alone cannot validate detection. Overcapacity, training contamination,
legitimate rare regimes and distribution shift can all break the reconstruction
assumption. Feature scaling can amplify noisy low-variance features; equal
standardized weights do not imply equal operational importance. A deterministic
seed aids replay on this environment but does not promise identical values
across library versions, hardware or thread settings.

For an actual telemetry or document-quality review queue, first define what
constitutes an actionable event and compare rules or supervised models if
labels exist. Keep a versioned feature schema, preprocessing, weights and
threshold together; validate temporal/group splits and monitor reviewed alerts,
missingness and regime shifts. No cloud service or deployment is implemented.

**Analytical follow-ups, not executed under this protocol:** repeat across seeds; vary bottleneck size; replace
linear normals with nonlinear structure; inject training contamination; compare
predeclared calibration quantiles; evaluate realistic chronological data and
cost/alert budgets. Each new run needs its own dated configuration and result.
Do not select a better configuration on this published test and continue to
call it an untouched evaluation.

The separate [visual laboratory](VISUAL_GUIDE.md) implements initialization-seed
bottleneck comparisons, a curved 3D generator, a capacity/contamination control
and the three calibration quantiles under its own configuration. Those runs
do not replace this dated analytical record or execute the analytical follow-up
protocol on its original data. The visual guide records their measurements
and limitations separately.

## Interview questions

**Why can PCA and a linear AE agree while their weights differ?** They minimize
the same centered rank-constrained squared reconstruction objective at a global
optimum, but latent invertible transformations leave reconstruction unchanged.

**Why can excellent reconstruction hide anomalies?** The objective rewards
reconstructing inputs, including unusual inputs to which the mapping generalizes.
Rare points in a learned subspace can have tiny residuals. A residual is not
a probability of fraud, failure or abnormality.

**Why not tune the threshold on the test?** That reuses final evaluation labels
for a decision, giving an optimistic assessment. Calibrate beforehand; use
another development set for architecture or supervised threshold selection.

**When is a nonlinear AE justified?** When relevant nonlinear structure and
held-out operational metrics demonstrate value beyond simpler methods, within
training, inference and maintenance constraints. This synthetic linear example
cannot establish that case.

**What would you say in an interview?** An autoencoder learns an encoder and
decoder through reconstruction. A compact representation can be useful without
external targets, but detection requires a separate assumption about residuals.
I would freeze preprocessing and the model, calibrate thresholds independently,
evaluate representative anomaly types and false-alert volume, and compare
simple baselines before making an operational claim.

# Interview questions: feature processing and depth

1. **Does a position-wise FFN ignore context?**
   It does not mix positions itself. Its input can already be contextualized
   by attention. The same nonlinear map is shared across tokens within a layer.

2. **Why expand and project back?**
   Expansion supplies intermediate features, the activation makes their
   composition nonlinear and projection restores the residual width. Without
   activation, two affine maps collapse into one; width alone adds no nonlinearity.

3. **Why do residuals help without guaranteeing stable gradients?**
   A residual branch has Jacobian $I+J_F$. The identity gives a direct
   contribution, but $J_F$ can amplify or cancel it. Products across depth
   still depend on initialization, norms, scaling and the objective.

4. **What exactly does LayerNorm normalize for shape $(B,T,d)$?**
   With `LayerNorm(d)`, each token's $d$ features independently, using
   population variance. It has no batch-running statistics. Affine parameters
   can change the standardized moments, and epsilon prevents exact unit variance.

5. **Can LayerNorm reconstruct what it removed?**
   Scalar shifts are lost. Positive scale is approximately normalized, with
   epsilon-dependent differences. Fixed learned scale/bias cannot recover
   arbitrary per-token means. A residual route around the norm can carry
   additional information, but does not guarantee invertibility of the block.

6. **How does Pre-LN change backpropagation?**
   It gives $I+J_FJ_{\mathrm{LN}}$ instead of
   $J_{\mathrm{LN}}(I+J_F)$. The direct identity term lies outside the norm.
   Also state the final readout: a final LayerNorm changes the loss boundary.

7. **Does the smaller random-target MSE identify the better architecture?**
   No. Here output scale affects MSE and no task has been learned. The
   [execution record](notes.md#execution-record--2026-11-25) is a propagation
   diagnostic. A quality comparison needs training, validation selection,
   independent evaluation and comparable tuning budgets.

8. **How would you investigate unstable fine-tuning?**
   Check finite losses, activation and parameter-gradient profiles, precision,
   learning rate/warmup, norm placement, initialization and residual scale.
   Verify the data and loss masking too. Change one controlled factor at a
   time; clipping alone may hide an underlying problem. This study does not
   execute that fine-tuning procedure.

9. **Why adjust width when replacing a dense FFN with SwiGLU?**
   The gate adds a third projection. Ignoring biases, match $3dh_g$ to
   $2dh_f$; a dense $h_f=4d$ corresponds to $h_g\approx8d/3$. Equal parameter
   counts still do not imply equal latency or learned quality.

10. **When could the FFN dominate compute or memory?**
    Dense FFNs can account for many block parameters and operations at short
    sequence lengths. Dense attention adds a quadratic sequence term.
    Decoding with KV caching changes the workload. Measure latency, bandwidth
    and memory on the intended device rather than treating asymptotic counts
    as timings.

An interview explanation should connect token mixing, position-wise feature
processing, the additive path and normalization axes, then state what evidence
would be needed to choose among architectural variants for an actual task.

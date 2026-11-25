# Multi-head attention: interview questions

1. **Why several heads instead of one wider head?** Each head can supply a
   different distribution over keys and a different weighted value combination.
   At fixed model width this trades per-head width for multiple routing patterns.
   It does not guarantee a better trained model for every task.

2. **Are heads fixed embedding slices?** No. Each projects all input features;
   packed output columns are reshaped into heads after projection. The learned
   spaces can overlap and need not be orthogonal or statistically independent.

3. **Explain the shapes for B=2, T=6, D=32, H=4.** Projected Q/K/V are
   `(2,6,32)`, split tensors `(2,4,6,8)`, maps `(2,4,6,6)` and head outputs
   `(2,4,6,8)`. Transpose then concatenate to `(2,6,32)` before output projection.

4. **Why scale by the square root of eight in that example?** Eight is the
   query/key head width. Under independent centered unit-variance components,
   dot-product variance grows with that width. Those assumptions are an
   initialization intuition, not a guarantee for trained representations.

5. **Does doubling heads double parameters?** With fixed standard width,
   projection parameters stay at $4D^2$ without biases. Explicit map entries
   double while head width halves. Leading attention arithmetic stays
   $O(BT^2D)$; actual runtime still needs measurement.

6. **Does standard MHA average heads?** It concatenates their outputs and
   applies $W^O$. Averaging returned maps is an inspection option, separate from
   the forward calculation. Inspecting averaged maps can hide differences.

7. **How would you detect a causal-mask bug?** Perturb future input tokens
   and compare prefix outputs with fixed weights. The prefix should stay
   invariant under the causal mask; use bidirectional attention as a control.
   Also check exact zeros, row sums and parity with a library reference.

8. **What does a padding mask do to padded queries?** A key padding mask
   excludes keys, not queries. Padded query outputs still exist; exclude them
   from downstream losses as needed. Combined masks may fully block a row,
   which requires an explicit policy. This implementation raises an error.

9. **Can identical maps justify pruning heads?** No. Their value projections
   and output blocks may differ. Evaluate controlled interventions on a trained
   model and measure task quality and real efficiency before choosing a policy.

10. **How do GQA and FlashAttention differ?** GQA shares K/V projections across
    groups of query heads, reducing theoretical cache payload. FlashAttention
    changes how dense attention executes to reduce memory traffic. Neither
    automatically establishes a quality or latency result for a deployment.

11. **What does NumPy/PyTorch agreement establish?** Agreement on the tested
    inputs, weights, masks and precision. It does not validate all inputs,
    trainability, optimized kernels, language understanding or production use.

12. **How would you explain this in a minute?** Multi-head attention projects
    each token into several query/key/value spaces, computes a distribution over
    keys for each query in each head, retrieves weighted values, and mixes the
    concatenated outputs. This supports multiple routing patterns at a fixed
    representation width. Correct axes, head-width scaling and masks matter
    as much as the formula. Quality, interpretability and serving efficiency
    each need their own evidence.

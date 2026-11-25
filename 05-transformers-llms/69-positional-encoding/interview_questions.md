# Interview questions — positional information

1. **Is self-attention permutation invariant?**
   Unmasked content-only self-attention is equivariant: $F(PX)=PF(X)$.
   Outputs follow the token permutation. Symmetric pooling can then make a
   sequence-level representation invariant. See the derivation in [notes](notes.md).

2. **Does a causal mask make positional mechanisms unnecessary?**
   It introduces asymmetric visibility and can provide implicit order signals.
   It does not explicitly encode arbitrary distances. The included fixed-mask
   control breaks general permutation symmetry without positional vectors;
   conclusions about trained task quality need a separate evaluation.

3. **What exactly changes in the additive-position permutation experiment?**
   Tokens move while coordinates remain assigned to fixed slots. Permuting both
   together instead preserves the equivariance of the underlying unmasked layer.
   The distinction prevents a misleading demonstration of position sensitivity.

4. **Why use paired sine/cosine frequencies?**
   They represent multiple scales, have constant pair norm and admit a linear
   offset transformation. Pure positional dot products depend on displacement.
   Arbitrary content additions and learned projections do not retain a general
   distance-only attention score.

5. **Why add rather than concatenate?**
   Addition preserves width and projection shapes. Concatenation can work with
   adjusted projections but changes the architecture and parameter budget.
   Addition alone does not guarantee unique recovery of content and position.

6. **How does RoPE turn absolute rotations into relative interactions?**
   Orthogonality gives $(R_pq_p)^\top R_rk_r=q_p^\top R_{r-p}k_r$.
   This holds with a consistent frequency/layout convention. The vectors still
   carry content; the score is not a function of distance alone.

7. **Does RoPE always favor nearby tokens?**
   Individual dot products can oscillate and increase at larger distances.
   ALiBi explicitly adds a monotonic distance penalty with content fixed;
   that is a different mechanism from rotary geometry.

8. **What cache error does the example detect?**
   Cached keys remain rotated with original IDs while new Q/K IDs restart at
   zero. Their relative geometry becomes inconsistent. Correct offsets match
   full one-layer causal attention. This does not validate a full decoding runtime.

9. **Can sinusoidal encoding or RoPE guarantee a larger effective context?**
   No. Computable indices, runtime acceptance and useful task performance are
   distinct. Check evidence-position accuracy, distractor robustness and citations
   under realistic token budgets; do not infer them from a numerical rotation test.

10. **What must be checked before reusing a positional implementation?**
    Pair layout, rotary subset, frequency base/scaling, position IDs, masks and
    checkpoint conventions. For absolute tables, check index bounds and adaptation
    policy. For ALiBi, add the bias after scaling content logits and verify slope
    policy. A pretrained model's positional scheme is part of its learned setup.

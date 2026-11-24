# Attention interview questions

**1. Explain attention without conflating keys and values.**
Queries and keys produce compatibility scores; row-wise softmax converts
scaled scores over permitted keys into weights. Those weights mix values
into an output vector. Matching features and transferred information need
not be the same, which motivates separate trainable projections.

**2. Why divide by $\sqrt{d_k}$?**
Under independent centered unit-variance coordinates, dot-product variance
grows as $d_k$. The scaling gives unit variance under those assumptions and
helps avoid excessively concentrated softmax. Learned features need not
satisfy those assumptions; it is not a gradient guarantee.

**3. How would you test the operation?**
Use a hand-computable case, row sums and exact masked zeros, unequal query/key
lengths and value widths, value-only interventions and causal future changes.
Check permutation equivariance when positional information is absent, invalid
inputs and agreement with a trusted implementation. Production use would also
require gradient, dtype, hardware and workload-specific checks absent here.

**4. What happens to a fully masked query?**
Ordinary softmax on all negative infinities is undefined. Define the API policy
explicitly. This reference rejects the row. If a system chooses a zero-output
convention, verify that downstream losses and residual paths handle it correctly.

**5. Does a high attention weight explain the prediction?**
It describes one aggregation coefficient. Value magnitudes/directions and
subsequent computation can change the final effect. It is not automatically
causal importance or confidence; an explanation needs additional evidence.

**6. Does self-attention encode order by itself?**
Unmasked attention without positional information is permutation-equivariant.
Position-dependent masks and positional representations change that symmetry.
The causal mask imposes visibility order; it does not provide every positional
feature a sequence model may need.

**7. How do self- and cross-attention differ?**
Self-attention derives Q/K/V from one sequence. Cross-attention derives queries
from one source and keys/values from another. The latter produces an n-by-m
weight matrix and n-by-$d_v$ output; sequence lengths need not match.

**8. What can go wrong when porting masks to PyTorch?**
Boolean True permits an interaction in functional SDPA, while True blocks
padding in `MultiheadAttention.key_padding_mask`. SDPA dropout must be disabled
explicitly for deterministic comparison. An offset decoding query also needs
an appropriate prefix mask rather than a naive rectangular lower triangle.
See [the versioned documentation](references.md).

**9. What are the long-context costs, and does FlashAttention remove them?**
Dense pair interactions have quadratic self-attention arithmetic. Conventional
implementations also store quadratic scores/weights. FlashAttention reduces
memory traffic and avoids full matrix materialization in high-bandwidth memory;
it does not make dense pair arithmetic linear. KV caching avoids recomputing
earlier K/V during decoding and adds context-dependent memory requirements.

**10. How is attention different from retrieval in RAG?**
Retrieval selects evidence from an external corpus. Attention aggregates
representations inside the model over available context. More supplied tokens
can increase cost without guaranteeing better answers. Evaluate evidence
coverage and answer quality separately from numerical attention correctness.

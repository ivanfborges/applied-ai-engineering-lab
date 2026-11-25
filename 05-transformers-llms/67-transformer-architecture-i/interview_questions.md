# Transformer architecture — interview questions

1. **What turns attention into a Transformer?**
   Attention routes information between positions. The architecture adds token
   and position representations, repeated attention/FFN sublayers, residuals,
   normalization and a task head. This study exposes those boundaries explicitly.

2. **Where do cross-attention queries, keys and values come from?**
   Queries come from decoder states after causal self-attention; keys and values
   come from final encoder memory. Target and source lengths need not agree.
   Each decoder layer has its own projections over that memory.

3. **Why are shifting and causal masking both required?**
   Shifting puts previously available tokens at the prediction position. Masking
   prevents access to subsequent inputs containing future labels. A mask alone
   cannot repair an unshifted label copied through the current-position residual.

4. **What numerical test would expose future-token leakage?**
   Keep source and earlier decoder inputs fixed, change only future inputs and
   compare earlier logits in evaluation mode. They should remain close with a
   causal mask. An unmasked control should respond, showing the test is sensitive.
   Also inspect the actual input/label shift and mask polarity.

5. **Is Transformer generation parallel?**
   Teacher-forced positions are available together, enabling parallel training.
   Standard autoregressive generation consumes its own prefix one token at a
   time. Caching avoids recomputing historical keys/values but does not remove
   that dependency. This example caches encoder memory only.

6. **Why measure token accuracy and exact match under free-running decoding?**
   A single wrong token invalidates exact reversal. Teacher forcing supplies
   correct prefixes; generation can propagate its own mistakes. Both metrics
   here include EOS and use a fixed horizon, which must be stated before comparison.

7. **Does self-attention know order without positional inputs?**
   Unrestricted content-only self-attention is permutation-equivariant. Positional
   features provide explicit order signals. A causal mask also imposes an
   order-dependent graph, so a blanket claim about all masked architectures
   would be too strong. Length extrapolation is a separate empirical question.

8. **How would you extend this study to variable-length text?**
   Introduce tokenization, source/target padding and memory masks, loss masking,
   EOS-aware decoding and length-aware evaluation. Then evaluate on separate
   validation/test data representative of the intended language task. Reversal
   accuracy does not establish translation or summarization quality.

9. **How do post-norm and pre-norm differ?**
   Post-norm applies normalization after residual addition; pre-norm normalizes
   the branch input and leaves a direct residual path. The placement changes
   optimization behavior. This implementation uses post-norm and reports no
   empirical superiority over pre-norm.

10. **Does RAG require encoder-decoder cross-attention?**
    No. Retrieved passages can be ordinary prompt tokens in a decoder-only model.
    Retrieval chooses external evidence; attention processes supplied representations.
    Grounded answer quality requires its own evaluation.

An interview-ready explanation: the original Transformer encodes a fully
available source and decodes a target conditioned on that source and its previous
tokens. Attention mixes positions, FFNs transform features, and residuals and
normalization support repeated composition. Correct training requires shifted
targets and causal visibility. Parallel teacher forcing and sequential generation
have different information available, so both must be evaluated deliberately.

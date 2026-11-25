# Interview questions

1. **What distinguishes the three families?**
   An encoder builds bidirectional input states. A standard causal decoder
   builds prefix states without separate encoder memory. An encoder-decoder
   adds a source stack and cross-attention to a causal target stack. Training
   targets determine what those states become useful for.

2. **Is masked language modeling a triangular attention mask?**
   No. MLM corrupts selected input tokens and predicts their original IDs;
   surrounding valid positions can remain bidirectionally visible. A causal
   mask restricts which positions can exchange information.

3. **Why do causal masking and shifting both matter?**
   Shifting makes the current input the preceding target. Causality then hides
   later inputs containing the current and future labels. Without shifting,
   even a residual connection can expose the answer. Some APIs shift labels
   internally, so inspect their loss contract.

4. **Can teacher-forced training be parallel?**
   Yes. The true prefixes are already available for all positions. Generating
   an unknown continuation ordinarily requires sequential token decisions;
   teacher-forced scores alone do not validate that behavior.

5. **Why can a decoder read the entire source without violating causality?**
   Causality is with respect to the target being generated. An already
   available source is conditioning information. If the application receives
   source tokens incrementally, its availability constraints must be encoded.

6. **Is a cross-encoder reranker an encoder-decoder?**
   Usually it is an encoder scoring concatenated query-document tokens with
   joint self-attention. Cross-attention instead takes queries from one stream
   and keys/values from another. A bi-encoder encodes items independently,
   enabling document precomputation at the cost of less joint interaction.

7. **What would you evaluate before choosing a family for RAG?**
   Evaluate retrieval recall and ranking on relevance judgments, grounded
   answer quality and citation correctness, plus latency and memory under the
   intended workload. Embedding training, candidate counts and decoding
   policy matter. Architecture alone supplies no quality-cost ranking.

8. **How would you detect a causal or padding bug?**
   Perturb a future target and compare earlier outputs. Perturb a padded
   source and check both valid source states and target outputs. Assert zero
   blocked probabilities and include a visible perturbation that changes an
   output. Tests in [test_architectures.py](tests/test_architectures.py) exercise
   those controls; positive effects are fixture-specific.

9. **Does BERT automatically supply retrieval embeddings?**
   Hidden states need a pooling policy and a suitable training/evaluation
   setup. Neither bidirectionality nor arbitrary mean pooling establishes
   retrieval quality. Causal decoders can also be adapted to representation
   tasks; model families are not exclusive task labels.

10. **What does the KV cache change?**
    It avoids recomputing prior keys/values during decoding, trading memory
    for computation. It does not make new token choices independent or remove
    attention over the prefix. Source memory and projected cross-attention
    keys/values can also be reused in an encoder-decoder. This topic does not
    implement or benchmark caching.

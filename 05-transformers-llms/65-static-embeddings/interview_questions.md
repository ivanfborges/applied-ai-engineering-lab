# Static embeddings: interview questions

1. **What makes CBOW different from Skip-gram?** CBOW aggregates context
   occurrences and predicts the center. Skip-gram predicts each context
   occurrence from the center. Their targets and event counts differ. CBOW's
   mean loses order; a center occurrence can yield several Skip-gram events.

2. **Why use two embedding tables?** Input vectors encode the conditioning
   side and output vectors score targets. A prediction dot product crosses
   these roles; cosine between two input vectors answers a different question.
   This study keeps input vectors for predictive models and sums both GloVe
   tables. That choice affects comparisons.

3. **Is negative sampling just a faster exact softmax?** It replaces normalized
   categorical likelihood with binary observed/noise discrimination. Its cost
   depends on the number of noise samples, and its optimum depends on the noise
   distribution. The executable learners here use full softmax.

4. **Under what assumptions does SGNS recover shifted PMI?** With independent,
   unconstrained dot products and unigram noise, the stationary score is
   $\operatorname{PMI}(w,c)-\log K$ for observed positive-probability pairs.
   Finite dimensions share parameters, and smoothed noise changes the expression.

5. **What does GloVe reconstruct?** Positive word-context log counts, with
   frequency-dependent weights and two bias terms. Counts aggregate local
   windows across the corpus. Zero cells are omitted; they are not targets with
   a finite logarithm. The example uses mean loss and batch gradient descent.

6. **Does a tiny reconstruction loss establish semantic quality?** No. In the
   [recorded synthetic run](notes.md#execution-record--2026-11-24), GloVe's first
   `doctor` neighbor was `vehicle`. Evaluate prespecified relations and
   downstream tasks; observed count fitting alone leaves semantic gaps.

7. **What exactly does a vector analogy test?** Whether a chosen offset and
   ranking convention recovers a candidate in the current vocabulary. Exclude
   source words, specify normalization, report coverage and test many relations.
   One success does not establish reasoning; the arithmetic test uses constructed
   vectors and has a different evidential role.

8. **Can a cosine of 0.9 justify synonym expansion?** It measures direction,
   not a probability of equivalence. Antonyms and topical associates can share
   contexts. Validate expansion against representative relevance judgments and
   inspect harmful substitutions before applying it in a domain workflow.

9. **How do polysemy, OOV and document pooling affect the design?** A static
   table supplies one row per entry, so it cannot select a `bank` sense. An OOV
   policy may reject, use an unknown row or use subword parameters. Mean pooling
   loses word order and can obscure negation; it does not make a word learner a
   retrieval-trained sentence encoder.

10. **What would a model migration require?** Keep vocabulary, normalization,
    embedding version and index consistent. Independently trained spaces can
    differ; coordinate comparisons need an explicit alignment assumption.
    Re-embed documents when necessary and evaluate downstream relevance,
    coverage, latency, memory and failure cases before switching a live system.

An interview explanation should connect distributional assumptions, the two
prediction directions, the reconstruction alternative and the gap between
training fit and task quality. The [notes](notes.md) contain the derivations;
none of the interpretation candidates has been approved on the author's behalf.

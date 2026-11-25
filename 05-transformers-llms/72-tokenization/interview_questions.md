# Tokenization interview questions

1. **Why subwords instead of words or characters?** They reuse fragments while
   controlling vocabulary size and sequence length. This is a distribution- and
   model-dependent compromise; coverage and semantic quality need separate checks.

2. **How do BPE training and inference differ?** Training repeatedly selects
   corpus-frequency merges. Inference applies stored priorities. The tests use
   conflicting `bc`/`ab` merges to distinguish BPE from longest-vocabulary lookup.

3. **How is WordPiece different?** BERT's inference uses greedy longest matching
   and continuation pieces. A failed segmentation can make the whole word unknown.
   The familiar association-score training formula is a reconstruction, not an
   exact claim about the original trainer.

4. **Is SentencePiece an alternative to BPE?** It is a framework supporting BPE
   and Unigram. Raw-sentence training, normalization, whitespace handling, fallback
   and sampling are separate aspects of a configuration.

5. **Can we change a pretrained model's tokenizer to reduce cost?** IDs index
   trained input/output weights. Replacing that mapping requires adaptation and
   training, and is not an independent hosted-model setting. Compare existing
   models on quality and usage rather than assigning one model's tokenizer to another.

6. **What does a larger vocabulary cost?** Embeddings use $Vd$ weights; untied
   output projections add roughly another $Vd$. Tying shares weights without
   removing vocabulary-sized logits. Sequence compression and rare-piece learning
   must also be considered; there is no universal best vocabulary size.

7. **Does a 2x token reduction guarantee 4x faster inference?** Dense prefill
   attention has quadratic score positions, but kernels, other layers, output
   projection, batch padding and cached decoding affect the total. Measure the
   actual serving workload; this study measures no latency.

8. **Why not independently decode token chunks?** Byte tokens can split a UTF-8
   character. Reconstruct the complete byte stream or use a suitable incremental
   decoder. Avoid corrupting text through replacement decoding at arbitrary splits.

9. **How would you budget a RAG request?** Enforce embedding input limits at
   ingestion. Count the final generator serialization including history, tools
   and delimiters; reserve output and obey all caps. Independently tokenized
   components may not add to the token count of their concatenation.

10. **How would you compare Portuguese and English efficiency?** Use representative
    parallel content and document code-point, byte and word-segmentation choices.
    A few unrelated strings cannot establish language fairness or answer quality.

11. **How should a cost estimate be used?** Use explicit, dated rates and provider
    billing categories, perform admission checks, then reconcile against returned
    usage. The example's USD 0.02 is hypothetical arithmetic, not an observed bill.

12. **Can tokenizer training leak evaluation data?** Yes. The learned vocabulary
    reflects corpus statistics even without labels. Keep custom-tokenizer training
    within the training partition for a controlled held-out comparison.

An interview-ready explanation: tokenization defines the categorical units a
model was trained to consume. Its vocabulary, segmentation and normalization
affect sequence length and information preservation. In an application, count
with the appropriate model interface, protect source text, reserve context and
evaluate quality alongside cost.

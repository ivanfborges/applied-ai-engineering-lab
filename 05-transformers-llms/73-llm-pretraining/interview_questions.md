# Interview questions: pretraining decisions

**Why call this self-supervised rather than unlabeled learning?**
Text supplies explicit targets: the next observed token for CLM, or original
IDs at selected corrupted positions for MLM. Target construction is part of the
method and can introduce leakage even without human labels.

**Why are both a target shift and a causal mask necessary?**
The shift aligns each input position with the next observed token. A causal
mask prevents that next token, present at a later input position, from becoming
visible. Test both alignment and prefix invariance; a decreasing loss tests neither.

**Why can a pretrained model be wrong despite low perplexity?**
Likelihood rewards predicting observed text, including plausible errors and
contradictions. It neither verifies truth nor guarantees attribution, calibration
or instruction following. Evaluate the intended behavior and evidence use separately.

**Can MLM and CLM perplexity be compared?**
Ordinary autoregressive perplexity exponentiates mean next-token NLL under a
specified tokenizer and context policy. MLM scores selected reconstruction
positions with different available information. Its exponentiated loss is not
the same sequence-likelihood metric. Different tokenizers also change the unit.

**Why does the toy CLM stop near 0.497 rather than zero training loss?**
Identical permitted prefixes lead to multiple valid targets. The empirical
conditional entropy is $(\log4+\log3)/5$. A deterministic tie-breaking argmax
cannot exceed the corresponding expected training accuracy of about 71.67%.
Uncertainty can be irreducible under the objective and observed distribution.

**What does Chinchilla change about choosing a model size?**
It motivates allocating training compute to both parameters and tokens within
the studied regime. A historical allocation is not a universal prescription.
Data repetition, data quality, context, optimization and expected inference
volume affect the useful operating point. This lab fits no scaling exponents.

**How would you audit pretraining validation leakage?**
Trace source provenance and split related documents before overlapping windows;
check exact and approximate duplicates and benchmark overlap. Confirm target,
attention, padding and packing masks. A disjoint sentence split alone does not
exclude all shared structure or guarantee independent evaluation.

**When would you investigate continued pretraining, SFT or retrieval?**
Diagnose the failure first. Domain-distribution adaptation can motivate continued
language modeling; desired task behavior can motivate supervised examples;
current, attributable evidence can motivate retrieval. These address different
needs and may be combined. Establish a baseline and task evaluation before
incurring training cost. None automatically guarantees factual reliability.

**What is missing from a `6ND` infrastructure estimate?**
Attention costs, activations, optimizer/gradient memory, communication,
recomputation, data throughput and failure recovery. Use a representative pilot
to measure sustained throughput, peak memory and checkpoint behavior. Peak
device specifications and a theoretical operation count do not establish wall time.

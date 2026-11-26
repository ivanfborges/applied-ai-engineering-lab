# Interview questions: choosing and validating adaptation

1. **How does instruction tuning differ from SFT?** SFT specifies supervised
   likelihood optimization. Instruction tuning organizes demonstrations around
   requests, often across tasks. Unseen-instruction generalization needs its
   own held-out evaluation; this category selector does not establish it.

2. **Can SFT, DPO and LoRA be combined?** Yes. SFT and DPO define objectives;
   LoRA defines which weight updates are possible. Here the same low-rank
   matrices continue from SFT into DPO, with a copied SFT reference.

3. **What does traditional RLHF add?** A preference-trained reward model and
   policy optimization, often PPO, using generated responses. Offline DPO
   instead fits policy log-ratios directly from preference pairs under a
   preference model, without an explicit reward-model training stage.

4. **Why is the reference a copy rather than an alias?** An alias would change
   when the policy updates, removing the intended baseline. Freeze all its
   parameters and exclude it from the optimizer; `eval()` alone does not freeze
   weights. Cache reference scores only while weights and scoring inputs stay fixed.

5. **Why does one zero LoRA matrix work but two zero matrices stall?** With
   $\Delta W=(\alpha/r)BA$, random A and zero B preserve initial outputs while
   allowing a first B gradient. If both are zero, both first gradients vanish.

6. **Does a lower DPO loss imply more probability on the chosen response?**
   A lower per-pair loss means a higher reference-adjusted ratio for that pair
   at fixed beta. A lower batch mean can hide worsening individual pairs.
   Neither establishes higher absolute probabilities or retained capabilities.
   Inspect probability, full argmax accuracy, margins and regressions separately.

7. **Does LoRA guarantee lower memory and better retention?** It reduces
   trainable counts for suitable ranks, but the base, activations and reference
   still consume resources. Retention and quality remain empirical. The toy
   counts 12 trainable entries but measures neither memory savings nor broad retention.

8. **How do adapters and QLoRA differ?** Bottleneck adapters insert small
   trainable modules. LoRA adds low-rank linear weight updates, which can be
   merged in compatible floating-point layers. QLoRA trains adapters over a
   quantized frozen base; quantization changes storage and numerical constraints.

9. **When would you fine-tune a RAG system?** First locate failures in retrieval,
   context assembly, grounding, prompting and application logic. Consider
   adaptation for a persistent measurable behavior gap with representative
   training data, a held-out baseline and acceptable regression/cost limits.

10. **What must a real acceptance plan include?** A fixed evaluation population,
    task metrics, blinded preferences where appropriate, capability/safety
    regressions, provenance, serving compatibility and rollback criteria.
    Passing numerical tests does not approve deployment or author conclusions.

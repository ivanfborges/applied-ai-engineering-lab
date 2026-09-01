# References

## Foundations

- Claude E. Shannon, [A Mathematical Theory of
  Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf),
  *Bell System Technical Journal*, 1948. Introduces the information and
  entropy framework.
- Solomon Kullback and Richard A. Leibler, [On Information and
  Sufficiency](https://doi.org/10.1214/aoms/1177729694), *The Annals of
  Mathematical Statistics*, 1951. Original source for the directed divergence.
- Thomas M. Cover and Joy A. Thomas, [Elements of Information
  Theory](https://onlinelibrary.wiley.com/doi/book/10.1002/047174882X), second
  edition, Wiley, 2006. Standard reference for entropy, coding, KL divergence,
  and related results.

## Machine-learning implementations

- PyTorch, [`CrossEntropyLoss`
  documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).
  Documents the logits interface, index and probability targets, reductions,
  class weights, and ignored indices.
- scikit-learn, [`log_loss`
  documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html).
  Defines classification log loss from predicted probabilities and describes
  its relationship to logistic-model NLL.
- Hugging Face Transformers, [Perplexity of fixed-length
  models](https://huggingface.co/docs/transformers/main/perplexity). Explains
  exponentiated average NLL, tokenization sensitivity, context-window
  treatment, and target masking.

## Scope

The NumPy implementation in this topic is intentionally educational. Consult a
framework's current documentation for production input shapes, weighting,
ignored-target, reduction, device, and numerical-precision behavior.

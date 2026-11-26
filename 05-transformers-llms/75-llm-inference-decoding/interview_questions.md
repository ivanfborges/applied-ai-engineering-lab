# Interview questions

**What distinguishes inference, decoding and stopping?**
Inference produces next-token logits from context. Decoding transforms them
into a selection policy. Stopping controls when to return, so a response can
end through EOS, a delimiter or a length cap with different completeness.

**Does temperature change token ranking or knowledge?**
For positive temperature in exact arithmetic, it preserves ranking and changes
relative probability ratios. It does not update weights or add evidence.
At zero, this implementation explicitly chooses a single argmax; tied maxima
make that different from the positive-temperature limit.

**Why is top-p not a percentage of vocabulary?**
Its threshold concerns cumulative mass. A peaked distribution may need one or
two candidates; a diffuse one may need many. Include the token that crosses
the threshold and renormalize. Filtering order changes which mass is measured.

**Can greedy decoding miss the most probable complete sequence?**
Yes. Choosing the locally best first token can lead to poor later conditionals.
The two-step counterexample in [notes.md](notes.md#model-policy-and-sequence)
has greedy joint probability 0.33 versus an alternative's 0.38. Sequence
likelihood itself also does not imply correctness or useful open-ended text.

**Does a fixed seed or zero temperature guarantee identical hosted responses?**
No. A seed controls a specified random stream; zero temperature controls token
selection given logits. Model revisions, precision, kernels and serving
configuration can change those logits. Record the execution conditions and
test reproducibility within an explicitly bounded scope.

**Why keep sampled IDs when returning text?**
A matched stop may begin inside a fragment and EOS is usually hidden. Token
IDs expose actual selection steps and explain why visible length differs
from consumed budget. Real API usage and billing require provider semantics.

**How would you evaluate a RAG or extraction decoder configuration?**
Choose representative development prompts and fixed model/retrieval versions.
Measure task accuracy, grounding, schema validity, completeness, retries and
latency across relevant seeds. Freeze the choice before held-out evaluation.
Low entropy alone cannot establish factual accuracy; lowering temperature
does not repair missing retrieval evidence.

**What changes when stops are streamed?**
Text already emitted cannot be retracted. Buffer suffixes that might become
stop strings, and account for tokenizer decoding and cross-token matches.
The local loop only matches an accumulated non-streaming fragment string.

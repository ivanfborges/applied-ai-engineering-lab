# Interview questions

**Why distinguish document frequency from corpus term frequency?**
IDF estimates document-level prevalence. A term repeated 100 times in one
document still has DF one. Confusing these counts changes the weighting rule.

**Does disabling smoothing recover classical $\log(N/DF)$?**
Not in scikit-learn: its unsmoothed variant retains the external $+1$.
Document the formula and normalization before comparing implementations.

**When can dot product replace cosine?**
For unit nonzero query and document vectors. Raw count dot product also depends
on vector magnitude. The count baseline here uses cosine to keep that comparison
consistent. Define zero-row behavior separately.

**Is a cosine score of 0.8 an 80% relevance probability?**
No. It describes vector alignment under a particular analyzer, vocabulary and
weighting. A relevance probability or operational threshold needs labeled data
and evaluation for the target population.

**How can TF-IDF leak information without using labels?**
Fitting on supervised test text exposes its vocabulary and term prevalence.
Fit vectorizers inside each training fold. Fitting a searchable document corpus
for retrieval is intentional, but does not permit tuning on held-out judgments.

**How should an all-OOV query behave?**
Detect the zero representation and return no lexical hits with a status that
allows the caller to choose a fallback. Returning arbitrary zero-score top-k
documents misrepresents absence of evidence. Partial-OOV queries still search
using their known terms; monitoring that loss would be a useful extension.

**Does a rare term deserve more trust?**
No. Typos and OCR noise can be rare. High IDF measures corpus distinctiveness,
not correctness or meaning. Consider character features, metadata and labeled
evaluation; these choices also introduce false matches and vocabulary costs.

**How do you scale this retriever?**
The example is a sparse scan followed by sorting positive scores. Large search
systems often use inverted postings, top-k pruning and explicit index refresh
policies. Choose those mechanisms from corpus size, query load, freshness and
measured latency rather than assuming sparse storage solves retrieval cost.

**What would justify adding BM25 or hybrid retrieval?**
A measured gap on representative queries, followed by an evaluation under the
same corpus, judgments, candidate budgets and tuning boundaries. BM25 adds TF
saturation and corpus-relative length handling; dense retrieval may add semantic
matching. Neither guarantees better retrieval or grounded generated answers.

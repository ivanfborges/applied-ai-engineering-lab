# Context objectives and static vector geometry

## What the representation assumes

An embedding table $V\in\mathbb{R}^{N\times d}$ assigns one row to each of $N$
vocabulary entries. A one-hot lookup selects that row. Context supplies the
self-supervised target; no human similarity labels are used in training here.
Words that share contexts may acquire related representations, including
synonyms, antonyms and topical associates. Related usage is weaker evidence
than interchangeability or logical equivalence.

All input sentences in this study are already tokenized lowercase strings.
Vocabulary IDs follow sorted lexical order. Windows are symmetric, fixed-width,
and clipped at sentence boundaries. Repeated word occurrences count separately;
two different positions with the same word can contribute a diagonal
co-occurrence. There is no stop-word filtering, frequency cutoff, dynamic
window, subsampling or phrase detection.

## Two prediction directions

The [Word2Vec architectures](https://arxiv.org/abs/1301.3781) use input vectors
$v_i$ and output vectors $u_j$. For CBOW, let $C_t$ contain context occurrences
around position $t$:

$$
h_t=\frac{1}{|C_t|}\sum_{c\in C_t}v_c,\qquad
p(j\mid C_t)=\frac{\exp(u_j^Th_t)}{\sum_{k=1}^{N}\exp(u_k^Th_t)}.
$$

The implemented objective is mean negative log-likelihood over eligible centers.
The context mean discards order. Repeated context words add repeated weight;
using a set would change the model. Boundary contexts average over their actual
number of occurrences, rather than a padded window size.

Skip-gram instead uses $h=v_w$ to predict each observed context occurrence:

$$
p(c\mid w)=\frac{\exp(u_c^Tv_w)}{\sum_{j=1}^{N}\exp(u_j^Tv_w)},\qquad
L_{SG}=-\frac{1}{M}\sum_{(w,c)}\log p(c\mid w).
$$

Here $M$ counts directed prediction events, including repetitions. CBOW and
Skip-gram therefore have different targets and event denominators, even with
identical data. Their losses are not comparable measures of semantic quality.

For feature matrix $A$, input table $V$, output table $U$ and one-hot targets $Y$:

$$
H=AV,\quad Z=HU^T,\quad D=\frac{\operatorname{softmax}(Z)-Y}{M},\quad
\nabla_U L=D^TH,\quad \nabla_V L=A^TDU.
$$

The code evaluates log-softmax after subtracting each row maximum. It computes
both gradients before changing either table. CBOW rows of $A$ contain context
frequencies divided by context length; Skip-gram rows select a single center.
Only the learned input table is retained for prediction-model cosine search.

## Negative sampling and the PMI connection

[Negative sampling](https://arxiv.org/abs/1310.4546) replaces vocabulary-wide
normalization with positive/noise discrimination. For a positive pair and $K$
noise draws from $q(c)$, its minimization objective is

$$
\ell=-\log\sigma(u_c^Tv_w)-\sum_{k=1}^{K}\log\sigma(-u_{n_k}^Tv_w).
$$

The approximate per-pair scoring cost falls from $O(Nd)$ to $O((K+1)d)$.
This optimizes a different objective from normalized softmax. Noise pairs can
contain semantically related words; a negative label records sampling origin.
The commonly used $q(c)\propto f(c)^{0.75}$ changes the noise distribution.

For positive-probability pairs, independent unconstrained dot-product optima
satisfy

$$
u_c^Tv_w=\log\frac{p(c\mid w)}{Kq(c)}.
$$

When $q(c)=p(c)$, this becomes $\operatorname{PMI}(w,c)-\log K$.
The [matrix-factorization analysis](https://papers.nips.cc/paper_files/paper/2014/hash/b78666971ceae55a8e87efb7cbfd9ad4-Abstract.html)
connects this idealized optimum with SGNS. Low-rank tables share parameters
across pairs, so they need not satisfy every independent optimum. Smoothed
noise also changes the expression. This is background theory; the executable
study uses full softmax and does not test SGNS or PMI factorization.

## Aggregated co-occurrences and GloVe

The count table adds $1/|t-s|$ for each ordered within-window position pair.
For `a b a`, radius two gives $X_{ab}=X_{ba}=2$ and $X_{aa}=1$.
The latter comes from two directed distance-two contributions of $1/2$.

Following the [GloVe reconstruction objective](https://aclanthology.org/D14-1162/),
define $\Omega=\{(i,j):X_{ij}>0\}$:

$$
r_{ij}=v_i^Tu_j+b_i+\tilde b_j-\log X_{ij},\qquad
f(x)=\min\left((x/x_{\max})^\alpha,1\right),
$$

$$
L=\frac{1}{|\Omega|}\sum_{(i,j)\in\Omega}f(X_{ij})r_{ij}^2.
$$

The local gradients are $2f(X_{ij})r_{ij}u_j$ for $v_i$,
$2f(X_{ij})r_{ij}v_i$ for $u_j$, and $2f(X_{ij})r_{ij}$ for each bias,
all divided by $|\Omega|$ for this mean-loss implementation. Zeros are omitted:
they neither produce $\log 0$ nor penalize an unobserved pair's score.
The weighting saturates at one and reduces the influence of small counts.

The example uses $x_{\max}=10$, $\alpha=0.75$, dense float64 arrays and batch
gradient descent. The paper uses AdaGrad and reports other experimental
configurations. This implementation is not a reproduction of those results.
It returns $V+U$ for inspection; that differs from the input-only policy above.
Optimizing observed reconstruction leaves unobserved relationships weakly
constrained, especially on this tiny artificial corpus.

## Similarity, analogies and evaluation boundaries

For nonzero vectors, $\cos(a,b)=a^Tb/(\|a\|_2\|b\|_2)$.
Unit-normalized vectors satisfy $\|a-b\|_2^2=2-2\cos(a,b)$.
The code scales rows before normalizing to handle very large or small finite
magnitudes; it rejects zero and nonfinite vectors. OOV words raise an explicit
error rather than returning a fabricated embedding.

For $a:b::c:?$, search by cosine to $v_b-v_a+v_c$, excluding $a,b,c$.
This study uses raw vectors for the offset before cosine normalization.
Other normalization conventions can change analogy rankings. Exact score ties
use lexical order; asking for more candidates than available returns all
eligible candidates. The analogy unit test uses constructed vectors only to
verify this arithmetic; it is not learned semantic evidence.

Do not compare raw coordinates across independently fitted models. Even a joint
orthogonal rotation changes coordinates while preserving cosine. More generally,
input/output factorizations can preserve their dot products without preserving
input-only cosine geometry. A fitted objective does not uniquely certify a
semantic metric. Version an embedding model with its tokenizer, vocabulary and
index; do not mix vectors across versions without validation and alignment.

Static `bank` cannot switch senses at lookup time. Average word vectors lose
order and can blur negation. Subword models can improve OOV coverage but do not
guarantee meaning for unseen terms. Contextual token states and retrieval-trained
sentence encoders answer different representation questions; neither a vector
index nor a pooling operation supplies missing relevance supervision.

For a real evaluation, hold out representative semantic judgments or downstream
labels and measure coverage, error categories and task metrics. Fit corpus
statistics only on the allowed training text for an inductive protocol.
Specify transductive access explicitly when the full document collection is
available by design. Intrinsic analogy scores alone do not establish retrieval
Recall@K, bias robustness, latency, domain suitability or deployment readiness.

## Execution record — 2026-11-24

New analytical run by Codex at **2026-11-24 16:24:28 UTC** on Windows 10 x64;
Python **3.11.14**, NumPy **2.4.6**. No prior Day 65 run record was replaced.

**Hypothesis:** each learner can reduce its own fitting objective; the deliberately
shared `doctor`/`physician` contexts may give a higher cosine than
`doctor`/`developer`. The analogy answer is inspected without assuming success.

**Configuration:** `example.py` defaults; 32 synthetic sentences, 96 tokens,
31 vocabulary entries, dimension 8, radius 2, seed 65, 400 full-batch updates,
constant learning rate 0.8, float64. Initial tables use independent draws from
$\mathcal{N}(0,0.1^2)$ with the same RNG initialization per learner. Loss history
includes initialization plus all 400 post-update states. CBOW has 96 events;
Skip-gram has 192 directed events. GloVe has 128 positive directed count cells,
inverse-distance weighting, $x_{\max}=10$ and $\alpha=0.75$.
The SHA-256 of compact JSON corpus serialization is
`e666399b290605017f0157ef62bb8702a13f7a425750b323b379ed95971960b6`.

**Observed results** (rounded; objectives have different units):

| Learner | Initial own loss | Final own loss | doctor/physician cosine | doctor/developer cosine | Difference |
|---|---:|---:|---:|---:|---:|
| CBOW, full softmax | 3.440722 | 0.804248 | 0.996281 | 0.488262 | 0.508018 |
| Skip-gram, full softmax | 3.440990 | 1.868886 | 0.993448 | 0.035482 | 0.957966 |
| GloVe, weighted residual | 0.132762 | 0.001938 | 0.498706 | -0.631771 | 1.130477 |

| Learner | doctor neighbors, descending | man:king::woman:? top three | queen rank among 28 candidates |
|---|---|---|---:|
| CBOW | physician, automobile, royal | prince, queen, princess | 2 |
| Skip-gram | physician, crown, royal | queen, prince, princess | 1 |
| GloVe | vehicle, computer, river | bank, princess, car | 12 |

The full-dimensional cosine scores for the first analogy candidate were
0.791416 (`prince`), 0.804291 (`queen`) and 0.759604 (`bank`), respectively.
`bank` neighbors were `borders, money, river` for CBOW;
`borders, river, money` for Skip-gram; and `princess, king, holds` for GloVe.
These lists are reported even where their apparent semantics are poor.

**Interpretation candidates — author review required:**

- All three learners fit their respective objectives in this configuration.
  Their final losses cannot identify a better embedding method.
- All selected margins are positive, but GloVe's nearest `doctor` neighbors
  demonstrate that this one pairwise contrast is insufficient to establish
  neighborhood quality. Small observed-pair residuals leave many relationships
  uncontrolled on the tiny fixture.
- Only Skip-gram returned `queen` first for this single engineered analogy.
  CBOW and GloVe failures remain part of the result. This is not analogy accuracy
  on a test set, a general architecture comparison or evidence of reasoning.
- The one-vector policy fixes both `bank` senses to one row by construction.
  Neighbor inspection does not measure contextual disambiguation accuracy.

**Limitations:** one seed and one artificial corpus; no held-out data,
hyperparameter search, pretrained baseline, relevance judgments, speed benchmark
or contextual encoder. Uniform predictive contexts and distance-weighted GloVe
counts differ; event counts, units and retained tables also differ. Equal update
counts and learning rates do not make optimization budgets or tuning equally
effective. Repeatability within an environment is distinct from robustness
across seeds, platforms or corpora.

### Separate standalone smoke — 2026-11-24

`from_scratch.py` was also executed in the same Python/NumPy environment.
**Hypothesis:** the small GloVe optimization completes with finite loss and fits
its observed log counts. **Configuration:** `ice water cold` and
`steam water hot` (synthetic; six occurrences, five vocabulary entries), with
the same dimension, window, seed, epochs, learning rate and weighting settings
as above. **Result:** initial mean weighted loss 0.01739498; final loss
0.0000030631. **Interpretation candidate — author review required:** optimization
fits this tiny fixture. **Limitation:** this is a separate smoke corpus, with
no semantic or analogy evaluation; its loss must not be substituted into the
32-sentence comparison.

### Automated numerical checks

The focused isolated test file passed **44 cases**, with pytest **9.2.0**, no
skips or warnings. It checks manual counts, every parameter group's finite
differences, simultaneous updates, stable softmax, cosine geometry, ranking
policies and input errors. Its repeatability tests use 80 updates per learner;
they do not claim cross-platform reproducibility. This is topic validation,
not the complete repository test suite, scientific approval or CI evidence.

Two additional full-comparison runs finished at **16:28:37 UTC** and
**16:29:06 UTC** on the same date, using the unchanged default configuration.
Their complete model-result and corpus-metadata dictionaries matched exactly
within that environment. Timestamped JSON reports remain ignored local outputs.
These runs check repeatability, not new hypotheses or robustness across seeds;
the interpretation candidates and limitations above still apply.

## Suggested experiments — not executed

Vary window radius, dimension and random seed without selecting favorable
analogies. Evaluate several prespecified relations and report failed ones.
Compare input-only versus summed GloVe vectors under a fixed evaluation rule.
Train an actual SGNS implementation with an explicit noise distribution.
Evaluate a domain corpus with held-out human judgments and an inductive
downstream task. These are proposals, not findings from the recorded run.

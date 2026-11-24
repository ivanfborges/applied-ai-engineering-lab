"""Deterministic NLP mechanisms and synthetic evaluation, independent of rendering."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import re
import unicodedata

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from from_scratch import document_term_matrix, generate_ngrams, normalize, tokenize


SENTENCES = (
    "Companies were operating without environmental licenses.",
    "I don't think this application was approved.",
    "A fiscalização identificou irregularidades ambientais.",
    "Decreto 47.838/2020.",
    "Unbelievable results!",
    "State-of-the-art models cost 123.45; quizzaciousness!",
)
STRATEGIES = ("whitespace", "regex", "linguistic", "character", "subword", "byte")
RAW = "  Apple apple US us  NÃO avó avô\tCafe\u0301 ① ﬁ 47.838/2020. NOT approved!  "
NORMALIZATIONS = ("lowercase", "casefold", "whitespace", "NFC", "NFD", "NFKC", "NFKD",
                  "accents", "punctuation", "numbers", "stopwords")
# Explicit teaching policy, deliberately containing negation for the ablation.
STOP_WORDS = frozenset({"the", "a", "an", "was", "is", "were", "with", "without", "not", "no", "never"})
NEGATION = frozenset({"not", "no", "never", "without"})
MORPH_WORDS = (
    ("connect", "v"), ("connected", "v"), ("connecting", "v"), ("connection", "n"), ("connections", "n"),
    ("study", "v"), ("studies", "v"), ("studying", "v"), ("studied", "v"),
    ("run", "v"), ("runs", "v"), ("running", "v"), ("ran", "v"),
    ("mouse", "n"), ("mice", "n"), ("be", "v"), ("is", "v"), ("was", "v"), ("were", "v"),
    ("operate", "v"), ("operated", "v"), ("operating", "v"), ("operation", "n"), ("operator", "n"),
    ("saw", "v"), ("saw", "n"), ("universe", "n"), ("university", "n"),
    ("leaves", "n"), ("leaves", "v"),
)
PT_WORDS = ("fiscalização", "fiscalizações", "ambiental", "ambientais", "operar", "operando", "operaram")
CONTRAST_DOCS = (
    "The license was approved.", "The license was not approved.",
    "The application was rejected.", "The application was not rejected.",
    "The company operated with a license.", "The company operated without a license.",
)
# Binary teaching judgments are specified before ranking, not inferred from scores.
CONTRAST_QUERIES = (
    ("license approved", (0,)), ("license not approved", (1,)),
    ("application rejected", (2,)), ("application not rejected", (3,)),
    ("company operated with a license", (4,)), ("company operated without a license", (5,)),
    ("companies operating without licenses", (5,)),
)
TOPIC_DOCS = (
    ("Environment", "Environmental inspection identifies pollution and license violations."),
    ("Environment", "The environmental license was approved for the company."),
    ("Environment", "The environmental license was not approved for the company."),
    ("Machine learning", "Machine learning models learn patterns from training data."),
    ("Machine learning", "Neural network training minimizes prediction loss."),
    ("Machine learning", "Supervised learning classifiers predict document categories."),
    ("Finance", "The bank transfer payment was confirmed on the account."),
    ("Finance", "Financial transactions require invoice payment verification."),
    ("Finance", "The account deposit transaction was rejected by the bank."),
    ("Medical", "The medical report describes patient treatment and diagnosis."),
    ("Medical", "A physician reviewed the clinical examination results."),
    ("Medical", "The hospital document records a patient prescription."),
    ("Administration", "The administrative application was rejected after review."),
    ("Administration", "The administrative application was not rejected after review."),
    ("Administration", "Companies operating without licenses submitted applications."),
)
SPACE_QUERIES = (
    "environmental license approved", "environmental license not approved",
    "machine learning training", "financial payment bank", "medical patient diagnosis",
    "doctor looked at illness findings", "company operated without license",
    "administrative application not rejected",
)
OCR_QUERIES = (
    ("environmental inspection pollution", (0,)), ("neural network training prediction", (4,)),
    ("bank transfer payment account", (6,)), ("medical report patient diagnosis", (9,)),
    ("administrative application review", (12, 13)),
    ("fiscalização ambiental licença", (15,)),
)
OCR_DOCS = tuple(text for _, text in TOPIC_DOCS) + ("A fiscalização ambiental verificou a licença da empresa.",)
REPRESENTATIONS = ("word", "word12", "char", "char_wb")
CONFIGS = (
    ("original unigram", "keep", "none", 1),
    ("drop stopwords", "drop", "none", 1),
    ("preserve negation", "preserve", "none", 1),
    ("Porter stemming", "keep", "porter", 1),
    ("WordNet supplied POS", "keep", "lemma", 1),
    ("original + bigram", "keep", "none", 2),
    ("preserved + bigram", "preserve", "none", 2),
)


class ResourceUnavailable(RuntimeError):
    """A requested optional linguistic dependency or resource is absent."""


@lru_cache(maxsize=1)
def nltk_tools() -> tuple:
    try:
        from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
        from nltk.tokenize import TreebankWordTokenizer, ToktokTokenizer
    except ImportError as exc:
        raise ResourceUnavailable('Install the canonical extra: python -m pip install -e ".[nlp]"') from exc
    return PorterStemmer(), SnowballStemmer("english"), SnowballStemmer("portuguese"), WordNetLemmatizer(), TreebankWordTokenizer(), ToktokTokenizer()


@lru_cache(maxsize=1)
def wordnet_resource():
    nltk_tools()
    import nltk
    from nltk.corpus import wordnet
    local_path = Path(__file__).resolve().parents[3] / ".local" / "nltk_data"
    if str(local_path) not in nltk.data.path:
        nltk.data.path.insert(0, str(local_path))
    try:
        wordnet.ensure_loaded()
    except LookupError as exc:
        raise ResourceUnavailable("WordNet unavailable. Explicit setup: python -m nltk.downloader -d .local/nltk_data wordnet") from exc
    return wordnet


def wordnet_lemma(word: str, pos: str) -> str:
    if pos not in ("n", "v", "a", "r", "s"):
        raise ValueError("WordNet POS must be n, v, a, r or s")
    wordnet_resource()
    return nltk_tools()[3].lemmatize(word, pos)


def wordnet_candidates(word: str, pos: str) -> list[str]:
    """Expose the resource's morphological candidates, not contextual disambiguation."""
    wordnet_lemma(word, pos)
    return nltk_tools()[3]._morphy(word, pos)


@dataclass
class EducationalBPE:
    """Character BPE with frequency merges, lexicographic ties and no word restriction."""

    alphabet: tuple[str, ...]
    merges: list[tuple[str, str]]
    corpus_sha256: str

    @classmethod
    def train(cls, corpus: tuple[str, ...], merge_count: int = 40) -> EducationalBPE:
        if not corpus or any(not isinstance(s, str) for s in corpus):
            raise ValueError("BPE requires a nonempty string corpus")
        if isinstance(merge_count, bool) or not isinstance(merge_count, int) or merge_count < 0:
            raise ValueError("merge_count must be a nonnegative integer")
        rows = [list(s) for s in corpus]
        alphabet = tuple(sorted(set("".join(corpus))))
        merges = []
        for _ in range(merge_count):
            counts = Counter(pair for row in rows for pair in zip(row, row[1:]))
            if not counts or max(counts.values()) < 2:
                break
            pair = min(counts, key=lambda p: (-counts[p], p))
            merges.append(pair)
            updated = []
            for row in rows:
                result, i = [], 0
                while i < len(row):
                    if i + 1 < len(row) and (row[i], row[i + 1]) == pair:
                        result.append(row[i] + row[i + 1]); i += 2
                    else:
                        result.append(row[i]); i += 1
                updated.append(result)
            rows = updated
        digest = hashlib.sha256("\n".join(corpus).encode("utf-8")).hexdigest()
        return cls(alphabet, merges, digest)

    @property
    def vocabulary(self) -> list[str]:
        return sorted(set(self.alphabet) | {a + b for a, b in self.merges} | {"<unk>"})

    def encode(self, text: str) -> list[dict]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        pieces = [{"token": c if c in self.alphabet else "<unk>", "start": i, "end": i + 1}
                  for i, c in enumerate(text)]
        for left, right in self.merges:
            updated, i = [], 0
            while i < len(pieces):
                a = pieces[i]
                if i + 1 < len(pieces) and (a["token"], pieces[i + 1]["token"]) == (left, right):
                    b = pieces[i + 1]
                    updated.append({"token": left + right, "start": a["start"], "end": b["end"]}); i += 2
                else:
                    updated.append(a); i += 1
            pieces = updated
        return pieces


@lru_cache(maxsize=1)
def trained_bpe() -> EducationalBPE:
    return EducationalBPE.train(SENTENCES, 40)


def segment(text: str, strategy: str) -> list[dict]:
    """Return real token labels and source offsets; byte offsets are explicitly UTF-8."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    if strategy == "subword":
        return trained_bpe().encode(text)
    if strategy == "byte":
        return [{"token": f"0x{b:02X}", "start": i, "end": i + 1} for i, b in enumerate(text.encode("utf-8"))]
    if strategy == "character":
        spans = [(i, i + 1) for i in range(len(text))]
    elif strategy == "linguistic":
        if text == SENTENCES[2] or text.startswith("Decreto"):
            tokens = nltk_tools()[5].tokenize(text)
            spans, cursor = [], 0
            for token in tokens:
                start = text.find(token, cursor)
                if start < 0:
                    raise ValueError("Toktok transformed a token; source alignment unavailable")
                spans.append((start, start + len(token))); cursor = start + len(token)
        else:
            spans = list(nltk_tools()[4].span_tokenize(text))
    else:
        pattern = r"\S+" if strategy == "whitespace" else r"[^\W_]+(?:['’][^\W_]+)*"
        spans = [m.span() for m in re.finditer(pattern, text)]
    return [{"token": text[a:b], "start": a, "end": b} for a, b in spans]


def normalization_step(text: str, operation: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if operation in ("NFC", "NFD", "NFKC", "NFKD"):
        return unicodedata.normalize(operation, text)
    if operation == "lowercase": return text.lower()
    if operation == "casefold": return text.casefold()
    if operation == "whitespace": return " ".join(text.split())
    if operation == "accents":
        return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    if operation == "punctuation":
        return "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    if operation == "numbers": return re.sub(r"\d+(?:[.,/]\d+)*", "<NUM>", text)
    if operation == "stopwords":
        return " ".join(t for t in tokenize(text) if t not in STOP_WORDS)
    raise ValueError(f"Unknown normalization operation: {operation}")


def normalization_trace(text: str, operations: list[str]) -> list[dict]:
    trace = [{"operation": "original", "text": text, "length": len(text)}]
    for op in operations:
        text = normalization_step(text, op)
        trace.append({"operation": op, "text": text, "length": len(text)})
    return trace


def codepoints(text: str) -> list[dict]:
    return [{"character": c, "codepoint": f"U+{ord(c):04X}", "name": unicodedata.name(c, "UNNAMED")}
            for c in text]


def morphology_rows(include_wordnet: bool = True) -> list[dict]:
    porter, snowball, portuguese, *_ = nltk_tools()
    rows = []
    for word, pos in MORPH_WORDS:
        row = {"word": word, "pos": pos, "porter": porter.stem(word), "snowball": snowball.stem(word)}
        if include_wordnet:
            row["lemma"] = wordnet_lemma(word, pos)
            row["candidates"] = wordnet_candidates(word, pos)
        rows.append(row)
    rows.extend({"word": w, "pos": "PT stem only", "portuguese": portuguese.stem(w)} for w in PT_WORDS)
    return rows


def synthetic_corpus() -> list[str]:
    """45 deterministic short documents with repeated phrases and unique IDs."""
    endings = ("was approved", "was not approved", "requires manual review")
    return [f"{text} Case {i} {endings[j]}." for i, (_, text) in enumerate(TOPIC_DOCS) for j in range(3)]


def feature_stats(documents: list[str], max_n: int = 1, min_df: int = 1,
                  max_features: int | None = None) -> tuple[dict, sparse.csr_matrix, list[str]]:
    if isinstance(max_n, bool) or not isinstance(max_n, int) or max_n not in (1, 2, 3, 4):
        raise ValueError("max_n must be 1, 2, 3 or 4")
    if isinstance(min_df, bool) or not isinstance(min_df, int) or min_df < 1:
        raise ValueError("min_df must be a positive document count")
    vectorizer = CountVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                                 ngram_range=(1, max_n), min_df=min_df, max_features=max_features)
    matrix = vectorizer.fit_transform(documents).tocsr()
    d, k = matrix.shape
    density = matrix.nnz / (d * k)
    return {"max_n": max_n, "documents": d, "features": k, "shape": [d, k], "nnz": int(matrix.nnz),
            "density": density, "sparsity": 1 - density, "dense_bytes": d * k * matrix.dtype.itemsize,
            "sparse_bytes": matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes}, matrix, vectorizer.get_feature_names_out().tolist()


def tfidf_from_counts(counts: np.ndarray, l2: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(counts, dtype=float)
    if x.ndim != 2 or x.shape[0] == 0 or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("counts must be a finite nonnegative 2D matrix with at least one document")
    df = (x > 0).sum(axis=0)
    idf = np.log((1 + x.shape[0]) / (1 + df)) + 1
    weighted = x * idf
    if l2:
        norms = np.linalg.norm(weighted, axis=1, keepdims=True)
        weighted = np.divide(weighted, norms, out=np.zeros_like(weighted), where=norms > 0)
    return df, idf, weighted


def tfidf_weight(documents: int, df: float | np.ndarray, tf: float | np.ndarray):
    if isinstance(documents, bool) or not isinstance(documents, int) or documents < 1:
        raise ValueError("D must be a positive integer")
    df, tf = np.asarray(df, dtype=float), np.asarray(tf, dtype=float)
    if not np.isfinite(df).all() or not np.isfinite(tf).all() or (df < 1).any() or (df > documents).any() or (df != np.floor(df)).any() or (tf < 0).any():
        raise ValueError("An observed term requires 1 <= df <= D and finite tf >= 0")
    return tf * (np.log((1 + documents) / (1 + df)) + 1)


def cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.ndim != 1 or b.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("vectors must be finite 1D arrays of equal dimension")
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return None if denominator == 0 else float(np.clip(np.dot(a, b) / denominator, -1, 1))


def make_vectorizer(representation: str = "word", **kwargs) -> TfidfVectorizer:
    if representation not in REPRESENTATIONS:
        raise ValueError(f"representation must be one of {REPRESENTATIONS}")
    if representation in ("char", "char_wb"):
        return TfidfVectorizer(analyzer=representation, preprocessor=normalize, lowercase=False,
                               ngram_range=(3, 5), **kwargs)
    return TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                           ngram_range=(1, 2 if representation == "word12" else 1), **kwargs)


def rank(scores: np.ndarray) -> list[int]:
    return sorted((i for i in range(len(scores)) if scores[i] > 0), key=lambda i: (-float(scores[i]), i))


def vector_space(representation: str, query: str, seed: int = 63) -> dict:
    vectorizer = make_vectorizer(representation)
    matrix = vectorizer.fit_transform([text for _, text in TOPIC_DOCS])
    q = vectorizer.transform([query])
    scores = cosine_similarity(q, matrix).ravel()
    svd = TruncatedSVD(n_components=3, random_state=seed)
    coords = svd.fit_transform(matrix)
    return {"documents": coords, "query": svd.transform(q)[0], "scores": scores,
            "ranking": rank(scores), "pairwise": cosine_similarity(matrix),
            "variance": float(svd.explained_variance_ratio_.sum()), "features": matrix.shape[1]}


class NGramLM:
    """Fixed-order Counter model with BOS contexts, EOS prediction and no backoff."""

    BOS, EOS = "<s>", "</s>"

    def __init__(self, sentences: list[list[str]], order: int = 2):
        if isinstance(order, bool) or not isinstance(order, int) or order not in (1, 2, 3):
            raise ValueError("order must be 1, 2 or 3")
        if not sentences or any(not isinstance(s, list) for s in sentences):
            raise ValueError("sentences must be a nonempty list of token lists")
        if any(not isinstance(w, str) or not w or w in (self.BOS, self.EOS) for s in sentences for w in s):
            raise ValueError("tokens must be nonempty strings without reserved boundaries")
        self.order = order
        self.vocabulary = tuple(sorted({w for s in sentences for w in s} | {self.EOS}))
        self.counts: dict[tuple, Counter] = defaultdict(Counter)
        for sentence in sentences:
            padded = [self.BOS] * (order - 1) + sentence + [self.EOS]
            for i in range(order - 1, len(padded)):
                context = tuple(padded[i - order + 1:i]) if order > 1 else ()
                self.counts[context][padded[i]] += 1

    def distribution(self, context: tuple[str, ...], alpha: float = 0) -> np.ndarray:
        if len(context) != self.order - 1:
            raise ValueError("context must have exactly order-1 tokens")
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        counts = self.counts.get(tuple(context), Counter())
        denominator = sum(counts.values()) + alpha * len(self.vocabulary)
        if denominator == 0:
            raise ValueError("unseen unsmoothed context has no defined distribution; use alpha > 0")
        return np.array([(counts[w] + alpha) / denominator for w in self.vocabulary])

    def generate(self, seed: int = 63, alpha: float = 0.1, max_tokens: int = 12) -> list[dict]:
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        rng = np.random.default_rng(seed)
        context = (self.BOS,) * (self.order - 1)
        history, steps = [], []
        for _ in range(max_tokens):
            probabilities = self.distribution(context, alpha)
            selected = str(rng.choice(self.vocabulary, p=probabilities))
            history.append(selected)
            steps.append({"context": list(context), "probabilities": probabilities.tolist(),
                          "selected": selected, "sequence": history.copy()})
            if selected == self.EOS: break
            context = tuple((list(context) + [selected])[-(self.order - 1):]) if self.order > 1 else ()
        return steps


LM_SENTENCES = ("the license was approved", "the license was not approved", "the application was rejected",
                "the application was not rejected", "the company operates", "the company needs a license")


def retrieval_metrics(ranking: list[int], relevant: tuple[int, ...], k: int = 3) -> dict:
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    if not relevant or len(set(ranking)) != len(ranking) or len(set(relevant)) != len(relevant):
        raise ValueError("relevant must be nonempty and both lists must contain unique IDs")
    rel = set(relevant)
    hits = [int(i in rel) for i in ranking[:k]]
    dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
    ideal = sum(1 / math.log2(index + 2) for index in range(min(k, len(rel))))
    first = next((1 / (i + 1) for i, doc in enumerate(ranking) if doc in rel), 0.0)
    return {"recall": sum(hits) / len(rel), "rr": first, "ndcg": dcg / ideal}


# Gold POS is supplied for this tiny synthetic vocabulary, not predicted.
GOLD_POS = {"approved": "v", "rejected": "v", "operated": "v", "operating": "v", "operates": "v",
            "was": "v", "were": "v", "is": "v", "company": "n", "companies": "n",
            "licenses": "n", "license": "n", "application": "n", "applications": "n"}


def analyze(text: str, stop_policy: str = "keep", morphology: str = "none") -> list[str]:
    if stop_policy not in ("keep", "drop", "preserve") or morphology not in ("none", "porter", "lemma"):
        raise ValueError("invalid stop-word policy or morphology")
    tokens = tokenize(text)
    removed = STOP_WORDS if stop_policy == "drop" else STOP_WORDS - NEGATION if stop_policy == "preserve" else set()
    tokens = [t for t in tokens if t not in removed]
    if morphology == "porter": return [nltk_tools()[0].stem(t) for t in tokens]
    if morphology == "lemma":
        return [wordnet_lemma(t, GOLD_POS.get(t, "n")) for t in tokens]
    return tokens


def retrieval_run(stop_policy: str = "keep", morphology: str = "none", max_n: int = 1,
                  query: str | None = None, k: int = 3) -> dict:
    if isinstance(max_n, bool) or not isinstance(max_n, int) or max_n not in (1, 2, 3): raise ValueError("max_n must be 1, 2 or 3")
    def analyzer(text: str) -> list[str]:
        tokens = analyze(text, stop_policy, morphology)
        return [" ".join(g) for n in range(1, max_n + 1) for g in generate_ngrams(tokens, n)]
    vectorizer = TfidfVectorizer(analyzer=analyzer)
    matrix = vectorizer.fit_transform(CONTRAST_DOCS)
    queries = CONTRAST_QUERIES if query is None else ((query, ()),)
    results = []
    for text, relevant in queries:
        scores = cosine_similarity(vectorizer.transform([text]), matrix).ravel()
        ranking = rank(scores)
        item = {"query": text, "relevant": list(relevant), "scores": scores.tolist(), "ranking": ranking,
                "failures": [i for i in relevant if i not in ranking[:k]]}
        if relevant: item["metrics"] = retrieval_metrics(ranking, relevant, k)
        results.append(item)
    means = {key: float(np.mean([r["metrics"][key] for r in results])) for key in ("recall", "rr", "ndcg")} if query is None else {}
    return {"stop_policy": stop_policy, "morphology": morphology, "max_n": max_n,
            "features": len(vectorizer.vocabulary_), "k": k, "results": results, "mean_metrics": means}


def corrupt_text(text: str, rate: float, seed: int = 63) -> str:
    """Edit selected original alphanumeric characters once; rate is selection probability."""
    if not isinstance(text, str): raise TypeError("text must be a string")
    if not np.isfinite(rate) or not 0 <= rate <= 1: raise ValueError("rate must be between 0 and 1")
    rng = np.random.default_rng(seed)
    confusions = {"o": "0", "l": "1", "i": "l", "m": "rn", "c": "e", "a": "o", "e": "c"}
    output = []
    for c in text:
        if not c.isalnum() or rng.random() >= rate:
            output.append(c); continue
        operation = int(rng.integers(0, 5))
        if operation == 0: output.append(str(rng.choice(list("abcdefghijklmnopqrstuvwxyz"))))
        elif operation == 1: continue
        elif operation == 2: output.append(c + str(rng.choice(list("abcdefghijklmnopqrstuvwxyz"))))
        elif operation == 3: output.append(confusions.get(c.casefold(), "?"))
        else:
            plain = normalization_step(c, "accents")
            output.append(plain if plain != c else c + "\u0301")
    return "".join(output)


def ocr_experiment(seed: int = 63, trials: int = 12, rates: tuple[float, ...] = (0, .1, .2, .35, .5)) -> dict:
    if not isinstance(trials, int) or trials < 1: raise ValueError("trials must be positive")
    rows, examples = [], []
    models = {rep: make_vectorizer(rep) for rep in REPRESENTATIONS}
    matrices = {rep: v.fit_transform(OCR_DOCS) for rep, v in models.items()}
    for rate in rates:
        for trial in range(trials):
            # Shared corruptions across representations provide paired comparisons.
            queries = [corrupt_text(q, rate, seed + 1009 * trial + i) for i, (q, _) in enumerate(OCR_QUERIES)]
            for rep, vectorizer in models.items():
                scores = cosine_similarity(vectorizer.transform(queries), matrices[rep])
                metrics = [retrieval_metrics(rank(score), relevant, 3) for score, (_, relevant) in zip(scores, OCR_QUERIES)]
                rows.append({"representation": rep, "rate": rate, "trial": trial,
                             **{key: float(np.mean([m[key] for m in metrics])) for key in ("recall", "rr", "ndcg")}})
                if trial == 0:
                    for i in (0, 5):
                        ranked = rank(scores[i])
                        examples.append({"representation": rep, "rate": rate, "query": queries[i],
                                         "relevant": list(OCR_QUERIES[i][1]), "ranking": ranked,
                                         "top_text": OCR_DOCS[ranked[0]] if ranked else "no match"})
    return {"seed": seed, "trials": trials, "rates": list(rates), "k": 3, "rows": rows, "examples": examples}

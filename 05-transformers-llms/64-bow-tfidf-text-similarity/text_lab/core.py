"""Deterministic fixtures, educational BM25, retrieval explanations and metrics."""

from collections import Counter

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from example import DOCUMENTS
from from_scratch import EducationalTfidf, tokenize, validate_documents


COLORS = {"binary": "#65758b", "count": "#2673b8", "tfidf": "#238457",
          "char": "#8962ab", "bm25": "#d07826", "rrf": "#ad4569"}
METHODS = ("binary", "count", "tfidf", "char", "bm25")
SEARCH_DOCUMENTS = (
    "Machine learning model calibration and prediction monitoring",
    "RAG retrieval augmented generation uses document search and citations",
    "Cloud infrastructure deployment uses containers and autoscaling",
    "Technical documentation for API error ERR-429 retry backoff",
    "Automobile servicing and maintenance guide",
    "Car repair workshop manual",
    "Document retrieval pipeline with sparse inverted index",
    "Fictional rule SYN-204 article 12: synthetic emissions reporting procedure",
    "Fictional rule SYN-205 article 12: synthetic water reporting procedure",
    "Docurnent retrieval pipeline after synthetic OCR corruption",
    "Reimbursement procedures for returned purchases",
    "Healthcare treatment not recommended without clinical evaluation",
    "Purchase order database backup and recovery documentation",
    "Neural networks deep learning model training",
)

# Fixed task judgments defined independently of model scores. Unlisted IDs have grade 0.
JUDGMENTS = (
    ("machine learning calibration", {0: 2, 13: 1}),
    ("RAG document citations", {1: 2, 6: 1}),
    ("ERR-429 retry", {3: 2}),
    ("SYN-204 article 12 emissions", {7: 2}),
    ("document retrieval", {6: 2, 1: 1, 9: 1}),
    ("refund", {10: 2}),
    ("automobile maintenance", {4: 2, 5: 1}),
    ("cloud containers", {2: 2}),
)


def finite_nonnegative(value, name):
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def idf_values(n, df):
    n, df = finite_nonnegative(n, "N"), finite_nonnegative(df, "DF")
    if (n < 1).any() or (df < 1).any() or (df > n).any():
        raise ValueError("IDF domain requires 1 <= DF <= N")
    return {"classical": np.log(n / df), "smoothed": 1 + np.log((n + 1) / (df + 1)),
            "bm25": np.log1p((n - df + .5) / (df + .5))}


def bm25_tf(tf, length_ratio=1., k1=1.2, b=.75):
    tf = finite_nonnegative(tf, "TF")
    length_ratio = finite_nonnegative(length_ratio, "length ratio")
    if not np.isfinite([k1, b]).all() or k1 <= 0 or not 0 <= b <= 1:
        raise ValueError("BM25 requires finite k1 > 0 and 0 <= b <= 1")
    denominator = tf + k1 * (1 - b + b * length_ratio)
    return np.divide(tf * (k1 + 1), denominator, out=np.zeros(np.broadcast_shapes(tf.shape, length_ratio.shape)),
                     where=denominator != 0)


class SearchIndex:
    """Small CPU teaching index; word query multiplicity is ignored for BM25.

    BM25 uses positive log1p IDF, analyzer token lengths including repetitions,
    and exact lengths (no Lucene length quantization). Search is an exhaustive
    sparse/dense fixture computation, not a production posting-list engine.
    """

    def __init__(self, documents=SEARCH_DOCUMENTS):
        self.documents = tuple(validate_documents(documents))
        if not self.documents:
            raise ValueError("documents must not be empty")
        self.vectorizers = {
            "binary": CountVectorizer(binary=True), "count": CountVectorizer(),
            "tfidf": TfidfVectorizer(),
            "char": TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)),
        }
        self.matrices = {method: vectorizer.fit_transform(self.documents).tocsr()
                         for method, vectorizer in self.vectorizers.items()}
        self.counts = self.matrices["count"].toarray().astype(float)
        self.lengths = self.counts.sum(axis=1)
        self.avgdl = float(self.lengths.mean())
        df = (self.counts > 0).sum(axis=0)
        self.bm25_idf = idf_values(len(self.documents), df)["bm25"]
        self.postings = {term: np.flatnonzero(self.counts[:, column]).tolist()
                         for term, column in self.vectorizers["count"].vocabulary_.items()}

    def explain(self, query, method="tfidf", *, k1=1.2, b=.75):
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        vectorizer = self.vectorizers["count" if method == "bm25" else method]
        query_vector = vectorizer.transform([query]).toarray()[0].astype(float)
        features = vectorizer.get_feature_names_out()
        analyzer_features = vectorizer.build_analyzer()(query)
        oov = sorted(set(analyzer_features) - set(vectorizer.vocabulary_))
        if method == "bm25":
            query_vector = (query_vector > 0).astype(float)
            contributions = bm25_tf(self.counts, self.lengths[:, None] / self.avgdl, k1, b)
            contributions *= self.bm25_idf * query_vector
        else:
            matrix = self.matrices[method]
            norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
            normalized = sparse.diags(np.divide(1, norms, out=np.zeros_like(norms, dtype=float),
                                                where=norms != 0)) @ matrix
            query_norm = np.linalg.norm(query_vector)
            if query_norm:
                query_vector /= query_norm
            contributions = normalized.toarray() * query_vector
        scores = contributions.sum(axis=1)
        ranking = sorted(np.flatnonzero(scores > 0).tolist(), key=lambda i: (-scores[i], i))
        return {"scores": scores, "contributions": contributions, "ranking": ranking,
                "features": features, "query_vector": query_vector, "tokens": tokenize(query),
                "analyzer_features": analyzer_features, "oov": oov,
                "excluded": np.flatnonzero(scores == 0).tolist(),
                "status": "ok" if ranking else "no_lexical_evidence"}


def rrf(rankings, constant=60):
    if not np.isfinite(constant) or constant <= 0:
        raise ValueError("RRF constant must be finite and positive")
    scores = Counter()
    for ranking in rankings:
        if len(set(ranking)) != len(ranking):
            raise ValueError("a ranking must not contain duplicate document IDs")
        for rank, document in enumerate(ranking, 1):
            scores[document] += 1 / (constant + rank)
    return sorted(scores, key=lambda document: (-scores[document], document)), dict(scores)


def ranking_metrics(ranking, grades, k=3):
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    if len(set(ranking)) != len(ranking):
        raise ValueError("ranking must have unique IDs")
    if not grades or any(not np.isfinite(g) or g < 0 for g in grades.values()):
        raise ValueError("grades must be nonempty, finite and nonnegative")
    relevant = {document for document, grade in grades.items() if grade > 0}
    if not relevant:
        raise ValueError("at least one relevant document is required")
    gains = np.array([2 ** grades.get(document, 0) - 1 for document in ranking[:k]])
    discounts = np.log2(np.arange(len(gains)) + 2)
    ideal = sorted([2 ** g - 1 for g in grades.values()], reverse=True)[:k]
    ideal_dcg = np.sum(np.array(ideal) / np.log2(np.arange(len(ideal)) + 2))
    hits = len(set(ranking[:k]) & relevant)
    return {"Precision@k": hits / k, "Recall@k": hits / len(relevant),
            "MRR": next((1 / rank for rank, doc in enumerate(ranking, 1) if doc in relevant), 0.),
            "nDCG@k": float(np.sum(gains / discounts) / ideal_dcg)}


def evaluation(k=3):
    index = SearchIndex()
    per_query = {method: [] for method in (*METHODS, "rrf")}
    rankings = {}
    for query, grades in JUDGMENTS:
        ranks = {method: index.explain(query, method)["ranking"] for method in METHODS}
        ranks["rrf"] = rrf([ranks["tfidf"][:5], ranks["char"][:5]])[0]
        rankings[query] = ranks
        for method, ranking in ranks.items():
            per_query[method].append(ranking_metrics(ranking, grades, k))
    summary = {method: {metric: float(np.mean([row[metric] for row in rows]))
                        for metric in rows[0]} for method, rows in per_query.items()}
    return {"k": k, "summary": summary, "per_query": per_query, "rankings": rankings,
            "judgments": [{"query": q, "grades": g} for q, g in JUDGMENTS],
            "boundary": "synthetic fixture; judgments fixed before scores; no tuning or held-out generalization"}


def representations(documents=DOCUMENTS):
    vectorizer = CountVectorizer()
    counts = vectorizer.fit_transform(documents).toarray().astype(float)
    df = (counts > 0).sum(axis=0)
    idf = idf_values(len(documents), df)["smoothed"]
    weighted = counts * idf
    lengths = counts.sum(axis=1, keepdims=True)
    from from_scratch import l2_normalize
    return vectorizer.get_feature_names_out(), {
        "Binary": (counts > 0).astype(float), "Counts": counts,
        "Relative TF": np.divide(counts, lengths, out=np.zeros_like(counts), where=lengths > 0),
        "TF-IDF (unnormalized)": weighted, "TF-IDF (L2)": l2_normalize(weighted),
    }, idf


def sparse_fixture(n=80, vocabulary=1000, unique=20, distribution="poisson", seed=64):
    for value in (n, vocabulary, unique):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("matrix dimensions and unique terms must be positive integers")
    if unique > vocabulary or n * vocabulary > 8_000_000:
        raise ValueError("unique terms exceed vocabulary or fixture exceeds bounded memory budget")
    if distribution not in ("poisson", "zipf"):
        raise ValueError("distribution must be poisson or zipf")
    rng = np.random.default_rng(seed)
    columns = np.concatenate([np.sort(rng.choice(vocabulary, unique, replace=False)) for _ in range(n)])
    values = (rng.poisson(2, n * unique) + 1 if distribution == "poisson"
              else np.minimum(rng.zipf(2, n * unique), 100))
    matrix = sparse.csr_matrix((values.astype(np.float64), columns,
                               np.arange(n + 1) * unique), shape=(n, vocabulary))
    dense = matrix.toarray()
    return matrix, {"shape": [n, vocabulary], "nnz": matrix.nnz,
                    "density": matrix.nnz / (n * vocabulary), "dense_bytes": dense.nbytes,
                    "csr_bytes": matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes,
                    "value_bytes": matrix.data.nbytes, "column_bytes": matrix.indices.nbytes,
                    "pointer_bytes": matrix.indptr.nbytes, "dtype": str(matrix.dtype)}


def three_vectors():
    documents = ["machine machine learning", "learning learning data", "machine data data data", "learning"]
    vectorizer = TfidfVectorizer(vocabulary={"machine": 0, "learning": 1, "data": 2}, norm=None)
    weighted = vectorizer.fit_transform(documents).toarray()
    counts = CountVectorizer(vocabulary=vectorizer.vocabulary_).fit_transform(documents).toarray()
    from from_scratch import l2_normalize
    return {"documents": documents, "terms": ["machine", "learning", "data"],
            "counts": counts, "weighted": weighted, "unit": l2_normalize(weighted), "idf": vectorizer.idf_}


def projection_fixture(seed=64):
    categories = {
        "ML": ["machine learning model", "neural training prediction", "model calibration validation"],
        "Cloud": ["cloud container deployment", "infrastructure autoscaling server", "cloud database backup"],
        "Fictional law": ["synthetic rule emissions reporting", "fictional article water permit", "synthetic legislation permit reporting"],
        "Automotive": ["automobile engine maintenance", "car repair workshop", "vehicle servicing inspection"],
        "IR": ["document retrieval index", "lexical sparse search", "query ranking retrieval"],
        "Healthcare": ["healthcare treatment evaluation", "clinical patient assessment", "patient treatment monitoring"],
    }
    documents, labels = [], []
    for category, phrases in categories.items():
        for phrase in phrases:
            for suffix in ("guide", "technical procedure"):
                documents.append(phrase + " " + suffix)
                labels.append(category)
    query = "machine learning prediction"
    result = {}
    for method, vectorizer in (("count", CountVectorizer()), ("tfidf", TfidfVectorizer())):
        matrix = vectorizer.fit_transform(documents)
        svd = TruncatedSVD(n_components=3, random_state=seed, n_iter=10)
        points = svd.fit_transform(matrix)
        q = vectorizer.transform([query])
        result[method] = {"points": points, "query": svd.transform(q)[0],
                          "variance": svd.explained_variance_ratio_,
                          "original_scores": cosine_similarity(q, matrix)[0],
                          "projected_scores": cosine_similarity(svd.transform(q), points)[0]}
    return documents, labels, query, result


def drift_fixture():
    # Alpha starts rarer than beta; appended alpha-only documents reverse that prevalence.
    initial = ["alpha alpha beta noise", "beta beta alpha noise", "beta noise gamma"]
    query = "alpha beta"
    frozen = TfidfVectorizer().fit(initial)
    steps = []
    for added in range(7):
        documents = initial + ["alpha novel" for _ in range(added)]
        current = TfidfVectorizer().fit(documents)
        def scores(vectorizer):
            return cosine_similarity(vectorizer.transform([query]), vectorizer.transform(documents))[0]
        fresh_scores, frozen_scores = scores(current), scores(frozen)
        steps.append({"added": added, "N": len(documents), "alpha_df": 2 + added,
                      "alpha_idf": current.idf_[current.vocabulary_["alpha"]],
                      "beta_idf": current.idf_[current.vocabulary_["beta"]],
                      "scores": fresh_scores.tolist(), "frozen_scores": frozen_scores.tolist(),
                      "ranking": sorted(range(len(documents)), key=lambda i: (-fresh_scores[i], i)),
                      "alpha_weight_d0": float(current.transform([documents[0]])[0, current.vocabulary_["alpha"]]),
                      "novel_in_frozen": "novel" in frozen.vocabulary_})
    return initial, query, steps


def validate_numerics():
    docs = list(DOCUMENTS) + ["", "machine machine learning"]
    errors = {}
    for sublinear in (False, True):
        for norm in (None, "l2"):
            own = EducationalTfidf(sublinear_tf=sublinear, norm=norm)
            reference = TfidfVectorizer(sublinear_tf=sublinear, norm=norm)
            actual = own.fit_transform(docs)
            expected = reference.fit_transform(docs).toarray()
            if own.vocabulary_ != reference.vocabulary_:
                raise AssertionError("vocabulary mismatch")
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(own.idf_, reference.idf_, atol=1e-12, rtol=1e-12)
            queries = ["machine model", "OOV", ""]
            np.testing.assert_allclose(own.transform(queries), reference.transform(queries).toarray(), atol=1e-12)
            errors[f"sublinear={sublinear};norm={norm}"] = float(np.max(np.abs(actual - expected)))
    return {"status": "passed", "atol": 1e-12, "max_absolute_errors": errors}

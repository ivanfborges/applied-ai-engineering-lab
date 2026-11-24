"""Optional local UI; all generation and numerical checks work without a server."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_lab import core


st.set_page_config(page_title="Day 64 · Lexical retrieval", layout="wide")
st.title("Lexical retrieval, one feature at a time")
st.caption("Day 64 · Synthetic educational documents · CPU only · Author interpretation review required")
view = st.segmented_control("Experiment", ["Search", "Sparse storage", "Evaluation"], default="Search", key="view")


@st.cache_resource(max_entries=1)
def get_index():
    return core.SearchIndex()


@st.cache_data(max_entries=1)
def get_evaluation():
    return core.evaluation()


if view == "Search":
    query = st.text_input("Search query", "document retrieval", max_chars=500, key="query")
    method = st.segmented_control("Retrieval method", list(core.METHODS), default="tfidf", key="method") or "tfidf"
    k = st.slider("Maximum results", 1, len(core.SEARCH_DOCUMENTS), 5, key="k")
    k1, b = 1.2, .75
    if method == "bm25":
        left, right = st.columns(2)
        k1 = left.slider("TF saturation k1", .1, 3., 1.2, .1, key="k1")
        b = right.slider("Length normalization b", 0., 1., .75, .05, key="b")
    index = get_index()
    result = index.explain(query, method, k1=k1, b=b)
    st.code("Word tokens: " + repr(result["tokens"]))
    st.caption("Active analyzer: character word-boundary 3–5-grams" if method == "char" else "Active analyzer: lowercased word tokens, at least two characters")
    st.code("Out-of-vocabulary analyzer features: " + repr(result["oov"]))
    active = np.flatnonzero(result["query_vector"])
    st.dataframe(pd.DataFrame({"Feature": result["features"][active], "Query coordinate": result["query_vector"][active]}),
                 hide_index=True, alt="Nonzero query coordinates in the fitted feature space")
    st.caption("Query coordinates are L2-normalized for cosine methods; BM25 shows binary query-term presence.")
    top = result["ranking"][:k]
    st.metric("Positive lexical matches", len(result["ranking"]))
    if not top:
        st.info("No lexical evidence: the query is empty or has no features in this index.")
    else:
        frame = pd.DataFrame({"Document ID": [f"D{i}" for i in top],
                              "Score": [result["scores"][i] for i in top],
                              "Text": [index.documents[i] for i in top]})
        st.bar_chart(frame, x="Document ID", y="Score", horizontal=True, sort="-Score",
                     alt="Positive lexical scores for the displayed document IDs")
        st.dataframe(frame, hide_index=True, alt="Ranked documents with calculated scores and full synthetic text")
        document = st.selectbox("Inspect document contributions", top, format_func=lambda i: f"D{i}", key="document")
        values = result["contributions"][document]
        matched = np.flatnonzero(values)
        st.dataframe(pd.DataFrame({"Matched feature": result["features"][matched], "Contribution": values[matched]}),
                     hide_index=True, alt="Individual matched-feature contributions to the selected document score")
        st.metric("Sum of contributions", f"{values.sum():.8f}")
        st.caption("Cosine methods sum products of normalized coordinates. BM25 sums positive IDF × saturated TF for distinct query terms. Their score scales differ.")
    st.code("Excluded IDs (zero overlap): " + repr(result["excluded"]))
    st.caption("Legal references SYN-204 and SYN-205 are fictional. No legislation, clinical advice or generated LLM answer is represented.")
elif view == "Sparse storage":
    n = st.slider("Documents", 10, 200, 80, 10, key="n")
    vocabulary = st.slider("Vocabulary size", 100, 5000, 1000, 100, key="vocabulary")
    unique = st.slider("Unique terms per document", 1, 100, 20, key="unique")
    distribution = st.segmented_control("Positive count distribution", ["poisson", "zipf"], default="poisson", key="distribution") or "poisson"
    matrix, memory = core.sparse_fixture(n, vocabulary, unique, distribution)
    with st.container(horizontal=True):
        st.metric("Dense array bytes", memory["dense_bytes"])
        st.metric("CSR component bytes", memory["csr_bytes"])
        st.metric("Nonzero fraction", f"{memory['density']:.4f}")
    rows, columns = matrix.nonzero()
    st.scatter_chart(pd.DataFrame({"Vocabulary column": columns, "Document": rows}),
                     x="Vocabulary column", y="Document", alt="Nonzero positions in the generated sparse matrix")
    st.code(f"data[:30] = {matrix.data[:30].tolist()}\nindices[:30] = {matrix.indices[:30].tolist()}\nindptr[:30] = {matrix.indptr[:30].tolist()}")
    st.caption("Seed 64; exactly the selected number of unique terms per row. Bytes exclude Python object overhead, peak memory and runtime. No latency benchmark.")
else:
    result = get_evaluation()
    st.dataframe(pd.DataFrame(result["summary"]).T, alt="Macro retrieval metrics on eight fixed judged synthetic queries")
    st.dataframe(pd.DataFrame([{ "Query": query, "Grades by document ID": repr(grades)} for query, grades in core.JUDGMENTS]),
                 hide_index=True, alt="Fixed synthetic relevance judgments")
    st.caption("k=3. Precision denominator=3; Recall uses all relevant IDs; MRR uses all positive results; nDCG uses exponential graded gain. RRF fuses word/character top-5. No held-out tuning, production benchmark or neural semantic scores.")

"""Optional local Streamlit dashboard; all essential exports also run from the CLI."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from nlp_lab import core as c
from nlp_lab.interactive import geometry_figure, morphology_figure, space_figure, tfidf_surface


SECTIONS = ("Tokenization", "Normalization", "Morphological transformations", "N-grams",
            "Document-term matrices", "Feature space", "TF-IDF", "Document vector space", "Cosine similarity",
            "Statistical language models", "Retrieval experiments", "OCR robustness")


@st.cache_data(max_entries=4)
def ocr_data(seed: int, trials: int) -> dict:
    return c.ocr_experiment(seed, trials)


def chart(fig, key: str, description: str) -> None:
    st.plotly_chart(fig, key=key, width="stretch", alt=description)


def main() -> None:
    st.set_page_config(page_title="Classical NLP · Day 63", layout="wide")
    st.title("Classical NLP: inspect the representation")
    st.caption("Synthetic educational lab · CPU only · Local computation · Author interpretation review required")
    section = st.sidebar.selectbox("Explore", SECTIONS, key="section")
    st.header(section)
    try:
        if section == "Tokenization":
            st.write("How do boundary rules trade vocabulary size for sequence length?")
            text = st.text_area("Text", c.SENTENCES[1], key="token_text")
            strategy = st.selectbox("Strategy", c.STRATEGIES, key="strategy")
            pieces = c.segment(text, strategy)
            st.metric("Tokens in input", len(pieces))
            st.dataframe(pd.DataFrame(pieces), alt="Tokens with source start and end offsets")
            st.caption("Bytes: UTF-8 offsets; other strategies: code-point offsets. Treebank is English-oriented; supplied PT examples use Toktok. Characters are not grapheme clusters.")
            if strategy == "subword":
                st.write(f"Local character BPE: {len(c.trained_bpe().vocabulary)} vocabulary entries, {len(c.trained_bpe().merges)} learned merges; unseen characters become <unk>.")
                st.json({"merges": c.trained_bpe().merges, "vocabulary": c.trained_bpe().vocabulary})
        elif section == "Normalization":
            text = st.text_area("Text", c.RAW, key="normalization_text")
            operations = st.multiselect("Apply in this selected order", c.NORMALIZATIONS, default=["NFC", "casefold", "whitespace"], key="operations")
            trace = c.normalization_trace(text, operations)
            st.dataframe(pd.DataFrame(trace), alt="Sequential normalization text and code-point lengths")
            st.dataframe(pd.DataFrame(c.codepoints(trace[-1]["text"])), alt="Actual Unicode code points after selected operations")
            st.warning("Destructive operations are experimental choices. Stopword removal uses the explicit English teaching list and can delete negation; accent and punctuation removal can damage names and identifiers.")
        elif section == "Morphological transformations":
            method = st.selectbox("Algorithm", ("porter", "snowball", "lemma", "portuguese"), key="morph_method")
            rows = c.morphology_rows(include_wordnet=method == "lemma")
            chart(morphology_figure(rows, method), "morph_chart", "Computed surface-to-stem or lemma mappings")
            st.dataframe(pd.DataFrame(rows), alt="Actual NLTK morphology outputs and supplied POS")
            st.caption("POS is supplied, not predicted. In 'I saw a bird' the contextual lemma should be see; WordNet can still return saw because it is also a verb. Inspect actual candidates. Leaves supplies a noun/verb contrast. Portuguese uses Portuguese Snowball.")
        elif section == "N-grams":
            text = st.text_input("Sentence", "machine learning models learn useful patterns", key="ngram_text")
            unit = st.selectbox("Unit", ("word", "character"), key="ngram_unit")
            n = st.slider("Order n", 1, 12, 2, key="ngram_n")
            tokens = c.tokenize(text) if unit == "word" else list(text)
            grams = c.generate_ngrams(tokens, n)
            st.latex(r"G(m,n)=\max(0,m-n+1)")
            st.write({"m": len(tokens), "n": n, "occurrences": len(grams)})
            st.json({"tokens": tokens, "ngrams": grams})
        elif section == "Document-term matrices":
            text = st.text_area("One document per line", "Machine learning models\nMachine learning learning\nDeep learning models", key="dtm_text")
            n = st.slider("Fixed order", 1, 3, 1, key="dtm_n")
            features, counts = c.document_term_matrix(text.splitlines(), n)
            st.write(f"Shape = {counts.shape}; dense educational counts; empty vocabularies are valid.")
            if len(features):
                names = [" ".join(g) for g in features]
                chart(px.imshow(counts, x=names, labels={"x": "Feature", "y": "Document ID", "color": "Count"}, text_auto=True, aspect="auto"), "dtm_chart", "Computed document-term counts")
        elif section == "Feature space":
            n = st.slider("Maximum word order", 1, 3, 2, key="feature_n")
            minimum = st.slider("min_df (documents)", 1, 20, 1, key="min_df")
            cap = st.selectbox("max_features", (None, 30, 60, 100), key="max_features")
            stats, matrix, features = c.feature_stats(c.synthetic_corpus(), n, minimum, cap)
            st.json(stats)
            chart(px.imshow(matrix.toarray(), aspect="auto", labels={"x": "Feature index", "y": "Document ID", "color": "Count"}), "feature_chart", "Synthetic matrix occupancy after feature pruning")
            st.caption("Actual CSR array storage excludes object overhead. K^n is a fixed-order bound, not an observed count or a memory prediction.")
        elif section == "TF-IDF":
            documents = st.number_input("Corpus size D", min_value=1, max_value=500, value=100, step=1, key="D")
            df = st.number_input("Document frequency df", min_value=1, max_value=int(documents), value=min(10, int(documents)), step=1, key="df")
            tf = st.slider("Term frequency TF", 0, 10, 3, key="tf")
            norm = st.selectbox("Document vector normalization", ("l2", "none"), key="tfidf_norm")
            st.latex(r"\mathrm{IDF}(t)=\log\frac{1+D}{1+\mathrm{df}(t)}+1")
            st.metric("Selected unnormalized weight", f"{c.tfidf_weight(int(documents), int(df), tf):.5f}")
            chart(tfidf_surface(int(documents), int(df), tf), "tfidf_surface", "Computed TF-IDF surface and selected term")
            _, counts = c.document_term_matrix(["document license license", "document payment", "document license"])
            _, _, weighted = c.tfidf_from_counts(counts, norm == "l2")
            st.dataframe(pd.DataFrame(weighted), alt="TF-IDF example with chosen document normalization")
        elif section == "Document vector space":
            rep = st.selectbox("Representation", ("word", "word12", "char"), key="space_rep")
            preset = st.selectbox("Query example", c.SPACE_QUERIES, key="space_preset")
            query = st.text_input("Query text", preset, key=f"space_query_{preset}")
            result = c.vector_space(rep, query)
            chart(space_figure(result, query), "space_chart", "SVD projection with original-space cosine in hover")
            st.dataframe(pd.DataFrame([{"document_id": i, "cosine": result["scores"][i], "text": c.TOPIC_DOCS[i][1]} for i in result["ranking"]]), alt="Authoritative ranking computed before SVD")
            st.warning("Projection distances can distort original relationships. Large points mean top lexical scores, not relevance judgments.")
        elif section == "Cosine similarity":
            angle = st.slider("Azimuth degrees", 0, 180, 45, key="angle")
            magnitude = st.slider("Magnitude", 0.0, 3.0, 1.0, .1, key="magnitude")
            elevation = st.slider("Elevation degrees", -90, 90, 0, key="elevation")
            chart(geometry_figure(angle, magnitude, elevation), "geometry_chart", "Actual signed vector geometry")
            st.latex(r"\cos(a,b)=\frac{a\cdot b}{\lVert a\rVert\lVert b\rVert}")
            st.caption("Zero magnitude: undefined geometric cosine. Nonnegative TF-IDF cannot have negative cosine.")
        elif section == "Statistical language models":
            order = st.selectbox("Model order", (1, 2, 3), index=1, key="lm_order")
            alpha = st.slider("Smoothing alpha", 0.0, 2.0, .1, .1, key="alpha")
            seed = st.number_input("Generation seed", 0, 10000, 63, key="lm_seed")
            model = c.NGramLM([c.tokenize(text) for text in c.LM_SENTENCES], order)
            contexts = list(model.counts)
            context = st.selectbox("Prediction context", contexts, format_func=lambda x: repr(x), key=f"context_{order}")
            probabilities = model.distribution(context, alpha)
            st.metric("Probability sum", f"{probabilities.sum():.10f}")
            chart(px.bar(x=model.vocabulary, y=probabilities, labels={"x": "Prediction token", "y": "Probability"}), "lm_chart", "Conditional next-token probabilities")
            st.dataframe(pd.DataFrame(model.generate(int(seed), alpha)), alt="Seeded generation steps, contexts and selected tokens")
            st.caption("No backoff: an unseen unsmoothed context is undefined. BOS is context only; EOS is predicted. Counter statistics do not supply Transformer context understanding.")
        elif section == "Retrieval experiments":
            stop = st.selectbox("Stop-word policy", ("keep", "drop", "preserve"), key="stop_policy")
            morph = st.selectbox("Morphology", ("none", "porter", "lemma"), key="retrieval_morph")
            n = st.slider("Maximum word order", 1, 3, 1, key="retrieval_n")
            query = st.text_input("Query text", c.CONTRAST_QUERIES[1][0], key="retrieval_query")
            result = c.retrieval_run(stop, morph, n, query)
            scores = result["results"][0]["scores"]
            chart(px.bar(x=scores, y=list(c.CONTRAST_DOCS), orientation="h", labels={"x": "Original-space cosine", "y": "Document"}), "ranking_chart", "Actual retrieval scores after selected preprocessing")
            judged = c.retrieval_run(stop, morph, n)
            st.json({"seven fixed judged-query means": judged["mean_metrics"], "k": 3})
            st.dataframe(pd.DataFrame(judged["results"]), alt="Fixed synthetic relevance judgments and individual misses")
            st.caption("Free-text query has no inferred judgments; displayed means always refer to the separate seven-query fixture. Supplied tiny-vocabulary POS, default noun for other tokens.")
        elif section == "OCR robustness":
            rate = st.slider("Corruption selection probability", 0.0, .7, .2, .05, key="ocr_rate")
            seed = st.number_input("Corruption seed", 0, 10000, 63, key="ocr_seed")
            rep = st.selectbox("Representation", c.REPRESENTATIONS, key="ocr_rep")
            original = st.text_input("Clean query", c.OCR_QUERIES[0][0], key="ocr_query")
            corrupted = c.corrupt_text(original, rate, int(seed))
            st.write({"clean": original, "corrupted": corrupted})
            vectorizer = c.make_vectorizer(rep)
            matrix = vectorizer.fit_transform(c.OCR_DOCS)
            scores = c.cosine_similarity(vectorizer.transform([corrupted]), matrix).ravel()
            st.dataframe(pd.DataFrame([{"document_id": i, "score": scores[i], "text": c.OCR_DOCS[i]} for i in c.rank(scores)]), alt="Actual retrieved clean documents for the corrupted query")
            report = ocr_data(int(seed), 4)
            rows = pd.DataFrame(report["rows"])
            chart(px.line(rows.groupby(["representation", "rate"], as_index=False)["rr"].mean(), x="rate", y="rr", color="representation", labels={"rr": "Mean reciprocal rank"}), "ocr_chart", "Paired four-trial synthetic corruption experiment")
            st.caption("The curve uses the fixed six judged queries. The editable query has no assumed relevance label. Rates are selection probabilities, not edit distance or real OCR error rates.")
    except c.ResourceUnavailable as exc:
        st.warning(str(exc))
        st.info("Other sections remain available. Resource downloads are explicit setup steps and never triggered by widgets.")
    except ValueError as exc:
        st.warning(str(exc))


if __name__ == "__main__":
    main()

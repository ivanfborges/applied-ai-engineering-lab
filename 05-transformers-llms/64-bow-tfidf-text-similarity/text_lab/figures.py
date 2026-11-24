"""Calculated scientific plots and bounded Matplotlib/Pillow teaching animations."""

from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from from_scratch import l2_normalize, tokenize
from . import core


plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.titlepad": 12})


class Exporter:
    def __init__(self, output, *, quick=False, targets=None):
        self.output = Path(output)
        self.quick = quick
        self.targets = set(targets) if targets else None
        self.paths = []

    def wants(self, filename):
        return self.targets is None or filename in self.targets

    def path(self, filename):
        folder = {".png": "png", ".gif": "gif", ".html": "html"}[Path(filename).suffix]
        path = self.output / folder / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def png(self, filename, figure):
        if self.wants(filename):
            figure.savefig(self.path(filename), dpi=100 if self.quick else 120, facecolor="white")
            self.paths.append(self.path(filename))
        plt.close(figure)

    def gif(self, filename, frames, draw, *, size=(12, 6), interval=850, bottom=.13):
        if not self.wants(filename):
            return
        figure = plt.figure(figsize=size)
        def update(frame):
            figure.clear()
            draw(figure, frame)
            figure.tight_layout(rect=(0, bottom, 1, .90))
        animation = FuncAnimation(figure, update, frames=frames, interval=interval, repeat=True)
        animation.save(self.path(filename), writer=PillowWriter(fps=1000 / interval),
                       dpi=75 if self.quick else 90)
        plt.close(figure)
        self.paths.append(self.path(filename))


def heatmap(ax, matrix, title, columns=None, *, vmax=None, annotate=True, cmap="YlGnBu"):
    matrix = np.asarray(matrix)
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Feature" if columns is not None else "Document ID")
    ax.set_ylabel("Document ID")
    ax.set_yticks(range(len(matrix)))
    ax.set_xticks(range(matrix.shape[1]), columns if columns is not None else range(matrix.shape[1]),
                  rotation=55 if columns is not None else 0, ha="right" if columns is not None else "center",
                  fontsize=8)
    if annotate:
        limit = image.get_clim()[1]
        for row, column in np.ndindex(matrix.shape):
            value = matrix[row, column]
            ax.text(column, row, f"{value:.2f}" if value % 1 else f"{value:.0f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if value > limit * .58 else "black")
    return image


def note(figure, text):
    figure.text(.5, .015, text, ha="center", fontsize=9, color="#475569", wrap=True)
    figure.subplots_adjust(bottom=.2, top=.85, hspace=.7, wspace=.35)


def bars(ax, ranking, scores, title, color="#238457"):
    ranking = list(ranking)
    if not ranking:
        ax.text(.5, .5, "No positive lexical evidence", ha="center", transform=ax.transAxes)
    else:
        values = [scores[i] for i in ranking]
        ax.barh([str(i) for i in ranking], values, color=color)
        ax.invert_yaxis()
        for row, value in enumerate(values):
            ax.text(value, row, f" {value:.4f}", va="center", fontsize=9)
        ax.set_xlim(0, max(values) * 1.28)
    ax.set_title(title)
    ax.set_xlabel("Calculated score (method-specific scale)")
    ax.set_ylabel("Document ID")


def tokens(export):
    terms, matrices, _ = core.representations()
    counts = matrices["Counts"]
    stages = ["Original documents", "Lowercase normalization", "Word tokenization",
              "Sorted vocabulary", "Vocabulary indices", "Term counts", "Document-term matrix", "CSR row storage"]
    def draw(figure, frame):
        ax = figure.add_subplot(211)
        ax.axis("off")
        stage = min(frame, len(stages) - 1)
        figure.suptitle(f"Text to vector — {stage + 1}/8: {stages[stage]}", fontsize=15)
        if stage < 2:
            lines = [f"D{i}: {document if stage == 0 else document.lower()}"
                     for i, document in enumerate(core.DOCUMENTS)]
        elif stage == 2:
            lines = [f"D{i}: {' | '.join(tokenize(doc))}" for i, doc in enumerate(core.DOCUMENTS)]
        elif stage in (3, 4):
            lines = textwrap.wrap("   ".join(f"{i}: {t}" if stage == 4 else t
                                            for i, t in enumerate(terms)), width=110)
        elif stage == 5:
            lines = [f"D0 counts: {dict(__import__('collections').Counter(tokenize(core.DOCUMENTS[0])))}"]
        else:
            from scipy.sparse import csr_matrix
            row = csr_matrix(counts).getrow(0)
            lines = ["D0 CSR nonzero columns: " + str(row.indices.tolist()),
                     "D0 CSR values: " + str(row.data.tolist()), "All other vocabulary columns are implicit zeros."]
        ax.text(.01, .95, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=10)
        bottom = figure.add_subplot(212)
        heatmap(bottom, counts, "Shared vocabulary; counts are computed from the documents", terms)
        if stage >= 2:
            token = tokenize(core.DOCUMENTS[0])[max(0, stage - 2) % len(tokenize(core.DOCUMENTS[0]))]
            column = list(terms).index(token)
            bottom.add_patch(Rectangle((column - .5, -.5), 1, 1, fill=False, lw=3, edgecolor="#d07826"))
            ax.text(.01, .02, f"Highlighted token '{token}' -> column {column}; D0 value={counts[0, column]:g}",
                    color="#a45710", transform=ax.transAxes)
    export.gif("01_text_to_vector.gif", 10, draw, size=(16, 9), interval=1100)
    return {"vocabulary": terms.tolist(), "counts": counts.tolist()}


def tfidf(export):
    terms, matrices, idf = core.representations()
    figure, axes = plt.subplots(5, 1, figsize=(19, 15))
    for ax, (name, matrix) in zip(axes, matrices.items()):
        heatmap(ax, matrix, name + " — own color scale", terms)
    figure.suptitle("Same documents and columns; different weighting conventions", fontsize=16)
    figure.tight_layout(rect=(0, .02, 1, .96))
    export.png("02_representation_comparison.png", figure)
    stages = ["Counts", "TF-IDF (unnormalized)", "TF-IDF (L2)"]
    def draw(figure, frame):
        stage = min(frame, 2)
        ax = figure.add_subplot(111)
        heatmap(ax, matrices[stages[stage]], stages[stage], terms)
        figure.suptitle("Count × corpus IDF -> divide each nonzero row by its L2 norm", fontsize=15)
        df = (matrices["Counts"] > 0).sum(axis=0)
        frequent = int(np.argmax(df)); rare = int(np.argmin(df))
        figure.text(.5, .02, f"'{terms[frequent]}': DF={df[frequent]}, IDF={idf[frequent]:.3f}; "
                    f"'{terms[rare]}': DF={df[rare]}, IDF={idf[rare]:.3f}. Equal local TF can have unequal weights.",
                    ha="center", fontsize=11)
    export.gif("03_tfidf_transformation.gif", 5, draw, size=(18, 6), interval=1500, bottom=.18)
    return {"idf": dict(zip(terms, idf.tolist())), "row_norms": np.linalg.norm(matrices[stages[-1]], axis=1).tolist()}


def sparse_storage(export, seed=64):
    matrix, memory = core.sparse_fixture(seed=seed)
    small, small_memory = core.sparse_fixture(n=4, vocabulary=8, unique=3, seed=seed)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].spy(matrix, markersize=1)
    axes[0].set(title=f"CSR occupancy: {matrix.shape}, density={memory['density']:.3f}", xlabel="Vocabulary column", ylabel="Document")
    heatmap(axes[1], small.toarray(), "Small dense view of the same storage idea", [f"t{i}" for i in range(8)])
    note(figure, "80 documents × 1,000 features; exactly 20 unique terms per document. Poisson(2)+1 counts; seed 64.")
    export.png("04_sparse_matrix_structure.png", figure)
    def draw(figure, frame):
        ax = figure.add_subplot(211)
        heatmap(ax, small.toarray(), f"Convert row {min(frame, 3)} to CSR", [f"t{i}" for i in range(8)])
        ax.add_patch(Rectangle((-.5, min(frame, 3) - .5), 8, 1, fill=False, edgecolor="#d07826", lw=3))
        end = small.indptr[min(frame + 1, 4)]
        bottom = figure.add_subplot(212); bottom.axis("off")
        bottom.text(.02, .9, f"data = {small.data[:end].tolist()}\n\nindices = {small.indices[:end].tolist()}\n\nindptr = {small.indptr[:min(frame + 2, 5)].tolist()}",
                    va="top", fontsize=12)
        figure.suptitle("CSR stores values, column indices and row boundaries — no implicit zeros")
    export.gif("05_csr_representation.gif", 6, draw, size=(13, 7), interval=1200)
    records = [core.sparse_fixture(n=n, vocabulary=v, unique=u, seed=seed)[1]
               for n, v, u in ((10, 50, 5), (80, 1000, 20), (300, 5000, 30))]
    figure, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(records))
    for offset, key, color in ((-.2, "dense_bytes", "#2673b8"), (.2, "csr_bytes", "#238457")):
        ax.bar(x + offset, [row[key] for row in records], width=.4, label=key, color=color)
    ax.set_yscale("log")
    ax.set_xticks(x, [str(row["shape"]) for row in records])
    ax.set(title="Actual array storage: dense.nbytes vs data + indices + indptr", xlabel="Matrix shape", ylabel="Bytes (log scale)")
    ax.legend(); note(figure, "float64 values; actual SciPy index dtypes. Excludes object overhead, peak memory, allocator and runtime cost.")
    export.png("06_sparse_vs_dense_memory.png", figure)
    return {"storage": records, "small_csr": {"values": small.data.tolist(), "indices": small.indices.tolist(), "indptr": small.indptr.tolist()},
            "small_memory": small_memory}


def frequency(export):
    tf = np.arange(101)
    values = {"Binary": (tf > 0).astype(float), "Raw": tf, "Relative TF (length=100)": tf / 100,
              "Sublinear": np.where(tf > 0, 1 + np.log(np.maximum(tf, 1)), 0),
              "BM25 factor": core.bm25_tf(tf, 1., 1.2, .75)}
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    tf_colors = ["#65758b", "#2673b8", "#8962ab", "#238457", "#d07826"]
    for (name, value), color in zip(values.items(), tf_colors):
        axes[0].plot(tf, value, label=name, color=color)
        if name != "Raw": axes[1].plot(tf, value, label=name, color=color)
    for ax in axes:
        ax.set(xlabel="Occurrences (0–100)", ylabel="TF factor, before IDF", title="Full range" if ax is axes[0] else "Non-raw factors")
        ax.legend(fontsize=8)
    note(figure, "Fixed document length=average length=100; other tokens are replaced. BM25 k1=1.2, b=.75. These factors have different scales.")
    export.png("07_term_frequency_comparison.png", figure)
    return {"tf_100": {key: float(value[-1]) for key, value in values.items()}}


def rarity(export):
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, n in zip(axes, (10, 100, 1000)):
        df = np.arange(1, n + 1)
        for (name, values), color in zip(core.idf_values(n, df).items(), ("#2673b8", "#238457", "#d07826")):
            ax.plot(df / n, values, label=name, color=color)
        ax.set(title=f"Corpus N={n}", xlabel="DF / N", ylabel="IDF (natural log)"); ax.legend()
    note(figure, "Classical IDF = −log(P(document contains term)); smoothed and positive BM25 variants use different formulas.")
    export.png("09_idf_comparison.png", figure)
    def draw(figure, frame):
        df = min(frame + 1, 10)
        ax = figure.add_subplot(121)
        for i in range(10):
            ax.add_patch(Rectangle((i % 5, i // 5), .85, .7, facecolor="#238457" if i < df else "#dce3eb"))
            ax.text(i % 5 + .42, i // 5 + .35, f"D{i}", ha="center")
        ax.set(xlim=(-.2, 5), ylim=(-.2, 2.1), title=f"Target present in {df}/10 documents"); ax.axis("off")
        ax = figure.add_subplot(122)
        names = core.idf_values(10, df)
        ax.bar(names.keys(), [float(v) for v in names.values()], color=["#2673b8", "#238457", "#d07826"])
        ax.set(ylim=(0, 4), ylabel="IDF", title=f"DF={df}; P={df/10:.1f}")
        for i, value in enumerate(names.values()): ax.text(i, float(value) + .08, f"{float(value):.3f}", ha="center")
        figure.suptitle("Rarity is document prevalence, not total term repetitions")
    export.gif("10_idf_rarity_animation.gif", 12, draw, interval=950)
    return {"N": 10, "DF": list(range(1, 11)), "smoothed": core.idf_values(10, np.arange(1, 11))["smoothed"].tolist()}


def vectors(export):
    data = core.three_vectors()
    def draw(figure, frame):
        stages = [("Raw counts", data["counts"]), ("IDF-weighted", data["weighted"]), ("L2-normalized", data["unit"])]
        name, matrix = stages[min(frame, 2)]
        ax = figure.add_subplot(111, projection="3d")
        for row, vector in enumerate(matrix):
            ax.quiver(0, 0, 0, *vector, color=plt.cm.tab10(row), arrow_length_ratio=.13)
            ax.text(*vector, f" D{row}")
        limit = float(data["weighted"].max()) * 1.15
        ax.set(xlim=(0, limit), ylim=(0, limit), zlim=(0, limit), xlabel="machine", ylabel="learning", zlabel="data", title=name)
        figure.suptitle("Three actual vocabulary dimensions; no projection")
        figure.text(.5, .04, "IDF changes relative coordinates; L2 then changes magnitude along the same ray. Axis limits remain fixed.", ha="center")
    export.gif("14_tfidf_vector_transformation.gif", 5, draw, size=(11, 7), interval=1500)
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in data.items()}


def cosine(export):
    def draw(figure, frame):
        angle = min(frame, 18) * 10
        radians = np.deg2rad(angle); vector = np.array([np.cos(radians), np.sin(radians)])
        ax = figure.add_subplot(111)
        circle = plt.Circle((0, 0), 1, fill=False, color="#cbd5e1"); ax.add_patch(circle)
        for v, color, name in (([1, 0], "#2673b8", "A"), (vector, "#238457", "B")):
            ax.arrow(0, 0, *v, head_width=.045, length_includes_head=True, color=color)
            ax.text(v[0] * 1.12, v[1] * 1.12, name)
        ax.axhline(0, color="grey", lw=.5); ax.axvline(0, color="grey", lw=.5)
        ax.set(xlim=(-1.4, 1.4), ylim=(-1.4, 1.4), aspect="equal", xlabel="x", ylabel="y",
               title=f"Angle={angle}°; cos={np.cos(radians):.3f}; dot={np.cos(radians):.3f}")
        figure.suptitle("Cosine: parallel +1 -> orthogonal 0 -> opposite −1")
        figure.text(.5, .025, "Signed general vectors can have negative cosine. Nonnegative TF-IDF cannot; zero-length cosine is undefined.", ha="center")
    export.gif("18_cosine_similarity_animation.gif", 21, draw, size=(10, 7), interval=550)
    return {"angles": [0, 90, 180], "cosine": [1, 0, -1]}


def normalization(export):
    raw = np.array([[3., 4.], [7., 1.], [0., 0.]])
    unit = l2_normalize(raw)
    def draw(figure, frame):
        progress = min(frame / 8, 1)
        matrix = raw * (1 - progress) + unit * progress
        ax = figure.add_subplot(111)
        ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="#238457"))
        for i, row in enumerate(matrix):
            ax.arrow(0, 0, *row, head_width=.1, length_includes_head=True, color=plt.cm.tab10(i))
            ax.text(row[0] + .1, row[1] + .1, f"D{i}: norm={np.linalg.norm(row):.2f}")
        ax.set(xlim=(-.4, 8), ylim=(-.4, 5.5), aspect="equal", xlabel="Feature 1", ylabel="Feature 2", title=f"Radial interpolation to L2 normalization: {progress:.0%}")
        figure.text(.5, .03, "Intermediate frames illustrate scaling along a fixed direction; zero stays at zero and is not on the unit circle.", ha="center")
    export.gif("19_l2_normalization.gif", 11, draw, size=(11, 7), interval=650)
    query = np.array([[1., 1.]])
    docs = np.array([[1., 1.], [100., 0.], [0., 0.]])
    qn, dn = l2_normalize(query), l2_normalize(docs)
    measures = np.column_stack([(query @ docs.T)[0], cosine_similarity(query, docs)[0],
                                (qn @ dn.T)[0], ((qn - dn) ** 2).sum(axis=1)])
    figure, ax = plt.subplots(figsize=(12, 4))
    heatmap(ax, measures, "Long D1 wins raw dot; aligned D0 wins cosine", ["Raw dot", "Cosine", "Unit dot", "Unit distance²"])
    note(figure, "Q=[1,1]; D0=[1,1], D1=[100,0], D2=[0,0]. Distance identity applies only to nonzero unit rows, not D2.")
    export.png("20_similarity_metrics_comparison.png", figure)
    return {"query": query.tolist(), "documents": docs.tolist(), "metrics": measures.tolist()}


def similarities(export):
    vectorizers = {"binary": CountVectorizer(binary=True), "count": CountVectorizer(),
                   "tfidf": TfidfVectorizer(), "char": TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))}
    matrices = {name: cosine_similarity(v.fit_transform(core.DOCUMENTS)) for name, v in vectorizers.items()}
    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (name, matrix) in zip(axes.ravel(), matrices.items()):
        heatmap(ax, matrix, f"{name}: actual cosine", vmax=1)
    figure.suptitle("Document similarity: shared 0–1 scale; all 36 cells annotated")
    figure.tight_layout(rect=(0, .03, 1, .96)); export.png("21_document_similarity_heatmaps.png", figure)
    distance = np.clip(1 - matrices["tfidf"], 0, 2); np.fill_diagonal(distance, 0)
    z = linkage(squareform(distance, checks=False), method="average")
    figure, ax = plt.subplots(figsize=(11, 5))
    dendrogram(z, labels=[f"D{i}" for i in range(6)], ax=ax)
    ax.set(title="Average linkage on cosine dissimilarity (1−cos)", xlabel="Document ID", ylabel="Average dissimilarity")
    note(figure, "Tiny synthetic corpus; clustering is lexical. No Ward linkage or claim that cosine dissimilarity is Euclidean distance.")
    export.png("22_document_similarity_clusters.png", figure)
    return {"similarities": {name: matrix.tolist() for name, matrix in matrices.items()}}


def projections(export, seed=64):
    documents, labels, query, data = core.projection_fixture(seed)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (method, values) in zip(axes, data.items()):
        for i, category in enumerate(sorted(set(labels))):
            selected = np.array(labels) == category
            ax.scatter(*values["points"][selected, :2].T, label=category, color=plt.cm.tab10(i))
        ax.scatter(*values["query"][:2], marker="*", s=180, c="black", label="query")
        ax.set(title=f"{method}; explained variance (2D)={sum(values['variance'][:2]):.3f}", xlabel="SVD 1", ylabel="SVD 2"); ax.legend(fontsize=7)
    note(figure, "36 synthetic documents, seed 64; separate fitted SVD per representation. Each query uses that same fitted transform. Projection distances can change retrieval order.")
    export.png("25_bow_vs_tfidf_projection.png", figure)
    return {"documents": documents, "labels": labels, "query": query,
            "variance": {method: value["variance"].tolist() for method, value in data.items()}}


NGRAM_DOCUMENTS = ["dog bites man", "man bites dog", "treatment recommended", "treatment not recommended",
                   "document retrieval", "retrieval document", "docurnent retrieval"]


def ngram_vectorizers():
    return {"unigram": CountVectorizer(), "bigram": TfidfVectorizer(ngram_range=(2, 2)),
            "uni+bi": TfidfVectorizer(ngram_range=(1, 2)),
            "char 3–5": TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))}


def ngrams(export):
    vectorizers = ngram_vectorizers()
    scores = {name: cosine_similarity(v.fit_transform(NGRAM_DOCUMENTS)) for name, v in vectorizers.items()}
    figure, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, (name, matrix) in zip(axes, scores.items()): heatmap(ax, matrix, name, vmax=1)
    figure.suptitle("D0/D1 reorder; D2/D3 negate; D4/D5 reorder; D4/D6 OCR")
    figure.tight_layout(rect=(0, .08, 1, .92)); export.png("26_ngram_feature_comparison.png", figure)
    def draw(figure, frame):
        name = list(vectorizers)[min(frame, 3)]; v = vectorizers[name]
        features = v.build_analyzer()(NGRAM_DOCUMENTS[4])
        ax = figure.add_subplot(111); ax.axis("off")
        figure.suptitle(f"Same input 'document retrieval' -> {name}", fontsize=16)
        lines = textwrap.wrap(" | ".join(features), width=85)
        ax.text(.02, .9, "\n\n".join(lines), va="top", fontsize=12)
        ax.text(.02, .06, f"{len(features)} occurrences; {len(set(features))} unique features. Character boundaries include word-edge spaces.", fontsize=11)
    export.gif("27_ngram_tokenization.gif", 6, draw, size=(13, 7), interval=1400)
    figure, ax = plt.subplots(figsize=(10, 5))
    values = [matrix[4, 6] for matrix in scores.values()]
    ax.bar(scores.keys(), values, color=["#2673b8", "#65758b", "#238457", "#8962ab"])
    ax.set(ylim=(0, 1.1), ylabel="Cosine", title="OCR fixture: document retrieval vs docurnent retrieval")
    for i, value in enumerate(values): ax.text(i, value + .03, f"{value:.3f}", ha="center")
    note(figure, "m -> rn corruption; character overlap is partial lexical evidence, not a semantic model or general OCR benchmark.")
    export.png("28_ocr_similarity_comparison.png", figure)
    return {"documents": NGRAM_DOCUMENTS, "similarities": {name: matrix.tolist() for name, matrix in scores.items()},
            "features": {name: v.get_feature_names_out().tolist() for name, v in vectorizers.items()}}


def retrieval(export):
    query = "document retrieval"
    index = core.SearchIndex()
    explanations = {method: index.explain(query, method) for method in core.METHODS}
    figure, axes = plt.subplots(1, 5, figsize=(20, 6))
    for ax, (method, result) in zip(axes, explanations.items()):
        bars(ax, result["ranking"][:5], result["scores"], method, core.COLORS[method])
    note(figure, "Query: document retrieval. Positive hits only. Each panel has its own score axis; BM25 is not a cosine score.")
    export.png("29_query_ranking_comparison.png", figure)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, method in zip(axes, ("tfidf", "bm25")):
        result = explanations[method]; document = result["ranking"][0]
        values = result["contributions"][document]; active = np.flatnonzero(values)
        ax.barh(result["features"][active], values[active], color=core.COLORS[method])
        ax.set(title=f"{method}: top D{document}, sum={values.sum():.4f}", xlabel="Feature contribution")
        for i, value in enumerate(values[active]): ax.text(value, i, f" {value:.4f}", va="center")
    note(figure, "TF-IDF: product of normalized query and document coordinates. BM25: positive IDF × saturated TF, unique query terms.")
    export.png("30_term_score_contributions.png", figure)
    return {"query": query, "rankings": {m: r["ranking"] for m, r in explanations.items()},
            "scores": {m: r["scores"].tolist() for m, r in explanations.items()}}


def inverted(export):
    index = core.SearchIndex(); query = "document retrieval"; words = tokenize(query)
    candidates = sorted(set().union(*(set(index.postings.get(word, [])) for word in words)))
    result = index.explain(query, "bm25")
    def draw(figure, frame):
        ax = figure.add_subplot(111); ax.axis("off")
        stages = ["Original collection", "Vocabulary and postings", "Tokenize query", "Lookup posting lists", "Union candidates", "Score candidates", "Top-k"]
        stage = min(frame, 6); figure.suptitle(f"Inverted index — {stage + 1}/7: {stages[stage]}", fontsize=16)
        if stage == 0:
            lines = [f"D{i}: {d}" for i, d in enumerate(index.documents[:7])] + ["… 14 synthetic documents total"]
        elif stage == 1:
            lines = [f"{word} -> " + str([(i, int(index.counts[i, index.vectorizers['count'].vocabulary_[word]])) for i in index.postings[word]])
                     for word in ("document", "retrieval", "cloud")]
        elif stage == 2: lines = [f"Query: {query}", f"Tokens: {words}"]
        elif stage == 3: lines = [f"{word}: postings={index.postings[word]}" for word in words]
        elif stage == 4: lines = [f"Union candidate IDs: {candidates}", "OR semantics; no document outside this union has a word match."]
        elif stage == 5: lines = [f"D{i}: BM25={result['scores'][i]:.4f}" for i in candidates]
        else: lines = [f"Rank {rank}: D{i} score={result['scores'][i]:.4f}" for rank, i in enumerate(result['ranking'][:3], 1)]
        ax.text(.03, .9, "\n\n".join(lines), va="top", fontsize=12)
        ax.text(.03, .03, f"Illustrative access: {len(candidates)} candidate IDs versus 14 documents. This is not a latency benchmark.", fontsize=10)
    export.gif("31_inverted_index_search.gif", 9, draw, size=(14, 8), interval=1200)
    return {"query": query, "candidates": candidates, "postings": {word: index.postings[word] for word in words}}


def bm25(export):
    documents = ["rare common", "rare " * 20 + "common", "rare " + "filler " * 60,
                 "common common", "other common filler"]
    index = core.SearchIndex(documents); query = "rare common"
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    results = {method: index.explain(query, method) for method in ("tfidf", "bm25")}
    for ax, (method, result) in zip(axes, results.items()): bars(ax, result["ranking"], result["scores"], method, core.COLORS[method])
    note(figure, "D0 short balanced; D1 repeated rare; D2 long filler; D3 common only; D4 common/other/filler. Scores have method-specific scales.")
    export.png("32_bm25_vs_tfidf_ranking.png", figure)
    def draw(figure, frame):
        b = min(frame, 10) / 10
        ax = figure.add_subplot(111)
        ratios = np.linspace(.1, 4, 100)
        for tf in (1, 3, 20): ax.plot(ratios, core.bm25_tf(tf, ratios, 1.2, b), label=f"TF={tf}")
        ax.set(xlim=(.1, 4), ylim=(0, 2.5), xlabel="Document length / average length", ylabel="BM25 TF factor", title=f"Length normalization b={b:.1f}; k1=1.2")
        ax.legend(); figure.suptitle("Fixed TF; vary document length by adding non-query tokens")
        figure.text(.5, .03, "At b=0 length has no effect; increasing b penalizes longer documents at the same TF. No direct cosine-score comparison.", ha="center")
    export.gif("34_bm25_length_normalization.gif", 13, draw, size=(12, 6), interval=700)
    return {"documents": documents, "lengths": index.lengths.tolist(), "avgdl": index.avgdl,
            "rankings": {m: r["ranking"] for m, r in results.items()},
            "scores": {m: r["scores"].tolist() for m, r in results.items()}}


def mismatch(export):
    pairs = [("car", "automobile"), ("purchase", "buy"), ("refund", "reimbursement"),
             ("document", "docurnent"), ("neural networks", "deep learning")]
    documents = [value for pair in pairs for value in pair]
    values = {}
    for name, vectorizer in (("word TF-IDF", TfidfVectorizer()),
                              ("character TF-IDF", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)))):
        matrix = vectorizer.fit_transform(documents)
        scores = cosine_similarity(matrix)
        values[name] = [scores[2 * i, 2 * i + 1] for i in range(len(pairs))]
    figure, ax = plt.subplots(figsize=(13, 5)); x = np.arange(len(pairs))
    for shift, (name, scores) in zip((-.18, .18), values.items()):
        ax.bar(x + shift, scores, width=.36, label=name, color="#238457" if name.startswith("word") else "#8962ab")
    ax.set_xticks(x, [f"{a}\n↔ {b}" for a, b in pairs]); ax.set(ylim=(0, 1), ylabel="Cosine", title="Lexical overlap can fail on synonyms and recover part of an OCR corruption")
    ax.legend(); note(figure, "Dense embeddings were not loaded or scored. Character overlap is lexical; these pairs do not evaluate semantic retrieval quality.")
    export.png("35_vocabulary_mismatch.png", figure)
    return {"pairs": pairs, "scores": values, "dense": "not executed; no model download; no semantic similarity scores"}


def drift(export):
    initial, query, steps = core.drift_fixture()
    def draw(figure, frame):
        entry = steps[min(frame, len(steps) - 1)]
        ax = figure.add_subplot(121)
        x = np.arange(entry["N"])
        ax.bar(x - .2, entry["frozen_scores"], .4, label="Frozen IDF")
        ax.bar(x + .2, entry["scores"], .4, label="Refitted IDF")
        ax.set(ylim=(0, 1), xlabel="Document ID", ylabel="Cosine", title=f"Query '{query}'; added={entry['added']}"); ax.legend()
        right = figure.add_subplot(122); right.axis("off")
        lines = [f"N={entry['N']}; alpha DF={entry['alpha_df']}",
                 f"alpha IDF={entry['alpha_idf']:.4f}; beta IDF={entry['beta_idf']:.4f}",
                 f"D0 normalized alpha weight={entry['alpha_weight_d0']:.4f}",
                 f"Fresh order: {entry['ranking']}", "Appended documents: 'alpha novel'",
                 "Frozen vocabulary ignores previously unseen 'novel'.", "Each score uses a consistent query/document IDF version."]
        right.text(.02, .9, "\n\n".join(lines), va="top", fontsize=11)
        figure.suptitle("Corpus drift: freeze versus refit — index version consistency matters")
    export.gif("37_corpus_drift_animation.gif", 9, draw, size=(15, 7), interval=1300)
    return {"initial": initial, "query": query, "steps": steps}


def hybrid(export):
    index = core.SearchIndex(); query = "SYN-204 article 12 emissions"
    word = index.explain(query, "tfidf")["ranking"][:5]
    char = index.explain(query, "char")["ranking"][:5]
    ranking, scores = core.rrf([word, char])
    figure, ax = plt.subplots(figsize=(14, 5)); ax.axis("off")
    labels = ["Synthetic query", "Word TF-IDF\nactual candidates", "Character TF-IDF\nactual candidates", "RRF (constant=60)", "Top-3 context\nhypothetical LLM"]
    positions = [(0, .5), (.25, .8), (.25, .2), (.55, .5), (.8, .5)]
    for label, (x, y) in zip(labels, positions): ax.text(x, y, label, bbox=dict(boxstyle="round,pad=.7", fc="#eaf1f8"), ha="center", va="center", fontsize=11)
    for a, c in ((0, 1), (0, 2), (1, 3), (2, 3), (3, 4)):
        ax.annotate("", xy=positions[c], xytext=positions[a],
                    arrowprops=dict(arrowstyle="->", color="#65758b", shrinkA=55, shrinkB=55))
    ax.set(xlim=(-.12, 1), ylim=(-.1, 1.1), title="Lexical rank fusion for RAG context — no dense retrieval or generation")
    export.png("38_hybrid_retrieval_pipeline.png", figure)
    def draw(figure, frame):
        consumed = min(frame + 1, max(len(word), len(char)))
        order, current = core.rrf([word[:consumed], char[:consumed]])
        ax = figure.add_subplot(121); ax.axis("off")
        ax.text(.02, .9, f"Word list: {word}\n\nCharacter list: {char}\n\nConsumed rank positions: {consumed}\n\nEach contribution = 1/(60 + one-based rank)", va="top", fontsize=13)
        bars(figure.add_subplot(122), order, current, "Actual RRF accumulation", "#ad4569")
        figure.suptitle("Two lexical ranked lists; missing documents contribute zero")
    export.gif("39_reciprocal_rank_fusion.gif", 7, draw, size=(13, 6), interval=1200)
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, method in zip(axes[:2], ("tfidf", "char")):
        result = index.explain(query, method); bars(ax, result["ranking"][:5], result["scores"], method, core.COLORS[method])
    bars(axes[2], ranking, scores, "RRF — lexical + lexical", "#ad4569")
    note(figure, "Fictional SYN-204: no real legislation. RRF scores have a different scale; context selection is shown, no LLM answer is generated.")
    export.png("40_hybrid_retrieval_comparison.png", figure)
    return {"query": query, "word_candidates": word, "char_candidates": char, "rrf_ranking": ranking,
            "rrf_scores": scores, "selected_context_ids": ranking[:3], "dense": "not loaded; lexical fusion only"}


def evaluate(export):
    result = core.evaluation(); names = list(result["summary"])
    metrics = list(result["summary"][names[0]])
    figure, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, metric in zip(axes, metrics):
        values = [result["summary"][name][metric] for name in names]
        ax.bar(names, values, color=[core.COLORS[name] for name in names]); ax.tick_params(axis="x", rotation=45)
        ax.set(ylim=(0, 1.1), title=metric, ylabel="Macro mean over eight queries")
        for i, value in enumerate(values): ax.text(i, value + .02, f"{value:.2f}", ha="center", fontsize=8)
    note(figure, "k=3; Precision denominator is 3 even with fewer hits; MRR uses all positive hits; nDCG uses 2^grade−1. Educational judgments, not held-out production evidence.")
    export.png("41_retrieval_evaluation.png", figure)
    matrix = np.array([[row["Recall@k"] for row in result["per_query"][name]] for name in names]).T
    figure, ax = plt.subplots(figsize=(12, 7)); heatmap(ax, matrix, "Per-query Recall@3: inspect failures, not only averages", names, vmax=1)
    ax.set_yticks(range(len(core.JUDGMENTS)), [query for query, _ in core.JUDGMENTS], fontsize=9); ax.set_ylabel("Query")
    figure.tight_layout(); export.png("42_per_query_failure_analysis.png", figure)
    return result

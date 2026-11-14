# Visual preview curation

[Study methodology](methodology.md) · [Validation boundaries](validation.md)

Keep a small selection that answers distinct technical questions. An exact
algorithm trace, a measured comparison and an interactive explorer need not
share a format. Regenerability alone is not a reason to remove a useful preview.
Generated galleries, run records and alternate exports stay ignored by default;
publish only intentional files referenced by public documentation.

## Review budgets

These are local curation targets for new selections, not platform limits,
performance measurements or automatic deletion rules. KiB/MiB use powers of 1024.

| Asset or reading surface | Default review target | When exceeding it can be justified |
|---|---:|---|
| Static PNG | 500 KiB per file | Legible labels, uncertainty intervals or a dense comparison |
| Animated GIF | 1 MiB per file | Exact states or changing distributions that a still cannot explain |
| Topic README | 3 MiB of inline images total | Document the additional learning value; prefer links for secondary views |
| Selected images in one topic | 6 MiB total, including linked images | Review selection before adding redundant exports |
| Standalone offline HTML | 6 MiB per file, linked rather than embedded | Bundled runtime and controls materially improve exploration |

For an exception, record its size, learning question, generator and why a lighter
form is insufficient in the topic guide or a dated selection record. Do not
compress text into illegibility, omit algorithm states or change playback time
merely to meet a budget. A file-byte budget does not bound decoded memory use.

## Comparing candidates

Try lossless encoding in a separate ignored destination first. Before replacing
an image, compare dimensions and decoded pixels; for GIFs compare every composited
frame, frame count, per-frame durations and looping behavior. Inspect labels and
representative states visually. Keep attribution, color profiles and relevant
metadata. Preserve the original until the candidate is accepted and keep a
reproducible encoding step if adopting it.

An initial replacement should normally save both 10% and 64 KiB to justify
binary churn; a larger collection can justify a separately measured aggregate
saving. This is a review heuristic, not a requirement to use lossy compression.
Do not add another encoder dependency for an unmeasured gain.

An offline HTML intentionally bundles its runtime. Removing licenses or replacing
it with a CDN changes the artifact's contract. Gzip size measures a possible
transport/archive representation, not the size of a directly opened HTML.
Do not publish both versions without a concrete distribution need. Changed HTML
controls require browser verification; image fidelity does not verify JavaScript.

## Selected examples — November 13, 2026

The following selections were reviewed without regenerating their experiments.
Sizes are for the existing tracked files, not predicted network transfers.

| Topic | Selection and total bytes | Decision |
|---|---|---|
| [37: DBSCAN](../02-unsupervised-recommender-systems/37-dbscan-density-based-clustering/) | Two GIFs and one PNG; 2,972,582 bytes, all inline | Keep the self-inclusive neighborhood, exact expansion and shared-coordinate comparison. The 2,114,365-byte expansion GIF is an intentional exception: 68 displayed frames preserve engine states and pauses across 36 seconds. |
| [46: power analysis](../03-statistics-experimentation/46-power-analysis/) | Three GIFs and one PNG; 3,991,187 bytes; only 1,325,119 bytes inline | Keep the sampling-distribution preview inline; power growth, effective information and effect-size planning are complementary linked views. The three GIFs exceed 1 MiB each but preserve different questions without adding inline loading weight for all three. |
| [48: multiple testing](../03-statistics-experimentation/48-multiple-testing/) | Two GIFs, one PNG and one linked HTML; 5,770,120 bytes; 785,723 bytes inline | Keep BH's step-up states, the FWER/FDR/power comparison and the disclosed peeking path. The 4,984,397-byte offline surface remains an explicit download, not an inline image. |

The tested Pillow re-encoding preserved decoded pixels, all 247 GIF frames,
durations and looping, but saved no GIF bytes and increased the three PNGs by
2.5–4.8%. Originals were retained; no claim is made about all possible encoders.
The HTML compressed to 1,521,906 bytes in a reversible gzip trial, but the plain
standalone file was retained for direct local opening. No browser or network
performance benchmark was run for this curation, and interpretation-review
status is unchanged.

# Evaluate PP-OCR failures first

> Historical experiment report. The CPU matcher now has an optional
> [Rust server integration](glyph-fallback.md); statements below describe the
> experiment at the time it was measured. GPU matching remains experimental.


The optimization target is incremental recovery of inputs PP-OCR actually gets
wrong, rather than dictionary accuracy averaged over arbitrary glyphs. The
earlier evaluation already measured actual empty-output recovery, but its
font-level matching diagnostics mixed successful and failed model inputs.
`tools/glyph_fallback/failures.py` now extracts explicit cohorts from a frozen
model snapshot and its dictionary replay, checking IDs, labels, groups and
model predictions agree before joining them.

## Current baseline

The existing 4,353-input corpus yields:

| PP-OCR cohort | Inputs | Correct dictionary proposals | Wrong/unsafe proposals | Dictionary abstentions |
| --- | ---: | ---: | ---: | ---: |
| Labeled glyph, empty prediction | 84 | 14 | 0 | 70 |
| Labeled glyph, wrong nonempty prediction | 2,241 | 551 | 1 | 1,689 |
| Labeled glyph, correct prediction | 1,862 | 617 | 1 | 1,244 |
| Negative control, empty prediction | 106 | 0 | 0 | 106 |
| Negative control, nonempty prediction | 60 | 0 | 2 | 58 |

The primary failure corpus contains **2,325 inputs**. Only the 84 empty labeled
inputs are eligible for the current serving proposal: **14/84 = 16.7% recovery**,
zero observed wrong fills. Nonempty proposals are diagnostic; the implementation
preserves every nonempty prediction. Automatically substituting all dictionary
proposals would corrupt one currently correct model prediction and introduce
unsafe proposals on two negative controls in this corpus.

The corpus deliberately overrepresents unsupported vocabulary, difficult fonts
and synthetic variants. These counts are not production failure rates. Several
images share a character or source rendering and are not independent examples.
The one confirmed production failure is included; the rest do not establish
representative real-world coverage.

## Iteration policy

1. Prioritize the **70 unrecovered empty glyph inputs** when evaluating font,
   rasterization and retrieval changes. Break down recovery by source font and
   corruption, and distinguish normalization rejection from matching abstention.
2. Analyze the **2,241 wrong nonempty inputs** separately. Their 551 correct
   dictionary proposals identify potential value beyond empty-only fallback,
   but require a separately evaluated trigger for consulting/replacing model
   predictions. Ground-truth failure labels are unavailable during inference.
3. Replay all 1,862 correct glyphs and 166 negative controls with every candidate
   change. Report regressions and unsafe proposals alongside incremental recovery.
4. Keep mined inputs as evaluation fixtures, not dictionary templates. Once an
   example informs tuning, treat it as development data and validate gains on
   fresh failure inputs, with held-out fonts and source images. Keep transformed
   siblings together when creating future development/evaluation splits.

Dictionary expansion should be justified by these failure cohorts. A larger
Unicode inventory alone does not address the observed font-style failures.
This change organizes and measures the existing evidence; it does not train a
new dictionary, tune thresholds, or alter the HTTP server.

## Reproduce

```sh
python3 tools/glyph_fallback/failures.py \
  --queries tmp/cpu-glyph-v2/queries-default.json \
  --results tmp/cpu-glyph-perf/expanded.json \
  --output tmp/glyph-failures
```

The output includes `glyph_failures.json`, separate manifests for each cohort,
and `summary.json` with per-group metrics and SHA-256 hashes of both source
snapshots. Manifests retain image paths and model outputs and can be passed
directly to the existing CPU evaluator or GPU exporter. Paths refer to the
original experimental assets and must exist on the host used for replay.

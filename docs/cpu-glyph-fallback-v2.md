# CPU glyph fallback: production crop and synthetic-font iteration

> Historical experiment report. The CPU matcher now has an optional
> [Rust server integration](glyph-fallback.md); statements below describe the
> experiment at the time it was measured. GPU matching remains experimental.


> Update: [the latency optimization](cpu-glyph-fallback-performance.md) preserves
> these acceptance results while reducing production-crop matching from about
> 21 ms to 3.38 ms. The timings below describe the earlier v2 implementation.

The confirmed production crop is **岂 (U+5C82)**. The unchanged recognition
server returns an empty result under `all`, `cjk_focus` and `cjk_focus_fallback`,
although 岂 is present in its vocabulary. Its timesteps are all CTC blanks. This
is a recognition failure, not an out-of-vocabulary limitation.

The revised offline Rust prototype recovers this crop as 岂. Across **4,353
images**, replay against fresh default-policy model outputs recovers **14 of 84
empty labeled glyphs**, including **nine out-of-vocabulary instances**, with
**zero wrong fills**. There are another 106 empty negative/control images, all
left empty. All 4,163 nonempty model results are preserved. Nothing is enabled
in the HTTP server; the temporary GPU baseline containers have been stopped.

This is substantially better than the [initial v1 experiment](cpu-glyph-fallback.md),
which recovers zero at its fixed operating point on the same expanded corpus.
The production crop and synthetic cases demonstrate that the first font bank
and binary comparison were too restrictive. Synthetic glyphs are useful for
this task, particularly when font, rasterization and vocabulary coverage are
explicitly varied. They do not establish accuracy on every production source.

## Changes and controlled checks

The production crop is evaluation/development data only. It was **not** inserted
as a dictionary template. The intended character was confirmed by the user.

| Experiment on the production crop | Best result | Similarity | Outcome |
|---|---|---:|---|
| v1: thin sans + regular serif, binary Dice | 岜 | 0.6675 | Abstain; correct glyph not first |
| Add regular sans, otherwise retain v1 | 岂 | 0.8956 | Still below threshold |
| v2: regular sans + preserve antialiased coverage | 岂 | 0.9541 | Accept |

All three rows were checked against the entire dictionary, not just a shortlist.
The v2 runner-up is 岜 at 0.8780, a margin of 0.0761. Similarity is **not an OCR
probability**. The prototype retains thresholds 0.93 and 0.04, but the similarity
metric changed, so equal numeric thresholds do not imply equal calibration.

The asset generator now builds 62,933 valid templates from Noto Sans CJK SC
weights 100 and 400 and Noto Serif CJK SC weight 400. It checks font cmap support
and covers U+4E00–U+9FFF; Extension A and later extension blocks are not included.

Version 2 retains grayscale ink coverage through aspect-preserving 28×28 resizing
on a 32×32 canvas. It compares normalized ink vectors with cosine similarity
across nine ±1-pixel translations, retaining the previous aspect-ratio penalty.
This avoids destroying antialiased/thin strokes by binarizing before resizing.
Binary occupancy descriptors still retrieve a shortlist, now 256 templates.

Before accepting a proposal, the matcher checks the **whole dictionary** for
any template that could violate the required different-character margin. A
Cauchy–Schwarz upper bound on each 4×4 block prunes provably weaker comparisons.
It evaluates the bound over the same nine alignments, using the same full-image
norms as the pixel metric, with a small floating-point slack. Surviving templates
receive exact pixel comparison. Templates are collapsed by character before
acceptance, so another font of the same character is not the runner-up.

This verification protects acceptance from missing shortlist competitors. It
does not guarantee the shortlist retrieves every recoverable character: an
initial abstention skips verification and can remain an unnecessary abstention.
The returned diagnostic runner-up can also be a shortlist/verified-subset result;
the guaranteed fact for an accepted proposal is separation by **at least the
configured margin**, not necessarily an exact global runner-up score. Use
`--exhaustive` when an exact global ranking is required.

The JSON dictionary is schema version 2, with base64-encoded grayscale coverage.
Version 1 assets are rejected rather than silently interpreted under the new
metric. No serving dependency, native interface or public runtime flag was added.
Original model scores, IDs and diagnostics remain under `original_model`;
applied results use `cpu_template_v2` provenance and separate similarity metadata.

## Expanded evaluation

The existing 2,240 cases were retained. Another 2,112 synthetic cases use **528
labels**: a seeded sample of 256 model-supported and 256 unsupported characters
beyond the initial U+4E00–U+51FF range, plus targeted confusables including 岂/岜/岩.
The extra targets bring the supported subset to 272. Every label is rendered in
two held-out fonts and two variants:

- Droid Sans Fallback: neither its files nor its crops enter the dictionary.
- [LXGW WenKai Regular v1.522](https://github.com/lxgw/LxgwWenKai/tree/v1.522): a
  held-out calligraphic font, obtained from the project's TTF assets. The downloaded
  font and all rendered crops remain in ignored experiment storage.
- Clean 39 px glyphs and smaller 29 px glyphs with Gaussian blur 0.35 and JPEG
  quality 80. These are rendered independently from the template images.

The confirmed production crop makes **4,353 total cases**. Font and production
image hashes, random seed, sampled codepoints and labeling provenance are saved
in `tmp/cpu-glyph-v2/expanded/provenance.json`. Font-family and degradation results
are reported separately. The threshold was not tuned against the new 528-label
set. The production crop guided this iteration and is therefore a development
case, not an independent production holdout.

Fresh model baselines were captured with `score_mode=model`, timestep IDs and all
three character policies. The table uses default `cjk_focus_fallback`; model
scores under this policy are conditional probabilities, not the default API's
binary accepted scores.

| Group | Images | Empty model outputs | Correct recoveries | Wrong fills |
|---|---:|---:|---:|---:|
| Historical synthetic glyphs | 1,024 | 14 | 9 | 0 |
| Original held-out Droid set | 1,036 | 12 | 2 | 0 |
| Inverted confusables | 14 | 0 | 0 | 0 |
| New Droid, clean, model-supported | 272 | 4 | 1 | 0 |
| New Droid, clean, out-of-vocabulary | 256 | 10 | 1 | 0 |
| New Droid, small/JPEG, supported | 272 | 4 | 0 | 0 |
| New Droid, small/JPEG, out-of-vocabulary | 256 | 10 | 0 | 0 |
| New WenKai, both variants and vocabulary strata | 1,056 | 29 | 0 | 0 |
| Confirmed production 岂 | 1 | 1 | 1 | 0 |
| Negative/damage controls | 166 | 106 | 0 | 0 |
| **Total** | **4,353** | **190** | **14** | **0** |

The recoveries are 丩, 丷, 傤, 僩, 僯, 僽, 儎, 儧, 儽, another 丩, 僒, 矃, 罾 and 岂.
Fallback frequency is **14/4,353 = 0.322%**. It abstains on **176/190** model-empty
inputs. Correct labeled outputs rise from 1,862 to 1,876 out of 4,187 labeled
images. This denominator deliberately includes many unsupported labels; it is
not a general production accuracy estimate.

Matching-only diagnostics expose remaining risks even when the current model
happens not to be empty:

| Matching group | Accepted if empty | Wrong/unsafe proposals |
|---|---:|---:|
| Historical synthetic | 620/1,024 | 1: 兔→免 |
| Original held-out Droid | 348/1,036 | 1: 丣→戼 |
| New Droid, clean | 117/528 | 0 |
| New Droid, small/JPEG | 90/528 | 0 |
| New WenKai, clean | 1/528 | 0 |
| New WenKai, small/JPEG | 3/528 | 0 |

There is also one damaged-glyph proposal and one single-stroke punctuation
proposal among the negative controls. These inputs have nonempty model results,
so they cause no actual fallback substitutions in this run. The damaged case
is conservatively labeled for abstention, not proven unreadable; a single stroke
can have different intended roles despite identical pixels. These observations
argue against replacing nonempty outputs and against claiming zero general
false-positive risk from the observed zero wrong fills.

## Verification, cost and limitations

An exhaustive audit covered 51 cases: all 14 actual recoveries, every wrong or
unsafe matching-only proposal, inverted confusables, and 20 seeded sampled
abstentions. Every proposed acceptance agreed with exhaustive acceptance. Two
shortlist abstentions could have recovered the correct 偃 and 土 under exhaustive
matching. Eleven top-two rankings and three winners differed overall; safety
verification addresses acceptance, not complete retrieval recall.

On the same Ryzen 7 7800X3D, the optimized release build scored just the **190
actual empty inputs**, which represents the work an integrated gate would admit:

- Match median **1.19 ms**, p95 **16.05 ms**; accepted cases median **16.60 ms**,
  worst observed **67.31 ms**, including full-dictionary margin verification.
- Production 岂 matching: **20.86 ms**, plus approximately **0.095 ms** decode and
  normalization. This is CPU matching latency, not an HTTP measurement.
- All 190 cases together: **523 ms** decode/normalize/match, excluding dictionary
  load, amortized over the 4,353-input experimental corpus only.
- Dictionary load and descriptor/bound construction: **283 ms**.
- JSON dictionary: **108,013,538 bytes**. Peak process RSS: **184.7 MiB**, including
  asset parsing buffers and diagnostic results. This is not a steady-state RSS
  measurement. The larger asset stores grayscale samples, not only binary rows.
- No additional VRAM; matching remains standalone Rust CPU code. Native inference
  was used only to collect the unchanged baseline.

Pixel norms are cached and the aligned dot products use contiguous row slices.
The optimized path produces **identical candidates, scores and final texts** for
all 190 empty inputs compared with the earlier v2 scalar implementation. The full
4,353 diagnostic run used that equivalent scalar implementation; latency numbers
above come from the optimized gate workload. Neither is an integrated HTTP test.

All **18 Rust tests** pass: the previous 12 server tests and six prototype tests.
The added checks include upper-bound coverage over deterministic pixel patterns
and a competing character deliberately hidden outside the shortlist. Clippy
passes with warnings denied. Original predictions and nonempty text are preserved
across replay. Native code and model bundles were not changed.

This iteration supports continuing the dictionary approach. Its next weakness
is font-style coverage, especially calligraphic fonts, followed by small-raster
coverage and overly conservative retrieval. Expand template families and test
new independent families before enabling it; do not insert evaluation crops into
the dictionary or tune thresholds on the final test set. Any integration also
needs explicit CPU work/concurrency bounds and request-level latency measurements.
The present result justifies further offline work, not automatic deployment.

## Reproduction

Use the current `tools/glyph_fallback/prepare.py` and the build/evaluate commands
from the initial report to generate a **v2** dictionary; it now includes regular
sans automatically. Keep the original development manifest separately so prior
results can still be audited. Optional font download and expanded evaluation:

```sh
mkdir -p tmp/cpu-glyph-v2
curl -fL -o tmp/cpu-glyph-v2/LXGWWenKai-Regular.ttf \
  https://raw.githubusercontent.com/lxgw/LxgwWenKai/v1.522/fonts/TTF/LXGWWenKai-Regular.ttf

tmp/cpu-glyph/venv/bin/python tools/glyph_fallback/extend.py \
  --base-queries tmp/cpu-glyph/data/queries.json \
  --droid /path/to/DroidSansFallbackFull.ttf \
  --wenkai tmp/cpu-glyph-v2/LXGWWenKai-Regular.ttf \
  --vocabulary models/classifier/characters.txt \
  --production /path/to/confirmed-production-qi.jpg \
  --output tmp/cpu-glyph-v2/expanded

# Optional: capture model outputs from an already running isolated server.
python3 tools/glyph_fallback/probe.py pack tmp/cpu-glyph-v2/expanded/queries.json \
  tmp/cpu-glyph-v2/expanded-requests.json
python3 tools/glyph_fallback/probe.py run http://127.0.0.1:18185 \
  tmp/cpu-glyph-v2/expanded-requests.json tmp/cpu-glyph-v2/model-expanded.json
python3 tools/glyph_fallback/probe.py attach tmp/cpu-glyph-v2/expanded/queries.json \
  tmp/cpu-glyph-v2/model-expanded.json cjk_focus_fallback tmp/cpu-glyph-v2/queries-default.json

cargo build --release --locked --manifest-path server/Cargo.toml --example cpu_glyph_fallback
server/target/release/examples/cpu_glyph_fallback evaluate \
  /path/to/v2-dictionary.json tmp/cpu-glyph-v2/queries-default.json \
  tmp/cpu-glyph-v2/expanded-results.json
python3 tools/glyph_fallback/summarize.py tmp/cpu-glyph-v2/expanded-results.json \
  tmp/cpu-glyph-v2/expanded-summary.json
cargo test --locked --manifest-path server/Cargo.toml --all-targets
```

`extend.py` is a case-study fixture tool: `--production` assumes the user-confirmed
岂 label for this case. Do not pass another crop and silently reuse that ground
truth. All generated corpora, attachments, font assets and benchmark JSON remain
under ignored `tmp/` paths. Post-hoc threshold sweeps in the summary use recorded
candidates only; a different operating point must rerun global verification.

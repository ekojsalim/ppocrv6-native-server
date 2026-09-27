# CPU glyph dictionary exploration

> Historical experiment report. The CPU matcher now has an optional
> [Rust server integration](glyph-fallback.md); statements below describe the
> experiment at the time it was measured. GPU matching remains experimental.


> Historical v1 findings. The current executable and generator implement v2;
> see [the production-crop iteration](cpu-glyph-fallback-v2.md) for current
> behavior, measurements and additional reproduction steps. The results below
> describe the original two-font, binary-mask experiment.

Explored 2026-09-27. A CPU-only Rust matcher is practical to implement without
adding serving dependencies, but this experiment does **not** justify enabling
it in the HTTP server. The initial conservative operating point recovers no
empty results. Looser points recover some synthetic failures, with poor
cross-font coverage and a punctuation ambiguity. No production behavior changed.

The offline executable is `server/examples/cpu_glyph_fallback/main.rs`.
`tools/glyph_fallback/` prepares assets, captures optional model baselines,
replays predictions and summarizes results. These are research tools, not an
additional endpoint, startup setting, native interface or model dependency.

## Existing path and actual failures

The Rust glyph path decodes/composites images, preserves aspect ratio when
containing wide crops within 48×80, and passes a normalized tensor to the native
recognizer. Dictionary matching must use the original decoded crop, not the
native padding. Native `cjk_focus_fallback` already has a narrow horizontal
stroke fallback for 一 requiring both shape and model evidence. It is unchanged.

The saved extraction run used `character_policy=all`, `score_mode=model`:
559/1,024 filename labels were correct, 10 predictions were empty. Seven of
those empty labels are absent from the classifier vocabulary:

| Empty label | In model vocabulary? |
|---|---|
| 乚 U+4E5A | No |
| 亗 U+4E97 | No |
| 傤 U+50A4 | No |
| 僩 U+50E9 | Yes |
| 僯 U+50EF | No |
| 僽 U+50FD | Yes |
| 儎 U+510E | No |
| 儧 U+5127 | No |
| 儽 U+513D | Yes |
| 兓 U+5153 | No |

The original crops were inspected, rather than assuming empty meant blank.
A separate historical crop named U+8DDD (距) actually contains a horizontal bar;
it is a negative case, not a verified example of 距. The available 98 line crops
contain English text, so they supply no real isolated CJK ground truth. No
verified corpus of real failed CJK crops was found in these development assets.

A temporary, unmodified extraction-checkpoint server on the validation GPU
reproduced all 1,024 saved `all` texts. Fresh responses also include timestep
IDs and raw scores for every experiment image, with all three character policies:
`all`, `cjk_focus`, and `cjk_focus_fallback`. The 10 historical empties have only
CTC blank IDs. Under CJK policy, 丨/丩/丷/冫 additionally become empty; under `all`
the corresponding predictions are `1`/`4`/`11`/`1`. This distinguishes
policy-induced empties from the original CTC blanks. The default policy scores
560/1,024 and has 14 empty results on these historical fixtures.

`score_mode=accepted` is binary and is not a confidence trigger. The probe uses
`model` scores: unrestricted probabilities for `all`, conditional probabilities
for the restricted policies. The native one-stroke fallback did not apply on
this corpus. The historical bar already produces 一 under CJK policy; an
empty-only dictionary cannot correct that nonempty model result.

## Prototype

The builder creates a versioned JSON dictionary from labeled PNG templates.
It does not need fonts or Python when loading/matching the dictionary. Queries
retain their complete original prediction, including scores, IDs and diagnostics.
An actual replacement is possible **only** if a supplied model result is empty.
An absent model result is a matching-only experiment, never counted as recovery.
Output marks `cpu_template_v1` provenance, and similarity is explicitly not an
OCR probability. No classifier class ID or OCR probability is fabricated for a
dictionary-only character. Nonempty predictions, even low-scoring ones, stay intact.

The algorithm:

1. Composite alpha on white; estimate polarity from the border median. Reject
   low contrast, too little ink, border-touching ink, extreme aspect ratios,
   small marks and extreme foreground density.
2. Crop the foreground bounds and contain them within 28×28 on a 32×32 canvas,
   preserving aspect ratio. Store binary rows and original foreground aspect.
3. Retrieve 64 templates by squared distance between 8×8 occupancy descriptors.
4. Rerank with binary Dice overlap across nine ±1-pixel translations, minus
   `0.1 × min(abs(log(aspect_query/aspect_template)), 1)`.
5. Collapse templates by character before finding the runner-up. Require
   similarity ≥0.93 and a margin ≥0.04 over a **different character**. Abstain
   with fewer than two candidates. These constants preceded the first run;
   they are exploratory, not calibrated probabilities or public runtime knobs.

Original crops are retained privately. Bounding-box normalization discards
absolute character size; thin strokes, font weight, serif shape, lost ink and
pairs such as 口/囗 remain weaknesses. This experiment neither solves those
ambiguities nor uses contextual language evidence.

The current shortlist margin is approximate. An exhaustive audit of 102 cases
(all 10 original empties, strict acceptances, held-out cases scoring ≥0.8, and
inverted confusables) changed the top two characters in 36 cases and the winner
in three. One missed winner was the correct 儎 for a historical failure. No
accept/reject decisions changed at the fixed 0.93/0.04 point. This does **not**
prove retrieval safety at other operating points or on unseen inputs. An
integration would need reliable competitor retrieval or exact verification
before accepting; simply increasing a margin cannot repair a missing competitor.

## Corpus and outcomes

Templates cover font-supported U+4E00–U+9FFF using Noto Sans CJK SC weight 100
and Noto Serif CJK SC weight 400, at 48 px on 64 px canvases. There are 41,968
rendered templates; nine rule-like shapes are rejected, leaving 41,959.
The generator checks font cmap coverage to avoid adding missing-glyph boxes.
Font paths, face indices, weights and SHA-256 hashes are recorded privately.

The 1,024 historical images are development fixtures, **not** a held-out font
claim: their original rendering provenance is unavailable and their appearance
resembles the thin template font. New held-out images use Droid Sans Fallback,
39 px on 57 px canvases; that font is absent from the dictionary. No evaluation
crop is inserted as a dictionary entry. Original HTTP response hashes and
per-image hashes are recorded with the experiment.

| Evaluation group | Images | Empty under default CJK policy | Strict dictionary proposals |
|---|---:|---:|---:|
| Historical synthetic glyphs | 1,024 | 14 | 22, all correct but already nonempty |
| Held-out font, including confusables | 1,036 | 12 | 0 |
| Inverted held-out confusables | 14 | 0 | 0 |
| Missing-stroke stress cases | 14 | 0 | 0 |
| Clipped stress cases | 14 | 0 | 0 |
| Punctuation and rules | 32 | 3 | 0 |
| Uniform blank images | 5 | 5 | 0 |
| Seeded noise | 100 | 98 | 0 |
| Historical mislabeled bar | 1 | 0 | 0 |
| **Total** | **2,240** | **132** | **22** |

Confusables explicitly include 未/末, 土/士, 口/囗, 日/曰, 己/已/巳 and 人/入/八.
Damage cases conservatively expect abstention; they are not adjudicated claims
that no human could recognize the character. Punctuation/rule cases include
single strokes that can also represent a CJK character. In particular, the
negative dot and positive 丶 have identical pixels: context-free image matching
cannot distinguish their intended roles.

At the fixed point, default-policy replay applies **0/2,240** fallbacks and
abstains on all 132 empties. It recovers zero errors and introduces zero wrong
fills. All 1,137 already-correct labeled predictions remain correct; all 2,108
nonempty predictions remain unchanged. The original `all` historical run stays
559/1,024. These no-regression counts partly follow from the empty-only gate;
they are not evidence that template predictions could safely replace nonempty OCR.

The following is a **post-hoc sensitivity analysis on this same corpus**, not
independent calibration. Margin stays at 0.04:

| Minimum similarity | Recovered empty results, default policy | Unsafe fills on default-policy empties | Held-out-font proposals if every image were empty | Wrong held-out-font proposals |
|---|---:|---:|---:|---:|
| 0.93 (prototype constant) | 0 | 0 | 0 | 0 |
| 0.90 | 1 | 0 | 2 | 0 |
| 0.85 | 3 | 0 | 8 | 1 |
| 0.80 | 9 | 1 | 46 | 5 |

At 0.90, 儧 is recovered. At 0.85, 亗 and 傤 are recovered as well. The 0.80
unsafe fill is the negative dot interpreted as 丶, not a wrong known positive
label. Under `all`, the corresponding recovery counts are 0/1/3/7, with no
unsafe empty fills. The incorrect forced-empty proposal at 0.85 is held-out
口→曰; the same failure occurs after polarity inversion. At 0.80 there is also
an unsafe damaged-glyph proposal. Those inputs currently have nonempty model
results, so the gate protects them in actual replay. There is no evidence that
future model failures would preserve that protection.

## CPU and memory

Measured locally on an AMD Ryzen 7 7800X3D with Rust 1.96 release builds, using
one thread. The first 2,240-query run reports:

- Historical-glyph matching median **0.585 ms**, p95 **0.598 ms**; decode and
  normalization median **0.041 ms** separately.
- Held-out matching median **0.573 ms**, p95 **0.653 ms**; decode/normalization
  median **0.065 ms**. Matching times exclude normalization rejections.
- Dictionary loading and descriptor construction: **41.5 ms**.
- Dictionary JSON: **14,159,151 bytes**; template structs: **7,720,456 bytes**
  excluding string allocations; descriptors: **2,685,376 bytes**.
- `/usr/bin/time -v` peak RSS: **41.6 MiB** for the initial harness and **48.2 MiB**
  for replay with complete model/timestep diagnostics. These include retained
  queries/results and parsing buffers, not just the dictionary's resident size.
- Exhaustive matching median: **26.2 ms** on the selected 102-case audit, versus
  about 0.6 ms for retrieval plus reranking. It is too expensive to silently
  substitute exhaustive matching on every image of a large request.

The harness intentionally scores nonempty images for evaluation. A future
serving path would skip them; do not read the all-image harness throughput as
end-to-end HTTP fallback performance. No HTTP integration latency was measured.
The Rust matcher does not load native inference or initialize CUDA and needs no
additional VRAM. GPU use during this investigation was only baseline capture by
the unchanged server; after stopping it, the node returned to 14 MiB idle usage.

## Library tradeoffs and next decision

The current implementation uses the existing `image`, `serde`, `serde_json` and
`anyhow` dependencies. Its small binary masks make Rust bit operations a useful
bounded baseline. It has no new C++ FFI or font-rendering dependency at serving time.

[imageproc's template matching methods](https://docs.rs/imageproc/latest/imageproc/template_matching/enum.MatchTemplateMethod.html)
include normalized correlation and normalized squared error; this is a sensible
pure-Rust option for a later grayscale comparison. Its sliding-template API
still leaves normalization, dictionary retrieval and abstention policy to us.
[OpenCV's template matching](https://docs.opencv.org/4.13.0/d4/dc6/tutorial_py_template_matching.html)
also supplies sliding similarity methods, but using the project's native
OpenCV dependency from Rust would add a binding or a CPU C++ interface. Neither
library was added or performance-benchmarked here. API availability alone is not
evidence that either fixes cross-font accuracy.

Keep the prototype offline. Before integrating, obtain verified real failure
crops and representative successful crops, split calibration and evaluation by
font/source, and test a stroke-width-tolerant comparison (for example symmetric
edge distance) alongside grayscale correlation. Verify candidate recall and
runner-up recall separately. A future Rust path should preserve original crops,
apply after native inference only to empty results, keep provenance/similarity
separate from model scores, and release the native recognizer lock before CPU
matching. Request concurrency and CPU work still need explicit bounds. Do not
extend to low-confidence nonempty replacements on this evidence.

## Reproduction and validation

Private outputs are under `tmp/cpu-glyph/`, including original failure contact
sheets, dictionary and manifests, query crops, initial and policy-specific JSON
results, threshold summaries, exhaustive audit and process memory logs. Model
captures and server logs also remain in the remote experiment directory.
None of these corpora/assets are source files or committed. Font generation used
Pillow 12.3.0 and fonttools 4.66.0 in a temporary virtual environment.

From the project root, with local font/fixture paths filled in:

```sh
uv venv tmp/cpu-glyph/venv
uv pip install --python tmp/cpu-glyph/venv/bin/python pillow==12.3.0 fonttools==4.66.0

tmp/cpu-glyph/venv/bin/python tools/glyph_fallback/prepare.py \
  --sans /path/to/NotoSansCJK-VF.ttc \
  --serif /path/to/NotoSerifCJK-VF.ttc \
  --heldout /path/to/DroidSansFallbackFull.ttf \
  --fixtures /path/to/synthetic-cjk-glyphs/rgb \
  --predictions tmp/validation/http/glyph1024.json \
  --vocabulary models/classifier/characters.txt \
  --output tmp/cpu-glyph/data
# Optional: --bar-fixture /path/to/visually-verified-historical-bar.png

cargo build --release --locked --manifest-path server/Cargo.toml --example cpu_glyph_fallback
server/target/release/examples/cpu_glyph_fallback build \
  tmp/cpu-glyph/data/templates.json tmp/cpu-glyph/dictionary.json
server/target/release/examples/cpu_glyph_fallback evaluate \
  tmp/cpu-glyph/dictionary.json tmp/cpu-glyph/data/queries.json tmp/cpu-glyph/results.json
python3 tools/glyph_fallback/summarize.py tmp/cpu-glyph/results.json tmp/cpu-glyph/summary.json

python3 tools/glyph_fallback/audit.py select tmp/cpu-glyph/results.json \
  tmp/cpu-glyph/data/queries.json tmp/cpu-glyph/audit-queries.json
server/target/release/examples/cpu_glyph_fallback evaluate \
  tmp/cpu-glyph/dictionary.json tmp/cpu-glyph/audit-queries.json \
  tmp/cpu-glyph/audit-results.json --exhaustive
python3 tools/glyph_fallback/audit.py compare tmp/cpu-glyph/results.json \
  tmp/cpu-glyph/audit-results.json tmp/cpu-glyph/audit-summary.json
```

Optional fresh model capture against an already running isolated test server:

```sh
python3 tools/glyph_fallback/probe.py pack tmp/cpu-glyph/data/queries.json tmp/cpu-glyph/requests.json
python3 tools/glyph_fallback/probe.py run http://127.0.0.1:18185 \
  tmp/cpu-glyph/requests.json tmp/cpu-glyph/model-results.json
python3 tools/glyph_fallback/probe.py attach tmp/cpu-glyph/data/queries.json \
  tmp/cpu-glyph/model-results.json cjk_focus_fallback tmp/cpu-glyph/queries-default.json
server/target/release/examples/cpu_glyph_fallback evaluate tmp/cpu-glyph/dictionary.json \
  tmp/cpu-glyph/queries-default.json tmp/cpu-glyph/results-default.json
python3 tools/glyph_fallback/summarize.py tmp/cpu-glyph/results-default.json tmp/cpu-glyph/summary-default.json
```

The packed request file and standard-library probe can run on the GPU node
without transferring fonts or changing its server installation. This experiment
used a temporary extraction-checkpoint container bound only to localhost, then
stopped it. A stale system CDI file referenced old driver libraries; a separate
CDI file was generated privately for this container. System configuration and
existing services were not modified.

Checks passed: the existing 12 Rust tests, four new example tests (polarity and
alpha, blank/rule/clipping rejection, different-character ambiguity, empty-only
replay), and Clippy with warnings denied for the example. All Python tools were
executed on the corpus and compile-checked. Native code and model bundles were
not changed; native/exporter tests were not rerun for this Rust/offline-only work.

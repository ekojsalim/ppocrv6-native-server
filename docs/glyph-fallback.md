# CPU glyph recovery

The Rust server can recover an empty isolated-glyph result with a CPU template
dictionary. Page and line OCR are unaffected. Existing native recognition,
including the narrow 一 fallback, runs first. Nonempty predictions are never
replaced, even when their model score is low.

## Enable and package

Put a schema-v2 dictionary at `MODEL_DIR/glyph-fallback/dictionary.json` before
starting the server. No additional server flag or serving dependency is needed.
Without that file, behavior is unchanged. An invalid installed dictionary is a
startup error. `/v1/glyphs/info` reports whether CPU fallback is enabled.

The dictionary is an offline-generated asset, excluded from Git. The packaging
helper validates the asset, removes development-machine font paths, copies font
license notices and adds SHA-256 entries to the existing model manifest:

```sh
python3 tools/package_glyph_dictionary.py /path/to/dictionary.json models \
  --font-license /path/to/sans/LICENSE --font-license /path/to/serif/LICENSE
```

To generate a new dictionary without any evaluation corpus, install Pillow and
fonttools in an offline build environment and supply the Noto variable TTC files
(the generator uses SC face index 2):

```sh
python3 tools/glyph_fallback/prepare.py --templates-only \
  --sans /path/to/NotoSansCJK-VF.ttc --serif /path/to/NotoSerifCJK-VF.ttc \
  --output tmp/glyph-dictionary
cargo build --release --locked --manifest-path server/Cargo.toml --example cpu_glyph_fallback
server/target/release/examples/cpu_glyph_fallback build \
  tmp/glyph-dictionary/templates.json tmp/glyph-dictionary/dictionary.json
```

Then package `tmp/glyph-dictionary/dictionary.json` using the command above.
The packager rejects an existing `glyph-fallback` directory. Use a fresh bundle for an
update. Restart the server when changing the dictionary. Font rasterization and
Python are offline tools only. See the [dictionary experiment](cpu-glyph-fallback-v2.md)
for generation and evaluation; its Rust example uses the same matcher as serving.
Retain font attribution appropriate to the fonts used in each asset.

## Response contract

Only predictions with empty `text` are eligible. Applied predictions have:

- Recovered `text` and `cpu_fallback.source = "cpu_template_v2"`.
- `original_model` containing the complete original native prediction.
- A separate `cpu_fallback.similarity`, which is **not a probability**.
- `score = 1` in `accepted` mode; `score = null` in `model` mode, with
  `score_type = "unavailable"`. No OCR probability is invented.
- Empty `class_ids` and `per_char_scores`: template recovery has no CTC alignment
  and can recover characters outside the model vocabulary. Original timestep
  diagnostics, when requested, remain under `original_model`.

Clients using `model` mode must handle the nullable score for recovered glyphs.
The single-image `prediction` alias and the batch `predictions` array agree.
Unchanged nonempty predictions retain their existing fields exactly. Empty
abstentions receive `cpu_fallback` metadata describing rejection or skipped work.
The response-level `cpu_fallback` reports attempted/applied/skipped counts and
CPU-stage elapsed time. Native timing and fallback counters retain their original
meaning; Rust blocking time includes the CPU stage.

## Work and memory bounds

One CPU fallback request runs at a time. The native recognizer mutex and shared
GPU inference permit are released before matching. If the CPU slot is busy,
empty results remain empty with `skipped = "busy"`; requests do not queue for it.
Each request attempts at most 32 empty glyphs, in input order. A 100 ms work budget
is checked before starting each attempt. An in-progress match finishes its full
competitor verification; the budget is not a hard response deadline. Remaining
empty glyphs report `request_limit` or `work_budget` rather than an unchecked guess.

The dictionary/index is loaded once and shared. The supported asset is bounded
to 128 MiB on disk and 100,000 templates, each with a 32×32 coverage image. The
current 62,933-template dictionary covers 20,990 characters in U+4E00–U+9FFF,
using three Noto SC renderings. It does not cover arbitrary fonts or all CJK
blocks; line-like characters are deliberately rejected by normalization. Existing
request byte/pixel limits apply before fallback. Only eligible images are decoded
again, avoiding retention of every full-resolution crop in a large request.

No extra VRAM is allocated. Host memory and request latency must be measured with
the complete server; standalone matcher peak RSS is not server steady-state RAM.

## Evidence and limitations

The extracted shared matcher reproduces all 4,353 offline decisions: 14 correct
recoveries among 84 empty labeled glyphs, no observed wrong fills, and no fills on
106 empty negative controls. These are development/held-out-font results, not a
production error-rate estimate. The confirmed 岂 crop guided development.
Known ambiguous proposals and weak calligraphic-font coverage remain. Low-score
nonempty replacement and GPU dictionary matching are experiments, not enabled
serving features.

## Integrated validation (2026-09-27)

On the RTX 5070 Ti / Ryzen 7 9700X node, using the packaged dictionary:

- All 21 Rust tests and Clippy with warnings denied pass.
- A fresh dictionary build with `--templates-only` reproduces all 62,933 evaluated
  templates exactly. Full offline replay preserves all 4,353 decisions.
- HTTP replay of the 190 originally empty inputs recovers the same 14 glyphs with
  zero wrong fills. The confirmed 岂 crop succeeds under all four character
  policies and both score modes, with consistent single/batch aliases.
- Median of 20 warmed single-crop requests: **2.82 ms HTTP**, including
  **2.18 ms CPU fallback**. These are integrated measurements.
- A 40-copy recovery request processes 32 glyphs and explicitly skips eight;
  concurrent requests exercise the busy-skip path while page OCR succeeds.
- All previously nonempty texts remain unchanged across 1,024 glyphs, 98 line
  crops and 98 page lines. Small/short/tall page smoke tests pass.
- The same image without a dictionary reports fallback disabled and preserves
  the original empty 岂 result without fallback fields.

Ten-request warmed benchmark medians were **164.4 ms per page** with the dictionary
and **164.2 ms** without it. The 1,024-glyph fixture batch takes **74.4 ms** versus
**54.3 ms**, reflecting the additional recovery work on empty glyphs. These are
sequential runs, not a broad statistical performance study.

After catalog warmup, both runs report **957 MiB GPU memory**. Sampled process
RSS is approximately **1,372 MiB enabled** versus **1,224 MiB disabled**. The
approximately 148 MiB difference includes allocator/workload effects (the enabled
run also exercised concurrent recovery), not a precise dictionary-only allocation.
Both isolated validation containers were stopped after testing.

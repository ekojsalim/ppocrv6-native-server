# CPU glyph fallback latency optimization

> Historical experiment report. The CPU matcher now has an optional
> [Rust server integration](glyph-fallback.md); statements below describe the
> experiment at the time it was measured. GPU matching remains experimental.


The production 岂 crop now matches in **3.38 ms**, down from **20.81 ms**
(6.2× faster), while retaining full-dictionary verification before acceptance.
The median accepted fallback is **3.49 ms** instead of **16.64 ms**. No model,
threshold, font, dictionary schema or HTTP behavior changed.

This optimizes the [v2 offline prototype](cpu-glyph-fallback-v2.md), not the native
PP-OCR recognizer. The fallback remains outside the HTTP path. Its cost would
only be paid after an empty model result; no nonempty-result replacement was
introduced.

## Where the time went

Phase timings in the example now separate retrieval, shortlist comparison and
global verification. On the production crop, the old implementation spent
approximately 0.80 ms retrieving candidates, 0.41 ms scoring the shortlist and
19.35 ms verifying separation from every other character. The loose 4×4-block
bound admitted 3,052 templates to expensive nine-alignment pixel comparison.
For the slowest empty case, it admitted 32,246 templates and spent about 67 ms
in verification alone. Normalization was not the main cost.

The optimized verifier uses two levels of Cauchy–Schwarz bounds: first 4×4
blocks, then tighter 2×2 blocks. Block energies are normalized and quantized
**upward** to fixed-point integers at scale 16,384, so rounding does not narrow
the upper bound. The comparison threshold is rounded downward and retains the
previous numerical slack. SSE2 integer multiply-add evaluates these bounds on
x86_64; other architectures use the scalar integer implementation. SSE2 is part
of the x86_64 baseline, so this does not require an AVX-specific build or change
the compiler target.

Each alignment is screened separately. Only alignments whose bounds can reach
the acceptance boundary proceed to exact pixel comparison. The query shifts in
the bound are mapped to opposite template shifts in the pixel matcher. Templates
proven too weak cannot change acceptance and need no exact comparison. The
production crop now needs pixel comparisons for only a handful of templates,
rather than thousands.

The pixel metric itself is unchanged. The shortlist still contains 256 entries,
and the similarity threshold and different-character margin remain 0.93 and
0.04. There is no new timeout, approximate acceptance, smaller dictionary,
thread-level parallelism or repeated-input cache hiding the cost.

## Paired measurements

Three sequential, alternating before/after runs used the same dictionary and
190 actual empty inputs from the 4,353-image evaluation set. Binaries ran on the
same Ryzen 7 7800X3D, with release builds and one thread. These are internal CPU
match times, excluding image decode/normalization and dictionary loading.

| Metric | Before | After |
|---|---:|---:|
| Production 岂, median of three | 20.81 ms | **3.38 ms** |
| Accepted fallback median | 16.64 ms | **3.49 ms** |
| Accepted fallback p95 | 64.83 ms | **5.21 ms** |
| All empty inputs, median | 1.20 ms | **1.24 ms** |
| All empty inputs, p95 | 16.01 ms | **3.24 ms** |
| Maximum across paired runs | 68.99 ms | **6.05 ms** |
| Mean total matching for all 190 empties | 516.54 ms | **199.21 ms** |

The common cheap abstention path is essentially unchanged; the gain is in
verification of possible recoveries and the latency tail. The new total is
about **0.046 ms CPU matching per original input**, amortized over this particular
4,353-image corpus. That is an aggregate work estimate, not the latency of an
individual request or a general production empty-rate prediction.

This remains slower than the previously measured 0.60 ms single-glyph GPU HTTP
baseline. That model measurement came from the GPU validation setup, whereas
these are local CPU matching measurements, so they are not a controlled combined
HTTP benchmark. A request needing a fallback still pays a few milliseconds;
ordinary nonempty predictions would skip this work completely.

The tradeoff is additional precomputed CPU data. Verification tables grow from
16,110,848 bytes to **40,277,120 bytes** (24,166,272 additional bytes, about 23 MiB).
Median dictionary load/index construction grows from 287 ms to **653 ms**.
These are startup costs and the tables should live across requests. The JSON
asset is unchanged. The after-run peak process RSS was **185.1 MiB**, including
parsing buffers, index and diagnostic output; it is not a steady-state server
memory measurement. Peak RSS can stay similar despite a larger live index
because transient JSON parsing dominated the old peak. No VRAM is used by this
matcher and no new dependency was added.

## Measurement on steven-pc

Repeated on `steven-pc`, an **AMD Ryzen 7 9700X**, using the exact same release
binaries, dictionary and 190 empty-result crops. All transferred files were
SHA-256 verified. Five sequential paired passes alternated execution order;
no GPU server was started and existing services were not changed. Host load
averages were approximately 0.5 on this 16-thread CPU.

| CPU matching metric on steven-pc | Before | After |
|---|---:|---:|
| Production 岂, median of five | 11.59 ms | **2.57 ms** |
| Accepted fallback median | 8.44 ms | **2.42 ms** |
| Accepted fallback p95 | 42.50 ms | **4.82 ms** |
| All empty inputs, median | 0.84 ms | **0.86 ms** |
| All empty inputs, p95 | 7.99 ms | **2.27 ms** |
| Maximum across paired runs | 43.32 ms | **4.88 ms** |
| Mean total matching for all 190 empties | 314.97 ms | **142.26 ms** |

Production-crop decode and normalization add a median **0.086 ms**; combined
CPU decode/normalize/match is **2.66 ms**. Matching alone is about **24% faster**
than the 3.38 ms measurement on the local 7800X3D, and **4.5× faster** than the
previous matcher running on steven-pc itself. All five runs preserve the same
14 recoveries, every accept/abstain decision and every accepted winner/score.

Dictionary load/index construction takes a median **529 ms**, excluded from
per-image matching. Peak standalone-process RSS is **185.3 MiB**. The dictionary
should remain loaded across requests; these numbers do not measure an integrated
HTTP fallback or combined GPU-plus-CPU request latency.

Local copies of the remote results, environment snapshot and file hashes are in
`tmp/cpu-glyph-perf/steven-pc/`; the isolated remote directory is
`/home/ekojs/Dev/ppocrv6-native-server/tmp/cpu-glyph-perf-20260927/`.

## Correctness and scope

Full replay of **all 4,353 cases** has identical:

- Actual fallback decisions and final text: **14 recoveries, zero wrong fills**.
- Matching-only accept/abstain decisions, including the known unsafe proposals
  documented in the v2 report.
- Accepted candidate character, source and pixel similarity score.
- Original model predictions and every nonempty final prediction.

Below-margin diagnostic rankings differ in 465 cases because weaker candidates
are now pruned earlier. They are not promised to be exact global rankings and
were not exact in the prior implementation either. The full-dictionary margin
check for accepted results is preserved. Use `--exhaustive` for an exact global
ranking; it continues to use the unchanged all-alignment pixel metric.

All 18 Rust tests pass, and Clippy passes with warnings denied. Bound tests
compare the SIMD dot product against a wide scalar reference, cover every
alignment, stress concentrated energy and image/block boundaries, and verify
that a different character deliberately omitted from the shortlist still blocks
acceptance. Quantization rounds outwards rather than relying on a calibrated
approximate screening threshold.

Further savings would need profiling in an integrated server. Reusing the
already-decoded image would avoid duplicate decode work, and an explicitly
bounded exact-input cache could help repeated crops if the real workload repeats
them. Neither benefit is included in these measurements. The remaining roughly
1.2 ms retrieval/shortlist floor and 2–4 ms verification cost are more useful
next targets than reducing model accuracy or weakening competitor checks.

## Reproduction

The subsequent [GPU experiment](glyph-fallback-gpu.md) measures 0.446 ms for
the same production crop on steven-pc, with 108 MiB of observed device memory
allocated after context initialization and unchanged corpus decisions.

The reusable paired benchmark is `tools/glyph_fallback/benchmark.py`. Preserve
the earlier executable before rebuilding; both executables must support the v2
asset and the same acceptance semantics. Generate an empty-only manifest from
fresh model snapshots for realistic gating. Local artifacts from this run are
under ignored `tmp/cpu-glyph-perf/`, including the preserved baseline source and
binary, phase profile, full-corpus replay, paired JSON runs and memory report.

```sh
cargo build --release --locked --manifest-path server/Cargo.toml --example cpu_glyph_fallback
python3 tools/glyph_fallback/benchmark.py \
  --before /path/to/preserved-v2-baseline \
  --after server/target/release/examples/cpu_glyph_fallback \
  --dictionary tmp/cpu-glyph-v2/dictionary.json \
  --queries tmp/cpu-glyph-v2/empty-queries.json \
  --output tmp/cpu-glyph-perf/paired --repeats 3
cargo test --locked --manifest-path server/Cargo.toml --all-targets
cargo clippy --locked --manifest-path server/Cargo.toml --example cpu_glyph_fallback -- -D warnings
```

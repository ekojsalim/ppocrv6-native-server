# GPU glyph fallback experiment

> Historical experiment report. The CPU matcher now has an optional
> [Rust server integration](glyph-fallback.md); statements below describe the
> experiment at the time it was measured. GPU matching remains experimental.


Measured on steven-pc on 2026-09-27: Ryzen 7 9700X, RTX 5070 Ti 16 GB,
driver 615.71.09, CUDA compiler 13.2.51. This is a standalone prototype;
the HTTP server is unchanged.

## Latency

The unchanged 62,933-template dictionary and acceptance rules from the
[CPU experiment](cpu-glyph-fallback-performance.md) run substantially faster
with CUDA offload on this machine:

| Warm matching, including GPU transfers and synchronization | CPU | GPU |
| --- | ---: | ---: |
| Production 岂 crop, median | 2.572 ms | **0.446 ms** |
| Accepted fallback inputs, median | 2.423 ms | **0.401 ms** |
| Accepted fallback inputs, p95 | 4.819 ms | **0.486 ms** |

The production GPU p95 is 0.460 ms. CPU decode/normalization took 0.077 ms
and query preparation another 0.048 ms in a separate export measurement.
Adding those to matching gives an estimated **0.571 ms** for the fallback.
That sum is not a measured integrated pipeline or HTTP latency. It excludes
the preceding PP-OCR inference.

GPU results use two complete warmup passes followed by 20 measured passes of
the 190 actual empty-model inputs (130 pass normalization). CPU figures are
from the earlier five paired runs on the same host. Timings are host wall
time, including pinned-buffer staging, transfers, kernel launches, CPU
candidate selection and synchronization. The GPU was otherwise idle;
contention with a serving OCR model and concurrent requests remain unmeasured.

## VRAM

| Device storage | Bytes | MiB |
| --- | ---: | ---: |
| Dictionary pixels, descriptors, bounds, norms and aspects | 109,251,688 | 104.190 |
| Reusable single-query scratch | 511,348 | 0.488 |
| Explicit allocations | 109,763,036 | **104.678** |
| Observed free-memory reduction after context initialization | 113,246,208 | **108.000** |

Budget approximately **110 MiB incremental VRAM** when sharing an existing
OCR CUDA context. This is an estimate for integration, based on allocations
measured after context initialization, not a measurement inside the server.
The dictionary stays resident; Unicode labels stay on the CPU.

A fresh standalone process peaked at 353 MiB total GPU memory versus 14 MiB
idle: **339 MiB above idle**, including its separate CUDA context/runtime.
This sampled process footprint should not be added unchanged to a server
that already owns a CUDA context. GPU memory returned to 14 MiB after exit.
Fresh context initialization took 198 ms; reading/uploading the exported
binary dictionary took 33 ms, both excluded from warm query timing.

## Implementation and validation

`server/examples/cpu_glyph_fallback/gpu_export.rs` exports the existing Rust
normalization, descriptors and outward-rounded bounds into experiment-only
binary tables. `tools/glyph_fallback/gpu_bench.cu` uploads them once, computes
dictionary descriptor distances on the GPU, selects 256 candidates on the CPU,
scores their nine alignments on the GPU, then verifies provisional accepts
against full-dictionary competitors using the existing coarse/fine bounds.
The CPU collapses scores by character and applies the same thresholds.
This hybrid implementation still transfers distance/score arrays to the CPU.

Integer dot products and FP32 scoring are used, with fused multiply-add
disabled. All **4,353 matching-only accept/abstain decisions and accepted
characters match the CPU reference**. Accepted score differences in serialized
JSON are below 5e-10. The 190 actual empty inputs yield the same **14 correct
recoveries and zero wrong fills** in every repeat. The known unsafe proposals
on hypothetical empty inputs remain unchanged; this does not establish
universal safety or improve accuracy beyond the CPU prototype.

All 18 Rust tests and Clippy with warnings denied pass. CUDA memcheck on three
representative queries reports zero errors with `--report-api-errors explicit`.
The default extended mode instead counts an internal driver warning,
“Selective device code recompilation in progress,” as one error, with no
invalid memory access reported. Both logs are retained. NVIDIA documents the
difference between internal logging and explicit API-return checks in its
[Compute Sanitizer reporting options](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html).

## Reproduction and artifacts

With CUDA 13.2 and the v2 dictionary/query manifests available:

```sh
cargo build --release --locked --manifest-path server/Cargo.toml --example cpu_glyph_fallback
server/target/release/examples/cpu_glyph_fallback export-gpu \
  tmp/cpu-glyph-v2/dictionary.json tmp/cpu-glyph-v2/empty-queries.json \
  tmp/gpu-glyph/data
nvcc -O3 -std=c++17 -arch=sm_120 --fmad=false \
  tools/glyph_fallback/gpu_bench.cu -o tmp/gpu-glyph/gpu_bench
tmp/gpu-glyph/gpu_bench tmp/gpu-glyph/data 20 > tmp/gpu-glyph/final.jsonl
python3 tools/glyph_fallback/gpu_summary.py \
  --results tmp/gpu-glyph/final.jsonl \
  --manifest tmp/gpu-glyph/data/manifest.json \
  --cpu-results tmp/cpu-glyph-perf/steven-pc/paired/after-0.json \
  --output tmp/gpu-glyph/summary.json
```

For a full parity replay, export all queries and compare against the matching
full CPU reference. Timing the export on the target host is necessary for
meaningful CPU preparation estimates. The benchmark accepts 1–100 repetitions.

Actual runs used the cached `localhost/ppocrv6-compiled:builder-final` container,
with private NVIDIA CDI metadata; no system configuration changes were needed.
Local raw results, exported manifests, memory samples, summaries and sanitizer
logs are under ignored `tmp/gpu-glyph/`. The remote experiment is under
`/home/ekojs/Dev/ppocrv6-native-server/tmp/gpu-glyph-experiment/`.
`all-results.jsonl` and `all-summary.json` cover all 4,353 inputs;
`final.jsonl` and `summary.json` cover the 20 empty-input repeats.

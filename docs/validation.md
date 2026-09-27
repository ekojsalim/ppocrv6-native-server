# Extraction validation

Validated on 2026-09-27 with an RTX 5070 Ti (16 GiB), CUDA 13.2, the default
FP16 implementation and all three HTTP APIs enabled. The fixture corpus and
performance harness remain outside this source repository.

- 12 Rust tests, 4 exporter CPU-reference tests and 2 native CTest tests passed.
- Clean application container and all 11 compiled model variants built.
- Fresh pinned upstream downloads, classifier preparation and a complete bundle
  build succeeded in a separate Python environment with the documented build
  requirements. Bundle hashes and weights were verified.
- Recognition and detection weights and generated graphs match the preceding
  memory-optimized prototype (apart from the generated header comment).
- HTTP output text matches that checkpoint for 1,024 glyphs, 98 line crops and
  all 98 lines on the page fixture.
- Page shape smoke checks include short, tiny, tall and square images.
- GPU scratch poisoning, graph replay and workspace growth tests passed for
  all 11 catalog entries across three passes. This run was not under Compute
  Sanitizer.

## Performance snapshot

Medians of 10 warmed HTTP requests on the validation node:

| Request | Time |
|---|---:|
| One glyph | 0.60 ms |
| 128 glyphs | 5.69 ms |
| 1,024 glyphs | 54.84 ms |
| 9 line crops | 19.61 ms |
| 98 line crops | 187.85 ms |
| Full page, 1224×1584, 98 lines | 163.39 ms |

After warming the shape catalog, `nvidia-smi` reported **957 MiB** total device
memory in use (approximately 0.93 GiB, including the node's small idle GPU
allocation). This is GPU memory, not host RAM or container image size. Cold
requests also initialize kernels and CUDA graphs and can take substantially
longer. Shared GPU workloads, input sizes and driver/library changes may affect
both latency and memory.

These results validate extraction fidelity on the existing fixtures, not
universal OCR accuracy or performance portability. Other GPUs are untested.

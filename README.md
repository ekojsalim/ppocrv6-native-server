# ppocrv6-native-server

PP-OCRv6 medium served through a Rust HTTP API and a native CUDA implementation.
Pages, cropped text lines, and individual glyphs share the same recognizer.
Inference uses CUDA, cuBLAS/cuBLASLt and cuDNN; it does not load TensorRT, ONNX
Runtime, Paddle, PyTorch, or Python.

The implementation uses compiled shape catalogs, FP16 storage, CUDA graphs,
shared weights and scratch memory. On an RTX 5070 Ti, the extracted implementation uses approximately 0.93 GiB
of warmed GPU memory and processes the validation page in 163 ms.
See [validation](docs/validation.md) for the tested scope.

## Run

With a compiled model bundle in `models/`:

```sh
podman build -t ppocrv6-native-server .
podman run --rm --device nvidia.com/gpu=all --security-opt label=disable \
  -p 127.0.0.1:8184:8184 -v "$PWD/models:/models:ro" ppocrv6-native-server
curl http://127.0.0.1:8184/health
```

The host needs an NVIDIA driver compatible with CUDA 13.2 and container GPU
support. `Containerfile` uses the CUDA/cuDNN development image only during the
build; the serving image contains the required runtime libraries and executable.
Docker users can build with `docker build -f Containerfile` and use `--gpus all`
in place of the Podman CDI device option.

Only four application settings are exposed:

| Option | Default | Purpose |
|---|---|---|
| `--model-dir` | `models` | Compiled bundle directory |
| `--listen` | `127.0.0.1:8184` | Bind address; the container uses `0.0.0.0:8184` |
| `--max-request-mib` | `16` | Image-data budget per request |
| `--queue-timeout-ms` | `30000` | Time allowed waiting for an inference slot |

All three endpoint families are always available. Precision, shared memory,
workspace sizing, model dimensions, batch capacities and detector thresholds
have fixed, tested defaults. The image also sets the small cuBLAS default pool;
operators retain their explicit 64 MiB workspace.

## API

POST JSON to `/v1/ocr/recognize`, `/v1/lines/recognize`, or
`/v1/glyphs/recognize`. Each also has a corresponding `/info` GET endpoint.
Images are base64-encoded PNG/JPEG/WebP/BMP/PNM files.

```json
{"image": "BASE64_IMAGE"}
```

The page endpoint returns detected line quadrilaterals and recognized text.
Lines and glyphs accept a batch:

```json
{"images": ["BASE64_IMAGE", "BASE64_IMAGE"], "batch_size": 8}
```

Glyph requests additionally accept `character_policy` (`all`, `suppress_ascii`,
`cjk_focus`, `cjk_focus_fallback`) and `score_mode` (`model`, `accepted`). The
existing glyph defaults are `cjk_focus_fallback` and `accepted`, intended for CJK
glyph recognition. Use `all` and `model` for unrestricted recognition and model
confidence. Accepted scores are binary acceptance indicators, not probabilities.
Page and line recognition use the full vocabulary. See [API details](docs/api.md).

## Build models

Model weights and generated GPU libraries are excluded from this repository.
The offline build needs Python 3.12+, NumPy, ONNX and PyYAML, plus Git and Podman.
Python/ONNX are build tools only. No Paddle or PyTorch export environment is needed.

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r tools/requirements-build.txt
CUDA_ARCH=120 bash tools/build_models.sh models
```

This downloads pinned PaddlePaddle model revisions, extracts the classifier,
fetches pinned CUTLASS C++ headers, and produces both catalogs plus provenance,
licenses and SHA-256 manifests. `CUDA_ARCH=120` targets RTX 5070 Ti; use the target
GPU's architecture for both the model build and `podman build --build-arg CUDA_ARCH=...`.
The generated libraries are architecture-specific executable code. Load trusted
bundles built for model ABI v2; older ABI v1 bundles are rejected.

To reuse already prepared source assets, pass their directory as the second
argument to `tools/build_models.sh`. A completed output directory is never
silently overwritten. [Build details](docs/build.md) describe layout and tests.

## License

Project code: MIT. Dependencies and model weights have separate licenses;
see [attribution](NOTICE.md). Generated models, CUDA libraries, benchmark data,
and development-machine configuration are not included in the source repository.

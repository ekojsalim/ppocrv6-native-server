# Build and dependencies

The supported container build uses CUDA 13.2, cuDNN, Ubuntu 24.04 and Rust 1.96.
`CUDA_ARCH` defaults to 120 (RTX 5070 Ti). Set the same target architecture for
both the application image and model bundle. Other GPUs have not been validated.

## Runtime

The Rust executable loads `../lib/libppocrv6_native.so` relative to itself.
Native inference uses CUDA, cuBLAS/cuBLASLt, cuDNN and compiled model libraries.
Detector postprocessing uses OpenCV core/imgproc and Clipper. The container
includes their runtime dependencies; no Python or graph inference framework is
needed to serve requests.

## Offline model build

Install Python 3.12+ and `tools/requirements-build.txt` (NumPy, ONNX, PyYAML).
Git fetches pinned CUTLASS headers; Podman runs the CUDA compiler image.
`tools/build_models.sh OUTPUT [PREPARED_SOURCE]` is the public entry point.
`PYTHON` selects the Python interpreter; `CUDA_ARCH` selects the GPU target.
An existing complete output is rejected. Retain prepared source assets to
rebuild without downloading the models again.

The output has `recognition/` and `detection/` catalogs (`models.tsv`, shared
weights and one `model.so` per shape), `classifier/`, `licenses/`, source
provenance and a SHA-256 `package.json` manifest. Recognition has seven shapes:
1/8/32/128 × width 80, batch 8 × width 384, and batch 1/8 × width 3200.
Detection has 256×256, 640×640, 1280×992 and 1280×1280 shapes.

ONNX is an offline input format only. ONNX Runtime is used only by the exporter
unit tests as a CPU reference and is absent from build and serving requirements.

## Development checks

```sh
cargo test --locked --manifest-path server/Cargo.toml
pip install -r tools/requirements-test.txt
python -m unittest discover -s tools/compiled_models/tests -v
cmake -S native -B build -DBUILD_TESTING=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Native development needs CUDA headers, CMake 3.24+, a C++20 compiler,
OpenCV core/imgproc development packages and libpolyclipping-dev. GPU test
executables are built separately from CPU CTest checks and take model paths.
For a manual installation, put the Rust binary in `PREFIX/bin` and install the
native library to `PREFIX/lib`; arrange the CUDA/cuDNN runtime library search
path and set `CUBLAS_WORKSPACE_CONFIG=:16:8` as the container does.

Model binaries, corpora, profiling traces and historical tuning programs are
kept outside the public source project.

# Attribution and distribution

The server and its native implementation are distributed under the MIT license
in `LICENSE`, preserving the original author's copyright. This project extracts
the native CUDA implementation from the PP-OCRv6 optimization work; no original
Git history, checkpoints, private fixture corpora, or model binaries are included.

The offline builder uses these separately licensed upstream components:

- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass), pinned to
  `0b55a2f691d69981583568fd9eb69687b1f0de8a`. Its C++ headers use BSD-3-Clause;
  the notice is in `LICENSES/CUTLASS.txt` and is copied into model bundles.
  This project does not use CuTe DSL's Python components.
- PaddlePaddle's [detector](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_onnx)
  and [recognizer](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx).
  The pinned source model cards declare Apache-2.0. The builder retains those
  cards and an Apache-2.0 license copy with the generated bundle. Modified graph
  representations, FP16 weights, and classifier extraction are produced locally.
  Source repository revisions and download hashes are recorded by the downloader.

CUDA, cuBLAS, cuDNN, OpenCV, Clipper, and Rust/Python dependencies retain their
own licenses. The container assembles NVIDIA runtime libraries from NVIDIA's
CUDA/cuDNN image; the project's MIT license does not relicense those libraries
or model weights. Publishing this source repository and redistributing built
GPU images/model bundles are separate distribution decisions.

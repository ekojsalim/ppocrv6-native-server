#!/usr/bin/env bash
# Build a versioned model library from an offline-exported bundle.
set -euo pipefail
if [[ $# != 2 || ( $1 != recognition && $1 != detection ) ]]; then
 echo 'usage: tools/compiled_models/build.sh recognition|detection BUNDLE' >&2;exit 2
fi
image=${BUILD_IMAGE:-docker.io/nvidia/cuda:13.2.0-cudnn-devel-ubuntu24.04}
podman image exists "$image" || podman pull "$image"
kind=$1;bundle=$(realpath "$2");repo=$(pwd)
cutlass=$(realpath "${CUTLASS_PATH:-tmp/native-cutlass}")
test -f "$bundle/model.inc";test -f "$cutlass/include/cutlass/gemm/device/gemm.h"
test -z "$(git -C "$cutlass" status --porcelain --untracked-files=all -- include)"
key=$({ sha256sum native/src/compiled/{cutlass_gemm.cu,cutlass_gemm.h} tools/compiled_models/build.sh;git -C "$cutlass" rev-parse HEAD;printf '%s\n' "${CUDA_ARCH:-120}";podman image inspect "$image" --format '{{.Id}}'; } | sha256sum | cut -d' ' -f1)
cache="$repo/tmp/compiled-model-cache/$key";mkdir -p "$cache"
podman run --rm --security-opt label=disable -v "$repo:/work:ro" -v "$bundle:/bundle" -v "$cutlass:/cutlass:ro" -v "$cache:/cache" -w /work -e CUDA_ARCH="${CUDA_ARCH:-120}" -e MODEL_KIND="$kind" --entrypoint bash "$image" -lc '
 set -euo pipefail
 if [[ ! -f /cache/cutlass.o ]];then
  nvcc -std=c++17 -O3 --expt-relaxed-constexpr -arch=sm_${CUDA_ARCH} -Xcompiler -fPIC,-fvisibility=hidden -I /cutlass/include -c native/src/compiled/cutlass_gemm.cu -o /tmp/cutlass.o
  mv /tmp/cutlass.o /cache/cutlass.o
 fi
 detector=0;if [[ $MODEL_KIND == detection ]];then detector=1;fi
 nvcc -std=c++17 -O3 --expt-relaxed-constexpr -arch=sm_${CUDA_ARCH} -Xcompiler -fPIC,-fvisibility=hidden -shared -DPPOCR_CUTLASS=1 -DPPOCR_MODEL_DETECTOR=$detector -I native/include -I /bundle -I /cutlass/include native/src/compiled/model_library.cu /cache/cutlass.o -lcudnn -lcublas -lcublasLt -o /bundle/model.so
 '
cp "$cutlass/LICENSE.txt" "$bundle/CUTLASS_LICENSE.txt"
git -C "$cutlass" rev-parse HEAD > "$bundle/cutlass-revision.txt"
(cd "$bundle";sha256sum model.so weights.f32 > SHA256SUMS)

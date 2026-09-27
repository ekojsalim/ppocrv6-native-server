#!/usr/bin/env bash
# Fetch only the pinned headers used by the fused CUDA kernels.
set -euo pipefail
destination=${1:-tmp/native-cutlass}
revision=0b55a2f691d69981583568fd9eb69687b1f0de8a
if [[ ! -e $destination ]]; then
  git init -q "$destination"
  git -C "$destination" remote add origin https://github.com/NVIDIA/cutlass.git
  git -C "$destination" fetch --depth 1 --filter=blob:none origin "$revision"
  git -C "$destination" sparse-checkout set include
  git -C "$destination" checkout --detach "$revision"
fi
test "$(git -C "$destination" rev-parse HEAD)" = "$revision"
git -C "$destination" diff --quiet HEAD -- include
test -f "$destination/include/cutlass/gemm/device/gemm.h"
echo "CUTLASS headers ready: $destination ($revision)"

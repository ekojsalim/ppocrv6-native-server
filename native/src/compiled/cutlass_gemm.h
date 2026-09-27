#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <functional>
#include <vector>

// Each returned callable launches a complete GEMM with its bias/activation or
// bias/residual epilogue in one CUDA kernel. Inputs/output must not overlap.
std::vector<std::function<void()>>
make_cutlass_candidates(const half *x, const half *w, const half *bias,
                        const half *residual, half *y, int rows, int ci, int co,
                        int activation, cudaStream_t stream);

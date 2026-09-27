#pragma once
// Fixed-shape FP16 model execution using CUDA, cuDNN, cuBLAS and cuBLASLt.
#include <algorithm>
#include <cmath>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <functional>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#if PPOCR_CUTLASS
#include "cutlass_gemm.h"
#endif

#define CU(x)                                                                  \
  do {                                                                         \
    auto e = (x);                                                              \
    if (e != cudaSuccess)                                                      \
      throw std::runtime_error(std::string(#x) + ": " +                        \
                               cudaGetErrorString(e));                         \
  } while (0)
#define DN(x)                                                                  \
  do {                                                                         \
    auto e = (x);                                                              \
    if (e != CUDNN_STATUS_SUCCESS)                                             \
      throw std::runtime_error(std::string(#x) + ": " +                        \
                               cudnnGetErrorString(e));                        \
  } while (0)
#define BL(x)                                                                  \
  do {                                                                         \
    auto e = (x);                                                              \
    if (e != CUBLAS_STATUS_SUCCESS)                                            \
      throw std::runtime_error(std::string(#x) + ": " + std::to_string(e));    \
  } while (0)
using Scalar = half;
constexpr bool half_storage = true;
using Shape = std::vector<int>;
int count(const Shape &s) {
  return std::accumulate(s.begin(), s.end(), 1, std::multiplies<int>());
}
struct Index {
  int rank = 0, shape[6] = {}, stride[6] = {}, offset = 0;
};
Index indexer(const Shape &source, const Shape &dest) {
  if (dest.size() > 6 || source.size() > dest.size())
    throw std::runtime_error("unsupported rank");
  Index m;
  m.rank = dest.size();
  int stride = 1;
  for (int j = int(dest.size()) - 1; j >= 0; --j) {
    m.shape[j] = dest[j];
    int k = j - int(dest.size() - source.size());
    if (k >= 0) {
      m.stride[j] = source[k] == 1 ? 0 : stride;
      stride *= source[k];
    }
  }
  return m;
}
__device__ int locate(int i, Index m) {
  int off = m.offset;
  for (int j = m.rank - 1; j >= 0; --j) {
    off += (i % m.shape[j]) * m.stride[j];
    i /= m.shape[j];
  }
  return off;
}
__global__ void binary_kernel(const Scalar *a, const Scalar *b, Scalar *y,
                              int n, Index ai, Index bi, int op) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float x = a[locate(i, ai)], z = b[locate(i, bi)];
  y[i] = op == 0   ? x + z
         : op == 1 ? x - z
         : op == 2 ? x * z
         : op == 3 ? x / z
                   : powf(x, z);
}
template <int Divisor, int Modulus> __device__ int flat_index(int i) {
  if constexpr (Modulus == 1)
    return 0;
  if constexpr (Divisor != 1)
    i /= Divisor;
  if constexpr (Modulus > 1)
    i %= Modulus;
  return i;
}
template <int Op, int AD, int AM, int BD, int BM>
__global__ void binary_fast_kernel(const Scalar *a, const Scalar *b, Scalar *y,
                                   int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float x = a[flat_index<AD, AM>(i)], z = b[flat_index<BD, BM>(i)];
  if constexpr (Op == 0)
    y[i] = x + z;
  else if constexpr (Op == 1)
    y[i] = x - z;
  else if constexpr (Op == 2)
    y[i] = x * z;
  else if constexpr (Op == 3)
    y[i] = x / z;
  else
    y[i] = powf(x, z);
}
__global__ void unary_kernel(const Scalar *x, Scalar *y, int n, int op,
                             float alpha, float beta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float v = x[i];
  y[i] = op == 0   ? fmaxf(v, 0)
         : op == 1 ? 1.f / (1.f + expf(-v))
         : op == 2 ? fminf(1.f, fmaxf(0.f, alpha * v + beta))
         : op == 3 ? erff(v)
         : op == 4 ? sqrtf(v)
                   : __half2float(__float2half_rn(v));
}
__global__ void gather_kernel(const Scalar *x, Scalar *y, int n, Index m) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = x[locate(i, m)];
}
__global__ void transpose_matrix_kernel(const Scalar *x, Scalar *y, int rows,
                                        int cols) {
  __shared__ Scalar tile[32][33];
  int c = blockIdx.x * 32 + threadIdx.x, r = blockIdx.y * 32 + threadIdx.y,
      b = blockIdx.z;
  for (int j = 0; j < 32; j += 8)
    if (c < cols && r + j < rows)
      tile[threadIdx.y + j][threadIdx.x] = x[(b * rows + r + j) * cols + c];
  __syncthreads();
  c = blockIdx.y * 32 + threadIdx.x;
  r = blockIdx.x * 32 + threadIdx.y;
  for (int j = 0; j < 32; j += 8)
    if (c < rows && r + j < cols)
      y[(b * cols + r + j) * rows + c] = tile[threadIdx.x][threadIdx.y + j];
}
__global__ void concat_kernel(const Scalar *a, const Scalar *b, Scalar *y,
                              int n, int na, int nb) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  int row = i / (na + nb), col = i % (na + nb);
  y[i] = col < na ? a[row * na + col] : b[row * nb + col - na];
}
__global__ void concat_vec8_kernel(const uint4 *a, const uint4 *b, uint4 *y,
                                   int n, int na, int nb) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  int row = i / (na + nb), col = i % (na + nb);
  y[i] = col < na ? a[row * na + col] : b[row * nb + col - na];
}
__global__ void pad_kernel(const Scalar *x, Scalar *y, int n, int h, int w,
                           int oh, int ow, int ph, int pw) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  int row = (i / ow) % oh - ph, col = i % ow - pw, bc = i / (oh * ow);
  y[i] = row >= 0 && row < h && col >= 0 && col < w
             ? float(x[(bc * h + row) * w + col])
             : 0.f;
}
__global__ void pool_kernel(const Scalar *x, Scalar *y, int n, int h, int w,
                            int oh, int ow, int kh, int kw, int sh, int sw,
                            int ph, int pw, bool maximum) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  int row = (i / ow) % oh * sh - ph, col = i % ow * sw - pw, bc = i / (oh * ow),
      valid = 0;
  float v = maximum ? -INFINITY : 0.f;
  for (int a = 0; a < kh; ++a)
    for (int b = 0; b < kw; ++b)
      if (row + a >= 0 && row + a < h && col + b >= 0 && col + b < w) {
        float z = x[(bc * h + row + a) * w + col + b];
        v = maximum ? fmaxf(v, z) : v + z;
        ++valid;
      }
  y[i] = maximum ? v : v / valid;
}
__global__ void reduce_kernel(const Scalar *x, Scalar *y, int cols,
                              bool softmax) {
  __shared__ float tmp[256];
  int row = blockIdx.x, tid = threadIdx.x;
  float v = softmax ? -INFINITY : 0.f;
  for (int j = tid; j < cols; j += 256)
    v = softmax ? fmaxf(v, float(x[row * cols + j]))
                : v + float(x[row * cols + j]);
  tmp[tid] = v;
  __syncthreads();
  for (int s = 128; s; s /= 2) {
    if (tid < s)
      tmp[tid] =
          softmax ? fmaxf(tmp[tid], tmp[tid + s]) : tmp[tid] + tmp[tid + s];
    __syncthreads();
  }
  if (!softmax) {
    if (tid == 0)
      y[row] = tmp[0] / cols;
    return;
  }
  float mx = tmp[0];
  __syncthreads();
  v = 0;
  for (int j = tid; j < cols; j += 256)
    v += expf(float(x[row * cols + j]) - mx);
  tmp[tid] = v;
  __syncthreads();
  for (int s = 128; s; s /= 2) {
    if (tid < s)
      tmp[tid] += tmp[tid + s];
    __syncthreads();
  }
  float total = tmp[0];
  for (int j = tid; j < cols; j += 256)
    y[row * cols + j] = expf(float(x[row * cols + j]) - mx) / total;
}

__global__ void layernorm_kernel(const Scalar *x, const Scalar *scale,
                                 const Scalar *bias, Scalar *y, int cols,
                                 float epsilon) {
  __shared__ float scratch[256];
  int row = blockIdx.x, tid = threadIdx.x;
  float sum = 0.f;
  for (int j = tid; j < cols; j += 256)
    sum += float(x[row * cols + j]);
  scratch[tid] = sum;
  __syncthreads();
  for (int step = 128; step; step /= 2) {
    if (tid < step)
      scratch[tid] += scratch[tid + step];
    __syncthreads();
  }
  float mean = scratch[0] / cols;
  __syncthreads();
  sum = 0.f;
  for (int j = tid; j < cols; j += 256) {
    float d = float(x[row * cols + j]) - mean;
    sum += d * d;
  }
  scratch[tid] = sum;
  __syncthreads();
  for (int step = 128; step; step /= 2) {
    if (tid < step)
      scratch[tid] += scratch[tid + step];
    __syncthreads();
  }
  float inv = rsqrtf(scratch[0] / cols + epsilon);
  for (int j = tid; j < cols; j += 256)
    y[row * cols + j] =
        (float(x[row * cols + j]) - mean) * inv * float(scale[j]) +
        float(bias[j]);
}

struct Conv {
  cudnnTensorDescriptor_t x = nullptr, y = nullptr;
  cudnnFilterDescriptor_t w = nullptr;
  cudnnConvolutionDescriptor_t c = nullptr;
  cudnnConvolutionFwdAlgo_t algo;
  size_t bytes = 0;
  half *weight = nullptr;
  Conv() {
    DN(cudnnCreateTensorDescriptor(&x));
    DN(cudnnCreateTensorDescriptor(&y));
    DN(cudnnCreateFilterDescriptor(&w));
    DN(cudnnCreateConvolutionDescriptor(&c));
  }
  ~Conv() {
    if (x)
      cudnnDestroyTensorDescriptor(x);
    if (y)
      cudnnDestroyTensorDescriptor(y);
    if (w)
      cudnnDestroyFilterDescriptor(w);
    if (c)
      cudnnDestroyConvolutionDescriptor(c);
  }
};

#include "spatial_kernels.cuh"

__global__ void mean_nhwc_kernel(const Scalar *x, Scalar *y, int channels,
                                 int spatial) {
  __shared__ float partial[8][32];
  int channel = blockIdx.x * 32 + threadIdx.x, batch = blockIdx.y,
      lane = threadIdx.y;
  float sum = 0.f;
  if (channel < channels)
    for (int j = lane; j < spatial; j += 8)
      sum += float(x[(batch * spatial + j) * channels + channel]);
  partial[lane][threadIdx.x] = sum;
  __syncthreads();
  if (lane == 0 && channel < channels) {
    float total = 0.f;
    for (int j = 0; j < 8; ++j)
      total += partial[j][threadIdx.x];
    y[batch * channels + channel] = total / spatial;
  }
}

__global__ void mean_partial_kernel(const Scalar *x, float *partial,
                                    int channels, int spatial, int chunks) {
  __shared__ float values[8][32];
  int c = blockIdx.x * 32 + threadIdx.x, b = blockIdx.z, lane = threadIdx.y,
      chunk = blockIdx.y;
  float sum = 0;
  if (c < channels)
    for (int p = chunk * 256 + lane; p < min(spatial, (chunk + 1) * 256);
         p += 8)
      sum += float(x[(b * spatial + p) * channels + c]);
  values[lane][threadIdx.x] = sum;
  __syncthreads();
  if (lane == 0 && c < channels) {
    for (int j = 1; j < 8; ++j)
      sum += values[j][threadIdx.x];
    partial[(b * chunks + chunk) * channels + c] = sum;
  }
}
__global__ void mean_finish_kernel(const float *partial, Scalar *y,
                                   int channels, int spatial, int chunks) {
  int c = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.y;
  if (c >= channels)
    return;
  float sum = 0;
  for (int j = 0; j < chunks; ++j)
    sum += partial[(b * chunks + j) * channels + c];
  y[b * channels + c] = sum / spatial;
}

struct LtGemm {
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
  cublasLtMatmulAlgo_t algo{};
  size_t bytes = 0;
  std::vector<cublasLtMatmulHeuristicResult_t> candidates;
  bool tuned = false;
  ~LtGemm() {
    if (desc)
      cublasLtMatmulDescDestroy(desc);
    if (a)
      cublasLtMatrixLayoutDestroy(a);
    if (b)
      cublasLtMatrixLayoutDestroy(b);
    if (c)
      cublasLtMatrixLayoutDestroy(c);
  }
};
__global__ void resize_nearest_kernel(const Scalar *x, Scalar *y, int n, int h,
                                      int w, int oh, int ow, float sh,
                                      float sw) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  int r = min(int(floorf(((i / ow) % oh) / sh)), h - 1),
      c = min(int(floorf((i % ow) / sw)), w - 1);
  y[i] = x[((i / (oh * ow)) * h + r) * w + c];
}
__global__ void resize_nhwc_kernel(const Scalar *x, Scalar *y, int n,
                                   int channels, int h, int w, int oh, int ow,
                                   float sh, float sw) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  int pixel = i / channels, channel = i % channels;
  int r = min(int(floorf(((pixel / ow) % oh) / sh)), h - 1),
      c = min(int(floorf((pixel % ow) / sw)), w - 1);
  y[i] = x[(((pixel / (oh * ow)) * h + r) * w + c) * channels + channel];
}
struct Runtime {
  cudaStream_t stream = nullptr;
  cudnnHandle_t dn = nullptr;
  cublasHandle_t bl = nullptr;
  std::function<cublasHandle_t()> acquire_blas;
  cublasLtHandle_t lt = nullptr;
  Scalar *weights = nullptr, *arena = nullptr;
  void *workspace = nullptr;
  size_t workspace_bytes = 0;
  static constexpr bool fp16 = true;
  static constexpr bool native_depthwise = true;
  static constexpr bool half_gemm_accumulation = true;
  static constexpr bool tune_gemm = true;
  static constexpr bool spatial_tune = true;
  static constexpr bool cutlass_enabled = true;
  static constexpr size_t workspace_limit = 64 * 1024 * 1024;
  std::vector<Scalar *> t;
  std::vector<std::function<void()>> ops;
  Runtime() {
    CU(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    DN(cudnnCreate(&dn));
    DN(cudnnSetStream(dn, stream));
  }
  void ensure_blas() {
    if (bl)
      return;
    if (acquire_blas)
      bl = acquire_blas();
    else
      BL(cublasCreate(&bl));
    BL(cublasSetStream(bl, stream));
    BL(cublasSetMathMode(bl, CUBLAS_DEFAULT_MATH));
    if (workspace)
      BL(cublasSetWorkspace(bl, workspace, workspace_limit));
  }
  ~Runtime() {
    if (stream)
      cudaStreamSynchronize(stream);
    ops.clear();
    if (lt)
      cublasLtDestroy(lt);
    if (bl && !acquire_blas)
      cublasDestroy(bl);
    if (dn)
      cudnnDestroy(dn);
    // Weights and scratch are borrowed from the engine/workspace lease.
    if (stream)
      cudaStreamDestroy(stream);
  }
  void deconv(int ai, int wi, int yi, Shape xs, Shape ws, Shape ys, Shape pad,
              Shape stride, Shape dilation, bool channels_last = false) {
    auto c = std::make_shared<Conv>();
    auto dtype = CUDNN_DATA_HALF;
    DN(cudnnSetTensor4dDescriptor(
        c->x, channels_last ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW, dtype,
        ys[0], ys[1], ys[2], ys[3]));
    DN(cudnnSetTensor4dDescriptor(
        c->y, channels_last ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW, dtype,
        xs[0], xs[1], xs[2], xs[3]));
    DN(cudnnSetFilter4dDescriptor(
        c->w, dtype, channels_last ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW,
        ws[0], ws[1], ws[2], ws[3]));
    DN(cudnnSetConvolution2dDescriptor(
        c->c, pad[0], pad[1], stride[0], stride[1], dilation[0], dilation[1],
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    DN(cudnnSetConvolutionMathType(c->c, CUDNN_TENSOR_OP_MATH));
    DN(cudnnGetConvolutionBackwardDataWorkspaceSize(
        dn, c->w, c->y, c->c, c->x, CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
        &c->bytes));
    workspace_bytes = std::max(workspace_bytes, c->bytes);
    ops.emplace_back([=] {
      float one = 1, zero = 0;
      DN(cudnnConvolutionBackwardData(dn, &one, c->w, t[wi], c->y, t[ai], c->c,
                                      CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
                                      workspace, c->bytes, &zero, c->x, t[yi]));
    });
  }
  void resize_nearest(int a, int y, Shape xs, Shape ys, float sh, float sw) {
    int n = count(ys);
    ops.emplace_back([=] {
      resize_nearest_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
          t[a], t[y], n, xs[2], xs[3], ys[2], ys[3], sh, sw);
    });
  }
  void resize_nearest_nhwc(int a, int y, Shape xs, Shape ys, float sh,
                           float sw) {
    int n = count(ys);
    ops.emplace_back([=] {
      resize_nhwc_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
          t[a], t[y], n, xs[1], xs[2], xs[3], ys[2], ys[3], sh, sw);
    });
  }
  void spatial_candidates(std::vector<std::function<void()>> candidates) {
    auto selected = std::make_shared<int>(-1);
    ops.back() = [=] {
      if (*selected < 0) {
        cudaEvent_t start, stop;
        CU(cudaEventCreate(&start));
        CU(cudaEventCreate(&stop));
        float best = INFINITY;
        for (int i = 0; i < int(candidates.size()); ++i) {
          candidates[i]();
          CU(cudaEventRecord(start, stream));
          for (int j = 0; j < 20; ++j)
            candidates[i]();
          CU(cudaEventRecord(stop, stream));
          CU(cudaEventSynchronize(stop));
          float ms;
          CU(cudaEventElapsedTime(&ms, start, stop));
          if (i == 0 || ms < best * 0.98f) {
            best = ms;
            *selected = i;
          }
        }
        CU(cudaEventDestroy(start));
        CU(cudaEventDestroy(stop));
      }
      candidates[*selected]();
    };
  }
  template <int KH, int KW, int SH, int PIX, int SW = 1>
  std::function<void()> spatial_depthwise(int ai, int wi, int yi, Shape xs,
                                          Shape ys, Shape pad, int bias_id,
                                          int activation) {
    return [=] {
      int n = xs[0] * ys[2] * ((ys[3] + PIX - 1) / PIX) * (xs[1] / 8);
      depthwise_nhwc_vec8<KH, KW, SH, PIX, SW>
          <<<(n + 127) / 128, 128, 0, stream>>>(
              reinterpret_cast<const half *>(t[ai]),
              reinterpret_cast<const half *>(t[wi]),
              reinterpret_cast<half *>(t[yi]), xs[0], xs[1], xs[2], xs[3],
              ys[2], ys[3], pad[0], pad[1],
              bias_id >= 0 ? reinterpret_cast<const half *>(t[bias_id])
                           : nullptr,
              activation);
    };
  }
  void gemm_epilogue(int ai, int wi, int bi, int ri, int yi, int rows, int ci,
                     int co, int activation) {
    if (!lt)
      BL(cublasLtCreate(&lt));
    auto g = std::make_shared<LtGemm>();
    BL(cublasLtMatmulDescCreate(&g->desc, CUBLAS_COMPUTE_16F, CUDA_R_16F));
    cublasOperation_t trans = CUBLAS_OP_T;
    BL(cublasLtMatmulDescSetAttribute(g->desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                      &trans, sizeof(trans)));
    cublasLtEpilogue_t epi = activation == 2   ? CUBLASLT_EPILOGUE_GELU_BIAS
                             : activation == 1 ? CUBLASLT_EPILOGUE_RELU_BIAS
                                               : CUBLASLT_EPILOGUE_BIAS;
    BL(cublasLtMatmulDescSetAttribute(g->desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                      &epi, sizeof(epi)));
    BL(cublasLtMatmulDescSetAttribute(
        g->desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &t[bi], sizeof(t[bi])));
    BL(cublasLtMatrixLayoutCreate(&g->a, CUDA_R_16F, ci, co, ci));
    BL(cublasLtMatrixLayoutCreate(&g->b, CUDA_R_16F, ci, rows, ci));
    BL(cublasLtMatrixLayoutCreate(&g->c, CUDA_R_16F, co, rows, co));
    cublasLtMatmulPreference_t pref;
    BL(cublasLtMatmulPreferenceCreate(&pref));
    size_t limit = workspace_limit;
    BL(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &limit, sizeof(limit)));
    uint32_t alignment =
        128; // Exporter's arena and weight packing guarantee 128 bytes.
    for (auto attr : {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                      CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                      CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                      CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES})
      BL(cublasLtMatmulPreferenceSetAttribute(pref, attr, &alignment,
                                              sizeof(alignment)));
    g->candidates.resize(tune_gemm ? 32 : 1);
    int found = 0;
    auto status = cublasLtMatmulAlgoGetHeuristic(
        lt, g->desc, g->a, g->b, g->c, g->c, pref, g->candidates.size(),
        g->candidates.data(), &found);
    cublasLtMatmulPreferenceDestroy(pref);
    BL(status);
    if (!found)
      throw std::runtime_error("No cuBLASLt epilogue algorithm for " +
                               std::to_string(rows) + "x" + std::to_string(ci) +
                               "x" + std::to_string(co));
    g->candidates.resize(found);
    g->algo = g->candidates[0].algo;
    g->bytes = g->candidates[0].workspaceSize;
    for (auto &choice : g->candidates)
      workspace_bytes = std::max(workspace_bytes, choice.workspaceSize);
    ops.emplace_back([=] {
      float one = 1.f, beta = ri >= 0 ? 1.f : 0.f;
      half one_h = __float2half_rn(one), beta_h = __float2half_rn(beta);
      const void *alpha_ptr = static_cast<void *>(&one_h);
      const void *beta_ptr = static_cast<void *>(&beta_h);
      auto launch = [&](const cublasLtMatmulAlgo_t &algo, size_t bytes) {
        return cublasLtMatmul(lt, g->desc, alpha_ptr, t[wi], g->a, t[ai], g->b,
                              beta_ptr, t[ri >= 0 ? ri : yi], g->c, t[yi], g->c,
                              &algo, workspace, bytes, stream);
      };
      // First invocation precedes CUDA graph capture and receives real
      // activations. Inputs and output have disjoint arena allocations,
      // including residual C.
      if (tune_gemm && !g->tuned) {
        cudaEvent_t start, stop;
        CU(cudaEventCreate(&start));
        CU(cudaEventCreate(&stop));
        float best = INFINITY;
        for (auto &candidate : g->candidates) {
          if (candidate.state != CUBLAS_STATUS_SUCCESS)
            continue;
          auto status = launch(candidate.algo, candidate.workspaceSize);
          if (status != CUBLAS_STATUS_SUCCESS)
            continue;
          CU(cudaEventRecord(start, stream));
          for (int repeat = 0; repeat < 5; ++repeat)
            BL(launch(candidate.algo, candidate.workspaceSize));
          CU(cudaEventRecord(stop, stream));
          CU(cudaEventSynchronize(stop));
          float ms;
          CU(cudaEventElapsedTime(&ms, start, stop));
          if (ms < best) {
            best = ms;
            g->algo = candidate.algo;
            g->bytes = candidate.workspaceSize;
          }
        }
        CU(cudaEventDestroy(start));
        CU(cudaEventDestroy(stop));
        g->tuned = true;
      }
      BL(launch(g->algo, g->bytes));
    });
#if PPOCR_CUTLASS
    if (half_gemm_accumulation && rows >= 32 && ci % 8 == 0 && co % 8 == 0 &&
        cutlass_enabled) {
      const half *x = reinterpret_cast<const half *>(t[ai]);
      const half *w = reinterpret_cast<const half *>(t[wi]);
      const half *bias = reinterpret_cast<const half *>(t[bi]);
      half *y = reinterpret_cast<half *>(t[yi]);
      const half *residual =
          ri >= 0 ? reinterpret_cast<const half *>(t[ri]) : nullptr;
      std::vector<std::function<void()>> candidates{ops.back()};
      auto custom = make_cutlass_candidates(x, w, bias, residual, y, rows, ci,
                                            co, activation, stream);
      candidates.insert(candidates.end(), custom.begin(), custom.end());
      auto selected = std::make_shared<int>(-1);
      ops.back() = [=] {
        if (*selected < 0) {
          cudaEvent_t start, stop;
          CU(cudaEventCreate(&start));
          CU(cudaEventCreate(&stop));
          float best = INFINITY;
          for (int index = 0; index < int(candidates.size()); ++index) {
            candidates[index]();
            CU(cudaEventRecord(start, stream));
            for (int repeat = 0; repeat < 7; ++repeat)
              candidates[index]();
            CU(cudaEventRecord(stop, stream));
            CU(cudaEventSynchronize(stop));
            float ms;
            CU(cudaEventElapsedTime(&ms, start, stop));
            if (index == 0 || ms < best * 0.98f) {
              best = ms;
              *selected = index;
            }
          }
          CU(cudaEventDestroy(start));
          CU(cudaEventDestroy(stop));
        }
        candidates[*selected]();
      };
    }
#endif
  }
  void conv(int ai, int wi, int yi, Shape xs, Shape ws, Shape ys, Shape pad,
            Shape stride, Shape dilation, int groups,
            bool channels_last = false, bool input_channels_last = false,
            int bias_id = -1, int activation = 0) {
    if (bias_id >= 0 && (!half_storage || !native_depthwise))
      throw std::runtime_error(
          "Depthwise epilogue requires native FP16 depthwise kernels");
    if (half_storage && channels_last && input_channels_last && groups == 1 &&
        ws[2] == 1 && ws[3] == 1 && stride == Shape{1, 1} &&
        pad == Shape{0, 0} && xs[2] == ys[2] && xs[3] == ys[3]) {
      int rows = xs[0] * xs[2] * xs[3], ci = xs[1], co = ys[1];
      ensure_blas();
      ops.emplace_back([=] {
        half one = __float2half_rn(1.f), zero = __float2half_rn(0.f);
        BL(cublasGemmEx(bl, CUBLAS_OP_T, CUBLAS_OP_N, co, rows, ci, &one, t[wi],
                        CUDA_R_16F, ci, t[ai], CUDA_R_16F, ci, &zero, t[yi],
                        CUDA_R_16F, co, CUBLAS_COMPUTE_16F,
                        CUBLAS_GEMM_DEFAULT));
      });
      return;
    }
    if (half_storage && native_depthwise && channels_last &&
        input_channels_last && groups == xs[1] && ws[0] == groups &&
        ws[1] == 1 && xs[1] % 8 == 0 && dilation == Shape{1, 1} &&
        ((ws[2] == 9 && ws[3] == 9 && stride == Shape{1, 1}) ||
         (ws[2] == 3 && ws[3] == 3 && stride == Shape{2, 2}))) {
      std::vector<std::function<void()>> candidates;
#define ADD_DETECTOR_DW(K, S)                                                  \
  candidates.push_back(spatial_depthwise<K, K, S, 1, S>(                       \
      ai, wi, yi, xs, ys, pad, bias_id, activation));                          \
  candidates.push_back(spatial_depthwise<K, K, S, 2, S>(                       \
      ai, wi, yi, xs, ys, pad, bias_id, activation));                          \
  candidates.push_back(spatial_depthwise<K, K, S, 4, S>(                       \
      ai, wi, yi, xs, ys, pad, bias_id, activation));
      if (ws[2] == 9) {
        ADD_DETECTOR_DW(9, 1)
      } else {
        ADD_DETECTOR_DW(3, 2)
      }
#undef ADD_DETECTOR_DW
      if (ws[2] == 9) {
#define ADD_PAIR(P)                                                            \
  candidates.emplace_back([=] {                                                \
    int n = xs[0] * ys[2] * ((ys[3] + P - 1) / P) * (xs[1] / 2);               \
    depthwise9_pair<P><<<(n + 127) / 128, 128, 0, stream>>>(                   \
        t[ai], t[wi], t[yi], xs[0], xs[1], xs[2], xs[3], ys[2], ys[3], pad[0], \
        pad[1], bias_id >= 0 ? t[bias_id] : nullptr, activation);              \
  });
        ADD_PAIR(2) ADD_PAIR(4) ADD_PAIR(8)
#undef ADD_PAIR
      }

      ops.push_back(candidates.front());
      if (spatial_tune)
        spatial_candidates(std::move(candidates));
      return;
    }
    if (half_storage && native_depthwise && groups == xs[1] &&
        ws[0] == groups && ws[1] == 1 && dilation == Shape{1, 1} &&
        stride[1] == 1 &&
        ((ws[2] == 3 && ws[3] == 3 && (stride[0] == 1 || stride[0] == 2)) ||
         (ws[2] == 1 && ws[3] == 7 && stride[0] == 1))) {
      int n = count(ys);
      ops.emplace_back([=] {
        auto x = reinterpret_cast<const half *>(t[ai]);
        auto w = reinterpret_cast<const half *>(t[wi]);
        auto y = reinterpret_cast<half *>(t[yi]);
        const half *bias =
            bias_id >= 0 ? reinterpret_cast<const half *>(t[bias_id]) : nullptr;
        if (channels_last != input_channels_last)
          throw std::runtime_error("Depthwise layout transition unsupported");
        if (channels_last && ws[2] == 3 && xs[1] % 2 == 0) {
          dim3 grid((ys[3] + 7) / 8, (ys[2] + 1) / 2,
                    xs[0] * ((xs[1] + 63) / 64));
          if (stride[0] == 2)
            depthwise_nhwc_tiled<2><<<grid, dim3(32, 8), 0, stream>>>(
                x, w, y, xs[1], xs[2], xs[3], ys[2], ys[3], pad[0], pad[1],
                bias, activation);
          else
            depthwise_nhwc_tiled<1><<<grid, dim3(32, 8), 0, stream>>>(
                x, w, y, xs[1], xs[2], xs[3], ys[2], ys[3], pad[0], pad[1],
                bias, activation);
          return;
        }
        if (ws[2] == 1)
          depthwise_half<1, 7, 1, 1><<<(n + 255) / 256, 256, 0, stream>>>(
              x, w, y, n, xs[1], xs[2], xs[3], ys[2], ys[3], pad[0], pad[1],
              channels_last, bias, activation);
        else if (stride[0] == 2)
          depthwise_half<3, 3, 2, 1><<<(n + 255) / 256, 256, 0, stream>>>(
              x, w, y, n, xs[1], xs[2], xs[3], ys[2], ys[3], pad[0], pad[1],
              channels_last, bias, activation);
        else
          depthwise_half<3, 3, 1, 1><<<(n + 255) / 256, 256, 0, stream>>>(
              x, w, y, n, xs[1], xs[2], xs[3], ys[2], ys[3], pad[0], pad[1],
              channels_last, bias, activation);
      });
      if (spatial_tune && channels_last && input_channels_last &&
          xs[1] % 8 == 0) {
        std::vector<std::function<void()>> candidates{ops.back()};
#define ADD_SPATIAL(KH, KW, SH)                                                \
  candidates.push_back(spatial_depthwise<KH, KW, SH, 1>(                       \
      ai, wi, yi, xs, ys, pad, bias_id, activation));                          \
  candidates.push_back(spatial_depthwise<KH, KW, SH, 2>(                       \
      ai, wi, yi, xs, ys, pad, bias_id, activation));                          \
  candidates.push_back(spatial_depthwise<KH, KW, SH, 4>(                       \
      ai, wi, yi, xs, ys, pad, bias_id, activation));
        if (ws[2] == 1) {
          ADD_SPATIAL(1, 7, 1)
        } else if (stride[0] == 2) {
          ADD_SPATIAL(3, 3, 2)
        } else {
          ADD_SPATIAL(3, 3, 1)
        }
#undef ADD_SPATIAL
        spatial_candidates(std::move(candidates));
      }
      return;
    }
    auto c = std::make_shared<Conv>();
    auto dtype = CUDNN_DATA_HALF;
    DN(cudnnSetTensor4dDescriptor(
        c->x, input_channels_last ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW,
        dtype, xs[0], xs[1], xs[2], xs[3]));
    DN(cudnnSetTensor4dDescriptor(
        c->y, channels_last ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW, dtype,
        ys[0], ys[1], ys[2], ys[3]));
    DN(cudnnSetFilter4dDescriptor(
        c->w, dtype, channels_last ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW,
        ws[0], ws[1], ws[2], ws[3]));
    DN(cudnnSetConvolution2dDescriptor(
        c->c, pad[0], pad[1], stride[0], stride[1], dilation[0], dilation[1],
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    DN(cudnnSetConvolutionGroupCount(c->c, groups));
    DN(cudnnSetConvolutionMathType(c->c, CUDNN_TENSOR_OP_MATH));
    cudnnConvolutionFwdAlgoPerf_t perf[8];
    int returned = 0;
    DN(cudnnGetConvolutionForwardAlgorithm_v7(dn, c->x, c->w, c->c, c->y, 8,
                                              &returned, perf));
    bool found = false;
    for (int j = 0; j < returned; ++j)
      if (perf[j].status == CUDNN_STATUS_SUCCESS &&
          perf[j].memory <= workspace_limit) {
        c->algo = perf[j].algo;
        DN(cudnnSetConvolutionMathType(c->c, perf[j].mathType));
        DN(cudnnGetConvolutionForwardWorkspaceSize(dn, c->x, c->w, c->c, c->y,
                                                   c->algo, &c->bytes));
        if (c->bytes <= workspace_limit) {
          found = true;
          break;
        }
      }
    if (half_storage)
      c->weight = reinterpret_cast<half *>(t[wi]);
    if (!found)
      throw std::runtime_error(
          "No convolution algorithm within workspace budget: input=" +
          std::to_string(xs[0]) + "x" + std::to_string(xs[1]) + "x" +
          std::to_string(xs[2]) + "x" + std::to_string(xs[3]) +
          " output_channels=" + std::to_string(ws[0]) +
          " groups=" + std::to_string(groups));
    workspace_bytes = std::max(workspace_bytes, c->bytes);
    ops.emplace_back([=] {
      float one = 1, zero = 0;
      DN(cudnnConvolutionForward(dn, &one, c->x, t[ai], c->w, c->weight, c->c,
                                 c->algo, workspace, c->bytes, &zero, c->y,
                                 t[yi]));
    });
  }

  void binary(int a, int b, int y, Shape as, Shape bs, Shape ys, int op) {
    auto am = indexer(as, ys), bm = indexer(bs, ys);
    int n = count(ys);
    ops.emplace_back([=] {
      binary_kernel<<<(n + 255) / 256, 256, 0, stream>>>(t[a], t[b], t[y], n,
                                                         am, bm, op);
    });
  }
  template <int Op, int AD, int AM, int BD, int BM>
  void binary_fast(int a, int b, int y, int n) {
    ops.emplace_back([=] {
      binary_fast_kernel<Op, AD, AM, BD, BM>
          <<<(n + 255) / 256, 256, 0, stream>>>(t[a], t[b], t[y], n);
    });
  }
  void unary(int a, int y, int n, int op, float alpha, float beta) {
    ops.emplace_back([=] {
      unary_kernel<<<(n + 255) / 256, 256, 0, stream>>>(t[a], t[y], n, op,
                                                        alpha, beta);
    });
  }
  void transpose(int a, int y, Shape xs, Shape ys, Shape perm) {
    int batch = 0, rows = 0, cols = 0;
    if (xs.size() == 3 && perm == Shape{0, 2, 1}) {
      batch = xs[0];
      rows = xs[1];
      cols = xs[2];
    }
    if (xs.size() == 4 && perm == Shape{0, 2, 3, 1}) {
      batch = xs[0];
      rows = xs[1];
      cols = xs[2] * xs[3];
    }
    if (xs.size() == 4 && perm == Shape{0, 3, 1, 2}) {
      batch = xs[0];
      rows = xs[1] * xs[2];
      cols = xs[3];
    }
    if (xs.size() == 4 && perm == Shape{0, 1, 3, 2}) {
      batch = xs[0] * xs[1];
      rows = xs[2];
      cols = xs[3];
    }
    if (batch && rows >= 16 && cols >= 16) {
      ops.emplace_back([=] {
        transpose_matrix_kernel<<<dim3((cols + 31) / 32, (rows + 31) / 32,
                                       batch),
                                  dim3(32, 8), 0, stream>>>(t[a], t[y], rows,
                                                            cols);
      });
      return;
    }
    Index m;
    m.rank = xs.size();
    Shape strides(xs.size());
    int stride = 1;
    for (int j = int(xs.size()) - 1; j >= 0; --j) {
      strides[j] = stride;
      stride *= xs[j];
    }
    for (int j = 0; j < m.rank; ++j) {
      m.shape[j] = ys[j];
      m.stride[j] = strides[perm[j]];
    }
    int n = count(ys);
    ops.emplace_back([=] {
      gather_kernel<<<(n + 255) / 256, 256, 0, stream>>>(t[a], t[y], n, m);
    });
  }
  void slice(int a, int y, Shape xs, Shape ys, Shape starts) {
    Index m;
    m.rank = xs.size();
    int stride = 1;
    for (int j = int(xs.size()) - 1; j >= 0; --j) {
      m.shape[j] = ys[j];
      m.stride[j] = stride;
      m.offset += starts[j] * stride;
      stride *= xs[j];
    }
    int n = count(ys);
    ops.emplace_back([=] {
      gather_kernel<<<(n + 255) / 256, 256, 0, stream>>>(t[a], t[y], n, m);
    });
  }
  void concat(int a, int b, int y, int rows, int na, int nb) {
    int n = rows * (na + nb);
    if (half_storage && na % 8 == 0 && nb % 8 == 0)
      ops.emplace_back([=] {
        concat_vec8_kernel<<<(n / 8 + 255) / 256, 256, 0, stream>>>(
            reinterpret_cast<const uint4 *>(t[a]),
            reinterpret_cast<const uint4 *>(t[b]),
            reinterpret_cast<uint4 *>(t[y]), n / 8, na / 8, nb / 8);
      });
    else
      ops.emplace_back([=] {
        concat_kernel<<<(n + 255) / 256, 256, 0, stream>>>(t[a], t[b], t[y], n,
                                                           na, nb);
      });
  }
  void pad(int a, int y, Shape xs, Shape ys, int ph, int pw) {
    int n = count(ys);
    ops.emplace_back([=] {
      pad_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
          t[a], t[y], n, xs[2], xs[3], ys[2], ys[3], ph, pw);
    });
  }
  void pool(int a, int y, Shape xs, Shape ys, Shape k, Shape s, Shape p,
            bool maximum) {
    int n = count(ys);
    ops.emplace_back([=] {
      pool_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
          t[a], t[y], n, xs[2], xs[3], ys[2], ys[3], k[0], k[1], s[0], s[1],
          p[0], p[1], maximum);
    });
  }
  void mean(int a, int y, int rows, int cols) {
    ops.emplace_back([=] {
      reduce_kernel<<<rows, 256, 0, stream>>>(t[a], t[y], cols, false);
    });
  }
  void mean_nhwc(int a, int y, int batch, int channels, int spatial) {
    if (spatial >= 1024) {
      int chunks = (spatial + 255) / 256;
      float *raw = nullptr;
      CU(cudaMalloc(&raw, size_t(batch) * channels * chunks * sizeof(float)));
      std::shared_ptr<float> partial(raw, [](float *p) { cudaFree(p); });
      ops.emplace_back([=] {
        mean_partial_kernel<<<dim3((channels + 31) / 32, chunks, batch),
                              dim3(32, 8), 0, stream>>>(
            t[a], partial.get(), channels, spatial, chunks);
        mean_finish_kernel<<<dim3((channels + 255) / 256, batch), 256, 0,
                             stream>>>(partial.get(), t[y], channels, spatial,
                                       chunks);
      });
    } else
      ops.emplace_back([=] {
        mean_nhwc_kernel<<<dim3((channels + 31) / 32, batch), dim3(32, 8), 0,
                           stream>>>(t[a], t[y], channels, spatial);
      });
  }
  void pad_nhwc(int a, int y, Shape xs, Shape ys, int ph, int pw) {
    int n = count(ys);
    ops.emplace_back([=] {
      if (half_storage && spatial_tune && xs[1] % 8 == 0)
        pad_nhwc_vec8<<<(n / 8 + 127) / 128, 128, 0, stream>>>(
            reinterpret_cast<const half *>(t[a]),
            reinterpret_cast<half *>(t[y]), n, xs[1], xs[2], xs[3], ys[2],
            ys[3], ph, pw);
      else
        pad_nhwc_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
            t[a], t[y], n, xs[1], xs[2], xs[3], ys[2], ys[3], ph, pw);
    });
  }
  void pool_nhwc(int a, int y, Shape xs, Shape ys, Shape k, Shape s, Shape p,
                 bool maximum) {
    int n = count(ys);
    ops.emplace_back([=] {
      pool_nhwc_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
          t[a], t[y], n, xs[1], xs[2], xs[3], ys[2], ys[3], k[0], k[1], s[0],
          s[1], p[0], p[1], maximum);
    });
    if (half_storage && spatial_tune && xs[1] % 8 == 0) {
      std::vector<std::function<void()>> candidates{ops.back()};
      if (maximum && k == Shape{2, 2} && s == Shape{1, 1})
        candidates.emplace_back([=] {
          pool_nhwc_vec8<2, 2, 1, 1, true>
              <<<(n / 8 + 127) / 128, 128, 0, stream>>>(
                  reinterpret_cast<const half *>(t[a]),
                  reinterpret_cast<half *>(t[y]), xs[0], xs[1], xs[2], xs[3],
                  ys[2], ys[3], p[0], p[1]);
        });
      if (!maximum && k == Shape{3, 2} && s == Shape{3, 2})
        candidates.emplace_back([=] {
          pool_nhwc_vec8<3, 2, 3, 2, false>
              <<<(n / 8 + 127) / 128, 128, 0, stream>>>(
                  reinterpret_cast<const half *>(t[a]),
                  reinterpret_cast<half *>(t[y]), xs[0], xs[1], xs[2], xs[3],
                  ys[2], ys[3], p[0], p[1]);
        });
      if (candidates.size() > 1)
        spatial_candidates(std::move(candidates));
    }
  }
  void layernorm(int a, int scale, int bias, int y, int rows, int cols,
                 float epsilon) {
    ops.emplace_back([=] {
      layernorm_kernel<<<rows, 256, 0, stream>>>(t[a], t[scale], t[bias], t[y],
                                                 cols, epsilon);
    });
  }
  void softmax(int a, int y, int rows, int cols) {
    ops.emplace_back([=] {
      reduce_kernel<<<rows, 256, 0, stream>>>(t[a], t[y], cols, true);
    });
  }
  // Attention products retain FP32 accumulation with FP16 inputs/output.
  void matmul(int a, int b, int y, int batches, int m, int n, int k) {
    ensure_blas();
    ops.emplace_back([=] {
      float one = 1, zero = 0;
      BL(cublasGemmStridedBatchedEx(
          bl, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one, t[b], CUDA_R_16F, n,
          static_cast<long long>(k) * n, t[a], CUDA_R_16F, k,
          static_cast<long long>(m) * k, &zero, t[y], CUDA_R_16F, n,
          static_cast<long long>(m) * n, batches, CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT));
    });
  }
  void run() {
    for (auto &op : ops)
      op();
    CU(cudaGetLastError());
  }
};

#pragma once
// NHWC spatial kernels; Scalar is defined by the including translation unit.

union SpatialPack8 {
  uint4 bits;
  half2 pair[4];
};

__global__ void pad_nhwc_vec8(const half *x, half *y, int n, int channels,
                              int h, int w, int oh, int ow, int ph, int pw) {
  int i = blockIdx.x * blockDim.x + threadIdx.x, groups = channels / 8;
  if (i >= n / 8)
    return;
  int channel = (i % groups) * 8, pixel = i / groups,
      row = (pixel / ow) % oh - ph, col = pixel % ow - pw,
      batch = pixel / (oh * ow);
  uint4 value = make_uint4(0, 0, 0, 0);
  if (row >= 0 && row < h && col >= 0 && col < w)
    value = *reinterpret_cast<const uint4 *>(
        x + ((batch * h + row) * w + col) * channels + channel);
  *reinterpret_cast<uint4 *>(y + i * 8) = value;
}

// Eight adjacent channels per thread, with horizontal reuse in registers.
// FP32 accumulation preserves the original tap order and output/bias rounding.
template <int KH, int KW, int SH, int PIX, int SW = 1>
__global__ void depthwise_nhwc_vec8(const half *x, const half *weights, half *y,
                                    int batch_size, int channels, int h, int w,
                                    int oh, int ow, int ph, int pw,
                                    const half *bias, int activation);
__device__ float depthwise_epilogue(float sum, const half *bias, int channel,
                                    int activation) {
  if (!bias)
    return sum;
  // Preserve the unfused convolution's FP16 output rounding.
  float x = __half2float(__float2half_rn(sum)) + __half2float(bias[channel]);
  if (activation == 1)
    return fmaxf(x, 0.f);
  if (activation == 2)
    return (x * (erff(x / 1.41421354f) + 1.f)) * 0.5f;
  return x;
}

template <int KH, int KW, int SH, int PIX, int SW>
__global__ void depthwise_nhwc_vec8(const half *x, const half *weights, half *y,
                                    int batch_size, int channels, int h, int w,
                                    int oh, int ow, int ph, int pw,
                                    const half *bias, int activation) {
  int i = blockIdx.x * blockDim.x + threadIdx.x, groups = channels / 8,
      tiles = (ow + PIX - 1) / PIX;
  if (i >= batch_size * oh * tiles * groups)
    return;
  int channel = (i % groups) * 8, t = i / groups, col = (t % tiles) * PIX,
      row = (t / tiles) % oh, batch = t / (tiles * oh);
  float2 sums[PIX][4];
#pragma unroll
  for (int p = 0; p < PIX; ++p) {
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      sums[p][v] = make_float2(0, 0);
    }
  }
#pragma unroll
  for (int a = 0; a < KH; ++a) {
    int r = row * SH - ph + a;
#pragma unroll
    for (int b = 0; b < KW; ++b) {
      half2 coeff[4];
#pragma unroll
      for (int v = 0; v < 4; ++v)
        coeff[v] = __halves2half2(
            weights[(channel + 2 * v) * KH * KW + a * KW + b],
            weights[(channel + 2 * v + 1) * KH * KW + a * KW + b]);
#pragma unroll
      for (int p = 0; p < PIX; ++p) {
        int c = (col + p) * SW - pw + b;
        if (r >= 0 && r < h && c >= 0 && c < w) {
          SpatialPack8 z;
          z.bits = *reinterpret_cast<const uint4 *>(
              x + ((batch * h + r) * w + c) * channels + channel);
#pragma unroll
          for (int v = 0; v < 4; ++v) {
            {
              float2 xx = __half22float2(z.pair[v]),
                     ww = __half22float2(coeff[v]);
              sums[p][v].x = fmaf(xx.x, ww.x, sums[p][v].x);
              sums[p][v].y = fmaf(xx.y, ww.y, sums[p][v].y);
            }
          }
        }
      }
    }
  }
#pragma unroll
  for (int p = 0; p < PIX; ++p)
    if (col + p < ow) {
      SpatialPack8 z;
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        float2 s = sums[p][v];
        z.pair[v] = __floats2half2_rn(
            depthwise_epilogue(s.x, bias, channel + 2 * v, activation),
            depthwise_epilogue(s.y, bias, channel + 2 * v + 1, activation));
      }
      *reinterpret_cast<uint4 *>(
          y + ((batch * oh + row) * ow + col + p) * channels + channel) =
          z.bits;
    }
}

// Smaller channel groups reduce register pressure for large detector filters.
template <int PIX>
__global__ void depthwise9_pair(const half *x, const half *weights, half *y,
                                int batch_size, int channels, int h, int w,
                                int oh, int ow, int ph, int pw,
                                const half *bias, int activation) {
  int i = blockIdx.x * blockDim.x + threadIdx.x, groups = channels / 2,
      tiles = (ow + PIX - 1) / PIX;
  if (i >= batch_size * oh * tiles * groups)
    return;
  int channel = (i % groups) * 2, t = i / groups, col = (t % tiles) * PIX,
      row = (t / tiles) % oh, batch = t / (tiles * oh);
  float2 sums[PIX];
#pragma unroll
  for (int p = 0; p < PIX; ++p)
    sums[p] = make_float2(0, 0);
#pragma unroll 1
  for (int a = 0; a < 9; ++a) {
    int r = row - ph + a;
#pragma unroll
    for (int b = 0; b < 9; ++b) {
      float w0 = __half2float(weights[channel * 81 + a * 9 + b]),
            w1 = __half2float(weights[(channel + 1) * 81 + a * 9 + b]);
#pragma unroll
      for (int p = 0; p < PIX; ++p) {
        int c = col + p - pw + b;
        if (r >= 0 && r < h && c >= 0 && c < w) {
          float2 v = __half22float2(*reinterpret_cast<const half2 *>(
              x + ((batch * h + r) * w + c) * channels + channel));
          sums[p].x = fmaf(v.x, w0, sums[p].x);
          sums[p].y = fmaf(v.y, w1, sums[p].y);
        }
      }
    }
  }
#pragma unroll
  for (int p = 0; p < PIX; ++p)
    if (col + p < ow)
      *reinterpret_cast<half2 *>(
          y + ((batch * oh + row) * ow + col + p) * channels + channel) =
          __floats2half2_rn(
              depthwise_epilogue(sums[p].x, bias, channel, activation),
              depthwise_epilogue(sums[p].y, bias, channel + 1, activation));
}

template <int KH, int KW, int SH, int SW, bool MAXIMUM>
__global__ void pool_nhwc_vec8(const half *x, half *y, int batch_size,
                               int channels, int h, int w, int oh, int ow,
                               int ph, int pw) {
  int i = blockIdx.x * blockDim.x + threadIdx.x, groups = channels / 8;
  if (i >= batch_size * oh * ow * groups)
    return;
  int channel = (i % groups) * 8, pixel = i / groups,
      row = (pixel / ow) % oh * SH - ph, col = pixel % ow * SW - pw,
      batch = pixel / (oh * ow), valid = 0;
  float2 sum[4];
#pragma unroll
  for (int v = 0; v < 4; ++v)
    sum[v] = make_float2(MAXIMUM ? -INFINITY : 0, MAXIMUM ? -INFINITY : 0);
#pragma unroll
  for (int a = 0; a < KH; ++a) {
#pragma unroll
    for (int b = 0; b < KW; ++b)
      if (row + a >= 0 && row + a < h && col + b >= 0 && col + b < w) {
        SpatialPack8 z;
        z.bits = *reinterpret_cast<const uint4 *>(
            x + ((batch * h + row + a) * w + col + b) * channels + channel);
#pragma unroll
        for (int v = 0; v < 4; ++v) {
          float2 s = __half22float2(z.pair[v]);
          sum[v].x = MAXIMUM ? fmaxf(sum[v].x, s.x) : sum[v].x + s.x;
          sum[v].y = MAXIMUM ? fmaxf(sum[v].y, s.y) : sum[v].y + s.y;
        }
        ++valid;
      }
  }
  SpatialPack8 z;
#pragma unroll
  for (int v = 0; v < 4; ++v)
    z.pair[v] = __floats2half2_rn(MAXIMUM ? sum[v].x : sum[v].x / valid,
                                  MAXIMUM ? sum[v].y : sum[v].y / valid);
  *reinterpret_cast<uint4 *>(y + i * 8) = z.bits;
}
template <int KH, int KW, int SH, int SW>
__global__ void depthwise_half(const half *x, const half *w, half *y, int n,
                               int channels, int h, int width, int oh, int ow,
                               int ph, int pw, bool nhwc,
                               const half *bias = nullptr, int activation = 0) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  int pixel = nhwc ? i / channels : i;
  int col = pixel % ow, row = (pixel / ow) % oh,
      bc = nhwc ? (pixel / (oh * ow)) * channels + i % channels : i / (oh * ow),
      channel = bc % channels;
  float sum = 0.f;
#pragma unroll
  for (int a = 0; a < KH; ++a) {
    int r = row * SH - ph + a;
#pragma unroll
    for (int b = 0; b < KW; ++b) {
      int c = col * SW - pw + b;
      if (r >= 0 && r < h && c >= 0 && c < width) {
        int index =
            nhwc ? (((bc / channels) * h + r) * width + c) * channels + channel
                 : (bc * h + r) * width + c;
        sum = fmaf(__half2float(x[index]),
                   __half2float(w[(channel * KH + a) * KW + b]), sum);
      }
    }
  }
  y[i] = __float2half_rn(depthwise_epilogue(sum, bias, channel, activation));
}

__global__ void pad_nhwc_kernel(const Scalar *x, Scalar *y, int n, int channels,
                                int h, int w, int oh, int ow, int ph, int pw) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  int channel = i % channels, pixel = i / channels,
      row = (pixel / ow) % oh - ph, col = pixel % ow - pw,
      batch = pixel / (oh * ow);
  y[i] = row >= 0 && row < h && col >= 0 && col < w
             ? float(x[((batch * h + row) * w + col) * channels + channel])
             : 0.f;
}
template <int SH>
__global__ void
depthwise_nhwc_tiled(const half *x, const half *weights, half *y, int channels,
                     int h, int w, int oh, int ow, int ph, int pw,
                     const half *bias = nullptr, int activation = 0) {
  // 2x8 output pixels and 64 channels, with each lane handling two channels.
  constexpr int tile_h = SH + 3;
  __shared__ half2 tile[tile_h * 10 * 32];
  int blocks_c = (channels + 63) / 64, batch = blockIdx.z / blocks_c,
      base_c = (blockIdx.z % blocks_c) * 64;
  int base_y = blockIdx.y * 2, base_x = blockIdx.x * 8,
      tid = threadIdx.y * 32 + threadIdx.x;
  for (int j = tid; j < tile_h * 10 * 32; j += 256) {
    int c = base_c + 2 * (j % 32), pixel = j / 32,
        r = base_y * SH - ph + pixel / 10, col = base_x - pw + pixel % 10;
    half2 value = __float2half2_rn(0.f);
    if (c + 1 < channels && r >= 0 && r < h && col >= 0 && col < w)
      value = *reinterpret_cast<const half2 *>(
          x + ((batch * h + r) * w + col) * channels + c);
    tile[j] = value;
  }
  __syncthreads();
  int channel = base_c + 2 * threadIdx.x, col = base_x + threadIdx.y;
  if (channel + 1 >= channels || col >= ow)
    return;
  float2 coeff[9];
#pragma unroll
  for (int k = 0; k < 9; ++k)
    coeff[k] = make_float2(__half2float(weights[channel * 9 + k]),
                           __half2float(weights[(channel + 1) * 9 + k]));
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    if (base_y + r >= oh)
      continue;
    float2 sum = make_float2(0.f, 0.f);
#pragma unroll
    for (int a = 0; a < 3; ++a) {
#pragma unroll
      for (int b = 0; b < 3; ++b) {
        float2 value = __half22float2(
            tile[((r * SH + a) * 10 + threadIdx.y + b) * 32 + threadIdx.x]);
        sum.x = fmaf(value.x, coeff[a * 3 + b].x, sum.x);
        sum.y = fmaf(value.y, coeff[a * 3 + b].y, sum.y);
      }
    }
    *reinterpret_cast<half2 *>(
        y + ((batch * oh + base_y + r) * ow + col) * channels + channel) =
        __floats2half2_rn(
            depthwise_epilogue(sum.x, bias, channel, activation),
            depthwise_epilogue(sum.y, bias, channel + 1, activation));
  }
}
__global__ void pool_nhwc_kernel(const Scalar *x, Scalar *y, int n,
                                 int channels, int h, int w, int oh, int ow,
                                 int kh, int kw, int sh, int sw, int ph, int pw,
                                 bool maximum) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  int channel = i % channels, pixel = i / channels,
      row = (pixel / ow) % oh * sh - ph, col = pixel % ow * sw - pw,
      batch = pixel / (oh * ow), valid = 0;
  float value = maximum ? -INFINITY : 0.f;
  for (int a = 0; a < kh; ++a)
    for (int b = 0; b < kw; ++b)
      if (row + a >= 0 && row + a < h && col + b >= 0 && col + b < w) {
        float z = x[((batch * h + row + a) * w + col + b) * channels + channel];
        value = maximum ? fmaxf(value, z) : value + z;
        ++valid;
      }
  y[i] = maximum ? value : value / valid;
}

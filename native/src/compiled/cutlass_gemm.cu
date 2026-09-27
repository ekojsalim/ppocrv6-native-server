#include "cutlass_gemm.h"
#include <memory>
#include <stdexcept>
#include <string>
// CUTLASS is a build-time template library; no additional shared library is
// used.
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/epilogue/thread/linear_combination_bias_elementwise.h>
#include <cutlass/epilogue/thread/linear_combination_generic.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal_with_broadcast.h>

inline void cutlass_check(cutlass::Status status) {
  if (status != cutlass::Status::kSuccess)
    throw std::runtime_error(std::string("CUTLASS: ") +
                             cutlassGetStatusString(status));
}

template <template <class> class Activation, int M, int N, int K,
          int Stages = 3>
std::function<void()> make_cutlass_gemm(const half *x, const half *w,
                                        const half *bias, half *y, int rows,
                                        int ci, int co, cudaStream_t stream) {
  using H = cutlass::half_t;
  using Epilogue =
      cutlass::epilogue::thread::LinearCombinationGeneric<Activation, H, 8, H,
                                                          float>;
  using Gemm = cutlass::gemm::device::Gemm<
      H, cutlass::layout::RowMajor, H, cutlass::layout::ColumnMajor, H,
      cutlass::layout::RowMajor, H, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<M, N, K>,
      cutlass::gemm::GemmShape<(M == 64 ? 32 : 64), (N == 64 ? 32 : 64), K>,
      cutlass::gemm::GemmShape<16, 8, 16>, Epilogue,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages>;
  // C has stride zero: every row reads the same channel bias vector.
  typename Gemm::Arguments args({rows, co, ci},
                                {reinterpret_cast<const H *>(x), ci},
                                {reinterpret_cast<const H *>(w), ci},
                                {reinterpret_cast<const H *>(bias), 0},
                                {reinterpret_cast<H *>(y), co}, {1.f, 1.f});
  auto op = std::make_shared<Gemm>();
  cutlass_check(op->can_implement(args));
  if (op->get_workspace_size(args))
    throw std::runtime_error("Unexpected CUTLASS workspace");
  cutlass_check(op->initialize(args, nullptr, stream));
  return [op, stream] { cutlass_check((*op)(stream)); };
}

template <int M, int N, int K, int Stages = 3>
std::function<void()>
cutlass_residual_gemm(const half *x, const half *w, const half *bias,
                      const half *residual, half *y, int rows, int ci, int co,
                      cudaStream_t stream) {
  using H = cutlass::half_t;
  using Epilogue = cutlass::epilogue::thread::LinearCombinationBiasElementwise<
      H, H, float, H, H, 8, cutlass::epilogue::thread::Identity<float>,
      cutlass::plus<float>, false, H>;
  using Gemm = cutlass::gemm::device::GemmUniversalWithBroadcast<
      H, cutlass::layout::RowMajor, H, cutlass::layout::ColumnMajor, H,
      cutlass::layout::RowMajor, H, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<M, N, K>,
      cutlass::gemm::GemmShape<(M == 64 ? 32 : 64), (N == 64 ? 32 : 64), K>,
      cutlass::gemm::GemmShape<16, 8, 16>, Epilogue,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages>;
  typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm,
                                {rows, co, ci}, 1, {1.f, 1.f}, x, w, residual,
                                y, const_cast<half *>(bias), nullptr, 0, 0, 0,
                                0, 0, 0, ci, ci, co, co, 0, 0);
  auto op = std::make_shared<Gemm>();
  cutlass_check(op->can_implement(args));
  if (op->get_workspace_size(args))
    throw std::runtime_error("Unexpected CUTLASS residual workspace");
  cutlass_check(op->initialize(args, nullptr, stream));
  return [op, stream] { cutlass_check((*op)(stream)); };
}

template <int M, int N, int K, int Stages = 3>
std::function<void()> cutlass_gemm(const half *x, const half *w,
                                   const half *bias, half *y, int rows, int ci,
                                   int co, int activation, cudaStream_t stream,
                                   const half *residual = nullptr) {
  using namespace cutlass::epilogue::thread;
  if (residual) {
    if (activation != 0)
      throw std::runtime_error("CUTLASS residual activation unsupported");
    return cutlass_residual_gemm<M, N, K, Stages>(x, w, bias, residual, y, rows,
                                                  ci, co, stream);
  }
  if (activation == 2)
    return make_cutlass_gemm<GELU_taylor, M, N, K, Stages>(x, w, bias, y, rows,
                                                           ci, co, stream);
  if (activation == 1)
    return make_cutlass_gemm<ReLu, M, N, K, Stages>(x, w, bias, y, rows, ci, co,
                                                    stream);
  return make_cutlass_gemm<Identity, M, N, K, Stages>(x, w, bias, y, rows, ci,
                                                      co, stream);
}

std::vector<std::function<void()>>
make_cutlass_candidates(const half *x, const half *w, const half *bias,
                        const half *residual, half *y, int rows, int ci, int co,
                        int activation, cudaStream_t stream) {
  std::vector<std::function<void()>> result;
  result.push_back(cutlass_gemm<128, 128, 64>(x, w, bias, y, rows, ci, co,
                                              activation, stream, residual));
  result.push_back(cutlass_gemm<128, 128, 32>(x, w, bias, y, rows, ci, co,
                                              activation, stream, residual));
  result.push_back(cutlass_gemm<128, 256, 32>(x, w, bias, y, rows, ci, co,
                                              activation, stream, residual));
  result.push_back(cutlass_gemm<256, 128, 32>(x, w, bias, y, rows, ci, co,
                                              activation, stream, residual));
  result.push_back(cutlass_gemm<64, 128, 64>(x, w, bias, y, rows, ci, co,
                                             activation, stream, residual));
  result.push_back(cutlass_gemm<128, 64, 32>(x, w, bias, y, rows, ci, co,
                                             activation, stream, residual));
  result.push_back(cutlass_gemm<64, 64, 64>(x, w, bias, y, rows, ci, co,
                                            activation, stream, residual));
  return result;
}

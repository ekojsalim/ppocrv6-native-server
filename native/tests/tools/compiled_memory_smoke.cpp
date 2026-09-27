// Development-only validation of shared scratch, graph replay and rebinding.
#include "ppocrv6_native/common/cuda_ptr.h"
#include "ppocrv6_native/common/workspace.h"
#include "ppocrv6_native/engine/model_engine.h"
#include <cmath>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
using namespace ppocrv6_native;
struct Case {
  engine::ModelEngine &model;
  engine::ModelContext &context;
  int profile;
  std::vector<float> expected;
};
int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::runtime_error("usage: compiled_memory_smoke REC DET");
    auto rec = engine::load_model_engine(argv[1]);
    auto det = engine::load_model_engine(argv[2]);
    Workspace workspace;
    workspace.reserve(std::max(rec->workspace_bytes(), det->workspace_bytes()));
    auto rc = rec->create_context(), dc = det->create_context();
    cudaStream_t stream;
    PPOCRV6_CUDA_CHECK(
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::vector<Case> cases;
    for (int p = 0; p < rec->profiles(); ++p)
      cases.push_back({*rec, *rc, p, {}});
    for (int p = 0; p < det->profiles(); ++p)
      cases.push_back({*det, *dc, p, {}});
    float max_change = 0;
    for (int pass = 0; pass < 3; ++pass) {
      // Moving storage must discard captured pointers and rebuild safely.
      if (pass == 2)
        workspace.reserve(workspace.size() + 256);
      for (auto &test : cases) {
        auto shape = test.model.profile_shape(test.profile, true);
        if (!test.context.select_profile(test.profile, stream) ||
            !test.context.set_input_shape(shape))
          throw std::runtime_error("shape failed");
        size_t count = 1;
        for (int i = 0; i < 4; ++i)
          count *= shape.d[i];
        std::vector<float> host(count);
        for (size_t i = 0; i < count; ++i)
          host[i] = float(int(i % 251) - 125) / 125;
        CudaPtr<float> input(count);
        PPOCRV6_CUDA_CHECK(cudaMemcpy(input.get(), host.data(), count * 4,
                                      cudaMemcpyHostToDevice));
        auto out = test.context.output_shape();
        size_t n = 1;
        for (int i = 0; i < out.nbDims; ++i)
          n *= out.d[i];
        bool fp32 = test.model.output_type() == engine::ElementType::Float32;
        CudaPtr<unsigned char> output(n * (fp32 ? 4 : 2));
        std::vector<float> actual(n);
        {
          Workspace::Lease lease(workspace, stream);
          test.context.bind_workspace(lease.data(), lease.size());
          // Every graph must initialize its live values after another shape
          // has used the same allocation; NaNs expose stale arena reads.
          PPOCRV6_CUDA_CHECK(
              cudaMemsetAsync(lease.data(), 0xff, lease.size(), stream));
          test.context.enqueue(input.get(), output.get(), stream);
          PPOCRV6_CUDA_CHECK(cudaStreamSynchronize(stream));
          if (fp32) {
            PPOCRV6_CUDA_CHECK(cudaMemcpy(actual.data(), output.get(), n * 4,
                                          cudaMemcpyDeviceToHost));
          } else {
            std::vector<half> packed(n);
            PPOCRV6_CUDA_CHECK(cudaMemcpy(packed.data(), output.get(), n * 2,
                                          cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < n; ++i)
              actual[i] = __half2float(packed[i]);
          }
        }
        double error_squared = 0, reference_squared = 0;
        float case_change = 0;
        for (size_t i = 0; i < n; ++i) {
          if (!std::isfinite(actual[i]))
            throw std::runtime_error(
                "Non-finite output after scratch overwrite");
          if (pass) {
            float error = std::abs(actual[i] - test.expected[i]);
            max_change = std::max(max_change, error);
            case_change = std::max(case_change, error);
            error_squared += double(error) * error;
            reference_squared += double(test.expected[i]) * test.expected[i];
            if (pass == 1 && error != 0)
              throw std::runtime_error("Output changed after graph replay");
          }
        }
        double relative_l2 =
            std::sqrt(error_squared / std::max(reference_squared, 1e-20));
        std::cout << "max_abs=" << case_change << " relative_l2=" << relative_l2
                  << ' ';
        if (relative_l2 > .02)
          throw std::runtime_error("Excessive output drift after retuning");
        if (!pass)
          test.expected = std::move(actual);
        std::cout << "pass=" << pass << " shape=" << shape.d[0] << 'x'
                  << shape.d[2] << 'x' << shape.d[3] << std::endl;
      }
    }
    rc.reset();
    dc.reset();
    PPOCRV6_CUDA_CHECK(cudaStreamDestroy(stream));
    std::cout << "PASS: all profiles, poisoned scratch, graph replay, "
                 "workspace growth; max_change="
              << max_change << '\n';
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

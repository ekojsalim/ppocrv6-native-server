#include "ppocrv6_native/common/cuda_check.h"
#include "ppocrv6_native/common/cuda_ptr.h"
#include "ppocrv6_native/detection/db_postprocess.h"
#include "ppocrv6_native/engine/model_engine.h"
#include "ppocrv6_native/kernels/classifier.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
using namespace ppocrv6_native;
int main(int argc, char **argv) {
  try {
    if (argc == 6 && std::string(argv[1]) == "--postprocess") {
      int h = std::stoi(argv[3]), w = std::stoi(argv[4]);
      if (h < 1 || w < 1)
        throw std::runtime_error("Invalid map shape");
      std::vector<float> map(size_t(h) * w);
      std::ifstream file(argv[2], std::ios::binary | std::ios::ate);
      if (!file || file.tellg() != std::streamoff(map.size() * 4))
        throw std::runtime_error("Map size mismatch");
      file.seekg(0);
      file.read(reinterpret_cast<char *>(map.data()), map.size() * 4);
      auto boxes = detection::postprocess_db_map(map.data(), h, w, h, w);
      std::ofstream output(argv[5]);
      output << detection::detection_boxes_to_json(boxes) << '\n';
      if (!output)
        throw std::runtime_error("Box output failed");
      return 0;
    }
    if (argc != 8 && argc != 9)
      throw std::runtime_error(
          "usage: model_engine_smoke CATALOG INPUT OUTPUT "
          "BATCH HEIGHT WIDTH REPEATS [CLASSIFIER_DIRECTORY]");
    int batch = std::stoi(argv[4]), h = std::stoi(argv[5]),
        w = std::stoi(argv[6]), repeats = std::stoi(argv[7]);
    if (batch < 1 || h < 1 || w < 1 || repeats < 1)
      throw std::runtime_error("Invalid shape/repeats");
    auto model = engine::load_model_engine(argv[1]);
    int profile = -1;
    for (int p = 0; p < model->profiles(); ++p) {
      auto lo = model->profile_shape(p, false),
           hi = model->profile_shape(p, true);
      if (lo.nbDims == 4 && batch >= lo.d[0] && batch <= hi.d[0] &&
          h >= lo.d[2] && h <= hi.d[2] && w >= lo.d[3] && w <= hi.d[3]) {
        profile = p;
        break;
      }
    }
    if (profile < 0)
      throw std::runtime_error("Unsupported test shape");
    cudaStream_t stream;
    PPOCRV6_CUDA_CHECK(
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    auto context = model->create_context();
    if (!context->select_profile(profile, stream) ||
        !context->set_input_shape({batch, 3, h, w}))
      throw std::runtime_error("Shape setup failed");
    size_t input_count = size_t(batch) * 3 * h * w;
    auto shape = context->output_shape();
    size_t output_count = 1;
    for (int i = 0; i < shape.nbDims; ++i)
      output_count *= shape.d[i];
    size_t element =
        model->output_type() == engine::ElementType::Float32 ? 4 : 2;
    std::vector<float> input(input_count);
    std::ifstream f(argv[2], std::ios::binary | std::ios::ate);
    if (!f || f.tellg() != std::streamoff(input_count * 4))
      throw std::runtime_error("Input size mismatch");
    f.seekg(0);
    f.read(reinterpret_cast<char *>(input.data()), input_count * 4);
    CudaPtr<float> x(input_count);
    CudaPtr<unsigned char> y(output_count * element);
    CudaPtr<unsigned char> workspace(
        std::max<int64_t>(1, model->workspace_bytes(profile)));
    context->bind_workspace(workspace.get(), model->workspace_bytes(profile));
    CudaPtr<half> weight, bias;
    CudaPtr<int> ids, partial_ids;
    CudaPtr<float> probability, logits, partial_max, partial_sum;
    int rows = 0, hidden = 0, vocab = 0;
    if (argc == 9) {
      if (element != 2 || shape.nbDims != 3)
        throw std::runtime_error(
            "Classifier requires rank-three FP16 hidden states");
      hidden = shape.d[2];
      rows = output_count / hidden;
      auto root = std::filesystem::path(argv[8]);
      vocab = std::filesystem::file_size(root / "bias.fp16.bin") / 2;
      auto load = [&](CudaPtr<half> &target, const char *name, size_t count) {
        std::vector<half> host(count);
        std::ifstream f(root / name, std::ios::binary | std::ios::ate);
        if (!f || f.tellg() != std::streamoff(count * 2))
          throw std::runtime_error("Classifier file size mismatch");
        f.seekg(0);
        f.read(reinterpret_cast<char *>(host.data()), count * 2);
        target.reset(count);
        PPOCRV6_CUDA_CHECK(cudaMemcpy(target.get(), host.data(), count * 2,
                                      cudaMemcpyHostToDevice));
      };
      load(weight, "weight.fp16.bin", size_t(vocab) * hidden);
      load(bias, "bias.fp16.bin", vocab);
      auto ws = kernels::tiled_classifier_workspace_shape(rows, vocab, 16, 256);
      ids.reset(rows);
      probability.reset(rows);
      logits.reset(rows);
      partial_ids.reset(ws.partial_count);
      partial_max.reset(ws.partial_count);
      partial_sum.reset(ws.partial_count);
    }
    auto classify = [&] {
      if (weight)
        kernels::cuda_linear_argmax_prob_wmma_mode(
            reinterpret_cast<half *>(y.get()), weight.get(), bias.get(),
            ids.get(), probability.get(), logits.get(), rows, hidden, vocab,
            256, partial_ids.get(), partial_max.get(), partial_sum.get(), true,
            stream);
    };

    PPOCRV6_CUDA_CHECK(cudaMemcpyAsync(x.get(), input.data(), input_count * 4,
                                       cudaMemcpyHostToDevice, stream));
    if (!context->enqueue(x.get(), y.get(), stream))
      throw std::runtime_error("Model enqueue failed");
    classify();
    PPOCRV6_CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaEvent_t start, stop;
    PPOCRV6_CUDA_CHECK(cudaEventCreate(&start));
    PPOCRV6_CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times;
    for (int i = 0; i < repeats; ++i) {
      PPOCRV6_CUDA_CHECK(cudaEventRecord(start, stream));
      if (!context->enqueue(x.get(), y.get(), stream))
        throw std::runtime_error("Model enqueue failed");
      classify();
      PPOCRV6_CUDA_CHECK(cudaEventRecord(stop, stream));
      PPOCRV6_CUDA_CHECK(cudaEventSynchronize(stop));
      float ms;
      PPOCRV6_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
      times.push_back(ms);
    }
    std::vector<unsigned char> output(output_count * element);
    PPOCRV6_CUDA_CHECK(cudaMemcpy(output.data(), y.get(), output.size(),
                                  cudaMemcpyDeviceToHost));
    std::ofstream out(argv[3], std::ios::binary);
    out.write(reinterpret_cast<char *>(output.data()), output.size());
    if (!out)
      throw std::runtime_error("Output write failed");
    if (weight) {
      std::vector<int> host(rows);
      PPOCRV6_CUDA_CHECK(cudaMemcpy(host.data(), ids.get(), rows * sizeof(int),
                                    cudaMemcpyDeviceToHost));
      std::ofstream f(std::string(argv[3]) + ".ids.i32", std::ios::binary);
      f.write(reinterpret_cast<char *>(host.data()), rows * sizeof(int));
      if (!f)
        throw std::runtime_error("Classifier output write failed");
    }
    std::sort(times.begin(), times.end());
    std::cout << "{\"backend\":\"" << model->backend_name()
              << "\",\"gpu_median_ms\":" << times[times.size() / 2]
              << ",\"output_element_bytes\":" << element
              << ",\"output_shape\":" << engine::dims_to_string(shape) << "}\n";
    context.reset();
    PPOCRV6_CUDA_CHECK(cudaEventDestroy(start));
    PPOCRV6_CUDA_CHECK(cudaEventDestroy(stop));
    PPOCRV6_CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

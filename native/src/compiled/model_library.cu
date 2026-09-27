#include "ppocrv6_native/compiled/model_abi.h"
#include "runtime.cuh"
#include <fstream>

// The generated graph requires Runtime and Scalar from runtime.cuh.
#include "model.inc"

#ifndef PPOCR_MODEL_DETECTOR
#define PPOCR_MODEL_DETECTOR 0
#endif
static_assert(half_storage,
              "Compiled model bundles use FP16 activation storage");
namespace {
thread_local std::string last_error;
__global__ void copy_input(const float *x, half *y, size_t valid,
                           size_t total) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < total)
    y[i] = i < valid ? __float2half_rn(x[i]) : __float2half_rn(0);
}
__global__ void copy_float_output(const half *x, float *y, size_t n) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = __half2float(x[i]);
}
struct Instance {
  Runtime r;
  cudaEvent_t ready = nullptr, done = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  bool warmed = false;
  ~Instance() {
    if (r.stream)
      cudaStreamSynchronize(r.stream);
    if (executable)
      cudaGraphExecDestroy(executable);
    if (graph)
      cudaGraphDestroy(graph);
    if (ready)
      cudaEventDestroy(ready);
    if (done)
      cudaEventDestroy(done);
  }
  void init(void *weights, void *scratch, size_t bytes,
            PpocrModelResources *resources) {
    constexpr size_t arena_bytes = (ARENA_ELEMENTS * 2 + 255) & ~size_t(255);
    if (!resources || !weights || !scratch ||
        bytes < arena_bytes + Runtime::workspace_limit)
      throw std::runtime_error("Invalid compiled model storage");
    r.weights = static_cast<half *>(weights);
    r.arena = static_cast<half *>(scratch);
    r.workspace = static_cast<unsigned char *>(scratch) + arena_bytes;
    r.acquire_blas = [resources] {
      if (!resources->blas) {
        cublasHandle_t handle = nullptr;
        BL(cublasCreate(&handle));
        resources->blas = handle;
      }
      return static_cast<cublasHandle_t>(resources->blas);
    };
    build_model(r);
    if (r.workspace_bytes > Runtime::workspace_limit)
      throw std::runtime_error("Compiled workspace exceeds declared capacity");
    CU(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    CU(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
  }
  void enqueue(const float *input, void *output, int batch,
               cudaStream_t caller) {
    if (!input || !output || batch < 1 || batch > MODEL_INPUT_SHAPE[0])
      throw std::runtime_error("Invalid compiled model input/batch");
    CU(cudaEventRecord(ready, caller));
    CU(cudaStreamWaitEvent(r.stream, ready));
    size_t input_n = INPUT_ELEMENTS / MODEL_INPUT_SHAPE[0] * batch;
    copy_input<<<(INPUT_ELEMENTS + 255) / 256, 256, 0, r.stream>>>(
        input, r.t[0], input_n, INPUT_ELEMENTS);
    if (!warmed) {
      r.run();
      CU(cudaStreamSynchronize(r.stream));
      CU(cudaStreamBeginCapture(r.stream, cudaStreamCaptureModeThreadLocal));
      r.run();
      CU(cudaStreamEndCapture(r.stream, &graph));
      CU(cudaGraphInstantiate(&executable, graph, 0));
      warmed = true;
    }
    CU(cudaGraphLaunch(executable, r.stream));
    size_t output_n = OUTPUT_ELEMENTS / MODEL_INPUT_SHAPE[0] * batch;
    if (PPOCR_MODEL_DETECTOR)
      copy_float_output<<<(output_n + 255) / 256, 256, 0, r.stream>>>(
          r.t[OUTPUT_ID], static_cast<float *>(output), output_n);
    else
      CU(cudaMemcpyAsync(output, r.t[OUTPUT_ID], output_n * 2,
                         cudaMemcpyDeviceToDevice, r.stream));
    CU(cudaGetLastError());
    CU(cudaEventRecord(done, r.stream));
    CU(cudaStreamWaitEvent(caller, done));
  }
};
} // namespace
#define API extern "C" __attribute__((visibility("default")))
API const PpocrModelInfo *ppocr_model_info() {
  static const PpocrModelInfo info = {
      PPOCR_MODEL_ABI_VERSION,
      {MODEL_INPUT_SHAPE[0], MODEL_INPUT_SHAPE[1], MODEL_INPUT_SHAPE[2],
       MODEL_INPUT_SHAPE[3]},
      MODEL_OUTPUT_RANK,
      {MODEL_OUTPUT_SHAPE[0], MODEL_OUTPUT_SHAPE[1], MODEL_OUTPUT_SHAPE[2],
       MODEL_OUTPUT_SHAPE[3]},
      PPOCR_MODEL_DETECTOR ? 4 : 2,
      WEIGHT_ELEMENTS * 4,
      ((ARENA_ELEMENTS * 2 + 255) & ~size_t(255)) + Runtime::workspace_limit};
  return &info;
}
API void *ppocr_model_load_weights(const char *path) {
  void *device = nullptr;
  try {
    last_error.clear();
    if (!path)
      throw std::runtime_error("Null weights path");
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f || f.tellg() != std::streamoff(WEIGHT_ELEMENTS * 4))
      throw std::runtime_error("Invalid compiled model weight size");
    std::vector<float> host(WEIGHT_ELEMENTS);
    f.seekg(0);
    f.read(reinterpret_cast<char *>(host.data()), host.size() * 4);
    if (!f)
      throw std::runtime_error("Cannot read compiled model weights");
    std::vector<half> packed(host.begin(), host.end());
    CU(cudaMalloc(&device, WEIGHT_ELEMENTS * 2));
    CU(cudaMemcpy(device, packed.data(), WEIGHT_ELEMENTS * 2,
                  cudaMemcpyHostToDevice));
    return device;
  } catch (const std::exception &e) {
    cudaFree(device);
    last_error = e.what();
    return nullptr;
  }
}
API void *ppocr_model_create(void *weights, void *scratch, size_t bytes,
                             PpocrModelResources *resources) {
  try {
    last_error.clear();
    auto p = std::make_unique<Instance>();
    p->init(weights, scratch, bytes, resources);
    return p.release();
  } catch (const std::exception &e) {
    last_error = e.what();
    return nullptr;
  }
}
API int ppocr_model_enqueue(void *handle, const float *input, void *output,
                            int batch, cudaStream_t stream) {
  try {
    last_error.clear();
    if (!handle)
      throw std::runtime_error("Null model handle");
    static_cast<Instance *>(handle)->enqueue(input, output, batch, stream);
    return 0;
  } catch (const std::exception &e) {
    // A failed enqueue may have submitted work without recording done. Keep
    // borrowed scratch exclusive until that work has completed too.
    if (handle)
      cudaStreamSynchronize(static_cast<Instance *>(handle)->r.stream);
    last_error = e.what();
    return -1;
  }
}
API void ppocr_model_release_resources(PpocrModelResources *resources) {
  if (resources && resources->blas) {
    cublasDestroy(static_cast<cublasHandle_t>(resources->blas));
    resources->blas = nullptr;
  }
}
API void ppocr_model_destroy(void *handle) {
  delete static_cast<Instance *>(handle);
}
API const char *ppocr_model_error() { return last_error.c_str(); }

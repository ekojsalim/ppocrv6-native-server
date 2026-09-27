#pragma once
#include <cuda_runtime_api.h>
#include <stddef.h>
#include <stdint.h>

// Versioned ABI for trusted, offline-compiled model bundles. One handle per
// compiled shape per worker. Device input is NCHW float32; output is contiguous
// FP16 recognition features or float32 detector probabilities. Calls enqueue work on the
// supplied stream. destroy waits for all work owned by the handle. Handles are
// not reentrant.
#define PPOCR_MODEL_ABI_VERSION 2
struct PpocrModelInfo {
  uint32_t abi_version;
  int input_shape[4];
  int output_rank;
  int output_shape[4];
  int output_element_bytes;
  size_t weight_file_bytes;
  size_t scratch_bytes;
};
// Owned by one worker context, shared by its serial shape handles.
struct PpocrModelResources {
  void *blas = nullptr;
};
extern "C" {
const PpocrModelInfo *ppocr_model_info();
// Weights are immutable, shared between shapes, and released with cudaFree.
void *ppocr_model_load_weights(const char *weights_path);
// Borrowed storage must outlive the handle; scratch is exclusive until stream
// completion. Recreate the handle if the scratch address changes.
void *ppocr_model_create(void *weights, void *scratch, size_t scratch_bytes,
                         PpocrModelResources *resources);
void ppocr_model_release_resources(PpocrModelResources *resources);
int ppocr_model_enqueue(void *handle, const float *device_input,
                        void *device_output, int batch, cudaStream_t stream);
void ppocr_model_destroy(void *handle);
const char *ppocr_model_error();
}

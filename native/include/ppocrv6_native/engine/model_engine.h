#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>

namespace ppocrv6_native::engine {
struct TensorShape {
  int nbDims = 0;
  int64_t d[8]{};
  TensorShape() = default;
  TensorShape(std::initializer_list<int64_t> values) {
    if (values.size() > 8)
      throw std::runtime_error("Tensor rank exceeds eight");
    for (auto v : values)
      d[nbDims++] = v;
  }
};
enum class ElementType { Float32, Float16 };
inline std::string dims_to_string(const TensorShape &shape) {
  std::string s = "[";
  for (int i = 0; i < shape.nbDims; ++i) {
    if (i)
      s += ",";
    s += std::to_string(shape.d[i]);
  }
  return s + "]";
}
class ModelContext {
public:
  virtual ~ModelContext() = default;
  virtual bool select_profile(int profile, cudaStream_t stream) = 0;
  virtual bool set_input_shape(const TensorShape &shape) = 0;
  virtual TensorShape output_shape() const = 0;
  virtual void bind_workspace(void *data, int64_t bytes) = 0;
  virtual bool enqueue(const float *input, void *output,
                       cudaStream_t stream) = 0;
};
class ModelEngine {
public:
  virtual ~ModelEngine() = default;
  virtual int profiles() const = 0;
  virtual TensorShape profile_shape(int index, bool maximum) const = 0;
  virtual int64_t workspace_bytes(int profile = -1) const = 0;
  virtual std::string input_name() const = 0;
  virtual std::string output_name() const = 0;
  virtual ElementType input_type() const = 0;
  virtual ElementType output_type() const = 0;
  virtual const char *backend_name() const = 0;
  virtual std::unique_ptr<ModelContext> create_context() = 0;
};
// Load an ABI-compatible compiled CUDA model catalog.
std::unique_ptr<ModelEngine> load_model_engine(const std::string &path);
} // namespace ppocrv6_native::engine

#include "ppocrv6_native/compiled/model_abi.h"
#include "ppocrv6_native/engine/model_engine.h"
#include <algorithm>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <vector>
namespace ppocrv6_native::engine {
namespace {
struct Weights {
  std::mutex mutex;
  void *data = nullptr;
  ~Weights() {
    if (data)
      cudaFree(data);
  }
};
struct Library {
  void *so = nullptr;
  PpocrModelInfo info{};
  std::string weights;
  std::shared_ptr<Weights> device_weights;
  decltype(&ppocr_model_load_weights) load_weights = nullptr;
  decltype(&ppocr_model_release_resources) release_resources = nullptr;
  decltype(&ppocr_model_create) create = nullptr;
  decltype(&ppocr_model_enqueue) enqueue = nullptr;
  decltype(&ppocr_model_destroy) destroy = nullptr;
  decltype(&ppocr_model_error) error = nullptr;
  ~Library() {
    if (so)
      dlclose(so);
  }
  template <class T> T symbol(const char *name) {
    auto p = dlsym(so, name);
    if (!p)
      throw std::runtime_error(std::string("Missing compiled model symbol: ") +
                               name);
    return reinterpret_cast<T>(p);
  }
  void load(const std::filesystem::path &binary,
            const std::filesystem::path &weight) {
    so = dlopen(binary.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!so)
      throw std::runtime_error(std::string("Cannot load compiled model: ") +
                               dlerror());
    auto get = symbol<decltype(&ppocr_model_info)>("ppocr_model_info");
    auto *metadata = get();
    if (!metadata || metadata->abi_version != PPOCR_MODEL_ABI_VERSION)
      throw std::runtime_error("Compiled model ABI mismatch");
    info = *metadata;
    if (info.output_rank < 1 || info.output_rank > 4 ||
        (info.output_element_bytes != 2 && info.output_element_bytes != 4))
      throw std::runtime_error("Invalid compiled model output metadata");
    for (int v : info.input_shape)
      if (v <= 0)
        throw std::runtime_error("Invalid compiled model input shape");
    for (int i = 0; i < info.output_rank; ++i)
      if (info.output_shape[i] <= 0)
        throw std::runtime_error("Invalid compiled model output shape");
    if (info.input_shape[1] != 3 || info.output_shape[0] != info.input_shape[0])
      throw std::runtime_error("Invalid compiled model batch/channels");
    if (std::filesystem::file_size(weight) != info.weight_file_bytes)
      throw std::runtime_error("Compiled weight size mismatch");
    weights = weight.string();
    load_weights = symbol<decltype(load_weights)>("ppocr_model_load_weights");
    release_resources =
        symbol<decltype(release_resources)>("ppocr_model_release_resources");
    create = symbol<decltype(create)>("ppocr_model_create");
    enqueue = symbol<decltype(enqueue)>("ppocr_model_enqueue");
    destroy = symbol<decltype(destroy)>("ppocr_model_destroy");
    error = symbol<decltype(error)>("ppocr_model_error");
  }
};
using Libraries = std::vector<std::shared_ptr<Library>>;
struct Context final : ModelContext {
  Libraries libraries;
  std::vector<void *> handles;
  PpocrModelResources resources;
  void *scratch = nullptr;
  size_t scratch_bytes = 0;
  int profile = -1;
  TensorShape input;
  explicit Context(Libraries l)
      : libraries(std::move(l)), handles(libraries.size(), nullptr) {}
  ~Context() {
    for (size_t i = 0; i < handles.size(); ++i)
      if (handles[i])
        libraries[i]->destroy(handles[i]);
    libraries.front()->release_resources(&resources);
  }
  bool select_profile(int p, cudaStream_t) override {
    if (p < 0 || p >= int(libraries.size()))
      return false;
    profile = p;
    return true;
  }
  bool set_input_shape(const TensorShape &s) override {
    if (profile < 0 || s.nbDims != 4)
      return false;
    const auto &m = libraries[profile]->info;
    if (s.d[0] < 1 || s.d[0] > m.input_shape[0])
      return false;
    for (int i = 1; i < 4; ++i)
      if (s.d[i] != m.input_shape[i])
        return false;
    input = s;
    return true;
  }
  TensorShape output_shape() const override {
    if (profile < 0 || input.nbDims != 4)
      throw std::runtime_error("Compiled input shape unset");
    const auto &m = libraries[profile]->info;
    TensorShape s;
    s.nbDims = m.output_rank;
    for (int i = 0; i < s.nbDims; ++i)
      s.d[i] = i ? m.output_shape[i] : input.d[0];
    return s;
  }
  void bind_workspace(void *data, int64_t bytes) override {
    if (!data || bytes <= 0)
      throw std::runtime_error("Missing compiled scratch storage");
    if (data != scratch || size_t(bytes) != scratch_bytes) {
      // Existing graphs embed scratch addresses. The caller holds the lease,
      // so no inference is in flight while old handles are discarded.
      for (size_t i = 0; i < handles.size(); ++i) {
        if (handles[i])
          libraries[i]->destroy(handles[i]);
        handles[i] = nullptr;
      }
      scratch = data;
      scratch_bytes = bytes;
    }
  }
  bool enqueue(const float *x, void *y, cudaStream_t s) override {
    if (profile < 0 || input.nbDims != 4)
      throw std::runtime_error("Compiled profile unset");
    auto &m = *libraries[profile];
    if (!handles[profile]) {
      std::lock_guard<std::mutex> lock(m.device_weights->mutex);
      if (!m.device_weights->data) {
        m.device_weights->data = m.load_weights(m.weights.c_str());
        if (!m.device_weights->data)
          throw std::runtime_error(m.error());
      }
      handles[profile] =
          m.create(m.device_weights->data, scratch, scratch_bytes, &resources);
      if (!handles[profile])
        throw std::runtime_error(m.error());
    }
    if (m.enqueue(handles[profile], x, y, input.d[0], s))
      throw std::runtime_error(m.error());
    return true;
  }
};
struct Engine final : ModelEngine {
  Libraries libraries;
  explicit Engine(const std::string &path) {
    std::filesystem::path manifest = path;
    if (std::filesystem::is_directory(manifest))
      manifest /= "models.tsv";
    std::ifstream f(manifest);
    if (!f)
      throw std::runtime_error("Cannot open compiled model catalog: " +
                               manifest.string());
    std::map<std::filesystem::path, std::shared_ptr<Weights>> weights_by_path;
    std::string line;
    while (std::getline(f, line)) {
      if (line.empty() || line[0] == '#')
        continue;
      std::istringstream row(line);
      std::string binary, weights, extra;
      if (!(row >> binary >> weights) || (row >> extra))
        throw std::runtime_error(
            "Catalog rows require library and weights relative paths");
      auto p = std::make_shared<Library>();
      p->load(manifest.parent_path() / binary,
              manifest.parent_path() / weights);
      if (!libraries.empty() &&
          (p->info.output_rank != libraries[0]->info.output_rank ||
           p->info.output_element_bytes !=
               libraries[0]->info.output_element_bytes))
        throw std::runtime_error("Mixed model types in catalog");
      auto key = std::filesystem::canonical(p->weights);
      auto &shared = weights_by_path[key];
      if (!shared)
        shared = std::make_shared<Weights>();
      p->device_weights = shared;
      libraries.push_back(std::move(p));
    }
    if (libraries.empty())
      throw std::runtime_error("Empty compiled model catalog");
    std::stable_sort(libraries.begin(), libraries.end(), [](auto &a, auto &b) {
      return a->info.input_shape[0] < b->info.input_shape[0];
    });
  }
  int profiles() const override { return libraries.size(); }
  TensorShape profile_shape(int p, bool max) const override {
    const auto &m = libraries.at(p)->info;
    return {max ? m.input_shape[0] : 1, m.input_shape[1], m.input_shape[2],
            m.input_shape[3]};
  }
  int64_t workspace_bytes(int profile) const override {
    if (profile >= 0)
      return libraries.at(profile)->info.scratch_bytes;
    size_t bytes = 0;
    for (const auto &library : libraries)
      bytes = std::max(bytes, library->info.scratch_bytes);
    return bytes;
  }
  std::string input_name() const override { return "x"; }
  std::string output_name() const override { return "output"; }
  ElementType input_type() const override { return ElementType::Float32; }
  ElementType output_type() const override {
    return libraries[0]->info.output_element_bytes == 4 ? ElementType::Float32
                                                        : ElementType::Float16;
  }
  const char *backend_name() const override { return "compiled_cuda"; }
  std::unique_ptr<ModelContext> create_context() override {
    return std::make_unique<Context>(libraries);
  }
};
} // namespace
std::unique_ptr<ModelEngine> load_model_engine(const std::string &path) {
  return std::make_unique<Engine>(path);
}
} // namespace ppocrv6_native::engine

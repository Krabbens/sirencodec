#include "sirencodec/litert_runner.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace sirencodec {
namespace {

using LiteRtStatus = int;
constexpr LiteRtStatus kLiteRtStatusOk = 0;
constexpr int kLiteRtHwAcceleratorCpu = 1 << 0;
constexpr int kLiteRtTensorBufferLockModeRead = 0;
constexpr int kLiteRtTensorBufferLockModeWrite = 1;

enum LiteRtElementType : int {
  kLiteRtElementTypeFloat32 = 1,
  kLiteRtElementTypeInt32 = 2,
};

enum LiteRtTensorBufferType : int {
  kLiteRtTensorBufferTypeHostMemory = 1,
};

constexpr int kLiteRtTensorMaxRank = 8;

struct LiteRtLayout {
  unsigned int rank : 7;
  bool has_strides : 1;
  std::int32_t dimensions[kLiteRtTensorMaxRank];
  std::uint32_t strides[kLiteRtTensorMaxRank];
};

struct LiteRtRankedTensorType {
  LiteRtElementType element_type;
  LiteRtLayout layout;
};

using LiteRtEnvironment = struct LiteRtEnvironmentT *;
using LiteRtModel = struct LiteRtModelT *;
using LiteRtCompiledModel = struct LiteRtCompiledModelT *;
using LiteRtOptions = struct LiteRtOptionsT *;
using LiteRtOpaqueOptions = struct LiteRtOpaqueOptionsT *;
using LiteRtSubgraph = struct LiteRtSubgraphT *;
using LiteRtTensor = struct LiteRtTensorT *;
using LiteRtTensorBuffer = struct LiteRtTensorBufferT *;
using LiteRtTensorBufferRequirements = struct LiteRtTensorBufferRequirementsT *;
using LiteRtParamIndex = std::size_t;

static_assert(sizeof(LiteRtLayout) == 68, "LiteRtLayout ABI mismatch");
static_assert(sizeof(LiteRtRankedTensorType) == 72, "LiteRtRankedTensorType ABI mismatch");

template <typename T>
class ScopedHandle {
public:
  using Deleter = void (*)(T);

  ScopedHandle() = default;
  ScopedHandle(T handle, Deleter deleter) : handle_(handle), deleter_(deleter) {}
  ~ScopedHandle() { reset(); }

  ScopedHandle(const ScopedHandle &) = delete;
  ScopedHandle &operator=(const ScopedHandle &) = delete;

  ScopedHandle(ScopedHandle &&other) noexcept
      : handle_(std::exchange(other.handle_, nullptr)), deleter_(std::exchange(other.deleter_, nullptr)) {}

  ScopedHandle &operator=(ScopedHandle &&other) noexcept {
    if (this != &other) {
      reset();
      handle_ = std::exchange(other.handle_, nullptr);
      deleter_ = std::exchange(other.deleter_, nullptr);
    }
    return *this;
  }

  [[nodiscard]] T get() const noexcept { return handle_; }
  [[nodiscard]] explicit operator bool() const noexcept { return handle_ != nullptr; }

  [[nodiscard]] T release() noexcept {
    deleter_ = nullptr;
    return std::exchange(handle_, nullptr);
  }

  void reset(T handle = nullptr, Deleter deleter = nullptr) noexcept {
    if (handle_ != nullptr && deleter_ != nullptr) {
      deleter_(handle_);
    }
    handle_ = handle;
    deleter_ = deleter;
  }

private:
  T handle_{nullptr};
  Deleter deleter_{nullptr};
};

class SharedLibrary {
public:
  explicit SharedLibrary(const std::filesystem::path &path) {
    const auto requested = path.empty() ? default_litert_library_path() : path;
    path_ = requested;
    handle_ = dlopen(requested.string().c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
    if (handle_ == nullptr) {
      throw std::runtime_error("cannot load LiteRT library '" + requested.string() + "': " + dlerror());
    }
  }

  ~SharedLibrary() = default;

  SharedLibrary(const SharedLibrary &) = delete;
  SharedLibrary &operator=(const SharedLibrary &) = delete;

  template <typename Fn>
  Fn load(const char *name) const {
    dlerror();
    auto *symbol = dlsym(handle_, name);
    const char *err = dlerror();
    if (err != nullptr || symbol == nullptr) {
      throw std::runtime_error("LiteRT symbol not found: " + std::string(name));
    }
    return reinterpret_cast<Fn>(symbol);
  }

  template <typename Fn>
  Fn try_load(const char *name) const noexcept {
    dlerror();
    auto *symbol = dlsym(handle_, name);
    const char *err = dlerror();
    if (err != nullptr || symbol == nullptr) {
      return nullptr;
    }
    return reinterpret_cast<Fn>(symbol);
  }

  [[nodiscard]] const std::filesystem::path &path() const noexcept { return path_; }

private:
  void *handle_{nullptr};
  std::filesystem::path path_;
};

struct LiteRtApi {
  using GetStatusStringFn = const char *(*)(LiteRtStatus);
  using CreateEnvironmentFn = LiteRtStatus (*)(int, const void *, LiteRtEnvironment *);
  using DestroyEnvironmentFn = void (*)(LiteRtEnvironment);
  using CreateModelFromFileFn = LiteRtStatus (*)(const char *, LiteRtModel *);
  using DestroyModelFn = void (*)(LiteRtModel);
  using CreateOptionsFn = LiteRtStatus (*)(LiteRtOptions *);
  using DestroyOptionsFn = void (*)(LiteRtOptions);
  using SetOptionsHardwareAcceleratorsFn = LiteRtStatus (*)(LiteRtOptions, int);
  using CreateOpaqueOptionsFn =
      LiteRtStatus (*)(const char *, void *, void (*)(void *), LiteRtOpaqueOptions *);
  using DestroyOpaqueOptionsFn = void (*)(LiteRtOpaqueOptions);
  using AddOpaqueOptionsFn = LiteRtStatus (*)(LiteRtOptions, LiteRtOpaqueOptions);
  using CreateCompiledModelFn = LiteRtStatus (*)(LiteRtEnvironment, LiteRtModel, LiteRtOptions, LiteRtCompiledModel *);
  using DestroyCompiledModelFn = void (*)(LiteRtCompiledModel);
  using GetModelSubgraphFn = LiteRtStatus (*)(LiteRtModel, LiteRtParamIndex, LiteRtSubgraph *);
  using GetNumSubgraphInputsFn = LiteRtStatus (*)(LiteRtSubgraph, LiteRtParamIndex *);
  using GetSubgraphInputFn = LiteRtStatus (*)(LiteRtSubgraph, LiteRtParamIndex, LiteRtTensor *);
  using GetNumSubgraphOutputsFn = LiteRtStatus (*)(LiteRtSubgraph, LiteRtParamIndex *);
  using GetSubgraphOutputFn = LiteRtStatus (*)(LiteRtSubgraph, LiteRtParamIndex, LiteRtTensor *);
  using GetRankedTensorTypeFn = LiteRtStatus (*)(LiteRtTensor, LiteRtRankedTensorType *);
  using GetInputRequirementsFn =
      LiteRtStatus (*)(LiteRtCompiledModel, LiteRtParamIndex, LiteRtParamIndex, LiteRtTensorBufferRequirements *);
  using GetOutputRequirementsFn =
      LiteRtStatus (*)(LiteRtCompiledModel, LiteRtParamIndex, LiteRtParamIndex, LiteRtTensorBufferRequirements *);
  using CreateManagedTensorBufferFromRequirementsFn =
      LiteRtStatus (*)(LiteRtEnvironment, const LiteRtRankedTensorType *, LiteRtTensorBufferRequirements, LiteRtTensorBuffer *);
  using LockTensorBufferFn = LiteRtStatus (*)(LiteRtTensorBuffer, void **, int);
  using UnlockTensorBufferFn = LiteRtStatus (*)(LiteRtTensorBuffer);
  using DestroyTensorBufferFn = void (*)(LiteRtTensorBuffer);
  using RunCompiledModelFn =
      LiteRtStatus (*)(LiteRtCompiledModel, LiteRtParamIndex, std::size_t, LiteRtTensorBuffer *, std::size_t, LiteRtTensorBuffer *);

  explicit LiteRtApi(const SharedLibrary &lib)
      : get_status_string(lib.try_load<GetStatusStringFn>("LiteRtGetStatusString")),
        create_environment(lib.load<CreateEnvironmentFn>("LiteRtCreateEnvironment")),
        destroy_environment(lib.load<DestroyEnvironmentFn>("LiteRtDestroyEnvironment")),
        create_model_from_file(lib.load<CreateModelFromFileFn>("LiteRtCreateModelFromFile")),
        destroy_model(lib.load<DestroyModelFn>("LiteRtDestroyModel")),
        create_options(lib.load<CreateOptionsFn>("LiteRtCreateOptions")),
        destroy_options(lib.load<DestroyOptionsFn>("LiteRtDestroyOptions")),
        set_options_hardware_accelerators(
            lib.load<SetOptionsHardwareAcceleratorsFn>("LiteRtSetOptionsHardwareAccelerators")),
        create_opaque_options(lib.load<CreateOpaqueOptionsFn>("LiteRtCreateOpaqueOptions")),
        destroy_opaque_options(lib.load<DestroyOpaqueOptionsFn>("LiteRtDestroyOpaqueOptions")),
        add_opaque_options(lib.load<AddOpaqueOptionsFn>("LiteRtAddOpaqueOptions")),
        create_compiled_model(lib.load<CreateCompiledModelFn>("LiteRtCreateCompiledModel")),
        destroy_compiled_model(lib.load<DestroyCompiledModelFn>("LiteRtDestroyCompiledModel")),
        get_model_subgraph(lib.load<GetModelSubgraphFn>("LiteRtGetModelSubgraph")),
        get_num_subgraph_inputs(lib.load<GetNumSubgraphInputsFn>("LiteRtGetNumSubgraphInputs")),
        get_subgraph_input(lib.load<GetSubgraphInputFn>("LiteRtGetSubgraphInput")),
        get_num_subgraph_outputs(lib.load<GetNumSubgraphOutputsFn>("LiteRtGetNumSubgraphOutputs")),
        get_subgraph_output(lib.load<GetSubgraphOutputFn>("LiteRtGetSubgraphOutput")),
        get_ranked_tensor_type(lib.load<GetRankedTensorTypeFn>("LiteRtGetRankedTensorType")),
        get_input_requirements(lib.load<GetInputRequirementsFn>("LiteRtGetCompiledModelInputBufferRequirements")),
        get_output_requirements(lib.load<GetOutputRequirementsFn>("LiteRtGetCompiledModelOutputBufferRequirements")),
        create_managed_tensor_buffer_from_requirements(
            lib.load<CreateManagedTensorBufferFromRequirementsFn>("LiteRtCreateManagedTensorBufferFromRequirements")),
        lock_tensor_buffer(lib.load<LockTensorBufferFn>("LiteRtLockTensorBuffer")),
        unlock_tensor_buffer(lib.load<UnlockTensorBufferFn>("LiteRtUnlockTensorBuffer")),
        destroy_tensor_buffer(lib.load<DestroyTensorBufferFn>("LiteRtDestroyTensorBuffer")),
        run_compiled_model(lib.load<RunCompiledModelFn>("LiteRtRunCompiledModel")) {}

  GetStatusStringFn get_status_string{};
  CreateEnvironmentFn create_environment{};
  DestroyEnvironmentFn destroy_environment{};
  CreateModelFromFileFn create_model_from_file{};
  DestroyModelFn destroy_model{};
  CreateOptionsFn create_options{};
  DestroyOptionsFn destroy_options{};
  SetOptionsHardwareAcceleratorsFn set_options_hardware_accelerators{};
  CreateOpaqueOptionsFn create_opaque_options{};
  DestroyOpaqueOptionsFn destroy_opaque_options{};
  AddOpaqueOptionsFn add_opaque_options{};
  CreateCompiledModelFn create_compiled_model{};
  DestroyCompiledModelFn destroy_compiled_model{};
  GetModelSubgraphFn get_model_subgraph{};
  GetNumSubgraphInputsFn get_num_subgraph_inputs{};
  GetSubgraphInputFn get_subgraph_input{};
  GetNumSubgraphOutputsFn get_num_subgraph_outputs{};
  GetSubgraphOutputFn get_subgraph_output{};
  GetRankedTensorTypeFn get_ranked_tensor_type{};
  GetInputRequirementsFn get_input_requirements{};
  GetOutputRequirementsFn get_output_requirements{};
  CreateManagedTensorBufferFromRequirementsFn create_managed_tensor_buffer_from_requirements{};
  LockTensorBufferFn lock_tensor_buffer{};
  UnlockTensorBufferFn unlock_tensor_buffer{};
  DestroyTensorBufferFn destroy_tensor_buffer{};
  RunCompiledModelFn run_compiled_model{};
};

LiteRtDataType to_dtype(LiteRtElementType type) {
  switch (type) {
  case kLiteRtElementTypeFloat32:
    return LiteRtDataType::Float32;
  case kLiteRtElementTypeInt32:
    return LiteRtDataType::Int32;
  default:
    throw std::runtime_error("unsupported LiteRT tensor element type: " + std::to_string(static_cast<int>(type)));
  }
}

LiteRtElementType to_litert_dtype(LiteRtDataType type) {
  switch (type) {
  case LiteRtDataType::Float32:
    return kLiteRtElementTypeFloat32;
  case LiteRtDataType::Int32:
    return kLiteRtElementTypeInt32;
  }
  throw std::runtime_error("unknown LiteRT tensor dtype");
}

LiteRtTensorSpec to_spec(const LiteRtRankedTensorType &type) {
  LiteRtTensorSpec spec;
  spec.dtype = to_dtype(type.element_type);
  spec.shape.reserve(type.layout.rank);
  for (unsigned int i = 0; i < type.layout.rank; ++i) {
    spec.shape.push_back(type.layout.dimensions[i]);
  }
  return spec;
}

LiteRtRankedTensorType to_ranked_type(const LiteRtTensorSpec &spec) {
  if (spec.shape.size() > kLiteRtTensorMaxRank) {
    throw std::runtime_error("LiteRT tensor rank exceeds supported ABI rank");
  }
  LiteRtRankedTensorType type{};
  type.element_type = to_litert_dtype(spec.dtype);
  *reinterpret_cast<unsigned char *>(&type.layout) = static_cast<unsigned char>(spec.shape.size());
  type.layout.has_strides = false;
  for (std::size_t i = 0; i < spec.shape.size(); ++i) {
    type.layout.dimensions[i] = spec.shape[i];
  }
  return type;
}

void check_status(const LiteRtApi &api, LiteRtStatus status, const std::string &what) {
  if (status == kLiteRtStatusOk) {
    return;
  }
  const char *status_text = api.get_status_string != nullptr ? api.get_status_string(status) : nullptr;
  std::ostringstream out;
  out << what << " failed with LiteRT status " << status;
  if (status_text != nullptr) {
    out << " (" << status_text << ')';
  }
  throw std::runtime_error(out.str());
}

void free_opaque_payload(void *payload) {
  std::free(payload);
}

char *make_c_payload(const std::string &text) {
  auto *payload = static_cast<char *>(std::malloc(text.size() + 1));
  if (payload == nullptr) {
    throw std::bad_alloc();
  }
  std::memcpy(payload, text.c_str(), text.size() + 1);
  return payload;
}

void add_cpu_options(const LiteRtApi &api, LiteRtOptions options, int num_threads) {
  if (num_threads <= 0) {
    return;
  }

  std::ostringstream toml;
  toml << "kernel_mode = \"xnnpack\"\n";
  toml << "num_threads = " << num_threads << '\n';

  std::unique_ptr<char, decltype(&std::free)> payload(make_c_payload(toml.str()), &std::free);
  LiteRtOpaqueOptions opaque_options{};
  check_status(api,
               api.create_opaque_options("xnnpack", payload.get(), free_opaque_payload, &opaque_options),
               "LiteRtCreateOpaqueOptions(cpu)");
  payload.release();

  ScopedHandle<LiteRtOpaqueOptions> opaque_handle(opaque_options, api.destroy_opaque_options);
  check_status(api,
               api.add_opaque_options(options, opaque_handle.get()),
               "LiteRtAddOpaqueOptions(cpu)");
  (void)opaque_handle.release();
}

std::vector<LiteRtTensorSpec> query_specs(const LiteRtApi &api, LiteRtModel model, bool inputs) {
  LiteRtSubgraph subgraph{};
  check_status(api, api.get_model_subgraph(model, 0, &subgraph), "LiteRtGetModelSubgraph");
  LiteRtParamIndex count{};
  check_status(api,
               inputs ? api.get_num_subgraph_inputs(subgraph, &count)
                      : api.get_num_subgraph_outputs(subgraph, &count),
               inputs ? "LiteRtGetNumSubgraphInputs" : "LiteRtGetNumSubgraphOutputs");
  std::vector<LiteRtTensorSpec> specs;
  specs.reserve(count);
  for (LiteRtParamIndex i = 0; i < count; ++i) {
    LiteRtTensor tensor{};
    check_status(api,
                 inputs ? api.get_subgraph_input(subgraph, i, &tensor)
                        : api.get_subgraph_output(subgraph, i, &tensor),
                 inputs ? "LiteRtGetSubgraphInput" : "LiteRtGetSubgraphOutput");
    LiteRtRankedTensorType type{};
    check_status(api, api.get_ranked_tensor_type(tensor, &type), "LiteRtGetRankedTensorType");
    specs.push_back(to_spec(type));
  }
  return specs;
}

class LockedBuffer {
public:
  LockedBuffer(const LiteRtApi &api, LiteRtTensorBuffer buffer, int mode) : api_(api), buffer_(buffer) {
    check_status(api_, api_.lock_tensor_buffer(buffer_, &ptr_, mode), "LiteRtLockTensorBuffer");
  }

  ~LockedBuffer() {
    if (ptr_ != nullptr) {
      (void)api_.unlock_tensor_buffer(buffer_);
    }
  }

  LockedBuffer(const LockedBuffer &) = delete;
  LockedBuffer &operator=(const LockedBuffer &) = delete;

  [[nodiscard]] void *get() const noexcept { return ptr_; }

private:
  const LiteRtApi &api_;
  LiteRtTensorBuffer buffer_{};
  void *ptr_{nullptr};
};

} // namespace

std::size_t LiteRtTensorSpec::element_size() const {
  switch (dtype) {
  case LiteRtDataType::Float32:
    return sizeof(float);
  case LiteRtDataType::Int32:
    return sizeof(std::int32_t);
  }
  throw std::runtime_error("unknown LiteRT dtype");
}

std::size_t LiteRtTensorSpec::element_count() const {
  if (shape.empty()) {
    return 0;
  }
  std::size_t count = 1;
  for (std::int32_t dim : shape) {
    if (dim <= 0) {
      throw std::runtime_error("dynamic LiteRT shapes are not supported by this runner");
    }
    const auto value = static_cast<std::size_t>(dim);
    if (count > std::numeric_limits<std::size_t>::max() / value) {
      throw std::runtime_error("LiteRT tensor element count overflow");
    }
    count *= value;
  }
  return count;
}

std::size_t LiteRtTensorSpec::byte_size() const {
  return element_count() * element_size();
}

std::string LiteRtTensorSpec::dtype_name() const {
  switch (dtype) {
  case LiteRtDataType::Float32:
    return "float32";
  case LiteRtDataType::Int32:
    return "int32";
  }
  return "unknown";
}

std::string LiteRtTensorSpec::shape_string() const {
  std::ostringstream out;
  out << '[';
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    out << shape[i];
  }
  out << ']';
  return out.str();
}

LiteRtTensorValue LiteRtTensorValue::from_floats(const LiteRtTensorSpec &spec, const std::vector<float> &values) {
  if (spec.dtype != LiteRtDataType::Float32 || values.size() != spec.element_count()) {
    throw std::runtime_error("float LiteRT input does not match tensor spec");
  }
  LiteRtTensorValue out;
  out.spec = spec;
  out.bytes.resize(values.size() * sizeof(float));
  std::memcpy(out.bytes.data(), values.data(), out.bytes.size());
  return out;
}

LiteRtTensorValue LiteRtTensorValue::from_int32(const LiteRtTensorSpec &spec, const std::vector<std::int32_t> &values) {
  if (spec.dtype != LiteRtDataType::Int32 || values.size() != spec.element_count()) {
    throw std::runtime_error("int32 LiteRT input does not match tensor spec");
  }
  LiteRtTensorValue out;
  out.spec = spec;
  out.bytes.resize(values.size() * sizeof(std::int32_t));
  std::memcpy(out.bytes.data(), values.data(), out.bytes.size());
  return out;
}

std::vector<float> LiteRtTensorValue::as_floats() const {
  if (spec.dtype != LiteRtDataType::Float32 || bytes.size() != spec.byte_size()) {
    throw std::runtime_error("LiteRT tensor is not float32");
  }
  std::vector<float> values(spec.element_count());
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

std::vector<std::int32_t> LiteRtTensorValue::as_int32() const {
  if (spec.dtype != LiteRtDataType::Int32 || bytes.size() != spec.byte_size()) {
    throw std::runtime_error("LiteRT tensor is not int32");
  }
  std::vector<std::int32_t> values(spec.element_count());
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

std::filesystem::path default_litert_library_path() {
  if (const char *env = std::getenv("SIRENCODEC_LITERT_LIB"); env != nullptr && env[0] != '\0') {
    return env;
  }
  const std::filesystem::path wheel_path =
      ".venv-tflite/lib/python3.12/site-packages/ai_edge_litert/libLiteRt.so";
  if (std::filesystem::exists(wheel_path)) {
    return wheel_path;
  }
  return "libLiteRt.so";
}

class LiteRtRunner::Impl {
public:
  Impl(const std::filesystem::path &model_path, Options options)
      : lib_(options.library_path),
        api_(lib_),
        model_path_(model_path) {
    if (!std::filesystem::exists(model_path_)) {
      throw std::runtime_error("LiteRT model does not exist: " + model_path_.string());
    }

    LiteRtEnvironment env{};
    check_status(api_, api_.create_environment(0, nullptr, &env), "LiteRtCreateEnvironment");
    environment_ = ScopedHandle<LiteRtEnvironment>(env, api_.destroy_environment);

    LiteRtModel model{};
    check_status(api_, api_.create_model_from_file(model_path_.string().c_str(), &model), "LiteRtCreateModelFromFile");
    model_ = ScopedHandle<LiteRtModel>(model, api_.destroy_model);

    LiteRtOptions compile_options{};
    check_status(api_, api_.create_options(&compile_options), "LiteRtCreateOptions");
    ScopedHandle<LiteRtOptions> options_handle(compile_options, api_.destroy_options);
    check_status(api_,
                 api_.set_options_hardware_accelerators(options_handle.get(), kLiteRtHwAcceleratorCpu),
                 "LiteRtSetOptionsHardwareAccelerators");
    add_cpu_options(api_, options_handle.get(), options.num_threads);

    LiteRtCompiledModel compiled{};
    check_status(api_,
                 api_.create_compiled_model(environment_.get(), model_.get(), options_handle.get(), &compiled),
                 "LiteRtCreateCompiledModel");
    compiled_ = ScopedHandle<LiteRtCompiledModel>(compiled, api_.destroy_compiled_model);

    input_specs_ = query_specs(api_, model_.get(), true);
    output_specs_ = query_specs(api_, model_.get(), false);
    create_buffers(true);
    create_buffers(false);
  }

  [[nodiscard]] std::vector<LiteRtTensorValue> run(const std::vector<LiteRtTensorValue> &inputs) const {
    if (inputs.size() != input_specs_.size()) {
      throw std::runtime_error("wrong number of LiteRT inputs");
    }

    for (std::size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i].spec.dtype != input_specs_[i].dtype || inputs[i].spec.shape != input_specs_[i].shape ||
          inputs[i].bytes.size() != input_specs_[i].byte_size()) {
        throw std::runtime_error("LiteRT input " + std::to_string(i) + " does not match model spec");
      }
      {
        LockedBuffer lock(api_, input_buffers_[i], kLiteRtTensorBufferLockModeWrite);
        std::memcpy(lock.get(), inputs[i].bytes.data(), inputs[i].bytes.size());
      }
    }

    check_status(api_,
                 api_.run_compiled_model(compiled_.get(), 0, input_buffers_.size(), input_buffers_.data(),
                                         output_buffers_.size(), output_buffers_.data()),
                 "LiteRtRunCompiledModel");

    std::vector<LiteRtTensorValue> outputs;
    outputs.reserve(output_specs_.size());
    for (std::size_t i = 0; i < output_specs_.size(); ++i) {
      LiteRtTensorValue value;
      value.spec = output_specs_[i];
      value.bytes.resize(output_specs_[i].byte_size());
      LockedBuffer lock(api_, output_buffers_[i], kLiteRtTensorBufferLockModeRead);
      std::memcpy(value.bytes.data(), lock.get(), value.bytes.size());
      outputs.push_back(std::move(value));
    }
    return outputs;
  }

  void create_buffers(bool inputs) {
    const auto &specs = inputs ? input_specs_ : output_specs_;
    auto &handles = inputs ? input_handles_ : output_handles_;
    auto &buffers = inputs ? input_buffers_ : output_buffers_;
    handles.clear();
    buffers.clear();
    handles.reserve(specs.size());
    buffers.reserve(specs.size());
    for (std::size_t i = 0; i < specs.size(); ++i) {
      LiteRtTensorBufferRequirements requirements{};
      check_status(api_,
                   inputs ? api_.get_input_requirements(compiled_.get(), 0, i, &requirements)
                          : api_.get_output_requirements(compiled_.get(), 0, i, &requirements),
                   inputs ? "LiteRtGetCompiledModelInputBufferRequirements"
                          : "LiteRtGetCompiledModelOutputBufferRequirements");
      auto type = to_ranked_type(specs[i]);
      LiteRtTensorBuffer buffer{};
      check_status(api_,
                   api_.create_managed_tensor_buffer_from_requirements(
                       environment_.get(), &type, requirements, &buffer),
                   inputs ? "LiteRtCreateManagedTensorBufferFromRequirements(input)"
                          : "LiteRtCreateManagedTensorBufferFromRequirements(output)");
      handles.emplace_back(buffer, api_.destroy_tensor_buffer);
      buffers.push_back(buffer);
    }
  }

  SharedLibrary lib_;
  LiteRtApi api_;
  std::filesystem::path model_path_;
  ScopedHandle<LiteRtEnvironment> environment_;
  ScopedHandle<LiteRtModel> model_;
  ScopedHandle<LiteRtCompiledModel> compiled_;
  std::vector<ScopedHandle<LiteRtTensorBuffer>> input_handles_;
  std::vector<ScopedHandle<LiteRtTensorBuffer>> output_handles_;
  mutable std::vector<LiteRtTensorBuffer> input_buffers_;
  mutable std::vector<LiteRtTensorBuffer> output_buffers_;
  std::vector<LiteRtTensorSpec> input_specs_;
  std::vector<LiteRtTensorSpec> output_specs_;
};

LiteRtRunner::LiteRtRunner(const std::filesystem::path &model_path)
    : LiteRtRunner(model_path, Options{}) {}

LiteRtRunner::LiteRtRunner(const std::filesystem::path &model_path, Options options)
    : impl_(std::make_unique<Impl>(model_path, std::move(options))) {}

LiteRtRunner::~LiteRtRunner() = default;
LiteRtRunner::LiteRtRunner(LiteRtRunner &&) noexcept = default;
LiteRtRunner &LiteRtRunner::operator=(LiteRtRunner &&) noexcept = default;

const std::vector<LiteRtTensorSpec> &LiteRtRunner::input_specs() const {
  return impl_->input_specs_;
}

const std::vector<LiteRtTensorSpec> &LiteRtRunner::output_specs() const {
  return impl_->output_specs_;
}

const std::filesystem::path &LiteRtRunner::model_path() const {
  return impl_->model_path_;
}

std::vector<LiteRtTensorValue> LiteRtRunner::run(const std::vector<LiteRtTensorValue> &inputs) const {
  return impl_->run(inputs);
}

} // namespace sirencodec

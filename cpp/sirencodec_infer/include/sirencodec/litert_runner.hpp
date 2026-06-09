#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace sirencodec {

enum class LiteRtDataType {
  Float32,
  Int32,
};

struct LiteRtTensorSpec {
  LiteRtDataType dtype{LiteRtDataType::Float32};
  std::vector<std::int32_t> shape;

  [[nodiscard]] std::size_t element_size() const;
  [[nodiscard]] std::size_t element_count() const;
  [[nodiscard]] std::size_t byte_size() const;
  [[nodiscard]] std::string dtype_name() const;
  [[nodiscard]] std::string shape_string() const;
};

struct LiteRtTensorValue {
  LiteRtTensorSpec spec;
  std::vector<std::byte> bytes;

  static LiteRtTensorValue from_floats(const LiteRtTensorSpec &spec, const std::vector<float> &values);
  static LiteRtTensorValue from_int32(const LiteRtTensorSpec &spec, const std::vector<std::int32_t> &values);

  [[nodiscard]] std::vector<float> as_floats() const;
  [[nodiscard]] std::vector<std::int32_t> as_int32() const;
};

class LiteRtRunner {
public:
  struct Options {
    Options() = default;
    explicit Options(std::filesystem::path library_path_, int num_threads_ = 0)
        : library_path(std::move(library_path_)), num_threads(num_threads_) {}

    std::filesystem::path library_path;
    int num_threads{0};
  };

  explicit LiteRtRunner(const std::filesystem::path &model_path);
  LiteRtRunner(const std::filesystem::path &model_path, Options options);
  ~LiteRtRunner();

  LiteRtRunner(const LiteRtRunner &) = delete;
  LiteRtRunner &operator=(const LiteRtRunner &) = delete;
  LiteRtRunner(LiteRtRunner &&) noexcept;
  LiteRtRunner &operator=(LiteRtRunner &&) noexcept;

  [[nodiscard]] const std::vector<LiteRtTensorSpec> &input_specs() const;
  [[nodiscard]] const std::vector<LiteRtTensorSpec> &output_specs() const;
  [[nodiscard]] const std::filesystem::path &model_path() const;

  [[nodiscard]] std::vector<LiteRtTensorValue> run(std::span<const LiteRtTensorValue> inputs) const;
  void run_into(std::span<const LiteRtTensorValue> inputs,
                std::vector<LiteRtTensorValue> &outputs) const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

std::filesystem::path default_litert_library_path();

} // namespace sirencodec

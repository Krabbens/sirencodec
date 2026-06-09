#include "sirencodec/audio_wav.hpp"
#include "sirencodec/litert_runner.hpp"
#include "sirencodec/metrics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
  std::filesystem::path input;
  std::filesystem::path full_model;
  std::filesystem::path compress_model;
  std::filesystem::path decompress_model;
  std::filesystem::path litert_lib;
  int sample_rate{16000};
  int num_threads{0};
  double tolerance{1.0e-3};
};

void print_usage(const char *argv0) {
  std::cerr
      << "usage: " << argv0 << " --input audio.wav [--full-model file] "
      << "[--compress-model file --decompress-model file] [options]\n\n"
      << "options:\n"
      << "  --litert-lib FILE    libLiteRt path (default: env/path probe)\n"
      << "  --sample-rate N      Audio sample rate for exported shape (default: 16000)\n"
      << "  --num-threads N      XNNPACK CPU threads via LiteRT opaque options\n"
      << "  --tolerance X        Max full-vs-codec absolute diff when both are provided (default: 1e-3)\n";
}

Args parse_args(int argc, char **argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    const auto require_value = [&](const std::string &option) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + option);
      }
      return argv[++i];
    };
    if (key == "--help" || key == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    }
    if (key == "--input" || key == "-i") {
      args.input = require_value(key);
    } else if (key == "--full-model") {
      args.full_model = require_value(key);
    } else if (key == "--compress-model") {
      args.compress_model = require_value(key);
    } else if (key == "--decompress-model") {
      args.decompress_model = require_value(key);
    } else if (key == "--litert-lib") {
      args.litert_lib = require_value(key);
    } else if (key == "--sample-rate") {
      args.sample_rate = std::stoi(require_value(key));
    } else if (key == "--num-threads") {
      args.num_threads = std::stoi(require_value(key));
    } else if (key == "--tolerance") {
      args.tolerance = std::stod(require_value(key));
    } else {
      throw std::runtime_error("unknown argument: " + key);
    }
  }

  if (args.input.empty()) {
    throw std::runtime_error("--input is required");
  }
  if (args.full_model.empty() && (args.compress_model.empty() || args.decompress_model.empty())) {
    throw std::runtime_error("provide --full-model or --compress-model plus --decompress-model");
  }
  if ((!args.compress_model.empty() || !args.decompress_model.empty()) &&
      (args.compress_model.empty() || args.decompress_model.empty())) {
    throw std::runtime_error("--compress-model and --decompress-model must be provided together");
  }
  if (args.sample_rate <= 0) {
    throw std::runtime_error("--sample-rate must be positive");
  }
  if (args.num_threads < 0) {
    throw std::runtime_error("--num-threads must be non-negative");
  }
  if (!(args.tolerance > 0.0)) {
    throw std::runtime_error("--tolerance must be positive");
  }
  return args;
}

std::vector<float> load_audio_for_spec(const Args &args, const sirencodec::LiteRtTensorSpec &input_spec) {
  if (input_spec.dtype != sirencodec::LiteRtDataType::Float32 || input_spec.shape.size() != 3 ||
      input_spec.shape[0] != 1 || input_spec.shape[2] != 1 || input_spec.shape[1] <= 0) {
    throw std::runtime_error("expected input model shape [1,N,1] float32");
  }

  auto audio = sirencodec::read_audio_mono(args.input);
  auto samples = sirencodec::resample_linear(audio.samples, audio.sample_rate, args.sample_rate);
  samples = sirencodec::normalize_peak(samples);
  const auto target = static_cast<std::size_t>(input_spec.shape[1]);
  if (samples.size() > target) {
    samples.resize(target);
  } else if (samples.size() < target) {
    samples.resize(target, 0.0F);
  }
  return samples;
}

void require_finite(const std::vector<float> &values, const std::string &name) {
  if (values.empty()) {
    throw std::runtime_error(name + " is empty");
  }
  const auto invalid = std::find_if(values.begin(), values.end(), [](float value) {
    return !std::isfinite(value);
  });
  if (invalid != values.end()) {
    throw std::runtime_error(name + " contains non-finite values");
  }
}

double max_abs_diff(const std::vector<float> &lhs, const std::vector<float> &rhs) {
  if (lhs.size() != rhs.size()) {
    throw std::runtime_error("cannot compare vectors with different sizes");
  }
  double max_diff = 0.0;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    max_diff = std::max(max_diff, static_cast<double>(std::abs(lhs[i] - rhs[i])));
  }
  return max_diff;
}

void print_specs(const std::string &label, const std::vector<sirencodec::LiteRtTensorSpec> &specs) {
  std::cout << label << ':';
  for (std::size_t i = 0; i < specs.size(); ++i) {
    std::cout << " #" << i << '=' << specs[i].dtype_name() << specs[i].shape_string();
  }
  std::cout << '\n';
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto args = parse_args(argc, argv);
    const sirencodec::LiteRtRunner::Options options{args.litert_lib, args.num_threads};

    std::vector<float> reference_audio;
    std::vector<float> full_recon;
    std::vector<float> codec_recon;

    if (!args.full_model.empty()) {
      sirencodec::LiteRtRunner full(args.full_model, options);
      print_specs("full inputs", full.input_specs());
      print_specs("full outputs", full.output_specs());
      reference_audio = load_audio_for_spec(args, full.input_specs().at(0));
      const std::array<sirencodec::LiteRtTensorValue, 1> input{
          sirencodec::LiteRtTensorValue::from_floats(full.input_specs().at(0), reference_audio)};
      full_recon = full.run(input).at(0).as_floats();
      full_recon.resize(reference_audio.size());
      require_finite(full_recon, "full reconstruction");
    }

    if (!args.compress_model.empty()) {
      sirencodec::LiteRtRunner compress(args.compress_model, options);
      sirencodec::LiteRtRunner decompress(args.decompress_model, options);
      print_specs("compress inputs", compress.input_specs());
      print_specs("compress outputs", compress.output_specs());
      print_specs("decompress inputs", decompress.input_specs());
      print_specs("decompress outputs", decompress.output_specs());

      auto audio = load_audio_for_spec(args, compress.input_specs().at(0));
      const std::array<sirencodec::LiteRtTensorValue, 1> input{
          sirencodec::LiteRtTensorValue::from_floats(compress.input_specs().at(0), audio)};
      auto packet = compress.run(input);
      if (packet.size() != 2 || packet[0].spec.dtype != sirencodec::LiteRtDataType::Int32 ||
          packet[1].spec.dtype != sirencodec::LiteRtDataType::Float32) {
        throw std::runtime_error("compress model did not return int32 indices and float32 norms");
      }
      codec_recon = decompress.run(packet).at(0).as_floats();
      codec_recon.resize(audio.size());
      require_finite(codec_recon, "codec reconstruction");
      if (reference_audio.empty()) {
        reference_audio = std::move(audio);
      }
    }

    double full_codec_max_abs_diff = 0.0;
    if (!full_recon.empty() && !codec_recon.empty()) {
      full_codec_max_abs_diff = max_abs_diff(full_recon, codec_recon);
      if (full_codec_max_abs_diff > args.tolerance) {
        throw std::runtime_error("full-vs-codec max abs diff " + std::to_string(full_codec_max_abs_diff) +
                                 " exceeds tolerance " + std::to_string(args.tolerance));
      }
    }

    const auto &recon = !codec_recon.empty() ? codec_recon : full_recon;
    const auto metrics = sirencodec::compute_quality_metrics_16k(reference_audio, recon);
    std::cout.setf(std::ios::fixed);
    std::cout.precision(9);
    std::cout << "ok: backend=litert"
              << " samples=" << reference_audio.size()
              << " num_threads=" << args.num_threads
              << " full_codec_max_abs_diff=" << full_codec_max_abs_diff
              << " si_sdr_db=" << metrics.si_sdr_db
              << " lsd_db=" << metrics.lsd_db
              << " l1=" << metrics.l1
              << " cos=" << metrics.cosine << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}

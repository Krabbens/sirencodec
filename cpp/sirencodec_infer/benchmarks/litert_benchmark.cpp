#include "sirencodec/audio_wav.hpp"
#include "sirencodec/litert_runner.hpp"
#include "sirencodec/metrics.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
  std::string profile{"codec"};
  std::filesystem::path input;
  std::filesystem::path full_model;
  std::filesystem::path compress_model;
  std::filesystem::path decompress_model;
  std::filesystem::path output_dir{"artifacts/benchmarks/litert_single_file"};
  std::filesystem::path litert_lib;
  int sample_rate{16000};
  int preload_runs{1};
  int benchmark_runs{5};
  int num_threads{0};
};

struct TimingSummary {
  std::size_t count{};
  double mean_seconds{};
  double min_seconds{};
  double p50_seconds{};
  double p90_seconds{};
  double max_seconds{};
  double x_realtime{};
};

void print_usage(const char *argv0) {
  std::cerr
      << "usage: " << argv0 << " --input audio.wav --profile full|codec|all [options]\n\n"
      << "options:\n"
      << "  --full-model FILE        Full waveform->waveform .tflite model\n"
      << "  --compress-model FILE    waveform->indices,norms .tflite model\n"
      << "  --decompress-model FILE  indices,norms->waveform .tflite model\n"
      << "  --output-dir DIR         Output directory (default: artifacts/benchmarks/litert_single_file)\n"
      << "  --litert-lib FILE        libLiteRt path (default: env/path probe)\n"
      << "  --sample-rate N          Audio sample rate for exported shape (default: 16000)\n"
      << "  --num-threads N          XNNPACK CPU threads via LiteRT opaque options\n"
      << "  --preload-runs N         Untimed warmups (default: 1)\n"
      << "  --benchmark-runs N       Timed runs (default: 5)\n";
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
    if (key == "--profile") {
      args.profile = require_value(key);
    } else if (key == "--input" || key == "-i") {
      args.input = require_value(key);
    } else if (key == "--full-model") {
      args.full_model = require_value(key);
    } else if (key == "--compress-model") {
      args.compress_model = require_value(key);
    } else if (key == "--decompress-model") {
      args.decompress_model = require_value(key);
    } else if (key == "--output-dir" || key == "-o") {
      args.output_dir = require_value(key);
    } else if (key == "--litert-lib") {
      args.litert_lib = require_value(key);
    } else if (key == "--sample-rate") {
      args.sample_rate = std::stoi(require_value(key));
    } else if (key == "--num-threads") {
      args.num_threads = std::stoi(require_value(key));
    } else if (key == "--preload-runs") {
      args.preload_runs = std::stoi(require_value(key));
    } else if (key == "--benchmark-runs") {
      args.benchmark_runs = std::stoi(require_value(key));
    } else {
      throw std::runtime_error("unknown argument: " + key);
    }
  }

  if (args.input.empty()) {
    throw std::runtime_error("--input is required");
  }
  if (args.profile != "full" && args.profile != "codec" && args.profile != "all") {
    throw std::runtime_error("--profile must be full, codec or all");
  }
  if ((args.profile == "full" || args.profile == "all") && args.full_model.empty()) {
    throw std::runtime_error("--full-model is required for --profile full/all");
  }
  if ((args.profile == "codec" || args.profile == "all") &&
      (args.compress_model.empty() || args.decompress_model.empty())) {
    throw std::runtime_error("--compress-model and --decompress-model are required for --profile codec/all");
  }
  if (args.sample_rate <= 0) {
    throw std::runtime_error("--sample-rate must be positive");
  }
  if (args.num_threads < 0) {
    throw std::runtime_error("--num-threads must be non-negative");
  }
  args.preload_runs = std::max(args.preload_runs, 0);
  args.benchmark_runs = std::max(args.benchmark_runs, 1);
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

void print_specs(const std::string &label, const std::vector<sirencodec::LiteRtTensorSpec> &specs) {
  std::cout << label << ':';
  for (std::size_t i = 0; i < specs.size(); ++i) {
    std::cout << " #" << i << '=' << specs[i].dtype_name() << specs[i].shape_string();
  }
  std::cout << '\n';
}

double run_timed_into(const sirencodec::LiteRtRunner &runner,
                      std::span<const sirencodec::LiteRtTensorValue> inputs,
                      std::vector<sirencodec::LiteRtTensorValue> &outputs) {
  const auto start = std::chrono::steady_clock::now();
  runner.run_into(inputs, outputs);
  const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return elapsed;
}

TimingSummary summarize(std::vector<double> seconds, double audio_seconds) {
  if (seconds.empty()) {
    throw std::runtime_error("empty benchmark result");
  }
  std::sort(seconds.begin(), seconds.end());
  const auto percentile = [&](double p) {
    const double pos = p * static_cast<double>(seconds.size() - 1);
    return seconds[static_cast<std::size_t>(std::lround(pos))];
  };
  const double mean =
      std::accumulate(seconds.begin(), seconds.end(), 0.0) / static_cast<double>(seconds.size());
  return TimingSummary{
      seconds.size(),
      mean,
      seconds.front(),
      percentile(0.50),
      percentile(0.90),
      seconds.back(),
      audio_seconds / mean,
  };
}

std::string json_escape(const std::filesystem::path &path) {
  std::ostringstream out;
  for (char ch : path.string()) {
    if (ch == '\\' || ch == '"') {
      out << '\\';
    }
    out << ch;
  }
  return out.str();
}

void print_summary(const std::string &name, const TimingSummary &summary) {
  std::cout << name
            << ": mean=" << summary.mean_seconds
            << "s p50=" << summary.p50_seconds
            << "s min=" << summary.min_seconds
            << "s max=" << summary.max_seconds
            << "s xrt=" << summary.x_realtime
            << '\n';
}

void write_summary_json(std::ostream &out, const TimingSummary &summary, bool trailing_comma) {
  out << "      \"count\": " << summary.count << ",\n";
  out << "      \"mean_seconds\": " << summary.mean_seconds << ",\n";
  out << "      \"min_seconds\": " << summary.min_seconds << ",\n";
  out << "      \"p50_seconds\": " << summary.p50_seconds << ",\n";
  out << "      \"p90_seconds\": " << summary.p90_seconds << ",\n";
  out << "      \"max_seconds\": " << summary.max_seconds << ",\n";
  out << "      \"x_realtime\": " << summary.x_realtime << '\n';
  out << "    }" << (trailing_comma ? "," : "") << '\n';
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto args = parse_args(argc, argv);
    const sirencodec::LiteRtRunner::Options options{args.litert_lib, args.num_threads};

    std::cout.setf(std::ios::fixed);
    std::cout.precision(6);
    std::cout << "litert_lib: " << (args.litert_lib.empty() ? sirencodec::default_litert_library_path() : args.litert_lib)
              << '\n';
    std::cout << "num_threads: " << (args.num_threads > 0 ? std::to_string(args.num_threads) : std::string("runtime-default"))
              << '\n';
    std::cout << "profile: " << args.profile << '\n';

    std::map<std::string, TimingSummary> reports;
    std::vector<float> quality_original;
    std::vector<float> quality_recon;
    double audio_seconds{0.0};

    if (args.profile == "full" || args.profile == "all") {
      sirencodec::LiteRtRunner full(args.full_model, options);
      print_specs("full inputs", full.input_specs());
      print_specs("full outputs", full.output_specs());

      const auto original = load_audio_for_spec(args, full.input_specs().at(0));
      audio_seconds = static_cast<double>(original.size()) / static_cast<double>(args.sample_rate);
      const std::array<sirencodec::LiteRtTensorValue, 1> inputs{
          sirencodec::LiteRtTensorValue::from_floats(full.input_specs().at(0), original)};
      std::vector<sirencodec::LiteRtTensorValue> outputs;

      for (int i = 0; i < args.preload_runs; ++i) {
        full.run_into(inputs, outputs);
      }

      std::vector<double> seconds;
      seconds.reserve(static_cast<std::size_t>(args.benchmark_runs));
      for (int i = 0; i < args.benchmark_runs; ++i) {
        const double elapsed = run_timed_into(full, inputs, outputs);
        if (i + 1 == args.benchmark_runs) {
          quality_original = original;
          quality_recon = outputs.at(0).as_floats();
        }
        seconds.push_back(elapsed);
      }
      reports.emplace("full", summarize(std::move(seconds), audio_seconds));
    }

    if (args.profile == "codec" || args.profile == "all") {
      sirencodec::LiteRtRunner compress(args.compress_model, options);
      sirencodec::LiteRtRunner decompress(args.decompress_model, options);
      print_specs("compress inputs", compress.input_specs());
      print_specs("compress outputs", compress.output_specs());
      print_specs("decompress inputs", decompress.input_specs());
      print_specs("decompress outputs", decompress.output_specs());

      const auto original = load_audio_for_spec(args, compress.input_specs().at(0));
      audio_seconds = static_cast<double>(original.size()) / static_cast<double>(args.sample_rate);
      const std::array<sirencodec::LiteRtTensorValue, 1> compress_inputs{
          sirencodec::LiteRtTensorValue::from_floats(compress.input_specs().at(0), original)};
      std::vector<sirencodec::LiteRtTensorValue> packet;
      std::vector<sirencodec::LiteRtTensorValue> decoded;

      for (int i = 0; i < args.preload_runs; ++i) {
        compress.run_into(compress_inputs, packet);
        decompress.run_into(packet, decoded);
      }

      std::vector<double> compress_seconds;
      std::vector<double> decompress_seconds;
      std::vector<double> full_seconds;
      compress_seconds.reserve(static_cast<std::size_t>(args.benchmark_runs));
      decompress_seconds.reserve(static_cast<std::size_t>(args.benchmark_runs));
      full_seconds.reserve(static_cast<std::size_t>(args.benchmark_runs));

      for (int i = 0; i < args.benchmark_runs; ++i) {
        const double compress_elapsed = run_timed_into(compress, compress_inputs, packet);
        const double decompress_elapsed = run_timed_into(decompress, packet, decoded);
        if (i + 1 == args.benchmark_runs) {
          quality_original = original;
          quality_recon = decoded.at(0).as_floats();
        }
        compress_seconds.push_back(compress_elapsed);
        decompress_seconds.push_back(decompress_elapsed);
        full_seconds.push_back(compress_elapsed + decompress_elapsed);
      }

      reports.emplace("compress_only", summarize(std::move(compress_seconds), audio_seconds));
      reports.emplace("decompress_only", summarize(std::move(decompress_seconds), audio_seconds));
      reports.emplace("codec_full", summarize(std::move(full_seconds), audio_seconds));
    }

    quality_recon.resize(quality_original.size());
    const auto metrics = sirencodec::compute_quality_metrics_16k(quality_original, quality_recon);

    std::filesystem::create_directories(args.output_dir);
    const auto report_path = args.output_dir / "benchmark_summary.json";
    std::ofstream report(report_path);
    report.setf(std::ios::fixed);
    report.precision(9);
    report << "{\n";
    report << "  \"backend\": \"litert-cpp\",\n";
    report << "  \"profile\": \"" << args.profile << "\",\n";
    report << "  \"input\": \"" << json_escape(args.input) << "\",\n";
    report << "  \"audio_seconds\": " << audio_seconds << ",\n";
    report << "  \"num_threads\": " << args.num_threads << ",\n";
    report << "  \"preload_runs\": " << args.preload_runs << ",\n";
    report << "  \"benchmark_runs\": " << args.benchmark_runs << ",\n";
    report << "  \"benchmarks\": {\n";
    std::size_t index = 0;
    for (const auto &[name, summary] : reports) {
      report << "    \"" << name << "\": {\n";
      write_summary_json(report, summary, index + 1 != reports.size());
      ++index;
    }
    report << "  },\n";
    report << "  \"metrics\": {\n";
    report << "    \"si_sdr_db\": " << metrics.si_sdr_db << ",\n";
    report << "    \"lsd_db\": " << metrics.lsd_db << ",\n";
    report << "    \"l1\": " << metrics.l1 << ",\n";
    report << "    \"cos\": " << metrics.cosine << '\n';
    report << "  }\n";
    report << "}\n";

    std::cout << "input: " << std::filesystem::absolute(args.input) << " (" << audio_seconds << "s)\n";
    std::cout << "preload_runs: " << args.preload_runs << '\n';
    std::cout << "benchmark_runs: " << args.benchmark_runs << '\n';
    for (const auto &[name, summary] : reports) {
      print_summary(name, summary);
    }
    std::cout << "metrics: si_sdr_db=" << metrics.si_sdr_db
              << " lsd_db=" << metrics.lsd_db
              << " l1=" << metrics.l1
              << " cos=" << metrics.cosine << '\n';
    std::cout << "wrote: " << report_path << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}

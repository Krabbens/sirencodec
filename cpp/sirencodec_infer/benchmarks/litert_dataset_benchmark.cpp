#include "sirencodec/audio_wav.hpp"
#include "sirencodec/litert_runner.hpp"
#include "sirencodec/metrics.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

namespace {

struct Args {
  std::filesystem::path manifest;
  std::filesystem::path full_model;
  std::filesystem::path compress_model;
  std::filesystem::path decompress_model;
  std::filesystem::path output_dir{"artifacts/benchmarks/litert_dataset"};
  std::filesystem::path litert_lib;
  std::string profile{"codec"};
  int sample_rate{16000};
  int num_threads{0};
  int warmup_files{16};
  int quality_files{32};
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

struct TimedOutput {
  std::vector<sirencodec::LiteRtTensorValue> outputs;
  double seconds{};
};

struct MemorySnapshot {
  std::string label;
  double rss_mib{};
  double rss_peak_mib{};
};

void usage(const char *argv0) {
  std::cerr
      << "usage: " << argv0 << " --manifest files.txt --profile full|codec|all [options]\n\n"
      << "options:\n"
      << "  --full-model FILE\n"
      << "  --compress-model FILE\n"
      << "  --decompress-model FILE\n"
      << "  --litert-lib FILE\n"
      << "  --num-threads N\n"
      << "  --warmup-files N\n"
      << "  --quality-files N\n"
      << "  --output-dir DIR\n";
}

Args parse_args(int argc, char **argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    const auto value = [&](const std::string &option) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + option);
      }
      return argv[++i];
    };
    if (key == "-h" || key == "--help") {
      usage(argv[0]);
      std::exit(0);
    } else if (key == "--manifest") {
      args.manifest = value(key);
    } else if (key == "--full-model") {
      args.full_model = value(key);
    } else if (key == "--compress-model") {
      args.compress_model = value(key);
    } else if (key == "--decompress-model") {
      args.decompress_model = value(key);
    } else if (key == "--profile") {
      args.profile = value(key);
    } else if (key == "--output-dir") {
      args.output_dir = value(key);
    } else if (key == "--litert-lib") {
      args.litert_lib = value(key);
    } else if (key == "--sample-rate") {
      args.sample_rate = std::stoi(value(key));
    } else if (key == "--num-threads") {
      args.num_threads = std::stoi(value(key));
    } else if (key == "--warmup-files") {
      args.warmup_files = std::stoi(value(key));
    } else if (key == "--quality-files") {
      args.quality_files = std::stoi(value(key));
    } else {
      throw std::runtime_error("unknown argument: " + key);
    }
  }
  if (args.manifest.empty()) {
    throw std::runtime_error("--manifest is required");
  }
  if (args.profile != "full" && args.profile != "codec" && args.profile != "all") {
    throw std::runtime_error("--profile must be full, codec, or all");
  }
  if ((args.profile == "full" || args.profile == "all") && args.full_model.empty()) {
    throw std::runtime_error("--full-model is required for profile full/all");
  }
  if ((args.profile == "codec" || args.profile == "all") &&
      (args.compress_model.empty() || args.decompress_model.empty())) {
    throw std::runtime_error("--compress-model and --decompress-model are required for profile codec/all");
  }
  args.warmup_files = std::max(args.warmup_files, 0);
  args.quality_files = std::max(args.quality_files, 0);
  return args;
}

std::vector<std::filesystem::path> read_manifest(const std::filesystem::path &path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot open manifest: " + path.string());
  }
  std::vector<std::filesystem::path> files;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty()) {
      files.emplace_back(line);
    }
  }
  if (files.empty()) {
    throw std::runtime_error("empty manifest: " + path.string());
  }
  return files;
}

std::vector<float> load_audio_for_spec(const std::filesystem::path &path,
                                       const sirencodec::LiteRtTensorSpec &input_spec,
                                       int sample_rate) {
  if (input_spec.dtype != sirencodec::LiteRtDataType::Float32 || input_spec.shape.size() != 3 ||
      input_spec.shape[0] != 1 || input_spec.shape[2] != 1 || input_spec.shape[1] <= 0) {
    throw std::runtime_error("expected input model shape [1,N,1] float32");
  }
  auto audio = sirencodec::read_audio_mono(path);
  auto samples = sirencodec::resample_linear(audio.samples, audio.sample_rate, sample_rate);
  samples = sirencodec::normalize_peak(samples);
  const auto target = static_cast<std::size_t>(input_spec.shape[1]);
  if (samples.size() > target) {
    samples.resize(target);
  } else if (samples.size() < target) {
    samples.resize(target, 0.0F);
  }
  return samples;
}

TimedOutput run_timed(const sirencodec::LiteRtRunner &runner,
                      const std::vector<sirencodec::LiteRtTensorValue> &inputs) {
  const auto start = std::chrono::steady_clock::now();
  auto outputs = runner.run(inputs);
  const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return {std::move(outputs), elapsed};
}

TimingSummary summarize(std::vector<double> values, double audio_seconds) {
  if (values.empty()) {
    throw std::runtime_error("empty timings");
  }
  std::sort(values.begin(), values.end());
  const auto pct = [&](double p) {
    const double pos = p * static_cast<double>(values.size() - 1);
    return values[static_cast<std::size_t>(std::lround(pos))];
  };
  const double mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
  return {values.size(), mean, values.front(), pct(0.50), pct(0.90), values.back(), audio_seconds / mean};
}

double current_rss_mib() {
#if defined(__linux__)
  std::ifstream in("/proc/self/status");
  std::string key;
  while (in >> key) {
    if (key == "VmRSS:") {
      double value_kib = 0.0;
      std::string unit;
      in >> value_kib >> unit;
      return value_kib / 1024.0;
    }
    std::string rest;
    std::getline(in, rest);
  }
#endif
  return 0.0;
}

double peak_rss_mib() {
#if defined(__linux__) || defined(__APPLE__)
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) == 0) {
#if defined(__APPLE__)
    return static_cast<double>(usage.ru_maxrss) / 1024.0 / 1024.0;
#else
    return static_cast<double>(usage.ru_maxrss) / 1024.0;
#endif
  }
#endif
  return current_rss_mib();
}

MemorySnapshot memory_snapshot(std::string label) {
  return {std::move(label), current_rss_mib(), peak_rss_mib()};
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

std::string json_escape_string(const std::string &text) {
  std::ostringstream out;
  for (char ch : text) {
    if (ch == '\\' || ch == '"') {
      out << '\\';
    }
    out << ch;
  }
  return out.str();
}

void write_summary(std::ostream &out, const TimingSummary &summary, bool trailing_comma) {
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
    const auto files = read_manifest(args.manifest);
    const sirencodec::LiteRtRunner::Options options{args.litert_lib, args.num_threads};
    std::vector<MemorySnapshot> memory_snapshots;
    memory_snapshots.push_back(memory_snapshot("start"));

    std::unique_ptr<sirencodec::LiteRtRunner> full;
    std::unique_ptr<sirencodec::LiteRtRunner> compress;
    std::unique_ptr<sirencodec::LiteRtRunner> decompress;
    if (args.profile == "full" || args.profile == "all") {
      full = std::make_unique<sirencodec::LiteRtRunner>(args.full_model, options);
    }
    if (args.profile == "codec" || args.profile == "all") {
      compress = std::make_unique<sirencodec::LiteRtRunner>(args.compress_model, options);
      decompress = std::make_unique<sirencodec::LiteRtRunner>(args.decompress_model, options);
    }
    memory_snapshots.push_back(memory_snapshot("after_model_load"));

    const auto &input_spec = full ? full->input_specs().at(0) : compress->input_specs().at(0);
    std::vector<std::vector<float>> audio;
    audio.reserve(files.size());
    for (const auto &file : files) {
      audio.push_back(load_audio_for_spec(file, input_spec, args.sample_rate));
    }
    memory_snapshots.push_back(memory_snapshot("after_audio_preload"));
    const double audio_seconds = static_cast<double>(input_spec.shape[1]) / static_cast<double>(args.sample_rate);

    std::vector<double> full_seconds;
    std::vector<double> compress_seconds;
    std::vector<double> decompress_seconds;
    std::vector<double> codec_seconds;
    std::vector<sirencodec::QualityMetrics> quality;

    const auto warmup = static_cast<std::size_t>(std::min(args.warmup_files, static_cast<int>(audio.size())));
    for (std::size_t i = 0; i < warmup; ++i) {
      const auto input = sirencodec::LiteRtTensorValue::from_floats(input_spec, audio[i]);
      if (full) {
        (void)full->run({input});
      }
      if (compress && decompress) {
        auto packet = compress->run({input});
        (void)decompress->run(packet);
      }
    }
    memory_snapshots.push_back(memory_snapshot("after_warmup"));

    for (std::size_t i = 0; i < audio.size(); ++i) {
      const auto input = sirencodec::LiteRtTensorValue::from_floats(input_spec, audio[i]);
      std::vector<float> recon;
      if (full) {
        auto y = run_timed(*full, {input});
        full_seconds.push_back(y.seconds);
        if (!compress) {
          recon = y.outputs.at(0).as_floats();
        }
      }
      if (compress && decompress) {
        auto packet = run_timed(*compress, {input});
        auto decoded = run_timed(*decompress, packet.outputs);
        compress_seconds.push_back(packet.seconds);
        decompress_seconds.push_back(decoded.seconds);
        codec_seconds.push_back(packet.seconds + decoded.seconds);
        recon = decoded.outputs.at(0).as_floats();
      }
      if (i < static_cast<std::size_t>(args.quality_files) && !recon.empty()) {
        recon.resize(audio[i].size());
        quality.push_back(sirencodec::compute_quality_metrics_16k(audio[i], recon));
      }
    }
    memory_snapshots.push_back(memory_snapshot("after_benchmark"));

    std::filesystem::create_directories(args.output_dir);
    const auto report_path = args.output_dir / "benchmark_summary.json";
    std::ofstream report(report_path);
    report.setf(std::ios::fixed);
    report.precision(9);
    report << "{\n";
    report << "  \"backend\": \"litert-cpp\",\n";
    report << "  \"profile\": \"" << args.profile << "\",\n";
    report << "  \"manifest\": \"" << json_escape(args.manifest) << "\",\n";
    report << "  \"selected_files\": " << files.size() << ",\n";
    report << "  \"samples_per_file\": " << input_spec.shape[1] << ",\n";
    report << "  \"audio_seconds_per_file\": " << audio_seconds << ",\n";
    report << "  \"timing_excludes_io\": true,\n";
    report << "  \"num_threads\": " << args.num_threads << ",\n";
    report << "  \"warmup_files\": " << args.warmup_files << ",\n";
    report << "  \"memory\": {\n";
    report << "    \"scope\": \"process RSS; includes LiteRT runtime, loaded models and preloaded audio buffers\",\n";
    report << "    \"rss_current_mib\": " << current_rss_mib() << ",\n";
    report << "    \"rss_peak_mib\": " << peak_rss_mib() << ",\n";
    report << "    \"snapshots\": [\n";
    for (std::size_t i = 0; i < memory_snapshots.size(); ++i) {
      const auto &snap = memory_snapshots[i];
      report << "      {\"label\": \"" << json_escape_string(snap.label) << "\", "
             << "\"rss_mib\": " << snap.rss_mib << ", "
             << "\"rss_peak_mib\": " << snap.rss_peak_mib << "}";
      report << (i + 1 < memory_snapshots.size() ? "," : "") << '\n';
    }
    report << "    ]\n";
    report << "  },\n";
    report << "  \"benchmarks\": {\n";
    bool wrote = false;
    if (!full_seconds.empty()) {
      report << "    \"full\": {\n";
      write_summary(report, summarize(full_seconds, audio_seconds), !compress_seconds.empty());
      wrote = true;
    }
    if (!compress_seconds.empty()) {
      if (wrote) {
        // comma already emitted by write_summary above
      }
      report << "    \"compress_only\": {\n";
      write_summary(report, summarize(compress_seconds, audio_seconds), true);
      report << "    \"decompress_only\": {\n";
      write_summary(report, summarize(decompress_seconds, audio_seconds), true);
      report << "    \"codec_full\": {\n";
      write_summary(report, summarize(codec_seconds, audio_seconds), false);
    }
    report << "  },\n";
    report << "  \"quality_files\": " << quality.size() << ",\n";
    if (!quality.empty()) {
      const auto avg = [&](auto getter) {
        double sum = 0.0;
        for (const auto &m : quality) {
          sum += getter(m);
        }
        return sum / static_cast<double>(quality.size());
      };
      report << "  \"metrics_mean\": {\n";
      report << "    \"si_sdr_db\": " << avg([](const auto &m) { return m.si_sdr_db; }) << ",\n";
      report << "    \"lsd_db\": " << avg([](const auto &m) { return m.lsd_db; }) << ",\n";
      report << "    \"l1\": " << avg([](const auto &m) { return m.l1; }) << ",\n";
      report << "    \"cos\": " << avg([](const auto &m) { return m.cosine; }) << '\n';
      report << "  }\n";
    } else {
      report << "  \"metrics_mean\": null\n";
    }
    report << "}\n";

    std::cout.setf(std::ios::fixed);
    std::cout.precision(6);
    std::cout << "litert-cpp: files=" << files.size() << " samples=" << input_spec.shape[1]
              << " profile=" << args.profile << '\n';
    std::cout << "memory: rss_peak=" << peak_rss_mib() << " MiB rss_current=" << current_rss_mib() << " MiB\n";
    if (!full_seconds.empty()) {
      const auto s = summarize(full_seconds, audio_seconds);
      std::cout << "full mean=" << s.mean_seconds << "s p50=" << s.p50_seconds
                << "s p90=" << s.p90_seconds << "s xrt=" << s.x_realtime << '\n';
    }
    if (!codec_seconds.empty()) {
      const auto s = summarize(codec_seconds, audio_seconds);
      std::cout << "codec_full mean=" << s.mean_seconds << "s p50=" << s.p50_seconds
                << "s p90=" << s.p90_seconds << "s xrt=" << s.x_realtime << '\n';
    }
    std::cout << "wrote: " << report_path << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}

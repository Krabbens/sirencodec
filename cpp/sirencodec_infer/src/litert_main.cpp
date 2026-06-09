#include "sirencodec/audio_wav.hpp"
#include "sirencodec/litert_runner.hpp"
#include "sirencodec/metrics.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
  std::string mode{"codec"};
  std::filesystem::path input;
  std::filesystem::path full_model;
  std::filesystem::path compress_model;
  std::filesystem::path decompress_model;
  std::filesystem::path output_dir{"artifacts/inference/litert"};
  std::filesystem::path litert_lib;
  int sample_rate{16000};
  int num_threads{0};
};

void print_usage(const char *argv0) {
  std::cerr
      << "usage: " << argv0 << " --input audio.wav --mode full|codec [options]\n\n"
      << "options:\n"
      << "  --full-model FILE        Full waveform->waveform .tflite model\n"
      << "  --compress-model FILE    waveform->indices,norms .tflite model\n"
      << "  --decompress-model FILE  indices,norms->waveform .tflite model\n"
      << "  --output-dir DIR         Output directory (default: artifacts/inference/litert)\n"
      << "  --litert-lib FILE        libLiteRt.so path (default: env/path probe)\n"
      << "  --sample-rate N          Audio sample rate for exported shape (default: 16000)\n"
      << "  --num-threads N          XNNPACK CPU threads via LiteRT opaque options\n";
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
    if (key == "--mode") {
      args.mode = require_value(key);
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
    } else {
      throw std::runtime_error("unknown argument: " + key);
    }
  }

  if (args.input.empty()) {
    throw std::runtime_error("--input is required");
  }
  if (args.mode != "full" && args.mode != "codec") {
    throw std::runtime_error("--mode must be full or codec");
  }
  if (args.mode == "full" && args.full_model.empty()) {
    throw std::runtime_error("--full-model is required for --mode full");
  }
  if (args.mode == "codec" && (args.compress_model.empty() || args.decompress_model.empty())) {
    throw std::runtime_error("--compress-model and --decompress-model are required for --mode codec");
  }
  if (args.sample_rate <= 0) {
    throw std::runtime_error("--sample-rate must be positive");
  }
  if (args.num_threads < 0) {
    throw std::runtime_error("--num-threads must be non-negative");
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

void print_specs(const std::string &label, const std::vector<sirencodec::LiteRtTensorSpec> &specs) {
  std::cout << label << ':';
  for (std::size_t i = 0; i < specs.size(); ++i) {
    std::cout << " #" << i << '=' << specs[i].dtype_name() << specs[i].shape_string();
  }
  std::cout << '\n';
}

struct TimedOutput {
  std::vector<sirencodec::LiteRtTensorValue> outputs;
  double seconds{0.0};
};

TimedOutput run_timed(const sirencodec::LiteRtRunner &runner,
                      const std::vector<sirencodec::LiteRtTensorValue> &inputs) {
  const auto start = std::chrono::steady_clock::now();
  auto outputs = runner.run(inputs);
  const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return {std::move(outputs), elapsed};
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto args = parse_args(argc, argv);
    sirencodec::LiteRtRunner::Options options{args.litert_lib, args.num_threads};

    std::cout.setf(std::ios::fixed);
    std::cout.precision(6);
    std::cout << "litert_lib: " << (args.litert_lib.empty() ? sirencodec::default_litert_library_path() : args.litert_lib)
              << '\n';
    std::cout << "num_threads: " << (args.num_threads > 0 ? std::to_string(args.num_threads) : std::string("runtime-default"))
              << '\n';

    std::vector<float> original;
    std::vector<float> recon;
    double full_seconds{0.0};
    double compress_seconds{0.0};
    double decompress_seconds{0.0};

    if (args.mode == "full") {
      sirencodec::LiteRtRunner full(args.full_model, options);
      print_specs("full inputs", full.input_specs());
      print_specs("full outputs", full.output_specs());
      original = load_audio_for_spec(args, full.input_specs().at(0));
      const auto input = sirencodec::LiteRtTensorValue::from_floats(full.input_specs().at(0), original);
      auto run = run_timed(full, {input});
      recon = run.outputs.at(0).as_floats();
      full_seconds = run.seconds;
    } else {
      sirencodec::LiteRtRunner compress(args.compress_model, options);
      sirencodec::LiteRtRunner decompress(args.decompress_model, options);
      print_specs("compress inputs", compress.input_specs());
      print_specs("compress outputs", compress.output_specs());
      print_specs("decompress inputs", decompress.input_specs());
      print_specs("decompress outputs", decompress.output_specs());
      original = load_audio_for_spec(args, compress.input_specs().at(0));
      const auto input = sirencodec::LiteRtTensorValue::from_floats(compress.input_specs().at(0), original);

      auto packet = run_timed(compress, {input});
      auto decoded = run_timed(decompress, packet.outputs);
      recon = decoded.outputs.at(0).as_floats();
      compress_seconds = packet.seconds;
      decompress_seconds = decoded.seconds;
      full_seconds = compress_seconds + decompress_seconds;
    }

    recon.resize(original.size());
    const auto metrics = sirencodec::compute_quality_metrics_16k(original, recon);
    const double audio_seconds = static_cast<double>(original.size()) / static_cast<double>(args.sample_rate);

    std::filesystem::create_directories(args.output_dir);
    const auto stem = args.input.stem().string();
    const auto orig_path = args.output_dir / (stem + "_orig.wav");
    const auto recon_path = args.output_dir / (stem + "_litert_recon.wav");
    const auto stats_path = args.output_dir / (stem + "_litert_stats.json");
    sirencodec::write_wav_mono_pcm16(orig_path, original, args.sample_rate);
    sirencodec::write_wav_mono_pcm16(recon_path, recon, args.sample_rate);

    std::ofstream stats(stats_path);
    stats.setf(std::ios::fixed);
    stats.precision(9);
    stats << "{\n";
    stats << "  \"backend\": \"litert-cpp\",\n";
    stats << "  \"mode\": \"" << args.mode << "\",\n";
    stats << "  \"audio_seconds\": " << audio_seconds << ",\n";
    stats << "  \"num_threads\": " << args.num_threads << ",\n";
    stats << "  \"full_seconds\": " << full_seconds << ",\n";
    stats << "  \"full_x_realtime\": " << audio_seconds / full_seconds << ",\n";
    if (args.mode == "codec") {
      stats << "  \"compress_seconds\": " << compress_seconds << ",\n";
      stats << "  \"compress_x_realtime\": " << audio_seconds / compress_seconds << ",\n";
      stats << "  \"decompress_seconds\": " << decompress_seconds << ",\n";
      stats << "  \"decompress_x_realtime\": " << audio_seconds / decompress_seconds << ",\n";
    }
    stats << "  \"metrics\": {\n";
    stats << "    \"si_sdr_db\": " << metrics.si_sdr_db << ",\n";
    stats << "    \"lsd_db\": " << metrics.lsd_db << ",\n";
    stats << "    \"l1\": " << metrics.l1 << ",\n";
    stats << "    \"cos\": " << metrics.cosine << "\n";
    stats << "  },\n";
    stats << "  \"files\": {\n";
    stats << "    \"original\": \"" << json_escape(orig_path) << "\",\n";
    stats << "    \"reconstruction\": \"" << json_escape(recon_path) << "\"\n";
    stats << "  }\n";
    stats << "}\n";

    std::cout << "mode:       " << args.mode << '\n';
    std::cout << "input:      " << std::filesystem::absolute(args.input) << " (" << audio_seconds << "s)\n";
    if (args.mode == "codec") {
      std::cout << "compress:   " << compress_seconds << " s, xrt=" << audio_seconds / compress_seconds << '\n';
      std::cout << "decompress: " << decompress_seconds << " s, xrt=" << audio_seconds / decompress_seconds << '\n';
    }
    std::cout << "full:       " << full_seconds << " s, xrt=" << audio_seconds / full_seconds << '\n';
    std::cout << "metrics:    si_sdr_db=" << metrics.si_sdr_db
              << " lsd_db=" << metrics.lsd_db
              << " l1=" << metrics.l1
              << " cos=" << metrics.cosine << '\n';
    std::cout << "wrote:      " << orig_path << '\n';
    std::cout << "            " << recon_path << '\n';
    std::cout << "            " << stats_path << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}

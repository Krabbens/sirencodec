#include "sirencodec/audio_wav.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

#if defined(SIRENCODEC_HAVE_SNDFILE)
#include <sndfile.h>
#endif

namespace sirencodec {
namespace {

template <typename T>
T read_pod(std::istream &in) {
  T value{};
  in.read(reinterpret_cast<char *>(&value), static_cast<std::streamsize>(sizeof(T)));
  if (!in) {
    throw std::runtime_error("unexpected end of WAV file");
  }
  return value;
}

void write_u16(std::ostream &out, std::uint16_t value) {
  out.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void write_u32(std::ostream &out, std::uint32_t value) {
  out.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

std::int32_t read_s24(const unsigned char *p) {
  std::int32_t value = static_cast<std::int32_t>(p[0]) |
                       (static_cast<std::int32_t>(p[1]) << 8) |
                       (static_cast<std::int32_t>(p[2]) << 16);
  if ((value & 0x00800000) != 0) {
    value |= static_cast<std::int32_t>(0xFF000000);
  }
  return value;
}

} // namespace

Audio read_audio_mono(const std::filesystem::path &path) {
#if defined(SIRENCODEC_HAVE_SNDFILE)
  SF_INFO info{};
  SNDFILE *file = sf_open(path.string().c_str(), SFM_READ, &info);
  if (file == nullptr) {
    throw std::runtime_error("cannot open audio with libsndfile: " + path.string());
  }
  if (info.channels <= 0 || info.frames <= 0 || info.samplerate <= 0) {
    sf_close(file);
    throw std::runtime_error("invalid audio file: " + path.string());
  }

  std::vector<float> interleaved(static_cast<std::size_t>(info.frames) * static_cast<std::size_t>(info.channels));
  const auto frames_read = sf_readf_float(file, interleaved.data(), info.frames);
  sf_close(file);
  if (frames_read != info.frames) {
    throw std::runtime_error("short audio read: " + path.string());
  }

  Audio audio;
  audio.sample_rate = info.samplerate;
  audio.samples.resize(static_cast<std::size_t>(info.frames));
  for (std::size_t i = 0; i < audio.samples.size(); ++i) {
    audio.samples[i] = interleaved[i * static_cast<std::size_t>(info.channels)];
  }
  return audio;
#else
  const auto ext = path.extension().string();
  if (ext == ".wav" || ext == ".WAV") {
    return read_wav_mono(path);
  }
  throw std::runtime_error("only WAV input is supported; rebuild with libsndfile for FLAC/OGG: " + path.string());
#endif
}

Audio read_wav_mono(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open WAV: " + path.string());
  }

  std::array<char, 4> riff{};
  in.read(riff.data(), 4);
  (void)read_pod<std::uint32_t>(in);
  std::array<char, 4> wave{};
  in.read(wave.data(), 4);
  if (std::string(riff.data(), 4) != "RIFF" || std::string(wave.data(), 4) != "WAVE") {
    throw std::runtime_error("unsupported WAV container: " + path.string());
  }

  std::uint16_t audio_format = 0;
  std::uint16_t channels = 0;
  std::uint32_t sample_rate = 0;
  std::uint16_t bits_per_sample = 0;
  std::vector<unsigned char> data;

  while (in && (!data.size() || sample_rate == 0)) {
    std::array<char, 4> chunk_id{};
    in.read(chunk_id.data(), 4);
    if (!in) {
      break;
    }
    const auto chunk_size = read_pod<std::uint32_t>(in);
    const auto id = std::string(chunk_id.data(), 4);
    if (id == "fmt ") {
      audio_format = read_pod<std::uint16_t>(in);
      channels = read_pod<std::uint16_t>(in);
      sample_rate = read_pod<std::uint32_t>(in);
      (void)read_pod<std::uint32_t>(in); // byte rate
      (void)read_pod<std::uint16_t>(in); // block align
      bits_per_sample = read_pod<std::uint16_t>(in);
      if (chunk_size > 16) {
        in.seekg(static_cast<std::streamoff>(chunk_size - 16), std::ios::cur);
      }
    } else if (id == "data") {
      data.resize(chunk_size);
      in.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(data.size()));
      if (!in) {
        throw std::runtime_error("truncated WAV data chunk: " + path.string());
      }
    } else {
      in.seekg(static_cast<std::streamoff>(chunk_size), std::ios::cur);
    }
    if ((chunk_size & 1U) != 0U) {
      in.seekg(1, std::ios::cur);
    }
  }

  if (channels == 0 || sample_rate == 0 || bits_per_sample == 0 || data.empty()) {
    throw std::runtime_error("incomplete WAV file: " + path.string());
  }
  if (audio_format != 1 && audio_format != 3) {
    throw std::runtime_error("only PCM and IEEE-float WAV are supported: " + path.string());
  }

  const auto bytes_per_sample = static_cast<std::size_t>(bits_per_sample / 8);
  const auto frame_size = bytes_per_sample * channels;
  if (bytes_per_sample == 0 || frame_size == 0 || data.size() < frame_size) {
    throw std::runtime_error("invalid WAV sample format: " + path.string());
  }
  const auto frames = data.size() / frame_size;
  Audio audio;
  audio.sample_rate = static_cast<int>(sample_rate);
  audio.samples.resize(frames);

  for (std::size_t i = 0; i < frames; ++i) {
    const auto *p = data.data() + i * frame_size;
    float value = 0.0F;
    if (audio_format == 3 && bits_per_sample == 32) {
      std::memcpy(&value, p, sizeof(float));
    } else if (audio_format == 1 && bits_per_sample == 16) {
      std::int16_t s = 0;
      std::memcpy(&s, p, sizeof(s));
      value = static_cast<float>(s) / 32768.0F;
    } else if (audio_format == 1 && bits_per_sample == 24) {
      value = static_cast<float>(read_s24(p)) / 8388608.0F;
    } else if (audio_format == 1 && bits_per_sample == 32) {
      std::int32_t s = 0;
      std::memcpy(&s, p, sizeof(s));
      value = static_cast<float>(static_cast<double>(s) / 2147483648.0);
    } else {
      throw std::runtime_error("unsupported WAV bit depth: " + path.string());
    }
    audio.samples[i] = value;
  }
  return audio;
}

void write_wav_mono_pcm16(const std::filesystem::path &path, const std::vector<float> &samples, int sample_rate) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("cannot write WAV: " + path.string());
  }

  const auto data_bytes = static_cast<std::uint32_t>(samples.size() * sizeof(std::int16_t));
  out.write("RIFF", 4);
  write_u32(out, 36U + data_bytes);
  out.write("WAVE", 4);
  out.write("fmt ", 4);
  write_u32(out, 16);
  write_u16(out, 1);
  write_u16(out, 1);
  write_u32(out, static_cast<std::uint32_t>(sample_rate));
  write_u32(out, static_cast<std::uint32_t>(sample_rate * 2));
  write_u16(out, 2);
  write_u16(out, 16);
  out.write("data", 4);
  write_u32(out, data_bytes);
  for (float sample : samples) {
    const auto clipped = std::clamp(sample, -1.0F, 1.0F);
    const auto scaled = std::lrint(static_cast<double>(clipped) * 32767.0);
    const auto s = static_cast<std::int16_t>(std::clamp<long>(scaled, -32768L, 32767L));
    out.write(reinterpret_cast<const char *>(&s), sizeof(s));
  }
}

std::vector<float> resample_linear(const std::vector<float> &samples, int source_rate, int target_rate) {
  if (source_rate == target_rate || samples.empty()) {
    return samples;
  }
  const auto output_size = std::max<std::size_t>(
      1, static_cast<std::size_t>(std::llround(static_cast<double>(samples.size()) * target_rate / source_rate)));
  std::vector<float> out(output_size);
  if (output_size == 1 || samples.size() == 1) {
    out[0] = samples.front();
    return out;
  }
  const double scale = static_cast<double>(samples.size() - 1) / static_cast<double>(output_size - 1);
  for (std::size_t i = 0; i < output_size; ++i) {
    const double pos = static_cast<double>(i) * scale;
    const auto left = static_cast<std::size_t>(std::floor(pos));
    const auto right = std::min(left + 1, samples.size() - 1);
    const auto frac = static_cast<float>(pos - static_cast<double>(left));
    out[i] = samples[left] * (1.0F - frac) + samples[right] * frac;
  }
  return out;
}

std::vector<float> normalize_peak(const std::vector<float> &samples) {
  float peak = 0.0F;
  for (float x : samples) {
    peak = std::max(peak, std::abs(x));
  }
  peak = std::max(peak, 1.0e-5F);
  std::vector<float> out(samples.size());
  for (std::size_t i = 0; i < samples.size(); ++i) {
    out[i] = samples[i] / peak;
  }
  return out;
}

} // namespace sirencodec

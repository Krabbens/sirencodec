#pragma once

#include <filesystem>
#include <vector>

namespace sirencodec {

struct Audio {
  int sample_rate{0};
  std::vector<float> samples;
};

Audio read_wav_mono(const std::filesystem::path &path);
Audio read_audio_mono(const std::filesystem::path &path);
void write_wav_mono_pcm16(const std::filesystem::path &path, const std::vector<float> &samples, int sample_rate);
std::vector<float> resample_linear(const std::vector<float> &samples, int source_rate, int target_rate);
std::vector<float> normalize_peak(const std::vector<float> &samples);

} // namespace sirencodec

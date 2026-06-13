#include "sirencodec/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>
#include <sstream>
#include <stdexcept>

#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__)
#include <arm_neon.h>
#define SIRENCODEC_HAVE_NEON 1
#else
#define SIRENCODEC_HAVE_NEON 0
#endif

namespace sirencodec {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

#if SIRENCODEC_HAVE_NEON
float hsum_f32(float32x4_t v) {
#if defined(__aarch64__)
  return vaddvq_f32(v);
#else
  const float32x2_t hi = vget_high_f32(v);
  const float32x2_t lo = vget_low_f32(v);
  const float32x2_t s = vadd_f32(lo, hi);
  return vget_lane_f32(vpadd_f32(s, s), 0);
#endif
}

double waveform_l1_neon(const float *reference, const float *estimate, std::size_t n) {
  std::size_t i = 0;
  float32x4_t acc0 = vdupq_n_f32(0.0F);
  float32x4_t acc1 = vdupq_n_f32(0.0F);
  for (; i + 8 <= n; i += 8) {
    const float32x4_t r0 = vld1q_f32(reference + i);
    const float32x4_t e0 = vld1q_f32(estimate + i);
    const float32x4_t r1 = vld1q_f32(reference + i + 4);
    const float32x4_t e1 = vld1q_f32(estimate + i + 4);
    acc0 = vaddq_f32(acc0, vabdq_f32(r0, e0));
    acc1 = vaddq_f32(acc1, vabdq_f32(r1, e1));
  }
  double total = static_cast<double>(hsum_f32(vaddq_f32(acc0, acc1)));
  for (; i < n; ++i) {
    total += std::abs(static_cast<double>(reference[i]) - static_cast<double>(estimate[i]));
  }
  return total;
}

struct DotNormSums {
  double dot{};
  double reference_norm{};
  double estimate_norm{};
};

DotNormSums dot_norms_neon(const float *reference, const float *estimate, std::size_t n) {
  std::size_t i = 0;
  float32x4_t dot0 = vdupq_n_f32(0.0F);
  float32x4_t dot1 = vdupq_n_f32(0.0F);
  float32x4_t ref0 = vdupq_n_f32(0.0F);
  float32x4_t ref1 = vdupq_n_f32(0.0F);
  float32x4_t est0 = vdupq_n_f32(0.0F);
  float32x4_t est1 = vdupq_n_f32(0.0F);
  for (; i + 8 <= n; i += 8) {
    const float32x4_t r0 = vld1q_f32(reference + i);
    const float32x4_t e0 = vld1q_f32(estimate + i);
    const float32x4_t r1 = vld1q_f32(reference + i + 4);
    const float32x4_t e1 = vld1q_f32(estimate + i + 4);
    dot0 = vmlaq_f32(dot0, r0, e0);
    dot1 = vmlaq_f32(dot1, r1, e1);
    ref0 = vmlaq_f32(ref0, r0, r0);
    ref1 = vmlaq_f32(ref1, r1, r1);
    est0 = vmlaq_f32(est0, e0, e0);
    est1 = vmlaq_f32(est1, e1, e1);
  }
  DotNormSums sums{
      static_cast<double>(hsum_f32(vaddq_f32(dot0, dot1))),
      static_cast<double>(hsum_f32(vaddq_f32(ref0, ref1))),
      static_cast<double>(hsum_f32(vaddq_f32(est0, est1))),
  };
  for (; i < n; ++i) {
    const double r = static_cast<double>(reference[i]);
    const double e = static_cast<double>(estimate[i]);
    sums.dot += r * e;
    sums.reference_norm += r * r;
    sums.estimate_norm += e * e;
  }
  return sums;
}
#endif

void fft_inplace(std::vector<std::complex<double>> &a) {
  const auto n = a.size();
  if (n == 0 || (n & (n - 1)) != 0) {
    throw std::runtime_error("FFT size must be a power of two");
  }

  for (std::size_t i = 1, j = 0; i < n; ++i) {
    std::size_t bit = n >> 1U;
    for (; (j & bit) != 0U; bit >>= 1U) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      std::swap(a[i], a[j]);
    }
  }

  for (std::size_t len = 2; len <= n; len <<= 1U) {
    const double angle = -2.0 * kPi / static_cast<double>(len);
    const std::complex<double> wlen(std::cos(angle), std::sin(angle));
    for (std::size_t i = 0; i < n; i += len) {
      std::complex<double> w(1.0, 0.0);
      for (std::size_t j = 0; j < len / 2; ++j) {
        const auto u = a[i + j];
        const auto v = a[i + j + len / 2] * w;
        a[i + j] = u + v;
        a[i + j + len / 2] = u - v;
        w *= wlen;
      }
    }
  }
}

double mean(const std::vector<double> &x) {
  if (x.empty()) {
    return 0.0;
  }
  return std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(x.size());
}

double si_sdr_db(const std::vector<float> &reference, const std::vector<float> &estimate) {
  const auto n = std::min(reference.size(), estimate.size());
  if (n == 0) {
    return -100.0;
  }
  std::vector<double> ref(n);
  std::vector<double> est(n);
  for (std::size_t i = 0; i < n; ++i) {
    ref[i] = static_cast<double>(reference[i]);
    est[i] = static_cast<double>(estimate[i]);
  }
  const auto ref_mean = mean(ref);
  const auto est_mean = mean(est);
  for (std::size_t i = 0; i < n; ++i) {
    ref[i] -= ref_mean;
    est[i] -= est_mean;
  }

  double dot = 0.0;
  double ref_energy = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    dot += ref[i] * est[i];
    ref_energy += ref[i] * ref[i];
  }

  double target_energy = 0.0;
  double noise_energy = 0.0;
  const double scale = dot / (ref_energy + 1.0e-12);
  for (std::size_t i = 0; i < n; ++i) {
    const double target = scale * ref[i];
    const double noise = est[i] - target;
    target_energy += target * target;
    noise_energy += noise * noise;
  }
  return 10.0 * std::log10(target_energy / (noise_energy + 1.0e-12) + 1.0e-12);
}

double waveform_l1(const std::vector<float> &reference, const std::vector<float> &estimate) {
  const auto n = std::min(reference.size(), estimate.size());
  if (n == 0) {
    return 0.0;
  }
  double total = 0.0;
#if SIRENCODEC_HAVE_NEON
  total = waveform_l1_neon(reference.data(), estimate.data(), n);
#else
  for (std::size_t i = 0; i < n; ++i) {
    total += std::abs(static_cast<double>(reference[i]) - static_cast<double>(estimate[i]));
  }
#endif
  return total / static_cast<double>(n);
}

double waveform_cosine(const std::vector<float> &reference, const std::vector<float> &estimate) {
  const auto n = std::min(reference.size(), estimate.size());
  if (n == 0) {
    return 0.0;
  }
  double dot = 0.0;
  double ref_norm = 0.0;
  double est_norm = 0.0;
#if SIRENCODEC_HAVE_NEON
  const auto sums = dot_norms_neon(reference.data(), estimate.data(), n);
  dot = sums.dot;
  ref_norm = sums.reference_norm;
  est_norm = sums.estimate_norm;
#else
  for (std::size_t i = 0; i < n; ++i) {
    const double r = reference[i];
    const double e = estimate[i];
    dot += r * e;
    ref_norm += r * r;
    est_norm += e * e;
  }
#endif
  return dot / (std::sqrt(ref_norm) * std::sqrt(est_norm) + 1.0e-12);
}

double log_spectral_distance_db(const std::vector<float> &reference,
                                const std::vector<float> &estimate,
                                std::size_t n_fft = 512,
                                std::size_t hop = 128) {
  const auto n = std::min(reference.size(), estimate.size());
  if (n == 0) {
    return 100.0;
  }
  const auto padded = std::max(n, n_fft);
  std::vector<double> ref(padded, 0.0);
  std::vector<double> est(padded, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    ref[i] = static_cast<double>(reference[i]);
    est[i] = static_cast<double>(estimate[i]);
  }

  std::vector<double> window(n_fft);
  for (std::size_t i = 0; i < n_fft; ++i) {
    window[i] = 0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(i) / static_cast<double>(n_fft - 1));
  }

  std::vector<double> frame_errors;
  std::vector<std::complex<double>> fr(n_fft);
  std::vector<std::complex<double>> fe(n_fft);
  const auto last_start = padded > n_fft ? padded - n_fft : 0;
  for (std::size_t start = 0; start <= last_start; start += hop) {
    for (std::size_t i = 0; i < n_fft; ++i) {
      fr[i] = {ref[start + i] * window[i], 0.0};
      fe[i] = {est[start + i] * window[i], 0.0};
    }
    fft_inplace(fr);
    fft_inplace(fe);

    double err = 0.0;
    for (std::size_t bin = 0; bin <= n_fft / 2; ++bin) {
      const double sr = std::abs(fr[bin]);
      const double se = std::abs(fe[bin]);
      const double dr = 20.0 * std::log10(std::max(sr, 1.0e-7));
      const double de = 20.0 * std::log10(std::max(se, 1.0e-7));
      const double diff = dr - de;
      err += diff * diff;
    }
    frame_errors.push_back(err / static_cast<double>(n_fft / 2 + 1));
  }

  if (frame_errors.empty()) {
    return 100.0;
  }
  const auto mse = std::accumulate(frame_errors.begin(), frame_errors.end(), 0.0) /
                   static_cast<double>(frame_errors.size());
  return std::sqrt(mse);
}

} // namespace

QualityMetrics compute_quality_metrics_16k(const std::vector<float> &reference, const std::vector<float> &estimate) {
  return {
      si_sdr_db(reference, estimate),
      log_spectral_distance_db(reference, estimate),
      waveform_l1(reference, estimate),
      waveform_cosine(reference, estimate),
  };
}

std::string metrics_json(const QualityMetrics &metrics) {
  std::ostringstream out;
  out.setf(std::ios::fixed);
  out.precision(9);
  out << "{\n";
  out << "  \"si_sdr_db\": " << metrics.si_sdr_db << ",\n";
  out << "  \"lsd_db\": " << metrics.lsd_db << ",\n";
  out << "  \"l1\": " << metrics.l1 << ",\n";
  out << "  \"cos\": " << metrics.cosine << "\n";
  out << "}\n";
  return out.str();
}

} // namespace sirencodec

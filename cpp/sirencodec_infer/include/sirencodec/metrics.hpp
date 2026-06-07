#pragma once

#include <string>
#include <vector>

namespace sirencodec {

struct QualityMetrics {
  double si_sdr_db{0.0};
  double lsd_db{0.0};
  double l1{0.0};
  double cosine{0.0};
};

QualityMetrics compute_quality_metrics_16k(const std::vector<float> &reference, const std::vector<float> &estimate);
std::string metrics_json(const QualityMetrics &metrics);

} // namespace sirencodec

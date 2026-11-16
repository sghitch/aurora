// Copyright (C) 2026 Stephen Hitchcock
//
// This file is part of Aurora.
//
// Aurora is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option)
// any later version.
//
// Aurora is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along
// with Aurora. If not, see <https://www.gnu.org/licenses/>.

#include "detection/cfar_detector.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

namespace aurora::detection {

namespace {

void validate_cfar_config(const CfarConfig& config) {
    // FIXED and UNSPECIFIED modes don't require validation
    if (config.mode() == CFAR_MODE_FIXED ||
        config.mode() == CFAR_MODE_UNSPECIFIED) {
        return;
    }

    std::ostringstream errors;

    if (config.guard_cells() == 0) {
        errors << "guard_cells must be > 0; ";
    }
    if (config.reference_cells() == 0) {
        errors << "reference_cells must be > 0; ";
    }
    if (config.threshold_factor() <= 0.0) {
        errors << "threshold_factor must be > 0; ";
    }
    if (config.min_threshold() <= 0.0) {
        errors << "min_threshold must be > 0; ";
    }
    if (config.mode() == CFAR_MODE_OS && config.os_rank() == 0) {
        errors << "os_rank must be > 0 for OS-CFAR mode; ";
    }

    std::string error_str = errors.str();
    if (!error_str.empty()) {
        throw DetectionError(DetectionError::Code::CONFIGURATION_ERROR,
                            "Invalid CFAR configuration: " + error_str);
    }
}

}  // namespace

CfarDetector::CfarDetector(const CfarConfig& config)
    : config_(config) {
    validate_cfar_config(config_);
}

bool CfarDetector::is_enabled() const {
    return config_.mode() != CFAR_MODE_UNSPECIFIED &&
           config_.mode() != CFAR_MODE_FIXED;
}

double CfarDetector::compute_threshold(double noise_estimate) const {
    double cfar_threshold = config_.threshold_factor() * noise_estimate;
    return std::max(cfar_threshold, config_.min_threshold());
}

std::vector<CfarDetection> CfarDetector::detect_1d(
    std::span<const double> magnitudes) const {

    std::vector<CfarDetection> detections;

    if (magnitudes.empty()) {
        return detections;
    }

    const size_t n = magnitudes.size();
    const uint32_t guard = config_.guard_cells();
    const uint32_t ref = config_.reference_cells();
    const uint32_t skip = config_.skip_bins();

    // Minimum cells needed for full window: guard + ref on each side
    const uint32_t margin = guard + ref;

    // Start after skip_bins (filters DC/near-zero range artifacts)
    for (size_t i = skip; i < n; ++i) {
        double noise_est;

        // Handle edges with reduced window
        if (i < margin || i >= n - margin) {
            noise_est = estimate_noise_edge(magnitudes, static_cast<uint32_t>(i));
        } else {
            // Full window available - use configured mode
            switch (config_.mode()) {
                case CFAR_MODE_CA:
                    noise_est = estimate_noise_ca(magnitudes, static_cast<uint32_t>(i));
                    break;
                case CFAR_MODE_GO:
                    noise_est = estimate_noise_go(magnitudes, static_cast<uint32_t>(i));
                    break;
                case CFAR_MODE_SO:
                    noise_est = estimate_noise_so(magnitudes, static_cast<uint32_t>(i));
                    break;
                case CFAR_MODE_OS:
                    noise_est = estimate_noise_os(magnitudes, static_cast<uint32_t>(i));
                    break;
                default:
                    // CFAR_MODE_UNSPECIFIED or CFAR_MODE_FIXED - use CA as fallback
                    noise_est = estimate_noise_ca(magnitudes, static_cast<uint32_t>(i));
                    break;
            }
        }

        // Compute threshold
        double threshold = compute_threshold(noise_est);

        // Check for detection
        if (magnitudes[i] > threshold) {
            CfarDetection det;
            det.set_bin_index(static_cast<uint32_t>(i));
            det.set_magnitude(magnitudes[i]);
            det.set_noise_estimate(noise_est);
            det.set_threshold(threshold);
            // SNR in dB: 10 * log10(signal / noise)
            double snr = (noise_est > 0.0)
                ? 10.0 * std::log10(magnitudes[i] / noise_est)
                : 0.0;
            det.set_snr_db(snr);
            detections.push_back(det);
        }
    }

    return detections;
}

double CfarDetector::estimate_noise_ca(
    std::span<const double> magnitudes,
    uint32_t cell_index) const {

    const uint32_t guard = config_.guard_cells();
    const uint32_t ref = config_.reference_cells();

    double sum = 0.0;
    uint32_t count = 0;

    // Leading reference cells (before guard region)
    uint32_t leading_start = cell_index - guard - ref;
    uint32_t leading_end = cell_index - guard;
    for (uint32_t i = leading_start; i < leading_end; ++i) {
        sum += magnitudes[i];
        ++count;
    }

    // Lagging reference cells (after guard region)
    uint32_t lagging_start = cell_index + guard + 1;
    uint32_t lagging_end = cell_index + guard + ref + 1;
    for (uint32_t i = lagging_start; i < lagging_end; ++i) {
        sum += magnitudes[i];
        ++count;
    }

    return (count > 0) ? sum / count : 0.0;
}

double CfarDetector::estimate_noise_go(
    std::span<const double> magnitudes,
    uint32_t cell_index) const {

    const uint32_t guard = config_.guard_cells();
    const uint32_t ref = config_.reference_cells();

    // Leading average
    double leading_sum = 0.0;
    uint32_t leading_start = cell_index - guard - ref;
    uint32_t leading_end = cell_index - guard;
    for (uint32_t i = leading_start; i < leading_end; ++i) {
        leading_sum += magnitudes[i];
    }
    double leading_avg = leading_sum / ref;

    // Lagging average
    double lagging_sum = 0.0;
    uint32_t lagging_start = cell_index + guard + 1;
    uint32_t lagging_end = cell_index + guard + ref + 1;
    for (uint32_t i = lagging_start; i < lagging_end; ++i) {
        lagging_sum += magnitudes[i];
    }
    double lagging_avg = lagging_sum / ref;

    // Greatest-Of: take maximum
    return std::max(leading_avg, lagging_avg);
}

double CfarDetector::estimate_noise_so(
    std::span<const double> magnitudes,
    uint32_t cell_index) const {

    const uint32_t guard = config_.guard_cells();
    const uint32_t ref = config_.reference_cells();

    // Leading average
    double leading_sum = 0.0;
    uint32_t leading_start = cell_index - guard - ref;
    uint32_t leading_end = cell_index - guard;
    for (uint32_t i = leading_start; i < leading_end; ++i) {
        leading_sum += magnitudes[i];
    }
    double leading_avg = leading_sum / ref;

    // Lagging average
    double lagging_sum = 0.0;
    uint32_t lagging_start = cell_index + guard + 1;
    uint32_t lagging_end = cell_index + guard + ref + 1;
    for (uint32_t i = lagging_start; i < lagging_end; ++i) {
        lagging_sum += magnitudes[i];
    }
    double lagging_avg = lagging_sum / ref;

    // Smallest-Of: take minimum
    return std::min(leading_avg, lagging_avg);
}

double CfarDetector::estimate_noise_os(
    std::span<const double> magnitudes,
    uint32_t cell_index) const {

    const uint32_t guard = config_.guard_cells();
    const uint32_t ref = config_.reference_cells();

    // Collect all reference cells
    std::vector<double> ref_cells;
    ref_cells.reserve(2 * ref);

    // Leading reference cells
    uint32_t leading_start = cell_index - guard - ref;
    uint32_t leading_end = cell_index - guard;
    for (uint32_t i = leading_start; i < leading_end; ++i) {
        ref_cells.push_back(magnitudes[i]);
    }

    // Lagging reference cells
    uint32_t lagging_start = cell_index + guard + 1;
    uint32_t lagging_end = cell_index + guard + ref + 1;
    for (uint32_t i = lagging_start; i < lagging_end; ++i) {
        ref_cells.push_back(magnitudes[i]);
    }

    if (ref_cells.empty()) {
        return 0.0;
    }

    // Sort and select K-th value
    std::sort(ref_cells.begin(), ref_cells.end());

    uint32_t rank = config_.os_rank();
    if (rank >= ref_cells.size()) {
        rank = static_cast<uint32_t>(ref_cells.size()) - 1;
    }

    return ref_cells[rank];
}

double CfarDetector::estimate_noise_edge(
    std::span<const double> magnitudes,
    uint32_t cell_index) const {

    const uint32_t guard = config_.guard_cells();
    const uint32_t ref = config_.reference_cells();
    const size_t n = magnitudes.size();

    double sum = 0.0;
    uint32_t count = 0;

    // Compute available reference window, excluding guard cells and CUT
    // Leading reference cells
    if (cell_index > guard) {
        uint32_t start = (cell_index > guard + ref) ? cell_index - guard - ref : 0;
        uint32_t end = cell_index - guard;
        for (uint32_t i = start; i < end; ++i) {
            sum += magnitudes[i];
            ++count;
        }
    }

    // Lagging reference cells
    if (cell_index + guard + 1 < n) {
        uint32_t start = cell_index + guard + 1;
        uint32_t end = std::min(static_cast<size_t>(cell_index + guard + ref + 1), n);
        for (size_t i = start; i < end; ++i) {
            sum += magnitudes[i];
            ++count;
        }
    }

    // If we have very few reference cells, use what we have
    // This provides degraded performance at edges but still adapts
    if (count == 0) {
        // Last resort: use neighbors excluding immediate guard region
        for (size_t i = 0; i < n; ++i) {
            // Skip cell under test and immediate guard region
            if (i >= cell_index - guard && i <= cell_index + guard && cell_index >= guard) {
                continue;
            }
            if (cell_index < guard && i <= cell_index + guard) {
                continue;
            }
            sum += magnitudes[i];
            ++count;
        }
    }

    return (count > 0) ? sum / count : config_.min_threshold();
}

}  // namespace aurora::detection

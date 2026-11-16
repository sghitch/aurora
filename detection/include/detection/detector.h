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

#ifndef AURORA_DETECTION_DETECTOR_H_
#define AURORA_DETECTION_DETECTOR_H_

/**
 * @file detector.h
 * @brief Main detection pipeline with phase-gradient angle estimation.
 *
 * The Detector class implements the 50 Hz object-level detection stream.
 * It integrates:
 * - CfarDetector (or a fixed threshold) for range bin discovery
 * - AngleEstimator for azimuth/elevation from the K-space phase gradient
 * - DopplerExtractor for radial velocity estimation
 *
 * ### Pipeline Overview
 *
 * 1. Find active range bins (energy above threshold / CFAR)
 * 2. For each active bin:
 *    a) Gather (K-vector, complex value) samples across the detector pixels
 *    b) Phase-gradient angle estimation
 *    c) Doppler extraction from up/down chirp pairs
 * 3. Output DetectionFrame with Detection records
 *
 * @see AngleEstimator for angle estimation
 * @see CfarDetector for adaptive thresholding
 * @see DopplerExtractor for velocity extraction
 */

#include <cstdint>
#include <memory>
#include <vector>

#include "detection/angle_estimator.h"
#include "detection/cfar_detector.h"
#include "detection/detection_error.h"
#include "detection/doppler_extractor.h"

#include "detection.pb.h"
#include "detection_config.pb.h"
#include "detection_frame.pb.h"
#include "forward_config.pb.h"
#include "measurement_tensor.pb.h"
#include "sequence_schedule.pb.h"

namespace aurora::detection {

/**
 * @brief Main detection pipeline with phase-gradient angle estimation.
 *
 * ### Example Usage
 * @code
 * DetectionConfig config;
 * config.set_magnitude_threshold_db(-30);
 * config.set_num_angular_bins(181);
 * config.set_noise_floor_db(-60);
 *
 * aurora::forward::ForwardConfig forward_config = ...;
 * aurora::acquisition::SequenceSchedule sequence_schedule = ...;
 *
 * Detector detector(config, forward_config, sequence_schedule);
 *
 * MeasurementTensor measurements = ...;
 * DetectionFrame frame = detector.detect(measurements);
 *
 * for (const auto& det : frame.detections()) {
 *     double azimuth = det.azimuth_rad();
 *     double az_uncertainty = det.azimuth_uncertainty_rad();
 * }
 * @endcode
 */
class Detector {
  public:
    /**
     * @brief Construct detector.
     *
     * @param config Detection configuration
     * @param forward_config Forward configuration (optics and detector geometry)
     * @param sequence_schedule Sequence schedule (chirp duration for Doppler)
     * @throws DetectionError(CONFIGURATION_ERROR) on invalid configuration
     */
    Detector(const DetectionConfig& config,
             const aurora::forward::ForwardConfig& forward_config,
             const aurora::acquisition::SequenceSchedule& sequence_schedule);

    /**
     * @brief Run detection on a measurement tensor.
     *
     * @param measurements Post-DSP measurements with K-vectors annotated
     * @return DetectionFrame with one Detection per active range bin
     */
    DetectionFrame detect(const aurora::shared::MeasurementTensor& measurements);

    /// Replace the detection configuration and rebuild analyzers.
    void update_config(const DetectionConfig& config);

    uint64_t next_frame_id() const { return next_frame_id_; }

  private:
    DetectionConfig config_;
    aurora::forward::ForwardConfig forward_config_;
    aurora::acquisition::SequenceSchedule sequence_schedule_;

    std::unique_ptr<CfarDetector> cfar_detector_;
    std::unique_ptr<AngleEstimator> angle_estimator_;
    std::unique_ptr<DopplerExtractor> doppler_extractor_;

    uint64_t next_frame_id_ = 0;
    double energy_threshold_ = 0.0;

    void initialize_analyzers();

    std::vector<uint32_t> find_active_range_bins(
        const aurora::shared::MeasurementTensor& m) const;

    /// Reduce runs of adjacent candidate bins to their energy peaks.
    static std::vector<uint32_t> select_run_peaks(
        const std::vector<uint32_t>& candidate_bins,
        const std::vector<double>& bin_energies);

    Detection process_range_bin(
        const aurora::shared::MeasurementTensor& m,
        uint32_t range_bin) const;

    /// Gather K-space samples across all detectors at one wavelength.
    std::vector<AngleEstimator::Sample> gather_samples(
        const aurora::shared::MeasurementTensor& m,
        uint32_t range_bin,
        uint32_t wavelength_index) const;

    void estimate_doppler(
        const aurora::shared::MeasurementTensor& m,
        uint32_t range_bin,
        Detection& det) const;

    double compute_range(
        const aurora::shared::MeasurementTensor& m,
        uint32_t range_bin) const;

    double compute_snr_db(double energy) const;

    double compute_range_bin_energy(
        const aurora::shared::MeasurementTensor& m,
        uint32_t range_bin,
        uint32_t wavelength_index) const;
};

}  // namespace aurora::detection

#endif  // AURORA_DETECTION_DETECTOR_H_

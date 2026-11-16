#pragma once

/**
 * @file test_forward_model_builder.h
 * @brief Test helpers for building detector geometry and ForwardModel instances.
 */

#include <cmath>
#include <string>
#include <vector>

#include "forward/forward_model.h"
#include "forward/k_space_geometry.h"
#include "detector_geometry.pb.h"
#include "forward_config.pb.h"
#include "optics_config.pb.h"

namespace aurora::test {

// =============================================================================
// Detector Geometry Helpers
// =============================================================================

/// Create a linear detector array with uniform pixel spacing.
///
/// @param array_id Unique identifier for this array
/// @param num_pixels Number of detector pixels
/// @param pitch_m Center-to-center pixel spacing in meters
/// @param wavelengths_m Wavelengths this detector is sensitive to
/// @param bandwidth_hz Detector bandwidth in Hz (default: 150 MHz)
/// @param sensitivity_type Detector sensitivity model (default: SINC for physical accuracy)
inline aurora::forward::DetectorArray create_linear_array(
    const std::string& array_id,
    uint32_t num_pixels,
    double pitch_m,
    const std::vector<double>& wavelengths_m,
    double bandwidth_hz = 150e6,
    aurora::forward::DetectorSensitivityType sensitivity_type =
        aurora::forward::DETECTOR_SENSITIVITY_SINC) {

    aurora::forward::DetectorArray array;
    array.set_array_id(array_id);
    array.set_nominal_pitch_m(pitch_m);
    array.set_bandwidth_hz(bandwidth_hz);
    array.set_sensitivity_type(sensitivity_type);

    for (double wl : wavelengths_m) {
        array.add_wavelengths_m(wl);
    }

    // Create pixels centered around origin
    double start = -static_cast<double>(num_pixels - 1) / 2.0 * pitch_m;
    for (uint32_t i = 0; i < num_pixels; ++i) {
        auto* pixel = array.add_pixels();
        pixel->set_index(i);
        pixel->set_x_m(start + i * pitch_m);
        pixel->set_y_m(0.0);
        pixel->set_width_m(pitch_m * 0.8);  // 80% fill factor
        pixel->set_height_m(pitch_m * 0.8);
    }

    return array;
}

/// Create the reference detector geometry: 8 pixels, 0.5 mm pitch, 905 nm + 1550 nm.
inline aurora::forward::DetectorGeometry create_reference_detector() {
    aurora::forward::DetectorGeometry detector;
    *detector.add_arrays() = create_linear_array(
        "reference_detector", 8, 0.5e-3, {905e-9, 1550e-9}, 150e6);
    return detector;
}

/// Builds ForwardModel instances for tests.
class TestForwardModelBuilder {
public:
    /// Build from a complete ForwardConfig.
    static aurora::forward::ForwardModel build(
        const aurora::forward::ForwardConfig& config) {
        return aurora::forward::ForwardModel(config);
    }

    /// Build from detector geometry and optics.
    static aurora::forward::ForwardModel build(
        const aurora::forward::DetectorGeometry& detector,
        const aurora::forward::OpticsConfig& optics) {
        return aurora::forward::ForwardModel(detector, optics);
    }
};

}  // namespace aurora::test

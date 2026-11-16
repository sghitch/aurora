#pragma once

#include <vector>

#include "forward/forward_model.h"
#include "measurement_tensor.pb.h"
#include "range_cube.pb.h"

namespace aurora::dsp {

/// Builds MeasurementTensor protobufs from RangeCube DSP output.
///
/// This class annotates each measurement with its corresponding K-vector
/// from the ForwardModel, producing the calibrated K-space coordinates
/// used by Detection and Reconstruction.
///
/// ### K-Vector Annotation
///
/// For each (wavelength, detector) pair in the RangeCube, the builder:
/// 1. Queries the ForwardModel for the pixel's scene K (detector K plus LO tilt)
/// 2. Attaches the K-vector to the DetectorChannel
///
/// Up/down chirp profiles present in the RangeCube are carried through as
/// `up_chirp_data` / `down_chirp_data` so Doppler can be extracted downstream.
///
/// ### Example Usage
/// @code
/// ForwardModel forward_model(detector, optics);
/// MeasurementBuilder builder(forward_model);
///
/// // Build tensor from single RangeCube
/// RangeCube cube = processor.process_range(burst);
/// MeasurementTensor tensor = builder.build(cube);
///
/// // Coherently average several bursts into one tensor (improves SNR)
/// std::vector<RangeCube> cubes = {cube1, cube2, cube3};
/// MeasurementTensor averaged = builder.build(cubes);
/// @endcode
class MeasurementBuilder {
public:
    /// Construct builder with ForwardModel reference.
    ///
    /// @param forward_model Reference to caller-owned ForwardModel
    explicit MeasurementBuilder(const aurora::forward::ForwardModel& forward_model);

    ~MeasurementBuilder();

    MeasurementBuilder(const MeasurementBuilder&) = delete;
    MeasurementBuilder& operator=(const MeasurementBuilder&) = delete;
    MeasurementBuilder(MeasurementBuilder&&) = delete;
    MeasurementBuilder& operator=(MeasurementBuilder&&) = delete;

    /// Build MeasurementTensor from a single RangeCube.
    aurora::shared::MeasurementTensor build(const RangeCube& range_cube);

    /// Build MeasurementTensor from multiple RangeCubes (coherent average).
    ///
    /// The range bins of all cubes are averaged element-wise; every cube must
    /// have the same wavelengths, detectors and range-bin count.
    ///
    /// @throws std::invalid_argument if cubes have incompatible structures
    aurora::shared::MeasurementTensor build(const std::vector<RangeCube>& range_cubes);

    const aurora::forward::ForwardModel& forward_model() const { return forward_model_; }

private:
    /// Get the scene K-vector (detector K plus LO tilt) for a (wavelength, detector) pair.
    aurora::shared::KVector get_calibrated_k(
        double wavelength_m,
        uint32_t detector_index) const;

    /// Convert range data to K-annotated wavelength channels.
    void convert_channels(
        const google::protobuf::RepeatedPtrField<WavelengthRangeData>& source,
        google::protobuf::RepeatedPtrField<aurora::shared::WavelengthChannel>* target) const;

    const aurora::forward::ForwardModel& forward_model_;
};

}  // namespace aurora::dsp

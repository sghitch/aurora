#ifndef AURORA_FORWARD_FORWARD_MODEL_H_
#define AURORA_FORWARD_FORWARD_MODEL_H_

/**
 * @file forward_model.h
 * @brief K-space mapping service for the coherent receiver.
 *
 * The ForwardModel answers one question for every module downstream of the
 * optics: which spatial frequency does detector pixel `j` sample at
 * wavelength `λ`? From Fourier optics,
 * @f[
 *   k_j(\lambda) = \frac{2\pi \, x_j}{f \, \lambda}
 * @f]
 * where @f$ x_j @f$ is the pixel position at the Fourier plane and @f$ f @f$
 * the lens focal length. The local-oscillator tilt adds a constant offset.
 *
 * ### Design Principles
 *
 * 1. **Single Responsibility**: K-space mapping only - no reconstruction algorithms
 * 2. **Lightweight**: Minimal dependencies (Eigen, protobufs)
 * 3. **Stable Interface**: Physics rarely changes; reconstruction algorithms evolve separately
 * 4. **Service Role**: Provides data that other modules consume
 * 5. **Protobuf Contracts**: KSpaceSlice is a protobuf, not a C++ struct
 *
 * Phase encoding: @f$ \phi = -\mathbf{k} \cdot \mathbf{r} @f$
 *
 * @see KSpaceGeometry for the underlying detector K-position calculations
 */

#include <complex>
#include <vector>

#include <Eigen/Core>

#include "forward/k_space_geometry.h"

#include "detector_geometry.pb.h"
#include "forward_config.pb.h"
#include "k_space_geometry.pb.h"
#include "k_space_slice.pb.h"
#include "measurement_tensor.pb.h"
#include "optics_config.pb.h"

namespace aurora::forward {

/**
 * @brief K-space mapping service.
 *
 * Provides calibrated K-vectors for simulation, DSP, and detection, plus
 * convenience methods for extracting K-space slices from a MeasurementTensor.
 *
 * ### Example Usage
 * @code
 * ForwardModel model(detector_geometry, optics);
 *
 * // K-sample for a detector at a wavelength
 * auto k_sample = model.get_k_sample(detector_idx, 1550e-9);
 *
 * // Extract K-space slice for reconstruction
 * auto slice = model.extract_kspace_slice(measurements, range_bin, wavelength);
 *
 * // Compute phase for target position
 * double phase = ForwardModel::compute_phase(k, target_xy);
 * @endcode
 */
class ForwardModel {
  public:
    /**
     * @brief Construct ForwardModel from detector geometry and optics.
     *
     * @param detector Detector geometry with pixel positions (first array is used)
     * @param optics Optical configuration (focal length, LO tilt)
     * @throws std::invalid_argument if the geometry has no detector array or
     *         pixels, or the focal length is not positive
     */
    ForwardModel(const DetectorGeometry& detector,
                 const OpticsConfig& optics);

    /// Construct from a complete ForwardConfig.
    explicit ForwardModel(const ForwardConfig& config);

    ~ForwardModel();

    ForwardModel(ForwardModel&&) noexcept;
    ForwardModel& operator=(ForwardModel&&) noexcept;

    // Non-copyable
    ForwardModel(const ForwardModel&) = delete;
    ForwardModel& operator=(const ForwardModel&) = delete;

    // =========================================================================
    // K-SPACE MAPPING
    // =========================================================================

    /**
     * @brief Get the K-sample a detector acquires at a wavelength.
     *
     * The sample is in scene K-space before LO tilt: `kx_eff`/`ky_eff` are the
     * geometric detector K-position and the bandwidths are the detector's
     * K-space acceptance window from its pixel size.
     *
     * @throws std::out_of_range if detector_idx >= num_detectors()
     * @throws std::invalid_argument if wavelength_m <= 0
     */
    DetectorKSample get_k_sample(uint32_t detector_idx,
                                 double wavelength_m) const;

    /**
     * @brief Get the effective K-vector in scene coordinates (includes LO tilt).
     *
     * @return 2D K-vector (kx, ky) in rad/m
     * @throws std::out_of_range if detector_idx >= num_detectors()
     */
    Eigen::Vector2d get_effective_k(uint32_t detector_idx,
                                    double wavelength_m) const;

    /// All K-samples at a wavelength, one per detector pixel.
    std::vector<DetectorKSample> get_all_k_samples(double wavelength_m) const;

    /// Number of detector pixels in the array.
    uint32_t num_detectors() const;

    // =========================================================================
    // K-SPACE EXTRACTION FROM MEASUREMENTS
    // =========================================================================

    /**
     * @brief Extract K-space slice for a specific range bin and wavelength.
     *
     * Iterates over the MeasurementTensor, extracting samples at the specified
     * range bin and wavelength, annotating each with its scene K-vector,
     * acceptance bandwidth and a unit sample weight.
     */
    aurora::shared::KSpaceSlice extract_kspace_slice(
        const aurora::shared::MeasurementTensor& measurements,
        uint32_t range_bin,
        double wavelength_m) const;

    /// Extract K-space slices for all range bins (batch processing).
    aurora::shared::KSpaceSliceCollection extract_all_range_slices(
        const aurora::shared::MeasurementTensor& measurements,
        double wavelength_m) const;

    // =========================================================================
    // PHYSICS UTILITIES
    // =========================================================================

    /// Compute K-space phase for target position: phi = -k . r
    static double compute_phase(const Eigen::Vector2d& k,
                                const Eigen::Vector2d& position_xy);

    /// Compute complex phasor: exp(-i * k . r)
    static std::complex<double> compute_phasor(
        const Eigen::Vector2d& k,
        const Eigen::Vector2d& position_xy);

    // =========================================================================
    // ACCESSORS
    // =========================================================================

    const DetectorGeometry& detector() const { return detector_; }
    const OpticsConfig& optics() const { return optics_; }
    const KSpaceGeometry& geometry() const { return geometry_; }

  private:
    DetectorGeometry detector_;
    OpticsConfig optics_;
    KSpaceGeometry geometry_;

    void compute_grid_hints(aurora::shared::KSpaceSlice& slice) const;

    static uint32_t get_num_range_bins(
        const aurora::shared::MeasurementTensor& measurements);
};

}  // namespace aurora::forward

#endif  // AURORA_FORWARD_FORWARD_MODEL_H_

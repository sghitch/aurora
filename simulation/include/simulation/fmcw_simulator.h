#ifndef AURORA_SIMULATION_FMCW_SIMULATOR_H_
#define AURORA_SIMULATION_FMCW_SIMULATOR_H_

/**
 * @file fmcw_simulator.h
 * @brief FMCW beat signal simulator.
 *
 * The FmcwSimulator class generates physically realistic time-domain FMCW
 * beat signals from scene descriptions. It ties together Scene, PupilModel,
 * NoiseModel, and AdcModel to produce complete acquisition bursts.
 *
 * ### Physical Model
 *
 * For each target, the beat signal contains:
 * - Beat frequency from range: f_beat = (2 * B * R) / (c * T)
 * - Doppler shift from velocity: f_d = 2 * v / λ
 * - K-space phase from position: φ = -k · r₀
 *
 * Multiple targets superpose coherently:
 *   s_beat(t) = Σ_n A_n * cos(2π * (f_beat,n + f_d,n) * t + φ_0,n)
 *
 * @see ChirpGenerator for chirp waveform generation
 * @see Scene for target definitions
 * @see NoiseModel for noise injection
 * @see AdcModel for signal digitization
 */

#include <complex>
#include <random>
#include <vector>

#include <Eigen/Core>

#include "forward/forward_model.h"
#include "forward/measurement_operator.h"
#include "simulation/adc_model.h"
#include "simulation/chirp_generator.h"
#include "simulation/noise_model.h"
#include "simulation/pupil_model.h"
#include "simulation/scene.h"

#include "burst.pb.h"
#include "sequence_plan.pb.h"
#include "sim_config.pb.h"

namespace aurora::simulation {

/**
 * @brief Generates realistic FMCW beat signals from scene descriptions.
 *
 * Simulates the complete signal chain from optical field to ADC samples,
 * including target responses, noise injection, and digitization.
 *
 * ### Example Usage
 * @code
 * // Create ForwardModel from detector geometry and optics
 * ForwardModel forward_model(detector_geometry, optics);
 *
 * // Create simulation config
 * SimConfig sim_config;
 * // ... configure scene, ADC, noise ...
 *
 * FmcwSimulator simulator(forward_model, sim_config);
 *
 * aurora::acquisition::SequencePlan plan;
 * // ... configure chirps ...
 *
 * Burst burst = simulator.simulate_burst(plan);
 * @endcode
 */
class FmcwSimulator {
  public:
    /**
     * @brief Construct simulator from ForwardModel and configuration.
     *
     * Initializes all sub-models (Scene, PupilModel, NoiseModel, AdcModel)
     * and the random number generator. The ForwardModel provides the scene
     * K-position and acceptance window of each (detector, wavelength) pair.
     *
     * @param forward_model Reference to caller-owned ForwardModel
     * @param config Simulation configuration (scene, noise, ADC settings)
     */
    FmcwSimulator(const aurora::forward::ForwardModel& forward_model,
                  const SimConfig& config);

    /**
     * @brief Simulate one complete burst.
     *
     * Generates ADC samples for all chirps and detectors
     * in the sequence plan.
     *
     * @param plan Sequence plan with chirp definitions
     * @return Burst containing all ADC samples
     */
    Burst simulate_burst(const aurora::acquisition::SequencePlan& plan);

    /**
     * @brief Simulate a single detector.
     *
     * Generates ADC samples for one detector pixel (all chirps).
     * Useful for debugging or partial simulations.
     *
     * @param plan Sequence plan with chirp definitions
     * @param detector_index Detector pixel index
     * @param rng Random number generator for noise and ADC effects
     * @return DetectorBurst containing ADC samples for this detector
     */
    DetectorBurst simulate_detector(const aurora::acquisition::SequencePlan& plan,
                                    uint32_t detector_index,
                                    std::mt19937_64& rng) const;

    /**
     * @brief Update scene state for dynamic targets.
     *
     * Advances target positions based on their velocities.
     *
     * @param delta_time_s Time step in seconds
     */
    void update_scene(double delta_time_s);

    /**
     * @brief Reset RNG to initial seed.
     *
     * Call this to reproduce identical simulation results.
     */
    void reset_rng();

    /**
     * @brief Get the current scene.
     * @return Reference to the scene
     */
    const Scene& scene() const { return scene_; }

    /**
     * @brief Get the pupil model.
     * @return Reference to the pupil model
     */
    const PupilModel& pupil_model() const { return pupil_model_; }

    /**
     * @brief Get the noise model.
     * @return Reference to the noise model
     */
    const NoiseModel& noise_model() const { return noise_model_; }

    /**
     * @brief Get the ADC model.
     * @return Reference to the ADC model
     */
    const AdcModel& adc_model() const { return adc_model_; }

  private:
    SimConfig config_;
    const aurora::forward::ForwardModel& forward_model_;

    Scene scene_;
    PupilModel pupil_model_;
    NoiseModel noise_model_;
    AdcModel adc_model_;

    std::mt19937_64 rng_;
    uint64_t burst_counter_;

    /**
     * @brief Generate beat signal for a single chirp and detector.
     *
     * Computes the coherent sum of all target contributions. The detector
     * pixel's acceptance window (scene K-position and K-space bandwidth)
     * comes from the ForwardModel.
     *
     * @param chirp_plan Chirp parameters
     * @param detector_index Detector pixel index
     * @param wavelength_m Wavelength in meters
     * @return Beat signal samples (analog, before noise/ADC)
     */
    std::vector<double> generate_beat_signal(
        const aurora::acquisition::ChirpPlan& chirp_plan,
        uint32_t detector_index,
        double wavelength_m) const;

    /**
     * @brief Compute single point target's contribution to beat signal.
     *
     * @param target Point target view
     * @param chirp_plan Chirp parameters
     * @param window Acceptance window of the detector pixel (scene K-position
     *               and K-space bandwidth)
     * @param wavelength_m Wavelength in meters
     * @param transmission Pupil transmission for this target
     * @return Beat signal contribution from this target
     */
    std::vector<double> compute_target_response(
        const PointTargetView& target,
        const aurora::acquisition::ChirpPlan& chirp_plan,
        const aurora::forward::MeasurementOperator::WindowParams& window,
        double wavelength_m,
        double transmission) const;

    /**
     * @brief Compute single plane target's contribution using analytic model.
     *
     * Integrates the plane's Fraunhofer spectrum (specular sinc envelope)
     * over the pixel's acceptance window and applies speckle for rough
     * surfaces, avoiding point sampling of the surface.
     *
     * @param plane Plane target view
     * @param chirp_plan Chirp parameters
     * @param window Acceptance window of the detector pixel
     * @param wavelength_m Wavelength in meters
     * @param amplitude_scale Overall amplitude scaling factor
     * @param speckle_seed Deterministic seed for speckle generation
     * @return Beat signal contribution from this plane target
     */
    std::vector<double> compute_plane_target_response(
        const PlaneTargetView& plane,
        const aurora::acquisition::ChirpPlan& chirp_plan,
        const aurora::forward::MeasurementOperator::WindowParams& window,
        double wavelength_m,
        double amplitude_scale,
        uint64_t speckle_seed) const;

    /**
     * @brief Compute single mesh target's contribution using analytic model.
     *
     * Coherently sums the integrated Fraunhofer response of every visible
     * facet (circular-aperture approximation for triangles, oriented
     * rectangular aperture for quads) with per-facet speckle.
     *
     * @param mesh Mesh target view
     * @param visible_faces Indices of front-facing, visible faces
     * @param chirp_plan Chirp parameters
     * @param window Acceptance window of the detector pixel
     * @param wavelength_m Wavelength in meters
     * @param amplitude_scale Overall amplitude scaling factor
     * @param speckle_seed Deterministic seed for speckle generation
     * @return Beat signal contribution from this mesh target
     */
    std::vector<double> compute_mesh_target_response(
        const MeshTargetView& mesh,
        const std::vector<size_t>& visible_faces,
        const aurora::acquisition::ChirpPlan& chirp_plan,
        const aurora::forward::MeasurementOperator::WindowParams& window,
        double wavelength_m,
        double amplitude_scale,
        uint64_t speckle_seed) const;

    /**
     * @brief Sensitivity type of the detector array.
     *
     * @return POINT (sample the scene spectrum at the pixel's scene K) or
     *         SINC (integrate it over the pixel's acceptance window)
     * @throws std::runtime_error if the ForwardModel has no detector array
     */
    aurora::forward::DetectorSensitivityType sensitivity_type() const;

    /**
     * @brief Apply noise and ADC to analog signal with external RNG.
     *
     * Thread-safe version that uses an externally provided RNG.
     * Used for parallel processing where each thread has its own RNG.
     *
     * @param signal Analog signal samples
     * @param optical_power_w Optical power for noise calculation
     * @param rng Random number generator (modified in place)
     * @return Digitized samples
     */
    std::vector<int16_t> apply_noise_and_adc(
        const std::vector<double>& signal,
        double optical_power_w,
        std::mt19937_64& rng) const;
};

}  // namespace aurora::simulation

#endif  // AURORA_SIMULATION_FMCW_SIMULATOR_H_

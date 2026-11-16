#ifndef AURORA_RECONSTRUCTION_PCG_PRECOMPUTATION_H_
#define AURORA_RECONSTRUCTION_PCG_PRECOMPUTATION_H_

/**
 * @file pcg_precomputation.h
 * @brief Offline precomputation pipeline for PCG reconstruction.
 *
 * Computes expensive-to-compute data when the K-space mapping or calibration
 * changes. This data is reused across frames for fast runtime performance.
 *
 * ### Precomputed Data
 *
 * | Data | Depends On | Compute Cost |
 * |------|------------|--------------|
 * | PSF | geometry, calibration, grid | O(NM) - expensive |
 * | Transfer function Ψ | PSF | O(N log N) - cheap |
 * | Regularizer spectrum Φ | grid, regularizer type | O(N) - cheap, analytical |
 * | Operator diagonal | geometry, calibration, grid | O(NM) - moderate |
 *
 * @see pcg_precomputed_data.proto for data layout
 * @see PcgConfig for configuration
 */

#include <string>

#include <Eigen/Core>

#include "reconstruction/direct_operator.h"
#include "reconstruction/reconstruction_operator.h"
#include "reconstruction/regularizer.h"
#include "calibration_artifact.pb.h"
#include "pcg_config.pb.h"
#include "pcg_precomputed_data.pb.h"

namespace aurora::reconstruction {

/**
 * @brief Static methods for PCG precomputation.
 *
 * Provides offline computation of PSF, transfer function, regularizer
 * spectrum, and operator diagonal for PCG reconstruction.
 */
class PcgPrecomputation {
  public:
    /**
     * @brief Compute precomputed data for a single wavelength.
     *
     * Computes PSF via A†WA·δ, transfer function via FFT, regularizer
     * spectrum analytically, and diagonal approximation.
     *
     * @param op Reconstruction operator (provides forward/adjoint)
     * @param weights Measurement weights (per K-sample)
     * @param grid Grid specification
     * @param wavelength_m Operating wavelength in meters
     * @param regularizer_type Type of regularizer
     * @param preconditioner_type Type of preconditioner
     * @param geometry_id Receiver geometry identifier
     * @param calibration_id Calibration artifact identifier
     * @return Precomputed data protobuf
     */
    static PcgPrecomputedData compute(
        const ReconstructionOperator& op,
        const Eigen::VectorXd& weights,
        const GridSpec& grid,
        double wavelength_m,
        RegularizerType regularizer_type,
        PreconditionerType preconditioner_type,
        const std::string& geometry_id,
        const std::string& calibration_id);

    /**
     * @brief Validate precomputed data matches current inputs.
     *
     * Checks that all input parameters match and array sizes are correct.
     *
     * @param data Precomputed data to validate
     * @param geometry_id Expected receiver geometry ID
     * @param calibration_id Expected calibration ID
     * @param grid Expected grid specification
     * @param wavelength_m Expected wavelength
     * @param config PCG configuration with regularizer/preconditioner types
     * @throws std::invalid_argument if any parameter mismatches or data is corrupted
     */
    static void validate(
        const PcgPrecomputedData& data,
        const std::string& geometry_id,
        const std::string& calibration_id,
        const GridSpec& grid,
        double wavelength_m,
        const PcgConfig& config);

    /**
     * @brief Load precomputed data set from file.
     *
     * @param path Path to protobuf file
     * @return Loaded data set
     * @throws std::runtime_error if file cannot be read or parsed
     */
    static PcgPrecomputedDataSet load(const std::string& path);

    /**
     * @brief Save precomputed data set to file.
     *
     * @param data Data set to save
     * @param path Path to output file
     * @throws std::runtime_error if file cannot be written
     */
    static void save(const PcgPrecomputedDataSet& data, const std::string& path);

    // =========================================================================
    // Low-level computation functions (exposed for testing)
    // =========================================================================

    /**
     * @brief Compute PSF via A†WA applied to centered impulse.
     *
     * Creates delta function at grid center, applies forward, weights,
     * then adjoint to get the point spread function.
     *
     * @param op Reconstruction operator
     * @param weights Measurement weights
     * @param grid Grid specification
     * @return Complex PSF matrix (Ny x Nx)
     */
    static Eigen::MatrixXcd compute_psf(
        const ReconstructionOperator& op,
        const Eigen::VectorXd& weights,
        const GridSpec& grid);

    /**
     * @brief Compute transfer function as FFT of PSF.
     *
     * @param psf Point spread function
     * @return Complex transfer function (Ny x Nx)
     */
    static Eigen::MatrixXcd compute_transfer_function(
        const Eigen::MatrixXcd& psf);

    /**
     * @brief Compute regularizer spectrum.
     *
     * @param grid Grid specification
     * @param type Regularizer type
     * @return Real spectrum (Ny x Nx)
     */
    static Eigen::MatrixXd compute_regularizer_spectrum(
        const GridSpec& grid,
        RegularizerType type);

    /**
     * @brief Compute approximate diagonal of A†WA.
     *
     * Uses A†WA applied to ones as proxy for diagonal energy.
     *
     * @param op Reconstruction operator
     * @param weights Measurement weights
     * @param grid Grid specification
     * @return Real diagonal values (Ny x Nx)
     */
    static Eigen::MatrixXd compute_diagonal_approximate(
        const ReconstructionOperator& op,
        const Eigen::VectorXd& weights,
        const GridSpec& grid);

    // =========================================================================
    // Helper functions for packing/unpacking protobuf arrays
    // =========================================================================

    /**
     * @brief Pack complex matrix into interleaved real/imag vector.
     *
     * @param matrix Complex matrix (Ny x Nx)
     * @param output Repeated double field to populate
     */
    static void pack_complex_array(
        const Eigen::MatrixXcd& matrix,
        google::protobuf::RepeatedField<double>* output);

    /**
     * @brief Unpack interleaved real/imag vector to complex matrix.
     *
     * @param input Interleaved [re, im, re, im, ...] data
     * @param rows Number of rows
     * @param cols Number of columns
     * @return Complex matrix
     */
    static Eigen::MatrixXcd unpack_complex_array(
        const google::protobuf::RepeatedField<double>& input,
        int rows, int cols);

    /**
     * @brief Pack real matrix into vector.
     *
     * @param matrix Real matrix (Ny x Nx)
     * @param output Repeated double field to populate
     */
    static void pack_real_array(
        const Eigen::MatrixXd& matrix,
        google::protobuf::RepeatedField<double>* output);

    /**
     * @brief Unpack vector to real matrix.
     *
     * @param input Real data
     * @param rows Number of rows
     * @param cols Number of columns
     * @return Real matrix
     */
    static Eigen::MatrixXd unpack_real_array(
        const google::protobuf::RepeatedField<double>& input,
        int rows, int cols);
};

}  // namespace aurora::reconstruction

#endif  // AURORA_RECONSTRUCTION_PCG_PRECOMPUTATION_H_

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

#ifndef AURORA_RECONSTRUCTION_PCG_PRECOMPUTATION_BUILDER_H_
#define AURORA_RECONSTRUCTION_PCG_PRECOMPUTATION_BUILDER_H_

/**
 * @file pcg_precomputation_builder.h
 * @brief Builder for PCG precomputed data from the receiver's K-space sampling.
 *
 * The PcgPrecomputationBuilder turns the K-samples the receiver acquires
 * (one per detector pixel and wavelength) into the PSF, transfer function,
 * and preconditioner data needed by the PCG regularized reconstruction
 * operator, one dataset per wavelength.
 *
 * ### Example Usage
 * @code
 * PcgPrecomputationBuilder builder(forward_model);
 * builder.set_grid_size(64, 64)
 *        .set_pixel_size(0.001)
 *        .set_bandwidth_config(bw_config);
 * auto data = builder.build(1550e-9);
 * @endcode
 */

#include <string>
#include <vector>

#include "reconstruction/pcg_precomputation.h"

#include "bandwidth_config.pb.h"
#include "pcg_config.pb.h"
#include "pcg_precomputed_data.pb.h"

namespace aurora::forward {
class ForwardModel;
}  // namespace aurora::forward

namespace aurora::reconstruction {

/**
 * @brief Builds PcgPrecomputedDataSet from the receiver's K-space samples.
 */
class PcgPrecomputationBuilder {
  public:
    /**
     * @brief Construct builder with the receiver's forward model.
     *
     * @param forward_model ForwardModel providing K-sample lookups (must outlive builder)
     */
    explicit PcgPrecomputationBuilder(const forward::ForwardModel& forward_model);

    PcgPrecomputationBuilder& set_grid_size(uint32_t size_x, uint32_t size_y);
    PcgPrecomputationBuilder& set_pixel_size(double pixel_size_m);
    PcgPrecomputationBuilder& set_pixel_spacing(double spacing_x_m, double spacing_y_m);
    PcgPrecomputationBuilder& set_origin(double origin_x_m, double origin_y_m);
    PcgPrecomputationBuilder& set_grid(const GridSpec& grid);
    PcgPrecomputationBuilder& set_regularizer(RegularizerType type);
    PcgPrecomputationBuilder& set_preconditioner(PreconditionerType type);

    /// Receiver geometry identifier recorded in the output.
    PcgPrecomputationBuilder& set_geometry_id(const std::string& id);
    PcgPrecomputationBuilder& set_calibration_id(const std::string& id);

    /// Required before build: window type and quadrature order for the operator.
    PcgPrecomputationBuilder& set_bandwidth_config(const forward::BandwidthConfig& config);

    /**
     * @brief Build precomputed data for one wavelength.
     * @throws std::invalid_argument if bandwidth config is not set
     */
    PcgPrecomputedDataSet build(double wavelength_m);

    /**
     * @brief Build precomputed data for several wavelengths.
     * @throws std::invalid_argument if wavelengths_m is empty or bandwidth config is not set
     */
    PcgPrecomputedDataSet build(const std::vector<double>& wavelengths_m);

    const GridSpec& grid() const { return grid_; }
    RegularizerType regularizer() const { return regularizer_; }
    PreconditionerType preconditioner() const { return preconditioner_; }

  private:
    const forward::ForwardModel& forward_model_;

    GridSpec grid_;
    RegularizerType regularizer_ = REGULARIZER_GRADIENT;
    PreconditionerType preconditioner_ = PRECONDITIONER_HYBRID;
    forward::BandwidthConfig bandwidth_config_;
    std::string geometry_id_ = "jit_geometry";
    std::string calibration_id_ = "jit_calibration";

    void validate_bandwidth_config() const;
};

}  // namespace aurora::reconstruction

#endif  // AURORA_RECONSTRUCTION_PCG_PRECOMPUTATION_BUILDER_H_

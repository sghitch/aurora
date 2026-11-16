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

/**
 * @file pcg_precompute.cpp
 * @brief CLI tool for offline PCG precomputation.
 *
 * Generates precomputed data (PSF, transfer function, regularizer spectrum,
 * operator diagonal) for PCG reconstruction. This data is computed once when
 * system configuration changes and reused across frames at runtime.
 *
 * Usage:
 *   aurora-pcg-precompute \
 *       --forward-config forward_config.pb \
 *       --calibration calibration.pb \
 *       --output pcg_precomputed.pb \
 *       [--grid-size 32x1] \
 *       [--pixel-size 0.001] \
 *       [--origin 0,0] \
 *       [--regularizer gradient] \
 *       [--preconditioner hybrid] \
 *       [--wavelength 905e-9] \
 *       [--verbose]
 */

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <stdexcept>

#include <Eigen/Core>

#include "reconstruction/direct_operator.h"
#include "reconstruction/pcg_precomputation.h"
#include "forward/forward_model.h"
#include "calibration_artifact.pb.h"
#include "detector_geometry.pb.h"
#include "k_space_slice.pb.h"
#include "optics_config.pb.h"
#include "pcg_config.pb.h"
#include "pcg_precomputed_data.pb.h"
#include "forward_config.pb.h"

namespace {

void print_usage(const char* program) {
    std::cerr << "Usage: " << program << " [options]\n"
              << "\n"
              << "Generate PCG precomputed data for offline use.\n"
              << "\n"
              << "Required options:\n"
              << "  --forward-config PATH    Path to ForwardConfig protobuf (detector + optics)\n"
              << "  --calibration PATH       Path to calibration artifact protobuf\n"
              << "  --output PATH            Output path for precomputed data\n"
              << "\n"
              << "Grid options:\n"
              << "  --grid-size WxH          Grid size in pixels (default: 32x1, matching the\n"
              << "                           one-dimensional K-sampling of a linear detector array)\n"
              << "  --pixel-size METERS      Pixel size in meters (default: 0.001)\n"
              << "  --origin X,Y             Grid origin in meters (default: 0,0)\n"
              << "\n"
              << "Algorithm options:\n"
              << "  --regularizer TYPE       tikhonov|gradient|laplacian (default: gradient)\n"
              << "  --preconditioner TYPE    none|jacobi|circulant|hybrid (default: hybrid)\n"
              << "\n"
              << "Other options:\n"
              << "  --wavelength METERS      Single wavelength (default: all in optics config)\n"
              << "  --verbose                Print progress information\n"
              << "  --help                   Show this help message\n";
}

struct Options {
    std::string forward_config_path;
    std::string calibration_path;
    std::string output_path;
    int grid_width = 32;
    int grid_height = 1;
    double pixel_size = 0.001;
    double origin_x = 0.0;
    double origin_y = 0.0;
    aurora::reconstruction::RegularizerType regularizer =
        aurora::reconstruction::REGULARIZER_GRADIENT;
    aurora::reconstruction::PreconditionerType preconditioner =
        aurora::reconstruction::PRECONDITIONER_HYBRID;
    double wavelength = 0.0;  // 0 = all wavelengths
    bool verbose = false;
};

aurora::reconstruction::RegularizerType parse_regularizer(const std::string& s) {
    if (s == "tikhonov") {
        return aurora::reconstruction::REGULARIZER_TIKHONOV;
    } else if (s == "gradient") {
        return aurora::reconstruction::REGULARIZER_GRADIENT;
    } else if (s == "laplacian") {
        return aurora::reconstruction::REGULARIZER_LAPLACIAN;
    }
    throw std::invalid_argument("Unknown regularizer type: " + s +
        " (expected: tikhonov, gradient, laplacian)");
}

aurora::reconstruction::PreconditionerType parse_preconditioner(const std::string& s) {
    if (s == "none") {
        return aurora::reconstruction::PRECONDITIONER_NONE;
    } else if (s == "jacobi") {
        return aurora::reconstruction::PRECONDITIONER_JACOBI;
    } else if (s == "circulant") {
        return aurora::reconstruction::PRECONDITIONER_CIRCULANT;
    } else if (s == "hybrid") {
        return aurora::reconstruction::PRECONDITIONER_HYBRID;
    }
    throw std::invalid_argument("Unknown preconditioner type: " + s +
        " (expected: none, jacobi, circulant, hybrid)");
}

bool parse_grid_size(const std::string& s, int& width, int& height) {
    size_t x_pos = s.find('x');
    if (x_pos == std::string::npos) {
        return false;
    }
    try {
        width = std::stoi(s.substr(0, x_pos));
        height = std::stoi(s.substr(x_pos + 1));
        return width > 0 && height > 0;
    } catch (...) {
        return false;
    }
}

bool parse_origin(const std::string& s, double& x, double& y) {
    size_t comma_pos = s.find(',');
    if (comma_pos == std::string::npos) {
        return false;
    }
    try {
        x = std::stod(s.substr(0, comma_pos));
        y = std::stod(s.substr(comma_pos + 1));
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_args(int argc, char* argv[], Options& opts) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--forward-config") {
            if (++i >= argc) {
                std::cerr << "Error: --forward-config requires an argument\n";
                return false;
            }
            opts.forward_config_path = argv[i];
        } else if (arg == "--calibration") {
            if (++i >= argc) {
                std::cerr << "Error: --calibration requires an argument\n";
                return false;
            }
            opts.calibration_path = argv[i];
        } else if (arg == "--output") {
            if (++i >= argc) {
                std::cerr << "Error: --output requires an argument\n";
                return false;
            }
            opts.output_path = argv[i];
        } else if (arg == "--grid-size") {
            if (++i >= argc) {
                std::cerr << "Error: --grid-size requires an argument\n";
                return false;
            }
            if (!parse_grid_size(argv[i], opts.grid_width, opts.grid_height)) {
                std::cerr << "Error: Invalid grid size format: " << argv[i]
                          << " (expected: WxH, e.g., 32x1)\n";
                return false;
            }
        } else if (arg == "--pixel-size") {
            if (++i >= argc) {
                std::cerr << "Error: --pixel-size requires an argument\n";
                return false;
            }
            try {
                opts.pixel_size = std::stod(argv[i]);
                if (opts.pixel_size <= 0) {
                    std::cerr << "Error: pixel-size must be positive\n";
                    return false;
                }
            } catch (...) {
                std::cerr << "Error: Invalid pixel-size: " << argv[i] << "\n";
                return false;
            }
        } else if (arg == "--origin") {
            if (++i >= argc) {
                std::cerr << "Error: --origin requires an argument\n";
                return false;
            }
            if (!parse_origin(argv[i], opts.origin_x, opts.origin_y)) {
                std::cerr << "Error: Invalid origin format: " << argv[i]
                          << " (expected: X,Y, e.g., 0,0)\n";
                return false;
            }
        } else if (arg == "--regularizer") {
            if (++i >= argc) {
                std::cerr << "Error: --regularizer requires an argument\n";
                return false;
            }
            try {
                opts.regularizer = parse_regularizer(argv[i]);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Error: " << e.what() << "\n";
                return false;
            }
        } else if (arg == "--preconditioner") {
            if (++i >= argc) {
                std::cerr << "Error: --preconditioner requires an argument\n";
                return false;
            }
            try {
                opts.preconditioner = parse_preconditioner(argv[i]);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Error: " << e.what() << "\n";
                return false;
            }
        } else if (arg == "--wavelength") {
            if (++i >= argc) {
                std::cerr << "Error: --wavelength requires an argument\n";
                return false;
            }
            try {
                opts.wavelength = std::stod(argv[i]);
                if (opts.wavelength <= 0) {
                    std::cerr << "Error: wavelength must be positive\n";
                    return false;
                }
            } catch (...) {
                std::cerr << "Error: Invalid wavelength: " << argv[i] << "\n";
                return false;
            }
        } else if (arg == "--verbose") {
            opts.verbose = true;
        } else {
            std::cerr << "Error: Unknown option: " << arg << "\n";
            return false;
        }
    }

    // Validate required options
    if (opts.forward_config_path.empty()) {
        std::cerr << "Error: --forward-config is required\n";
        return false;
    }
    if (opts.calibration_path.empty()) {
        std::cerr << "Error: --calibration is required\n";
        return false;
    }
    if (opts.output_path.empty()) {
        std::cerr << "Error: --output is required\n";
        return false;
    }

    return true;
}

// Load a protobuf from file
template <typename T>
T load_protobuf(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }

    T message;
    if (!message.ParseFromIstream(&file)) {
        throw std::runtime_error("Failed to parse protobuf from file: " + path);
    }

    return message;
}

// Wavelengths from the optics configuration
std::vector<double> extract_wavelengths(const aurora::forward::OpticsConfig& optics) {
    std::set<double> wavelength_set(optics.wavelengths_m().begin(),
                                    optics.wavelengths_m().end());
    return std::vector<double>(wavelength_set.begin(), wavelength_set.end());
}

// Create a KSpaceSlice (no measurement values) from the receiver's K-samples
aurora::shared::KSpaceSlice create_k_space_slice(
    const aurora::forward::ForwardModel& forward_model,
    double wavelength_m) {
    aurora::shared::KSpaceSlice slice;
    slice.set_wavelength_m(wavelength_m);
    slice.set_range_bin(0);
    for (const auto& s : forward_model.get_all_k_samples(wavelength_m)) {
        Eigen::Vector2d k = forward_model.get_effective_k(s.detector_index(), wavelength_m);
        auto* sample = slice.add_samples();
        sample->set_kx(k.x());
        sample->set_ky(k.y());
        sample->set_weight(1.0);
        sample->set_detector_index(s.detector_index());
        sample->set_kx_bandwidth(s.kx_bandwidth());
        sample->set_ky_bandwidth(s.ky_bandwidth());
    }
    return slice;
}

std::string format_size(uintmax_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB"};
    int unit_idx = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024.0 && unit_idx < 3) {
        size /= 1024.0;
        ++unit_idx;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(unit_idx == 0 ? 0 : 2) << size
        << " " << units[unit_idx];
    return oss.str();
}

std::string regularizer_name(aurora::reconstruction::RegularizerType type) {
    switch (type) {
        case aurora::reconstruction::REGULARIZER_TIKHONOV: return "tikhonov";
        case aurora::reconstruction::REGULARIZER_GRADIENT: return "gradient";
        case aurora::reconstruction::REGULARIZER_LAPLACIAN: return "laplacian";
        default: return "unknown";
    }
}

std::string preconditioner_name(aurora::reconstruction::PreconditionerType type) {
    switch (type) {
        case aurora::reconstruction::PRECONDITIONER_NONE: return "none";
        case aurora::reconstruction::PRECONDITIONER_JACOBI: return "jacobi";
        case aurora::reconstruction::PRECONDITIONER_CIRCULANT: return "circulant";
        case aurora::reconstruction::PRECONDITIONER_HYBRID: return "hybrid";
        default: return "unknown";
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc == 1) {
        print_usage(argv[0]);
        return 1;
    }

    Options opts;
    if (!parse_args(argc, argv, opts)) {
        return 1;
    }

    try {
        // Load forward configuration (detector geometry + optics)
        if (opts.verbose) {
            std::cout << "Loading forward configuration from "
                      << opts.forward_config_path << "..." << std::endl;
        }

        auto forward_config = load_protobuf<aurora::forward::ForwardConfig>(
            opts.forward_config_path);

        if (forward_config.detector().arrays_size() == 0 ||
            forward_config.detector().arrays(0).pixels_size() == 0) {
            std::cerr << "Error: forward config has no detector pixels\n";
            return 1;
        }

        aurora::forward::ForwardModel forward_model(forward_config);

        // Load calibration
        if (opts.verbose) {
            std::cout << "Loading calibration from "
                      << opts.calibration_path << "..." << std::endl;
        }

        auto calibration = load_protobuf<aurora::shared::CalibrationArtifact>(
            opts.calibration_path);

        // Geometry and calibration IDs
        std::string geometry_id = forward_config.detector().arrays(0).array_id();
        std::string calibration_id = calibration.calibration_id();

        if (geometry_id.empty()) {
            geometry_id = std::filesystem::path(opts.forward_config_path)
                .stem().string();
        }
        if (calibration_id.empty()) {
            calibration_id = std::filesystem::path(opts.calibration_path)
                .stem().string();
        }

        // Build grid spec
        aurora::reconstruction::GridSpec grid;
        grid.set_size_x(static_cast<uint32_t>(opts.grid_width));
        grid.set_size_y(static_cast<uint32_t>(opts.grid_height));
        grid.set_spacing_x_m(opts.pixel_size);
        grid.set_spacing_y_m(opts.pixel_size);
        grid.set_origin_x_m(opts.origin_x);
        grid.set_origin_y_m(opts.origin_y);

        // Determine wavelengths to process
        std::vector<double> wavelengths;
        if (opts.wavelength > 0) {
            wavelengths.push_back(opts.wavelength);
        } else {
            wavelengths = extract_wavelengths(forward_config.optics());
            if (wavelengths.empty()) {
                std::cerr << "Error: No wavelengths found in optics configuration\n";
                return 1;
            }
        }

        if (opts.verbose) {
            std::cout << "Grid: " << opts.grid_width << "x" << opts.grid_height
                      << ", pixel size: " << opts.pixel_size * 1000.0 << " mm"
                      << std::endl;
            std::cout << "Regularizer: " << regularizer_name(opts.regularizer)
                      << ", preconditioner: " << preconditioner_name(opts.preconditioner)
                      << std::endl;
            std::cout << "Processing " << wavelengths.size() << " wavelength(s)..."
                      << std::endl;
        }

        // Compute precomputed data
        aurora::reconstruction::PcgPrecomputedDataSet result;
        result.set_geometry_id(geometry_id);
        result.set_calibration_id(calibration_id);
        *result.mutable_grid() = grid;

        auto total_start = std::chrono::steady_clock::now();

        for (size_t i = 0; i < wavelengths.size(); ++i) {
            double wavelength = wavelengths[i];

            if (opts.verbose) {
                std::cout << "  Computing wavelength " << wavelength * 1e9
                          << " nm..." << std::flush;
            }

            auto start = std::chrono::steady_clock::now();

            // Build the K-space slice for this wavelength
            auto slice = create_k_space_slice(forward_model, wavelength);

            Eigen::Vector2i grid_size(opts.grid_width, opts.grid_height);
            Eigen::Vector2d grid_spacing(opts.pixel_size, opts.pixel_size);
            Eigen::Vector2d grid_origin(opts.origin_x, opts.origin_y);

            // Build operator and weights using MeasurementOperator physics
            aurora::forward::BandwidthConfig bw_config;
            bw_config.set_window_type(aurora::forward::BANDWIDTH_WINDOW_SINC);
            bw_config.set_quadrature_order(4);

            aurora::reconstruction::DirectOperator op(
                slice, forward_model,
                bw_config,
                grid_size, grid_spacing, grid_origin);
            Eigen::VectorXd weights(slice.samples_size());
            for (int s = 0; s < slice.samples_size(); ++s) {
                weights(s) = slice.samples(s).weight();
            }

            // Compute precomputed data
            auto data = aurora::reconstruction::PcgPrecomputation::compute(
                op, weights, grid, wavelength,
                opts.regularizer, opts.preconditioner,
                geometry_id, calibration_id);

            *result.add_wavelength_data() = data;

            auto end = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();

            if (opts.verbose) {
                std::cout << " done (" << std::fixed << std::setprecision(2)
                          << elapsed << "s, " << slice.samples_size() << " K-samples)"
                          << std::endl;
            }
        }

        auto total_end = std::chrono::steady_clock::now();
        double total_elapsed = std::chrono::duration<double>(
            total_end - total_start).count();

        if (result.wavelength_data_size() == 0) {
            std::cerr << "Error: No precomputed data generated\n";
            return 1;
        }

        // Save output
        if (opts.verbose) {
            std::cout << "Saving to " << opts.output_path << "..." << std::endl;
        }

        aurora::reconstruction::PcgPrecomputation::save(result, opts.output_path);

        // Get file size
        uintmax_t file_size = std::filesystem::file_size(opts.output_path);

        // Print summary
        std::cout << "Done!" << std::endl;
        std::cout << "  Wavelengths: " << result.wavelength_data_size() << std::endl;
        std::cout << "  Grid size: " << opts.grid_width << "x" << opts.grid_height
                  << std::endl;
        std::cout << "  Computation time: " << std::fixed << std::setprecision(2)
                  << total_elapsed << " seconds" << std::endl;
        std::cout << "  Output size: " << format_size(file_size) << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

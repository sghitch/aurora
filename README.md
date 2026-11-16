# Project Aurora

A modular C++20 signal-processing framework for coherent (FMCW) LiDAR with
protobuf-based data contracts. This repository is a simulation-first software
prototype: it generates physically realistic time-domain beat signals from a
scene, processes them into range-resolved measurements from an 8-pixel coherent
detector array, and runs object-level detection and scene reconstruction on
the result.

## Overview

- **Coherent 8-pixel receiver**: a linear InGaAs array at the Fourier plane of
  the receive optics. Each pixel samples one spatial frequency of the scene,
  `k_j = 2π·x_j/(f·λ)`, so a burst yields eight K-space samples per wavelength.
- **Dual-rate outputs**: a 50 Hz detection stream (range, azimuth,
  velocity per object) and a 10-20 Hz reconstruction stream (azimuth profiles
  per range bin), both computed from the same bursts.
- **Phase-gradient localization**: K-space phase encodes target position
  (φ = -k·r), so azimuth is recovered from the phase gradient across the
  detector pixels without forming an image.
- **Dual wavelength**: 905 nm and 1550 nm time-interleaved on one detector
  array, giving a spectral ratio per detection.
- **Doppler**: up/down chirp pairs give radial velocity, either by phase
  comparison (slow targets) or by pairing up- and down-chirp range peaks.

## Pipeline

```
Scene ──▶ FMCW simulation ──▶ Range DSP ──▶ MeasurementTensor ──┬──▶ Detection (50 Hz)
            (time-domain ADC)   (range FFT)   Y[λ][detector][bin] │
                                                                 └──▶ Reconstruction (10-20 Hz)
```

Every module consumes and produces protobuf messages defined under `proto/`;
there are no shadow structs. The `MeasurementTensor` is the central contract.

## Project Structure

```
aurora/
├── proto/                      # Protobuf definitions (authoritative data contracts)
│   ├── shared/                 # Complex, Vector3, Axis, MeasurementTensor, KSpaceSlice, ...
│   ├── forward/                # Detector geometry, optics, K-space geometry
│   ├── acquisition/            # Sequence plans and schedules
│   ├── simulation/             # Scene, noise, ADC, simulator configuration
│   ├── dsp/                    # Range cube, Doppler, DSP configuration
│   ├── detection/              # Detections, detection configuration
│   └── reconstruction/         # Reconstruction configuration and outputs
├── forward/                    # K-space geometry, forward model, measurement operator
├── acquisition/                # Chirp timing and wavelength interleaving
├── simulation/                 # FMCW beat-signal generation, scene and target models
├── dsp/                        # IQ demodulation, range FFT, Doppler processing
├── calibration/                # Calibration artifact handling (placeholder)
├── detection/                  # CFAR, phase-gradient angle estimation, Doppler extraction
├── reconstruction/             # Gridding / NUFFT / direct adjoint / PCG imaging
├── tools/                      # aurora-pcg-precompute CLI
├── test/                       # Integration tests and benchmarks
└── docs/examples/              # End-to-end examples
```

## Modules

| Module | Description |
|--------|-------------|
| **forward** | `KSpaceGeometry` derives each pixel's K-position and acceptance window from Fourier optics; `ForwardModel` serves those K-samples and extracts K-space slices from measurements; `MeasurementOperator` models the K-space measurement window shared by simulation and reconstruction |
| **acquisition** | `SequencePlanner` turns a `SequenceSchedule` into per-burst chirp timing with wavelength interleaving |
| **simulation** | `FmcwSimulator` produces ADC samples for point, plane and mesh targets with speckle, occlusion, pupil vignetting and a detector noise model |
| **dsp** | `RangeDopplerProcessor` performs IQ demodulation, windowed range FFT and up/down-chirp Doppler processing, producing the `MeasurementTensor` |
| **detection** | `Detector` finds active range bins (fixed threshold or CFAR), estimates azimuth with `AngleEstimator`, and extracts radial velocity |
| **reconstruction** | `Reconstructor` images each range bin from its K-space slice using gridding, NUFFT adjoint, direct adjoint, fast online or PCG-regularized operators |

### Detection

```cpp
#include "detection/detector.h"

DetectionConfig config;
config.set_noise_floor_db(0.0);
config.mutable_cfar()->set_mode(CFAR_MODE_CA);
// ... guard/reference cells, threshold factor

Detector detector(config, forward_config, sequence_schedule);
DetectionFrame frame = detector.detect(measurements);

for (const auto& det : frame.detections()) {
    std::cout << "Range: " << det.range_m() << " m, "
              << "Azimuth: " << det.azimuth_rad() << " rad\n";
}
```

`AngleEstimator` performs a matched-filter search over the field of view
(robust to 2π phase wrapping) followed by a weighted least-squares fit of the
linear phase model `φ_j = -k_j·r + φ_0` across the detector pixels at a range
bin. A uniformly spaced array makes the response periodic in angle; the
unambiguous span is `2π/(f·Δk)` and the alias nearest boresight is reported.

### Reconstruction

```cpp
#include "reconstruction/reconstructor.h"

ReconstructionConfig config;
config.set_algorithm(ALGORITHM_DIRECT_ADJOINT);
config.set_grid_size_x(32);
config.set_grid_size_y(1);
config.set_pixel_size_m(20e-6);
config.mutable_bandwidth_config()->set_window_type(BANDWIDTH_WINDOW_SINC);
config.mutable_bandwidth_config()->set_quadrature_order(11);
config.mutable_direct_adjoint_config()->set_apodization(DirectAdjointConfig::APOD_NONE);

Reconstructor reconstructor(config, forward_model);
ReconstructedScene scene = reconstructor.reconstruct(measurements);  // one MeasurementTensor
```

`ReconstructionConfig` selects the algorithm, output grid and bandwidth window.
Algorithms: `ALGORITHM_GRIDDING_IFFT` (fast preview), `ALGORITHM_NUFFT_ADJOINT`,
`ALGORITHM_DIRECT_ADJOINT` (reference), `ALGORITHM_FAST_ONLINE`, and
`ALGORITHM_PCG_REGULARIZED` (with optional offline precomputation via
`aurora-pcg-precompute`). The operators are general K-space imaging operators;
with the 8-pixel linear array they produce azimuth profiles.

## Requirements

- CMake 3.20+
- C++20 compiler (GCC 12+ or Clang 15+)
- Protocol Buffers
- Eigen3
- Google Test, Google Benchmark and KissFFT are fetched automatically

Optional: Doxygen for API documentation. A dev container with all
dependencies is provided under `.devcontainer/`.

## Building

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

## Testing

```bash
cd build
ctest --output-on-failure

# Run a subset
ctest -R DetectionIntegration --output-on-failure
ctest -R SignalChain --output-on-failure
ctest -R reconstruction --output-on-failure
```

## Examples

| Example | Description |
|---------|-------------|
| `docs/examples/detection_example.cpp` | Scene → simulation → DSP → `Detector`: range, azimuth and velocity of simulated targets, plus up/down peak pairing for fast targets |
| `docs/examples/reconstruction_example.cpp` | Scene → simulation → DSP → `Reconstructor`: azimuth profile of the target range bin |

```bash
cd build
cmake --build . --target detection_example reconstruction_example
./docs/examples/detection_example
./docs/examples/reconstruction_example
```

## Documentation

```bash
cd build
cmake --build . --target docs
# Open docs/html/index.html
```

## Performance Targets

| Benchmark | Target |
|-----------|--------|
| DSP latency (one burst, 8 channels) | < 5 ms |
| Detection latency | < 2 ms |
| Reconstruction latency (azimuth profile) | < 50 ms |
| Memory footprint | < 500 MB |

## License

TBD

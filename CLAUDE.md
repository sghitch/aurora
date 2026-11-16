# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

**Project:** Aurora coherent LiDAR simulation framework
**Codebase:** `aurora/` — simulation-first software prototype
**Language:** C++20, Protocol Buffers for data contracts
**Build:** CMake 3.20+, no copyleft dependencies
**FFT:** KissFFT (not FFTW — licensing)

## Build Commands

```bash
# Configure and build
mkdir build && cd build
cmake ..
cmake --build .

# Run all tests (using preset - works on all platforms)
ctest --preset default

# Run all tests (alternative for single-config generators like Make/Ninja)
cd build && ctest --output-on-failure

# Run a single test by name (regex)
ctest --preset default -R <regex>

# Run a specific module's tests
ctest --preset default -R forward

# Rebuild after changes
cd build && cmake --build .
```

---

## What This System Does

Aurora simulates a coherent FMCW LiDAR with an 8-pixel detector array at the
Fourier plane of the receive optics. Each pixel samples one spatial frequency
(K-vector) of the scene, `k_j(λ) = 2π · x_j / (f · λ)`, so every burst yields
eight K-space samples per wavelength. The detector geometry and optics fully
determine the K-space sampling.

**Core relation:** K-space phase encodes target position, `φ = -k · r`.
The phase gradient across detector pixels gives fine azimuth without image
reconstruction.

**Dual-rate architecture:**
- **Detection stream (50 Hz):** object-level outputs (range, azimuth, velocity, spectral ratio) via phase-gradient fitting
- **Reconstruction stream (10-20 Hz):** azimuth profiles per range bin from the K-space samples

---

## Architecture Principles

### Single Physical Pathway

```
Scene → FMCW Time-Domain Signals → DSP → MeasurementTensor → Detection / Reconstruction
```

**No shortcuts to K-space.** Every module consumes or produces well-defined protobuf contracts.

### Protocol Buffers Are Authoritative

- All data structures defined in `proto/` directories
- No shadow C++ structs duplicating protobuf fields
- No derived state duplication (e.g., no `num_chirps` alongside `repeated ChirpPlan`)
- Work directly with protobuf-generated classes
- **Never assume default values** — always check `proto/` files for field definitions and existing defaults before implementing config handling

### Module Boundaries

| Module | Responsibility | Does NOT Do |
|--------|---------------|-------------|
| **Forward** | Detector K geometry, K-sample lookup, measurement windows | Signal simulation, DSP |
| **Acquisition** | Chirp timing, wavelength interleaving | Optics, signal processing |
| **Simulation** | FMCW time-domain signal generation | Direct K-space output |
| **DSP** | Range FFT, Doppler processing | Scene awareness, reconstruction |
| **Detection** | Phase gradient, CFAR, Doppler extraction | Image reconstruction |
| **Reconstruction** | Gridding / NUFFT / PCG imaging | Object-level detection |

---

## Key Data Structures

### MeasurementTensor (`shared/measurement_tensor.proto`)

The authoritative post-DSP representation of one burst. Conceptually: `Y[λ][detector][range_bin] : complex`

```protobuf
message MeasurementTensor {
    Axis range_axis = 1;
    optional Axis doppler_axis = 2;
    repeated WavelengthChannel data = 3;            // per-detector k_vector + range data
    repeated WavelengthChannel up_chirp_data = 6;   // optional, for Doppler
    repeated WavelengthChannel down_chirp_data = 7; // optional, for Doppler
    uint64 burst_id = 4;
    double timestamp_s = 5;
}
```

**This is what Detection and Reconstruction consume.** Everything upstream produces it.

### DetectorKSample (`forward/k_space_geometry.proto`)

What one detector pixel samples at one wavelength: `kx_eff`, `ky_eff` (the
geometric K-position) and the K-space acceptance bandwidth from the pixel
size. `ForwardModel::get_k_sample()` computes it from geometry;
`get_effective_k()` adds the LO tilt offset to give scene K-space coordinates.

### CalibrationArtifact (`shared/calibration_artifact.proto`)

Immutable calibration state. Applied to convert raw measurements to calibrated K-space.

---

## Critical Physics

### Phase Gradient Localization

For a point target at **r₀**, the K-space phase is:

```
φᵢ = -kᵢ · r₀
```

`AngleEstimator` runs a matched-filter search over the field of view (robust to
2π wrapping; the uniformly spaced array aliases with period `2π/(f·Δk)`, and the
alias nearest boresight is chosen) followed by a weighted linear regression:

```cpp
// Weight by magnitude, solve for position
// φ_i = -k_i · r₀ + φ_common
Eigen::VectorXd p = (A.transpose() * Aw).ldlt().solve(A.transpose() * bw);
azimuth = p(0) / focal_length;   // r = f · θ
```

### K-Vector from Geometry

```
k_j(λ) = 2π · x_j / (f · λ)
```

Where `x_j` = detector position, `f` = Fourier lens focal length, `λ` = wavelength.

### Measurement Window

Each detector pixel has one K-space window, `W(k) = H_d(k − k_d)`, centered
on the pixel's scene K-position `k_d` (detector K plus LO tilt) with a width
set by the pixel size (`bandwidth_x/y`). A sample integrates the scene spectrum
over this window, and a point target at `r` contributes with weight
`W`-filtered `exp(-i k·r)`. `MeasurementOperator::WindowParams` holds
`k_center`, `bandwidth_x`, `bandwidth_y`; `build_window_params(forward_model,
det, λ)` fills it from the `ForwardModel`. Simulation and reconstruction share
`MeasurementOperator` so both sides use the same physics.

---

## Development Practices

### Keep Changes Small and Reviewable

- **One logical change per commit** — don't mix refactoring with new features
- **< 400 lines per PR** — if larger, break into sequential PRs
- **Each commit should compile and pass tests** — no "WIP" commits in main history
- **Explain the "why" in commit messages** — the code shows "what"

### Write Modular Code

- **Single responsibility** — each class/function does one thing well
- **Explicit dependencies** — pass dependencies via constructor, not globals
- **No God objects** — if a class has > 7 methods, consider splitting
- **Prefer composition over inheritance** — inherit for interface, compose for behavior
- **Keep functions short** — if > 50 lines, extract helper functions

### Design for Testability

- **Write tests first for complex logic** — especially physics calculations
- **Mock external dependencies** — file I/O, hardware, time
- **Test edge cases explicitly** — empty inputs, boundary conditions, error paths
- **Group related assertions** — test one logical behavior per test, with multiple asserts as needed

### Incremental Development

- **Get something working first, then optimize** — correctness before performance
- **Prototype in isolation** — new algorithms start as standalone tests
- **Validate against known results** — compare to analytical solutions or reference implementations

### Code Review Checklist

Before submitting:
- [ ] All tests pass
- [ ] No compiler warnings
- [ ] New public APIs have documentation
- [ ] Complex algorithms have explanatory comments
- [ ] No hardcoded magic numbers (use named constants)
- [ ] 4-space indentation throughout

---

## Code Standards

### General

- **C++20** with concepts and ranges where appropriate
- **4-space indentation** — no tabs
- **Metric units everywhere** — meters, seconds, radians
- **All numeric values are `double`** unless hardware-constrained
- **Deterministic RNG** via explicit seeding

### Naming

- Protobuf messages: `PascalCase`
- C++ classes: `PascalCase`
- C++ methods: `snake_case`
- C++ member variables: `snake_case_` (trailing underscore)

### No Copyleft Libraries

| Use | Don't Use |
|-----|-----------|
| KissFFT | FFTW (GPL) |
| Eigen (MPL-2) | GSL (GPL) |
| spdlog (MIT) | Any GPL/LGPL dependency |

---

## Testing Requirements

### Required Physics Invariants

1. **Adjoint consistency:** `⟨Ax, y⟩ ≈ ⟨x, A*y⟩`
2. **Phase gradient accuracy:** Error < 3× Cramer-Rao bound
3. **Wavelength K-scaling:** `k ∝ 1/λ`
4. **DSP integrity:** Range peak at correct bin
5. **Detection robustness:** Results independent of reconstruction settings
6. **Simulation determinism:** Same seed → identical output

### Performance Targets

| Benchmark | Target |
|-----------|--------|
| DSP latency (one burst, 8 channels) | < 5 ms |
| Detection latency | < 2 ms |
| Reconstruction latency (azimuth profile) | < 50 ms |
| Memory footprint | < 500 MB |

---

## Common Tasks

### Adding a New Protobuf

1. Create `.proto` file in appropriate `proto/` subdirectory
2. Add to `CMakeLists.txt` if needed (glob should catch it)
3. Rebuild to generate C++ headers
4. Include generated header: `#include "message.pb.h"`

### Adding a New Module

1. Create directory structure: `include/`, `src/`, `test/`
2. Add `CMakeLists.txt` with library target
3. Link to `aurora_proto` and required dependencies
4. Add subdirectory in top-level `CMakeLists.txt`

### Building Test Geometry

Use `test/common/test_forward_model_builder.h`: `create_linear_array()` builds
a detector array and `TestForwardModelBuilder::build(detector, optics)` the
`ForwardModel`.

### Debugging K-Space Issues

1. Check `ForwardModel::get_all_k_samples(λ)` — do the K-vectors match `2π·x_j/(f·λ)` and are the pixel widths set?
2. Verify calibration is applied
3. Dump `MeasurementTensor` phases and check gradient
4. Compare magnitude profiles across pixels

---

## Reference Geometry

| Component | Spec | Notes |
|-----------|------|-------|
| Detector | 8-pixel linear InGaAs array | 0.5 mm pitch, 150 MHz BW |
| ADC | 8 channels | 125 MS/s |
| Wavelengths | 905 nm + 1550 nm | Both on the single array, time-interleaved |

---

## Key Files to Know

```
aurora/
├── proto/
│   ├── shared/
│   │   ├── measurement_tensor.proto  ← Central data structure
│   │   ├── k_space_slice.proto       ← Reconstruction input
│   │   └── calibration_artifact.proto
│   └── forward/
│       ├── k_space_geometry.proto    ← DetectorKSample
│       └── forward_config.proto
├── forward/
│   └── include/forward/
│       ├── forward_model.h           ← K-sample service, slice extraction
│       ├── k_space_geometry.h        ← Detector K positions
│       └── measurement_operator.h    ← Shared window physics
├── detection/
│   └── include/detection/
│       ├── detector.h                ← Detection pipeline
│       └── angle_estimator.h         ← Phase gradient fitting
└── reconstruction/
    └── include/reconstruction/
        └── reconstructor.h           ← Imaging facade
```

---

## Gotchas

1. **Phase wrapping** — K-space phases wrap at ±π. The uniformly spaced array makes the angle response periodic with span `2π/(f·Δk)` (about ±1.5 mrad at the reference geometry); `AngleEstimator` reports the alias nearest boresight.

2. **Wavelength interleaving** — 905nm and 1550nm share one detector array, time-interleaved. K-vectors scale as 1/λ, so each wavelength is its own `WavelengthChannel`.

3. **Calibration is per-(wavelength, detector)** — K-vector offsets and phase corrections vary across both dimensions.

4. **Pixel size is required** — `ForwardModel::extract_kspace_slice()` throws if a pixel has zero width or height (its K-space bandwidth would be zero).

5. **Doppler at optical wavelengths** — a few m/s shifts the beat frequency by MHz. The Detector's phase-based Doppler is unambiguous only for `|v| < λ/(4T)`; faster targets need `RangeDopplerProcessor::process_doppler_peaks()` (up/down peak pairing).

---

## Formulas Quick Reference

| Quantity | Formula |
|----------|---------|
| Range resolution | `ΔR = c / (2B)` |
| Range from beat freq | `R = c · f_beat · T_chirp / (2B)` |
| K-vector | `k = 2π · x_det / (f · λ)` |
| Phase encoding | `φ = -k · r₀` |
| Velocity from Doppler | `v = λ · f_doppler / 2` |

---

## Error Handling & Defensive Programming

### Fail Fast, Fail Loud

- **Validate inputs at module boundaries** — check protobuf fields before processing
- **Use assertions for invariants** — `assert()` for things that should never happen
- **Throw exceptions for recoverable errors** — invalid user input, missing files
- **Log warnings for suspicious but valid states** — unexpected but handleable conditions

### Handle Edge Cases Explicitly

```cpp
// Good: explicit handling
if (measurements.data().empty()) {
    return DetectionFrame{};  // No detections for empty input
}

// Bad: implicit crash
auto first = measurements.data(0);  // Crashes if empty
```

### Resource Management

- **RAII everywhere** — no manual `new`/`delete`
- **Use smart pointers** — `std::unique_ptr` for ownership, `std::shared_ptr` only when truly shared
- **Prefer stack allocation** — heap only when necessary
- **Clean up in destructors** — no explicit cleanup methods

---

## Debugging Tips

### Common Failure Modes

| Symptom | Likely Cause |
|---------|--------------|
| All zeros in output | Missing calibration, wrong array indexing |
| NaN/Inf values | Division by zero, log of negative |
| Phase gradient fails | Too few K-samples, degenerate (identical) K-vectors |
| Wrong range | Chirp parameters inconsistent with FFT size |
| Zero signal with SINC window | Quadrature aliasing: `quadrature_order=4` with SINC lands on zeros. Use odd orders (11) or GAUSSIAN |
| Aliased azimuth | Target outside the array's unambiguous span `2π/(f·Δk)` |

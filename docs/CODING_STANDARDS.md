# Aurora Coding Standards

This document defines the C++20 coding standards and engineering conventions for the Aurora codebase. All contributors must follow these guidelines to ensure consistency and maintainability.

---

## Table of Contents

1. [Language Standard](#language-standard)
2. [Naming Conventions](#naming-conventions)
3. [File Organization](#file-organization)
4. [Formatting](#formatting)
5. [Engineering Principles](#engineering-principles)
6. [Error Handling](#error-handling)
7. [Documentation](#documentation)
8. [Dependencies](#dependencies)
9. [Examples](#examples)

---

## Language Standard

### C++20

Aurora uses **C++20** with modern features where they improve clarity:

```cpp
// Concepts for template constraints
template <std::floating_point T>
T normalize_phase(T phase) {
    return std::fmod(phase + std::numbers::pi_v<T>,
                     2 * std::numbers::pi_v<T>) - std::numbers::pi_v<T>;
}

// Designated initializers
DetectorConfig config{
    .num_pixels = 8,
    .pixel_pitch_m = 0.0005,
    .bandwidth_hz = 150e6
};

// Ranges for cleaner algorithms
auto valid_peaks = measurements
    | std::views::filter([](auto& m) { return m.snr() > threshold; })
    | std::views::take(max_detections);

// std::span for non-owning array views
void process_samples(std::span<const double> samples);
```

### Recommended C++20 Features

| Feature | Use Case |
|---------|----------|
| Concepts | Template constraints, clearer error messages |
| Ranges | Data transformations, pipelines |
| `std::span` | Non-owning views of contiguous data |
| `std::numbers` | Mathematical constants (`pi_v`, `e_v`) |
| Designated initializers | Struct initialization with named fields |
| `[[nodiscard]]` | Functions where ignoring return value is likely a bug |
| `[[likely]]`/`[[unlikely]]` | Branch prediction hints in hot paths |

---

## Naming Conventions

### Summary

| Element | Style | Example |
|---------|-------|---------|
| Classes/Structs | `PascalCase` | `MeasurementTensor` |
| Functions/Methods | `snake_case` | `compute_phase_gradient()` |
| Variables | `snake_case` | `range_bin_count` |
| Member Variables | `snake_case_` | `detector_count_` |
| Constants | `kPascalCase` | `kSpeedOfLight` |
| Macros | `UPPER_SNAKE_CASE` | `AURORA_ASSERT` |
| Namespaces | `snake_case` | `aurora::forward` |
| Template Parameters | `PascalCase` | `template <typename ValueType>` |
| Files | `snake_case` | `forward_model.h` |
| Protobuf Messages | `PascalCase` | `message CalibrationArtifact` |
| Protobuf Fields | `snake_case` | `double wavelength_m = 1;` |

### Detailed Guidelines

```cpp
namespace aurora::detection {

// Constants use kPascalCase
constexpr double kSpeedOfLight = 299792458.0;  // m/s
constexpr int kDefaultDetectorCount = 8;

// Classes use PascalCase
class AngleEstimator {
public:
    // Methods use snake_case
    std::optional<Position> fit_position(
        std::span<const KVector> k_vectors,
        std::span<const double> phases) const;

    // Accessors match the field name without trailing underscore
    int detector_count() const { return detector_count_; }

private:
    // Member variables use snake_case with trailing underscore
    int detector_count_;
    double wavelength_m_;
    std::vector<double> weights_;
};

}  // namespace aurora::detection
```

---

## File Organization

### Directory Structure

```
module/
├── include/module/      # Public headers
│   ├── module.h         # Main module header
│   └── types.h          # Module-specific types
├── src/                 # Implementation files
│   ├── module.cpp
│   └── internal.h       # Private headers (if needed)
├── test/                # Unit tests
│   ├── CMakeLists.txt
│   └── module_test.cpp
└── CMakeLists.txt
```

### Header Organization

```cpp
// file: include/forward/forward_model.h
#ifndef AURORA_FORWARD_FORWARD_MODEL_H_
#define AURORA_FORWARD_FORWARD_MODEL_H_

// 1. Related header (for .cpp files)
// 2. C system headers
#include <cmath>
#include <cstdint>

// 3. C++ standard library headers
#include <memory>
#include <span>
#include <vector>

// 4. Other library headers
#include <Eigen/Dense>

// 5. Project headers
#include "forward/types.h"
#include "shared/measurement_tensor.pb.h"

namespace aurora::forward {

// Class declarations...

}  // namespace aurora::forward

#endif  // AURORA_FORWARD_FORWARD_MODEL_H_
```

### Include Guard Format

Use `#ifndef` guards of the form `AURORA_MODULE_FILENAME_H_`

```cpp
#ifndef AURORA_DETECTION_PHASE_FITTER_H_
#define AURORA_DETECTION_PHASE_FITTER_H_
// ...
#endif  // AURORA_DETECTION_PHASE_FITTER_H_
```

---

## Formatting

### Indentation and Spacing

- **4 spaces** for indentation (no tabs)
- **120 character** line limit (soft limit, prefer 100)
- Braces on same line for functions and control structures
- Single space after control keywords (`if`, `for`, `while`)

```cpp
void process_measurement(const MeasurementTensor& tensor) {
    if (tensor.data().empty()) {
        return;
    }

    for (const auto& wavelength : tensor.data()) {
        for (const auto& detector : wavelength.detectors()) {
            process_detector_channel(detector);
        }
    }
}
```

### Clang-Format

Use the project's `.clang-format` file. Format before committing:

```bash
clang-format -i src/*.cpp include/**/*.h
```

---

## Engineering Principles

### 1. No Shadow Structs

Work directly with protobuf-generated classes. Never create C++ structs that duplicate protobuf message fields.

```cpp
// BAD: Shadow struct duplicating protobuf
struct MeasurementData {
    int chirp_count;            // Duplicates repeated field size
    std::vector<double> phases; // Duplicates protobuf field
};

// GOOD: Use protobuf directly
void process(const aurora::shared::MeasurementTensor& tensor) {
    for (const auto& wavelength : tensor.data()) {
        // Access protobuf fields directly
    }
}
```

### 2. No Derived State Duplication

Don't store values that can be computed from existing data.

```cpp
// BAD: Storing derived count
class ChirpSequence {
    std::vector<Chirp> chirps_;
    int chirp_count_;  // Redundant! Use chirps_.size()
};

// GOOD: Compute when needed
class ChirpSequence {
    std::vector<Chirp> chirps_;
public:
    size_t chirp_count() const { return chirps_.size(); }
};
```

### 3. Metric Units Everywhere

All physical quantities use SI units. Include unit suffix in variable names.

```cpp
// GOOD: Clear units in names
double range_m = 150.0;           // meters
double velocity_mps = 25.0;       // meters per second
double wavelength_m = 905e-9;     // meters (905 nm)
double frequency_hz = 10e9;       // hertz
double angle_rad = 0.5;           // radians
double time_s = 0.001;            // seconds
double chirp_rate_hzps = 1e12;    // Hz per second

// BAD: Ambiguous units
double range = 150.0;      // meters? feet? kilometers?
double wavelength = 905;   // nm? um? m?
```

### 4. Double Precision by Default

Use `double` for all numeric computations unless hardware-constrained.

```cpp
// GOOD: Double precision
double phase = compute_phase(k_vector, position);
std::vector<double> samples(1024);

// ACCEPTABLE: Hardware interface requires specific type
int16_t adc_sample = read_adc();  // ADC returns 16-bit
uint8_t packed_flags[1024];       // Hardware register is 1 bit per entry
```

### 5. Deterministic RNG

Always seed random number generators explicitly for reproducibility.

```cpp
// GOOD: Explicit seed for reproducibility
std::mt19937_64 rng(config.random_seed());

// GOOD: Document when using random seed
uint64_t seed = config.has_random_seed()
    ? config.random_seed()
    : std::random_device{}();
spdlog::info("Using RNG seed: {}", seed);
std::mt19937_64 rng(seed);

// BAD: Non-deterministic
std::mt19937_64 rng(std::random_device{}());  // Can't reproduce results
```

### 6. RAII Resource Management

Use RAII for all resource management. No manual `new`/`delete`.

```cpp
// GOOD: Smart pointers and RAII
auto processor = std::make_unique<SignalProcessor>(config);
std::vector<double> buffer(size);
std::ifstream file(path);

// BAD: Manual memory management
SignalProcessor* processor = new SignalProcessor(config);
// ... code that might throw ...
delete processor;  // Might leak if exception thrown
```

---

## Error Handling

### Validation at Boundaries

Validate inputs at module boundaries, trust internal data.

```cpp
// Public API: validate inputs
std::optional<DetectionResult> detect(const MeasurementTensor& tensor) {
    if (tensor.data().empty()) {
        spdlog::warn("Empty measurement tensor");
        return std::nullopt;
    }
    if (!tensor.has_range_axis()) {
        throw std::invalid_argument("MeasurementTensor missing range_axis");
    }

    // Internal processing can trust data is valid
    return process_validated(tensor);
}
```

### Use std::optional for "No Result"

```cpp
// Return nullopt when no valid result exists
std::optional<Position> fit_position(std::span<const Phase> phases) {
    if (phases.size() < kMinimumPixels) {
        return std::nullopt;
    }
    // ... fitting logic ...
    return Position{x, y};
}
```

### Use Exceptions for Unrecoverable Errors

```cpp
// Throw for programming errors or unrecoverable states
void set_wavelength(double wavelength_m) {
    if (wavelength_m <= 0) {
        throw std::invalid_argument(
            fmt::format("Wavelength must be positive, got {}", wavelength_m));
    }
    wavelength_m_ = wavelength_m;
}
```

### Assertions for Invariants

```cpp
// Assert conditions that should never fail if code is correct
void process_pixel(int pixel_index) {
    assert(pixel_index >= 0 && pixel_index < kDetectorCount);
    // ...
}
```

---

## Documentation

### Public API Documentation

Document all public classes and functions with Doxygen-style comments.

```cpp
/// Fits a linear phase gradient to extract target position.
///
/// Uses weighted least-squares fitting with magnitude-based weights.
/// Phase wrapping is handled by fitting in the complex domain.
///
/// @param k_vectors K-space sample positions for each detector pixel
/// @param measurements Complex measurements (magnitude and phase)
/// @return Fitted position, or nullopt if fit fails (e.g., too few pixels)
///
/// @note Requires at least 3 valid measurements for 2D position fit
/// @see compute_k_vectors() for K-vector computation
std::optional<Position> fit_phase_gradient(
    std::span<const KVector> k_vectors,
    std::span<const std::complex<double>> measurements);
```

### Implementation Comments

Comment the "why", not the "what". Code should be self-documenting.

```cpp
// BAD: Describes what code does (obvious from reading it)
// Multiply by 2 pi
double angular_freq = 2 * std::numbers::pi * frequency_hz;

// GOOD: Explains why
// Convert to angular frequency for phasor calculation
double angular_freq = 2 * std::numbers::pi * frequency_hz;

// GOOD: Explains non-obvious physics
// Phase wraps at ±pi, so we fit in complex domain to avoid discontinuities
auto complex_phase = std::polar(magnitude, phase);
```

---

## Dependencies

### Approved Libraries

| Purpose | Library | License |
|---------|---------|---------|
| FFT | KissFFT | BSD |
| Linear Algebra | Eigen | MPL-2.0 |
| Logging | spdlog | MIT |
| Serialization | Protocol Buffers | BSD |
| Testing | Google Test | BSD |
| Formatting | fmt | MIT |

### Prohibited Libraries

| Library | Reason | Alternative |
|---------|--------|-------------|
| FFTW | GPL license | KissFFT |
| GSL | GPL license | Eigen |
| Any GPL/LGPL | License incompatible | See approved list |

---

## Examples

### Complete Module Example

```cpp
// file: include/detection/peak_finder.h
#ifndef AURORA_DETECTION_PEAK_FINDER_H_
#define AURORA_DETECTION_PEAK_FINDER_H_

#include <optional>
#include <span>
#include <vector>

namespace aurora::detection {

/// Configuration for peak detection algorithm.
struct PeakFinderConfig {
    double threshold_db = 10.0;    ///< Minimum SNR for valid peak
    int max_peaks = 16;            ///< Maximum peaks to return
    int guard_bins = 2;            ///< Guard cells for CFAR
};

/// Represents a detected peak in the range profile.
struct Peak {
    int bin_index;
    double magnitude;
    double snr_db;
};

/// Finds peaks in range-FFT magnitude data using CFAR detection.
class PeakFinder {
public:
    explicit PeakFinder(PeakFinderConfig config);

    /// Detect peaks in magnitude spectrum.
    /// @param magnitudes Range-FFT magnitude values
    /// @return Vector of detected peaks, sorted by magnitude (descending)
    [[nodiscard]] std::vector<Peak> find_peaks(
        std::span<const double> magnitudes) const;

private:
    PeakFinderConfig config_;
};

}  // namespace aurora::detection

#endif  // AURORA_DETECTION_PEAK_FINDER_H_
```

```cpp
// file: src/peak_finder.cpp
#include "detection/peak_finder.h"

#include <algorithm>
#include <numeric>

namespace aurora::detection {

PeakFinder::PeakFinder(PeakFinderConfig config)
    : config_(std::move(config)) {}

std::vector<Peak> PeakFinder::find_peaks(
    std::span<const double> magnitudes) const {

    if (magnitudes.size() < static_cast<size_t>(2 * config_.guard_bins + 1)) {
        return {};  // Not enough bins for CFAR
    }

    std::vector<Peak> peaks;
    peaks.reserve(config_.max_peaks);

    // CFAR detection with guard cells
    for (size_t i = config_.guard_bins;
         i < magnitudes.size() - config_.guard_bins; ++i) {

        double noise_floor = estimate_noise_floor(magnitudes, i);
        double snr_db = 20.0 * std::log10(magnitudes[i] / noise_floor);

        if (snr_db >= config_.threshold_db) {
            peaks.push_back({
                .bin_index = static_cast<int>(i),
                .magnitude = magnitudes[i],
                .snr_db = snr_db
            });
        }
    }

    // Sort by magnitude descending, keep top N
    std::partial_sort(
        peaks.begin(),
        peaks.begin() + std::min(peaks.size(),
                                  static_cast<size_t>(config_.max_peaks)),
        peaks.end(),
        [](const Peak& a, const Peak& b) {
            return a.magnitude > b.magnitude;
        });

    if (peaks.size() > static_cast<size_t>(config_.max_peaks)) {
        peaks.resize(config_.max_peaks);
    }

    return peaks;
}

}  // namespace aurora::detection
```

---

## Checklist

Before submitting code for review:

- [ ] Code compiles without warnings (`-Wall -Wextra`)
- [ ] All tests pass
- [ ] Public APIs have documentation
- [ ] No hardcoded magic numbers (use named constants)
- [ ] Units are explicit in variable names
- [ ] No manual memory management (use smart pointers/RAII)
- [ ] Formatted with clang-format
- [ ] No new dependencies without approval

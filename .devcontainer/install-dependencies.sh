#!/usr/bin/env bash
#
# Install Aurora build dependencies into the devcontainer image.
#
# - Eigen, Ninja, Doxygen from apt
# - Protobuf (with bundled Abseil + utf8_range) built from source, because the
#   Debian package (3.21.x) predates Abseil and ships no absl/utf8_range CMake
#   configs, which the top-level CMakeLists.txt requires. Version matches CI
#   (.github/workflows/ci.yml).
#
# GoogleTest, KissFFT and Google Benchmark are fetched by CMake (FetchContent).
#
set -euo pipefail

PROTOBUF_VERSION=${1:-"29.2"}
INSTALL_PREFIX=/usr/local

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get -y install --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
    ninja-build \
    libeigen3-dev \
    doxygen
rm -rf /var/lib/apt/lists/*

TMP_DIR=$(mktemp -d -t protobuf-XXXXXXXXXX)
cleanup() { rm -rf "${TMP_DIR}"; }
trap cleanup EXIT

echo "Building protobuf v${PROTOBUF_VERSION}..."
git clone --depth 1 --branch "v${PROTOBUF_VERSION}" --recursive --shallow-submodules \
    https://github.com/protocolbuffers/protobuf.git "${TMP_DIR}/src"

cmake -S "${TMP_DIR}/src" -B "${TMP_DIR}/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -Dprotobuf_BUILD_TESTS=OFF \
    -Dprotobuf_BUILD_CONFORMANCE=OFF \
    -Dprotobuf_ABSL_PROVIDER=module \
    -Dutf8_range_ENABLE_TESTS=OFF \
    -DABSL_PROPAGATE_CXX_STD=ON

cmake --build "${TMP_DIR}/build" -j "$(nproc)"
cmake --install "${TMP_DIR}/build"
ldconfig

echo "protoc: $(protoc --version)"

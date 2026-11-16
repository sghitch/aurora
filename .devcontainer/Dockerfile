FROM mcr.microsoft.com/devcontainers/cpp:3-debian13

ARG REINSTALL_CMAKE_VERSION_FROM_SOURCE="none"
ARG PROTOBUF_VERSION="29.2"

# Optionally reinstall a specific CMake version from source
COPY ./reinstall-cmake.sh /tmp/

RUN if [ "${REINSTALL_CMAKE_VERSION_FROM_SOURCE}" != "none" ]; then \
        chmod +x /tmp/reinstall-cmake.sh && /tmp/reinstall-cmake.sh ${REINSTALL_CMAKE_VERSION_FROM_SOURCE}; \
    fi \
    && rm -f /tmp/reinstall-cmake.sh

# Aurora build dependencies (Eigen, Ninja, protobuf + abseil from source)
COPY ./install-dependencies.sh /tmp/

RUN chmod +x /tmp/install-dependencies.sh \
    && /tmp/install-dependencies.sh ${PROTOBUF_VERSION} \
    && rm -f /tmp/install-dependencies.sh

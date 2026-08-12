set(GGML_XDNA_XRT_ROOT "$ENV{XILINX_XRT}" CACHE PATH "ggml-xdna: XRT installation root")

find_package(XRT CONFIG REQUIRED
    HINTS "${GGML_XDNA_XRT_ROOT}/share/cmake/XRT"
)

if (NOT TARGET XRT::xrt_coreutil)
    message(FATAL_ERROR
        "ggml-xdna requires an XRT CMake package that exports XRT::xrt_coreutil. "
        "Source the XRT setup script or set GGML_XDNA_XRT_ROOT to the XRT prefix.")
endif()

ggml_add_backend_library(ggml-xdna
    ggml-xdna.cpp
    xdna-artifact.cpp
    xdna-artifact.h
    xdna-lazy-cache.h
    xdna-kernel-registry.cpp
    xdna-kernel-registry.h
    xdna-runtime.cpp
    xdna-runtime.h
    ../../include/ggml-xdna.h
)

target_compile_features(ggml-xdna PRIVATE cxx_std_17)
target_link_libraries(ggml-xdna PRIVATE XRT::xrt_coreutil)

if (GGML_XDNA_XRT_ROOT)
    set_property(TARGET ggml-xdna APPEND PROPERTY BUILD_RPATH "${GGML_XDNA_XRT_ROOT}/lib")
endif()

set(GGML_XDNA_XRT_ROOT "$ENV{XILINX_XRT}" CACHE PATH "ggml-xdna: XRT installation root")

find_path(GGML_XDNA_XRT_INCLUDE_DIR
    NAMES xrt/xrt_device.h xrt/experimental/xrt_system.h
    HINTS ${GGML_XDNA_XRT_ROOT}
    PATH_SUFFIXES include
)

find_library(GGML_XDNA_XRT_COREUTIL_LIBRARY
    NAMES xrt_coreutil
    HINTS ${GGML_XDNA_XRT_ROOT}
    PATH_SUFFIXES lib lib64
)

if (NOT GGML_XDNA_XRT_INCLUDE_DIR OR NOT GGML_XDNA_XRT_COREUTIL_LIBRARY)
    message(FATAL_ERROR
        "ggml-xdna requires an XRT installation. Source /opt/xilinx/xrt/setup.sh "
        "or set GGML_XDNA_XRT_ROOT to the XRT prefix.")
endif()

ggml_add_backend_library(ggml-xdna
    ggml-xdna.cpp
    xdna-runtime.cpp
    xdna-runtime.h
    ../../include/ggml-xdna.h
)

target_compile_features(ggml-xdna PRIVATE cxx_std_17)
target_include_directories(ggml-xdna SYSTEM PRIVATE ${GGML_XDNA_XRT_INCLUDE_DIR})
target_link_libraries(ggml-xdna PRIVATE ${GGML_XDNA_XRT_COREUTIL_LIBRARY})

get_filename_component(GGML_XDNA_XRT_LIBRARY_DIR "${GGML_XDNA_XRT_COREUTIL_LIBRARY}" DIRECTORY)
set_property(TARGET ggml-xdna APPEND PROPERTY BUILD_RPATH "${GGML_XDNA_XRT_LIBRARY_DIR}")

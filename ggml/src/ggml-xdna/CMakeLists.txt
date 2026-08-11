set(GGML_XDNA_XRT_ROOT "$ENV{XILINX_XRT}" CACHE PATH "ggml-xdna: XRT installation root")

find_package(XRT CONFIG QUIET
    HINTS "${GGML_XDNA_XRT_ROOT}/share/cmake/XRT"
)

if (TARGET XRT::xrt_coreutil)
    set(GGML_XDNA_XRT_TARGET XRT::xrt_coreutil)
else()
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
            "ggml-xdna requires XRT development files. Source the XRT setup script "
            "or set GGML_XDNA_XRT_ROOT to the XRT prefix.")
    endif()
endif()

ggml_add_backend_library(ggml-xdna
    ggml-xdna.cpp
    xdna-runtime.cpp
    xdna-runtime.h
    ../../include/ggml-xdna.h
)

target_compile_features(ggml-xdna PRIVATE cxx_std_17)
if (TARGET XRT::xrt_coreutil)
    target_link_libraries(ggml-xdna PRIVATE ${GGML_XDNA_XRT_TARGET})
else()
    target_include_directories(ggml-xdna SYSTEM PRIVATE ${GGML_XDNA_XRT_INCLUDE_DIR})
    target_link_libraries(ggml-xdna PRIVATE ${GGML_XDNA_XRT_COREUTIL_LIBRARY})
endif()

if (GGML_XDNA_XRT_ROOT)
    set_property(TARGET ggml-xdna APPEND PROPERTY BUILD_RPATH "${GGML_XDNA_XRT_ROOT}/lib")
endif()

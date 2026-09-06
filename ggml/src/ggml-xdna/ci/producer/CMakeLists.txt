cmake_minimum_required(VERSION 3.16)
project(xdna_package_producer LANGUAGES C CXX ASM)

if (NOT IS_DIRECTORY "${LLAMA_SOURCE}/ggml")
    message(FATAL_ERROR "Set LLAMA_SOURCE to the llama.cpp checkout")
endif()
add_subdirectory("${LLAMA_SOURCE}/ggml" ggml)

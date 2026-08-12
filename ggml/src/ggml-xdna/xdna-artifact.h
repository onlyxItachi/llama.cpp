#pragma once

#include "xdna-kernel-registry.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ggml_xdna {
namespace detail {

struct artifact_contents {
    std::vector<char> xclbin_data;
    std::vector<uint32_t> instructions;
};

artifact_contents read_artifact_bundle(const std::string & path, const xdna_kernel_variant & variant);

} // namespace detail
} // namespace ggml_xdna

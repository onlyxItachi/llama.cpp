#pragma once

#include "ggml-xdna.h"
#include "xdna-kernel-registry.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_tensor;

namespace ggml_xdna {

struct device_info {
    unsigned int index = 0;
    std::string name;
    std::string bdf;
    device_architecture arch = device_architecture::unknown;
    std::string architecture;
};

struct kernel_artifact_configuration {
    const xdna_kernel_variant * variant = nullptr;
    std::string artifact_path;
    std::vector<char> xclbin_data;
    std::vector<uint32_t> instructions;
};

// The validated artifact inventory for one physical device.  Registry order
// is selector priority and each entry owns the payload needed to construct one
// persistent XRT kernel state.
struct kernel_configuration {
    bool available = false;
    std::vector<kernel_artifact_configuration> artifacts;
    std::string status;
};

std::vector<device_info> discover_devices(std::string * error = nullptr) noexcept;
kernel_configuration probe_kernel_configuration(const device_info & info) noexcept;
const kernel_artifact_configuration * select_kernel_configuration(
        const kernel_configuration & configuration,
        const xdna_problem & problem) noexcept;

class runtime {
public:
    runtime(const device_info & info, const kernel_configuration & configuration);
    ~runtime();

    runtime(const runtime &) = delete;
    runtime & operator=(const runtime &) = delete;

    const device_info & info() const noexcept;
    bool kernel_available() const noexcept;
    const std::string & kernel_status() const noexcept;

    bool supports_op(const ggml_tensor * op) const noexcept;
    int compute(ggml_tensor * op) noexcept;

    void get_stats(ggml_backend_xdna_stats * stats) const noexcept;
    void get_stats_v2(ggml_backend_xdna_stats_v2 * stats) const noexcept;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

} // namespace ggml_xdna

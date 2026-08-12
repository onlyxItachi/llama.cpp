#pragma once

#include "ggml-xdna.h"
#include "xdna-lazy-cache.h"
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

struct readonly_userptr_capability {
    bool supported = false;
    std::string status;
};

struct kernel_artifact_configuration {
    const xdna_kernel_variant * variant = nullptr;
    std::string artifact_path;
    std::vector<char> xclbin_data;
    std::vector<uint32_t> instructions;
};

// The validated artifact inventory for one physical device. Registry order is
// selector priority and each entry owns one host-validated payload.
struct kernel_configuration {
    bool available = false;
    std::vector<kernel_artifact_configuration> artifacts;
    std::string status;
};

std::vector<device_info> discover_devices(std::string * error = nullptr) noexcept;
readonly_userptr_capability probe_readonly_userptr(const device_info & info) noexcept;
kernel_configuration probe_kernel_configuration(const device_info & info) noexcept;

struct kernel_program;

class program_cache {
public:
    program_cache(const device_info & info, kernel_configuration configuration);
    ~program_cache();

    program_cache(const program_cache &) = delete;
    program_cache & operator=(const program_cache &) = delete;

    const device_info & info() const noexcept;
    bool inventory_available() const noexcept;
    bool program_available() const noexcept;
    std::string status() const;

    const xdna_kernel_variant * resolve_variant(const xdna_problem & problem) noexcept;
    std::shared_ptr<kernel_program> resolve_program(const xdna_problem & problem) noexcept;
    std::shared_ptr<kernel_program> resolve_program_after(
            const xdna_problem & problem,
            const kernel_program * previous) noexcept;
    detail::execution_gate::guard acquire_execution();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

class runtime {
public:
    explicit runtime(std::shared_ptr<program_cache> programs);
    ~runtime();

    runtime(const runtime &) = delete;
    runtime & operator=(const runtime &) = delete;

    const device_info & info() const noexcept;
    bool kernel_available() const noexcept;
    std::string kernel_status() const;

    bool supports_op(const ggml_tensor * op) const noexcept;
    int compute(ggml_tensor * op) noexcept;

    void get_stats(ggml_backend_xdna_stats * stats) const noexcept;
    void get_stats_v2(ggml_backend_xdna_stats_v2 * stats) const noexcept;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

} // namespace ggml_xdna

#pragma once

#include "ggml-xdna.h"

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
    std::string architecture;
};

enum class kernel_kind {
    none,
    bf16_gemv_288,
    q4_0_gemv_288,
};

struct kernel_configuration {
    bool available = false;
    kernel_kind kind = kernel_kind::none;
    std::string artifact_path;
    std::vector<char> xclbin_data;
    std::vector<uint32_t> instructions;
    std::string status;
};

std::vector<device_info> discover_devices(std::string * error = nullptr) noexcept;
kernel_configuration probe_kernel_configuration(const device_info & info) noexcept;
bool supports_op_contract(const ggml_tensor * op, kernel_kind kind) noexcept;

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

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

} // namespace ggml_xdna

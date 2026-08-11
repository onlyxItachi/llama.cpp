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

std::vector<device_info> discover_devices(std::string * error = nullptr) noexcept;

class runtime {
public:
    explicit runtime(const device_info & info);
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

#include "xdna-runtime.h"

#include "ggml.h"

#include <xrt/experimental/xrt_system.h>
#include <xrt/xrt_device.h>

#include <atomic>
#include <exception>
#include <utility>

namespace ggml_xdna {

namespace {

std::string architecture_from_name(const std::string & name) {
    if (name == "RyzenAI-npu4") {
        return "AIE2P/XDNA2";
    }
    if (name.rfind("RyzenAI-npu", 0) == 0) {
        return "XDNA (generation not validated)";
    }
    return "unknown XDNA";
}

bool is_xdna_device_name(const std::string & name) {
    return name.rfind("RyzenAI-npu", 0) == 0;
}

} // namespace

std::vector<device_info> discover_devices(std::string * error) noexcept {
    std::vector<device_info> result;
    try {
        const unsigned int count = xrt::system::enumerate_devices();
        for (unsigned int i = 0; i < count; ++i) {
            xrt::device device(i);
            const std::string name = device.get_info<xrt::info::device::name>();
            if (!is_xdna_device_name(name)) {
                continue;
            }

            device_info info;
            info.index = i;
            info.name = name;
            info.bdf = device.get_info<xrt::info::device::bdf>();
            info.architecture = architecture_from_name(name);
            result.emplace_back(std::move(info));
        }
    } catch (const std::exception & e) {
        if (error != nullptr) {
            *error = e.what();
        }
    } catch (...) {
        if (error != nullptr) {
            *error = "unknown XRT device discovery failure";
        }
    }
    return result;
}

struct runtime::impl {
    explicit impl(device_info value) : info(std::move(value)), device(info.index) {
        kernel_status = "no XDNA compute kernel configured";
    }

    device_info info;
    xrt::device device;
    std::string kernel_status;

    std::atomic<uint64_t> bo_creations { 0 };
    std::atomic<uint64_t> weight_registrations { 0 };
    std::atomic<uint64_t> weight_registration_hits { 0 };
    std::atomic<uint64_t> sync_to_device_calls { 0 };
    std::atomic<uint64_t> sync_from_device_calls { 0 };
    std::atomic<uint64_t> sync_to_device_bytes { 0 };
    std::atomic<uint64_t> sync_from_device_bytes { 0 };
    std::atomic<uint64_t> host_copy_calls { 0 };
    std::atomic<uint64_t> host_copy_bytes { 0 };
    std::atomic<uint64_t> kernel_submissions { 0 };
    std::atomic<uint64_t> kernel_time_ns { 0 };
};

runtime::runtime(const device_info & info) : pimpl(new impl(info)) {}

runtime::~runtime() = default;

const device_info & runtime::info() const noexcept {
    return pimpl->info;
}

bool runtime::kernel_available() const noexcept {
    return false;
}

const std::string & runtime::kernel_status() const noexcept {
    return pimpl->kernel_status;
}

bool runtime::supports_op(const ggml_tensor * op) const noexcept {
    (void) op;
    return false;
}

int runtime::compute(ggml_tensor * op) noexcept {
    (void) op;
    return -1;
}

void runtime::get_stats(ggml_backend_xdna_stats * stats) const noexcept {
    stats->bo_creations = pimpl->bo_creations.load();
    stats->weight_registrations = pimpl->weight_registrations.load();
    stats->weight_registration_hits = pimpl->weight_registration_hits.load();
    stats->sync_to_device_calls = pimpl->sync_to_device_calls.load();
    stats->sync_from_device_calls = pimpl->sync_from_device_calls.load();
    stats->sync_to_device_bytes = pimpl->sync_to_device_bytes.load();
    stats->sync_from_device_bytes = pimpl->sync_from_device_bytes.load();
    stats->host_copy_calls = pimpl->host_copy_calls.load();
    stats->host_copy_bytes = pimpl->host_copy_bytes.load();
    stats->kernel_submissions = pimpl->kernel_submissions.load();
    stats->kernel_time_ns = pimpl->kernel_time_ns.load();
}

} // namespace ggml_xdna

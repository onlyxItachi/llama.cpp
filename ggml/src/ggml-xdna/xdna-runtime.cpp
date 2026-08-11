#include "xdna-runtime.h"

#include "ggml-impl.h"
#include "ggml.h"

#include <xrt/experimental/xrt_system.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_xclbin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#if defined(__x86_64__)
#include <emmintrin.h>
#endif

namespace ggml_xdna {

namespace {

constexpr size_t artifact_header_bytes = 64;
constexpr char artifact_magic[8] = { 'G', 'G', 'X', 'D', 'N', 'A', '1', '\0' };

device_architecture architecture_from_device_name(const std::string & name) {
    if (name == "RyzenAI-npu4" || name == "RyzenAI-npu5" || name == "RyzenAI-npu6") {
        return device_architecture::aie2p;
    }
    if (name == "RyzenAI-npu1") {
        return device_architecture::aie2;
    }
    return device_architecture::unknown;
}

std::string architecture_description(const std::string & name, device_architecture architecture) {
    if (architecture == device_architecture::unknown && name.rfind("RyzenAI-npu", 0) == 0) {
        return "XDNA (generation not validated)";
    }
    return architecture_name(architecture);
}

bool is_xdna_device_name(const std::string & name) {
    return name.rfind("RyzenAI-npu", 0) == 0;
}

uint32_t read_le32(const unsigned char * value) {
    return static_cast<uint32_t>(value[0]) |
           (static_cast<uint32_t>(value[1]) << 8u) |
           (static_cast<uint32_t>(value[2]) << 16u) |
           (static_cast<uint32_t>(value[3]) << 24u);
}

uint64_t read_le64(const unsigned char * value) {
    return static_cast<uint64_t>(read_le32(value)) |
           (static_cast<uint64_t>(read_le32(value + 4)) << 32u);
}

struct artifact_contents {
    std::vector<char> xclbin_data;
    std::vector<uint32_t> instructions;
};

artifact_contents read_artifact_bundle(const std::string & path, const xdna_kernel_variant & variant) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("failed to open XDNA artifact bundle " + path);
    }

    const auto end = stream.tellg();
    if (end < static_cast<std::streamoff>(artifact_header_bytes)) {
        throw std::runtime_error("truncated XDNA artifact bundle " + path);
    }

    std::vector<unsigned char> bytes(static_cast<size_t>(end));
    stream.seekg(0);
    stream.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(end));
    if (!stream) {
        throw std::runtime_error("failed to read XDNA artifact bundle " + path);
    }

    if (std::memcmp(bytes.data(), artifact_magic, sizeof(artifact_magic)) != 0 ||
            read_le32(bytes.data() + 8) != artifact_header_bytes ||
            read_le32(bytes.data() + 12) != variant.artifact_abi_version ||
            read_le32(bytes.data() + 16) != variant.artifact_kind ||
            read_le32(bytes.data() + 20) != variant.m ||
            read_le32(bytes.data() + 24) != variant.k ||
            read_le32(bytes.data() + 28) != variant.weight_bytes ||
            read_le32(bytes.data() + 32) != variant.device_activation_bytes ||
            read_le32(bytes.data() + 36) != variant.device_output_bytes ||
            read_le64(bytes.data() + 56) != 0 || variant.n != 1) {
        throw std::runtime_error("XDNA artifact ABI metadata mismatch in " + path);
    }

    const uint64_t xclbin_bytes_u64 = read_le64(bytes.data() + 40);
    const uint64_t instruction_bytes_u64 = read_le64(bytes.data() + 48);
    const size_t payload_bytes = bytes.size() - artifact_header_bytes;
    if (xclbin_bytes_u64 == 0 || instruction_bytes_u64 == 0 ||
            xclbin_bytes_u64 > payload_bytes ||
            instruction_bytes_u64 != payload_bytes - xclbin_bytes_u64 ||
            instruction_bytes_u64 % sizeof(uint32_t) != 0) {
        throw std::runtime_error("invalid XDNA artifact payload sizes in " + path);
    }

    const size_t xclbin_bytes = static_cast<size_t>(xclbin_bytes_u64);
    const size_t instruction_bytes = static_cast<size_t>(instruction_bytes_u64);
    artifact_contents result;
    result.xclbin_data.assign(
        reinterpret_cast<const char *>(bytes.data() + artifact_header_bytes),
        reinterpret_cast<const char *>(bytes.data() + artifact_header_bytes + xclbin_bytes));
    result.instructions.resize(instruction_bytes / sizeof(uint32_t));
    std::memcpy(
        result.instructions.data(),
        bytes.data() + artifact_header_bytes + xclbin_bytes,
        instruction_bytes);
    return result;
}

uint64_t elapsed_ns(
        std::chrono::steady_clock::time_point start,
        std::chrono::steady_clock::time_point stop) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count());
}

void host_memory_fence() {
#if defined(__x86_64__)
    _mm_mfence();
#else
    std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

struct host_buffer_span {
    ggml_backend_buffer_t owner = nullptr;
    void * logical_base = nullptr;
    size_t logical_bytes = 0;
    void * page_base = nullptr;
    size_t page_bytes = 0;
};

host_buffer_span get_host_buffer_span(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->buffer == nullptr || tensor->data == nullptr) {
        throw std::runtime_error("tensor has no host buffer backing");
    }

    host_buffer_span span;
    span.owner = tensor->buffer;
    span.logical_base = ggml_backend_buffer_get_base(tensor->buffer);
    span.logical_bytes = ggml_backend_buffer_get_size(tensor->buffer);
    if (span.logical_base == nullptr || span.logical_bytes == 0) {
        throw std::runtime_error("tensor host buffer has no addressable range");
    }

    const uintptr_t logical_begin = reinterpret_cast<uintptr_t>(span.logical_base);
    if (span.logical_bytes > std::numeric_limits<uintptr_t>::max() - logical_begin) {
        throw std::runtime_error("tensor host buffer range overflows the address space");
    }
    const uintptr_t logical_end = logical_begin + span.logical_bytes;

    const uintptr_t tensor_begin = reinterpret_cast<uintptr_t>(tensor->data);
    const size_t tensor_bytes = ggml_nbytes(tensor);
    if (tensor_bytes > std::numeric_limits<uintptr_t>::max() - tensor_begin ||
            tensor_begin < logical_begin || tensor_begin + tensor_bytes > logical_end) {
        throw std::runtime_error("tensor range is outside its GGML host buffer");
    }

    const long page_size_value = ::sysconf(_SC_PAGESIZE);
    if (page_size_value <= 0) {
        throw std::runtime_error("failed to query the system page size");
    }
    const uintptr_t page_size = static_cast<uintptr_t>(page_size_value);
    const uintptr_t page_begin = logical_begin / page_size * page_size;
    if (logical_end > std::numeric_limits<uintptr_t>::max() - (page_size - 1)) {
        throw std::runtime_error("page-rounded host buffer range overflows the address space");
    }
    const uintptr_t page_end = (logical_end + page_size - 1) / page_size * page_size;

    span.page_base = reinterpret_cast<void *>(page_begin);
    span.page_bytes = static_cast<size_t>(page_end - page_begin);
    return span;
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
            info.arch = architecture_from_device_name(name);
            info.architecture = architecture_description(name, info.arch);
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

kernel_configuration probe_kernel_configuration(const device_info & info) noexcept {
    kernel_configuration result;
    try {
        size_t variant_count = 0;
        const xdna_kernel_variant * variants = kernel_variants(&variant_count);
        bool architecture_has_variant = false;
        for (size_t i = 0; i < variant_count; ++i) {
            const xdna_kernel_variant & variant = variants[i];
            if (variant.architecture != info.arch) {
                continue;
            }
            architecture_has_variant = true;
            const char * artifact = std::getenv(variant.environment_variable);
            if (artifact == nullptr || artifact[0] == '\0') {
                continue;
            }

            result.variant = &variant;
            result.artifact_path = artifact;
            result.status = std::string(variant.description) + " configured";
            artifact_contents contents = read_artifact_bundle(result.artifact_path, variant);
            result.xclbin_data = std::move(contents.xclbin_data);
            result.instructions = std::move(contents.instructions);
            result.available = true;
            return result;
        }

        if (!architecture_has_variant) {
            result.status = "no validated kernel for " + info.architecture;
        } else {
            result.status = "no registered kernel artifact bundle configured for " + info.architecture;
        }
    } catch (const std::exception & e) {
        result.available = false;
        result.variant = nullptr;
        result.status = std::string("kernel configuration failed: ") + e.what();
    } catch (...) {
        result.available = false;
        result.variant = nullptr;
        result.status = "unknown kernel configuration failure";
    }
    return result;
}

struct runtime::impl {
    struct root_registration {
        root_registration(xrt::device & device, host_buffer_span value) :
            span(value),
            parent(device, value.page_base, value.page_bytes, xrt::bo::flags::host_only, 0) {}

        host_buffer_span span;
        xrt::bo parent;
    };

    struct weight_view {
        weight_view(root_registration & root, const ggml_tensor * tensor) :
            owner(tensor->buffer),
            data(tensor->data),
            bytes(ggml_nbytes(tensor)),
            view(root.parent, bytes,
                 reinterpret_cast<uintptr_t>(data) - reinterpret_cast<uintptr_t>(root.span.page_base)) {}

        ggml_backend_buffer_t owner;
        const void * data;
        size_t bytes;
        xrt::bo view;
    };

    struct kernel_state {
        kernel_state(xrt::device & device, const kernel_configuration & configuration) {
            if (configuration.variant == nullptr) {
                throw std::runtime_error("XDNA kernel configuration has no selected variant");
            }
            const xdna_kernel_variant & variant = *configuration.variant;
            xclbin = std::make_unique<xrt::xclbin>(configuration.xclbin_data);
            device.register_xclbin(*xclbin);
            context = std::make_unique<xrt::hw_context>(device, xclbin->get_uuid());
            kernel = std::make_unique<xrt::kernel>(*context, variant.xrt_kernel_name);
            instructions = configuration.instructions;

            instruction_bo = std::make_unique<xrt::bo>(
                device, instructions.size() * sizeof(uint32_t), XCL_BO_FLAGS_CACHEABLE, kernel->group_id(1));
            std::memcpy(instruction_bo->map<void *>(), instructions.data(), instructions.size() * sizeof(uint32_t));
            const auto instruction_sync_start = std::chrono::steady_clock::now();
            instruction_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            instruction_sync_time_ns = elapsed_ns(instruction_sync_start, std::chrono::steady_clock::now());

            activation_bo = std::make_unique<xrt::bo>(
                device, variant.device_activation_bytes, xrt::bo::flags::host_only, kernel->group_id(4));
            activation_data = activation_bo->map<void *>();
            output_bo = std::make_unique<xrt::bo>(
                device, variant.device_output_bytes, xrt::bo::flags::host_only, kernel->group_id(5));
            output_data = output_bo->map<void *>();
            temporary_bo = std::make_unique<xrt::bo>(
                device, 1, xrt::bo::flags::host_only, kernel->group_id(6));
            trace_bo = std::make_unique<xrt::bo>(
                device, 1, xrt::bo::flags::host_only, kernel->group_id(7));

            run = std::make_unique<xrt::run>(*kernel);
            run->set_arg(0, variant.runtime_opcode);
            run->set_arg(1, *instruction_bo);
            run->set_arg(2, instructions.size());
            run->set_arg(4, *activation_bo);
            run->set_arg(5, *output_bo);
            run->set_arg(6, *temporary_bo);
            run->set_arg(7, *trace_bo);
        }

        std::unique_ptr<xrt::xclbin> xclbin;
        std::unique_ptr<xrt::hw_context> context;
        std::unique_ptr<xrt::kernel> kernel;
        std::vector<uint32_t> instructions;
        std::unique_ptr<xrt::bo> instruction_bo;
        std::unique_ptr<xrt::bo> activation_bo;
        std::unique_ptr<xrt::bo> output_bo;
        std::unique_ptr<xrt::bo> temporary_bo;
        std::unique_ptr<xrt::bo> trace_bo;
        std::unique_ptr<xrt::run> run;
        void * activation_data = nullptr;
        void * output_data = nullptr;
        uint64_t instruction_sync_time_ns = 0;
    };

    impl(device_info value, const kernel_configuration & configuration) :
        info(std::move(value)), device(info.index), variant(configuration.variant), kernel_status(configuration.status) {
        if (!configuration.available) {
            return;
        }

        try {
            state = std::make_unique<kernel_state>(device, configuration);
            explicit_bo_creations.fetch_add(5);
            explicit_bo_creation_bytes.fetch_add(
                state->instructions.size() * sizeof(uint32_t) + variant->device_activation_bytes +
                variant->device_output_bytes + 2);
            sync_to_device_calls.fetch_add(1);
            sync_to_device_bytes.fetch_add(state->instructions.size() * sizeof(uint32_t));
            sync_to_device_time_ns.fetch_add(state->instruction_sync_time_ns);
            kernel_status = configuration.status + " (loaded " + configuration.artifact_path + ")";
        } catch (const std::exception & e) {
            state.reset();
            kernel_status = std::string("AIE2P GEMV artifact load failed: ") + e.what();
        }
    }

    root_registration & register_root(const ggml_tensor * tensor) {
        const host_buffer_span wanted = get_host_buffer_span(tensor);
        for (auto & root : roots) {
            const auto & current = root->span;
            if (current.owner == wanted.owner && current.logical_base == wanted.logical_base &&
                    current.logical_bytes == wanted.logical_bytes) {
                buffer_registration_hits.fetch_add(1);
                return *root;
            }
        }

        // Destroy child BOs before replacing a registration for the same GGML
        // buffer owner. Model weight buffers are immutable and expected to
        // outlive the backend; a future XDNA-owned buft will provide precise
        // free-time eviction for more general transient allocations.
        weights.erase(std::remove_if(weights.begin(), weights.end(), [&](const std::unique_ptr<weight_view> & weight) {
            return weight->owner == wanted.owner;
        }), weights.end());
        roots.erase(std::remove_if(roots.begin(), roots.end(), [&](const std::unique_ptr<root_registration> & root) {
            return root->span.owner == wanted.owner;
        }), roots.end());

        const auto start = std::chrono::steady_clock::now();
        auto root = std::make_unique<root_registration>(device, wanted);
        const auto stop = std::chrono::steady_clock::now();
        buffer_registrations.fetch_add(1);
        registered_bytes.fetch_add(wanted.page_bytes);
        registration_time_ns.fetch_add(elapsed_ns(start, stop));
        explicit_bo_creations.fetch_add(1);
        explicit_bo_creation_bytes.fetch_add(wanted.page_bytes);
        roots.emplace_back(std::move(root));
        return *roots.back();
    }

    void sync_weight(weight_view & weight) {
        const auto sync_start = std::chrono::steady_clock::now();
        weight.view.sync(XCL_BO_SYNC_BO_TO_DEVICE, weight.bytes, 0);
        host_memory_fence();
        const uint64_t sync_ns = elapsed_ns(sync_start, std::chrono::steady_clock::now());
        weight_sync_to_device_calls.fetch_add(1);
        weight_sync_to_device_bytes.fetch_add(weight.bytes);
        weight_sync_to_device_time_ns.fetch_add(sync_ns);
        sync_to_device_calls.fetch_add(1);
        sync_to_device_bytes.fetch_add(weight.bytes);
        sync_to_device_time_ns.fetch_add(sync_ns);
    }

    xrt::bo & register_weight(const ggml_tensor * tensor) {
        if (ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            throw std::runtime_error("XDNA requires an immutable GGML weight buffer");
        }

        for (auto & weight : weights) {
            if (weight->owner == tensor->buffer && weight->data == tensor->data &&
                    weight->bytes == ggml_nbytes(tensor)) {
                weight_registration_hits.fetch_add(1);
                return weight->view;
            }
        }

        root_registration & root = register_root(tensor);
        weights.emplace_back(std::make_unique<weight_view>(root, tensor));
        weight_registrations.fetch_add(1);
        weight_registered_bytes.fetch_add(ggml_nbytes(tensor));
        sync_weight(*weights.back());
        return weights.back()->view;
    }

    device_info info;
    xrt::device device;
    const xdna_kernel_variant * variant = nullptr;
    std::string kernel_status;
    std::unique_ptr<kernel_state> state;
    std::vector<std::unique_ptr<root_registration>> roots;
    std::vector<std::unique_ptr<weight_view>> weights;
    std::mutex compute_mutex;

    std::atomic<uint64_t> initialization_time_ns { 0 };
    std::atomic<uint64_t> explicit_bo_creations { 0 };
    std::atomic<uint64_t> explicit_bo_creation_bytes { 0 };
    std::atomic<uint64_t> buffer_registrations { 0 };
    std::atomic<uint64_t> buffer_registration_hits { 0 };
    std::atomic<uint64_t> registered_bytes { 0 };
    std::atomic<uint64_t> registration_time_ns { 0 };
    std::atomic<uint64_t> weight_registrations { 0 };
    std::atomic<uint64_t> weight_registration_hits { 0 };
    std::atomic<uint64_t> weight_registered_bytes { 0 };
    std::atomic<uint64_t> weight_sync_to_device_calls { 0 };
    std::atomic<uint64_t> weight_sync_to_device_bytes { 0 };
    std::atomic<uint64_t> weight_sync_to_device_time_ns { 0 };
    std::atomic<uint64_t> sync_to_device_calls { 0 };
    std::atomic<uint64_t> sync_from_device_calls { 0 };
    std::atomic<uint64_t> sync_to_device_bytes { 0 };
    std::atomic<uint64_t> sync_from_device_bytes { 0 };
    std::atomic<uint64_t> sync_to_device_time_ns { 0 };
    std::atomic<uint64_t> sync_from_device_time_ns { 0 };
    std::atomic<uint64_t> host_copy_calls { 0 };
    std::atomic<uint64_t> host_copy_bytes { 0 };
    std::atomic<uint64_t> host_copy_time_ns { 0 };
    std::atomic<uint64_t> weight_copy_bytes { 0 };
    std::atomic<uint64_t> kernel_submissions { 0 };
    std::atomic<uint64_t> first_kernel_time_ns { 0 };
    std::atomic<uint64_t> first_compute_time_ns { 0 };
    std::atomic<uint64_t> kernel_time_ns { 0 };
    std::atomic<uint64_t> total_compute_time_ns { 0 };
};

runtime::runtime(const device_info & info, const kernel_configuration & configuration) {
    const auto initialization_start = std::chrono::steady_clock::now();
    pimpl = std::make_unique<impl>(info, configuration);
    pimpl->initialization_time_ns.store(elapsed_ns(initialization_start, std::chrono::steady_clock::now()));
}

runtime::~runtime() = default;

const device_info & runtime::info() const noexcept {
    return pimpl->info;
}

bool runtime::kernel_available() const noexcept {
    return pimpl->state != nullptr;
}

const std::string & runtime::kernel_status() const noexcept {
    return pimpl->kernel_status;
}

bool runtime::supports_op(const ggml_tensor * op) const noexcept {
    if (!kernel_available() || pimpl->variant == nullptr) {
        return false;
    }
    xdna_problem problem;
    if (!problem_from_ggml(op, pimpl->info.arch, &problem)) {
        return false;
    }
    const xdna_kernel_variant * candidates[] = { pimpl->variant };
    return select_kernel_variant(problem, candidates, 1) != nullptr;
}

int runtime::compute(ggml_tensor * op) noexcept {
    if (!supports_op(op)) {
        return -1;
    }

    const auto total_start = std::chrono::steady_clock::now();
    try {
        // One persistent command BO/run is intentionally reused. Serialize it
        // until a future implementation provisions a bounded run pool.
        const std::lock_guard<std::mutex> lock(pimpl->compute_mutex);
        const xdna_kernel_variant & variant = *pimpl->variant;
        if (ggml_nbytes(op->src[0]) != variant.weight_bytes ||
                variant.activation_type != data_type::f32 ||
                variant.device_activation_type != data_type::bf16 ||
                variant.output_type != data_type::f32 ||
                variant.device_output_type != data_type::f32) {
            throw std::runtime_error("selected XDNA kernel variant has an unsupported host storage conversion");
        }
        xrt::bo & weight_bo = pimpl->register_weight(op->src[0]);

        const auto host_copy_start = std::chrono::steady_clock::now();
        ggml_fp32_to_bf16_row(
            static_cast<const float *>(op->src[1]->data),
            static_cast<ggml_bf16_t *>(pimpl->state->activation_data),
            variant.k * variant.n);
        pimpl->host_copy_time_ns.fetch_add(
            elapsed_ns(host_copy_start, std::chrono::steady_clock::now()));
        pimpl->host_copy_calls.fetch_add(1);
        pimpl->host_copy_bytes.fetch_add(variant.device_activation_bytes);
        const auto activation_sync_start = std::chrono::steady_clock::now();
        pimpl->state->activation_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE, variant.device_activation_bytes, 0);
        pimpl->sync_to_device_time_ns.fetch_add(
            elapsed_ns(activation_sync_start, std::chrono::steady_clock::now()));
        pimpl->sync_to_device_calls.fetch_add(1);
        pimpl->sync_to_device_bytes.fetch_add(variant.device_activation_bytes);
        host_memory_fence();

        pimpl->state->run->set_arg(3, weight_bo);
        const auto kernel_start = std::chrono::steady_clock::now();
        pimpl->state->run->start();
        const ert_cmd_state state = pimpl->state->run->wait();
        const auto kernel_stop = std::chrono::steady_clock::now();
        const uint64_t submission = pimpl->kernel_submissions.fetch_add(1) + 1;
        const uint64_t kernel_ns = elapsed_ns(kernel_start, kernel_stop);
        pimpl->kernel_time_ns.fetch_add(kernel_ns);
        if (state != ERT_CMD_STATE_COMPLETED) {
            GGML_LOG_ERROR("ggml_xdna: kernel returned ERT state %d\n", static_cast<int>(state));
            return -1;
        }

        const auto output_sync_start = std::chrono::steady_clock::now();
        pimpl->state->output_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE, variant.device_output_bytes, 0);
        pimpl->sync_from_device_time_ns.fetch_add(
            elapsed_ns(output_sync_start, std::chrono::steady_clock::now()));
        pimpl->sync_from_device_calls.fetch_add(1);
        pimpl->sync_from_device_bytes.fetch_add(variant.device_output_bytes);
        host_memory_fence();

        const auto output_copy_start = std::chrono::steady_clock::now();
        std::memcpy(op->data, pimpl->state->output_data, variant.device_output_bytes);
        pimpl->host_copy_time_ns.fetch_add(
            elapsed_ns(output_copy_start, std::chrono::steady_clock::now()));
        pimpl->host_copy_calls.fetch_add(1);
        pimpl->host_copy_bytes.fetch_add(variant.device_output_bytes);

        const uint64_t total_ns = elapsed_ns(total_start, std::chrono::steady_clock::now());
        pimpl->total_compute_time_ns.fetch_add(total_ns);
        if (submission == 1) {
            pimpl->first_kernel_time_ns.store(kernel_ns);
            pimpl->first_compute_time_ns.store(total_ns);
            GGML_LOG_INFO(
                "ggml_xdna: executed GGML_OP_MUL_MAT node '%s' on XDNA variant '%s': "
                "%s[M=%" PRId64 ",K=%" PRId64 "] x %s[K=%" PRId64 ",N=%" PRId64 "], "
                "kernel %.3f ms, total %.3f ms, weight copies 0 bytes\n",
                op->name, variant.id, data_type_name(variant.weights_type), variant.m, variant.k,
                data_type_name(variant.activation_type), variant.k, variant.n,
                kernel_ns / 1.0e6, total_ns / 1.0e6);
        }
        return 0;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ggml_xdna: MUL_MAT execution failed: %s\n", e.what());
    } catch (...) {
        GGML_LOG_ERROR("ggml_xdna: MUL_MAT execution failed with an unknown XRT error\n");
    }
    return -1;
}

void runtime::get_stats(ggml_backend_xdna_stats * stats) const noexcept {
    stats->initialization_time_ns = pimpl->initialization_time_ns.load();
    stats->explicit_bo_creations = pimpl->explicit_bo_creations.load();
    stats->explicit_bo_creation_bytes = pimpl->explicit_bo_creation_bytes.load();
    stats->buffer_registrations = pimpl->buffer_registrations.load();
    stats->buffer_registration_hits = pimpl->buffer_registration_hits.load();
    stats->registered_bytes = pimpl->registered_bytes.load();
    stats->registration_time_ns = pimpl->registration_time_ns.load();
    stats->weight_registrations = pimpl->weight_registrations.load();
    stats->weight_registration_hits = pimpl->weight_registration_hits.load();
    stats->weight_registered_bytes = pimpl->weight_registered_bytes.load();
    stats->weight_sync_to_device_calls = pimpl->weight_sync_to_device_calls.load();
    stats->weight_sync_to_device_bytes = pimpl->weight_sync_to_device_bytes.load();
    stats->weight_sync_to_device_time_ns = pimpl->weight_sync_to_device_time_ns.load();
    stats->sync_to_device_calls = pimpl->sync_to_device_calls.load();
    stats->sync_from_device_calls = pimpl->sync_from_device_calls.load();
    stats->sync_to_device_bytes = pimpl->sync_to_device_bytes.load();
    stats->sync_from_device_bytes = pimpl->sync_from_device_bytes.load();
    stats->sync_to_device_time_ns = pimpl->sync_to_device_time_ns.load();
    stats->sync_from_device_time_ns = pimpl->sync_from_device_time_ns.load();
    stats->host_copy_calls = pimpl->host_copy_calls.load();
    stats->host_copy_bytes = pimpl->host_copy_bytes.load();
    stats->host_copy_time_ns = pimpl->host_copy_time_ns.load();
    stats->weight_copy_bytes = pimpl->weight_copy_bytes.load();
    stats->kernel_submissions = pimpl->kernel_submissions.load();
    stats->first_kernel_time_ns = pimpl->first_kernel_time_ns.load();
    stats->first_compute_time_ns = pimpl->first_compute_time_ns.load();
    stats->kernel_time_ns = pimpl->kernel_time_ns.load();
    stats->total_compute_time_ns = pimpl->total_compute_time_ns.load();
}

} // namespace ggml_xdna

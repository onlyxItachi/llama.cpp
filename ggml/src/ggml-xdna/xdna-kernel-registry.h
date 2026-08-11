#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

struct ggml_tensor;

namespace ggml_xdna {

enum class device_architecture {
    unknown,
    aie2,
    aie2p,
};

enum class operation {
    unknown,
    mul_mat,
};

enum class data_type {
    unknown,
    f32,
    bf16,
    q4_0,
};

enum class tensor_layout {
    contiguous,
    strided,
};

enum class weight_usage {
    unknown,
    immutable,
    mutable_buffer,
};

// Backend-neutral description translated from one GGML operation.  Kernel
// selection consumes this value; shapes are not properties of the XDNA
// backend or runtime itself.
struct xdna_problem {
    device_architecture architecture = device_architecture::unknown;
    operation op = operation::unknown;
    data_type weights_type = data_type::unknown;
    data_type activation_type = data_type::unknown;
    data_type output_type = data_type::unknown;
    int64_t m = 0;
    int64_t n = 0;
    int64_t k = 0;
    std::array<int64_t, 2> weights_batch { 0, 0 };
    std::array<int64_t, 2> activation_batch { 0, 0 };
    std::array<int64_t, 2> output_batch { 0, 0 };
    tensor_layout weights_layout = tensor_layout::strided;
    tensor_layout activation_layout = tensor_layout::strided;
    tensor_layout output_layout = tensor_layout::strided;
    weight_usage weights_usage = weight_usage::unknown;
    bool default_precision = false;
    bool src0_hadamard = false;
};

// A specialized compiled implementation.  Adding a shape or architecture is
// a registry entry plus its artifact; it must not add shape checks throughout
// the backend/runtime.
struct xdna_kernel_variant {
    const char * id;
    const char * description;
    const char * environment_variable;
    const char * xrt_kernel_name;
    device_architecture architecture;
    operation op;
    data_type weights_type;
    data_type activation_type;
    data_type output_type;
    data_type device_activation_type;
    data_type device_output_type;
    int64_t m;
    int64_t n;
    int64_t k;
    std::array<int64_t, 2> weights_batch;
    std::array<int64_t, 2> activation_batch;
    std::array<int64_t, 2> output_batch;
    tensor_layout weights_layout;
    tensor_layout activation_layout;
    tensor_layout output_layout;
    uint32_t artifact_kind;
    uint32_t artifact_abi_version;
    uint32_t runtime_opcode;
    size_t weight_bytes;
    size_t device_activation_bytes;
    size_t device_output_bytes;
    uint32_t rows_per_worker;
    uint32_t worker_count;
    bool prefer_for_offload;
};

const char * architecture_name(device_architecture architecture) noexcept;
const char * data_type_name(data_type type) noexcept;

bool problem_from_ggml(
        const ggml_tensor * op,
        device_architecture architecture,
        xdna_problem * problem) noexcept;

const xdna_kernel_variant * kernel_variants(size_t * count) noexcept;

bool kernel_variant_supports(
        const xdna_kernel_variant & variant,
        const xdna_problem & problem) noexcept;

const xdna_kernel_variant * select_kernel_variant(
        const xdna_problem & problem,
        const xdna_kernel_variant * const * candidates,
        size_t candidate_count) noexcept;

bool kernel_variant_prefers_offload(
        const xdna_kernel_variant & variant,
        bool force_preference) noexcept;

} // namespace ggml_xdna

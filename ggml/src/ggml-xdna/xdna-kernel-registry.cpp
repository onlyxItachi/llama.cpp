#include "xdna-kernel-registry.h"

#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml.h"

#include <algorithm>

namespace ggml_xdna {

namespace {

struct gemv_format {
    data_type weights_type;
    uint32_t artifact_kind;
    size_t block_values;
    size_t block_bytes;
};

constexpr gemv_format bf16_format = { data_type::bf16, 1, 1, sizeof(uint16_t) };
constexpr gemv_format q4_0_format = { data_type::q4_0, 2, 32, 18 };
constexpr gemv_format q8_0_format = { data_type::q8_0, 3, 32, 34 };

constexpr xdna_kernel_variant make_aie2p_gemv(
        const char * id,
        const char * description,
        const char * environment_variable,
        gemv_format format,
        int64_t m,
        int64_t k,
        uint32_t rows_per_worker,
        uint32_t worker_count,
        bool prefer_for_offload = false) {
    return {
        /* .id                      = */ id,
        /* .description             = */ description,
        /* .environment_variable    = */ environment_variable,
        /* .xrt_kernel_name         = */ "MLIR_AIE",
        /* .architecture            = */ device_architecture::aie2p,
        /* .op                      = */ operation::mul_mat,
        /* .weights_type            = */ format.weights_type,
        /* .activation_type         = */ data_type::f32,
        /* .output_type             = */ data_type::f32,
        /* .device_activation_type  = */ data_type::bf16,
        /* .device_output_type      = */ data_type::f32,
        /* .m                       = */ m,
        /* .n                       = */ 1,
        /* .k                       = */ k,
        /* .weights_batch           = */ { 1, 1 },
        /* .activation_batch        = */ { 1, 1 },
        /* .output_batch            = */ { 1, 1 },
        /* .weights_layout          = */ tensor_layout::contiguous,
        /* .activation_layout       = */ tensor_layout::contiguous,
        /* .output_layout           = */ tensor_layout::contiguous,
        /* .artifact_kind           = */ format.artifact_kind,
        /* .artifact_abi_version    = */ 1,
        /* .runtime_opcode          = */ 3,
        /* .weight_bytes            = */ static_cast<size_t>(m) * (static_cast<size_t>(k) / format.block_values) * format.block_bytes,
        /* .device_activation_bytes = */ static_cast<size_t>(k) * sizeof(uint16_t),
        /* .device_output_bytes     = */ static_cast<size_t>(m) * sizeof(float),
        /* .rows_per_worker         = */ rows_per_worker,
        /* .worker_count            = */ worker_count,
        /* .prefer_for_offload      = */ prefer_for_offload,
    };
}

constexpr xdna_kernel_variant variants[] = {
    make_aie2p_gemv(
        "aie2p-q4_0-gemv-m288-n1-k288", "AIE2P Q4_0xF32 decode GEMV M=288 N=1 K=288",
        "GGML_XDNA_AIE2P_Q4_0_GEMV_BUNDLE", q4_0_format, 288, 288, 32, 9),
    make_aie2p_gemv(
        "aie2p-bf16-gemv-m288-n1-k288", "AIE2P BF16xF32 decode GEMV M=288 N=1 K=288",
        "GGML_XDNA_AIE2P_BF16_GEMV_BUNDLE", bf16_format, 288, 288, 32, 1),
    make_aie2p_gemv(
        "aie2p-bf16-gemv-m8192-n1-k2048", "AIE2P BF16xF32 decode GEMV M=8192 N=1 K=2048",
        "GGML_XDNA_AIE2P_BF16_GEMV_M8192_K2048_BUNDLE", bf16_format, 8192, 2048, 8, 32),
    make_aie2p_gemv(
        "aie2p-bf16-gemv-m8192-n1-k3072", "AIE2P BF16xF32 decode GEMV M=8192 N=1 K=3072",
        "GGML_XDNA_AIE2P_BF16_GEMV_M8192_K3072_BUNDLE", bf16_format, 8192, 3072, 8, 32),
    make_aie2p_gemv(
        "aie2p-q4_0-gemv-m512-n1-k2560", "AIE2P Q4_0xF32 decode GEMV M=512 N=1 K=2560",
        "GGML_XDNA_AIE2P_Q4_0_GEMV_M512_K2560_BUNDLE", q4_0_format, 512, 2560, 16, 32),
    make_aie2p_gemv(
        "aie2p-q4_0-gemv-m10240-n1-k2560", "AIE2P Q4_0xF32 decode GEMV M=10240 N=1 K=2560",
        "GGML_XDNA_AIE2P_Q4_0_GEMV_M10240_K2560_BUNDLE", q4_0_format, 10240, 2560, 16, 32),
    make_aie2p_gemv(
        "aie2p-q8_0-gemv-m9216-n1-k2560", "AIE2P Q8_0xF32 decode GEMV M=9216 N=1 K=2560",
        "GGML_XDNA_AIE2P_Q8_0_GEMV_M9216_K2560_BUNDLE", q8_0_format, 9216, 2560, 16, 32),
};

data_type translate_type(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return data_type::f32;
        case GGML_TYPE_BF16:
            return data_type::bf16;
        case GGML_TYPE_Q4_0:
            return data_type::q4_0;
        case GGML_TYPE_Q8_0:
            return data_type::q8_0;
        default:
            return data_type::unknown;
    }
}

tensor_layout translate_layout(const ggml_tensor * tensor) {
    return ggml_is_contiguous(tensor) ? tensor_layout::contiguous : tensor_layout::strided;
}

} // namespace

const char * architecture_name(device_architecture architecture) noexcept {
    switch (architecture) {
        case device_architecture::aie2:
            return "AIE2/XDNA";
        case device_architecture::aie2p:
            return "AIE2P/XDNA2";
        case device_architecture::unknown:
            return "unknown XDNA";
    }
    return "unknown XDNA";
}

const char * data_type_name(data_type type) noexcept {
    switch (type) {
        case data_type::f32:
            return "F32";
        case data_type::bf16:
            return "BF16";
        case data_type::q4_0:
            return "Q4_0";
        case data_type::q8_0:
            return "Q8_0";
        case data_type::unknown:
            return "unknown";
    }
    return "unknown";
}

bool problem_from_ggml(
        const ggml_tensor * op,
        device_architecture architecture,
        xdna_problem * problem) noexcept {
    if (op == nullptr || problem == nullptr || op->op != GGML_OP_MUL_MAT ||
            op->src[0] == nullptr || op->src[1] == nullptr) {
        return false;
    }

    const ggml_tensor * weights = op->src[0];
    const ggml_tensor * activation = op->src[1];
    if (weights->ne[0] != activation->ne[0] || op->ne[0] != weights->ne[1] ||
            op->ne[1] != activation->ne[1]) {
        return false;
    }

    xdna_problem result;
    result.architecture = architecture;
    result.op = operation::mul_mat;
    result.weights_type = translate_type(weights->type);
    result.activation_type = translate_type(activation->type);
    result.output_type = translate_type(op->type);
    result.m = weights->ne[1];
    result.n = activation->ne[1];
    result.k = weights->ne[0];
    result.weights_batch = { weights->ne[2], weights->ne[3] };
    result.activation_batch = { activation->ne[2], activation->ne[3] };
    result.output_batch = { op->ne[2], op->ne[3] };
    result.weights_layout = translate_layout(weights);
    result.activation_layout = translate_layout(activation);
    result.output_layout = translate_layout(op);
    result.default_precision = ggml_get_op_params_i32(op, 0) == GGML_PREC_DEFAULT;
    result.src0_hadamard = ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD;
    if (weights->buffer == nullptr) {
        result.weights_usage = weight_usage::unknown;
    } else if (ggml_backend_buffer_get_usage(weights->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        result.weights_usage = weight_usage::immutable;
    } else {
        result.weights_usage = weight_usage::mutable_buffer;
    }

    *problem = result;
    return true;
}

const xdna_kernel_variant * kernel_variants(size_t * count) noexcept {
    if (count != nullptr) {
        *count = sizeof(variants) / sizeof(variants[0]);
    }
    return variants;
}

bool kernel_variant_supports(
        const xdna_kernel_variant & variant,
        const xdna_problem & problem) noexcept {
    return problem.architecture == variant.architecture && problem.op == variant.op &&
           problem.weights_type == variant.weights_type &&
           problem.activation_type == variant.activation_type &&
           problem.output_type == variant.output_type &&
           problem.m == variant.m && problem.n == variant.n && problem.k == variant.k &&
           problem.weights_batch == variant.weights_batch &&
           problem.activation_batch == variant.activation_batch &&
           problem.output_batch == variant.output_batch &&
           problem.weights_layout == variant.weights_layout &&
           problem.activation_layout == variant.activation_layout &&
           problem.output_layout == variant.output_layout &&
           problem.default_precision && !problem.src0_hadamard;
}

const xdna_kernel_variant * select_kernel_variant(
        const xdna_problem & problem,
        const xdna_kernel_variant * const * candidates,
        size_t candidate_count) noexcept {
    for (size_t i = 0; i < candidate_count; ++i) {
        if (candidates[i] != nullptr && kernel_variant_supports(*candidates[i], problem)) {
            return candidates[i];
        }
    }
    return nullptr;
}

bool kernel_variant_prefers_offload(
        const xdna_kernel_variant & variant,
        bool force_preference) noexcept {
    return force_preference || variant.prefer_for_offload;
}

} // namespace ggml_xdna

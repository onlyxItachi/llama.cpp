#include "xdna-kernel-registry.h"

#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml.h"

#include <algorithm>

namespace ggml_xdna {

namespace {

constexpr size_t ggml_q4_0_block_values = 32;
constexpr size_t ggml_q4_0_block_bytes = 18;

constexpr xdna_kernel_variant variants[] = {
    {
        /* .id                      = */ "aie2p-q4_0-gemv-m288-n1-k288",
        /* .description             = */ "AIE2P Q4_0xF32 decode GEMV M=288 N=1 K=288",
        /* .environment_variable    = */ "GGML_XDNA_AIE2P_Q4_0_GEMV_BUNDLE",
        /* .xrt_kernel_name         = */ "MLIR_AIE",
        /* .architecture            = */ device_architecture::aie2p,
        /* .op                      = */ operation::mul_mat,
        /* .weights_type            = */ data_type::q4_0,
        /* .activation_type         = */ data_type::f32,
        /* .output_type             = */ data_type::f32,
        /* .device_activation_type  = */ data_type::bf16,
        /* .device_output_type      = */ data_type::f32,
        /* .m                       = */ 288,
        /* .n                       = */ 1,
        /* .k                       = */ 288,
        /* .weights_batch           = */ { 1, 1 },
        /* .activation_batch        = */ { 1, 1 },
        /* .output_batch            = */ { 1, 1 },
        /* .weights_layout          = */ tensor_layout::contiguous,
        /* .activation_layout       = */ tensor_layout::contiguous,
        /* .output_layout           = */ tensor_layout::contiguous,
        /* .artifact_kind           = */ 2,
        /* .artifact_abi_version    = */ 1,
        /* .runtime_opcode          = */ 3,
        /* .weight_bytes            = */ 288 * (288 / ggml_q4_0_block_values) * ggml_q4_0_block_bytes,
        /* .device_activation_bytes = */ 288 * sizeof(uint16_t),
        /* .device_output_bytes     = */ 288 * sizeof(float),
        /* .rows_per_worker         = */ 32,
        /* .worker_count            = */ 9,
    },
    {
        /* .id                      = */ "aie2p-bf16-gemv-m288-n1-k288",
        /* .description             = */ "AIE2P BF16xF32 decode GEMV M=288 N=1 K=288",
        /* .environment_variable    = */ "GGML_XDNA_AIE2P_BF16_GEMV_BUNDLE",
        /* .xrt_kernel_name         = */ "MLIR_AIE",
        /* .architecture            = */ device_architecture::aie2p,
        /* .op                      = */ operation::mul_mat,
        /* .weights_type            = */ data_type::bf16,
        /* .activation_type         = */ data_type::f32,
        /* .output_type             = */ data_type::f32,
        /* .device_activation_type  = */ data_type::bf16,
        /* .device_output_type      = */ data_type::f32,
        /* .m                       = */ 288,
        /* .n                       = */ 1,
        /* .k                       = */ 288,
        /* .weights_batch           = */ { 1, 1 },
        /* .activation_batch        = */ { 1, 1 },
        /* .output_batch            = */ { 1, 1 },
        /* .weights_layout          = */ tensor_layout::contiguous,
        /* .activation_layout       = */ tensor_layout::contiguous,
        /* .output_layout           = */ tensor_layout::contiguous,
        /* .artifact_kind           = */ 1,
        /* .artifact_abi_version    = */ 1,
        /* .runtime_opcode          = */ 3,
        /* .weight_bytes            = */ 288 * 288 * sizeof(uint16_t),
        /* .device_activation_bytes = */ 288 * sizeof(uint16_t),
        /* .device_output_bytes     = */ 288 * sizeof(float),
        /* .rows_per_worker         = */ 32,
        /* .worker_count            = */ 1,
    },
    {
        /* .id                      = */ "aie2p-q4_0-gemv-m512-n1-k2560",
        /* .description             = */ "AIE2P Q4_0xF32 decode GEMV M=512 N=1 K=2560",
        /* .environment_variable    = */ "GGML_XDNA_AIE2P_Q4_0_GEMV_M512_K2560_BUNDLE",
        /* .xrt_kernel_name         = */ "MLIR_AIE",
        /* .architecture            = */ device_architecture::aie2p,
        /* .op                      = */ operation::mul_mat,
        /* .weights_type            = */ data_type::q4_0,
        /* .activation_type         = */ data_type::f32,
        /* .output_type             = */ data_type::f32,
        /* .device_activation_type  = */ data_type::bf16,
        /* .device_output_type      = */ data_type::f32,
        /* .m                       = */ 512,
        /* .n                       = */ 1,
        /* .k                       = */ 2560,
        /* .weights_batch           = */ { 1, 1 },
        /* .activation_batch        = */ { 1, 1 },
        /* .output_batch            = */ { 1, 1 },
        /* .weights_layout          = */ tensor_layout::contiguous,
        /* .activation_layout       = */ tensor_layout::contiguous,
        /* .output_layout           = */ tensor_layout::contiguous,
        /* .artifact_kind           = */ 2,
        /* .artifact_abi_version    = */ 1,
        /* .runtime_opcode          = */ 3,
        /* .weight_bytes            = */ 512 * (2560 / ggml_q4_0_block_values) * ggml_q4_0_block_bytes,
        /* .device_activation_bytes = */ 2560 * sizeof(uint16_t),
        /* .device_output_bytes     = */ 512 * sizeof(float),
        /* .rows_per_worker         = */ 16,
        /* .worker_count            = */ 32,
    },
    {
        /* .id                      = */ "aie2p-q4_0-gemv-m10240-n1-k2560",
        /* .description             = */ "AIE2P Q4_0xF32 decode GEMV M=10240 N=1 K=2560",
        /* .environment_variable    = */ "GGML_XDNA_AIE2P_Q4_0_GEMV_M10240_K2560_BUNDLE",
        /* .xrt_kernel_name         = */ "MLIR_AIE",
        /* .architecture            = */ device_architecture::aie2p,
        /* .op                      = */ operation::mul_mat,
        /* .weights_type            = */ data_type::q4_0,
        /* .activation_type         = */ data_type::f32,
        /* .output_type             = */ data_type::f32,
        /* .device_activation_type  = */ data_type::bf16,
        /* .device_output_type      = */ data_type::f32,
        /* .m                       = */ 10240,
        /* .n                       = */ 1,
        /* .k                       = */ 2560,
        /* .weights_batch           = */ { 1, 1 },
        /* .activation_batch        = */ { 1, 1 },
        /* .output_batch            = */ { 1, 1 },
        /* .weights_layout          = */ tensor_layout::contiguous,
        /* .activation_layout       = */ tensor_layout::contiguous,
        /* .output_layout           = */ tensor_layout::contiguous,
        /* .artifact_kind           = */ 2,
        /* .artifact_abi_version    = */ 1,
        /* .runtime_opcode          = */ 3,
        /* .weight_bytes            = */ 10240 * (2560 / ggml_q4_0_block_values) * ggml_q4_0_block_bytes,
        /* .device_activation_bytes = */ 2560 * sizeof(uint16_t),
        /* .device_output_bytes     = */ 10240 * sizeof(float),
        /* .rows_per_worker         = */ 16,
        /* .worker_count            = */ 32,
    },
};

data_type translate_type(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return data_type::f32;
        case GGML_TYPE_BF16:
            return data_type::bf16;
        case GGML_TYPE_Q4_0:
            return data_type::q4_0;
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
           problem.weights_usage != weight_usage::mutable_buffer &&
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

} // namespace ggml_xdna

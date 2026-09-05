#include "xdna-kernel-registry.h"
#include "ggml.h"
#include "ggml-backend-impl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static void require(bool condition, const char * message) {
    if (!condition) {
        fprintf(stderr, "test-xdna-offload-policy: %s\n", message);
        abort();
    }
}

static ggml_type weight_type(ggml_xdna::data_type type) {
    switch (type) {
        case ggml_xdna::data_type::bf16: return GGML_TYPE_BF16;
        case ggml_xdna::data_type::q4_0: return GGML_TYPE_Q4_0;
        case ggml_xdna::data_type::q8_0: return GGML_TYPE_Q8_0;
        default: abort();
    }
}

struct expected_variant_contract {
    const char * id;
    const char * environment_variable;
    ggml_xdna::data_type weights_type;
    int64_t m;
    int64_t k;
    uint32_t artifact_kind;
    size_t weight_bytes;
    size_t device_activation_bytes;
    size_t device_output_bytes;
    uint32_t rows_per_worker;
    uint32_t worker_count;
    bool prefer_for_offload;
};

static constexpr expected_variant_contract expected_variants[] = {
    {
        "aie2p-q4_0-gemv-m288-n1-k288", "GGML_XDNA_AIE2P_Q4_0_GEMV_BUNDLE",
        ggml_xdna::data_type::q4_0,
        288, 288, 2, 46656, 576, 1152, 32, 9, false,
    },
    {
        "aie2p-bf16-gemv-m288-n1-k288", "GGML_XDNA_AIE2P_BF16_GEMV_BUNDLE",
        ggml_xdna::data_type::bf16,
        288, 288, 1, 165888, 576, 1152, 32, 1, false,
    },
    {
        "aie2p-bf16-gemv-m8192-n1-k2048", "GGML_XDNA_AIE2P_BF16_GEMV_M8192_K2048_BUNDLE",
        ggml_xdna::data_type::bf16,
        8192, 2048, 1, 33554432, 4096, 32768, 8, 32, false,
    },
    {
        "aie2p-bf16-gemv-m8192-n1-k3072", "GGML_XDNA_AIE2P_BF16_GEMV_M8192_K3072_BUNDLE",
        ggml_xdna::data_type::bf16,
        8192, 3072, 1, 50331648, 6144, 32768, 8, 32, false,
    },
    {
        "aie2p-q4_0-gemv-m512-n1-k2560", "GGML_XDNA_AIE2P_Q4_0_GEMV_M512_K2560_BUNDLE",
        ggml_xdna::data_type::q4_0,
        512, 2560, 2, 737280, 5120, 2048, 16, 32, false,
    },
    {
        "aie2p-q4_0-gemv-m10240-n1-k2560", "GGML_XDNA_AIE2P_Q4_0_GEMV_M10240_K2560_BUNDLE",
        ggml_xdna::data_type::q4_0,
        10240, 2560, 2, 14745600, 5120, 40960, 16, 32, false,
    },
    {
        "aie2p-q8_0-gemv-m9216-n1-k2560", "GGML_XDNA_AIE2P_Q8_0_GEMV_M9216_K2560_BUNDLE",
        ggml_xdna::data_type::q8_0,
        9216, 2560, 3, 25067520, 5120, 36864, 16, 32, false,
    },
};

static void test_variant_inventory(
        const ggml_xdna::xdna_kernel_variant * variants,
        size_t variant_count) {
    using namespace ggml_xdna;
    require(
            variant_count == sizeof(expected_variants) / sizeof(expected_variants[0]),
            "the kernel registry specialization count changed");

    for (const expected_variant_contract & expected : expected_variants) {
        const xdna_kernel_variant * actual = nullptr;
        for (size_t i = 0; i < variant_count; ++i) {
            if (variants[i].id != nullptr && strcmp(variants[i].id, expected.id) == 0) {
                actual = &variants[i];
                break;
            }
        }
        require(actual != nullptr, "an expected kernel specialization is missing");
        require(
                actual->environment_variable != nullptr &&
                    strcmp(actual->environment_variable, expected.environment_variable) == 0,
                "a kernel specialization environment variable changed");
        require(
                actual->xrt_kernel_name != nullptr && strcmp(actual->xrt_kernel_name, "MLIR_AIE") == 0,
                "a kernel specialization XRT entry point changed");
        require(
                actual->architecture == device_architecture::aie2p && actual->op == operation::mul_mat &&
                    actual->weights_type == expected.weights_type && actual->activation_type == data_type::f32 &&
                    actual->output_type == data_type::f32 && actual->device_activation_type == data_type::bf16 &&
                    actual->device_output_type == data_type::f32,
                "a kernel specialization type contract changed");
        require(
                actual->m == expected.m && actual->n == 1 && actual->k == expected.k &&
                    actual->weights_batch[0] == 1 && actual->weights_batch[1] == 1 &&
                    actual->activation_batch[0] == 1 && actual->activation_batch[1] == 1 &&
                    actual->output_batch[0] == 1 && actual->output_batch[1] == 1,
                "a kernel specialization shape contract changed");
        require(
                actual->weights_layout == tensor_layout::contiguous &&
                    actual->activation_layout == tensor_layout::contiguous &&
                    actual->output_layout == tensor_layout::contiguous,
                "a kernel specialization layout contract changed");
        require(
                actual->artifact_kind == expected.artifact_kind && actual->artifact_abi_version == 1 &&
                    actual->runtime_opcode == 3 && actual->weight_bytes == expected.weight_bytes &&
                    actual->device_activation_bytes == expected.device_activation_bytes &&
                    actual->device_output_bytes == expected.device_output_bytes,
                "a kernel specialization artifact contract changed");
        require(
                actual->rows_per_worker == expected.rows_per_worker &&
                    actual->worker_count == expected.worker_count &&
                    actual->prefer_for_offload == expected.prefer_for_offload,
                "a kernel specialization execution policy changed");
    }
}

static void test_variant(const ggml_xdna::xdna_kernel_variant & variant) {
    using namespace ggml_xdna;
    ggml_context * ctx = ggml_init({ 4 * ggml_tensor_overhead(), nullptr, true });
    require(ctx != nullptr, "context allocation failed");
    ggml_tensor * weights = ggml_new_tensor_2d(ctx, weight_type(variant.weights_type), variant.k, variant.m);
    ggml_tensor * activation = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, variant.k, variant.n);
    ggml_tensor * output = ggml_mul_mat(ctx, weights, activation);
    xdna_problem problem;
    require(problem_from_ggml(output, variant.architecture, &problem), "GGML extraction failed");
    require(problem.weights_usage == weight_usage::unknown, "unallocated weights were marked immutable");
    require(kernel_variant_supports(variant, problem), "registry rejected its exact GGML problem");
    require(variant.weight_bytes == ggml_nbytes(weights), "native weight byte count mismatch");
    require(variant.device_activation_bytes == variant.k * variant.n * sizeof(uint16_t), "activation byte count mismatch");
    require(variant.device_output_bytes == ggml_nbytes(output), "output byte count mismatch");
    require(variant.rows_per_worker > 0 && variant.worker_count > 0 &&
            variant.m % (variant.rows_per_worker * variant.worker_count) == 0, "invalid worker geometry");
    const xdna_kernel_variant * inventory[] = { nullptr, &variant };
    require(select_kernel_variant(problem, inventory, 2) == &variant, "partial inventory selection failed");
    require(select_kernel_variant(problem, inventory, 1) == nullptr, "missing artifact was selected");
    require(select_kernel_variant(problem, nullptr, 0) == nullptr, "empty inventory was selected");

    const auto reject = [&](xdna_problem other) {
        require(!kernel_variant_supports(variant, other), "near-miss problem was accepted");
        require(select_kernel_variant(other, inventory, 2) == nullptr, "near-miss inventory selection succeeded");
    };
    xdna_problem other = problem;
    other.architecture = device_architecture::unknown; reject(other);
    other = problem; other.op = operation::unknown; reject(other);
    other = problem; other.weights_type = data_type::unknown; reject(other);
    other = problem; other.activation_type = data_type::bf16; reject(other);
    other = problem; other.output_type = data_type::bf16; reject(other);
    other = problem; ++other.m; reject(other);
    other = problem; ++other.n; reject(other);
    other = problem; other.k += 32; reject(other);
    for (size_t d = 0; d < 2; ++d) {
        other = problem; ++other.weights_batch[d]; reject(other);
        other = problem; ++other.activation_batch[d]; reject(other);
        other = problem; ++other.output_batch[d]; reject(other);
    }
    other = problem; other.weights_layout = tensor_layout::strided; reject(other);
    other = problem; other.activation_layout = tensor_layout::strided; reject(other);
    other = problem; other.output_layout = tensor_layout::strided; reject(other);
    other = problem; other.default_precision = false; reject(other);
    other = problem; other.src0_hadamard = true; reject(other);

    ggml_backend_buffer_t owner = ggml_backend_buffer_init(nullptr, {}, nullptr, 0);
    weights->buffer = owner;
    require(problem_from_ggml(output, variant.architecture, &other) &&
            other.weights_usage == weight_usage::mutable_buffer, "mutable weight usage was lost");
    ggml_backend_buffer_set_usage(owner, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    require(problem_from_ggml(output, variant.architecture, &other) &&
            other.weights_usage == weight_usage::immutable, "immutable weight usage was lost");
    weights->buffer = nullptr;
    ggml_backend_buffer_free(owner);

    ggml_mul_mat_set_prec(output, GGML_PREC_F32);
    require(problem_from_ggml(output, variant.architecture, &other) &&
            !kernel_variant_supports(variant, other), "explicit F32 precision was ignored");
    ggml_mul_mat_set_prec(output, GGML_PREC_DEFAULT);
    ggml_mul_mat_set_hint(output, GGML_HINT_SRC0_IS_HADAMARD);
    require(problem_from_ggml(output, variant.architecture, &other) &&
            !kernel_variant_supports(variant, other), "Hadamard hint was ignored");
    require(!problem_from_ggml(nullptr, variant.architecture, &other), "null op was accepted");
    require(!problem_from_ggml(output, variant.architecture, nullptr), "null result was accepted");
    ggml_free(ctx);
}

int main() {
    size_t variant_count = 0;
    const ggml_xdna::xdna_kernel_variant * variants = ggml_xdna::kernel_variants(&variant_count);
    require(variants != nullptr && variant_count > 0, "the kernel registry is empty");
    test_variant_inventory(variants, variant_count);

    for (size_t i = 0; i < variant_count; ++i) {
        test_variant(variants[i]);
        for (size_t j = 0; j < i; ++j) {
            require(strcmp(variants[i].id, variants[j].id) != 0, "duplicate variant identity");
            require(strcmp(variants[i].environment_variable, variants[j].environment_variable) != 0,
                    "duplicate artifact configuration key");
        }
        require(
                !ggml_xdna::kernel_variant_prefers_offload(variants[i], false),
                "a current variant requested automatic offload");
        require(
                ggml_xdna::kernel_variant_prefers_offload(variants[i], true),
                "the explicit override did not prefer a current variant");
    }

    const ggml_xdna::xdna_kernel_variant * bf16_m8192 = nullptr;
    const ggml_xdna::xdna_kernel_variant * bf16_m8192_k3072 = nullptr;
    for (size_t i = 0; i < variant_count; ++i) {
        if (strcmp(variants[i].id, "aie2p-bf16-gemv-m8192-n1-k2048") == 0) {
            bf16_m8192 = &variants[i];
        } else if (strcmp(variants[i].id, "aie2p-bf16-gemv-m8192-n1-k3072") == 0) {
            bf16_m8192_k3072 = &variants[i];
        }
    }
    require(bf16_m8192 != nullptr, "the BF16 M8192 K2048 variant is missing");
    require(
            bf16_m8192->weight_bytes == 8192 * 2048 * sizeof(uint16_t) &&
                bf16_m8192->device_activation_bytes == 2048 * sizeof(uint16_t) &&
                bf16_m8192->device_output_bytes == 8192 * sizeof(float) &&
                bf16_m8192->rows_per_worker == 8 && bf16_m8192->worker_count == 32,
            "the BF16 M8192 K2048 artifact contract changed");
    require(bf16_m8192_k3072 != nullptr, "the BF16 M8192 K3072 variant is missing");
    require(
            bf16_m8192_k3072->architecture == ggml_xdna::device_architecture::aie2p &&
                bf16_m8192_k3072->artifact_kind == 1 && bf16_m8192_k3072->artifact_abi_version == 1 &&
                bf16_m8192_k3072->weight_bytes == 8192 * 3072 * sizeof(uint16_t) &&
                bf16_m8192_k3072->device_activation_bytes == 3072 * sizeof(uint16_t) &&
                bf16_m8192_k3072->device_output_bytes == 8192 * sizeof(float) &&
                bf16_m8192_k3072->rows_per_worker == 8 && bf16_m8192_k3072->worker_count == 32 &&
                !bf16_m8192_k3072->prefer_for_offload,
            "the BF16 M8192 K3072 artifact contract changed");

    ggml_xdna::xdna_problem problem;
    problem.architecture = ggml_xdna::device_architecture::aie2p;
    problem.op = ggml_xdna::operation::mul_mat;
    problem.weights_type = ggml_xdna::data_type::bf16;
    problem.activation_type = ggml_xdna::data_type::f32;
    problem.output_type = ggml_xdna::data_type::f32;
    problem.m = 8192;
    problem.n = 1;
    problem.k = 2048;
    problem.weights_batch = { 1, 1 };
    problem.activation_batch = { 1, 1 };
    problem.output_batch = { 1, 1 };
    problem.weights_layout = ggml_xdna::tensor_layout::contiguous;
    problem.activation_layout = ggml_xdna::tensor_layout::contiguous;
    problem.output_layout = ggml_xdna::tensor_layout::contiguous;
    problem.weights_usage = ggml_xdna::weight_usage::immutable;
    problem.default_precision = true;
    problem.src0_hadamard = false;
    require(
            ggml_xdna::kernel_variant_supports(*bf16_m8192, problem),
            "the BF16 M8192 K2048 problem did not match its variant");
    const ggml_xdna::xdna_kernel_variant * candidates[] = { bf16_m8192 };
    require(
            ggml_xdna::select_kernel_variant(problem, candidates, 1) == bf16_m8192,
            "generic selection did not choose the BF16 M8192 K2048 variant");

    problem.k = 3072;
    require(
            !ggml_xdna::kernel_variant_supports(*bf16_m8192, problem),
            "the BF16 M8192 K2048 variant accepted K3072");
    require(
            ggml_xdna::kernel_variant_supports(*bf16_m8192_k3072, problem),
            "the BF16 M8192 K3072 problem did not match its variant");
    const ggml_xdna::xdna_kernel_variant * k3072_candidates[] = { bf16_m8192, bf16_m8192_k3072 };
    require(
            ggml_xdna::select_kernel_variant(problem, k3072_candidates, 2) == bf16_m8192_k3072,
            "generic selection did not choose the BF16 M8192 K3072 variant");

    problem.n = 2;
    require(
            !ggml_xdna::kernel_variant_supports(*bf16_m8192_k3072, problem),
            "the BF16 M8192 K3072 variant accepted a batched problem");

    ggml_xdna::xdna_kernel_variant variant = {};

    require(
            !ggml_xdna::kernel_variant_prefers_offload(variant, false),
            "an unpreferred variant requested automatic offload");
    require(
            ggml_xdna::kernel_variant_prefers_offload(variant, true),
            "the explicit override did not prefer an unpreferred variant");

    variant.prefer_for_offload = true;
    require(
            ggml_xdna::kernel_variant_prefers_offload(variant, false),
            "variant metadata did not request automatic offload");

    return 0;
}

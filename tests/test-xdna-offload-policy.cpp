#include "xdna-kernel-registry.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static void require(bool condition, const char * message) {
    if (!condition) {
        fprintf(stderr, "test-xdna-offload-policy: %s\n", message);
        abort();
    }
}

int main() {
    size_t variant_count = 0;
    const ggml_xdna::xdna_kernel_variant * variants = ggml_xdna::kernel_variants(&variant_count);
    require(variants != nullptr && variant_count > 0, "the kernel registry is empty");

    for (size_t i = 0; i < variant_count; ++i) {
        require(
                !ggml_xdna::kernel_variant_prefers_offload(variants[i], false),
                "a current variant requested automatic offload");
        require(
                ggml_xdna::kernel_variant_prefers_offload(variants[i], true),
                "the explicit override did not prefer a current variant");
    }

    const ggml_xdna::xdna_kernel_variant * bf16_m8192 = nullptr;
    for (size_t i = 0; i < variant_count; ++i) {
        if (strcmp(variants[i].id, "aie2p-bf16-gemv-m8192-n1-k2048") == 0) {
            bf16_m8192 = &variants[i];
            break;
        }
    }
    require(bf16_m8192 != nullptr, "the BF16 M8192 K2048 variant is missing");
    require(
            bf16_m8192->weight_bytes == 8192 * 2048 * sizeof(uint16_t) &&
                bf16_m8192->device_activation_bytes == 2048 * sizeof(uint16_t) &&
                bf16_m8192->device_output_bytes == 8192 * sizeof(float) &&
                bf16_m8192->rows_per_worker == 8 && bf16_m8192->worker_count == 32,
            "the BF16 M8192 K2048 artifact contract changed");

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

    problem.n = 2;
    require(
            !ggml_xdna::kernel_variant_supports(*bf16_m8192, problem),
            "the BF16 M8192 K2048 variant accepted a batched problem");

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

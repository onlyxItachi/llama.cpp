#include "xdna-kernel-registry.h"

#include <cstdio>
#include <cstdlib>

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

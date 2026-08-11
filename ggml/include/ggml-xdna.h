#pragma once

#include "ggml-backend.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_backend_xdna_stats {
    uint64_t bo_creations;
    uint64_t weight_registrations;
    uint64_t weight_registration_hits;
    uint64_t sync_to_device_calls;
    uint64_t sync_from_device_calls;
    uint64_t sync_to_device_bytes;
    uint64_t sync_from_device_bytes;
    uint64_t host_copy_calls;
    uint64_t host_copy_bytes;
    uint64_t kernel_submissions;
    uint64_t kernel_time_ns;
};

GGML_BACKEND_API ggml_backend_t ggml_backend_xdna_init(int device);

GGML_BACKEND_API bool ggml_backend_is_xdna(ggml_backend_t backend);

GGML_BACKEND_API bool ggml_backend_xdna_get_stats(
        ggml_backend_t backend,
        struct ggml_backend_xdna_stats * stats);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_xdna_reg(void);

#ifdef __cplusplus
}
#endif

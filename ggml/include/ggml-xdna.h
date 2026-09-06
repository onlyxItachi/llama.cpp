#pragma once

#include "ggml-backend.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_backend_xdna_stats {
    uint64_t initialization_time_ns;
    uint64_t explicit_bo_creations;
    uint64_t explicit_bo_creation_bytes;
    uint64_t buffer_registrations;
    uint64_t buffer_registration_hits;
    uint64_t registered_bytes;
    uint64_t registration_time_ns;
    uint64_t weight_registrations;
    uint64_t weight_registration_hits;
    uint64_t weight_registered_bytes;
    uint64_t weight_sync_to_device_calls;
    uint64_t weight_sync_to_device_bytes;
    uint64_t weight_sync_to_device_time_ns;
    uint64_t sync_to_device_calls;
    uint64_t sync_from_device_calls;
    uint64_t sync_to_device_bytes;
    uint64_t sync_from_device_bytes;
    uint64_t sync_to_device_time_ns;
    uint64_t sync_from_device_time_ns;
    uint64_t host_copy_calls;
    uint64_t host_copy_bytes;
    uint64_t host_copy_time_ns;
    uint64_t weight_copy_bytes;
    uint64_t kernel_submissions; // Successful start returns, including runs whose wait fails.
    uint64_t first_kernel_time_ns;
    uint64_t first_compute_time_ns;
    uint64_t kernel_time_ns; // All start/wait attempts; excludes abort.
    uint64_t total_compute_time_ns; // Successful compute calls only.
};

struct ggml_backend_xdna_stats_v2 {
    struct ggml_backend_xdna_stats base;
    uint64_t successful_compute_calls;
    uint64_t activation_pack_calls;
    uint64_t activation_pack_input_bytes;
    uint64_t activation_pack_output_bytes;
    uint64_t activation_pack_time_ns;
    uint64_t activation_sync_calls;
    uint64_t activation_sync_bytes;
    uint64_t activation_sync_time_ns;
    uint64_t run_start_calls; // API attempts, including exceptions.
    uint64_t run_start_time_ns;
    uint64_t run_wait_calls; // API attempts, including exceptions.
    uint64_t run_wait_time_ns;
    uint64_t first_run_start_time_ns;
    uint64_t first_run_wait_time_ns;
    uint64_t output_sync_calls;
    uint64_t output_sync_bytes;
    uint64_t output_sync_time_ns;
    uint64_t output_copy_calls;
    uint64_t output_copy_bytes;
    uint64_t output_copy_time_ns;
};

typedef bool (*ggml_backend_xdna_get_stats_t)(
        ggml_backend_t backend,
        struct ggml_backend_xdna_stats * stats);

typedef bool (*ggml_backend_xdna_get_stats_v2_t)(
        ggml_backend_t backend,
        struct ggml_backend_xdna_stats_v2 * stats,
        size_t stats_size);

GGML_BACKEND_API ggml_backend_t ggml_backend_xdna_init(int device);

GGML_BACKEND_API bool ggml_backend_is_xdna(ggml_backend_t backend);

GGML_BACKEND_API bool ggml_backend_xdna_get_stats(
        ggml_backend_t backend,
        struct ggml_backend_xdna_stats * stats);

GGML_BACKEND_API bool ggml_backend_xdna_get_stats_v2(
        ggml_backend_t backend,
        struct ggml_backend_xdna_stats_v2 * stats,
        size_t stats_size);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_xdna_reg(void);

#ifdef __cplusplus
}
#endif

#include <ggml-cpp.h>
#include <ggml-cpu.h>
#include <ggml-xdna.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <sys/prctl.h>
#include <vector>

int main(int argc, char ** argv) {
    if (prctl(PR_SET_DUMPABLE, 0) != 0) {
        std::perror("FAIL: disabling process core dumps");
        return 1;
    }
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc != 4) {
        std::fputs("usage: failure-test XDNA_MODULE CPU_MODULE MODE\n", stderr);
        return 2;
    }
    const bool start_failure = std::strcmp(argv[3], "start-throw") == 0;
    const bool wait_failure = std::strcmp(argv[3], "wait-throw") == 0;
    const bool expect_failstop = !start_failure && !wait_failure;
    auto reset = reinterpret_cast<bool (*)(const char *)>(dlsym(RTLD_DEFAULT, "xdna_failure_test_reset"));
    auto snapshot = reinterpret_cast<bool (*)(uint64_t *, size_t, int *)>(
        dlsym(RTLD_DEFAULT, "xdna_failure_test_snapshot"));
    if (!reset || !snapshot) {
        std::fputs("FAIL: required post-completion injection shim is not loaded\n", stderr);
        return 1;
    }
    ggml_backend_reg_t xdna_reg = ggml_backend_load(argv[1]);
    ggml_backend_reg_t cpu_reg = ggml_backend_load(argv[2]);
    if (!xdna_reg || !cpu_reg || std::strcmp(ggml_backend_reg_name(xdna_reg), "XDNA") != 0 ||
            std::strcmp(ggml_backend_reg_name(cpu_reg), "CPU") != 0 ||
            ggml_backend_reg_dev_count(xdna_reg) == 0 || ggml_backend_reg_dev_count(cpu_reg) == 0) {
        std::fputs("FAIL: explicit XDNA/CPU modules or devices unavailable\n", stderr);
        return 1;
    }
    ggml_backend_ptr xdna(ggml_backend_dev_init(ggml_backend_reg_dev_get(xdna_reg, 0), nullptr));
    ggml_backend_ptr cpu(ggml_backend_dev_init(ggml_backend_reg_dev_get(cpu_reg, 0), nullptr));
    auto get_stats = reinterpret_cast<ggml_backend_xdna_get_stats_v2_t>(
        ggml_backend_reg_get_proc_address(xdna_reg, "ggml_backend_xdna_get_stats_v2"));
    auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
        ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_set_n_threads"));
    if (!xdna || !cpu || !get_stats || !set_threads) {
        std::fputs("FAIL: backend initialization or statistics procedure\n", stderr);
        return 1;
    }
    set_threads(cpu.get(), 1);
    constexpr int64_t m = 288, k = 288;
    constexpr size_t graph_size = 8;
    ggml_context_ptr weight_ctx(ggml_init({ggml_tensor_overhead(), nullptr, true}));
    ggml_context_ptr work_ctx(ggml_init({
        3 * ggml_tensor_overhead() + 2 * ggml_graph_overhead_custom(graph_size, false), nullptr, true}));
    if (!weight_ctx || !work_ctx) {
        return 1;
    }
    ggml_tensor * weight = ggml_new_tensor_2d(weight_ctx.get(), GGML_TYPE_BF16, k, m);
    ggml_tensor * activation = ggml_new_tensor_2d(work_ctx.get(), GGML_TYPE_F32, k, 1);
    ggml_tensor * actual = ggml_mul_mat(work_ctx.get(), weight, activation);
    ggml_tensor * expected = ggml_mul_mat(work_ctx.get(), weight, activation);
    ggml_set_name(weight, "immutable_fault_weights");
    ggml_set_name(actual, "post_completion_retry");
    ggml_set_name(expected, "cpu_reference");
    ggml_cgraph * xdna_graph = ggml_new_graph_custom(work_ctx.get(), graph_size, false);
    ggml_cgraph * cpu_graph = ggml_new_graph_custom(work_ctx.get(), graph_size, false);
    ggml_build_forward_expand(xdna_graph, actual);
    ggml_build_forward_expand(cpu_graph, expected);
    ggml_backend_buffer_ptr weight_buffer(ggml_backend_alloc_ctx_tensors(weight_ctx.get(), cpu.get()));
    ggml_backend_buffer_ptr work_buffer(ggml_backend_alloc_ctx_tensors(work_ctx.get(), cpu.get()));
    if (!weight_buffer || !work_buffer) {
        return 1;
    }
    ggml_backend_buffer_set_usage(weight_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    if (!ggml_backend_supports_op(xdna.get(), actual) || !ggml_backend_supports_op(cpu.get(), expected)) {
        std::fputs("FAIL: required BF16 288x288 operation is unsupported\n", stderr);
        return 1;
    }
    std::vector<ggml_bf16_t> weights(m * k);
    std::vector<float> activations(k), output(m), reference(m), first_reference(m);
    for (int64_t row = 0; row < m; ++row) {
        for (int64_t col = 0; col < k; ++col) {
            weights[row * k + col] = ggml_fp32_to_bf16(float(row % 5 - 2) / 16 + float(col % 7 - 3) / 32);
        }
    }
    for (int64_t col = 0; col < k; ++col) {
        activations[col] = float(col % 11 - 5) / 8;
    }
    ggml_backend_tensor_set(weight, weights.data(), 0, ggml_nbytes(weight));
    const std::vector<uint32_t> poison(m, UINT32_C(0x7fc00000));
    std::vector<uint32_t> raw_output(m);
    ggml_backend_xdna_stats_v2 baseline{};
    if (!get_stats(xdna.get(), &baseline, sizeof(baseline))) {
        return 1;
    }
    if (!reset(argv[3])) {
        std::fputs("FAIL: unknown failure mode\n", stderr);
        return 2;
    }
    std::printf("mode=%s physical_in_flight_failure=false expected_failstop=%d\n", argv[3], int(expect_failstop));
    bool passed = true;
    auto require = [&](bool condition, const char * label) {
        if (!condition) {
            std::fprintf(stderr, "FAIL: %s\n", label);
            passed = false;
        }
    };
    auto delta = [&](const char * label, uint64_t before, uint64_t after, uint64_t wanted) {
        if (after < before || after - before != wanted) {
            std::fprintf(stderr, "FAIL: %s before=%" PRIu64 " after=%" PRIu64 " expected_delta=%" PRIu64 "\n",
                label, before, after, wanted);
            passed = false;
        }
    };
    for (int call = 0; call < 2; ++call) {
        if (call == 1) {
            for (float & value : activations) {
                value = -value;
            }
        }
        ggml_backend_tensor_set(activation, activations.data(), 0, ggml_nbytes(activation));
        ggml_backend_tensor_set(actual, poison.data(), 0, ggml_nbytes(actual));
        ggml_backend_tensor_set(expected, poison.data(), 0, ggml_nbytes(expected));
        if (ggml_backend_graph_compute(cpu.get(), cpu_graph) != GGML_STATUS_SUCCESS) {
            std::fputs("FAIL: CPU reference execution\n", stderr);
            return 1;
        }
        ggml_backend_tensor_get(expected, reference.data(), 0, ggml_nbytes(expected));
        const ggml_status status = ggml_backend_graph_compute(xdna.get(), xdna_graph);
        if (expect_failstop) {
            std::fputs("FAIL: the requested unprovable state returned without fail-stop\n", stderr);
            return 1;
        }
        ggml_backend_tensor_get(actual, raw_output.data(), 0, ggml_nbytes(actual));
        if (call == 0) {
            first_reference = reference;
            require(status == GGML_STATUS_FAILED, "first compute must report failure");
            require(raw_output == poison, "failed compute must not copy output over poison");
        } else {
            require(status == GGML_STATUS_SUCCESS, "same-backend retry must succeed");
            ggml_backend_tensor_get(actual, output.data(), 0, ggml_nbytes(actual));
            bool matched = true, changed = false;
            float max_error = 0;
            for (int64_t row = 0; row < m; ++row) {
                const float error = std::fabs(output[row] - reference[row]);
                matched = matched && std::isfinite(output[row]) && std::isfinite(reference[row]) &&
                    error <= 5e-4f + 5e-4f * std::fabs(reference[row]) && reference[row] == -first_reference[row];
                changed = changed || reference[row] != first_reference[row];
                max_error = std::max(max_error, error);
            }
            require(matched && changed, "retry with changed activation must match changed CPU reference");
            std::printf("retry_max_error=%.9g reference_changed=%d\n", max_error, int(changed));
        }
        ggml_backend_xdna_stats_v2 stats{};
        if (get_stats(xdna.get(), &stats, sizeof(stats))) {
            const uint64_t calls = call + 1;
            const uint64_t observed_submissions = calls - uint64_t(start_failure);
            delta("submissions", baseline.base.kernel_submissions, stats.base.kernel_submissions, observed_submissions);
            delta("start calls", baseline.run_start_calls, stats.run_start_calls, calls);
            delta("wait calls", baseline.run_wait_calls, stats.run_wait_calls, observed_submissions);
            delta("successful computes", baseline.successful_compute_calls, stats.successful_compute_calls, call);
            delta("output copies", baseline.output_copy_calls, stats.output_copy_calls, call);
            delta("output syncs", baseline.output_sync_calls, stats.output_sync_calls, call);
            delta("output copy bytes", baseline.output_copy_bytes, stats.output_copy_bytes, call * ggml_nbytes(actual));
            delta("roots", baseline.base.buffer_registrations, stats.base.buffer_registrations, 1);
            delta("root hits", baseline.base.buffer_registration_hits, stats.base.buffer_registration_hits, 0);
            delta("weight views", baseline.base.weight_registrations, stats.base.weight_registrations, 1);
            delta("weight hits", baseline.base.weight_registration_hits, stats.base.weight_registration_hits, call);
            delta("weight sync calls", baseline.base.weight_sync_to_device_calls, stats.base.weight_sync_to_device_calls, 1);
            delta("weight sync bytes", baseline.base.weight_sync_to_device_bytes, stats.base.weight_sync_to_device_bytes, ggml_nbytes(weight));
            delta("immutable copy bytes", baseline.base.weight_copy_bytes, stats.base.weight_copy_bytes, 0);
            std::printf("call=%d status=%d submissions=%" PRIu64 " starts=%" PRIu64 " waits=%" PRIu64
                " successful=%" PRIu64 " output_copies=%" PRIu64 " weight_hits=%" PRIu64 "\n",
                call + 1, int(status), stats.base.kernel_submissions, stats.run_start_calls, stats.run_wait_calls,
                stats.successful_compute_calls, stats.output_copy_calls, stats.base.weight_registration_hits);
        } else {
            require(false, "statistics retrieval after compute");
        }
        std::array<uint64_t, 7> counts{};
        int abort_result = -1;
        require(snapshot(counts.data(), counts.size(), &abort_result), "shim snapshot");
        const uint64_t starts = uint64_t(call + 1);
        const std::array<uint64_t, 7> wanted = start_failure ?
            std::array<uint64_t, 7>{starts, uint64_t(call), 0, 1, starts, 1, 1} :
            std::array<uint64_t, 7>{starts, starts, 1, 1, starts, 0, 0};
        require(counts == wanted, "exact real starts/waits, quiescence check, completion and injection counts");
        require(abort_result == (start_failure ? -1 : 4), "abort absent or real already-completed abort result");
        std::printf("shim_call=%d starts=%" PRIu64 " waits=%" PRIu64 " aborts=%" PRIu64
            " injections=%" PRIu64 " completed_waits=%" PRIu64 " state_queries=%" PRIu64
            " direct_waits=%" PRIu64 " abort_state=%d\n",
            call + 1, counts[0], counts[1], counts[2], counts[3], counts[4], counts[5], counts[6], abort_result);
    }
    std::puts(passed ? "PASS: post-completion failure accounting and persistent retry; NOT in-flight recovery" :
        "FAIL: post-completion failure accounting or retry; NOT in-flight recovery");
    return passed ? 0 : 1;
}

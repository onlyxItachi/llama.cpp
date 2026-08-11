#include "ggml-backend.h"
#include "ggml-cpp.h"

#include "../ggml/src/ggml-backend-impl.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

struct mock_device_context {
    const char * name;
    enum ggml_backend_dev_type type;
    ggml_backend_buffer_type_t buft;
    ggml_backend_buffer_type_t host_buft;
    ggml_backend_buffer_type_t compatible_buft;
    ggml_backend_buffer_type_t scheduler_buft;
    bool offload_sqr;
};

static const char * mock_buffer_type_name(ggml_backend_buffer_type_t) {
    return "MockDevice";
}

static void mock_buffer_free(ggml_backend_buffer_t buffer) {
    free(buffer->context);
}

static void * mock_buffer_base(ggml_backend_buffer_t buffer) {
    return buffer->context;
}

static void mock_buffer_memset(ggml_backend_buffer_t, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((char *) tensor->data + offset, value, size);
}

static void mock_buffer_set(
        ggml_backend_buffer_t, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    memcpy((char *) tensor->data + offset, data, size);
}

static void mock_buffer_get(
        ggml_backend_buffer_t, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy(data, (const char *) tensor->data + offset, size);
}

static void mock_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

static ggml_backend_buffer_t mock_buffer_type_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = malloc(size);
    if (data == nullptr) {
        return nullptr;
    }

    static const ggml_backend_buffer_i iface = {
        /* .free_buffer     = */ mock_buffer_free,
        /* .get_base        = */ mock_buffer_base,
        /* .init_tensor     = */ nullptr,
        /* .memset_tensor   = */ mock_buffer_memset,
        /* .set_tensor      = */ mock_buffer_set,
        /* .get_tensor      = */ mock_buffer_get,
        /* .set_tensor_2d   = */ nullptr,
        /* .get_tensor_2d   = */ nullptr,
        /* .cpy_tensor      = */ nullptr,
        /* .clear           = */ mock_buffer_clear,
        /* .reset           = */ nullptr,
    };

    return ggml_backend_buffer_init(buft, iface, data, size);
}

static size_t mock_buffer_type_alignment(ggml_backend_buffer_type_t) {
    return alignof(std::max_align_t);
}

static bool mock_buffer_type_is_host(ggml_backend_buffer_type_t) {
    return true;
}

static const char * mock_device_name(ggml_backend_dev_t dev) {
    return static_cast<mock_device_context *>(dev->context)->name;
}

static const char * mock_device_description(ggml_backend_dev_t dev) {
    return mock_device_name(dev);
}

static enum ggml_backend_dev_type mock_device_type(ggml_backend_dev_t dev) {
    return static_cast<mock_device_context *>(dev->context)->type;
}

static ggml_backend_buffer_type_t mock_device_buffer_type(ggml_backend_dev_t dev) {
    return static_cast<mock_device_context *>(dev->context)->buft;
}

static ggml_backend_buffer_type_t mock_device_host_buffer_type(ggml_backend_dev_t dev) {
    return static_cast<mock_device_context *>(dev->context)->host_buft;
}

static bool mock_device_supports_op(ggml_backend_dev_t, const ggml_tensor *) {
    return true;
}

static bool mock_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    const auto * ctx = static_cast<mock_device_context *>(dev->context);
    return buft == ctx->buft || buft == ctx->host_buft || buft == ctx->compatible_buft || buft == ctx->scheduler_buft;
}

static bool mock_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    return static_cast<mock_device_context *>(dev->context)->offload_sqr && op->op == GGML_OP_SQR;
}

static const ggml_backend_device_i mock_device_iface = {
    /* .get_name             = */ mock_device_name,
    /* .get_description      = */ mock_device_description,
    /* .get_memory           = */ nullptr,
    /* .get_type             = */ mock_device_type,
    /* .get_props            = */ nullptr,
    /* .init_backend         = */ nullptr,
    /* .get_buffer_type      = */ mock_device_buffer_type,
    /* .get_host_buffer_type = */ mock_device_host_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ mock_device_supports_op,
    /* .supports_buft        = */ mock_device_supports_buft,
    /* .offload_op           = */ mock_device_offload_op,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static const char * mock_backend_name(ggml_backend_t backend) {
    return mock_device_name(backend->device);
}

static const ggml_backend_i mock_backend_iface = {
    /* .get_name               = */ mock_backend_name,
    /* .free                   = */ nullptr,
    /* .set_tensor_async       = */ nullptr,
    /* .get_tensor_async       = */ nullptr,
    /* .set_tensor_2d_async    = */ nullptr,
    /* .get_tensor_2d_async    = */ nullptr,
    /* .cpy_tensor_async       = */ nullptr,
    /* .synchronize            = */ nullptr,
    /* .graph_plan_create      = */ nullptr,
    /* .graph_plan_free        = */ nullptr,
    /* .graph_plan_update      = */ nullptr,
    /* .graph_plan_compute     = */ nullptr,
    /* .graph_compute          = */ nullptr,
    /* .event_record           = */ nullptr,
    /* .event_wait             = */ nullptr,
    /* .graph_optimize         = */ nullptr,
};

static ggml_backend_buffer_type make_mock_buffer_type(bool host) {
    return {
        /* .iface   = */ {
            /* .get_name         = */ mock_buffer_type_name,
            /* .alloc_buffer     = */ mock_buffer_type_alloc,
            /* .get_alignment    = */ mock_buffer_type_alignment,
            /* .get_max_size     = */ nullptr,
            /* .get_alloc_size   = */ nullptr,
            /* .is_host          = */ host ? mock_buffer_type_is_host : nullptr,
        },
        /* .device  = */ nullptr,
        /* .context = */ nullptr,
    };
}

struct mock_backend_storage {
    mock_device_context context;
    ggml_backend_device device;
    ggml_backend backend;

    mock_backend_storage(
            const char * name,
            enum ggml_backend_dev_type type,
            ggml_backend_buffer_type_t buft,
            ggml_backend_buffer_type_t host_buft,
            bool offload_sqr = false) :
        context { name, type, buft, host_buft, nullptr, nullptr, offload_sqr },
        device { mock_device_iface, nullptr, &context },
        backend { guid(), mock_backend_iface, &device, nullptr } {
    }

private:
    static ggml_guid_t guid() {
        static ggml_guid value = {
            0x6d, 0x6f, 0x63, 0x6b, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
        };
        return &value;
    }
};

static void require(bool condition, const char * message) {
    if (!condition) {
        fprintf(stderr, "test-backend-sched: %s\n", message);
        abort();
    }
}

static void test_cpu_mapped_buffer_type_identity() {
    alignas(64) unsigned char storage[64] = {};
    ggml_backend_buffer_t mapped = ggml_backend_cpu_buffer_from_ptr(storage, sizeof(storage));
    require(mapped != nullptr, "failed to wrap CPU mapped test storage");
    require(
            ggml_backend_buft_is_cpu_mapped(ggml_backend_buffer_get_type(mapped)),
            "CPU_Mapped buffer type was not identified");
    require(
            !ggml_backend_buft_is_cpu_mapped(ggml_backend_cpu_buffer_type()),
            "ordinary CPU buffer type was misidentified as CPU_Mapped");

    ggml_backend_buffer_type fake_host = make_mock_buffer_type(true);
    require(
            !ggml_backend_buft_is_cpu_mapped(&fake_host),
            "unrelated host buffer type was misidentified as CPU_Mapped");
    ggml_backend_buffer_free(mapped);
}

struct buffer_observer_test_state {
    int sequence = 0;
    int provider_calls = 0;
    int provider_order = 0;
    int observer_calls[3] = {};
    int observer_order[3] = {};
};

struct buffer_observer_test_data {
    buffer_observer_test_state * state;
    size_t index;
};

static void buffer_observer_test_provider_free(ggml_backend_buffer_t buffer) {
    auto * state = static_cast<buffer_observer_test_state *>(buffer->context);
    state->provider_calls++;
    state->provider_order = ++state->sequence;
}

static void buffer_observer_test_callback(ggml_backend_buffer_t buffer, void * user_data) {
    auto * data = static_cast<buffer_observer_test_data *>(user_data);
    require(buffer->context == data->state, "observer received the wrong buffer");
    require(buffer->free_observers == nullptr, "observer list was not detached before notification");
    require(buffer->is_freeing, "buffer was not marked as freeing before notification");
    require(data->state->provider_calls == 0, "provider freed its allocation before observer notification");
    data->state->observer_calls[data->index]++;
    data->state->observer_order[data->index] = ++data->state->sequence;
}

static void test_buffer_free_observers() {
    ggml_backend_buffer_type buft = make_mock_buffer_type(true);
    buffer_observer_test_state state;
    buffer_observer_test_data observer_data[] = {
        { &state, 0 },
        { &state, 1 },
        { &state, 2 },
    };

    static const ggml_backend_buffer_i iface = {
        /* .free_buffer     = */ buffer_observer_test_provider_free,
        /* .get_base        = */ nullptr,
        /* .init_tensor     = */ nullptr,
        /* .memset_tensor   = */ nullptr,
        /* .set_tensor      = */ nullptr,
        /* .get_tensor      = */ nullptr,
        /* .set_tensor_2d   = */ nullptr,
        /* .get_tensor_2d   = */ nullptr,
        /* .cpy_tensor      = */ nullptr,
        /* .clear           = */ nullptr,
        /* .reset           = */ nullptr,
    };

    ggml_backend_buffer_t buffer = ggml_backend_buffer_init(&buft, iface, &state, 0);
    require(buffer != nullptr, "failed to create observer test buffer");

    ggml_backend_buffer_observer_t observer_0 =
            ggml_backend_buffer_add_free_observer(buffer, buffer_observer_test_callback, &observer_data[0]);
    ggml_backend_buffer_observer_t observer_1 =
            ggml_backend_buffer_add_free_observer(buffer, buffer_observer_test_callback, &observer_data[1]);
    ggml_backend_buffer_observer_t observer_2 =
            ggml_backend_buffer_add_free_observer(buffer, buffer_observer_test_callback, &observer_data[2]);
    require(observer_0 != nullptr && observer_1 != nullptr && observer_2 != nullptr,
            "failed to create buffer observers");

    ggml_backend_buffer_remove_free_observer(observer_1);
    ggml_backend_buffer_free(buffer);

    require(state.observer_calls[0] == 1, "first observer was not called exactly once");
    require(state.observer_calls[1] == 0, "removed observer was called");
    require(state.observer_calls[2] == 1, "second active observer was not called exactly once");
    require(state.provider_calls == 1, "provider free callback was not called exactly once");
    require(state.observer_order[0] < state.provider_order && state.observer_order[2] < state.provider_order,
            "provider free callback ran before an observer");
}

enum class memory_kind {
    host,
    device,
};

enum class tensor_kind {
    compute,
    weights,
};

static void test_shared_input(
        enum ggml_backend_dev_type target_type,
        enum ggml_backend_dev_type source_type,
        memory_kind source_memory,
        memory_kind target_memory,
        tensor_kind tensor,
        bool preallocated = true) {
    const bool source_uses_host_memory = source_memory == memory_kind::host;
    const bool target_uses_host_memory = target_memory == memory_kind::host;
    const bool weights = tensor == tensor_kind::weights;
    ggml_backend_buffer_type target_buft = make_mock_buffer_type(false);
    ggml_backend_buffer_type host_buft = make_mock_buffer_type(true);
    ggml_backend_buffer_type target_host_buft = make_mock_buffer_type(true);
    ggml_backend_buffer_type source_buft = make_mock_buffer_type(false);

    mock_backend_storage cpu_backend("MockCPU", GGML_BACKEND_DEVICE_TYPE_CPU, &host_buft, &host_buft);
    host_buft.device = &cpu_backend.device;

    const char * target_name = target_type == GGML_BACKEND_DEVICE_TYPE_GPU ? "MockGPU" :
            target_type == GGML_BACKEND_DEVICE_TYPE_IGPU ? "MockIGPU" : "MockCPU";
    mock_backend_storage target_backend(target_name, target_type, &target_buft, &host_buft);
    target_buft.device = &target_backend.device;
    target_host_buft.device = &target_backend.device;
    target_backend.context.scheduler_buft = &target_host_buft;

    ggml_backend_buffer_type_t source_alloc_buft = source_uses_host_memory ? &host_buft : &source_buft;
    mock_backend_storage source_backend_storage(
            "MockACCEL", GGML_BACKEND_DEVICE_TYPE_ACCEL, source_alloc_buft, &host_buft);
    source_buft.device = &source_backend_storage.device;
    target_backend.context.compatible_buft = &source_buft;
    ggml_backend_t source_backend = source_type == GGML_BACKEND_DEVICE_TYPE_CPU ?
            &cpu_backend.backend : &source_backend_storage.backend;

    ggml_backend_buffer_type_t target_sched_buft = target_uses_host_memory ? &target_host_buft : &target_buft;
    ggml_backend_t backends[3] = { &target_backend.backend, nullptr, &cpu_backend.backend };
    ggml_backend_buffer_type_t bufts[3] = { target_sched_buft, nullptr, &host_buft };
    int n_backends = 2;
    if (target_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
        require(
                source_type == GGML_BACKEND_DEVICE_TYPE_ACCEL,
                "CPU target test requires a different source backend");
        backends[0] = source_backend;
        backends[1] = &target_backend.backend;
        bufts[0] = source_alloc_buft;
        bufts[1] = target_sched_buft;
    } else if (source_type == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
        backends[1] = source_backend;
        bufts[1] = source_alloc_buft;
        n_backends = 3;
    } else {
        backends[1] = &cpu_backend.backend;
        bufts[1] = &host_buft;
    }

    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, n_backends, 16, false, false);

    ggml_init_params params = {
        /* .mem_size   = */ 8 * ggml_tensor_overhead() + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr ctx = ggml_context_ptr(ggml_init(params));
    require(ctx != nullptr, "failed to create graph context");

    ggml_tensor * source;
    ggml_backend_buffer_t source_buffer = nullptr;
    if (weights) {
        require(preallocated, "weight test requires a buffer with WEIGHTS usage");
        source = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 32);
        ggml_set_name(source, "weight");
        source_buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), source_alloc_buft);
        require(source_buffer != nullptr, "failed to allocate weight buffer");
        ggml_backend_buffer_set_usage(source_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    } else {
        ggml_tensor * input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 32);
        ggml_set_name(input, "input");
        source = ggml_sqr(ctx.get(), input);
        ggml_set_name(source, "host_compute");
        if (preallocated) {
            source_buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), source_alloc_buft);
            require(source_buffer != nullptr, "failed to allocate compute buffer");
            ggml_backend_buffer_set_usage(source_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        }
        ggml_backend_sched_set_tensor_backend(sched, input, source_backend);
    }

    ggml_tensor * consumer = ggml_sqr(ctx.get(), source);
    ggml_set_name(consumer, "device_consumer");
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, consumer);

    ggml_backend_sched_set_tensor_backend(sched, source, source_backend);
    ggml_backend_sched_set_tensor_backend(sched, consumer, &target_backend.backend);

    ggml_backend_buffer_type_t effective_source_buft =
            source_buffer != nullptr ? ggml_backend_buffer_get_type(source_buffer) : source_alloc_buft;
    require(
            ggml_backend_buft_is_host(effective_source_buft) == source_uses_host_memory,
            "source buffer memory kind does not match the test case");
    require(
            ggml_backend_buft_is_host(target_sched_buft) == target_uses_host_memory,
            "target buffer memory kind does not match the test case");
    require(
            ggml_backend_supports_buft(&target_backend.backend, effective_source_buft),
            "target does not support the source buffer type");

    ggml_backend_sched_split_graph(sched, graph);
    require(
            ggml_backend_sched_get_tensor_backend(sched, source) == source_backend,
            "source backend assignment changed");
    require(
            ggml_backend_sched_get_tensor_backend(sched, consumer) == &target_backend.backend,
            "consumer backend assignment changed");
    const bool target_is_gpu =
            target_type == GGML_BACKEND_DEVICE_TYPE_GPU || target_type == GGML_BACKEND_DEVICE_TYPE_IGPU;
    const bool expect_copy = target_is_gpu && source_uses_host_memory && !target_uses_host_memory && !weights;
    if (expect_copy) {
        require(consumer->src[0] != source, "transient host compute input was not materialized");
        require(
                strncmp(
                        consumer->src[0]->name,
                        target_backend.context.name,
                        strlen(target_backend.context.name)) == 0,
                "transient copy was not assigned to the target backend");
    } else {
        require(consumer->src[0] == source, "compatible shared input was copied");
    }

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(source_buffer);
}

static void test_accel_offload_preference(
        enum ggml_backend_dev_type target_type,
        bool scheduler_allows_offload,
        bool accel_requests_offload) {
    ggml_backend_buffer_type target_buft = make_mock_buffer_type(false);
    ggml_backend_buffer_type host_buft = make_mock_buffer_type(true);

    mock_backend_storage cpu_backend("MockCPU", GGML_BACKEND_DEVICE_TYPE_CPU, &host_buft, &host_buft);
    host_buft.device = &cpu_backend.device;
    mock_backend_storage target_backend(
            target_type == GGML_BACKEND_DEVICE_TYPE_GPU ? "MockGPU" : "MockIGPU",
            target_type,
            &target_buft,
            &host_buft);
    target_buft.device = &target_backend.device;
    mock_backend_storage accel_backend(
            "MockACCEL",
            GGML_BACKEND_DEVICE_TYPE_ACCEL,
            &host_buft,
            &host_buft,
            accel_requests_offload);

    ggml_backend_t backends[] = { &target_backend.backend, &accel_backend.backend, &cpu_backend.backend };
    ggml_backend_buffer_type_t bufts[] = {
        &target_buft,
        &host_buft,
        &host_buft,
    };
    ggml_backend_sched_t sched =
            ggml_backend_sched_new(backends, bufts, 3, 16, false, scheduler_allows_offload);

    ggml_init_params params = {
        /* .mem_size   = */ 4 * ggml_tensor_overhead() + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr ctx = ggml_context_ptr(ggml_init(params));
    require(ctx != nullptr, "failed to create graph context");

    ggml_tensor * weight = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 32);
    ggml_set_name(weight, "weight");
    ggml_backend_buffer_t weight_buffer =
            ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), &host_buft);
    require(weight_buffer != nullptr, "failed to allocate weight buffer");
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    ggml_tensor * consumer = ggml_sqr(ctx.get(), weight);
    ggml_set_name(consumer, "offload_candidate");
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, consumer);
    ggml_backend_sched_split_graph(sched, graph);

    const bool expect_accel = scheduler_allows_offload && accel_requests_offload;
    require(
            ggml_backend_sched_get_tensor_backend(sched, weight) == &target_backend.backend,
            "shared weight did not retain first-priority affinity");
    require(
            ggml_backend_sched_get_tensor_backend(sched, consumer) ==
                    (expect_accel ? &accel_backend.backend : &target_backend.backend),
            expect_accel ? "ACCEL offload request was ignored" : "ACCEL was selected without both offload gates");
    require(consumer->src[0] == weight, "immutable shared weight was copied during offload selection");

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(weight_buffer);
}

int main() {
    test_buffer_free_observers();
    test_cpu_mapped_buffer_type_identity();

    const enum ggml_backend_dev_type target_types[] = {
        GGML_BACKEND_DEVICE_TYPE_GPU,
        GGML_BACKEND_DEVICE_TYPE_IGPU,
    };
    const enum ggml_backend_dev_type source_types[] = {
        GGML_BACKEND_DEVICE_TYPE_CPU,
        GGML_BACKEND_DEVICE_TYPE_ACCEL,
    };

    for (enum ggml_backend_dev_type target_type : target_types) {
        for (enum ggml_backend_dev_type source_type : source_types) {
            test_shared_input(
                    target_type, source_type, memory_kind::host, memory_kind::device, tensor_kind::compute);
            test_shared_input(
                    target_type, source_type, memory_kind::host, memory_kind::device, tensor_kind::weights);
            test_shared_input(
                    target_type, source_type, memory_kind::host, memory_kind::host, tensor_kind::compute);
        }
        test_shared_input(
                target_type,
                GGML_BACKEND_DEVICE_TYPE_ACCEL,
                memory_kind::device,
                memory_kind::device,
                tensor_kind::compute);
        test_shared_input(
                target_type,
                GGML_BACKEND_DEVICE_TYPE_ACCEL,
                memory_kind::host,
                memory_kind::device,
                tensor_kind::compute,
                false);
        test_accel_offload_preference(target_type, true, true);
        test_accel_offload_preference(target_type, false, true);
        test_accel_offload_preference(target_type, true, false);
    }
    test_shared_input(
            GGML_BACKEND_DEVICE_TYPE_CPU,
            GGML_BACKEND_DEVICE_TYPE_ACCEL,
            memory_kind::host,
            memory_kind::host,
            tensor_kind::compute);

    return 0;
}

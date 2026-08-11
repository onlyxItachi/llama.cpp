#include "ggml-xdna.h"

#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ggml.h"
#include "xdna-runtime.h"

#include <cinttypes>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

struct xdna_device_context {
    ggml_xdna::device_info info;
    ggml_xdna::kernel_configuration kernel_configuration;
    std::string name;
    std::string description;
};

struct xdna_backend_context {
    xdna_backend_context(
            const ggml_xdna::device_info & info,
            const ggml_xdna::kernel_configuration & configuration) : runtime(info, configuration) {}

    ggml_xdna::runtime runtime;
};

static ggml_guid_t ggml_backend_xdna_guid() {
    static ggml_guid guid = { 0x58, 0x44, 0x4e, 0x41, 0xf4, 0x1d, 0x4b, 0xe7, 0x98, 0x57, 0x6a, 0xc4, 0x8c, 0x26, 0x63, 0x11 };
    return &guid;
}

static const char * ggml_backend_xdna_name(ggml_backend_t backend) {
    return ggml_backend_dev_name(backend->device);
}

static void ggml_backend_xdna_free(ggml_backend_t backend) {
    auto * ctx = static_cast<xdna_backend_context *>(backend->context);
    ggml_backend_xdna_stats_v2 stats_v2 = {};
    ctx->runtime.get_stats_v2(&stats_v2);
    const ggml_backend_xdna_stats & stats = stats_v2.base;
    GGML_LOG_INFO(
            "ggml_xdna: init_ms=%.3f submissions=%" PRIu64 " first_kernel_ms=%.3f "
            "first_total_ms=%.3f kernel_ms=%.3f total_ms=%.3f "
            "root_registrations=%" PRIu64 " registration_hits=%" PRIu64 " registration_ms=%.3f "
            "registered_MiB=%.3f explicit_payload_BOs=%" PRIu64 "/%" PRIu64 "B "
            "weight_views=%" PRIu64 " weight_hits=%" PRIu64 " weight_sync=%" PRIu64 "/%" PRIu64 "B/%.3fms "
            "weight_copy_bytes=%" PRIu64 " "
            "sync_to=%" PRIu64 "/%" PRIu64 "B/%.3fms sync_from=%" PRIu64 "/%" PRIu64 "B/%.3fms "
            "host_copies=%" PRIu64 "/%" PRIu64 "B/%.3fms\n",
            stats.initialization_time_ns / 1.0e6,
            stats.kernel_submissions,
            stats.first_kernel_time_ns / 1.0e6,
            stats.first_compute_time_ns / 1.0e6,
            stats.kernel_time_ns / 1.0e6,
            stats.total_compute_time_ns / 1.0e6,
            stats.buffer_registrations,
            stats.buffer_registration_hits,
            stats.registration_time_ns / 1.0e6,
            stats.registered_bytes / (1024.0 * 1024.0),
            stats.explicit_bo_creations,
            stats.explicit_bo_creation_bytes,
            stats.weight_registrations,
            stats.weight_registration_hits,
            stats.weight_sync_to_device_calls,
            stats.weight_sync_to_device_bytes,
            stats.weight_sync_to_device_time_ns / 1.0e6,
            stats.weight_copy_bytes,
            stats.sync_to_device_calls,
            stats.sync_to_device_bytes,
            stats.sync_to_device_time_ns / 1.0e6,
            stats.sync_from_device_calls,
            stats.sync_from_device_bytes,
            stats.sync_from_device_time_ns / 1.0e6,
            stats.host_copy_calls,
            stats.host_copy_bytes,
            stats.host_copy_time_ns / 1.0e6);
    GGML_LOG_INFO(
            "ggml_xdna: successful_calls=%" PRIu64 " start=%" PRIu64 "/%.3fms wait=%" PRIu64 "/%.3fms "
            "first_start_ms=%.3f first_wait_ms=%.3f activation_pack=%" PRIu64 "/%" PRIu64 "B->%" PRIu64 "B/%.3fms "
            "activation_sync=%" PRIu64 "/%" PRIu64 "B/%.3fms output_sync=%" PRIu64 "/%" PRIu64 "B/%.3fms "
            "output_copy=%" PRIu64 "/%" PRIu64 "B/%.3fms\n",
            stats_v2.successful_compute_calls,
            stats_v2.run_start_calls,
            stats_v2.run_start_time_ns / 1.0e6,
            stats_v2.run_wait_calls,
            stats_v2.run_wait_time_ns / 1.0e6,
            stats_v2.first_run_start_time_ns / 1.0e6,
            stats_v2.first_run_wait_time_ns / 1.0e6,
            stats_v2.activation_pack_calls,
            stats_v2.activation_pack_input_bytes,
            stats_v2.activation_pack_output_bytes,
            stats_v2.activation_pack_time_ns / 1.0e6,
            stats_v2.activation_sync_calls,
            stats_v2.activation_sync_bytes,
            stats_v2.activation_sync_time_ns / 1.0e6,
            stats_v2.output_sync_calls,
            stats_v2.output_sync_bytes,
            stats_v2.output_sync_time_ns / 1.0e6,
            stats_v2.output_copy_calls,
            stats_v2.output_copy_bytes,
            stats_v2.output_copy_time_ns / 1.0e6);
    delete ctx;
    delete backend;
}

static enum ggml_status ggml_backend_xdna_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * ctx = static_cast<xdna_backend_context *>(backend->context);
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;
            case GGML_OP_MUL_MAT:
                if (ctx->runtime.compute(node) != 0) {
                    GGML_LOG_ERROR("%s: XDNA execution failed for %s\n", __func__, node->name);
                    return GGML_STATUS_FAILED;
                }
                break;
            default:
                GGML_LOG_ERROR("%s: unsupported op %s reached XDNA backend\n", __func__, ggml_op_desc(node));
                return GGML_STATUS_FAILED;
        }
    }
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_xdna_i = {
    /* .get_name                = */ ggml_backend_xdna_name,
    /* .free                    = */ ggml_backend_xdna_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ nullptr,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_xdna_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

static const char * ggml_backend_xdna_device_get_name(ggml_backend_dev_t dev) {
    auto * ctx = static_cast<xdna_device_context *>(dev->context);
    return ctx->name.c_str();
}

static const char * ggml_backend_xdna_device_get_description(ggml_backend_dev_t dev) {
    auto * ctx = static_cast<xdna_device_context *>(dev->context);
    return ctx->description.c_str();
}

static void ggml_backend_xdna_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    *free = 0;
    *total = 0;
    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_xdna_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_xdna_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    auto * ctx = static_cast<xdna_device_context *>(dev->context);
    props->name = ggml_backend_xdna_device_get_name(dev);
    props->description = ggml_backend_xdna_device_get_description(dev);
    props->type = GGML_BACKEND_DEVICE_TYPE_ACCEL;
    props->device_id = ctx->info.bdf.c_str();
    ggml_backend_xdna_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
        // The installed amdxdna driver must be able to pin PROT_READ mappings before
        // host-pointer wrapping and mmap loading can be enabled. Writable CPU buffers
        // are registered lazily by the compute runtime instead.
        /* .mmap_support         = */ false,
    };
}

static ggml_backend_t ggml_backend_xdna_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);

    auto * dev_ctx = static_cast<xdna_device_context *>(dev->context);
    try {
        auto * ctx = new xdna_backend_context(dev_ctx->info, dev_ctx->kernel_configuration);
        if (dev_ctx->kernel_configuration.available && !ctx->runtime.kernel_available()) {
            GGML_LOG_ERROR("ggml_xdna: failed to load configured kernel on %s: %s\n",
                    dev_ctx->info.name.c_str(), ctx->runtime.kernel_status().c_str());
            delete ctx;
            return nullptr;
        }
        GGML_LOG_INFO("ggml_xdna: initialized %s at %s (%s); %s\n",
                dev_ctx->info.name.c_str(), dev_ctx->info.bdf.c_str(), dev_ctx->info.architecture.c_str(),
                ctx->runtime.kernel_status().c_str());
        return new ggml_backend {
            /* .guid    = */ ggml_backend_xdna_guid(),
            /* .iface   = */ ggml_backend_xdna_i,
            /* .device  = */ dev,
            /* .context = */ ctx,
        };
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ggml_xdna: failed to initialize %s: %s\n", dev_ctx->info.name.c_str(), e.what());
        return nullptr;
    }

}

static ggml_backend_buffer_type_t ggml_backend_xdna_device_get_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cpu_buffer_type();
}

static bool ggml_backend_xdna_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    auto * dev_ctx = static_cast<xdna_device_context *>(dev->context);
    if (!dev_ctx->kernel_configuration.available) {
        return false;
    }

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            break;
    }

    ggml_xdna::xdna_problem problem;
    if (!ggml_xdna::problem_from_ggml(op, dev_ctx->info.arch, &problem)) {
        return false;
    }
    return ggml_xdna::select_kernel_configuration(dev_ctx->kernel_configuration, problem) != nullptr;
}

static bool ggml_backend_xdna_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    if (buft == ggml_backend_cpu_buffer_type()) {
        return true;
    }

    // Device-owned host buffer types are the existing GGML contract for
    // writable/pinned system memory.  In particular, this admits ROCm_Host on
    // an integrated GPU without also admitting CPU_Mapped, whose device is
    // null and whose underlying mapping may be read-only.
    ggml_backend_dev_t owner = ggml_backend_buft_get_device(buft);
    if (owner == nullptr || ggml_backend_dev_host_buffer_type(owner) != buft) {
        return false;
    }

    return ggml_backend_buft_is_host(buft);
}

static bool ggml_backend_xdna_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    // Opt into scheduler preference only for an exact registered kernel.  This
    // keeps large-batch prefill on HIP while allowing a shared host-backed
    // batch-one decode weight to select XDNA.
    if (op == nullptr || op->src[0] == nullptr || op->src[0]->buffer == nullptr ||
            !ggml_backend_xdna_device_supports_buft(dev, op->src[0]->buffer->buft)) {
        return false;
    }

    ggml_xdna::xdna_problem problem;
    auto * dev_ctx = static_cast<xdna_device_context *>(dev->context);
    return ggml_xdna::problem_from_ggml(op, dev_ctx->info.arch, &problem) &&
           problem.weights_usage == ggml_xdna::weight_usage::immutable &&
           ggml_xdna::select_kernel_configuration(dev_ctx->kernel_configuration, problem) != nullptr;
}

static const ggml_backend_device_i ggml_backend_xdna_device_i = {
    /* .get_name             = */ ggml_backend_xdna_device_get_name,
    /* .get_description      = */ ggml_backend_xdna_device_get_description,
    /* .get_memory           = */ ggml_backend_xdna_device_get_memory,
    /* .get_type             = */ ggml_backend_xdna_device_get_type,
    /* .get_props            = */ ggml_backend_xdna_device_get_props,
    /* .init_backend         = */ ggml_backend_xdna_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_xdna_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_xdna_device_supports_op,
    /* .supports_buft        = */ ggml_backend_xdna_device_supports_buft,
    /* .offload_op           = */ ggml_backend_xdna_device_offload_op,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

struct xdna_registry_context {
    explicit xdna_registry_context(ggml_backend_reg_t reg) {
        std::string error;
        auto found = ggml_xdna::discover_devices(&error);
        if (found.empty() && !error.empty()) {
            GGML_LOG_WARN("ggml_xdna: XRT device discovery failed: %s\n", error.c_str());
        }

        contexts.reserve(found.size());
        devices.reserve(found.size());
        for (size_t ordinal = 0; ordinal < found.size(); ++ordinal) {
            auto & info = found[ordinal];
            auto ctx = std::make_unique<xdna_device_context>();
            ctx->info = std::move(info);
            ctx->kernel_configuration = ggml_xdna::probe_kernel_configuration(ctx->info);
            ctx->name = "XDNA" + std::to_string(ordinal);
            ctx->description = ctx->info.name + " (" + ctx->info.architecture + ")";
            devices.push_back({
                /* .iface   = */ ggml_backend_xdna_device_i,
                /* .reg     = */ reg,
                /* .context = */ ctx.get(),
            });
            contexts.emplace_back(std::move(ctx));
        }
    }

    std::vector<std::unique_ptr<xdna_device_context>> contexts;
    std::vector<ggml_backend_device> devices;
};

static const char * ggml_backend_xdna_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "XDNA";
}

static size_t ggml_backend_xdna_reg_get_device_count(ggml_backend_reg_t reg) {
    auto * ctx = static_cast<xdna_registry_context *>(reg->context);
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_xdna_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto * ctx = static_cast<xdna_registry_context *>(reg->context);
    GGML_ASSERT(index < ctx->devices.size());
    return &ctx->devices[index];
}

static void * ggml_backend_xdna_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (std::strcmp(name, "ggml_backend_xdna_get_stats") == 0) {
        ggml_backend_xdna_get_stats_t fct = ggml_backend_xdna_get_stats;
        return (void *) fct;
    }
    if (std::strcmp(name, "ggml_backend_xdna_get_stats_v2") == 0) {
        ggml_backend_xdna_get_stats_v2_t fct = ggml_backend_xdna_get_stats_v2;
        return (void *) fct;
    }
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_xdna_reg_i = {
    /* .get_name         = */ ggml_backend_xdna_reg_get_name,
    /* .get_device_count = */ ggml_backend_xdna_reg_get_device_count,
    /* .get_device       = */ ggml_backend_xdna_reg_get_device,
    /* .get_proc_address = */ ggml_backend_xdna_reg_get_proc_address,
};

} // namespace

ggml_backend_t ggml_backend_xdna_init(int device) {
    ggml_backend_reg_t reg = ggml_backend_xdna_reg();
    if (device < 0 || static_cast<size_t>(device) >= ggml_backend_reg_dev_count(reg)) {
        GGML_LOG_ERROR("ggml_xdna: invalid device index %d\n", device);
        return nullptr;
    }
    return ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, static_cast<size_t>(device)), nullptr);
}

bool ggml_backend_is_xdna(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_xdna_guid());
}

bool ggml_backend_xdna_get_stats(ggml_backend_t backend, ggml_backend_xdna_stats * stats) {
    if (!ggml_backend_is_xdna(backend) || stats == nullptr) {
        return false;
    }
    auto * ctx = static_cast<xdna_backend_context *>(backend->context);
    ctx->runtime.get_stats(stats);
    return true;
}

bool ggml_backend_xdna_get_stats_v2(
        ggml_backend_t backend,
        ggml_backend_xdna_stats_v2 * stats,
        size_t stats_size) {
    if (!ggml_backend_is_xdna(backend) || stats == nullptr || stats_size < sizeof(*stats)) {
        return false;
    }
    auto * ctx = static_cast<xdna_backend_context *>(backend->context);
    ctx->runtime.get_stats_v2(stats);
    return true;
}

ggml_backend_reg_t ggml_backend_xdna_reg() {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_xdna_reg_i,
        /* .context     = */ nullptr,
    };
    static xdna_registry_context ctx(&reg);
    reg.context = &ctx;
    return &reg;
}

static int ggml_backend_xdna_score() {
    return ggml_xdna::discover_devices().empty() ? 0 : 50;
}

GGML_BACKEND_DL_IMPL(ggml_backend_xdna_reg)
GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_xdna_score)

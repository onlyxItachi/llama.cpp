#include <xrt/xrt_kernel.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <stdexcept>

namespace {
enum class failure_mode {
    wait_throw, start_throw, wait_noresponse, wait_submitted, abort_noresponse,
    start_state_new, start_state_submitted, start_state_throw, prestart_throw, abort_throw,
};

failure_mode mode = failure_mode::wait_throw;
std::atomic<uint64_t> starts{0}, waits{0}, aborts{0}, injections{0}, completions{0}, states{0}, direct_waits{0};
std::atomic<bool> armed{false}, state_override{false};
std::atomic<int> abort_state{-1};

void * real_symbol(const char * name) {
    void * symbol = dlsym(RTLD_NEXT, name);
    if (!symbol) {
        // Dynamically loaded backend dependencies can be outside RTLD_NEXT's scope.
        const char * path = std::getenv("XDNA_FAILURE_XRT_LIBRARY");
        void * library = path ? dlopen(path, RTLD_NOW | RTLD_NOLOAD) : nullptr;
        if (library) {
            symbol = dlsym(library, name);
            dlclose(library);
        }
    }
    if (!symbol) {
        std::fprintf(stderr, "failure-shim: cannot resolve selected XRT symbol %s\n", name);
        std::abort();
    }
    return symbol;
}

using wait_fn = ert_cmd_state (*)(const xrt::run *, const std::chrono::milliseconds &);
wait_fn real_wait() {
    static auto fn = reinterpret_cast<wait_fn>(
        real_symbol("_ZNK3xrt3run4waitERKNSt6chrono8durationIlSt5ratioILl1ELl1000EEEE"));
    return fn;
}

using state_fn = ert_cmd_state (*)(const xrt::run *);
state_fn real_state() {
    static auto fn = reinterpret_cast<state_fn>(real_symbol("_ZNK3xrt3run5stateEv"));
    return fn;
}
}

extern "C" bool xdna_failure_test_reset(const char * name) {
    if (!name) {
        return false;
    }
    const char * names[] = { "wait-throw", "start-throw", "wait-noresponse", "wait-submitted", "abort-noresponse",
        "start-state-new", "start-state-submitted", "start-state-throw", "prestart-throw", "abort-throw" };
    size_t selected = 0;
    for (; selected < sizeof(names) / sizeof(names[0]); ++selected) {
        if (std::strcmp(name, names[selected]) == 0) {
            break;
        }
    }
    if (selected == sizeof(names) / sizeof(names[0])) {
        return false;
    }
    mode = static_cast<failure_mode>(selected);
    starts = waits = aborts = injections = completions = states = direct_waits = 0;
    abort_state = -1;
    state_override = false;
    armed = true;
    return true;
}

extern "C" bool xdna_failure_test_snapshot(uint64_t * values, size_t count, int * state) {
    if (!values || count != 7 || !state) {
        return false;
    }
    values[0] = starts.load();
    values[1] = waits.load();
    values[2] = aborts.load();
    values[3] = injections.load();
    values[4] = completions.load();
    values[5] = states.load();
    values[6] = direct_waits.load();
    *state = abort_state.load();
    return true;
}

void xrt::run::start() {
    static auto real = reinterpret_cast<void (*)(xrt::run *)>(real_symbol("_ZN3xrt3run5startEv"));
    if (mode == failure_mode::prestart_throw && armed.exchange(false)) {
        const auto state = real_state()(this);
        if (state != ERT_CMD_STATE_NEW) {
            std::fprintf(stderr, "failure-shim: prestart fixture expected actual NEW, got %d\n", int(state));
            std::abort();
        }
        injections.fetch_add(1);
        std::fputs("failure-shim: category=pre-submit phase=start actual_state=1 real_submissions=0 device_setup_required=1\n", stderr);
        throw std::runtime_error("injected pre-submission start exception; no device command submitted");
    }
    starts.fetch_add(1);
    real(this);
    const bool start_failure = mode == failure_mode::start_throw || mode == failure_mode::start_state_new ||
        mode == failure_mode::start_state_submitted || mode == failure_mode::start_state_throw;
    if (start_failure && armed.exchange(false)) {
        direct_waits.fetch_add(1);
        const auto state = real_wait()(this, std::chrono::milliseconds(0));
        if (state != ERT_CMD_STATE_COMPLETED) {
            std::fprintf(stderr, "failure-shim: cannot inject after unsuccessful real start/wait, state=%d\n", int(state));
            std::abort();
        }
        completions.fetch_add(1);
        injections.fetch_add(1);
        state_override = mode != failure_mode::start_throw;
        std::fprintf(stderr, "failure-shim: category=%c phase=start real_state=4 injected_exception=1\n",
            mode == failure_mode::start_throw ? 'B' : 'C');
        throw std::runtime_error("injected post-completion start exception; physical command already COMPLETED");
    }
}

ert_cmd_state xrt::run::wait(const std::chrono::milliseconds & timeout) const {
    waits.fetch_add(1);
    const ert_cmd_state state = real_wait()(this, timeout);
    if (state == ERT_CMD_STATE_COMPLETED) {
        completions.fetch_add(1);
        if (armed.exchange(false)) {
            injections.fetch_add(1);
            if (mode == failure_mode::wait_noresponse || mode == failure_mode::wait_submitted) {
                const auto reported = mode == failure_mode::wait_submitted ? ERT_CMD_STATE_SUBMITTED : ERT_CMD_STATE_NORESPONSE;
                std::fprintf(stderr, "failure-shim: category=C phase=wait real_state=4 synthetic_state=%d\n", int(reported));
                return reported;
            }
            std::fputs("failure-shim: category=B phase=wait real_state=4 injected_exception=1\n", stderr);
            throw std::runtime_error("injected post-completion wait exception; physical command already COMPLETED");
        }
    }
    return state;
}

ert_cmd_state xrt::run::abort() {
    static auto real = reinterpret_cast<ert_cmd_state (*)(xrt::run *)>(real_symbol("_ZN3xrt3run5abortEv"));
    aborts.fetch_add(1);
    const auto state = real(this);
    abort_state = int(state);
    if (state == ERT_CMD_STATE_COMPLETED && mode == failure_mode::abort_throw) {
        // The backend cannot generally rule out a pending internal ERT_ABORT command after an exception.
        std::fputs("failure-shim: category=B phase=abort real_state=4 injected_exception=1\n", stderr);
        throw std::runtime_error("injected abort exception after physical completion");
    }
    if (state == ERT_CMD_STATE_COMPLETED && mode == failure_mode::abort_noresponse) {
        std::fputs("failure-shim: category=C phase=abort real_state=4 synthetic_state=9\n", stderr);
        return ERT_CMD_STATE_NORESPONSE;
    }
    return state;
}

ert_cmd_state xrt::run::state() const {
    states.fetch_add(1);
    const auto state = real_state()(this);
    if (state_override.exchange(false)) {
        if (state != ERT_CMD_STATE_COMPLETED) {
            std::fputs("failure-shim: state injection requires real COMPLETED\n", stderr);
            std::abort();
        }
        if (mode == failure_mode::start_state_throw) {
            std::fputs("failure-shim: category=C phase=state real_state=4 injected_exception=1\n", stderr);
            throw std::runtime_error("injected state-query exception after physical completion");
        }
        const auto reported = mode == failure_mode::start_state_submitted ? ERT_CMD_STATE_SUBMITTED : ERT_CMD_STATE_NEW;
        std::fprintf(stderr, "failure-shim: category=C phase=state real_state=4 synthetic_state=%d\n", int(reported));
        return reported;
    }
    return state;
}

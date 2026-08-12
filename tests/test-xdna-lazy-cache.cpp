#include "xdna-lazy-cache.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

struct value {
    explicit value(int id_value) : id(id_value) {}

    int id;
};

void require(bool condition, const char * message) {
    if (!condition) {
        fprintf(stderr, "test-xdna-lazy-cache: %s\n", message);
        abort();
    }
}

void test_partial_failure_fallback() {
    ggml_xdna::detail::lazy_cache<int, value> cache(std::vector<int> { 1, 2, 3 });
    std::atomic<int> first_loads { 0 };
    std::atomic<int> second_loads { 0 };

    const ggml_xdna::detail::lazy_cache_counts inventory = cache.counts();
    require(
        inventory.untried == 3 && inventory.ready == 0 && inventory.failed == 0,
        "the host inventory did not start with three untried entries");

    auto loader = [&](int key) -> std::shared_ptr<value> {
        if (key == 1) {
            first_loads.fetch_add(1);
            throw std::runtime_error("first program rejected");
        }
        second_loads.fetch_add(1);
        return std::make_shared<value>(key);
    };

    std::shared_ptr<value> selected = cache.resolve_if([](int key) { return key <= 2; }, loader);
    require(selected != nullptr && selected->id == 2, "a failed candidate blocked the next matching program");
    require(first_loads.load() == 1 && second_loads.load() == 1, "the first resolution used an unexpected loader count");

    const ggml_xdna::detail::lazy_cache_counts first_counts = cache.counts();
    require(
        first_counts.untried == 1 && first_counts.ready == 1 && first_counts.failed == 1,
        "the first resolution did not retain untried, ready, and failed states");
    const auto failures = cache.failures();
    require(
        failures.size() == 1 && failures[0].first == 1 && failures[0].second == "first program rejected",
        "the failed program did not retain its diagnostic");

    selected = cache.resolve_if([](int key) { return key <= 2; }, loader);
    require(selected != nullptr && selected->id == 2, "the ready program was not reused");
    require(first_loads.load() == 1 && second_loads.load() == 1, "a cached program was loaded again");
}

void test_backend_local_state_is_lazy() {
    auto program = std::make_shared<value>(11);
    ggml_xdna::detail::lazy_cache<std::shared_ptr<value>, value> cache;
    std::atomic<int> loads { 0 };

    require(cache.counts().ready == 0, "a backend execution state existed before compute");
    auto loader = [&](const std::shared_ptr<value> & key) {
        loads.fetch_add(1);
        return std::make_shared<value>(key->id + 1);
    };
    std::shared_ptr<value> state = cache.resolve_or_add(program, loader);
    require(state != nullptr && state->id == 12 && loads.load() == 1, "the first compute did not create its execution state");
    require(cache.counts().ready == 1, "the backend execution state was not cached");
    require(cache.resolve_or_add(program, loader) == state && loads.load() == 1, "the backend execution state was recreated");
    const std::vector<std::shared_ptr<value>> ready = cache.ready_values();
    require(ready.size() == 1 && ready[0] == state, "the ready-state snapshot did not retain the cached value");

    auto bad_program = std::make_shared<value>(13);
    auto failing_loader = [&](const std::shared_ptr<value> &) -> std::shared_ptr<value> {
        loads.fetch_add(1);
        throw std::runtime_error("execution state failed");
    };
    require(cache.resolve_or_add(bad_program, failing_loader) == nullptr, "a failed execution state returned a value");
    require(cache.resolve_or_add(bad_program, failing_loader) == nullptr, "a failed execution state was not retained");
    require(loads.load() == 2 && cache.counts().failed == 1, "a failed execution state was retried");
}

void test_execution_state_failure_falls_back() {
    ggml_xdna::detail::lazy_cache<int, value> programs(std::vector<int> { 21, 22 });
    ggml_xdna::detail::lazy_cache<std::shared_ptr<value>, value> states;
    std::atomic<int> program_loads { 0 };
    std::atomic<int> state_loads { 0 };

    auto select = [&]() -> std::shared_ptr<value> {
        std::shared_ptr<value> previous;
        while (true) {
            bool passed_previous = previous == nullptr;
            std::shared_ptr<value> program = programs.resolve_if(
                [&](int key) {
                    if (!passed_previous) {
                        passed_previous = key == previous->id;
                        return false;
                    }
                    return true;
                },
                [&](int key) {
                    program_loads.fetch_add(1);
                    return std::make_shared<value>(key);
                });
            if (program == nullptr) {
                return nullptr;
            }

            std::shared_ptr<value> state = states.resolve_or_add(
                program,
                [&](const std::shared_ptr<value> & key) -> std::shared_ptr<value> {
                    state_loads.fetch_add(1);
                    if (key->id == 21) {
                        throw std::runtime_error("first execution state failed");
                    }
                    return std::make_shared<value>(key->id + 100);
                });
            if (state != nullptr) {
                return state;
            }
            previous = std::move(program);
        }
    };

    std::shared_ptr<value> selected = select();
    require(selected != nullptr && selected->id == 122, "a failed first execution state blocked the later program");
    require(program_loads.load() == 2 && state_loads.load() == 2, "fallback did not resolve each candidate exactly once");
    require(programs.counts().ready == 2, "the fallback programs were not retained");
    require(states.counts().ready == 1 && states.counts().failed == 1, "the fallback execution states were not retained");

    require(select() == selected, "the fallback execution state was not reused");
    require(program_loads.load() == 2 && state_loads.load() == 2, "cached fallback state was loaded again");
}

void test_concurrent_resolution() {
    ggml_xdna::detail::lazy_cache<int, value> cache(std::vector<int> { 7 });
    std::atomic<int> loads { 0 };
    std::atomic<int> ready { 0 };
    std::atomic<bool> start { false };
    std::vector<std::shared_ptr<value>> results(16);
    std::vector<std::thread> threads;
    threads.reserve(results.size());

    for (size_t i = 0; i < results.size(); ++i) {
        threads.emplace_back([&, i]() {
            ready.fetch_add(1);
            while (!start.load()) {
                std::this_thread::yield();
            }
            results[i] = cache.resolve_if(
                [](int key) { return key == 7; },
                [&](int key) {
                    loads.fetch_add(1);
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    return std::make_shared<value>(key);
                });
        });
    }
    while (ready.load() != static_cast<int>(results.size())) {
        std::this_thread::yield();
    }
    start.store(true);
    for (std::thread & thread : threads) {
        thread.join();
    }

    require(loads.load() == 1, "concurrent callers loaded one program more than once");
    require(results[0] != nullptr && results[0]->id == 7, "concurrent resolution returned no program");
    for (const auto & result : results) {
        require(result == results[0], "concurrent callers did not share the resolved program");
    }
}

void test_concurrent_failure_is_cached() {
    ggml_xdna::detail::lazy_cache<int, value> cache(std::vector<int> { 9 });
    std::atomic<int> loads { 0 };
    std::vector<std::thread> threads;
    threads.reserve(16);

    for (int i = 0; i < 16; ++i) {
        threads.emplace_back([&]() {
            std::shared_ptr<value> result = cache.resolve_if(
                [](int key) { return key == 9; },
                [&](int) -> std::shared_ptr<value> {
                    loads.fetch_add(1);
                    throw std::runtime_error("load failed");
                });
            require(result == nullptr, "a failed loader returned a program");
        });
    }
    for (std::thread & thread : threads) {
        thread.join();
    }

    const ggml_xdna::detail::lazy_cache_counts counts = cache.counts();
    require(loads.load() == 1, "concurrent callers retried a failed program");
    require(counts.failed == 1 && counts.ready == 0 && counts.untried == 0, "the failed program state was not cached");
}

void test_shared_execution_gate_serializes_backends() {
    ggml_xdna::detail::execution_gate gate;
    std::atomic<int> ready { 0 };
    std::atomic<int> active { 0 };
    std::atomic<int> visits { 0 };
    std::atomic<bool> start { false };
    std::atomic<bool> overlap { false };
    std::vector<std::thread> threads;
    threads.reserve(16);

    for (int i = 0; i < 16; ++i) {
        threads.emplace_back([&]() {
            ready.fetch_add(1);
            while (!start.load()) {
                std::this_thread::yield();
            }
            ggml_xdna::detail::execution_gate::guard guard = gate.acquire();
            if (active.fetch_add(1) != 0) {
                overlap.store(true);
            }
            visits.fetch_add(1);
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            active.fetch_sub(1);
        });
    }
    while (ready.load() != 16) {
        std::this_thread::yield();
    }
    start.store(true);
    for (std::thread & thread : threads) {
        thread.join();
    }

    require(!overlap.load() && visits.load() == 16, "the shared device execution gate allowed overlapping commands");
}

} // namespace

int main() {
    test_partial_failure_fallback();
    test_backend_local_state_is_lazy();
    test_execution_state_failure_falls_back();
    test_concurrent_resolution();
    test_concurrent_failure_is_cached();
    test_shared_execution_gate_serializes_backends();
    return 0;
}

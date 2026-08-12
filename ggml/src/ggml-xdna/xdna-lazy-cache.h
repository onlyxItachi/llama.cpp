#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ggml_xdna {
namespace detail {

enum class lazy_cache_state {
    untried,
    ready,
    failed,
};

struct lazy_cache_counts {
    size_t untried = 0;
    size_t ready = 0;
    size_t failed = 0;
};

class execution_gate {
public:
    using guard = std::unique_lock<std::mutex>;

    guard acquire() {
        return guard(mutex);
    }

private:
    std::mutex mutex;
};

template<typename Key, typename Value>
class lazy_cache {
public:
    lazy_cache() = default;

    explicit lazy_cache(std::vector<Key> keys) {
        entries.reserve(keys.size());
        for (Key & key : keys) {
            entries.emplace_back(std::make_unique<entry>(std::move(key)));
        }
    }

    lazy_cache(const lazy_cache &) = delete;
    lazy_cache & operator=(const lazy_cache &) = delete;

    template<typename Matches, typename Loader>
    std::shared_ptr<Value> resolve_if(Matches matches, Loader loader) {
        const std::lock_guard<std::mutex> lock(mutex);
        for (const auto & item : entries) {
            if (!matches(item->key)) {
                continue;
            }
            std::shared_ptr<Value> value = resolve(*item, loader);
            if (value != nullptr) {
                return value;
            }
        }
        return nullptr;
    }

    template<typename Loader>
    std::shared_ptr<Value> resolve_or_add(const Key & key, Loader loader) {
        const std::lock_guard<std::mutex> lock(mutex);
        for (const auto & item : entries) {
            if (item->key == key) {
                return resolve(*item, loader);
            }
        }

        auto item = std::make_unique<entry>(key);
        entry * added = item.get();
        entries.emplace_back(std::move(item));
        return resolve(*added, loader);
    }

    lazy_cache_counts counts() const {
        const std::lock_guard<std::mutex> lock(mutex);
        lazy_cache_counts result;
        for (const auto & item : entries) {
            switch (item->state) {
                case lazy_cache_state::untried:
                    ++result.untried;
                    break;
                case lazy_cache_state::ready:
                    ++result.ready;
                    break;
                case lazy_cache_state::failed:
                    ++result.failed;
                    break;
            }
        }
        return result;
    }

    std::vector<std::pair<Key, std::string>> failures() const {
        const std::lock_guard<std::mutex> lock(mutex);
        std::vector<std::pair<Key, std::string>> result;
        for (const auto & item : entries) {
            if (item->state == lazy_cache_state::failed) {
                result.emplace_back(item->key, item->failure);
            }
        }
        return result;
    }

    std::vector<std::shared_ptr<Value>> ready_values() const {
        const std::lock_guard<std::mutex> lock(mutex);
        std::vector<std::shared_ptr<Value>> result;
        for (const auto & item : entries) {
            if (item->state == lazy_cache_state::ready) {
                result.push_back(item->value);
            }
        }
        return result;
    }

    void clear() {
        const std::lock_guard<std::mutex> lock(mutex);
        entries.clear();
    }

private:
    struct entry {
        explicit entry(Key value) : key(std::move(value)) {}

        Key key;
        lazy_cache_state state = lazy_cache_state::untried;
        std::shared_ptr<Value> value;
        std::string failure;
    };

    template<typename Loader>
    static std::shared_ptr<Value> resolve(entry & item, Loader & loader) {
        if (item.state == lazy_cache_state::ready) {
            return item.value;
        }
        if (item.state == lazy_cache_state::failed) {
            return nullptr;
        }

        try {
            item.value = loader(item.key);
            if (item.value == nullptr) {
                throw std::runtime_error("lazy cache loader returned no value");
            }
            item.state = lazy_cache_state::ready;
            return item.value;
        } catch (const std::exception & error) {
            item.failure = error.what();
        } catch (...) {
            item.failure = "unknown lazy cache loader failure";
        }
        item.value.reset();
        item.state = lazy_cache_state::failed;
        return nullptr;
    }

    mutable std::mutex mutex;
    std::vector<std::unique_ptr<entry>> entries;
};

} // namespace detail
} // namespace ggml_xdna

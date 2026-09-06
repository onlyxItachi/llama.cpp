#include <ggml-xdna.h>

#include <cstdio>
#ifdef PACKAGE_DL
#include <dlfcn.h>
#endif

int main(int argc, char ** argv) {
    if (ggml_type_size(GGML_TYPE_F32) != sizeof(float)) {
        std::fputs("FAIL: installed ggml type API\n", stderr);
        return 1;
    }
#ifdef PACKAGE_OFF
    (void) argc;
    (void) argv;
    if (ggml_backend_reg_by_name("XDNA") != nullptr || ggml_backend_reg_by_name("CPU") == nullptr) {
        std::fputs("FAIL: expected CPU registration without XDNA\n", stderr);
        return 1;
    }
#else
#ifdef PACKAGE_DL
    if (argc != 2) {
        std::fputs("usage: package-consumer INSTALLED_XDNA_MODULE\n", stderr);
        return 2;
    }
    void * handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        std::fprintf(stderr, "FAIL: installed plugin ELF load: %s\n", dlerror());
        return 1;
    }
    if (!dlsym(handle, "ggml_backend_init")) {
        std::fputs("FAIL: installed plugin backend entry point\n", stderr);
        dlclose(handle);
        return 1;
    }
    auto is_xdna = reinterpret_cast<bool (*)(ggml_backend_t)>(dlsym(handle, "ggml_backend_is_xdna"));
    auto get_stats = reinterpret_cast<ggml_backend_xdna_get_stats_t>(dlsym(handle, "ggml_backend_xdna_get_stats"));
    auto get_stats_v2 = reinterpret_cast<ggml_backend_xdna_get_stats_v2_t>(dlsym(handle, "ggml_backend_xdna_get_stats_v2"));
#else
    (void) argc;
    (void) argv;
    auto is_xdna = ggml_backend_is_xdna;
    auto get_stats = ggml_backend_xdna_get_stats;
    auto get_stats_v2 = ggml_backend_xdna_get_stats_v2;
#endif
    ggml_backend_xdna_stats stats {};
    ggml_backend_xdna_stats_v2 stats_v2 {};
    const bool valid = is_xdna && get_stats && get_stats_v2 && !is_xdna(nullptr) &&
        !get_stats(nullptr, &stats) && !get_stats_v2(nullptr, &stats_v2, sizeof(stats_v2));
#ifdef PACKAGE_DL
    if (dlclose(handle) != 0) {
        std::fprintf(stderr, "FAIL: installed plugin ELF unload: %s\n", dlerror());
        return 1;
    }
#endif
    if (!valid) {
        std::fputs("FAIL: installed XDNA symbols or null-backend API contract\n", stderr);
        return 1;
    }
#endif
    std::puts("PASS: installed package consumer; no XDNA registry or device initialization");
    return 0;
}

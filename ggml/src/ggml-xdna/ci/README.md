# Linux package smoke test

Run as an unprivileged user with CMake, Ninja, native C/C++ compilers, GNU coreutils, Bash, binutils (`readelf`), and an XRT SDK exporting `XRT::xrt_coreutil`:

```sh
bash ggml/src/ggml-xdna/ci/package-smoke.sh /path/to/empty-output /path/to/xrt-prefix 2
```

The output path must be new or empty. All builds and installs remain below it; no system installation or cleanup is performed. The caller owns workload serialization. Compiler selection may use `CC` and `CXX`.

The harness builds ggml only, installs static/XDNA, dynamic-backend/XDNA, and static/XDNA-disabled variants, then moves each install before configuring an independent consumer. Imported ggml paths must remain inside the relocated prefix. The static consumer must preserve its own XRT root. Dynamic and disabled consumers configure with XRT discovery disabled, and their ELF dependencies must not directly require XRT. Both the producer and static consumer must reject a required SDK disabled by CMake.

Runtime checks cover static API linkage, dynamic plugin ELF load/unload, null-backend API rejection, and CPU-only registration. They never initialize the XDNA registry or device and need no AOT artifacts, Python, MLIR-AIE, or Peano. Dynamic ELF loading still requires the XRT runtime. This is not an absent-runtime test: clearing `LD_LIBRARY_PATH` does not remove a globally installed runtime from the loader cache.

Stage logs and `terminal.txt` identify the stopping point. A successful result establishes package consumption on the tested host, not physical kernel correctness or a hermetic distribution across Linux versions.

## Failure-policy smoke test (physical AIE2P required)

```sh
bash ggml/src/ggml-xdna/ci/failure-smoke.sh \
    /path/to/build/bin /path/to/xrt-prefix \
    /path/to/gemv-bf16-288.ggmlxdna /path/to/new-output
```

This Linux x86_64 test requires the dynamic CPU/XDNA modules from a coherent
build, the BF16 288x288 artifact, an idle compatible device at `/dev/accel/accel0`,
a C++17 compiler, `flock`, `fuser` and coreutils. It builds an interposition
library against the explicitly selected XRT. Use it only in a dedicated test
process: eight cases deliberately terminate by SIGABRT. The runner checks Linux
`PR_SET_DUMPABLE=0` before device setup, preventing its core dumps even with a
piped system core handler. A shell relays the numeric exit status so `timeout`
does not re-raise SIGABRT in its own process. The selected SDK comes first in the
loader path; the test rejects a backend resolving XRT coreutil outside that SDK.
The caller must also serialize against compilers and other accelerator workloads;
`XDNA_CI_PHYSICAL_LOCK` selects the test's cooperative device lock.

Thirteen cases check post-completion start/wait exceptions with poisoned output and
same-backend changing-input CPU-reference retry; synthetic nonterminal/error-query
states; synthetic terminal ERROR/ABORT/TIMEOUT returns with CPU-exact retry; an
abort exception after completion; and a pre-submission host exception. The three
terminal-state cases substitute results only after a real COMPLETED wait. They
are category C, not actual hardware errors or timed-wait API timeout evidence.
Each case requires both the exact injection marker and expected runtime diagnostic
or success receipt, not merely a signal or exit code. Source/library/artifact hashes
are checked before and after. `kernel_submissions` counts successful `start()`
returns, not submissions hidden by an exception inside XRT; independent interposed
counters distinguish the latter.

These are category B (post-completion exceptions), C (synthetic states), and a
pre-submit injection that still requires device setup. None is a host-only test
or proves recovery from an actual in-flight hardware failure. Real cancellation
experiments must additionally account for XRT's internal abort command; observing
the original command complete is insufficient if abort itself threw.

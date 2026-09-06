# Native AMD XDNA backend: bring-up design and evidence

Status: pre-upstream candidate (3) for the documented Linux/AIE2P scope; not an upstream-quality candidate (4), updated 2026-09-06.
This document records what was inspected and physically exercised; it is not a claim of production-ready XDNA decode.

## Upstream synchronization and failure-path validation (2026-09-06)

The previous pass had already closed the large upstream gap. This continuation starts at published `44e26d284c3c818c9de8fa518dfddf68624d1b23` and rebases onto `9e0e220594af405a62835dc3a27495729fd8506b`, six newer upstream commits. All 50 development changes remain patch-equivalent; `backup/xdna-before-upstream-20260906-44e26d2` preserves the prior branch. A second fetch before final validation found the same upstream head. Scheduler/buffer interfaces and XDNA/AOT sources did not change in this upstream delta. The host backend was rebuilt; the unchanged seven fresh AOT bundles were physically revalidated.

Upstream now supports JSONL logging through both a CLI switch and environment variables. The new real-GGUF runner explicitly selects `--no-log-jsonl` and clears inherited logging variables, keeping generated-text comparisons separate from logs. Final smoke runs deliberately inherit `LLAMA_ARG_LOG_JSONL=1` to exercise this guard.

The existing backend operation test now includes a persistent BF16 288x288 trainable-weight case. It marks the tensor as `PARAM` before use, negates all weights for its second invocation, and checks CPU agreement, exact XDNA sign reversal, per-call staging/synchronization, and absence of immutable registrations. The retained old backend fails this in-tree case; current XDNA and HIP pass it. This extends coverage without adding a kernel or changing scheduling capability.

Failure accounting now counts start/wait attempts at their call boundaries and records a known submission immediately after `start()` returns. A throwing wait no longer erases evidence of device work. Attempt timings exclude subsequent abort work; successful-compute and first-success timings retain their existing meaning. Diagnostics identify the failing execution phase, and the statistics ABI is unchanged.

The installed XRT 2.20 headers and matching source `021204355eeaa034ff69aae407ace2265adf047a` distinguish a reset timeout from `NORESPONSE`, which means reset failed. Synchronous abort can return an already-completed command's existing state without another reset. The runtime therefore requires an explicit supported terminal state before resetting bindings or releasing execution ownership. Blocking wait and abort accept `COMPLETED`, `ERROR`, `ABORT`, and scheduler `TIMEOUT`; live, unknown, and failed-reset states terminate rather than release potentially active borrowed storage. This is contract hardening, not a reproduced hardware reset defect.

A cache-local interposer waits for actual device completion before injecting one exception. Before the accounting fix, both real launches occur but the backend reports only one submission/start/wait. After the fix, it reports both, leaves the first failed destination poisoned, and successfully retries the same graph with a changed activation, exact CPU output, one registration/synchronization, and zero immutable-weight copies. Separate synthetic `NORESPONSE` wait/abort returns and a `SUBMITTED` wait return exercise the fail-stop guard after the real command is already complete. These tests do not establish in-flight abort or failed-reset recovery.

All four focused host tests pass with XRT disabled, with runtime enabled, under ThreadSanitizer, and under AddressSanitizer/UndefinedBehaviorSanitizer. Final physical tests pass all seven immutable specializations with fourteen changing-input launches, plus the trainable-weight case. The HIP operation control also passes all seven cases and the trainable case.

Final real-GGUF tests cover `stories15M-q4_0.gguf` and a native BF16 1B model, each with CPU, XDNA, HIP, forced HIP+XDNA, and ordinary HIP-first+XDNA placement. All five configurations agree on generated text for each model: SHA-256 `afbaea16edb09763443b80c13cd58d7c61a40b9d0eb3a4c7807fde9ccfb19c3f` and `5d78704928839cc3450ec9f4bc7f6aa8cc65924afb93dd027201965be5c1d891`, respectively. The 1B forced-hybrid run completes 994 NPU calls from shared host-backed model storage, with 32 weight views, 962 reuse hits, one-time synchronization of 1 GiB, and zero immutable-weight copies. Ordinary HIP-first placement records zero NPU calls. These remain 32-token functional smoke tests, not every-logit validation or a performance comparison.

Evidence, sealed input identities, failure-test sources, and terminal results are retained in `/home/hamza-usta/.cache/llama-xdna-upstream-20260906-vmwizI`. Earlier evidence remains intact. No performance preference changed, no unused fusion matcher was added, and inference still has no compiler/Python dependency. Readiness remains level 3: independent human review of generic core changes, broader driver/failure coverage, portable artifact/package CI, and matched end-to-end performance remain open.

## Current-upstream completion pass (2026-09-05)

The pass started from clean local and remote `feat/ggml-xdna` head `1b90c9ac2b3ba27e2c4d2b0412ba738cbe4756d9`.
The local branch was rebased onto upstream `6a1a922d269908a29cbd4b49c27e6a8e7fd10fae`, producing `f1f7964541aea1b82f9897939796beed1f0bc3c5`; all 40 original changes were retained.
`backup/xdna-before-upstream-20260905` preserves the original history; the completion changes were left uncommitted for the initial handoff and then checkpointed under the user's private-fork development authorization.
Historical measurements below retain their original source and protocol boundaries; they are not new measurements on this upstream base.

Distinct backend instances can now concurrently add and remove their own observers on a shared live buffer.
The observer list is serialized in GGML, detached before notification, and callbacks run outside the list lock.
Buffer destruction must still be serialized against buffer use, observer mutation, and backend teardown; observers do not extend allocation lifetime.
A two-context host reproducer reports a ThreadSanitizer race before the fix and none after it, and existing scheduler tests cover concurrent subscriptions and callbacks that touch another buffer.
XDNA registry context publication is also performed only once during static initialization.

All seven specializations now use one private constexpr GEMV descriptor constructor with format data and exact specialization rows.
The AOT Makefile shares compile, spatial-generation, artifact, and packaging rules; the two large BF16 variants use the existing parameterized compute source and controller.
All descriptor fields and all 43 generated build commands match the previous implementation.
No compute source, controller topology, artifact ABI, or inference-time dependency changed, so the existing validated artifacts were retained.
Independent seven-row contract tests guard the inventory, and native Q4_0 fixtures now include adversarial scales, nibble ordering, and row/block boundaries.

A fresh dynamic CPU/XDNA build passes the four focused host tests and all seven physical backend operation cases against CPU.
The real `stories15M-q4_0.gguf` native-layout smoke completes 744 XDNA submissions over 31 decode steps with 24 weight views, 720 view hits, fixed first-use weight synchronization, and zero immutable-weight copy bytes.
Its generated text matches the CPU control byte-for-byte; unsupported operations remain on CPU.
A fresh current-base HIP module also passes the matched-text smoke, and forced HIP+XDNA executes the same 744 NPU calls from a shared `ROCm_Host` model allocation with zero immutable-weight copies.
With all artifacts configured but no priority override, the ordinary HIP+XDNA smoke matches the reference and records zero XDNA submissions.
These small native gfx1150 runs use no HSA architecture override and do not establish larger-model rocBLAS coverage.
The corrected cache-local BF16 runners separately pass two changing-input, per-launch-poisoned invocations for both candidate artifacts.
These are functional results, not performance results: the strict performance preflight rejected host swap activity before invoking its runner, with the stop reason recorded.
Old unfinished performance protocols remain unfinished.

Current upstream assigns backends and creates splits before calling per-split `graph_optimize`.
CUDA and SYCL perform runtime pattern selection before single-node dispatch, and GGML already supplies subgraph-closure helpers.
No new XDNA matcher is added without an accepted executable composite: the retained gate/up candidate loses to two standalone XDNA calls, while the measured K/V candidate is slower than HIP and its target graph separates the projections into different backend splits.
Other graphs can place siblings together; this is not a claim that all K/V pairs are split.
Future useful fusion belongs in the existing XDNA split compute scan, with explicit materialized outputs, source identity and capability checks, and single-node fallback on a match miss.
No graph-builder rewrite, second IR, runtime compiler, or new scheduler API is justified by the present evidence.

Current limitations still include read-only mmap rejection on the installed driver, unvalidated concurrent allocation destruction, separate AIE2 implementation requirements, large-model allocation and hybrid performance gates, and missing cross-platform/physical CI.
The observer, shared-host scheduling, transient-copy, and CPU-mapped-buffer core changes need independent maintainer review before any upstream proposal.

## Private-fork development validation (2026-09-05)

The repair checkpoint was published to `onlyxItachi/llama.cpp:feat/ggml-xdna` under explicit private-fork authorization, followed by further reviewed development commits.
Final executable-source validation uses `3335e9d913d28eb54544098ffe8ad751158fb064`, based on upstream `6a1a922d269908a29cbd4b49c27e6a8e7fd10fae`; repeated upstream fetches found no newer base during this pass.
No upstream submission, issue, discussion, or comment was created or modified.

The build now exposes the four focused scheduler, artifact, cache, and policy tests without enabling XDNA or finding XRT. They pass in the ordinary Linux build, a Clang ThreadSanitizer build, and a Clang AddressSanitizer/UndefinedBehaviorSanitizer build. The last configuration also exposed pre-existing upstream null-pointer arithmetic in graph-overhead calculation; integer-address increment fixes it without an API or layout change, with exact-overhead allocation checks across graph sizes and gradient-storage modes.
Installed static, dynamic-backend, and XDNA-disabled packages pass isolated external-consumer checks. Static consumers retain their selected XRT SDK root. Relocated dynamic consumers require no XRT SDK at configuration time, but loading the plugin requires the declared XRT runtime. Removing that runtime from the controlled loader search rejects XDNA while the installed CPU plugin remains loadable.

Trainable tensors in `WEIGHTS` buffers are not constants in GGML. XDNA now follows the upstream `WEIGHTS && !PARAM` predicate and uses its existing per-call staging path for parameters. A two-call physical reproducer returned stale data before the fix: output `0.1171875` instead of the updated CPU reference `-1.1328125`. Both calls are exact after the fix, with two native-weight copies and synchronizations and no persistent registration. Immutable model payloads must remain frozen until their owner is freed; changing flags is not a cache-invalidation API. Backend construction also keeps its context in RAII ownership until publication.

All seven AOT artifacts were rebuilt from a clean source export at `e6fd8a5a292e6b352a7ff7f25f8f3b744bc8608c`, using MLIR-AIE 1.3.4, Peano `21.0.0.2026062301+cb664e8c`, Python 3.12.13, XRT 2.20, and bounded outer/compiler parallelism. Makefile recipe changes now invalidate generated outputs; a host regression fails on the old Makefile and passes on the new one. This establishes a fresh source rebuild on the retained toolchain, not bit-reproducibility across clean machines: xclbin metadata differs, and cross-host/toolchain-distribution CI remains open.

The final native-weight operation test uses the original activation and then its whole-vector negation, poisons the GGML destination before each invocation, and constructs a fresh CPU comparison while retaining the primary graph and registrations. Every XDNA output must negate exactly, in addition to ordinary numerical comparison and finite/non-vacuous output checks. This avoids a rejected intermediate harness revision whose changed BF16 edge-case lane masked smaller errors in aggregate NMSE. All seven cases pass with fourteen launches, seven first-use weight synchronizations, seven reuse hits, and zero immutable-weight copy bytes; the same seven cases pass the HIP CPU-reference comparison. A physical acceptance wrapper additionally requires the exact seven-case inventory: ordinary `test-backend-ops` success alone can include skipped unsupported cases.

The final real-GGUF smoke repeats CPU, configured CPU+XDNA with and without priority override, HIP, forced HIP+XDNA, and ordinary HIP-first+XDNA on the native-layout `stories15M-q4_0.gguf`. All six text outputs have SHA-256 `afbaea16edb09763443b80c13cd58d7c61a40b9d0eb3a4c7807fde9ccfb19c3f`. Forced hybrid runs 744 successful NPU calls, with 24 weight views, 720 hits, and zero immutable-weight copies; ordinary HIP-first runs zero NPU calls. Binaries, CPU oracle libraries, all seven artifacts, and wrappers are hashed before and after final physical runs. These are functional results only, not matched performance or larger-model rocBLAS validation.

The development evidence is retained in `/home/hamza-usta/.cache/llama-xdna-development-20260905-gelKfY`, including the factual human-ownership handoff, source/build inputs, before/after failures, sanitizer logs, package consumers, and `*-core-final` physical results. Historical evidence directories and validated artifacts were preserved.

Level 3 here means coherent selective integration, current upstream base, strong correctness gates, safe documented lifetime semantics, a reproducible local AOT build, conservative GPU preference, and real GGUF execution. Level 4 is not claimed. Required follow-up includes independent human review of the generic core surfaces, portable package/artifact CI, cross-host and driver coverage, positive mmap validation, injected XRT failure-path testing, and broader current-base GGUF coverage. No current result establishes an end-to-end HIP performance win, energy advantage, or reason to enable automatic priority. No unused graph matcher, runtime compiler, or model-specific selection logic was introduced.

Related upstream work inspected read-only: [XDNA request #21725](https://github.com/ggml-org/llama.cpp/issues/21725), [AMD HRX draft #27218](https://github.com/ggml-org/llama.cpp/pull/27218), and [its architecture discussion #27219](https://github.com/ggml-org/llama.cpp/discussions/27219). These are references for human review, not claims of endorsement or equivalent implementation.

## Revisions and repository state

The initial XDNA bring-up began from clean upstream llama.cpp commit
[`6e62ba538478202094edc6c100c782719e310aa3`](https://github.com/ggml-org/llama.cpp/commit/6e62ba538478202094edc6c100c782719e310aa3)
on branch `feat/ggml-xdna`. The fork is `onlyxItachi/llama.cpp` (`origin`) and
`ggml-org/llama.cpp` is `upstream`. No user work was reset.

Before the vector-performance sprint, the XDNA head was
`31ab68b083e1b6baa00bb36bdf0c48766235f255`. It was backed up at
`backup/ggml-xdna-pre-vector-20260811`, then rebased non-destructively onto
current upstream [`70dfba5aee36793fb51ae649723b3c30ed9e99d3`](https://github.com/ggml-org/llama.cpp/commit/70dfba5aee36793fb51ae649723b3c30ed9e99d3).
The patch-equivalent post-rebase XDNA baseline was
`9b625b68fedad18249e9f970919d18421e426796`. The optimization commits remain
separate local milestones on `feat/ggml-xdna`; none was pushed by this sprint.

The production-shape phase started from XDNA head
`194668600a5852b4ae54a24502248288631c7042`. It fetched and rebased those
commits onto upstream
[`f785fc9ea485e6cfdda129978310aa52939c3619`](https://github.com/ggml-org/llama.cpp/commit/f785fc9ea485e6cfdda129978310aa52939c3619).
The clean post-rebase head was
`73ab9f23ecb5b1c3fc7e34b966236287dfe7b8d1`. No XDNA commit or user work was
dropped, and no commit from this phase was pushed.

After the shared-host scheduler and real-GGUF work, the pre-rebase XDNA head was
`8d6e9a84caf2b1b467079e0d97dfe3e9264f0c00`. A backup ref named
`backup/ggml-xdna-pre-rebase-20260811-2349` preserves it. The branch was then
rebased cleanly onto upstream
[`7b13a8404d7e219c13d1a243e2a21a857a6e99d9`](https://github.com/ggml-org/llama.cpp/commit/7b13a8404d7e219c13d1a243e2a21a857a6e99d9), producing post-rebase head
`7cb812a91a440699b7ee2268d01d10a98d51a0fd`. `git range-diff` reports all 20
XDNA commits patch-equivalent. The rebased history was published to
`origin/feat/ggml-xdna` with an explicit SHA-bound `--force-with-lease`; later
validated milestones have been pushed normally on top of it.

The lifetime and mmap audit below starts from XDNA head `62661d56d393c142155a1e4fa47cb7c3b83ea9f0`.
Its relevant implementation milestones are the generic observer at `2a27ff2065d46e3406828562d9079d55c0f70b09`, XDNA owner-lifetime eviction at `9d55fecb6cfb221489f052f8b324910c23883a69`, run quiescence at `1771128c472824fdf7a0f84c3bbe4bdb31ac8f77`, and the read-only mmap capability probe at the audit head.

The current default-branch heads inspected with Git and GitHub CLI were:

| Project | Exact revision |
| --- | --- |
| Xilinx/mlir-aie | [`a8fea363dfd63015cffd380e0679950419a0a7cd`](https://github.com/Xilinx/mlir-aie/commit/a8fea363dfd63015cffd380e0679950419a0a7cd) |
| Installed mlir-aie / Peano | mlir-aie `1.3.4` (release source [`ed23bba7381b6f0680775fd3b6110d9268d27f43`](https://github.com/Xilinx/mlir-aie/commit/ed23bba7381b6f0680775fd3b6110d9268d27f43)); llvm-aie [`cb664e8cc3eb42a12e3ad3cee28729785ffa97a3`](https://github.com/Xilinx/llvm-aie/commit/cb664e8cc3eb42a12e3ad3cee28729785ffa97a3), wheel `21.0.0.2026062301+cb664e8c` |
| amd/xdna-driver | [`d73478232a7057bfea73a24f74b09df4e5580ad8`](https://github.com/amd/xdna-driver/commit/d73478232a7057bfea73a24f74b09df4e5580ad8) |
| Xilinx/XRT | [`27bed122dd01ca3e21dbf2b07000078ec842c98c`](https://github.com/Xilinx/XRT/commit/27bed122dd01ca3e21dbf2b07000078ec842c98c); driver pin [`8b60ae7a90bfbc873e181497fd34ca520b4ef504`](https://github.com/Xilinx/XRT/commit/8b60ae7a90bfbc873e181497fd34ca520b4ef504) |
| amd/Triton-XDNA | [`434dc3ec3de0fa0de2d2e26236e363967a00e265`](https://github.com/amd/Triton-XDNA/commit/434dc3ec3de0fa0de2d2e26236e363967a00e265) |
| ROCm/FastFlowLM v1.0.0 | [`39fb58cc7f9e23d97b764a1ef21ff0ade088047b`](https://github.com/ROCm/FastFlowLM/commit/39fb58cc7f9e23d97b764a1ef21ff0ade088047b) |
| amd/RyzenAI-SW | [`06fdce7eeacf88c07fa84451c22e306ebca6de5e`](https://github.com/amd/RyzenAI-SW/commit/06fdce7eeacf88c07fa84451c22e306ebca6de5e) |
| RyzenAI-SW historical llama branch | [`d78a8cf57cfbda319ec09dcac941ddfc34f5793f`](https://github.com/amd/RyzenAI-SW/commit/d78a8cf57cfbda319ec09dcac941ddfc34f5793f) |

The live machine has XRT 2.20.0 (`021204355eeaa034ff69aae407ace2265adf047a`),
amdxdna 0.7.0 (`srcversion 3873C4BB9348F32F5C9E515`), firmware 1.1.2.64,
and an accessible `RyzenAI-npu4` at `0000:67:00.1` through
`/dev/accel/accel0`. The device exposes eight columns and uses KMQ.
`xrt-smi validate` passed its supplied INT8 GEMM test at 51.0 TOPS and its
submission-latency test at about 46 us. Those are stack-health results, not
GGML performance.

## Current GGML integration surface

Current GGML has the required generic seam:

- Devices can report `GGML_BACKEND_DEVICE_TYPE_ACCEL`, host-pointer wrapping,
  and mmap capability independently
  ([API](https://github.com/ggml-org/llama.cpp/blob/6e62ba538478202094edc6c100c782719e310aa3/ggml/include/ggml-backend.h#L134-L194)).
- Dynamic backends are registered and loaded by name; `ggml-xdna` follows that
  mechanism rather than modifying HIP
  ([registry](https://github.com/ggml-org/llama.cpp/blob/6e62ba538478202094edc6c100c782719e310aa3/ggml/src/ggml-backend-reg.cpp#L193-L268),
  [loader](https://github.com/ggml-org/llama.cpp/blob/6e62ba538478202094edc6c100c782719e310aa3/ggml/src/ggml-backend-dl.cpp#L34-L119)).
- BLAS is the useful small ACCEL reference: it exposes host-backed buffers and
  restricts `MUL_MAT` in `supports_op`
  ([BLAS device](https://github.com/ggml-org/llama.cpp/blob/6e62ba538478202094edc6c100c782719e310aa3/ggml/src/ggml-blas/ggml-blas.cpp#L355-L468)). Hexagon is useful for runtime
  separation, but currently labels its device as GPU and is not copied as the
  type model.
- The mmap wrapping path checks `buffer_from_host_ptr` and whether the chosen
  buffer is that device's default; `mmap_support` is a separate AUTO-mode gate
  applied only to model GPU devices, not ACCEL devices
  ([loader](https://github.com/ggml-org/llama.cpp/blob/6e62ba538478202094edc6c100c782719e310aa3/src/llama-model.cpp#L1276-L1285),
  [buffer creation](https://github.com/ggml-org/llama.cpp/blob/6e62ba538478202094edc6c100c782719e310aa3/src/llama-model.cpp#L1577-L1607)).
  CPU's wrapper returns a distinct private `CPU_Mapped` buffer type
  ([implementation](https://github.com/ggml-org/llama.cpp/blob/6e62ba538478202094edc6c100c782719e310aa3/ggml/src/ggml-backend.cpp#L2405-L2430)),
  which gives XDNA a backend-local way to reject unregistrable mappings.

There is an important scheduler constraint. llama.cpp orders model GPU devices,
then all ACCEL devices, then CPU
([model device ordering](https://github.com/ggml-org/llama.cpp/blob/6e62ba538478202094edc6c100c782719e310aa3/src/llama-model.cpp#L884-L932),
[context ordering](https://github.com/ggml-org/llama.cpp/blob/6e62ba538478202094edc6c100c782719e310aa3/src/llama-context.cpp#L318-L362)). The scheduler first prefers a backend that
owns/supports an input buffer; it consults `offload_op` while reconsidering an
otherwise CPU assignment
([assignment](https://github.com/ggml-org/llama.cpp/blob/6e62ba538478202094edc6c100c782719e310aa3/ggml/src/ggml-backend.cpp#L877-L968)). Therefore decode-selective
`supports_op` alone is insufficient to select a lower-priority ACCEL for a
weight already supported by HIP. The integrated scheduler adds a narrow generic
preference: only an ACCEL's affirmative `offload_op` for an immutable
host-backed weight can override ordinary owner priority. The XDNA callback then
requires an exact registered kernel and consults per-variant preference
metadata. All current variants remain supported but default to no automatic
preference because the measured HIP implementations are faster. Thus normal
`[ROCm0, XDNA0, CPU]` order keeps a mutually supported operation on ROCm, while
an explicitly configured CPU+XDNA configuration still selects XDNA as the first capable backend.
Set `GGML_XDNA_PREFER_OFFLOAD=1` to request the exact XDNA kernels explicitly
for controlled heterogeneous experiments; unsupported shapes such as prefill
remain on HIP. This policy is capability-driven and does not add model-specific
placement logic.
An unset preference override or `--no-op-offload` is not a CPU-only switch when XDNA is the first capable backend.
For a CPU-only control, leave all XDNA artifact bundle variables unset or do not load the XDNA plugin.
For native quantized XDNA placement, use `--no-repack`; CPU-repacked weights are correctly declined.

## Selected open stack and generation split

The selected path is MLIR-AIE/IRON plus its pinned Peano compiler at build time,
and native XRT C++ at runtime. Current IRON drives Peano directly
([MLIR-AIE overview](https://github.com/Xilinx/mlir-aie/blob/a8fea363dfd63015cffd380e0679950419a0a7cd/README.md#L59-L70)); Triton,
FastFlowLM, HRX, and proprietary Ryzen AI operators are not runtime
dependencies.

`npu1` means Phoenix/Hawk Point XDNA/AIE2; `npu2` means Strix, Strix Halo, and
Krackan XDNA2/AIE2P in MLIR-AIE
([device mapping](https://github.com/Xilinx/mlir-aie/blob/a8fea363dfd63015cffd380e0679950419a0a7cd/python/aie_lit_utils/lit_config_helpers.py#L62-L67)). XRT names the present
XDNA2 device `RyzenAI-npu4`. The backend keeps architecture discovery generic,
but the only registered and physically validated compute artifacts are AIE2P. XDNA1 has no registry entry and is not claimed as tested.

FastFlowLM demonstrates persistent runtime resources, but its Q4NX layout and
dequant/GEMM kernels are opaque prebuilt libraries, use a different packed
layout, and have restrictive/conflicting binary terms
([public Q4NX boundary](https://github.com/ROCm/FastFlowLM/blob/39fb58cc7f9e23d97b764a1ef21ff0ade088047b/src/include/tensor_utils/q4_npu_eXpress.hpp#L11-L59),
[prebuilt linkage](https://github.com/ROCm/FastFlowLM/blob/39fb58cc7f9e23d97b764a1ef21ff0ade088047b/src/CMakeLists.txt#L406-L430),
[terms](https://github.com/ROCm/FastFlowLM/blob/39fb58cc7f9e23d97b764a1ef21ff0ade088047b/TERMS.md#L1-L18)). Triton-XDNA demonstrates
the current Peano path and persistent multi-launch resources
([compiler invocation](https://github.com/amd/Triton-XDNA/blob/434dc3ec3de0fa0de2d2e26236e363967a00e265/amd_triton_npu/backend/driver.py#L1834-L1973),
[persistent runner](https://github.com/amd/Triton-XDNA/blob/434dc3ec3de0fa0de2d2e26236e363967a00e265/amd_triton_npu/backend/multilaunch.py#L232-L330)), but has no native GGML Q4
kernel. Its Qwen `hetero-fast` example explicitly chooses NPU prefill plus GPU
decode and pads decode `M=1` to 128
([policy](https://github.com/amd/Triton-XDNA/blob/434dc3ec3de0fa0de2d2e26236e363967a00e265/examples/qwen2_5/model.py#L693-L722)); that is an implementation choice, not a
hardware result for direct compressed GEMV. Neither project is linked or
vendored. Current MLIR-AIE has a WIP int16 matrix-vector example, not a ready Q4
GEMV. Its new AIE2P lowering for canonical packed i4 unpack is a useful building
block
([conversion test](https://github.com/Xilinx/mlir-aie/blob/a8fea363dfd63015cffd380e0679950419a0a7cd/test/Conversion/AIEVecToLLVM/test-unpack-int4-aie2p.mlir#L10-L31)).

The historical RyzenAI-SW llama hook is useful only as evidence: it recognized
GGML Q4_0's 32-value blocks, accepted contiguous F32 activations, converted
activations to BF16, and cached a proprietary reordered weight representation.
It predates the current backend API, depends on proprietary operators, and
duplicates/transforms weights, so no code or architecture was carried forward.
Current RyzenAI-SW instead points users to pre-optimized OGA hybrid/NPU models
([OGA example](https://github.com/amd/RyzenAI-SW/blob/06fdce7eeacf88c07fa84451c22e306ebca6de5e/LLM-examples/oga_api/README.md#L1-L28)).

## System-memory design and measured constraints

The intended representation is one system-RAM model allocation shared by the
host devices. XDNA registers page-rounded windows of the immutable tensors it
actually selects, with each tensor expressed as a child offset:

```text
GGML host allocation (observed buffer owner)
  -> cached XRT host_only userptr BO for a selected tensor's page window
     -> root-derived immutable weight sub-BOs
        -> shim DMA / memory tile / AIE compute tile
```

Transient activations and results use small persistent XRT BOs sized by the
selected kernel variant. For the 288x288 slice, the host copies 576 bytes of
BF16 activation into the input BO and 1,152 bytes of F32 result out of the
output BO. Immutable model weights are not payload-copied; mutable/test weights
use the explicit staging path described below.

XRT sub-BOs retain their parent, expose the parent device address plus an
arbitrary byte offset, and bind a kernel argument as handle plus offset/size
([sub-BO](https://github.com/Xilinx/XRT/blob/8b60ae7a90bfbc873e181497fd34ca520b4ef504/src/runtime_src/core/common/api/xrt_bo.cpp#L941-L999),
[binding](https://github.com/Xilinx/XRT/blob/8b60ae7a90bfbc873e181497fd34ca520b4ef504/src/runtime_src/core/common/api/xrt_kernel.cpp#L1154-L1161)). Registration uses `host_only`, not XRT's
`svm` flag. Root pages are long-term pinned and count against `RLIMIT_MEMLOCK`.
On this KMQ device, sync is range cache maintenance, not a second payload copy.

Physical probes on this machine established:

- One 16 KiB page-aligned userptr parent and sub-BOs at byte offsets 64 and
  8,320 executed a real AIE2P increment kernel correctly for 32 iterations.
- BO creation ioctls stayed constant at five while submissions rose from one
  to 32 when the parent, sub-BOs, and `xrt::run` were reused. Recreating the run
  added one command BO per invocation.
- Unaligned root pointers fail; arbitrary sub-BO offsets work. The backend
  validates each tensor against its live GGML owner, page-rounds the tensor
  window, and preserves the owner identity in the registration cache.
- The installed kernel driver rejects a `PROT_READ` GGUF-like mmap for both
  ordinary and read-access extended BO constructors with CREATE_BO errno 12.
  A 16 MiB writable private mapping then COWed the full 16 MiB, so that is not a
  workaround. Current upstream driver added read-only pinning and read-only
  IOMMU permissions in
  [`ed8fb2dd172bde623d7112a1bd674fc0e3c4cae4`](https://github.com/amd/xdna-driver/commit/ed8fb2dd172bde623d7112a1bd674fc0e3c4cae4), but the installed module must be
  upgraded before that path can pass locally.

Backend statistics label their counter `explicit_payload_BOs`: it covers
payload/root BO allocations made directly by ggml-xdna, not lightweight
subviews, command BOs, or other internal BOs allocated by XRT. Driver-level
creation evidence comes from the ioctl trace above; those counts, too, remain
constant rather than growing with repeated submissions.

The backend does not infer mmap safety from a driver version. When at least one
validated kernel artifact is configured, XDNA device registration creates one
page of file-backed `PROT_READ`, `MAP_SHARED` storage and asks XRT to construct
and destroy a `host_only` user-pointer parent BO over that page. It also creates
a nonzero-offset child BO and performs the same range-limited TO_DEVICE sync
used by an immutable weight. Any mapping, parent, subview, sync, or driver
failure is caught and logged, `mmap_support` remains false, and `CPU_Mapped`
remains unsupported. Only a successful full-path probe enables both
properties. The probe covers the CREATE_BO/pinning/subview/coherency path used
later by a GGUF tensor, while keeping `buffer_from_host_ptr=false`: llama.cpp
continues to create the non-owning CPU wrapper around its mmap. With no usable
artifact the capability is explicitly reported as unprobed and disabled, and
backend discovery does not issue a userptr CREATE_BO merely to list an
unusable device.

GGML now exposes an internal identity predicate for the otherwise-private `CPU_Mapped` singleton.
XDNA uses that exact identity instead of accepting all device-less host buffer types.
Writable `CPU` and an owning device's declared host buffer (including the physically validated `ROCm_Host`) remain separate supported cases.
On the installed driver the physical probe logged `DRM_IOCTL_AMDXDNA_CREATE_BO` errno 12 (`ENOMEM`) and failed closed, so `--load-mode none` is still required for an XDNA-eligible model allocation.
The evidence log `mmap-capability-combined-20260812.log` has SHA-256 `19669d9e7b6b6bb2c9e6c7c7aea2d6cfa34899289458f3635b4f717f459e9b42`.
After a driver upgrade, a successful run of the complete probe is the positive gate that enables both `mmap_support` and exact `CPU_Mapped` acceptance; that positive outcome has not been physically validated here.
The already-installed Ubuntu 7.0.0-29 kernel retains the same unconditional
`FOLL_WRITE` pin as the running 7.0.0-27 kernel, so merely rebooting into it
will not enable read-only GGUF registration. Mainline Linux also reverted its
earlier implementation; the reviewed capability boundary is AMD xdna-driver
commit `ed8fb2dd172bde623d7112a1bd674fc0e3c4cae4` or a tested descendant, not
an XRT version string. The current behavior-based probe should remain the
admission rule. After an attended coherent driver/XRT upgrade and reboot, the
positive gate must additionally run a real kernel from an exact read-only
file mapping, match CPU output, report a nonzero XDNA submission and zero
weight-copy bytes, then repeat registration teardown before real-GGUF mmap is
claimed. The primary-source stack audit has SHA-256
`25566727961b37ab2622ab52631587229ed91b7d5bbaf52f17b013e7fd5503a5`.

Immutable `GGML_BACKEND_BUFFER_USAGE_WEIGHTS` take the persistent userptr path.
Ordinary mutable/test buffers use a separate state-owned XRT BO: the runtime
copies and synchronizes the exact native weight bytes on every call and records
them as weight-copy bytes. A seven-call physical stress test changed the Q4
payload on every call, including free/reallocation at the same virtual address;
all results matched CPU, one staging BO was reused, and no userptr registration
was created. This makes pre-allocation capability queries and the standard
backend test executable without weakening the no-secondary-copy claim for
immutable model weights. It also avoids the stale-data failure previously seen
when mutable userptr registrations were recreated at a reused address.

Independent HIP and XRT registrations of exactly the same `ROCm_Host` pages are now physically proven; dma-buf export is not required for this read/read weight case.
The registration cache remains backend scoped, but registrations no longer require their immutable owners to outlive the backend.
GGML now provides a generic, cancelable buffer-free observer.
XDNA subscribes once per owner; the callback resets every run bound to that owner, destroys tensor child views, and then destroys the userptr parent registrations before the provider releases the allocation.
If the XDNA runtime is destroyed first, it cancels each remaining subscription before releasing runtime state.
Scheduler output is not registered: each kernel state owns a persistent output BO sized for that variant and copies it back after synchronization, so context reserve/reallocation cannot leave a transient scheduler allocation pinned.

An `xrt::run` retains its bound weight BO, so owner eviction also clears that binding by destroying the run.
The next call recreates the run lazily and binds the current weight.
A binding exception likewise discards a partially bound run.
If `wait()` throws after submission, the runtime synchronously calls `xrt::run::abort()` before releasing borrowed storage, then resets the run; failure to abort terminates the process rather than risk an active userptr.
Distinct observer handles may be added and removed concurrently while the owner buffer is live.
The caller must serialize owner destruction against buffer use, observer mutation, and backend teardown; observers do not extend allocation lifetime.

Page-window registration avoids pinning an unrelated full model allocation.
Adjacent GGML weights can share a boundary page, so their rounded XRT parents
can overlap. The current driver/XRT pair accepted that arrangement in a real
six-layer GGUF run: 24 distinct weight windows registered and all 24 were reused
on the next token. Registration accounting deliberately reports rounded bytes,
so a shared boundary page can be counted in more than one root.

The pre-observer raw-pointer cache had reproduced stale reuse after an immutable owner was freed and its addresses were recycled.
The post-fix physical ABA regression kept one XDNA backend alive and obtained exactly the same owner, base, and tensor-data addresses for the replacement allocation.
The payload hash changed from `8823c6106c634768` to `58f6252aff3024af`; eviction occurred before the second registration; both CPU-reference comparisons passed; and the final counters reported two root registrations, zero registration hits, two weight views, zero weight-view hits, and two weight syncs.
The evidence log `physical-6.log` has SHA-256 `cde292dfd2adbf704368c2e4945ec5a244470de2918f3f9314f3b971fe7ec7b6`.
This proves serialized destruction and exact-address reuse without stale cache hits; it does not prove concurrent teardown safety.

## Implemented vertical slice

The backend lives in `ggml/src/ggml-xdna/`, builds with `GGML_XDNA`, loads as
`libggml-xdna.so`, discovers the real XRT device as `XDNA0`, and reports ACCEL.
If no XDNA device exists, its dynamic-backend score is zero. If no valid kernel
artifact bundle is configured, the hardware-visible device supports no
compute operations. Neither path emulates XDNA on CPU. With the installed
driver's read-only pinning limitation, default mmap-backed weights are
deliberately unsupported and remain on the scheduler's fallback backend. Use
`--load-mode none --no-host 1` for the controlled generic-CPU path, or
`--load-mode none` with an available HIP iGPU host buffer for the physically
validated shared `ROCm_Host` path.

Both dynamic and static GGML builds were exercised. The installed static CMake
package reconstructs the `XRT::xrt_coreutil` dependency; a fresh external
consumer configured against the install, linked XRT, initialized `XDNA0`, and
loaded the Q4 bundle successfully.

The host no longer treats 288 as a backend property. `problem_from_ggml()`
translates an operation and its tensor metadata into an `xdna_problem`:

```text
GGML_OP_MUL_MAT + tensor metadata
  -> operation and host/device dtypes
  -> M/N/K, batch dimensions, layouts, weight lifetime, device architecture
  -> registry capability match
  -> specialized xdna_kernel_variant
```

Each registry entry owns its shape, architecture, artifact ABI, XRT symbol,
opcode, byte sizes, and worker geometry. The AIE2P Q4_0 288x1x288,
512x1x2560, and 10240x1x2560 kernels, the Q8_0 9216x1x2560 kernel, and the
288x1x288, 8192x1x2048, and 8192x1x3072 BF16 kernels are entries, not switch
cases scattered across the backend/runtime. The artifact packer
likewise receives metadata from the shape-specific build instead of
maintaining a second dtype/shape table. This is deliberately a specialization
registry, not a universally dynamic kernel.

Backend registration discovers every configured bundle and validates its complete header and payload without loading an XRT program. Each device owns one shared resolver whose entries start as `untried`. The first exact `supports_op` query constructs and retains that entry's XCLBIN, hardware context, and kernel, then marks the entry `ready` or permanently `failed`. Concurrent queries serialize this transition, and a failed entry does not invalidate other configured variants. Both device capability reporting and execution use the same `xdna_problem` selector and the same resolved program.

Backend instances do not allocate per-kernel BOs or runs during initialization. The first compute for a ready program creates that backend's instruction, activation, output, temporary, trace, and persistent run state. This second lazy state is also cached, including construction failure. If multiple variants match, a local-state failure advances to the next program in registry order. It keeps mutable staging, output storage, weight registrations, statistics, and command quiescence backend-local while sharing the immutable XRT program objects across backend instances.

All current artifacts own the full AIE array. One execution gate in the shared device cache therefore serializes XCLBIN/context/kernel resolution, backend-local kernel BO/run construction, and steady run creation/binding through `wait()` or error quiescence across every backend instance and artifact on that device. Host packing, registration, activation sync, and completed-output drain remain outside this gate. Teardown and owner callbacks keep the lock order backend compute mutex then shared execution gate.

The resolver inventory, partial-failure fallback, failure caching, concurrent first-use behavior, shared-value lifetime, and execution gate have host-only unit coverage. A physical lazy-resolution process configured all seven current AIE2P bundles and started with `ready=0`, `untried=7`, `failed=0`. It resolved the Q4_0 288x288, BF16 288x288, BF16 8192x2048 and 8192x3072, Q4_0 512x2560 and 10240x2560, and Q8_0 9216x2560 programs only as their exact tests were reached, and passed all seven CPU references. It reported seven successful calls/submissions, seven root registrations (`118.898 MiB`), seven weight views, 42 explicit payload BOs totaling 125,084,806 bytes, seven one-time weight syncs totaling 124,649,024 bytes, and zero weight-copy bytes. The retained raw-log SHA256 is `912be8cf9728dd20aa9ad6b8d52310c4929e71998840704f2909bef442a7959d`; the durable report and evidence-manifest SHA256 values are `2c5c11571669b1482335b82e92055772ec441c7fddde634c868e0214146f87f5` and `3db7f3c853bc6f29d16784b82459f70434146400d83536cb3ef0b9088b036c34`. The device process exited successfully, but the first wrapper post-parser returned a false negative because lazy-resolution messages split six `OK` markers from their `MUL_MAT` lines. The frozen raw log passed an exact seven-case/counter audit; the corrected parser passed `bash -n` and `shellcheck`, and the already complete device process was not repeated.

A separate physical probe created two backend instances from the same registered XDNA device and therefore the same lazily resolved program cache. Both instances executed the Q4_0 288x288 CPU-reference case successfully, then were destroyed in `A -> B` and `B -> A` order in separate processes. The retained log SHA256 values are `f8ffc7ff8fe9c78a4f5f09bb5f8ccc865f27b860777d84f7bfb0baf3b2887225` and `c4fb9596ecd46137fc94afc5f11163281e705a7e66a4e617fe629225e0c5f9ef`. Together these runs validate sequential multi-artifact and shared-program backend lifetimes on the installed XRT stack; they do not claim overlapping AIE execution or exercise an injected `wait()` failure.
UMA weight registrations remain shared across those states.
Malformed optional bundles are rejected without hiding valid variants. An XRT load failure is cached against only the selected artifact, so its exact operation falls back while independently ready or untried variants remain available.

The BF16 8192x2048 artifact also passed seven directed long-double
CPU-reference patterns and both 1,001- and 10,001-launch stress runs in a
dedicated persistent XRT process. Its generic-source replacement then passed
old/new `edge` and `special_finite` A/B launches exactly, with zero mismatches,
non-finites, numerical error, mutations, or weight-payload copies. The old/new
raw-log SHA256 values are
`a2b2a2c04ccbb874fe7e46b4b4fca7783606b8fb2a616db6f56124f0b6617423` /
`878a73bfe524db3ce99561776fa1500592a2623ca20aac4ca00f0f7b772f1331`
for `edge` and
`f850aed5685b790ce2c0ad312f8320ac1de281f9eae188059cb3a6ddbe37a25a` /
`127be011d6c5d7e0711a07cfb239ed8441d5e75be0f39d8d7e1f607006222396`
for `special_finite`. The seven-artifact check above independently proves
coexistence and ordinary GGML backend integration.

The BF16 8192x3072 standalone candidate passed the same directed and sustained
physical gates plus one exact pinned real-GGUF tensor. Its final report has
SHA256
`360e9c724daa27c35c4883e5ed962c931d42468e0788ada437991008c670ca34`.
The generic-source K=2048 regression and seven-artifact process above close
the replacement and integrated-coexistence gates for K=3072.

The configured variant accepts only its exact contract:

- `GGML_OP_MUL_MAT`;
- one of the contiguous native contracts
  `Q4_0[288,288] x F32[288,1] -> F32[288,1]`,
  `Q4_0[2560,512] x F32[2560,1] -> F32[512,1]`,
  `Q4_0[2560,10240] x F32[2560,1] -> F32[10240,1]`,
  `Q8_0[2560,9216] x F32[2560,1] -> F32[9216,1]`,
  the retained `BF16[288,288]` plumbing reference, or
  `BF16[2048,8192] x F32[2048,1] -> F32[8192,1]`, or
  `BF16[3072,8192] x F32[3072,1] -> F32[8192,1]`;
- one AIE2P/XDNA2 device and an explicitly configured versioned artifact bundle.

The bundle couples a self-identifying kernel kind, ABI version, M and K, weight/activation/output byte sizes, architecture identity, xclbin, and instruction stream; N=1 is enforced by the selected variant rather than encoded in the artifact header. The 64-byte container keeps existing AIE2P ABI-v1 bundles compatible with bytes 56-63 reserved as zero. ABI v1 is rejected for AIE2 variants. ABI v2 stores a stable little-endian architecture ID at bytes 56-59 (`1` for AIE2 and `2` for AIE2P) and requires bytes 60-63 to remain zero. The runtime rejects unknown IDs, unknown variant architectures, and AIE2/AIE2P cross-pairing before `supports_op` becomes true. This parser and wire-format separation prepares an AIE2 boundary but does not add an AIE2 registry entry or claim XDNA1 execution.

A deliberate test labeling the BF16 bundle as Q4_0 was rejected with an ABI metadata mismatch and zero submissions, preventing accidental cross-pairing of the generated bundles. The configured bundle is still a trusted executable input; its header does not authenticate the semantics of a manually relabeled arbitrary xclbin.

Every prefill/batched shape and every other unregistered dtype/shape is
rejected. The Q4_0 kernel reads the native 18-byte `block_q4_0` representation
and decodes scale/nibbles on-tile; the Q8_0 kernel reads the native 34-byte
`block_q8_0` scale/int8 representation. Neither kernel repacks nor materializes
a dequantized matrix. The host converts only the small activation vector to
BF16. For immutable model weights it
page-registers each selected tensor window once, uses root-derived sub-BOs,
explicitly cache-flushes each new weight view once, reuses one command/run per
selected artifact, and uses persistent activation/output BOs. That path
performs no weight payload copy. Mutable test weights instead use the explicitly
measured per-call staging path described above. The backend records backend-local
initialization, registration, backend-created BOs, weight and other host-copy
time, TO/FROM sync time, separate XRT `start()` and `wait()` time, and complete
call time. A size-aware v2 statistics API exposes those stage counters through
dynamic backend lookup while preserving the original stats ABI. Artifact parsing happens during backend registration before the runtime initialization timer. Shared XCLBIN/context/kernel construction happens during capability resolution, logs its own wall time, and is not charged to a later backend instance's initialization counter. Kernel source and reproduction instructions are in
`ggml/src/ggml-xdna/kernels/aie2p/`.

The clean-built nine-worker Q4_0 bundle used for the final physical tests is
intentionally ignored by Git and has SHA-256
`31c0d6bcd4e570b298fac81da801934a99682f0af62789170c5a3383487c583f`.
It contains a 44,505-byte xclbin with SHA-256
`38a1dc1dff9bb361ffcee9feed90c3942bdf91305ec0c40cf0de6faa1ece922a`
and a 988-byte instruction stream with SHA-256
`9e85c37d45e085ef2f597a554b913e9744bdabc558313ae5473808f4e0da15ed`.
The generated MLIR and Peano object hashes are respectively
`942590208bd114b5c92d77b3ee3c4634831fa02f65eea2f9eb64fabc356f10e5`
and `2d7c750e15a93991e6b5b475aab633b3575930291a3019a3972761a0aabc0755`.
The xclbin and therefore bundle are not bit-reproducible because the current
toolchain writes a fresh UUID, timestamp, and unique ID; the kernel object,
instruction stream, ABI header, and generated MLIR were deterministic across
clean serial builds.

`test-backend-ops` compared the BF16 XDNA result with CPU successfully. A steady
run of 8,190 measured iterations reported 695.70 us/run, 238.45 MFLOP/s, one
root registration, one weight view, 8,190 weight-view hits, and zero weight-copy
bytes. The 165,888-byte BF16 matrix achieved about 0.238 GB/s.

## Llama 3.2 1B BF16 production shape

The first production BF16 specialization is the Llama 3.2 1B gate/up
projection: native `BF16[M=8192,K=2048] x F32[K=2048,N=1]`. The exact source
model is `unsloth/Llama-3.2-1B-Instruct-GGUF`, file
`Llama-3.2-1B-Instruct-BF16.gguf`, revision
`b69aef112e9f895e6f98d7ae0949f72ff09aa401`. The 2,479,595,264-byte file has
SHA-256
`f5d05e90d179fa27c01a049be1574a0201bdc09ba0b345eb85e1dbf3b87610f3`;
its tensor census reports gate/up dimensions `[2048,8192]`.

A native row is 4,096 bytes. The AIE2P core-DMA length field accepts at most
16,383 32-bit granules, so a 16-row, 65,536-byte object is illegal by one
granule. The selected eight-row object uses 8,192 granules. Eight columns by
four compute rows provide 32 workers, 256 output rows per wave, and 32 waves.
One BF16 activation acquisition remains live across all waves. Eight memtile
groups each split four worker-weight objects and join four outputs. The placed
core uses 41,004 of 65,536 local bytes and each group uses 131,200 of 524,288
memtile bytes. The current shim schedule repeats one task per stream across all
32 waves while retaining depth-one object FIFOs.

The final three-row plus two-row compute grouping has no stack references.
Peano `-O2` and `-O3` produced the same 2,036-byte object with SHA-256
`e20e70fa41d2c778d17bbc5b337555609cf8f04a0ebb49cfae229e152c7d3413`.
The retained 513-task ABI-v1 bundle is 188,051 bytes with SHA-256
`6f9132673cf1f6fc90445766dcadb07edddffa6d62f9b3a6e1574dc62fb62d1f`;
its 72,884-byte instruction stream has SHA-256
`91d5b29a0b6ff922d2a7ef1ae2ab1eee2316eaeccf82b402f71721771f3adcfc`.
These are the physically validated artifact hashes. A clean Makefile rebuild
reproduced the MLIR, compute object, addressed MLIR, PDI, and instruction
stream byte-for-byte. The rebuilt xclbin container has a newly generated UUID,
so its xclbin and enclosing bundle hashes differ.
Generated artifacts remain ignored by Git.

Seven physical patterns covered zero weights, zero activation, first/last K
impulses, finite BF16 normal/subnormal extremes, worker/wave/lane boundaries,
and deterministic dense input. Every F32 output matched a long-double sum of
the exact BF16 inputs, with zero maximum error and zero NMSE. Both 1,001- and
10,001-launch persistent runs completed. The 10,001-call kernel-only result was
721.268 us p50, 767.737 us p95, and 46.521 GB/s over the 33,554,432-byte
matrix. Including F32-to-BF16 activation conversion plus activation/output
synchronization measured 722.667 us p50. Setup registered one parent, created
three child views, synchronized the native weight once, and copied zero weight
payload bytes per call.

The equivalent single-call gfx1151 control used the same native BF16 bytes,
BF16-exact logical F32 activation, CPU reference, persistent weights, and one
synchronization per sample. A clean D-H-H-D order used two 1,001-call trials
per HIP placement. Device-resident trial medians were 406.148 and 370.059 us;
`ROCm_Host` medians were 370.748 and 413.128 us. The mean of each balanced pair
was 388.104 and 391.938 us, making XDNA 1.858x and 1.840x slower respectively.
An earlier 10,001-call HIP sequence overlapped a low-priority compiler and is
excluded. The variant remains available for explicit coverage and XDNA-only
execution but sets `prefer_for_offload=false`.

## Llama 3.2 3B BF16 production shape

The next specialization is native
`BF16[M=8192,K=3072] x F32[K=3072,N=1]`, matching the gate/up projections in
`unsloth/Llama-3.2-3B-Instruct-GGUF`, file
`Llama-3.2-3B-Instruct-BF16.gguf`, revision
`e7d0997e49c9cb00d88b4c1a6a16aa894b0bbc31`. The 6,433,687,744-byte file has
SHA-256
`9b8dce13b6cbcd8b20037bd6383ee8e747b5034ca32f40b5b8ee2efa9ebf56b8`.
Its 28 layers contain 28 gate and 28 up tensors of this shape, or 56 exact
candidate GEMVs per generated token. The M=3072, K=8192 down projections need
a different kernel despite having the same matrix byte count.

A strict cache-only real-GGUF runner used one bounded `pread` for only
`blk.0.ffn_gate.weight` at byte offset 846,186,688. The 50,331,648-byte BF16
tensor has SHA-256
`f38192e76c00a2829113cb3202b42df45687d46b68a8682f5b1b0e46073e5ee0`.
One full XDNA launch with a deterministic exact-BF16 activation matched the
long-double reference across all 8,192 rows with zero mismatches and
non-finites, NMSE `5.2182310884410422e-15`, unchanged native weight bytes, and
zero post-ingest weight-payload copies. The cold 3.262 ms launch is correctness
evidence, not a steady performance result.

The K=3072 program preserves the K=2048 spatial mapping: 32 workers, eight rows
per worker, 256 rows per wave, 32 waves, and depth-one FIFOs. A worker weight
object uses 12,288 of the 16,383 available 32-bit core-DMA granules. The pinned
toolchain's placed MLIR reports a 12-byte IRON state object per worker; the
generator budgets this toolchain-observed footprint rather than treating it
as an architecture guarantee. Stack, weights, activation, output, and that
state use 59,436 of 65,536 compute-tile bytes, leaving 6,100. Each memtile
group uses 196,736 of 524,288 bytes.

Peano `-O2` and `-O3` produced byte-identical 2,056-byte objects with
SHA-256
`709be05f3ea7a728633af78428575706b3f13afc38c5cd4281da522d2ce8d564`,
so the build retains `-O2`. The retained 513-task ABI-v1/kind-1 bundle is 188,563
bytes with SHA-256
`6a682f17a066a76aef2e865ef3fb56427c0ab55d42132438655994db8d4ef641`.
Its 72,884-byte instruction stream has SHA-256
`687b057d10918c2cda192c175934dc4b70b3e805b1c5d44145657a2be661b14c`.

Seven directed patterns passed twice, and both 1,001- and 10,001-launch random
closures completed with zero mismatches and invalid iterations. The
10,001-call kernel-only p50 was 1,045.054 us (48.162 GB/s); verified-full p50
was 1,044.585 us. Clean order-balanced 1,001-call HIP controls measured
526.994 us kernel and 549.750 us full p50 for `ROCm0`, and 539.854 us and
583.869 us for `ROCm_Host`. XDNA is 1.789x to 1.983x slower for the matched
single-projection controls, so this descriptor also keeps
`prefer_for_offload=false`.

The reusable source parameterizes K at compile time; it does not claim a
dynamically shaped artifact. At K=2048 it reproduced the generated and
addressed MLIR, PDI, and instruction stream byte-for-byte. The complete
compute ELF differs only in source-file metadata; its code, allocated,
relocation, comment, and stack sections and normalized symbols match, as do
the corresponding sections in all 32 placed ELFs. The regenerated program
then passed exact old/new `edge` and `special_finite` physical A/B tests, and
both generic K variants passed in the single seven-artifact ordinary backend
process. These gates extend the standalone K=3072 evidence to tracked
integration without changing its opt-in policy.

### Repeated shim-task fusion

The tracked generator now replaces 256 one-wave weight tasks and 256 one-wave
output tasks with eight 32-wave repeated weight tasks and eight repeated
output tasks. Including activation, this intentionally reduces the shim task
count from 513 to 17 and the controller stream from 72,884 to 2,452 bytes.
The numerical kernel, 32 cores, 112 flows, 16 links, depth-one FIFOs, routes,
locks, memory placement, and host ABI are unchanged. Static comparison found
the non-runtime addressed MLIR identical. The K=2048 placed core ELFs were
byte-identical. Each K=3072 placed core differed only in non-loadable `STT_FILE`
source-name metadata; its executable sections and normalized executable symbols
were identical. A post-`aiecc` verifier checks the exact v0.1 operation order,
absolute shim addresses, reserved fields, repeats, encoded 20-bit strides,
host offsets, and final buffer endpoints before packaging either K value.

For K=2048, the 15/15 phase-one matrix passed FIFO parity, all directed cases,
1,001-launch closures, and exact bounded reads of both Llama 1B
`blk.0.ffn_gate.weight` and `blk.0.ffn_up.weight`. The balanced phase-two A/B
matrix reduced full p50 from 724.2425 to 668.540 us (7.691%) and kernel p50
from 721.792 to 664.927 us (7.878%). One fused full-path attempt was excluded
only for swap movement; its retry passed, and every accepted cell passed the
strict numerical and state checks. The durable report and top-level evidence
manifest have SHA-256
`39b3deaf9aca6dd83b984095e0961ee71d1e743c52fe4b6dd83006eb5c4f1ac9`
and
`25d36584702974fe7b577e5ffd1b7545131a90d48144956f85b09db68ec18bd8`.

For K=3072, the repeated-task artifact retained exact correctness for the
pinned Llama 3B gate tensor and reduced balanced full p50 from 1,055.645 to
979.295 us (7.232%) and kernel p50 from 1,049.432 to 961.773 us (8.353%). Its
final report has SHA-256
`5aa8131d9bdc3b681a87d80848b71464f7f8ee77b9f5b92f44d1fe3b54e70b6c`.
HIP remains faster for both shapes, so task fusion changes neither descriptor
to automatic placement: both keep `prefer_for_offload=false`.

A source-only alternative tested whether explicit ping-pong could lift the
bandwidth plateau. It reduced each worker to four rows, doubled the wave count
to 64, and used depth-two weight/output FIFOs. Seven directed patterns, the
exact pinned tensor, and 1,001 alternating FIFO-parity launches passed, so the
layout is functionally viable. Its generated and addressed MLIR SHA-256 values
are
`d52039080a9f79fc34edef8ec75d9b711c64120804d1987497c71e4c8d6ae434`
and
`ae106f308c6b534d413140b90f2f36790b81b996ac7951eb5516f2d7182aa7a0`;
the bundle and instruction SHA-256 values are
`2443282a1dd8300b4313012fc62f42b581eeae93d5b33fd6c0cfd3bdaeafaadd`
and
`9a1862a4db88d770df617b12d86ad729b9c16e9166a9bfa8dbc9bc4eeba3b6ee`.
Clean balanced full p50 regressed from 1.0663185 ms at depth one to 1.2232145
ms (14.714%), while kernel-only p50 regressed from 1.0363775 ms to 1.197059 ms
(15.504%). Depth two is rejected; the tracked generator retains the
eight-row, depth-one layout. The physical-log and balanced-result manifests
have SHA-256
`497f0fa807c218c0781f48ba5f442e380e9d829adcc29b22c6ddcdb73e3c2e2e`
and
`cdf06776693f86702f1a0d54ccf0a8b893365790ffd003ba91940637c3e41def`.
The final rejection report has SHA-256
`1cee68e52bdf4c296fd1e0755f353836e024ba0c51f57a1c3c84981d94d25306`.

## AIE2P vector and multi-worker performance sprint

The original scalar, one-worker artifact was re-run for 101 persistent launches
after the rebase: 12,973.193 us steady and 0.003596 GB/s of native compressed
weights. The optimized compute primitive now performs packed `uint4` vector
loads, AIE2P nibble unpack/deinterleave, vector subtract-eight, int8-to-BF16
conversion, vector MAC, and vector reduction. The final packed block in each
32-row FIFO object uses an exact staged load because the fast unaligned AIE
primitive may physically read a wider surrounding region.

AIE2P/arch21 does not provide IEEE-FP16 arithmetic for GGML's scale. The kernel
decodes the binary16 bits exactly to FP32, then deliberately rounds the scale to
BF16 so block scaling stays on the vector datapath. That is a numerical contract
change, not a bit-exact reordering. GGML's random CPU-reference test passes its
Q4 tolerance. Separate physical patterns covered zero quant values, min/max
nibbles, positive/negative small and large scales, deterministic random
weights/activations, zero activation, and small/large activation magnitudes;
all seven completed five persistent launches with zero mismatches. A 101-run
structured test also had zero mismatches and zero maximum absolute error.

Three row accumulators reuse each activation vector load. Four rows regressed
because the pinned Peano compiler spilled vector state. The IRON program then
instantiates the proven 32-row primitive nine times: workers `(0..7,2)` plus
`(0,3)`. One shim stream broadcasts the activation to all workers. Native GGML
weight rows and output rows are split/joined in `4 + 4 + 1` groups across memory
tiles, respecting the five-input/five-output DMA-channel limit. No weight is
repacked or dequantized, and no secondary weight allocation is introduced.

An eight-worker, 36-row-per-worker topology also compiled, allocated, and
routed, but returned deterministic incorrect results on hardware. A matching
single-worker 36-row probe failed too, isolating the problem to that compiled
36-row primitive rather than the multi-tile routes. It was not committed or
used for timing; the proven 32-row primitive was retained and scaled to nine
workers instead.

The serial standalone XRT ladder below uses the same 46,656-byte matrix, 101
persistent launches, and excludes the first launch from the steady average:

| Artifact | Steady start+wait | Effective weight GB/s | Speedup vs scalar |
| --- | ---: | ---: | ---: |
| scalar single worker | 12,973.193 us | 0.003596 | 1.00x |
| vector unpack/reduction, exact FP16 scale | 373.288 us | 0.124986 | 34.75x |
| BF16-scaled, two-row activation reuse | 170.975 us | 0.272882 | 75.88x |
| BF16-scaled, three-row activation reuse | 135.050 us | 0.345471 | 96.06x |
| nine workers, three rows per inner step | **74.301 us** | **0.627932** | **174.60x** |

Two final-source GGML perf runs measured 16,384 steady calls at 70.41 and 71.56
us/run, 2.32-2.36 GFLOP/s, and **0.652-0.663 GB/s**. In the 70.41 us run, a
fresh process spent 127.649 ms in XRT device/kernel initialization; its first
kernel start+wait was 0.792 ms and its first complete backend call was 0.875
ms. Across 16,385 successful calls, XRT
`start()` averaged 1.962 us and `wait()` 67.296 us. Activation conversion,
activation sync, output sync, and output copy averaged respectively 0.096,
0.039, 0.059, and 0.230 us/call. One weight view was created and synced once;
16,384 subsequent calls hit it, with zero weight-copy bytes.

A minimal AIE2P increment design measured 65.224 us average start+wait (52.480
us minimum) over 100 steady launches. It is a different program, so subtracting
it does not yield an exact compute time, but it shows that the final 288x288
kernel is now close to the installed XRT/AIE command envelope rather than
spending milliseconds in scalar arithmetic. The retained single-worker BF16
reference remains slow at 693.76 us/run and is not an optimized comparator.

The exact GGML HIP test on the Radeon 890M (`gfx1150`) measured 5.91 us/run,
28.07 GFLOP/s, and about 7.895 GB/s. The optimized XDNA result is therefore
about 12x slower than HIP for this very small operation, versus more than
2,000x for the original scalar artifact. Vectorization and multi-worker scaling
removed the catastrophic underutilization, but this fixed shape is not yet a
decode-performance win over HIP.

The final nine-worker source was also built independently with Peano `-O2` and
`-O3`. The worker objects, MLIR, instruction stream, PDI, and all nine core ELFs
were byte-identical; only generated xclbin UUID/path metadata differed. Three
10,001-launch trials measured 67.179 us median for `-O2` and 67.086 us for
`-O3`, a 0.14% difference with identical executable code. `-O2` remains the
defensible default. Replacing `xrt::run::wait()` with `wait2()` changed the
median from 70.627 to 70.014 us, only 0.613 us or 0.87%; host wait API choice
does not explain the roughly 60-70 us command floor.

## First production-shape kernel

The first real-model specialization is the Gemma 4 E4B sliding-window K/V
projection: native `Q4_0[M=512,K=2560] x F32[K=2560,N=1]`. It maps 16 output
rows to each of all 32 AIE2P compute tiles. Eight memory-tile groups split one
737,280-byte native-weight matrix into four 23,040-byte worker objects each,
broadcast one 5,120-byte BF16 activation, and join one 2,048-byte F32 output.
The 23,040-byte core DMA occupies 5,760 of the descriptor's 16,383 32-bit
address granules. No weight repack or dequantized matrix is created.

The current independently rebuilt bundle has SHA-256
`040d2d30463f990be44aa40449cc1066628476f3451e2fb966b502c144cedf3e`
and declares artifact kind 2, ABI 1, M=512, K=2560, 737,280/5,120/2,048-byte
weight/activation/output operands. The physical CPU-reference test passes.

The initial production artifact measured a 107.93 us median start+wait and
6.83 decimal GB/s. Replacing repeated block-index address arithmetic with
three-row pointer induction reduced the order-balanced physical mean from
107.844 to 106.757 us over 8,008 launches of each variant, a 1.01% gain and
6.91 GB/s. All 16,016 A/B launches were exact. The change also removed 11 VLIW
bundles and shrank each core kernel by 80 bytes; Peano `-O2` and `-O3` again
produced identical core objects, so `-O2` remains selected.

HIP comparison depends materially on dispatch methodology. The earlier
`test-backend-ops perf` graph packed 8,191 identical nodes into one submission
and measured 8.55 us/op; that is throughput, not one synchronized token call.
A later fair single-invocation test measured 21.40 us with a device-resident
weight and 24.25 us reading the same shape directly from `ROCm_Host`. The
current XDNA result is therefore about 4.99x and 4.40x slower respectively for
one isolated invocation. The larger shape proves that XDNA escapes the
few-MB/s regime, but it does not support a hybrid-performance win claim yet.

A native integer alternative was also taken through physical A/B rather than
accepted from code generation alone. A one-tile M=16/K=64 kernel consumed
native GGML Q4_0 and Q8_0 blocks and matched every directed, lane-basis,
scale-edge, and random reference exactly; its 10,001-launch `wait2()` average
was 60.746 us. Scaled to the same 32-tile M=512/K=2560 topology, the exact-FP32
per-block-scale implementation remained correct (maximum observed absolute
error 2.86e-6) but measured 196.42 us steady start+wait and 196.85 us for the
complete quantize/sync/run/copy path. Current GGML F32-to-Q8_0 quantization
itself costs only 0.30 us hot, so host quantization is not the regression.
The AIE2P exact-scale instruction sequence and native-layout transpose/gather
work make this path about 82% slower than the BF16-activation production
kernel. Rounding the scales to BF16 was both slower in a short run and failed
the GGML NMSE limit (0.00262 random, 0.01129 min/max versus 0.0005). Native
Q4_0xQ8_0 is therefore a correctness **GO** but a performance **NO-GO** for
this variant; it is not integrated into the registry.

The next experiment targeted the dominant E4B decode projection rather than
extrapolating from K/V: native `Q4_0[M=10240,K=2560] x F32[K=2560,N=1]` for
the gate/up matrices. The AIE2P program uses all 32 compute tiles in twenty
512-row waves, holds one BF16 activation acquisition across all waves, and
keeps the depth-one forwarding contract. The selected topology uses eight
independent memory-tile weight groups. A 15-stream alternative occupied all 16
shim MM2S channels but nearly doubled the controller instruction stream and
was consistently 0.9-1.3% slower, so it was rejected.

Both physical stream-count artifacts matched their deterministic
BF16-semantics references exactly. With the required activation/output
synchronization enabled, the original eight-stream schedule measured 1.050877
ms median start+wait and 1.051317 ms for the complete pack/sync/run/copy path,
or 14.03 GB/s over the 14,745,600-byte native Q4_0 matrix. Registration cost
0.862 ms once and the one-time weight sync cost 0.118 ms; the persistent
parent/view/run path performed no weight payload copy.

The same depth-one topology then batched multiple waves into each runtime task
group. Two waves measured 865.620 us and four measured 778.540 us over 10,001
exact calls; four waves improve latency by 25.9% and effective compressed-weight
throughput by 34.9% to 18.94 GB/s. Eight waves measured 741.500 us over 101
calls but hung in `wait2()` during the 10,001-call stress test and were rejected.
All twenty waves fail lowering when shim BD IDs are exhausted at wave nine.
The integrated four-wave artifact rebuilt independently with MLIR-AIE 1.3.4
and Peano `cb664e8c`; its instruction stream is byte-identical to the physically
tested artifact, and the current GGML backend correctness test passes on the
NPU. The three-row pointer-induction change was then validated in an
order-balanced physical A/B: mean latency fell from 778.872 to 765.908 us
(1.67%), all four adjacent M10240 pairs improved, and all 8,008 launches per
variant were exact. Effective weight bandwidth is 19.25 GB/s from the aggregate
mean. The current rebuilt bundle is 187,411 bytes with SHA-256
`3d911dfbf792c9e1eae13c73bf15abe41533f5793df995931302424c67483605`.

The exact current HIP graph-exported operation measured 167.24, 157.53, and
162.45 us in three independent runs (162.45 us median). A separate synchronized
single-call benchmark measured 155.87 us device-resident and 161.96 us directly
from `ROCm_Host`. The current XDNA artifact is therefore about 4.7-4.9x slower.
A paired gate/up artifact was also physically correct, but its 2.052723 ms
complete median is about 34% slower than two current independently optimized
XDNA calls and about 6.3x slower than two direct-host HIP calls. The current
paired topology is therefore rejected; fusion helps
command-bound K/V more than bandwidth-bound gate/up. This shape remains
valuable because it covers the majority of compressed E4B linear-weight traffic
and exposes the real XDNA bandwidth regime, not because it currently wins
device selection.

## Native Q8_0 production-shape milestone

The first non-Q4 quantized production specialization targets the exact Qwen
3.5 4B gate/up decode operation: native
`Q8_0[M=9216,K=2560] x F32[K=2560,N=1]`. The 32-layer model executes two such
projections per layer, so this one descriptor represents 64 calls and
1,604,321,280 native Q8_0 weight bytes per generated token. Selection remains
generic: architecture, operation, dtype, layout, M/N/K, batches, and weight
lifetime determine capability; neither the registry nor runtime checks a model
name.

Each native `block_q8_0` is the GGML 34-byte layout: one IEEE FP16 scale and 32
signed int8 lanes. A K=2560 row is 2,720 bytes and the complete M=9216 matrix
is 25,067,520 bytes. The retained eight-stream IRON topology maps four 16-row
workers to each of eight memory-tile groups, uses all 32 compute tiles for one
512-row wave, and completes the matrix in 18 waves. A worker transfer is
43,520 bytes, or 10,880 of the AIE2P core DMA descriptor's 16,383 available
32-bit granules. The activation is broadcast once and held across all waves;
FIFOs remain depth one and four waves share each controller task group.

The host packs only the 5,120-byte F32 activation to BF16. On-tile code converts
the FP16 block scale to FP32 and then rounds it to BF16, converts the signed
lanes to BF16, rounds the scaled lane vector to BF16, and performs BF16 MACs
into F32 accumulators. This is an explicit numerical contract rather than an
exact FP16-scale reorder. The final row and final 34-byte block use exact local
tail staging so the fast unaligned vector load cannot read outside the worker
FIFO object.

The MIT compute source and Apache-2.0-with-LLVM-exception IRON generator were
built with MLIR-AIE 1.3.4 and Peano commit
`cb664e8cc3eb42a12e3ad3cee28729785ffa97a3`. Explicit Peano `-O2` and `-O3`
builds were byte-identical: a 2,892-byte object with a 1,920-byte text section
and SHA-256
`231e12d85ed26c1f5585f21fa1876e6de72d0154e854900f35e84aa3604c9f9f`.
The Makefile therefore retains `-O2`. The local generated ABI-v1/kind-3 bundle
is 237,139 bytes with SHA-256
`1a8b653d0de85f65825b389e50529bf5b17091c7780feae33367713985c73cdf`;
its 41,076-byte instruction stream has SHA-256
`cd1f298f20b075ef3b8d22e3402e56d49028227e39ab63a190b55087b85d7482`.
Generated binaries remain ignored by Git.

The deterministic GGML backend test passed against the CPU backend at the exact
shape. A standalone physical CPU-reference suite additionally covered zero
weights, zero activation, three fixed random seeds, int8 -128/+127 lanes,
zero/positive/negative/small/large FP16 scales, and the last row/block. The
zero cases were exact. Against a CPU calculation of the kernel's explicit
BF16-rounded dequant semantics, the seeded cases measured NMSE from
`3.99554e-13` through `4.17706e-13`, maximum absolute error from `0.0185547`
through `0.0234375`, and final-row error from `0.00146484` through `0.00634766`.
No NaN or tolerance failure occurred. The GGML CPU comparison separately
guards the public F32-input tolerance rather than treating this internal
BF16-semantics reference as a substitute.
The committed deterministic test includes FP16 scales derived from `0.0007f`
and `-0.0137f`, which are not BF16-exact, so it directly exercises the
production FP16-to-BF16 scale-rounding difference in addition to zero and
power-of-two edge values.

Five persistent `test-backend-ops` processes completed 10,605 physical
submissions. Their per-process means were 792.26, 799.01, 794.53, 790.70, and
799.07 us/call; the median is 794.53 us, 31.55 decimal GB/s over native Q8_0
bytes, and 59.39 GFLOP/s. Each immutable weight was registered and synchronized
once per process; steady calls copied zero weight payload bytes.

A bounded 15-weight-stream experiment used the same compute object, 32 tiles,
16 rows per worker, 18 waves, depth-one FIFOs, and four-wave task grouping. It
passed the same directed reference, but enlarged the controller instruction
stream from 41,076 to 76,860 bytes. In balanced adjacent order, the retained
eight-stream medians were 794.821, 786.061, 785.790, and 786.281 us; the
15-stream medians were 882.902, 883.482, 875.522, and 875.182 us. The candidate
lost every pair and was 11.84% slower by aggregate median, so it remains
cache-only and is not a registry variant.

The exact `gfx1151` HIP operation independently passed a deterministic CPU test
and measured 269.80 us median across five device-resident `test-backend-ops`
processes, or 92.91 GB/s and 174.89 GFLOP/s. A separate synchronized allocation
probe measured 247.858 and 268.538 us from ROCm device memory and 290.578 and
282.668 us directly from `ROCm_Host`; direct-host output was bit-identical and
incurred zero weight copy. Explicit generic-CPU staging plus compute was about
786-819 us and is not conflated with the HIP kernel ceiling. The comparable
GGML throughput result makes the current XDNA artifact about 2.94x slower.
Consequently the policy-enabled descriptor must set
`prefer_for_offload=false`; Q8_0 capability is a format/shape milestone, not an
automatic hybrid-performance win.

A pinned-toolchain defect remains isolated: depth-two forwarding corrupted one
row on alternating launches. The nine-worker artifact keeps every FIFO at depth
one and gains parallelism through independent workers. The production gate/up
artifact safely amortizes controller work with four queued depth-one waves, but
makes no DMA/compute-overlap claim; ping-pong remains a separate artifact and
physical A/B test.

## Hybrid scheduling and real-GGUF evidence

A controlled scheduler probe first used backend order `[ROCm0, XDNA0, CPU]`,
one generic CPU `Q4_0[288,288]` weight, and `op_offload=true`. For `n=1`, HIP
did not accept the buffer or offload the small operation, so XDNA executed it.
For `n=32`, XDNA rejected the shape and HIP accepted the prefill operation,
copying its split inputs. Both matched their references. This established the
decode-selective policy before a shared allocation was available.

A second guard probe wrapped the same writable test storage with GGML's
`CPU_Mapped` buffer type. XDNA reported that buffer unsupported and the batch-one
node stayed on CPU; changing only the buffer to an owned `CPU` allocation selected
XDNA and executed correctly. This proves the installed-driver mmap limitation is
handled before scheduling rather than surfacing as a failed XRT submission.

With all three Q4_0 artifacts configured, the current E4B graph export reports
both `Kcur-0` (512x1x2560) and `ffn_gate-0` (10240x1x2560) supported on XDNA,
while their otherwise identical `N=16` prompt variants are rejected. The
production specializations therefore preserve the intended decode-selective
capability policy instead of stealing prefill from ROCm.

The decisive shared-memory probe used a real 737,280-byte `ROCm_Host`
allocation. Its base was page-aligned, HIP consumed it directly, and XRT
registered that exact pointer as one parent plus one tensor view. The sequence
XDNA decode -> HIP prefill -> XDNA decode retained one registration, one
first-use 737,280-byte cache-maintenance sync, one registration hit, and zero
XDNA weight-copy bytes. Both engines matched CPU and the two XDNA outputs were
bit-identical. This proves one shared system-memory weight allocation, not zero
data movement.

The integrated scheduler lets an ACCEL backend's exact `offload_op` request
override ordinary owner priority only for immutable host-backed weights. Before
the evidence-based XDNA preference gate, three physical processes using
`[ROCm0, XDNA0, CPU]` selected XDNA for `N=1` and ROCm for `N=32` from the same
`ROCm_Host` pointer. XDNA admits that buffer through the generic
owning-device/declared-host-buft relationship, without a `ROCm_Host` name check,
while continuing to reject `CPU_Mapped`. The XDNA registration survived the
intervening HIP execution and again reported zero weight-copy bytes. Current
default policy leaves mutually supported decode on the earlier ROCm backend;
`GGML_XDNA_PREFER_OFFLOAD=1` intentionally restores the measured XDNA placement
for mode-C experiments without widening XDNA capability.
Large-model pinned-allocation failure remains an open productization gate; serialized immutable-buffer free-time eviction now passes the exact-address regression described above.

The local `stories15M-q4_0.gguf` is a real 24.41-million-parameter, six-layer
Llama model with embedding width 288. A graph exported by the current
`test-export-graph-ops` contains both of these actual nodes:

```text
MUL_MAT Qcur-0: Q4_0[288,288] x F32[288, 1] -> F32[288, 1]   SUPPORTED
MUL_MAT Qcur-0: Q4_0[288,288] x F32[288,32] -> F32[288,32]   NOT SUPPORTED
```

With `--load-mode none --no-host 0 -ngl 0`, llama.cpp allocated the 17.50 MiB
model in `ROCm_Host`: one writable system-memory allocation declared by HIP and
accepted by XDNA. A current `llama-bench` run used `pp32` and `tg2` as separate
tests. Prompt processing produced zero XDNA submissions; the quantized prefill
splits remained on `ROCm0` while unsupported attention work used CPU. The
batch-one graph then executed 24 matching attention projections per token on
XDNA. The backend logged the real `Qcur-0` node and completed 48 submissions for
the two generated tokens.

That real graph exposed a scheduler correctness bug before the final pass. A
CPU/ACCEL-produced transient such as `ffn_inp` lived in `ROCm_Host`, and HIP
reported that buffer type supported. The scheduler therefore omitted a split
copy; a later HIP compute operation treated the host pointer as a device-local
intermediate and reported an illegal access. With fusion disabled, the same
fault surfaced directly in HIP `ADD`, so fusion changed where the error was
observed rather than causing it. Staging XDNA's weight instead of
registering it did not change the failure, ruling out XRT weight registration
as the cause. The generic scheduler now materializes cross-backend transient
host values into a GPU/IGPU compute buffer while leaving immutable shared-host
weights direct. The passing trace shows the affected ROCm splits explicitly
listing both `norm` and `ffn_inp` inputs.

The final two-token decode run registered 24 tensor-local page windows and 24
weight views, then recorded 24 weight-view hits. Rounded registered storage was
1.160 MiB rather than the full 17.50 MiB model. Weight synchronization remained
fixed at 24 calls / 1,119,744 bytes and `weight_copy_bytes` remained zero. All
48 XRT start+wait intervals totaled 4.661 ms and complete XDNA calls totaled
5.340 ms; registration itself took 0.538 ms. Activation packing/sync, output
sync, and output copy totaled 0.008/0.008/0.011/0.015 ms. The run reported
700.02 prompt tokens/s and 157.95 generation tokens/s, but it deliberately used
one no-warmup repetition and a tiny model; these are liveness numbers, not a
stable end-to-end performance comparison.

An otherwise identical HIP-only control completed, as did the final hybrid
run. No fair all-HIP versus hybrid token-throughput result is claimed here: the
current evidence isolates scheduler correctness, placement, XRT reuse, and
per-kernel costs. A representative model with all required XDNA shapes and
order-balanced repeated inference remains the end-to-end performance gate.

For a focused correctness rerun after building the generated kernel artifacts:

```sh
export GGML_XDNA_AIE2P_Q4_0_GEMV_BUNDLE=$PWD/ggml/src/ggml-xdna/kernels/aie2p/gemv-q4_0-288.ggmlxdna
build/bin/test-backend-ops test -b XDNA0 -o MUL_MAT \
  -p 'type_a=q4_0,type_b=f32,m=288,n=1,k=288'
```

## Four-model ROCm reference and KV-cache matrix

The end-to-end baseline now covers the exact four-file model matrix below. All
files were read from their Hugging Face cache snapshots and independently
matched the immutable LFS size and SHA-256; no alternate quantization was
substituted.

| Model repository and revision | Exact file | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `unsloth/Llama-3.2-1B-Instruct-GGUF@b69aef112e9f895e6f98d7ae0949f72ff09aa401` | `Llama-3.2-1B-Instruct-BF16.gguf` | 2,479,595,264 | `f5d05e90d179fa27c01a049be1574a0201bdc09ba0b345eb85e1dbf3b87610f3` |
| `unsloth/Llama-3.2-3B-Instruct-GGUF@e7d0997e49c9cb00d88b4c1a6a16aa894b0bbc31` | `Llama-3.2-3B-Instruct-BF16.gguf` | 6,433,687,744 | `9b8dce13b6cbcd8b20037bd6383ee8e747b5034ca32f40b5b8ee2efa9ebf56b8` |
| `unsloth/Qwen3.5-4B-GGUF@e87f176479d0855a907a41277aca2f8ee7a09523` | `Qwen3.5-4B-Q8_0.gguf` | 4,482,403,488 | `10cc391b403021dd11c614679d2fd92f611c3681d29e29651b717316965d61e1` |
| `unsloth/Qwen3.5-9B-GGUF@3885219b6810b007914f3a7950a8d1b469d598a5` | `Qwen3.5-9B-Q4_K_M.gguf` | 5,680,522,464 | `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8` |

The A1 control isolated ROCm and CPU plugins; XDNA was not loaded. Every cell
used context 2,048, `pp512`, `tg128`, batch/ubatch 512/512, 12 threads, three
benchmark repetitions, same-type K/V, Flash Attention enabled, and ROCm0
resident-fit with a 768 MiB margin. Each configuration first passed a fixed,
deterministic 128-token generation sanity gate and a graph-placement gate.
Values below are mean prompt/decode tokens per second; the parenthesized value
is the sample standard deviation.

| Model (weight type) | F16 K/V, pp / tg | BF16 K/V, pp / tg | Q8_0 K/V, pp / tg | Q4_0 K/V, pp / tg |
| --- | ---: | ---: | ---: | ---: |
| Llama 3.2 1B (BF16) | 2535.754 (115.019) / 35.570 (0.852) | 2353.492 (518.048) / 34.948 (0.517) | 2374.220 (490.327) / 35.712 (0.944) | 2284.177 (629.483) / 35.448 (1.056) |
| Llama 3.2 3B (BF16) | 1010.482 (103.455) / 13.357 (0.167) | 1036.403 (111.176) / 13.185 (0.086) | 1084.211 (56.175) / 13.749 (0.079) | 1027.002 (89.202) / 13.714 (0.089) |
| Qwen3.5 4B (Q8_0) | 530.053 (6.695) / 17.561 (0.113) | 510.584 (7.472) / 17.289 (0.074) | 521.133 (3.431) / 17.516 (0.523) | 506.150 (11.912) / 17.306 (0.099) |
| Qwen3.5 9B (Q4_K_M mixed) | 423.394 (4.438) / 14.863 (0.147) | 421.661 (6.212) / 14.866 (0.064) | 413.127 (5.463) / 14.888 (0.107) | 429.201 (2.805) / 14.651 (0.023) |

All 16 outputs were coherent and finite. Llama 1B BF16 K/V was byte-identical
to its F16 reference; the other lower-precision outputs had sane wording
differences. This is a functional gate, not a quality-equivalence result. No KV
format produced a robust short-context decode-speed advantage. Llama 1B prompt
samples were noisy, while the Llama 3B F16/BF16 and Qwen 9B F16 cells recorded
material host-global swap-out. Those rows remain valid for correctness and
placement but cannot establish a fine-grained precision ranking. The matrix is
internally build-stable within each model, not a single-binary certification
across all four models.

The expected context-2,048 state payload explains the useful tradeoff. These
figures exclude weights, allocator padding, compute buffers, and driver state;
they are not measured total memory.

| Model | F16/BF16 attention K+V | Q8_0 attention K+V | Q4_0 attention K+V | Fixed F32 recurrent state |
| --- | ---: | ---: | ---: | ---: |
| Llama 3.2 1B | 64 MiB | 34 MiB | 18 MiB | 0 |
| Llama 3.2 3B | 224 MiB | 119 MiB | 63 MiB | 0 |
| Qwen3.5 4B | 64 MiB | 34 MiB | 18 MiB | about 50.25 MiB |
| Qwen3.5 9B | 64 MiB | 34 MiB | 18 MiB | about 50.25 MiB |

Q8_0 and Q4_0 reduce attention-KV payload by 46.9% and 71.9% relative to
F16/BF16. The Qwen values must additionally include the fixed recurrent state
for 24 recurrent layers, so their combined reductions are 26.3% and 40.3%.
All graph probes retained ROCm Flash Attention. XDNA node count is zero because
the plugin was absent, not because a node-level scheduler census was collected;
HIP node counts were not recorded. TTFT, sampled CPU/GPU/NPU utilization, and
power or energy per token were also not measured and remain `N/A`.

The frozen aggregate report has SHA-256
`be2d9ad49d7c3a7d45dccea405c8f0e67f2da81138b986de13f387eb379b3de6`;
its 16-row, 72-column TSV has SHA-256
`3bb917ed316cb382fe11301e35c336946e497a8631bb4d13012b47586e8f6af1`.
The next execution gates remain separate datasets: A2 measures identical HIP
execution with shared host placement, B enables XDNA with the automatic policy
and is expected to retain zero XDNA submissions because every measured
descriptor currently declines preference, and C forces exact XDNA placement
only for registered shapes, including the Llama 1B and 3B BF16 gate/up shapes
and the Qwen 4B Q8_0 gate/up shape. A2/B require order-balanced throughput and
placement parity; C additionally requires nonzero XDNA submissions, zero
repeated immutable-weight payload copies, and a CPU-reference sanity pass.
Qwen 9B must reject C until matching tracked kernels exist.

A swap-guarded first A2/B pass completed six functional correctness/placement
cells before host pressure invalidated further timing work. Llama 1B completed
F16 and Q4_0 K/V in both A2 and B; Llama 3B completed F16 K/V in both. All six
generated coherent deterministic output, retained ROCm Flash Attention, and
exited successfully. The B scheduler census assigned zero compute nodes to
XDNA: Llama 1B used 246 ROCm plus 64 CPU nodes for F16 and 310 ROCm plus 64 CPU
nodes for Q4_0, while Llama 3B F16 used 426 ROCm plus 112 CPU nodes. Every
`MUL_MAT` stayed on ROCm, XDNA reported zero submissions and zero weight-copy
bytes, and `GGML_XDNA_PREFER_OFFLOAD` was explicitly unset. This confirms the
automatic no-preference policy on the completed real graphs; it is not an
end-to-end performance result.

The run stopped before Llama 3B Q4_0 after its F16 cells incurred 2.4-124.7 MiB
swap-in and 31.3-635.8 MiB swap-out per stage. All Qwen A2/B cells and all
sixteen planned A2/B benchmarks therefore remain unrun. The retained partial
report has SHA-256
`200214a5f9bb3e0136c889c90f73ac311d6f1e189e947e0dbd594532fd88842a`.
Single correctness-smoke rates are deliberately excluded from the performance
table because cache state and host paging differed between profiles.

An exact-head forced-C functional diagnostic subsequently exercised the pinned
Llama 3.2 3B BF16 model at
`df255f7cf15f720995799b23b579ca05b5394d3e`. The correctness action generated
the exact frozen profile-B reference text, SHA-256
`8f30d3a7df78841054291d938139e9e3d46b52955484fac1c3e1061e28cabf94`,
and recorded 7,118 successful XDNA calls, 56 root registrations, and zero
immutable-weight copy bytes. The frozen stdout, stderr, and normalized XDNA
statistics have SHA-256
`dfb9e1ae37d3d972628de1d5b01c8c8af15e60f14fe0b8080fd81c94ad16152d`,
`970a1d5ebb02a4ac1b1d4ba8780e1f7b6a5c1e201949845a5e8d4f34f61cca08`,
and `9a418e22adcd3a52ac1f48b88f6e972e93d27e37dc769484b1a9edef66b7df0f`.

A separate `pp16` plus `tg4` functional graph action produced the exact 56
XDNA decode identities `ffn_gate-0..27` and `ffn_up-0..27`. The `pp16`
topology had 195 ROCm `MUL_MAT` identities plus `ffn_gate-27` and `ffn_up-27`
on XDNA, with two XDNA calls and two registrations. The `tg4` topology had 141
ROCm `MUL_MAT` identities plus all 56 gate/up identities on XDNA, with 224
XDNA calls and 56 registrations. Their deduplicated union contains 197 logical
`MUL_MAT` identities, 197 ROCm-tagged entries, and the 56 additional
XDNA-tagged gate/up entries, for 253 backend-tagged entries total. All 28 Flash
Attention identities were on ROCm. The graph action therefore recorded 226
successful XDNA calls, 58 root registrations, and zero immutable-weight copy
bytes. Its stderr, census
summary, full census, and identical expected/actual XDNA identity lists have
SHA-256
`372871296b3988f5520db46aeca4049c8bca471eb913da72eb031da585f5e56d`,
`e05b53ac9f18ea8e5d438a1c879e77e1cbf1b5398648239e4198bdc5b0dd13dc`,
`81fc96193af264a0fdc229e2769600cafde77b19b0d547225ebfb76b698dd575`,
and `245635bcbac594536dd49c91bfb6eb5845791acafcb187168d1c27ef9586184b`.
The first post-run validator rejected the preserved exit-zero graph log because
it assumed prefixed placement lines and counted the backend-tagged union as
unique logical nodes. The corrected offline parser passed the frozen raw
evidence without rerunning the device process.

Both C actions moved host-global swap: correctness recorded 406,206 pages in
and 511,267 pages out, while graph placement recorded 2,747 pages in and
10,035 pages out. Their timings are invalid and support no performance claim.
A matched exact-head H correctness/graph control and order-balanced H/C
performance runs remain unrun. This is forced-C functional and placement GO,
not comparative performance evidence, and it does not change either BF16
descriptor's `prefer_for_offload=false` policy. The final durable report and
manifest have SHA-256
`2ed271d3e84e78be89291e8e34dd5298c99390ac76224bc12b9e86b6a198cf0a`
and
`d410c481b7855980902241be179c8116e6425ad45ae3aeed977dd2754747205b`.

## Native Q4_K research frontier

A cache-only AIE2P experiment implements the exact native GGML
`Q4_K[M=12288,K=4096] x Q8_K[K=4096] -> F32` dot product used by the Qwen3.5
9B gate/up projection. It consumes the 28,311,552-byte Q4_K matrix directly,
quantizes only the 16,384-byte F32 activation to the native 4,672-byte Q8_K
layout, and preserves FP16 `d`/`dmin`, FP32 Q8_K `d`, and the signed block-sum
minimum compensation. Zero weights, zero activation, scale/minimum edge cases,
multiple random seeds, sentinels, and the exact real GGUF
`blk.0.ffn_gate.weight` tensor all passed their CPU references. This establishes
**correctness GO** for the format algebra and physical AIE2P implementation;
it does not add Q4_K to the backend's type translation or registry.

The best organization processes each worker's 24 rows as three eight-row
groups with the Q8_K super-block pair outside the row loop. Relative to the
repaired row-at-a-time kernel, this reduces logical core-local activation reads
from 112,128 to 14,016 bytes per worker and improves full p50 from 2,407.381 to
1,763.780 us over 10,001 measured iterations. All 34 retained reference checks
were exact within the stated tolerance, with no non-finite value or sentinel
corruption. Effective native-weight bandwidth is 16.052 GB/s.

The matched device-resident HIP control used the same `0x12345678` native
weight and F32 activation payloads and measured 337.339 us full p50, or
83.926 GB/s. XDNA is therefore 5.229x slower and provides 19.13% of HIP's
effective weight throughput. The decision is **integration NO-GO**: do not add
this artifact to the tracked registry or automatic offload policy. If further
research resumes, the 8+8+8 implementation is the baseline and must remain
explicitly opt-in until it clears a matched end-to-end gate.

The corrected Q4_K report has SHA-256
`20e06be0bb4eb15d1f73f1dc68778e4c70465325fc0733158a6c056f391a81ca`.
The selected package's directed/random/real-GGUF correctness log has SHA-256
`163bf96b096fd8825ee147064621865d1670fb61e89d302b8f4b4abb0824730a`;
its XCLBIN has SHA-256
`2afd0da10043d61efa8ea40e2d0187ae41c949593869f29792bcf05a367f4b4c`.
The retained 10,001-iteration XDNA log has SHA-256
`19efddcff8f9a9f0628d74f60c9bd88d02ff0d0fefad9e33a3d86bbbb1d21703`;
the matched HIP stdout has SHA-256
`7a5da4134480ee855cca6aadf1d3ee520b9aa4b55bbe7323c2dabd523375b334`.

## Decode shape census and next compute step

GGUF tensor dimensions are listed below as decode `M x K` (`N=1`), not the
reader's `[ne0, ne1]` storage display:

| Model | Projection | Decode shape | Stored type |
| --- | --- | ---: | --- |
| Llama 3.2 1B (hidden 2048, FFN 8192) | gate/up | 8192 x 2048 | BF16 |
| Llama 3.2 1B | down | 2048 x 8192 | BF16 |
| Llama 3.2 3B (hidden 3072, FFN 8192) | Q/output | 3072 x 3072 | BF16 |
| Llama 3.2 3B | K/V | 1024 x 3072 | BF16 |
| Llama 3.2 3B | gate/up | 8192 x 3072 | BF16 |
| Llama 3.2 3B | down | 3072 x 8192 | BF16 |
| stories15M (hidden 288, FFN 768) | Q/K/V/output | 288 x 288 | Q4_0 |
| stories15M | gate/up | 768 x 288 | Q4_0 |
| stories15M | down | 288 x 768 | Q4_0 |
| Gemma 4 E4B (hidden 2560, FFN 10240) | Q | 2048 x 2560 | Q4_0 |
| Gemma 4 E4B | K/V | 512 x 2560 | Q4_0 |
| Gemma 4 E4B | attention output | 2560 x 2048 | Q4_0 |
| Gemma 4 E4B | gate/up | 10240 x 2560 | Q4_0 |
| Gemma 4 E4B | down | 2560 x 10240 | Q4_1 |
| Qwen 3.5 4B (hidden 2560, FFN 9216) | gate/up | 9216 x 2560 | Q8_0 |
| Qwen 3.5 4B | down | 2560 x 9216 | Q8_0 |
| Qwen 3.5 4B | recurrent QKV | 8192 x 2560 | Q8_0 |
| Qwen 3.5 4B | full-attention K/V | 1024 x 2560 | Q8_0 |
| Qwen 3.5 9B (hidden 4096, FFN 12288) | gate/up | 12288 x 4096 | Q4_K |
| Qwen 3.5 9B | down | 4096 x 12288 | Q6_K |
| Qwen 3.5 9B | recurrent QKV | 8192 x 4096 | Q5_K |
| Qwen 3.5 9B | full-attention Q | 8192 x 4096 | Q4_K |
| Gemma 4 12B (hidden 3840, FFN 15360) | Q | 4096 x 3840 | Q4_0 |
| Gemma 4 12B | K/V | 2048 x 3840 | Q4_0 |
| Gemma 4 12B | attention output | 3840 x 4096 | Q4_0 |
| Gemma 4 12B | gate/up | 15360 x 3840 | Q4_0 |
| Gemma 4 12B | down | 3840 x 15360 | Q4_1 |

The local E4B GGUF contains 42 layers, but its shared-KV policy executes K/V
projections only in the first 24 KV-owning layers. Twenty of those are
sliding-window layers, so the 512x2560 specialization covers 20 K plus 20 V
calls per decode token. That is 40 of 253 Q4_0 linear calls (15.8%), but only
29,491,200 of 2,114,519,040 compressed Q4_0 weight bytes (1.39%). The dominant
E4B byte traffic is 84 gate/up calls at 10240x2560
(1,238,630,400 bytes/token), followed by 37 Q4_0 down calls at 2560x10240
(545,587,200 bytes/token; five other down matrices are Q4_1). This makes K/V a
low-risk real-graph entry point, not the highest-impact final specialization.

K and V share the same activation in source, but they are not adjacent in the
actual scheduled Gemma 4 graph. A current graph export and scheduler probe show
`K MUL_MAT -> reshape/RMS_NORM/MUL/ROPE -> V MUL_MAT`, which necessarily forms
two XDNA splits because the intervening operations are unsupported. A generic
shared-RHS fused artifact compiled successfully for two weights, one activation,
and two outputs on 32 tiles. A 10,001-submission physical run matched both CPU
references exactly and measured 145.430 us median start+`wait2()` and 146.990
us median complete time, or 9.99 GB/s over both 737,280-byte weight matrices.
That saves about 72 us (33%) versus two separate complete XDNA calls by
removing one command envelope and reusing the activation. It is still about
8.6x slower than two isolated HIP projections. Backend `graph_compute` cannot
currently see both nodes in one split, so the smallest semantic-preserving
integration experiment is to expand the raw Q/K/V projections together before
their postprocessing, matching the existing `build_attn()` intent; a wider
backend-driven graph-reordering API is not justified yet.

The measured format frontier is now broad enough to choose the next work by
hardware evidence rather than model size alone. Statistics retain the protocol
used by each physical experiment; the comparison column reports its matched
HIP conclusion rather than mixing p50 and order-balanced means.

| Native weight / decode shape | Best current XDNA complete result | Effective weight BW | Matched HIP result |
| --- | ---: | ---: | --- |
| Q4_0 512 x 2560 | 106.757 us mean | 6.91 GB/s | 4.4-5.0x faster |
| Q4_0 10240 x 2560 | 765.908 us mean | 19.25 GB/s | 4.7-4.9x faster |
| Q8_0 9216 x 2560 | 794.53 us process-median | 31.55 GB/s | 269.80 us, 2.94x faster |
| BF16 8192 x 2048, repeated tasks | 668.540 us balanced full p50 | 50.73 GB/s sustained | 388.104/391.938 us; XDNA remains 1.70-1.72x slower |
| BF16 8192 x 3072, repeated tasks, opt-in | 979.295 us balanced full p50 | 51.52 GB/s sustained | 526.994-583.869 us; XDNA remains 1.68-1.83x slower |
| Q4_K 12288 x 4096, cache-only | 1,763.780 us full p50 | 16.052 GB/s | 337.339 us, 5.229x faster |

The Llama 3.2 3B gate/up probe raised sustained task-fused XDNA effective
bandwidth by only about 1.6% from K=2048 while HIP remained faster. Larger
native BF16 matrices therefore narrow the gap compared with the quantized
families but do not establish a crossover. The same-byte down projection has
different row geometry, and another fixed BF16 shape is not justified without
a topology or overlap change that addresses the measured bandwidth ceiling.
The K=3072 descriptor remains opt-in because the measured HIP controls are
faster; its in-tree seven-artifact correctness/coexistence gate has passed.

Q4_K correctness does not justify tracking a 5.229x-losing artifact, and the
mixed Qwen 9B policy would still require Q5_K and Q6_K for broad coverage.
Q4_1 down projections likewise require a separate format contract. Any next
Q8_0 or Q4_0 specialization must be selected from graph byte/call coverage and
must beat the corresponding matched controls rather than extrapolating from a
single format. Each admitted shape remains a registry entry with its own
worker/DMA geometry and correctness tolerance; it does not add model-name
switches to GGML or the XDNA runtime.

The remaining high-value questions are whether compute-tile codegen or a safe
explicit ping-pong topology can move BF16 beyond the measured bandwidth
plateau, whether the model loader can provision large `ROCm_Host`
allocations reliably, whether the read-only mmap gate passes after a driver
upgrade, and whether concurrent compute/free/teardown can be given a safe
contract.
Depth-two forwarding remains disabled. The repeated-task controller safely covers all 32 waves with depth-one FIFOs; an earlier eight-wave multi-task batch hung under sustained load and is not used.
AIE2 requires its own implementation and is not inferred from the tested AIE2P kernel.

## Go/no-go checkpoint

| Question | Current evidence |
| --- | --- |
| Clean ACCEL registration and discovery? | **GO**: dynamic `XDNA0`, real XRT/plugin/device-node path. |
| Real GGML `MUL_MAT` on XDNA? | **GO for the seven tracked fixed Q4_0, Q8_0, and BF16 shapes**. All seven passed together in one ordinary physical backend process; the exact Q8_0 production shape passes physically, a real Q4_0 GGUF `Qcur-0` node executed, and BF16 K=3072 passed the exact pinned `blk.0.ffn_gate.weight`. Native Q4_K x Q8_K is physically correct only in a cache-local research harness and is not advertised by the backend. |
| Fundamentally UMA/system-backed weights? | **GO for writable CPU and `ROCm_Host` allocations**: a physical HIP/XDNA probe consumed the same page-aligned system-memory weight pointer, with one XRT registration and no secondary weight copy. |
| Reuse without per-token weight copy? | **GO for serialized immutable-buffer lifetimes**: persistent parent/view/run reuse has zero weight-copy bytes, and observer-driven eviction passes exact owner/base/data ABA reuse with two registrations and no hits. Mutable/test weights use one explicit native-byte staging copy per call. Concurrent compute/free/backend teardown is not claimed. |
| Ordinary read-only GGUF mmap? | **NO on the installed driver, fail-closed automatically**: the complete file-backed `PROT_READ` parent/subview/sync probe fails with errno 12, leaving `mmap_support` and `CPU_Mapped` acceptance disabled. Upstream driver commit `ed8fb2dd172bde623d7112a1bd674fc0e3c4cae4` contains the required read-only pin/IOMMU path; only a successful post-upgrade runtime probe enables the positive path, which is not physically validated here. |
| Fixed batch-one GEMV? | **GO for seven integrated variants**. Native 288x288, 512x2560, and 10240x2560 Q4_0, native 9216x2560 Q8_0, plus 288x288, 8192x2048, and 8192x3072 BF16 pass together in one physical backend instance. The two production BF16 shapes now use the physically validated 17-task controller; K=2048 passed both exact Llama 1B gate/up tensors and K=3072 passed the exact Llama 3B gate tensor. The cache-only Q4_K row-batched kernel is correctness GO but integration NO-GO at 16.052 GB/s and 5.229x slower than its matched HIP control. Broader dtype/shape coverage remains open. |
| Four-model real-GGUF inference matrix? | **GO for the 16-cell ROCm-resident A1 correctness, placement, pp512, and tg128 reference; PARTIAL functional GO for six A2/B cells; forced-C functional and placement GO for exact-head Llama 3B BF16; OPEN for comparative A2/B/C performance.** All requested exact files and four same-type KV modes completed A1 with Flash Attention. Completed A2/B real graphs kept every `MUL_MAT` on ROCm and recorded zero XDNA submissions/copies under automatic policy. Forced C produced the exact frozen reference text. Across its combined `pp16`/`tg4` evidence, every one of the 197 logical `MUL_MAT` identities had a ROCm-tagged entry and the 56 decode gate/up identities additionally had XDNA-tagged entries; all 28 Flash Attention identities were on ROCm and no immutable-weight bytes were copied. Host-global swap invalidated C timing and stopped the wider matrix. A matched exact-head H control, every A2/B/C benchmark, TTFT, utilization, energy, and full-matrix HIP node counts remain unmeasured. |
| ROCm comparison and hybrid value? | **Capability scheduling, shared weights, and a real small-GGUF hybrid graph GO; performance NO-GO for the measured isolated shapes.** The physical pre-policy path kept `pp32` off XDNA and executed 48 XDNA projections for `tg2` from one `ROCm_Host` model allocation, with persistent page-window registrations and zero immutable-weight copy. Current normal `[ROCm,XDNA,CPU]` order leaves mutually supported decode on ROCm; `GGML_XDNA_PREFER_OFFLOAD=1` is the explicit control for reproducing intentional XDNA placement. Fair synchronized HIP is about 4.4-5.0x faster at Q4_0 512x2560, 4.7-4.9x at Q4_0 10240x2560, 2.94x at Q8_0 9216x2560, 1.70-1.72x at task-fused BF16 8192x2048, 1.68-1.83x at task-fused BF16 8192x3072, and 5.229x at cache-only Q4_K 12288x4096; HIP is about 12x faster at Q4_0 288x288. Larger native-BF16 shapes narrow the gap materially but do not reverse it. All production descriptors therefore keep `prefer_for_offload=false`. Fused K/V saves one XDNA command floor but remains slower than HIP, and the tested gate/up pair is slower than two optimized XDNA calls. No end-to-end hybrid performance win is claimed. |
| NPU utilization and energy? | **OPEN**: installed XRT/sysfs exposes neither a per-kernel utilization/energy counter nor estimated NPU power (`Estimated Power: N/A`); external SoC telemetry is required. |

The experiment is a no-go if direct-layout Q4_0 cannot beat HIP after activation
transfer and synchronization, if immutable weights require per-token payload
copies, or if safe scheduler selection requires model-specific changes. Raw
TOPS alone is not a success criterion; report total GEMV latency, effective
compressed-weight GB/s, sync/CPU overhead, and projected token throughput.

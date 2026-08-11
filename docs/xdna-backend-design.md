# Native AMD XDNA backend: bring-up design and evidence

Status: experimental vertical slice, 2026-08-11. This document records what
was inspected and physically exercised; it is not a claim of production-ready
XDNA decode.

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
requires an exact registered kernel, so batch-one decode can select XDNA while
unsupported prefill remains on HIP. This policy is capability-driven and does
not add model-specific placement logic.

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
but the only compute artifact currently accepted is AIE2P. XDNA1 is not claimed
as tested.

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
GGML host allocation (model-lifetime owner)
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
  workaround. Current upstream driver added read-only pinning in
  [`97cfa4c673f09d321c613e55761d44a4cc378d11`](https://github.com/amd/xdna-driver/commit/97cfa4c673f09d321c613e55761d44a4cc378d11), but the installed module must be
  upgraded and re-tested before `mmap_support` can be enabled.

Backend statistics label their counter `explicit_payload_BOs`: it covers
payload/root BO allocations made directly by ggml-xdna, not lightweight
subviews, command BOs, or other internal BOs allocated by XRT. Driver-level
creation evidence comes from the ioctl trace above; those counts, too, remain
constant rather than growing with repeated submissions.

The present slice therefore registers writable GGML host allocations with no
XDNA weight shadow. It deliberately reports both `buffer_from_host_ptr=false`
and `mmap_support=false`. GGML distinguishes its writable allocation as `CPU`
from an externally wrapped/mmap allocation as the private `CPU_Mapped` buffer
type. XDNA admits exact `CPU` storage plus a non-null device owner's exact
declared host buffer type; this accepts the physically validated `ROCm_Host`
allocation but continues to reject device-less `CPU_Mapped`. Normal automatic
mmap loading therefore remains safe and its nodes stay on fallback backends.
`--load-mode none` is still required on this driver for an XDNA-eligible model
allocation.

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

Independent HIP and XRT registrations of exactly the same `ROCm_Host` pages are
now physically proven; dma-buf export is not required for this read/read weight
case. The lazy XRT registration cache remains backend-lifetime scoped and is
limited to immutable model/test weight buffers that must outlive their cached
views. Scheduler output is not registered: each kernel state owns a persistent
output BO sized for that variant and copies it back after synchronization, so
context reserve/reallocation cannot leave a transient scheduler allocation
pinned. Before supporting arbitrary immutable-buffer lifetimes, registration
ownership and eviction must move into an XDNA buffer wrapper so an XRT BO can
never outlive its GGML mapping.

Page-window registration avoids pinning an unrelated full model allocation.
Adjacent GGML weights can share a boundary page, so their rounded XRT parents
can overlap. The current driver/XRT pair accepted that arrangement in a real
six-layer GGUF run: 24 distinct weight windows registered and all 24 were reused
on the next token. Registration accounting deliberately reports rounded bytes,
so a shared boundary page can be counted in more than one root.

That lifetime restriction is demonstrated, not hypothetical. A stress probe
kept one XDNA backend alive, freed an immutable-weight GGML buffer, then
allocated a replacement at the same owner/data addresses. The raw-pointer cache
reported two hits and only the original sync; subsequent results had NMSE
0.0481 and 0.189 instead of the first call's 9.55e-8. The current slice is thus
safe only under the llama.cpp model/context ordering used here, where model
weights remain alive until the XDNA context/backend is destroyed. It is not a
general-purpose upstream-ready buffer cache. A custom XDNA host buffer with a
free-time registration callback is the required fix before broadening use.

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
512x1x2560, and 10240x1x2560 kernels plus the BF16 reference are entries, not
switch cases scattered across the backend/runtime. The artifact packer
likewise receives metadata from the shape-specific build instead of
maintaining a second dtype/shape table. This is deliberately a specialization
registry, not a universally dynamic kernel.

The runtime now discovers every configured registered bundle, validates each
one independently, and constructs one persistent XCLBIN/context/kernel/run
state per valid artifact. Both device capability reporting and execution use
the same `xdna_problem` selector. A physical process loaded the 288x288,
512x2560, and 10240x2560 full-array artifacts simultaneously and passed all
three CPU-reference tests, proving that the installed XRT stack permits
multiple persistent full-array contexts. UMA weight registrations remain
shared across those states. Malformed optional bundles are rejected without
hiding valid variants; an XRT load failure rejects the complete backend
instance so capability reporting cannot advertise an unavailable state.

The configured variant accepts only its exact contract:

- `GGML_OP_MUL_MAT`;
- contiguous native `Q4_0[288,288] x F32[288,1] -> F32[288,1]`,
  `Q4_0[2560,512] x F32[2560,1] -> F32[512,1]`,
  `Q4_0[2560,10240] x F32[2560,1] -> F32[10240,1]`, or the retained
  `BF16[288,288]` plumbing reference;
- one AIE2P/XDNA2 device and an explicitly configured versioned artifact bundle.

The bundle couples a self-identifying kernel kind, ABI version, dimensions,
weight/activation/output byte sizes, xclbin, and instruction stream. The runtime
parses and validates the complete bundle before `supports_op` becomes true. A
deliberate test labeling the BF16 bundle as Q4_0 was rejected with an ABI
metadata mismatch and zero submissions, preventing accidental cross-pairing of
the generated bundles. The configured bundle is still a trusted executable
input; its header does not authenticate the semantics of a manually relabeled
arbitrary xclbin.

Every prefill/batched shape and every other dtype is rejected. The Q4_0 kernel
reads the native 18-byte `block_q4_0` representation and decodes scale/nibbles
on-tile; it does not repack or dequantize the matrix. The host converts only
the small activation vector to BF16. For immutable model weights it
page-registers each selected tensor window once, uses root-derived sub-BOs,
explicitly cache-flushes each new weight view once, reuses one command/run per
selected artifact, and uses persistent activation/output BOs. That path
performs no weight payload copy. Mutable test weights instead use the explicitly
measured per-call staging path described above. The backend records XRT device/kernel
initialization, registration, backend-created BOs, weight and other host-copy
time, TO/FROM sync time, separate XRT `start()` and `wait()` time, and complete
call time. A size-aware v2 statistics API exposes those stage counters through
dynamic backend lookup while preserving the original stats ABI. Artifact
parsing happens during backend registration before the runtime initialization
timer. Kernel source and reproduction instructions are in
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

The integrated scheduler now lets an ACCEL backend's exact `offload_op` request
override ordinary owner priority only for immutable host-backed weights. With
normal production order `[ROCm0, XDNA0, CPU]`, three physical processes selected
XDNA for `N=1` and ROCm for `N=32` from the same `ROCm_Host` pointer. XDNA
admits that buffer through the generic owning-device/declared-host-buft
relationship, without a `ROCm_Host` name check, while continuing to reject
`CPU_Mapped`. The XDNA registration survived the intervening HIP execution and
again reported zero weight-copy bytes. Large-model pinned-allocation failure
and immutable-buffer free-time eviction remain open productization gates.

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

## Decode shape census and next compute step

GGUF tensor dimensions are listed below as decode `M x K` (`N=1`), not the
reader's `[ne0, ne1]` storage display:

| Model | Projection | Decode shape | Stored type |
| --- | --- | ---: | --- |
| stories15M (hidden 288, FFN 768) | Q/K/V/output | 288 x 288 | Q4_0 |
| stories15M | gate/up | 768 x 288 | Q4_0 |
| stories15M | down | 288 x 768 | Q4_0 |
| Gemma 4 E4B (hidden 2560, FFN 10240) | Q | 2048 x 2560 | Q4_0 |
| Gemma 4 E4B | K/V | 512 x 2560 | Q4_0 |
| Gemma 4 E4B | attention output | 2560 x 2048 | Q4_0 |
| Gemma 4 E4B | gate/up | 10240 x 2560 | Q4_0 |
| Gemma 4 E4B | down | 2560 x 10240 | Q4_1 |
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

The next specialized family should target the remaining E4B Q4_0 Q and
attention-output projections. Q4_1 down projections require a separate format
contract. Each admitted shape remains a registry entry with its own worker/DMA
geometry and correctness tolerance; it does not add shape switches to GGML or
the XDNA runtime.

The measured regimes now separate cleanly: 288x288 and 512x2560 are dominated
by the command envelope, while 10240x2560 exposes sustained kernel/dataflow
throughput. The next high-value questions are whether compute-tile codegen or a
safe explicit ping-pong topology can move beyond 19.25 GB/s, whether the model
loader can provision large `ROCm_Host` allocations reliably, and how to attach
XRT registration eviction to the backing buffer's actual lifetime. Depth-two
forwarding remains disabled; four-wave depth-one task groups are the largest
stress-tested controller batch, while an eight-wave batch hung under sustained
load. AIE2 requires its own implementation and is not inferred from the tested
AIE2P kernel.

## Go/no-go checkpoint

| Question | Current evidence |
| --- | --- |
| Clean ACCEL registration and discovery? | **GO**: dynamic `XDNA0`, real XRT/plugin/device-node path. |
| Real GGML `MUL_MAT` on XDNA? | **GO for fixed Q4_0 and BF16 shapes**; CPU comparisons pass and a real GGUF `Qcur-0` node executed. |
| Fundamentally UMA/system-backed weights? | **GO for writable CPU and `ROCm_Host` allocations**: a physical HIP/XDNA probe consumed the same page-aligned system-memory weight pointer, with one XRT registration and no secondary weight copy. |
| Reuse without per-token weight copy? | **GO for model-lifetime immutable buffers**: persistent parent/view/run, constant registrations, zero weight-copy bytes. Mutable/test weights use one explicit native-byte staging copy per call. **NO for arbitrary immutable-buffer lifetimes** until registration ownership moves into an XDNA buffer type. |
| Ordinary read-only GGUF mmap? | **NO on the installed driver**; XDNA rejects `CPU_Mapped`, so default loading falls back safely. Upstream's driver fix exists but is not yet physically validated here. |
| Quantized batch-one GEMV? | **GO for native 288x288, 512x2560, and 10240x2560 Q4_0**. All three registered variants pass together in one physical backend instance; the production variants reach 6.91 and 19.25 GB/s without repacking or a secondary immutable-weight copy. Broader dtype/shape coverage remains open. |
| ROCm comparison and hybrid value? | **Capability scheduling, shared weights, and a real small-GGUF hybrid graph GO; performance NO-GO for the measured isolated shapes.** Normal `[ROCm,XDNA,CPU]` order keeps `pp32` off XDNA and executes 48 XDNA projections for `tg2` from one `ROCm_Host` model allocation, with persistent page-window registrations and zero immutable-weight copy. Fair synchronized HIP is about 4.4-5.0x faster at 512x2560 and 4.7-4.9x at 10240x2560; HIP is about 12x faster at 288x288. Fused K/V saves one XDNA command floor but remains slower than HIP, and the tested gate/up pair is slower than two optimized XDNA calls. No end-to-end hybrid performance win is claimed. |
| NPU utilization and energy? | **OPEN**: installed XRT/sysfs exposes neither a per-kernel utilization/energy counter nor estimated NPU power (`Estimated Power: N/A`); external SoC telemetry is required. |

The experiment is a no-go if direct-layout Q4_0 cannot beat HIP after activation
transfer and synchronization, if immutable weights require per-token payload
copies, or if safe scheduler selection requires model-specific changes. Raw
TOPS alone is not a success criterion; report total GEMV latency, effective
compressed-weight GB/s, sync/CPU overhead, and projected token throughput.

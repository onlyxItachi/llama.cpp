# Native AMD XDNA backend: bring-up design and evidence

Status: experimental vertical slice, 2026-08-11. This document records what
was inspected and physically exercised; it is not a claim of production-ready
XDNA decode.

## Revisions and repository state

XDNA work began from clean upstream llama.cpp commit
[`6e62ba538478202094edc6c100c782719e310aa3`](https://github.com/ggml-org/llama.cpp/commit/6e62ba538478202094edc6c100c782719e310aa3)
on branch `feat/ggml-xdna`. The fork is `onlyxItachi/llama.cpp` (`origin`) and
`ggml-org/llama.cpp` is `upstream`. No user work was reset.

A final non-destructive fetch found that upstream had advanced to
[`2468576f241235452013308597e6de1b78866996`](https://github.com/ggml-org/llama.cpp/commit/2468576f241235452013308597e6de1b78866996)
while the hardware work was running. Its three intervening commits do not
change any GGML/backend/model/scheduler file inspected for this design, so the
validated branch remains on the exact recorded start SHA rather than rewriting
tested history at the end of the experiment.

The current default-branch heads inspected with Git and GitHub CLI were:

| Project | Exact revision |
| --- | --- |
| Xilinx/mlir-aie | [`a8fea363dfd63015cffd380e0679950419a0a7cd`](https://github.com/Xilinx/mlir-aie/commit/a8fea363dfd63015cffd380e0679950419a0a7cd) |
| mlir-aie Peano/llvm-aie pin | [`c9c5ecb725fc8c765e4b687356e6ec1e54da7a0e`](https://github.com/Xilinx/llvm-aie/commit/c9c5ecb725fc8c765e4b687356e6ec1e54da7a0e), wheel `21.0.0.2026080301+c9c5ecb7` |
| amd/xdna-driver | [`d73478232a7057bfea73a24f74b09df4e5580ad8`](https://github.com/amd/xdna-driver/commit/d73478232a7057bfea73a24f74b09df4e5580ad8) |
| Xilinx/XRT | [`27bed122dd01ca3e21dbf2b07000078ec842c98c`](https://github.com/Xilinx/XRT/commit/27bed122dd01ca3e21dbf2b07000078ec842c98c); driver pin [`8b60ae7a90bfbc873e181497fd34ca520b4ef504`](https://github.com/Xilinx/XRT/commit/8b60ae7a90bfbc873e181497fd34ca520b4ef504) |
| amd/Triton-XDNA | [`434dc3ec3de0fa0de2d2e26236e363967a00e265`](https://github.com/amd/Triton-XDNA/commit/434dc3ec3de0fa0de2d2e26236e363967a00e265) |
| ROCm/FastFlowLM v1.0.0 | [`39fb58cc7f9e23d97b764a1ef21ff0ade088047b`](https://github.com/ROCm/FastFlowLM/commit/39fb58cc7f9e23d97b764a1ef21ff0ade088047b) |
| amd/RyzenAI-SW | [`06fdce7eeacf88c07fa84451c22e306ebca6de5e`](https://github.com/amd/RyzenAI-SW/commit/06fdce7eeacf88c07fa84451c22e306ebca6de5e) |
| RyzenAI-SW historical llama branch | [`d78a8cf57cfbda319ec09dcac941ddfc34f5793f`](https://github.com/amd/RyzenAI-SW/commit/d78a8cf57cfbda319ec09dcac941ddfc34f5793f) |

The live machine has XRT 2.20.0 (`021204355eeaa034ff69aae407ace2265adf047a`),
firmware 1.1.2.64, and an accessible `RyzenAI-npu4` at `0000:67:00.1` through
`/dev/accel/accel0`. `xrt-smi validate` passed its supplied INT8 GEMM test at
51.0 TOPS and its submission-latency test at about 46 us. Those are stack-health
results, not GGML performance.

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
`supports_op` is necessary but not sufficient to steal a weight already owned
by HIP. A controlled capability-only experiment can keep weights in CPU host
buffers so HIP accepts large prefill and XDNA accepts batch one. A production
shared-buffer HIP/XDNA policy will need a small generic scheduler preference or
explicit assignment experiment; it must not be hard-coded in model code.

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

The intended representation is one system-RAM allocation, registered once with
XRT, with tensor views expressed as offsets:

```text
GGML host/mmap owner (page-aligned root, page-rounded extent)
  -> cached XRT host_only userptr BO
     -> root-derived immutable weight sub-BOs
        -> shim DMA / memory tile / AIE compute tile
```

Transient activations and results use small persistent XRT BOs. The host copies
576 bytes of BF16 activation into the input BO and 1,152 bytes of F32 result out
of the output BO per invocation; no model-weight payload is copied.

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
- Unaligned root pointers fail; arbitrary sub-BO offsets work. The backend must
  round the known parent allocation, never an isolated tensor pointer.
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
type. XDNA conservatively supports only the exact `CPU` buffer type, and an
allocated `MUL_MAT` source must be explicitly marked
`GGML_BACKEND_BUFFER_USAGE_WEIGHTS`. Normal automatic mmap loading therefore
remains safe and its nodes stay on fallback backends. `--load-mode none` is the
explicit opt-in that creates the writable immutable-weight allocation used by
this experiment. Mutable CPU buffers are rejected: a live stress probe found
that re-registering the same virtual address after changing Q4 bytes left stale
weight data visible on the installed KMQ/XRT stack. This avoids claiming an
unvalidated cache-coherency path and avoids globally disabling mmap merely
because an experimental ACCEL is visible. A future XDNA-owned buffer type must
record registration, lifetime, mutability, and mapping capabilities before
`CPU_Mapped` or other host buffers are accepted.

Current XRT userptr BOs also cannot be exported as dma-bufs, so exactly shared
physical weight pages between independent HIP and XRT registrations remain to
be proven. The lazy registration cache is intentionally backend-lifetime scoped
and is limited to immutable model/test weight buffers that must outlive their
cached views. Scheduler output is not registered: a persistent 1,152-byte XRT
output BO is copied back after synchronization, so context reserve/reallocation
cannot leave a transient scheduler allocation pinned. Before supporting more
general lifetimes, registration ownership and eviction must move into an XDNA
buffer wrapper so an XRT BO can never outlive its GGML mapping.

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
deliberately unsupported and remain on the scheduler's fallback backend; use
`--load-mode none --no-host` to opt into XDNA execution.

Both dynamic and static GGML builds were exercised. The installed static CMake
package reconstructs the `XRT::xrt_coreutil` dependency; a fresh external
consumer configured against the install, linked XRT, initialized `XDNA0`, and
loaded the Q4 bundle successfully.

The backend selects one explicitly configured kernel bundle and accepts only
its exact contract:

- `GGML_OP_MUL_MAT`;
- contiguous native `Q4_0[288,288] x F32[288,1] -> F32[288,1]`, or the retained
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
on-tile; it does not repack or dequantize the matrix. Because one 162-byte row
does not meet the AIE DMA four-byte length constraint, the IRON transfer pairs
two adjacent native rows into exact 324-byte transactions. The host converts
only the 288-element activation to BF16. It page-registers the immutable GGML
weight buffer once, uses root-derived sub-BOs for weights, explicitly
cache-flushes each new weight view once, reuses one command/run, and uses
persistent activation/output BOs. It performs no weight copy and records XRT
device/kernel initialization, registration, backend-created BOs, host-copy
time, TO/FROM sync time, combined kernel start/wait time, and complete call
time. Artifact parsing happens during backend registration before the runtime
initialization timer. Kernel source and reproduction instructions are in
`ggml/src/ggml-xdna/kernels/aie2p/`.

The generated Q4_0 bundle used for the final reviewed-code test is intentionally
ignored by Git and has SHA-256
`4aae43e71b1ef6eaca4189e35e15093baacdc6bddaae4e82a053ccc9dfa2a5f3`.
It contains an xclbin with SHA-256
`86be4509f87dedc2a0c756eae915bb70970fe0be5ab01fbc2fadaf020a403774`
and a 420-byte instruction stream with SHA-256
`50fa4300155c11d5cc1fa05619cdbb4dc5af3f3b3251d2a395003af363018390`.
The xclbin and therefore bundle are not bit-reproducible because the current
toolchain writes a fresh UUID, timestamp, and unique ID; the kernel object,
instruction stream, ABI header, and generated MLIR were deterministic across
clean serial builds.

`test-backend-ops` compared the BF16 XDNA result with CPU successfully. A steady
run of 8,190 measured iterations reported 695.70 us/run, 238.45 MFLOP/s, one
root registration, one weight view, 8,190 weight-view hits, and zero weight-copy
bytes. The 165,888-byte BF16 matrix achieved about 0.238 GB/s.

The direct Q4_0 artifact first completed 101 persistent hardware launches with
zero mismatches and zero maximum error against a BF16-activation CPU reference.
GGML's own CPU-comparison test then passed with random `Q4_0` tensor data. The
final 8,192-run perf case measured 13,130.40 us/run, 12.63 MFLOP/s, and about
0.003553 GB/s of native compressed weights. Backend counters showed 8,193
submissions, one root registration, one weight view, 8,192 weight-view hits,
exactly one 46,656-byte weight sync, and zero weight-copy bytes. The scalar
one-core result is proof of native compressed execution rather than a
performance candidate. The final fresh-process correctness run measured 115.516
ms of XRT device/kernel initialization, a 13.071 ms first kernel, and a 13.183
ms first complete call. In the long run, combined kernel start/waits totaled
107,443.733 ms and complete calls 107,550.442 ms. Measured TO-device cache
maintenance was 5.342 ms total (0.652 us/call), FROM-device maintenance was
24.214 ms (2.955 us/call), and the two small host copies totaled 10.483 ms
(1.280 us/call). The remaining measured call overhead was about 8.14 us/call.
Each repeat maintained 576 activation bytes, 1,152 output bytes, converted 576
bytes of F32 activation to BF16, and copied 1,152 output bytes to GGML. These
byte counters describe cache maintenance and small activation/result movement
on KMQ, not payload DMA to a second weight buffer.

The same GGML test in a simultaneous HIP+XDNA build passed on the Radeon 890M
(`gfx1150`). The final focused HIP run measured 5.86 us/run and 28.33 GFLOP/s,
or about 7.96 GB/s of compressed weights. HIP was about 2,241x faster than the
current XDNA validation kernel. This is a firm no-go for the current scalar
implementation, not for the optimized decode hypothesis. A pinned-toolchain
defect was also isolated: depth-two forwarding corrupted one row on alternating
launches; depth-one FIFOs passed both 20- and 101-launch tests and are
intentionally used.

## Hybrid scheduling and real-GGUF evidence

A controlled scheduler probe used backend order `[ROCm0, XDNA0, CPU]`, one CPU
host-backed `Q4_0[288,288]` weight, and `op_offload=true`. For `n=1`, HIP did
not accept the CPU buffer or offload the small operation, XDNA accepted it, and
the scheduler produced one zero-input XDNA split which executed correctly. For
`n=32`, XDNA rejected the shape, HIP's `offload_op` accepted it, and the
scheduler produced one ROCm split with the weight and activation as its two
inputs. Both computations matched their reference exactly. This proves the
intended capability policy without a model-layer special case. It does not
solve the separate priority problem for weights already owned by a HIP buffer.

A second guard probe wrapped the same writable test storage with GGML's
`CPU_Mapped` buffer type. XDNA reported that buffer unsupported and the batch-one
node stayed on CPU; changing only the buffer to an owned `CPU` allocation selected
XDNA and executed correctly. This proves the installed-driver mmap limitation is
handled before scheduling rather than surfacing as a failed XRT submission.

The local `stories15M-q4_0.gguf` is a real 24.41-million-parameter, six-layer
Llama model with embedding width 288. A graph exported by the current
`test-export-graph-ops` contains both of these actual nodes:

```text
MUL_MAT Qcur-0: Q4_0[288,288] x F32[288, 1] -> F32[288, 1]   SUPPORTED
MUL_MAT Qcur-0: Q4_0[288,288] x F32[288,32] -> F32[288,32]   NOT SUPPORTED
```

With `--load-mode none --no-repack --no-host -ngl 0`, llama.cpp loaded one
17.50 MiB writable CPU model buffer. A combined HIP+XDNA scheduler trace for a
52-token prompt assigned the prompt's quantized linear operations to `ROCm0`
(unsupported attention work remained on CPU), then assigned the matching
batch-one attention projections to 24 zero-input XDNA splits per decode graph.
The backend logged the actual `Qcur-0` node on XDNA. Each ROCm projection split
listed its CPU-owned weight as an input; the six layers therefore staged at
least 3,373,056 bytes of listed linear/norm weights during this prompt. This is
temporary scheduler traffic rather than a persistent second full-model copy,
but it proves that current HIP does not directly consume the same plain CPU
buffer that XRT registers.

A final three-token CLI run on the reviewed output-buffer design issued 48
XDNA kernels across two repeated decode graphs: 24 unique weight views followed
by 24 cache hits. The one-time weight sync remained fixed at 24 calls /
1,119,744 bytes, one page-root registration covered the 17.504 MiB model
allocation, and weight-copy bytes remained zero. Combined kernel start/waits
totaled 609.096 ms and complete XDNA calls 610.432 ms; TO/FROM cache maintenance
and activation/result host copies were separately 0.031/0.043/0.022 ms. That
projects to about 3.28 steady tokens/s from the 24 scalar NPU projections alone,
before optimizing the kernel. The CLI reported 2,492.1 prompt tokens/s and 4.8
generation tokens/s for this very short aggregate; the first sampled token is
produced by prompt evaluation, so the projected repeated-decode figure is the
relevant one.

An all-HIP end-to-end comparison could not run on this host: the locally staged
rocBLAS package lacks a `gfx1150` Tensile library and aborts with `Illegal seek
for GPU arch : gfx1150`. The exact-operation HIP benchmark above remains valid
because llama.cpp's native HIP quantized kernel does not enter that missing
rocBLAS path. No all-HIP end-to-end number is claimed.

For a focused correctness rerun after building the generated kernel artifacts:

```sh
export GGML_XDNA_AIE2P_Q4_0_GEMV_BUNDLE=$PWD/ggml/src/ggml-xdna/kernels/aie2p/gemv-q4_0-288.ggmlxdna
build/bin/test-backend-ops test -b XDNA0 -o MUL_MAT \
  -p 'type_a=q4_0,type_b=f32,m=288,n=1,k=288'
```

## Next compute step

The direct Q4_0 contract now executes genuine attention projections in a small
LLM, but the implementation remains a scalar, single-tile validation kernel.
The next gate is a production-size LLM linear layer with AIE2P packed-int4
vector unpack, multi-tile row distribution, and vector/F32 accumulation. It
must retain native GGML bytes and validate CPU equivalence before each shape
enters `supports_op`. AIE2 needs a separate unpack implementation.

## Go/no-go checkpoint

| Question | Current evidence |
| --- | --- |
| Clean ACCEL registration and discovery? | **GO**: dynamic `XDNA0`, real XRT/plugin/device-node path. |
| Real GGML `MUL_MAT` on XDNA? | **GO for fixed Q4_0 and BF16 shapes**; CPU comparisons pass and a real GGUF `Qcur-0` node executed. |
| Fundamentally UMA/system-backed weights? | **GO for writable host allocations on XDNA**; XRT maps the original pointer. HIP currently stages CPU-owned prompt weights instead of directly sharing that buffer. |
| Reuse without per-token weight copy? | **GO for model-lifetime immutable buffers**: persistent parent/view/run, constant registrations, zero weight-copy bytes. **NO for arbitrary buffer lifetimes** until registration ownership moves into an XDNA buffer type. |
| Ordinary read-only GGUF mmap? | **NO on the installed driver**; XDNA rejects `CPU_Mapped`, so default loading falls back safely. Upstream's driver fix exists but is not yet physically validated here. |
| Quantized batch-one GEMV? | **GO for native 288x288 Q4_0 and a small real LLM**; production-size shapes and optimization remain open. |
| ROCm comparison and hybrid value? | **Scheduling GO, current kernel performance NO-GO**: constructed and real graphs select ROCm prefill/XDNA decode, but HIP is about 2,241x faster for the exact operation. Production shared-HIP-buffer policy remains open. |
| NPU utilization and energy? | **OPEN**: installed XRT/sysfs exposes neither a per-kernel utilization/energy counter nor estimated NPU power (`Estimated Power: N/A`); external SoC telemetry is required. |

The experiment is a no-go if direct-layout Q4_0 cannot beat HIP after activation
transfer and synchronization, if immutable weights require per-token payload
copies, or if safe scheduler selection requires model-specific changes. Raw
TOPS alone is not a success criterion; report total GEMV latency, effective
compressed-weight GB/s, sync/CPU overhead, and projected token throughput.

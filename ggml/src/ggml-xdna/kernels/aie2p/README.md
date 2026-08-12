# AIE2P bring-up kernels

This directory contains MIT-licensed kernels and artifact tooling written
for this experiment, plus Apache-2.0-with-LLVM-exception IRON generators adapted
from Xilinx/mlir-aie. The meaningful path directly consumes native GGML
`Q4_0[288,288] x F32[288,1]`; the BF16 version is retained as a simpler plumbing
reference. Production-shape specializations cover the Gemma 4 E4B
sliding-window K/V projection `Q4_0[2560,512] x F32[2560,1]` and its gate/up
projection `Q4_0[2560,10240] x F32[2560,1]`, plus the Llama 3.2 1B gate/up
projection `BF16[2048,8192] x F32[2048,1]` and the Llama 3.2 3B gate/up
projection `BF16[3072,8192] x F32[3072,1]`. A native
`Q8_0[2560,9216] x F32[2560,1]` specialization covers the Qwen 3.5 4B gate/up
shape. The host converts only the small activation to BF16, while weights
remain in their registered GGML host allocation. The Q4_0 compute kernel uses
AIE2P packed-nibble vector unpack, int8-to-BF16 conversion, vector multiply and
vector reduction; the Q8_0 kernel consumes the native signed-byte lanes
directly. The optimized Q4_0 artifact distributes the nine 32-row groups over
nine workers. The 288x288 BF16 reference remains single-worker; the production
BF16 specialization uses all 32 compute tiles.

Native Q4_K x Q8_K was also proven physically correct for the Qwen3.5 9B
`M=12288, K=4096` gate/up shape in a cache-only row-batched experiment. It is
deliberately not generated or registered here: the best 10,001-iteration XDNA
result was 1,763.780 us full p50 versus 337.339 us for the exact-payload HIP
control, or 5.229x slower. The retained research report and the four-model
ROCm/KV baseline are summarized in `docs/xdna-backend-design.md`; they are
evidence for the next kernel decision, not additional advertised capability.

Kernel artifacts are generated, not committed. With an MLIR-AIE IRON
environment containing Peano:

```sh
python3.12 -m venv /path/to/ironenv
/path/to/ironenv/bin/python -m pip install \
    "mlir_aie==1.3.4" \
    -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.3.4
/path/to/ironenv/bin/python -m pip install \
    "llvm-aie==21.0.0.2026062301+cb664e8c" \
    -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
```

Those are the independently reproduced versions used for the physical results
below; Torch or Torch-XDNA is not a build dependency. Then generate the local
artifacts:

```sh
source /opt/xilinx/xrt/setup.sh
make IRON_ENV=/path/to/ironenv
export GGML_XDNA_AIE2P_Q4_0_GEMV_BUNDLE="$PWD/gemv-q4_0-288.ggmlxdna"
```

For the E4B K/V and gate/up projection artifacts, configure their registered
bundles:

```sh
export GGML_XDNA_AIE2P_Q4_0_GEMV_M512_K2560_BUNDLE="$PWD/gemv-q4_0-m512-k2560.ggmlxdna"
export GGML_XDNA_AIE2P_Q4_0_GEMV_M10240_K2560_BUNDLE="$PWD/gemv-q4_0-m10240-k2560.ggmlxdna"
export GGML_XDNA_AIE2P_Q8_0_GEMV_M9216_K2560_BUNDLE="$PWD/gemv-q8_0-m9216-k2560.ggmlxdna"
export GGML_XDNA_AIE2P_BF16_GEMV_M8192_K2048_BUNDLE="$PWD/gemv-bf16-m8192-k2048.ggmlxdna"
export GGML_XDNA_AIE2P_BF16_GEMV_M8192_K3072_BUNDLE="$PWD/gemv-bf16-m8192-k3072.ggmlxdna"
```

The runtime discovers and host-validates every configured registered bundle. An exact `supports_op` query lazily resolves one shared device XRT program, and the first compute creates the backend-local BO and run state. It is therefore valid to configure multiple shape specializations in one backend instance without loading programs for shapes absent from the graph. Registry order is used only when more than one variant matches the same problem. Current full-array artifacts share one per-device XRT state and command gate across backend instances.

Configured variants remain available through `supports_op`, but the current
measured variants do not request automatic scheduler priority over an earlier
GPU. Set `GGML_XDNA_PREFER_OFFLOAD=1` only for an intentional heterogeneous
experiment. Exact shape, buffer, and immutability gates still apply, and an
XDNA-only configuration continues to select XDNA without the override.

To make the BF16 reference available as well, set
`GGML_XDNA_AIE2P_BF16_GEMV_BUNDLE`. The bundle header couples the declared
kernel kind, fixed tensor ABI, xclbin, and instruction stream. It prevents
accidentally selecting a correctly generated BF16, Q4_0, or Q8_0 bundle
through another dtype's variable before advertising `MUL_MAT` support. Bundles
remain trusted executable inputs: the header does not prove that a manually
relabeled arbitrary xclbin implements its declared contract.

## Artifact architecture identity

The 64-byte `GGXDNA1` container header has two supported ABI layouts. Existing AIE2P artifacts use ABI v1 unchanged: bytes 56 through 63 are zero, and the runtime accepts this legacy layout only for an AIE2P registry variant. ABI v2 assigns stable little-endian architecture IDs independently of C++ enum values:

| Header bytes | ABI v1 | ABI v2 |
| --- | --- | --- |
| 56-59 | zero | `1` for AIE2, `2` for AIE2P |
| 60-63 | zero | zero |

ABI v2 requires an exact match between the header ID and the registry variant. Unknown IDs, unknown variant architectures, cross-generation IDs, and nonzero reserved bytes are rejected before XRT sees the payload. A future AIE2 artifact must use ABI v2; this format support does not add an AIE2 registry entry or a physical XDNA1 support claim.

Every AIE2P Makefile rule names `--abi-version 1 --architecture aie2p` explicitly. ABI v2 must likewise name both the version and architecture, for example `pack-artifact.py --abi-version 2 --architecture aie2p ...`. `--architecture aie2` with ABI v1, ABI v2 without `--architecture`, unsupported versions, and unknown architecture names fail closed.

The artifact exercised during initial bring-up was built with MLIR-AIE Python
package 1.3.4 and Peano/llvm-aie commit
`cb664e8cc3eb42a12e3ad3cee28729785ffa97a3`. Before distributing artifacts,
rebuild and validate them with the current MLIR-AIE pin recorded in
`docs/xdna-backend-design.md`.

The IRON-generator provenance and full upstream license are retained in this
directory. `IRON_ENV` may use any Python version; the Makefile derives its
site-packages location from that interpreter. `MLIR_AIE` and `PEANO` remain
explicit overrides for nonstandard environments.
Run `make clean` before changing the compiler/toolchain path; Make cannot infer
toolchain identity changes from timestamps alone.

The Q4_0 kernel decodes the IEEE FP16 scale bits and packed `q - 8` values
on-tile without a dequantized weight copy. AIE2P/arch21 has no native IEEE-FP16
arithmetic, so the decoded scale is rounded to BF16 before vector dequantization;
the backend correctness tolerance covers that explicit numerical contract.
Peano lowers the inner Q4 path to AIE2P vector loads, unpack/shuffle,
conversion, multiply/MAC and reduction without scalar soft-float helpers.
Three row accumulators reuse each activation vector load; four rows were slower
because the pinned compiler spills vector state. The last packed block in each
FIFO object is staged through an aligned local buffer because the fast
unaligned vector primitive may issue wider surrounding loads.

The IRON design places eight workers across compute row 2 and the ninth at
column 0, row 3. One direct shim stream broadcasts the BF16 activation once.
Weights remain in native row order and are split as 4 + 4 + 1 worker groups;
outputs use matching memory-tile joins. This grouping respects the pinned
toolchain's five-input/five-output memory-tile DMA-channel limit. Every FIFO
remains depth one, so the known depth-two forwarding corruption is not
reintroduced. Further work should measure dispatch and DMA bounds and add a
separate, validated pipelined artifact rather than mutating this topology.

The E4B K/V design uses all 32 compute tiles as eight four-worker groups. Each
worker receives 16 contiguous rows (23,040 bytes) and the activation is
broadcast once. The core DMA descriptor length is measured in 32-bit address
granules on AIE2P, so that payload occupies 5,760 of the available 16,383
granules. Weights stay in native row-major GGML Q4_0 layout.

The E4B gate/up specialization uses the same 32-tile and eight-stream spatial
mapping for twenty 512-row waves. Each worker holds the single BF16 activation
object across all waves, while weights and outputs advance in 16-row tiles.
Depth-one FIFOs preserve the validated forwarding behavior of the pinned
toolchain. Four consecutive waves share one runtime task group, which
amortizes controller waits without using depth-two storage. A 10,001-call
physical stress test matched the CPU reference and measured 778.540 us median
(18.94 GB/s over the 14,745,600-byte native-Q4_0 matrix), 25.9% lower latency
than finishing one task group per wave. Eight-wave groups were faster in a
101-call test but hung during the sustained run and are deliberately rejected;
twenty waves exhaust shim BD IDs during lowering. A 15-weight-stream experiment
was also slower and nearly doubled the controller instruction stream, so it is
not included.

The shared K=2560 compute source subsequently replaced indexed three-row block
addressing with pointer induction. An order-balanced physical A/B over 16,016
launches was exact and reduced M=10240 mean latency from 778.872 to 765.908 us
(1.67%, 19.25 GB/s) and M=512 from 107.844 to 106.757 us (1.01%). The Peano
object lost 11 VLIW bundles and 80 code bytes; `-O2` and `-O3` produced
byte-identical objects, so the build retains `-O2`.

The Qwen 3.5 4B specialization consumes native GGML `block_q8_0`: one IEEE
FP16 scale followed by 32 signed int8 lanes, or 34 bytes per block and 2,720
bytes per K=2560 row. It does not repack the 25,067,520-byte matrix. The host
packs the 2,560-element F32 activation to BF16; on-tile dequantization converts
the FP16 scale to FP32 and deliberately rounds it to BF16, converts each int8
lane to BF16, and rounds the scaled lane to BF16 before the BF16 MAC/F32
accumulation. This BF16-rounded scale and dequant product are part of the
kernel's numerical contract.

The retained IRON topology uses all 32 compute tiles as eight four-worker
groups. Each worker receives 16 rows (43,520 bytes, or 10,880 of the 16,383
available 32-bit DMA length granules); one 512-row wave spans the array and 18
waves cover M=9216. The activation remains acquired across all waves, every
FIFO stays depth one, and four consecutive waves share one runtime task group.
The generator retains the Apache-2.0-with-LLVM-exception provenance used by the
in-tree Xilinx/mlir-aie-derived IRON programs; the compute source is original
MIT-licensed code in this directory. The generic registry descriptor is keyed
only by architecture, operation, dtype, layout, and exact tensor metadata--it
contains no model-name policy.

Peano `-O2` and `-O3` independently produced byte-identical 2,892-byte objects
(SHA-256 `231e12d85ed26c1f5585f21fa1876e6de72d0154e854900f35e84aa3604c9f9f`),
with a 1,920-byte text section and no unresolved helpers, so the Makefile
retains `-O2`. The generated MLIR has SHA-256
`64b41c9a7cc96a386145d1c8364255851be065d352be9a84ab2e42e002a8bcb3`.
The physically tested ABI-v1/kind-3 bundle is 237,139 bytes with SHA-256
`1a8b653d0de85f65825b389e50529bf5b17091c7780feae33367713985c73cdf`;
its 41,076-byte instruction stream has SHA-256
`cd1f298f20b075ef3b8d22e3402e56d49028227e39ab63a190b55087b85d7482`.
These generated files are local evidence and remain ignored by Git.

Deterministic `test-backend-ops` coverage compares the native-Q8_0 XDNA result
with the GGML CPU backend at the exact M=9216/N=1/K=2560 shape. A separate
physical directed runner covered zero weights, zero activation, three fixed
random seeds, int8 -128/+127 lanes, zero/positive/negative/small/large FP16
scales, and the final row/block. Zero cases were exact. The seeded
BF16-semantics CPU references measured NMSE `3.99554e-13` to `4.17706e-13`,
maximum absolute error `0.0185547` to `0.0234375`, and final-row absolute error
`0.00146484` to `0.00634766`, with no NaN or tolerance failure.
After integration, one backend process loaded this artifact together with all
four existing Q4_0/BF16 artifacts and passed all five CPU-reference cases.
That run registered 38.891 MiB across five roots, issued five one-time weight
syncs, and copied zero weight payload bytes; its log has SHA-256
`66e21a1318ea4b5701297b988b87b7612fcbaa0f5f972f6c9cafc5b8d14726fd`.
The committed backend test also includes the FP16 encodings derived from
`0.0007f` and `-0.0137f`; these are not BF16-exact and preserve coverage of the
production FP16-to-BF16 scale-rounding difference.

Five independent persistent GGML perf processes completed 10,605 submissions
with means of 792.26, 799.01, 794.53, 790.70, and 799.07 us/call. Their median
is 794.53 us, or 31.55 decimal GB/s over native weight bytes and 59.39 GFLOP/s;
each process registered and synchronized the immutable weight once and copied
zero weight payload bytes per steady call. A balanced direct-XRT A/B retained
2,000 launches per trial. The eight-stream medians were 794.821, 786.061,
785.790, and 786.281 us, while a cache-only 15-weight-stream candidate measured
882.902, 883.482, 875.522, and 875.182 us. The 15-stream candidate lost every
adjacent pair, was 11.84% slower by aggregate median, and enlarged the
instruction stream from 41,076 to 76,860 bytes, so it is rejected and not
tracked.

On the same M=9216/N=1/K=2560 operation, a `gfx1151` HIP build independently
passed a deterministic CPU comparison and measured 269.80 us median across five
`test-backend-ops` processes (92.91 GB/s, 174.89 GFLOP/s). A separate
synchronized allocator probe found direct `ROCm_Host` medians of 290.578 and
282.668 us with bit-identical output and zero weight copy. Thus XDNA is about
2.94x slower than the comparable device-resident HIP throughput measurement;
this variant must keep `prefer_for_offload=false` in policy-enabled registry
descriptors. Q8_0 coverage is an engineering/format milestone, not evidence of
a hybrid performance win.

These results used MLIR-AIE 1.3.4, Peano
`cb664e8cc3eb42a12e3ad3cee28729785ffa97a3`, and XRT 2.20.0; generated artifacts
remain local and must be rebuilt for distribution. The original task-group
stress artifact was an ABI-v1 189,971-byte bundle with SHA-256
`d51bc1954a357b105bb2a6f294b5208b7143767f5318ac0ccb3e660bdd0e5c9e`;
its 45,620-byte instruction stream has SHA-256
`72d2c1961c729a4c75733ac081b433f29ec885e8e6bacf720cb89fe489b8e913`.
The current pointer-induction M=10240 rebuild keeps that instruction stream,
is 187,411 bytes, and has SHA-256
`3d911dfbf792c9e1eae13c73bf15abe41533f5793df995931302424c67483605`.
The corresponding current M=512 bundle has SHA-256
`040d2d30463f990be44aa40449cc1066628476f3451e2fb966b502c144cedf3e`.

The BF16 M=8192, K=2048 specialization comes from
`unsloth/Llama-3.2-1B-Instruct-GGUF`, file
`Llama-3.2-1B-Instruct-BF16.gguf`, revision
`b69aef112e9f895e6f98d7ae0949f72ff09aa401`. That 2,479,595,264-byte GGUF has
SHA-256
`f5d05e90d179fa27c01a049be1574a0201bdc09ba0b345eb85e1dbf3b87610f3`.
Its gate/up tensors establish the stored `[2048,8192]` shape.

The kernel maps eight rows to each of 32 workers and covers 256 rows per wave
for 32 waves. A native row is 4,096 bytes. Sixteen rows would require 16,384
32-bit core-DMA granules, one more than the 16,383-granule limit, so the
eight-row object is an ABI/topology requirement rather than an arbitrary tile
choice. The placed design uses 41,004 of 65,536 local bytes per core and
131,200 of 524,288 bytes per memtile group. One activation acquisition is
retained across all waves. The current shim schedule repeats one task per
stream across all 32 waves while keeping every object FIFO at depth one.

Seven physical directed patterns, including first/last-column impulses and
finite BF16 normal/subnormal extremes, matched a long-double CPU reference
exactly. A 10,001-launch persistent run measured 721.268 us p50, 767.737 us
p95, and 46.521 GB/s over the 33,554,432-byte native matrix. Including the
F32-to-BF16 activation conversion and activation/output synchronization
measured 722.667 us p50. The generated ABI-v1 bundle is 188,051 bytes and has
SHA-256
`6f9132673cf1f6fc90445766dcadb07edddffa6d62f9b3a6e1574dc62fb62d1f`;
its instruction stream
has SHA-256
`91d5b29a0b6ff922d2a7ef1ae2ab1eee2316eaeccf82b402f71721771f3adcfc`.
Those are the original physically validated artifact hashes. A fresh
generic-source Makefile rebuild reproduced the generated and addressed MLIR,
PDI, and instruction stream byte-for-byte. The complete compute ELF differs
only in source-file metadata; its code, allocated, relocation, comment, and
stack sections and normalized symbols match, as do the corresponding sections
in all 32 placed ELFs. `xclbinutil` assigned a new xclbin UUID and embedded
path, so the rebuilt xclbin and enclosing bundle have different container
hashes: respectively
`4eeea09a4b41f023fe8fde02ad7f9d2610b9de0d5b11d094b89005334d0374e9`
and
`59d1c4b6aa0313c12a37bf72cb8f6ff5f8dc29eb55bc3afc1b80af6e0b2bcb2c`.
The old and rebuilt programs each passed one full `verify_each` launch for
both `edge` and `special_finite`, exactly matching the long-double reference
with zero mismatches, non-finites, absolute/relative error, NMSE, sentinel
corruption, weight mutation, or weight-payload copies. The old/new raw-log
SHA-256 values are, for `edge`,
`a2b2a2c04ccbb874fe7e46b4b4fca7783606b8fb2a616db6f56124f0b6617423`
and
`878a73bfe524db3ce99561776fa1500592a2623ca20aac4ca00f0f7b772f1331`,
and for `special_finite`,
`f850aed5685b790ce2c0ad312f8320ac1de281f9eae188059cb3a6ddbe37a25a`
and
`127be011d6c5d7e0711a07cfb239ed8441d5e75be0f39d8d7e1f607006222396`.

One ordinary dynamic-backend process then configured all seven AIE2P bundles.
Its inventory began at `ready=0`, `untried=7`, `failed=0`; lazy resolution
reached every specialization and all seven exact CPU-reference cases passed.
The final counters report seven successful calls/submissions, seven root
registrations (`118.898 MiB`), seven weight views, 42 explicit payload BOs
totalling 125,084,806 bytes, seven one-time weight syncs totalling 124,649,024
bytes, and zero weight-payload copies. The raw coexistence log has SHA-256
`912be8cf9728dd20aa9ad6b8d52310c4929e71998840704f2909bef442a7959d`.
The durable K=2048/integration report and evidence manifest have SHA-256
`2c5c11571669b1482335b82e92055772ec441c7fddde634c868e0214146f87f5`
and
`3db7f3c853bc6f29d16784b82459f70434146400d83536cb3ef0b9088b036c34`.
The device process itself exited successfully. Its first wrapper revision then
reported a false negative because lazy-resolution messages split six `OK`
markers from their `MUL_MAT` lines; the frozen raw log passed an exact
seven-case/counter audit, and the corrected post-parser passed `bash -n` and
`shellcheck` without rerunning the already complete device process.

The fair single-call gfx1151 control was faster. Across a clean D-H-H-D order,
the mean of two trial medians was 388.104 us for a device-resident BF16 weight
and 391.938 us for the same persistent weight in `ROCm_Host`. XDNA was 1.858x
and 1.840x slower respectively. The registry therefore exposes this
specialization for explicit coverage and XDNA-only use but keeps
`prefer_for_offload=false`.

The BF16 M=8192, K=3072 specialization is the Llama 3.2 3B gate/up shape. Its
source is `unsloth/Llama-3.2-3B-Instruct-GGUF`, file
`Llama-3.2-3B-Instruct-BF16.gguf`, revision
`e7d0997e49c9cb00d88b4c1a6a16aa894b0bbc31`. The 6,433,687,744-byte file has
SHA-256
`9b8dce13b6cbcd8b20037bd6383ee8e747b5034ca32f40b5b8ee2efa9ebf56b8`.
A strict one-launch runner read only `blk.0.ffn_gate.weight` at byte offset
846,186,688. That 50,331,648-byte BF16 tensor has SHA-256
`f38192e76c00a2829113cb3202b42df45687d46b68a8682f5b1b0e46073e5ee0`.
All 8,192 outputs passed a long-double reference with zero mismatches and
non-finites, NMSE `5.2182310884410422e-15`, unchanged weight bytes, and zero
post-ingest weight-payload copies.

The fixed K=2048 topology scales to K=3072 with 32 workers, eight rows per
worker, 32 waves, and depth-one FIFOs. The pinned
toolchain's placed MLIR shows a 12-byte IRON state object per worker; this is an
observed toolchain footprint budgeted by the generator, not an architectural
constant. Each compute tile uses 59,436 of 65,536 local bytes, leaving 6,100
bytes. Peano `-O2` and `-O3` produced byte-identical K=3072 compute objects,
so the Makefile retains `-O2`.

The cache-only candidate passed seven directed patterns twice and independent
1,001- and 10,001-launch closures. The 10,001-call kernel-only p50 was
1,045.054 us (48.162 GB/s); verified full p50 was 1,044.585 us. Clean,
order-balanced 1,001-call HIP controls measured 526.994 us kernel p50 and
549.750 us verified-full p50 for `ROCm0`, and 539.854 us and 583.869 us for
`ROCm_Host`. XDNA was 1.789x to 1.983x slower, so the descriptor keeps
`prefer_for_offload=false`. The complete standalone K=3072 report has SHA-256
`360e9c724daa27c35c4883e5ed962c931d42468e0788ada437991008c670ca34`.

The generic K=2048 A/B regression and the ordinary seven-artifact
`test-backend-ops` process described above close the remaining integration
gates. K=3072 is therefore physically validated as part of the tracked
seven-specialization backend while remaining deliberately opt-in on the
measured HIP comparison. The final generic-source K=3072 xclbin and bundle
have SHA-256
`d6d93d241749c25f6523d989d4f100c31e2003931d940fa599f6f35d0e831dfd`
and
`6c8022bd11b30b9e4fcd0824f4f72524b3ebf14dd4c2cc2c83dbb8572c012385`;
the instruction stream remains byte-identical to the standalone candidate.

The tracked generator now replaces the 256 one-wave weight tasks and 256
one-wave output tasks with eight 32-wave repeated tasks for each stream. With
activation, this deliberately reduces shim tasks from 513 to 17 and controller
bytes from 72,884 to 2,452. The compute kernel, depth-one FIFOs, memory layout,
locks, routes, 32 cores, 112 flows, and 16 links remain unchanged. Non-runtime
addressed MLIR matched the retained controls, and the K=2048 placed core ELFs
were byte-identical. Each K=3072 placed core differed only in non-loadable
`STT_FILE` source-name metadata; its executable sections and normalized
executable symbols were identical. The Makefile runs a fail-closed controller
decoder after `aiecc` for both K values and before packaging.

At K=2048, the 15/15 physical matrix passed all parity/directed/sustained
checks and exact bounded reads of both Llama 1B gate and up tensors. A balanced
A/B matrix reduced full p50 from 724.2425 to 668.540 us (7.691%) and kernel
p50 from 721.792 to 664.927 us (7.878%); one attempt required a swap-only
retry. The report and evidence manifest have SHA-256
`39b3deaf9aca6dd83b984095e0961ee71d1e743c52fe4b6dd83006eb5c4f1ac9`
and
`25d36584702974fe7b577e5ffd1b7545131a90d48144956f85b09db68ec18bd8`.
At K=3072, the exact Llama 3B gate tensor remained correct and balanced full
and kernel p50 improved by 7.232% and 8.353%. Its final report has SHA-256
`5aa8131d9bdc3b681a87d80848b71464f7f8ee77b9f5b92f44d1fe3b54e70b6c`.
HIP remains faster for both shapes, so both descriptors retain
`prefer_for_offload=false`.

An exact-head forced-C functional diagnostic then exercised the pinned Llama
3.2 3B BF16 GGUF at commit
`df255f7cf15f720995799b23b579ca05b5394d3e`. The correctness action matched
the frozen profile-B generated text exactly, SHA-256
`8f30d3a7df78841054291d938139e9e3d46b52955484fac1c3e1061e28cabf94`,
and recorded 7,118 successful XDNA calls, 56 root registrations, and zero
immutable-weight copy bytes. A separate `pp16` plus `tg4` graph action placed
exactly `ffn_gate-0..27` and `ffn_up-0..27` on XDNA. The `pp16` topology had
195 ROCm `MUL_MAT` identities plus `ffn_gate-27` and `ffn_up-27` on XDNA, with
two XDNA calls and two registrations. The `tg4` topology had 141 ROCm
`MUL_MAT` identities plus all 56 gate/up identities on XDNA, with 224 XDNA
calls and 56 registrations. Their deduplicated union contains 197 logical
`MUL_MAT` identities, 197 ROCm-tagged entries, and 56 additional XDNA-tagged
entries, for 253 backend-tagged entries total. All 28 Flash Attention
identities were on ROCm. The graph action therefore recorded 226 successful
XDNA calls, 58 root registrations, and zero immutable-weight copy bytes. The
frozen correctness stdout/stderr and graph
stderr/census summary have SHA-256
`dfb9e1ae37d3d972628de1d5b01c8c8af15e60f14fe0b8080fd81c94ad16152d`,
`970a1d5ebb02a4ac1b1d4ba8780e1f7b6a5c1e201949845a5e8d4f34f61cca08`,
`372871296b3988f5520db46aeca4049c8bca471eb913da72eb031da585f5e56d`,
and `e05b53ac9f18ea8e5d438a1c879e77e1cbf1b5398648239e4198bdc5b0dd13dc`.

Both actions moved host-global swap, so every recorded time is invalid:
correctness moved 406,206 pages in and 511,267 pages out, and graph placement
moved 2,747 pages in and 10,035 pages out. A matched exact-head H control and
order-balanced H/C performance remain unrun. This establishes forced-C
functional and placement GO only; it does not change
`prefer_for_offload=false`. The final durable report and manifest have SHA-256
`2ed271d3e84e78be89291e8e34dd5298c99390ac76224bc12b9e86b6a198cf0a`
and
`d410c481b7855980902241be179c8116e6425ad45ae3aeed977dd2754747205b`.

A separate K=3072 ping-pong experiment used four rows per worker, 128 rows per
wave, 64 waves, and depth-two weight/output FIFOs. All seven directed patterns,
the exact pinned tensor, and a 1,001-launch alternating FIFO-parity case passed.
Its bundle and instruction SHA-256 values are
`2443282a1dd8300b4313012fc62f42b581eeae93d5b33fd6c0cfd3bdaeafaadd`
and
`9a1862a4db88d770df617b12d86ad729b9c16e9166a9bfa8dbc9bc4eeba3b6ee`.
Clean balanced p50 was 1.2232145 ms full versus 1.0663185 ms for depth one
(14.714% slower), and 1.197059 ms kernel-only versus 1.0363775 ms (15.504%
slower). The retained design therefore remains eight rows per worker with
depth-one FIFOs. The physical-log and balanced-result manifests have SHA-256
`497f0fa807c218c0781f48ba5f442e380e9d829adcc29b22c6ddcdb73e3c2e2e`
and
`cdf06776693f86702f1a0d54ccf0a8b893365790ffd003ba91940637c3e41def`.
The final depth-two rejection report has SHA-256
`1cee68e52bdf4c296fd1e0755f353836e024ba0c51f57a1c3c84981d94d25306`.

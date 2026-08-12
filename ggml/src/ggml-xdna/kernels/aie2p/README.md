# AIE2P bring-up kernels

This directory contains MIT-licensed kernels and artifact tooling written
for this experiment, plus Apache-2.0-with-LLVM-exception IRON generators adapted
from Xilinx/mlir-aie. The meaningful path directly consumes native GGML
`Q4_0[288,288] x F32[288,1]`; the BF16 version is retained as a simpler plumbing
reference. Production-shape specializations cover the Gemma 4 E4B
sliding-window K/V projection `Q4_0[2560,512] x F32[2560,1]` and its gate/up
projection `Q4_0[2560,10240] x F32[2560,1]`, plus the Llama 3.2 1B gate/up
projection `BF16[2048,8192] x F32[2048,1]`. A native
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

The packer continues to emit ABI-v1 AIE2P bundles for existing Makefile calls. ABI v2 must name both the version and architecture explicitly, for example `pack-artifact.py --abi-version 2 --architecture aie2p ...`. `--architecture aie2` with ABI v1, ABI v2 without `--architecture`, unsupported versions, and unknown architecture names fail closed.

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
retained across all waves and four waves share each depth-one task group.

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
Those are the physically validated artifact hashes. A fresh Makefile rebuild
reproduced the MLIR, compute object, addressed MLIR, PDI, and instruction
stream byte-for-byte; `xclbinutil` assigned a new xclbin UUID, so the rebuilt
xclbin and enclosing bundle have different container hashes.

The ordinary GGML backend test also loaded the physically validated bundle
alongside the other five registered artifacts and passed all six CPU-reference
cases with zero weight-copy bytes. That coexistence log has SHA-256
`3558dd263fb259326b14e41f00669ebe6b85ba5798195a8d64c75a6fcf547f27`.

The fair single-call gfx1151 control was faster. Across a clean D-H-H-D order,
the mean of two trial medians was 388.104 us for a device-resident BF16 weight
and 391.938 us for the same persistent weight in `ROCm_Host`. XDNA was 1.858x
and 1.840x slower respectively. The registry therefore exposes this
specialization for explicit coverage and XDNA-only use but keeps
`prefer_for_offload=false`.

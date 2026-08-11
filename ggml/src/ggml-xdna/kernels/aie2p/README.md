# AIE2P bring-up kernel

This directory contains MIT-licensed kernels and artifact tooling written
for this experiment, plus Apache-2.0-with-LLVM-exception IRON generators adapted
from Xilinx/mlir-aie. The meaningful path directly consumes native GGML
`Q4_0[288,288] x F32[288,1]`; the BF16 version is retained as a simpler plumbing
reference. Production-shape specializations cover the Gemma 4 E4B
sliding-window K/V projection `Q4_0[2560,512] x F32[2560,1]` and its gate/up
projection `Q4_0[2560,10240] x F32[2560,1]`. The host converts only the small
activation to BF16, while weights
remain in their registered GGML host allocation. The Q4_0 compute kernel uses
AIE2P packed-nibble vector unpack, int8-to-BF16 conversion, vector multiply and
vector reduction. The optimized Q4_0 artifact distributes the nine 32-row
groups over nine workers, while the BF16 reference remains single-worker.

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
```

The runtime discovers every configured registered bundle, creates one
persistent XRT state per artifact, and selects among them from the generic
`xdna_problem` metadata. It is therefore valid to configure multiple shape
specializations in one backend instance. Registry order is used only when more
than one variant matches the same problem.

To make the BF16 reference available as well, set
`GGML_XDNA_AIE2P_BF16_GEMV_BUNDLE`. The bundle header couples the declared
kernel kind, fixed tensor ABI, xclbin, and instruction stream. It prevents
accidentally selecting a correctly generated BF16 bundle through the Q4
variable (and vice versa) before advertising `MUL_MAT` support. Bundles remain
trusted executable inputs: the header does not prove that a manually relabeled
arbitrary xclbin implements its declared contract.

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

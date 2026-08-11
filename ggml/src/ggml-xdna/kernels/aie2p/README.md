# AIE2P bring-up kernel

This directory contains MIT-licensed kernels and artifact tooling written
for this experiment, plus Apache-2.0-with-LLVM-exception IRON generators adapted
from Xilinx/mlir-aie. The meaningful path directly consumes native GGML
`Q4_0[288,288] x F32[288,1]`; the BF16 version is retained as a simpler plumbing
reference. The host converts only the small activation to BF16, while weights
remain in their registered GGML host allocation. The Q4_0 compute kernel uses
AIE2P packed-nibble vector unpack, int8-to-BF16 conversion, vector multiply and
vector reduction. Its IRON design remains a single-worker performance step;
multi-tile row distribution is the next specialization.

Kernel artifacts are generated, not committed. With an MLIR-AIE IRON
environment containing Peano:

```sh
source /opt/xilinx/xrt/setup.sh
make IRON_ENV=/path/to/ironenv
export GGML_XDNA_AIE2P_Q4_0_GEMV_BUNDLE="$PWD/gemv-q4_0-288.ggmlxdna"
```

To select the BF16 reference instead, set
`GGML_XDNA_AIE2P_BF16_GEMV_BUNDLE`. Q4_0 takes priority if both bundles are
set. The bundle header couples the declared kernel kind, fixed tensor ABI,
xclbin, and instruction stream. It prevents accidentally selecting a correctly
generated BF16 bundle through the Q4 variable (and vice versa) before
advertising `MUL_MAT` support. Bundles remain trusted executable inputs: the
header does not prove that a manually relabeled arbitrary xclbin implements its
declared contract.

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

The Q4_0 kernel decodes the exact IEEE FP16 scale and packed `q - 8` values
on-tile without a dequantized weight copy. Peano lowers its inner Q4 path to
AIE2P vector loads, unpack/shuffle, conversion, multiply and reduction. The
last packed block in each FIFO object is staged through an aligned local buffer
because the fast unaligned vector primitive may issue wider surrounding loads.
The remaining single-worker performance work is to remove the per-block scalar
FP32 helper dependency without violating the GGML numerical contract, then
distribute output rows over multiple workers.
